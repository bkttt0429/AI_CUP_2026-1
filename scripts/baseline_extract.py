#!/usr/bin/env python
"""Feature extraction of the official rank baseline (Tutorial rank-prediction.ipynb §1) under
the project's player-grouped protocol.

Training pool = players of folds 1-4; validation = whole players of those folds, about 10,000
games per rank (aicup.baseline.split_players), so it matches the notebook's VAL_SIZE 100,000;
players of other folds (fold 0) are never written. Everything else is the notebook's: every
train_*.csv concatenated in name order and shuffled with seed 42, NPROC worker processes each
writing its own webdataset .tar.gz shards of PER_SHARD samples, keys '<rank>_<game_id>'.
--seen-fold 0 is the seen-player check of METHOD_NOTES §2.1: training also gets every fold-0
game that no question of scripts/baseline_eval.py asks (aicup.baseline.eval_questions), so
the evaluated players are known to the model through their other games; validation is unchanged.
Writes
  <out>/{val,train}/<tag>-w{i}-%06d.tar.gz   features.npy, label.txt, rank_idx.txt, game_name.txt
  <out>/{val,train}_games.csv                 game_key, player_id, rank, fold of every game sent
  <out>/split.json                            counts per split (players, games, written, bad, shards, bytes),
                                              seen_fold and the seen games added to training
"""
import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.baseline import eval_questions, game_features, split_players  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.scoring import TRAIN_DIR, player_games  # noqa: E402


def seen_mask(df, fold, asked):
    """True for the games (rows of df: rank, game_id, fold) of `fold` players whose key 'RANK:game_id' is not in `asked`."""
    return (df["fold"] == fold) & ~(df["rank"] + ":" + df["game_id"]).isin(asked)


def extract_worker(df_part, pattern, wid, per_shard, stats_path):
    """Extract one slice and write it as shards; each process owns its parser and ShardWriter."""
    import webdataset as wds

    written = bad = 0
    t0 = time.time()
    with wds.ShardWriter(pattern, maxcount=per_shard, verbose=0) as sink:
        for _, row in df_part.iterrows():
            feat = game_features(row["sgf_content"])
            if feat is None:
                bad += 1
                continue
            sink.write({
                "__key__": "%s_%s" % (row["rank"], row["game_id"]),
                "features.npy": feat,
                "label.txt": str(row["rank"]),
                "rank_idx.txt": str(RANK_TO_IDX[row["rank"]]),
                "game_name.txt": str(row["game_id"]),
            })
            written += 1
            if written % 20000 == 0:
                print("  [w%d] %d/%d  %.0f min" % (wid, written, len(df_part), (time.time() - t0) / 60), flush=True)
    Path(stats_path).write_text(json.dumps({"written": written, "bad": bad}))
    print("  [w%d] wrote %d (parse failures %d) %.1f min" % (wid, written, bad, (time.time() - t0) / 60), flush=True)


def extract_split(df_part, out_dir, tag, nproc, per_shard, min_shard_bytes=10 * 1024):
    """Run one split with `nproc` workers and check that the shards really hold data."""
    outdir = Path(out_dir) / tag
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"{outdir} is not empty; delete it before extracting again")
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    procs = []
    for i in range(nproc):
        pat = str(outdir / f"{tag}-w{i}-%06d.tar.gz")
        p = ctx.Process(target=extract_worker, args=(df_part.iloc[i::nproc], pat, i, per_shard, outdir / f"{tag}-w{i}.json"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit("%s: worker exited abnormally (exitcode %s)" % (tag, p.exitcode))
    stats = [json.loads((outdir / f"{tag}-w{i}.json").read_text()) for i in range(nproc)]
    shards = sorted(outdir.glob("*.tar.gz"))
    total = sum(s.stat().st_size for s in shards)
    avg = total / len(shards) if shards else 0
    print("%s: %d shards, %.2f GB, %.1f KB per shard" % (tag, len(shards), total / 1e9, avg / 1024), flush=True)
    # An erroring write loop still counts samples but leaves empty tars (46 bytes gzipped);
    # the mean shard size catches that, counting shards does not.
    if avg < min_shard_bytes:
        raise SystemExit("%s: shards average only %.0f bytes, judged empty" % (tag, avg))
    return {"written": sum(s["written"] for s in stats), "bad": sum(s["bad"] for s in stats),
            "shards": len(shards), "bytes": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "baseline_official"))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--train-folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--val-games-per-rank", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N games of each split (smoke tests)")
    ap.add_argument("--seen-fold", type=int, default=None,
                    help="also train on this held-out fold's games that no evaluation question asks (seen-player check)")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()

    folds = pd.read_csv(ROOT / "splits" / "player_folds.csv")
    train_ids, val_ids = split_players(folds, a.train_folds, a.val_games_per_rank, a.seed)
    held_out = set(folds.loc[~folds["fold"].isin(a.train_folds), "player_id"])
    if (set(train_ids) | set(val_ids)) & held_out:
        raise SystemExit("players of held-out folds in the train/val lists")

    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(TRAIN_DIR / "train_*.csv")))], ignore_index=True)
    df = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    df["fold"] = df["player_id"].map(folds.set_index("player_id")["fold"])
    if df["fold"].isna().any():
        raise SystemExit(f"{int(df['fold'].isna().sum())} games of players missing from player_folds.csv")
    in_train, asked, seen_ids = df["player_id"].isin(set(train_ids)), set(), set()
    if a.seen_fold is not None:
        if a.seen_fold in a.train_folds:
            raise SystemExit(f"--seen-fold {a.seen_fold} is a training fold")
        qsets = eval_questions(player_games(), folds, pd.read_csv(ROOT / "splits" / f"questions_fold{a.seen_fold}.csv"),
                               a.seen_fold)
        asked = {k for qs in qsets.values() for ks in qs["game_keys"] for k in ks}
        in_train |= seen_mask(df, a.seen_fold, asked)
        seen_ids = set(folds.loc[folds["fold"] == a.seen_fold, "player_id"])
    parts = {"val": df[df["player_id"].isin(set(val_ids))], "train": df[in_train]}
    if a.limit:
        parts = {tag: part.iloc[:a.limit] for tag, part in parts.items()}
    print(f"{len(df)} games; val {len(parts['val'])} games of {parts['val']['player_id'].nunique()} players, "
          f"train {len(parts['train'])} games of {parts['train']['player_id'].nunique()} players", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "seed": a.seed, "train_folds": a.train_folds,
               "val_games_per_rank": a.val_games_per_rank, "nproc": a.nproc, "per_shard": a.per_shard, "limit": a.limit,
               "seen_fold": a.seen_fold}
    if a.seen_fold is not None:
        added = parts["train"][parts["train"]["fold"] == a.seen_fold]
        summary["seen"] = {"players": int(added["player_id"].nunique()), "games": int(len(added)),
                           "question_games_kept_out": len(asked)}
        print(f"seen fold {a.seen_fold}: train gets {len(added)} games of {added['player_id'].nunique()} players; "
              f"{len(asked)} question games kept out", flush=True)
    for tag, part in parts.items():   # val first, as in the notebook
        if set(part["player_id"]) & (held_out - (seen_ids if tag == "train" else set())):
            raise SystemExit(f"{tag}: players of held-out folds")
        if (part["rank"] + ":" + part["game_id"]).isin(asked).any():
            raise SystemExit(f"{tag}: games of evaluation questions")
        pd.DataFrame({"game_key": part["rank"] + ":" + part["game_id"], "player_id": part["player_id"],
                      "rank": part["rank"], "fold": part["fold"].astype(int)}).to_csv(out / f"{tag}_games.csv", index=False)
        stats = extract_split(part, out, tag, a.nproc, a.per_shard)
        summary[tag] = {"players": int(part["player_id"].nunique()), "games": int(len(part)), **stats}
    summary["val_games_by_rank"] = {lv: int((parts["val"]["rank"] == lv).sum()) for lv in LEVELS}
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "split.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {summary['wall_minutes']} min: {json.dumps({t: summary[t] for t in parts})}", flush=True)


if __name__ == "__main__":
    main()
