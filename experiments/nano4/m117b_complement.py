#!/usr/bin/env python
"""M117b: is the rank information of the unprojected trunk rows already in the base's per-move scores? (report only)

M117: a per-game learner reads the level better from the unprojected per-game trunk statistics (1,536 dims) than from the usual
projected row (256). The base ensemble's tokens, however, already hold a supervised read-out of the whole trunk: the per-move
log p of the cross-fitted fine-tuned heads (L1b, level + 10 centred) and of the 18 Human SL profiles (level + 18 centred). This
script asks whether the unprojected rows still add once those scores are in the input. Games: the intersection of the peer
session's row sample (/work/u2507115/pid_v2/fullrows/rows, folds 1-4) with this project's scored games (20 + x40 per player).
Per-game score row S (71 dims): mean / std of the 30 score columns over the target's plies, mean of the 11 L1b columns over the
opponent's plies (the per-ply arrays the base members train on: results/trunk_seq/scores[_x40]_fold<f>.npy).
Representations: S, S+rp256, S+full, rp256, full; learners, folds, seeds and metrics as M117 (m117_fullrows_rank.py); the
all-game question of a player needs >= 5 of its games in the intersection.

  python -I logs/nano4/m117b_complement.py --gpus 0 1 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import m117_fullrows_rank as m117  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import completed_parts, part_paths, player_games  # noqa: E402

N_S = ts.N_SCORES
FOLDS = m117.FOLDS


def score_rows(wanted, cache):
    """{game_key: [2 * 30 + 11]} for the wanted games found in the trunk-vector parts of folds 1-4 (base and x40 sets)."""
    out = {}
    for f in FOLDS:
        for vec in (ts.VEC, ts.EXTRAS[40]["vec"]):
            fold_dir = vec / f"fold{f}"
            sc = np.load(ts.scores_path(cache, f, vec, 3, False), mmap_mode="r")
            o = 0
            for s, k in completed_parts(fold_dir):
                csv_path, npz_path = part_paths(fold_dir, s, k)
                tab = pd.read_csv(csv_path, usecols=["game_key", "n_plies"], dtype={"game_key": str}, keep_default_na=False)
                lens = tab["n_plies"].to_numpy().astype(np.int64)
                starts = np.r_[0, np.cumsum(lens)]
                want = [i for i, key in enumerate(tab["game_key"]) if key in wanted]
                if want:
                    with np.load(npz_path) as z:
                        target = z["target"].astype(bool)
                    if len(target) != starts[-1]:
                        raise ValueError(f"{npz_path}: {len(target)} plies, the games table says {starts[-1]}")
                    for i in want:
                        x = np.asarray(sc[o + starts[i]:o + starts[i + 1]]).astype(np.float32)
                        tg = target[starts[i]:starts[i + 1]]
                        if tg.sum() < 2 or (~tg).sum() < 1:
                            continue
                        out[tab["game_key"][i]] = np.concatenate([x[tg].mean(0), x[tg].std(0), x[~tg][:, :11].mean(0)])
                o += int(starts[-1])
            if o != len(sc):
                raise ValueError(f"fold {f} {vec.name}: {len(sc)} score plies, the parts hold {o}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="/work/u2507115/pid_v2/fullrows/rows")
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"))
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="m117b_complement")
    a = ap.parse_args(argv)
    t0 = time.time()
    keys, RP, FULL = m117.load_rows(a.rows)
    S_of = score_rows(set(keys.tolist()), a.cache)
    keep = np.array([k in S_of for k in keys])
    print(f"{len(keys)} sampled games, {int(keep.sum())} of them among the scored games ({time.time() - t0:.0f} s)", flush=True)
    keys, RP, FULL = keys[keep], RP[keep], FULL[keep]
    S = np.stack([S_of[k] for k in keys]).astype(np.float32)
    games = player_games()
    folds = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold", "n_games"]]
    meta = pd.DataFrame({"game_key": keys}).merge(games, on="game_key", how="left").merge(folds, on="player_id", how="left")
    assert len(meta) == len(keys) and meta.player_id.notna().all()
    fold = meta.fold.to_numpy().astype(int)
    assert not (fold == 0).any(), "fold 0 is never read here"
    y = meta["rank"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy().astype(np.int64)
    player = meta.player_id.to_numpy().astype(str)
    n_games = meta.n_games.to_numpy().astype(int)
    print("by fold: " + ", ".join(f"{v}: {int((fold == v).sum())} games / {len(set(player[fold == v]))} players" for v in FOLDS), flush=True)
    reps = {"S": S, "S+rp256": np.hstack([S, RP]), "S+full": np.hstack([S, FULL]), "rp256": RP, "full": FULL}
    out_dir = m117.OUT
    paths = {v: out_dir / f"{a.tag}_fold{v}.npz" for v in FOLDS}
    kids = []
    for i, v in enumerate(FOLDS):
        pid = os.fork()
        if pid == 0:
            try:
                m117.run_fold(v, a.gpus[i % len(a.gpus)], reps, y, fold, player, paths[v])
                os._exit(0)
            except BaseException:
                import traceback
                traceback.print_exc()
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(1)
        kids.append(pid)
    if [k for k in kids if os.waitpid(k, 0)[1] != 0]:
        raise SystemExit("fold workers failed")
    names = [f"{r}|{l}" for r in reps for l in m117.LEARNERS]
    L = {n: np.zeros((len(keys), len(LEVELS)), np.float32) for n in names}
    for v in FOLDS:
        z = np.load(paths[v])
        for n in names:
            L[n][z["idx"]] = z[n]
    for l in m117.LEARNERS:  # late fusion: the score learner's log-probabilities plus w x a trunk-row learner's
        for w in (0.5, 1.0):
            for r in ("rp256", "full"):
                L[f"S(+){w:g}*{r}|{l}"] = L[f"S|{l}"] + w * L[f"{r}|{l}"]
    q = m117.questions(player, keys, np.arange(len(keys)), min_all=5)
    rng = np.random.default_rng(0)
    out = {"games": int(len(keys)), "players": int(len(set(player))), "scores": {}, "diffs": {}}
    Sc = {}
    for n in L:
        row = {"game_exact": float((L[n].argmax(1) == y).mean())}
        for size, qs in q.items():
            pred = np.array([L[n][g].sum(0).argmax() for g in qs])
            truth = np.array([y[g[0]] for g in qs])
            Sc[(n, size)] = m117.score(pred, truth)
            row[f"q{size}"] = float(Sc[(n, size)].mean())
        out["scores"][n] = row
    cl = {size: np.array([player[g[0]] for g in qs]) for size, qs in q.items()}
    fo = {size: np.array([fold[g[0]] for g in qs]) for size, qs in q.items()}
    act = np.array([n_games[g[0]] for g in q["all"]])
    pairs = []
    for l in m117.LEARNERS:
        pairs += [(f"S+full|{l}", f"S+rp256|{l}"), (f"S+rp256|{l}", f"S|{l}"), (f"S+full|{l}", f"S|{l}"), (f"full|{l}", f"rp256|{l}"),
                  (f"S(+)0.5*full|{l}", f"S(+)0.5*rp256|{l}"), (f"S(+)1*full|{l}", f"S(+)1*rp256|{l}"), (f"S(+)0.5*full|{l}", f"S|{l}")]
    print(f"\nquestions: 5 games {len(q['5'])}, 10 games {len(q['10'])}, all games {len(q['all'])} (mean {np.mean([len(g) for g in q['all']]):.1f} games)")
    print(f"{'representation|learner':26s} game exact   q5      q10     qall")
    for n in L:
        r = out["scores"][n]
        print(f"{n:26s} {r['game_exact']:.4f}       {r['q5']:.4f}  {r['q10']:.4f}  {r['qall']:.4f}")
    print("\npaired differences, bootstrap over players:")
    for x, b in pairs:
        row = {}
        for size in q:
            d = Sc[(x, size)] - Sc[(b, size)]
            row[f"q{size}"] = {"diff": float(d.mean()), "ci95": m117.boot(d, cl[size], a.boot, rng), "by_fold": [float(d[fo[size] == v].mean()) for v in FOLDS]}
        d = Sc[(x, "all")] - Sc[(b, "all")]
        row["qall_le40"] = {"diff": float(d[act <= 40].mean()), "ci95": m117.boot(d[act <= 40], cl["all"][act <= 40], a.boot, rng)}
        out["diffs"][f"{x} - {b}"] = row
        print(f"  {x:18s} - {b:20s} " + "  ".join(f"q{s} {row[f'q{s}']['diff']:+.4f} [{row[f'q{s}']['ci95'][0]:+.4f}, {row[f'q{s}']['ci95'][1]:+.4f}]" for s in q)
              + f"  folds(qall) {' '.join(f'{v:+.3f}' for v in row['qall']['by_fold'])}  <=40 g {row['qall_le40']['diff']:+.4f}")
    (out_dir / f"{a.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    for p in paths.values():
        p.unlink()
    print(f"wrote {out_dir / (a.tag + '.json')} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
