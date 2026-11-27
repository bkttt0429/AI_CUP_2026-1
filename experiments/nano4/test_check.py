#!/usr/bin/env python
"""11/04 official test-file check (read-only; run through sbatch): format, question sizes, per-game validity, handicap
presence and distribution, colours, game lengths and passes, duplicate games -- against the training sample.

  python -I logs/nano4/test_check.py --test data/test/rank_prediction_test_public.csv data/test/rank_prediction_test_private.csv

Writes results/data/test_check.json (summary) and results/data/test_games.csv (one row per game: question_id, index, colour,
ok, error, moves, passes, target_moves, hd, sgf hash) for the later adversarial-validation step. Nothing here touches labels
or models; the training reference is the fold-1 scored sample (game lengths) and results/data/hd_table.csv (handicap shares).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.submission import PREFIXES, check_game, read_test_lenient  # noqa: E402

TRAIN_REF, HD_TABLE = ROOT / "results/l1b/scores3_B_hd/fold1", ROOT / "results/data/hd_table.csv"


def _check(task):
    file, qid, n, i, sgf, color = task
    ok, info = check_game(sgf, color)
    return {"file": file, "question_id": qid, "n": n, "i": i, "color": color, "ok": bool(ok), "error": info["error"], "moves": info["moves"],
            "passes": info["passes"], "target_moves": info["target_moves"], "hd": info["handicap"], "sha": hashlib.sha1(sgf.encode()).hexdigest()[:16],
            "moves_sha": hashlib.sha1(sgf.split(";", 2)[-1].encode()).hexdigest()[:16] if ";" in sgf else ""}


def quantiles(x):
    x = np.asarray(x, dtype=float)
    return [float(v) for v in np.quantile(x, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])] if len(x) else []


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/data/test_check.json"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--train-overlap", action="store_true",
                    help="also hash every training record (data/training/train_*.csv) and count test games that appear there verbatim, "
                         "with the training players they belong to (the signal for the paused identification route)")
    a = ap.parse_args()
    rep, tasks = {"files": {}}, []
    for path in a.test:
        qs, warnings = read_test_lenient(path)
        n = np.array([len(q["games"]) for q in qs])
        declared = np.array([q["declared_games"] for q in qs])
        ids = pd.Series([q["question_id"] for q in qs])
        rep["files"][path] = {"questions": int(len(qs)), "n_warnings": len(warnings), "warnings_head": warnings[:20],
                              "prefix_counts": dict(Counter(next((p for p in PREFIXES if q["question_id"].startswith(p)), "other") for q in qs)),
                              "num_games_hist": {int(k): int(v) for k, v in zip(*np.unique(n, return_counts=True))},
                              "num_games_mean": float(n.mean()) if len(n) else None, "declared_ne_filled": int((n != declared).sum()),
                              "duplicate_ids": int(ids.duplicated().sum()), "empty_colours": int(sum(c == "" for q in qs for _, c in q["games"]))}
        tasks += [(Path(path).name, q["question_id"], len(q["games"]), i, sgf, color) for q in qs for i, (sgf, color) in enumerate(q["games"])]
        print(f"{path}: {len(qs)} questions, {len(warnings)} format warnings, num_games {rep['files'][path]['num_games_hist']}", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_check, tasks, chunksize=16)
    g = pd.DataFrame(rows)
    out_csv = Path(a.out).with_name("test_games.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_csv, index=False)
    hd_present = g["hd"].notna()
    hd = g.loc[hd_present, "hd"].astype(int)
    rep["games"] = {"total": int(len(g)), "ok_share": float(g["ok"].mean()), "errors": dict(Counter(str(e).split(":")[0] for e in g.loc[~g["ok"], "error"])),
                    "colour_share": {c: float((g["color"] == c).mean()) for c in ("B", "W")},
                    "hd_present_share": float(hd_present.mean()), "hd_hist": {int(k): int(v) for k, v in hd.value_counts().sort_index().items()},
                    "hd_gt0_share": float((hd > 0).mean()) if len(hd) else None, "hd_ge2_share": float((hd >= 2).mean()) if len(hd) else None,
                    "hd_ge2_white_share": float((g.loc[hd_present & (g["hd"].fillna(0).astype(int) >= 2), "color"] == "W").mean()) if (hd >= 2).any() else None,
                    "moves_quantiles": quantiles(g.loc[g["ok"], "moves"]), "target_moves_quantiles": quantiles(g.loc[g["ok"], "target_moves"]),
                    "games_with_pass_share": float((g["passes"] > 0).mean()), "moves_lt_100_share": float((g.loc[g["ok"], "moves"] < 100).mean()),
                    "duplicate_sgf_within_test": int(g["sha"].duplicated().sum()),
                    "sgf_shared_by_two_questions": int((g.groupby("sha")["question_id"].nunique() > 1).sum()),
                    "duplicate_move_sequences": int(g["moves_sha"].duplicated().sum())}
    # per-question handicap mix (what the HD document would see)
    q_hd = g.assign(h=g["hd"].fillna(0).astype(int)).groupby("question_id")["h"].agg(["max", lambda s: (s > 0).mean()])
    rep["questions"] = {"with_any_handicap_share": float((q_hd["max"] > 0).mean()), "with_hd_ge2_share": float((q_hd["max"] >= 2).mean()),
                        "handicap_game_share_quantiles": quantiles(q_hd.iloc[:, 1])}
    # training reference
    ref = {}
    try:
        t = read_scored_games(TRAIN_REF)
        ref["train_target_moves_quantiles_fold1_sample"] = quantiles(t["n_moves"].astype(int))  # the scored table counts the target's moves only
        ref["train_colour_share_fold1_sample"] = {c: float((t["color"] == c).mean()) for c in ("B", "W")}
    except Exception as e:  # reference is optional
        ref["train_ref_error"] = str(e)
    if HD_TABLE.exists():
        h = pd.read_csv(HD_TABLE, usecols=["hd"])["hd"].astype(int)
        ref["train_hd_gt0_share"] = float((h > 0).mean())
        ref["train_hd_ge2_share"] = float((h >= 2).mean())
        ref["train_hd_hist"] = {int(k): int(v) for k, v in h.value_counts().sort_index().items()}
    if a.train_overlap:  # verbatim duplicates between the test games and the 1M training records
        seen = {}
        for p in sorted((ROOT / "data" / "training").glob("train_*.csv")):
            df = pd.read_csv(p, usecols=["player_id", "sgf_content"], dtype=str, keep_default_na=False)
            for pid, sgf in zip(df["player_id"], df["sgf_content"]):
                seen.setdefault(hashlib.sha1(sgf.strip().encode()).hexdigest()[:16], set()).add(pid)
        hit = g["sha"].map(lambda s: s in seen)
        players = set().union(*(seen[s] for s in g.loc[hit, "sha"])) if hit.any() else set()
        per_q = g.assign(hit=hit).groupby("question_id")["hit"].agg(["sum", "count"])
        rep["train_overlap"] = {"training_records_hashed": int(sum(len(v) for v in seen.values())), "test_games_in_training": int(hit.sum()),
                                "test_games_in_training_share": float(hit.mean()), "training_players_matched": len(players),
                                "questions_with_any_hit": int((per_q["sum"] > 0).sum()), "questions_fully_in_training": int((per_q["sum"] == per_q["count"]).sum())}
        print(f"train overlap: {rep['train_overlap']['test_games_in_training']} of {len(g)} test games appear verbatim in the training files "
              f"({rep['train_overlap']['training_players_matched']} training players); questions with any hit {rep['train_overlap']['questions_with_any_hit']}, "
              f"fully in training {rep['train_overlap']['questions_fully_in_training']}", flush=True)
    rep["train_reference"] = ref
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    gm = rep["games"]
    print(f"games {gm['total']}: ok {gm['ok_share']:.4f} errors {gm['errors']}; colours {gm['colour_share']}; HD present {gm['hd_present_share']:.3f}, "
          f"HD>0 {gm['hd_gt0_share']}, HD>=2 {gm['hd_ge2_share']} (train {ref.get('train_hd_gt0_share')}, {ref.get('train_hd_ge2_share')}); "
          f"HD>=2 target white share {gm['hd_ge2_white_share']}", flush=True)
    print(f"moves quantiles {gm['moves_quantiles']}; target moves {gm['target_moves_quantiles']} (train target moves {ref.get('train_target_moves_quantiles_fold1_sample')}); "
          f"<100 moves {gm['moves_lt_100_share']:.4f}; "
          f"games with a pass {gm['games_with_pass_share']:.4f}; duplicate sgf {gm['duplicate_sgf_within_test']}, shared by two questions "
          f"{gm['sgf_shared_by_two_questions']}, duplicate move sequences {gm['duplicate_move_sequences']}", flush=True)
    print(f"questions with any handicap game {rep['questions']['with_any_handicap_share']:.3f}, with HD>=2 {rep['questions']['with_hd_ge2_share']:.3f}; written {a.out}, {out_csv}", flush=True)


if __name__ == "__main__":
    main()
