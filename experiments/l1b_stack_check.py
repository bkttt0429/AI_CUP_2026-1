#!/usr/bin/env python
"""Does the L1b fused MLP add to ens_top3? One pre-specified comparison on the folds 1-4 CV questions.

Issue #6, after decision point 2 (2026-10-26). ens_top3 is the submission rule on the ladder's CV files:
each member's (r3_pma, r4_xattn_t1, r5_ctm) probabilities averaged over seeds 0-2, then the members
averaged uniformly, then the expected-score decision. ens_top3_plus adds the fused MLP of scripts/l1b_cv.py
(Human SL + L1b question features, MLPModel(hidden=32), the same training questions) as a fourth member
with the same weight. The fused fit is rerun here with l1b_cv's own functions, so its probabilities are
those behind results/l1b/cv_summary.json; the questions are matched to the ladder files by question_id
(both are pilot.pilot_questions) and must match one to one.

Pre-specified: ens_top3_plus vs ens_top3, passing on conditions (1) player-bootstrap CI lower bound > 0,
(3) > 0 in at least 3 of the 4 validation folds and (4) >= 0.005, as l1b_cv's; (2) the difference > 0 for
every seed (seed s: the members' seed-s probabilities, the fused MLP being deterministic) is reported, not
required. Reported, also: fused alone, ens_top3 vs fused.

  python -I scripts/l1b_stack_check.py --l1b results/l1b/scores results/l1b/scores_B --out results/l1b/stack_check.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):  # `python -I` adds neither; scripts/ holds l1b_cv
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, load_cv, to_json  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
import l1b_cv  # noqa: E402

MEMBERS = ("r3_pma", "r4_xattn_t1", "r5_ctm")
SEEDS = (0, 1, 2)
PCOLS = [f"p{k}" for k in range(10)]


def fused_probabilities(hsl_dir, l1b_dirs, folds, per_player, log=print):
    """DataFrame (fold, question_id, player_id, y, n_games, p0..p9) of the fused MLP on every validation fold."""
    where = l1b_cv.locate_folds(l1b_dirs, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, where[f], f)
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "players": players, "val": l1b_cv.pilot_questions(players)}
    out = []
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds) if f != v], ignore_index=True)
        X_tr, y_tr, _ = l1b_cv.feature_sets(pd.concat([data[f]["hsl"] for f in folds if f != v], ignore_index=True),
                                            pd.concat([data[f]["l1b"] for f in folds if f != v], ignore_index=True), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = l1b_cv.feature_sets(data[v]["hsl"], data[v]["l1b"], val)
        P = MLPModel(hidden=32).fit(X_tr["fused"], y_tr).predict_proba(X_va["fused"])
        res, _ = evaluate(P, y_va, val)
        log(f"fold {v}: fused {res['balanced_mean_over_n']:.4f} on {len(val)} questions", flush=True)
        out.append(pd.DataFrame({"fold": v, "question_id": val["question_id"].to_numpy(), "player_id": val["player_id"].to_numpy(dtype=str),
                                 "y": y_va, "n_games": n_va, **{c: P[:, k] for k, c in enumerate(PCOLS)}}))
    return pd.concat(out, ignore_index=True)


def member_probabilities(cv_dir):
    """{member: {seed: frame}} of the ladder CV files."""
    return {m: {s: load_cv(Path(cv_dir) / f"{m}_s{s}.npz")[0] for s in SEEDS} for m in MEMBERS}


def align(frame, keys):
    """frame's p0..p9 in the order of keys (fold, question_id); ValueError unless the question sets are equal."""
    f = frame.assign(fold=frame["fold"].astype(int), question_id=frame["question_id"].astype(str)).set_index(["fold", "question_id"])
    idx = pd.MultiIndex.from_frame(keys[["fold", "question_id"]])
    if len(f) != len(idx) or not f.index.isin(idx).all() or not idx.isin(f.index).all():
        raise ValueError(f"question sets differ: {len(f)} vs {len(idx)} questions")
    return f.loc[idx, PCOLS].to_numpy(dtype=float), f.loc[idx, "y"].to_numpy(dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--cv-dir", default=str(ROOT / "results" / "ladder" / "cv"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    fused = fused_probabilities(a.hsl, a.l1b, folds, a.per_player)
    keys = fused[["fold", "question_id"]].astype({"fold": int, "question_id": str})
    y, pid, n, fold = fused["y"].to_numpy(dtype=int), fused["player_id"].to_numpy(dtype=str), fused["n_games"].to_numpy(), fused["fold"].to_numpy()
    P_fused = fused[PCOLS].to_numpy(dtype=float)
    members = member_probabilities(a.cv_dir)
    per_seed = {m: {} for m in MEMBERS}
    for m in MEMBERS:
        for s in SEEDS:
            P, y_m = align(members[m][s], keys)
            if not np.array_equal(y_m, y):
                raise ValueError(f"{m} seed {s}: labels differ from the fused questions")
            per_seed[m][s] = P
    P_mem = {m: np.mean([per_seed[m][s] for s in SEEDS], axis=0) for m in MEMBERS}
    P3 = np.mean([P_mem[m] for m in MEMBERS], axis=0)
    P4 = np.mean([P_mem[m] for m in MEMBERS] + [P_fused], axis=0)

    def scores(P):
        return evaluate(P, y, fused)[1]

    s3, s4, sf = scores(P3), scores(P4), scores(P_fused)
    head = {name: {"balanced_mean_over_n": balanced_mean_over_n(s, y, n),
                   "balanced_by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},
                   "per_fold": [balanced_mean_over_n(s[fold == v], y[fold == v], n[fold == v]) for v in folds]}
            for name, s in (("ens_top3", s3), ("ens_top3_plus", s4), ("fused", sf))}
    d = paired_diff(s4, s3, y, pid, n, n=a.n_boot)
    per_fold = [x - z for x, z in zip(head["ens_top3_plus"]["per_fold"], head["ens_top3"]["per_fold"])]
    seed_diff = []
    for s in SEEDS:
        P3s = np.mean([per_seed[m][s] for m in MEMBERS], axis=0)
        P4s = np.mean([per_seed[m][s] for m in MEMBERS] + [P_fused], axis=0)
        seed_diff.append(balanced_mean_over_n(scores(P4s), y, n) - balanced_mean_over_n(scores(P3s), y, n))
    cond = {"1": bool(d["ci95"][0] > 0), "2": bool(all(x > 0 for x in seed_diff)), "3": bool(sum(x > 0 for x in per_fold) >= len(folds) - 1),
            "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
    ref = paired_diff(s3, sf, y, pid, n, n=a.n_boot)
    summary = {"folds": folds, "questions": int(len(y)), "members": list(MEMBERS), "seeds": list(SEEDS), "rule": "seed mean, then uniform over members",
               "scores": head, "comparison": {"a": "ens_top3_plus", "b": "ens_top3", **d, "per_fold_diff": per_fold, "per_seed_diff": seed_diff,
                                              "conditions": cond, "pass": cond["1"] and cond["3"] and cond["4"]},
               "report_only": {"ens_top3_vs_fused": ref}, **environment("cpu")}
    for name, h in head.items():
        print(f"{name:14s} {h['balanced_mean_over_n']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    lo, hi = d["ci95"]
    print(f"ens_top3_plus vs ens_top3: {d['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]  seeds " + " ".join(f"{x:+.4f}" for x in seed_diff)
          + "  folds " + " ".join(f"{x:+.4f}" for x in per_fold) + "  conditions " + " ".join(f"{k} {'✓' if v else '✗'}" for k, v in cond.items())
          + f"  -> {'PASS' if summary['comparison']['pass'] else 'no'}")
    print(f"(report only) ens_top3 vs fused: {ref['diff']:+.4f} [{ref['ci95'][0]:+.4f}, {ref['ci95'][1]:+.4f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(summary, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
