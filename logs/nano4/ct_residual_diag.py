"""D4: is the label-side signal still on the table for the submission's architecture (ct3 = C / T, 40 games per player)?
(user 2026-11-05: ct3 Public 0.7625 is the best; a signal only matters if it applies there.) No training of C / T: reads the
out-of-fold question probabilities the section 4M M8 CV members already wrote (results/trunk_seq/es/val<v>/<arm>_s<seed>.npz,
the validation questions are l1b_cv.pilot_questions in the same order; the stored per-question scores are checked against the
recomputed ones). fold 0 is not touched.

Models: CTx40-6 = 3 C-x40 + 3 T-x40 (seeds 0-2, ct3's recipe without fold 0), CT-6 = 3 C + 3 T (ct's recipe), and the R-level
MLP of label_side_diag.npz when that file exists. Must reproduce ens_m8.json's 6_new / 6_old per fold.

  (a) persistent / sampling split of the class-and-size-centred error E[rank] - label (n=5: four quarters, n=10: two halves).
  (b) Spearman of that error with the question's label-side covariates (opp - own pmi, opp pmi, own pmi, signed HD gap, HD1
      share), per question by n and per player. Section 2.20's reading for fused_opp_diff: per question +0.034 / -0.014 / -0.072
      (n = 5 / 10 / 20), per player -0.123. Near zero at every n -> ct3 already uses that linear information.
  (c) Cross-fitted stacker on CTx40-6 (fit on three folds' OOF, apply to the fourth; SoftmaxModel, class-balanced, L2 1e-3):
      S0 = [log P] (calibration control), S1 = [log P, x, x * log n] with x = (opp - own, opp pmi, opp pmi std, signed gap,
      HD1 share, log n). PRE-SPECIFIED: S1 vs S0, conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0, (4) >= 0.005.
      Report only: S0 vs base, S1 vs base. A pass says ct3 leaves label-side signal unused that a reweighting recovers; it is not
      yet a submission rule (the CV members are not ct3's 12 models, which train on five folds and are sharper).

  sbatch -J ctres -t 01:00:00 scripts/nano4/run.sbatch python -I logs/nano4/ct_residual_diag.py --out results/r_variants/ct_residual_diag.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import SoftmaxModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from label_side_diag import SIZES, centred_errors, player_moments  # noqa: E402

K = len(LEVELS)
ES = ROOT / "results/trunk_seq/es"
MODELS = {"CTx40-6": [("C-x40", s) for s in range(3)] + [("T-x40", s) for s in range(3)],
          "CT-6": [("C", s) for s in range(3)] + [("T", s) for s in range(3)]}
COV = ["opp_minus_own", "opp_pmi", "own_pmi", "signed_gap", "hd1_share"]


def spearman(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rlevel-npz", default=str(ROOT / "results/r_variants/label_side_diag.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    Q = {k: [] for k in ["y", "n", "pid", "fold", "X"] + [f"P_{m}" for m in MODELS]}
    for v in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[v], v)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[v] / f"fold{v}")
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
        val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl, folds_df))
        X, y, n, _ = hd_check.features(hsl, l1b, opp, hd, val)
        B = X["fused_opp_diff_hd"]
        Q["y"].append(y); Q["n"].append(n); Q["pid"].append(val["player_id"].to_numpy(dtype=str)); Q["fold"].append(np.full(len(y), v))
        Q["X"].append(B)
        for name, members in MODELS.items():
            Ps = []
            for arm, s in members:
                z = np.load(ES / f"val{v}" / f"{arm}_s{s}.npz")
                P, stored = z["proba"].astype(float), z["scores"]
                if len(P) != len(y):
                    raise SystemExit(f"fold {v} {arm}_s{s}: {len(P)} questions, expected {len(y)}")
                mine = evaluate(P, y, val)[1]
                bad = int((~np.isclose(mine, stored, atol=1e-9)).sum())  # float32 storage can flip a near-tie; a wrong order flips thousands
                if bad > 0.005 * len(y):
                    raise SystemExit(f"fold {v} {arm}_s{s}: stored scores disagree with the recomputed ones on {bad} questions (order?)")
                if bad:
                    print(f"fold {v} {arm}_s{s}: {bad} near-tie score flips from float32 storage", flush=True)
                Ps.append(P)
            Q[f"P_{name}"].append(np.mean(Ps, axis=0))
        print(f"fold {v}: {len(y)} questions, members aligned ({time.time() - t0:.0f} s)", flush=True)
    Q = {k: np.concatenate(v) for k, v in Q.items()}
    y, n, pid, fold, X = Q["y"], Q["n"], Q["pid"], Q["fold"], Q["X"]
    rl = Path(a.rlevel_npz)
    if rl.exists():
        Z = np.load(rl)
        if np.array_equal(Z["y_val"], y) and np.array_equal(Z["pid_val"], pid):
            Q["P_R-MLP"] = Z["P_val_base"]
        else:
            print("label_side_diag.npz does not match these questions; R-MLP skipped")
    names = [k[2:] for k in Q if k.startswith("P_")]
    cov = {"opp_minus_own": X[:, 56], "opp_pmi": X[:, 57], "own_pmi": X[:, 57] - X[:, 56], "signed_gap": X[:, 62] - X[:, 63], "hd1_share": X[:, 60]}
    rep = {"folds": a.folds, "models": {}}
    S = {}
    for name in names:
        P = Q[f"P_{name}"]
        S[name] = evaluate(P, y, pd.DataFrame({"n_games": n}))[1]
        per_fold = [balanced_mean_over_n(S[name][fold == v], y[fold == v], n[fold == v]) for v in a.folds]
        m = {"score": float(np.mean(per_fold)), "per_fold": per_fold, "by_n": {str(s): balanced_mean_over_n(S[name], y, n, sizes=(s,)) for s in SIZES}}
        e = centred_errors(P, y.astype(float), n)
        m["error_variance"] = {}
        for size in (5, 10):
            M = player_moments(e, e, y, pid, n, size)
            m["error_variance"][str(size)] = {"total": float(M[:, 2].mean()), "persistent": float(M[:, 1].mean()), "persistent_share": float(M[:, 1].mean() / M[:, 2].mean())}
        m["residual_spearman"] = {}
        for c in COV:
            row = {str(s): spearman(e[n == s], cov[c][n == s]) for s in SIZES}
            pe = pd.DataFrame({"pid": pid, "e": e, "x": cov[c]}).groupby("pid")[["e", "x"]].mean()
            row["per_player"] = spearman(pe["e"], pe["x"])
            m["residual_spearman"][c] = row
        rep["models"][name] = m
        print(f"{name:8s} {m['score']:.4f} folds {[round(x, 4) for x in per_fold]} by n " + " ".join(f"{k}: {v:.4f}" for k, v in m["by_n"].items())
              + f" | persistent share n5 {m['error_variance']['5']['persistent_share']:.3f} n10 {m['error_variance']['10']['persistent_share']:.3f}")
        for c in COV:
            r = m["residual_spearman"][c]
            print(f"   residual ~ {c:13s} n5 {r['5']:+.3f}  n10 {r['10']:+.3f}  n20 {r['20']:+.3f}  per player {r['per_player']:+.3f}")
    # (c) stacker on CTx40-6
    P = Q["P_CTx40-6"]
    logn = np.log(n.astype(float))
    xs = np.column_stack([cov["opp_minus_own"], cov["opp_pmi"], X[:, 58], cov["signed_gap"], cov["hd1_share"], logn])
    feats = {"S0": np.log(np.clip(P, 1e-9, 1)), "S1": np.column_stack([np.log(np.clip(P, 1e-9, 1)), xs, xs * logn[:, None]])}
    stack = {}
    for k, F in feats.items():
        out = np.zeros_like(P)
        for v in a.folds:
            tr, te = fold != v, fold == v
            out[te] = SoftmaxModel(l2=1e-3).fit(F[tr], y[tr]).predict_proba(F[te])
        stack[k] = evaluate(out, y, pd.DataFrame({"n_games": n}))[1]
    base = S["CTx40-6"]
    rep["stacker"] = {}
    for k in stack:
        pf = [balanced_mean_over_n(stack[k][fold == v], y[fold == v], n[fold == v]) for v in a.folds]
        rep["stacker"][k] = {"score": float(np.mean(pf)), "per_fold": pf}
    for (a_, b_) in (("S1", "S0"), ("S0", "base"), ("S1", "base")):
        sa = stack[a_]
        sb = stack[b_] if b_ != "base" else base
        dd = paired_diff(sa, sb, y, pid, n, n=a.n_boot)
        pfa = rep["stacker"][a_]["per_fold"]
        pfb = rep["stacker"][b_]["per_fold"] if b_ != "base" else rep["models"]["CTx40-6"]["per_fold"]
        pfd = [p - q for p, q in zip(pfa, pfb)]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(x > 0 for x in pfd) >= 3), "4": bool(dd["diff"] >= 0.005)}
        tag = "PRE-SPECIFIED " if (a_, b_) == ("S1", "S0") else ""
        rep["stacker"][f"{a_} vs {b_}"] = {**dd, "per_fold_diff": pfd, "conditions": cond, "pass": all(cond.values())}
        print(f"{tag}stacker {a_} vs {b_}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{x:+.4f}" for x in pfd)
              + " by n " + " ".join(f"{s}: {x:+.4f}" for s, x in dd["by_n"].items()) + f" -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
