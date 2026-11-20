"""Section 4M M51, R side redone with the shift measured WITHIN question size (2026-11-07). m51_shift.py's R counterfactuals compared the
test (n = 5 / 10 / 15 / 20, a quarter each) with the validation questions (n = 5 for 59%), so the evidence columns log_games / log_moves
(+0.85 sd) dominated the shift and were added to every validation question regardless of its size -- a design difference, not a
population shift. Here delta_R[n] = mean of the test questions of size n - mean of the validation questions of size n (n = 5, 10, 20),
each validation question receives its own size's delta, and the shift table averages the per-size shifts. Report only.
  1. per-column shift (validation sd of that size, averaged over sizes) and the largest columns.
  2. counterfactual blocks: own HSL curve (rel_<profile>, 0-17), own HSL summary (18-23), own L1b curve (24-33), own L1b summary (34-39),
     opponent curve (40-49), opponent summary (50-55), diff (56-58), HD (59-63), own HSL + L1b, all 64; and the eight largest single
     columns. Same metrics as m51_shift.py (confusion flows, L1 to ct9's observed test shares under the uniform / validation prior, the
     class-balanced score).
  3. the direct check (ct9's decisions among true-A / true-1D questions by tertile of hd_sente and of opp_pmi_mean), as before.
  python -I logs/nano4/m51_shift_r.py --val-fold 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_seq_check as ts  # noqa: E402
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES as R_NAMES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n  # noqa: E402
from feature_importance import probs_from, z_and_logits_fn  # noqa: E402
from m43_gate import decide, qscore  # noqa: E402
from m51_shift import confusion  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
A, D1 = RANK["A"], RANK["1D"]
SIZES = (5, 10, 20)
R_BLOCKS = {"own HSL curve": range(0, 18), "own HSL summary": range(18, 24), "own L1b curve": range(24, 34), "own L1b summary": range(34, 40),
            "opponent curve": range(40, 50), "opponent summary": range(50, 56), "diff": range(56, 59), "HD": range(59, 64),
            "own HSL + L1b": range(0, 40), "all 64": range(0, 64)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); a = ap.parse_args()
    t0, v, device = time.time(), a.val_fold, torch.device("cuda")
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.json"))["val"]
    keys = m["game_keys"]; y = np.array([RANK[r] for r in m["rank"]]); n = np.array(m["n_games"])
    R_raw = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
    X_raw, offsets = ts.raw_tokens([(ts.VEC / f"fold{v}", ts.scores_path(ROOT / "results/trunk_seq", v, ts.VEC, 3, True))])
    members = [ts.load_checkpoint(ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.pt", device) for arm in ("C", "T") for s in (0, 1, 2)]
    test = np.load(ROOT / "results/trunk_seq/es/m51_test_inputs.npz", allow_pickle=False)
    R_test = test["R"].astype(np.float64).mean(0); n_test = test["n_games"].astype(int)
    obs = pd.read_csv(ROOT / "submission_rank_ct9.csv")["pred_rank"].map(RANK).to_numpy()
    obs_shares = np.bincount(obs, minlength=10) / len(obs)
    print(f"fold {v}: {len(keys)} questions (n=5 {int((n == 5).sum())}, 10 {int((n == 10).sum())}, 20 {int((n == 20).sum())}), {len(members)} members; "
          f"test sizes " + " ".join(f"{s}:{int((n_test == s).sum())}" for s in (5, 10, 15, 20)) + f" ({time.time() - t0:.0f} s)", flush=True)

    # 1. shift within size
    d_R, shift = {}, {}
    for s in SIZES:
        mu_v, sd_v = R_raw[n == s].astype(np.float64).mean(0), R_raw[n == s].astype(np.float64).std(0) + 1e-9
        d_R[s] = (R_test[n_test == s].mean(0) - mu_v); shift[s] = d_R[s] / sd_v
    shift_avg = np.mean([shift[s] for s in SIZES], axis=0)
    top = np.argsort(-np.abs(shift_avg))[:14]
    print("R shift within size (test - validation, validation sd, mean over n = 5 / 10 / 20), largest: " + ", ".join(f"{R_NAMES[i]} {shift_avg[i]:+.2f}" for i in top), flush=True)
    for blk, cols in R_BLOCKS.items():
        if blk not in ("own HSL + L1b", "all 64"):
            print(f"    {blk:16s} mean |shift| {np.abs(shift_avg[list(cols)]).mean():.2f}; mean signed {shift_avg[list(cols)].mean():+.2f}", flush=True)
    rep = {"val_fold": v, "r_shift_sd_within_n": dict(zip(R_NAMES, shift_avg.tolist())), "r_shift_by_n": {str(s): dict(zip(R_NAMES, shift[s].tolist())) for s in SIZES},
           "observed_test_shares": obs_shares.tolist()}

    # 2. counterfactuals
    val_prior = np.bincount(y, minlength=10) / len(y); uni = np.full(10, 0.1)
    Z0 = []
    for model, tstats, _, _ in members:
        Xs = torch.as_tensor(ts.standardise_tokens(X_raw.copy(), tstats), device=device)
        Z0.append(z_and_logits_fn(model, Xs, offsets, keys, device)); del Xs; torch.cuda.empty_cache()
    del X_raw

    def probs(R_in):
        return np.mean([probs_from(mod, ((R_in - rstats[0]) / rstats[1]).astype(np.float32), Z, device) for (mod, _, rstats, _), Z in zip(members, Z0)], axis=0)

    def summary(P):
        d = decide(P); C = confusion(d, y, n)
        return {"l1_uniform": float(np.abs(C @ uni - obs_shares).sum()), "l1_val": float(np.abs(C @ val_prior - obs_shares).sum()),
                "shares_uniform": (C @ uni).tolist(), "shares_val": (C @ val_prior).tolist(),
                "flow": {"A->1D": float(C[D1, A]), "A->A": float(C[A, A]), "A->B": float(C[RANK['B'], A]), "1D->A": float(C[A, D1]), "1D->1D": float(C[D1, D1]),
                         "2D->3D": float(C[RANK['3D'], RANK['2D']]), "B->A": float(C[A, RANK['B']])},
                "balanced_score": float(balanced_mean_over_n(qscore(d, y), y, n)), "decisions": d}

    base = summary(probs(R_raw)); rep["base"] = {k: val for k, val in base.items() if k != "decisions"}
    print(f"base: score {base['balanced_score']:.4f}; L1 to the test's observed shares: uniform prior {base['l1_uniform']:.3f}, validation prior {base['l1_val']:.3f}; "
          f"flows A->1D {base['flow']['A->1D']:.3f} A->A {base['flow']['A->A']:.3f} A->B {base['flow']['A->B']:.3f} 1D->A {base['flow']['1D->A']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    rep["counterfactuals"] = {}

    def run(name, cols):
        cols = list(cols); R_cf = R_raw.copy()
        for s in SIZES:
            R_cf[np.ix_(n == s, cols)] += d_R[s][cols].astype(np.float32)
        sm = summary(probs(R_cf)); sm = {k: val for k, val in sm.items() if k != "decisions"}
        rep["counterfactuals"][name] = sm
        du = np.array(sm["shares_uniform"]) - np.array(base["shares_uniform"])
        print(f"{name:18s} score {sm['balanced_score'] - base['balanced_score']:+.4f}  L1 uniform {sm['l1_uniform'] - base['l1_uniform']:+.3f}  L1 val {sm['l1_val'] - base['l1_val']:+.3f}  "
              f"A->1D {sm['flow']['A->1D'] - base['flow']['A->1D']:+.3f}  A->A {sm['flow']['A->A'] - base['flow']['A->A']:+.3f}  A->B {sm['flow']['A->B'] - base['flow']['A->B']:+.3f}  "
              f"1D->A {sm['flow']['1D->A'] - base['flow']['1D->A']:+.3f}  d shares (uniform) " + " ".join(f"{lv} {x * 100:+.1f}" for lv, x in zip(LEVELS, du)), flush=True)

    for blk, cols in R_BLOCKS.items():
        run(blk, cols)
    for i in np.argsort(-np.abs(shift_avg))[:8]:
        run(f"col {R_NAMES[i]}", [int(i)])

    # 3. direct check
    rep["direct"] = {}
    for col_name, col in (("hd_sente", 60), ("opp_pmi_mean", 57)):
        x = R_raw[:, col].astype(np.float64)
        for c, cname in ((A, "A"), (D1, "1D")):
            sel = y == c; cuts = np.quantile(x[sel], [1 / 3, 2 / 3]); tert = np.digitize(x[sel], cuts); d_sel = base["decisions"][sel]
            rows_out = [{"tertile": t, "n": int((tert == t).sum()), "x_mean": float(x[sel][tert == t].mean()),
                         **{f"decide_{lv}": float((d_sel[tert == t] == RANK[lv]).mean()) for lv in ("B", "A", "1D", "2D")}} for t in range(3)]
            above = float((R_test[:, col] > cuts[1]).mean())
            rep["direct"][f"{col_name}|{cname}"] = {"tertiles": rows_out, "test_share_above_upper_tertile_cut": above, "cuts": cuts.tolist()}
            print(f"true {cname} by {col_name} tertile: " + " | ".join(f"x {r['x_mean']:.3f}: B {r['decide_B']:.2f} A {r['decide_A']:.2f} 1D {r['decide_1D']:.2f} 2D {r['decide_2D']:.2f}" for r in rows_out)
                  + f"   (test questions above the upper cut {cuts[1]:.3f}: {above:.2f})", flush=True)
    Path(ROOT / f"results/trunk_seq/es/m51_shiftr_val{v}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"written results/trunk_seq/es/m51_shiftr_val{v}.json ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
