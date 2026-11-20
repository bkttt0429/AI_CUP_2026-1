"""Section 4M M51 (2026-11-07): does the test set's INPUT shift explain ct9's test prediction pattern (fewer A, more kyu, more 1D) and
ct10's Public gain (1D -> A flips that scored)? Report only. On validation fold v with ct9's six members of that fold
(C/T-pc-f0-x40-oh seeds 0-2; the submission averages the four folds' members):
  1. shift table: (test mean - validation mean) / validation sd per R column (64) and per raw token column (0-157) on target and
     opponent plies; the test side from m51_test_inputs.npz (averaged over its two L1b models).
  2. counterfactuals: the test's mean shift of ONE input block added to every validation question's inputs, the members re-run
     (the question embedding Z once per token variant, the last MLP per R variant):
       R blocks   own HSL (0-23), own L1b (24-39), opponent (40-58), HD frequency (59-61), HD gap (62-63), HD all (59-63), all 64;
       token blocks   own HSL (139-157 on target plies), opponent HSL (same columns on opponent plies), own L1b (128-138, target),
                      opponent L1b (128-138, opponent), trunk (0-127, both roles with their own shift).
     For each: the n-balanced confusion C_cf (P(decide i | Y = j)); its predicted shares under the uniform and the validation prior
     against ct9's observed test shares (L1 over 10 classes); the flows P(decide 1D | A), P(decide A | A), P(decide B | A),
     P(decide A | 1D), P(decide 3D | 2D); the class-balanced official score. A block whose shift moves the shares toward the test's
     and raises P(decide 1D | A) is a candidate mechanism.
  3. direct check: among true-A and true-1D questions, ct9's decision shares by tertile of hd_sente (R 60) and of opp_pmi_mean (R 57).
  python -I logs/nano4/m51_shift.py --val-fold 1
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

RANK = {lv: i for i, lv in enumerate(LEVELS)}
A, D1 = RANK["A"], RANK["1D"]
R_BLOCKS = {"R own HSL": range(0, 24), "R own L1b": range(24, 40), "R opponent": range(40, 59), "R HD frequency": range(59, 62),
            "R HD gap": range(62, 64), "R HD all": range(59, 64), "R all 64": range(0, 64)}
L1B_COLS, HSL_COLS, TRUNK_COLS = slice(128, 139), slice(139, 158), slice(0, 128)
TOKEN_BLOCKS = {"tok own HSL": (HSL_COLS, "target"), "tok opponent HSL": (HSL_COLS, "opp"), "tok own L1b": (L1B_COLS, "target"),
                "tok opponent L1b": (L1B_COLS, "opp"), "tok trunk": (TRUNK_COLS, "both")}
N_TOKEN_COLS = 158


def column_stats(X, rows):
    s, s2, cnt = np.zeros(N_TOKEN_COLS), np.zeros(N_TOKEN_COLS), 0
    for i in range(0, len(rows), 500_000):
        blk = X[rows[i:i + 500_000], :N_TOKEN_COLS].astype(np.float64)
        s += blk.sum(0); s2 += (blk * blk).sum(0); cnt += len(blk)
    mean = s / max(cnt, 1)
    return mean, np.sqrt(np.maximum(s2 / max(cnt, 1) - mean * mean, 0.0))


def confusion(d, y, n):
    C = np.zeros((10, 10))
    for j in range(10):
        C[:, j] = np.mean([np.bincount(d[(y == j) & (n == m)], minlength=10) / max(((y == j) & (n == m)).sum(), 1) for m in (5, 10, 20)], axis=0)
    return C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); a = ap.parse_args()
    t0, v, device = time.time(), a.val_fold, torch.device("cuda")
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.json"))["val"]
    keys = m["game_keys"]; y = np.array([RANK[r] for r in m["rank"]]); n = np.array(m["n_games"])
    R_raw = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
    X_raw, offsets = ts.raw_tokens([(ts.VEC / f"fold{v}", ts.scores_path(ROOT / "results/trunk_seq", v, ts.VEC, 3, True))])
    tgt_rows = np.flatnonzero(X_raw[:, 158].astype(np.float32) > 0.5); opp_rows = np.flatnonzero(X_raw[:, 158].astype(np.float32) <= 0.5)
    members = [ts.load_checkpoint(ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.pt", device) for arm in ("C", "T") for s in (0, 1, 2)]
    test = np.load(ROOT / "results/trunk_seq/es/m51_test_inputs.npz", allow_pickle=False)
    names = [str(x) for x in test["l1b_names"]]
    R_test = test["R"].astype(np.float64).mean(0)  # [800, 64], the two L1b models averaged
    obs = pd.read_csv(ROOT / "submission_rank_ct9.csv")["pred_rank"].map(RANK).to_numpy()
    obs_shares = np.bincount(obs, minlength=10) / len(obs)
    print(f"fold {v}: {len(keys)} questions, {len(X_raw)} plies, {len(members)} members; test R from {names} ({time.time() - t0:.0f} s)", flush=True)

    # 1. shift table
    r_mean_val, r_sd_val = R_raw.astype(np.float64).mean(0), R_raw.astype(np.float64).std(0) + 1e-9
    d_R = R_test.mean(0) - r_mean_val
    shift_R = d_R / r_sd_val
    tok_shift, d_tok = {}, {}
    for role, rows in (("target", tgt_rows), ("opp", opp_rows)):
        mu, sd = column_stats(X_raw, rows)
        test_mu = np.mean([test[f"tok_{nm}_{role}_mean"] for nm in names if f"tok_{nm}_{role}_mean" in test], axis=0)
        d_tok[role] = (test_mu - mu).astype(np.float32); tok_shift[role] = (test_mu - mu) / (sd + 1e-9)
    rep = {"val_fold": v, "r_shift_sd": dict(zip(R_NAMES, shift_R.tolist())), "r_delta_raw": dict(zip(R_NAMES, d_R.tolist())),
           "r_test_mean": dict(zip(R_NAMES, R_test.mean(0).tolist())), "r_val_mean": dict(zip(R_NAMES, r_mean_val.tolist())),
           "token_shift_sd": {role: tok_shift[role].tolist() for role in tok_shift}}
    top = np.argsort(-np.abs(shift_R))[:12]
    print("R shift (test - validation, in validation sd), largest: " + ", ".join(f"{R_NAMES[i]} {shift_R[i]:+.2f}" for i in top), flush=True)
    for blk, cols in R_BLOCKS.items():
        if blk != "R all 64":
            print(f"    {blk:16s} mean |shift| {np.abs(shift_R[list(cols)]).mean():.2f}", flush=True)
    for role in ("target", "opp"):
        s = tok_shift[role]
        print(f"token shift on {role} plies: trunk mean |shift| {np.abs(s[0:128]).mean():.2f}, L1b block {np.abs(s[128:139]).mean():.2f} (level col {s[128]:+.2f}), "
              f"HSL block {np.abs(s[139:158]).mean():.2f} (level col {s[139]:+.2f})", flush=True)

    # 2. counterfactuals
    val_prior = np.bincount(y, minlength=10) / len(y); uni = np.full(10, 0.1)

    def token_pass(X_mod):
        Zs = []
        for model, tstats, _, _ in members:
            Xs = torch.as_tensor(ts.standardise_tokens(X_mod.copy(), tstats), device=device)
            Zs.append(z_and_logits_fn(model, Xs, offsets, keys, device)); del Xs; torch.cuda.empty_cache()
        return Zs

    def probs(R_in, Zs):
        return np.mean([probs_from(mod, ((R_in - rstats[0]) / rstats[1]).astype(np.float32), Z, device) for (mod, _, rstats, _), Z in zip(members, Zs)], axis=0)

    def summary(P):
        d = decide(P); C = confusion(d, y, n)
        return {"l1_uniform": float(np.abs(C @ uni - obs_shares).sum()), "l1_val": float(np.abs(C @ val_prior - obs_shares).sum()),
                "shares_uniform": (C @ uni).tolist(), "shares_val": (C @ val_prior).tolist(),
                "flow": {"A->1D": float(C[D1, A]), "A->A": float(C[A, A]), "A->B": float(C[RANK['B'], A]), "1D->A": float(C[A, D1]), "1D->1D": float(C[D1, D1]),
                         "2D->3D": float(C[RANK['3D'], RANK['2D']]), "B->A": float(C[A, RANK['B']])},
                "balanced_score": float(balanced_mean_over_n(qscore(d, y), y, n)), "decisions": d}

    Z0 = token_pass(X_raw)
    base = summary(probs(R_raw, Z0)); rep["base"] = {k: val for k, val in base.items() if k != "decisions"}
    rep["observed_test_shares"] = obs_shares.tolist()
    print(f"base: score {base['balanced_score']:.4f}; L1 to the test's observed shares: uniform prior {base['l1_uniform']:.3f}, validation prior {base['l1_val']:.3f}; "
          f"flows A->1D {base['flow']['A->1D']:.3f} A->A {base['flow']['A->A']:.3f} 1D->A {base['flow']['1D->A']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    rep["counterfactuals"] = {}

    def report(name, s):
        s = {k: val for k, val in s.items() if k != "decisions"}
        rep["counterfactuals"][name] = s
        du = np.array(s["shares_uniform"]) - np.array(base["shares_uniform"])
        print(f"{name:18s} score {s['balanced_score']:.4f} ({s['balanced_score'] - base['balanced_score']:+.4f})  L1 uniform {s['l1_uniform']:.3f} ({s['l1_uniform'] - base['l1_uniform']:+.3f})  "
              f"L1 val {s['l1_val']:.3f} ({s['l1_val'] - base['l1_val']:+.3f})  A->1D {s['flow']['A->1D']:.3f} ({s['flow']['A->1D'] - base['flow']['A->1D']:+.3f})  "
              f"1D->A {s['flow']['1D->A']:.3f}  A->B {s['flow']['A->B']:.3f}  d shares (uniform prior) " + " ".join(f"{lv} {x * 100:+.1f}" for lv, x in zip(LEVELS, du))
              + f"  ({time.time() - t0:.0f} s)", flush=True)

    for blk, cols in R_BLOCKS.items():
        cols = list(cols); R_cf = R_raw.copy(); R_cf[:, cols] += d_R[cols].astype(np.float32)
        report(blk, summary(probs(R_cf, Z0)))
    for blk, (cols, role) in TOKEN_BLOCKS.items():
        X_mod = X_raw.copy()
        for rl, rows in (("target", tgt_rows), ("opp", opp_rows)):
            if role in (rl, "both"):
                for i in range(0, len(rows), 500_000):
                    rr = rows[i:i + 500_000]
                    X_mod[rr, cols] = (X_mod[rr, cols].astype(np.float32) + d_tok[rl][cols]).astype(np.float16)
        report(blk, summary(probs(R_raw, token_pass(X_mod)))); del X_mod

    # 3. direct check
    rep["direct"] = {}
    for col_name, col in (("hd_sente", 60), ("opp_pmi_mean", 57)):
        x = R_raw[:, col].astype(np.float64)
        for c, cname in ((A, "A"), (D1, "1D")):
            sel = y == c; cuts = np.quantile(x[sel], [1 / 3, 2 / 3]); tert = np.digitize(x[sel], cuts)
            d_sel = base["decisions"][sel]
            rows_out = []
            for t in range(3):
                dd = d_sel[tert == t]
                rows_out.append({"tertile": t, "n": int(len(dd)), "x_mean": float(x[sel][tert == t].mean()),
                                 **{f"decide_{lv}": float((dd == RANK[lv]).mean()) for lv in ("B", "A", "1D", "2D")}})
            share_test_above = float((R_test[:, col] > cuts[1]).mean())
            rep["direct"][f"{col_name}|{cname}"] = {"tertiles": rows_out, "test_share_above_upper_tertile_cut": share_test_above, "cuts": cuts.tolist()}
            print(f"true {cname} by {col_name} tertile: " + " | ".join(f"x {r['x_mean']:.3f}: B {r['decide_B']:.2f} A {r['decide_A']:.2f} 1D {r['decide_1D']:.2f} 2D {r['decide_2D']:.2f}" for r in rows_out)
                  + f"   (test questions above the upper cut {cuts[1]:.3f}: {share_test_above:.2f})", flush=True)
    Path(ROOT / f"results/trunk_seq/es/m51_shift_val{v}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"written results/trunk_seq/es/m51_shift_val{v}.json ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
