"""Section 4M M51: aggregate m51_shift_val{1..4}.json -- the counterfactual table averaged over the folds, the shift table from fold 1.
  python -I logs/nano4/m51_agg.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="shift", help="shift (m51_shift.py) or shiftr (m51_shift_r.py, within-size R deltas)"); a = ap.parse_args()
    reps = [json.load(open(ROOT / f"results/trunk_seq/es/m51_{a.tag}_val{v}.json")) for v in (1, 2, 3, 4) if (ROOT / f"results/trunk_seq/es/m51_{a.tag}_val{v}.json").exists()]
    print(f"{len(reps)} folds")
    r0 = reps[0]
    sh = r0.get("r_shift_sd_within_n", r0.get("r_shift_sd")); order = sorted(sh, key=lambda k: -abs(sh[k]))
    print(f"R shift (test - validation, validation sd{', within size' if 'r_shift_sd_within_n' in r0 else ''}), fold 1, |shift| >= 0.15: " + ", ".join(f"{k} {sh[k]:+.2f}" for k in order if abs(sh[k]) >= 0.15))
    obs = np.array(r0["observed_test_shares"])
    print("observed test shares (ct9): " + " ".join(f"{lv} {x * 100:.1f}" for lv, x in zip(LEVELS, obs)))
    base_u = np.mean([r["base"]["shares_uniform"] for r in reps], axis=0); base_v = np.mean([r["base"]["shares_val"] for r in reps], axis=0)
    print("base shares, uniform prior:    " + " ".join(f"{lv} {x * 100:.1f}" for lv, x in zip(LEVELS, base_u)) + f"   L1 {np.abs(base_u - obs).sum():.3f}")
    print("base shares, validation prior: " + " ".join(f"{lv} {x * 100:.1f}" for lv, x in zip(LEVELS, base_v)) + f"   L1 {np.abs(base_v - obs).sum():.3f}")
    bs = np.mean([r["base"]["balanced_score"] for r in reps]); bf = {k: np.mean([r["base"]["flow"][k] for r in reps]) for k in r0["base"]["flow"]}
    print(f"base score {bs:.4f}; flows A->1D {bf['A->1D']:.3f} A->A {bf['A->A']:.3f} A->B {bf['A->B']:.3f} 1D->A {bf['1D->A']:.3f} 2D->3D {bf['2D->3D']:.3f}")
    print(f"{'counterfactual':18s} {'score':>8s} {'L1 unif':>8s} {'L1 val':>8s} {'A->1D':>7s} {'A->A':>6s} {'A->B':>6s} {'1D->A':>6s}  d shares under the uniform prior (points)")
    for name in r0["counterfactuals"]:
        cs = [r["counterfactuals"][name] for r in reps if name in r["counterfactuals"]]
        su = np.mean([c["shares_uniform"] for c in cs], axis=0); sv = np.mean([c["shares_val"] for c in cs], axis=0)
        f = {k: np.mean([c["flow"][k] for c in cs]) for k in cs[0]["flow"]}
        sc = np.mean([c["balanced_score"] for c in cs])
        du = (su - base_u) * 100
        print(f"{name:18s} {sc - bs:+8.4f} {np.abs(su - obs).sum() - np.abs(base_u - obs).sum():+8.3f} {np.abs(sv - obs).sum() - np.abs(base_v - obs).sum():+8.3f} "
              f"{f['A->1D'] - bf['A->1D']:+7.3f} {f['A->A'] - bf['A->A']:+6.3f} {f['A->B'] - bf['A->B']:+6.3f} {f['1D->A'] - bf['1D->A']:+6.3f}  "
              + " ".join(f"{lv} {x:+.1f}" for lv, x in zip(LEVELS, du)))
    print("direct check (fold means), decision shares by tertile of the feature among true-A / true-1D questions:")
    for key in r0["direct"]:
        rows = []
        for t in range(3):
            rows.append({k: np.mean([r["direct"][key]["tertiles"][t][k] for r in reps]) for k in ("x_mean", "decide_B", "decide_A", "decide_1D", "decide_2D")})
        above = np.mean([r["direct"][key]["test_share_above_upper_tertile_cut"] for r in reps])
        print(f"  {key:18s} " + " | ".join(f"x {r['x_mean']:.3f}: B {r['decide_B']:.2f} A {r['decide_A']:.2f} 1D {r['decide_1D']:.2f} 2D {r['decide_2D']:.2f}" for r in rows)
              + f"   test share above the upper cut {above:.2f}")


if __name__ == "__main__":
    main()
