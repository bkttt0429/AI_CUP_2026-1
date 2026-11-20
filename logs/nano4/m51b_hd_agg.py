"""Section 4M M51b check 2: fold means of m51b_hd_subset_val{1..4}.json.
  python -I logs/nano4/m51b_hd_agg.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()


def main():
    reps = [json.load(open(p)) for p in sorted(ROOT.glob("results/trunk_seq/es/m51b_hd_subset_val*.json"))]
    print(f"{len(reps)} folds; HD >= 2 questions " + " ".join(f"fold {r['val_fold']} {r['hd2_questions']}/{r['questions']}" for r in reps))
    for name in ("all", "hd_free", "hd2"):
        rs = [r["subsets"][name] for r in reps]
        line = f"{name:8s} questions {sum(r['questions'] for r in rs):6d}: intact {np.mean([r['score_intact'] for r in rs]):.4f}"
        for alt in ("permuted", "zeroed"):
            d = [r[f'drop_{alt}']['diff'] for r in rs]; lo = [r[f'drop_{alt}']['ci95'][0] for r in rs]; hi = [r[f'drop_{alt}']['ci95'][1] for r in rs]
            line += f"; HD {alt}: drop {np.mean(d):+.4f} (folds " + " ".join(f"{x:+.4f}" for x in d) + f"; mean CI [{np.mean(lo):+.4f}, {np.mean(hi):+.4f}])"
        f0 = {k: np.mean([r["flows_intact"][k] for r in rs]) for k in ("A->1D", "1D->A", "A->A")}; f1 = {k: np.mean([r["flows_zeroed"][k] for r in rs]) for k in f0}
        line += f"; flows intact/zeroed A->1D {f0['A->1D']:.3f}/{f1['A->1D']:.3f} 1D->A {f0['1D->A']:.3f}/{f1['1D->A']:.3f} A->A {f0['A->A']:.3f}/{f1['A->A']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
