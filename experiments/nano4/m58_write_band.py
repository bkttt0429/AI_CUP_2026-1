"""Section 4M M58: a probe that flips one A|1D odds band. ct13 + every question ct13 decides 1D whose ct9 odds P(A)/P(1D) lies in [LO, HI)
-> A (public and private alike, like ct13's own rule). ct29 = the band [1/2.6, 1/2): public 4, private 9 questions.
  python -I logs/nano4/m58_write_band.py --lo 0.384615 --hi 0.5 --out submission_rank_ct29.csv
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.submission import validate_submission, write_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lo", type=float, required=True); ap.add_argument("--hi", type=float, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id")
    base = pd.read_csv(ROOT / "submission_rank_ct13.csv", dtype=str)
    d = base.set_index("question_id")["pred_rank"].copy()
    odds = pr["p_A"] / pr["p_1D"]
    flip = [q for q in d.index if d[q] == "1D" and a.lo <= odds[q] < a.hi]
    d[flip] = "A"
    out = ROOT / a.out
    write_submission(list(zip(base["question_id"], d.reindex(base["question_id"]))), out)
    problems = validate_submission(out, base["question_id"].tolist(), 800, {"pub_": 400, "priv_": 400})
    pub = [q for q in flip if q.startswith("pub")]
    print(f"{out.name}: 1D->A for odds in [{a.lo:.4f}, {a.hi:.4f}): public {len(pub)} {pub}, private {len(flip) - len(pub)}; problems {problems or 'none'}; "
          f"md5 {hashlib.md5(out.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
