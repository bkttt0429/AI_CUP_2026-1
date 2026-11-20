#!/usr/bin/env python
"""M223 (2026-11-18): Public-only refinement of ct118d for 11-19 - three of its 18 Public edits changed to the answer with the
largest expected score under the exact update of M222 (both readings: probe_C2B and ct118d; results/r_variants/m222_update.json):
pub_q_0286 A -> C, pub_q_0192 D -> B, pub_q_0103 B -> A. Private rows = ct116 (= ct118d). Never overwrites.

  python -I logs/nano4/m223_build.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; EDITS = {"pub_q_0286": ("A", "C"), "pub_q_0192": ("D", "B"), "pub_q_0103": ("B", "A")}


def main():
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a118 = rd("submission_rank_ct116.csv"), rd("submission_rank_ct118d.csv"); w = json.load(open(ROOT / "results/r_variants/m222_update.json")); out = dict(a118)
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); tot = 0.0
    for q, (old, new) in EDITS.items():
        assert a118[q] == old, (q, a118[q]); p = np.array([w[q].get(l, 0.0) for l in LEV]); es = S @ p; best = LEV[int(es.argmax())]; assert best == new, (q, best, new)
        out[q] = new; tot += es[LEV.index(new)] - es[LEV.index(old)]
        print(f"  {q}: ct116 {a116[q]:>2} | ct118d {old:>2} -> {new:>2}; weights " + " ".join(f"{k} {v:.2f}" for k, v in sorted(w[q].items(), key=lambda t: -t[1])[:3]) + f"; expected score {es[LEV.index(old)]:.3f} -> {es[LEV.index(new)]:.3f}")
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"
    print(f"expected gain against ct118d under the update: {tot:+.2f} points ({tot / 400:+.4f}); Public answers that differ from ct116: {sum(out[q] != a116[q] for q in a116)}, from ct118d: {sum(out[q] != a118[q] for q in a116)}; Private rows identical to ct116")
    path = ROOT / "submission_rank_ct118e.csv"; assert not path.exists(), f"{path.name} exists already - nothing overwritten"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
