#!/usr/bin/env python
"""Section 4M M164 (2026-11-15): the label-free choice among identifier compositions (design doc M161q, registered rule).

For the tags given (in order of simplicity, simplest first; each built by m164_ident.py): the share of TEST opponents recognised
with a clear margin (largest excess of the opponent side's m15 over the target side's, from m84_<tag>_eval.json). Chosen = the
first tag in the given order whose value is within --tol (0.003) of the best one. No level label is involved.
The table goes to stderr and results/r_variants/m164_selected.json; stdout carries the chosen tag only.

  best=$(python -I logs/nano4/m164_select.py SQ1 SQ2 SQ3)
"""
import argparse
import json
import sys
from pathlib import Path

RV = Path.cwd() / "results/r_variants"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--tol", type=float, default=0.003); a_ = ap.parse_args(); rows = []
    for t in a_.tags:
        f = RV / f"m84_{t}_eval.json"
        if not f.exists():
            print(f"{t}: no evaluation file, skipped", file=sys.stderr); continue
        d = json.loads(f.read_text()); rows.append((t, d["test opponent side"]["largest_excess"], d["validation opponent side"]["largest_excess"], d["heldout"]["fused with the prior"]["top1_rows"], d["beta"], d["sources"]))
    if not rows:
        sys.exit("no identifier to choose from")
    best = max(r[1] for r in rows); chosen = next(r for r in rows if r[1] >= best - a_.tol)
    for t, te, va, h, beta, src in rows:
        print(f"{t:6s} test-side largest excess {te:.4f} | validation side {va:.4f} | held-out top-1 by rows (with the prior) {h:.3f} | beta {beta:.2f} | block + {'+'.join(src)}" + ("   <- chosen" if t == chosen[0] else ""), file=sys.stderr)
    (RV / "m164_selected.json").write_text(json.dumps({"chosen": chosen[0], "tol": a_.tol, "rows": [{"tag": r[0], "test": r[1], "validation": r[2], "heldout_rows": r[3], "beta": r[4], "sources": r[5]} for r in rows]}, indent=1))
    print(chosen[0])


if __name__ == "__main__":
    main()
