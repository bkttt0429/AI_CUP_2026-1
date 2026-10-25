#!/usr/bin/env python
"""The L1b early signal: per-rank per-move NLL and adjacent-rank AUC, L1b heads against Human SL on the same moves.

Plan 2026-10-25-l1b-rank-heads, Task 6 (decision point 1). --l1b is scripts/l1b_score.py's output, --hsl is
results/scores (the 18 Human SL profiles of the same evaluation games); only folds the heads never saw
belong here (model A: folds 3 4; model B: folds 1 2). Prints the two tables of aicup.l1beval.format_report
and writes the JSON report to --out.

  python -I scripts/l1b_early_signal.py --l1b results/l1b/scores --hsl results/scores --folds 3 4 --out results/l1b/early_signal_A.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.l1beval import early_signal, format_report  # noqa: E402
from aicup.ladder import environment, to_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", required=True, help="l1b_score.py output directory (fold{F}/ inside)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"), help="Human SL scores of the same games")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="JSON report")
    a = ap.parse_args(argv)
    try:
        res = early_signal(a.l1b, a.hsl, a.folds)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    print(format_report(res))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json({**res, **environment("cpu")}, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
