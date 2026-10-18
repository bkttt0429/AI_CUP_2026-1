#!/usr/bin/env python
"""One-shot fold-0 evaluation of the ladder rung chosen from the folds 1-4 CV (aicup.ladder_fold0).

Run it once, and only after the rung has been chosen from the CV summary (--chosen-from, default
results/ladder/cv_summary.json). It retrains the rung on folds 1-4 (a static rung once; a set rung
once per --seeds seed on the 4-fold cross-fitted base, scores averaged over seeds per question),
scores fold 0's pilot questions and writes --out (default results/ladder/fold0_eval.json): the
headline with its player-bootstrap CI and the paired differences vs the same retrain's base and vs
the production model (--rank-model, default results/models/rank_model.json).

Exit 2 before any data is read when --out exists (fold 0 was evaluated already), when the summary
is missing, or when it has no CV results for the configuration (--rung with --per-player and
--no-counts as given to scripts/ladder_cv.py).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import DEFAULT_PER_PLAYER, RUNGS  # noqa: E402
from aicup.ladder_fold0 import CHOSEN_FROM, OUT, RANK_MODEL, VECTORS, Refused, run  # noqa: E402


def _ci(c):
    return f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rung", required=True, choices=list(RUNGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="set rungs only; a static rung is fitted once")
    ap.add_argument("--per-player", type=int, default=DEFAULT_PER_PLAYER)
    ap.add_argument("--no-counts", action="store_true", help="game-length ablation (ladder_data counts=False)")
    ap.add_argument("--chosen-from", default=str(CHOSEN_FROM), help="the CV summary the rung was chosen from")
    ap.add_argument("--vectors", default=str(VECTORS))
    ap.add_argument("--rank-model", default=str(RANK_MODEL), help="the production model to compare with")
    ap.add_argument("--out", default=str(OUT), help="refused when it exists: fold 0 is evaluated once")
    ap.add_argument("--device", default="cuda", help="device of the set models")
    a = ap.parse_args(argv)
    if len(set(a.seeds)) != len(a.seeds):
        ap.error(f"--seeds must be distinct, got {a.seeds}")
    try:
        r = run(a.rung, out=a.out, chosen_from=a.chosen_from, vectors_dir=a.vectors, rank_model=a.rank_model, seeds=a.seeds,
                per_player=a.per_player, counts=not a.no_counts, device=a.device)
    except Refused as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 2
    h = r["headline"]
    by_n = "  ".join(f"n={m} {v:.4f}" for m, v in h["balanced_by_n"].items())
    print(f"{r['cv_label']} on fold 0: {h['balanced_mean_over_n']:.4f} [{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}]  {by_n}")
    print(f"  vs base:       {_ci(r['vs_base'])}")
    print(f"  vs production: {_ci(r['vs_production'])}  (production {r['production']['balanced_mean_over_n']:.4f})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
