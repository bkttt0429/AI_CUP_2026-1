#!/usr/bin/env python
"""Merge a partial re-scoring over a scored directory, fold by fold (issue #42, the handicap board fix).

The HD >= 2 games (3.6%) were scored on boards missing their handicap stones and, per Q13, with the recorded (reversed)
colours. After the fix in aicup.sgf.setup_and_moves they are scored again, only them, with `--game-keys` of
scripts/score_folds.py / scripts/l1b_score.py into a PATCH directory; this script writes OUT = BASE with those games' table
rows (n_moves, sum_*) and per-move arrays replaced (aicup.scoring.patch_fold), every other game and file unchanged, so
every downstream script runs on OUT exactly as on BASE. Each fold's out/fold{F}/patch_meta.json and --summary record what
changed (moves before/after, mean |Δ| and mean Δ of every sum column, the patched keys).

  python -I scripts/patch_scored.py --base results/scores --patch results/scores_hdpatch --out results/scores_hd --folds 0 1 2 3 4 \\
      --summary results/data/patch_scores_hd.json
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.scoring import patch_fold  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="the scored directory (fold{F} subdirectories)")
    ap.add_argument("--patch", required=True, help="the re-scoring of some of its games, same layout")
    ap.add_argument("--out", required=True, help="written: base with the patched games replaced")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--summary", default=None, help="JSON with every fold's summary")
    a = ap.parse_args(argv)
    t0, out = time.time(), {}
    for f in a.folds:
        try:
            s = patch_fold(Path(a.base) / f"fold{f}", Path(a.patch) / f"fold{f}", Path(a.out) / f"fold{f}")
        except (FileNotFoundError, FileExistsError, ValueError) as e:
            print(f"fold {f}: {e}", file=sys.stderr)
            return 2
        out[str(f)] = s
        top = sorted(s["sum_mean_abs_change"].items(), key=lambda kv: -kv[1])[:3]
        print(f"fold {f}: {s['patched']} of {s['games']} games patched, target moves {s['n_moves_before']} -> {s['n_moves_after']}; "
              f"largest mean |change| " + " ".join(f"{c} {v:.3f}" for c, v in top) + f"  ({time.time() - t0:.0f} s)", flush=True)
    if a.summary:
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(to_json({"folds": out, "base": a.base, "patch": a.patch, "out": a.out, **environment("cpu")}, indent=1) + "\n")
        print(f"wrote {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
