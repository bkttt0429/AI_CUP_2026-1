#!/usr/bin/env python
"""Leave-one-fold-out CV of the L3 aggregator ladder on folds 1-4, then the pooled summary and gates.

For every requested rung and seed (static rungs: seed 0 only, they are deterministic; set rungs:
--seeds, default 0 1 2) runs aicup.ladder.run_rung_cv and writes
<out>/{rung}[_pp{N}][_nocounts]_s{seed}.npz; existing files are skipped, so a rerun resumes. The set
rungs' base logits are cached in <out>/base_v{v}[_pp{N}][_nocounts].npz. At the end every file in
<out> is summarised (aicup.ladder.summarize, gates) into <out>/../cv_summary.json, by default
results/ladder/cv_summary.json.

The iterative rungs (r4_*, r5_*) run only when the summary of <out> holds the r3 rungs and their
controls and G_set passes; otherwise the reason is printed and the exit code is 2 (--force overrides).
Folds containing 0 also exit with 2: fold 0 is evaluated once, by scripts/ladder_eval_fold0.py.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import (CV_FOLDS, DEFAULT_PER_PLAYER, RUNGS, STATIC_RUNGS, cv_filename, g_set_ready, gates,  # noqa: E402
                          load_players, run_rung_cv, save_cv, summarize, to_json)
from aicup.qeval import balanced_mean_over_n  # noqa: E402


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rungs", nargs="+", default=[], choices=list(RUNGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="set rungs only; static rungs use seed 0")
    ap.add_argument("--folds", type=int, nargs="+", default=list(CV_FOLDS))
    ap.add_argument("--per-player", type=int, default=DEFAULT_PER_PLAYER)
    ap.add_argument("--vectors", default=str(ROOT / "results" / "ladder" / "vectors"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ladder" / "cv"))
    ap.add_argument("--no-counts", action="store_true", help="game-length ablation (ladder_data counts=False)")
    ap.add_argument("--force", action="store_true", help="run r4/r5 even when G_set does not hold")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--device", default="cuda", help="device of the set models")
    a = ap.parse_args(argv)
    if 0 in a.folds:
        print("refusing --folds with fold 0: fold 0 is evaluated once (scripts/ladder_eval_fold0.py), never in the CV",
              file=sys.stderr)
        return 2
    if not a.rungs and not a.summarize_only:
        ap.error("--rungs is required unless --summarize-only")
    out, counts = Path(a.out), not a.no_counts

    if not a.summarize_only:
        if any(r.startswith(("r4_", "r5_")) for r in a.rungs) and not a.force:
            ok, reason = g_set_ready(summarize(out))
            if not ok:
                print(reason, file=sys.stderr)
                return 2
        for f in a.folds:  # fail before the ~30 s of reading the players
            if not (Path(a.vectors) / f"fold{f}.npz").exists():
                raise FileNotFoundError(f"{Path(a.vectors) / f'fold{f}.npz'} not built (scripts/ladder_build_vectors.py)")
        players = load_players(a.folds)
        for rung in a.rungs:
            for seed in [0] if rung in STATIC_RUNGS else a.seeds:
                path = out / cv_filename(rung, seed, a.per_player, counts)
                if path.exists():
                    print(f"skip {path.name} (exists)", flush=True)
                    continue
                t0 = time.time()
                df = run_rung_cv(rung, a.vectors, folds=a.folds, seed=seed, per_player=a.per_player, counts=counts,
                                 device=a.device, base_cache=out, players=players)
                save_cv(df, path)
                per_fold = "  ".join(f"fold {f} {balanced_mean_over_n(d['score'], d['y'], d['n_games']):.4f}"
                                     for f, d in df.groupby("fold", sort=True))
                pooled = balanced_mean_over_n(df["score"], df["y"], df["n_games"])
                print(f"{path.name}: {per_fold}  pooled {pooled:.4f}  ({time.time() - t0:.0f} s)", flush=True)

    summary = summarize(out)
    summary["gates"] = gates(summary)
    summary_path = out.parent / "cv_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(to_json(summary, indent=1) + "\n")
    for label, r in summary["rungs"].items():
        sd = "" if r["seed_sd"] is None else f"  seed SD {r['seed_sd']:.4f}"
        folds = " ".join(_fmt(x) for x in r["per_fold"].values())
        print(f"  {label:28s} {_fmt(r['balanced_mean_over_n'])}  folds {folds}{sd}")
    for name, g in summary["gates"].items():
        print(f"{name}: {'PASS' if g['pass'] else 'fail'}")
        for key, c in g["comparisons"].items():
            print(f"  {key}: {'pass' if c['pass'] else '; '.join(c['reasons'])}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
