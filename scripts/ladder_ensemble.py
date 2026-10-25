#!/usr/bin/env python
"""Probability-mean ensemble of set rungs from their CV files, then the pooled summary (issue #2, item 3).

For each seed (default 0 1 2) reads the members' CV files <out>/{member}_s{seed}.npz (default members
aicup.ladder.ENSEMBLE_MEMBERS: r3_pma r4_xattn_t1 r5_ctm, the three best set rungs of the 2026-10-19
ladder), averages their class probabilities, re-decides every question (the expected-score decision on
the mean probabilities) and writes <out>/{label}_s{seed}.npz (default label ens_top3; see
aicup.ladder.build_ensemble for the attrs). Then the summary of <out> is rewritten exactly as
ladder_cv.py --summarize-only writes it (<out.parent>/<out.name>_summary.json, default
results/ladder/cv_summary.json) and the pre-specified follow-up comparisons are printed with their four
conditions: aicup.ladder.FOLLOWUP_COMPARISONS -- ens_top3, r5_ctm_minmax_pc and r5_ctm_unet, each vs
r5_ctm and vs r3_pma; reported, never gated. Nothing is trained: this runs on the CPU in minutes.

An existing ensemble file is refused unless --overwrite, a missing member file always; both exit 2.
--skip-ensemble only rewrites the summary and prints the reports. --across-seeds RUNG ... adds, per rung,
how one decision on the seed-averaged probabilities (the submission's rule) scores against the CV
statistic of per-seed decisions (aicup.ladder.across_seeds; report only).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):  # `python -I` adds neither; scripts/ holds the sibling ladder_cv
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import (DEFAULT_PER_PLAYER, ENSEMBLE_MEMBERS, ENSEMBLE_RUNG, FOLLOWUP_COMPARISONS, across_seeds,  # noqa: E402
                          build_ensemble, check_comparison)
import ladder_cv  # noqa: E402  (the sibling script: the summary is written exactly as it writes it)


def _fmt(c):
    """One line of a summary comparison: the difference and CI, per seed, per fold, the four conditions."""
    seeds = " ".join(f"{d:+.4f}" for d in c["per_seed"].values())
    folds = " ".join(f"{d:+.4f}" for d in c["per_fold"].values())
    chk = check_comparison(c)
    conds = "all four conditions" if chk["pass"] else ", ".join(f"{k} {'ok' if v else 'no'}" for k, v in chk["conditions"].items())
    return (f"diff {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  seeds {seeds}  folds {folds}  {conds}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--members", nargs="+", default=list(ENSEMBLE_MEMBERS), help="set rungs whose CV files are averaged")
    ap.add_argument("--label", default=ENSEMBLE_RUNG, help="the ensemble's rung label in the files and the summary")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-player", type=int, default=DEFAULT_PER_PLAYER)
    ap.add_argument("--no-counts", action="store_true", help="read and write the _nocounts files")
    ap.add_argument("--out", default=str(ROOT / "results" / "ladder" / "cv"), help="the CV directory (ladder_cv.py --out)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing ensemble files")
    ap.add_argument("--skip-ensemble", action="store_true", help="only rewrite the summary and print the reports")
    ap.add_argument("--across-seeds", nargs="*", default=[], metavar="RUNG",
                    help="report the seed-averaged-probability decision of these rungs against the CV statistic")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.skip_ensemble:
        try:
            written = build_ensemble(out, a.members, a.label, a.seeds, a.per_player, not a.no_counts, a.overwrite)
        except (FileExistsError, FileNotFoundError, ValueError) as e:
            print(e, file=sys.stderr)
            return 2
        for path in written:
            print(f"wrote {path}", flush=True)
    rc = ladder_cv.main(["--summarize-only", "--out", str(out)])
    if rc:
        return rc
    summary = json.loads((out.parent / f"{out.name}_summary.json").read_text())
    print("follow-up comparisons (pre-specified; reported, never gated):")
    for x, y in FOLLOWUP_COMPARISONS:
        key = f"{x} vs {y}"
        c = summary["comparisons"].get(key)
        print(f"  {key}: {_fmt(c) if c is not None else 'no CV results yet'}")
    if a.across_seeds:
        print("seed-averaged probabilities (one decision) vs the CV statistic (per-seed decisions, scores averaged); report only:")
        for rung in a.across_seeds:
            try:
                r = across_seeds(out, rung, a.per_player, not a.no_counts)
            except (KeyError, ValueError) as e:
                print(f"  {rung}: {e}")
                continue
            by_n = " ".join(f"{m}: {d:+.4f}" for m, d in r["by_n"].items())
            print(f"  {r['label']} (seeds {r['seeds']}): {r['score_seedmean_proba']:.4f} vs {r['score_cv']:.4f}, diff {r['diff']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  by n {by_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
