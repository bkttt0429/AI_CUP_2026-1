#!/usr/bin/env python
"""M236 (2026-11-18), report only: the Public readings as a test of the WEIGHT and of a SHIFT of the look-alike reading
on the test side. For every setting of results/r_variants/m236_grid.npz (factor on the look-alike reading, its shift in
levels; the pooled posterior of the ct112 rule on the 800 test questions) the composite log-likelihood of the
Public readings is computed exactly as in m229_like.py (pairs of uploads, exact distribution of the change by a dynamic
programme) plus the term of the exact Public composition. The pairs are split BEFORE reading into model-driven steps,
posterior-band probes (both fair tests of the posterior) and the rest (steps chosen with Public information: not fair).

  python -I logs/nano4/m236_like.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m229_like as ml  # noqa: E402  (pair_dist, LATER)

LEV = led.LEVELS
MODEL = {"ct40", "ct42", "ct44", "ct64", "ct71", "ct72", "ct73", "ct75n", "ct75s", "ct75u", "ct75s_out", "ct75s_out_up", "ct104", "ct104b", "ct105", "ct106", "ct109", "ct112"}
BAND = {"probe_4D5D_locked", "probe_A2B", "probe_2D3D", "probe_3D4D", "probe_C2B", "probe_5D4D"}


def main():
    z = np.load(ROOT / "results/r_variants/m236_grid.npz", allow_pickle=True); grid = [tuple(round(float(v), 3) for v in g) for g in z["grid"]]; PA = z["P"]; qid = z["qid"].astype(str); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); i0 = grid.index((1.0, 0.0))
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *ml.LATER]; ups = [(f, s) for f, s in ups if (ROOT / f).exists() or print(f"   (missing, left out: {f})")]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k])
        if len(idx) == 0:
            continue
        sols = led.decompose(400 * (S[k] - S[p]), len(idx))
        if sols:
            pairs.append((k, p, idx, sols))
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[l] for l in LEV], float)
    name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    jm = [j for j, x in enumerate(pairs) if name(x[0]) in MODEL]; jb = [j for j, x in enumerate(pairs) if name(x[0]) in BAND]; jr = [j for j, x in enumerate(pairs) if name(x[0]) not in MODEL | BAND]
    print(f"{len(ups)} uploads, {len(pairs)} pairs: model-driven {len(jm)}, band probes {len(jb)}, the rest (chosen with Public information) {len(jr)}")
    per = np.zeros((len(grid), len(pairs))); co = np.zeros(len(grid)); mean_lv = np.zeros(len(grid))
    for g in range(len(grid)):
        Pp = PA[g][pi]
        for j, (k, p, idx, sols) in enumerate(pairs):
            D, n = ml.pair_dist(Pp, A[p], A[k], idx); per[g, j] = math.log(max(sum(D[n + x, n + y] for x, y in sols), 1e-300))
        mu = Pp.sum(0); var = (Pp * (1 - Pp)).sum(0); co[g] = float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum()); mean_lv[g] = float((Pp @ np.arange(10.0)).mean())
    true_mean = float((cnt * np.arange(10.0)).sum() / cnt.sum())
    d = lambda g, js: float(per[g, js].sum() - per[i0, js].sum())  # noqa: E731
    fair = lambda g: d(g, jm) + d(g, jb) + float(co[g] - co[i0])  # noqa: E731
    print(f"current setting: pooled posterior's mean level on the Public questions {mean_lv[i0]:.3f}, true mean level {true_mean:.3f} (difference {mean_lv[i0] - true_mean:+.3f})")
    for ax, nm, other in ((0, "factor on the look-alike reading", 0.0), (1, "shift of the look-alike reading", 1.0)):
        print(f"\n== {nm} (the other at its current value): change of the log-likelihood against the current setting  [FAIR = model-driven + band probes + composition | model-driven | band probes | composition | the rest (not fair)]   mean level - truth")
        for g, st in enumerate(grid):
            if st[1 - ax] == other:
                print(f"   {st[ax]:+5.2f}: {fair(g):+7.2f} | {d(g, jm):+7.2f} | {d(g, jb):+7.2f} | {co[g] - co[i0]:+7.2f} | {d(g, jr):+7.2f}   {mean_lv[g] - true_mean:+.3f}")
    order = sorted(range(len(grid)), key=lambda g: -fair(g))[:6]
    print("\n== best settings of the grid by the fair log-likelihood (look-alike factor, look-alike shift): " + "; ".join(f"{grid[g]} {fair(g):+.2f}" for g in order))
    lev = np.array(LEV); dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(PA[i0]); print("\n== answers changed against the ct112 rule (Public / Private) by setting")
    for g in sorted(set([*order, *[g for g, st in enumerate(grid) if st[0] == 1.0 or st[1] == 0.0]]), key=lambda g: grid[g]):
        dd = dec(PA[g]); ch = dd != d0; print(f"   {grid[g]}: {int((ch & pub).sum())} / {int((ch & ~pub).sum())}" + ("   kinds (Private): " + ", ".join(f"{lev[a]}>{lev[b_]}" for a, b_ in zip(d0[ch & ~pub], dd[ch & ~pub]))[:200] if (ch & ~pub).any() else ""))
    (ROOT / "results/r_variants/m236_like.json").write_text(json.dumps({"grid": grid, "fair": [fair(g) for g in range(len(grid))], "model": [d(g, jm) for g in range(len(grid))], "band": [d(g, jb) for g in range(len(grid))], "composition": (co - co[i0]).tolist(), "rest": [d(g, jr) for g in range(len(grid))],
                                                                    "mean_level_minus_truth": (mean_lv - true_mean).tolist()}, indent=1)); print("\nwritten results/r_variants/m236_like.json")


if __name__ == "__main__":
    main()
