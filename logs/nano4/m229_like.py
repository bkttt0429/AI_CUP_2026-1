#!/usr/bin/env python
"""M229 (2026-11-18), report only: the Public readings as a test of the pool's global weights on the test side.

For every setting of results/r_variants/m229_grid.npz (factors on the anchor evidence, the base log posterior and the
look-alike reading; the pooled posterior of the ct112 rule on the 800 test questions) the composite log-likelihood of the
Public readings is computed: every upload is paired with the earlier upload whose Public answers are closest; the observed
change (exact units, within-one units; all solutions compatible with the two truncated scores) gets its exact probability
under independent truths drawn from the setting's posterior (dynamic programme over the changed questions). A second term
scores the exact Public composition (level counts, normal approximation per level).
Printed: the table along each axis, the best settings, the split into the earlier and the later half of the uploads, the
expected and the obtained change per step under the current setting, and how many answers the best settings would change.

  python -I logs/nano4/m229_like.py
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

LEV = led.LEVELS; E1 = led.E1
LATER = [("upload_probe/used_20261117/probe_4D5D_locked.csv", 0.8347546), ("upload_probe/used_20261117/probe_A2B.csv", 0.8363349), ("submission_rank_ct116.csv", 0.8394955), ("upload_probe/probe_2D3D.csv", 0.8347546),
         ("upload_probe/probe_3D4D.csv", 0.8369955), ("upload_probe/probe_C2B.csv", 0.8417364), ("upload_probe/probe_5D4D.csv", 0.8363349), ("submission_rank_ct118d.csv", 0.8546379)]
XY = {(d0, d1): [led.xy(d0, d1, c) for c in range(10)] for d0 in range(10) for d1 in range(10) if d0 != d1}


def pair_dist(P, a0, a1, idx):
    """Distribution [2n+1, 2n+1] of (exact units, within-one units) of the change a0 -> a1 on the questions idx under truths ~ P."""
    n = len(idx); D = np.zeros((2 * n + 1, 2 * n + 1)); D[n, n] = 1.0
    for q in idx:
        sh = {}
        for c, (x, y) in enumerate(XY[(int(a0[q]), int(a1[q]))]):
            sh[(x, y)] = sh.get((x, y), 0.0) + P[q, c]
        N = np.zeros_like(D)
        for (x, y), p in sh.items():
            if p > 0:
                N += p * np.roll(np.roll(D, x, 0), y, 1)
        D = N
    return D, n


def main():
    z = np.load(ROOT / "results/r_variants/m229_grid.npz", allow_pickle=True); grid = [tuple(float(v) for v in g) for g in z["grid"]]; PA = z["P"]; qid = z["qid"].astype(str); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); i0 = grid.index((1.0, 1.0, 1.0))
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *LATER]; ups = [(f, s) for f, s in ups if (ROOT / f).exists() or print(f"   (missing, left out: {f})")]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k])
        if len(idx) == 0:
            continue
        sols = led.decompose(400 * (S[k] - S[p]), len(idx)); pairs.append((k, p, idx, sols))
        if not sols:
            print(f"   no solution for {ups[k][0]} against {ups[p][0]} ({len(idx)} changed, {400 * (S[k] - S[p]):+.4f} points) - left out")
    pairs = [x for x in pairs if x[3]]; half = len(pairs) // 2; comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[l] for l in LEV], float)
    print(f"{len(ups)} uploads, {len(pairs)} pairs (changed questions: median {int(np.median([len(x[2]) for x in pairs]))}, max {max(len(x[2]) for x in pairs)}; pairs with more than one compatible solution: {sum(len(x[3]) > 1 for x in pairs)}); halves split after {ups[pairs[half - 1][0]][0]}")
    CL = np.zeros((len(grid), 3)); per = np.zeros((len(grid), len(pairs)))
    for g in range(len(grid)):
        Pp = PA[g][pi]
        for j, (k, p, idx, sols) in enumerate(pairs):
            D, n = pair_dist(Pp, A[p], A[k], idx); per[g, j] = math.log(max(sum(D[n + x, n + y] for x, y in sols), 1e-300))
        mu = Pp.sum(0); var = (Pp * (1 - Pp)).sum(0); CL[g] = (per[g, :half].sum(), per[g, half:].sum(), float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum()))
    tot = CL[:, 0] + CL[:, 1]; base = tot[i0]
    print(f"\ncurrent setting (1, 1, 1): composite log-likelihood of the pairs {base:.2f} (earlier half {CL[i0, 0]:.2f}, later half {CL[i0, 1]:.2f}); composition term {CL[i0, 2]:.2f}")
    for ax, nm in ((0, "anchor evidence"), (1, "base posterior"), (2, "look-alike reading")):
        print(f"\n== factor on the {nm} (the other two at 1): change of the log-likelihood against the current setting [pairs | earlier half | later half | composition]")
        for g, st in enumerate(grid):
            if all(st[a] == 1.0 for a in range(3) if a != ax):
                print(f"   {st[ax]:5.2f}: {tot[g] - base:+7.2f} | {CL[g, 0] - CL[i0, 0]:+7.2f} | {CL[g, 1] - CL[i0, 1]:+7.2f} | {CL[g, 2] - CL[i0, 2]:+7.2f}")
    name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    MODEL = {"ct40", "ct42", "ct44", "ct64", "ct71", "ct72", "ct73", "ct75n", "ct75s", "ct75u", "ct75s_out", "ct75s_out_up", "ct104", "ct104b", "ct105", "ct106", "ct109", "ct112"}; BAND = {"probe_4D5D_locked", "probe_A2B", "probe_2D3D", "probe_3D4D", "probe_C2B", "probe_5D4D"}
    sub = {"model-driven steps only (new identifiers / pool versions chosen by cross-validation)": [j for j, x in enumerate(pairs) if name(x[0]) in MODEL], "band probes only (bands defined on the posterior alone)": [j for j, x in enumerate(pairs) if name(x[0]) in BAND],
           "the rest (pushes of the Public-mining period, files edited from Public readings)": [j for j, x in enumerate(pairs) if name(x[0]) not in MODEL | BAND]}
    print("\n== added after the first reading (2026-11-18): the anchor-evidence axis by kind of step - a step chosen with Public information is not a fair test of the posterior")
    for nm, js in sub.items():
        print(f"   {nm}: {len(js)} pairs: " + "  ".join(f"{st[0]:g}: {per[g, js].sum() - per[i0, js].sum():+.2f}" for g, st in enumerate(grid) if st[1] == 1.0 and st[2] == 1.0))
    for nm, js in sub.items():
        print(f"   {nm}, look-alike reading axis: " + "  ".join(f"{st[2]:g}: {per[g, js].sum() - per[i0, js].sum():+.2f}" for g, st in enumerate(grid) if st[0] == 1.0 and st[1] == 1.0) + " | base axis: " + "  ".join(f"{st[1]:g}: {per[g, js].sum() - per[i0, js].sum():+.2f}" for g, st in enumerate(grid) if st[0] == 1.0 and st[2] == 1.0))
    order = np.argsort(-tot)[:6]; print("\n== best settings of the grid by the pairs (anchor, base, reading): " + "; ".join(f"{grid[g]} {tot[g] - base:+.2f} (composition {CL[g, 2] - CL[i0, 2]:+.2f})" for g in order))
    print("\n== expected and obtained change per step under the current setting (points): expected | sd | obtained | changed | file <- parent")
    P0 = PA[i0][pi]; Sm = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); se = so = 0.0; zs = []
    for k, p, idx, sols in pairs:
        dq = np.array([[Sm[A[k][q], c] - Sm[A[p][q], c] for c in range(10)] for q in idx]); m = (P0[idx] * dq).sum(1); v = (P0[idx] * dq ** 2).sum(1) - m ** 2; ob = 400 * (S[k] - S[p]); se += m.sum(); so += ob; zs.append((ob - m.sum()) / math.sqrt(max(v.sum(), 1e-9)))
        print(f"   {m.sum():+6.2f} | {math.sqrt(v.sum()):4.2f} | {ob:+6.2f} | {len(idx):3d} | {ups[k][0].split('/')[-1]} <- {ups[p][0].split('/')[-1]}")
    zs = np.array(zs); print(f"   sums: expected {se:+.2f}, obtained {so:+.2f}; standardised residuals: mean {zs.mean():+.2f}, sd {zs.std():.2f} (calibrated = 0 and 1), share above zero {float((zs > 0).mean()):.2f}")
    lev = np.array(LEV); dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(PA[i0]); print("\n== answers changed against the ct112 rule (Public / Private) by setting")
    for g in sorted(set([*order.tolist(), *[g for g, st in enumerate(grid) if st[1] == 1.0 and st[2] == 1.0]]), key=lambda g: grid[g]):
        d = dec(PA[g]); ch = d != d0; print(f"   {grid[g]}: {int((ch & pub).sum())} / {int((ch & ~pub).sum())}" + ("   kinds (Private): " + ", ".join(f"{lev[a]}>{lev[b_]}" for a, b_ in zip(d0[ch & ~pub], d[ch & ~pub]))[:160] if (ch & ~pub).any() else ""))
    (ROOT / "results/r_variants/m229_like.json").write_text(json.dumps({"grid": grid, "pairs": tot.tolist(), "halves": CL[:, :2].tolist(), "composition": CL[:, 2].tolist(), "n_pairs": len(pairs)}, indent=1)); print("\nwritten results/r_variants/m229_like.json")


if __name__ == "__main__":
    main()
