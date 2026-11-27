#!/usr/bin/env python
"""M239 (2026-11-18), report only: the Public readings as a test of pool coefficients fitted on a test-like population (a copy of m237_like.py).
For every setting stored in results/r_variants/m239_grid.npz (the pooled posterior of the ct112 rule on the 800 test
questions: current centring, level-balanced centring, variants) the composite log-likelihood of the Public readings is
computed exactly as in m229_like.py, plus the term of the exact Public composition. The pairs are split BEFORE reading into
model-driven steps, posterior-band probes (both fair) and the rest (steps chosen with Public information: not fair).

  python -I logs/nano4/m239_like.py
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
    z = np.load(ROOT / "results/r_variants/m239_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; PA = z["P"]; qid = z["qid"].astype(str); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); i0 = 0
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
    per = np.zeros((len(names), len(pairs))); co = np.zeros(len(names)); mass = []
    for g in range(len(names)):
        Pp = PA[g][pi]
        for j, (k, p, idx, sols) in enumerate(pairs):
            D, n = ml.pair_dist(Pp, A[p], A[k], idx); per[g, j] = math.log(max(sum(D[n + x, n + y] for x, y in sols), 1e-300))
        mu = Pp.sum(0); var = (Pp * (1 - Pp)).sum(0); co[g] = float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum()); mass.append(mu - cnt)
    d = lambda g, js: float(per[g, js].sum() - per[i0, js].sum())  # noqa: E731
    lev = np.array(LEV); dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(PA[i0]); out = {}
    print("\n== change of the log-likelihood against the current setting  [FAIR = model-driven + band probes + composition | model-driven | band probes | composition | the rest (not fair)]")
    for g, nm in enumerate(names):
        fair = d(g, jm) + d(g, jb) + float(co[g] - co[i0]); dd = dec(PA[g]); ch = dd != d0
        print(f"   {nm:30s}: {fair:+7.2f} | {d(g, jm):+7.2f} | {d(g, jb):+7.2f} | {co[g] - co[i0]:+7.2f} | {d(g, jr):+7.2f}   posterior mass minus the true Public counts: " + " ".join(f"{lv} {x:+.1f}" for lv, x in zip(LEV, mass[g])) + f"   | mean level - truth {float((PA[g][pi] @ np.arange(10.0)).mean() - (cnt * np.arange(10.0)).sum() / cnt.sum()):+.3f}")
        out[nm] = {"fair": fair, "model": d(g, jm), "band": d(g, jb), "composition": float(co[g] - co[i0]), "rest": d(g, jr), "changed_public": int((ch & pub).sum()), "changed_private": int((ch & ~pub).sum())}
    print("\n== answers changed against the ct112 rule (locked questions and guards as in the rule): Public / Private, kinds")
    for g, nm in enumerate(names[1:], 1):
        dd = dec(PA[g]); ch = dd != d0; kinds = {}
        for a, b_ in zip(d0[ch], dd[ch]):
            kinds[f"{lev[a]}>{lev[b_]}"] = kinds.get(f"{lev[a]}>{lev[b_]}", 0) + 1
        print(f"   {nm:30s}: {int((ch & pub).sum())} / {int((ch & ~pub).sum())}   " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    (ROOT / "results/r_variants/m239_like.json").write_text(json.dumps(out, indent=1)); print("\nwritten results/r_variants/m239_like.json")


if __name__ == "__main__":
    main()
