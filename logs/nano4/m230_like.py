#!/usr/bin/env python
"""M230 (2026-11-18), report only: which identifier's pooled test posterior explains the Public readings best?
For every results/r_variants/m230_post_<id>.npz: the composite log-likelihood of the upload pairs (as m229_like.py), split into
the kinds of steps fixed BEFORE this reading (M229's lesson): model-driven steps (18), posterior-band probes (6), the rest
(Public-informed, not a fair test). Printed as differences against the first identifier (the current one).

  python -I logs/nano4/m230_like.py SQ4 SZ3 SY3 SV1 SC2 SC3 SW1 SW2
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m229_like as ml  # noqa: E402

MODEL = ["ct40", "ct42", "ct44", "ct64", "ct71", "ct72", "ct73", "ct75n", "ct75s", "ct75u", "ct75s_out", "ct75s_out_up", "ct104", "ct104b", "ct105", "ct106", "ct109", "ct112"]
BAND = ["probe_4D5D_locked", "probe_A2B", "probe_2D3D", "probe_3D4D", "probe_C2B", "probe_5D4D"]


def main():
    ids = sys.argv[1:]; Z = {i: np.load(ROOT / f"results/r_variants/m230_post_{i}.npz", allow_pickle=True) for i in ids}; qid = Z[ids[0]]["qid"].astype(str); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub)
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *ml.LATER]; ups = [(f, s) for f, s in ups if (ROOT / f).exists()]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []; name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k]); sols = led.decompose(400 * (S[k] - S[p]), len(idx)) if len(idx) else []
        if sols:
            pairs.append((k, p, idx, sols))
    kind = np.array([0 if name(x[0]) in MODEL else 1 if name(x[0]) in BAND else 2 for x in pairs]); mj = np.flatnonzero(kind == 0); h1, h2 = mj[:len(mj) // 2], mj[len(mj) // 2:]
    print(f"{len(pairs)} pairs: model-driven {int((kind == 0).sum())} (earlier half ends with {name(pairs[h1[-1]][0])}), band probes {int((kind == 1).sum())}, the rest {int((kind == 2).sum())}")
    per = {}; comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[l] for l in led.LEVELS], float); cterm = {}
    for i in ids:
        P = Z[i]["P"][pi]; per[i] = np.array([math.log(max(sum(ml.pair_dist(P, A[p], A[k], idx)[0][len(idx) + x, len(idx) + y] for x, y in sols), 1e-300)) for k, p, idx, sols in pairs])
        mu = P.sum(0); var = (P * (1 - P)).sum(0); cterm[i] = float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum())
    r = ids[0]; d0 = Z[r]["fin"]; print(f"\nreference {r}: log-likelihood model-driven {per[r][kind == 0].sum():.2f}, band probes {per[r][kind == 1].sum():.2f}, the rest {per[r][kind == 2].sum():.2f}; composition {cterm[r]:.2f}")
    print("\nidentifier: difference against the reference [fair = model-driven + band probes | model-driven | its earlier half | its later half | band probes | the rest | composition]; answers that differ from the reference (Public / Private)")
    out = {}
    for i in ids[1:]:
        dd = per[i] - per[r]; f = dd[kind != 2].sum(); ch = Z[i]["fin"] != d0
        print(f"   {i:5s}: {f:+6.2f} | {dd[kind == 0].sum():+6.2f} | {dd[h1].sum():+6.2f} | {dd[h2].sum():+6.2f} | {dd[kind == 1].sum():+6.2f} | {dd[kind == 2].sum():+6.2f} | {cterm[i] - cterm[r]:+6.2f}   {int((ch & pub).sum())} / {int((ch & ~pub).sum())}")
        out[i] = {"fair": float(f), "model": float(dd[kind == 0].sum()), "halves": [float(dd[h1].sum()), float(dd[h2].sum())], "band": float(dd[kind == 1].sum()), "rest": float(dd[kind == 2].sum()), "composition": cterm[i] - cterm[r], "per_pair": dd.tolist()}
    best = max(out, key=lambda i: out[i]["fair"]); dd = per[best] - per[r]; print(f"\nlargest fair difference: {best}; its per-pair differences (model-driven, then band probes):")
    print("   " + "  ".join(f"{name(pairs[j][0])} {dd[j]:+.2f}" for j in np.flatnonzero(kind != 2)))
    (ROOT / "results/r_variants/m230_like.json").write_text(json.dumps({"reference": r, "pairs": [name(x[0]) for x in pairs], "kind": kind.tolist(), "out": out}, indent=1)); print("written results/r_variants/m230_like.json")


if __name__ == "__main__":
    main()
