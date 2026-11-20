#!/usr/bin/env python
"""M214 (2026-11-17), exploration for the next Public probes: for each pair of adjacent levels (a -> b), the questions
answered a by the final file (ct116) ranked by log P(b) - log P(a) of the pooled posterior, on both halves.

Only the stored pooled posterior and the answer files are used (no ledger, no reading of any upload): a band defined here
by a threshold on the ratio is the same rule on Public and on Private, so that a Public probe of the Public band says
something about the Private band. Printed per pair: how many questions pass each threshold on each half, and the top of
the list. Also: the level counts of ct116's answers on both halves against the exact Public composition.

  python -I logs/nano4/m214_bands.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137f_build2 as mf7  # noqa: E402

LEV = led.LEVELS; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])
PAIRS = [("1D", "A"), ("D", "C"), ("C", "B"), ("C", "D"), ("B", "C"), ("B", "A"), ("A", "B"), ("A", "1D"), ("2D", "1D"), ("2D", "3D"), ("3D", "2D"), ("3D", "4D"), ("4D", "3D"), ("4D", "5D"), ("5D", "4D"), ("5D", "6D"), ("6D", "5D")]
TAUS = (1.0, 0.8, 0.65, 0.5, 0.4)


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = z["learned"].astype(float); P = P / P.sum(1, keepdims=True); Pb = z["base"].astype(float); Pb = Pb / Pb.sum(1, keepdims=True); pub = np.char.startswith(qid, "pub")
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112, c116 = (sub(f) for f in ("ct9", "ct28", "ct106", "ct112", "ct116")); locked = c28 != c9; D4, D5 = LEV.index("4D"), LEV.index("5D"); lock5 = locked & (c9 == D4) & (c28 == D5); plain = (P @ S.T).argmax(1)
    assert (mf7.guard2(np.where(locked, c28, plain), c106) == c112).all(), "ct112 is not reproduced"; pipe = mf7.guard2(np.where(locked & ~lock5, c28, plain), c106); lockA = locked & ~lock5
    d = np.flatnonzero(pipe != c116); print(f"pipeline with the 4D->5D lock removed against ct116: {len(d)} differences (Public {int(pub[d].sum())}, Private {int((~pub[d]).sum())}): " + ", ".join(f"{qid[i]} {LEV[pipe[i]]}->{LEV[c116[i]]}" for i in d))
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]
    print("level                " + " ".join(f"{l:>4}" for l in LEV)); print("Public truth         " + " ".join(f"{int(comp[l]):4d}" for l in LEV))
    for nm, m in (("Public", pub), ("Private", ~pub)):
        print(f"{nm:8s}ct116 answers " + " ".join(f"{int((c116[m] == k).sum()):4d}" for k in range(10))); print(f"{nm:8s}pooled mass   " + " ".join(f"{P[m][:, k].sum():4.0f}" for k in range(10))); print(f"{nm:8s}plain answers " + " ".join(f"{int((plain[m] == k).sum()):4d}" for k in range(10)))
    print(f"locked 1D->A answers still in force: Public {int((lockA & pub).sum())}, Private {int((lockA & ~pub).sum())}; questions held by a guard (pipeline answer != plain decision, not locked): Public {int((~locked & (pipe != plain) & pub).sum())}, Private {int((~locked & (pipe != plain) & ~pub).sum())}")
    out = {}
    for a_, b_ in PAIRS:
        a, b = LEV.index(a_), LEV.index(b_); cand = (c116 == a) & (pipe == a) & ~locked; r = P[:, b] / np.maximum(P[:, a], 1e-9)
        line = f"\n== {a_} -> {b_}: answered {a_} (not locked): Public {int((cand & pub).sum())}, Private {int((cand & ~pub).sum())}; passing P({b_})/P({a_}) >= tau: " + "; ".join(f"{t}: {int((cand & pub & (r >= t)).sum())} / {int((cand & ~pub & (r >= t)).sum())}" for t in TAUS) + "  (Public / Private)"
        print(line); out[f"{a_}->{b_}"] = {}
        for nm, m in (("Public", pub), ("Private", ~pub)):
            idx = np.flatnonzero(cand & m); idx = idx[np.argsort(-r[idx])][:12]; out[f"{a_}->{b_}"][nm] = [(str(qid[i]), float(r[i])) for i in idx]
            for i in idx:
                if r[i] < 0.3:
                    break
                print(f"   {nm:7s} {qid[i]}: ratio {r[i]:5.2f}  pooled {a_} {P[i, a]:.2f} / {b_} {P[i, b]:.2f}; base {a_} {Pb[i, a]:.2f} / {b_} {Pb[i, b]:.2f}; plain {LEV[plain[i]]}{'  HELD by a guard' if plain[i] != pipe[i] else ''}; ct106 {LEV[c106[i]]}")
    (ROOT / "results/r_variants/m214_bands.json").write_text(json.dumps(out, indent=1)); print("\nwritten results/r_variants/m214_bands.json")


if __name__ == "__main__":
    main()
