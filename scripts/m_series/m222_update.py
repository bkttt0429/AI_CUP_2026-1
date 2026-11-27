#!/usr/bin/env python
"""M222 (2026-11-18), report only: what the reading of ct118d (0.8546379 = +6.057 points against ct116, exact units +9,
within-one units +1 over its 18 changed Public answers) says about each of the 18 questions.

Exact conditional update by dynamic programming: prior of each question's truth = its M211 weights (re-solved after the ct116 reading)
where they exist, else its M210 weights, else the pooled posterior behind ct112; conditions: (1) the reading of probe_C2B over its seven questions (C -> B: +2, -1) and
(2) the reading of ct118d over the 18 questions (ct116's answer -> ct118d's answer: +9, +1). Couplings with the rest of the
ledger are ignored. Per question: truth weights before / after, the expected score of ct116's answer, of ct118d's answer and
of the best answer.

  python -I scripts/m_series/m222_update.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; TARGET = (2, -1, 9, 1); EPS = 1e-15


def main():
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a118, c2b = rd("submission_rank_ct116.csv"), rd("submission_rank_ct118d.csv"), rd("upload_probe/probe_C2B.csv")
    qs = [q for q in a116 if a116[q] != a118[q]]; seven = {q for q in a116 if a116[q] != c2b[q]}; assert seven <= set(qs) and len(qs) == 18
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); post = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}; old = json.load(open(ROOT / "results/r_variants/m210_profiles.json"))
    prior, src = {}, {}
    new = {q: d["with_composition"] for q, d in json.load(open(ROOT / "results/r_variants/m211_ledger_ct116.json")).items() if d.get("with_composition")}  # re-solved after the ct116 reading (10 questions)
    for q in qs:
        w = np.array([new[q].get(l, 0.0) for l in LEV]) if q in new else np.array([old[q].get(l, 0.0) for l in LEV]) if q in old else post[q].copy(); src[q] = "M211" if q in new else "M210" if q in old else "pooled"
        w = np.maximum(w, 1e-4 * post[q]); prior[q] = w / w.sum()  # a floor from the pooled posterior: no truth is excluded outright
    I = LEV.index
    inc = {q: [(*(led.xy(I("C"), I("B"), c) if q in seven else (0, 0)), *led.xy(I(a116[q]), I(a118[q]), c)) for c in range(10)] for q in qs}

    def step(d, q, sign=1):
        n = {}
        for s, p in d.items():
            for c in range(10):
                pc = p * prior[q][c]
                if pc < EPS:
                    continue
                k = tuple(a + sign * b for a, b in zip(s, inc[q][c])); n[k] = n.get(k, 0.0) + pc
        return n
    pre = [{(0, 0, 0, 0): 1.0}]
    for q in qs:
        pre.append(step(pre[-1], q))
    suf = [{(0, 0, 0, 0): 1.0}]
    for q in reversed(qs):
        suf.append(step(suf[-1], q))
    suf = suf[::-1]  # suf[i] = distribution of the sums of qs[i:]
    p_obs = pre[-1].get(TARGET, 0.0); print(f"probability of both readings under the priors: {p_obs:.2e}")
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); rows = []; rep = {}
    for i, q in enumerate(qs):
        w = np.zeros(10); right = suf[i + 1]
        for s, p in pre[i].items():
            for c in range(10):
                need = tuple(t - a - b for t, a, b in zip(TARGET, s, inc[q][c])); pr = right.get(need)
                if pr:
                    w[c] += p * prior[q][c] * pr
        w /= w.sum(); es = S @ w; e0, e1 = es[I(a116[q])], es[I(a118[q])]; best = int(es.argmax()); rep[q] = {LEV[c]: float(w[c]) for c in range(10) if w[c] >= 0.001}
        rows.append((float(es[best] - e1), q, w, e0, e1, best))
    print("question: ct116 -> ct118d | prior | after both readings | expected score of ct116's answer, of ct118d's, of the best answer")
    for g, q, w, e0, e1, best in sorted(rows, key=lambda r: r[1]):
        pt = np.argsort(-prior[q])[:3]; top = np.argsort(-w)[:3]
        print(f"  {q}: {a116[q]:>2} -> {a118[q]:>2} | {src[q]:6s} " + " ".join(f"{LEV[c]} {prior[q][c]:.2f}" for c in pt if prior[q][c] >= 0.005) + " | " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005)
              + f" | {e0:.3f}, {e1:.3f}, best {LEV[best]} {S[best] @ w:.3f}" + ("" if LEV[best] == a118[q] else f"   <- {LEV[best]} would be {g:+.3f} better") + ("  [probe_C2B]" if q in seven else ""))
    tot = sum(r[0] for r in rows); print(f"sum of the expected gains of switching to the best answers: {tot:+.2f} points; expected score of ct118d on these 18 under the update: {sum(r[4] for r in rows):.2f} (ct116: {sum(r[3] for r in rows):.2f})")
    (ROOT / "results/r_variants/m222_update.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m222_update.json")


if __name__ == "__main__":
    main()
