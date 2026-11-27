#!/usr/bin/env python
"""M220 fallback (2026-11-18): the Public-only file ct118 without the full re-profile (after the four readings of 11-18 the
profile solves of m220_resolve.py no longer return within their time limit).

Exact local update for the seven questions of probe_C2B: prior of each question's truth = its M210 weights (ledger + probes of
11-17 + composition) where they exist, else the pooled posterior behind ct112; conditioned on the reading of the probe
(C -> B on the seven: +2 units of 1 - e^-1, -1 unit of e^-1) by enumeration (dynamic programme over the two counts). The
coupling of these seven with the rest of the ledger through tonight's reading is ignored (the other three probes do not touch them).

ct118 = ct116 with
  the edits of ct117 kept as they are (their weights predate tonight's probes; none of them is in a probe except pub_q_0286),
  pub_q_0286: the answer with the largest expected score under the local update,
  the other six questions of probe_C2B -> B (their summed contribution is positive whatever the truth of pub_q_0286 is).
Private rows = ct116.

  python -I scripts/m_series/m220_local.py
"""
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; X_OBS, Y_OBS = 2, -1


def main():
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a117, c2b = rd("submission_rank_ct116.csv"), rd("submission_rank_ct117.csv"), rd("upload_probe/probe_C2B.csv"); seven = [q for q in a116 if a116[q] != c2b[q]]; assert len(seven) == 7 and all(a116[q] == "C" and c2b[q] == "B" for q in seven)
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); post = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}; old = json.load(open(ROOT / "results/r_variants/m210_profiles.json"))
    prior = {}
    for q in seven:
        if q in old:
            w = np.array([old[q].get(l, 0.0) for l in LEV]); src = "M210 weights"
        else:
            w = post[q].copy(); src = "pooled posterior"
        prior[q] = (w / w.sum(), src)
    inc = [led.xy(LEV.index("C"), LEV.index("B"), c) for c in range(10)]  # (x, y) of one question by its truth

    def dist(qs):
        d = {(0, 0): 1.0}
        for q in qs:
            n = {}
            for (x, y), p in d.items():
                for c in range(10):
                    k = (x + inc[c][0], y + inc[c][1]); n[k] = n.get(k, 0.0) + p * prior[q][0][c]
            d = n
        return d
    p_obs = dist(seven).get((X_OBS, Y_OBS), 0.0); print(f"probability of the reading (+2, -1) under the priors of the seven questions: {p_obs:.4f}")
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); postw = {}
    for q in seven:
        rest = dist([r for r in seven if r != q]); w = np.array([prior[q][0][c] * rest.get((X_OBS - inc[c][0], Y_OBS - inc[c][1]), 0.0) for c in range(10)]); w = w / w.sum(); postw[q] = w
        es = S @ w; top = np.argsort(-w)[:4]; pt = np.argsort(-prior[q][0])[:3]
        print(f"  {q}: prior ({prior[q][1]}) " + " ".join(f"{LEV[c]} {prior[q][0][c]:.2f}" for c in pt) + " | after the reading " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005)
              + f" | expected score C {es[1]:.3f}, B {es[2]:.3f}, A {es[3]:.3f}, D {es[0]:.3f}; ct117 answers {a117[q]}")
    out = dict(a117); why = {q: "ct117 edit kept" for q in a116 if a117[q] != a116[q]}
    q0 = "pub_q_0286"; es = S @ postw[q0]; best = max(("C", "A", "B", "D"), key=lambda l: es[LEV.index(l)]); out[q0] = best; why[q0] = f"local update: expected score C {es[1]:.3f}, B {es[2]:.3f}, A {es[3]:.3f} -> {best}"
    best_mode = "--best" in sys.argv  # amendment of 2026-11-18 01:5x: each of the six gets the answer with the largest expected score under the local update
    for q in seven:
        if q != q0:
            e_ = S @ postw[q]; lv = max(("C", "B", "A", "D"), key=lambda l: e_[LEV.index(l)]) if best_mode else "B"
            out[q] = lv; why[q] = (f"local update: expected score C {e_[1]:.3f}, B {e_[2]:.3f}, D {e_[0]:.3f} -> {lv}" if best_mode else "probe_C2B (the six together are positive for every truth of pub_q_0286)")
    six = [q for q in seven if q != q0]; e6 = sum(float((S @ postw[q])[2] - (S @ postw[q])[1]) for q in six)
    worst = min(0.8964 - ((1 - led.E1) * x + led.E1 * y) for x, y in set(inc)); print(f"the six other questions -> B: expected {e6 * 1:+.3f} points under the local update; guaranteed minimum over the truths of {q0}: {worst:+.3f} points")
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"
    ch = [q for q in a116 if out[q] != a116[q]]; print(f"\n== {'ct118d' if best_mode else 'ct118'}: {len(ch)} Public answers differ from ct116; Private rows identical")
    for q in sorted(set(ch) | set(why)):
        print(f"  {q}: ct116 {a116[q]:>2} | ct117 {a117[q]:>2} | ct118 {out[q]:>2}   {why.get(q, '')}")
    buf = io.StringIO(newline=""); wr = csv.writer(buf, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, a116[q]) for q in a116)
    assert buf.getvalue().encode() == (ROOT / "submission_rank_ct116.csv").read_bytes(), "the writer does not reproduce ct116 byte for byte"
    path = ROOT / ("submission_rank_ct118d.csv" if best_mode else "submission_rank_ct118.csv"); assert not path.exists(), f"{path.name} exists already - nothing overwritten"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
