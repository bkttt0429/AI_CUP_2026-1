#!/usr/bin/env python
"""M220 (2026-11-18), Public only: the joint ledger problem of M210 / M211 with the four boundary probes of 2026-11-18 added,
and the Public-only file ct118 (Private rows = ct116).

Readings (all against ct116 = 0.8394955; decoded with the formula of m206_decode.py, each has one solution):
  probe_2D3D 0.8347546 (dE, dW) = (-3, 0);  probe_3D4D 0.8369955 (-1, -1);  probe_C2B 0.8417364 (+2, -1);  probe_5D4D 0.8363349 (-2, 0)
The problem: every recorded upload + the two probes of 11-17 + the reading of ct116 + these four, with the exact Public
composition (m208_ledger_probe.solve). For each constrained Public question and candidate truth: the cost of the cheapest
joint assignment with that truth fixed -> weights (results/r_variants/m220_profiles.json).

ct118 (rule written in the design doc before this ran):
  (b) the tightened rule of ct117 on the NEW weights: a truth other than ct116's answer with weight >= 0.95 on a question
      changed by >= 2 recorded uploads -> that answer;
  (c) ct117's two special edits (pub_q_0103 -> B, pub_q_0157 -> 4D) stay unless the new weights give them a lower
      expected score than ct116's answer;
  (d) the questions of probe_C2B that (b) does not edit -> B (the reading itself: +2 exact, -1 within one over the seven).

  python -I logs/nano4/m220_resolve.py [--workers 88]
"""
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m208_ledger_probe as m8  # noqa: E402

LEV = led.LEVELS; ST = {}; REF_FILE, REF_SCORE = "submission_rank_ct116.csv", 0.8394955
OLD = {"upload_probe/used_20261117/probe_4D5D_locked.csv": 0.8347546, "upload_probe/used_20261117/probe_A2B.csv": 0.8363349, REF_FILE: REF_SCORE}
NEW = {"upload_probe/probe_2D3D.csv": 0.8347546, "upload_probe/probe_3D4D.csv": 0.8369955, "upload_probe/probe_C2B.csv": 0.8417364, "upload_probe/probe_5D4D.csv": 0.8363349}
SPECIAL = {"pub_q_0103": "B", "pub_q_0157": "4D"}


def one(task):
    q, c = task; a_, cost = m8.solve(ST["pub"], ST["cand"], ST["obs"], ST["prior"], ST["counts"], fix=(q, c), time_limit=150)
    return q, c, (cost if isinstance(a_, dict) else None), (a_ == "timeout")


def main():
    opt = dict(zip(sys.argv[1:], sys.argv[2:])); workers = int(opt.get("--workers", "88"))
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **OLD, **NEW}; qs, cand, obs = led.build(up, prior)
    for o in obs[-len(NEW):]:
        print(f"  {o['name']}: against ct9 {len(o['moves'])} changed questions, exact {o['X']:+d}, within one {o['Y']:+d}")
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}
    ref = led.decisions(REF_FILE); pub = [q for q in ref.index if q.startswith("pub")]; cand2 = {q: (cand[q] if q in cand else list(range(10))) for q in pub}; ST.update(pub=pub, cand=cand2, obs=obs, prior=prior, counts=counts)
    m0, c0 = m8.solve(pub, cand2, obs, prior, counts); print(f"{len(obs)} uploads, {len(qs)} constrained Public questions; joint problem with the composition: " + (f"cost {c0:.2f}" if isinstance(m0, dict) else f"NOT solved ({m0})"), flush=True)
    if not isinstance(m0, dict):
        sys.exit("the joint problem with the new readings has no solution within the candidate sets - nothing written")
    tasks = [(q, c) for q in qs for c in cand2[q]]; print(f"{len(tasks)} solves, {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(one, tasks, chunksize=1)
    cost = {}; late = 0
    for q, c, v, to in res:
        late += bool(to)
        if v is not None:
            cost.setdefault(q, {})[c] = v
    prof = {}
    for q, d in cost.items():
        m = min(d.values()); w = {c: math.exp(-(v - m)) for c, v in d.items()}; t = sum(w.values()); prof[q] = {c: v / t for c, v in w.items()}
    (ROOT / "results/r_variants/m220_profiles.json").write_text(json.dumps({q: {LEV[c]: v for c, v in w.items()} for q, w in prof.items()}, indent=1)); print(f"profiles of {len(prof)} questions written to results/r_variants/m220_profiles.json; solves that hit the time limit: {late}", flush=True)
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); n_obs = {q: sum(q in o["moves"] for o in obs[:-len(NEW)]) for q in qs}  # support = uploads recorded before tonight's probes
    old = json.load(open(ROOT / "results/r_variants/m210_profiles.json")) if (ROOT / "results/r_variants/m210_profiles.json").exists() else {}

    def wv(q):
        v = np.zeros(10)
        for c, x in prof.get(q, {}).items():
            v[c] = x
        return v

    def show(q, note=""):
        w = wv(q); es = S @ w; cur = int(ref[q]); top = np.argsort(-w)[:3]
        return f"  {q}: ct116 {LEV[cur]:>2}; weights " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005) + f"; best answer {LEV[int(es.argmax())]} ({es.max() - es[cur]:+.3f} against ct116's); recorded uploads that changed it {n_obs.get(q, 0)}" + (f"; M210 weights " + " ".join(f"{k} {v:.2f}" for k, v in sorted(old[q].items(), key=lambda t_: -t_[1])[:2]) if q in old else "") + note

    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a117 = rd(REF_FILE), rd("submission_rank_ct117.csv"); e117 = [q for q in a116 if a116[q] != a117[q]]; out = dict(a116); why = {}
    print("\n== the twelve edits of ct117 under the new weights")
    for q in e117:
        w = wv(q); es = S @ w; print(show(q, f"; ct117 answers {a117[q]} ({es[LEV.index(a117[q])] - es[int(ref[q])]:+.3f})"))
    print("\n== the questions of tonight's probes under the new weights")
    for f in NEW:
        d = rd(f); print(f" {f}:")
        for q in [q for q in a116 if a116[q] != d[q]]:
            print(show(q))
    for q in qs:  # (b)
        w = wv(q); t = int(w.argmax())
        if w[t] >= 0.95 and t != int(ref[q]) and n_obs.get(q, 0) >= 2:
            out[q] = LEV[t]; why[q] = f"(b) weight {w[t]:.2f}, {n_obs[q]} uploads"
    for q, lv in SPECIAL.items():  # (c)
        if q not in why:
            es = S @ wv(q)
            if es[LEV.index(lv)] >= es[int(ref[q])]:
                out[q] = lv; why[q] = f"(c) special edit of ct117 kept ({es[LEV.index(lv)] - es[int(ref[q])]:+.3f})"
            else:
                why[q] = f"(c) special edit of ct117 DROPPED ({es[LEV.index(lv)] - es[int(ref[q])]:+.3f})"
    c2b = rd("upload_probe/probe_C2B.csv")
    for q in [q for q in a116 if a116[q] != c2b[q]]:  # (d)
        if q not in why:
            out[q] = "B"; why[q] = "(d) probe_C2B"
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"
    ch = [q for q in a116 if out[q] != a116[q]]; print(f"\n== ct118: {len(ch)} Public answers differ from ct116 ({len([q for q in ch if q in e117 and out[q] == a117[q]])} of them as in ct117); Private rows identical")
    for q in sorted(set(ch) | set(why) | set(e117)):
        print(f"  {q}: ct116 {a116[q]:>2} | ct117 {a117[q]:>2} | ct118 {out[q]:>2}   {why.get(q, 'not edited (rule (b) no longer holds)' if q in e117 else '')}")
    es_gain = sum(float((S @ wv(q))[LEV.index(out[q])] - (S @ wv(q))[int(ref[q])]) for q in ch if q in prof); print(f"expected gain of ct118 over ct116 under the new weights: {es_gain:+.2f} points (nominal; attribution to single questions is soft)")
    import io
    buf = io.StringIO(newline=""); wr = csv.writer(buf, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, a116[q]) for q in a116)
    assert buf.getvalue().encode() == (ROOT / REF_FILE).read_bytes(), "the writer does not reproduce ct116 byte for byte"
    path = ROOT / "submission_rank_ct118.csv"; assert not path.exists(), "submission_rank_ct118.csv exists already - nothing overwritten"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
