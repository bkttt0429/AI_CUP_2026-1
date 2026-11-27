#!/usr/bin/env python
"""M210 (2026-11-17), Public only: truth weights of every Public question that some upload has constrained, under the ledger
+ the two probes of M206 + the exact Public composition (the joint problem of m208_ledger_probe.py), in parallel.

For each such question and each candidate truth: the cost of the cheapest joint assignment with that truth fixed; weights =
exp(-(cost - min)), normalised. Output: results/r_variants/m210_profiles.json {question: {level: weight}} and a table of the
questions where an answer other than the reference file's has a larger expected score.

  python -I scripts/m_series/m210_profile_all.py [--ref submission_rank_ct116.csv] [--workers 44]
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m58_decode as led  # noqa: E402
import m208_ledger_probe as m8  # noqa: E402

LEV = led.LEVELS; ST = {}


def one(task):
    q, c = task; a_, cost = m8.solve(ST["pub"], ST["cand"], ST["obs"], ST["prior"], ST["counts"], fix=(q, c), time_limit=150)
    return q, c, (cost if isinstance(a_, dict) else None), (a_ == "timeout")


def main():
    opt = dict(zip(sys.argv[1:], sys.argv[2:])); ref_file = opt.get("--ref", "submission_rank_ct116.csv"); workers = int(opt.get("--workers", "44"))
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **m8.PROBES}; qs, cand, obs = led.build(up, prior); comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}
    ref = led.decisions(ref_file); pub = [q for q in ref.index if q.startswith("pub")]; cand2 = {q: (cand[q] if q in cand else list(range(10))) for q in pub}; ST.update(pub=pub, cand=cand2, obs=obs, prior=prior, counts=counts)
    tasks = [(q, c) for q in qs for c in cand2[q]]; print(f"{len(qs)} constrained Public questions, {len(tasks)} solves, {workers} workers", flush=True)
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
    (ROOT / "results/r_variants/m210_profiles.json").write_text(json.dumps({q: {LEV[c]: v for c, v in w.items()} for q, w in prof.items()}, indent=1)); print(f"profiles of {len(prof)} questions written to results/r_variants/m210_profiles.json; solves that hit the time limit: {late}", flush=True)
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); rows = []
    for q, w in prof.items():
        wv = np.zeros(10)
        for c, v in w.items():
            wv[c] = v
        es = S @ wv; cur = int(ref[q]); best = int(es.argmax()); rows.append((float(es[best] - es[cur]), q, cur, best, float(wv[best]), float(wv[cur])))
    rows.sort(reverse=True); print(f"questions where another answer has a larger expected score than {ref_file} (gain, question, answer -> better answer, weight of the better answer as the truth, weight of the current answer):")
    tot = 0.0
    for g, q, cur, best, wb, wc in rows:
        if best != cur and g > 0.02:
            tot += g; print(f"  {g:+.3f}  {q}: {LEV[cur]} -> {LEV[best]}   truth {LEV[best]} {wb:.2f}, {LEV[cur]} {wc:.2f}; pooled posterior " + " ".join(f"{LEV[c]} {prior[q][c]:.2f}" for c in np.argsort(-prior[q])[:3]))
    print(f"sum of the expected gains listed: {tot:+.2f} points")


if __name__ == "__main__":
    main()
