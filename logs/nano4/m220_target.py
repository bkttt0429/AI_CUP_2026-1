#!/usr/bin/env python
"""M220 (2026-11-18), report only: the joint problem of m220_resolve.py, profiles of the questions that matter for ct118 only
(ct117's twelve edits + the questions of tonight's four probes), with the time of every solve - the full re-profile of all 202
constrained questions turned out to need hours after the four readings were added.

  python -I logs/nano4/m220_target.py [--workers 24] [--limit 150]
"""
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m208_ledger_probe as m8  # noqa: E402
import m220_resolve as m220  # noqa: E402

LEV = led.LEVELS; ST = {}


def one(task):
    q, c = task; t0 = time.time(); a_, cost = m8.solve(ST["pub"], ST["cand"], ST["obs"], ST["prior"], ST["counts"], fix=(q, c), time_limit=ST["limit"])
    return q, c, (cost if isinstance(a_, dict) else None), (a_ == "timeout"), time.time() - t0


def main():
    opt = dict(zip(sys.argv[1:], sys.argv[2:])); workers = int(opt.get("--workers", "24")); ST["limit"] = int(opt.get("--limit", "150")); t0 = time.time()
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **m220.OLD, **m220.NEW}; qs, cand, obs = led.build(up, prior)
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}
    ref = led.decisions(m220.REF_FILE); pub = [q for q in ref.index if q.startswith("pub")]; cand2 = {q: (cand[q] if q in cand else list(range(10))) for q in pub}; ST.update(pub=pub, cand=cand2, obs=obs, prior=prior, counts=counts)
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a117 = rd(m220.REF_FILE), rd("submission_rank_ct117.csv"); groups = {"ct117 edits": [q for q in a116 if a116[q] != a117[q]]}
    for f in m220.NEW:
        d = rd(f); groups[f] = [q for q in a116 if a116[q] != d[q]]
    target = list(dict.fromkeys(q for g in groups.values() for q in g)); tasks = [(q, c) for q in target for c in cand2[q]]
    t1 = time.time(); m0, c0 = m8.solve(pub, cand2, obs, prior, counts); print(f"{len(obs)} uploads; joint problem with the composition: cost {c0:.2f} in {time.time() - t1:.1f} s; {len(target)} target questions, {len(tasks)} solves, {workers} workers, limit {ST['limit']} s", flush=True)
    cost = {}; late = 0; secs = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (q, c, v, to, dt) in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            late += bool(to); secs.append(dt)
            if v is not None:
                cost.setdefault(q, {})[c] = v
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1}/{len(tasks)} solves done ({time.time() - t0:.0f} s); time per solve median {np.median(secs):.1f} s, max {max(secs):.1f} s; at the limit {late}", flush=True)
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); old = json.load(open(ROOT / "results/r_variants/m210_profiles.json")); prof = {}
    for q, d in cost.items():
        m = min(d.values()); w = {c: math.exp(-(v - m)) for c, v in d.items()}; t = sum(w.values()); prof[q] = {c: v / t for c, v in w.items()}
    (ROOT / "results/r_variants/m220_target_profiles.json").write_text(json.dumps({q: {LEV[c]: v for c, v in w.items()} for q, w in prof.items()}, indent=1))
    n_obs = {q: sum(q in o["moves"] for o in obs[:-len(m220.NEW)]) for q in target}
    for g, ql in groups.items():
        print(f"\n== {g}")
        for q in ql:
            w = np.zeros(10)
            for c, v in prof.get(q, {}).items():
                w[c] = v
            es = S @ w; cur = LEV.index(a116[q]); top = np.argsort(-w)[:3]
            print(f"  {q}: ct116 {a116[q]:>2}, ct117 {a117[q]:>2}; weights " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005) + f"; best answer {LEV[int(es.argmax())]} ({es.max() - es[cur]:+.3f} against ct116's; ct117's answer {es[LEV.index(a117[q])] - es[cur]:+.3f}); earlier uploads that changed it {n_obs[q]}"
                  + ("; M210 weights " + " ".join(f"{k_} {v:.2f}" for k_, v in sorted(old[q].items(), key=lambda t_: -t_[1])[:2]) if q in old else "") + (f"; candidates solved {len(prof.get(q, {}))}/{len(cand2[q])}"), flush=True)
    print(f"\nwritten results/r_variants/m220_target_profiles.json; solves at the time limit: {late} of {len(tasks)}")


if __name__ == "__main__":
    main()
