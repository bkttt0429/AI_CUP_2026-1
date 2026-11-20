"""Section 4M M102 (2026-11-13): a split file for the Public ledger and its back-test. ct73 and ct75n both add changes on top of
ct44; on the Public half: A = the changes they share, B = only in ct73, C = only in ct75n. Known: A + B (ct73 vs ct44) and A + C
(ct75n vs ct44). The split file = ct44 + A alone, so one reading separates the three groups.
Back-test with the ledger decoder (m58_decode.py: every upload's exact Public score is a pair of linear constraints on the hidden
Public labels, all uploads solved jointly):
 (1) are the recorded readings jointly feasible - without today's upload, with it recorded as ct75n, with it recorded as ct75;
 (2) every reading the split file can have, given everything recorded so far, and what each one means for A, B and C;
 (3) the questions of A, B, C whose label is already determined, and the most probable label of the others.
Writes submission_rank_split_A.csv (also into upload_next/), upload_next/split_A_decode.md and results/r_variants/m102_split.json.
  python -I logs/nano4/m102_split.py [--read 0.80xxxxx]     # with --read: decode the split file's reading and re-check everything
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as m  # noqa: E402
from aicup.submission import write_submission  # noqa: E402

NEW = {"submission_rank_ct42.csv": 0.7880062, "submission_rank_ct44.csv": 0.8067107, "submission_rank_ct64.csv": 0.8038092, "submission_rank_ct71.csv": 0.8053895,
       "submission_rank_ct72.csv": 0.8038092, "submission_rank_ct73.csv": 0.8060501}
TODAY, SPLIT, E1 = 0.8044698, "submission_rank_split_A.csv", math.exp(-1.0)


def try_solve(uploads, prior, extra=()):
    try:
        qs, cand, obs = m.build(uploads, prior, extra)
    except SystemExit as e:
        return None, None, None, f"not decodable: {str(e)[:160]}"
    sol = m.solve(qs, cand, obs, prior)
    return (qs, cand, obs, "feasible") if sol is not None else (qs, cand, obs, "INFEASIBLE")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--read", type=float, default=None); a_ = ap.parse_args()
    led = json.load(open(m.LEDGER)); prior = m.prior_from_ct9(); base = {**m.KNOWN_SCORES, **led, **NEW}; print(f"{len(base)} recorded uploads (reference ct9)", flush=True)
    for label, extra in (("without today's upload", {}), ("today's 0.8044698 recorded as ct75n", {"submission_rank_ct75n.csv": TODAY}), ("today's 0.8044698 recorded as ct75 (the full version)", {"submission_rank_ct75.csv": TODAY})):
        print(f"(1) {label}: {try_solve({**base, **extra}, prior)[3]}", flush=True)
    up = {**base, "submission_rank_ct75n.csv": TODAY}; qs, cand, obs, st = try_solve(up, prior); assert st == "feasible", st
    c44, c73, c75n = (m.decisions(f"submission_rank_{k}.csv") for k in ("ct44", "ct73", "ct75n")); pub = [q for q in c44.index if q.startswith("pub")]
    A = [q for q in pub if c73[q] != c44[q] and c75n[q] == c73[q]]; B = [q for q in pub if c73[q] != c44[q] and c75n[q] == c44[q]]; C = [q for q in pub if c75n[q] != c44[q] and c73[q] == c44[q]]; odd = [q for q in pub if c73[q] != c44[q] and c75n[q] != c44[q] and c75n[q] != c73[q]]
    print(f"groups on the Public half: A (shared) {len(A)}, B (ct73 only) {len(B)}, C (ct75n only) {len(C)}, changed differently {len(odd)}"); assert not odd
    d_split = c44.copy(); d_split[A] = c73[A]; inv = {v: k for k, v in m.RANK.items()}; f = ROOT / SPLIT; write_submission([(q, inv[int(d_split[q])]) for q in d_split.index], f); (ROOT / "upload_next" / SPLIT).write_bytes(f.read_bytes()); md5 = hashlib.md5(f.read_bytes()).hexdigest()[:12]
    chk = m.decisions(SPLIT); assert (chk.reindex(c44.index) == d_split).all() and int((chk != c44).sum()) == len(A)
    print(f"{SPLIT} written, md5 {md5}: ct44 with the {len(A)} shared Public changes (no Private question differs from ct44: {bool(all(q.startswith('pub') for q in c44.index[chk != c44]))})")
    o44 = next(o for o in obs if o["name"] == "submission_rank_ct44.csv"); X44, Y44 = o44["X"], o44["Y"]; S44 = NEW["submission_rank_ct44.csv"]; rows = []
    for a in range(-len(A), len(A) + 1):
        for b_ in range(-len(A), len(A) + 1):
            if abs(a - b_) > len(A):
                continue
            qs2, cand2, obs2 = m.build(up, prior, [(d_split, X44 + a, Y44 + b_)]); sol = m.solve(qs2, cand2, obs2, prior)
            if sol is not None:
                cost = float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items())); rows.append({"a": a, "b": b_, "score": S44 + ((1 - E1) * a + E1 * b_) / 400, "cost": cost})
    cmin = min(r["cost"] for r in rows); z = sum(math.exp(-(r["cost"] - cmin)) for r in rows)
    for r in rows:
        r["weight"] = math.exp(-(r["cost"] - cmin)) / z; r["B"] = (-1 - r["a"], 1 - r["b"]); r["C"] = (-2 - r["a"], 1 - r["b"])
    rows.sort(key=lambda r: -r["weight"]); print(f"(2) feasible readings of the split file: {len(rows)} (weight = relative plausibility of the best label assignment under ct9's posterior; rough)")
    for r in rows:
        print(f"    reading {r['score']:.7f}  A: exact {r['a']:+d}, within one {r['b']:+d} | B: exact {r['B'][0]:+d}, within one {r['B'][1]:+d} | C: exact {r['C'][0]:+d}, within one {r['C'][1]:+d} | weight {r['weight']:.3f}")
    mapa = m.solve(qs, cand, obs, prior); grp = {**{q: "A" for q in A}, **{q: "B" for q in B}, **{q: "C" for q in C}}; fs = m.feasible_sets(qs, cand, obs, prior, mapa, only=[q for q in grp if q in qs], time_limit=20)
    print("(3) per question (group, ct44 -> new answer, labels still possible, most probable label):")
    for q in A + B + C:
        new = c73[q] if q in A or q in B else c75n[q]; poss = fs.get(q); print(f"    {q} {grp[q]}: {inv[int(c44[q])]} -> {inv[int(new)]} | possible {[inv[c] for c in poss] if poss else '?'} | most probable {inv[mapa[q]] if q in mapa else '?'}{'  DETERMINED' if poss and len(poss) == 1 else ''}")
    L = ["# 拆解檔 `submission_rank_split_A.csv` 的讀數對照表（自動產生）", "", f"檔案 md5 `{md5}`：ct44 加上 ct73 與 ct75n 共同的 {len(A)} 題 Public 改動（Private 的答案與 ct44 完全相同）。", "",
         f"ct73 獨有 {len(B)} 題（B 組）、ct75n 獨有 {len(C)} 題（C 組）。已知 A＋B：正確 −1、一級內 +1；A＋C：正確 −2、一級內 +1。", "",
         "讀數必須是下表其中之一；不在表內就代表傳錯檔或讀數抄錯。", "", "| Public 讀數 | A 組（共同 8 題） | B 組（ct73 獨有） | C 組（ct75n 獨有） | 相對可能性 |", "|---|---|---|---|---|"]
    L += [f"| {r['score']:.7f} | 正確 {r['a']:+d}、一級內 {r['b']:+d} | 正確 {r['B'][0]:+d}、一級內 {r['B'][1]:+d} | 正確 {r['C'][0]:+d}、一級內 {r['C'][1]:+d} | {r['weight']:.2f} |" for r in rows]
    L += ["", "「正確」＝完全答對的題數變化，「一級內」＝差一級以內的題數變化，都是相對 ct44。分數換算：每多一題正確 +0.00158，每多一題一級內 +0.00092。", "", "傳完後請把讀數貼給 Claude，會用全部歷次讀數一起回測；最後一筆上傳請留你選定的最終檔。"]
    (ROOT / "upload_next/split_A_decode.md").write_text("\n".join(L) + "\n"); (ROOT / "results/r_variants/m102_split.json").write_text(json.dumps({"md5": md5, "A": A, "B": B, "C": C, "outcomes": rows}, indent=1, default=float))
    if a_.read is not None:
        st2 = try_solve({**up, SPLIT: a_.read}, prior)[3]; hit = [r for r in rows if abs(r["score"] - a_.read) < 2e-7]; print(f"reading {a_.read}: {'in the table' if hit else 'NOT in the table'}; all recorded readings together: {st2}")
        for r in hit:
            print(f"   -> A: exact {r['a']:+d}, within one {r['b']:+d}; B: exact {r['B'][0]:+d}, within one {r['B'][1]:+d}; C: exact {r['C'][0]:+d}, within one {r['C'][1]:+d}")
    print("written")


if __name__ == "__main__":
    main()
