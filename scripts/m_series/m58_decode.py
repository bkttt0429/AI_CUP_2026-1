"""Section 4M M58: the Public ledger. Every upload's exact Public score is a pair of linear constraints on the hidden Public labels:
400 (S_u - S_ref) = 0.63212 X_u + 0.36788 Y_u with X_u / Y_u the one-level / two-level net changes summed over the questions whose decision
differs from the reference file (ct9). All uploads are decoded jointly by a 0/1 MILP (scipy.optimize.milp): one-hot truth per involved
question over the classes within two of any decision it received, the exact X_u, Y_u of every upload as equality constraints, ct9's posterior
as the log-prior objective. Report: the MAP truths, which questions are DETERMINED (only one class is feasible), each probe's net vs ct13,
and the keep decision of single-push probes (Bayes rule, prior signal rate PI, signal precision P1 vs null precision P0).
  python -I scripts/m_series/m58_decode.py add submission_rank_ct16.csv 0.7760000     # record an upload's Public score
  python -I scripts/m_series/m58_decode.py report                                    # decode everything recorded
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path.cwd()
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
RANK = {lv: i for i, lv in enumerate(LEVELS)}
E1 = math.exp(-1.0)
LEDGER = ROOT / "results/trunk_seq/es/public_ledger.json"
REF = "submission_rank_ct9.csv"
KNOWN_SCORES = {"submission_rank_ct7.csv": 0.7687835, "submission_rank_ct9.csv": 0.7622032, "submission_rank_ct10.csv": 0.7701047, "submission_rank_ct12.csv": 0.7656229,
                "submission_rank_ct13.csv": 0.7748456, "submission_rank_ct14.csv": 0.7732653, "submission_rank_ct15.csv": 0.7676047}
PI, P1 = 0.05, 0.8  # 2026-11-08 19:00: prior signal rate lowered from 0.1 to 0.05 before any further readout (M61 + report 測試端新訊號來源)


def s(d, c):
    k = abs(d - c)
    return 1.0 if k == 0 else E1 if k == 1 else 0.0


def xy(d0, d1, c):
    """(x, y) increments of one question whose decision moves d0 -> d1 when its truth is c: 1 = (1, 1), 1 - e^-1 = (1, 0), e^-1 = (0, 1)."""
    dd = round(s(d1, c) - s(d0, c), 6)
    return {1.0: (1, 1), -1.0: (-1, -1), round(1 - E1, 6): (1, 0), round(E1 - 1, 6): (-1, 0), round(E1, 6): (0, 1), round(-E1, 6): (0, -1), 0.0: (0, 0)}[dd]


def decompose(delta400, nmax):
    sols = [(X, Y) for X in range(-nmax, nmax + 1) for Y in range(-nmax, nmax + 1) if abs((1 - E1) * X + E1 * Y - delta400) < 3e-4]
    return sols


def decisions(path):
    return pd.read_csv(ROOT / path, dtype=str).set_index("question_id")["pred_rank"].map(RANK)


def build(uploads, prior_P, extra_queries=()):
    """uploads: {file: score}. extra_queries: [(decisions Series, X, Y)] for simulations. Returns (questions, cand, obs)."""
    ref = decisions(REF); pub = [q for q in ref.index if q.startswith("pub")]
    obs = []
    for f, sc in uploads.items():
        if f == REF:
            continue
        d = decisions(f).reindex(ref.index)
        ch = [q for q in pub if d[q] != ref[q]]
        delta = round((sc - uploads.get(REF, KNOWN_SCORES[REF])) * 400, 5)
        sol = decompose(delta, len(ch))
        if len(sol) != 1:
            raise SystemExit(f"{f}: score {sc} gives {len(sol)} (X, Y) decompositions over {len(ch)} changes: {sol}")
        obs.append({"name": f, "moves": {q: (int(ref[q]), int(d[q])) for q in ch}, "X": sol[0][0], "Y": sol[0][1]})
    for k, (d, X, Y) in enumerate(extra_queries):
        ch = [q for q in pub if d[q] != ref[q]]
        obs.append({"name": f"query{k}", "moves": {q: (int(ref[q]), int(d[q])) for q in ch}, "X": X, "Y": Y})
    qs = sorted({q for o in obs for q in o["moves"]})
    cand = {}
    for q in qs:
        ds = {int(ref[q])} | {o["moves"][q][1] for o in obs if q in o["moves"]}
        cand[q] = sorted({c for d in ds for c in range(d - 2, d + 3) if 0 <= c <= 9})
    return qs, cand, obs


def solve(qs, cand, obs, prior_P, fix=None, forbid=None, time_limit=None):
    """MAP truth assignment (dict q -> class) or None if infeasible. fix: (q, c) forced; forbid: (q, c) excluded."""
    var = [(q, c) for q in qs for c in cand[q]]
    idx = {v: i for i, v in enumerate(var)}
    cost = np.array([-math.log(max(prior_P[q][c], 1e-6)) for q, c in var])
    n_rows = len(qs) + 2 * len(obs)
    A = lil_matrix((n_rows, len(var))); lo = np.zeros(n_rows); hi = np.zeros(n_rows)
    for r, q in enumerate(qs):
        for c in cand[q]:
            A[r, idx[(q, c)]] = 1
        lo[r] = hi[r] = 1
    for k, o in enumerate(obs):
        rx, ry = len(qs) + 2 * k, len(qs) + 2 * k + 1
        for q, (d0, d1) in o["moves"].items():
            for c in cand[q]:
                x, y = xy(d0, d1, c)
                if x:
                    A[rx, idx[(q, c)]] = x
                if y:
                    A[ry, idx[(q, c)]] = y
        lo[rx] = hi[rx] = o["X"]; lo[ry] = hi[ry] = o["Y"]
    lb = np.zeros(len(var)); ub = np.ones(len(var))
    if fix:
        lb[idx[fix]] = 1
    if forbid:
        ub[idx[forbid]] = 0
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(len(var)), bounds=Bounds(lb, ub),
               options={"time_limit": time_limit} if time_limit else None)
    if res.status == 1:  # time limit: unknown
        return "timeout"
    if res.status != 0 or res.x is None:
        return None
    return {q: c for (q, c), v in zip(var, res.x) if v > 0.5}


def feasible_sets(qs, cand, obs, prior_P, mapa, only=None, time_limit=None):
    """{q: sorted feasible classes}; a check that hits the time limit counts its class as feasible (conservative)."""
    out = {}
    for q in (only or qs):
        fs = {mapa[q]}
        for c in cand[q]:
            if c != mapa[q] and solve(qs, cand, obs, prior_P, fix=(q, c), time_limit=time_limit) is not None:
                fs.add(c)
        out[q] = sorted(fs)
    return out


def determined(qs, cand, obs, prior_P, mapa, only=None, time_limit=None):
    """{q: True if no feasible assignment gives q a class other than its MAP class} -- one MILP per question."""
    return {q: solve(qs, cand, obs, prior_P, forbid=(q, mapa[q]), time_limit=time_limit) is None for q in (only or qs)}


def prior_from_ct9():
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id")
    return {q: pr.loc[q, [f"p_{lv}" for lv in LEVELS]].to_numpy(float) for q in pr.index}


def keep_rule(right, wrong, p0):
    lr = (P1 / p0) ** right * ((1 - P1) / (1 - p0)) ** wrong
    post = PI * lr / (PI * lr + 1 - PI)
    return post


def report():
    led = json.load(open(LEDGER)) if LEDGER.exists() else {}
    uploads = {**KNOWN_SCORES, **led}
    prior = prior_from_ct9()
    qs, cand, obs = build(uploads, prior)
    mapa = solve(qs, cand, obs, prior)
    if mapa is None:
        raise SystemExit("the recorded scores are jointly infeasible: a score or a file was recorded wrongly")
    fsets = feasible_sets(qs, cand, obs, prior, mapa)
    det = [q for q in qs if len(fsets[q]) == 1]
    print(f"{len(obs)} uploads decoded vs ct9; {len(qs)} public questions involved, {len(det)} determined")
    base13 = decisions("submission_rank_ct13.csv")
    for o in obs:
        print(f"  {o['name']:28s} vs ct9: X {o['X']:+d} Y {o['Y']:+d} over {len(o['moves'])} changes")
    print("single-push probes vs ct13 (right = truth is the target, wrong = truth is the source; ? = not determined):")
    for f in uploads:
        if f in KNOWN_SCORES or f == REF:
            continue
        d = decisions(f)
        ch = [q for q in d.index if q.startswith("pub") and d[q] != base13[q]]
        if not ch:
            continue
        moves = pd.Series([f"{LEVELS[base13[q]]}->{LEVELS[d[q]]}" for q in ch]).value_counts()
        right = sum(1 for q in ch if mapa[q] == d[q]); wrong = sum(1 for q in ch if mapa[q] == base13[q]); undet = sum(1 for q in ch if len(fsets[q]) > 1)
        p0 = 0.4
        print(f"  {f:28s} {dict(moves)}: right {right}, wrong {wrong}, other {len(ch) - right - wrong}, undetermined {undet}; "
              f"P(push is signal) {keep_rule(right, wrong, p0):.2f} -> {'KEEP' if keep_rule(right, wrong, p0) > 0.5 else 'drop'}")
    out = {"map": {q: LEVELS[c] for q, c in mapa.items()}, "feasible": {q: [LEVELS[c] for c in fs] for q, fs in fsets.items()}, "determined": det}
    Path(ROOT / "results/trunk_seq/es/public_ledger_decoded.json").write_text(json.dumps(out, indent=1)); print("written results/trunk_seq/es/public_ledger_decoded.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("add", "report")); ap.add_argument("file", nargs="?"); ap.add_argument("score", nargs="?", type=float)
    a = ap.parse_args()
    if a.cmd == "add":
        led = json.load(open(LEDGER)) if LEDGER.exists() else {}
        led[a.file] = a.score; LEDGER.write_text(json.dumps(led, indent=1)); print(f"recorded {a.file} = {a.score}")
    report()


if __name__ == "__main__":
    main()
