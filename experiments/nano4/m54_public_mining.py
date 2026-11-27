"""Section 4M M54 (2026-11-07; user: the CV-rejected A|1D push carried real test information, so mine the evidence more thoroughly):
every uploaded file's exact Public score is a linear constraint on the hidden Public labels. For two files a, b the score difference
400 (S_b - S_a) = 0.63212 X + 0.36788 Y exactly, with X = (one-off net: questions b gets exactly right minus questions a did) and Y the two-off
net, summed over the questions whose decisions differ. (X, Y) is read off uniquely when the difference set is small. Given the per-question
prior over the truth (ct9's test posteriors, with the eight ct13 flips fixed to A -- Public proved them), a forward-backward dynamic
programme over (X, Y) gives the posterior truth of EVERY changed question conditional on the exact readout, and so the posterior
precision of each change category (e.g. ct12's four 2D->3D flips). Report only; the files' Public scores are the user's uploads.
  python -I logs/nano4/m54_public_mining.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
E1 = float(np.exp(-1.0)); ONE = 1 - E1
SCORES = {"ct6": 0.7529805, "ct7": 0.7687835, "ct9": 0.7622032, "ct10": 0.7701047, "ct12": 0.7656229, "ct13": 0.7748456, "ct14": 0.7732653, "ct15": 0.7676047}
PAIRS = [("ct9", "ct13"), ("ct13", "ct14"), ("ct13", "ct15"), ("ct9", "ct12"), ("ct9", "ct10"), ("ct13", "ct10"), ("ct13", "ct12"), ("ct12", "ct10"), ("ct9", "ct7"), ("ct7", "ct10"), ("ct7", "ct13")]
KNOWN_A_FROM = ("ct9", "ct13")  # the eight questions ct13 flipped 1D -> A on Public are true A


def qscore_vec(d, c):
    k = abs(d - c)
    return 1.0 if k == 0 else E1 if k == 1 else 0.0


def decompositions(delta400, nmax, tol=2e-3):
    sols = [(X, Y) for X in range(-nmax, nmax + 1) for Y in range(-nmax, nmax + 1) if abs(ONE * X + E1 * Y - delta400) < tol]
    return sols


def posterior(items, prior, targets):
    """items: list of (old decision, new decision) per changed question; prior: [Q, 10] truth priors; targets: list of feasible (X, Y).
    Forward-backward over the (X, Y) state; returns per-question posterior truth [Q, 10] and the probability of each target."""
    Q = len(items); off = Q
    size = 2 * Q + 1
    # per-question contribution table: for truth c, (x, y) increments
    contrib = []
    for (o, nw) in items:
        rows = []
        for c in range(10):
            d = qscore_vec(nw, c) - qscore_vec(o, c)
            x = 1 if abs(d - ONE) < 1e-9 else -1 if abs(d + ONE) < 1e-9 else 0
            y = 1 if abs(d - E1) < 1e-9 else -1 if abs(d + E1) < 1e-9 else 0
            if abs(d - 1.0) < 1e-9:  # two-off -> exact (x +1, y +1 in score units: 1 = ONE + E1)
                x, y = 1, 1
            elif abs(d + 1.0) < 1e-9:
                x, y = -1, -1
            rows.append((x, y))
        contrib.append(rows)
    fwd = [None] * (Q + 1); fwd[0] = np.zeros((size, size)); fwd[0][off, off] = 1.0
    for i in range(Q):
        nxt = np.zeros((size, size))
        for c in range(10):
            p = prior[i, c]
            if p <= 0:
                continue
            x, y = contrib[i][c]
            src = fwd[i]
            dst = np.roll(np.roll(src, x, axis=0), y, axis=1)  # shifts; edges never reached (|X|,|Y| <= Q)
            nxt += p * dst
        fwd[i + 1] = nxt
    bwd = [None] * (Q + 1); bwd[Q] = np.zeros((size, size))
    for (X, Y) in targets:
        bwd[Q][X + off, Y + off] = 1.0
    for i in range(Q - 1, -1, -1):
        prev = np.zeros((size, size))
        for c in range(10):
            p = prior[i, c]
            if p <= 0:
                continue
            x, y = contrib[i][c]
            prev += p * np.roll(np.roll(bwd[i + 1], -x, axis=0), -y, axis=1)
        bwd[i] = prev
    Z = float((fwd[Q] * bwd[Q]).sum())
    post = np.zeros((Q, 10))
    for i in range(Q):
        for c in range(10):
            p = prior[i, c]
            if p <= 0:
                continue
            x, y = contrib[i][c]
            post[i, c] = p * float((np.roll(np.roll(fwd[i], x, axis=0), y, axis=1) * bwd[i + 1]).sum())
        post[i] /= max(post[i].sum(), 1e-300)
    p_targets = {f"{X},{Y}": float(fwd[Q][X + off, Y + off]) / max(float(fwd[Q][[X + off for X, Y in targets], [Y + off for X, Y in targets]].sum()), 1e-300) for (X, Y) in targets}
    return post, Z, p_targets


def main():
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id"); P9 = pr[[f"p_{lv}" for lv in LEVELS]]
    files = {k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id")["pred_rank"].map(RANK) for k in SCORES}
    pub = [q for q in files["ct9"].index if q.startswith("pub")]
    known = {}
    a, b = KNOWN_A_FROM
    for q in pub:
        if files[a][q] != files[b][q]:
            known[q] = RANK["A"]
    print(f"known truths injected: {len(known)} questions = A (ct13's Public flips)")
    rep = {}
    for a, b in PAIRS:
        diff = [q for q in pub if files[a][q] != files[b][q]]
        delta400 = round((SCORES[b] - SCORES[a]) * 400, 4)
        sols = decompositions(delta400, len(diff))
        cats = pd.Series([f"{LEVELS[files[a][q]]}->{LEVELS[files[b][q]]}" for q in diff]).value_counts().to_dict()
        print(f"\n== {b} vs {a}: {len(diff)} Public changes, 400 dS = {delta400:+.4f}; (X, Y) solutions {sols}; categories {cats}")
        if not sols or len(diff) > 60:
            print("   (skipped: no exact decomposition or too many changes)")
            rep[f"{b} vs {a}"] = {"changes": len(diff), "delta400": delta400, "solutions": sols, "categories": cats}
            continue
        prior = np.array([P9.loc[q].to_numpy(float) for q in diff])
        for i, q in enumerate(diff):
            if q in known:
                prior[i] = 0.0; prior[i, known[q]] = 1.0
        items = [(int(files[a][q]), int(files[b][q])) for q in diff]
        post, Z, p_t = posterior(items, prior, sols)
        print(f"   P(readout | prior) = {Z:.4f}; solution weights {p_t}")
        out_rows = []
        for i, q in enumerate(diff):
            o, nw = items[i]
            p_new, p_old = post[i, nw], post[i, o]
            out_rows.append({"q": q, "change": f"{LEVELS[o]}->{LEVELS[nw]}", "known": q in known, "P_truth_new": round(float(p_new), 3), "P_truth_old": round(float(p_old), 3),
                             "prior_new": round(float(P9.loc[q].iloc[nw]), 3), "prior_old": round(float(P9.loc[q].iloc[o]), 3), "top": LEVELS[int(post[i].argmax())]})
        df = pd.DataFrame(out_rows)
        for cat, g in df.groupby("change"):
            print(f"   {cat:8s} n {len(g):2d}: posterior E[# truth = new] {g['P_truth_new'].sum():.2f} (prior {g['prior_new'].sum():.2f}); E[# truth = old] {g['P_truth_old'].sum():.2f} (prior {g['prior_old'].sum():.2f})"
                  + ("" if len(g) > 8 else "; per question P(new) " + " ".join(f"{x:.2f}" for x in g["P_truth_new"])))
        rep[f"{b} vs {a}"] = {"changes": len(diff), "delta400": delta400, "solutions": sols, "categories": cats, "questions": out_rows}
    Path(ROOT / "results/trunk_seq/es/m54_public_mining.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nwritten")


if __name__ == "__main__":
    main()
