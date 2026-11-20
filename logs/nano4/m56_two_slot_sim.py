"""Section 4M M56 (2026-11-08): can one upload test two boundary pushes? Simulation only, no upload.
Design ("two-slot"): push A is applied forward as usual (decision i -> j on its public candidate set); push B (hypothesis: questions decided i'
near i'+1 are truly i'+1) is applied REVERSED, i' -> i'-1, so a true i'+1 in B shows up as a two-off loss (Y - 1) instead of an X change.
The upload's exact readout (X, Y) (400 dS = 0.63212 X + 0.36788 Y) is decoded by Bayes: per-question truth distributions under each of the
four hypotheses (A null/signal x B null/signal) give P(X, Y | H) by dynamic programming; posterior P(push signal | readout) with prior pi;
keep a push if that posterior > 0.5. Compared with single-push probes decoded the same way (one upload per push).
Truth model per question (public members of the candidate sets, ct9 posteriors as the base):
  null   = ct9's posterior;   signal = the target class's probability multiplied by BOOST, renormalised (precision of the push ~0.75-0.85).
Pairs: (A fwd, B reversed) from the probe files ct16-ct28 (each = ct13 + one push).
  python -I logs/nano4/m56_two_slot_sim.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
E1 = float(np.exp(-1.0))
BOOST, PI, SIMS = 4.0, 0.2, 3000
PROBES = {"2D->1D": 16, "3D->2D": 17, "4D->3D": 18, "5D->4D": 19, "6D->5D": 20, "A->B": 21, "B->C": 22, "C->D": 23, "D->C": 24, "C->B": 25,
          "1D->2D": 26, "3D->4D": 27, "4D->5D": 28}
PAIRS = [("2D->1D", "3D->2D"), ("3D->2D", "2D->1D"), ("1D->2D", "4D->3D"), ("B->C", "C->D"), ("2D->1D", "1D->2D"), ("A->B", "5D->4D")]


def s(d, c):
    k = abs(d - c)
    return 1.0 if k == 0 else E1 if k == 1 else 0.0


def xy(d0, d1, c):
    dd = s(d1, c) - s(d0, c)
    table = {1.0: (1, 1), -1.0: (-1, -1), round(1 - E1, 6): (1, 0), round(E1 - 1, 6): (-1, 0), round(E1, 6): (0, 1), round(-E1, 6): (0, -1), 0.0: (0, 0)}
    return table[round(dd, 6)]


def dist_xy(items, probs):
    """P(X, Y) for independent questions: items [(d0, d1)], probs [Q, 10] truth distributions. Returns dict-like array with offset."""
    Q = len(items); off = 2 * Q + 1; size = 2 * off + 1
    D = np.zeros((size, size)); D[off, off] = 1.0
    for (d0, d1), p in zip(items, probs):
        N = np.zeros_like(D)
        for c in range(10):
            if p[c] <= 0:
                continue
            x, y = xy(d0, d1, c)
            N += p[c] * np.roll(np.roll(D, x, 0), y, 1)
        D = N
    return D, off


def boosted(P, target):
    Q = P.copy(); Q[:, target] *= BOOST; return Q / Q.sum(1, keepdims=True)


def main():
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id"); P9 = pr[[f"p_{lv}" for lv in LEVELS]]
    base = pd.read_csv(ROOT / "submission_rank_ct13.csv", dtype=str).set_index("question_id")["pred_rank"]
    pub = base.index[base.index.str.startswith("pub")]
    sets = {}
    for name, k in PROBES.items():
        f = pd.read_csv(ROOT / f"submission_rank_ct{k}.csv", dtype=str).set_index("question_id")["pred_rank"].reindex(base.index)
        q = [x for x in pub if f[x] != base[x]]
        src, dst = name.split("->")
        Pq = np.array([P9.loc[x].to_numpy(float) for x in q]); Pq = Pq / Pq.sum(1, keepdims=True)
        sets[name] = {"q": q, "src": RANK[src], "dst": RANK[dst], "P": Pq}
    rng = np.random.default_rng(0)
    rep = {}
    for a_name, b_name in PAIRS:
        A, B = sets[a_name], sets[b_name]
        rev = A["src"] if False else None  # noqa: F841
        b_rev_to = B["src"] - (B["dst"] - B["src"])  # reversed: move away from the hypothesised target
        if not 0 <= b_rev_to <= 9:
            print(f"skip {a_name} + {b_name}: reversed move out of range"); continue
        items_A = [(A["src"], A["dst"])] * len(A["q"]); items_B = [(B["src"], b_rev_to)] * len(B["q"])
        truth = {("A", False): A["P"], ("A", True): boosted(A["P"], A["dst"]), ("B", False): B["P"], ("B", True): boosted(B["P"], B["dst"])}
        # likelihood tables
        L2, offs = {}, None
        for ha in (False, True):
            for hb in (False, True):
                D, off = dist_xy(items_A + items_B, np.vstack([truth[("A", ha)], truth[("B", hb)]])); L2[(ha, hb)] = D; offs = off
        LA = {h: dist_xy(items_A, truth[("A", h)]) for h in (False, True)}
        LBf = {h: dist_xy([(B["src"], B["dst"])] * len(B["q"]), truth[("B", h)]) for h in (False, True)}
        prior = {True: PI, False: 1 - PI}
        res = {}
        for ha in (False, True):
            for hb in (False, True):
                ok2 = {"A": 0, "B": 0}; ok1 = {"A": 0, "B": 0}
                for _ in range(SIMS):
                    ca = np.array([rng.choice(10, p=p) for p in truth[("A", ha)]]); cb = np.array([rng.choice(10, p=p) for p in truth[("B", hb)]])
                    X = sum(xy(d0, d1, c)[0] for (d0, d1), c in zip(items_A + items_B, list(ca) + list(cb)))
                    Y = sum(xy(d0, d1, c)[1] for (d0, d1), c in zip(items_A + items_B, list(ca) + list(cb)))
                    post = {(u, w): prior[u] * prior[w] * L2[(u, w)][X + offs, Y + offs] for u in (False, True) for w in (False, True)}
                    z = sum(post.values()) or 1e-300
                    pa = (post[(True, False)] + post[(True, True)]) / z; pb = (post[(False, True)] + post[(True, True)]) / z
                    ok2["A"] += (pa > 0.5) == ha; ok2["B"] += (pb > 0.5) == hb
                    # single probes
                    Xa = sum(xy(A["src"], A["dst"], c)[0] for c in ca); Ya = sum(xy(A["src"], A["dst"], c)[1] for c in ca)
                    Da = {h: LA[h][0][Xa + LA[h][1], Ya + LA[h][1]] * prior[h] for h in (False, True)}
                    ok1["A"] += (Da[True] / (Da[True] + Da[False] + 1e-300) > 0.5) == ha
                    Xb = sum(xy(B["src"], B["dst"], c)[0] for c in cb); Yb = sum(xy(B["src"], B["dst"], c)[1] for c in cb)
                    Db = {h: LBf[h][0][Xb + LBf[h][1], Yb + LBf[h][1]] * prior[h] for h in (False, True)}
                    ok1["B"] += (Db[True] / (Db[True] + Db[False] + 1e-300) > 0.5) == hb
                res[f"A {'sig' if ha else 'null'} / B {'sig' if hb else 'null'}"] = {"two_slot": {k: v / SIMS for k, v in ok2.items()}, "single": {k: v / SIMS for k, v in ok1.items()}}
        rep[f"{a_name} fwd + {b_name} rev"] = {"nA": len(A["q"]), "nB": len(B["q"]), "results": res}
        print(f"\n{a_name} forward ({len(A['q'])} public q) + {b_name} reversed as {LEVELS[B['src']]}->{LEVELS[b_rev_to]} ({len(B['q'])} public q): "
              f"correct-decision rate, two-slot vs single probes")
        for k, v in res.items():
            print(f"   truth {k:22s}: push A {v['two_slot']['A']:.3f} vs {v['single']['A']:.3f}; push B {v['two_slot']['B']:.3f} vs {v['single']['B']:.3f}", flush=True)
    Path(ROOT / "results/trunk_seq/es/m56_two_slot_sim.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
