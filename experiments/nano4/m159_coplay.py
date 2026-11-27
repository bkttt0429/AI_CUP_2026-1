#!/usr/bin/env python
"""Section 4M M159 (2026-11-15, report only): do a player's opponents form a circle beyond level and activity? (design doc M159)

The identifier matches every game's opponent side against all 8,007 candidates on its own; who was recognised in the other games
of the question plays no part. If the data hold circles beyond the level (ids = account x period, fixed hours, fixed groups), the
recognised opponents of a question shrink the candidate list of its other games. Measured on the pairs recognised INSIDE the
training set (M96: even games, margin >= q90, 72,673 rows; true-pair share of the tiers 0.54 / 0.76 / 0.98):
  (1) rematches: share of rows in a pair that occurs more than once;
  (2) closure: two different opponents of one player are themselves a pair (clustering coefficient);
      null of (1) and (2): opponents permuted inside every (owner level, opponent level) cell, 20 times - every player's counts
      and the level-by-level pairing table are kept, so what is left is beyond level and activity;
  (3) twins (M154: mutual nearest prototypes, similarity >= 0.9): overlap of the two opponent sets and a direct pair, against
      random pairs of the same levels and similar degree;
  (4) use for identification (leave one owner out): an owner P with >= 3 different opponents in his own rows, one of them (R)
      hidden; every candidate's links to the others (S) in the graph without P's edges: direct pairs and common neighbours; AUC
      of R against 50 random candidates of R's level and similar degree; the same on a shuffled graph (must be 0.5).
Registered reading: closure ratio >= 2 and AUC >= 0.65 -> circles exist and help identification (next: a candidate prior from
the question's recognised opponents, CV test); ratio < 1.3 or AUC < 0.58 -> closed; in between -> record, estimate the ceiling.

  python -I logs/nano4/m159_coplay.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; N_NULL = 20


def graph(ow, op, C):
    """Binary symmetric adjacency of the distinct undirected pairs (self pairs dropped) and the number of rows per pair."""
    m = ow != op; a, b = np.minimum(ow[m], op[m]), np.maximum(ow[m], op[m]); key, cnt = np.unique(a * C + b, return_counts=True); i, j = key // C, key % C
    A = sp.coo_matrix((np.ones(len(key), np.float64), (i, j)), shape=(C, C)).tocsr(); A = A + A.T
    return A, cnt


def closure(A):
    d = np.asarray(A.sum(1)).ravel(); wedges = float((d * (d - 1) / 2).sum()); tri3 = float((A @ A).multiply(A).sum()) / 2  # = 3 x triangles
    return tri3 / max(wedges, 1.0), wedges, tri3 / 3


def shuffle(ow, op, lab, rng):
    """Opponents permuted inside every (owner level, opponent level) cell."""
    out = op.copy(); cell = lab[ow] * 10 + lab[op]
    for c in np.unique(cell):
        ix = np.flatnonzero(cell == c); out[ix] = op[ix][rng.permutation(len(ix))]
    return out


def link_auc(ow, op, lab, C, rng, n_owner=4000, n_rand=50):
    """Leave one owner out: a hidden opponent R of owner P against random candidates of R's level and similar degree, scored by
    the links to P's other opponents S in the graph without P's edges (direct pairs; common neighbours)."""
    A, _ = graph(ow, op, C); A2 = (A @ A).tocsr(); deg = np.asarray(A.sum(1)).ravel(); by_lab = [np.flatnonzero((lab == c) & (deg > 0)) for c in range(10)]
    own_sets = {}
    for o, q in zip(ow, op):
        if o != q:
            own_sets.setdefault(int(o), set()).add(int(q))
    owners = [o for o, s in own_sets.items() if len(s) >= 3]; rng.shuffle(owners); owners = owners[:n_owner]
    res = {"direct": [], "common": [], "both": []}; pos_any = {"direct": 0, "common": 0}; neg_any = {"direct": 0.0, "common": 0.0}; n_used = 0; sizes = []
    for P in owners:
        opps = np.array(sorted(own_sets[P])); R = int(opps[rng.integers(len(opps))]); S = opps[opps != R]
        pool = by_lab[lab[R]]; pool = pool[(deg[pool] >= deg[R] / 1.5) & (deg[pool] <= deg[R] * 1.5)]; nbP = set(A.indices[A.indptr[P]:A.indptr[P + 1]].tolist()) | {P, R}; pool = np.array([c for c in pool if c not in nbP])
        if len(pool) < 10:
            continue
        cand = pool if len(pool) <= n_rand else rng.choice(pool, n_rand, replace=False); rows = np.concatenate([[R], cand])
        direct = np.asarray(A[rows][:, S].sum(1)).ravel(); common = np.asarray(A2[rows][:, S].sum(1)).ravel()
        # paths c - P - s are removed: P is a neighbour of every s in S; of the candidates only R is a neighbour of P
        common[0] -= len(S)
        for name, sc in (("direct", direct), ("common", common), ("both", direct * 5 + common)):
            res[name].append(float((sc[0] > sc[1:]).mean() + 0.5 * (sc[0] == sc[1:]).mean()))
        for name, sc in (("direct", direct), ("common", common)):
            pos_any[name] += int(sc[0] > 0); neg_any[name] += float((sc[1:] > 0).mean())
        n_used += 1; sizes.append(len(S))
    out = {"owners": n_used, "median_S": float(np.median(sizes)) if sizes else 0.0}
    for name in res:
        v = np.array(res[name]); out[f"auc_{name}"] = float(v.mean()); out[f"auc_{name}_se"] = float(v.std() / np.sqrt(max(len(v), 1)))
    for name in pos_any:
        out[f"any_{name}_true"] = pos_any[name] / max(n_used, 1); out[f"any_{name}_random"] = neg_any[name] / max(n_used, 1)
    return out


def main():
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); lab = q["plab"].astype(int); C = len(players); assert C == 8007 and (np.sort(players) == players).all(), "the player list is not the sorted list M96 used"
    z = np.load(RV / "m96_pairs.npz"); OW, OP, TIER = z["owner"].astype(np.int64), z["opp"].astype(np.int64), z["tier"].astype(int); rep = {}
    d6 = (lab[OP] - lab[OW])[TIER == 6]; print(f"pairs: {len(OW)} rows (tiers q90 / q95 / q98: {int((TIER == 4).sum())} / {int((TIER == 5).sum())} / {int((TIER == 6).sum())}); at q98 the two labels are the same in {float((d6 == 0).mean()):.3f}, within one in {float((np.abs(d6) <= 1).mean()):.3f}")
    assert float((d6 == 0).mean()) > 0.35, "the indices do not match the player list"
    rng = np.random.default_rng(159)
    for name, sel in (("q95 and above", TIER >= 5), ("all (q90 and above)", TIER >= 4), ("q98 only", TIER >= 6)):
        ow, op = OW[sel], OP[sel]; A, cnt = graph(ow, op, C); cl, wedges, tri = closure(A); deg = np.asarray(A.sum(1)).ravel(); multi = float(cnt[cnt > 1].sum() / cnt.sum())
        nulls = []
        for _ in range(N_NULL):
            An, cn = graph(ow, shuffle(ow, op, lab, rng), C); c0, w0, t0 = closure(An); nulls.append((c0, float(cn[cn > 1].sum() / cn.sum()), t0, An.nnz / 2))
        nulls = np.array(nulls); r = {"rows": int(sel.sum()), "players": int((deg > 0).sum()), "pairs": int(A.nnz // 2), "median_degree": float(np.median(deg[deg > 0])), "rematch_share": multi, "rematch_share_null": float(nulls[:, 1].mean()),
                                      "closure": cl, "closure_null": float(nulls[:, 0].mean()), "closure_null_sd": float(nulls[:, 0].std()), "triangles": tri, "triangles_null": float(nulls[:, 2].mean()), "pairs_null": float(nulls[:, 3].mean())}
        r["closure_ratio"] = cl / max(r["closure_null"], 1e-12); rep[name] = r
        print(f"\n=== {name}: {r['rows']} rows, {r['players']} players with a pair, {r['pairs']} different pairs (shuffled {r['pairs_null']:.0f}), median degree {r['median_degree']:.0f}")
        print(f"   (1) rows in a pair that occurs more than once: {multi:.3f} (shuffled {r['rematch_share_null']:.3f})")
        print(f"   (2) closure (two opponents of one player are a pair): {cl:.4f}; shuffled {r['closure_null']:.4f} +/- {r['closure_null_sd']:.4f}; ratio {r['closure_ratio']:.2f}; triangles {tri:.0f} against {r['triangles_null']:.0f}")
        # closure by level of the centre player
        A2 = (A @ A).multiply(A); t_node = np.asarray(A2.sum(1)).ravel() / 2; w_node = deg * (deg - 1) / 2
        print("       by the level of the centre player: " + "  ".join(f"{LV[c]} {float(t_node[lab == c].sum() / max(w_node[lab == c].sum(), 1)):.3f}" for c in range(10)))
    # (3) twins
    try:
        P = [np.load(ROOT / f"results/m120/out_-1{t}.npz")["protos"].astype(np.float64) for t in ("", "_s1", "_s3")]; S = sum(p @ p.T for p in P) / len(P); np.fill_diagonal(S, -2); j = S.argmax(1); s = S[np.arange(C), j]
        tw = np.flatnonzero((j[j] == np.arange(C)) & (s >= 0.9) & (np.arange(C) < j)); ow, op = OW[TIER >= 5], OP[TIER >= 5]; A, _ = graph(ow, op, C); deg = np.asarray(A.sum(1)).ravel(); A = A.tocsr()
        nb = lambda x: set(A.indices[A.indptr[x]:A.indptr[x + 1]].tolist())  # noqa: E731
        ov, ov0, ed, ed0, used = [], [], [], [], 0
        for a in tw:
            b = int(j[a])
            if deg[a] < 3 or deg[b] < 3:
                continue
            na, nbb = nb(a), nb(b); pool = np.flatnonzero((lab == lab[b]) & (deg >= deg[b] / 1.5) & (deg <= deg[b] * 1.5)); pool = pool[(pool != a) & (pool != b)]
            if len(pool) < 5:
                continue
            used += 1; ov.append(len(na & nbb) / min(len(na), len(nbb))); ed.append(float(b in na))
            pick = pool if len(pool) <= 20 else rng.choice(pool, 20, replace=False); ov0.append(float(np.mean([len(na & nb(c)) / min(len(na), len(nb(c))) for c in pick]))); ed0.append(float(np.mean([c in na for c in pick])))
        rep["twins"] = {"pairs": int(len(tw)), "used": used, "overlap": float(np.mean(ov)) if ov else None, "overlap_random": float(np.mean(ov0)) if ov0 else None, "any_overlap": float(np.mean(np.array(ov) > 0)) if ov else None, "direct": float(np.mean(ed)) if ed else None, "direct_random": float(np.mean(ed0)) if ed0 else None,
                        "same_level": float((lab[tw] == lab[j[tw]]).mean())}
        print(f"\n=== (3) twins: {len(tw)} mutual nearest pairs at similarity >= 0.9 (same level {rep['twins']['same_level']:.3f}); with >= 3 neighbours on both sides: {used}")
        if used:
            print(f"   overlap of the two opponent sets (share of the smaller set): twins {np.mean(ov):.3f} (some overlap in {np.mean(np.array(ov) > 0):.3f}); random pair of the same levels and similar degree {np.mean(ov0):.3f}")
            print(f"   the two are a recognised pair themselves: twins {np.mean(ed):.3f}; random {np.mean(ed0):.4f}")
    except FileNotFoundError as e:
        print(f"\n(3) twins skipped: {e}")
    # (4) use for identification
    print("\n=== (4) leave one owner out: the hidden opponent against 50 random candidates of the same level and similar degree")
    for name, sel in (("q95 and above", TIER >= 5), ("all (q90 and above)", TIER >= 4)):
        ow, op = OW[sel], OP[sel]; r = link_auc(ow, op, lab, C, np.random.default_rng(1590)); r0 = link_auc(ow, shuffle(ow, op, lab, np.random.default_rng(1591)), lab, C, np.random.default_rng(1590)); rep[f"link {name}"] = {"real": r, "shuffled": r0}
        print(f"   {name}: {r['owners']} owners (median {r['median_S']:.0f} other opponents); AUC direct pairs {r['auc_direct']:.3f} +/- {r['auc_direct_se']:.3f}, common neighbours {r['auc_common']:.3f} +/- {r['auc_common_se']:.3f}, both {r['auc_both']:.3f} +/- {r['auc_both_se']:.3f}")
        print(f"       a direct pair with one of the others: hidden opponent {r['any_direct_true']:.3f}, random candidate {r['any_direct_random']:.3f}; a common neighbour: {r['any_common_true']:.3f} against {r['any_common_random']:.3f}")
        print(f"       shuffled graph: AUC direct {r0['auc_direct']:.3f}, common {r0['auc_common']:.3f}, both {r0['auc_both']:.3f}; direct pair {r0['any_direct_true']:.3f} against {r0['any_direct_random']:.3f}")
    main_r = rep["q95 and above"]["closure_ratio"]; main_auc = rep["link q95 and above"]["real"]["auc_both"]
    verdict = "circles exist and help identification -> next step" if (main_r >= 2 and main_auc >= 0.65) else ("closed" if (main_r < 1.3 or main_auc < 0.58) else "in between: record, estimate the ceiling")
    print(f"\nREGISTERED reading: closure ratio {main_r:.2f} (>= 2 / < 1.3), AUC {main_auc:.3f} (>= 0.65 / < 0.58) -> {verdict}"); rep["verdict"] = verdict
    (RV / "m159_coplay.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m159_coplay.json")


if __name__ == "__main__":
    sys.exit(main())
