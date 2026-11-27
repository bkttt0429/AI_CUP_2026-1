#!/usr/bin/env python
"""Section 4M M140 (2026-11-14, report only): opponent-side to opponent-side matching inside a validation fold (pilot of the
two-hop anchors). For every validation game the most similar opponent-side vector among the OTHER players' games of the same fold
(same network); bands of that similarity; the reading (the other player's level against this player's level, even games only),
split by whether the direct anchor recognised the opponent; reliability against confident direct identifications; coverage.

  python -I logs/nano4/m140_twohop_pilot.py [--var S4e]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; CACHE = ROOT / "results/m84_cache"
BANDS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var", nargs="+", default=["S4e", "S2e"]); a_ = ap.parse_args()
    d = np.load(RV / "m82_val.npz"); key, pid, fold, y, hd = d["key"].astype(str), d["pid"].astype(str), d["fold"], d["y"], d["hd"]; b5 = d["f5e_opp_best"].astype(str)
    s2 = np.load(RV / "m84_S2_val.npz"); bs, ms = s2["f_opp_best"].astype(str), s2["f_opp_m"]
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), key)
    conf = (ms > np.quantile(ms, 0.95)) & (bs == b5)          # the direct anchor is sure who the opponent is (two independent identifiers agree)
    unrec = (ms <= np.quantile(ms, 0.75)) & (bs != b5)        # the direct anchor has nothing: low margin and no agreement
    print(f"{Q} validation games; direct anchor sure of the opponent {conf.mean():.3f}, nothing {unrec.mean():.3f}, in between {1 - conf.mean() - unrec.mean():.3f}; even games {float((hd == 0).mean()):.3f}")
    rep = {}
    for var in a_.var:
        QV = np.load(CACHE / f"var_{var}.npz")["qvec"]; M = QV.shape[0]; top_s = np.full(Q, -1.0); top_j = np.full(Q, -1, np.int64); sec_s = np.full(Q, -1.0); n5 = np.zeros((Q, 5), np.int64); s5 = np.zeros((Q, 5))
        pos, neg = [], []
        for v in (1, 2, 3, 4):
            idx = np.flatnonzero(fold == v); X = QV[:, idx].astype(np.float32); X /= np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-9); pv = pid[idx]; cv, idv = conf[idx], bs[idx]
            for a in range(0, len(idx), 2000):
                S = sum(X[m, a:a + 2000] @ X[m].T for m in range(M)) / M
                S[pv[a:a + 2000, None] == pv[None, :]] = -2.0   # never a game of the same player
                part = np.argpartition(-S, 5, axis=1)[:, :5]; ps = np.take_along_axis(S, part, 1); o = np.argsort(-ps, axis=1); part, ps = np.take_along_axis(part, o, 1), np.take_along_axis(ps, o, 1)
                top_s[idx[a:a + 2000]], top_j[idx[a:a + 2000]], sec_s[idx[a:a + 2000]] = ps[:, 0], idx[part[:, 0]], ps[:, 1]; n5[idx[a:a + 2000]], s5[idx[a:a + 2000]] = idx[part], ps
                rows = np.flatnonzero(cv[a:a + 2000])            # reliability: pairs of games whose opponents the direct anchor is sure about
                if len(rows):
                    cols = np.flatnonzero(cv); Sc = S[np.ix_(rows, cols)]; same = idv[a:a + 2000][rows][:, None] == idv[cols][None, :]; ok = Sc > -1.5
                    pos.append(Sc[same & ok]); sub = Sc[~same & ok]; neg.append(sub[:: max(1, sub.size // 200000)])
        pos, neg = np.concatenate(pos), np.concatenate(neg)
        print(f"\n=== vectors {var} ({M} members)")
        print(f"reliability against sure direct identifications: same opponent {len(pos)} pairs, similarity median {np.median(pos):.3f} (10% {np.quantile(pos, 0.1):.3f}); different opponents similarity median {np.median(neg):.3f}, 99% {np.quantile(neg, 0.99):.3f}, 99.9% {np.quantile(neg, 0.999):.3f}, 99.99% {np.quantile(neg, 0.9999):.3f}")
        for t in (0.5, 0.6, 0.7, 0.8):
            print(f"    similarity >= {t}: share of same-opponent pairs kept {float((pos >= t).mean()):.3f}; share of different-opponent pairs passing {float((neg >= t).mean()):.2e}")
        ev = (hd == 0) & (hd[top_j] == 0); dl = np.abs(y - y[top_j]); cuts = np.quantile(top_s, BANDS); edges = np.r_[-np.inf, cuts, np.inf]
        rng = np.random.default_rng(0); rj = rng.integers(0, Q, Q); rb = (hd == 0) & (hd[rj] == 0) & (pid != pid[rj]) & (fold == fold[rj])
        print(f"random pairs of even games in a fold: same level {float((y[rb] == y[rj][rb]).mean()):.3f}, within one {float((np.abs(y[rb] - y[rj][rb]) <= 1).mean()):.3f}")
        print("band of the best similarity | all games: n, same level, within one | direct anchor has NOTHING: n, same, within one | direct anchor SURE: n, within one, neighbour sure and same person")
        rows = []
        for i in range(len(edges) - 1):
            m = (top_s > edges[i]) & (top_s <= edges[i + 1]) & ev; mu, mc = m & unrec, m & conf; both = mc & conf[top_j]
            f = lambda k: (int(k.sum()), float((dl[k] == 0).mean()) if k.any() else float("nan"), float((dl[k] <= 1).mean()) if k.any() else float("nan"))  # noqa: E731
            A, U, C = f(m), f(mu), f(mc); sp = float((bs[both] == bs[top_j][both]).mean()) if both.any() else float("nan")
            lab = f"q{BANDS[i - 1] if i else 0:g}-{BANDS[i] if i < len(BANDS) else 1:g} (sim {max(edges[i], -1):.2f}..{min(edges[i + 1], 1):.2f})"
            print(f"  {lab:34s} | {A[0]:6d} {A[1]:.3f} {A[2]:.3f} | {U[0]:6d} {U[1]:.3f} {U[2]:.3f} | {C[0]:6d} {C[2]:.3f} {sp:.3f} ({int(both.sum())})")
            rows.append(dict(band=lab, all=A, unrecognised=U, sure=C, same_person_when_both_sure=sp))
        for t in (0.5, 0.6, 0.7):
            cu = float((top_s[unrec] >= t).mean()); ca = float((top_s >= t).mean())
            k = unrec & (top_s >= t) & ev; print(f"coverage at similarity >= {t}: all games {ca:.4f}, games where the direct anchor has nothing {cu:.4f} (their reading: same level {float((dl[k] == 0).mean()) if k.any() else float('nan'):.3f}, within one {float((dl[k] <= 1).mean()) if k.any() else float('nan'):.3f}, n {int(k.sum())}); with 40 times the candidates (independent approximation) {1 - (1 - cu) ** 40:.3f}")
        rep[var] = rows
    (RV / "m140_twohop_pilot.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nwritten results/r_variants/m140_twohop_pilot.json")


if __name__ == "__main__":
    main()
