#!/usr/bin/env python
"""Section 4M M234 (2026-11-18), report only: which labelling rule fits our own data, without using any time order?
Even games pair two players of the same displayed rank, so (opponent's label - owner's label) d = e_owner - e_opponent with
e = (displayed rank at the game) - (label). Several identified opponents of one owner give the within-owner variance A, the
stable between-owner variance V_o = Var(owner's centre - label), the shares P(d = 0), P(|d| = 1) and owner-level shares.
A rounded centre (mean / median / mode) keeps the centre within half a level of the label (V_o about 1/12 whatever the
fluctuation); a snapshot makes the label one of the player's ranks (V_o about the fluctuation variance, more when ranks move
slowly); the maximum puts every offset on one side.
Data: validation games, identifier SQ4 (statistic m15), even games in the calibrated bands with the highest share of true pairs.
Simulation: rating = c + AR(1) (sd s, correlation phi) + linear drift (sd g over the record), rank = floor; labels by six
rules; readings generated with the data's readings per owner, purities and false-match table; grid search per rule.

  python -I logs/nano4/m234_rule.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

m3, b, s2, FOLDS, STAT = mb7.m3, mb7.b, mb7.s2, mb7.FOLDS, mb7.STAT
m87 = m3.m87
LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; RULES = ("mean", "median", "mode", "max", "last", "middle")
KEYS = ("p0", "p1", "var", "A", "Vo", "round0", "nonzero_major", "big")


def stats(owner, d, cls=None):
    """The statistic vector of a set of readings: owner index (ints), d (ints), cls = class of every reading's owner (for centring)."""
    d = d.astype(float); dc = d.copy()
    if cls is not None:
        for c in np.unique(cls):
            dc[cls == c] -= d[cls == c].mean()
    else:
        dc -= d.mean()
    u, inv, n = np.unique(owner, return_inverse=True, return_counts=True); s1 = np.bincount(inv, weights=dc); s2_ = np.bincount(inv, weights=dc * dc); m = s1 / n
    k = n >= 2; ssw = float((s2_[k] - n[k] * m[k] ** 2).sum()); dfw = float((n[k] - 1).sum()); A = ssw / max(dfw, 1)
    N = float(n[k].sum()); g = int(k.sum()); gm = float((n[k] * m[k]).sum() / N); msb = float((n[k] * (m[k] - gm) ** 2).sum() / max(g - 1, 1)); n0 = (N - float((n[k] ** 2).sum()) / N) / max(g - 1, 1); Vo = (msb - A) / max(n0, 1e-9)
    raw = m; k3 = n >= 3; z0 = np.bincount(inv, weights=(d == 0)); best_nz = np.zeros(len(u))  # owner means of the centred readings: the data's small mean shift does not enter
    for x in (-3, -2, -1, 1, 2, 3):
        best_nz = np.maximum(best_nz, np.bincount(inv, weights=(d == x)))
    return {"p0": float((d == 0).mean()), "p1": float((np.abs(d) == 1).mean()), "var": float(dc.var()), "A": A, "Vo": Vo, "round0": float((np.floor(raw[k3] + 0.5) == 0).mean()),
            "nonzero_major": float((best_nz[k3] > z0[k3]).mean()), "big": float((np.abs(raw[k3]) >= 1.0).mean())}


def simulate(s, phi, g, rng, N=6000, H=300):
    """Rank histories [N, H] (ints, relative ladder) and the six labels."""
    c = rng.random(N); x = rng.normal(0, s, N); R = np.empty((N, H)); sl = rng.normal(0, g, N) if g else np.zeros(N); q = np.sqrt(max(1 - phi * phi, 0.0))
    for t in range(H):
        R[:, t] = c + x + sl * (t / (H - 1) - 0.5); x = phi * x + q * rng.normal(0, s, N)
    B = np.floor(R).astype(np.int64); lo = B.min(); cnt = np.stack([(B == v).sum(1) for v in range(lo, B.max() + 1)], 1)
    lab = {"mean": np.floor(B.mean(1) + 0.5).astype(np.int64), "median": np.floor(np.median(B, 1) + 0.5).astype(np.int64), "mode": cnt.argmax(1) + lo, "max": B.max(1), "last": B[:, -1], "middle": B[:, H // 2]}
    return B, lab


def readings(B, lab, n_own, pur, p0_vals, rng):
    """Readings d = e_owner(t) - e_opponent(t') for owners with n_own readings each; a reading is a true pair with its purity, else a false match drawn from p0_vals."""
    N, H = B.shape; own = np.repeat(np.arange(len(n_own)) % N, n_own); M = len(own); e = B - lab[:, None]
    d = e[own, rng.integers(0, H, M)] - e[rng.integers(0, N, M), rng.integers(0, H, M)]; false = rng.random(M) >= pur[rng.integers(0, len(pur), M)]
    d[false] = p0_vals[rng.integers(0, len(p0_vals), int(false.sum()))]
    return np.repeat(np.arange(len(n_own)), n_own), d


def main():
    rng = np.random.default_rng(234); V, _, _, used = b.sources([f"SQ4~{STAT}"]); z = V[f"SQ4~{STAT}"]; parts = []
    for v in FOLDS:
        tr = z["fold"] != v; edges, cal, _ = m87.corrected(z, tr); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
        band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); a = np.zeros(len(te))
        for k in (0, 1, 2):
            a[g == k] = cal[k]["a"][band[g == k]]
        parts.append((z["pid"][te].astype(str), z["y"][te].astype(int), z["opp_lab"][te].astype(int), a, g, band))
        if v == FOLDS[0]:
            print("fold 1 calibration, even games: true-pair share by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]) + " | share of the games " + " ".join(f"{x:.3f}" for x in cal[0].get("share", [])), flush=True)
    pid, y, lab, a, g, band = (np.concatenate([p[i] for p in parts]) for i in range(6)); ev = g == 0; d_all = lab - y; rep = {}
    print(f"validation games {len(y)}, even {int(ev.sum())}; players {len(np.unique(pid))}", flush=True)
    for cname, lo_c, hi_c in (("2D-4D", 5, 7), ("1D-5D", 4, 8)):
        cls_ok = (y >= lo_c) & (y <= hi_c); null = ev & cls_ok & (band <= 1); p0_vals = d_all[null]
        for amin in (0.9, 0.75):
            sel = ev & cls_ok & (a >= amin)
            if sel.sum() < 2000:
                print(f"\n{cname}, a >= {amin}: only {int(sel.sum())} readings - skipped"); continue
            own = np.unique(pid[sel], return_inverse=True)[1]; d = d_all[sel]; cls = y[sel]; pur = a[sel]; n_own = np.bincount(own); T = stats(own, d, cls); bs = []
            groups = [np.flatnonzero(own == i) for i in range(len(n_own))]
            for _ in range(200):
                pick = rng.integers(0, len(groups), len(groups)); idx = np.concatenate([groups[i] for i in pick]); bs.append(stats(np.repeat(np.arange(len(pick)), [len(groups[i]) for i in pick]), d[idx], cls[idx]))
            se = {k: float(np.std([x[k] for x in bs])) for k in KEYS}
            print(f"\n== owners {cname}, even games with true-pair share >= {amin}: {int(sel.sum())} readings of {len(n_own)} owners (mean purity {float(pur.mean()):.3f}; readings per owner: mean {float(n_own.mean()):.1f}, >= 3: {int((n_own >= 3).sum())} owners); mean d {float(d.mean()):+.3f}; false-match table from {int(null.sum())} low-band games (same level {float((p0_vals == 0).mean()):.3f}, sd {float(p0_vals.std()):.2f})")
            print("   data          : " + "  ".join(f"{k} {T[k]:.3f}±{se[k]:.3f}" for k in KEYS) + f"   | Vo sd {np.sqrt(max(T['Vo'], 0)):.3f}, implied fluctuation variance (A - Vo) / 2 = {(T['A'] - T['Vo']) / 2:.3f}", flush=True)
            best = {r: (1e18, None, None) for r in RULES}; agree = {}
            for s in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8):
                for phi in (0.0, 0.9, 0.98, 0.995):
                    for gd in (0.0, 0.5, 1.0):
                        B, L = simulate(s, phi, gd, rng)
                        for r in RULES:
                            o_, d_ = readings(B, L[r], n_own, pur, p0_vals, rng); S = stats(o_, d_); chi = float(sum(((S[k] - T[k]) / max(se[k], 1e-4)) ** 2 for k in KEYS))
                            if chi < best[r][0]:
                                best[r] = (chi, (s, phi, gd), S); agree[r] = {q: float((L[r] == L[q]).mean()) for q in RULES}; agree[r]["w2"] = float(B.var(1).mean())
            order = sorted(RULES, key=lambda r: best[r][0])
            for r in order:
                chi, par, S = best[r]
                print(f"   {r:7s} misfit {chi:7.1f}  best s {par[0]}, phi {par[1]}, drift {par[2]} (rank variance within a player {agree[r]['w2']:.3f}): " + "  ".join(f"{k} {S[k]:.3f} ({(S[k] - T[k]) / max(se[k], 1e-4):+.1f})" for k in KEYS), flush=True)
            for r0 in order[:2]:
                print(f"   at the best setting of '{r0}': share of players whose label under '{r0}' equals the label under " + ", ".join(f"{q} {agree[r0][q]:.3f}" for q in RULES if q != r0), flush=True)
            rep[f"{cname}|a>={amin}"] = {"data": T, "se": se, "n_readings": int(sel.sum()), "n_owners": int(len(n_own)), "purity": float(pur.mean()), "fits": {r: {"misfit": best[r][0], "params": best[r][1], "stats": best[r][2], "agree": agree[r]} for r in RULES}}
    (ROOT / "results/r_variants/m234_rule.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nREPORT ONLY; written m234_rule.json")


if __name__ == "__main__":
    sys.exit(main())
