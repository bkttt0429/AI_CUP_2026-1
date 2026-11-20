#!/usr/bin/env python
"""Section 4M M232 part A (2026-11-18), report only: is the label a centre of the period or the peak rank of the period?
Identified even-game pairs inside the training set (results/r_variants/m96_pairs.npz: owner, opp, tier, weight, pos).
For every owner the rows are split into two interleaved halves (by the position in the owner's record): the spread (sd) of
the opponents' labels from one half against the mean (opponent label - own label) from the other half, both centred inside
the owner's level. Peak rule: the slope is negative in the low AND the high levels; wrong identifications alone give opposite
signs on the two sides (their labels lean towards the middle); a centre rule gives zero.
Second reading: the time trend of the opponents' labels (one half) against the mean offset (other half), signed and absolute.

  python -I logs/nano4/m232_peak.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; RANK = {r: i for i, r in enumerate(LEV)}


def boot_slope(x, y, grp, rng, nb=2000):
    """Slope of y on x (both centred already), resampling the owners (grp holds the owner index of every stacked row)."""
    u, inv = np.unique(grp, return_inverse=True); sxy = np.bincount(inv, weights=x * y); sxx = np.bincount(inv, weights=x * x)
    est = sxy.sum() / max(sxx.sum(), 1e-12); bs = []
    for _ in range(nb):
        i = rng.integers(0, len(u), len(u)); bs.append(sxy[i].sum() / max(sxx[i].sum(), 1e-12))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def main():
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "rank"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)
    one = rows.drop_duplicates(); assert one.player_id.is_unique, "a player carries two levels"
    players = np.array(sorted(one.player_id)); ylab = one.set_index("player_id")["rank"].map(RANK).reindex(players).to_numpy(); C = len(players)
    z = np.load(ROOT / "results/r_variants/m96_pairs.npz"); ow, op, tier, pos = z["owner"], z["opp"], z["tier"], z["pos"]
    assert max(int(ow.max()), int(op.max())) < C
    dd = (ylab[op] - ylab[ow]).astype(float); rng = np.random.default_rng(232); rep = {}
    top = tier >= 5; nb_ = np.bincount(ow[top], minlength=C); m_ = nb_ >= 10
    print(f"players {C}; pairs {len(ow)}; check against M96 (1): owners with >= 10 pairs at >= q95: {int(m_.sum())} (M96: 1125), mean offset {float((np.bincount(ow[top], weights=dd[top], minlength=C)[m_] / nb_[m_]).mean()):+.3f} (M96: -0.18); same level {float((dd[top] == 0).mean()):.3f}", flush=True)
    for name, sel, half_min in (("q90", tier >= 4, 4), ("q95", tier >= 5, 4), ("q98", tier >= 6, 3)):
        o, d, p = ow[sel], dd[sel], pos[sel]; order = np.lexsort((p, o)); o, d, p = o[order], d[order], p[order]
        par = (np.arange(len(o)) - np.searchsorted(o, o, side="left")) % 2

        def stats(mask):
            n = np.bincount(o[mask], minlength=C).astype(float); s = np.bincount(o[mask], weights=d[mask], minlength=C); q = np.bincount(o[mask], weights=d[mask] ** 2, minlength=C)
            sp = np.bincount(o[mask], weights=p[mask], minlength=C); spp = np.bincount(o[mask], weights=p[mask] ** 2, minlength=C); spd = np.bincount(o[mask], weights=p[mask] * d[mask], minlength=C)
            with np.errstate(all="ignore"):
                m = s / n; sd = np.sqrt(np.maximum((q / n - m * m) * n / (n - 1), 0)); vp = spp / n - (sp / n) ** 2; tr = (spd / n - (sp / n) * m) / np.where(vp > 1e-4, vp, np.nan)
            return n, m, sd, tr
        H = [stats(par == k) for k in (0, 1)]; keep = (H[0][0] >= half_min) & (H[1][0] >= half_min); idx = np.flatnonzero(keep); cls = ylab[idx]
        cen = lambda a: a - np.array([np.nanmean(a[cls == c]) if (cls == c).any() else 0.0 for c in range(10)])[cls]  # noqa: E731  centred inside the owner's level
        M = [cen(H[k][1][idx]) for k in (0, 1)]; S = [cen(H[k][2][idx]) for k in (0, 1)]; T = [H[k][3][idx] for k in (0, 1)]
        ok_t = np.isfinite(T[0]) & np.isfinite(T[1]); Tc = [cen(np.where(ok_t, T[k], np.nan)) for k in (0, 1)]; Ta = [cen(np.where(ok_t, np.abs(T[k]), np.nan)) for k in (0, 1)]
        x = np.concatenate([S[0], S[1]]); y = np.concatenate([M[1], M[0]]); g = np.concatenate([idx, idx]); cl2 = np.concatenate([cls, cls])
        out = {"owners": int(len(idx)), "rows_per_owner_median": float(np.median(H[0][0][idx] + H[1][0][idx])), "sd_mean": float(np.mean((H[0][2][idx] + H[1][2][idx]) / 2))}
        print(f"\n== {name}: {len(idx)} owners with >= {half_min} pairs in each half (median {out['rows_per_owner_median']:.0f} pairs); mean sd of the opponents' labels {out['sd_mean']:.3f}; split-half correlation of the sd {float(np.corrcoef(S[0], S[1])[0, 1]):+.3f}, of the mean offset {float(np.corrcoef(M[0], M[1])[0, 1]):+.3f}")
        for lab_, msk in (("all levels", np.ones(len(x), bool)), ("low  D-A", cl2 <= 3), ("high 1D-6D", cl2 >= 4), ("interior C-5D", (cl2 >= 1) & (cl2 <= 8))):
            e, lo, hi = boot_slope(x[msk], y[msk], g[msk], rng); r = float(np.corrcoef(x[msk], y[msk])[0, 1]); out[f"slope_sd|{lab_}"] = (e, lo, hi, r)
            print(f"  mean offset (other half) on sd, {lab_:14s}: slope {e:+.3f} [{lo:+.3f}, {hi:+.3f}]  r {r:+.3f}  (owners {int(msk.sum() // 2)})")
        print("  by level: " + "  ".join(f"{LEV[c]} {boot_slope(x[cl2 == c], y[cl2 == c], g[cl2 == c], rng, 300)[0]:+.2f} ({int((cls == c).sum())})" for c in range(10) if (cls == c).sum() >= 30))
        q1, q2 = np.quantile(S[0], [1 / 3, 2 / 3]); lo_m, hi_m = float(M[1][S[0] <= q1].mean()), float(M[1][S[0] >= q2].mean()); out["tertiles"] = (lo_m, hi_m)
        print(f"  owners in the lowest / highest third of the sd (first half): mean offset in the second half {lo_m:+.3f} / {hi_m:+.3f} levels (difference {hi_m - lo_m:+.3f})")
        f = np.concatenate([ok_t, ok_t]); xt = np.concatenate([Tc[0], Tc[1]])[f]; xa = np.concatenate([Ta[0], Ta[1]])[f]; yt = y[f]; gt = g[f]
        e, lo, hi = boot_slope(xt, yt, gt, rng); out["slope_trend"] = (e, lo, hi); e2, lo2, hi2 = boot_slope(xa, yt, gt, rng); out["slope_abs_trend"] = (e2, lo2, hi2)
        print(f"  mean offset (other half) on the time trend of the opponents' labels (levels over the whole record): signed {e:+.4f} [{lo:+.4f}, {hi:+.4f}], absolute {e2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]; mean trend {float(np.nanmean(np.where(ok_t, (T[0] + T[1]) / 2, np.nan))):+.3f}, split-half correlation of the trend {float(np.corrcoef(Tc[0][ok_t], Tc[1][ok_t])[0, 1]):+.3f}")
        rep[name] = out
    (ROOT / "results/r_variants/m232_peak.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nREPORT ONLY; written m232_peak.json")


if __name__ == "__main__":
    sys.exit(main())
