"""Section 4M diagnostic (2026-11-11, report only): where is the anchor channel's bottleneck? For the HD0 opponent queries of the
current identifier (5e), per margin band: the share of queries, the mixture weight a_b used today (from the excess over the impostor
quantiles) and an independent, label-based estimate pi_b of the share of real matches in that band (least-squares fit of the band's
label-difference histogram as pi K + (1 - pi) P0, P0 = the lowest band's histogram, K = the top bands' real-match distribution).
The sum over bands gives the mass of correct top-1 matches that exists vs the mass the evidence actually uses.
  python -I logs/nano4/m87_bottleneck_diag.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m71_step2 as s2  # noqa: E402

d = dict(np.load(ROOT / "results/r_variants/m82_val.npz"))
for name, pre, top1 in (("five + matcher (ct44)", "f5e_", 0.268), ("five components (ct42)", "f5_", 0.157)):
    z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d[pre + "opp_lab"], "opp_m": d[pre + "opp_m"], "imp_lab": d[pre + "imp_lab"], "imp_m": d[pre + "imp_m"]}
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0; dd = np.clip(z["opp_lab"] - z["y"] + 9, 0, 18); nb = len(s2.QS) + 1
    H = np.stack([np.bincount(dd[s & (band == b)], minlength=19) / max((s & (band == b)).sum(), 1) for b in range(nb)]); share = np.array([(s & (band == b)).sum() for b in range(nb)]) / s.sum()
    P0, K = H[0], cal[0]["K"]; v = K - P0; pi = np.clip(((H - P0) @ v) / (v @ v), 0, 1); a = cal[0]["a"]
    print(f"{name}: HD0 opponent queries {int(s.sum())}")
    print("  band (impostor quantile)  share   a used   label-based real share   exact-label rate")
    for b, lab in enumerate(("< q50", "q50-80", "q80-90", "q90-95", "q95-98", "q98-99", "q99-99.5", "q99.5-99.9", ">= q99.9")):
        print(f"  {lab:12s} {share[b]:7.4f}  {a[b]:5.2f}   {pi[b]:5.2f}   {H[b][9]:.3f}")
    used, real = float((share * a).sum()), float((share * pi).sum()); low = float((share[1:3] * pi[1:3]).sum())
    print(f"  mass of real matches the evidence uses (sum share x a): {used:.4f}; label-based estimate of real top-1 matches (bands above q50, lowest band taken as pure null): {real:.4f}; of which in the two bands with a ~ 0 (q50-q90): {low:.4f}")
    print(f"  implied share of games whose opponent is a training player = real / top-1 accuracy ({top1}): {real / top1:.2f}  (the held-out top-1 is measured on in-pool players the network has seen; a lower true rate makes this share larger)")
