#!/usr/bin/env python
"""Section 4M M188 (2026-11-16, report only): what a second stage of the recogniser would have to fix.

Held-out own games of the pool players (3,244 queries, the owner known), fused scores of ct112's identifier SQ4 (block + S6at +
S7t + S7tc + activity prior, stored weights). All rates weighted by the owner's number of games (opponents are met in
proportion to activity) and unweighted.
(a) where the owner ranks: first / 2-5 / 6-20 / beyond; and the margin statistic: owner first AND above the null's q95
    ("recognised") against owner first but below it ("right but not believed").
(b) when the owner is not first, who is: an alt account of the owner (prototype similarity >= 0.85 to the owner's), a much
    busier candidate (>= 3 x the owner's games), a candidate of the owner's level, another;
    when the owner is first but below the line: is the runner-up an alt account of the owner.
(c) what merging alt accounts into one candidate would do: owner-or-alt first, and the margin after the alt is removed from
    the list (share above the same q95 line).
(d) by the owner's number of games: first / recognised - where the misses concentrate.
The null line: the test target side's q95 of m15 (stored identifier file).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

CACHE, RV = mb7.CACHE, ROOT / "results/r_variants"


def std(S, has=None):
    if has is not None:
        out = np.zeros_like(S); out[:, has] = std(S[:, has]); return out
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); sd[sd < 1e-12] = 1.0; return (S - mu) / sd


def main():
    q = np.load(CACHE / "queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); players = q["players"].astype(str); plab = q["plab"].astype(int); C = len(players); truth = q["held_truth"].astype(int); hq = np.arange(2 * Q, 2 * Q + H)
    ev = json.loads((RV / "m84_SQ4_eval.json").read_text()); w = ev["weights"]; beta = float(ev["beta"]); ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players).to_numpy()
    F = w[0] * np.load(CACHE / "block_held.npy").astype(np.float32); Pm = 0
    for k, v in enumerate(ev["sources"]):
        V = np.load(CACHE / f"var_{v}.npz"); M = V["protos"].shape[0]; z = np.zeros((H, C), np.float32)
        for m in range(M):
            P = V["protos"][m, 4].astype(np.float32); z += std(V["qvec"][m, hq].astype(np.float32) @ P.T, V["has"][m, 4] if "has" in V.files else None)
            if v == "S7t":
                Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12); Pm = Pm + Pn @ Pn.T / M
        F += w[k + 1] * z / M
    F += beta * np.log(ng)[None].astype(np.float32); np.fill_diagonal(Pm, -1.0)
    zz = np.load(RV / "m127_SQ4.npz"); st_all = zz["TV"][:, 0] - zz["TV"][:, 1:5].mean(1); line = float(np.quantile(st_all[2 * Q + T:2 * Q + 2 * T], 0.95)); hv = zz["HV"]; assert abs(float(np.sort(F, 1)[:, -1].mean() - hv[:, 0].mean())) < 0.05, (float(np.sort(F, 1)[:, -1].mean()), float(hv[:, 0].mean()))
    order = np.argsort(-F, axis=1)[:, :21]; ar = np.arange(H); rank = (F > F[ar, truth][:, None]).sum(1); top = np.take_along_axis(F, order, 1); m15 = top[:, 0] - top[:, 1:5].mean(1); first = order[:, 0]; second = order[:, 1]
    wa = ng[truth] / ng[truth].sum(); wu = np.full(H, 1.0 / H); alt_of = Pm[truth] >= 0.85  # [H, C] candidates that are alt accounts of the owner
    has_alt = alt_of.any(1); f1 = rank == 0; rec = f1 & (m15 >= line)
    for lab, ww in (("weighted by the owner's games", wa), ("unweighted", wu)):
        s = lambda m: float(ww[m].sum())  # noqa: E731
        print(f"{lab}:")
        print(f"  (a) owner first {s(f1):.3f} | 2-5 {s((rank >= 1) & (rank < 5)):.3f} | 6-20 {s((rank >= 5) & (rank < 20)):.3f} | beyond {s(rank >= 20):.3f}; recognised (first and above the q95 line {line:.2f}) {s(rec):.3f}; first but below the line {s(f1 & ~rec):.3f}; a wrong first above the line {s(~f1 & (m15 >= line)):.3f}")
        nf = ~f1; alt1 = alt_of[ar, first]; busy = ng[first] >= 3 * ng[truth]; same = plab[first] == plab[truth]
        print(f"  (b) owner not first ({s(nf):.3f}): the first is an alt account of the owner {s(nf & alt1) / s(nf):.3f}, a candidate with >= 3 x the owner's games {s(nf & ~alt1 & busy) / s(nf):.3f}, another of the owner's level {s(nf & ~alt1 & ~busy & same) / s(nf):.3f}, other {s(nf & ~alt1 & ~busy & ~same) / s(nf):.3f}; first's level equals the owner's {s(nf & same) / s(nf):.3f}, within one {s(nf & (np.abs(plab[first] - plab[truth]) <= 1)) / s(nf):.3f}")
        low = f1 & ~rec; print(f"      owner first but below the line ({s(low):.3f}): the runner-up is an alt account {s(low & alt_of[ar, second]) / max(s(low), 1e-9):.3f}; owners with an alt account at all {s(has_alt):.3f}")
        # (c) merge the alt accounts of the owner into the owner: take the best of them, drop the others from the list
        Fm = F.copy(); best_alt = np.where(alt_of, F, -np.inf).max(1); Fm[ar, truth] = np.maximum(F[ar, truth], best_alt); Fm[alt_of] = -np.inf; om = np.argsort(-Fm, axis=1)[:, :5]; tm = np.take_along_axis(Fm, om, 1); m15m = tm[:, 0] - tm[:, 1:5].mean(1); f1m = om[:, 0] == truth
        print(f"  (c) alt accounts merged into the owner: owner-or-alt first {s(f1m):.3f} (was {s(f1):.3f}); recognised {s(f1m & (m15m >= line)):.3f} (was {s(rec):.3f})")
    print("(d) by the owner's number of games (queries | owner first | recognised | owner in the first five):")
    for lo, hi in ((1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)):
        m = (ng[truth] >= lo) & (ng[truth] < hi); print(f"    [{lo}, {hi if hi < 10 ** 9 else 'inf'}): {int(m.sum())} | {float(f1[m].mean()):.3f} | {float(rec[m].mean()):.3f} | {float((rank[m] < 5).mean()):.3f}   share of the encounters (weights) {float(wa[m].sum()):.3f}")
    # how separable are "right" and "wrong" first candidates beyond m15: gap to the second, and agreement of the sources
    print(f"owner first: median m15 {float(np.median(m15[f1])):.2f}; wrong first: median m15 {float(np.median(m15[~f1])):.2f}; share of wrong firsts above the line {float((m15[~f1] >= line).mean()):.3f}; share of right firsts above the line {float((m15[f1] >= line).mean()):.3f}")


if __name__ == "__main__":
    main()
