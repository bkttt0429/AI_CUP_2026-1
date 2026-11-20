#!/usr/bin/env python
"""Section 4M M190 pilot (2026-11-16, report only): does the fused score need a slope that depends on the candidate's games?

A prototype built from few games is a noisier direction, so the true owner's standardised score is lower for a candidate
with 50 games than for one with 500 - while the score of a stranger is the same. The fused score is  F = sum_k w_k z_k +
beta log n  with ONE slope for all candidates; the likelihood-ratio form for a candidate whose true matches score
N(mu(n), 1) against strangers N(0, 1) is  mu(n) z - mu(n)^2 / 2 + log prior(n): slope and offset both depend on n.
On the held-out own games (3,244, the owner known; ct112's identifier SQ4, stored weights):
(a) the owner's fused standardised score (before the prior) by the owner's number of games: mean, sd; the strangers' top
    scores for reference;
(b) fit, by the likelihood weighted with the owner's games (as the prior beta was fitted),
      F' = exp(a0 + a1 log n) * Z + b1 log n + b2 (log n)^2
    against the current one-parameter form (beta alone), out of fold over the OWNERS (five folds): log-likelihood, owner
    first (weighted by games / unweighted / by band), and the share of the owner-first queries above a line placed so that
    the same share of wrong firsts passes as now (a like-for-like "recognised" share).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

CACHE, RV = mb7.CACHE, ROOT / "results/r_variants"


def std(S, has=None):
    if has is not None:
        out = np.zeros_like(S); out[:, has] = std(S[:, has]); return out
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); sd[sd < 1e-12] = 1.0; return (S - mu) / sd


def main():
    q = np.load(CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); players = q["players"].astype(str); C = len(players); truth = q["held_truth"].astype(int); hq = np.arange(2 * Q, 2 * Q + H)
    ev = json.loads((RV / "m84_SQ4_eval.json").read_text()); w = ev["weights"]; beta0 = float(ev["beta"]); ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players).to_numpy()
    Z = w[0] * np.load(CACHE / "block_held.npy").astype(np.float32)
    for k, v in enumerate(ev["sources"]):
        V = np.load(CACHE / f"var_{v}.npz"); M = V["protos"].shape[0]; z = np.zeros((H, C), np.float32)
        for m in range(M):
            z += std(V["qvec"][m, hq].astype(np.float32) @ V["protos"][m, 4].astype(np.float32).T, V["has"][m, 4] if "has" in V.files else None)
        Z += w[k + 1] * z / M
    ar = np.arange(H); zt = Z[ar, truth]; nt = ng[truth]; Zs = np.sort(Z, axis=1)
    print("(a) the owner's fused score before the prior, by the owner's games (queries | mean | sd | share above the strangers' best) - strangers' best: median " + f"{float(np.median(np.where(Z.argmax(1) == truth, Zs[:, -2], Zs[:, -1]))):.2f}")
    for lo, hi in ((1, 30), (30, 60), (60, 100), (100, 200), (200, 400), (400, 10 ** 9)):
        m = (nt >= lo) & (nt < hi)
        if m.sum() >= 20:
            best_other = np.where(Z.argmax(1) == truth, Zs[:, -2], Zs[:, -1]); print(f"    [{lo}, {hi if hi < 10 ** 9 else 'inf'}): {int(m.sum())} | {float(zt[m].mean()):.2f} | {float(zt[m].std()):.2f} | {float((zt[m] > best_other[m]).mean()):.3f}")
    Zt = torch.tensor(Z, dtype=torch.float64); ln = torch.tensor(np.log(ng), dtype=torch.float64); tr = torch.tensor(truth); wa = torch.tensor(nt / nt.sum(), dtype=torch.float64)
    owners = np.unique(truth); rng = np.random.default_rng(0); fo_ = dict(zip(owners, rng.integers(0, 5, len(owners)))); fo = np.array([fo_[t] for t in truth])

    def F_of(th, full):
        if full:
            return torch.exp(th[0] + th[1] * (ln - ln.mean()))[None] * Zt + th[2] * ln[None] + th[3] * (ln[None] - ln.mean()) ** 2
        return Zt + th[0] * ln[None]

    def fit(mask, full):
        th = torch.zeros(4 if full else 1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([th], lr=0.5, max_iter=200, line_search_fn="strong_wolfe"); idx = torch.tensor(np.flatnonzero(mask)); ww = wa[idx] / wa[idx].sum()
        def closure():
            opt.zero_grad(); F = F_of(th, full)[idx]; loss = -(ww * torch.log_softmax(F, dim=1)[torch.arange(len(idx)), tr[idx]]).sum() + 1e-4 * (th ** 2).sum(); loss.backward(); return loss
        opt.step(closure); return th.detach()
    res = {}
    for name, full in (("current form (beta log n)", False), ("slope and offset by games", True)):
        Fo = torch.zeros_like(Zt); ths = []
        for k in range(5):
            th = fit(fo != k, full); ths.append(th.numpy()); Fo[torch.tensor(np.flatnonzero(fo == k))] = F_of(th, full)[torch.tensor(np.flatnonzero(fo == k))]
        F = Fo.numpy(); ll = float((wa * torch.log_softmax(Fo, dim=1)[torch.arange(H), tr]).sum()); first = F.argmax(1) == truth; top = -np.partition(-F, 5, axis=1)[:, :5]; top.sort(axis=1); top = top[:, ::-1]; m15 = top[:, 0] - top[:, 1:5].mean(1)
        res[name] = (first, m15); print(f"\n(b) {name}: parameters (mean of the five fits) " + ", ".join(f"{x:+.3f}" for x in np.mean(ths, 0)) + f"; out-of-fold log-likelihood (weighted) {ll:.4f}; owner first: weighted {float((wa.numpy() * first).sum()):.3f}, unweighted {float(first.mean()):.3f}; by the owner's games " + "  ".join(f"[{lo},{hi if hi < 10 ** 9 else 'inf'}) {float(first[(nt >= lo) & (nt < hi)].mean()):.3f}" for lo, hi in ((30, 100), (100, 300), (300, 1000))))
    f0, m0 = res["current form (beta log n)"]; f1, m1 = res["slope and offset by games"]; wn = wa.numpy()
    for fp in (0.02, 0.034, 0.06):
        t0 = float(np.quantile(m0[~f0], 1 - fp)); t1 = float(np.quantile(m1[~f1], 1 - fp)); print(f"  line letting {fp:.3f} of the wrong firsts pass: owner first and above it (weighted by games) - current {float((wn * (f0 & (m0 >= t0))).sum()):.3f}, new {float((wn * (f1 & (m1 >= t1))).sum()):.3f}; unweighted {float((f0 & (m0 >= t0)).mean()):.3f} -> {float((f1 & (m1 >= t1)).mean()):.3f}")


if __name__ == "__main__":
    main()
