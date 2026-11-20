#!/usr/bin/env python
"""Section 4M M194 (2026-11-16, report only): four cheap checks suggested in the outside discussion, on the per-game vectors
of M192 (all-players networks, seeds 30 / 31; neural score, standardised over the candidates, + 0.4 log games).

(a) train against held-out, like for like: one random TRAINED game of every held-out query's owner, scored against the same
    candidates with the owner's prototype rebuilt WITHOUT that game - against the held-out game of the same owner.
(b) is it the number of support games? the owners with >= 300 games get their prototype rebuilt from 16 / 32 / 64 / 128
    random games (prior kept at the true number of games); their held-out queries.
(c) learned centres against prototypes: the owner first under each, the overlap of the misses and of the wrong first.
(d) hubs: do a few candidates take more wrong firsts than their share of the games; a candidate-side correction (score
    minus the candidate's mean of its 10 highest scores over a background of 20,000 trained games of other owners).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); W = Path("/work/u2507115/m192"); BETA = 0.4


def nrm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def main():
    m = np.load(W / "meta.npz"); owner, held = m["owner"].astype(np.int64), m["held"]; C = int(owner.max()) + 1; use = np.flatnonzero(~held); hr = np.flatnonzero(held); truth = owner[hr]; ou = owner[use]; H = len(hr); ar = np.arange(H)
    n_c = np.bincount(ou, minlength=C).astype(float); ok = n_c > 0; prior = BETA * np.where(ok, np.log(np.maximum(n_c, 1.0)), 0.0); wa = n_c[truth] / n_c[truth].sum(); rng = np.random.default_rng(0)
    E = {s: np.load(W / f"emb_s{s}.npz") for s in (30, 31)}; own = {s: E[s]["own"].astype(np.float32) for s in E}; hq = {s: E[s]["held"].astype(np.float32) for s in E}; Ssum = {}
    for s in E:
        S = np.zeros((C, own[s].shape[1])); np.add.at(S, ou, own[s].astype(np.float64)); Ssum[s] = S
    P = {s: nrm(Ssum[s]).astype(np.float32) for s in E}
    def zrow(S):
        mu = S[:, ok].mean(1, keepdims=True); sd = S[:, ok].std(1, keepdims=True); Z = (S - mu) / np.maximum(sd, 1e-9); Z[:, ~ok] = -1e9; return Z
    def fused(q, Pd, own_override=None, tr=None):
        F = 0
        for s in E:
            S = q[s] @ Pd[s].T
            if own_override is not None:
                S[np.arange(len(tr)), tr] = (q[s] * own_override[s]).sum(1)
            F = F + zrow(S) / len(E)
        return F + prior[None]
    def rate(F, tr, w=None, sel=None):
        f = (F > F[np.arange(len(tr)), tr][:, None]).sum(1) == 0; sel = np.ones(len(tr), bool) if sel is None else sel; w = np.ones(len(tr)) if w is None else w; return float((w[sel] * f[sel]).sum() / w[sel].sum()), float(f[sel].mean())
    F0 = fused(hq, P); r0 = rate(F0, truth, wa); first0 = F0.argmax(1); hit0 = first0 == truth; print(f"reference, held-out games: owner first {r0[0]:.3f} (weighted by games) | {r0[1]:.3f}")
    # ---------------------------------------------------------------- (a)
    order = np.argsort(ou, kind="stable"); bounds = np.searchsorted(ou[order], np.arange(C + 1)); pick = np.array([order[bounds[t] + rng.integers(0, bounds[t + 1] - bounds[t])] for t in truth]); qtr = {s: own[s][pick] for s in E}
    loo = {s: nrm(Ssum[s][truth] - own[s][pick].astype(np.float64)).astype(np.float32) for s in E}; Ft = fused(qtr, P, loo, truth); rt = rate(Ft, truth, wa)
    print(f"(a) a TRAINED game of the same owners, the owner's prototype rebuilt without it: owner first {rt[0]:.3f} | {rt[1]:.3f}  (held-out: {r0[0]:.3f} | {r0[1]:.3f}); by the owner's games " + "  ".join(f"[{lo},{hi}) trained {rate(Ft, truth, None, (n_c[truth] >= lo) & (n_c[truth] < hi))[1]:.3f} / held-out {rate(F0, truth, None, (n_c[truth] >= lo) & (n_c[truth] < hi))[1]:.3f}" for lo, hi in ((30, 100), (100, 300), (300, 1000))))
    ct = np.mean([(qtr[s] * loo[s]).sum(1).mean() for s in E]); ch = np.mean([(hq[s] * P[s][truth]).sum(1).mean() for s in E]); print(f"    mean cosine to the owner's prototype: trained game (left out of it) {ct:.3f}, held-out game {ch:.3f}")
    # ---------------------------------------------------------------- (b)
    big = np.flatnonzero(n_c >= 300); selq = n_c[truth] >= 300; print(f"(b) owners with >= 300 games ({int(selq.sum())} held-out queries; full prototypes: owner first {rate(F0, truth, None, selq)[1]:.3f}); prototype rebuilt from k random games, prior unchanged:")
    for k in (16, 32, 64, 128):
        vals = []
        for rep in range(3):
            Pk = {s: P[s].copy() for s in E}
            for c in big:
                idx = order[bounds[c]:bounds[c + 1]]; sub = rng.choice(idx, k, replace=False)
                for s in E:
                    Pk[s][c] = nrm(own[s][sub].astype(np.float64).sum(0))
            vals.append(rate(fused(hq, Pk), truth, None, selq)[1])
        print(f"    k = {k:3d}: owner first {np.mean(vals):.3f} (three draws {min(vals):.3f}-{max(vals):.3f})")
    print(f"    for reference, owners who really have 30-100 games: {rate(F0, truth, None, (n_c[truth] >= 30) & (n_c[truth] < 100))[1]:.3f}; 100-300 games: {rate(F0, truth, None, (n_c[truth] >= 100) & (n_c[truth] < 300))[1]:.3f}")
    # ---------------------------------------------------------------- (c)
    Cen = {}; hasc = np.ones(C, bool)
    for s in E:
        z = np.load(ROOT / f"results/m161q/out_-1_s{s}.npz"); Cen[s] = nrm(z["centres"].astype(np.float32)); hasc &= z["has_centre"].astype(bool)
    Fc = 0
    for s in E:
        S = hq[s] @ Cen[s].T; mu = S[:, hasc].mean(1, keepdims=True); sd = S[:, hasc].std(1, keepdims=True); Z = (S - mu) / np.maximum(sd, 1e-9); Z[:, ~hasc] = -1e9; Fc = Fc + Z / len(E)
    Fc = Fc + prior[None]; sel = hasc[truth]; firstc = Fc.argmax(1); hitc = firstc == truth; w = wa[sel] / wa[sel].sum()
    print(f"(c) owners with a centre ({int(sel.sum())} queries): owner first - prototypes {float((w * hit0[sel]).sum()):.3f}, centres {float((w * hitc[sel]).sum()):.3f}; both right {float((w * (hit0 & hitc)[sel]).sum()):.3f}, only the centres {float((w * (~hit0 & hitc)[sel]).sum()):.3f}, only the prototypes {float((w * (hit0 & ~hitc)[sel]).sum()):.3f}, neither {float((w * (~hit0 & ~hitc)[sel]).sum()):.3f}; when both are wrong, the same wrong first {float(((first0 == firstc) & ~hit0 & ~hitc)[sel].sum() / max((~hit0 & ~hitc)[sel].sum(), 1)):.3f}")
    # ---------------------------------------------------------------- (d)
    miss = ~hit0; cnt = np.bincount(first0[miss], minlength=C).astype(float); share_w = cnt / cnt.sum(); share_g = n_c / n_c.sum(); top = np.argsort(-cnt)[:50]
    print(f"(d) wrong firsts ({int(miss.sum())}): the 50 candidates that take the most hold {float(share_w[top].sum()):.3f} of them and {float(share_g[top].sum()):.3f} of the games; candidates that are a wrong first at least 5 times: {int((cnt >= 5).sum())}; correlation of log(1 + wrong firsts) with log games {float(np.corrcoef(np.log1p(cnt[ok]), np.log(n_c[ok]))[0, 1]):.3f}")
    bg = rng.choice(len(use), 20000, replace=False); Fd = 0
    for s in E:
        B = own[s][bg] @ P[s].T; B[np.arange(len(bg)), ou[bg]] = -1.0; rc = np.sort(B, axis=0)[-10:].mean(0); Fd = Fd + zrow(hq[s] @ P[s].T - rc[None]) / len(E)
    Fd = Fd + prior[None]; rd = rate(Fd, truth, wa); print(f"    candidate-side correction (score minus the candidate's mean of its 10 highest background scores): owner first {rd[0]:.3f} | {rd[1]:.3f}  (reference {r0[0]:.3f} | {r0[1]:.3f})")


if __name__ == "__main__":
    main()
