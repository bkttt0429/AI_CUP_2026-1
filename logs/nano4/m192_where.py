#!/usr/bin/env python
"""Section 4M M192 step 2 (2026-11-16, report only): WHERE does the recogniser fail to separate?

Held-out own games (3,244; the owner known, the game not in any prototype), all-players networks of the second production
set (seeds 30, 31; plain vectors of m192_embed.py), neural score only: cosine to the prototypes, standardised over the
candidates, mean of the two seeds, + log games (prior, beta 1). "first" = the owner ranks first; rates weighted by the owner's
games (opponents are met in proportion to activity) | unweighted.
A  colour: first by the colour the query was played with; prototypes built from the candidate's games of ONE colour (the
   query's / the other one) and blends; the owner's own two colour prototypes against each other.
B  the query game: by its length; by handicap.
C  the candidate: how tight its games sit around the prototype (leave-one-out cosine) - first by tightness inside one band
   of games; two sub-prototypes per candidate (split of its games, refined) scored by the better one.
D  how early identity shows: the held-out game cut to its first 40 / 80 / 120 / 160 plies against the full prototypes.
E  the wrong first candidate: is it one of the owner's nearest prototypes (a look-alike) or far away (the game itself is
   off the owner's style)? and how typical the held-out game is for its owner (percentile of its cosine to the owner's
   prototype among the owner's own games, leave-one-out) - for hits and for misses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); W = Path("/work/u2507115/m192")


def zs(S, ok):
    mu = S[:, ok].mean(1, keepdims=True); sd = S[:, ok].std(1, keepdims=True); Z = (S - mu) / np.maximum(sd, 1e-9); Z[:, ~ok] = -1e9; return Z


def nrm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def main():
    m = np.load(W / "meta.npz"); owner, subj, length, held, key = m["owner"].astype(np.int64), m["subj"].astype(int), m["length"].astype(int), m["held"], m["key"].astype(str); C = int(owner.max()) + 1; use = np.flatnonzero(~held); hr = np.flatnonzero(held); truth = owner[hr]; qcol = subj[hr]; qlen = length[hr]
    n_c = np.bincount(owner[use], minlength=C).astype(float); ok = n_c > 0; BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; prior = BETA * np.where(ok, np.log(np.maximum(n_c, 1.0)), 0.0); wa = n_c[truth] / n_c[truth].sum(); H = len(hr); ar = np.arange(H)
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "hd"], dtype={"game_key": str}).set_index("game_key").hd.astype(float).reindex(key[hr]).to_numpy()
    E = {s: np.load(W / f"emb_s{s}.npz") for s in (30, 31)}; own = {s: E[s]["own"].astype(np.float32) for s in E}; ou = owner[use]; su = subj[use]

    def protos(mask=None):
        out = {}
        for s in E:
            S = np.zeros((C, own[s].shape[1])); sel = np.ones(len(ou), bool) if mask is None else mask; np.add.at(S, ou[sel], own[s][sel].astype(np.float64)); out[s] = S
        return out
    Sall = protos(); P = {s: nrm(Sall[s]) for s in E}

    def score(qname, Pd, extra=None):
        F = 0
        for s in E:
            q = E[s][qname].astype(np.float32); Z = zs(q @ Pd[s].T.astype(np.float32), ok)
            if extra is not None:
                Z2 = zs(q @ extra[s].T.astype(np.float32), ok); Z = np.maximum(Z, Z2)
            F = F + Z / len(E)
        return F + prior[None]

    def rates(F, sel=None):
        r = (F > F[ar, truth][:, None]).sum(1); sel = np.ones(H, bool) if sel is None else sel; w = wa[sel] / wa[sel].sum(); return float((w * (r[sel] == 0)).sum()), float((r[sel] == 0).mean()), float((w * (r[sel] < 5)).sum())
    show = lambda t: f"{t[0]:.3f} | {t[1]:.3f} (first five {t[2]:.3f})"  # noqa: E731
    F0 = score("held", P); base_r = rates(F0); rank0 = (F0 > F0[ar, truth][:, None]).sum(1); hit = rank0 == 0; print(f"reference (single prototype, neural score + {BETA} x log games): first {show(base_r)}; by the owner's games " + "  ".join(f"[{lo},{hi}) {rates(F0, (n_c[truth] >= lo) & (n_c[truth] < hi))[1]:.3f}" for lo, hi in ((30, 100), (100, 300), (300, 1000))))
    # ---------------------------------------------------------------- A
    print("\nA. colour")
    for c in (0, 1):
        print(f"  query played with colour {c}: {int((qcol == c).sum())} games, first {show(rates(F0, qcol == c))}")
    Sc = {c: protos(su == c) for c in (0, 1)}; n_col = {c: np.bincount(ou[su == c], minlength=C).astype(float) for c in (0, 1)}
    def colour_protos(lam, same=True):
        out = {0: {}, 1: {}}
        for qc in (0, 1):
            a, b = (qc, 1 - qc) if same else (1 - qc, qc)
            for s in E:
                out[qc][s] = nrm(Sc[a][s] + lam * Sc[b][s] + 1e-6 * Sall[s])
        return out
    for lab, lam, same in (("prototype from the candidate's games of the query's colour only", 0.0, True), ("of the other colour only", 0.0, False), ("query's colour + 0.25 x the other", 0.25, True), ("query's colour + 0.5 x the other", 0.5, True)):
        Pc = colour_protos(lam, same); F = np.zeros_like(F0)
        for qc in (0, 1):
            F[qcol == qc] = score("held", Pc[qc])[qcol == qc]
        print(f"  {lab}: first {show(rates(F))}")
    cs = np.mean([(nrm(Sc[0][s]) * nrm(Sc[1][s])).sum(1)[(n_col[0] >= 20) & (n_col[1] >= 20)].mean() for s in E]); rnd = np.mean([float((nrm(Sc[0][s])[:2000] @ nrm(Sc[1][s])[2000:4000].T).mean()) for s in E]); print(f"  cosine between a candidate's two colour prototypes (>= 20 games each): {cs:.3f}; between different candidates' prototypes: {rnd:.3f}")
    # ---------------------------------------------------------------- B
    print("\nB. the query game")
    for lo, hi in ((0, 120), (120, 170), (170, 220), (220, 256), (256, 10 ** 6)):
        sel = (qlen >= lo) & (qlen < hi)
        if sel.sum() >= 50:
            print(f"  length [{lo}, {hi if hi < 10 ** 6 else 'inf'}): {int(sel.sum())} games, first {show(rates(F0, sel))}")
    for lab, sel in (("even", hd == 0), ("handicap", hd > 0)):
        if sel.sum() >= 50:
            print(f"  {lab}: {int(sel.sum())} games, first {show(rates(F0, sel))}")
    # ---------------------------------------------------------------- C
    print("\nC. the candidate")
    s0 = 30; e0 = own[s0]; S0 = Sall[s0]; loo = (e0.astype(np.float64) * nrm(S0[ou] - e0)).sum(1); tight = np.zeros(C); np.add.at(tight, ou, loo); tight = tight / np.maximum(n_c, 1)
    for lo, hi in ((30, 100), (100, 300), (300, 1000)):
        b = (n_c[truth] >= lo) & (n_c[truth] < hi); t = tight[truth]; qs = np.quantile(t[b], [1 / 3, 2 / 3]); print(f"  owners with {lo}-{hi} games, by tightness of their own games (leave-one-out cosine to the prototype; loose / middle / tight): " + "  ".join(f"{float(t[b & g].mean()):.3f}: first {float(hit[b & g].mean()):.3f}" for g in (t < qs[0], (t >= qs[0]) & (t < qs[1]), t >= qs[1])))
    sub = {s: (P[s].copy(), P[s].copy()) for s in E}; cnt2 = 0; order = np.argsort(ou, kind="stable"); bounds = np.searchsorted(ou[order], np.arange(C + 1))
    for c in np.flatnonzero(n_c >= 40):
        idx = order[bounds[c]:bounds[c + 1]]; X = e0[idx].astype(np.float64); Xc = X - X.mean(0); v = np.linalg.svd(Xc, full_matrices=False)[2][0]; lab = (Xc @ v) > 0
        for _ in range(4):
            if lab.sum() < 5 or (~lab).sum() < 5:
                break
            c1, c2 = nrm(X[lab].sum(0)), nrm(X[~lab].sum(0)); lab = (X @ c1) > (X @ c2)
        if lab.sum() >= 5 and (~lab).sum() >= 5:
            cnt2 += 1
            for s in E:
                Xs = own[s][idx].astype(np.float64); sub[s][0][c] = nrm(Xs[lab].sum(0)); sub[s][1][c] = nrm(Xs[~lab].sum(0))
    F2 = score("held", {s: sub[s][0] for s in E}, {s: sub[s][1] for s in E}); print(f"  two sub-prototypes for the {cnt2} candidates with >= 40 games (the better of the two counts): first {show(rates(F2))}; by the owner's games " + "  ".join(f"[{lo},{hi}) {rates(F2, (n_c[truth] >= lo) & (n_c[truth] < hi))[1]:.3f} (single {rates(F0, (n_c[truth] >= lo) & (n_c[truth] < hi))[1]:.3f})" for lo, hi in ((30, 100), (100, 300), (300, 1000))))
    # ---------------------------------------------------------------- D
    print("\nD. how early identity shows (held-out game cut to its first plies, full prototypes)")
    for c in (40, 80, 120, 160):
        print(f"  first {c} plies: first {show(rates(score(f'held_cut{c}', P)))}")
    print(f"  whole game (up to 256 plies): first {show(base_r)}; games longer than 256 plies: {float((qlen > 256).mean()):.3f} of the held-out games (training games: {float((length[use] > 256).mean()):.3f}; exactly 256 stored: {float((length[use] == 256).mean()):.3f})")
    # ---------------------------------------------------------------- E
    print("\nE. the wrong first candidate and the typicality of the game")
    PP = np.mean([P[s] @ P[s].T for s in E], axis=0); np.fill_diagonal(PP, -2.0); first = F0.argmax(1); miss = ~hit; nb_rank = (PP[truth[miss]] > PP[truth[miss], first[miss]][:, None]).sum(1); wm = wa[miss] / wa[miss].sum()
    print(f"  misses ({int(miss.sum())}): the wrong first is among the owner's nearest prototypes - nearest 5: {float((wm * (nb_rank < 5)).sum()):.3f}, nearest 20: {float((wm * (nb_rank < 20)).sum()):.3f}, nearest 100: {float((wm * (nb_rank < 100)).sum()):.3f}, beyond 500: {float((wm * (nb_rank >= 500)).sum()):.3f} (weighted by the owner's games); median rank {int(np.median(nb_rank))}")
    qh = E[s0]["held"].astype(np.float64); cq = (qh * P[s0][truth]).sum(1); pct = np.zeros(H)
    for i in range(H):
        l = loo[order[bounds[truth[i]]:bounds[truth[i] + 1]]]; pct[i] = float((l < cq[i]).mean()) if len(l) else np.nan
    for lab, sel in (("hits", hit), ("misses", miss), ("misses where the owner is not in the first twenty", rank0 >= 20)):
        print(f"  {lab}: typicality of the game for its owner (percentile among the owner's own games) - median {float(np.nanmedian(pct[sel])):.2f}, share below the 10th percentile {float((pct[sel] < 0.10).mean()):.3f}, below the 25th {float((pct[sel] < 0.25).mean()):.3f}")
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = (pct >= lo) & (pct < hi); print(f"    games in the owner's percentile band [{lo}, {min(hi, 1.0)}): {float(sel.mean()):.3f} of the held-out games, first {float(hit[sel].mean()):.3f}")


if __name__ == "__main__":
    main()
