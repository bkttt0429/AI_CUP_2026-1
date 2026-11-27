#!/usr/bin/env python
"""Section 4M M217 (2026-11-17), report only: how much of the single-game recogniser is a generic style metric (works for players
the network never trained on) and how much is exposure (the network trained on the candidate's games), by the candidate's games.

For each cross-fitted production network v (second set, seed 30; trained without the players of fold v): the own-side vector of
every stored training game; prototype of each of the 8,007 players = unit sum of the vectors of the player's games that are not
held out (trained games for the players the network saw, never-trained games for fold v's players). Queries, the owner known:
  seen-held      held-out own games of players the network trained on (>= 16 trained games)        -> generic + exposure
  unseen-held    held-out own games of fold v's players                                             -> generic only
  unseen-extra   up to K_EXTRA further games of every fold-v player, the game taken out of its owner's prototype
  seen-trained   one trained game of every trained player, taken out of the prototype              -> memorised reference
  unseen-proto16 / 48   as unseen-extra for fold-v players with >= 64 games, the owner's prototype from 16 / 48 random other games
                        (unseen-n64 = the same queries with the full prototype)
Two scorings: the cosine to the prototypes (as the training log), and the cosine standardised over the candidates + 0.4 log(games)
(as M192). Top-1 / top-5 / top-20 by the owner's number of non-held games; "w" = weighted by the bands' share of all games.

  sbatch --gres=gpu:1 -c 12 ... run.sbatch python -I logs/nano4/m217_generic.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

CK = Path("/work/u2507115/m161q_ckpt"); BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; K_EXTRA, BETA, SUB, MIN_TR = 5, 0.4, (16, 48), 16


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=30); ap.add_argument("--nets", type=int, nargs="+", default=[1, 2, 3, 4]); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT = True, None; base.D, base.FFN = 512, 2048; G, _ = base.load_games(a.smoke, log); dev = "cuda"
    q = np.load(base.CACHE / "queries.npz"); pidx = {p: i for i, p in enumerate(q["players"])}; C = len(pidx); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]], np.int64); is_held = np.array([k in held for k in G["key"]]); fold = G["fold"].astype(np.int64); use = np.flatnonzero(~is_held); hr = np.flatnonzero(is_held)
    N_all = np.bincount(owner[use], minlength=C); has = N_all > 0; pfold = np.full(C, -9, np.int64); pfold[owner] = fold; band_p = np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi), 0) for n in N_all])
    share = np.array([N_all[(N_all >= lo) & (N_all < hi)].sum() for lo, hi in BANDS]) / max(N_all.sum(), 1); prior = BETA * np.log(np.maximum(N_all, 1))
    o_use = owner[use]; order = np.argsort(o_use, kind="stable"); first = np.searchsorted(o_use[order], np.arange(C)); last = np.searchsorted(o_use[order], np.arange(C), side="right")  # the games of player p = order[first[p]:last[p]] (indices into use)
    log(f"{len(use)} stored games not held out, {len(hr)} held out, {C} players ({int(has.sum())} with a prototype); share of games by band {np.round(share, 3).tolist()}")
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def embed(net, rows):
        out = np.zeros((len(rows), base.D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
        return out
    acc = {}  # (group, scoring) -> [bands, 4]: top-1, top-5, top-20 hits and the number of queries

    def add(group, Eq, truth, P, own_cos=None):
        if not len(truth):
            return
        S = Eq @ P.T
        if own_cos is not None:
            S[np.arange(len(truth)), truth] = own_cos
        S[:, ~has] = -2.0; Z = (S - S[:, has].mean(1, keepdims=True)) / (S[:, has].std(1, keepdims=True) + 1e-9) + prior[None]; Z[:, ~has] = -1e9
        for name, M in (("cosine", S), ("prior", Z)):
            r = (M > M[np.arange(len(truth)), truth][:, None]).sum(1); A = acc.setdefault((group, name), np.zeros((len(BANDS), 4)))
            for k in range(len(BANDS)):
                m = band_p[truth] == k; A[k] += [(r[m] == 0).sum(), (r[m] < 5).sum(), (r[m] < 20).sum(), m.sum()]
    for v in a.nets:
        st = torch.load(CK / f"ckpt_{v}_s{a.seed}.pt", map_location=dev, weights_only=False); net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); rng = np.random.default_rng(217 + v)
        E = embed(net, use); Eh = embed(net, hr); S = np.zeros((C, base.D_OUT)); np.add.at(S, o_use, E.astype(np.float64)); P = (S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        ref = np.load(ROOT / f"results/m161q/out_{v}_s{a.seed}.npz")["protos"]; cnt_tr = np.bincount(o_use[fold[use] != v], minlength=C); trained = (pfold != v) & (cnt_tr >= MIN_TR); unseen = pfold == v
        log(f"net {v}: vectors done; prototypes against the training run's stored ones, mean cosine {float((P * ref).sum(1)[has].mean()):.5f}; players trained {int(trained.sum())}, unseen {int((unseen & has).sum())}")
        th = owner[hr]
        for name, m in (("seen-held", trained[th]), ("unseen-held", unseen[th]), ("fewgames-held", ~trained[th] & ~unseen[th])):
            add(name, Eh[m], th[m], P)

        def loo(j):
            """Own-prototype cosine of the games use[j] with the game itself taken out of the owner's sum."""
            e = E[j].astype(np.float64); rest = S[o_use[j]] - e
            return ((e * rest).sum(1) / np.maximum(np.linalg.norm(rest, axis=1), 1e-12)).astype(np.float32)
        ju, big = [], []
        for p in np.flatnonzero(unseen & (N_all >= 2)):
            rows_p = order[first[p]:last[p]]; pick = rng.choice(rows_p, min(K_EXTRA, len(rows_p)), replace=False); ju.append(pick)
            if N_all[p] >= 64:
                big.append((p, pick, np.setdiff1d(rows_p, pick)))
        ju = np.concatenate(ju); add("unseen-extra", E[ju], o_use[ju], P, loo(ju))
        jt = np.array([rng.choice(order[first[p]:last[p]]) for p in np.flatnonzero(trained)]); add("seen-trained", E[jt], o_use[jt], P, loo(jt))
        if big:
            jb = np.concatenate([x[1] for x in big]); add("unseen-n64", E[jb], o_use[jb], P, loo(jb))
            for size in SUB:
                oc = []
                for p, pick, rest in big:
                    pr = E[rng.choice(rest, size, replace=False)].astype(np.float64).sum(0); pr /= max(np.linalg.norm(pr), 1e-12); oc.append(E[pick] @ pr.astype(np.float32))
                add(f"unseen-proto{size}", E[jb], o_use[jb], P, np.concatenate(oc))
        log(f"net {v}: scored")
        del E, Eh, net
    rep = {"bands": BANDS, "share_games": share.tolist(), "nets": a.nets, "seed": a.seed, "beta": BETA, "groups": {}}
    for (group, name), A in sorted(acc.items()):
        t = A[:, :3] / np.maximum(A[:, 3:], 1); ok = A[:, 3] > 0; w = (share[ok, None] * t[ok]).sum(0) / max(share[ok].sum(), 1e-12); tot = A.sum(0)
        rep["groups"][f"{group}|{name}"] = {"n": A[:, 3].astype(int).tolist(), "top1": t[:, 0].tolist(), "top5": t[:, 1].tolist(), "top20": t[:, 2].tolist(), "all": (tot[:3] / max(tot[3], 1)).tolist(), "weighted": w.tolist()}
        print(f"{group:15s} {name:6s} n {int(tot[3]):6d}  top-1 by games " + "  ".join(f"{lo}+ {('-' if not o_ else format(x, '.3f'))} ({int(n)})" for (lo, hi), x, n, o_ in zip(BANDS, t[:, 0], A[:, 3], ok)) + f"  | all {tot[0] / max(tot[3], 1):.3f}  w {w[0]:.3f}  | top-5 {tot[1] / max(tot[3], 1):.3f}  top-20 {tot[2] / max(tot[3], 1):.3f}", flush=True)
    out = ROOT / f"results/r_variants/m217_generic{'_smoke' if a.smoke else ''}.json"; out.write_text(json.dumps(rep, indent=1)); log(f"written {out}")


if __name__ == "__main__":
    main()
