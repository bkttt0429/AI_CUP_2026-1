#!/usr/bin/env python
"""Section 4M M193 (2026-11-16, report only): is a prototype built from games the network has NOT trained on a better
prototype for unseen queries - or is the unseen query itself the problem?

M192: a held-out game falls below the 10th percentile of its owner's trained games in 55 % of the cases. Two readings:
 (1) the prototype is in the wrong place for unseen games (it is the mean of memorised games) -> build prototypes from games
     the network never trained on;
 (2) the prototype is fine, unseen games are simply embedded more loosely -> only better generalisation helps.
Tool: the width-768 pilot network that was trained while results/trunk_vec_x80 was missing (846,807 games; snapshot of its
running checkpoint). The 149,949 x80 games are games of TRAINED players that this network has never seen.
Players with >= 24 unseen games: their unseen games are split in two halves, A (enrolment) and B (queries). Candidates = these
players only (closed set, no prior). Prototypes: all seen games / a random subset of the seen games as large as A / A alone /
seen + A / seen + A with A counted k times. Queries: B. Reported: first (unweighted), by the number of seen games; the
percentile of a B game against the owner's A games (should be uniform) and against the owner's seen games.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402


def nrm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT = True, None; base.D, base.FFN = 768, 3072; G, _ = base.load_games(False, log); dev = "cuda"
    st = torch.load("/work/u2507115/m192/w768b_snapshot.pt", map_location=dev, weights_only=False); net = base.build_model().to(dev); net.load_state_dict(st["net"]); net.eval(); log(f"snapshot at step {st['step']}; best so far {st['best']}")
    q = np.load(base.CACHE / "queries.npz"); pidx = {p: i for i, p in enumerate(q["players"])}; held = set(json.load(open(base.M82 / "held_keys.json"))); owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]])
    xk = set(pd.read_csv(ROOT / "results/data/x80_keys.csv", dtype=str).game_key); is_x = np.array([k in xk for k in G["key"]]); seen = ~is_held & ~is_x; unseen = ~is_held & is_x; log(f"seen games {int(seen.sum())}, unseen (x80) {int(unseen.sum())}")
    tb = lambda x: torch.as_tensor(x, device=dev); E = np.zeros((len(owner), base.D_OUT), np.float32)  # noqa: E731
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(owner), 1024):
            r = np.arange(i, min(i + 1024, len(owner))); E[r] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
    log("embedded"); C = len(pidx); nu = np.bincount(owner[unseen], minlength=C); ns = np.bincount(owner[seen], minlength=C); cand = np.flatnonzero((nu >= 24) & (ns >= 10)); cid = -np.ones(C, int); cid[cand] = np.arange(len(cand)); rng = np.random.default_rng(0); M = len(cand)
    S_seen = np.zeros((M, E.shape[1])); S_sub = np.zeros_like(S_seen); S_A = np.zeros_like(S_seen); qv, qo = [], []; pct_A, pct_S = [], []; order = np.argsort(owner, kind="stable"); bounds = np.searchsorted(owner[order], np.arange(C + 1))
    for j, c in enumerate(cand):
        idx = order[bounds[c]:bounds[c + 1]]; s_i = idx[seen[idx]]; u_i = rng.permutation(idx[unseen[idx]]); a_i, b_i = u_i[:len(u_i) // 2], u_i[len(u_i) // 2:]
        S_seen[j] = E[s_i].sum(0); S_sub[j] = E[rng.choice(s_i, min(len(a_i), len(s_i)), replace=False)].sum(0); S_A[j] = E[a_i].sum(0); qv.append(E[b_i]); qo += [j] * len(b_i)
        pa = nrm(S_A[j]); ps = nrm(S_seen[j]); la = (E[a_i] * nrm(S_A[j][None] - E[a_i])).sum(1); ls = (E[s_i] * nrm(S_seen[j][None] - E[s_i])).sum(1); pct_A += [float((la < x).mean()) for x in E[b_i] @ pa]; pct_S += [float((ls < x).mean()) for x in E[b_i] @ ps]
    Qv = np.concatenate(qv); qo = np.array(qo); nsj = ns[cand]; naj = np.array([len(x) for x in qv]); pct_A, pct_S = np.array(pct_A), np.array(pct_S)
    log(f"candidates {M} (>= 24 unseen and >= 10 seen games); queries {len(qo)}; seen games per candidate median {int(np.median(nsj))}, enrolment (A) games median {int(np.median(naj))}")
    print(f"percentile of an unseen query game among the owner's games (leave-one-out cosine to the prototype): against the owner's UNSEEN enrolment games - below the 10th percentile {float((pct_A < 0.1).mean()):.3f}, median {float(np.median(pct_A)):.2f}; against the owner's SEEN games - below the 10th percentile {float((pct_S < 0.1).mean()):.3f}, median {float(np.median(pct_S)):.2f}")
    def first(Pm):
        Z = Qv @ nrm(Pm).T.astype(np.float32); Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True); f = Z.argmax(1) == qo; return f
    res = {"all seen games": first(S_seen), "seen games, as many as A": first(S_sub), "unseen games A alone": first(S_A), "seen + A": first(S_seen + S_A)}
    for k in (3, 10, 30):
        res[f"seen + {k} x A"] = first(S_seen + k * S_A)
    res["seen and A as two unit prototypes (sum)"] = first(nrm(S_seen) + nrm(S_A))
    bands = [(10, 40), (40, 100), (100, 300), (300, 10 ** 6)]
    print("prototype built from | first (all queries) | by the candidate's seen games " + " ".join(f"[{lo},{hi if hi < 10 ** 6 else 'inf'})" for lo, hi in bands))
    for nm, f in res.items():
        print(f"  {nm:42s} | {float(f.mean()):.3f} | " + "  ".join(f"{float(f[(nsj[qo] >= lo) & (nsj[qo] < hi)].mean()):.3f}" for lo, hi in bands))
    print("  queries per band: " + "  ".join(str(int(((nsj[qo] >= lo) & (nsj[qo] < hi)).sum())) for lo, hi in bands))


if __name__ == "__main__":
    main()
