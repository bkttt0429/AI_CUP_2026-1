#!/usr/bin/env python
"""M253 (2026-11-19), report only, label-free: cross-competition linkage re-measured with game-by-game votes (design doc M253).
How many rank-test players can be recognised as candidates of competition 1's added pools (pub2 / priv2, 600 candidates x 10 games each)?

Vectors: results/m95 (own-side vectors of the competition-1 games under the 15 M90 networks; nothing of the player-identification
session is read here); the 10,000 test games are embedded again with the same networks.
Two-way votes: every game of a query unit votes for the candidate unit with the highest per-game standardised cosine to the unit's
prototype (mean of the networks), and every game of a candidate unit votes among the query units. Pair = mutual favourites,
strength = the smaller vote share.
Queries    = 800 test questions + 400 positive controls (games 11.. of the old Public pool's candidates, one colour, 5 / 10 / 15 / 20 games)
Candidates = pub2 / priv2 candidates + 400 control candidates (the first 10 games of the same old Public candidates; the old Public pool
             has no overlap with the test players, M95)

  python -I logs/nano4/m253_xvotes.py
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m88_seqid as m  # noqa: E402

X95 = ROOT / "results/m95"; RV = ROOT / "results/r_variants"; TAGS = [f"{v}{f'_s{s}' if s else ''}" for s in (0, 1, 2) for v in m.NETS]


def unit_norm(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def vote(vq, vc, uq, uc, nq, nc):
    """Votes of the query games (rows of vq, unit uq) among nc candidate units (prototype = normalised sum of the rows of vc by uc).
    vq / vc: lists over the networks. Returns the vote matrix [nq, nc]."""
    F = 0
    for a, b in zip(vq, vc):
        T = np.zeros((nc, b.shape[1])); np.add.at(T, uc, b); T = unit_norm(T); S = a @ T.T; F = F + (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)
    W = np.zeros((nq, nc)); np.add.at(W, (uq, F.argmax(1)), 1.0); return W


def mutual(W1, W2, kq, kc, lo):
    """Mutual favourites (query a, candidate b) with min share >= lo; W1 [nq, nc] votes of the queries, W2 [nc, nq] votes of the candidates."""
    f1 = W1.argmax(1); s1 = W1[np.arange(len(f1)), f1] / np.maximum(kq, 1); f2 = W2.argmax(1); s2 = W2[np.arange(len(f2)), f2] / np.maximum(kc, 1)
    return [(int(a), int(f1[a]), float(s1[a]), float(s2[f1[a]])) for a in range(len(f1)) if f2[f1[a]] == a and min(s1[a], s2[f1[a]]) >= lo]


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; tf = sorted(m.TEST.glob("games_s*_p*.csv"))
    with mp.get_context("fork").Pool(12) as pool:
        rt = pool.map(m.read_part, tf, chunksize=2)
    cat = lambda j: np.concatenate([x[j] for x in rt])  # noqa: E731
    o = np.argsort(cat(0)); assert np.array_equal(cat(0)[o], np.arange(len(o))); qid = cat(1)[o].astype(str); Tt = dict(GL=cat(2)[o], POS=cat(3)[o], MOV=cat(4)[o], LEN=cat(5)[o], SUBJ=cat(6)[o]); del rt
    ref = np.load(RV / "m82_test.npz")["qid"].astype(str); agree = float((ref == qid).mean()); log(f"test games {len(qid)}; question ids equal to m82_test.npz in {agree:.4f} of the rows"); assert agree == 1.0
    g = np.load(X95 / "games.npz"); grp = g["group"].astype(str); subj = g["subj"].astype(int); G = len(grp); log(f"competition-1 games {G}; groups: " + ", ".join(f"{p} {len(set(grp[np.char.startswith(grp, p)]))}" for p in ("oldpub:c:", "oldpub:q:", "pub2:c:", "pub2:q:", "priv2:c:", "priv2:q:")))
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    VT, VX = [], []
    for tag in TAGS:
        st = torch.load(ROOT / f"results/m90/ckpt_{tag}.pt", map_location="cpu", weights_only=False); net = m.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); del st; out = np.zeros((len(qid), m.D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(qid), 2048):
                s = slice(i, i + 2048); out[s] = net(tb(Tt["GL"][s]), tb(Tt["POS"][s].astype(np.int64)), tb(Tt["MOV"][s].astype(np.int64)), tb(Tt["LEN"][s].astype(np.int64)), tb(Tt["SUBJ"][s].astype(np.int64))).float().cpu().numpy()
        VT.append(unit_norm(out.astype(np.float64))); VX.append(unit_norm(np.load(X95 / f"xemb_{tag}.npz")["own"].astype(np.float64))); del net
    log(f"{len(TAGS)} networks: test vectors embedded, pool vectors loaded")
    uq_names, uq_t = np.unique(qid, return_inverse=True); nt = len(uq_names); size_t = np.bincount(uq_t, minlength=nt)
    # sanity: within-question coherence of the test vectors (first network)
    rng = np.random.default_rng(2530); a_ = rng.integers(0, len(qid), 200000); b_ = rng.integers(0, len(qid), 200000); cs = (VT[0][a_] * VT[0][b_]).sum(1); same = (uq_t[a_] == uq_t[b_]) & (a_ != b_)
    T0 = np.zeros((nt, 128)); np.add.at(T0, uq_t, VT[0]); own_c = ((VT[0] * (T0[uq_t] - VT[0])).sum(1) / np.maximum(size_t[uq_t] - 1, 1)).mean(); log(f"sanity: mean cosine of a test game to the other games of its question {own_c:.3f}; random pairs of different questions {float(cs[~same].mean()):.3f}")
    # ---- units
    old_c = sorted(set(grp[np.char.startswith(grp, "oldpub:c:")])); sizes = (5, 10, 15, 20); ctl_c_rows, ctl_q_rows, ctl_q_size = [], [], []
    for j, name in enumerate(old_c):
        r = np.flatnonzero(grp == name); ctl_c_rows.append(r[:10]); rest = r[10:]; col = j % 2; n = sizes[j % 4]; pick = rest[subj[rest] == col][:n]
        if len(pick) < n:
            pick = rest[subj[rest] == 1 - col][:n]
        ctl_q_rows.append(pick); ctl_q_size.append(len(pick))
    nctl = len(old_c); log(f"controls: {nctl} old Public candidates; control question sizes " + ", ".join(f"{s}: {int((np.array(ctl_q_size) == s).sum())}" for s in sizes))
    rep = {}

    def run(cand_prefixes, label):
        cn = sorted(set(grp[np.logical_or.reduce([np.char.startswith(grp, p) for p in cand_prefixes])])); ci = {n_: i for i, n_ in enumerate(cn)}; ncand = len(cn) + nctl
        rows_c = np.concatenate([np.flatnonzero(np.isin(grp, cn)), *ctl_c_rows]); uc = np.concatenate([np.array([ci[x] for x in grp[np.isin(grp, cn)]], int), *[np.full(len(r), len(cn) + j) for j, r in enumerate(ctl_c_rows)]]); kc = np.bincount(uc, minlength=ncand)
        rows_qx = np.concatenate(ctl_q_rows); uq = np.concatenate([uq_t, *[np.full(len(r), nt + j) for j, r in enumerate(ctl_q_rows)]]); nq = nt + nctl; kq = np.bincount(uq, minlength=nq)
        vq = [np.vstack([VT[i], VX[i][rows_qx]]) for i in range(len(TAGS))]; vc = [VX[i][rows_c] for i in range(len(TAGS))]
        W1 = vote(vq, vc, uq, uc, nq, ncand); W2 = vote(vc, vq, uc, uq, ncand, nq)
        out = {"candidates": len(cn), "candidate_games_median": float(np.median(kc[:len(cn)]))}
        print(f"\n== {label}: {len(cn)} candidate units (median {np.median(kc[:len(cn)]):.0f} games) + {nctl} control candidates; {nt} test questions + {nctl} control questions")
        for lo in (0.5, 0.7):
            P = mutual(W1, W2, kq, kc, lo); ctl_ok = [p_ for p_ in P if p_[0] >= nt and p_[1] == len(cn) + (p_[0] - nt)]; ctl_bad = [p_ for p_ in P if p_[0] >= nt and p_[1] != len(cn) + (p_[0] - nt)]
            t_fp = [p_ for p_ in P if p_[0] < nt and p_[1] >= len(cn)]; t_hit = [p_ for p_ in P if p_[0] < nt and p_[1] < len(cn)]
            pw = {s: sum(1 for p_ in ctl_ok if ctl_q_size[p_[0] - nt] == s) / max(sum(1 for x in ctl_q_size if x == s), 1) for s in sizes}; power = len(ctl_ok) / nctl
            fp_rate = len(t_fp) / nctl; exp_fp = fp_rate * len(cn); pub = sum(1 for p_ in t_hit if uq_names[p_[0]].startswith("pub")); by_n = {int(s): sum(1 for p_ in t_hit if size_t[p_[0]] == s) for s in sizes}
            pw_t = sum(pw[s] * (size_t == s).sum() for s in sizes) / nt; est = (len(t_hit) - exp_fp) / max(pw_t, 1e-9)
            print(f"   min share >= {lo}: positive control found {len(ctl_ok)}/{nctl} = {power:.3f} (by size " + ", ".join(f"{s}: {pw[s]:.3f}" for s in sizes) + f"), control paired with a wrong unit {len(ctl_bad)}; test questions paired with a control candidate (false) {len(t_fp)} -> expected false among the real candidates {exp_fp:.1f}")
            print(f"      TEST QUESTIONS PAIRED WITH A REAL CANDIDATE: {len(t_hit)} (Public {pub}, Private {len(t_hit) - pub}; by size " + ", ".join(f"{s}: {by_n[s]}" for s in sizes) + f"); net of false {len(t_hit) - exp_fp:.1f}; implied overlapping players at the test's size mix (power {pw_t:.3f}): {est:.0f}", flush=True)
            out[f"lo_{lo}"] = {"power": power, "power_by_size": pw, "control_wrong": len(ctl_bad), "test_false": len(t_fp), "expected_false": exp_fp, "test_hits": len(t_hit), "public": pub, "private": len(t_hit) - pub, "by_size": by_n, "implied_overlap": est,
                               "pairs": [(str(uq_names[a]), cn[b], int(size_t[a]), s1, s2) for a, b, s1, s2 in sorted(t_hit, key=lambda x: -min(x[2], x[3]))]}
            if lo == 0.5:
                for a, b, s1, s2 in sorted(t_hit, key=lambda x: -min(x[2], x[3])):
                    print(f"         {uq_names[a]} ({size_t[a]} games) <-> {cn[b]} ({kc[b]} games): {s1 * size_t[a]:.0f}/{size_t[a]} | {s2 * kc[b]:.0f}/{kc[b]}")
        f1 = W1.argmax(1); known = [("pub_q_0298", "pub2_p_0353"), ("pub_q_0256", "pub2_p_0176"), ("pub_q_0335", "pub2_p_0456"), ("priv_q_0381", "priv2_p_0127")]; ix = {n_: i for i, n_ in enumerate(uq_names)}
        print("   M95's four named pairs: " + "; ".join(f"{a} -> favourite {cn[f1[ix[a]]] if f1[ix[a]] < len(cn) else 'control'} ({W1[ix[a], f1[ix[a]]]:.0f}/{size_t[ix[a]]}; named {b})" for a, b in known if a in ix), flush=True)
        rep[label] = out
    run(("pub2:c:", "priv2:c:"), "candidates of pub2 / priv2")
    run(("pub2:q:", "priv2:q:"), "questions of pub2 / priv2 (report only)")
    (RV / "m253_xvotes.json").write_text(json.dumps(rep, indent=1, default=float)); log("written results/r_variants/m253_xvotes.json")


if __name__ == "__main__":
    main()
