#!/usr/bin/env python
"""M251 (2026-11-19), report only, label-free: does one test player own two test questions (one black, one white)?
(design doc M251, registered before any reading; closes the blind spot of the 2026-11-04 linkage check, which compared
questions of the same colour only)

Question prototype = mean of the normalised own-side identity vectors of its games (the look-alike members); similarity of two
questions = mean cosine over the members. Per question: best other question, its similarity, the margin (best minus the mean of
ranks 2-5) and whether the two are each other's best.
Controls on the validation side (every fold with its own cross-fitted network, so the players are unseen as on the test side):
(player, colour) units of 5 or 10 scored games;
  P1  a pool of 400 players with a black and a white unit each (800 units, every unit has a partner)
  P0  a pool of 800 different players with one unit each (no partner anywhere)

  python -I logs/nano4/m251_pairs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402

la, CACHE = mb7.la, mb7.CACHE; LEV = led.LEVELS; RNG = np.random.default_rng(251); REPS = 6; QTL = (0.5, 0.9, 0.99, 0.999)


def protos(vec, unit, nu):
    """[M, n, d] game vectors -> [M, nu, d] normalised means of the normalised vectors."""
    out = np.zeros((vec.shape[0], nu, vec.shape[2]))
    for m in range(vec.shape[0]):
        U = vec[m].astype(np.float64); U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9); np.add.at(out[m], unit, U)
    return out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-9)


def sim(P):
    S = np.mean([p @ p.T for p in P], axis=0); np.fill_diagonal(S, -np.inf); return S


def stats(S):
    """Best other unit, its similarity, margin over ranks 2-5, mutual-best flag."""
    o = np.argsort(-S, axis=1)[:, :5]; top = o[:, 0]; s1 = S[np.arange(len(S)), top]; rest = np.take_along_axis(S, o[:, 1:5], 1).mean(1); mut = top[top] == np.arange(len(S))
    return top, s1, s1 - rest, mut


def qt_(x, qs=QTL):
    return " / ".join(f"{float(np.quantile(x, q)):.3f}" for q in qs) if len(x) else "-"


def main():
    rep = {}
    dv = np.load(ROOT / "results/r_variants/m82_val.npz"); keyv = dv["key"].astype(str); pidv = dv["pid"].astype(str); hdv = dv["hd"].astype(int); tbv = dv["target_black"].astype(bool); foldv = dv["fold"].astype(int); yv = dv["y"].astype(int)
    dt = np.load(ROOT / "results/r_variants/m82_test.npz"); qt = dt["qid"].astype(str); tbt = dt["target_black"].astype(bool)
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), keyv); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); qvec = v_["qvec"]; M = qvec.shape[0]
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(qt)
    # ------------------------------------------------------------ test side
    ut, names = pd.factorize(qt); nq = len(names); size = np.bincount(ut, minlength=nq); col = np.zeros(nq, int); col[ut] = tbt.astype(int); pub = np.char.startswith(names.astype(str), "pub")
    assert all(len(set(tbt[ut == j].tolist())) == 1 for j in range(nq)), "a test question with two colours"
    ans = led.decisions("submission_rank_ct116.csv").reindex(names).to_numpy().astype(int)
    St = sim(protos(np.stack([qvec[m, tgt] for m in range(M)]), ut, nq)); top_t, s1_t, mg_t, mut_t = stats(St)
    print(f"test: {nq} questions ({int(col.sum())} black, {int((1 - col).sum())} white; sizes " + ", ".join(f"{s}: {int((size == s).sum())}" for s in sorted(set(size.tolist()))) + f"); members {M} ({la.VAR})", flush=True)
    # ------------------------------------------------------------ validation controls
    ok = hdv <= 1; P1 = dict(hit=[], mut=[], true=[], other=[], mg=[], k=[]); P0 = dict(mut=[], s1=[], mg=[], s1_mut=[], k=[]); half = dict(same=[], cross=[])
    for f in sorted(set(foldv.tolist())):
        idx = np.flatnonzero(ok & (foldv == f)); df = pd.DataFrame({"i": idx, "p": pidv[idx], "c": tbv[idx].astype(int)}); grp = {k_: g_.i.to_numpy() for k_, g_ in df.groupby(["p", "c"])}
        elig = {k_: (10 if len(g_) >= 10 else 5) for k_, g_ in grp.items() if len(g_) >= 5}; both = sorted({p_ for p_, c_ in elig if (p_, 1 - c_) in elig}); players = sorted({p_ for p_, _ in elig})
        print(f"fold {f}: {len(idx)} games, units with >= 5 games {len(elig)} (with >= 10: {sum(v == 10 for v in elig.values())}), players with both colours eligible {len(both)}, players with any {len(players)}", flush=True)

        def build(units):
            rows, owner = [], []
            for j, u in enumerate(units):
                g_ = RNG.choice(grp[u], elig[u], replace=False); rows.append(g_); owner.append(np.full(len(g_), j))
            rows = np.concatenate(rows); owner = np.concatenate(owner); return sim(protos(np.stack([qvec[m, Q + rows] for m in range(M)]), owner, len(units)))
        for _ in range(REPS):
            pl = RNG.choice(both, min(400, len(both)), replace=False); units = [(p_, c_) for p_ in pl for c_ in (1, 0)]; S = build(units); top, s1, mg, mut = stats(S); partner = np.arange(len(units)) ^ 1
            P1["hit"].append(float((top == partner).mean())); P1["mut"].append(float(((top == partner) & mut).mean())); P1["true"] += S[np.arange(len(units)), partner].tolist(); Sx = S.copy(); Sx[np.arange(len(units)), partner] = -np.inf; P1["other"] += Sx.max(1).tolist(); P1["mg"] += mg[top == partner].tolist()
            P1["k"] += [min(elig[units[j]], elig[units[j ^ 1]]) for j in range(len(units))]
            pl0 = RNG.choice(players, min(800, len(players)), replace=False); units0 = []
            for p_ in pl0:
                cs = [c_ for c_ in (0, 1) if (p_, c_) in elig]; units0.append((p_, int(RNG.choice(cs))))
            S0 = build(units0); top0, s10, mg0, mut0 = stats(S0); P0["mut"].append(float(mut0.mean())); P0["s1"] += s10.tolist(); P0["mg"] += mg0.tolist(); P0["s1_mut"].append(s10[mut0]); P0["k"] += [elig[u] for u in units0]
        # split halves of the same unit (same colour) and the two colours of one player, 5 games each: the true-pair similarity by kind
        ten = [u for u, k_ in elig.items() if k_ == 10][:600]
        if ten:
            rows, owner = [], []
            for j, u in enumerate(ten):
                g_ = RNG.choice(grp[u], 10, replace=False); rows.append(g_); owner.append(np.repeat([2 * j, 2 * j + 1], 5))
            Pp = protos(np.stack([qvec[m, Q + np.concatenate(rows)] for m in range(M)]), np.concatenate(owner), 2 * len(ten)); half["same"] += np.mean([(p[0::2] * p[1::2]).sum(1) for p in Pp], axis=0).tolist()
        bb = both[:600]; rows, owner = [], []
        for j, p_ in enumerate(bb):
            for c_ in (1, 0):
                rows.append(RNG.choice(grp[(p_, c_)], 5, replace=False)); owner.append(np.full(5, 2 * j + (1 - c_)))
        Pp = protos(np.stack([qvec[m, Q + np.concatenate(rows)] for m in range(M)]), np.concatenate(owner), 2 * len(bb)); half["cross"] += np.mean([(p[0::2] * p[1::2]).sum(1) for p in Pp], axis=0).tolist()
    tr, ot, k1 = np.array(P1["true"]), np.array(P1["other"]), np.array(P1["k"]); s0, m0, k0 = np.array(P0["s1"]), np.array(P0["mg"]), np.array(P0["k"]); thr = float(np.quantile(s0, 0.999)); thr99 = float(np.quantile(s0, 0.99))
    print(f"\n== P1 (every unit has a partner of the other colour; {len(P1['hit'])} pools of up to 800 units): the best other unit IS the partner {np.mean(P1['hit']):.3f} (min {min(P1['hit']):.3f}), mutual {np.mean(P1['mut']):.3f}")
    print(f"   similarity of true partners, quantiles 10 / 50 / 90 %: {qt_(tr, (0.1, 0.5, 0.9))}   (smaller unit of 5 games: {qt_(tr[k1 == 5], (0.1, 0.5, 0.9))}; both of 10: {qt_(tr[k1 == 10], (0.1, 0.5, 0.9))})")
    print(f"   best NON-partner in the same pools, quantiles 50 / 90 / 99 / 99.9 %: {qt_(ot)};  margin of the correct hits 10 / 50 / 90 %: {qt_(np.array(P1['mg']), (0.1, 0.5, 0.9))}")
    print(f"   true pairs of 5 + 5 games: same colour (split halves) {qt_(np.array(half['same']), (0.1, 0.5, 0.9))}; opposite colours {qt_(np.array(half['cross']), (0.1, 0.5, 0.9))}")
    print(f"\n== P0 (800 different players, no partner; {len(P0['mut'])} pools): mutual-best share {np.mean(P0['mut']):.3f}; best similarity 50 / 90 / 99 / 99.9 %: {qt_(s0)} (units of 5 games: {qt_(s0[k0 == 5])}; of 10: {qt_(s0[k0 == 10])}); margin: {qt_(m0)}")
    exp0 = float(np.mean([(x > thr).sum() for x in P0["s1_mut"]])) / 2; exp0_99 = float(np.mean([(x > thr99).sum() for x in P0["s1_mut"]])) / 2
    power = float((tr > thr).mean()); print(f"   threshold = 99.9 % quantile of the best similarity under P0 = {thr:.3f} (99 %: {thr99:.3f}); true partners above it: {power:.3f} (above the 99 % one: {float((tr > thr99).mean()):.3f}); mutual-best pairs above it per P0 pool: {exp0:.2f} (99 %: {exp0_99:.2f})")
    # ------------------------------------------------------------ test reading
    print(f"\n== TEST: mutual-best share {float(mut_t.mean()):.3f} (P0 {np.mean(P0['mut']):.3f}, P1 {np.mean(P1['mut']):.3f} for true partners); best similarity 50 / 90 / 99 / 99.9 %: {qt_(s1_t)}; margin: {qt_(mg_t)}")
    for s in sorted(set(size.tolist())):
        print(f"   questions of {s} games: best similarity {qt_(s1_t[size == s])}; mutual-best share {float(mut_t[size == s].mean()):.3f}")
    i_ = np.flatnonzero(mut_t & (np.arange(nq) < top_t)); pairs = [(int(a), int(top_t[a]), float(s1_t[a])) for a in i_]; n999 = sum(s > thr for _, _, s in pairs); n99 = sum(s > thr99 for _, _, s in pairs)
    cross = sum(col[a] != col[b] for a, b, s in pairs if s > thr99); mixed = sum(pub[a] != pub[b] for a, b, s in pairs if s > thr99)
    print(f"   mutual-best pairs {len(pairs)}; above the P0 99.9 % threshold: {n999} (P0 expects {exp0:.2f}); above the 99 % threshold: {n99} (P0 expects {exp0_99:.2f}; of these opposite colours {cross}, one Public + one Private {mixed})")
    print(f"   questions whose best similarity exceeds the P0 99.9 % threshold: {int((s1_t > thr).sum())} of {nq} (P0 expects {0.001 * nq:.1f}); the 99 % one: {int((s1_t > thr99).sum())} (P0 expects {0.01 * nq:.0f}); exceeding the median of true partners ({float(np.median(tr)):.3f}): {int((s1_t > np.median(tr)).sum())}")
    print("   the 15 most similar pairs of test questions (similarity, margin of the first, the two questions: colour, games, ct116 answer):")
    for a in np.argsort(-s1_t)[:15]:
        b_ = int(top_t[a]); print(f"      {s1_t[a]:.3f}  {mg_t[a]:+.3f}  {names[a]} ({'B' if col[a] else 'W'}, {size[a]}, {LEV[ans[a]]})  ->  {names[b_]} ({'B' if col[b_] else 'W'}, {size[b_]}, {LEV[ans[b_]]})  mutual {bool(mut_t[a])}")
    near = np.abs(ans - ans[top_t]) <= 1; print(f"   answers of a question and of its best other question within one level: all {float(near.mean()):.3f}; top 5 % by similarity {float(near[s1_t >= np.quantile(s1_t, 0.95)].mean()):.3f}; random pairs {float((np.abs(ans[:, None] - ans[None, :]) <= 1).mean()):.3f}")
    excess = n999 - exp0; verdict = "PAIRS EXIST" if excess >= 20 else ("no pairs: every test question is a different player" if excess < 5 and power >= 0.5 else "in between - inspect the listed pairs")
    print(f"\nREGISTERED: mutual-best pairs above the P0 99.9 % threshold minus the P0 expectation = {excess:+.1f}; power of the threshold on true partners {power:.3f}  ->  {verdict}")
    rep = {"p1_hit": float(np.mean(P1["hit"])), "p1_mutual": float(np.mean(P1["mut"])), "p0_mutual": float(np.mean(P0["mut"])), "thr999": thr, "thr99": thr99, "power999": power, "test_mutual": float(mut_t.mean()), "test_pairs_above_999": int(n999), "p0_expect_999": exp0,
           "test_pairs_above_99": int(n99), "p0_expect_99": exp0_99, "test_s1_q": [float(np.quantile(s1_t, q_)) for q_ in QTL], "p0_s1_q": [float(np.quantile(s0, q_)) for q_ in QTL], "true_q": [float(np.quantile(tr, q_)) for q_ in (0.1, 0.5, 0.9)], "verdict": verdict}
    (ROOT / "results/r_variants/m251_pairs.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m251_pairs.json")


if __name__ == "__main__":
    main()
