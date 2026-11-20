"""Section 4M M123 (2026-11-14): M115 unchanged except for the identity vectors (var_S4e: the three symmetry-augmented seeds) and the
file name; registered as a single re-test (design doc M123). Original docstring:
Section 4M M115 (2026-11-14): M114's look-alike reading with (a) the average push per level removed (the log reading is centred
per level over the evaluated questions) and (b) a weight per question size picked by cross-fitting (design doc M115).
Below: M114's description; the grid is now sharpness 20 only and weights per size.
Section 4M M114: the target's look-alikes in the identity space as a second estimate of the level (design doc M114).
Reading of a question: mean target-side identity vector of its games (five members, var_S2y) against the prototypes of the training
players the network has seen (validation: the validation fold left out; test: all 8,007); level weights sum exp(beta x similarity)
divided by the level's number of candidates. Base probabilities P' ~ P x reading^w; then the whole ct75u rule on P'.
(beta, w) from {20, 30} x {0.2, 0.3, 0.5}, picked on one half of the folds (seeds 0-2) and judged on the other; seeds 6-8 reuse.
M37: the ct75u rule on P' against the ct75u rule on P - CI lower bound > 0, >= 3 of 4 folds, difference >= 0.0025, both seed sets.
On a pass (or --build): ct76 = the ct75u composition rebuilt on P9' (ct28's 38 locked answers untouched; with w = 0 the builder
must reproduce the files ct44 and ct75u), the reading's distribution on test against validation, and the feasible Public readings.
  python -I logs/nano4/m114_lookalike_test.py [--build]
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0); BE = 20.0; NAME = "ct102"; VAR = "S4e"  # M123: the same test on the symmetry-augmented networks' vectors (ct76-ct99 are taken); E1 = math.exp(-1.0)


def reading(tq_by_member, protos_by_member, cand_lab, beta):
    """log of the balanced level distribution of the look-alikes: tq [members, Q, 128] unit vectors, protos [members, C, 128]."""
    S = 0.0
    for m in range(len(tq_by_member)):
        S = S + tq_by_member[m] @ protos_by_member[m].T / len(tq_by_member)
    oh = np.eye(10)[cand_lab]; w = np.exp(beta * (S - S.max(1, keepdims=True))) @ oh / oh.sum(0); w = (w + 1e-9) / (w + 1e-9).sum(1, keepdims=True)
    return np.log(w), S.max(1), -(w * np.log(w)).sum(1)


def pooled(P, lr, w):
    w = np.asarray(w, float).reshape(-1, 1) if np.ndim(w) else w; x = P * np.exp(w * (lr - lr.max(1, keepdims=True))); return x / x.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a_ = ap.parse_args()
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    Fd = {}; val_stats = []
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); RD = {}
        lr_, smax, ent = reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], BE); RD = lr_ - lr_.mean(0, keepdims=True); val_stats.append(np.stack([n, smax, ent], 1))  # centred per level over the fold's questions
        Fd[v] = (y, n, pid, Ls, RD)
    half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; picks = None; ok = []; rep = {}; ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2))
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S0, SU = {}, {}
        for v in FOLDS:
            yv, nv, _, Ls, RD = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); d0, du = mf.rule75u(P, Ls); S0.setdefault("base", []).append(SM[d0, yv]); SU.setdefault("base", []).append(SM[du, yv])
            for w in WS[1:]:
                d0, du = mf.rule75u(pooled(P, RD, w), Ls); S0.setdefault(w, []).append(SM[d0, yv]); SU.setdefault(w, []).append(SM[du, yv])
            S0[0.0] = S0["base"]; SU[0.0] = SU["base"]
        fs = lambda S, k: np.array([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731
        fsz = lambda S, k, m: np.array([balanced_mean_over_n(s_, yy, nn, sizes=(m,)) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731
        if picks is None:
            best = lambda fl, m: max(WS, key=lambda c: fsz(SU, c, m)[[fi[v] for v in fl]].mean())  # noqa: E731
            picks = {**{v: {m: best(half[1], m) for m in SIZES} for v in half[0]}, **{v: {m: best(half[0], m) for m in SIZES} for v in half[1]}}; final = {m: best(list(FOLDS), m) for m in SIZES}; print(f"weights by question size picked on folds {half[0]}: { {m: best(half[0], m) for m in SIZES} } | on folds {half[1]}: { {m: best(half[1], m) for m in SIZES} } | on all four: {final}")
            for m in SIZES:
                print(f"    n={m}: after the ct75u rule, by weight: " + "  ".join(f"{w}: {fsz(SU, w, m).mean() - fsz(SU, 'base', m).mean():+.4f}" for w in WS[1:]))
        new = [np.where(Fd[v][1] == SIZES[0], SU[picks[v][SIZES[0]]][fi[v]], np.where(Fd[v][1] == SIZES[1], SU[picks[v][SIZES[1]]][fi[v]], SU[picks[v][SIZES[2]]][fi[v]])) for v in FOLDS]; old = SU["base"]; dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new, old, ys, ns)]
        good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); ok.append(good); A_, B_ = np.concatenate(new), np.concatenate(old)
        print(f"seeds {seeds}: ct75u rule on P' against the ct75u rule on P (cross-fitted picks): {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}   (scores {fs(SU, 'base').mean():.4f} -> {np.mean([balanced_mean_over_n(a, yy, nn, sizes=SIZES) for a, yy, nn in zip(new, ys, ns)]):.4f}; answers changed {float((A_ != B_).mean()):.3f} of the questions by score)")
        print("    by size: " + "  ".join(f"n={m}: {float(np.mean([balanced_mean_over_n(a, yy, nn, sizes=(m,)) - balanced_mean_over_n(c, yy, nn, sizes=(m,)) for a, c, yy, nn in zip(new, old, ys, ns)])):+.4f}" for m in SIZES) + " | by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)), flush=True)
        ng_ = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(pid.astype(str)).to_numpy()
        print("    by the player's number of games in the data set (plain mean of the score difference; all sizes | 5-game questions): " + "  ".join(f"{nm_}: {float((A_ - B_)[s_].mean()):+.4f} | {float((A_ - B_)[s_ & (n == 5)].mean()):+.4f}" for nm_, s_ in (("1-25", ng_ <= 25), ("26-40", (ng_ > 25) & (ng_ <= 40)), ("41-70", (ng_ > 40) & (ng_ <= 70)), ("71-170", (ng_ > 70) & (ng_ <= 170)), ("171+", ng_ > 170))), flush=True)
        rep[f"seeds_{seeds[0]}"] = {**dd, "per_fold": pf, "pass": bool(good)}
    passed = bool(all(ok)); print(f"PRE-REGISTERED (M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep.update(passed=passed, final={str(k): v for k, v in final.items()}); (ROOT / "results/r_variants/m123_lookalike_sym.json").write_text(json.dumps(rep, indent=1, default=float))
    if not (passed or a_.build):
        return
    # ---------------------------------------------------------------- test side
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75u")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}
    for name in ("v44",) + FIVE:
        idn, enc, cc, temp, pi0, bmin, scale = REC[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, scale, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()
        LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)

    def compose(P):
        dR, dA = {}, {}
        for name in LQ:
            d0, da, dr, _, region = mb.decisions(P, LQ[name], 0.5, 3.0, 0.6); dR[name] = np.where(locked, l28, dr); dA[name] = np.where(locked, l28, da)
        base = np.where(locked, l28, d0); d44 = dR["v44"]; keep = d44 != base; fin = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE])
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != d44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != base); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]
        return d44, fin
    d44_0, fin_0 = compose(P9); assert (lev[d44_0] == c44).all(), int((lev[d44_0] != c44).sum()); assert (lev[fin_0] == c75u).all(), int((lev[fin_0] != c75u).sum()); print("checks: with weight 0 the builder reproduces the files ct44 and ct75u")
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); own = float(((QV[0, tgt].astype(np.float64) @ tq[0].T).argmax(1) == gq).mean()); print(f"alignment check (same member): a test game's target-side vector is closest to its own question's mean vector in {own:.3f} of the games"); assert own > 0.5, "test games and questions are not aligned"
    lr, smax20, ent20 = reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, BE); lr = lr - lr.mean(0, keepdims=True); wq = np.array([{5: final[5], 10: final[10], 15: (final[10] + final[20]) / 2, 20: final[20]}[int(m)] for m in nq_t]); vs = np.concatenate(val_stats)
    print("the reading on test against validation (sharpness 20), by question size: mean top similarity | mean entropy of the level distribution")
    for m in (5, 10, 15, 20):
        s_t = nq_t == m; s_v = vs[:, 0] == m; print(f"    n={m}: test {smax20[s_t].mean():.3f} | {ent20[s_t].mean():.3f}" + (f"   validation {vs[s_v, 1].mean():.3f} | {vs[s_v, 2].mean():.3f}" if s_v.any() else "   (no validation questions of this size)"))
    d44_1, fin_1 = compose(pooled(P9, lr, wq)); fin = lev[fin_1]
    for nm, o in (("ct75u", c75u), ("ct44", c44), ("ct28", c28)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{NAME} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    print("predicted levels, ct75u -> " + NAME + ": " + "  ".join(f"{lv} {int((c75u == lv).sum())}->{int((fin == lv).sum())}" for lv in LEVELS))
    f = ROOT / f"submission_rank_{NAME}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw); print(f"{f.name} written, md5 {md5}; weights by size = {final} (15 games: the mean of 10 and 20); passed the registered test: {passed}")


if __name__ == "__main__":
    main()
