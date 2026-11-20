#!/usr/bin/env python
"""M256 (2026-11-20), report only: why do two "units" of the same person read at different levels? (design doc M256,
registered before any result)

(A) exact opponents of every training id (the other row with the identical SGF text): for the validation twins (mutual
    game-by-game favourites inside a fold, vote share >= 0.5 - the list of m255_val.py) the overlap of the opponent sets
    (controls: one id split in two halves, strangers of the same label), the slope of the opponents' mean label (even games)
    on the label difference, games per id, direct games between twins; ids that appear in two rank files.
(B) the level gap of a pair split in two parts - base model (expected level) and anchors (expected level under pi x SQ2
    evidence) - for same-id pairs (two disjoint 10-game questions), twins (one 10-game question each; equal / different
    labels) and the 20 test pairs of M255.
(C) identifier-level shared opponents: among the clearly identified opponents (m15 above the impostor q95) of the two sides,
    the share of (game, game) combinations with the same first candidate: same id, twins, strangers, test pairs.
(D) the 20 test pairs by the colour of the test question and the HD shares.

  python -I logs/nano4/m256_why.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m105_voters as m5  # noqa: E402,F401
import m127_conf_family as m127  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m251b_votes as mv  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

m3, b, s2, ES, FOLDS, CACHE = mb7.m3, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.CACHE
RV = ROOT / "results/r_variants"; OUT = ROOT / "results/m255"; STAT, X = "m15", "SQ2"; RNG = np.random.default_rng(2560); LV = np.arange(10.0); HD = re.compile(r"HD\[(\d+)\]"); RANK = {lv: i for i, lv in enumerate(LEVELS)}


def soft(L):
    L = L - L.max(-1, keepdims=True); P = np.exp(L); return P / P.sum(-1, keepdims=True)


def zscore(Sc, ok=None):
    if ok is None:
        return (Sc - Sc.mean(1, keepdims=True)) / np.maximum(Sc.std(1, keepdims=True), 1e-12)
    cnt = max(int(ok.sum()), 1); mu = (Sc * ok).sum(1, keepdims=True) / cnt; sd = np.sqrt((((Sc - mu) ** 2) * ok).sum(1, keepdims=True) / cnt); return np.where(ok[None], (Sc - mu) / np.maximum(sd, 1e-12), 0.0)


def desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"n {len(x)}, mean {x.mean():+.2f}, sd {x.std():.2f}, |.| median {np.median(np.abs(x)):.2f}" if len(x) else "n 0"


def corr(a, c):
    a, c = np.asarray(a, float), np.asarray(c, float); m = np.isfinite(a) & np.isfinite(c)
    return float(np.corrcoef(a[m], c[m])[0, 1]) if m.sum() >= 5 else float("nan")


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    val = json.load(open(RV / "m255_val.json")); xscale = val["density_ratio"]; pi = np.array(val["seeds_0"]["pi"]); lpi = np.log(pi); rep = {}
    name = f"{X}~{STAT}"; V, T, key_pid, used = b.sources([name]); zv = dict(np.load(RV / f"m84_{name}_val.npz")); thr = float(np.quantile(zv["f_imp_m"], 0.95)); vkey = zv["key"].astype(str); vrow = {k: i for i, k in enumerate(vkey)}; vbest = zv["f_opp_best"].astype(str); vm = zv["f_opp_m"].astype(float)
    LF = {v: m3.load_fold(v) for v in FOLDS}; base_of = lambda v: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)  # noqa: E731
    # ---------------------------------------------------------------- units of the validation side: every 10-game question
    U = {}; U20 = {}  # (fold, pid) -> list of 10-game units / the 20-game unit
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y).astype(int), np.asarray(n).astype(int), np.asarray(pid).astype(str); gk = [list(map(str, g_)) for g_ in qd["game_keys"]]; PB = base_of(v); ev = {str(k): e for k, e in m3.evidence(V[name], v, 1, 1, 1, xscale).items()}
        for j in np.flatnonzero(n == 20):
            r = [vrow[k] for k in gk[j] if k in vrow]; cl = [i for i in r if vm[i] > thr]; E = sum((ev.get(k, m3.Z10) for k in gk[j]), m3.Z10)
            U20[(v, pid[j])] = dict(y=int(y[j]), games=set(gk[j]), mb=float(PB[j] @ LV), ma=float(soft(lpi + E) @ LV) if len(cl) >= 2 else np.nan, clear=[vbest[i] for i in cl], hd1=float(np.mean(zv["hd"][r] > 0)) if r else np.nan)
        for j in np.flatnonzero(n == 10):
            r = [vrow[k] for k in gk[j] if k in vrow]; cl = [i for i in r if vm[i] > thr]; E = sum((ev.get(k, m3.Z10) for k in gk[j]), m3.Z10)
            U.setdefault((v, pid[j]), []).append(dict(y=int(y[j]), games=set(gk[j]), mb=float(PB[j] @ LV), ma=float(soft(lpi + E) @ LV) if len(cl) >= 2 else np.nan, clear=[vbest[i] for i in cl], hd1=float(np.mean(zv["hd"][r] > 0)) if r else np.nan))
    same = [(u[0], u[1]) for u in U.values() if len(u) >= 2 and not (u[0]["games"] & u[1]["games"])]; log(f"validation: {len(U)} players with 10-game questions, {len(same)} same-id pairs of disjoint 10-game questions; clear-identification line (impostor q95 of m15) {thr:.3f}")
    # twins (m255_val.py's list)
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); vecs = []
    for nm in mv.NETS:
        Vv = np.load(CACHE / f"var_{nm}.npz")["qvec"]; vecs += [Vv[m] for m in range(Vv.shape[0])]
    dv = np.load(RV / "m82_val.npz"); pidv = dv["pid"].astype(str); foldv = dv["fold"].astype(int); yv = dv["y"].astype(int); tw = []
    for v in FOLDS:
        idx = np.flatnonzero(foldv == v); u, names = pd.factorize(pidv[idx]); k = np.bincount(u); keep = k[u] >= 10; idx = idx[keep]; u, names = pd.factorize(pidv[idx]); nu = len(names); lab = np.zeros(nu, int); lab[u] = yv[idx]
        Fs = mv.scores(vecs, Q + idx, Q + idx, u, nu, u); W, fav, share, mut, _, _, kk = mv.votes(Fs, u, nu)
        tw += [(v, str(names[a]), str(names[fav[a]]), float(min(share[a], share[fav[a]])), int(lab[a]), int(lab[fav[a]])) for a in range(nu) if mut[a] and a < fav[a] and min(share[a], share[fav[a]]) >= 0.5]
    del vecs; log(f"twins (mutual favourites inside a fold, vote share >= 0.5): {len(tw)} pairs; labels equal {np.mean([t[4] == t[5] for t in tw]):.3f}, one apart {np.mean([abs(t[4] - t[5]) == 1 for t in tw]):.3f}, two or more {np.mean([abs(t[4] - t[5]) >= 2 for t in tw]):.3f}")
    # ---------------------------------------------------------------- (A) exact opponents from identical SGF texts
    rows = m81.load_rows(); rows["lab"] = rows["rank"].map(RANK); nr = rows.groupby("player_id")["rank"].nunique(); print(f"\n(A) training rows {len(rows)}, ids {rows.player_id.nunique()}; ids that appear under two or more ranks: {int((nr > 1).sum())}; games per id 10 / 50 / 90 %: " + " / ".join(str(int(x)) for x in rows.groupby('player_id').size().quantile([0.1, 0.5, 0.9])))
    dup = rows.sgf_content.duplicated(keep=False); print(f"    rows whose SGF text also appears in another row: {int(dup.sum())} -> every game is listed under ONE player only (no exact opponent lists; the opponent analysis below uses the identifier)")
    ngames = rows.groupby("player_id").size().to_dict(); ng_t = [ngames[x] for t in tw for x in (t[1], t[2])]; print(f"    games per twin id median {int(np.median(ng_t))} (all ids {int(np.median(list(ngames.values())))}); twin pairs by fold: " + ", ".join(f"{v}: {sum(t[0] == v for t in tw)}" for v in FOLDS))
    rep["A"] = {"twins": len(tw), "duplicated_rows": int(dup.sum()), "ids_two_ranks": int((nr > 1).sum())}; del rows
    # ---------------------------------------------------------------- (B) + (C) on the validation side
    def gap(pairs):
        db = np.array([b2["mb"] - a2["mb"] for a2, b2 in pairs]); da = np.array([b2["ma"] - a2["ma"] for a2, b2 in pairs]); num = sum(sum(x == y_ for x in a2["clear"] for y_ in b2["clear"]) for a2, b2 in pairs); den = sum(len(a2["clear"]) * len(b2["clear"]) for a2, b2 in pairs)
        any_ = float(np.mean([bool(set(a2["clear"]) & set(b2["clear"])) for a2, b2 in pairs])); return db, da, num / max(den, 1), any_
    first = lambda v, p_: U[(v, p_)][0] if (v, p_) in U else None  # noqa: E731
    tw_u = [(first(v, a), first(v, b_), la, lb) for v, a, b_, sh, la, lb in tw]; tw_u = [x for x in tw_u if x[0] and x[1]]; keys = list(U.keys()); bylab = {}
    for k_ in keys:
        bylab.setdefault((k_[0], U[k_][0]["y"]), []).append(k_)
    strangers = []
    for k_ in keys[::3]:
        o = bylab[(k_[0], U[k_][0]["y"])]; k2 = o[RNG.integers(len(o))]
        if k2 != k_:
            strangers.append((U[k_][0], U[k2][0]))
    print("\n(B) the level gap of a pair, split in two parts (B minus A): base model | anchors (pi x SQ2 evidence; both sides need >= 2 clear opponents) | correlation of the two parts")
    print("(C) identifier-level shared opponents: share of (clear game of A, clear game of B) combinations with the same first candidate | pairs with at least one")
    G = {}
    for nm, pr in (("same id (two disjoint 10-game questions)", same), ("twins, all", [(x[0], x[1]) for x in tw_u]), ("twins, equal labels", [(x[0], x[1]) for x in tw_u if x[2] == x[3]]), ("twins, different labels", [(x[0], x[1]) for x in tw_u if x[2] != x[3]]), ("strangers of the same label", strangers)):
        db, da, rate, any_ = gap(pr); G[nm] = (db, da); rep[nm] = {"n": len(pr), "base_abs_median": float(np.median(np.abs(db))), "anchor_abs_median": float(np.nanmedian(np.abs(da))), "corr": corr(db, da), "shared_rate": rate, "any_shared": any_}
        print(f"   {nm:42s}: base {desc(db)} | anchors {desc(da)} | corr {corr(db, da):+.2f} || shared-opponent rate {rate:.4f}, pairs with a shared clear opponent {any_:.3f}")
    t20 = [(U20[(v, a_)], U20[(v, b_)], la, lb) for v, a_, b_, sh, la, lb in tw if (v, a_) in U20 and (v, b_) in U20]
    for nm, pr in (("twins, all, 20-game units", [(x[0], x[1]) for x in t20]), ("twins, equal labels, 20-game units", [(x[0], x[1]) for x in t20 if x[2] == x[3]]), ("twins, different labels, 20-game units", [(x[0], x[1]) for x in t20 if x[2] != x[3]])):
        db, da, rate, any_ = gap(pr); rep[nm] = {"n": len(pr), "base_abs_median": float(np.median(np.abs(db))), "anchor_abs_median": float(np.nanmedian(np.abs(da))), "corr": corr(db, da), "shared_rate": rate, "any_shared": any_}
        print(f"   {nm:42s}: base {desc(db)} | anchors {desc(da)} | corr {corr(db, da):+.2f} || shared-opponent rate {rate:.4f}, pairs with a shared clear opponent {any_:.3f}")
    d20 = np.array([x[3] - x[2] for x in t20], float); b20 = np.array([x[1]["mb"] - x[0]["mb"] for x in t20]); a20 = np.array([x[1]["ma"] - x[0]["ma"] for x in t20]); mfin = np.isfinite(a20)
    print(f"   twins, 20-game units: gap (b - a) on the label difference (b - a): base slope {float(np.polyfit(d20, b20, 1)[0]):+.2f}, anchors slope {float(np.polyfit(d20[mfin], a20[mfin], 1)[0]):+.2f}; by label difference: " + "; ".join(f"{int(d_):+d}: base {b20[d20 == d_].mean():+.2f}, anchors {np.nanmean(a20[d20 == d_]):+.2f} (n {int((d20 == d_).sum())})" for d_ in sorted(set(d20.tolist()))))
    td = [(x[0], x[1], x[3] - x[2]) for x in tw_u if x[2] != x[3]]
    if td:
        d_ = np.array([t[2] for t in td], float); print(f"   twins with different labels ({len(td)}): slope on the label difference - base {float(np.polyfit(d_, [t[1]['mb'] - t[0]['mb'] for t in td], 1)[0]):+.2f}; anchors " + (lambda m_: f"{float(np.polyfit(d_[m_], np.array([t[1]['ma'] - t[0]['ma'] for t in td])[m_], 1)[0]):+.2f} ({int(m_.sum())} pairs)")(np.isfinite([t[1]['ma'] - t[0]['ma'] for t in td])))
    # ---------------------------------------------------------------- the 20 test pairs
    games = []
    for f in ("bq.csv", "ctl.csv"):
        for d in pd.read_csv(OUT / "inputs" / f, dtype=str, keep_default_na=False).to_dict("records"):
            games += [(d["question_id"], d[f"color_{i}"] == "B", int(HD.search(d[f"sgf_{i}"]).group(1))) for i in range(1, int(d["num_games"]) + 1)]
    gq = np.array([g_[0] for g_ in games]); tblack = np.array([g_[1] for g_ in games]); hdn = np.array([g_[2] for g_ in games]); z = np.load(OUT / "s7t_new.npz"); assert np.array_equal(z["qid"].astype(str), gq)
    blk = np.load(OUT / "block_new.npy").astype(np.float64); qv = z["qv_opp"].astype(np.float64); players = q["players"].astype(str); plq = q["plab"].astype(int)
    ngp = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players).to_numpy(); ev_ = json.load(open(RV / f"m84_{X}_eval.json")); w = ev_["weights"]; beta = ev_["beta"]
    Vp = np.load(CACHE / "var_S7t.npz")["protos"][:, 4].astype(np.float64); Vc = np.load(CACHE / "var_S7tc.npz"); cen = Vc["protos"][:, 4].astype(np.float64); hs = Vc["has"][:, 4]
    F = w[0] * blk + w[1] * np.mean([zscore(qv[m] @ Vp[m].T) for m in range(2)], 0) + w[2] * np.mean([zscore(qv[m] @ cen[m].T, hs[m]) for m in range(2)], 0) + beta * np.log(ngp)[None]
    order = np.argsort(-F, 1)[:, :20]; TV = np.take_along_axis(F, order, 1).astype(np.float32); bestn = players[order[:, 0]]; m15 = m127.simple_stats(TV)[STAT].astype(float); labn = plq[order[:, 0]]
    t82 = np.load(RV / "m82_test.npz"); tq = t82["qid"].astype(str); tS = np.load(RV / f"m84_{name}_test.npz")
    logE = m3.test_evidence({name: V[name]}, {name: {"target_black": tblack, "hd": hdn.astype(t82["hd"].dtype), "opp_m": m15, "opp_lab": labn}}, name, 1, 1, xscale, 1)
    logS = m3.test_evidence({name: V[name]}, {name: {"target_black": t82["target_black"], "hd": t82["hd"], "opp_m": tS["f_opp_m"], "opp_lab": tS["f_opp_lab"]}}, name, 1, 1, xscale, 1); tbest = tS["f_opp_best"].astype(str); tm = tS["f_opp_m"].astype(float)
    cols = [f"p_{lv}" for lv in LEVELS]; pb = pd.read_csv(OUT / "base/submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id"); p9 = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id"); mp_ = pd.read_csv(OUT / "inputs/map.csv", dtype=str); tp = []
    print("\n(D) the 20 test pairs: question (games, colour, share of HD1 games) | candidate's 10 games (share of HD1 games) | base gap | anchor gap | clear opponents A / B | shared")
    for r in mp_.to_dict("records"):
        ia = np.flatnonzero(tq == r["test_question"]); ib = np.flatnonzero(gq == r["b_question"]); ca = [tbest[i] for i in ia if tm[i] > thr]; cb = [bestn[i] for i in ib if m15[i] > thr]
        A = dict(mb=float(p9.loc[r["test_question"], cols].to_numpy(float) @ LV), ma=float(soft(lpi + logS[ia].sum(0)) @ LV) if len(ca) >= 2 else np.nan, clear=ca, hd1=float(np.mean(t82["hd"][ia] > 0))); B_ = dict(mb=float(pb.loc[r["b_question"], cols].to_numpy(float) @ LV), ma=float(soft(lpi + logE[ib].sum(0)) @ LV) if len(cb) >= 2 else np.nan, clear=cb, hd1=float(np.mean(hdn[ib] > 0)))
        tp.append((A, B_, r["test_color"], int(r["n_test"]), r["test_question"])); sh_ = sorted(set(ca) & set(cb))
        print(f"   {r['test_question']} ({int(r['n_test']):2d}, {r['test_color']}, HD1 {A['hd1']:.2f}) | {r['candidate']} (HD1 {B_['hd1']:.2f}) | base {B_['mb'] - A['mb']:+.2f} | anchors {B_['ma'] - A['ma']:+.2f} | {len(ca)} / {len(cb)} | {len(sh_)}" + (f" ({', '.join(sh_)})" if sh_ else ""))
    db, da, rate, any_ = gap([(x[0], x[1]) for x in tp]); rep["test pairs"] = {"n": len(tp), "base_abs_median": float(np.median(np.abs(db))), "anchor_abs_median": float(np.nanmedian(np.abs(da))), "corr": corr(db, da), "shared_rate": rate, "any_shared": any_}
    print(f"   {'TEST PAIRS (20)':42s}: base {desc(db)} | anchors {desc(da)} | corr {corr(db, da):+.2f} || shared-opponent rate {rate:.4f}, pairs with a shared clear opponent {any_:.3f}")
    for nm in ("same id (two disjoint 10-game questions)", "twins, all"):
        d0 = G[nm]; pr = {k_: float(np.mean([np.median(np.abs(RNG.choice(x[np.isfinite(x)], n_))) >= obs for _ in range(20000)])) for k_, x, n_, obs in (("base", d0[0], len(db), np.median(np.abs(db))), ("anchors", d0[1], int(np.isfinite(da).sum()), np.nanmedian(np.abs(da))))}
        print(f"   chance that {len(db)} pairs drawn from '{nm}' reach the test pairs' median |gap|: base {pr['base']:.4f}, anchors {pr['anchors']:.4f}")
    for c_ in ("B", "W"):
        m_ = np.array([x[2] == c_ for x in tp]); print(f"   test question played as {c_}: base gap {desc(db[m_])} | anchor gap {desc(da[m_])}")
    hdA = np.array([x[0]["hd1"] for x in tp]); hdB = np.array([x[1]["hd1"] for x in tp]); print(f"   share of HD1 games: test side mean {hdA.mean():.3f}, candidate side mean {hdB.mean():.3f} (all test games {float(np.mean(t82['hd'] > 0)):.3f}, validation games {float(np.mean(zv['hd'] > 0)):.3f}); correlation of the base gap with (HD1 share B - A) {corr(db, hdB - hdA):+.2f}; with the HD1 share of A, black questions {corr(db[[x[2] == 'B' for x in tp]], hdA[[x[2] == 'B' for x in tp]]):+.2f}, white questions {corr(db[[x[2] == 'W' for x in tp]], hdA[[x[2] == 'W' for x in tp]]):+.2f}")
    (RV / "m256_why.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m256_why.json")


if __name__ == "__main__":
    main()
