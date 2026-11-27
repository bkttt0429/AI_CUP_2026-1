#!/usr/bin/env python
"""M255 (2026-11-20), test side of step 2 of the cross-competition linkage (design doc M255 / M255b).

For the 20 pairs of M253: the matched candidate's 10 games as a second question B -
  identifier SQ2 on B's games (block_new.npy + s7t_new.npz, the stored weights / beta of m84_SQ2_eval.json, statistic m15),
  anchor evidence with the validation calibration (m103_stability.test_evidence, density ratio of m255_val.json),
  base posterior of B from results/m255/base/submission_rank_ct9_proba.csv,
  log P' = log P_A + a [log P_B - log pi] + b E_B with the all-fold coefficients of m255_val.json (A of 15 / 20 games: A = 10's).
Checks on the six control questions (original test questions run through the same new code): base posterior against
submission_rank_ct9_proba.csv, SQ2's first candidate and m15 against m84_SQ2~m15_test.npz.
Report per pair; the label-free statistic D; the decisions under the production guards and the standing rules relative to
ct116; the Public pairs against the ledger's truth weights where they exist. Files go to results/m255/ only (never the root).

  python -I logs/nano4/m255_apply.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m127_conf_family as m127  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m58_decode as led  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

m3, b, CACHE = mb7.m3, mb7.b, mb7.CACHE; OUT = ROOT / "results/m255"; RV = ROOT / "results/r_variants"; STAT, X = "m15", "SQ2"; LV = np.arange(10.0); HD = re.compile(r"HD\[(\d+)\]")
S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])


def soft(L):
    L = L - L.max(-1, keepdims=True); P = np.exp(L); return P / P.sum(-1, keepdims=True)


def zscore(Sc, ok=None):
    if ok is None:
        return (Sc - Sc.mean(1, keepdims=True)) / np.maximum(Sc.std(1, keepdims=True), 1e-12)
    cnt = max(int(ok.sum()), 1); mu = (Sc * ok).sum(1, keepdims=True) / cnt; sd = np.sqrt((((Sc - mu) ** 2) * ok).sum(1, keepdims=True) / cnt); return np.where(ok[None], (Sc - mu) / np.maximum(sd, 1e-12), 0.0)


def top3(p):
    return " ".join(f"{LEVELS[k]} {p[k]:.2f}" for k in np.argsort(-p)[:3] if p[k] >= 0.01)


def main():
    val = json.load(open(RV / "m255_val.json")); xscale = val["density_ratio"]; pi = np.array(val["seeds_0"]["pi"]); lpi = np.log(pi); co = {int(s): (d["all_folds"]["a"], d["all_folds"]["b"]) for s, d in val["seeds_0"]["sizes"].items()}; q99 = {int(s): d["D_same"]["abs_quantiles"]["0.99"] for s, d in val["seeds_0"]["sizes"].items()}
    co[15] = co[20] = co[10]; q99[15] = q99[20] = q99[10]; print("coefficients (a, b) by the size of A: " + ", ".join(f"{s}: ({co[s][0]:.3f}, {co[s][1]:.3f})" for s in (5, 10, 15, 20)) + f"; density ratio {xscale:.3f}")
    # ---------------------------------------------------------------- the new games (task order)
    games = []
    for f in ("bq.csv", "ctl.csv"):
        for d in pd.read_csv(OUT / "inputs" / f, dtype=str, keep_default_na=False).to_dict("records"):
            games += [(d["question_id"], d[f"color_{i}"] == "B", int(HD.search(d[f"sgf_{i}"]).group(1))) for i in range(1, int(d["num_games"]) + 1)]
    gq = np.array([g[0] for g in games]); tblack = np.array([g[1] for g in games]); hd = np.array([g[2] for g in games]); N = len(games); z = np.load(OUT / "s7t_new.npz"); assert np.array_equal(z["qid"].astype(str), gq), "vector parts and the input files are not in the same order"
    blk = np.load(OUT / "block_new.npy").astype(np.float64); qv = z["qv_opp"].astype(np.float64); assert blk.shape[0] == N == qv.shape[1]
    q = np.load(CACHE / "queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"])
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players).to_numpy(); logn = np.log(ng); ev_ = json.load(open(RV / f"m84_{X}_eval.json")); w = ev_["weights"]; beta = ev_["beta"]; assert ev_["sources"] == ["S7t", "S7tc"]
    Vp = np.load(CACHE / "var_S7t.npz")["protos"][:, 4].astype(np.float64); Vc = np.load(CACHE / "var_S7tc.npz"); cen = Vc["protos"][:, 4].astype(np.float64); has = Vc["has"][:, 4]
    z1 = np.mean([zscore(qv[m] @ Vp[m].T) for m in range(2)], 0); z2 = np.mean([zscore(qv[m] @ cen[m].T, has[m]) for m in range(2)], 0); F = w[0] * blk + w[1] * z1 + w[2] * z2 + beta * logn[None]
    order = np.argsort(-F, 1)[:, :20]; TV = np.take_along_axis(F, order, 1).astype(np.float32); best = order[:, 0]; m15 = m127.simple_stats(TV)[STAT]; lab = plab[best]
    # ---------------------------------------------------------------- checks on the control questions
    cm = pd.read_csv(OUT / "inputs/ctl_map.csv", dtype=str); orig = dict(zip(cm.ctl_question, cm.test_question)); t82 = np.load(RV / "m82_test.npz"); tq = t82["qid"].astype(str); tS = np.load(RV / f"m84_{X}~{STAT}_test.npz"); seen = {}; same, dm, okf = [], [], []
    for i, x in enumerate(gq):
        if x in orig:
            j = seen.get(x, 0); seen[x] = j + 1; task = int(np.flatnonzero(tq == orig[x])[j]); same.append(str(tS["f_opp_best"][task]) == players[best[i]]); dm.append(abs(float(tS["f_opp_m"][task]) - float(m15[i]))); okf.append(bool(t82["target_black"][task]) == bool(tblack[i]) and int(t82["hd"][task]) == int(hd[i]))
    print(f"CHECK identifier {X} on {len(same)} control games: first candidate equal {np.mean(same):.3f}; max |m15 difference| {np.max(dm):.4f}; colour / HD fields equal {np.mean(okf):.3f} -> {'PASS' if np.mean(same) >= 0.95 and np.mean(okf) == 1.0 else 'FAIL'}")
    pb = pd.read_csv(OUT / "base/submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id"); p9 = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id"); cols = [f"p_{lv}" for lv in LEVELS]
    dmax = max(float(np.abs(pb.loc[c, cols].to_numpy(float) - p9.loc[o, cols].to_numpy(float)).max()) for c, o in orig.items()); dsame = all(int(np.asarray(b.decide(pb.loc[[c], cols].to_numpy(float)), int)[0]) == int(np.asarray(b.decide(p9.loc[[o], cols].to_numpy(float)), int)[0]) for c, o in orig.items())
    print(f"CHECK base posterior on the {len(orig)} control questions against submission_rank_ct9_proba.csv: max |dp| {dmax:.4f}, decisions equal {dsame} -> {'PASS' if dmax <= 0.004 and dsame else 'FAIL'}")
    # ---------------------------------------------------------------- evidence of B's games
    V, _, _, used = b.sources([f"{X}~{STAT}"]); name = f"{X}~{STAT}"; Tn = {"target_black": tblack, "hd": hd.astype(t82["hd"].dtype), "opp_m": m15.astype(np.float64), "opp_lab": lab}
    logE = m3.test_evidence({name: V[name]}, {name: Tn}, name, 1, 1, xscale, 1); EQ = pd.DataFrame(logE).groupby(gq).sum()
    # control: the same games' evidence from the stored test file
    Ts = {"target_black": t82["target_black"], "hd": t82["hd"], "opp_m": tS["f_opp_m"], "opp_lab": tS["f_opp_lab"]}; logS = m3.test_evidence({name: V[name]}, {name: Ts}, name, 1, 1, xscale, 1); ES_ = pd.DataFrame(logS).groupby(tq).sum()
    print("CHECK evidence of the control questions against the stored test games: max |difference| " + f"{max(float(np.abs(EQ.loc[c].to_numpy() - ES_.loc[o].to_numpy()).max()) for c, o in orig.items()):.4f}")
    # ---------------------------------------------------------------- the pairs
    mp_ = pd.read_csv(OUT / "inputs/map.csv", dtype=str); zt = np.load(RV / "m163_test_posterior_ct112.npz", allow_pickle=True); qid = zt["qid"].astype(str); P = zt["learned"].astype(float); P = P / P.sum(1, keepdims=True); pos = {x: i for i, x in enumerate(qid)}
    sub = lambda f: led.decisions(ROOT / f).reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112, c116, c118e = (sub(f"submission_rank_{f}.csv") for f in ("ct9", "ct28", "ct106", "ct112", "ct116", "ct118e")); locked = c28 != c9; plain = np.asarray(b.decide(P), int); assert (mf7.guard2(np.where(locked, c28, plain), c106) == c112).all(), "ct112 is not reproduced"
    prof = {}
    for f, key in (("m210_profiles.json", None), ("m211_ledger_ct116.json", "with_composition"), ("m254_update.json", None)):
        for k, d in json.load(open(RV / f)).items():
            d = d.get(key) if key else d
            if d:
                prof[k] = np.array([d.get(l, 0.0) for l in LEVELS])
    Pn = P.copy(); rows = []; Dall = []
    print("\nquestion (games, colour) <- candidate | A: ct112-rule posterior | B alone | product | ct116 answer -> new plain answer")
    for r in mp_.to_dict("records"):
        i = pos[r["test_question"]]; nA = int(r["n_test"]); a_, b_ = co[nA]; lb = np.log(np.maximum(pb.loc[r["b_question"], cols].to_numpy(float), 1e-12)) - lpi; eb = EQ.loc[r["b_question"]].to_numpy(); nanc = int((np.abs(logE[gq == r["b_question"]]).max(1) > 0.5).sum())
        PB = soft(lpi + a_ * lb + b_ * eb); Pn[i] = soft(np.log(np.maximum(P[i], 1e-12)) + a_ * lb + b_ * eb); D = float(PB @ LV - P[i] @ LV); Dall.append(D); dn = int(np.asarray(b.decide(Pn[i][None]), int)[0])
        rows.append(dict(q=r["test_question"], i=i, n=nA, cand=r["candidate"], D=D, drop=abs(D) > q99[nA], plain_new=dn, anchors=nanc, base_B=top3(pb.loc[r["b_question"], cols].to_numpy(float))))
        print(f"  {r['test_question']} ({nA:2d}, {r['test_color']}) <- {r['candidate']} | A {top3(P[i])} | B base {rows[-1]['base_B']}; clear anchors {nanc}; B alone {top3(PB)} | product {top3(Pn[i])} | D {D:+.2f}{' DROPPED (|D| above the same-player 99 % line)' if rows[-1]['drop'] else ''} | {LEVELS[c116[i]]} -> {LEVELS[dn]}")
    Dall = np.array(Dall); se = float(Dall.std(ddof=1) / np.sqrt(len(Dall))); print(f"\nconsistency statistic D over the 20 pairs: mean {Dall.mean():+.3f} (se {se:.3f}), sd {Dall.std(ddof=1):.3f}, |D| median {np.median(np.abs(Dall)):.2f}, max {np.abs(Dall).max():.2f}; validation: same player sd 0.99 / 1.25 (|D| median 0.36 / 0.40), strangers sd 3.9 (median 2.8); registered gate |mean| <= 0.35: {'PASS' if abs(Dall.mean()) <= 0.35 else 'FAIL'}")
    # ---------------------------------------------------------------- decisions: production guards (relative to ct106), then the standing rules relative to ct116
    use = np.zeros(len(qid), bool); use[[r["i"] for r in rows if not r["drop"]]] = True; plain2 = np.where(use, np.asarray(b.decide(Pn), int), plain); d112 = mf7.guard2(np.where(locked, c28, plain2), c106); edited = c116 != c112
    prop = np.where(edited | ~use, c116, d112); fin = mf7.guard2(prop, c116); pub = np.char.startswith(qid, "pub")
    print("\nchanges against ct116 (after the guards relative to ct106 and to ct116; locked and user-set answers untouched):")
    for r in rows:
        i = r["i"]; tag = []
        if locked[i]:
            tag.append("locked by ct28")
        if edited[i]:
            tag.append("answer set by the user 11-17")
        if plain2[i] != plain[i] and fin[i] == c116[i] and not (locked[i] or edited[i]):
            tag.append(f"plain decision {LEVELS[plain[i]]} -> {LEVELS[plain2[i]]} held by a guard")
        line = f"  {r['q']}: ct116 {LEVELS[c116[i]]}" + (f" (ct118e {LEVELS[c118e[i]]})" if pub[i] and c118e[i] != c116[i] else "") + f" -> {LEVELS[fin[i]]}" + ("  CHANGED" if fin[i] != c116[i] else "") + (" [" + "; ".join(tag) + "]" if tag else "")
        if r["q"] in prof:
            wv = prof[r["q"]] / max(prof[r["q"]].sum(), 1e-12); cur = c118e[i]; line += f" | ledger truth {top3(wv)}: expected score of the Public answer now {float(S[cur] @ wv):.3f}, of the new answer {float(S[fin[i]] @ wv):.3f}; ledger weight of the truth under A {float(P[i] @ wv):.3f}, under the product {float(Pn[i] @ wv):.3f}"
        print(line)
    ch = np.flatnonzero(fin != c116); print(f"\ntotal: {len(ch)} answers change (Public {int(pub[ch].sum())}, Private {int((~pub[ch]).sum())}); expected score change by the product posterior itself: Public {float(sum(S[fin[i]] @ Pn[i] - S[c116[i]] @ Pn[i] for i in ch if pub[i])):+.2f}, Private {float(sum(S[fin[i]] @ Pn[i] - S[c116[i]] @ Pn[i] for i in ch if not pub[i])):+.2f} points")
    kn = [r for r in rows if r["q"] in prof]
    if kn:
        la = np.mean([np.log(max(float(P[r["i"]] @ (prof[r["q"]] / prof[r["q"]].sum())), 1e-9)) for r in kn]); ln_ = np.mean([np.log(max(float(Pn[r["i"]] @ (prof[r["q"]] / prof[r["q"]].sum())), 1e-9)) for r in kn]); print(f"Public pairs with ledger truth weights ({len(kn)}): mean log weight of the truth under A {la:+.3f}, under the product {ln_:+.3f}")
    # ---------------------------------------------------------------- files (results/m255 only)
    order = pd.read_csv(ROOT / "submission_rank_ct116.csv", dtype=str); idx = np.array([pos[x] for x in order.question_id]); a_pub = np.where(pub, np.where(fin != c116, fin, c118e), c116); a_all = np.where(pub, a_pub, fin)
    for nm, ans in (("m255_public_check.csv", a_pub), ("m255_candidate.csv", a_all)):
        f = OUT / nm; pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEVELS[x] for x in ans[idx]]}).to_csv(f, index=False, lineterminator="\n")
        print(f"written results/m255/{nm}: md5 {hashlib.md5(f.read_bytes()).hexdigest()[:12]}; Public rows differing from ct118e {int((ans != c118e)[pub].sum())}, Private rows differing from ct116 {int((ans != c116)[~pub].sum())}")
    np.savez(OUT / "m255_posterior.npz", qid=qid, product=Pn, base=P, used=use); (OUT / "m255_apply.json").write_text(json.dumps({"pairs": [{k: (v if not isinstance(v, (np.bool_, np.integer, np.floating)) else v.item()) for k, v in r.items()} for r in rows], "D_mean": float(Dall.mean()), "D_sd": float(Dall.std(ddof=1)), "changed": [str(qid[i]) for i in ch]}, indent=1)); print("written results/m255/m255_apply.json")


if __name__ == "__main__":
    main()
