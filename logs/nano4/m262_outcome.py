#!/usr/bin/env python
"""M262 step 1 (2026-11-20), report only: CLEAN game outcomes (strong KataGo net, no search) on the test side and in validation.
(design doc M262, registered before any reading)

Outcome of a game = sign of the labelled player's mean score lead over the last four plies (scripts/ai_state_extract.py states; the
features were computed with komi 7.5 for every game, so Black's lead gets +1.0 in even games and +7.5 in HD1 games).
  (A) label-free: dispersion of the win share inside a question (even games), test questions against validation questions of
      the same size: phi = mean squared binomial z; ratio test / validation, bootstrap
  (B) label-free: correlation of the win share with "mean level of the confident anchors - base expected level", test and validation
  (D) validation by account size (< 20 listed games | >= 20): slope of "true level - pooled expected level" on the win share
Writes results/r_variants/m262_outcomes.npz (per test question) for m262_like.py (the Public fair likelihood, part C).

  python -I logs/nano4/m262_outcome.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.scoring import completed_parts, part_paths  # noqa: E402
import m103_stability as m3  # noqa: E402

AI = Path("/work/u2507115/m262_ai"); RV = ROOT / "results/r_variants"; s2, b, ES, FOLDS = m3.s2, m3.b, m3.ES, m3.FOLDS; LV = np.arange(10.0); B = 500


def games_of(vec_dir, ai_dir):
    """Per game of a vec directory: key, last-four-ply means of the labelled player's win probability and score lead, did the player move last, plies; plus the part table's columns."""
    out = []
    for s, k in completed_parts(vec_dir):
        csv_path, npz_path = part_paths(vec_dir, s, k); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False); ai = np.load(ai_dir / f"ai_s{s}_p{k:03d}.npy").astype(np.float32)
        with np.load(npz_path) as z:
            gr, tg = z["game_row"].astype(np.int64), z["target"].astype(bool)
        assert len(ai) == len(gr) and gr.max() == len(tab) - 1, (csv_path, len(ai), len(gr))
        d = pd.DataFrame({"g": gr, "win": np.where(tg, ai[:, 4], 1 - ai[:, 4]), "lead": np.where(tg, ai[:, 5], -ai[:, 5]), "tg": tg.astype(float)}); last = d.groupby("g").tail(4).groupby("g").mean(); fin = d.groupby("g").tail(1).set_index("g")
        tab = tab.assign(win4=last["win"].reindex(range(len(tab))).to_numpy(), lead4=last["lead"].reindex(range(len(tab))).to_numpy(), last_own=fin["tg"].reindex(range(len(tab))).to_numpy()); out.append(tab)
    return pd.concat(out, ignore_index=True)


def outcome(df):
    """Adds lead_adj (komi correction) and won (0 / 1)."""
    adj = np.where(df["hd"].to_numpy(float) == 0, 1.0, 7.5) * np.where(df["color"].to_numpy() == "B", 1.0, -1.0); df = df.assign(lead_adj=df["lead4"].to_numpy(float) + adj); return df.assign(won=(df["lead_adj"] > 0).astype(float))


def phi(wins, ne, p):
    ok = ne >= 3; z = (wins[ok] - ne[ok] * p) / np.sqrt(ne[ok] * p * (1 - p)); return float((z ** 2).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--test-only", action="store_true", help="write the per-question outcomes of the test set and stop (the validation states are not needed)"); a_ = ap.parse_args()
    rng = np.random.default_rng(262); rep = {}
    # ------------------------------------------------------------ test games
    t82 = np.load(RV / "m82_test.npz"); T = games_of(ROOT / "results/ct_submission/vec", AI / "test"); task = T["task"].astype(int).to_numpy(); assert len(T) == len(t82["qid"]) and np.array_equal(t82["qid"].astype(str)[task], T["question_id"].to_numpy())
    T = outcome(T.assign(hd=t82["hd"].astype(float)[task])); assert np.array_equal(t82["target_black"].astype(bool)[task], (T["color"] == "B").to_numpy())
    if a_.test_only:
        e = T[T["hd"] == 0]; print(f"test: {len(T)} games ({len(e)} even); win rate even {e['won'].mean():.3f}; sign of the lead == win probability > 0.5 in {float(((e['win4'] > 0.5) == (e['won'] > 0.5)).mean()):.3f}; 'moved last' == won in {float((e['last_own'] == e['won']).mean()):.3f}", flush=True)
        gt = T.groupby("question_id"); qid = np.array(sorted(gt.groups)); even_t = T[T["hd"] == 0].groupby("question_id")["won"]
        np.savez(RV / "m262_outcomes.npz", qid=qid, n=gt.size().reindex(qid).to_numpy(), n_even=even_t.size().reindex(qid).fillna(0).to_numpy(), wins_even=even_t.sum().reindex(qid).fillna(0).to_numpy(), share_all=gt["won"].mean().reindex(qid).to_numpy(), share_last=gt["last_own"].mean().reindex(qid).to_numpy())
        print("written results/r_variants/m262_outcomes.npz (test only)"); return
    # ------------------------------------------------------------ validation games (folds 1-4, the <= 20 scored games of every player)
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"].astype(float)
    V = pd.concat([games_of(ROOT / f"results/trunk_vec/fold{f}", AI / f"base/fold{f}").assign(fold=f) for f in FOLDS], ignore_index=True); V = outcome(V.assign(hd=hd_all.reindex(V["game_key"]).to_numpy())); V = V[V["hd"] <= 1]
    for nm, d in (("test", T), ("validation", V)):
        e = d[d["hd"] == 0]; agree = float(((e["win4"] > 0.5) == (e["won"] > 0.5)).mean()); prox = float((e["last_own"] == e["won"]).mean())
        print(f"{nm}: {len(d)} games ({len(e)} even); win rate even {e['won'].mean():.3f}, HD1 as Black {d[(d['hd'] == 1) & (d['color'] == 'B')]['won'].mean():.3f}, HD1 as White {d[(d['hd'] == 1) & (d['color'] == 'W')]['won'].mean():.3f}; even games: sign of the lead == win probability > 0.5 in {agree:.3f}; 'moved last' == won in {prox:.3f}; |lead| median {e['lead_adj'].abs().median():.1f}, share with |lead| < 3: {float((e['lead_adj'].abs() < 3).mean()):.3f}", flush=True)
    won_of = dict(zip(V["game_key"], V["won"])); hd_of = dict(zip(V["game_key"], V["hd"])); last_of = dict(zip(V["game_key"], V["last_own"]))
    # ------------------------------------------------------------ test questions
    gt = T.groupby("question_id"); qid = np.array(sorted(gt.groups)); even_t = T[T["hd"] == 0].groupby("question_id")["won"]; n_t = gt.size().reindex(qid).to_numpy(); ne_t = even_t.size().reindex(qid).fillna(0).to_numpy(); w_t = even_t.sum().reindex(qid).fillna(0).to_numpy()
    wall_t = gt["won"].mean().reindex(qid).to_numpy(); last_t = gt["last_own"].mean().reindex(qid).to_numpy()
    np.savez(RV / "m262_outcomes.npz", qid=qid, n=n_t, n_even=ne_t, wins_even=w_t, share_all=wall_t, share_last=last_t)
    # ------------------------------------------------------------ validation questions (the pilot questions of every fold) with the base posterior
    LF = {v: m3.load_fold(v) for v in FOLDS}; rowsV = []; zp = np.load(RV / "m251_val_post.npz"); off = 0
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid).astype(str); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)
        same = off + len(y) <= len(zp["y"]) and np.array_equal(zp["y"][off:off + len(y)], y) and np.array_equal(zp["pid"][off:off + len(y)].astype(str), pid) and np.array_equal(zp["n"][off:off + len(y)], n)
        if not same:
            print(f"   (fold {v}: m251_val_post.npz is not in this fold's question order - the pooled posterior is replaced by the base posterior for this fold)", flush=True)
        post = zp["post"][off:off + len(y)].astype(np.float64) if same else P; off += len(y)
        for i, gk in enumerate(qd["game_keys"]):
            ev = [k for k in gk if hd_of.get(k, 9) == 0]; rowsV.append((v, pid[i], int(y[i]), int(n[i]), len(ev), sum(won_of[k] for k in ev), float(P[i] @ LV), float(post[i] @ LV / post[i].sum()), i, float(np.mean([last_of[k] for k in gk if k in last_of]) if any(k in last_of for k in gk) else 0.5)))
    Q = pd.DataFrame(rowsV, columns=["fold", "pid", "y", "n", "ne", "wins", "eb", "ep", "i", "last"])
    folds_df = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id"); Q["games_total"] = folds_df["n_games"].reindex(Q["pid"]).to_numpy(); up, uq = np.unique(Q["pid"], return_inverse=True)
    # ------------------------------------------------------------ (A) dispersion
    print("\n== (A) dispersion of the win share inside a question, even games (phi = mean squared binomial z; 1 = pure chance at the population's win rate)")
    print("   size | test: questions, win rate, phi, share all-won or all-lost (>= 5 even games) | validation: questions, win rate, phi, share | phi ratio test / validation [95 %]")
    pooled = []
    for size in (5, 10, 15, 20):
        mt = (n_t == size) & (ne_t >= 3); vs = (10, 20) if size == 15 else (size,); mv = Q["n"].isin(vs).to_numpy() & (Q["ne"].to_numpy() >= 3)
        pt = w_t[mt].sum() / ne_t[mt].sum(); pv = Q["wins"][mv].sum() / Q["ne"][mv].sum(); ft, fv = phi(w_t[mt], ne_t[mt], pt), phi(Q["wins"].to_numpy()[mv], Q["ne"].to_numpy()[mv], pv)
        ext = lambda w, ne: float(((w == 0) | (w == ne))[ne >= 5].mean())  # noqa: E731
        bt = []
        for _ in range(B):
            it = rng.choice(np.flatnonzero(mt), int(mt.sum())); wv = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[uq] * mv; iv = np.repeat(np.arange(len(Q)), wv)
            bt.append(phi(w_t[it], ne_t[it], w_t[it].sum() / ne_t[it].sum()) / phi(Q["wins"].to_numpy()[iv], Q["ne"].to_numpy()[iv], Q["wins"].to_numpy()[iv].sum() / Q["ne"].to_numpy()[iv].sum()))
        lo, hi = np.percentile(bt, [2.5, 97.5]); pooled.append((ft / fv, np.array(bt)))
        print(f"   {size:2d} | {int(mt.sum())}, {pt:.3f}, {ft:.3f}, {ext(w_t[mt], ne_t[mt]):.3f} | {int(mv.sum())}, {pv:.3f}, {fv:.3f}, {ext(Q['wins'].to_numpy()[mv], Q['ne'].to_numpy()[mv]):.3f} | {ft / fv:.3f} [{lo:.3f}, {hi:.3f}]")
        rep[f"A_size_{size}"] = {"phi_test": ft, "phi_val": fv, "ratio": ft / fv, "ci": [float(lo), float(hi)], "win_rate_test": float(pt), "win_rate_val": float(pv)}
    gm = float(np.exp(np.mean([np.log(r) for r, _ in pooled]))); gb = np.exp(np.mean([np.log(x) for _, x in pooled], axis=0)); lo, hi = np.percentile(gb, [2.5, 97.5])
    print(f"   all sizes (geometric mean of the four ratios): {gm:.3f} [{lo:.3f}, {hi:.3f}]"); rep["A_all"] = {"ratio": gm, "ci": [float(lo), float(hi)]}
    # validation by account size (the same statistic): do small accounts scatter more?
    for nm, m in (("accounts with < 20 listed games", Q["games_total"] < 20), ("20-39", (Q["games_total"] >= 20) & (Q["games_total"] < 40)), (">= 40", Q["games_total"] >= 40)):
        for size in (5, 10):
            mm = (m & (Q["n"] == size) & (Q["ne"] >= 3)).to_numpy()
            if mm.sum() >= 50:
                pv = Q["wins"][mm].sum() / Q["ne"][mm].sum(); print(f"   validation, {nm}, {size}-game questions: {int(mm.sum())} questions, win rate {pv:.3f}, phi {phi(Q['wins'].to_numpy()[mm], Q['ne'].to_numpy()[mm], pv):.3f}")
    # ------------------------------------------------------------ (B) win share against anchors - base
    print("\n== (B) correlation of the win share (even games) with 'mean level of the confident even anchors - base expected level'")
    zt = np.load(RV / "m163_test_posterior_ct112.npz", allow_pickle=True); pos = {str(x): i for i, x in enumerate(zt["qid"])}; baseT = zt["base"].astype(np.float64); baseT = baseT / baseT.sum(1, keepdims=True); ebT = np.array([baseT[pos[x]] @ LV for x in qid]); epT = np.array([(zt["learned"][pos[x]] / zt["learned"][pos[x]].sum()) @ LV for x in qid])
    st = np.load(RV / "m84_SQ4~m15_test.npz"); om, ol = st["f_opp_m"].astype(float), st["f_opp_lab"].astype(float); qg = t82["qid"].astype(str); hdg = t82["hd"].astype(int); thr = float(np.quantile(om, 0.7)); c = (om >= thr) & (hdg == 0)
    aT = pd.Series(ol[c]).groupby(qg[c]).mean().reindex(qid).to_numpy(); shT = np.where(ne_t > 0, w_t / np.maximum(ne_t, 1), np.nan)
    sv = np.load(RV / "m84_SQ4~m15_val.npz"); dv = np.load(RV / "m82_val.npz"); assert np.array_equal(sv["key"], dv["key"]); thv = float(np.quantile(sv["f_opp_m"], 0.7)); cv = (sv["f_opp_m"] >= thv) & (dv["hd"] == 0); lab_of = dict(zip(dv["key"].astype(str)[cv], sv["f_opp_lab"].astype(float)[cv]))
    aV = []
    for v in FOLDS:
        for gk in LF[v][3]["game_keys"]:
            x = [lab_of[k] for k in gk if k in lab_of]; aV.append(float(np.mean(x)) if x else np.nan)
    Q["anc"] = aV; Q["share"] = np.where(Q["ne"] > 0, Q["wins"] / np.maximum(Q["ne"], 1), np.nan)

    def corr(x, y_, cl=None):
        ok = np.isfinite(x) & np.isfinite(y_); x, y_ = x[ok], y_[ok]; r = float(np.corrcoef(x, y_)[0, 1]); sl = float(np.polyfit(x, y_, 1)[0]); bs = []
        for _ in range(B):
            if cl is None:
                i = rng.integers(0, len(x), len(x))
            else:
                u, inv = np.unique(cl[ok], return_inverse=True); i = np.repeat(np.arange(len(x)), np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv])
            bs.append(np.polyfit(x[i], y_[i], 1)[0])
        lo_, hi_ = np.percentile(bs, [2.5, 97.5]); return r, sl, float(lo_), float(hi_), int(ok.sum())
    for size in (5, 10, 15, 20):
        mt = (n_t == size) & (ne_t >= 3); r, sl, lo, hi, k = corr(shT[mt], (aT - ebT)[mt]); line = f"   size {size:2d} | test: r {r:+.3f}, slope {sl:+.2f} levels per unit share [{lo:+.2f}, {hi:+.2f}] ({k} questions)"
        if size != 15:
            mv = ((Q["n"] == size) & (Q["ne"] >= 3)).to_numpy(); r2, sl2, lo2, hi2, k2 = corr(Q["share"].to_numpy()[mv], (Q["anc"] - Q["eb"]).to_numpy()[mv], Q["pid"].to_numpy()[mv]); line += f" | validation: r {r2:+.3f}, slope {sl2:+.2f} [{lo2:+.2f}, {hi2:+.2f}] ({k2})"; rep[f"B_size_{size}"] = {"test_slope": sl, "test_ci": [lo, hi], "val_slope": sl2, "val_ci": [lo2, hi2]}
        print(line)
    mt = ne_t >= 3; r, sl, lo, hi, k = corr(shT[mt], (epT - ebT)[mt]); print(f"   test, all sizes: win share against 'pooled expected level - base expected level': r {r:+.3f}, slope {sl:+.2f} [{lo:+.2f}, {hi:+.2f}] ({k})")
    r, sl, lo, hi, k = corr(shT[mt], ebT[mt]); print(f"   test, all sizes: win share against the base expected level itself: r {r:+.3f}, slope {sl:+.2f} [{lo:+.2f}, {hi:+.2f}]"); mv = (Q["ne"] >= 3).to_numpy(); r, sl, lo, hi, k = corr(Q["share"].to_numpy()[mv], Q["eb"].to_numpy()[mv], Q["pid"].to_numpy()[mv]); print(f"   validation, all sizes: win share against the base expected level: r {r:+.3f}, slope {sl:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    # ------------------------------------------------------------ (D) validation: residual on the win share by account size
    print("\n== (D) validation: slope of 'true level - expected level' on the win share (even games), by the account's listed games; clean outcome | 'moved last' proxy")
    for nm, m in (("< 20 listed games", Q["games_total"] < 20), ("20-39", (Q["games_total"] >= 20) & (Q["games_total"] < 40)), (">= 40", Q["games_total"] >= 40), ("all", Q["games_total"] >= 0)):
        for size in (5, 10, 20):
            mm = (m & (Q["n"] == size) & (Q["ne"] >= 3)).to_numpy()
            if mm.sum() < 100:
                continue
            x = Q["share"].to_numpy()[mm]; cl = Q["pid"].to_numpy()[mm]; rp = corr(x, (Q["y"] - Q["ep"]).to_numpy()[mm], cl); rb = corr(x, (Q["y"] - Q["eb"]).to_numpy()[mm], cl); rl = corr(Q["last"].to_numpy()[mm], (Q["y"] - Q["ep"]).to_numpy()[mm], cl)
            print(f"   {nm:18s} size {size:2d} ({int(mm.sum()):5d} questions, mean true level {Q['y'].to_numpy()[mm].mean():.2f}): against the pooled posterior {rp[1]:+.3f} [{rp[2]:+.3f}, {rp[3]:+.3f}]; against the base {rb[1]:+.3f} [{rb[2]:+.3f}, {rb[3]:+.3f}] | proxy, pooled {rl[1]:+.3f} [{rl[2]:+.3f}, {rl[3]:+.3f}]")
            rep[f"D_{nm}_{size}"] = {"n": int(mm.sum()), "pooled_slope": rp[1], "pooled_ci": [rp[2], rp[3]], "base_slope": rb[1], "base_ci": [rb[2], rb[3]]}
    (RV / "m262_outcome.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nwritten results/r_variants/m262_outcome.json and m262_outcomes.npz")


if __name__ == "__main__":
    main()
