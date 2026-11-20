"""Section 4M M113 (2026-11-14, report only, CPU): four candidate signals that follow from M111 (70% of the opponents are labelled
players, very concentrated) and from the list of sources never measured. Validation folds, cross-fitted; seeds 0-2 and 6-8.
 1  HOW OFTEN the opponents of a question are recognised, as evidence of its own: P(margin band | level) per game (the current
    evidence uses the recognised opponent's label and treats "not recognised" as no information). Table by level; the band term
    added to the current evidence at several temperatures; the same bands on the test set.
 2  the TARGET's own look-alikes: the question's mean target-side identity vector against the prototypes of the training players
    (own fold left out), level distribution of the nearest players; alone, pooled into the base, and on top of the current evidence.
 3  order of the test questions: predicted level against the question number, and neighbouring questions sharing a level, against
    permutations (a leak of the file layout would show here).
 4  test players as each other's opponents: every test game's opponent-side vector against the 800 questions' own prototypes and
    against the 8,007 training prototypes; how often the best match is another test question, and whether the linked questions
    have neighbouring predicted levels more often than chance.
  python -I logs/nano4/m113_signal_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m111_label_readings as m1  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM; RV = ROOT / "results/r_variants"; CACHE = ROOT / "results/m84_cache"; rng = np.random.default_rng(1130); TAU = (0.15, 0.3, 0.5, 1.0); BETA = (20.0, 40.0, 80.0); WK = (0.1, 0.2, 0.3, 0.5)


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; t = T["SP2e"]; row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool); edges, cal, om = m87.corrected(z, ones); _, _, band, g = s2.fit(z, ones)
    q = np.load(CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"]), "query order differs from the validation rows"; var_ = np.load(CACHE / "var_S2e.npz"); var = {"qvec": var_["qvec"], "protos": var_["protos"]}; players = q["players"].astype(str); plab = q["plab"]; pidx = {p: i for i, p in enumerate(players)}
    pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    # ---------------------------------------------------------------- 1: recognition density
    print("1. share of a player's games whose opponent is recognised, by the player's level (validation, even games): bands >= q90 | bands >= q98")
    ev = g == 0; print("     " + "  ".join(f"{LEVELS[c]} {float((band[ev & (z['y'] == c)] >= 3).mean()):.3f}|{float((band[ev & (z['y'] == c)] >= 5).mean()):.3f}" for c in range(10)))
    for k, nm in ((1, "HD1, target Black"), (2, "HD1, target White")):
        s = g == k; print(f"     {nm}: " + "  ".join(f"{LEVELS[c]} {float((band[s & (z['y'] == c)] >= 3).mean()):.3f}" for c in range(10)))
    tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"])
    print("   test set against validation, share of the games by band (even games): " + "  ".join(f"{m1.BAND[x]} {float((tb[tg == 0] == x).mean()):.3f}/{float((band[ev] == x).mean()):.3f}" for x in range(9)))
    res1 = {}; res2 = {}; ys, ns, ps = [], [], []; F = {}
    for v in FOLDS:
        tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid).astype(str); Qn = len(y)
        qi_all = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(qd["game_keys"])]); gi_all = np.concatenate([[row[k] for k in g_ if k in row] for g_ in qd["game_keys"]]).astype(int); ok = g[gi_all] >= 0; qi, gi = qi_all[ok], gi_all[ok]; kk, bb, lab = g[gi], band[gi], z["opp_lab"][gi]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]
        L_std = np.zeros((Qn, 10)); np.add.at(L_std, qi, np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab])); L_std *= Tn
        TB = np.zeros((3, 10, 9))  # P(band | level, group) from the other folds
        for k in (0, 1, 2):
            for c in range(10):
                s = tr & (gg == k) & (z["y"] == c); TB[k, c] = (np.bincount(bnd[s], minlength=9) + 2.0) / (s.sum() + 18.0)
        L_band = np.zeros((Qn, 10)); np.add.at(L_band, qi, np.log(TB[kk, :, bb]))
        # 2: the target's own look-alikes in the identity space
        j = v - 1; cand = np.flatnonzero((pfold != v)); tq = np.zeros((3, Qn, 128)); cnt_ = np.bincount(qi_all, minlength=Qn)[:, None]
        for s_ in range(3):
            np.add.at(tq[s_], qi_all, var["qvec"][s_, Q + gi_all].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); S_ = np.mean([tq[s_] @ var["protos"][s_, j][cand].astype(np.float64).T for s_ in range(3)], 0); oh = np.eye(10)[plab[cand]]; PK = {}
        for be in BETA:
            w_ = np.exp(be * (S_ - S_.max(1, keepdims=True))); pk = w_ @ oh; PK[be] = (pk + 1e-4) / (pk + 1e-4).sum(1, keepdims=True)
        F[v] = dict(y=y, n=n, pid=pid, L_std=L_std, L_band=L_band, PK=PK); ys.append(y); ns.append(n); ps.append(pid)
    yy, nn, pp = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps)
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {}
        for v in FOLDS:
            f = F[v]; y, n = f["y"], f["n"]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = lambda P_, L_: mb.decisions(P_, L_, 0.5, 3.0, 0.6)[1]; Z = np.zeros_like(P)  # noqa: E731
            S.setdefault("base", []).append(SM[dec(P, Z), y]); S.setdefault("current evidence", []).append(SM[dec(P, f["L_std"]), y])
            for tau in TAU:
                S.setdefault(("band only", tau), []).append(SM[dec(P, tau * f["L_band"]), y]); S.setdefault(("evidence + band", tau), []).append(SM[dec(P, f["L_std"] + tau * f["L_band"]), y])
            for be in BETA:
                S.setdefault(("look-alikes alone", be), []).append(SM[dec(f["PK"][be], Z), y])
                for w in WK:
                    lk = w * np.log(f["PK"][be]); S.setdefault(("base + look-alikes", be, w), []).append(SM[dec(P, lk), y]); S.setdefault(("evidence + look-alikes", be, w), []).append(SM[dec(P, f["L_std"] + lk), y])
        sc = lambda k_: float(np.mean([balanced_mean_over_n(s_, y_, n_, sizes=SIZES) for s_, y_, n_ in zip(S[k_], ys, ns)]))  # noqa: E731

        def cmp(new, old):
            d = paired_diff(np.concatenate(S[new]), np.concatenate(S[old]), yy, pp, nn, sizes=SIZES, n=2000); pf = [float(balanced_mean_over_n(a_, y_, n_, sizes=SIZES) - balanced_mean_over_n(c_, y_, n_, sizes=SIZES)) for a_, c_, y_, n_ in zip(S[new], S[old], ys, ns)]
            return f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf)
        print(f"\n=== seeds {seeds}: base {sc('base'):.4f}; current evidence {sc('current evidence'):.4f} ===")
        print("1. recognition density as evidence (temperature): alone over the base | added to the current evidence")
        for tau in TAU:
            print(f"     x{tau:4.2f}: {sc(('band only', tau)) - sc('base'):+.4f} | {cmp(('evidence + band', tau), 'current evidence')}", flush=True)
        print("2. the target's look-alikes: alone | best pooling into the base | best pooling on top of the current evidence")
        for be in BETA:
            wb = max(WK, key=lambda w: sc(("base + look-alikes", be, w))); we = max(WK, key=lambda w: sc(("evidence + look-alikes", be, w)))
            print(f"     sharpness {be:4.0f}: alone {sc(('look-alikes alone', be)):.4f} | weight {wb}: {cmp(('base + look-alikes', be, wb), 'base')} | weight {we}: {cmp(('evidence + look-alikes', be, we), 'current evidence')}", flush=True)
        res1[seeds[0]] = {str(k_): sc(k_) for k_ in S}
    # ---------------------------------------------------------------- 3: order of the test questions
    sub = pd.read_csv(ROOT / "submission_rank_ct75u.csv", dtype=str); rk = sub.pred_rank.map(b.RANK).to_numpy(); num = sub.question_id.str.extract(r"(\d+)$")[0].astype(int).to_numpy(); pub = sub.question_id.str.startswith("pub_").to_numpy(); ng = pd.Series(t["qid"]).value_counts().reindex(sub.question_id).to_numpy()
    print("\n3. order of the test questions (ct75u's levels):")
    for nm, s in (("Public", pub), ("Private", ~pub)):
        o = np.argsort(num[s]); r_, n_ = rk[s][o], ng[s][o]; same = float((r_[1:] == r_[:-1]).mean()); near = float((np.abs(r_[1:] - r_[:-1]) <= 1).mean()); perm = np.array([[(x[1:] == x[:-1]).mean(), (np.abs(x[1:] - x[:-1]) <= 1).mean()] for x in (rng.permutation(r_) for _ in range(2000))]); rho = float(pd.Series(r_).corr(pd.Series(np.arange(len(r_))), method="spearman")); rn = float(pd.Series(n_).corr(pd.Series(np.arange(len(n_))), method="spearman")); rl = float(pd.Series(r_).corr(pd.Series(n_), method="spearman"))
        print(f"     {nm}: level against the question number, Spearman {rho:+.3f}; number of games against the question number {rn:+.3f}; level against the number of games {rl:+.3f}; neighbours with the same level {same:.3f} (permutations {perm[:, 0].mean():.3f} +- {perm[:, 0].std():.3f}), within one level {near:.3f} ({perm[:, 1].mean():.3f} +- {perm[:, 1].std():.3f})")
    # ---------------------------------------------------------------- 4: test players as each other's opponents
    ti = np.flatnonzero(q["qtest"] >= 0); side = q["side"][ti]; tgame = q["qtest"][ti]; qid_of_game = t["qid"].astype(str); assert len(qid_of_game) == 10000
    opp = ti[side == 1][np.argsort(tgame[side == 1])]; tgt = ti[side == 0][np.argsort(tgame[side == 0])]; qids = sub.question_id.to_numpy(); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in qid_of_game]); St = np.zeros((10000, 800)); Sr = np.zeros(10000); Sa = np.zeros(10000, int)
    for s_ in range(3):
        tv = var["qvec"][s_, tgt].astype(np.float64); pr = np.zeros((800, 128)); np.add.at(pr, gq, tv); pr /= np.maximum(np.linalg.norm(pr, axis=1, keepdims=True), 1e-9); ov = var["qvec"][s_, opp].astype(np.float64); St += ov @ pr.T / 3; x = ov @ var["protos"][s_, 4].astype(np.float64).T; Sr += x.max(1) / 3; Sa = x.argmax(1)
    own_first = float((St.argmax(1) == gq).mean()); print(f"\n(sanity: a test game's opponent-side vector is closest to its OWN question's prototype in {own_first:.3f} of the games; the target-side alignment check: {float(((var['qvec'][0, tgt].astype(np.float64) @ pr.T).argmax(1) == gq).mean()):.3f} of the games are closest to their own question)")
    St[np.arange(10000), gq] = -9.0; bt = St.argmax(1); sm = St.max(1); win = sm > Sr; mar = sm - Sr; d_lv = np.abs(rk[gq] - rk[bt]); base_near = float(np.mean([(np.abs(rk[gq] - rk[rng.permutation(800)[gq]]) <= 1).mean() for _ in range(200)]))
    print(f"\n4. test games whose opponent looks more like another test question's player than like any training player: {int(win.sum())} of 10,000 ({float(win.mean()):.3f}); expected if the 800 test prototypes were just 800 more candidates among 8,807: about {800 / 8807:.3f}")
    for top in (50, 200, 500):
        o = np.argsort(-mar)[:top]; print(f"     the {top} largest margins: margin >= {mar[o].min():+.3f}; linked questions within one predicted level {float((d_lv[o] <= 1).mean()):.3f}, same level {float((d_lv[o] == 0).mean()):.3f} (random pairs within one: {base_near:.3f})")
    (RV / "m113_signal_sweep.json").write_text(json.dumps({"scores": res1}, indent=1, default=float))


if __name__ == "__main__":
    main()
