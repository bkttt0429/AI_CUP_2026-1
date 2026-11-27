#!/usr/bin/env python
"""M203 (2026-11-16), REPORT ONLY: is the test set (Public: exact composition from M202; Private: no labels) distributed like
the training data? Design doc M203 (registered before the run).

(1) level composition: several ways of counting the training data against the exact Public counts (C + D merged, 9 cells);
(2) given the level, does the model behave on the test set as in cross-validation: the cross-validated confusion matrix
    P(answer | truth, size) times the Public composition = the answer distribution and the score the Public half should
    show if only the composition differed; against the actual answers / the obtained score;
(3) the Private composition by inverting the same confusion matrix (checked on Public against the truth first);
(4) label-free quantities of the games: validation (all, by the player's number of games), Public, Private.

  python -I logs/nano4/m203_train_vs_test.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; RV = ROOT / "results/r_variants"; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])  # S[answer, truth]
rng = np.random.default_rng(203)


def read_len(path):
    with np.load(path, allow_pickle=False) as z:
        return z["key"], z["LEN"]


def merge9(v):  # 10 levels -> C + D merged first, then B .. 6D
    v = np.asarray(v, float); return np.concatenate([[v[0] + v[1]], v[2:]])


def gof(counts, q, sims=20000):
    n = int(counts.sum()); e = n * q; chi = float(((counts - e) ** 2 / np.maximum(e, 1e-9)).sum()); d = rng.multinomial(n, q, sims); cs = ((d - e) ** 2 / np.maximum(e, 1e-9)).sum(1)
    return chi, float((cs >= chi - 1e-12).mean()), float((counts * np.log(np.maximum(q, 1e-12))).sum())


def nnls_simplex(A, b, iters=20000):
    """min |A x - b| over the simplex (projected gradient; small problem)."""
    x = np.full(A.shape[1], 1.0 / A.shape[1]); L = np.linalg.norm(A, 2) ** 2
    for _ in range(iters):
        x = x - (A.T @ (A @ x - b)) / L; u = np.sort(x)[::-1]; css = np.cumsum(u); k = np.nonzero(u * np.arange(1, len(x) + 1) > (css - 1))[0][-1]; x = np.maximum(x - (css[k] - 1) / (k + 1), 0)
    return x


def main():
    pc = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; pub9 = np.array([400 - sum(pc.values())] + [pc[l] for l in LEV[2:]], float); assert pub9.sum() == 400
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}); pf["lv"] = pf["rank"].map({l: i for i, l in enumerate(LEV)}); ng = pf.n_games.to_numpy(); lv = pf.lv.to_numpy()
    comp = lambda m, w=None: np.bincount(lv[m], weights=None if w is None else w[m], minlength=10) / (m.sum() if w is None else w[m].sum())  # noqa: E731
    cands = {"by games (each level 10%)": np.full(10, 0.1), "by players (all 8,007)": comp(np.ones(len(pf), bool))}
    for k in (5, 10, 15, 20, 30, 40):
        cands[f"players with >= {k} games"] = comp(ng >= k)
    cands["a size-n question from a player with >= n games (5/10/15/20 equally)"] = np.mean([comp(ng >= k) for k in (5, 10, 15, 20)], axis=0)
    cands["by games of players with >= 20 games"] = comp(ng >= 20, ng.astype(float))
    print("== (1) level composition (percent): training data counted in several ways against the exact Public counts\n" + " " * 74 + "  C+D     B     A    1D    2D    3D    4D    5D    6D   | chi2    p     logL")
    print(f"  {'Public, measured (M202)':72s}" + " ".join(f"{100 * x / 400:5.1f}" for x in pub9))
    for nm, q in cands.items():
        q9 = merge9(q); chi, p_, ll = gof(pub9, q9); print(f"  {nm:72s}" + " ".join(f"{100 * x:5.1f}" for x in q9) + f"   | {chi:5.1f}  {p_:.4f}  {ll:.1f}")
    print("  players per level (all): " + ", ".join(f"{l} {int((lv == i).sum())}" for i, l in enumerate(LEV)) + "; share of D within C + D by the candidates: " + ", ".join(f"{q[0] / (q[0] + q[1]):.2f}" for q in cands.values()))
    # ------------------------------------------------------------------ (2) conditional distribution: confusion matrices of the cross-validation
    z = np.load(RV / "m183_dump.npz", allow_pickle=True); y, n = z["y"].astype(int), z["n"].astype(int); dv = {"pooled posterior, no guards": z["anat_plain"].astype(int), "base": z["anat_base"].astype(int), "rule with guards (SQ4)": z["SQ4"].astype(int)}

    def confusion(d):
        C = {}
        for s_ in (5, 10, 20):
            m = n == s_; M = np.zeros((10, 10)); np.add.at(M, (y[m], d[m]), 1); C[s_] = M / M.sum(1, keepdims=True)
        C[15] = (C[10] + C[20]) / 2
        return C
    tp = np.load(RV / "m163_test_posterior_ct112.npz", allow_pickle=True); qid = tp["qid"].astype(str); P = {"learned": tp["learned"].astype(float), "base": tp["base"].astype(float)}; pub = np.char.startswith(qid, "pub")
    T = np.load(RV / "m84_SQ4~m15_test.npz", allow_pickle=True); pos = {q: i for i, q in enumerate(qid)}; gi = np.array([pos[q] for q in T["qid"].astype(str)]); nq = np.bincount(gi, minlength=len(qid))
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    dt = {"pooled posterior, no guards": (P["learned"] @ S.T).argmax(1), "base": sub("ct9"), "rule with guards (SQ4)": None}; c112, c9, c28 = sub("ct112"), sub("ct9"), sub("ct28"); locked = c28 != c9
    dt["rule with guards (SQ4)"] = np.where(locked, dt["pooled posterior, no guards"], c112)  # ct112 without the 38 locked pushes (those questions: the posterior's own answer)
    ledger = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}
    print("\n== (2) given the level: cross-validated confusion x the Public composition, against what the Public half shows")
    best_split = {}
    for nm, d in dv.items():
        C = confusion(d); obs = np.bincount(dt[nm][pub], minlength=10).astype(float); sizes = (5, 10, 15, 20); cnt_s = {s_: int((pub & (nq == s_)).sum()) for s_ in sizes}

        def expect(pi):
            return sum(cnt_s[s_] * (pi @ C[s_]) for s_ in sizes)
        fits = []
        for D_ in range(0, int(pub9[0]) + 1):  # the split of C + D is not measured: the likelihood-best split
            pi = np.concatenate([[D_, pub9[0] - D_], pub9[1:]]) / 400.0; e = expect(pi); fits.append((float((obs * np.log(np.maximum(e / 400, 1e-12))).sum()), D_))
        ll, D_ = max(fits); best_split[nm] = D_; pi = np.concatenate([[D_, pub9[0] - D_], pub9[1:]]) / 400.0; e = expect(pi); chi = float(((obs - e) ** 2 / np.maximum(e, 1e-9)).sum()); sims = []
        for _ in range(4000):
            o = np.zeros(10)
            for s_ in sizes:
                tr = rng.multinomial(cnt_s[s_], pi)
                for c in range(10):
                    if tr[c]:
                        o += rng.multinomial(tr[c], C[s_][c])
            sims.append(((o - e) ** 2 / np.maximum(e, 1e-9)).sum())
        sc = sum(cnt_s[s_] * float((pi[:, None] * C[s_] * S.T).sum()) for s_ in sizes) / 400.0
        e_bal = sum(cnt_s[s_] * (np.full(10, 0.1) @ C[s_]) for s_ in sizes); sc_bal = sum(cnt_s[s_] * float((0.1 * C[s_] * S.T).sum()) for s_ in sizes) / 400.0
        print(f"  {nm}: likelihood-best split D {D_} / C {int(pub9[0]) - D_}; chi-square {chi:.1f}, simulated p {float((np.array(sims) >= chi).mean()):.3f}; score expected under the Public composition {sc:.4f} (under equal levels {sc_bal:.4f})" + (f"; obtained on Public: ct9 {ledger['submission_rank_ct9.csv']:.4f}" if nm == "base" else (f"; obtained: ct112 {ledger['submission_rank_ct112.csv']:.4f} (with the 17 Public pushes, known +8.22 points at their introduction)" if nm.startswith("rule") else "")))
        print("      answers        " + " ".join(f"{l:>5}" for l in LEV)); print("      expected       " + " ".join(f"{x:5.1f}" for x in e)); print("      Public actual  " + " ".join(f"{int(x):5d}" for x in obs)); print("      if levels equal" + " ".join(f"{x:5.1f}" for x in e_bal))
        prv = np.bincount(dt[nm][~pub], minlength=10).astype(float); print("      Private actual " + " ".join(f"{int(x):5d}" for x in prv))
        # ---------------------------------------------------------- (3) composition by inverting the confusion matrix
        if nm != "base":
            Cm = sum(0.25 * C[s_] for s_ in sizes).T  # answers = Cm @ pi
            est = {h: nnls_simplex(Cm, np.bincount(dt[nm][m], minlength=10) / m.sum()) for h, m in (("Public", pub), ("Private", ~pub))}; boot = {h: [] for h in est}; idx = {"Public": np.flatnonzero(pub), "Private": np.flatnonzero(~pub)}
            for _ in range(300):
                for h in est:
                    r = rng.choice(idx[h], len(idx[h])); boot[h].append(nnls_simplex(Cm, np.bincount(dt[nm][r], minlength=10) / len(r), iters=3000))
            bp, bq = np.array(boot["Public"]), np.array(boot["Private"]); truth9 = pub9 / 400; bp9 = np.array([merge9(b_) for b_ in bp]); dd9 = np.array([merge9(b_) for b_ in bq - bp]); lo9, hi9 = np.quantile(dd9, [0.05, 0.95], axis=0)
            print("      (3) composition inverted from these answers (C + D merged; percent)")
            print("          Public truth      " + " ".join(f"{100 * x:5.1f}" for x in truth9))
            print("          Public estimate   " + " ".join(f"{100 * x:5.1f}" for x in merge9(est["Public"])) + "   (bootstrap sd " + " ".join(f"{100 * x:.1f}" for x in bp9.std(0)) + ")")
            print("          Private estimate  " + " ".join(f"{100 * x:5.1f}" for x in merge9(est["Private"])))
            print("          Private - Public  " + " ".join(f"{100 * x:+5.1f}" for x in merge9(est["Private"]) - merge9(est["Public"])))
            print("          its 90% interval  " + " ".join(f"[{100 * a:+.0f},{100 * b_:+.0f}]" for a, b_ in zip(lo9, hi9)))
    # ------------------------------------------------------------------ (4) label-free quantities of the games
    print("\n== (4) label-free quantities: validation games (by the player's number of games), Public, Private")
    V = np.load(RV / "m84_SQ4~m15_val.npz", allow_pickle=True); files = sorted((ROOT / "results/m197_choice64").glob("part_s*of*_p*.npz"))
    with mp.get_context("fork").Pool(12) as pool:
        res = pool.map(read_len, files, chunksize=64)
    LEN = dict(zip(np.concatenate([r[0] for r in res]).tolist(), np.concatenate([r[1] for r in res]).tolist())); del res
    import m81_tactic_id as m81
    tq = [q for q, _ in m81.load_tests()]; tl = np.array([LEN[f"test:{i}"] for i in range(len(tq))]); tpub = np.array([q.startswith("pub") for q in tq]); assert len(tq) == len(T["qid"]) and (np.array(tq) == T["qid"].astype(str)).all(), "test games are not in the order of the identifier file"
    vl = np.array([LEN.get(k, -1) for k in V["key"].astype(str)]); vng = pf.set_index("player_id").n_games.reindex(V["pid"].astype(str)).to_numpy(); vq90 = np.quantile(V["f_imp_m"], 0.90); tq90 = np.quantile(T["f_tgt_m"], 0.90)

    def row(nm, hd, tb, om, q90, ln):
        print(f"  {nm:34s} games {len(hd):6d}; handicap {float((hd != 0).mean()):.3f}; subject black {float(np.mean(tb)):.3f}; opponent recognised (above the own-side q90) {float((om > q90).mean()):.3f}; plies mean {ln[ln >= 0].mean():.0f}, cut at 256 {float((ln >= 256).mean()):.3f}, under 100 plies {float((ln[ln >= 0] < 100).mean()):.3f}")
    row("validation, all", V["hd"], V["target_black"], V["f_opp_m"], vq90, vl)
    for lo, hi in ((20, 25), (26, 40), (41, 110), (111, 260), (261, 10 ** 9)):
        m = (vng >= lo) & (vng <= hi); row(f"validation, players with {lo}-{hi if hi < 10 ** 8 else ''} games", V["hd"][m], V["target_black"][m], V["f_opp_m"][m], vq90, vl[m])
    row("Public", T["hd"][tpub], T["target_black"][tpub], T["f_opp_m"][tpub], tq90, tl[tpub]); row("Private", T["hd"][~tpub], T["target_black"][~tpub], T["f_opp_m"][~tpub], tq90, tl[~tpub])
    hdv, hdt = V["hd"], T["hd"]; vals = sorted(set(np.unique(hdv)) | set(np.unique(hdt)))
    print("  handicap value distribution (share of games): " + " | ".join(f"hd {int(h)}: val {float((hdv == h).mean()):.3f} pub {float((hdt[tpub] == h).mean()):.3f} priv {float((hdt[~tpub] == h).mean()):.3f}" for h in vals[:9]))
    post = z["anat_post"].astype(float); post = post / post.sum(1, keepdims=True); pl = P["learned"] / P["learned"].sum(1, keepdims=True)
    print("  largest probability of the pooled posterior, by size: " + " | ".join(f"{s_}: validation {post[n == s_].max(1).mean():.3f}, Public {pl[pub & (nq == s_)].max(1).mean():.3f}, Private {pl[~pub & (nq == s_)].max(1).mean():.3f}" for s_ in (5, 10, 20)) + f" | 15: Public {pl[pub & (nq == 15)].max(1).mean():.3f}, Private {pl[~pub & (nq == 15)].max(1).mean():.3f}")
    print("  validation questions are class-balanced only through weights; players per level among the validation players: " + ", ".join(f"{LEV[i]} {int(c)}" for i, c in enumerate(np.bincount(pd.Series(y[n == 20], index=z['pid'][n == 20].astype(str)).groupby(level=0).first().to_numpy(), minlength=10))))


if __name__ == "__main__":
    main()
