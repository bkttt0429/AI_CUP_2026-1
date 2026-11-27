"""Section 4M M103 (2026-11-13, report only): which new changes are stable? ct75s (changes shared by two recipes) reads +1.0 point
above ct44 on the Public half while the changes of a single recipe lose points. Generalisation as stability selection: a change of
the reference recipe (bundle v5: SP2e + encoder + class-conditional tables + tempering + pi0, all bands, test-side rules, no reversal
of ct44's changes) is characterised by
  support   how many of four other recipes, built on other identifier networks, make the same change
            (SP = M88 nets + matcher, S2 = M90 seed 0 alone, S3c = seed-3 20k-step nets, S3 = the same with external players);
  loo       the share of leave-one-game-out versions of the question in which the reference recipe still makes that change;
  margin    the expected-score margin of the change under the reference posterior.
CV under the test rules (as m90_testmimic.py): precision and mean gain of the changes by each characteristic, and the scores of
composition rules against "shared with SP" (ct75s's rule). Then the same characteristics on the test set for the 20 Public
questions whose group outcome is known (A shared: +1.0 point; B: -1.26; C: -1.90) - the back-test of every rule.
  python -I logs/nano4/m103_stability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

R_ENC = 0.713; REC = {"v44": ("5e", 0, 0, 0, 0, 3, 0.851), "v3": ("SP", 1, 1, 1, 1, 1, 0.875), "v5": ("SP2e", 1, 1, 1, 1, 1, 0.852), "s2": ("S2", 1, 1, 1, 1, 1, 0.734), "s3c": ("S3c", 1, 1, 1, 1, 1, 0.855), "s3": ("S3", 1, 1, 1, 1, 1, 0.85)}
OTH = ("v3", "s2", "s3c", "s3"); SM = b.SM; Z10 = np.zeros(10)


def evidence(z, v, cc, pi0, band_min, scale):
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)
    if pi0:
        _, cal, _ = m87.corrected(z, tr)
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out))


def test_evidence(V, T, name, cc, pi0, r, floor):
    v, tt = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1] if pi0 else cal
    tb = np.where(~tt["target_black"], np.searchsorted(edges[True], tt["opp_m"]), np.searchsorted(edges[False], tt["opp_m"])); tg = s2.group_of(tt["hd"], tt["target_black"]); R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10))
    for k in (0, 1, 2):
        s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = tt["opp_lab"][s]; ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio)
    return logE


def loo_share(P_row, G, T, new):
    """Share of leave-one-game-out versions (evidence of one game removed) whose region decision is still `new`."""
    if len(G) < 2:
        return 0.0
    L = (G.sum(0)[None, :] - G) * T; d = mb.decisions(np.repeat(P_row[None, :], len(G), 0), L, 0.5, 3.0, 0.6)[2]
    return float((d == new).mean())


def margin_of(P, L, new, old):
    Pa = P * np.exp(L - L.max(1, keepdims=True)); Pa /= Pa.sum(1, keepdims=True); E = Pa @ mb.S.T; r = np.arange(len(P))
    return E[r, new] - E[r, old]


def table(title, key, gain, bins, names):
    print(f"  {title}:")
    for bn, nm in zip(bins, names):
        s = bn(key); n = int(s.sum())
        if n:
            print(f"      {nm:26s} {n:6d} changes ({n / len(key):.3f}); better {float((gain[s] > 0).mean()):.3f}, worse {float((gain[s] < 0).mean()):.3f}; mean gain {float(gain[s].mean()):+.3f}; total {float(gain[s].sum()):+.1f}")


def main():
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); assert all(x[0] in V for x in REC.values()), used; Fd = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; G5 = None
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = evidence(V[idn], v, cc, pi0, bmin, scale); ee = evidence(V["enc"], v, 0, pi0, bmin, R_ENC) if enc else {}
            G = [np.array([ev.get(k, Z10) + b.U_ENC * ee.get(k, Z10) for k in g_]) if len(g_) else np.zeros((0, 10)) for g_ in gk]; L = np.stack([g_.sum(0) if len(g_) else Z10 for g_ in G]); Ls[name] = L * Tn if temp else L
            if name == "v5":
                G5 = G
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls, G5, Tn)
    print("CV under the test rules; reference = bundle v5's recipe; new changes = questions the ct44 recipe leaves at the base decision", flush=True); rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        rows = {k: [] for k in ("gain", "sup", "loo", "mar", "v3")}; S = {k: [] for k in ("v44", "all v5", "shared with SP (ct75s rule)", "support >= 1", "support >= 2", "support >= 3", "support = 4", "loo = 1", "loo = 1 and support >= 1", "loo = 1 or support >= 2", "shared with SP or support >= 3", "majority of the five")}; ys, ns, ps = [], [], []
        for v in FOLDS:
            y, n, pid, Ls, G5, Tn = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = {}
            for name in REC:
                d0, _, dec[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
            d44 = dec["v44"]; keep = d44 != d0; nr = {k: np.where(keep, d44, dec[k]) for k in REC if k != "v44"}; n5 = nr["v5"]; sel = n5 != d44; sup = sum((nr[k] == n5).astype(int) for k in OTH); mar = margin_of(P, Ls["v5"], n5, d44)
            loo = np.zeros(len(y)); idx = np.flatnonzero(sel)
            for i in idx:
                loo[i] = loo_share(P[i], G5[i], float(Tn[i, 0]), n5[i])
            g = SM[n5, y] - SM[d44, y]; rows["gain"].append(g[sel]); rows["sup"].append(sup[sel]); rows["loo"].append(loo[sel]); rows["mar"].append(mar[sel]); rows["v3"].append((nr["v3"] == n5)[sel])
            votes = np.stack([nr[k] for k in ("v3", "v5", "s2", "s3c", "s3")]); maj = d44.copy()
            for c in range(10):
                cnt = (votes == c).sum(0); maj = np.where((cnt >= 3) & (c != d44), c, maj)
            rule = {"v44": d44, "all v5": n5, "shared with SP (ct75s rule)": np.where(nr["v3"] == n5, n5, d44), "support >= 1": np.where(sup >= 1, n5, d44), "support >= 2": np.where(sup >= 2, n5, d44), "support >= 3": np.where(sup >= 3, n5, d44), "support = 4": np.where(sup >= 4, n5, d44),
                    "loo = 1": np.where(loo >= 1, n5, d44), "loo = 1 and support >= 1": np.where((loo >= 1) & (sup >= 1), n5, d44), "loo = 1 or support >= 2": np.where((loo >= 1) | (sup >= 2), n5, d44), "shared with SP or support >= 3": np.where((nr["v3"] == n5) | (sup >= 3), n5, d44), "majority of the five": maj}
            for k, d in rule.items():
                S[k].append(SM[d, y])
            ys.append(y); ns.append(n); ps.append(pid)
        R = {k: np.concatenate(x) for k, x in rows.items()}; y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps); gain = R["gain"]; print(f"seeds {seeds}: {len(gain)} new changes of the reference recipe ({len(gain) / len(y):.4f} of the questions); mean gain {gain.mean():+.3f}")
        table("by support (other recipes making the same change)", R["sup"], gain, [lambda k, c=c: k == c for c in range(5)], [f"support {c}" for c in range(5)])
        table("by leave-one-game-out share", R["loo"], gain, [lambda k: k >= 1, lambda k: (k >= 0.75) & (k < 1), lambda k: (k >= 0.5) & (k < 0.75), lambda k: k < 0.5], ["stable in all", "0.75 to 1", "0.5 to 0.75", "below 0.5"])
        qs_ = np.quantile(R["mar"], [1 / 3, 2 / 3]); table("by expected-score margin (thirds)", R["mar"], gain, [lambda k: k <= qs_[0], lambda k: (k > qs_[0]) & (k <= qs_[1]), lambda k: k > qs_[1]], [f"low (<= {qs_[0]:.3f})", "middle", f"high (> {qs_[1]:.3f})"])
        table("shared with SP or not", R["v3"], gain, [lambda k: k, lambda k: ~k], ["shared with SP", "not shared"])
        print("  composition rules, class-balanced score against the ct44 recipe and against the ct75s rule:"); res = {}
        for k in S:
            if k == "v44":
                continue
            A_ = np.concatenate(S[k]); d1 = paired_diff(A_, np.concatenate(S["v44"]), y, pid, n, sizes=SIZES, n=3000); d2 = paired_diff(A_, np.concatenate(S["shared with SP (ct75s rule)"]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(a_, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(c_, Fd[v][0], Fd[v][1], sizes=SIZES) for a_, c_, v in zip(S[k], S["shared with SP (ct75s rule)"], FOLDS)]
            print(f"      {k:32s} vs ct44 recipe {d1['diff']:+.4f} | vs ct75s rule {d2['diff']:+.4f} [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" | changed vs ct44 recipe {float((A_ != np.concatenate(S['v44'])).mean()):.4f}", flush=True); res[k] = {"vs_ct44": d1["diff"], "vs_ct75s_rule": d2["diff"], "ci": d2["ci95"]}
        rep[f"seeds_{seeds[0]}"] = res
    # test side: the same characteristics for the reference recipe's new changes
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c73, c75n, c75s = (sub(k) for k in ("ct9", "ct28", "ct44", "ct73", "ct75n", "ct75s")); locked = c28 != c9; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; qpos = {q: i for i, q in enumerate(qids)}
    Le = test_evidence(V, T, "enc", 0, 1, R_ENC, 1); Leq = pd.DataFrame(Le).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); tdec = {}; G5t = {}; L5 = None; same_order = bool(np.array_equal(T["SP2e"]["qid"], T["enc"]["qid"]))
    for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
        if name == "v44":
            continue
        Eg = test_evidence(V, T, idn, cc, pi0, scale, bmin); Lq = (pd.DataFrame(Eg).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq
        d = np.where(locked, c28, lev[mb.decisions(P9, Lq, 0.5, 3.0, 0.6)[2]]); tdec[name] = np.where(c44 != c28, c44, d)
        if name == "v3":
            print(f"check: the SP recipe rebuilt on the test differs from ct73 in {int((tdec[name] != c73).sum())} questions")
        if name == "v5":
            L5 = Lq; assert (tdec[name] == c75n).all(), int((tdec[name] != c75n).sum()); Eg5 = Eg + (b.U_ENC * Le if same_order else 0.0); qv = T[idn]["qid"]
            for q_ in np.unique(qv):
                G5t[q_] = Eg5[qv == q_]
            print(f"check: the reference recipe rebuilt on the test equals ct75n; encoder evidence aligned game by game: {same_order}")
    n5 = tdec["v5"]; sel = np.flatnonzero(n5 != c44); sup = sum((tdec[k] == n5).astype(int) for k in OTH); mar = margin_of(P9, L5, rk(n5), rk(c44)); pub = np.char.startswith(qids.astype(str), "pub_"); out = []
    print(f"\ntest set: the reference recipe's {len(sel)} new changes (Public {int(pub[sel].sum())}, Private {int((~pub[sel]).sum())}); group of the Public ones: A = also in ct73 (8 questions, +1.0 point together), C = only in ct75n (8 questions, -1.9 points together)")
    for i in sel:
        loo = loo_share(P9[i], G5t.get(qids[i], np.zeros((0, 10))), float(Tq[i, 0]), int(rk(n5[i]))); grp = ("A" if c73[i] == n5[i] else "C") if pub[i] else ("a" if c73[i] == n5[i] else "c"); out.append((qids[i], grp, c44[i], n5[i], int(sup[i]), loo, float(mar[i]), int(nq_t[i])))
    df = pd.DataFrame(out, columns=["question", "group", "ct44", "new", "support", "loo", "margin", "games"])
    for g_ in ("A", "C", "a", "c"):
        d_ = df[df.group == g_]; print(f"  group {g_} ({len(d_)}): support counts {dict(d_.support.value_counts().sort_index())}; loo = 1: {int((d_.loo >= 1).sum())}; mean margin {d_.margin.mean():.3f}")
    print(df[df.group.isin(["A", "C"])].sort_values(["group", "support"], ascending=[True, False]).to_string(index=False))
    for name, keep in (("support >= 1", df.support >= 1), ("support >= 2", df.support >= 2), ("support >= 3", df.support >= 3), ("support = 4", df.support >= 4), ("loo = 1", df.loo >= 1), ("loo = 1 or support >= 2", (df.loo >= 1) | (df.support >= 2))):
        print(f"  rule {name:26s} keeps of the Public A group {int((keep & (df.group == 'A')).sum())}/8, of C {int((keep & (df.group == 'C')).sum())}/8; Private: of the shared {int((keep & (df.group == 'a')).sum())}/{int((df.group == 'a').sum())}, of the others {int((keep & (df.group == 'c')).sum())}/{int((df.group == 'c').sum())}")
    df.to_csv(ROOT / "results/r_variants/m103_test_changes.csv", index=False); (ROOT / "results/r_variants/m103_stability.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
