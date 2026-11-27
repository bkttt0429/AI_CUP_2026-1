"""Section 4M M86 (2026-11-11): bundle test of small gains that act at different stages of the anchor pipeline. Switches on top of
ct44's pipeline (5e identifier, pooled evidence, current mixture weights, expected-score decision):
  id    identifier: one of --ids (5e = ct44's; P3 = all games + phase rows + 3 seeds; S / SP = M88's sequence model, alone / with P3)
  enc   the whole-game encoder's (M77) evidence added with weight 0.5 (its own cross-fitted mixture calibration)
  cc    class-conditional evidence tables (M83, kappa 50) for the main identifier
  temp  the question's anchor log-evidence times T_n = 1.0 / 1.0 / 0.6 for n = 5 / 10 / 20 (15 -> 0.8 on the test)
  pi0   (with --pi0) M87's corrected mixture weights, for both evidence sources
Selection and estimation use different folds: the best combination on folds {1, 2} (class-balanced score, seeds 0-2) is scored on folds
{3, 4} against "all off", and the other way round; the two halves together are the cross-fitted estimate. Rule: CI lower bound > 0,
difference >= 0.0025, >= 3 of 4 folds positive, on seeds 0-2 and 6-8 (same picks). If it passes, the combination chosen on all four
folds is written as --out (ct28 locked + boundary region, each source with its own lower density ratio, evidence from the margin bands
>= --floor: 3 = q90 as ct44, 1 = every band with a positive weight, which is what the CV comparison scores); the builder must reproduce
ct44 exactly with every switch off and floor 3.
  v1 (ct64):  python -I scripts/m_series/m86_bundle.py --ids 5e,P3 --floor 3 --out ct64 --json m86_bundle.json
  v2 (ct70):  ... --ids 5e,P3 --pi0 --floor 3 --out ct70 --json m86b_bundle.json
  v2, all bands (ct71):  ... --ids 5e,P3 --pi0 --floor 1 --out ct71 --json m86c_bundle.json
  v3 (ct72):  ... --ids 5e,P3,S,SP --pi0 --floor 1 --out ct72 --json m86d_bundle.json
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

RV = ROOT / "results/r_variants"; RANK = {lv: i for i, lv in enumerate(LEVELS)}
TN = {5: 1.0, 10: 1.0, 15: 0.8, 20: 0.6}; U_ENC = 0.5
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)


def sources(ids):
    d82 = dict(np.load(RV / "m82_val.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; t82 = dict(np.load(RV / "m82_test.npz")); tb_ = {k: t82[k] for k in ("qid", "hd", "target_black")}
    parts = [dict(np.load(ROOT / f"results/m77/val{v}.npz")) for v in FOLDS]; e = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}; te = dict(np.load(ROOT / "results/m77/test.npz"))
    V = {"5e": {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_lab": d82["f5e_imp_lab"], "imp_m": d82["f5e_imp_m"]}, "enc": e}
    T = {"5e": {**tb_, "opp_lab": t82["f5e_opp_lab"], "opp_m": t82["f5e_opp_m"]}, "enc": {k: te[k] for k in ("qid", "hd", "target_black", "opp_lab", "opp_m")}}; used = ["5e"]
    for name in ids:
        if name == "5e":
            continue
        fv, ft = RV / f"m84_{name}_val.npz", RV / f"m84_{name}_test.npz"
        if not (fv.exists() and ft.exists()):
            print(f"identifier {name}: files missing, left out", flush=True); continue
        dp, tp = dict(np.load(fv)), dict(np.load(ft)); assert np.array_equal(dp["key"], base["key"])
        V[name] = {**base, "opp_lab": dp["f_opp_lab"], "opp_m": dp["f_opp_m"], "imp_lab": dp["f_imp_lab"], "imp_m": dp["f_imp_m"]}; T[name] = {**tb_, "opp_lab": tp["f_opp_lab"], "opp_m": tp["f_opp_m"]}; used.append(name)
    return V, T, dict(zip(base["key"], base["pid"])), used


def evidence(z, v, cc, pi0):
    """{game_key: log e [10]} of fold v, calibrated on the other folds: pooled or class-conditional tables, current or corrected weights."""
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)
    if pi0:
        _, cal, _ = m87.corrected(z, tr)
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s_ = g == k; a = cal[k]["a"][band[s_]][:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", default="5e,P3"); ap.add_argument("--pi0", action="store_true"); ap.add_argument("--floor", type=int, default=3); ap.add_argument("--out", default="ct64"); ap.add_argument("--json", default="m86_bundle.json"); ap.add_argument("--require", default="", help="identifiers that must be present; the run stops if one is missing"); a_ = ap.parse_args()
    V, T, key_pid, IDS = sources(a_.ids.split(",")); missing = [x for x in a_.require.split(",") if x and x not in IDS]
    if missing:
        sys.exit(f"required identifier(s) missing: {missing}; nothing written")
    PS = (0, 1) if a_.pi0 else (0,); COMBOS = list(itertools.product(range(len(IDS)), (0, 1), (0, 1), (0, 1), PS)); off = COMBOS[0]; rep = {"ids": IDS, "floor": a_.floor}
    label = lambda c: "+".join(x for x in ((f"id={IDS[c[0]]}" if c[0] else ""), "enc" * c[1], "cc" * c[2], "temp" * c[3], "pi0" * c[4]) if x) or "all off"  # noqa: E731
    F = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y)

        def qsum(ev):
            L = np.zeros((nq, 10))
            for i, g in enumerate(gk):
                for k in g:
                    if k in ev:
                        L[i] += ev[k]
            return L
        F[v] = {"y": np.asarray(y), "n": np.asarray(n), "pid": np.asarray(pid), "main": {(a, c, p_): qsum(evidence(V[IDS[a]], v, c, p_)) for a in range(len(IDS)) for c in (0, 1) for p_ in PS}, "enc": {p_: qsum(evidence(V["enc"], v, 0, p_)) for p_ in PS}}
        F[v]["T"] = np.array([TN[int(m)] for m in F[v]["n"]])[:, None]
    L_of = lambda v, c: (F[v]["main"][(c[0], c[2], c[4])] + c[1] * U_ENC * F[v]["enc"][c[4]]) * (F[v]["T"] if c[3] else 1.0)  # noqa: E731
    half = (list(FOLDS[:2]), list(FOLDS[2:])); picks = None; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {}
        for v in FOLDS:
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            for c in COMBOS:
                L = L_of(v, c); S[(c, v)] = SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), F[v]["y"]]
        fs = {c: np.array([balanced_mean_over_n(S[(c, v)], F[v]["y"], F[v]["n"], sizes=SIZES) for v in FOLDS]) for c in COMBOS}
        if picks is None:
            fi = {v: i for i, v in enumerate(FOLDS)}; best = lambda fl: max(COMBOS, key=lambda c: fs[c][[fi[v] for v in fl]].mean())  # noqa: E731
            picks = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}; final = best(list(FOLDS))
            print(f"scores of the {len(COMBOS)} combinations (mean over the four folds, seeds 0-2; difference to all off; the best 14 and the single switches):")
            singles = [c for c in COMBOS if sum(1 for x in c if x) == 1]
            for c in sorted(COMBOS, key=lambda c: -fs[c].mean())[:14] + singles:
                print(f"  {label(c):30s} {fs[c].mean():.4f}  {fs[c].mean() - fs[off].mean():+.4f}   folds " + " ".join(f"{x:+.4f}" for x in fs[c] - fs[off]))
            print(f"picked on folds {half[0]}: {label(best(half[0]))} | picked on folds {half[1]}: {label(best(half[1]))} | on all four: {label(final)}")
            rep["table"] = {label(c): {"mean": float(fs[c].mean()), "diff": float(fs[c].mean() - fs[off].mean()), "per_fold_diff": (fs[c] - fs[off]).tolist()} for c in COMBOS}
            rep["picks"] = {"from_first_half": label(best(half[0])), "from_second_half": label(best(half[1])), "all_folds": label(final)}
        y, n, pid = (np.concatenate([F[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); new = np.concatenate([S[(picks[v], v)] for v in FOLDS]); old = np.concatenate([S[(off, v)] for v in FOLDS])
        dd = paired_diff(new, old, y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[picks[v]][i] - fs[off][i]) for i, v in enumerate(FOLDS)]; good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
        print(f"seeds {seeds}: cross-fitted bundle vs all off {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((new != old).mean()):.3f} -> {'PASS' if good else 'no'}")
        fb = np.concatenate([S[(final, v)] for v in FOLDS]); df = paired_diff(fb, old, y, pid, n, sizes=SIZES, n=5000)
        print(f"seeds {seeds}: (not cross-fitted, report) the all-folds pick {label(final)} vs all off {df['diff']:+.4f} [{df['ci95'][0]:+.4f}, {df['ci95'][1]:+.4f}]")
        rep[f"seeds_{seeds[0]}"] = {"bundle_vs_off": {**dd, "per_fold": pf, "pass": bool(good)}, "final_pick_vs_off_not_crossfitted": df}; ok.append(good)
    passed = bool(all(ok) and final != off); rep["passed_cv"] = passed; print(f"PRE-REGISTERED: bundle passes: {'YES' if passed else 'NO'}")
    # ---- test side
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); dec = {}
    for f in FOLDS:
        y, n, pid, q = load_fold(f); P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(decide(P), int)) if n_ == 20})
    ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))

    def test_evidence(name, cc, pi0, floor):
        """Question-level log-evidence [800, 10] of one source on the test (m78_build_candidate.py's rules) and its density ratio."""
        v, t = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1] if pi0 else cal
        band_of = lambda m, mb_: np.where(mb_, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731
        vb_imp = band_of(v["imp_m"], v["target_black"]); tb = band_of(t["opp_m"], ~t["target_black"]); tg = s2.group_of(t["hd"], t["target_black"]); assert all(k in key_pid for k in v["key"]), f"{name}: validation keys outside the anchor query set"
        vd = np.array([dec[key_pid[k]] for k in v["key"]]); td = ct9.reindex(t["qid"]).to_numpy(); num = den = 0.0
        for b in range(4, 9):
            w1 = lambda lab, bb, d_, s: float((np.abs(lab[s & (bb == b)] - d_[s & (bb == b)]) <= 1).mean()) if (s & (bb == b)).any() else np.nan  # noqa: E731
            av, ai, at = w1(v["opp_lab"], vband, vd, vg >= 0), w1(v["imp_lab"], vb_imp, vd, vg >= 0), w1(t["opp_lab"], tb, td, tg >= 0); w = int(((tg >= 0) & (tb == b)).sum())
            if w and np.isfinite([av, ai, at]).all():
                num += w * (at - ai); den += w * (av - ai)
        r_agree = float(np.clip(num / den, 0, 1)); sh_v, sh_t = float((vband[vg >= 0] >= 3).mean() - nominal[3:].sum()), float((tb[tg >= 0] >= 3).mean() - nominal[3:].sum()); r = float(min(r_agree, np.clip(sh_t / sh_v, 0, 1)))
        R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10))
        for k in (0, 1, 2):
            s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = t["opp_lab"][s]
            ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio)
        return pd.DataFrame(logE).groupby(t["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(), r
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44 = sub("ct9"), sub("ct28"), sub("ct44"); lev = np.array(LEVELS); locked = c28 != c9; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([TN[int(m)] for m in nq_t])[:, None]; cache = {}

    def build(c, floor):
        km = (IDS[c[0]], c[2], c[4], floor)
        if km not in cache:
            cache[km] = test_evidence(*km)
        L, r_main = cache[km]; r_enc = None
        if c[1]:
            ke = ("enc", 0, c[4], floor)
            if ke not in cache:
                cache[ke] = test_evidence(*ke)
            L = L + U_ENC * cache[ke][0]; r_enc = cache[ke][1]
        L = L * Tq if c[3] else L
        return np.where(locked, c28, lev[mb.decisions(P9, L, 0.5, 3.0, 0.6)[2]]), r_main, r_enc
    base44, r0, _ = build(off, 3); repro = bool((base44 == c44).all()); print(f"builder with every switch off (floor q90) reproduces ct44: {'exact' if repro else f'FAILED, {int((base44 != c44).sum())} questions differ'} (density ratio {r0:.3f})")
    fin, r_main, r_enc = build(final, a_.floor); ch = fin != c44; pub = np.char.startswith(qids.astype(str), "pub_"); jump = np.abs(np.array([RANK[x] for x in fin]) - np.array([RANK[x] for x in c44]))
    print(f"test, combination {label(final)}, bands >= index {a_.floor}: density ratio main {r_main:.3f}" + (f", encoder {r_enc:.3f}" if r_enc is not None else "") + f"; {int(ch.sum())} questions differ from ct44 (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}; one level {int((jump == 1).sum())}, two or more {int((jump >= 2).sum())}); vs ct28 {int((fin != c28).sum())}")
    rep["test"] = {"repro_ct44": repro, "r_main": r_main, "r_enc": r_enc, "changed_vs_ct44": int(ch.sum()), "public": int(ch[pub].sum()), "private": int(ch[~pub].sum()), "two_or_more": int((jump >= 2).sum())}
    rep["adopt"] = bool(passed and repro); rep["file"] = a_.out
    if rep["adopt"]:
        f = ROOT / f"submission_rank_{a_.out}.csv"; write_submission(list(zip(qids, fin)), f); rep["md5"] = hashlib.md5(f.read_bytes()).hexdigest()[:12]; print(f"{a_.out} written, md5 {rep['md5']}")
        for other in ("ct64", "ct70", "ct71"):
            if other != a_.out and (ROOT / f"submission_rank_{other}.csv").exists():
                co = sub(other); print(f"   vs {other}: {int((fin != co).sum())} questions differ (public {int(((fin != co) & pub).sum())}, private {int(((fin != co) & ~pub).sum())})"); rep["test"][f"changed_vs_{other}"] = int((fin != co).sum())
    (RV / a_.json).write_text(json.dumps(rep, indent=1, default=float)); print("written")
    subprocess.run([sys.executable, "-I", "scripts/m_series/m84_variant.py", "--stage", "finalize"], cwd=ROOT)


if __name__ == "__main__":
    main()
