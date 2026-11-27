#!/usr/bin/env python
"""M259 (2026-11-20), report only, no test labels: is ONE test-side multiplier per identifier enough for the anchors' strength, or does
the correction depend on the confidence band? (design doc M259, registered before any reading)

Bands = quantiles 0.5 / 0.8 / 0.9 / 0.95 / 0.98 / 0.99 / 0.995 / 0.999 of the validation impostor margins (as m71_step2.fit). Per
identifier and band: (i) excess share e_b = share - q * nominal (q as m87_pi0), test / validation; (ii) agreement "label of the
recognised opponent within one level of the base decision": validation true opponents, validation impostors, test; r_b = (test -
impostor) / (validation - impostor). Primary (SQ4): top bands (7-8) minus middle bands (4-6) for both ratios, bootstrap over
validation players and test questions.

  python -I logs/nano4/m259_bands.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402

b, s2, ES, FOLDS = m3.b, m3.s2, m3.ES, m3.FOLDS
NAMES = ("SQ4~m15", "SQ2~m15", "SP", "SP2e", "S2", "S3c", "S3", "5e"); NOM = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); NB = len(NOM); TOP, MID = (7, 8), (4, 5, 6); B = 500


def counts(band, ok, agree, w):
    """Weighted counts per band [NB] and weighted agreeing counts."""
    n = np.bincount(band[ok], weights=w[ok], minlength=NB); a = np.bincount(band[ok], weights=(w * agree)[ok], minlength=NB); return n, a


def ratios(nv, av, ni, ai, nt, at):
    """Per band: excess ratio test / validation and agreement ratio; and the same for the pooled top and middle bands."""
    sv, st = nv / nv.sum(), nt / nt.sum(); qv, qt = min(1.0, sv[0] / NOM[0]), min(1.0, st[0] / NOM[0]); ev, et = sv - qv * NOM, st - qt * NOM

    def one(bs):
        bs = list(bs); rho = et[bs].sum() / ev[bs].sum() if ev[bs].sum() > 0 else np.nan; pv, pi, pt = av[bs].sum() / max(nv[bs].sum(), 1e-9), ai[bs].sum() / max(ni[bs].sum(), 1e-9), at[bs].sum() / max(nt[bs].sum(), 1e-9)
        return rho, (pt - pi) / (pv - pi) if abs(pv - pi) > 1e-9 else np.nan, pv, pi, pt
    per = [one((k,)) for k in range(NB)]; return per, one(TOP), one(MID), (sv, st, ev, et, qv, qt)


def main():
    V, T, key_pid, used = b.sources(list(NAMES)); LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); rng = np.random.default_rng(259); rep = {}
    print(f"identifiers found: {used}; bands = null quantiles {s2.QS}; top = bands {TOP}, middle = bands {MID}; bootstrap {B}", flush=True)
    for name in NAMES:
        if name not in V:
            continue
        v, t = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones)
        band_of = lambda m, mb_: np.where(mb_, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731
        vimp = band_of(v["imp_m"], v["target_black"]); tband = band_of(t["opp_m"], ~t["target_black"]); tg = s2.group_of(t["hd"], t["target_black"]); assert (tg >= 0).all()
        pidv = np.array([key_pid[k] for k in v["key"]]); has = np.array([p_ in dec for p_ in pidv]); vd = np.array([dec.get(p_, 0) for p_ in pidv]); td = ct9.reindex(t["qid"].astype(str)).to_numpy().astype(int)
        agv = (np.abs(v["opp_lab"] - vd) <= 1).astype(float); agi = (np.abs(v["imp_lab"] - vd) <= 1).astype(float); agt = (np.abs(t["opp_lab"] - td) <= 1).astype(float)
        up, uv = np.unique(pidv, return_inverse=True); uq, ut = np.unique(t["qid"].astype(str), return_inverse=True); rep[name] = {}
        for label, okv, okt in (("even + HD1", (vg >= 0) & has, tg >= 0), ("even only", (vg == 0) & has, tg == 0)):
            def run(wv, wt):
                nv, av = counts(vband, okv, agv, wv); ni, ai = counts(vimp, okv, agi, wv); nt, at = counts(tband, okt, agt, wt); return ratios(nv, av, ni, ai, nt, at)
            per, top, mid, (sv, st, ev, et, qv, qt) = run(np.ones(len(vband)), np.ones(len(tband))); boot = []
            for _ in range(B):
                wv = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[uv]; wt = np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq)).astype(float)[ut]; _, tp, md, _ = run(wv, wt); boot.append((tp[0] - md[0], tp[1] - md[1], tp[0], md[0], tp[1], md[1]))
            boot = np.array(boot); ci = lambda j: (float(np.nanpercentile(boot[:, j], 2.5)), float(np.nanpercentile(boot[:, j], 97.5)))  # noqa: E731
            print(f"\n== {name} | {label}: validation games {int(okv.sum())}, test games {int(okt.sum())}; null-mass factor q validation {qv:.3f}, test {qt:.3f}")
            print("   band | share validation, test | excess validation, test, ratio | within one level of the base decision: validation, impostor, test | agreement ratio")
            for k in range(NB):
                rho, r_, pv, pi, pt = per[k]; print(f"   {k} | {sv[k]:.4f} {st[k]:.4f} | {ev[k]:+.4f} {et[k]:+.4f} {rho:6.3f} | {pv:.3f} {pi:.3f} {pt:.3f} | {r_:6.3f}")
            d1, d2 = top[0] - mid[0], top[1] - mid[1]; c1, c2 = ci(0), ci(1)
            print(f"   excess ratio: top {top[0]:.3f} [{ci(2)[0]:.3f}, {ci(2)[1]:.3f}], middle {mid[0]:.3f} [{ci(3)[0]:.3f}, {ci(3)[1]:.3f}], top - middle {d1:+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}]")
            print(f"   agreement ratio: top {top[1]:.3f} [{ci(4)[0]:.3f}, {ci(4)[1]:.3f}], middle {mid[1]:.3f} [{ci(5)[0]:.3f}, {ci(5)[1]:.3f}], top - middle {d2:+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}]", flush=True)
            rep[name][label] = {"excess_ratio_top": top[0], "excess_ratio_mid": mid[0], "excess_diff": d1, "excess_ci": c1, "agree_ratio_top": top[1], "agree_ratio_mid": mid[1], "agree_diff": d2, "agree_ci": c2,
                                "bands": [{"share_val": float(sv[k]), "share_test": float(st[k]), "excess_ratio": float(per[k][0]), "agree_val": float(per[k][2]), "agree_imp": float(per[k][3]), "agree_test": float(per[k][4]), "agree_ratio": float(per[k][1])} for k in range(NB)]}
    p = rep["SQ4~m15"]["even + HD1"]; s1 = not (p["excess_ci"][0] <= 0 <= p["excess_ci"][1]); s2_ = not (p["agree_ci"][0] <= 0 <= p["agree_ci"][1]); same = np.sign(p["excess_diff"]) == np.sign(p["agree_diff"])
    verdict = "the correction depends on the band (next step registered separately)" if s1 and s2_ and same else ("one multiplier is enough: closed" if not s1 and not s2_ else "one of the two only: recorded, no second step")
    print(f"\nREGISTERED (SQ4, even + HD1): excess ratio top - middle {p['excess_diff']:+.3f} [{p['excess_ci'][0]:+.3f}, {p['excess_ci'][1]:+.3f}]; agreement ratio top - middle {p['agree_diff']:+.3f} [{p['agree_ci'][0]:+.3f}, {p['agree_ci'][1]:+.3f}]  ->  {verdict}")
    rep["verdict"] = verdict; (ROOT / "results/r_variants/m259_bands.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m259_bands.json")


if __name__ == "__main__":
    main()
