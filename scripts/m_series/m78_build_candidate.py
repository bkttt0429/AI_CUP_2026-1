"""Section 4M (2026-11-10): test-side candidate from a given opponent identifier (fused five components: m73_*.npz; later the encoder).
 1 label-free density ratio r: within-one agreement of the matched label with the model's own decision, bands above the impostor q95,
   test opponents vs validation opponents, both net of the impostors (as m71_test_diag A)
 2 CV expectation at density r (weights x r): anchors on all levels / inside the boundary region only / pairwise act-at-0.6
 3 files: ct<A> = ct28 locked on the pushed questions + anchored decision inside the boundary region (bands >= q90, a = r x validation a);
          ct<B> = the same with the pairwise posterior >= 0.6 rule
  python -I scripts/m_series/m78_build_candidate.py --source fused --out 42 43
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, paired_diff  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
import m71_cv_density as cd  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=("fused", "encoder", "matcher", "file"), default="fused"); ap.add_argument("--out", nargs=2, default=["42", "43"])
ap.add_argument("--val"); ap.add_argument("--test"); ap.add_argument("--prefix", default="f_"); ap.add_argument("--ref", default=None); a = ap.parse_args()
REF = a.ref or ("ct42" if a.source == "matcher" else "ct41")
if a.source == "file":  # M84: any identifier's files in the m82_eval.py layout (--val, --test, --prefix)
    d = dict(np.load(a.val)); t = dict(np.load(a.test)); p_ = a.prefix
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d[p_ + "opp_lab"], "opp_m": d[p_ + "opp_m"], "imp_lab": d[p_ + "imp_lab"], "imp_m": d[p_ + "imp_m"]}
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t[p_ + "opp_lab"], "opp_m": t[p_ + "opp_m"], "tgt_lab": t[p_ + "tgt_lab"], "tgt_m": t[p_ + "tgt_m"]}
elif a.source == "matcher":  # M82: five components + the learned conditional matcher (m82_eval.py's files, prefix f5e_)
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); t = dict(np.load(ROOT / "results/r_variants/m82_test.npz"))
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t["f5e_opp_lab"], "opp_m": t["f5e_opp_m"], "tgt_lab": t["f5e_tgt_lab"], "tgt_m": t["f5e_tgt_m"]}
elif a.source == "fused":
    d = dict(np.load(ROOT / "results/r_variants/m73_val.npz")); t = dict(np.load(ROOT / "results/r_variants/m73_test.npz"))
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f_opp_lab"], "opp_m": d["f_opp_m"], "imp_lab": d["f_imp_lab"], "imp_m": d["f_imp_m"]}
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t["f_opp_lab"], "opp_m": t["f_opp_m"], "tgt_lab": t["f_tgt_lab"], "tgt_m": t["f_tgt_m"]}
else:
    parts = [dict(np.load(ROOT / f"results/m77/val{f}.npz")) for f in (1, 2, 3, 4)]; v = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "opp_best"}
    t = dict(np.load(ROOT / "results/m77/test.npz")); T = {k: t[k] for k in ("qid", "hd", "target_black", "opp_lab", "opp_m", "tgt_lab", "tgt_m")}
edges, cal, vband, vg = s2.fit(v, np.ones(len(v["y"]), bool))
band_of = lambda m, model_black: np.where(model_black, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731
vb_imp = band_of(v["imp_m"], v["target_black"]); tb = band_of(T["opp_m"], ~T["target_black"]); tg = s2.group_of(T["hd"], T["target_black"])
dec = {}
for f in (1, 2, 3, 4):
    y, n, pid, q = load_fold(f); P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dd = np.asarray(decide(P), int)
    dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, dd) if n_ == 20})
vd = np.array([dec[p_] for p_ in v["pid"]]); ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(RANK); td = ct9.reindex(T["qid"]).to_numpy()
num = den = 0.0
for b in range(4, 9):
    w1 = lambda lab, bb, d_, s: float((np.abs(lab[s & (bb == b)] - d_[s & (bb == b)]) <= 1).mean()) if (s & (bb == b)).any() else np.nan  # noqa: E731
    av, ai, at = w1(v["opp_lab"], vband, vd, vg >= 0), w1(v["imp_lab"], vb_imp, vd, vg >= 0), w1(T["opp_lab"], tb, td, tg >= 0); w = int(((tg >= 0) & (tb == b)).sum())
    if w and np.isfinite([av, ai, at]).all():
        num += w * (at - ai); den += w * (av - ai)
r_agree = float(np.clip(num / den, 0, 1)); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
sh_v, sh_t = float((vband[vg >= 0] >= 3).mean() - nominal[3:].sum()), float((tb[tg >= 0] >= 3).mean() - nominal[3:].sum())
# under-weighting is free in CV (full density, weights x0.74: +0.0080 vs +0.0078 matched) while over-weighting costs about 0.001, so the LOWER of the two label-free estimates is used
r = float(min(r_agree, np.clip(sh_t / sh_v, 0, 1)))
print(f"source {a.source}: density ratio: agreement-based {r_agree:.3f}, share-based {sh_t / sh_v:.3f} -> used r = {r:.3f} (the lower one); excess above q90: test {sh_t:+.4f}, validation {sh_v:+.4f}")
# report only: does the impostor null transfer to the test? the test questions' OWN side against the pool (their players are not in it)
tt_b = band_of(T["tgt_m"], T["target_black"]); print("null check, share of own-side queries at or above the impostor q90 / q99 / q99.9 (nominal 0.100 / 0.010 / 0.001): test " + " / ".join(f"{float((tt_b[tg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8))
                                                      + "; validation " + " / ".join(f"{float((vb_imp[vg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8)) + "; opponent side: test " + " / ".join(f"{float((tb[tg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8))
                                                      + "; validation " + " / ".join(f"{float((vband[vg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8)))
print(f"CV expectation at density {r:.2f} with weights x{r:.2f} (three thinning draws):")
for seeds in ((0, 1, 2), (6, 7, 8)):
    acc = {"all levels": [], "region only": [], "pair >= 0.6": []}
    for draw in (1, 2, 3):
        data = mb.fold_data(v, cd.thin(v, r, draw), r, seeds); y, n, pid = (np.concatenate([x[k] for x in data]) for k in (0, 1, 2)); cols = {k: [] for k in acc}; d0s = []
        for yy, nn, pp, P, L in data:
            d0, dA, dAr, dB, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); d0s.append(d0); cols["all levels"].append(dA); cols["region only"].append(dAr); cols["pair >= 0.6"].append(dB)
        d0 = np.concatenate(d0s)
        for k in acc:
            dk = np.concatenate(cols[k]); q = paired_diff(mb.S[dk, y], mb.S[d0, y], y, pid, n, sizes=SIZES, n=2000); acc[k].append((q["diff"], q["ci95"][0], float((dk != d0).mean())))
    print(f"  seeds {seeds}: " + " | ".join(f"{k}: {np.mean(x, axis=0)[0]:+.4f} (CI lower {np.mean(x, axis=0)[1]:+.4f}, changed {np.mean(x, axis=0)[2]:.3f})" for k, x in acc.items()))
# test evidence
logE = np.zeros((len(tb), 10))
for k in (0, 1, 2):
    s = tg == k; aw = r * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= 3); dd = T["opp_lab"][s][:, None] - np.arange(10)[None, :]
    logE[s] = np.log((1 - aw[tb[s]])[:, None] + aw[tb[s]][:, None] * cal[k]["ratio"][dd + 9])
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); cols_ = [f"p_{lv}" for lv in LEVELS]
L = pd.DataFrame(logE).groupby(T["qid"]).sum().reindex(pr.question_id).fillna(0.0).to_numpy(); P = pr[cols_].to_numpy(float)
d0, dA, dAr, dB, region = mb.decisions(P, L, 0.5, 3.0, 0.6)
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(pr.question_id).to_numpy()  # noqa: E731
c9, c28, c41 = sub("ct9"), sub("ct28"), sub(REF); lev = np.array(LEVELS); assert (lev[d0] == c9).all()
locked = c28 != c9; pub = pr.question_id.str.startswith("pub_").to_numpy()
for name, dk in ((a.out[0], dAr), (a.out[1], dB)):
    final = np.where(locked, c28, lev[dk]); write_submission(list(zip(pr.question_id, final)), ROOT / f"submission_rank_ct{name}.csv")
    for ref_n, ref in (("ct28", c28), (REF, c41)):
        ch = final != ref; print(f"ct{name} vs {ref_n}: {int(ch.sum())} differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())})" + (("; moves " + str(pd.Series([f'{x}->{y_}' for x, y_ in zip(ref[ch], final[ch])]).value_counts().head(12).to_dict())) if ref_n == "ct28" else ""))
    jump = np.abs(np.array([RANK[x] for x in final]) - np.array([RANK[x] for x in c28])); print(f"   changes vs ct28 by size: one level {int((jump == 1).sum())}, two or more {int((jump >= 2).sum())}; md5 {hashlib.md5((ROOT / f'submission_rank_ct{name}.csv').read_bytes()).hexdigest()[:12]}")
Pn = P * np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); out = pr.copy(); out[cols_] = Pn; out.to_csv(ROOT / f"submission_rank_ct{a.out[0]}_proba.csv", index=False)
print(f"boundary region covers {float(region.mean()):.3f} of the test questions; locked {int(locked.sum())}")
