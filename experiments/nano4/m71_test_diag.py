"""Section 4M M71 test-side diagnostics (2026-11-10): the pre-registered check failed (the test TARGET side exceeds the impostor bands).
Two label-free questions:
 A  Are the test's high-margin OPPONENT matches real anchors? Agreement of the matched label with the model's own decision for the
    question (ct9 on the test; the 6-member OOF decision of the player's n = 20 question on validation), by margin band, for:
    validation opponents, validation impostors, test opponents, test target side. Real anchors agree far more than impostors.
    rho_agree = (test opp - impostor) / (validation opp - impostor) in the bands above q95.
 B  Are some test TARGETS in the training pool? Per question, the plurality count of the target-side top-1 candidate (how many of the
    question's games point to the same training player), against the validation impostor of the same question size (the first n games
    of each validation player's 20; own player excluded, so "one person's games agree with each other" is in the null).
  python -I logs/nano4/m71_test_diag.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
v = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); t = dict(np.load(ROOT / "results/r_variants/m71_test_queries.npz"))
edges, cal, vband, vg = s2.fit({**v, "imp_m": v["imp_m"]}, np.ones(len(v["y"]), bool))
band = lambda m, model_black: np.where(model_black, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731
vb_opp, vb_imp = band(v["opp_m"], ~v["target_black"]), band(v["imp_m"], v["target_black"])
tb_opp, tb_tgt = band(t["opp_m"], ~t["target_black"]), band(t["tgt_m"], t["target_black"])
# the model's own decision per player (validation: n = 20 question, 6 members) and per test question (ct9)
dec = {}
for f in (1, 2, 3, 4):
    y, n, pid, q = load_fold(f)
    P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); d = np.asarray(decide(P), int)
    for p_, n_, d_ in zip(pid, n, d):
        if n_ == 20:
            dec[p_] = d_
vd = np.array([dec[p_] for p_ in v["pid"]])
ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(RANK); td = ct9.reindex(t["qid"]).to_numpy()
names = ["<=q50", "q50-80", "q80-90", "q90-95", "q95-98", "q98-99", "q99-99.5", "q99.5-99.9", ">q99.9"]; rep = {"A": {}, "B": {}}
print("A  agreement of the matched label with the MODEL'S decision (exact / within one), HD0 + HD1 queries")
print(f"   {'band':11s} | val opp      | val impostor | test opp     | test target  ")
sv, st_ = vg >= 0, s2.group_of(t["hd"], t["target_black"]) >= 0
num = den = 0.0
for b, nm in enumerate(names):
    cells = []
    for lab, bb, d_, s in ((v["opp_lab"], vb_opp, vd, sv), (v["imp_lab"], vb_imp, vd, sv), (t["opp_lab"], tb_opp, td, st_), (t["tgt_lab"], tb_tgt, td, st_)):
        m = s & (bb == b); x = lab[m] - d_[m]; cells.append((float((x == 0).mean()) if m.any() else np.nan, float((np.abs(x) <= 1).mean()) if m.any() else np.nan, int(m.sum())))
    rep["A"][nm] = cells
    print(f"   {nm:11s} | " + " | ".join(f"{e:.3f} {w:.3f} {k:5d}" for e, w, k in cells))
    if b >= 4 and cells[2][2] > 0:
        w = cells[2][2]; num += w * (cells[2][1] - cells[1][1]); den += w * (cells[0][1] - cells[1][1])
rho_agree = float(num / den) if den else float("nan"); rep["rho_agree_within1_above_q95"] = rho_agree
print(f"   rho from within-one agreement above q95: {rho_agree:.2f}  (share-based rho was 0.78)")
# B plurality of the target-side top-1 candidate within a question
print("B  share of questions whose target-side top-1 candidate repeats in >= k games")
tq = pd.DataFrame({"qid": t["qid"], "best": t["tgt_best"], "hi": tb_tgt >= 6, "lab": t["tgt_lab"]})
plur = tq.groupby("qid").agg(n=("best", "size"), top=("best", lambda x: Counter(x).most_common(1)[0][1]), who=("best", lambda x: Counter(x).most_common(1)[0][0]))
hi_same = tq[tq.hi].groupby("qid").best.agg(lambda x: Counter(x).most_common(1)[0][1]).reindex(plur.index).fillna(0).astype(int); plur["hi_same"] = hi_same
vq = pd.DataFrame({"pid": v["pid"], "best": v["imp_best"], "hi": vb_imp >= 6}); vq["k"] = vq.groupby("pid").cumcount()
for n_ in (5, 10, 15, 20):
    sub = vq[vq.k < n_]; null = sub.groupby("pid").best.agg(lambda x: Counter(x).most_common(1)[0][1]); null_hi = sub[sub.hi].groupby("pid").best.agg(lambda x: Counter(x).most_common(1)[0][1]).reindex(null.index).fillna(0)
    te = plur[plur.n == n_]
    row = {f"test_ge{k}": float((te.top >= k).mean()) for k in (2, 3, 4)} | {f"null_ge{k}": float((null >= k).mean()) for k in (2, 3, 4)} | {"test_hi2": float((te.hi_same >= 2).mean()), "null_hi2": float((null_hi >= 2).mean()), "questions": int(len(te))}
    rep["B"][str(n_)] = row
    print(f"   n={n_:2d} ({len(te)} test questions): >=2 {row['test_ge2']:.3f} (null {row['null_ge2']:.3f})  >=3 {row['test_ge3']:.3f} (null {row['null_ge3']:.3f})  >=4 {row['test_ge4']:.3f} (null {row['null_ge4']:.3f})"
          f"  | >=2 games above q99 on the same player: {row['test_hi2']:.3f} (null {row['null_hi2']:.4f})")
strong = plur[(plur.top >= 4) | (plur.hi_same >= 2)].copy()
lab_of = pd.Series(t["tgt_lab"], index=t["tgt_best"]); lab_of = lab_of[~lab_of.index.duplicated()]
strong["label"] = [LEVELS[int(lab_of[w])] for w in strong.who]; strong["ct9"] = [LEVELS[int(ct9[q])] for q in strong.index]
exact, within = float((strong.label == strong.ct9).mean()) if len(strong) else None, float(np.mean([abs(RANK[a] - RANK[b]) <= 1 for a, b in zip(strong.label, strong.ct9)])) if len(strong) else None
print(f"   test questions with a strongly repeated target-side candidate (>= 4 games, or >= 2 above q99): {len(strong)} (public {int(strong.index.str.startswith('pub_').sum())}, private {int(strong.index.str.startswith('priv_').sum())});"
      f" that player's label equals ct9's decision in {exact}, within one level in {within}; by n {strong.n.value_counts().sort_index().to_dict()}")
rep["B"]["strong"] = {"questions": int(len(strong)), "label_eq_ct9": exact, "within1": within}
Path(ROOT / "results/r_variants/m71_test_diag.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
