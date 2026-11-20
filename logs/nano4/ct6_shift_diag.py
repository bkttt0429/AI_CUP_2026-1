"""ct6 vs ct3 (2026-11-05, Public 0.7530 vs 0.7625): is the opponent-HSL channel distributed on the test set as in the training folds?
T1 per-question mean (opponent HSL pmi - own HSL pmi), own pmi, opponent pmi, HD1 share: test (ct6 shards) vs folds 1-4 pilot questions
   (training tables), overall, by n and by class (test: the ct3 decision)
T2 the shift the channel induces: E_ct6 - E_ct3 against the gap on the test; E_oh - E_x40 against the gap on the CV questions (slopes, correlations)
T3 the questions whose decision changed: public / private, +1 / -1, their gap and HD1 share against the unchanged ones
  python -I logs/nano4/ct6_shift_diag.py --out results/r_variants/ct6_shift_diag.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COLS = [f"sum_{p}" for p in PROFILES]
PC = [f"p_{lv}" for lv in LEVELS]


def q(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std()), "q10": float(np.quantile(x, .1)), "q50": float(np.quantile(x, .5)), "q90": float(np.quantile(x, .9)), "n": int(len(x))}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.polyfit(x, y, 1)[0]
    return {"slope": float(b), "corr": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def test_side():
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS)
    tasks, owner, _, _ = ps.build_tasks(questions)
    own, opp, hd = {}, {}, {}
    for i in range(8):
        with np.load(ROOT / "results/ct6" / f"submission_rank_shard{i}of8.npz", allow_pickle=False) as z:
            for k, s, o in zip(z["task"].tolist(), z["sums"], z["hsl_opp_sums"]):
                own[k], opp[k] = s, o
    for k, sgf, _ in tasks:
        m = re.search(r"HD\[(\d+)\]", sgf)
        hd[k] = float(m.group(1)) if m else np.nan
    rows = []
    for qi, qq in enumerate(questions):
        ks = [k for k, o in enumerate(owner) if o == qi]
        if not ks:
            continue
        op, ow = posterior_mean_index(np.stack([opp[k] for k in ks])), posterior_mean_index(np.stack([own[k] for k in ks]))
        h = np.array([hd[k] for k in ks])
        rows.append({"question_id": qq["question_id"], "n": len(ks), "own": ow.mean(), "opp": op.mean(), "gap": (op - ow).mean(), "gap_sd": (op - ow).std(),
                     "hd1": float(np.nanmean(h >= 1))})
    T = pd.DataFrame(rows)
    for tag in ("ct3", "ct6"):
        P = pd.read_csv(ROOT / f"submission_rank_{tag}_proba.csv", dtype={"question_id": str}).set_index("question_id").loc[T.question_id, PC].to_numpy()
        T[f"E_{tag}"] = P @ np.arange(len(LEVELS))
        T[f"pred_{tag}"] = pd.read_csv(ROOT / f"submission_rank_{tag}.csv", dtype=str).set_index("question_id").loc[T.question_id, "pred_rank"].map(RANK).to_numpy()
    T["public"] = T.question_id.str.startswith("pub_")
    return T


def train_side(folds=(1, 2, 3, 4)):
    out = []
    for v in folds:
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        hsl = read_scored_games(ts.HSL / f"fold{v}").set_index("game_key")
        oh = read_scored_games(ts.OPP_HSL / f"fold{v}").set_index("game_key")
        hdt = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"]
        own = pd.Series(posterior_mean_index(hsl[COLS].to_numpy(float)), index=hsl.index)
        opp = pd.Series(posterior_mean_index(oh[COLS].to_numpy(float)), index=oh.index)
        E = {}
        for name, arms in (("x40", ("C-x40", "T-x40")), ("oh", ("C-x40-oh", "T-x40-oh"))):
            P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{a}_s{s}.npz")["proba"].astype(float) for a in arms for s in (0, 1, 2)], axis=0)
            E[name] = P @ np.arange(len(LEVELS))
        for i, keys in enumerate(meta["game_keys"]):
            keys = list(keys)
            g = (opp.loc[keys] - own.loc[keys]).to_numpy()
            out.append({"fold": v, "n": int(meta["n_games"].iat[i]), "y": RANK[meta["rank"].iat[i]], "own": float(own.loc[keys].mean()), "opp": float(opp.loc[keys].mean()),
                        "gap": float(g.mean()), "gap_sd": float(g.std()), "hd1": float((hdt.reindex(keys).fillna(0) >= 1).mean()), "E_x40": float(E["x40"][i]), "E_oh": float(E["oh"][i])})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    T, R = test_side(), train_side()
    rep = {"test_questions": int(len(T)), "train_questions": int(len(R))}
    print(f"test {len(T)} questions, train (folds 1-4 pilot) {len(R)} questions")
    # T1
    rep["T1"] = {"gap": {"test": q(T.gap), "train": q(R.gap)}, "gap_sd_within_q": {"test": q(T.gap_sd), "train": q(R.gap_sd)},
                 "own": {"test": q(T.own), "train": q(R.own)}, "opp": {"test": q(T.opp), "train": q(R.opp)}, "hd1": {"test": q(T.hd1), "train": q(R.hd1)}}
    for k, v in rep["T1"].items():
        print(f"T1 {k:16s} test mean {v['test']['mean']:+.3f} sd {v['test']['sd']:.3f} [q10 {v['test']['q10']:+.2f} q90 {v['test']['q90']:+.2f}] | train mean {v['train']['mean']:+.3f} sd {v['train']['sd']:.3f} [q10 {v['train']['q10']:+.2f} q90 {v['train']['q90']:+.2f}]")
    rep["T1_gap_by_n"] = {"test": {int(n): q(g.gap) for n, g in T.groupby("n")}, "train": {int(n): q(g.gap) for n, g in R.groupby("n")}}
    print("T1 gap by n: test " + " ".join(f"{n}:{v['mean']:+.3f}" for n, v in rep["T1_gap_by_n"]["test"].items()) + " | train " + " ".join(f"{n}:{v['mean']:+.3f}" for n, v in rep["T1_gap_by_n"]["train"].items()))
    byc_t = {LEVELS[c]: {"own": float(g.own.mean()), "opp": float(g.opp.mean()), "gap": float(g.gap.mean()), "n": int(len(g))} for c, g in T.groupby("pred_ct3")}
    byc_r = {LEVELS[c]: {"own": float(g.own.mean()), "opp": float(g.opp.mean()), "gap": float(g.gap.mean()), "n": int(len(g))} for c, g in R.groupby("y")}
    rep["T1_by_class"] = {"test_by_ct3_decision": byc_t, "train_by_label": byc_r}
    print("T1 by class (own pmi | opp pmi | gap): test by ct3 decision / train by label")
    for lv in LEVELS:
        t, r = byc_t.get(lv), byc_r.get(lv)
        print(f"   {lv:2s} test {t['own']:.2f} | {t['opp']:.2f} | {t['gap']:+.3f} (n {t['n']})   train {r['own']:.2f} | {r['opp']:.2f} | {r['gap']:+.3f} (n {r['n']})")
    # T2
    rep["T2"] = {"test_dE_vs_gap": slope(T.gap, T.E_ct6 - T.E_ct3), "train_dE_vs_gap_all": slope(R.gap, R.E_oh - R.E_x40), "train_dE_vs_gap_n20": slope(R[R.n == 20].gap, (R.E_oh - R.E_x40)[R.n == 20]),
                 "test_dE_mean": float((T.E_ct6 - T.E_ct3).mean()), "train_dE_mean": float((R.E_oh - R.E_x40).mean()),
                 "test_dE_sd": float((T.E_ct6 - T.E_ct3).std()), "train_dE_sd": float((R.E_oh - R.E_x40).std())}
    print(f"T2 dE vs gap: test slope {rep['T2']['test_dE_vs_gap']['slope']:+.3f} corr {rep['T2']['test_dE_vs_gap']['corr']:+.3f}; train all slope {rep['T2']['train_dE_vs_gap_all']['slope']:+.3f} corr {rep['T2']['train_dE_vs_gap_all']['corr']:+.3f}; "
          f"train n=20 slope {rep['T2']['train_dE_vs_gap_n20']['slope']:+.3f}; mean dE test {rep['T2']['test_dE_mean']:+.3f} (sd {rep['T2']['test_dE_sd']:.3f}) train {rep['T2']['train_dE_mean']:+.3f} (sd {rep['T2']['train_dE_sd']:.3f})")
    # train: did the oh ensemble's gain concentrate where the gap is large? score by gap tercile for both ensembles
    # T3
    T["changed"] = T.pred_ct3 != T.pred_ct6; T["shift"] = T.pred_ct6 - T.pred_ct3
    ch = T[T.changed]
    rep["T3"] = {"changed": int(len(ch)), "public_changed": int(ch.public.sum()), "private_changed": int((~ch.public).sum()),
                 "shift_counts_public": {int(k): int(v) for k, v in ch[ch.public]["shift"].value_counts().items()}, "shift_counts_private": {int(k): int(v) for k, v in ch[~ch.public]["shift"].value_counts().items()},
                 "gap_changed_up": float(ch[ch["shift"] > 0].gap.mean()), "gap_changed_down": float(ch[ch["shift"] < 0].gap.mean()), "gap_unchanged": float(T[~T.changed].gap.mean()),
                 "hd1_changed": float(ch.hd1.mean()), "hd1_unchanged": float(T[~T.changed].hd1.mean()),
                 "E_ct3_changed_frac": float(((ch.E_ct3 - np.round(ch.E_ct3)).abs()).mean()), "E_ct3_unchanged_frac": float(((T[~T.changed].E_ct3 - np.round(T[~T.changed].E_ct3)).abs()).mean()),
                 "n_changed": {int(k): int(v) for k, v in ch.n.value_counts().sort_index().items()}}
    print(f"T3 changed {len(ch)} (public {rep['T3']['public_changed']}, private {rep['T3']['private_changed']}); shifts public {rep['T3']['shift_counts_public']} private {rep['T3']['shift_counts_private']}; "
          f"gap: up {rep['T3']['gap_changed_up']:+.3f} down {rep['T3']['gap_changed_down']:+.3f} unchanged {rep['T3']['gap_unchanged']:+.3f}; hd1 changed {rep['T3']['hd1_changed']:.3f} unchanged {rep['T3']['hd1_unchanged']:.3f}")
    # T2b: what drives the shift -- dE against own pmi and opponent pmi separately, test vs train; dE by n
    rep["T2b"] = {"test_dE_vs_own": slope(T.own, T.E_ct6 - T.E_ct3), "test_dE_vs_opp": slope(T.opp, T.E_ct6 - T.E_ct3),
                  "train_dE_vs_own": slope(R.own, R.E_oh - R.E_x40), "train_dE_vs_opp": slope(R.opp, R.E_oh - R.E_x40),
                  "test_dE_by_n": {int(n): float((g.E_ct6 - g.E_ct3).mean()) for n, g in T.groupby("n")}, "train_dE_by_n": {int(n): float((g.E_oh - g.E_x40).mean()) for n, g in R.groupby("n")}}
    print(f"T2b dE vs own pmi: test slope {rep['T2b']['test_dE_vs_own']['slope']:+.3f} train {rep['T2b']['train_dE_vs_own']['slope']:+.3f}; dE vs opp pmi: test {rep['T2b']['test_dE_vs_opp']['slope']:+.3f} train {rep['T2b']['train_dE_vs_opp']['slope']:+.3f}; "
          f"dE by n test {rep['T2b']['test_dE_by_n']} train {rep['T2b']['train_dE_by_n']}")
    # T4: does the oh gain in CV depend on the gap? score of each ensemble by gap bin, and reweighted to the test's gap distribution
    SM = np.array([[score(k, j) for j in range(len(LEVELS))] for k in range(len(LEVELS))])
    def decide(E_cols):
        return None
    Rsc = {}
    for v in sorted(R.fold.unique()):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        y = meta["rank"].map(RANK).to_numpy()
        for name, arms in (("x40", ("C-x40", "T-x40")), ("oh", ("C-x40-oh", "T-x40-oh"))):
            P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{a_}_s{s_}.npz")["proba"].astype(float) for a_ in arms for s_ in (0, 1, 2)], axis=0)
            pred = np.array([expected_score_decision(p) for p in P])
            Rsc.setdefault(name, []).append(np.array([score(int(p_), int(t_)) for p_, t_ in zip(pred, y)]))
    for name in Rsc:
        R[f"s_{name}"] = np.concatenate(Rsc[name])
    R["dscore"] = R.s_oh - R.s_x40
    bins = [-9, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 9]
    R["gbin"] = pd.cut(R.gap, bins); T["gbin"] = pd.cut(T.gap, bins)
    tab = R.groupby("gbin", observed=True).agg(n=("dscore", "size"), s_x40=("s_x40", "mean"), s_oh=("s_oh", "mean"), gain=("dscore", "mean"))
    tab["test_share"] = T.groupby("gbin", observed=True).size().reindex(tab.index).fillna(0) / len(T); tab["train_share"] = tab.n / len(R)
    rep["T4_gain_by_gap_bin"] = {str(k): {c: float(v[c]) for c in tab.columns} for k, v in tab.iterrows()}
    print("T4 CV gain (oh - x40 per-question score) by gap bin: bin | train n | x40 | oh | gain | test share | train share")
    for k, v in tab.iterrows():
        print(f"   {str(k):14s} {int(v.n):6d}  {v.s_x40:.4f}  {v.s_oh:.4f}  {v.gain:+.4f}  test {v.test_share:.3f} train {v.train_share:.3f}")
    w = (tab.test_share / tab.train_share).reindex(R.gbin).to_numpy(); w = np.nan_to_num(w, nan=0.0)
    rep["T4_reweighted"] = {"gain_plain": float(R.dscore.mean()), "gain_reweighted_to_test_gap": float(np.average(R.dscore, weights=w)),
                            "gain_test_like_gap_lt_0": float(R.loc[R.gap < 0, "dscore"].mean()), "gain_gap_ge_0": float(R.loc[R.gap >= 0, "dscore"].mean())}
    # also reweight jointly on (n, gap bin)
    jt = T.groupby(["n", "gbin"], observed=True).size() / len(T); jr = R.groupby(["n", "gbin"], observed=True).size() / len(R)
    T["n_cv"] = T.n.map({5: 5, 10: 10, 15: 20, 20: 20}); jt2 = T.groupby(["n_cv", "gbin"], observed=True).size() / len(T)
    wj = (jt2 / jr).reindex(pd.MultiIndex.from_arrays([R.n, R.gbin])).to_numpy(); wj = np.nan_to_num(wj, nan=0.0)
    rep["T4_reweighted"]["gain_reweighted_n_and_gap"] = float(np.average(R.dscore, weights=wj))
    print(f"T4 CV gain: plain {rep['T4_reweighted']['gain_plain']:+.4f}; reweighted to the test gap distribution {rep['T4_reweighted']['gain_reweighted_to_test_gap']:+.4f}; "
          f"reweighted on (n, gap) {rep['T4_reweighted']['gain_reweighted_n_and_gap']:+.4f}; gap<0 {rep['T4_reweighted']['gain_test_like_gap_lt_0']:+.4f} gap>=0 {rep['T4_reweighted']['gain_gap_ge_0']:+.4f}")
    T.to_csv(Path(a.out).with_suffix(".questions.csv"), index=False)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
