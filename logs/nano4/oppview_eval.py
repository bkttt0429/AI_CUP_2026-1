"""Opponent-view pilot evaluation (section 4M M13).  Player level: the opponent-view pmi of the target (posterior over r_t from the
opponent's moves, r_o marginalised or at the opponent's best r_o), its Spearman with the label, the partial Spearman given the
target's own HSL pmi (and given the opponent's L1b pmi), and the correlation of its residual (on own pmi) with the persistent bias
E - y of floor_decomp; the KL sensitivity control through the same statistics; GPT's oracle bound on adjacent top-2 confusions.
  python -I logs/nano4/oppview_eval.py --root results/oppview/fold1 --fold 1 --out results/r_variants/oppview_pilot.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, score  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from oppview_score import R_O, R_T  # noqa: E402
from pilot_stats import spearman  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}; PIDX = {p: i for i, p in enumerate(PROFILES)}


def lse(x, axis):
    m = x.max(axis=axis, keepdims=True); return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def ranks(v):
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def partial(x, y, Z):
    rx, ry = ranks(x), ranks(y); RZ = np.column_stack([np.ones(len(rx))] + [ranks(z) for z in Z])
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]; ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def resid_on(x, z):
    A = np.column_stack([np.ones(len(x)), np.asarray(z, float)]); return np.asarray(x, float) - A @ np.linalg.lstsq(A, np.asarray(x, float), rcond=None)[0]


def boot(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def oracle_bound():
    ans = pd.read_csv(ROOT / "data/mock_tests/fold1_n20/answers.csv", dtype=str); y = ans["rank"].map(RANK).to_numpy()
    P = pd.read_csv(ROOT / "results/tta_val/sym0/submission_rank_ct_proba.csv").set_index("question_id").loc[ans["question_id"], [f"p_{l}" for l in LEVELS]].to_numpy()
    SM = np.array([[score(k, j) for j in range(10)] for k in range(10)]); ES = P @ SM.T
    top = np.argsort(-ES, axis=1)[:, :2]; s1 = np.array([score(int(t), int(t_)) for t, t_ in zip(top[:, 0], y)]); s2 = np.array([score(int(t), int(t_)) for t, t_ in zip(top[:, 1], y)])
    adj = np.abs(top[:, 0] - top[:, 1]) == 1; dan = (top.min(axis=1) >= RANK["1D"])
    out = {"questions": int(len(y)), "current": float(s1.mean())}
    for name, m in (("adjacent_top2", adj), ("adjacent_top2_dan_only", adj & dan), ("any_top2", np.ones(len(y), bool))):
        gain = np.where(m, np.maximum(s1, s2) - s1, 0.0); out[name] = {"share": float(m.mean()), "oracle_gain_per_question": float(gain.mean())}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True); a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str}) for f in sorted(glob.glob(str(Path(a.root) / "shard*.csv")))]); T = T[T.n_opp_moves >= 20].reset_index(drop=True)
    LL = T[[f"ll_{ro}_{rt}" for ro in R_O for rt in R_T]].to_numpy().reshape(len(T), len(R_O), len(R_T))
    tidx = np.array([PIDX[r] for r in R_T], float); tc = tidx - tidx.mean()
    marg = lse(LL, axis=1); p_t = np.exp(marg - lse(marg, axis=1)[:, None])
    best_o = lse(LL, axis=2).argmax(axis=1); fixed = LL[np.arange(len(T)), best_o, :]; p_f = np.exp(fixed - lse(fixed, axis=1)[:, None])
    T["ov_pmi"] = p_t @ tidx; T["ov_pmi_fixed"] = p_f @ tidx; T["opp_best_o"] = [PIDX[R_O[o]] for o in best_o]
    per_move_fixed = fixed / T.n_opp_moves.to_numpy()[:, None]; T["ov_slope"] = (per_move_fixed - per_move_fixed.mean(axis=1, keepdims=True)) @ tc / (tc @ tc)
    klcols = [c for c in T.columns if c.startswith("kl_")]; T["kl_mean"] = T[klcols].sum(axis=1) / T.n_opp_moves / len(klcols)
    P = T.groupby("player_id").agg(y=("rank", lambda s: RANK[s.iloc[0]]), ov=("ov_pmi", "mean"), ovf=("ov_pmi_fixed", "mean"), slope=("ov_slope", "mean"),
                                   kl=("kl_mean", "mean"), oppo=("opp_best_o", "mean"), games=("game_key", "count"))
    P = P[P.games >= 10]
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}"); cols = [f"sum_{p}" for p in PROFILES]
    H = hsl[hsl.player_id.isin(P.index)].copy(); L18 = H[cols].to_numpy(dtype=float); H["own_pmi"] = np.exp(L18 - lse(L18, 1)[:, None]) @ np.arange(18)
    P["own"] = H.groupby("player_id")["own_pmi"].mean()
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[a.fold] / f"fold{a.fold}"); O = opp[opp.player_id.isin(P.index)].copy()
    L10 = O[sum_cols(LEVELS)].to_numpy(dtype=float); O["opp_pmi"] = np.exp(L10 - lse(L10, 1)[:, None]) @ np.arange(10); P["oppl1b"] = O.groupby("player_id")["opp_pmi"].mean()
    bias = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str}); bias["b"] = (bias.e0 + bias.e1) / 2
    P["bias"] = P.index.map(dict(zip(bias.player_id, bias.b)))
    P = P.dropna(subset=["own", "oppl1b"]); n = len(P); y = P.y.to_numpy(); own = P.own.to_numpy(); oppl = P.oppl1b.to_numpy(); hasb = P.bias.notna().to_numpy()
    rep = {"players": int(n), "games": int(len(T)), "with_bias": int(hasb.sum()), "own_vs_y": spearman(own, y), "oppl1b_vs_y": spearman(oppl, y), "features": {}}
    for feat in ("ov", "ovf", "slope", "kl", "oppo"):
        x = P[feat].to_numpy(); r_own = resid_on(x, own)
        d = {"vs_y": spearman(x, y), "vs_own": spearman(x, own), "partial_y_given_own": partial(x, y, [own]), "partial_y_given_own_oppl1b": partial(x, y, [own, oppl]),
             "ci_partial_given_own": boot(lambda i: partial(x[i], y[i], [own[i]]), n)}
        if hasb.sum() > 50:
            b = P.bias.to_numpy()[hasb]; d["resid_vs_bias"] = spearman(r_own[hasb], b); d["ci_resid_vs_bias"] = boot(lambda i: spearman(r_own[hasb][i], b[i]), int(hasb.sum()))
        q = pd.qcut(own, 5, labels=False, duplicates="drop"); d["within_own_quintile_spearman_y"] = float(np.mean([spearman(x[q == k], y[q == k]) for k in np.unique(q) if (q == k).sum() > 20]))
        rep["features"][feat] = d
        print(feat, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
    rep["oracle"] = oracle_bound()
    print(json.dumps(rep["oracle"], indent=1)); Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
