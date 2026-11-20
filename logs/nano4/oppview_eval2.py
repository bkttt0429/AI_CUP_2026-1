"""Second pass over the opponent-view pilot: separate 'the opponent adapts to the target's displayed rank' (ov given the opponent's
own strength) from 'the opponent's strength is a matchmaking proxy of the label' (oppo given the L1b opponent channel), and ask how
much of the full model's persistent bias each explains.  CPU, seconds.
  python -I logs/nano4/oppview_eval2.py --root results/oppview/fold1 --fold 1 --out results/r_variants/oppview_pilot2.json
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
from aicup.metric import LEVELS  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from oppview_eval import boot, lse, partial, resid_on  # noqa: E402
from oppview_score import R_O, R_T  # noqa: E402
from pilot_stats import spearman  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}; PIDX = {p: i for i, p in enumerate(PROFILES)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True); a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str}) for f in sorted(glob.glob(str(Path(a.root) / "shard*.csv")))]); T = T[T.n_opp_moves >= 20].reset_index(drop=True)
    LL = T[[f"ll_{ro}_{rt}" for ro in R_O for rt in R_T]].to_numpy().reshape(len(T), len(R_O), len(R_T))
    tidx = np.array([PIDX[r] for r in R_T], float); oidx = np.array([PIDX[r] for r in R_O], float)
    marg_t = lse(LL, axis=1); p_t = np.exp(marg_t - lse(marg_t, 1)[:, None]); marg_o = lse(LL, axis=2); p_o = np.exp(marg_o - lse(marg_o, 1)[:, None])
    T["ov"] = p_t @ tidx; T["oppo_pm"] = p_o @ oidx  # posterior means (continuous), not argmax
    P = T.groupby("player_id").agg(y=("rank", lambda s: RANK[s.iloc[0]]), ov=("ov", "mean"), oppo=("oppo_pm", "mean"), ov_sd=("ov", "std"), oppo_sd=("oppo_pm", "std"), games=("game_key", "count"))
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}"); cols = [f"sum_{p}" for p in PROFILES]
    H = hsl[hsl.player_id.isin(P.index)].copy(); L18 = H[cols].to_numpy(float); H["own"] = np.exp(L18 - lse(L18, 1)[:, None]) @ np.arange(18); P["own"] = H.groupby("player_id")["own"].mean()
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[a.fold] / f"fold{a.fold}"); O = opp[opp.player_id.isin(P.index)].copy()
    L10 = O[sum_cols(LEVELS)].to_numpy(float); O["oppl1b"] = np.exp(L10 - lse(L10, 1)[:, None]) @ np.arange(10); P["oppl1b"] = O.groupby("player_id")["oppl1b"].mean()
    own_l = read_scored_games(ROOT / "results/l1b" / ts.L1B[a.fold] / f"fold{a.fold}"); Q = own_l[own_l.player_id.isin(P.index)].copy()
    L10b = Q[sum_cols(LEVELS)].to_numpy(float); Q["ownl1b"] = np.exp(L10b - lse(L10b, 1)[:, None]) @ np.arange(10); P["ownl1b"] = Q.groupby("player_id")["ownl1b"].mean()
    bias = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str}); bias["b"] = (bias.e0 + bias.e1) / 2; bias["E"] = bias["b"] + bias["y"]
    P["bias"] = P.index.map(dict(zip(bias.player_id, bias.b))); P["E"] = P.index.map(dict(zip(bias.player_id, bias.E)))
    P = P.dropna(); n = len(P); y, own, ownl, oppl, ov, oppo, b, E = (P[c].to_numpy(float) for c in ("y", "own", "ownl1b", "oppl1b", "ov", "oppo", "bias", "E"))
    rep = {"players": int(n), "spearman_vs_y": {k: spearman(v, y) for k, v in (("own_hsl", own), ("own_l1b", ownl), ("opp_l1b", oppl), ("oppo_hsl", oppo), ("ov", ov), ("model_E", E))}}
    rep["partials_vs_y"] = {
        "ov | own, oppo": partial(ov, y, [own, oppo]), "ci": boot(lambda i: partial(ov[i], y[i], [own[i], oppo[i]]), n),
        "ov | own, ownl1b, oppl1b, oppo": partial(ov, y, [own, ownl, oppl, oppo]),
        "oppo | own, ownl1b, oppl1b": partial(oppo, y, [own, ownl, oppl]), "ci_oppo": boot(lambda i: partial(oppo[i], y[i], [own[i], ownl[i], oppl[i]]), n),
        "oppl1b | own, ownl1b, oppo": partial(oppl, y, [own, ownl, oppo]),
        "oppo | model_E": partial(oppo, y, [E]), "ci_oppo_E": boot(lambda i: partial(oppo[i], y[i], [E[i]]), n),
        "ov | model_E, oppo": partial(ov, y, [E, oppo]), "oppl1b | model_E": partial(oppl, y, [E]), "own_hsl | model_E": partial(own, y, [E])}
    r_oppo = resid_on(oppo, own); r_oppo2 = oppo - np.column_stack([np.ones(n), own, ownl, oppl]) @ np.linalg.lstsq(np.column_stack([np.ones(n), own, ownl, oppl]), oppo, rcond=None)[0]
    rep["vs_bias"] = {"resid(oppo|own) vs bias": spearman(r_oppo, b), "resid(oppo|own,ownl1b,oppl1b) vs bias": spearman(r_oppo2, b), "ci": boot(lambda i: spearman(r_oppo2[i], b[i]), n),
                      "resid(ov|own,oppo) vs bias": spearman(ov - np.column_stack([np.ones(n), own, oppo]) @ np.linalg.lstsq(np.column_stack([np.ones(n), own, oppo]), ov, rcond=None)[0], b),
                      "oppo - E vs bias": spearman(oppo - E * 0, b)}
    # how much of the bias variance does oppo explain on top of E?  OOF linear: b ~ 1 + E + oppo  vs  b ~ 1 + E  (5-fold)
    rng = np.random.default_rng(0); f = rng.integers(0, 5, n); ss = {"E": 0.0, "E+oppo": 0.0, "E+oppo+oppl1b": 0.0}; tot = ((b - b.mean()) ** 2).sum()
    for name, cols_ in (("E", [E]), ("E+oppo", [E, oppo]), ("E+oppo+oppl1b", [E, oppo, oppl])):
        X = np.column_stack([np.ones(n)] + cols_); pred = np.zeros(n)
        for k in range(5):
            tr = f != k; beta = np.linalg.lstsq(X[tr], b[tr], rcond=None)[0]; pred[~tr] = X[~tr] @ beta
        ss[name] = float(1 - ((b - pred) ** 2).sum() / tot)
    rep["oof_r2_of_bias"] = ss
    rep["grid_edges"] = {"share_ov_at_edges": float(((P.ov <= PIDX["6k"] + 0.5) | (P.ov >= PIDX["6d"] - 0.5)).mean()), "share_oppo_at_edges": float(((P.oppo <= PIDX["6k"] + 0.5) | (P.oppo >= PIDX["6d"] - 0.5)).mean())}
    print(json.dumps(rep, indent=1)); Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
