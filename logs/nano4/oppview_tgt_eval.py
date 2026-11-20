"""Section 4M M17 evaluation: the target's moves under the (opponent rank r_o, target rank r_t) grid (oppview_score.py --target).
Per game the conditioned curve = the ll row at the r_o nearest the opponent's HSL pmi (results/scores_opp_hd), the soft version weights
the r_o rows by the opponent's HSL posterior, the marginal lse over r_o is the grid-scale control; each gives a posterior-mean index on the
R_T grid (profile-index units). Player level (20 games): Spearman with the label, partials given the standard own pmi (own = opp = p),
given the marginal (isolates the conditioning), given the oh ensemble's expected rank E and the opponent strength; the residual against the
oh ensemble's bias. Continue condition (pre-registered): partial(cpmi, y | E, oppo) CI lower bound > 0 and >= 0.08.
  python -I logs/nano4/oppview_tgt_eval.py --root results/oppview_tgt/fold1 --fold 1 --out results/r_variants/tgtview_pilot.json
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
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from oppview_eval import boot, partial, resid_on  # noqa: E402
from oppview_score import R_O, R_T  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
PIDX = {p: i for i, p in enumerate(PROFILES)}
IO, IT = np.array([PIDX[r] for r in R_O]), np.array([PIDX[r] for r in R_T], dtype=float)  # grid ranks as profile indices
COLS = [f"sum_{p}" for p in PROFILES]


def pmi_on_grid(L):
    """Posterior-mean profile index over the R_T grid from log-likelihood rows [n, len(R_T)] (uniform prior)."""
    w = np.exp(L - L.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    return w @ IT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(f"{a.root}/shard*.csv"))], ignore_index=True)
    LL = np.stack([T[[f"ll_{ro}_{rt}" for rt in R_T]].to_numpy(dtype=float) for ro in R_O], axis=1)  # [games, n_o, n_t]
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}").set_index("game_key")
    oh = read_scored_games(ROOT / "results/scores_opp_hd" / f"fold{a.fold}").set_index("game_key")
    own_pmi = posterior_mean_index(hsl.loc[T.game_key, COLS].to_numpy(dtype=float))
    O = oh.loc[T.game_key, COLS].to_numpy(dtype=float)
    oppo = posterior_mean_index(O)
    near = np.abs(oppo[:, None] - IO[None, :]).argmin(axis=1)  # nearest grid r_o per game
    cond = LL[np.arange(len(T)), near]  # [games, n_t]
    wo = np.exp(O[:, IO] - O[:, IO].max(axis=1, keepdims=True)); wo /= wo.sum(axis=1, keepdims=True)  # the opponent's posterior over the grid r_o
    soft = np.log(np.einsum("go,got->gt", wo, np.exp(LL - LL.max(axis=(1, 2), keepdims=True))) + 1e-300)
    marg = np.log(np.exp(LL - LL.max(axis=(1, 2), keepdims=True)).sum(axis=1) + 1e-300)
    T["cpmi"], T["spmi"], T["mpmi"], T["own_pmi"], T["oppo"] = pmi_on_grid(cond), pmi_on_grid(soft), pmi_on_grid(marg), own_pmi, oppo
    T["ll_cond_minus_marg"] = cond.max(axis=1) - marg.max(axis=1)
    g = T.groupby("player_id").agg(y=("rank", "first"), cpmi=("cpmi", "mean"), spmi=("spmi", "mean"), mpmi=("mpmi", "mean"), own=("own_pmi", "mean"), oppo=("oppo", "mean"), n=("game_key", "size"))
    g["y"] = g["y"].map(RANK)
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); m20 = meta[meta["n_games"] == 20].drop_duplicates("player_id").set_index("player_id")
    g["E"] = m20["E"].reindex(g.index); g = g.dropna(subset=["E"]); g["bias"] = g["E"] - g["y"]
    n = len(g); print(f"{len(T)} games, {n} players with an oh-ensemble n=20 question; games/player {g['n'].mean():.1f}", flush=True)
    y = g["y"].to_numpy(dtype=float)
    rep = {"games": int(len(T)), "players": n, "spearman": {}, "partial": {}}
    for k in ("cpmi", "spmi", "mpmi", "own", "oppo", "E"):
        rep["spearman"][k] = spearman(g[k].to_numpy(), y)
    print("Spearman with y: " + "  ".join(f"{k} {v:.3f}" for k, v in rep["spearman"].items()))
    specs = {"cpmi|own": ("cpmi", ["own"]), "cpmi|mpmi": ("cpmi", ["mpmi"]), "cpmi|own,oppo": ("cpmi", ["own", "oppo"]), "cpmi|E": ("cpmi", ["E"]), "cpmi|E,oppo": ("cpmi", ["E", "oppo"]),
             "spmi|E,oppo": ("spmi", ["E", "oppo"]), "mpmi|own": ("mpmi", ["own"]), "mpmi|E,oppo": ("mpmi", ["E", "oppo"]), "own|E,oppo": ("own", ["E", "oppo"])}
    for name, (x, Z) in specs.items():
        xv, Zv = g[x].to_numpy(), [g[z].to_numpy() for z in Z]
        est = partial(xv, y, Zv); ci = boot(lambda idx: partial(xv[idx], y[idx], [z[idx] for z in Zv]), n)
        rep["partial"][name] = {"estimate": est, "ci95": ci}
        print(f"partial {name:14s} {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    r = resid_on(g["cpmi"].to_numpy(), np.column_stack([g["own"], g["oppo"]])); b = g["bias"].to_numpy()
    est = spearman(r, b); ci = boot(lambda idx: spearman(r[idx], b[idx]), n)
    rep["resid_cpmi_vs_bias_oh"] = {"estimate": est, "ci95": ci}
    print(f"resid(cpmi | own, oppo) vs oh-ensemble bias: {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    rep["ll_cond_minus_marg_mean"] = float(T["ll_cond_minus_marg"].mean())
    rep["nearest_ro_counts"] = {R_O[i]: int(c) for i, c in enumerate(np.bincount(near, minlength=len(R_O)))}
    print("nearest r_o counts", rep["nearest_ro_counts"])
    cont = rep["partial"]["cpmi|E,oppo"]
    rep["continue"] = bool(cont["ci95"][0] > 0 and cont["estimate"] >= 0.08)
    print(f"continue condition (partial cpmi | E, oppo: CI lower > 0 and >= 0.08): {'YES' if rep['continue'] else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
