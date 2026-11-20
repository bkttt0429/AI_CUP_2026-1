"""Two-stage feasibility (2026-11-05, user: 先找出官方標籤不一致的玩家再分類; 對手等級造成的狀態差異): can the oh ensemble's per-player bias
(E - y on the n = 20 pilot question, folds 1-4) or its magnitude be predicted out of fold from opponent-related observables?
Features per player (20 games): mean / sd / skew of the opponent HSL pmi, mean and sd of (opponent - own) pmi, share of games against an
opponent >= 1 profile stronger / weaker than the player's own mean, the player's own slope of own pmi on opponent pmi (performance vs
opponent level), own pmi sd, HD1 share, W share, the last-mover win proxy rate; controls: own pmi mean and the ensemble's E.
Models: ridge and a small MLP, 5-fold OOF over players. Pre-registered rule: delta OOF R^2 of (controls + opponent features) over
(controls only) >= 0.02 for the signed bias -> a second stage has fuel; otherwise the opponent-level route is closed.
  python -I logs/nano4/bias_pred.py --out results/r_variants/bias_pred.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COLS = [f"sum_{p}" for p in PROFILES]


def player_features(folds):
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    eg = pd.read_csv(ROOT / "results/endgame/game_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")
    rows = []
    for f in folds:
        hsl = read_scored_games(ts.HSL / f"fold{f}"); oh = read_scored_games(ts.OPP_HSL / f"fold{f}").set_index("game_key")
        own = posterior_mean_index(hsl[COLS].to_numpy(float)); opp = posterior_mean_index(oh.loc[hsl.game_key, COLS].to_numpy(float))
        g = pd.DataFrame({"player_id": hsl.player_id.to_numpy(), "rank": hsl["rank"].to_numpy(), "own": own, "opp": opp, "color": hsl.color.to_numpy(),
                          "hd": hsl.game_key.map(hd).fillna(0).to_numpy(), "lm": eg["last_mover_is_target"].reindex(hsl.game_key).to_numpy(), "v0": eg["v_0"].reindex(hsl.game_key).to_numpy()})
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{f}.json"))["val"])
        P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{f}/{a}_s{s}.npz")["proba"].astype(float) for a in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
        for pid, s in g.groupby("player_id"):
            if pid not in E.index or len(s) < 10:
                continue
            d = s.opp - s.own
            slope = np.polyfit(s.opp, s.own, 1)[0] if s.opp.std() > 1e-6 else 0.0
            rows.append({"fold": f, "player_id": pid, "y": RANK[s["rank"].iat[0]], "E": float(E[pid]), "own_mean": s.own.mean(), "own_sd": s.own.std(),
                         "opp_mean": s.opp.mean(), "opp_sd": s.opp.std(), "opp_skew": float(((s.opp - s.opp.mean()) ** 3).mean() / (s.opp.std() ** 3 + 1e-9)),
                         "gap_mean": d.mean(), "gap_sd": d.std(), "share_stronger": float((s.opp > s.own.mean() + 1).mean()), "share_weaker": float((s.opp < s.own.mean() - 1).mean()),
                         "slope_own_on_opp": float(slope), "hd1_share": float((s.hd >= 1).mean()), "w_share": float((s.color == "W").mean()),
                         "win_lm": float(np.nanmean(s.lm)), "win_v": float(np.nanmean(s.v0 > 0.5)), "n": len(s)})
    return pd.DataFrame(rows)


def oof_r2(X, y, folds, kind):
    pred = np.zeros_like(y)
    for k in range(5):
        te = folds == k; tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if kind == "ridge":
            A = np.column_stack([np.ones(tr.sum()), Xtr]); w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[tr])
            pred[te] = np.column_stack([np.ones(te.sum()), Xte]) @ w
        else:
            torch.manual_seed(0)
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
            xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(y[tr], dtype=torch.float32)[:, None]
            for _ in range(300):
                opt.zero_grad(); loss = ((net(xt) - yt) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                pred[te] = net(torch.tensor(Xte, dtype=torch.float32)).numpy()[:, 0]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = player_features((1, 2, 3, 4))
    D["bias"] = D.E - D.y; D["abias"] = D.bias.abs()
    print(f"{len(D)} players; bias sd {D.bias.std():.3f}, mean |bias| {D.abias.mean():.3f}")
    folds = np.random.RandomState(0).randint(0, 5, len(D))
    controls = ["E", "own_mean"]
    oppf = ["opp_mean", "opp_sd", "opp_skew", "gap_mean", "gap_sd", "share_stronger", "share_weaker", "slope_own_on_opp", "own_sd", "hd1_share", "w_share"]
    resf = ["win_lm", "win_v"]
    sets = {"controls": controls, "controls+opp": controls + oppf, "controls+opp+results": controls + oppf + resf, "controls+results": controls + resf}
    rep = {"players": int(len(D)), "bias_sd": float(D.bias.std()), "r2": {}}
    for target in ("bias", "abias"):
        y = D[target].to_numpy(float)
        for name, cols in sets.items():
            X = D[cols].to_numpy(float)
            for kind in ("ridge", "mlp"):
                rep["r2"][f"{target}|{name}|{kind}"] = oof_r2(X, y, folds, kind)
        print(f"{target}: " + "  ".join(f"{name} ridge {rep['r2'][f'{target}|{name}|ridge']:.4f} mlp {rep['r2'][f'{target}|{name}|mlp']:.4f}" for name in sets))
    rep["delta_r2_bias_opp_ridge"] = rep["r2"]["bias|controls+opp|ridge"] - rep["r2"]["bias|controls|ridge"]
    rep["delta_r2_bias_opp_mlp"] = rep["r2"]["bias|controls+opp|mlp"] - rep["r2"]["bias|controls|mlp"]
    rep["delta_r2_bias_results_ridge"] = rep["r2"]["bias|controls+results|ridge"] - rep["r2"]["bias|controls|ridge"]
    rep["uni_spearman_with_bias"] = {c: float(pd.Series(D[c]).rank().corr(D.bias.rank())) for c in oppf + resf}
    print("univariate Spearman with bias:", {k: round(v, 3) for k, v in rep["uni_spearman_with_bias"].items()})
    print(f"delta OOF R^2 (bias): opponent features ridge {rep['delta_r2_bias_opp_ridge']:+.4f} mlp {rep['delta_r2_bias_opp_mlp']:+.4f}; result proxies ridge {rep['delta_r2_bias_results_ridge']:+.4f}  -> two-stage fuel: {'YES' if max(rep['delta_r2_bias_opp_ridge'], rep['delta_r2_bias_opp_mlp']) >= 0.02 else 'no'}")
    D.to_csv(Path(a.out).with_suffix(".players.csv"), index=False)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
