"""Result proxies and the rating disequilibrium (2026-11-05): the records carry no RE and no passes; two proxies for the target's win:
(a) the last mover is the target (the loser resigns on their turn), (b) the human SL value at the final position > 0.5 (4I's o_win).
They agree on 79% of fold-1 games.  Which one tracks the label-side bias?  Per player: win rate under each proxy, the win rate expected
from the strength gap (logistic fit of wins on own - opponent pmi over all games), disequilibrium = actual - expected; Spearman with the
oh ensemble's bias (E - y), partial given E and the gap.  Pre-registered rule: |Spearman(diseq, bias)| >= 0.10 with the CI excluding 0 for
either proxy -> the results channel has fuel for a second stage; otherwise closed (4I's -0.0024 stands).
  python -I logs/nano4/result_proxy.py --out results/r_variants/result_proxy.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from oppview_eval import boot, partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    g = pd.read_csv(ROOT / "results/endgame/game_table.csv", dtype={"game_key": str, "player_id": str})
    g = g[g.fold.isin([1, 2, 3, 4])].copy()
    g["win_lm"] = g.last_mover_is_target.astype(float); g["win_v"] = (g.v_0 > 0.5).astype(float); g["gap"] = g.own_pmi - g.opp_pmi
    rep = {"games": int(len(g)), "agree_lm_v": float((g.win_lm == g.win_v).mean()), "decisive_share": float(((g.v_0 - 0.5).abs() > 0.4).mean()),
           "p_win_lm": float(g.win_lm.mean()), "p_win_v": float(g.win_v.mean())}
    print(f"{len(g)} games: P(last mover = target) {rep['p_win_lm']:.3f}, P(v_T > 0.5) {rep['p_win_v']:.3f}, agreement {rep['agree_lm_v']:.3f}, decisive {rep['decisive_share']:.3f}")
    # expected win probability from the strength gap: logistic fit per proxy
    def fit_logit(x, w):
        b = np.zeros(2)
        for _ in range(50):
            z = b[0] + b[1] * x; p = 1 / (1 + np.exp(-z)); W = p * (1 - p) + 1e-9
            X = np.column_stack([np.ones_like(x), x]); H = X.T @ (X * W[:, None]); grad = X.T @ (w - p)
            b = b + np.linalg.solve(H + 1e-6 * np.eye(2), grad)
        return b
    for proxy in ("win_lm", "win_v"):
        b = fit_logit(g.gap.to_numpy(float), g[proxy].to_numpy(float)); g[f"exp_{proxy}"] = 1 / (1 + np.exp(-(b[0] + b[1] * g.gap)))
        rep[f"logit_{proxy}"] = {"intercept": float(b[0]), "slope_per_profile": float(b[1])}
        # calibration of the proxy: win rate by gap decile
        dec = pd.qcut(g.gap, 10, labels=False); rep[f"winrate_by_gap_decile_{proxy}"] = [float(v) for v in g.groupby(dec)[proxy].mean()]
        print(f"{proxy}: logistic slope {b[1]:+.3f}/profile; win rate by own-opp decile {np.round(rep[f'winrate_by_gap_decile_{proxy}'], 2).tolist()}")
    # per player
    rows = []
    for f in (1, 2, 3, 4):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{f}.json"))["val"])
        P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{f}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
        for pid, s in g[g.fold == f].groupby("player_id"):
            if pid not in E.index:
                continue
            rows.append({"player_id": pid, "y": RANK[s["rank"].iat[0]], "E": float(E[pid]), "gap": s.gap.mean(), "n": len(s),
                         **{f"{p}_rate": s[p].mean() for p in ("win_lm", "win_v")}, **{f"diseq_{p}": (s[p] - s[f"exp_{p}"]).mean() for p in ("win_lm", "win_v")}})
    D = pd.DataFrame(rows); D["bias"] = D.E - D.y; n = len(D); y = D.y.to_numpy(float); bias = D.bias.to_numpy()
    rep["players"] = n
    print(f"{n} players")
    for p in ("win_lm", "win_v"):
        for col in (f"{p}_rate", f"diseq_{p}"):
            x = D[col].to_numpy()
            est = spearman(x, bias); ci = boot(lambda idx: spearman(x[idx], bias[idx]), n)
            pb = partial(x, y, [D.E.to_numpy(), D.gap.to_numpy()]); cib = boot(lambda idx: partial(x[idx], y[idx], [D.E.to_numpy()[idx], D.gap.to_numpy()[idx]]), n)
            rep[col] = {"spearman_vs_bias": est, "ci95": ci, "partial_vs_y_given_E_gap": pb, "ci95_partial": cib, "spearman_vs_y": spearman(x, y)}
            print(f"{col:14s} Spearman vs bias {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  partial vs y | E, gap {pb:+.3f} [{cib[0]:+.3f}, {cib[1]:+.3f}]  Spearman vs y {rep[col]['spearman_vs_y']:+.3f}")
    fuel = any(abs(rep[f"diseq_{p}"]["spearman_vs_bias"]) >= 0.10 and (rep[f"diseq_{p}"]["ci95"][0] > 0 or rep[f"diseq_{p}"]["ci95"][1] < 0) for p in ("win_lm", "win_v"))
    rep["fuel"] = bool(fuel); print(f"results-channel fuel for a second stage: {'YES' if fuel else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
