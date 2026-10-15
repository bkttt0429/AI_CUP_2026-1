#!/usr/bin/env python
"""Analyse the Human SL pilot (METHOD_NOTES §2.4) -> results/pilot_humansl.json.

Fold 0 is the evaluation set and nothing is fitted on it. Fold 1 is used only to fit the
monotone thresholds in (e). Sections:
  (a) likelihood curves per class and best-profile counts (players, all 20 games)
  (b) zero-shot hard bins (D = 10-12k ... 6D = 6d), official score by question size
  (c) the same with per-game-normalised sums
  (d) adjacent-class AUC of the posterior-mean profile index (no fitting at all)
  (e) monotone thresholds fitted on fold 1, applied to fold 0
  (f) opponent-rank sensitivity
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.evidence import (apply_thresholds, auc, bootstrap_ci, class_scores_hard_bins, fit_thresholds,  # noqa: E402
                            posterior_mean_index, question_matrix)
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX, expected_score_decision, score  # noqa: E402
from aicup.pilot import pilot_questions, select_players  # noqa: E402

PILOT = ROOT / "results" / "pilot"
OUT = ROOT / "results" / "pilot_humansl.json"
SIZES = (5, 10, 20)


def load_fold(f):
    games = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(PILOT / f"games_fold{f}_shard*.csv")))], ignore_index=True)
    assert games["game_key"].is_unique and not games.isna().any().any(), f"fold {f}: duplicate or missing values"
    players = select_players(pd.read_csv(ROOT / "splits" / f"questions_fold{f}.csv"), f)
    qs = pilot_questions(players)
    want = {k for ks in qs["game_keys"] for k in ks}
    assert want == set(games["game_key"]), f"fold {f}: scored games {len(games)} != selected {len(want)}"
    L_sum, L_norm, n_moves = question_matrix(games, qs, PROFILES)
    y = qs["rank"].map(RANK_TO_IDX).to_numpy()
    return {"games": games, "players": players, "qs": qs, "L_sum": L_sum, "L_norm": L_norm, "n_moves": n_moves, "y": y}


def decide(class_scores):
    z = class_scores - class_scores.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return np.array([expected_score_decision(row) for row in p])


def evaluate(pred, y, qs, mask):
    s = np.array([score(p, t) for p, t in zip(pred[mask], y[mask])])
    conf = np.zeros((10, 10), dtype=int)
    for p, t in zip(pred[mask], y[mask]):
        conf[t, p] += 1
    lo, hi = bootstrap_ci(s, qs["player_id"].to_numpy()[mask], y[mask])
    return {
        "score": float(s.mean()), "ci95": [lo, hi], "n_questions": int(mask.sum()),
        "exact": float((pred[mask] == y[mask]).mean()), "within1": float((abs(pred[mask] - y[mask]) <= 1).mean()),
        "per_class": {LEVELS[k]: float(s[y[mask] == k].mean()) for k in range(10)},
        "confusion_rows_truth_cols_pred": conf.tolist(),
    }


def curves(d):
    g = d["games"]
    cols = [f"sum_{p}" for p in PROFILES]
    per_player = g.groupby(["rank", "player_id"])[cols + ["n_moves"]].sum()
    mean = per_player[cols].to_numpy() / per_player["n_moves"].to_numpy()[:, None]
    rel = mean - mean.max(axis=1, keepdims=True)
    ranks = per_player.index.get_level_values("rank")
    out, best = {}, {}
    for lv in LEVELS:
        m = np.asarray(ranks == lv)
        out[lv] = rel[m].mean(axis=0).round(4).tolist()
        best[lv] = np.bincount(mean[m].argmax(axis=1), minlength=18).tolist()
    return {"profiles": PROFILES, "relative_mean_logp_by_class": out, "best_profile_counts_by_class": best,
            "mean_target_moves_per_game": float(g["n_moves"].mean())}


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def opp_sweep(d0):
    paths = sorted(glob.glob(str(PILOT / "opp_sweep_shard*.csv")))
    if not paths:
        return None
    sw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    main = d0["games"].set_index("game_key")
    res, worst = {}, 0.0
    for lv in LEVELS:
        sub = sw[sw["rank"] == lv]
        if sub.empty:
            continue
        own = sub["own"].iloc[0]
        d_opp, d_own = [], []
        for key, gsub in sub.groupby("game_key"):
            gsub = gsub.set_index("opp").loc[PROFILES]
            n = gsub["n_moves"].iloc[0]
            base = gsub.loc[own, "sum_logp"]
            d_opp.append((gsub["sum_logp"].to_numpy() - base) / n)
            m = main.loc[key]
            own_curve = np.array([m[f"sum_{p}"] for p in PROFILES])
            d_own.append((own_curve - m[f"sum_{own}"]) / n)
            worst = max(worst, abs(base - m[f"sum_{own}"]) / n)
        res[lv] = {"own": own, "games": int(sub["game_key"].nunique()),
                   "delta_per_move_vs_opponent_rank": np.mean(d_opp, axis=0).round(4).tolist(),
                   "delta_per_move_vs_own_rank": np.mean(d_own, axis=0).round(4).tolist()}
    return {"profiles": PROFILES, "by_class": res, "consistency_max_abs_per_move": worst}


def main():
    d0, d1 = load_fold(0), load_fold(1)
    out = {"profiles": PROFILES, "levels": LEVELS,
           "players": {"fold0": int(len(d0["players"])), "fold1": int(len(d1["players"]))},
           "questions_per_size_fold0": {n: int((d0["qs"]["n_games"] == n).sum()) for n in SIZES},
           "curves_fold0": curves(d0)}
    for variant in ("sum", "norm"):
        L = d0[f"L_{variant}"]
        pred = decide(class_scores_hard_bins(L))
        out[f"hard_bin_{variant}"] = {n: evaluate(pred, d0["y"], d0["qs"], d0["qs"]["n_games"].to_numpy() == n) for n in SIZES}
        s = posterior_mean_index(L)
        out[f"auc_adjacent_{variant}"] = {
            n: {f"{LEVELS[k]}|{LEVELS[k + 1]}": auc(s[(d0["y"] == k + 1) & (d0["qs"]["n_games"].to_numpy() == n)],
                                                    s[(d0["y"] == k) & (d0["qs"]["n_games"].to_numpy() == n)])
                for k in range(9)} for n in SIZES}
        out[f"spearman_{variant}"] = {n: spearman(s[d0["qs"]["n_games"].to_numpy() == n], d0["y"][d0["qs"]["n_games"].to_numpy() == n])
                                      for n in SIZES}
    s0, s1 = posterior_mean_index(d0["L_sum"]), posterior_mean_index(d1["L_sum"])
    out["thresholds_fold1_to_fold0"] = {}
    for n in SIZES:
        m0, m1 = d0["qs"]["n_games"].to_numpy() == n, d1["qs"]["n_games"].to_numpy() == n
        cuts = fit_thresholds(s1[m1], d1["y"][m1])
        pred = np.zeros(len(s0), dtype=int)
        pred[m0] = apply_thresholds(s0[m0], cuts)
        r = evaluate(pred, d0["y"], d0["qs"], m0)
        r["cuts_profile_index"] = [None if not np.isfinite(c) else round(float(c), 3) for c in cuts]
        fit_pred = apply_thresholds(s1[m1], cuts)
        r["fold1_fit_score"] = float(np.mean([score(p, t) for p, t in zip(fit_pred, d1["y"][m1])]))
        out["thresholds_fold1_to_fold0"][n] = r
    out["opp_sweep_fold0"] = opp_sweep(d0)
    out["runs"] = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(PILOT / "run_fold*_shard*.json")))]
    OUT.write_text(json.dumps(out, indent=1))

    print(f"players fold0 {out['players']['fold0']} fold1 {out['players']['fold1']}; questions {out['questions_per_size_fold0']}")
    for key, title in (("hard_bin_sum", "zero-shot hard bins, summed"), ("hard_bin_norm", "zero-shot hard bins, per-game normalised"),
                       ("thresholds_fold1_to_fold0", "monotone thresholds fitted on fold 1")):
        print(title + ":")
        for n in SIZES:
            r = out[key][n]
            print(f"  n={n:2d}: score {r['score']:.4f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  exact {r['exact']:.3f}  within1 {r['within1']:.3f}")
    for n in SIZES:
        a = out["auc_adjacent_sum"][n]
        print(f"AUC adjacent n={n}: " + "  ".join(f"{k} {v:.2f}" for k, v in a.items()) + f"  | spearman {out['spearman_sum'][n]:.3f}")
    print("best profile counts (fold 0 players, rows = class):")
    for lv in LEVELS:
        print(f"  {lv:>2}: " + " ".join(f"{c:2d}" for c in out["curves_fold0"]["best_profile_counts_by_class"][lv]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
