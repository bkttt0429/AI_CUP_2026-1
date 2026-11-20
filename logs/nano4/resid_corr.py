"""Is the persistent per-player error shared across model families (data/label floor) or model-specific? Per-player signed error
(E[rank] - label) on the two disjoint 10-game halves for: the MLP (floor_decomp players.csv), the 12-member C+T ensemble and the
6C / 6T ensembles (stored validation probabilities), then the cross-model correlations of the player mean error and the split-half
reliability of each model's error.

  python -I logs/nano4/resid_corr.py --out results/r_variants/resid_corr.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
ES = ROOT / "results/trunk_seq/es"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mlp = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str})
    rows = []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        q = pd.DataFrame(m)
        sets = {"CT12": [(arm, s) for arm in ("C", "T") for s in range(6)], "C6": [("C", s) for s in range(6)], "T6": [("T", s) for s in range(6)]}
        E = {}
        for k, members in sets.items():
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm, s in members], axis=0)
            E[k] = (P * np.arange(10)).sum(axis=1)
        q["y"] = q["rank"].map(RANK)
        for k in sets:
            q[f"e_{k}"] = E[k] - q["y"]
        h = q[q.n_games == 10].copy()
        h["half"] = h.groupby("player_id").cumcount()  # pilot_questions order: the two disjoint 10-game halves j = 0, 1
        for pid, g in h.groupby("player_id"):
            if len(g) != 2:
                continue
            g = g.sort_values("half")
            rows.append({"player_id": pid, "y": int(g["y"].iloc[0]), **{f"{k}_e0": float(g[f"e_{k}"].iloc[0]) for k in sets}, **{f"{k}_e1": float(g[f"e_{k}"].iloc[1]) for k in sets}})
    ct = pd.DataFrame(rows)
    T = ct.merge(mlp[["player_id", "e0", "e1"]].rename(columns={"e0": "MLP_e0", "e1": "MLP_e1"}), on="player_id")
    models = ["MLP", "CT12", "C6", "T6"]
    rep = {"players": int(len(T)), "split_half_r": {}, "persistent_share": {}, "cross_model_r_mean_error": {}, "cross_model_r_persistent_only": {}}
    for k in models:
        e0, e1 = T[f"{k}_e0"].to_numpy(), T[f"{k}_e1"].to_numpy()
        rep["split_half_r"][k] = float(np.corrcoef(e0, e1)[0, 1])
        e = np.concatenate([e0, e1])
        rep["persistent_share"][k] = float(1 - 0.5 * np.mean((e0 - e1) ** 2) / e.var())
    for i, k1 in enumerate(models):
        for k2 in models[i + 1:]:
            b1, b2 = (T[f"{k1}_e0"] + T[f"{k1}_e1"]) / 2, (T[f"{k2}_e0"] + T[f"{k2}_e1"]) / 2
            rep["cross_model_r_mean_error"][f"{k1}~{k2}"] = float(np.corrcoef(b1, b2)[0, 1])
            # cross-half, cross-model correlation isolates the persistent part (sampling noise of the two halves is independent)
            rep["cross_model_r_persistent_only"][f"{k1}~{k2}"] = float(0.5 * (np.corrcoef(T[f"{k1}_e0"], T[f"{k2}_e1"])[0, 1] + np.corrcoef(T[f"{k1}_e1"], T[f"{k2}_e0"])[0, 1]))
    # how much of MLP's persistent error does CT12 remove, and vice versa (variance of the difference of player biases)
    bm, bc = (T["MLP_e0"] + T["MLP_e1"]) / 2, (T["CT12_e0"] + T["CT12_e1"]) / 2
    rep["bias_var"] = {"MLP": float(bm.var()), "CT12": float(bc.var()), "MLP_minus_CT12": float((bm - bc).var())}
    print(f"{len(T)} players")
    print("split-half r of the signed error:", {k: round(v_, 3) for k, v_ in rep["split_half_r"].items()})
    print("persistent share:", {k: round(v_, 3) for k, v_ in rep["persistent_share"].items()})
    print("cross-model r of the player mean error:", {k: round(v_, 3) for k, v_ in rep["cross_model_r_mean_error"].items()})
    print("cross-model, cross-half r (persistent part only):", {k: round(v_, 3) for k, v_ in rep["cross_model_r_persistent_only"].items()})
    print("player-bias variance:", {k: round(v_, 3) for k, v_ in rep["bias_var"].items()})
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
