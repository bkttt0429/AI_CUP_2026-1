"""Section 4M M65 (2026-11-09; user: 不能用多個模型推出偏差嗎). Can the PATTERN of disagreement among model families on one 10-game half
predict the shared persistent bias on the player's OTHER half?  Same players / halves as resid_corr.py.
Features from half A: E (expected level) of CT12, MLP, C6, T6; disagreement d1 = E_MLP - E_CT12, d2 = E_C6 - E_T6; spread of E across the
12 CT members; CT12 posterior pmax and entropy; MLP pmax.  Target: shared bias on half B = (e_MLP,B + e_CT12,B) / 2.
Nested OOF over the four validation folds (players grouped by fold):  S0 = E_CT12 + its pmax/entropy (what a single model already
knows about its own shrinkage);  S1 = S0 + disagreement features.  Ridge and HistGradientBoosting.  Report OOF R^2 and the increment,
plus the direct Spearman of each disagreement feature with the cross-half bias.  Both half orderings (A->B and B->A) pooled.
  python -I logs/nano4/m65_disagreement.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9; Z = (X - self.mu) / self.sd; self.ym = y.mean()
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1]); self.w = np.linalg.solve(A, Z.T @ (y - self.ym)); return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.ym


class GBT:
    def fit(self, X, y):
        self.m = xgb.train({"max_depth": 3, "eta": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "reg:squarederror", "nthread": 4},
                           xgb.DMatrix(X, label=y), num_boost_round=300); return self

    def predict(self, X):
        return self.m.predict(xgb.DMatrix(X))


ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
ES = ROOT / "results/trunk_seq/es"
mlp = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str})
rows = []
for v in (1, 2, 3, 4):
    q = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
    Pm = {(arm, s): np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in ("C", "T") for s in range(6)}
    Em = {k: (P * np.arange(10)).sum(1) for k, P in Pm.items()}
    P12 = np.mean(list(Pm.values()), axis=0)
    q["E_CT12"] = (P12 * np.arange(10)).sum(1); q["E_C6"] = np.mean([Em[("C", s)] for s in range(6)], axis=0); q["E_T6"] = np.mean([Em[("T", s)] for s in range(6)], axis=0)
    q["spread12"] = np.std(np.stack([Em[k] for k in Pm]), axis=0)
    q["pmax12"] = P12.max(1); q["ent12"] = -(P12 * np.log(P12 + 1e-12)).sum(1)
    q["y"] = q["rank"].map(RANK)
    h = q[q.n_games == 10].copy(); h["half"] = h.groupby("player_id").cumcount()
    for pid, g in h.groupby("player_id"):
        if len(g) != 2:
            continue
        g = g.sort_values("half"); r = {"player_id": pid, "fold": v, "y": int(g.y.iloc[0])}
        for j in (0, 1):
            for c in ("E_CT12", "E_C6", "E_T6", "spread12", "pmax12", "ent12"):
                r[f"{c}_{j}"] = float(g[c].iloc[j])
        rows.append(r)
T = pd.DataFrame(rows).merge(mlp[["player_id", "e0", "e1", "pmax0", "pmax1", "y"]].rename(columns={"y": "y_mlp"}), on="player_id")
assert (T.y == T.y_mlp).all(); print(f"{len(T)} players")
T["E_MLP_0"] = T.e0 + T.y; T["E_MLP_1"] = T.e1 + T.y
# stack both orderings: features from half a, target = shared bias on half b
parts = []
for a, b in ((0, 1), (1, 0)):
    d = pd.DataFrame({"fold": T.fold, "y": T.y, "E_CT12": T[f"E_CT12_{a}"], "pmax12": T[f"pmax12_{a}"], "ent12": T[f"ent12_{a}"],
                      "d_mlp_ct": T[f"E_MLP_{a}"] - T[f"E_CT12_{a}"], "d_c_t": T[f"E_C6_{a}"] - T[f"E_T6_{a}"], "spread12": T[f"spread12_{a}"],
                      "pmax_mlp": T[f"pmax{a}"], "d_pmax": T[f"pmax{a}"] - T[f"pmax12_{a}"],
                      "target": ((T[f"E_MLP_{b}"] - T.y) + (T[f"E_CT12_{b}"] - T.y)) / 2, "target_ct": T[f"E_CT12_{b}"] - T.y})
    parts.append(d)
D = pd.concat(parts, ignore_index=True)
S0 = ["E_CT12", "pmax12", "ent12"]; DIS = ["d_mlp_ct", "d_c_t", "spread12", "pmax_mlp", "d_pmax"]; S1 = S0 + DIS
out = {"players": int(len(T)), "rows": int(len(D))}
print("direct Spearman of each disagreement feature with the cross-half shared bias:")
for c in DIS:
    r = spearmanr(D[c], D.target); out[f"rho_{c}"] = float(r.statistic); print(f"  {c:10s} rho {r.statistic:+.3f} p {r.pvalue:.2g}")
def oof(cols, target, model):
    pred = np.zeros(len(D))
    for v in (1, 2, 3, 4):
        tr, te = D.fold != v, D.fold == v
        m = model(); m.fit(D.loc[tr, cols].to_numpy(), D.loc[tr, target].to_numpy()); pred[te] = m.predict(D.loc[te, cols].to_numpy())
    t = D[target].to_numpy(); return float(1 - np.mean((t - pred) ** 2) / np.var(t)), pred
for target in ("target", "target_ct"):
    for name, mk in [("ridge", lambda: Ridge(alpha=1.0))] + ([("xgb", GBT)] if xgb is not None else []):
        r0, p0 = oof(S0, target, mk); r1, p1 = oof(S1, target, mk); rd, _ = oof(DIS, target, mk)
        # does S1's extra prediction help the sign of the bias?  share of players where sign(p1 - p0) == sign(target - p0), among |p1-p0| > 0.1
        delta = p1 - p0; big = np.abs(delta) > 0.1; agree = float(np.mean(np.sign(delta[big]) == np.sign((D[target].to_numpy() - p0)[big]))) if big.any() else float("nan")
        out[f"{target}_{name}"] = {"R2_S0": r0, "R2_S1": r1, "R2_disagreement_only": rd, "increment": r1 - r0, "n_big_adjust": int(big.sum()), "sign_agree_big": agree}
        print(f"{target:9s} {name:5s}: OOF R^2  S0 (own E + shape) {r0:.4f} | S1 (+ disagreement) {r1:.4f} | disagreement only {rd:.4f} | increment {r1 - r0:+.4f}; "
              f"adjustments > 0.1 level: {big.sum()} rows, sign correct {agree:.2f}")
# the agreement structure itself: when the two families DISAGREE in decision on half a, who is right on half b?
Path(ES / "m65_disagreement.json").write_text(json.dumps(out, indent=1)); print("written")
