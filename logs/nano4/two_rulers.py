"""GPT round 6 (2026-11-05): how much of the cross-source error correlation is one shared stable offset? Folds 1-4 players with 80 scored
games, L1b per-game expected ranks (0-9): own-side estimate S and opponent-side anchor O (opponent estimate + HD rule), 40/40 random halves
(A, B). Per class (1D-4D singled out) and overall: r_S = corr(S_A, S_B), r_O = corr(O_A, O_B), r_x = mean(corr(S_A, O_B), corr(S_B, O_A)),
rho_stable = r_x / sqrt(r_S r_O) (disattenuated), with player bootstrap CIs; the stability of the divergence D = S - O across halves and the
within-class correlation of the mean divergence with the oh ensemble's bias E - y (does a persistent own-vs-opponent disagreement tell who
ct7 over- or under-rates?). Errors are taken relative to the label (S - y, O - y) and class-demeaned for the within-class versions.
  python -I logs/nano4/two_rulers.py --out results/r_variants/two_rulers.json
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
from anchor_kalman import per_game  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def stats(H):
    y, SA, SB, OA, OB = H[:, 0], H[:, 1], H[:, 2], H[:, 3], H[:, 4]
    eSA, eSB, eOA, eOB = SA - y, SB - y, OA - y, OB - y
    rS, rO = corr(eSA, eSB), corr(eOA, eOB); rx = 0.5 * (corr(eSA, eOB) + corr(eSB, eOA))
    DA, DB = SA - OA, SB - OB
    return {"r_S": rS, "r_O": rO, "r_x": rx, "rho_stable": rx / np.sqrt(max(rS * rO, 1e-9)), "r_D": corr(DA, DB)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    G = pd.concat([per_game(f, hd) for f in (1, 2, 3, 4)], ignore_index=True)
    cnt = G.groupby("player_id").size(); G = G[G.player_id.isin(cnt[cnt >= 80].index)]
    Es = []
    for v in (1, 2, 3, 4):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        meta["E"] = P @ np.arange(10); Es.append(meta[meta.n_games == 20].drop_duplicates("player_id")[["player_id", "E"]])
    E = pd.concat(Es).set_index("player_id")["E"]
    rng = np.random.default_rng(0); rows = []
    for pid, s in G.groupby("player_id"):
        s = s.iloc[rng.permutation(len(s))[:80]]; y = int(s.y.iat[0])
        rows.append((y, s.e_own.iloc[:40].mean(), s.e_own.iloc[40:].mean(), s.anchor.iloc[:40].mean(), s.anchor.iloc[40:].mean(), float(E.get(pid, np.nan)), pid))
    H = np.array([r[:6] for r in rows], dtype=float); n = len(H)
    print(f"{n} players with 80 games")
    rep = {"players": n, "by_class": {}}
    def with_ci(idx_mask, label):
        Hm = H[idx_mask]; base = stats(Hm); boots = {k: [] for k in base}
        for _ in range(500):
            b = rng.integers(0, len(Hm), len(Hm)); st = stats(Hm[b])
            for k in st: boots[k].append(st[k])
        out = {k: {"estimate": base[k], "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]} for k in base}
        print(f"{label:10s} n={len(Hm):4d}  r_S {base['r_S']:.3f}  r_O {base['r_O']:.3f}  r_x {base['r_x']:.3f}  rho_stable {base['rho_stable']:.3f} [{out['rho_stable']['ci95'][0]:.3f}, {out['rho_stable']['ci95'][1]:.3f}]  r_D(S-O stable) {base['r_D']:.3f}")
        return out
    # overall (class means removed from every estimate so the between-class structure does not inflate anything)
    Hd = H.copy()
    for c in np.unique(H[:, 0].astype(int)):
        m = H[:, 0] == c
        for j in (1, 2, 3, 4):
            Hd[m, j] -= (H[m, j] - H[m, 0]).mean()
    rep["all_within_class"] = with_ci(np.ones(n, bool), "all(wc)") if False else None
    Hsave = H; H = Hd; rep["all_within_class"] = with_ci(np.ones(n, bool), "all(wc)"); H = Hsave
    for c in range(10):
        m = H[:, 0] == c
        if m.sum() >= 60:
            rep["by_class"][LEVELS[c]] = with_ci(m, LEVELS[c])
    # the persistent divergence vs the ensemble's bias
    ok = np.isfinite(H[:, 5]); Hk = H[ok]; y = Hk[:, 0]; D = 0.5 * ((Hk[:, 1] - Hk[:, 3]) + (Hk[:, 2] - Hk[:, 4])); bias = Hk[:, 5] - y
    def dm(x):
        out = x.copy()
        for c in np.unique(y): out[y == c] -= x[y == c].mean()
        return out
    rD, bD = dm(D), dm(bias); rho = float(pd.Series(rD).rank().corr(pd.Series(bD).rank()))
    bs = [pd.Series(rD[b]).rank().corr(pd.Series(bD[b]).rank()) for b in (rng.integers(0, len(rD), len(rD)) for _ in range(500))]
    rep["divergence_vs_bias_within_class"] = {"players": int(len(rD)), "spearman": rho, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print(f"within-class Spearman(mean own - anchor divergence, oh bias E - y) = {rho:+.3f} [{rep['divergence_vs_bias_within_class']['ci95'][0]:+.3f}, {rep['divergence_vs_bias_within_class']['ci95'][1]:+.3f}] over {len(rD)} players")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
