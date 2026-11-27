"""Train vs Public vs Private distributions without test labels (2026-11-07, report only).
1. Game-level covariates from the raw SGFs: moves per game, HD share (HD0 / HD1 / HD>=2), the target colour, first move on a star point.
   Train = all 1,000,000 games (balanced by games: 100,000 per level), also per level.
2. ct9's predictions: the expected level E and the predicted level shares, train validation questions (out-of-fold, n = 5 / 10 / 20) vs
   Public vs Private (n = 5 / 10 / 15 / 20); two-sample KS per n.
3. BBSE (as logs/nano4/m47_bbse.py) for Public and Private separately.
  python -I logs/nano4/dist_compare.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide  # noqa: E402
from m47_bbse import conf, simplex_ls  # noqa: E402

STAR = {"dd", "dp", "pd", "pp"}
MOVE = re.compile(r";[BW]\[([a-s]{0,2})\]")


def game_stats(sgf):
    s = pd.Series(sgf)
    n = s.str.count(r";[BW]\[")
    hd = s.str.extract(r"HD\[(\d+)\]")[0].astype(float).fillna(0)
    first = s.str.extract(r";B\[([a-s]{2})\]")[0]
    return pd.DataFrame({"moves": n, "hd": hd, "first_star": first.isin(STAR).astype(float)})


def ks(a, b):
    a, b = np.sort(a), np.sort(b); grid = np.union1d(a, b)
    Fa, Fb = np.searchsorted(a, grid, side="right") / len(a), np.searchsorted(b, grid, side="right") / len(b)
    D = float(np.abs(Fa - Fb).max()); ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D
    p = float(min(1.0, 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))))
    return D, p


def summ(df, name):
    return {"set": name, "games": len(df), "moves mean": df.moves.mean(), "moves sd": df.moves.std(), "HD0": (df.hd == 0).mean(), "HD1": (df.hd == 1).mean(),
            "HD>=2": (df.hd >= 2).mean(), "target B": (df.color == "B").mean(), "first move star": df.first_star.mean()}


def main():
    rows, per_level = [], []
    tr = []
    for lv in LEVELS:
        t = pd.read_csv(ROOT / f"data/training/train_{lv}.csv", usecols=["color", "sgf_content"])
        g = game_stats(t["sgf_content"].to_numpy()); g["color"] = t["color"].to_numpy(); g["rank"] = lv; tr.append(g)
        per_level.append(summ(g, f"train {lv}"))
    tr = pd.concat(tr, ignore_index=True); rows.append(summ(tr, "train (all, 100k a level)"))
    test = {}
    for part in ("public", "private"):
        q = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{part}.csv")
        sg, co = [], []
        for i in range(1, 21):
            m = q[f"sgf_{i}"].notna()
            sg += q.loc[m, f"sgf_{i}"].tolist(); co += q.loc[m, f"color_{i}"].tolist()
        g = game_stats(np.array(sg)); g["color"] = co; test[part] = g; rows.append(summ(g, part))
    T = pd.DataFrame(rows + per_level).set_index("set")
    pd.set_option("display.width", 200)
    print("== game-level covariates"); print(T.round(3).to_string())
    for a, b in (("public", "private"),):
        D, p = ks(test[a].moves.to_numpy(), test[b].moves.to_numpy()); print(f"moves KS {a} vs {b}: D {D:.3f} p {p:.3f}")
    for part in ("public", "private"):
        D, p = ks(test[part].moves.to_numpy(), tr.moves.sample(200000, random_state=0).to_numpy()); print(f"moves KS {part} vs train: D {D:.3f} p {p:.3g}")
    # 2. predictions
    Y, P, N = [], [], []
    for v in FOLDS:
        y, n, pid, qq = load_fold(v)
        P.append(np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(float) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)); Y.append(y); N.append(n)
    Pv, yv, nv = np.concatenate(P), np.concatenate(Y), np.concatenate(N)
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); cols = [f"p_{lv}" for lv in LEVELS]; Pt = pr[cols].to_numpy(float)
    ng = pd.concat([pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{s}.csv", usecols=["question_id", "num_games"]) for s in ("public", "private")]).set_index("question_id")["num_games"]
    nt = ng.reindex(pr["question_id"]).to_numpy(); pub = pr["question_id"].str.startswith("pub").to_numpy()
    Ev, Et = Pv @ np.arange(10), Pt @ np.arange(10)
    print("\n== ct9 expected level E by n (mean, sd); KS p-values")
    for m in (5, 10, 15, 20):
        e_pub, e_pri = Et[pub & (nt == m)], Et[~pub & (nt == m)]
        line = f"n {m:2d}: public {e_pub.mean():.2f} ({e_pub.std():.2f}, {len(e_pub)})  private {e_pri.mean():.2f} ({e_pri.std():.2f}, {len(e_pri)})  KS pub-priv p {ks(e_pub, e_pri)[1]:.3f}"
        if m in (5, 10, 20):
            ev = Ev[nv == m]
            line += f"  | train-val {ev.mean():.2f} ({ev.std():.2f}, {len(ev)})  KS pub-val p {ks(e_pub, ev)[1]:.3g}  priv-val p {ks(e_pri, ev)[1]:.3g}"
        print(line)
    dv = decide(Pv); dt = decide(Pt)
    S = pd.DataFrame({"train-val (natural)": np.bincount(dv, minlength=10) / len(dv), "public": np.bincount(dt[pub], minlength=10) / pub.sum(),
                      "private": np.bincount(dt[~pub], minlength=10) / (~pub).sum()}, index=LEVELS)
    C = conf(yv, dv); S["expected if uniform"] = C @ np.full(10, 0.1); S["expected if validation prior"] = C @ (np.bincount(yv, minlength=10) / len(yv))
    print("\n== predicted level shares"); print(S.round(3).to_string())
    exp_pub = np.bincount(dt[pub], minlength=10); exp_pri = np.bincount(dt[~pub], minlength=10)
    tot = exp_pub + exp_pri; e1 = tot * pub.sum() / len(dt); e2 = tot * (~pub).sum() / len(dt)
    chi = float(((exp_pub - e1) ** 2 / np.maximum(e1, 1e-9) + (exp_pri - e2) ** 2 / np.maximum(e2, 1e-9)).sum())
    print(f"public vs private predicted shares: chi-square {chi:.1f} on 9 df (5% critical value 16.9)")
    print("\n== BBSE (noisy; adjacent levels are confusable)")
    for name, m in (("public", pub), ("private", ~pub)):
        w = simplex_ls(C, np.bincount(dt[m], minlength=10) / m.sum())
        print(f"{name:8s}: " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, w)))


if __name__ == "__main__":
    main()
