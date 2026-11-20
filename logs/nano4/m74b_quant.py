"""Section 4M M74 part B (2026-11-10): quant-style engineered features of the per-ply game state (results/ai_state/base, strong net, no
search), per question, both sides, on all validation players (folds 1-4):
 (a) conditional attribution: mean slog point loss in 3 phases (ply < 60, 60-150, > 150) x 5 bands of the mover's win probability
     (< .1, .1-.35, .35-.65, .65-.9, > .9), shrunk to the question mean (pseudo-count 5) and centred on it; the 5 band shares; the mean
 (b) performance statistics of the per-move win-probability change r = -(win loss): mean, sd, mean / sd, downside sd; mean max drawdown
     of the side's win-probability path within a game; time under water (drawdown > 0.1); CVaR90 of the point loss; skewness of the loss
 (c) dynamics: AR(1) correlation of consecutive own losses; correlation of the own loss with the opponent's previous loss; excess loss
     over the next 3 own moves after a blunder (> 3 points)
Evaluation: (1) player level (n = 20): partial correlation of the cross-fitted ridge composite with the label given the ensemble's
expected level E; (2) question level: multinomial logistic stack on [ensemble log-probabilities, log n, features] vs the same stack
without the features, cross-fitted over the validation folds, M37 rule on seeds 0-2 and 6-8.
  python -I logs/nano4/m74b_quant.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.scoring import completed_parts, part_paths  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

VEC, AI = ROOT / "results/trunk_vec", ROOT / "results/ai_state/base"
STD = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh")
NS = 30 + 4 + 2 + 1 + 4 + 6 + 6 + 2  # sufficient statistics per side
slog = lambda x: np.sign(x) * np.log1p(np.abs(x))  # noqa: E731


def corr_sums(a, b):
    return [a.sum(), b.sum(), (a * b).sum(), (a * a).sum(), (b * b).sum(), float(len(a))] if len(a) else [0.0] * 6


def game_stats(win, lead, mover, target):
    """[2, NS]: sufficient statistics of one game for the target side (row 0) and the opponent (row 1)."""
    n = len(win); same = np.zeros(n, bool); same[:-1] = mover[1:] == mover[:-1]; has = np.zeros(n, bool); has[:-1] = True
    ln, wn = np.roll(lead, -1), np.roll(win, -1)
    pl = np.where(has, lead - np.where(same, ln, -ln), 0.0); wl = np.where(has, win - np.where(same, wn, 1.0 - wn), 0.0)
    plc = np.clip(pl, -5, 40); x = slog(plc); t = np.arange(n)
    cell = np.where(t < 60, 0, np.where(t < 150, 1, 2)) * 5 + np.searchsorted([0.1, 0.35, 0.65, 0.9], win)
    out = np.zeros((2, NS))
    for k, side in enumerate((target, ~target)):
        idx = np.flatnonzero(side & has)
        if len(idx) < 4:
            continue
        o = out[k]
        o[:15] = np.bincount(cell[idx], weights=x[idx], minlength=15); o[15:30] = np.bincount(cell[idx], minlength=15)
        r = -wl[idx]; o[30:34] = [len(idx), r.sum(), (r * r).sum(), (np.minimum(r, 0) ** 2).sum()]
        w = np.where(side, win, 1.0 - win); dd = np.maximum.accumulate(w) - w; o[34:36] = [dd.max(), (dd > 0.1).mean()]; o[36] = 1.0
        p = plc[idx]; top = np.sort(p)[-max(1, len(p) // 10):]; o[37] = top.mean(); o[38:41] = [p.sum(), (p * p).sum(), (p ** 3).sum()]
        xs = x[idx]; o[41:47] = corr_sums(xs[:-1], xs[1:])
        prev = idx[idx > 0]; prev = prev[~side[prev - 1]]; o[47:53] = corr_sums(x[prev - 1], x[prev])
        bl = np.flatnonzero(p > 3)
        bl = bl[bl + 3 < len(xs)]
        if len(bl):
            o[53] = (np.stack([xs[bl + 1], xs[bl + 2], xs[bl + 3]]).mean(0) - xs.mean()).sum(); o[54] = len(bl)
    return out


def fold_stats(f):
    res = {}
    for s, k in completed_parts(VEC / f"fold{f}"):
        csv_path, npz_path = part_paths(VEC / f"fold{f}", s, k)
        keys = pd.read_csv(csv_path, usecols=["game_key"], dtype=str)["game_key"].tolist()
        with np.load(npz_path) as z:
            mover, target, row = z["mover"], z["target"], z["game_row"]
        ai = np.load(AI / f"fold{f}" / f"ai_s{s}_p{k:03d}.npy").astype(np.float32)
        b = np.flatnonzero(np.diff(row)) + 1; st = np.concatenate([[0], b]); en = np.concatenate([b, [len(row)]])
        for g, key in enumerate(keys):
            a, e = st[g], en[g]; res[key] = game_stats(ai[a:e, 4], ai[a:e, 5], mover[a:e], target[a:e])
    return res


def corr_of(c):
    sa, sb, sab, saa, sbb, m = c.T; m = np.maximum(m, 1)
    cov = sab / m - sa * sb / m ** 2; va = saa / m - (sa / m) ** 2; vb = sbb / m - (sb / m) ** 2
    return cov / np.sqrt(np.maximum(va * vb, 1e-12))


def features(S):
    """[Q, 2, NS] summed statistics -> [Q, 64] and the column names."""
    cols, names = [], []
    for k, side in enumerate(("t", "o")):
        s = S[:, k]; tot_n = np.maximum(s[:, 15:30].sum(1, keepdims=True), 1); m_all = s[:, :15].sum(1, keepdims=True) / tot_n
        cells = (s[:, :15] + 5 * m_all) / (s[:, 15:30] + 5) - m_all
        share = s[:, 15:30].reshape(-1, 3, 5).sum(1) / tot_n
        n = np.maximum(s[:, 30], 1); mr = s[:, 31] / n; sdr = np.sqrt(np.maximum(s[:, 32] / n - mr ** 2, 1e-12)); dsd = np.sqrt(s[:, 33] / n)
        g = np.maximum(s[:, 36], 1); mdd, tuw, cvar = s[:, 34] / g, s[:, 35] / g, s[:, 37] / g
        mp_ = s[:, 38] / n; vp = np.maximum(s[:, 39] / n - mp_ ** 2, 1e-12); skew = (s[:, 40] / n - 3 * mp_ * vp - mp_ ** 3) / vp ** 1.5
        ar, cross = corr_of(s[:, 41:47]), corr_of(s[:, 47:53]); post = s[:, 53] / np.maximum(s[:, 54], 1)
        cols += [cells, m_all, share, np.stack([mr, sdr, mr / sdr, dsd, mdd, tuw, cvar, skew, ar, cross, post], axis=1)]
        names += [f"{side}_cell{p}{b}" for p in range(3) for b in range(5)] + [f"{side}_mean"] + [f"{side}_share{b}" for b in range(5)] + [f"{side}_{x}" for x in ("mr", "sdr", "sharpe", "dsd", "mdd", "tuw", "cvar", "skew", "ar1", "cross", "post")]
    return np.nan_to_num(np.hstack(cols)), names


def stack(Xtr, ytr, Xte, n_logp, l2=1e-3, iters=300):
    """Class-balanced multinomial logistic regression, the log-probability block initialised at the identity."""
    Xtr, Xte = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Xte, dtype=torch.float64); y = torch.tensor(ytr)
    W = torch.zeros(Xtr.shape[1], 10, dtype=torch.float64); W[:n_logp] = torch.eye(10, dtype=torch.float64)[:n_logp]; W.requires_grad_(True); b = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    cw = (1.0 / torch.bincount(y, minlength=10).double())[y]; cw = cw / cw.sum()
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = (cw * torch.nn.functional.cross_entropy(Xtr @ W + b, y, reduction="none")).sum() + l2 * (W[n_logp:] ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(Xte @ W + b, dim=1).numpy()


def main():
    with mp.get_context("fork").Pool(4) as pool:
        per = {}
        for d in pool.map(fold_stats, FOLDS):
            per.update(d)
    print(f"{len(per)} games with state statistics", flush=True); rep = {}
    data = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v); S = np.stack([np.sum([per[k] for k in gk if k in per], axis=0) if any(k in per for k in gk) else np.zeros((2, NS)) for gk in q["game_keys"]])
        F, names = features(S); data[v] = (y, n, pid, q, F)
    for block, sel in (("all", lambda c: True), ("a attribution", lambda c: "cell" in c or "share" in c or c.endswith("_mean")), ("b performance", lambda c: c.split("_")[1] in ("mr", "sdr", "sharpe", "dsd", "mdd", "tuw", "cvar", "skew")), ("c dynamics", lambda c: c.split("_")[1] in ("ar1", "cross", "post"))):
        use = np.array([sel(c) for c in names]); comp, ys, Es = [], [], []
        for v in FOLDS:
            y, n, pid, q, F = data[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in (0, 1, 2)], axis=0); E = P @ np.arange(10)
            tr = [data[u] for u in FOLDS if u != v]; Xtr = np.vstack([t[4][t[1] == 20][:, use] for t in tr]); ytr = np.concatenate([t[0][t[1] == 20] for t in tr])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = (Xtr - mu) / sd; w = np.linalg.solve(Z.T @ Z + 10 * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
            m = n == 20; comp.append(((F[m][:, use] - mu) / sd) @ w); ys.append(y[m]); Es.append(E[m])
        c, yy, ee = np.concatenate(comp), np.concatenate(ys).astype(float), np.concatenate(Es)
        res = lambda a: a - np.polyval(np.polyfit(ee, a, 1), ee)  # noqa: E731
        rng = np.random.default_rng(0); rc, ry = res(c), res(yy); r = float(np.corrcoef(rc, ry)[0, 1])
        bs = [np.corrcoef(rc[i], ry[i])[0, 1] for i in (rng.integers(0, len(c), len(c)) for _ in range(1000))]
        rep[f"partial_{block}"] = {"r": r, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "raw_spearman_with_y": float(pd.Series(c).corr(pd.Series(yy), method="spearman"))}
        print(f"(1) {block:14s} composite: partial r with the label given E {r:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] ({len(c)} players; composite vs label Spearman {rep[f'partial_{block}']['raw_spearman_with_y']:.3f})", flush=True)
    ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X0, X1, Y = {}, {}, {}
        for v in FOLDS:
            y, n, pid, q, F = data[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in seeds], axis=0)
            X0[v] = np.hstack([np.log(np.clip(P, 1e-6, 1)), np.log(n)[:, None] - 2.3]); X1[v] = F; Y[v] = y
        sn, so, sb, pf, ys, ns, pids = [], [], [], [], [], [], []
        for v in FOLDS:
            y, n, pid, q, F = data[v]; tr = [u for u in FOLDS if u != v]
            mu, sd = np.vstack([X1[u] for u in tr]).mean(0), np.vstack([X1[u] for u in tr]).std(0) + 1e-9
            A0, B0 = np.vstack([X0[u] for u in tr]), X0[v]; A1, B1 = np.hstack([A0, (np.vstack([X1[u] for u in tr]) - mu) / sd]), np.hstack([B0, (X1[v] - mu) / sd]); ytr = np.concatenate([Y[u] for u in tr])
            p0, p1 = stack(A0, ytr, B0, 10), stack(A1, ytr, B1, 10)
            s0, s1 = evaluate(p0, y, q, sizes=SIZES)[1], evaluate(p1, y, q, sizes=SIZES)[1]; sraw = evaluate(np.exp(B0[:, :10]), y, q, sizes=SIZES)[1]
            sn.append(s1); so.append(s0); sb.append(sraw); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(s1, y, n, sizes=SIZES) - balanced_mean_over_n(s0, y, n, sizes=SIZES))
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
        d = paired_diff(np.concatenate(sn), np.concatenate(so), y, pid, n, sizes=SIZES, n=5000); d0 = paired_diff(np.concatenate(so), np.concatenate(sb), y, pid, n, sizes=SIZES, n=2000)
        good = (d["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (d["diff"] >= 0.0025); ok.append(good)
        rep[f"stack_seeds_{seeds[0]}"] = {**d, "per_fold": pf, "pass": bool(good), "stack_without_features_vs_ensemble": d0["diff"]}
        print(f"(2) seeds {seeds}: stack with the features vs the same stack without {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if good else 'no'}   (the plain stack vs the raw ensemble {d0['diff']:+.4f})", flush=True)
    rep["adopt"] = bool(all(ok)); print(f"PRE-REGISTERED: adopt the engineered state features as a stacking stage: {'YES' if all(ok) else 'NO'}")
    Path(ROOT / "results/r_variants/m74b_quant.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
