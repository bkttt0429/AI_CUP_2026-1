"""Probe: how much rank information the full trunk means carry vs their 64-dim random projection (and 16..256).  Player level
(mean of the 20 game means -> one vector -> multinomial logistic regression, 5-fold OOF over players, expected-score decision,
class-balanced official score) and game level (per-game logistic regression, posterior mean over the player's 20 games).
Paired bootstrap of full - 64 over players.  CPU only.
  python -I logs/nano4/probe_dim.py --root results/probe_dim/fold1 --out results/r_variants/probe_dim.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
import trunk_extract as te  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
DIMS = (16, 32, 64, 128, 256, 0)  # 0 = full


def project(Gm, Lm, d, seed):
    C = Gm.shape[1]
    if d == 0:
        return np.concatenate([Gm, Lm], axis=1)
    if d == 64 and seed == te.PROJ_SEED:
        pg, pl = te.projections(C, torch.device("cpu")); pg, pl = pg.numpy(), pl.numpy()
    else:
        gen = torch.Generator().manual_seed(seed + d); pg = (torch.randn(C, d, generator=gen) / C ** 0.5).numpy(); pl = (torch.randn(C, d, generator=gen) / C ** 0.5).numpy()
    return np.concatenate([Gm @ pg, Lm @ pl], axis=1)


def fit_logreg(X, y, Xte, wd):
    X, y, Xte = torch.as_tensor(X), torch.as_tensor(y), torch.as_tensor(Xte)
    W = torch.zeros(X.shape[1], 10, requires_grad=True); b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(X @ W + b, y) + wd * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(Xte @ W + b, dim=1).numpy()


def oof_probs(X, y, groups, folds=5, seed=0):
    """OOF posteriors with wd chosen on an inner split of each training part (groups = player index)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); rng.shuffle(ug); assign = {g: i % folds for i, g in enumerate(ug)}
    fa = np.array([assign[g] for g in groups]); P = np.zeros((len(y), 10))
    for k in range(folds):
        tr, te_ = fa != k, fa == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
        inner = (np.arange(len(y)) % 5 == 0) & tr; fit = tr & ~inner
        best = None
        for wd in (1e-4, 1e-3, 1e-2, 1e-1):
            pr = fit_logreg(Xs[fit].astype(np.float32), y[fit], Xs[inner].astype(np.float32), wd)
            nll = -np.log(pr[np.arange(inner.sum()), y[inner]] + 1e-9).mean()
            if best is None or nll < best[0]:
                best = (nll, wd)
        P[te_] = fit_logreg(Xs[tr].astype(np.float32), y[tr], Xs[te_].astype(np.float32), best[1])
    return P


def official(P, y):
    pred = np.array([expected_score_decision(p) for p in P]); s = np.array([score(int(a), int(b)) for a, b in zip(pred, y)])
    return s, pred


def balanced(s, y):
    return float(np.mean([s[y == k].mean() for k in range(10) if (y == k).any()]))


def paired_boot(sa, sb, y, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(balanced(sa[i], y[i]) - balanced(sb[i], y[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    torch.set_num_threads(12)
    parts = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(str(Path(a.root) / "shard*.npz")))]
    key = np.concatenate([p["game_key"] for p in parts]); pid = np.concatenate([p["player_id"] for p in parts]); rank = np.concatenate([p["rank"] for p in parts])
    Gm = np.concatenate([p["g_mean"] for p in parts]); Lm = np.concatenate([p["l_mean"] for p in parts]); n_own = np.concatenate([p["n_own"] for p in parts])
    ok = n_own > 0; key, pid, rank, Gm, Lm = key[ok], pid[ok], rank[ok], Gm[ok], Lm[ok]
    y_game = np.array([RANK[r] for r in rank]); up, gidx = np.unique(pid, return_inverse=True)
    y_pl = np.zeros(len(up), int); y_pl[gidx] = y_game
    C = Gm.shape[1]
    # variance spectrum of the full means: how many PCs for 95% (is the trunk mean low-rank?)
    Z = np.concatenate([Gm, Lm], axis=1); Z = Z - Z.mean(0); ev = np.linalg.svd(Z, compute_uv=False) ** 2; cum = np.cumsum(ev) / ev.sum()
    rep = {"games": int(len(key)), "players": int(len(up)), "C": int(C), "pcs_for_90_95_99": [int((cum < q).sum() + 1) for q in (0.9, 0.95, 0.99)], "player_level": {}, "game_level": {}}
    scores_pl, scores_gm = {}, {}
    for d in DIMS:
        X = project(Gm, Lm, d, te.PROJ_SEED)
        # player level: mean over games
        Xp = np.zeros((len(up), X.shape[1])); np.add.at(Xp, gidx, X); Xp /= np.bincount(gidx)[:, None]
        P = oof_probs(Xp, y_pl, np.arange(len(up))); s, pred = official(P, y_pl); scores_pl[d] = s
        rep["player_level"][str(d or C)] = {"balanced": balanced(s, y_pl), "exact": float((pred == y_pl).mean()), "within1": float((np.abs(pred - y_pl) <= 1).mean())}
        # game level, then posterior mean over the player's games
        Pg = oof_probs(X, y_game, gidx); pg_pred = Pg.argmax(1)
        Pp = np.zeros((len(up), 10)); np.add.at(Pp, gidx, Pg); Pp /= np.bincount(gidx)[:, None]
        s2, pred2 = official(Pp, y_pl); scores_gm[d] = s2
        rep["game_level"][str(d or C)] = {"game_exact": float((pg_pred == y_game).mean()), "game_within1": float((np.abs(pg_pred - y_game) <= 1).mean()),
                                         "agg20_balanced": balanced(s2, y_pl), "agg20_exact": float((pred2 == y_pl).mean())}
        print(d or C, json.dumps(rep["player_level"][str(d or C)]), json.dumps(rep["game_level"][str(d or C)]), flush=True)
    rep["full_minus_64"] = {"player_level": {"diff": balanced(scores_pl[0], y_pl) - balanced(scores_pl[64], y_pl), "ci95": paired_boot(scores_pl[0], scores_pl[64], y_pl)},
                            "game_agg20": {"diff": balanced(scores_gm[0], y_pl) - balanced(scores_gm[64], y_pl), "ci95": paired_boot(scores_gm[0], scores_gm[64], y_pl)}}
    rep["d256_minus_64"] = {"player_level": {"diff": balanced(scores_pl[256], y_pl) - balanced(scores_pl[64], y_pl), "ci95": paired_boot(scores_pl[256], scores_pl[64], y_pl)}}
    print(json.dumps(rep, indent=1)); Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
