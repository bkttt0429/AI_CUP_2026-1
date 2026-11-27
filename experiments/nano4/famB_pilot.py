"""Research report family B (reports/圍棋等級預測新特徵家族.md): cross-profile disagreement weighting and -log p tail order statistics
from the stored per-move log p (humanv0 18 profiles, L1b 10 heads; target plies), pre-registered:
  step 0  per validation player two random halves of 10 games -> 8 statistics per half (x2 sources): split-half r, W/between,
          adjacent-rank AUC, Spearman with rank, partial Spearman given own/opp pmi; G0 = split-half >= 0.75 and |partial| >= 0.05
  step 1  R + X vs R (hd_check protocol, MLP h32 seeds 0-2, folds 1-4): X = the 8 humanv0 statistics, the 8 L1b statistics, the
          G0-passing subset, and 8 iid N(0, 1) noise columns as the control; conditions (1)(3)(4)
The eight statistics (per question, over the target plies of its games; D_t = std over profiles of ll_t):
  hi_share   share of plies with D_t above the training 75th percentile
  pmi_w      posterior-mean index of sum_t (D_t / mean D) ll_t
  pmi_gap    pmi of the top-quartile-D plies minus pmi of the bottom-quartile-D plies
  argmax_H   entropy (nats) of the argmax-profile histogram over plies
  q10, q25   10th / 25th percentile of ll_{t, r*} under the posterior-mode profile r* of the question
  tail_share share of ll_{t, r*} < -5
  jensen     mean(exp ll_{t, r*}) - exp(mean ll_{t, r*})

  python -I logs/nano4/famB_pilot.py --out results/famB/pilot.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.evidence import auc, posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import _part_moves, read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from pilot_stats import partial_spearman, spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
STATS = ["hi_share", "pmi_w", "pmi_gap", "argmax_H", "q10", "q25", "tail_share", "jensen"]
_DATA = None


def stats8(LL, d75):
    """The eight statistics of a [plies, profiles] log p block."""
    if len(LL) < 4:
        return np.full(8, np.nan)
    D = LL.std(axis=1)
    w = D / max(D.mean(), 1e-9)
    total = LL.sum(axis=0)
    r_star = int(np.argmax(total))
    ll = LL[:, r_star]
    lo, hi = np.quantile(D, 0.25), np.quantile(D, 0.75)
    top, bot = LL[D >= hi], LL[D <= lo]
    hist = np.bincount(LL.argmax(axis=1), minlength=LL.shape[1]) / len(LL)
    hist = hist[hist > 0]
    return np.array([(D > d75).mean(), posterior_mean_index((w[:, None] * LL).sum(axis=0)),
                     (posterior_mean_index(top.sum(axis=0)) - posterior_mean_index(bot.sum(axis=0))) if len(top) and len(bot) else 0.0,
                     float(-(hist * np.log(hist)).sum()), np.quantile(ll, 0.10), np.quantile(ll, 0.25), (ll < -5).mean(),
                     float(np.exp(ll).mean() - np.exp(ll.mean()))])


def _load_fold(f):
    hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
    mh = _part_moves(ts.HSL / f"fold{f}")
    ml = _part_moves(ROOT / "results/l1b" / ts.L1B[f] / f"fold{f}")
    ll18 = {k: np.asarray(v[1], dtype=np.float32) for k, v in mh.items()}
    ll10 = {k: np.asarray(v[1], dtype=np.float32) for k, v in ml.items()}
    return {"hsl": hsl, "l1b": l1b, "opp": opp, "ll18": ll18, "ll10": ll10}


def _question_stats(args):
    """16 statistics (8 humanv0 + 8 L1b) of each question in a chunk: game key lists."""
    chunk, d75_18, d75_10 = args
    out = np.zeros((len(chunk), 16))
    for i, keys in enumerate(chunk):
        out[i, :8] = stats8(np.concatenate([_DATA["ll18"][k] for k in keys]), d75_18)
        out[i, 8:] = stats8(np.concatenate([_DATA["ll10"][k] for k in keys]), d75_10)
    return out


def question_matrix(questions, d75_18, d75_10, workers):
    keys = [list(k) for k in questions["game_keys"]]
    chunks = [keys[i:i + 500] for i in range(0, len(keys), 500)]
    with mp.get_context("fork").Pool(min(workers, max(1, len(chunks)))) as pool:
        parts = pool.map(_question_stats, [(c, d75_18, d75_10) for c in chunks])
    return np.concatenate(parts) if parts else np.zeros((0, 16))


def half_stats(H, cols):
    A, B = H[H.half == 0].set_index("player_id"), H[H.half == 1].set_index("player_id")
    both = A.index.intersection(B.index)
    out = {}
    for c in cols:
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        w, between = 0.5 * np.mean((a[ok] - b[ok]) ** 2), np.var(0.5 * (a[ok] + b[ok]))
        hv = H[c].to_numpy(dtype=float)
        aucs = []
        for k in range(9):
            lo, hi = hv[(H.y == k) & np.isfinite(hv)], hv[(H.y == k + 1) & np.isfinite(hv)]
            u = auc(hi, lo)
            aucs.append(max(u, 1 - u))
        out[c] = {"split_half_r": r, "w_between": float(w / between) if between > 0 else None, "adjacent_auc": float(np.mean(aucs)),
                  "spearman_rank": spearman(H[c], H["y"]), "partial_rho_given_own_opp": partial_spearman(H[c], H["y"], H[["own_pmi", "opp_pmi"]].to_numpy()),
                  "corr_own_pmi": spearman(H[c], H["own_pmi"]), "mean_by_rank": [float(np.nanmean(H.loc[H.y == k, c])) for k in range(10)]}
        out[c]["g0"] = bool(r >= 0.75 and abs(out[c]["partial_rho_given_own_opp"]) >= 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    workers = max(1, min(len(os.sched_getaffinity(0)), 48))
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    with mp.get_context("fork").Pool(min(workers, len(a.folds))) as pool:
        loaded = pool.map(_load_fold, a.folds)
    global _DATA
    data = {}
    _DATA = {"ll18": {}, "ll10": {}}
    for f, d in zip(a.folds, loaded):
        d["hd"] = hd_all[hd_all["game_key"].isin(set(d["hsl"]["game_key"]))]
        d["players"] = l1b_cv.players_from_scores(d["hsl"], folds_df)
        d["val"] = l1b_cv.pilot_questions(d["players"])
        _DATA["ll18"].update(d["ll18"]); _DATA["ll10"].update(d["ll10"])
        data[f] = d
    print(f"folds loaded: {len(_DATA['ll18'])} games ({time.time() - t0:.0f} s); {workers} CPUs", flush=True)
    # disagreement thresholds from all plies (training-side statistic; the same for every fold here, folds differ negligibly)
    D18 = np.concatenate([v.std(axis=1) for v in _DATA["ll18"].values()])
    D10 = np.concatenate([v.std(axis=1) for v in _DATA["ll10"].values()])
    d75_18, d75_10 = float(np.quantile(D18, 0.75)), float(np.quantile(D10, 0.75))
    rep = {"stats": STATS, "d75": {"humanv0": d75_18, "l1b": d75_10}, "disagreement_quantiles": {"humanv0": [float(np.quantile(D18, q)) for q in (0.25, 0.5, 0.75, 0.95)], "l1b": [float(np.quantile(D10, q)) for q in (0.25, 0.5, 0.75, 0.95)]}}
    print("D quantiles (25/50/75/95): humanv0", rep["disagreement_quantiles"]["humanv0"], "l1b", rep["disagreement_quantiles"]["l1b"], flush=True)
    # step 0: halves
    lcols = sum_cols(LEVELS)
    rows, keysets = [], []
    for f in a.folds:
        d = data[f]
        L, O = d["l1b"].set_index("game_key")[lcols], d["opp"].set_index("game_key")[lcols]
        for p in d["players"].itertuples():
            keys = list(p.game_keys)
            if len(keys) < 10:
                continue
            rng = np.random.RandomState((int(p.player_id[:8], 16) + 7) % 2**32)
            order = rng.permutation(len(keys)); k = len(keys) // 2
            for half, idx in ((0, order[:k]), (1, order[k:2 * k])):
                ks = [keys[i] for i in idx]
                rows.append({"player_id": p.player_id, "y": RANK[p.rank], "half": half,
                             "own_pmi": float(posterior_mean_index(L.loc[ks].to_numpy(dtype=float).sum(axis=0))),
                             "opp_pmi": float(posterior_mean_index(O.loc[ks].to_numpy(dtype=float).sum(axis=0)))})
                keysets.append(ks)
    H = pd.DataFrame(rows)
    M = question_matrix(pd.DataFrame({"game_keys": keysets}), d75_18, d75_10, workers)
    cols = [f"h_{s}" for s in STATS] + [f"l_{s}" for s in STATS]
    for j, c in enumerate(cols):
        H[c] = M[:, j]
    rep["step0"] = {"players": int(H["player_id"].nunique()), "stats": half_stats(H, cols)}
    print(f"\n== step 0: {rep['step0']['players']} players, halves of 10 games (G0 = split-half r >= 0.75 and |partial rho| >= 0.05) ({time.time() - t0:.0f} s)")
    print(f"{'column':14s} {'split-half':>10s} {'W/betw':>7s} {'adjAUC':>7s} {'rho':>7s} {'partial':>8s} {'r(own)':>7s} G0   mean by rank D..6D")
    g0 = []
    for c, s in rep["step0"]["stats"].items():
        print(f"{c:14s} {s['split_half_r']:10.3f} {(s['w_between'] if s['w_between'] is not None else float('nan')):7.2f} {s['adjacent_auc']:7.3f} {s['spearman_rank']:7.3f} {s['partial_rho_given_own_opp']:8.3f} {s['corr_own_pmi']:7.2f} {'yes' if s['g0'] else 'no ':3s} " + " ".join(f"{m:.3f}" for m in s["mean_by_rank"]))
        if s["g0"]:
            g0.append(c)
    rep["step0"]["g0_pass"] = g0
    print("G0 pass:", g0 or "none")
    # step 1: R + X vs R
    names = ["base", "B_humanv0_8", "B_l1b_8", "noise8"] + (["B_g0"] if g0 else [])
    scores, per_fold, ys, pids, ns = {k: [] for k in names}, {k: [] for k in names}, [], [], []
    g0_idx = [cols.index(c) for c in g0]
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        B_tr, B_va = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        F_tr, F_va = question_matrix(train_q, d75_18, d75_10, workers), question_matrix(d["val"], d75_18, d75_10, workers)
        F_tr, F_va = np.nan_to_num(F_tr), np.nan_to_num(F_va)
        noise = np.random.RandomState(99 + v)
        variants = {"base": (B_tr, B_va), "B_humanv0_8": (np.hstack([B_tr, F_tr[:, :8]]), np.hstack([B_va, F_va[:, :8]])),
                    "B_l1b_8": (np.hstack([B_tr, F_tr[:, 8:]]), np.hstack([B_va, F_va[:, 8:]])),
                    "noise8": (np.hstack([B_tr, noise.randn(len(B_tr), 8)]), np.hstack([B_va, noise.randn(len(B_va), 8)]))}
        if g0:
            variants["B_g0"] = (np.hstack([B_tr, F_tr[:, g0_idx]]), np.hstack([B_va, F_va[:, g0_idx]]))
        jobs = [(name, Xt, y_tr, Xv, s) for name, (Xt, Xv) in variants.items() for s in a.seeds]
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            probas = pool.map(_fit, jobs)
        for name in variants:
            P = np.mean([p for (nm, *_), p in zip(jobs, probas) if nm == name], axis=0)
            res, sc = evaluate(P, y_va, d["val"])
            scores[name].append(sc); per_fold[name].append(res["balanced_mean_over_n"])
        ys.append(y_va); pids.append(d["val"]["player_id"].to_numpy(dtype=str)); ns.append(n_va)
        print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in variants) + f"  ({time.time() - t0:.0f} s)", flush=True)
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
    S = {k: np.concatenate(scores[k]) for k in names}
    rep["step1"] = {"arms": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": {str(m): balanced_mean_over_n(S[k], y, n, sizes=(m,)) for m in SIZES}} for k in names}, "pairs": {}}
    print("\n== step 1: R + X vs R (MLP h32, seeds averaged)")
    for k in names:
        print(f"{k:12s} {rep['step1']['arms'][k]['score']:.4f}  folds {[round(x, 4) for x in per_fold[k]]}")
    for k in names[1:]:
        dd = paired_diff(S[k], S["base"], y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[k], per_fold["base"])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
        rep["step1"]["pairs"][f"{k} vs base"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{k} vs base: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


def _fit(job):
    name, Xt, y, Xv, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return MLPModel(hidden=32, seed=seed).fit(Xt, y).predict_proba(Xv)


if __name__ == "__main__":
    main()
