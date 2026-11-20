"""Section 4M M10 diagnostic (user: 研究一下人類指紋／風格一致性 5 欄): are the five style-consistency columns signal, noise, or
redundant? (1) Per validation player two random halves of its scored games -> the columns on each half: monotonicity by rank,
adjacent-rank AUC, split-half reliability, partial Spearman with the rank given own/opp pmi (the G0 frame of section 4L). (2) The
MLP penalty of the five columns against five iid N(0, 1) columns (same questions, folds 1-4, seeds averaged). (3) Their
distribution on the official test questions (same colour within a question) against the validation questions by size and by
colour composition.

  python -I logs/nano4/fp_diag.py --out results/r_variants/fp_diag.json
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
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from pilot_stats import partial_spearman, spearman  # noqa: E402
from r_variants import canonical, opening_keys  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
FP = ["pmi_std", "curve_pairdist", "first1_agree", "first3_agree", "nmoves_cv"]


class Fast:
    """Per-game arrays of one table set; fp(keys) gives the five columns of any game list."""

    def __init__(self, hsl_curves, pmi, opening):
        self.curves, self.pmi, self.opening = hsl_curves, pmi, opening

    @staticmethod
    def from_tables(hsl, l1b, opening):
        h = hsl.set_index("game_key")
        hcols = sum_cols(l1b_cv.PROFILES)
        C = h[hcols].to_numpy(dtype=float) / np.maximum(h["n_moves"].to_numpy(dtype=float), 1.0)[:, None]
        C = C - C.max(axis=1, keepdims=True)
        li = l1b.set_index("game_key")
        P = posterior_mean_index(li[sum_cols(LEVELS)].to_numpy(dtype=float))
        return Fast(dict(zip(h.index, C)), dict(zip(li.index, P)), opening)

    @staticmethod
    def agree(labels):
        _, counts = np.unique(np.array([repr(x) for x in labels]), return_counts=True)  # one string per game (a tuple would be flattened)
        k = len(labels)
        return float((counts * (counts - 1) / 2).sum() / (k * (k - 1) / 2)) if k > 1 else 0.0

    def fp(self, keys):
        keys = list(keys)
        C = np.stack([self.curves[k] for k in keys])
        D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(keys), 1)
        o = [self.opening[k] for k in keys]
        nm = np.array([x[2] for x in o], dtype=float)
        return np.array([np.std([self.pmi[k] for k in keys]), D[iu].mean() if len(iu[0]) else 0.0,
                         self.agree([x[0] for x in o]), self.agree([x[1] for x in o]), nm.std() / max(nm.mean(), 1.0)])

    def matrix(self, questions):
        return np.stack([self.fp(ks) for ks in questions["game_keys"]]) if len(questions) else np.zeros((0, 5))


def halves(data, folds, seed=7):
    lcols = sum_cols(LEVELS)
    rows = []
    for f in folds:
        d, fast = data[f], data[f]["fast"]
        L, O = d["l1b"].set_index("game_key")[lcols], d["opp"].set_index("game_key")[lcols]
        for p in d["players"].itertuples():
            keys = list(p.game_keys)
            if len(keys) < 10:
                continue
            rng = np.random.RandomState((int(p.player_id[:8], 16) + seed) % 2**32)
            order = rng.permutation(len(keys))
            k = len(keys) // 2
            for half, idx in ((0, order[:k]), (1, order[k:2 * k])):
                ks = [keys[i] for i in idx]
                rows.append({"fold": f, "player_id": p.player_id, "rank": p.rank, "y": RANK[p.rank], "half": half, "k": k,
                             "own_pmi": float(posterior_mean_index(L.loc[ks].to_numpy(dtype=float).sum(axis=0))),
                             "opp_pmi": float(posterior_mean_index(O.loc[ks].to_numpy(dtype=float).sum(axis=0))), **dict(zip(FP, fast.fp(ks)))})
    return pd.DataFrame(rows)


def half_stats(H):
    A, B = H[H.half == 0].set_index("player_id"), H[H.half == 1].set_index("player_id")
    both = A.index.intersection(B.index)
    out = {}
    for c in FP + ["own_pmi", "opp_pmi"]:
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1])
        w, between = 0.5 * np.mean((a - b) ** 2), np.var(0.5 * (a + b))
        hv = H[c].to_numpy(dtype=float)
        aucs = [max(auc(hv[H.y == k + 1], hv[H.y == k]), 1 - auc(hv[H.y == k + 1], hv[H.y == k])) for k in range(9)]
        out[c] = {"split_half_r": r, "w_between": float(w / between) if between > 0 else None, "adjacent_auc": float(np.mean(aucs)),
                  "spearman_rank": spearman(H[c], H["y"]), "partial_rho_given_own_opp": partial_spearman(H[c], H["y"], H[["own_pmi", "opp_pmi"]].to_numpy()),
                  "corr_own_pmi": spearman(H[c], H["own_pmi"]), "corr_opp_pmi": spearman(H[c], H["opp_pmi"]),
                  "mean_by_rank": [float(H.loc[H.y == k, c].mean()) for k in range(10)]}
        out[c]["g0"] = bool(r >= 0.75 and abs(out[c]["partial_rho_given_own_opp"]) >= 0.05)
    return out


_DATA = None


def _load_fold(f):
    hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
    return {"hsl": hsl, "l1b": l1b, "opp": opp, "opening": opening_keys(hsl["game_key"])}


def _halves_of_fold(f):
    return halves(_DATA, [f])


def _fit(job):
    name, Xt, y, Xv, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return MLPModel(hidden=32, seed=seed).fit(Xt, y).predict_proba(Xv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shards", default=str(ROOT / "submission_rank.csv"))
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    workers = max(1, min(len(os.sched_getaffinity(0)), 48))
    print(f"{workers} CPUs", flush=True)
    with mp.get_context("fork").Pool(min(workers, len(a.folds))) as pool:
        loaded = pool.map(_load_fold, a.folds)
    data = {}
    for f, d in zip(a.folds, loaded):
        d["hd"] = hd_all[hd_all["game_key"].isin(set(d["hsl"]["game_key"]))]
        d["players"] = l1b_cv.players_from_scores(d["hsl"], folds_df)
        d["val"] = l1b_cv.pilot_questions(d["players"])
        d["fast"] = Fast.from_tables(d["hsl"], d["l1b"], d["opening"])
        d["colour"] = dict(zip(d["hsl"]["game_key"], d["hsl"]["color"]))
        data[f] = d
    print(f"folds loaded ({time.time() - t0:.0f} s)", flush=True)
    rep = {"columns": FP}
    # (1) halves, one process per fold
    global _DATA
    _DATA = data
    with mp.get_context("fork").Pool(min(workers, len(a.folds))) as pool:
        H = pd.concat(pool.map(_halves_of_fold, a.folds), ignore_index=True)
    rep["halves"] = {"players": int(H["player_id"].nunique()), "games_per_half": int(H["k"].median()), "stats": half_stats(H)}
    print("\n== (1) per-player halves: reliability and signal (G0 = split-half r >= 0.75 and |partial rho| >= 0.05)")
    print(f"{'column':15s} {'split-half r':>12s} {'W/between':>10s} {'adj AUC':>8s} {'rho(rank)':>10s} {'partial':>8s} {'r(own)':>7s} {'r(opp)':>7s}  G0   mean by rank D..6D")
    for c, s in rep["halves"]["stats"].items():
        print(f"{c:15s} {s['split_half_r']:12.3f} {s['w_between'] if s['w_between'] is not None else float('nan'):10.2f} {s['adjacent_auc']:8.3f} {s['spearman_rank']:10.3f} "
              f"{s['partial_rho_given_own_opp']:8.3f} {s['corr_own_pmi']:7.2f} {s['corr_opp_pmi']:7.2f}  {'yes' if s['g0'] else 'no ':3s} " + " ".join(f"{m:.3f}" for m in s["mean_by_rank"]))
    # (2) MLP: base, +fp, +rand5
    scores, per_fold, ys, pids, ns = {k: [] for k in ("base", "fp", "rand5")}, {k: [] for k in ("base", "fp", "rand5")}, [], [], []
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        B_tr, B_va = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        # the training questions mix folds: pool the training folds' per-game arrays
        pool = Fast({k: v_ for f in train for k, v_ in data[f]["fast"].curves.items()}, {k: v_ for f in train for k, v_ in data[f]["fast"].pmi.items()},
                    {k: v_ for f in train for k, v_ in data[f]["opening"].items()})
        F_tr, F_va = pool.matrix(train_q), d["fast"].matrix(d["val"])
        noise = np.random.RandomState(99 + v)
        variants = {"base": (B_tr, B_va), "fp": (np.hstack([B_tr, F_tr]), np.hstack([B_va, F_va])),
                    "rand5": (np.hstack([B_tr, noise.randn(len(B_tr), 5)]), np.hstack([B_va, noise.randn(len(B_va), 5)]))}
        jobs = [(name, Xt, y_tr, Xv, s) for name, (Xt, Xv) in variants.items() for s in a.seeds]
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            probas = pool.map(_fit, jobs)
        for name in variants:
            P = np.mean([p for (nm, *_), p in zip(jobs, probas) if nm == name], axis=0)
            res, sc = evaluate(P, y_va, d["val"])
            scores[name].append(sc)
            per_fold[name].append(res["balanced_mean_over_n"])
        ys.append(y_va); pids.append(d["val"]["player_id"].to_numpy(dtype=str)); ns.append(n_va)
        print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in scores) + f"  ({time.time() - t0:.0f} s)", flush=True)
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
    S = {k: np.concatenate(scores[k]) for k in scores}
    rep["mlp"] = {"arms": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k]} for k in scores}, "pairs": {}}
    print("\n== (2) MLP h32 (seeds averaged): the five columns against five random columns")
    for x in ("fp", "rand5"):
        dd = paired_diff(S[x], S["base"], y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold["base"])]
        rep["mlp"]["pairs"][f"{x} vs base"] = {**dd, "per_fold_diff": pf}
        print(f"{x} vs base: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()))
    dd = paired_diff(S["fp"], S["rand5"], y, pid, n, n=a.n_boot)
    rep["mlp"]["pairs"]["fp vs rand5"] = dd
    print(f"fp vs rand5: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]")
    # (3) test questions vs validation questions, by size and by colour composition
    val_rows = []
    for f in a.folds:
        d = data[f]
        F = d["fast"].matrix(d["val"])
        same = np.array([len({d["colour"][k] for k in ks}) == 1 for ks in d["val"]["game_keys"]])
        for i in range(len(F)):
            val_rows.append({"n": int(d["val"]["n_games"].iloc[i]), "same_colour": bool(same[i]), **dict(zip(FP, F[i]))})
    V = pd.DataFrame(val_rows)
    inputs = [str(ROOT / "data/tests/rank_prediction_test_public.csv"), str(ROOT / "data/tests/rank_prediction_test_private.csv")]
    questions, _ = ps.load_questions(inputs)
    tasks, owner, _, _ = ps.build_tasks(questions)
    sha = ps.questions_sha(inputs, questions)
    per_task, _, _ = ps.load_shards(a.shards, a.nshards, len(tasks), sha)
    _, l1b_per_task, _, _ = ps.load_shards_l1b(a.shards, a.nshards)
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    curves, pmi, opening = {}, {}, {}
    for k, sgf, color in tasks:
        n_t, hsl_sums = per_task[k]
        c = np.asarray(hsl_sums[:18], dtype=float) / max(n_t, 1)
        curves[k] = c - c.max()
        pmi[k] = float(posterior_mean_index(np.asarray(l1b_per_task[k][0], dtype=float)))
        _, moves = setup_and_moves(sgf)
        own = [(x, y_) for cc, x, y_ in moves if cc == color]
        opening[k] = (canonical(own[:1]), canonical(own[:3]), len(own))
    fast_t = Fast(curves, pmi, opening)
    test_rows = [{"n": len(by_q[qi]), **dict(zip(FP, fast_t.fp(by_q[qi])))} for qi in sorted(by_q)]
    T = pd.DataFrame(test_rows)
    rep["distribution"] = {"validation_by_n": {}, "validation_same_colour": {}, "test_by_n": {}}
    print("\n== (3) means by question size: validation (mixed colours) / validation same-colour questions / official test (same colour)")
    print(f"{'col':15s} " + " ".join(f"{'val n=' + str(m):>12s} {'val same n=' + str(m):>16s} {'test n=' + str(m):>12s}" for m in (5, 10, 20)))
    for c in FP:
        parts = []
        for m in (5, 10, 20):
            v_all, v_same, t_ = V.loc[V.n == m, c], V.loc[(V.n == m) & V.same_colour, c], T.loc[T.n == m, c]
            rep["distribution"]["validation_by_n"][f"{c}@{m}"] = [float(v_all.mean()), float(v_all.std()), int(len(v_all))]
            rep["distribution"]["validation_same_colour"][f"{c}@{m}"] = [float(v_same.mean()), float(v_same.std()), int(len(v_same))]
            rep["distribution"]["test_by_n"][f"{c}@{m}"] = [float(t_.mean()), float(t_.std()), int(len(t_))]
            parts.append(f"{v_all.mean():12.3f} {v_same.mean() if len(v_same) else float('nan'):16.3f} {t_.mean():12.3f}")
        print(f"{c:15s} " + " ".join(parts))
    rep["distribution"]["validation_same_colour_share_by_n"] = {str(m): float(V.loc[V.n == m, "same_colour"].mean()) for m in (5, 10, 20)}
    rep["distribution"]["test_n15"] = {c: float(T.loc[T.n == 15, c].mean()) for c in FP}
    print("validation questions with one colour:", rep["distribution"]["validation_same_colour_share_by_n"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
