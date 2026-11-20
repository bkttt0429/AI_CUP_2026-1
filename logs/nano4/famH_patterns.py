"""Research report family H: rank-reference local-pattern likelihood (Moudřík-style), out-of-fold, pre-registered.
Phase 1: every HD0/HD1 training game is replayed; for each ply of the LABELLED player the canonical (8 symmetries, colours
relative to the mover) 3x3 ring (8 points, base 4: empty/own/opp/off) and the 5x5 diamond (12 points at Manhattan distance <= 2)
of the position BEFORE the move are stored. Phase 2: per book fold set, P(pattern | rank) with Dirichlet smoothing over a
vocabulary (3x3: all; diamond: the top 1000 patterns + one out-of-vocabulary bucket); question features = the 10-dim
log-likelihood vector sum_t log P(pat_t | r) minus its max (two books -> two 10-dim arms) + coverage.
  step 0  halves (10 games) per validation player with the book excluding the player's fold: split-half, AUC, partial rho
  step 1  R + H10 vs R (nested OOF books), control R + noise10

  python -I logs/nano4/famH_patterns.py --out results/famH/pilot.json
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(ROOT / "repo/KataGo/python")]
from aicup.evidence import auc, posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import TRAIN_DIR, read_scored_games  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from pilot_stats import partial_spearman, spearman  # noqa: E402
from katago.game.board import Board  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
RING = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
DIAMOND = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3) if 0 < abs(dx) + abs(dy) <= 2]
SYMS = [lambda x, y: (x, y), lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (-x, -y), lambda x, y: (y, x), lambda x, y: (-y, x), lambda x, y: (y, -x), lambda x, y: (-y, -x)]


def _perms(points):
    """For each symmetry, the index permutation that maps the point list onto its image."""
    idx = {p: i for i, p in enumerate(points)}
    return [np.array([idx[s(*p)] for p in points]) for s in SYMS]


RING_PERMS, DIA_PERMS = _perms(RING), _perms(DIAMOND)
RING_POW, DIA_POW = 4 ** np.arange(len(RING), dtype=np.int64), 4 ** np.arange(len(DIAMOND), dtype=np.int64)


def canonical_code(vals, perms, pows):
    best = None
    for perm in perms:
        c = int((vals[perm] * pows).sum())
        if best is None or c < best:
            best = c
    return best


def game_patterns(args):
    """(ring codes uint16 [n_t], diamond codes uint32 [n_t]) over the labelled player's plies of one HD0/HD1 record."""
    sgf, color = args
    moves = parse_moves(sgf)
    b = Board(19)
    rings, dias = [], []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        if x is None:
            try:
                b.play(pla, Board.PASS_LOC)
            except Exception:
                break
            continue
        if c == color:
            own = pla
            def cell(dx, dy):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < 19 and 0 <= yy < 19):
                    return 3
                v = b.board[b.loc(xx, yy)]
                return 0 if v == Board.EMPTY else (1 if v == own else 2)
            rings.append(canonical_code(np.array([cell(dx, dy) for dx, dy in RING]), RING_PERMS, RING_POW))
            dias.append(canonical_code(np.array([cell(dx, dy) for dx, dy in DIAMOND]), DIA_PERMS, DIA_POW))
        try:
            b.play(pla, b.loc(x, y))
        except Exception:
            break
    return np.asarray(rings, dtype=np.uint16), np.asarray(dias, dtype=np.uint32)


def phase1(hd, workers, out):
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return list(z["rings"]), list(z["dias"])
    t0 = time.time()
    rings, dias = [None] * len(hd), [None] * len(hd)
    row_of = {k: i for i, k in enumerate(hd["game_key"])}
    with mp.get_context("fork").Pool(workers) as pool:
        for r in sorted(set(hd["rank"])):
            df = pd.read_csv(Path(TRAIN_DIR) / f"train_{r}.csv", usecols=["game_id", "color", "sgf_content"], dtype=str, keep_default_na=False)
            df["game_key"] = r + ":" + df["game_id"]
            df = df[df["game_key"].isin(row_of)]
            rows = [row_of[k] for k in df["game_key"]]
            for i, (rg, dm) in zip(rows, pool.imap(game_patterns, zip(df["sgf_content"], df["color"]), chunksize=100)):
                rings[i], dias[i] = rg, dm
            print(f"patterns rank {r}: {len(df)} games ({time.time() - t0:.0f} s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rings=np.array(rings, dtype=object), dias=np.array(dias, dtype=object), game_key=hd["game_key"].to_numpy(dtype=str))
    return rings, dias


class PatternBook:
    """log P(pattern | rank) over a vocabulary with Dirichlet smoothing (alpha per cell); unknown patterns -> the OOV row."""

    def __init__(self, codes_per_game, ranks, vocab_size=None, alpha=1.0):
        codes = np.concatenate([c for c in codes_per_game]) if len(codes_per_game) else np.zeros(0, dtype=np.int64)
        rk = np.concatenate([np.full(len(c), r) for c, r in zip(codes_per_game, ranks)]) if len(codes_per_game) else np.zeros(0, dtype=np.int64)
        uniq, counts = np.unique(codes, return_counts=True)
        if vocab_size is not None and len(uniq) > vocab_size:
            keep = uniq[np.argsort(-counts)[:vocab_size]]
        else:
            keep = uniq
        self.index = {int(c): i for i, c in enumerate(sorted(keep))}
        V = len(self.index) + 1  # + OOV
        M = np.zeros((V, 10))
        ids = np.array([self.index.get(int(c), V - 1) for c in codes])
        np.add.at(M, (ids, rk.astype(int)), 1)
        self.logp = np.log((M + alpha) / (M.sum(axis=0, keepdims=True) + alpha * V))
        self.V = V

    def loglik(self, codes):
        ids = np.array([self.index.get(int(c), self.V - 1) for c in codes]) if len(codes) else np.zeros(0, dtype=int)
        L = self.logp[ids].sum(axis=0) if len(ids) else np.zeros(10)
        return L, float(np.mean(ids != self.V - 1)) if len(ids) else 0.0


def question_feats(keys_list, row_of, codes, book):
    out = np.zeros((len(keys_list), 11))
    for qi, keys in enumerate(keys_list):
        g = [row_of[k] for k in keys if k in row_of]
        c = np.concatenate([codes[i] for i in g]) if g else np.zeros(0, dtype=np.int64)
        L, cov = book.loglik(c)
        out[qi, :10] = L - L.max()
        out[qi, 10] = cov
    return out


def half_stats(H, cols):
    A, B = H[H.half == 0].set_index("player_id"), H[H.half == 1].set_index("player_id")
    both = A.index.intersection(B.index)
    out = {}
    for c in cols:
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1])
        hv = H[c].to_numpy(dtype=float)
        aucs = [max(auc(hv[H.y == k + 1], hv[H.y == k]), 1 - auc(hv[H.y == k + 1], hv[H.y == k])) for k in range(9)]
        out[c] = {"split_half_r": r, "adjacent_auc": float(np.mean(aucs)), "spearman_rank": spearman(H[c], H["y"]),
                  "partial_rho_given_own_opp": partial_spearman(H[c], H["y"], H[["own_pmi", "opp_pmi"]].to_numpy()), "corr_own_pmi": spearman(H[c], H["own_pmi"]),
                  "mean_by_rank": [float(H.loc[H.y == k, c].mean()) for k in range(10)]}
        out[c]["g0"] = bool(r >= 0.75 and abs(out[c]["partial_rho_given_own_opp"]) >= 0.05)
    return out


def _fit(job):
    name, Xt, y, Xv, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return MLPModel(hidden=32, seed=seed).fit(Xt, y).predict_proba(Xv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    workers = max(1, min(len(os.sched_getaffinity(0)), 48))
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    fold_of_player = dict(zip(folds_df["player_id"], folds_df["fold"]))
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    hd = hd_all[hd_all["hd"].isin([0.0, 1.0])].reset_index(drop=True)
    rings, dias = phase1(hd, workers, ROOT / "results/famH/patterns.npz")
    print(f"patterns ready: {len(hd)} games, {sum(len(r) for r in rings)} labelled plies ({time.time() - t0:.0f} s)", flush=True)
    row_of = {k: i for i, k in enumerate(hd["game_key"])}
    fold_g = hd["player_id"].map(fold_of_player).to_numpy()
    ranks = hd["rank"].map(RANK).to_numpy()

    def books(book_folds):
        sel = np.flatnonzero(np.isin(fold_g, list(book_folds)))
        return (PatternBook([rings[i] for i in sel], ranks[sel]), PatternBook([dias[i] for i in sel], ranks[sel], vocab_size=a.vocab))

    data = {}
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], "players": players, "val": l1b_cv.pilot_questions(players)}
    rep = {"vocab": a.vocab}
    # step 0
    lcols = sum_cols(LEVELS)
    rows, F = [], []
    for v in a.folds:
        bk_ring, bk_dia = books([f for f in a.folds if f != v])
        rep.setdefault("book", {})[str(v)] = {"ring_vocab": bk_ring.V, "diamond_vocab": bk_dia.V}
        d = data[v]
        L, O = d["l1b"].set_index("game_key")[lcols], d["opp"].set_index("game_key")[lcols]
        keysets = []
        for p in d["players"].itertuples():
            keys = list(p.game_keys)
            if len(keys) < 10:
                continue
            rng = np.random.RandomState((int(p.player_id[:8], 16) + 7) % 2**32)
            order = rng.permutation(len(keys)); k2 = len(keys) // 2
            for half, idx in ((0, order[:k2]), (1, order[k2:2 * k2])):
                ks = [keys[i] for i in idx]
                rows.append({"player_id": p.player_id, "y": RANK[p.rank], "half": half,
                             "own_pmi": float(posterior_mean_index(L.loc[ks].to_numpy(dtype=float).sum(axis=0))), "opp_pmi": float(posterior_mean_index(O.loc[ks].to_numpy(dtype=float).sum(axis=0)))})
                keysets.append(ks)
        Fr, Fd = question_feats(keysets, row_of, rings, bk_ring), question_feats(keysets, row_of, dias, bk_dia)
        F.append(np.column_stack([posterior_mean_index(Fr[:, :10]), Fr[:, 10], posterior_mean_index(Fd[:, :10]), Fd[:, 10]]))
        print(f"step 0 fold {v} ({time.time() - t0:.0f} s)", flush=True)
    H = pd.DataFrame(rows)
    F = np.concatenate(F)
    cols = ["ring_pmi", "ring_cov", "dia_pmi", "dia_cov"]
    for j, c in enumerate(cols):
        H[c] = F[:, j]
    rep["step0"] = {"players": int(H["player_id"].nunique()), "stats": half_stats(H, cols)}
    print(f"\n== step 0 ({rep['step0']['players']} players; G0 = split-half >= 0.75 and |partial| >= 0.05)")
    for c, s in rep["step0"]["stats"].items():
        print(f"{c:10s} split-half {s['split_half_r']:.3f} adjAUC {s['adjacent_auc']:.3f} rho {s['spearman_rank']:+.3f} partial {s['partial_rho_given_own_opp']:+.3f} r(own) {s['corr_own_pmi']:+.2f} G0 {'yes' if s['g0'] else 'no'}  by rank " + " ".join(f"{m:.2f}" for m in s["mean_by_rank"]))
    # step 1 (nested books)
    names = ["base", "H_ring10", "H_dia10", "noise10"]
    scores, per_fold, ys, pids, ns = {k: [] for k in names}, {k: [] for k in names}, [], [], []
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        B_tr, B_va = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        q_fold = train_q["player_id"].map(fold_of_player).to_numpy()
        Ftr_r, Ftr_d = np.zeros((len(train_q), 11)), np.zeros((len(train_q), 11))
        for f in train:
            br, bd = books([x for x in a.folds if x not in (v, f)])
            sel = np.flatnonzero(q_fold == f)
            kl = [list(x) for x in train_q["game_keys"].iloc[sel]]
            Ftr_r[sel], Ftr_d[sel] = question_feats(kl, row_of, rings, br), question_feats(kl, row_of, dias, bd)
        br, bd = books(train)
        kv = [list(x) for x in d["val"]["game_keys"]]
        Fva_r, Fva_d = question_feats(kv, row_of, rings, br), question_feats(kv, row_of, dias, bd)
        noise = np.random.RandomState(99 + v)
        variants = {"base": (B_tr, B_va), "H_ring10": (np.hstack([B_tr, Ftr_r[:, :10]]), np.hstack([B_va, Fva_r[:, :10]])),
                    "H_dia10": (np.hstack([B_tr, Ftr_d[:, :10]]), np.hstack([B_va, Fva_d[:, :10]])),
                    "noise10": (np.hstack([B_tr, noise.randn(len(B_tr), 10)]), np.hstack([B_va, noise.randn(len(B_va), 10)]))}
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
    rep["step1"] = {"arms": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k]} for k in names}, "pairs": {}}
    print("\n== step 1: R + pattern book vs R")
    for k in names:
        print(f"{k:10s} {rep['step1']['arms'][k]['score']:.4f}  folds {[round(x, 4) for x in per_fold[k]]}")
    for k in names[1:]:
        dd = paired_diff(S[k], S["base"], y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[k], per_fold["base"])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
        rep["step1"]["pairs"][f"{k} vs base"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{k} vs base: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
