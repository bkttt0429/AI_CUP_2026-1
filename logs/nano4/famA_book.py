"""Research report family A (reports/圍棋等級預測新特徵家族.md): the out-of-fold opening rank book, pre-registered.

Phase 1 (hash): every HD0/HD1 training game -> the canonical (min over the 8 board symmetries) 64-bit hash of the position after
each of its first K_MAX plies, plus which plies the labelled player made. Phase 2: books = counts of the labelled player's rank per
position reached BY the labelled player's move (and an 'opponent book' for positions reached by the opponent's move), built only
from the players of the book folds; Dirichlet smoothing alpha = 10 towards the uniform class prior; support >= 30 games and >= 15
distinct players. Question features (7): book pmi, std of per-game book pmi, mean first off-book ply, coverage, entropy of the
book posterior, opponent-book pmi, opponent-book coverage.
  step 0  validation players' two random halves (10 games) with the book that excludes their fold: split-half r, W/between,
          adjacent-rank AUC, Spearman, partial Spearman given own/opp pmi; stop rule: split-half of book pmi < 0.5
  step 1  R + book7 vs R (hd_check protocol, MLP h32 seeds 0-2, folds 1-4): training questions of fold f use the book from the
          folds other than {v, f} (nested), validation questions the book from the folds other than v; control R + noise7;
          report only: K = 8 and K = 20 versions of the same seven columns

  python -I logs/nano4/famA_book.py --out results/famA/pilot.json
"""
import argparse
import hashlib
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
K_MAX, ALPHA, MIN_GAMES, MIN_PLAYERS = 20, 10.0, 30, 15
FEATS = ["book_pmi", "book_pmi_std", "first_offbook", "coverage", "book_entropy", "opp_book_pmi", "opp_coverage"]
_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        b = Board(19)
        _IDX = np.array([[b.loc(x, y) for x in range(19)] for y in range(19)], dtype=np.int64)
    return _IDX


def canonical_hash(grid):
    """min over the 8 symmetries of blake2b-8 of the 19x19 int8 grid."""
    best = None
    g = grid
    for _ in range(4):
        for arr in (g, g[:, ::-1]):
            h = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=8).digest(), "little")
            if best is None or h < best:
                best = h
        g = np.rot90(g)
    return best


def hash_game(args):
    """(hashes [K_MAX] uint64, is_target [K_MAX] bool, n) of one HD0/HD1 record: positions after each of the first K_MAX plies."""
    sgf, color = args
    idx = _index()
    moves = parse_moves(sgf)
    b = Board(19)
    hashes, tgt = np.zeros(K_MAX, dtype=np.uint64), np.zeros(K_MAX, dtype=bool)
    n = 0
    for k, (c, x, y) in enumerate(moves[:K_MAX]):
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else b.loc(x, y)
        try:
            b.play(pla, loc)
        except Exception:
            break
        hashes[k] = canonical_hash(b.board[idx])
        tgt[k] = (c == color)
        n = k + 1
    return hashes, tgt, n


def phase1(hd, workers, out):
    """Hashes of every HD0/HD1 game, in hd's row order (cached in results/famA/hashes.npz)."""
    if out.exists():
        z = np.load(out, allow_pickle=False)
        return z["hashes"], z["is_target"], z["n"]
    t0 = time.time()
    H, T, N = np.zeros((len(hd), K_MAX), dtype=np.uint64), np.zeros((len(hd), K_MAX), dtype=bool), np.zeros(len(hd), dtype=np.int16)
    row_of = {k: i for i, k in enumerate(hd["game_key"])}
    with mp.get_context("fork").Pool(workers) as pool:
        for r in sorted(set(hd["rank"])):
            df = pd.read_csv(Path(TRAIN_DIR) / f"train_{r}.csv", usecols=["game_id", "color", "sgf_content"], dtype=str, keep_default_na=False)
            df["game_key"] = r + ":" + df["game_id"]
            df = df[df["game_key"].isin(row_of)]
            rows = [row_of[k] for k in df["game_key"]]
            for i, (h, t, n) in zip(rows, pool.imap(hash_game, zip(df["sgf_content"], df["color"]), chunksize=200)):
                H[i], T[i], N[i] = h, t, n
            print(f"hashed rank {r}: {len(df)} games ({time.time() - t0:.0f} s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, hashes=H, is_target=T, n=N, game_key=hd["game_key"].to_numpy(dtype=str))
    return H, T, N


class Book:
    """Rank counts per position (sorted hash keys): counts [M, 10], n_games [M], n_players [M]."""

    def __init__(self, hashes, ranks, players):
        order = np.argsort(hashes, kind="stable")
        h, r, p = hashes[order], ranks[order], players[order]
        self.keys, start = np.unique(h, return_index=True)
        idx = np.repeat(np.arange(len(self.keys)), np.diff(np.append(start, len(h))))
        self.counts = np.zeros((len(self.keys), 10), dtype=np.int32)
        np.add.at(self.counts, (idx, r), 1)
        self.n_games = self.counts.sum(axis=1)
        pairs = np.unique(np.stack([h, p.astype(np.uint64)], axis=1), axis=0)
        self.n_players = np.bincount(np.searchsorted(self.keys, pairs[:, 0]), minlength=len(self.keys))
        self.logp = np.log((self.counts + ALPHA / 10.0) / (self.n_games[:, None] + ALPHA))
        self.ok = (self.n_games >= MIN_GAMES) & (self.n_players >= MIN_PLAYERS)

    def lookup(self, hashes):
        """(hit [n] bool, logp [n, 10]) for an array of hashes."""
        pos = np.searchsorted(self.keys, hashes)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        hit = (self.keys[pos] == hashes) & self.ok[pos] & (hashes != 0)
        return hit, np.where(hit[:, None], self.logp[pos], 0.0)


def build_books(H, T, folds_of_game, ranks, players, book_folds, K):
    """(own book, opponent book) from the games of players in book_folds, plies 1..K."""
    sel = np.isin(folds_of_game, list(book_folds))
    g = np.flatnonzero(sel)
    out = []
    for target in (True, False):
        mask = T[g, :K] if target else ~T[g, :K]
        mask = mask & (H[g, :K] != 0)
        rows, cols = np.nonzero(mask)
        out.append(Book(H[g][rows, cols], ranks[g][rows], players[g][rows]))
    return out


def question_features(keys_list, row_of, H, T, books, K):
    """[Q, 7] book features of questions given as lists of game keys."""
    own, opp = books
    out = np.zeros((len(keys_list), len(FEATS)))
    for qi, keys in enumerate(keys_list):
        g = np.array([row_of.get(k, -1) for k in keys])
        g = g[g >= 0]
        if len(g) == 0:
            continue
        Hq, Tq = H[g, :K], T[g, :K]
        hit_o, lp_o = own.lookup(np.where(Tq, Hq, 0).ravel())
        hit_x, lp_x = opp.lookup(np.where(~Tq, Hq, 0).ravel())
        hit_o, lp_o = hit_o.reshape(len(g), K), lp_o.reshape(len(g), K, 10)
        hit_x, lp_x = hit_x.reshape(len(g), K), lp_x.reshape(len(g), K, 10)
        n_t = Tq.sum()
        total = lp_o.sum(axis=(0, 1))
        per_game = lp_o.sum(axis=1)
        has = hit_o.any(axis=1)
        pm = posterior_mean_index(per_game[has]) if has.any() else np.array([])
        # first off-book ply: the first target ply (1-based position among the first K plies) without a hit; K + 1 if none
        first = []
        for i in range(len(g)):
            miss = np.flatnonzero(Tq[i] & ~hit_o[i])
            first.append(int(miss[0]) + 1 if len(miss) else K + 1)
        post = np.exp(total - total.max()); post /= post.sum()
        out[qi] = [posterior_mean_index(total) if hit_o.any() else 4.5, pm.std() if len(pm) > 1 else 0.0, float(np.mean(first)),
                   hit_o.sum() / max(n_t, 1), float(-(post * np.log(post + 1e-12)).sum()),
                   posterior_mean_index(lp_x.sum(axis=(0, 1))) if hit_x.any() else 4.5, hit_x.sum() / max((~Tq).sum(), 1)]
    return out


def half_stats(Hdf, cols):
    A, B = Hdf[Hdf.half == 0].set_index("player_id"), Hdf[Hdf.half == 1].set_index("player_id")
    both = A.index.intersection(B.index)
    out = {}
    for c in cols:
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1])
        w, between = 0.5 * np.mean((a - b) ** 2), np.var(0.5 * (a + b))
        hv = Hdf[c].to_numpy(dtype=float)
        aucs = [max(auc(hv[Hdf.y == k + 1], hv[Hdf.y == k]), 1 - auc(hv[Hdf.y == k + 1], hv[Hdf.y == k])) for k in range(9)]
        out[c] = {"split_half_r": r, "w_between": float(w / between) if between > 0 else None, "adjacent_auc": float(np.mean(aucs)),
                  "spearman_rank": spearman(Hdf[c], Hdf["y"]), "partial_rho_given_own_opp": partial_spearman(Hdf[c], Hdf["y"], Hdf[["own_pmi", "opp_pmi"]].to_numpy()),
                  "corr_own_pmi": spearman(Hdf[c], Hdf["own_pmi"]), "mean_by_rank": [float(Hdf.loc[Hdf.y == k, c].mean()) for k in range(10)]}
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
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--report-k", type=int, nargs="*", default=[8, 20])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    workers = max(1, min(len(os.sched_getaffinity(0)), 48))
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    fold_of_player = dict(zip(folds_df["player_id"], folds_df["fold"]))
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    hd = hd_all[hd_all["hd"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"{len(hd)} HD0/HD1 games of {hd['player_id'].nunique()} players; {workers} CPUs", flush=True)
    H, T, N = phase1(hd, workers, ROOT / "results/famA/hashes.npz")
    print(f"hashes ready: {len(H)} games, mean plies {N.mean():.1f} ({time.time() - t0:.0f} s)", flush=True)
    row_of = {k: i for i, k in enumerate(hd["game_key"])}
    folds_of_game = hd["player_id"].map(fold_of_player).to_numpy()
    ranks = hd["rank"].map(RANK).to_numpy(dtype=np.int64)
    players = pd.factorize(hd["player_id"])[0].astype(np.int64)
    rep = {"features": FEATS, "k": a.k, "games": int(len(hd)), "mean_plies": float(N.mean())}
    # scored tables (validation questions, own/opp pmi, R features)
    data = {}
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        players_f = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], "players": players_f, "val": l1b_cv.pilot_questions(players_f)}
    # step 0: halves with the book excluding the player's fold
    lcols = sum_cols(LEVELS)
    rows, feats = [], []
    for v in a.folds:
        books = build_books(H, T, folds_of_game, ranks, players, [f for f in a.folds if f != v], a.k)
        rep.setdefault("book_sizes", {})[str(v)] = {"own_keys": int(len(books[0].keys)), "own_supported": int(books[0].ok.sum()), "opp_supported": int(books[1].ok.sum())}
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
                             "own_pmi": float(posterior_mean_index(L.loc[ks].to_numpy(dtype=float).sum(axis=0))),
                             "opp_pmi": float(posterior_mean_index(O.loc[ks].to_numpy(dtype=float).sum(axis=0)))})
                keysets.append(ks)
        feats.append(question_features(keysets, row_of, H, T, books, a.k))
        print(f"step 0 fold {v}: book keys {len(books[0].keys)}, supported {books[0].ok.sum()} ({time.time() - t0:.0f} s)", flush=True)
    Hdf = pd.DataFrame(rows)
    F = np.concatenate(feats)
    for j, c in enumerate(FEATS):
        Hdf[c] = F[:, j]
    rep["step0"] = {"players": int(Hdf["player_id"].nunique()), "stats": half_stats(Hdf, FEATS)}
    print(f"\n== step 0 ({rep['step0']['players']} players, halves of 10 games; G0 = split-half >= 0.75 and |partial| >= 0.05)")
    print(f"{'column':14s} {'split-half':>10s} {'W/betw':>7s} {'adjAUC':>7s} {'rho':>7s} {'partial':>8s} {'r(own)':>7s} G0   mean by rank D..6D")
    for c, s in rep["step0"]["stats"].items():
        print(f"{c:14s} {s['split_half_r']:10.3f} {(s['w_between'] if s['w_between'] is not None else float('nan')):7.2f} {s['adjacent_auc']:7.3f} {s['spearman_rank']:7.3f} {s['partial_rho_given_own_opp']:8.3f} {s['corr_own_pmi']:7.2f} {'yes' if s['g0'] else 'no ':3s} " + " ".join(f"{m:.3f}" for m in s["mean_by_rank"]))
    stop = rep["step0"]["stats"]["book_pmi"]["split_half_r"] < 0.5
    rep["step0"]["stop_rule_triggered"] = bool(stop)
    print("stop rule (book pmi split-half < 0.5):", "TRIGGERED -> step 1 is report-only" if stop else "not triggered")
    # step 1: nested books
    ks_all = [a.k] + [k for k in a.report_k if k != a.k]
    names = ["base"] + [f"A_book7_K{k}" for k in ks_all] + ["noise7"]
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
        variants = {"base": (B_tr, B_va)}
        for k in ks_all:
            F_tr = np.zeros((len(train_q), len(FEATS)))
            for f in train:  # nested: fold f's training questions use the book without {v, f}
                books_f = build_books(H, T, folds_of_game, ranks, players, [x for x in a.folds if x not in (v, f)], k)
                sel = np.flatnonzero(q_fold == f)
                F_tr[sel] = question_features([list(x) for x in train_q["game_keys"].iloc[sel]], row_of, H, T, books_f, k)
            books_v = build_books(H, T, folds_of_game, ranks, players, train, k)
            F_va = question_features([list(x) for x in d["val"]["game_keys"]], row_of, H, T, books_v, k)
            variants[f"A_book7_K{k}"] = (np.hstack([B_tr, F_tr]), np.hstack([B_va, F_va]))
        noise = np.random.RandomState(99 + v)
        variants["noise7"] = (np.hstack([B_tr, noise.randn(len(B_tr), 7)]), np.hstack([B_va, noise.randn(len(B_va), 7)]))
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
    print("\n== step 1: R + book vs R (MLP h32, seeds averaged; nested OOF books)")
    for k in names:
        print(f"{k:14s} {rep['step1']['arms'][k]['score']:.4f}  folds {[round(x, 4) for x in per_fold[k]]}")
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
