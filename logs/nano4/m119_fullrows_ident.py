#!/usr/bin/env python
"""M119 (pilot, report only): do the UNPROJECTED per-game trunk rows identify SEEN players from unseen games better than the
projected rows this project's matcher uses, and what is left of the difference on top of the sequence identity nets?

Candidates: the players of --folds (default 0: the rows of logs/nano4/m118_fullrows_extract.py; folds 1-4 read the peer session's
/work/u2507115/pid_v2/fullrows_train once it is complete). Rows of a game (the owner's plies): `rp` = the stored 256-dim projected
row (results/m82/rows_train.npz, m84_rowsp), `full` = the 1,536-dim unprojected statistics. Queries:
  Q1  the official held-out own games (results/m82/held_keys.json) of the candidates: the sequence nets (five members, var_S2y,
      network -1) never trained on them, so the row matcher can be tested on top of them;
  Q2  two more games of every candidate with >= 12 games left, held out here (rows only).
Row matcher: the peer's PlayerEmbedding (episodes of 512 players, single-game queries against up to 40 support games, 6,000 steps,
two seeds averaged), trained on the candidates' remaining games (the seen-player setting of M82's network -1); a candidate's set
vector = all his remaining games. Fusion with the sequence nets: per query the scores are standardised over the candidates,
Z_seq + w * Z_row, w chosen on one half of the players and scored on the other (both ways).
Reported: top-1 / top-5 among the candidates and paired bootstrap (over players) differences.

  python -I logs/nano4/m119_fullrows_ident.py --folds 0
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
CACHE, M82 = ROOT / "results/m84_cache", ROOT / "results/m82"
MINE = ROOT / "results/m118_fullrows/fold0"
PEER = Path("/work/u2507115/pid_v2/fullrows_train/rows")
ROWSP = Path("/work/u2507115/aicup1_moved/m84_rowsp")
WS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)


def projected_rows(want):
    """{key: [256]} of the wanted games from the two stores of projected target-side rows."""
    out = {}
    R = np.load(M82 / "rows_train.npz")
    for k, r in zip(R["key"].astype(str), R["T"]):
        if k in want:
            out[k] = r
    for f in sorted(glob.glob(str(ROWSP / "rows_*.npz"))):
        z = np.load(f)
        keys = z["key"].astype(str)
        hit = [i for i, k in enumerate(keys) if k in want]
        if hit:
            T = z["T"]
            for i in hit:
                out[keys[i]] = T[i]
    return out


def full_rows(folds):
    """{key: [1536] fp16} from this project's fold-0 parts and / or the peer's parts of folds 1-4."""
    out = {}
    if 0 in folds:
        for f in sorted(MINE.glob("part_p*.npz")):
            z = np.load(f)
            out.update(zip(z["key"].astype(str), z["T_full"]))
    if any(v > 0 for v in folds):
        for f in sorted(PEER.glob("part_s*_p*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            out.update(zip(z["keys"].astype(str), z["full"].astype(np.float16)))
    return out


def zs(S):
    return (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)


def rank_of(S, truth):
    return (S > S[np.arange(len(truth)), truth][:, None]).sum(1)


def boot(d, cluster, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num, den = np.bincount(inv, weights=d, minlength=len(ids)), np.bincount(inv, minlength=len(ids)).astype(float)
    pick = rng.integers(0, len(ids), (B, len(ids)))
    bs = num[pick].sum(1) / den[pick].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="m119_fullrows_ident")
    a = ap.parse_args(argv)
    import torch
    from aicup_pid.embed import PlayerEmbedding
    from aicup.scoring import player_games
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    games = player_games()
    folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str})
    fold_of = dict(zip(folds.player_id, folds.fold))
    games = games[games.player_id.map(fold_of).isin(a.folds)]
    FULL = full_rows(a.folds)
    games = games[games.game_key.isin(FULL.keys())]
    RP = projected_rows(set(games.game_key))
    games = games[games.game_key.isin(RP.keys())].sort_values(["player_id", "game_key"])
    log(f"folds {a.folds}: {len(games)} games of {games.player_id.nunique()} players with both rows")
    q = np.load(CACHE / "queries.npz")
    Q, H = int(q["Q"]), int(q["H"])
    all_players = q["players"].astype(str)
    held_keys = json.load(open(M82 / "held_keys.json"))
    held_truth = q["held_truth"]
    assert len(held_keys) == H and np.array_equal(q["qkey"][2 * Q:2 * Q + H].astype(str), np.array(held_keys))
    by = games.groupby("player_id")["game_key"].apply(list).to_dict()
    heldset = set(held_keys)
    rng = np.random.RandomState(20261114)
    train, q2 = {}, []
    for p in sorted(by):
        ks = [k for k in by[p] if k not in heldset]
        if len(ks) >= 12:
            pick = set(rng.choice(len(ks), 2, replace=False).tolist())
            q2 += [(ks[i], p) for i in sorted(pick)]
            ks = [k for i, k in enumerate(ks) if i not in pick]
        if len(ks) >= 8:
            train[p] = ks
    cands = sorted(train)
    cidx = {p: i for i, p in enumerate(cands)}
    q1 = [(k, all_players[t], h) for h, (k, t) in enumerate(zip(held_keys, held_truth)) if all_players[t] in cidx and k in FULL and k in RP]
    q2 = [(k, p) for k, p in q2 if p in cidx]
    log(f"candidates {len(cands)}; Q1 (official held-out games of the candidates) {len(q1)} of {len(set(p for _, p, _ in q1))} players; Q2 (held out here) {len(q2)}")

    def matcher(store):
        """[len(q1) + len(q2), C] mean over the seeds of the standardised row-matcher scores."""
        rows_by = {p: np.stack([store[k] for k in train[p]]).astype(np.float32) for p in cands}
        Xq = np.stack([store[k] for k, *_ in q1] + [store[k] for k, _ in q2]).astype(np.float32)
        out = 0
        for s in a.seeds:
            m = PlayerEmbedding(steps=a.steps, players_per_episode=512, support=40, query_sizes=(1,), p_aug=0.0, seed=s).fit(rows_by)
            Cm = np.stack([m.set_vector(rows_by[p]) for p in cands])
            Zq = m.embed(Xq)
            Zq /= np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-12
            out = out + zs(Zq @ Cm.T)
            log(f"  seed {s}: episode top-1 at the end {m.history[-1]['acc']:.3f}, scale {m.scale:.1f}")
            del m
            torch.cuda.empty_cache()
        return out / len(a.seeds)

    Z = {}
    for name, store in (("rp", RP), ("full", FULL)):
        log(f"row matcher on {name} ({len(next(iter(store.values())))} dims)")
        Z[name] = matcher(store)
    n1 = len(q1)
    truth1 = np.array([cidx[p] for _, p, _ in q1])
    truth2 = np.array([cidx[p] for _, p in q2])
    truth = np.r_[truth1, truth2]
    out = {"folds": a.folds, "candidates": len(cands), "n_q1": n1, "n_q2": len(q2), "rows_only": {}, "fusion_q1": {}}
    rngb = np.random.default_rng(0)
    for name in Z:
        r = rank_of(Z[name], truth)
        out["rows_only"][name] = {"q1_top1": float((r[:n1] == 0).mean()), "q1_top5": float((r[:n1] < 5).mean()), "q2_top1": float((r[n1:] == 0).mean()), "q2_top5": float((r[n1:] < 5).mean())}
    d = (rank_of(Z["full"], truth)[n1:] == 0).astype(float) - (rank_of(Z["rp"], truth)[n1:] == 0).astype(float)
    out["rows_only"]["full_minus_rp_q2_top1"] = {"diff": float(d.mean()), "ci95": boot(d, truth2, a.boot, rngb)}
    # the sequence nets on Q1: five members, network -1 (index 4), prototypes of the candidates
    v = np.load(CACHE / "var_S2y.npz")
    ci = np.array([int(np.flatnonzero(all_players == p)[0]) for p in cands])
    hq = np.array([h for *_, h in q1])
    Zs = 0
    for m in range(v["protos"].shape[0]):
        Zs = Zs + zs(v["qvec"][m, 2 * Q + hq].astype(np.float32) @ v["protos"][m, 4][ci].T)
    Zs = zs(Zs)
    r_seq = rank_of(Zs, truth1)
    out["fusion_q1"]["seq"] = {"top1": float((r_seq == 0).mean()), "top5": float((r_seq < 5).mean())}
    half = np.array([int(p[:8], 16) % 2 for _, p, _ in q1])
    hit = {"seq": (r_seq == 0).astype(float)}
    for name in Z:
        Zr = Z[name][:n1]
        h_ = np.zeros(n1)
        r5 = np.zeros(n1)
        ws = {}
        for b in (0, 1):
            w = max(WS, key=lambda w_: (rank_of(Zs[half != b] + w_ * Zr[half != b], truth1[half != b]) == 0).mean())
            r = rank_of(Zs[half == b] + w * Zr[half == b], truth1[half == b])
            h_[half == b], r5[half == b] = r == 0, r < 5
            ws[str(b)] = w
        hit[f"seq+{name}"] = h_
        out["fusion_q1"][f"seq+{name}"] = {"top1": float(h_.mean()), "top5": float(r5.mean()), "w": ws,
                                             "top1_by_w": {str(w_): float((rank_of(Zs + w_ * Zr, truth1) == 0).mean()) for w_ in WS}}
    for x, b in (("seq+full", "seq+rp"), ("seq+full", "seq"), ("seq+rp", "seq")):
        d = hit[x] - hit[b]
        out["fusion_q1"][f"{x} - {b}"] = {"diff": float(d.mean()), "ci95": boot(d, truth1, a.boot, rngb)}
    print(json.dumps(out, indent=1))
    (ROOT / "results/r_variants" / f"{a.tag}_f{''.join(map(str, a.folds))}.json").write_text(json.dumps(out, indent=1) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
