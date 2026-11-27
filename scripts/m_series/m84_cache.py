"""Section 4M M84 tooling (2026-11-11): cache of the five-component block score of every query, so that a new matcher variant is fused
in minutes instead of re-scoring the five components (40 minutes x 4 jobs). Same set-up as m82_cond_id.py's score stage (pools, held-out
games, queries, exclusions). block_i[c] = sum_j w_j z_ij[c] over the five components with M82's "5e" sub-weights (frozen), each z
standardised over the valid candidates of the query; the excluded candidate (a validation query's own player) is -inf. float16.
  --shard k/K -> results/m84_cache/block_{k}of{K}.npy (+ idx); shard 0 also writes block_held.npy and queries.npz (the query table:
  game key / test index, side, network, exclusion; players, labels, held-out truth)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COMPS = ip.COMPS
M82 = ROOT / "results/m82"; CACHE = ROOT / "results/m84_cache"
OTHER = {"B": "W", "W": "B"}
G = {}


def block_chunk(idx):
    C = G["C"]; out = np.empty((len(idx), C), np.float16)
    for r, i in enumerate(idx):
        v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); S = np.stack([sc[c] for c in COMPS]); ex = G["excl"][i]; valid = np.ones(C, bool)
        if ex >= 0:
            valid[ex] = False
        b = G["w5"] @ ((S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12))
        if ex >= 0:
            b[ex] = -np.inf
        out[r] = b
    return out


def main(k, K):
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    w = json.loads((M82 / "report.json").read_text())["weights"]["5e"]; w5 = np.array([w[c] for c in COMPS]); log("frozen sub-weights " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w5)))
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}; sgf = rows.sgf_content.tolist(); col = rows.color.tolist()
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))
    marks = np.array([v[0] for v in views]); qrow, qfold, qpid = [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, ng, gk in zip(m["player_id"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k_ in gk:
                    qrow.append(key_index[k_]); qfold.append(v); qpid.append(pid)
    qrow = np.array(qrow); qfold = np.array(qfold); Q = len(qrow)
    with ctx.Pool(12) as pool:
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))
    held_keys = json.load(open(M82 / "held_keys.json")); held = np.array([key_index[k_] for k_ in held_keys]); H = len(held); include = np.ones(len(rows), bool); include[held] = False
    tg = m81.load_tests(); T = len(tg)
    with ctx.Pool(12) as pool:
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt; excl = np.array([pidx[p_] for p_ in qpid] * 2 + [-1] * H + [-1] * (2 * T))
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log); G.update(qviews=qviews, pools=pools, excl=excl, C=C, w5=w5)
    n0 = 2 * Q; N = len(qviews)
    if k == 0:
        with ctx.Pool(12) as pool:
            hb = np.concatenate(pool.map(block_chunk, np.array_split(np.arange(n0, n0 + H), 48)))
        np.save(CACHE / "block_held.npy", hb)
        net = np.full(N, 4, np.int8); net[:Q] = net[Q:n0] = qfold - 1  # index into (1, 2, 3, 4, -1)
        side = np.zeros(N, np.int8); side[:Q] = 1; side[n0 + H:n0 + H + T] = 1  # 1 = the opponent's plies of the game
        qkey = np.array([keys[r] for r in qrow] * 2 + list(held_keys) + [""] * (2 * T)); qtest = np.full(N, -1, np.int32); qtest[n0 + H:n0 + H + T] = qtest[n0 + H + T:] = np.arange(T)
        np.savez_compressed(CACHE / "queries.npz", qkey=qkey, qtest=qtest, side=side, net=net, excl=excl, players=players, plab=plab, Q=Q, H=H, T=T, held_truth=pl[held], w5=w5)
        log(f"held-out block and the query table written ({N} queries)")
    rest = np.concatenate([np.arange(n0), np.arange(n0 + H, N)]); mine = np.array_split(rest, K)[k]
    with ctx.Pool(12) as pool:
        blk = np.concatenate(pool.map(block_chunk, np.array_split(mine, 240)))
    np.save(CACHE / f"block_{k}of{K}.npy", blk); np.save(CACHE / f"idx_{k}of{K}.npy", mine); log(f"shard {k} of {K}: {len(mine)} queries, block {blk.shape} {blk.dtype} written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args(); main(*map(int, a_.shard.split("/")))
