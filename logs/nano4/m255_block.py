#!/usr/bin/env python
"""M255 (2026-11-20): the five-component block score (m84_cache.py's definition, frozen "5e" sub-weights) of the OPPONENT side of
every game of results/m255/inputs/{bq,ctl}.csv against all training players (design doc M255).

Same pools as m84_cache.py (all training rows except the held-out own games); rows in task order (bq.csv first, then ctl.csv; a
question's games in file order) -> results/m255/block_new.npy [games, C] float16 + block_new.json. Check: the control games'
rows against the cached blocks of the same test games (opponent side).

  python -I logs/nano4/m255_block.py
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m81_tactic_id as m81  # noqa: E402

COMPS = ip.COMPS; M82 = ROOT / "results/m82"; CACHE = ROOT / "results/m84_cache"; OUT = ROOT / "results/m255"; OTHER = {"B": "W", "W": "B"}


def new_games():
    out = []
    for f in ("bq.csv", "ctl.csv"):
        for d in pd.read_csv(OUT / "inputs" / f, dtype=str, keep_default_na=False).to_dict("records"):
            out += [(d["question_id"], d[f"sgf_{i}"], d[f"color_{i}"]) for i in range(1, int(d["num_games"]) + 1)]
    return out


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    w = json.loads((M82 / "report.json").read_text())["weights"]["5e"]; w5 = np.array([w[c] for c in COMPS]); q = np.load(CACHE / "queries.npz"); assert np.allclose(w5, q["w5"]); log("frozen sub-weights " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w5)))
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); assert np.array_equal(players.astype(str), q["players"].astype(str)); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}; sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); ng = new_games(); log(f"{len(rows)} training rows, {C} players, {len(ng)} new games")
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512)); qviews = list(pool.imap(ip.game_view, [(s, OTHER[c]) for _, s, c in ng], chunksize=16))
    marks = np.array([v[0] for v in views]); held_keys = json.load(open(M82 / "held_keys.json")); held = np.array([key_index[k_] for k_ in held_keys]); include = np.ones(len(rows), bool); include[held] = False
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]; pools = ip.fit_scenario(views, marks, pl, include, need, C, log)
    blk = np.empty((len(qviews), C), np.float16)
    for i, v in enumerate(qviews):
        sc = pools[v[0]].scores([v]); S = np.stack([sc[c] for c in COMPS]); blk[i] = w5 @ ((S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-12))
    np.save(OUT / "block_new.npy", blk); log(f"block of the new games written {blk.shape}")
    # ---------------------------------------------------------------- check against the cache (control games = original test games, opponent side)
    Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); tq = np.load(ROOT / "results/r_variants/m82_test.npz")["qid"].astype(str); cm = pd.read_csv(OUT / "inputs/ctl_map.csv", dtype=str); orig = dict(zip(cm.ctl_question, cm.test_question))
    idx = [np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]; where = {}
    for k in range(4):
        where.update({int(g): (k, r) for r, g in enumerate(idx[k])})
    B = [np.load(CACHE / f"block_{k}of4.npy", mmap_mode="r") for k in range(4)]; seen = {}; cors, dmax, same = [], [], []
    for i, (qid, _, _) in enumerate(ng):
        if qid not in orig:
            continue
        j = seen.get(qid, 0); seen[qid] = j + 1; task = int(np.flatnonzero(tq == orig[qid])[j]); k, r = where[2 * Q + H + task]; old = np.asarray(B[k][r], np.float32); new = blk[i].astype(np.float32)
        cors.append(float(np.corrcoef(old, new)[0, 1])); dmax.append(float(np.abs(old - new).max())); same.append(int(old.argmax() == new.argmax()))
    rep = {"games": len(ng), "control_games": len(cors), "min_corr": float(np.min(cors)), "mean_corr": float(np.mean(cors)), "max_abs_diff": float(np.max(dmax)), "same_first": float(np.mean(same))}
    (OUT / "block_new.json").write_text(json.dumps(rep, indent=1)); log(f"CHECK against the cached blocks, {len(cors)} control games: correlation min {rep['min_corr']:.5f} mean {rep['mean_corr']:.5f}; max |diff| {rep['max_abs_diff']:.3f}; same first candidate {rep['same_first']:.3f} -> {'PASS' if rep['min_corr'] >= 0.999 else 'FAIL'}")


if __name__ == "__main__":
    main()
