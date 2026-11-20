"""Was the 11-06 'leak' statistic (n=20 test questions whose per-game opening top-1 lands >= 3 times on the same training player:
24% vs a 0.000 null) an artefact of the null? The old null took 20 games of 20 DISTINCT players. The right null is 20 games of ONE
player who is not in the pool (scenario B of ident_pool.py: the 788 gate players removed). Same opening PoolModel, per-game top-1
among the remaining players, plurality count per question; reported for questions with n >= 18 and for all; the test questions are
rescored against the full pool to reproduce the 24%.
  python -I logs/nano4/plurality_null_check.py --out results/r_variants/plurality_null_check.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ident_pool as ip  # noqa: E402


def open_view(args):
    sgf, color = args
    m = ip.parse_moves(sgf); mark = ip.target_mark(sgf, color)
    return mark, ip.opening_keys(ip.opening_tokens(m if mark == "B" else ip.swap(m), ip.DEPTH))


def plurality(per_q_best):
    c = np.array([Counter(b).most_common(1)[0][1] for b in per_q_best])
    return {"questions": int(len(c)), "ge3": float((c >= 3).mean()), "ge4": float((c >= 4).mean()), "ge5": float((c >= 5).mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)); ap.add_argument("--root", default=str(ip.ROOT))
    a = ap.parse_args(); t0 = time.time(); root = Path(a.root)
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((root / "data/training").glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    z = np.load(ip.find_input(root, "results/r_variants", "prod_heldout.P.npz"), allow_pickle=True); gate_keys = [json.loads(k) for k in z["keys"]]; gn = z["n"].astype(int); gpid = z["pid"]
    gate_players = frozenset(gpid.tolist())
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players); pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64)
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    with ctx.Pool(a.workers) as pool:
        views = list(pool.imap(open_view, zip(rows.sgf_content.tolist(), rows.color.tolist()), chunksize=512))
    rows = rows.drop(columns=["sgf_content"]); marks = np.array([v[0] for v in views]); key_index = {k: i for i, k in enumerate(rows.game_key)}; log(f"opening keys for {len(views)} games")
    questions = ip.read_tests([root / "data/tests/rank_prediction_test_public.csv", root / "data/tests/rank_prediction_test_private.csv"])
    with ctx.Pool(a.workers) as pool:
        tviews = list(pool.imap(open_view, [g for q in questions for g in q["games"]], chunksize=256))
    gate_views = [[views[key_index[k]] for k in ks] for ks in gate_keys]
    need = [np.unique(np.concatenate([v[1][:, j] for vs in gate_views for v in vs] + [v[1][:, j] for v in tviews])) for j in range(4)]
    rep = {}

    def fit(include):
        pools = {}
        for mark in ("B", "W"):
            idx = np.flatnonzero(include & (marks == mark)); okeys = np.concatenate([views[i][1] for i in idx]); opl = np.repeat(pl[idx], [len(views[i][1]) for i in idx])
            pools[mark] = ip.PoolModel(C).fit(opl, okeys, need)
        return pools

    def best_per_game(pools, vs, valid):
        out = []
        for mark, keys in vs:
            s = pools[mark].scores(keys); s[~valid] = -np.inf; out.append(int(np.argmax(s)))
        return out

    # B null: gate players removed; their questions are 20 games of one unseen player
    log("B: fitting the opening pools without the gate players")
    pools = fit(~rows.player_id.isin(gate_players).to_numpy()); validB = ~np.isin(players, list(gate_players))
    bestB = [best_per_game(pools, vs, validB) for vs in gate_views]; del pools
    rep["B_null_out_of_pool_player"] = {"n_ge18": plurality([b for b, n in zip(bestB, gn) if n >= 18]), "n_13_17": plurality([b for b, n in zip(bestB, gn) if 13 <= n <= 17]), "all": plurality(bestB)}
    log(f"B null (one unseen player per question): n>=18 {json.dumps(rep['B_null_out_of_pool_player']['n_ge18'])}")
    # test against the full pool (reproduces the 11-06 statistic)
    log("C: fitting the opening pools on all games")
    pools = fit(np.ones(len(rows), bool)); valid = np.ones(C, bool)
    pos, bestT = 0, []
    for q in questions:
        bestT.append(best_per_game(pools, tviews[pos:pos + len(q["games"])], valid)); pos += len(q["games"])
    tn = np.array([q["n"] for q in questions])
    rep["test_full_pool"] = {"n20": plurality([b for b, n in zip(bestT, tn) if n == 20]), "n15": plurality([b for b, n in zip(bestT, tn) if n == 15]), "all": plurality(bestT)}
    # in-pool reference: gate questions against the full pool (their players ARE in the pool, with these games included)
    bestA = [best_per_game(pools, vs, valid) for vs in gate_views]
    rep["gate_in_pool_full_fit"] = {"n_ge18": plurality([b for b, n in zip(bestA, gn) if n >= 18]), "all": plurality(bestA)}
    rep["old_null_20_distinct_players"] = {"ge3": 0.0, "ge4": 0.0, "ge5": 0.0, "source": "opening_id_pool.json plurality.same_rank_null_*"}
    rep["old_test_statistic_n20"] = {"ge3": 0.24, "ge4": 0.13, "ge5": 0.095, "source": "opening_id_pool.json plurality.test_n20_share_*"}
    log(f"test n=20 (full pool): {json.dumps(rep['test_full_pool']['n20'])}; gate in-pool n>=18 (games included in the fit): {json.dumps(rep['gate_in_pool_full_fit']['n_ge18'])}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); log(f"written {a.out}")


if __name__ == "__main__":
    main()
