#!/usr/bin/env python
"""Section 4M M135: external games whose record cannot be replayed (an illegal move under the replay the scoring scripts use).

Every game of the external table is replayed move by move (aicup.sgf.setup_and_moves, aicup.pilot.start_state, the production
rules); a game that raises is listed with the index of the failing move. --fix rewrites the table with those games cut before
their first illegal move (the game keys, their order and every other game are unchanged, so the scoring passes resume where they
stopped); the list goes to <root>/illegal_games.csv. A game already cut replays cleanly, so the fix is idempotent.

  python -I logs/nano4/m135_scan_illegal.py [--fix] [--root /work/u2507115/ext_rank] [--procs 40]
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from aicup.pilot import start_state  # noqa: E402
from aicup.sgf import _MOVE_RE, setup_and_moves  # noqa: E402

RULES = dict(GameState.RULES_CHINESE, whiteKomi=7.5)


def check(job):
    key, sgf = job
    try:
        stones, moves = setup_and_moves(sgf)
        gs = start_state(stones, RULES)
    except Exception as e:  # noqa: BLE001
        return key, -1, 0, f"{type(e).__name__}: {e}"
    for t, (c, x, y) in enumerate(moves):
        try:
            gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))
        except Exception as e:  # noqa: BLE001
            return key, t, len(moves), f"{type(e).__name__}: {e}"
    return key, None, len(moves), ""


def cut(sgf, t):
    """The record with its first t moves only."""
    ms = list(_MOVE_RE.finditer(sgf))
    return sgf[:ms[t].start()] + ")"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/work/u2507115/ext_rank")
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    root = Path(a.root); files = sorted((root / "training").glob("train_F*.csv"))
    tabs = {f: pd.read_csv(f, dtype=str, keep_default_na=False) for f in files}
    jobs = [(r + ":" + g, s) for t in tabs.values() for r, g, s in zip(t["rank"], t["game_id"], t["sgf_content"])]
    with mp.Pool(a.procs) as pool:
        res = pool.map(check, jobs, chunksize=200)
    bad = [(k, t, n, e) for k, t, n, e in res if t is not None]
    lens = pd.Series([n for _, t, n, _ in res if t is None])
    print(f"{len(jobs)} games replayed; {len(bad)} cannot be replayed ({len(bad) / len(jobs):.4%}); clean games: moves median {lens.median():.0f}, min {lens.min()}")
    if not bad:
        return
    b = pd.DataFrame(bad, columns=["game_key", "fail_index", "n_moves", "error"]); b["rank"] = b["game_key"].str.split(":").str[0]
    print("by error: " + "; ".join(f"{e}: {c}" for e, c in b["error"].value_counts().items()))
    print("by rank: " + "; ".join(f"{r}: {c}" for r, c in b["rank"].value_counts().sort_index().items()))
    print(f"failing move index: min {b['fail_index'].min()}, 10% {b['fail_index'].quantile(0.1):.0f}, median {b['fail_index'].median():.0f}, max {b['fail_index'].max()}; share of the record kept by cutting: median {(b['fail_index'] / b['n_moves']).median():.2f}; cut below 40 moves: {int((b['fail_index'] < 40).sum())}, below 80: {int((b['fail_index'] < 80).sum())}")
    b.to_csv(root / "illegal_games.csv", index=False)
    if not a.fix:
        print("list written (no change to the table; --fix cuts these games before their first illegal move)"); return
    where = dict(zip(b["game_key"], b["fail_index"])); changed = 0
    for f, t in tabs.items():
        keys = t["rank"] + ":" + t["game_id"]; m = keys.isin(where)
        if not m.any():
            continue
        t.loc[m, "sgf_content"] = [cut(s, int(where[k])) for k, s in zip(keys[m], t.loc[m, "sgf_content"])]
        again = [check((k, s)) for k, s in zip(keys[m], t.loc[m, "sgf_content"])]
        assert all(r[1] is None for r in again), "a cut record still does not replay"
        assert all(r[2] == int(where[r[0]]) for r in again), "a cut record has an unexpected number of moves"
        tmp = f.with_name(f.name + f".tmp{os.getpid()}"); t.to_csv(tmp, index=False); os.replace(tmp, f); changed += int(m.sum())
    print(f"table rewritten: {changed} games cut before their first illegal move (keys and order unchanged)")


if __name__ == "__main__":
    main()
