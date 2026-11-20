"""Data for the rank-evidence replay wall (viz/rank_replay.html).

Each board replays one game; under it, the posterior-mean Human SL profile of the moves the
target player has made so far (uniform prior over the 18 profiles) shows how the rank
evidence builds up move by move, against the band of best-fitting profiles that the pilot
measured for the player's true class.

Page capacity comes from the lab01 GPU measurements (input-to-paint P95, one headless
Chrome page): 50 boards 28 ms, 100 boards 54 ms, 200 boards 105 ms; 20 / 40 separate
windows 24 / 52 ms. So a page holds at most MAX_BOARDS boards unless forced.
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.evidence import posterior_mean_index
from aicup.humansl import PROFILES
from aicup.metric import LEVELS
from aicup.sgf import handicap, setup_and_moves

MAX_BOARDS = 100
DEFAULT_BOARDS = 50
MAX_WINDOWS = 40


def check_board_count(n, force=False):
    if n > MAX_BOARDS and not force:
        raise ValueError(f"{n} boards on one page; the measured comfortable limit is {MAX_BOARDS} "
                         f"(P95 54 ms at 100, 105 ms at 200). Use fewer boards, separate windows, or force.")


def evidence_series(logp):
    """Per target move: posterior-mean profile index and best profile of the cumulative log p."""
    cum = np.cumsum(np.asarray(logp, dtype=float), axis=0)
    return {"pmi": [round(float(v), 3) for v in posterior_mean_index(cum)], "best": [int(i) for i in cum.argmax(axis=1)]}


def class_bands(best_profile_counts, lo=0.1, hi=0.9):
    """{class: [low, high]} profile indices covering the middle of each class's best-profile counts."""
    out = {}
    for cls, counts in best_profile_counts.items():
        cum = np.cumsum(counts) / max(sum(counts), 1)
        out[cls] = [int(np.searchsorted(cum, lo - 1e-12)), int(np.searchsorted(cum, hi - 1e-12))]
    return out


def select_games(games, per_rank, seed=42, ranks=LEVELS):
    """per_rank players per rank, one random game each, in weak-to-strong order."""
    rng = np.random.RandomState(seed)
    picked = []
    for rank in ranks:
        sub = games[games["rank"] == rank]
        players = sorted(sub["player_id"].unique())
        for pid in [players[i] for i in sorted(rng.choice(len(players), size=min(per_rank, len(players)), replace=False))]:
            keys = sorted(sub.loc[sub["player_id"] == pid, "game_key"])
            picked.append({"game_key": keys[rng.randint(len(keys))], "player_id": pid, "rank": rank})
    return pd.DataFrame(picked, columns=["game_key", "player_id", "rank"])


def _pairs(directory, fold, layout):
    d = Path(directory)
    if layout == "pilot":
        for csv in sorted(d.glob(f"games_fold{fold}_shard*.csv")):
            yield csv, d / csv.name.replace("games_", "moves_").replace(".csv", ".npz")
    elif layout == "scores":
        d = d / f"fold{fold}"
        for csv in sorted(d.glob("games_s*_p*.csv")):
            if re.fullmatch(r"games_s\d+_p\d+\.csv", csv.name):
                yield csv, d / csv.name.replace("games_", "moves_").replace(".csv", ".npz")
    else:
        raise ValueError(f"unknown layout {layout!r}")


def load_move_logp(directory, fold, layout, keys=None):
    """{game_key: {"logp": [n, 18] float32, "t": [n]}} from CSV/npz pairs; finished pairs only."""
    want = None if keys is None else set(keys)
    out = {}
    for csv, npz in _pairs(directory, fold, layout):
        if not npz.exists():
            continue
        names = pd.read_csv(csv, usecols=["game_key"])["game_key"].tolist()
        if want is not None and not want.intersection(names):
            continue
        z = np.load(npz)
        rows, logp, ts = z["game_row"], z["logp"], z["t"]
        order = np.argsort(rows, kind="stable")
        bounds = np.searchsorted(rows[order], np.arange(len(names) + 1))
        for r, key in enumerate(names):
            if want is None or key in want:
                idx = order[bounds[r]:bounds[r + 1]]
                out[key] = {"logp": logp[idx].astype(np.float32), "t": ts[idx]}
    return out


def build_payload(rows, sgf_map, logp_map, bands, title, source):
    games = []
    for r in rows.itertuples():
        sgf, color = sgf_map[r.game_key]
        ev = logp_map.get(r.game_key)
        stones, moves = setup_and_moves(sgf)
        games.append({
            "key": r.game_key, "player": r.player_id[:8], "rank": r.rank, "color": color,
            "moves": [None if x is None else [x, y] for _, x, y in moves],
            "stones": [[x, y] for x, y in stones], "first": moves[0][0] if moves else "B", "handicap": handicap(sgf),
            "band": bands.get(r.rank),
            "evidence": None if ev is None else {"t": [int(t) for t in ev["t"]], **evidence_series(ev["logp"])},
        })
    return {"title": title, "source": source, "created": datetime.now().isoformat(timespec="seconds"),
            "profiles": PROFILES, "levels": LEVELS, "games": games}
