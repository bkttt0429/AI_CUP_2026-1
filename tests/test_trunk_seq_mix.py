"""trunk_seq_check.mixed_questions (architecture v2 section 4L.4): same-rank mixed-player training questions."""
import numpy as np
import pandas as pd

import trunk_seq_check as ts


def _players(fold, n_per_rank=12, games=20):
    rows = []
    for r in ts.LEVELS if hasattr(ts, "LEVELS") else ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"):
        for i in range(n_per_rank):
            pid = f"{fold}{r}{i:02d}".ljust(16, "0")
            rows.append({"rank": r, "player_id": pid, "game_keys": [f"{r}:g_{fold}{i:02d}{k:02d}" for k in range(games)]})
    return pd.DataFrame(rows)


def test_mixed_questions_same_rank_distinct_other_players_never_held_out():
    data = {f: {"players": _players(f)} for f in (2, 3, 4)}
    rng = np.random.RandomState(0)
    rows = []
    for f in (2, 3, 4):
        for p in data[f]["players"].itertuples():
            for _ in range(4):
                n = int(rng.choice([5, 10, 20]))
                rows.append({"player_id": p.player_id, "rank": p.rank, "n_games": n, "game_keys": list(rng.permutation(p.game_keys)[:n])})
    train_q = pd.DataFrame(rows)
    q = ts.mixed_questions(train_q, data, [2, 3, 4], 1, 0.5, 0.1)
    q2 = ts.mixed_questions(train_q, data, [2, 3, 4], 1, 0.5, 0.1)
    assert q.equals(q2)  # deterministic
    assert list(q.columns) == ["player_id", "rank", "n_games", "game_keys", "mixed"] and len(q) == len(train_q)
    players = np.array(sorted(set(train_q["player_id"])))
    held = set(np.random.RandomState(12345 + ts.fold_code(1)).permutation(players)[:int(0.1 * len(players))])
    not_held = ~q["player_id"].isin(held)
    share = q.loc[not_held, "mixed"].mean()
    assert 0.35 < share < 0.65 and not q.loc[~not_held, "mixed"].any()  # held-out players' questions are never mixed (the inner set)
    owner = {k: (p.player_id, p.rank) for f in (2, 3, 4) for p in data[f]["players"].itertuples() for k in p.game_keys}
    for row, orig in zip(q.itertuples(), train_q.itertuples()):
        assert len(row.game_keys) == row.n_games == len(orig.game_keys)
        if not row.mixed:
            assert row.game_keys == orig.game_keys
            continue
        owners = [owner[k] for k in row.game_keys]
        assert all(r == row.rank for _, r in owners)  # same rank
        pids = [p for p, _ in owners]
        assert len(set(pids)) == len(pids) and row.player_id not in pids  # distinct other players
        assert not (set(pids) & held)  # never the inner held-out players
        assert row.game_keys != orig.game_keys
    # a rank with too few other players is left unmixed
    small = {2: {"players": _players(2, n_per_rank=2)}}
    tq = pd.DataFrame([{"player_id": p.player_id, "rank": p.rank, "n_games": 5, "game_keys": p.game_keys[:5]} for p in small[2]["players"].itertuples()])
    qs = ts.mixed_questions(tq, small, [2], 1, 1.0, 0.0)
    assert not qs["mixed"].any() and all(a == b for a, b in zip(qs["game_keys"], tq["game_keys"]))
