import numpy as np
import pandas as pd

from aicup.scoring import fold_players, pending_parts, plan_parts


def _frames():
    games = pd.DataFrame(
        [("aaaa0000aaaa0000", "A", f"A:g{i:04d}") for i in range(30)]
        + [("bbbb1111bbbb1111", "A", f"A:g{100 + i:04d}") for i in range(4)]
        + [("cccc2222cccc2222", "6D", f"6D:g{i:04d}") for i in range(12)]
        + [("dddd3333dddd3333", "B", f"B:g{i:04d}") for i in range(25)],
        columns=["player_id", "rank", "game_key"])
    folds = pd.DataFrame([("A", "aaaa0000aaaa0000", 30, 1), ("A", "bbbb1111bbbb1111", 4, 1),
                          ("6D", "cccc2222cccc2222", 12, 1), ("B", "dddd3333dddd3333", 25, 2)],
                         columns=["rank", "player_id", "n_games", "fold"])
    return games, folds


def test_fold_players_caps_at_20_and_drops_players_under_5_games():
    games, folds = _frames()
    sel = fold_players(games, folds, fold=1).set_index("player_id")
    assert sorted(sel.index) == ["aaaa0000aaaa0000", "cccc2222cccc2222"]
    a = sel.loc["aaaa0000aaaa0000", "game_keys"]
    assert len(a) == 20 and len(set(a)) == 20 and set(a) <= {f"A:g{i:04d}" for i in range(30)}
    assert sorted(sel.loc["cccc2222cccc2222", "game_keys"]) == [f"6D:g{i:04d}" for i in range(12)]


def test_fold_players_selection_ignores_row_order_and_other_players():
    games, folds = _frames()
    base = fold_players(games, folds, fold=1).set_index("player_id")["game_keys"].to_dict()
    shuffled = fold_players(games.sample(frac=1.0, random_state=3), folds.sample(frac=1.0, random_state=4), fold=1)
    assert shuffled.set_index("player_id")["game_keys"].to_dict() == base
    only_a = fold_players(games[games["player_id"] == "aaaa0000aaaa0000"], folds, fold=1)
    assert only_a.iloc[0]["game_keys"] == base["aaaa0000aaaa0000"]


def test_plan_parts_covers_items_in_order():
    assert plan_parts(1001, 500) == [(0, 500), (500, 1000), (1000, 1001)]
    assert plan_parts(0, 500) == []


def test_pending_parts_needs_both_csv_and_npz(tmp_path):
    (tmp_path / "games_s0_p000.csv").write_text("x")
    np.savez(tmp_path / "moves_s0_p000.npz", a=np.zeros(1))
    (tmp_path / "games_s0_p001.csv").write_text("x")
    (tmp_path / "games_s1_p002.csv").write_text("x")
    np.savez(tmp_path / "moves_s1_p002.npz", a=np.zeros(1))
    assert pending_parts(tmp_path, 0, [(0, 1), (1, 2), (2, 3)]) == [1, 2]
