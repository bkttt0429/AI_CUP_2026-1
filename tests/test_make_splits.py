import pandas as pd

from make_splits import assign_folds, make_questions


def test_assign_folds_round_robin_by_descending_games_within_rank():
    players = pd.DataFrame({
        "rank": ["A"] * 7 + ["D"] * 3,
        "player_id": [f"a{i}" for i in range(7)] + [f"d{i}" for i in range(3)],
        "n_games": [50, 40, 30, 20, 10, 5, 1, 9, 8, 7],
    })
    out = assign_folds(players, n_folds=5, seed=42)
    assert len(out) == 10 and out["player_id"].is_unique
    a = out[out["rank"] == "A"].sort_values("n_games", ascending=False)
    assert a["fold"].tolist() == [0, 1, 2, 3, 4, 0, 1]
    d = out[out["rank"] == "D"].sort_values("n_games", ascending=False)
    assert d["fold"].tolist() == [0, 1, 2]
    counts = out[out["rank"] == "A"]["fold"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_make_questions_round_robin_over_sizes_without_reusing_games():
    keys = [f"1D:g{i:03d}" for i in range(35)]
    qs = make_questions(keys, fold=2, rank="1D", player_id="abcdef0123456789", sizes=(5, 10, 20), max_q=2)
    assert [q["n_games"] for q in qs] == [5, 10, 20]
    used = [k for q in qs for k in q["game_keys"].split(";")]
    assert len(used) == len(set(used)) == 35
    assert qs[0]["question_id"] == "f2_1D_abcdef01_n5_0"
    assert qs[2]["question_id"] == "f2_1D_abcdef01_n20_0"


def test_make_questions_skips_sizes_that_no_longer_fit():
    keys = [f"A:g{i}" for i in range(24)]
    assert [q["n_games"] for q in make_questions(keys, 0, "A", "p" * 16)] == [5, 10, 5]
    assert make_questions(keys[:4], 0, "A", "p" * 16) == []
