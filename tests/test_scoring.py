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


def _write_part(out_dir, shard, k, keys, npz=True):
    pd.DataFrame({"game_key": keys}).to_csv(out_dir / f"games_s{shard}_p{k:03d}.csv", index=False)
    if npz:
        np.savez(out_dir / f"moves_s{shard}_p{k:03d}.npz", a=np.zeros(1))


def test_done_game_keys_and_next_index_ignore_incomplete_parts(tmp_path):
    from aicup.scoring import done_game_keys, next_part_index

    _write_part(tmp_path, 0, 0, ["g1", "g2"])
    _write_part(tmp_path, 0, 1, ["g3"], npz=False)   # interrupted between the two renames
    _write_part(tmp_path, 1, 0, ["g9"])              # another shard
    assert done_game_keys(tmp_path, 0) == {"g1", "g2"}
    assert next_part_index(tmp_path, 0) == 2
    assert next_part_index(tmp_path, 5) == 0


def test_resume_plan_scores_each_game_once_even_if_part_size_changes(tmp_path):
    from aicup.scoring import resume_plan

    keys = [f"g{i}" for i in range(10)]
    first = resume_plan(keys, tmp_path, 0, part_size=4)
    assert [k for k, _ in first] == [0, 1, 2] and first[0][1] == keys[:4]
    for k, part in first[:2]:                        # two parts finish, then the run is paused
        _write_part(tmp_path, 0, k, part)
    second = resume_plan(keys, tmp_path, 0, part_size=3)
    assert [k for k, _ in second] == [2]
    assert second[0][1] == ["g8", "g9"]
    done_twice = [g for _, part in first[:2] for g in part] + [g for _, part in second for g in part]
    assert sorted(done_twice) == sorted(keys)


def test_resume_plan_rejects_finished_games_outside_the_selection(tmp_path):
    import pytest

    from aicup.scoring import resume_plan

    _write_part(tmp_path, 0, 0, ["stranger"])
    with pytest.raises(ValueError):
        resume_plan(["g1", "g2"], tmp_path, 0, part_size=1)


def test_pause_requested(tmp_path):
    from aicup.scoring import pause_requested

    flag = tmp_path / "PAUSE"
    assert not pause_requested(flag)
    flag.write_text("")
    assert pause_requested(flag)


def test_read_scored_games_skips_temporary_files_and_rejects_duplicates(tmp_path):
    import pytest

    from aicup.scoring import read_scored_games

    _write_part(tmp_path, 0, 0, ["g1", "g2"])
    _write_part(tmp_path, 1, 0, ["g3"])
    pd.DataFrame({"game_key": ["g4"]}).to_csv(tmp_path / "games_s0_p001.tmp.csv", index=False)
    assert sorted(read_scored_games(tmp_path)["game_key"]) == ["g1", "g2", "g3"]
    _write_part(tmp_path, 1, 1, ["g1"])
    with pytest.raises(ValueError):
        read_scored_games(tmp_path)


def test_compare_scored_games_reports_per_move_differences_and_gaps():
    import pytest

    from aicup.scoring import compare_scored_games

    ref = pd.DataFrame({"game_key": ["g1", "g2", "g3"], "n_moves": [100, 50, 80],
                        "sum_1k": [-200.0, -90.0, -150.0], "sum_1d": [-190.0, -95.0, -140.0]})
    new = pd.DataFrame({"game_key": ["g1", "g2", "g9"], "n_moves": [100, 51, 70],
                        "sum_1k": [-200.0, -90.0, -1.0], "sum_1d": [-189.8, -95.0, -1.0]})
    res = compare_scored_games(new, ref)
    assert res["games"] == 2
    assert res["missing_in_ref"] == 1
    assert res["n_moves_mismatch"] == 1
    assert res["max_abs_sum"] == pytest.approx(0.2)
    assert res["max_abs_per_move"] == pytest.approx(0.2 / 100)
    assert res["mean_abs_per_move"] == pytest.approx(0.2 / 100 / 4)
    assert res["best_profile_agreement"] == 1.0


def test_score_stream_default_scorer_is_humansl_and_scorer_overrides(monkeypatch):
    """score_stream(pool, model, tasks, rows) still scores with humansl.score_positions; scorer= replaces it."""
    import multiprocessing as mp

    from katago.train import modelconfigs

    from aicup import humansl, scoring

    calls = []

    def fake_humansl(model, bins, globs, legal, target, colors, rows):
        calls.append(("humansl", model, rows))
        return np.full((len(target), 3), -1.0, dtype=np.float32)

    monkeypatch.setattr(humansl, "score_positions", fake_humansl)
    tasks = [(0, "(;SZ[19];B[pd];W[dp];B[pq])", "B"), (1, "(;SZ[19];B[pd];W[dp])", "W")]
    with mp.get_context("fork").Pool(1, initializer=scoring.init_worker, initargs=(modelconfigs.config_of_name["b2c16"],)) as pool:
        per_task, arrays, n_pos, wall = scoring.score_stream(pool, "MODEL", tasks, "ROWS", log=lambda *a, **k: None)
        assert calls == [("humansl", "MODEL", "ROWS")] and n_pos == 3 and per_task[0][0] == 2 and per_task[1][0] == 1
        np.testing.assert_array_equal(per_task[0][1], [-2.0, -2.0, -2.0])
        assert arrays[0].shape == (3, 3) and arrays[1].tolist() == [0, 0, 1] and arrays[2].tolist() == [0, 2, 1]

        def l1b_like(bins, globs, legal, target, colors, rows):
            return np.tile(np.arange(10, dtype=np.float32), (len(target), 1))

        per_task2, arrays2, _, _ = scoring.score_stream(pool, None, tasks, None, scorer=l1b_like, log=lambda *a, **k: None)
    assert len(calls) == 1 and arrays2[0].shape == (3, 10) and per_task2[0][1].tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
