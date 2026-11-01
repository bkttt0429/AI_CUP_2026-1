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


def test_read_game_keys_csv_or_lines(tmp_path):
    from aicup.scoring import read_game_keys

    (tmp_path / "k.txt").write_text("# the HD >= 2 games\nA:g1\n\n6D:g7\n")
    assert read_game_keys(tmp_path / "k.txt") == ["A:g1", "6D:g7"]
    pd.DataFrame({"game_key": ["A:g1", "B:g2"], "hd": [2, 9]}).to_csv(tmp_path / "k.csv", index=False)
    assert read_game_keys(tmp_path / "k.csv") == ["A:g1", "B:g2"]
    (tmp_path / "empty.txt").write_text("")
    assert read_game_keys(tmp_path / "empty.txt") == []


def test_restrict_players_keeps_the_selection_order_and_drops_empty_players():
    from aicup.scoring import restrict_players

    games, folds = _frames()
    players = fold_players(games, folds, fold=1)
    sel = restrict_players(players, ["A:g0003", "A:g0001", "6D:g0005", "Z:none"])
    a = sel[sel["player_id"] == "aaaa0000aaaa0000"].iloc[0]["game_keys"]
    full = players[players["player_id"] == "aaaa0000aaaa0000"].iloc[0]["game_keys"]
    assert set(a) <= {"A:g0001", "A:g0003"} and a == [k for k in full if k in ("A:g0001", "A:g0003")]
    assert set(sel["player_id"]) <= {"aaaa0000aaaa0000", "cccc2222cccc2222"} and list(sel.columns) == ["rank", "player_id", "game_keys"]
    assert restrict_players(players, []).empty


def test_patch_fold_replaces_rows_and_arrays_of_the_patched_games_only(tmp_path):
    import json

    import pytest

    from aicup.l1beval import read_moves
    from aicup.scoring import patch_fold, read_scored_games
    from tests.test_l1b_scripts import _write_scored

    levels = ["x", "y"]
    base, patch, out = tmp_path / "base" / "fold1", tmp_path / "patch" / "fold1", tmp_path / "out" / "fold1"
    games = [{"game_key": f"A:g{i}", "player_id": "0000aaaa", "rank": "A", "color": "BW"[i % 2]} for i in range(5)]
    lp = [np.full((3 + i, 2), -1.0 - i) for i in range(5)]
    ts = [np.arange(3 + i) * 2 for i in range(5)]
    _write_scored(base, 0, 0, games[:3], lp[:3], ts[:3], levels)
    _write_scored(base, 1, 0, games[3:], lp[3:], ts[3:], levels)
    (base / "meta_s0.json").write_text("{}")
    # the patch scores games 1 and 4 again: other move counts and values
    _write_scored(patch, 0, 0, [games[1], games[4]], [np.full((2, 2), -5.0), np.full((9, 2), -0.5)], [np.array([1, 3]), np.arange(9) * 2 + 1], levels)
    s = patch_fold(base, patch, out)
    assert (s["games"], s["patched"], s["n_moves_before"], s["n_moves_after"]) == (5, 2, 4 + 7, 2 + 9)
    assert [q["game_key"] for q in s["patched_games"]] == ["A:g1", "A:g4"] and (out / "meta_s0.json").exists() and (out / "patch_meta.json").exists()
    assert json.loads((out / "patch_meta.json").read_text())["patched"] == 2
    t_base, t_out = read_scored_games(base).set_index("game_key"), read_scored_games(out).set_index("game_key")
    assert list(t_out.index) == list(t_base.index) and (t_out["color"] == t_base["color"]).all()
    for k in ("A:g0", "A:g2", "A:g3"):
        assert t_out.loc[k].equals(t_base.loc[k])
    assert int(t_out.loc["A:g1", "n_moves"]) == 2 and float(t_out.loc["A:g1", "sum_x"]) == -10.0 and int(t_out.loc["A:g4", "n_moves"]) == 9
    assert s["sum_mean_abs_change"]["sum_x"] == pytest.approx((abs(-10.0 - (-2.0 * 4)) + abs(-4.5 - (-5.0 * 7))) / 2)
    g_out, lp_out, row_out, t_out_moves = read_moves(out)
    g_base, lp_base, row_base, t_base_moves = read_moves(base)
    pos = {k: i for i, k in enumerate(g_out["game_key"])}
    assert pos == {k: i for i, k in enumerate(g_base["game_key"])}
    for k, n, val in (("A:g0", 3, -1.0), ("A:g1", 2, -5.0), ("A:g2", 5, -3.0), ("A:g3", 6, -4.0), ("A:g4", 9, -0.5)):
        m = row_out == pos[k]
        assert m.sum() == n and (lp_out[m] == val).all(), k
    assert (t_out_moves[row_out == pos["A:g1"]] == [1, 3]).all() and (t_out_moves[row_out == pos["A:g0"]] == t_base_moves[row_base == pos["A:g0"]]).all()
    assert (np.diff(row_out) >= 0).all()
    # refusals: an existing output, a patch game the base lacks, a colour mismatch
    with pytest.raises(FileExistsError):
        patch_fold(base, patch, out)
    stray = tmp_path / "stray" / "fold1"
    _write_scored(stray, 0, 0, [{**games[1], "game_key": "A:g9"}], [np.zeros((1, 2))], [np.array([0])], levels)
    with pytest.raises(ValueError, match="not in"):
        patch_fold(base, stray, tmp_path / "out2" / "fold1")
    wrong = tmp_path / "wrong" / "fold1"
    _write_scored(wrong, 0, 0, [{**games[1], "color": "B"}], [np.zeros((1, 2))], [np.array([0])], levels)
    with pytest.raises(ValueError, match="color"):
        patch_fold(base, wrong, tmp_path / "out3" / "fold1")
    with pytest.raises(FileNotFoundError):
        patch_fold(base, tmp_path / "nothing", tmp_path / "out4" / "fold1")


def test_patch_scored_cli(tmp_path, capsys):
    import json

    import patch_scored
    from tests.test_l1b_scripts import _write_scored

    levels = ["x"]
    games = [{"game_key": f"A:g{i}", "player_id": "0000aaaa", "rank": "A", "color": "B"} for i in range(2)]
    for f in (1, 2):
        _write_scored(tmp_path / "base" / f"fold{f}", 0, 0, games, [np.ones((2, 1)), np.ones((3, 1))], [np.arange(2), np.arange(3)], levels)
        _write_scored(tmp_path / "patch" / f"fold{f}", 0, 0, games[:1], [np.zeros((4, 1))], [np.arange(4)], levels)
    summary = tmp_path / "summary.json"
    args = ["--base", str(tmp_path / "base"), "--patch", str(tmp_path / "patch"), "--out", str(tmp_path / "out"), "--folds", "1", "2", "--summary", str(summary)]
    assert patch_scored.main(args) == 0
    res = json.loads(summary.read_text())
    assert set(res["folds"]) == {"1", "2"} and res["folds"]["1"]["patched"] == 1 and res["folds"]["2"]["n_moves_after"] == 4
    assert "fold 1: 1 of 2 games patched" in capsys.readouterr().out
    assert patch_scored.main(args) == 2 and "already holds" in capsys.readouterr().err
