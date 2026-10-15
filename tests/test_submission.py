import pandas as pd

from aicup.submission import check_game, read_test, validate_submission, write_submission

GOOD = "(;SZ[19];B[pd];W[dp];B[pq];W[dd])"


def _test_csv(tmp_path, rows):
    cols = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
    df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows], columns=cols)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_read_test_ignores_empty_columns_after_num_games(tmp_path):
    path = _test_csv(tmp_path, [
        {"question_id": "pub_q_0001", "num_games": 2, "sgf_1": GOOD, "color_1": "B", "sgf_2": GOOD, "color_2": "W"},
        {"question_id": "pub_q_0002", "num_games": 1, "sgf_1": GOOD, "color_1": "W"},
    ])
    qs = read_test(path)
    assert [q["question_id"] for q in qs] == ["pub_q_0001", "pub_q_0002"]
    assert qs[0]["games"] == [(GOOD, "B"), (GOOD, "W")]
    assert qs[1]["games"] == [(GOOD, "W")]
    assert qs[0]["declared_games"] == 2


def test_read_test_skips_blank_games_inside_the_declared_range(tmp_path):
    path = _test_csv(tmp_path, [{"question_id": "priv_q_0001", "num_games": 3, "sgf_1": GOOD, "color_1": "B",
                                 "sgf_3": GOOD, "color_3": "B"}])
    q = read_test(path)[0]
    assert q["games"] == [(GOOD, "B"), (GOOD, "B")] and q["declared_games"] == 3


def test_check_game_accepts_passes_and_rejects_bad_games():
    ok, info = check_game("(;SZ[19];B[pd];W[];B[pq];W[tt];B[dd])", "W")
    assert ok and info["passes"] == 2 and info["target_moves"] == 2
    assert check_game(GOOD, "B")[0]
    assert not check_game("(;SZ[13];B[dd];W[jj])", "B")[0]
    assert not check_game("(;SZ[19];B[pd];W[pd])", "B")[0]   # second stone on an occupied point
    assert not check_game(GOOD, "X")[0]
    assert not check_game("(;SZ[19])", "B")[0]                # no target moves
    assert not check_game("(;SZ[19];B[pd];B[dp];W[pq])", "W")[0]  # colours do not alternate


def test_validate_submission_reports_problems(tmp_path):
    ids = ["pub_q_0001", "pub_q_0002", "priv_q_0001"]
    good = tmp_path / "good.csv"
    write_submission([("pub_q_0001", "3D"), ("pub_q_0002", "D"), ("priv_q_0001", "6D")], good)
    assert validate_submission(good, ids) == []
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"question_id": ["pub_q_0001", "pub_q_0001", "pub_q_9999"], "pred_rank": ["3D", "3D", "7D"]}).to_csv(bad, index=False)
    problems = " | ".join(validate_submission(bad, ids))
    assert "missing" in problems and "duplicate" in problems and "unknown" in problems and "invalid rank" in problems
