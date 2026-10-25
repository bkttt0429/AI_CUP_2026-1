import pandas as pd
import pytest

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


def test_read_test_refuses_num_games_that_disagree_with_the_filled_columns(tmp_path):
    """The official format fills exactly the first num_games sgf columns; anything else stops the pipeline
    before scoring, naming the questions."""
    blank_inside = {"question_id": "priv_q_0001", "num_games": 3, "sgf_1": GOOD, "color_1": "B", "sgf_3": GOOD, "color_3": "B"}
    with pytest.raises(ValueError, match=r"priv_q_0001: num_games 3 but the non-empty sgf columns are \[1, 3\]"):
        read_test(_test_csv(tmp_path, [blank_inside]))
    too_few = {"question_id": "pub_q_0007", "num_games": 1, "sgf_1": GOOD, "color_1": "B", "sgf_2": GOOD, "color_2": "W"}
    with pytest.raises(ValueError, match=r"pub_q_0007: num_games 1 but the non-empty sgf columns are \[1, 2\]"):
        read_test(_test_csv(tmp_path, [too_few]))
    for bad in ("0", "21", "", "x"):
        row = {"question_id": "pub_q_0001", "num_games": bad, "sgf_1": GOOD, "color_1": "B"}
        with pytest.raises(ValueError, match=rf"pub_q_0001: num_games {bad!r} is not in 1..20"):
            read_test(_test_csv(tmp_path, [row]))
    # every malformed question is counted, the first max_bad are named, the file too
    rows = [{"question_id": f"pub_q_{i:04d}", "num_games": 2, "sgf_1": GOOD, "color_1": "B"} for i in range(7)]
    path = _test_csv(tmp_path, rows)
    with pytest.raises(ValueError, match=r"test.csv: 7 malformed questions, e.g. .*pub_q_0004.* \(and 2 more\)") as e:
        read_test(path)
    assert "pub_q_0005" not in str(e.value)
    with pytest.raises(ValueError, match="pub_q_0006"):
        read_test(path, max_bad=7)
    # "5.0" is still read as 5 games; the games keep their columns' order and colours
    ok = {"question_id": "pub_q_0002", "num_games": "2.0", "sgf_1": GOOD, "color_1": "w", "sgf_2": GOOD, "color_2": "B"}
    q = read_test(_test_csv(tmp_path, [ok]))[0]
    assert q["declared_games"] == 2 and q["games"] == [(GOOD, "W"), (GOOD, "B")]


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


def test_validate_submission_checks_order_prefixes_and_expected_counts(tmp_path):
    ids = ["pub_q_0001", "pub_q_0002", "priv_q_0001"]
    rows = [("pub_q_0001", "3D"), ("pub_q_0002", "D"), ("priv_q_0001", "6D")]
    good = tmp_path / "good.csv"
    write_submission(rows, good)
    assert validate_submission(good, ids, expected_total=3, expected_prefix_counts={"pub_": 2, "priv_": 1}) == []
    # the same rows in another order: every id is there once, but the order is not the inputs' and private
    # comes before public
    swapped = tmp_path / "swapped.csv"
    write_submission([rows[2], rows[0], rows[1]], swapped)
    problems = validate_submission(swapped, ids)
    assert any("order differs" in p and "row 0 is 'priv_q_0001', expected 'pub_q_0001'" in p for p in problems)
    assert any("priv_ question (row 0) comes before a pub_ question (row 2)" in p for p in problems)
    assert not any("missing" in p or "duplicate" in p or "unknown" in p for p in problems)
    # the official expectations are independent of the inputs: a truncated input is caught
    problems = validate_submission(good, ids, expected_total=800, expected_prefix_counts={"pub_": 400, "priv_": 400})
    assert "3 questions read from the inputs, expected 800" in problems
    assert "2 pub_ questions read from the inputs, expected 400" in problems
    assert "1 priv_ questions read from the inputs, expected 400" in problems
    assert len(problems) == 3
    # an id with neither prefix
    odd_ids = ["pub_q_0001", "q_0002"]
    odd = tmp_path / "odd.csv"
    write_submission([("pub_q_0001", "3D"), ("q_0002", "D")], odd)
    problems = validate_submission(odd, odd_ids)
    assert problems == ["ids without a pub_/priv_ prefix: 1, e.g. ['q_0002']"]
    # the public-first order holds whatever the inputs' order was: inputs given private first are refused too
    rev = tmp_path / "rev.csv"
    write_submission([rows[2], rows[0], rows[1]], rev)
    problems = validate_submission(rev, ["priv_q_0001", "pub_q_0001", "pub_q_0002"])
    assert problems == ["a priv_ question (row 0) comes before a pub_ question (row 2): public must come first"]
