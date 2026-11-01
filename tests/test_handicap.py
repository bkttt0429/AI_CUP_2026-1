"""aicup.handicap and scripts/hd_table.py: HD per game and the question-level handicap features (issue #42, proposal (a))."""
import numpy as np
import pandas as pd
import pytest

import hd_table as ht
from aicup import handicap as hc
from tests.test_hd_census import _train_dir


def test_hd_table_and_summary(tmp_path):
    d = _train_dir(tmp_path)
    t = hc.hd_table(d)
    assert list(t.columns) == hc.HD_COLUMNS and len(t) == 12 and t["game_key"].str.startswith(("D:", "A:", "6D:")).all()
    assert t["hd"].dtype == float and sorted(t["hd"].unique()) == [0.0, 1.0, 2.0]
    s = hc.hd_summary(t)
    assert s == {"games": 12, "hd": {"0": 6, "1": 3, "2": 3}, "share_hd1": 0.5, "share_hd2": 0.25}
    v10 = hc.hd_table(_train_dir(tmp_path / "v10", with_hd=False))
    assert v10["hd"].isna().all() and hc.hd_summary(v10)["hd"] == {"absent": 12}
    path = tmp_path / "hd.csv"
    t.to_csv(path, index=False)
    back = hc.read_hd_table(path)
    assert back.equals(t)
    v10.to_csv(path, index=False)
    assert hc.read_hd_table(path)["hd"].isna().all()
    with pytest.raises(FileNotFoundError):
        hc.hd_table(tmp_path / "missing")


def test_game_features_signed_gap_and_question_means():
    hd, color = [np.nan, 0, 1, 1, 2, 9], ["B", "W", "W", "B", "W", "B"]
    assert hc.signed_gap(hd, color).tolist() == [0.0, 0.0, 1.0, -1.0, 2.0, -9.0]
    F = hc.game_hd_features(hd, color)
    assert F.shape == (6, 5) and F[0].tolist() == [0, 0, 0, 0, 0]  # a v1.0 record: nothing known
    assert F[1].tolist() == [1, 0, 0, 0, 0] and F[2].tolist() == [0, 1, 0, 1, 0] and F[3].tolist() == [0, 1, 0, 0, 1]
    assert F[4].tolist() == [0, 0, 1, 2, 0] and F[5].tolist() == [0, 0, 1, 0, 9]
    table = pd.DataFrame({"game_key": [f"g{i}" for i in range(6)], "player_id": "p", "rank": "A", "color": color, "hd": hd})
    qs = pd.DataFrame({"game_keys": [["g1", "g2"], ["g4", "g5", "g0"], ["g3"]]})
    X = hc.question_hd_features(table, qs)
    assert X.shape == (3, 5) and np.allclose(X[0], [0.5, 0.5, 0, 0.5, 0]) and np.allclose(X[1], [0, 0, 2 / 3, 2 / 3, 3]) and X[2].tolist() == [0, 1, 0, 0, 1]
    assert hc.question_hd_features(table, pd.DataFrame({"game_keys": []})).shape == (0, 5)
    with pytest.raises(KeyError):
        hc.question_hd_features(table, pd.DataFrame({"game_keys": [["nope"]]}))
    # the submission path reads the same features from (sgf, colour) pairs
    games = [("(;SZ[19]HD[2];B[qf])", "W"), ("(;SZ[19]HD[0];B[pd])", "B"), ("(;SZ[19];B[pd])", "B")]
    assert np.allclose(hc.question_hd_features_from_games(games), [1 / 3, 0, 1 / 3, 2 / 3, 0]) and hc.question_hd_features_from_games([]).tolist() == [0] * 5


def test_hd_table_cli(tmp_path, capsys):
    d = _train_dir(tmp_path)
    out, keys = tmp_path / "hd_table.csv", tmp_path / "hd2_keys.csv"
    assert ht.main(["--train-dir", str(d), "--out", str(out), "--keys-out", str(keys)]) == 0
    t = hc.read_hd_table(out)
    assert len(t) == 12 and list(t.columns) == hc.HD_COLUMNS
    k = pd.read_csv(keys, dtype=str)
    assert list(k.columns) == ["game_key", "color", "hd"] and len(k) == 3 and set(k["game_key"]) <= set(t["game_key"])
    printed = capsys.readouterr().out
    assert "12 games" in printed and "3 games with HD >= 2" in printed
    assert ht.main(["--train-dir", str(tmp_path / "missing"), "--out", str(out)]) == 2
