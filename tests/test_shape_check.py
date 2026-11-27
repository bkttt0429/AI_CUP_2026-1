"""aicup.seqshape and experiments/shape_check.py: the independent shape channel on top of fused_opp (issue #21)."""
import json

import numpy as np
import pandas as pd
import pytest

import seq_diag
import shape_check
from aicup import seqshape as ss
from aicup.scoring import read_scored_games
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def test_question_shape_means_and_stds_over_the_questions_games():
    table = pd.DataFrame({"game_key": ["a", "b", "c"], "std_pmi": [1.0, 3.0, 5.0], "ac1_pmi": [0.0, 0.5, 1.0]})
    qs = pd.DataFrame({"game_keys": [["a", "b"], ["c"], ["a", "b", "c"]]})
    X = ss.question_shape(table, qs, cols=["std_pmi", "ac1_pmi"])
    assert X.shape == (3, 2) and np.allclose(X, [[2.0, 0.25], [5.0, 1.0], [3.0, 0.5]])
    X2 = ss.question_shape(table, qs, cols=["std_pmi", "ac1_pmi"], with_std=True)
    assert X2.shape == (3, 4) and np.allclose(X2[:, :2], X) and np.allclose(X2[:, 2:], [[1.0, 0.25], [0.0, 0.0], [np.std([1, 3, 5]), np.std([0, 0.5, 1])]])
    with pytest.raises(KeyError):
        ss.question_shape(table, pd.DataFrame({"game_keys": [["zzz"]]}), cols=["std_pmi"])


def test_fold_shape_table_and_seq_diag_use_the_same_features(tmp_path):
    l1b_dir, _, _ = _synthetic_scores(tmp_path, folds=(2,))
    table = ss.fold_shape_table(l1b_dir / "fold2")
    games = read_scored_games(l1b_dir / "fold2")
    assert list(table.columns) == ["game_key", "player_id", "rank", "n_moves"] + ss.FEATURES and len(table) == len(games)
    assert set(table["game_key"]) == set(games["game_key"]) and np.allclose(table["log_moves"], np.log(5))
    assert ss.SHAPE == [f for f in ss.FEATURES if f != "log_moves"] and seq_diag.FEATURES is ss.FEATURES and seq_diag.game_shape is ss.game_shape
    diag = seq_diag.fold_table(l1b_dir / "fold2")
    assert list(diag.columns) == ["player_id", "y"] + ss.FEATURES and np.allclose(diag[ss.FEATURES].to_numpy(), table[ss.FEATURES].to_numpy())
    # the ranks of the synthetic games are recoverable from mean_pmi alone (peaked scores), so the per-rank means are ordered
    by_rank = table.groupby("rank")["mean_pmi"].mean()
    assert by_rank["D"] < by_rank["A"] < by_rank["6D"]


def test_shape_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "shape_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert shape_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert set(res["scores"]) == set(shape_check.SETS) and res["folds"] == [1, 2, 3, 4] and res["shape_features"] == ss.SHAPE
    assert {s: h["n_features"] for s, h in res["scores"].items()} == {"fused_opp": 56, "fused_opp_shape": 64, "fused_opp_shape2": 72}
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_shape vs fused_opp"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_shape2 vs fused_opp", "fused_opp_shape2 vs fused_opp_shape"}
    for h in res["scores"].values():
        assert len(h["per_fold"]) == 4 and 0 <= h["balanced_mean_over_n"] <= 1 and set(h["balanced_by_n"]) == {"5", "10", "20"}
    c = res["comparisons"]["pre_specified"]["fused_opp_shape vs fused_opp"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    assert "[pre_specified] fused_opp_shape vs fused_opp" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        shape_check.main(args + ["--folds", "0", "1"])
    assert shape_check.main(args[:2] + ["--opp", str(tmp_path / "missing")] + args[4:]) == 2
