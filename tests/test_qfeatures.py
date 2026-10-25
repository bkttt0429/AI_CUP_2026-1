import numpy as np
import pandas as pd

from aicup.humansl import PROFILES
from aicup.qfeatures import FEATURE_NAMES, build_matrix, question_features, sample_questions


def test_feature_vector_layout_and_relative_curve():
    sums = np.tile(-np.linspace(3.0, 1.0, 18), (5, 1)) * 100  # per-game sums, best profile = last
    f = question_features(sums, np.full(5, 100))
    assert f.shape == (len(FEATURE_NAMES),)
    rel = f[:18]
    assert rel.max() == 0.0 and rel[17] == 0.0 and np.all(rel <= 0)
    named = dict(zip(FEATURE_NAMES, f))
    assert named["best_logp"] == np.float64(-1.0)
    assert named["log_moves"] == np.log(500) and named["log_games"] == np.log(5)
    assert named["game_argmax_std"] == 0.0


def test_features_finite_for_short_five_game_question():
    rng = np.random.RandomState(0)
    sums = -rng.rand(5, 18) * 3
    f = question_features(sums, np.array([1, 1, 2, 1, 1]))
    assert np.all(np.isfinite(f))


def test_peak_interp_is_between_neighbours_and_exact_for_symmetric_peak():
    curve = -((np.arange(18) - 7.0) ** 2) * 0.01
    named = dict(zip(FEATURE_NAMES, question_features(curve[None, :] * 50, np.array([50]))))
    assert abs(named["peak_interp"] - 7.0) < 1e-9
    skew = -((np.arange(18) - 7.3) ** 2) * 0.01
    p = dict(zip(FEATURE_NAMES, question_features(skew[None, :] * 50, np.array([50]))))["peak_interp"]
    assert 7.0 < p < 7.5


def test_build_matrix_maps_labels_and_keeps_question_order():
    cols = {f"sum_{p}": [-1.0 * (i + 1), -2.0 * (i + 1)] for i, p in enumerate(PROFILES)}
    games = pd.DataFrame({"game_key": ["g1", "g2"], "n_moves": [10, 20], **cols})
    qs = pd.DataFrame({"question_id": ["q1", "q2"], "rank": ["3D", "D"], "game_keys": [["g1", "g2"], ["g2"]]})
    X, y, n_games = build_matrix(games, qs)
    assert X.shape == (2, len(FEATURE_NAMES)) and y.tolist() == [6, 0] and n_games.tolist() == [2, 1]


def test_sample_questions_respects_player_game_counts_and_seed():
    players = pd.DataFrame({"rank": ["A", "6D"], "player_id": ["aaaa0000aaaa0000", "bbbb1111bbbb1111"],
                            "game_keys": [[f"A:{i}" for i in range(20)], [f"6D:{i}" for i in range(7)]]})
    a = sample_questions(players, per_player=6, rng=np.random.RandomState(1))
    b = sample_questions(players, per_player=6, rng=np.random.RandomState(1))
    assert a["game_keys"].tolist() == b["game_keys"].tolist()
    assert len(a) == 12 and a["n_games"].between(5, 20).all()
    six = a[a["player_id"] == "bbbb1111bbbb1111"]
    assert (six["n_games"] <= 7).all()
    assert all(len(set(k)) == len(k) for k in a["game_keys"])


def test_profiles_parameter():
    from aicup.qfeatures import SUM_COLS, feature_names, sum_cols

    assert feature_names(["a", "b"]) == ["rel_a", "rel_b", "best_logp", "pmi_sum", "peak_interp", "log_moves", "log_games", "game_argmax_std"]
    assert sum_cols(["a", "b"]) == ["sum_a", "sum_b"] and FEATURE_NAMES == feature_names(PROFILES) and SUM_COLS == sum_cols(PROFILES)
    games = pd.DataFrame({"game_key": ["g1", "g2", "g3"], "n_moves": [50, 60, 70], "sum_a": [-100.0, -120.0, -140.0], "sum_b": [-90.0, -125.0, -130.0]})
    questions = pd.DataFrame({"question_id": ["q1", "q2"], "rank": ["1D", "6D"], "game_keys": [["g1", "g2"], ["g3"]]})
    X, y, n = build_matrix(games, questions, profiles=["a", "b"])
    assert X.shape == (2, 8) and y.tolist() == [4, 9] and n.tolist() == [2, 1] and np.isfinite(X).all()
