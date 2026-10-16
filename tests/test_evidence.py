import math

import numpy as np
import pandas as pd
import pytest

from aicup.evidence import (HARD_BINS, apply_thresholds, auc, bootstrap_ci, class_scores_hard_bins,
                            fit_thresholds, posterior_mean_index, question_matrix)
from aicup.metric import mean_score


def test_hard_bins_cover_18_profiles_once_in_order():
    flat = [i for b in HARD_BINS for i in b]
    assert flat == list(range(18)) and len(HARD_BINS) == 10
    assert HARD_BINS[0] == [0, 1, 2] and HARD_BINS[3] == [9, 10, 11] and HARD_BINS[4] == [12]


def test_class_scores_take_the_best_profile_inside_each_bin():
    L = np.arange(18, dtype=float)[None, :]
    got = class_scores_hard_bins(L)
    assert got.shape == (1, 10)
    assert got[0].tolist() == [2, 5, 8, 11, 12, 13, 14, 15, 16, 17]


def test_posterior_mean_index_is_the_dominant_index_for_extreme_likelihoods():
    L = np.full((2, 18), -1000.0)
    L[0, 7] = 0.0
    L[1, 3] = L[1, 4] = 0.0
    assert posterior_mean_index(L).tolist() == pytest.approx([7.0, 3.5])


def test_auc_perfect_reversed_and_ties():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_fit_thresholds_separable_data_gets_full_score_and_monotone_cuts():
    s = np.array([0.1, 0.2, 1.1, 1.2, 5.0, 9.0])
    y = np.array([0, 0, 1, 1, 4, 9])
    cuts = fit_thresholds(s, y)
    assert cuts.shape == (9,) and np.all(np.diff(cuts) >= 0)
    pred = apply_thresholds(s, cuts)
    assert pred.tolist() == y.tolist() and mean_score(pred, y) == 1.0


def test_fit_thresholds_keeps_tied_values_in_one_class():
    s = np.array([1.0, 1.0, 1.0, 2.0])
    y = np.array([3, 3, 4, 7])
    pred = apply_thresholds(s, fit_thresholds(s, y))
    assert pred[0] == pred[1] == pred[2]
    # 3,3,4 share a class: predicting 3 scores 1 + 1 + e^-1, the best any single class can do
    assert pred[0] == 3 and pred[3] == 7


def test_bootstrap_ci_brackets_the_mean_and_is_degenerate_for_constant_scores():
    rng = np.random.RandomState(0)
    scores = rng.rand(200)
    players = np.repeat(np.arange(40), 5)
    ranks = np.repeat(np.arange(10), 20)
    lo, hi = bootstrap_ci(scores, players, ranks, n=300, seed=1)
    assert lo < scores.mean() < hi
    assert bootstrap_ci(np.ones(10), np.arange(10), np.zeros(10), n=50) == (1.0, 1.0)


def test_question_matrix_sums_and_per_game_normalised_sums():
    games = pd.DataFrame({"game_key": ["g1", "g2"], "n_moves": [10, 20],
                          **{f"sum_{p}": [-10.0 * (1 + i), -40.0] for i, p in enumerate(["a", "b"])}})
    q = pd.DataFrame({"question_id": ["q"], "game_keys": [["g1", "g2"]]})
    L_sum, L_norm, n_moves = question_matrix(games, q, ["a", "b"])
    assert L_sum.tolist() == [[-50.0, -60.0]]
    assert L_norm.tolist() == [[pytest.approx(-1.0 - 2.0), pytest.approx(-2.0 - 2.0)]]
    assert n_moves.tolist() == [30]
    assert math.isclose(L_norm[0, 0], -3.0)


def test_bootstrap_ci_class_balanced_uses_the_mean_of_class_means():
    # class 0: 9 players scoring 0; class 1: 1 player scoring 1 -> question mean 0.1, class-balanced mean 0.5
    scores = np.array([0.0] * 9 + [1.0])
    players = np.arange(10)
    ranks = np.array([0] * 9 + [1])
    lo, hi = bootstrap_ci(scores, players, ranks, n=200, seed=0, balanced=True)
    assert lo == hi == 0.5
    lo_q, hi_q = bootstrap_ci(scores, players, ranks, n=200, seed=0)
    assert lo_q == hi_q == pytest.approx(0.1)


def test_bootstrap_ci_mean_over_sizes_matches_the_headline_statistic():
    from aicup.evidence import bootstrap_ci_mean_over_sizes

    # two ranks, two players each; size-5 questions score 0.4, size-20 questions score 0.8,
    # and there are three times as many size-5 questions: pooled mean 0.5, mean over sizes 0.6
    players = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    ranks = np.array([0] * 8 + [1] * 8)
    sizes = np.array([5, 5, 5, 20] * 4)
    scores = np.where(sizes == 5, 0.4, 0.8)
    lo, hi = bootstrap_ci_mean_over_sizes(scores, players, ranks, sizes, n=100, seed=0)
    assert lo == pytest.approx(0.6) and hi == pytest.approx(0.6)
