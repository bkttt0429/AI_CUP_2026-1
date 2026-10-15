import math

import numpy as np
import pytest

from aicup.metric import LEVELS, expected_score_decision, mean_score, score


def test_levels_order_weak_to_strong():
    assert LEVELS == ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def test_score_exact_adjacent_other():
    assert score(3, 3) == 1.0
    assert score(3, 4) == pytest.approx(math.exp(-1))
    assert score(4, 3) == pytest.approx(math.exp(-1))
    assert score(3, 5) == 0.0
    assert score(0, 9) == 0.0


def test_mean_score():
    assert mean_score([0, 1, 2], [0, 2, 4]) == pytest.approx((1 + math.exp(-1) + 0) / 3)


def test_expected_score_decision_prefers_neighbour_mass():
    p = np.array([0.30, 0.30, 0.40, 0, 0, 0, 0, 0, 0, 0])
    # index 1: 0.30 + e^-1 (0.30 + 0.40) = 0.557 beats index 2: 0.40 + e^-1 * 0.30 = 0.510
    assert expected_score_decision(p) == 1


def test_expected_score_decision_edges_have_one_neighbour():
    assert expected_score_decision(np.eye(10)[9]) == 9
    assert expected_score_decision(np.eye(10)[0]) == 0
