import math

import numpy as np
import pandas as pd
import pytest

from aicup import qeval
from aicup.metric import LEVELS

# Output of the pre-move `scripts/fit_rank_model.evaluate` on `_inputs()` (seed 20261019, 300 questions).
GOLDEN = {
    "balanced_by_n": {5: 0.552474286041349, 10: 0.5807497995396633, 20: 0.539028803923224},
    "balanced_mean_over_n": 0.5574176298347454,
    "question_mean": 0.5678243273522876,
    "per_class": {"D": 0.6193707047505511, "C": 0.5043182634797637, "B": 0.6079904821977454, "A": 0.47937168627301996, "1D": 0.4991862218481649, "2D": 0.4986437030802747, "3D": 0.6344277001627003, "4D": 0.5411189313721623, "5D": 0.604145532940573, "6D": 0.6587461490867009},
    "exact": 0.52,
    "within1": 0.65,
    "confusion_rows_truth_cols_pred": [
        [15, 3, 0, 1, 2, 2, 0, 1, 0, 2],
        [0, 11, 3, 2, 1, 2, 1, 2, 1, 1],
        [1, 3, 22, 0, 2, 1, 1, 2, 5, 1],
        [0, 1, 4, 10, 4, 2, 0, 0, 2, 4],
        [1, 3, 4, 2, 16, 2, 1, 4, 2, 0],
        [1, 0, 0, 3, 2, 9, 2, 1, 2, 1],
        [2, 1, 3, 1, 3, 2, 21, 3, 0, 0],
        [3, 2, 3, 2, 1, 3, 3, 20, 0, 2],
        [1, 1, 0, 1, 1, 1, 3, 1, 14, 2],
        [1, 3, 0, 1, 0, 1, 1, 1, 3, 18],
    ],
}
# Per-question scores of the same run: 1 -> exact, e -> adjacent (e^-1), 0 -> miss.
GOLDEN_SCORES = (
    "11101111000111e1e10110011011e00000e1111e0010111001001100101e"
    "10e0110e1101e1111e101000e111e1011e1e0100010010000011e01011e1"
    "001000e01001e11e001e0110110e10111111e1001e1e11e011e110101000"
    "00e110110011110101111111111111110e10010e0111011e011110111101"
    "011111010000ee010ee1101000101e1011111e1001e10e111e0010110110"
)


def _inputs(seed=20261019, n=300):
    """Seeded random questions; the recipe that produced `GOLDEN`."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 10, size=n)
    alpha = np.full((n, 10), 0.5)
    alpha[np.arange(n), y] += 1.5
    proba = np.vstack([rng.dirichlet(a) for a in alpha])
    qs = pd.DataFrame({"player_id": [f"p{k}_{i % 7}" for i, k in enumerate(y)],
                       "n_games": rng.choice([5, 10, 20], size=n)})
    return proba, y, qs


def test_evaluate_golden_against_fit_rank_model():
    """`GOLDEN` is the output of the pre-move `fit_rank_model.evaluate`: every original key must match."""
    proba, y, qs = _inputs()
    got, s = qeval.evaluate(proba, y, qs)
    assert set(GOLDEN) <= set(got)
    assert list(got["balanced_by_n"]) == [5, 10, 20]
    assert list(got["per_class"]) == LEVELS
    for key in ("balanced_by_n", "per_class"):
        assert got[key] == pytest.approx(GOLDEN[key], rel=1e-12), key
    for key in ("balanced_mean_over_n", "question_mean", "exact", "within1"):
        assert got[key] == pytest.approx(GOLDEN[key], rel=1e-12), key
    assert got["confusion_rows_truth_cols_pred"] == GOLDEN["confusion_rows_truth_cols_pred"]
    levels = {"1": 1.0, "0": 0.0, "e": math.exp(-1)}
    np.testing.assert_allclose(s, [levels[c] for c in "".join(GOLDEN_SCORES)], rtol=0, atol=1e-15)


def test_paired_identical_is_zero():
    proba, y, qs = _inputs()
    _, s = qeval.evaluate(proba, y, qs)
    got = qeval.paired_diff(s, s.copy(), y, qs["player_id"], qs["n_games"], n=200)
    assert got["diff"] == 0.0
    assert got["ci95"] == [0.0, 0.0]
    assert got["by_n"] == {5: 0.0, 10: 0.0, 20: 0.0}


def test_paired_constant_offset():
    proba, y, qs = _inputs()
    _, s_a = qeval.evaluate(proba, y, qs)
    got = qeval.paired_diff(s_a, s_a - 0.05, y, qs["player_id"], qs["n_games"], n=200)
    assert got["diff"] == pytest.approx(0.05, abs=1e-12)
    assert got["ci95"] == pytest.approx([0.05, 0.05], abs=1e-12)
    assert got["by_n"] == pytest.approx({5: 0.05, 10: 0.05, 20: 0.05}, abs=1e-12)


def test_paired_matches_headline_difference():
    proba_a, y, qs = _inputs()
    proba_b = _inputs(seed=7)[0]
    ev_a, s_a = qeval.evaluate(proba_a, y, qs)
    ev_b, s_b = qeval.evaluate(proba_b, y, qs)
    assert ev_a["balanced_mean_over_n"] != ev_b["balanced_mean_over_n"]
    got = qeval.paired_diff(s_a, s_b, y, qs["player_id"], qs["n_games"], n=200)
    assert got["diff"] == pytest.approx(ev_a["balanced_mean_over_n"] - ev_b["balanced_mean_over_n"], abs=1e-12)
    assert got["by_n"] == pytest.approx({m: ev_a["balanced_by_n"][m] - ev_b["balanced_by_n"][m] for m in (5, 10, 20)}, abs=1e-12)
    assert got["diff"] == pytest.approx(np.mean(list(got["by_n"].values())), abs=1e-12)
    assert qeval.balanced_mean_over_n(s_a, y, qs["n_games"]) == pytest.approx(ev_a["balanced_mean_over_n"], abs=1e-12)
    lo, hi = got["ci95"]
    assert lo <= hi
    assert lo < got["diff"] < hi
    assert got["ci95"] == qeval.paired_diff(s_a, s_b, y, qs["player_id"], qs["n_games"], n=200)["ci95"]


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:invalid value encountered")
def test_nll_brier_values():
    proba = np.zeros((2, 10))
    proba[0, :3] = [0.5, 0.3, 0.2]
    proba[1, :3] = [0.1, 0.6, 0.3]
    y = np.array([0, 1])
    qs = pd.DataFrame({"player_id": ["a", "b"], "n_games": [5, 5]})
    got, _ = qeval.evaluate(proba, y, qs)
    assert got["nll_balanced"] == pytest.approx((-np.log(0.5) - np.log(0.6)) / 2, abs=1e-12)
    brier_0 = 0.5 ** 2 + 0.3 ** 2 + 0.2 ** 2
    brier_1 = 0.1 ** 2 + 0.4 ** 2 + 0.3 ** 2
    assert got["brier"] == pytest.approx((brier_0 + brier_1) / 2, abs=1e-12)


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:invalid value encountered")
def test_nll_brier_are_class_balanced():
    """Three class-0 questions weigh as much as one class-1 question."""
    proba = np.zeros((4, 10))
    proba[:3, 0], proba[:3, 1] = 0.9, 0.1
    proba[3, 0], proba[3, 1] = 0.5, 0.5
    y = np.array([0, 0, 0, 1])
    qs = pd.DataFrame({"player_id": list("abcd"), "n_games": [5, 5, 5, 5]})
    got, _ = qeval.evaluate(proba, y, qs)
    assert got["nll_balanced"] == pytest.approx((-np.log(0.9) - np.log(0.5)) / 2, abs=1e-12)
    assert got["brier"] == pytest.approx((0.02 + 0.5) / 2, abs=1e-12)


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:invalid value encountered")
def test_nll_is_infinite_for_a_one_hot_miss_and_brier_is_two():
    """One-hot models (ThresholdModel) have no probability on a wrong class: NLL is inf, Brier 2 per miss."""
    proba = np.eye(10)[[0, 0]]
    y = np.array([0, 1])
    qs = pd.DataFrame({"player_id": ["a", "b"], "n_games": [5, 5]})
    got, _ = qeval.evaluate(proba, y, qs)
    assert got["nll_balanced"] == np.inf
    assert got["brier"] == pytest.approx((0.0 + 2.0) / 2, abs=1e-12)
