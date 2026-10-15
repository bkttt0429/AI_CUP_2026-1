import numpy as np
import pytest

from aicup.evidence import apply_thresholds, fit_thresholds
from aicup.rankmodel import MLPModel, SoftmaxModel, ThresholdModel, class_weights, from_state


def _separable(n=600, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 10, size=n)
    X = np.eye(10)[y] * 3 + rng.randn(n, 10) * 0.1
    return np.hstack([X, rng.randn(n, 4)]), y


def test_class_weights_equalise_class_totals():
    y = np.array([0, 0, 0, 1])
    w = class_weights(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert w.mean() == pytest.approx(1.0)


def test_softmax_model_learns_separable_classes_and_returns_distributions():
    X, y = _separable()
    m = SoftmaxModel(l2=1e-4).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y), 10) and np.allclose(p.sum(axis=1), 1.0) and np.all(np.isfinite(p))
    assert (p.argmax(axis=1) == y).mean() >= 0.95


def test_mlp_model_is_deterministic_and_learns():
    X, y = _separable(seed=1)
    p1 = MLPModel(hidden=16, epochs=150, seed=3).fit(X, y).predict_proba(X)
    p2 = MLPModel(hidden=16, epochs=150, seed=3).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)
    assert (p1.argmax(axis=1) == y).mean() >= 0.95


def test_threshold_model_matches_fit_thresholds_on_its_feature():
    rng = np.random.RandomState(2)
    s = np.sort(rng.rand(200) * 17)
    y = np.clip((s / 1.8).astype(int), 0, 9)
    X = np.column_stack([rng.rand(200), s])
    m = ThresholdModel(feature=1).fit(X, y)
    expect = apply_thresholds(s, fit_thresholds(s, y))
    assert m.predict_proba(X).argmax(axis=1).tolist() == expect.tolist()


def test_state_round_trip_preserves_predictions():
    X, y = _separable(seed=4)
    for m in (SoftmaxModel(l2=1e-3).fit(X, y), MLPModel(hidden=8, epochs=50, seed=0).fit(X, y), ThresholdModel(feature=0).fit(X, y)):
        again = from_state(m.state_dict())
        assert np.allclose(again.predict_proba(X), m.predict_proba(X), atol=1e-6)
