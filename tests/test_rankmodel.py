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


def test_ordinal_targets_shape_mass_and_edges():
    from aicup.rankmodel import ordinal_targets
    T = ordinal_targets([0, 4, 9], 0.2)
    assert T.shape == (3, 10) and np.allclose(T.sum(axis=1), 1.0)
    assert T[0, 0] == pytest.approx(0.8) and T[0, 1] == pytest.approx(0.2)  # the edge class gives its whole eps to its one neighbour
    assert T[1, 4] == pytest.approx(0.8) and T[1, 3] == pytest.approx(0.1) and T[1, 5] == pytest.approx(0.1)
    assert T[2, 9] == pytest.approx(0.8) and T[2, 8] == pytest.approx(0.2) and T[2, 7] == 0.0
    assert np.array_equal(ordinal_targets([3, 7], 0.0), np.eye(10)[[3, 7]])
    with pytest.raises(ValueError):
        ordinal_targets([1], 1.0)
    with pytest.raises(ValueError):
        MLPModel(label_smoothing=-0.1)


def test_mlp_label_smoothing_zero_is_bit_identical_and_positive_spreads_mass_to_the_neighbours():
    X, y = _separable(seed=5)
    plain = MLPModel(hidden=16, epochs=100, seed=1).fit(X, y)
    zero = MLPModel(hidden=16, epochs=100, seed=1, label_smoothing=0.0).fit(X, y)
    assert np.array_equal(plain.predict_proba(X), zero.predict_proba(X))
    smooth = MLPModel(hidden=16, epochs=100, seed=1, label_smoothing=0.2).fit(X, y)
    p, q = plain.predict_proba(X), smooth.predict_proba(X)
    assert (q.argmax(axis=1) == y).mean() >= 0.95
    rows = np.arange(len(y))
    neigh = lambda P: np.array([P[i, max(y[i] - 1, 0)] * (y[i] > 0) + P[i, min(y[i] + 1, 9)] * (y[i] < 9) for i in rows])  # noqa: E731
    assert neigh(q).mean() > neigh(p).mean() and q[rows, y].mean() < p[rows, y].mean()
    # the state carries the setting; an old document without it is a plain MLP
    d = smooth.state_dict()
    assert d["label_smoothing"] == 0.2 and from_state(d).label_smoothing == 0.2 and np.allclose(from_state(d).predict_proba(X), q, atol=1e-6)
    old = {k: v for k, v in plain.state_dict().items() if k != "label_smoothing"}
    assert from_state(old).label_smoothing == 0.0 and np.allclose(from_state(old).predict_proba(X), p, atol=1e-6)


def test_mlp_l1_and_early_stopping_defaults_are_bit_identical_and_options_act():
    X, y = _separable(seed=5)
    base = MLPModel(hidden=16, epochs=120, seed=2).fit(X, y)
    same = MLPModel(hidden=16, epochs=120, seed=2, l1=0.0).fit(X, y)
    assert np.array_equal(base.predict_proba(X), same.predict_proba(X)) and base.best_epoch is None and same.state_dict()["l1"] == 0.0
    # L1 on the input layer shrinks its weights
    sparse = MLPModel(hidden=16, epochs=120, seed=2, l1=5e-2).fit(X, y)
    assert float(sparse.net[0].weight.abs().sum()) < float(base.net[0].weight.abs().sum())
    assert (sparse.predict_proba(X).argmax(axis=1) == y).mean() >= 0.9
    # early stopping on a validation set keeps the best evaluation and records where it stopped
    Xv, yv = _separable(seed=6)
    es = MLPModel(hidden=16, epochs=120, seed=2).fit(X, y, val=(Xv, yv), eval_every=10, patience=2)
    assert es.best_epoch is not None and es.best_epoch % 10 == 0 and 10 <= es.best_epoch <= 120 and np.isfinite(es.val_ce)
    # with random labels on the validation set the stop comes early
    rng = np.random.RandomState(0)
    stop = MLPModel(hidden=16, epochs=300, seed=2).fit(X, y, val=(Xv, rng.randint(0, 10, size=len(yv))), eval_every=5, patience=2)
    assert stop.best_epoch is not None and stop.best_epoch < 300
    again = from_state(es.state_dict())
    assert np.allclose(again.predict_proba(X), es.predict_proba(X), atol=1e-6) and again.best_epoch == es.best_epoch
    with pytest.raises(ValueError):
        MLPModel(l1=-1.0)
