import dataclasses
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import settrain
from aicup.aggregators import DeepSets, PooledDelta, SetTransformer
from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.ladder_data import count_cols, pack_questions
from aicup.metric import RANK_TO_IDX
from aicup.rankmodel import MLPModel, _softmax, _standardise
from aicup.settrain import TrainCfg, crossfit_base_logits, final_loss, fit_set_model, predict_set_model

VECTORS = Path(__file__).resolve().parents[1] / "results" / "ladder" / "vectors"
STAGED, WHOLE_NAMES = vec_names(SEGMENTS), vec_names(WHOLE)
RANKS = list(RANK_TO_IDX)
M = 20


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


def _vec(n_games=120, seed=0):
    """Synthetic vectors with the constant columns of real data: stages 1 and 2 are always present
    with 20 and 30 target moves; s1_mean_12k is the same in every game."""
    rng = np.random.RandomState(seed)
    staged = rng.randn(n_games, len(STAGED)).astype(np.float32)
    s3, s4 = rng.randint(1, 40, size=n_games), rng.randint(0, 60, size=n_games)
    for s, n in ((1, 20), (2, 30), (3, s3), (4, s4)):
        staged[:, STAGED.index(f"s{s}_n_eff")] = n
        staged[:, STAGED.index(f"s{s}_present")] = np.asarray(n) > 0
    staged[np.ix_(s4 == 0, [i for i, name in enumerate(STAGED) if name.startswith("s4_")])] = 0.0
    staged[:, STAGED.index("s1_mean_12k")] = 0.25
    n_moves = (50 + s3 + s4).astype(np.int64)
    keys = np.asarray([f"{RANKS[g % 10]}:g_{g:07d}" for g in range(n_games)])
    return {"game_key": keys, "staged": staged, "whole": rng.randn(n_games, len(WHOLE_NAMES)).astype(np.float32),
            "sums": -rng.rand(n_games, 18) * n_moves[:, None], "n_moves": n_moves}


def _questions(vec, n_q=60, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for j in range(n_q):
        m = int(rng.randint(5, M + 1))
        rows.append({"question_id": f"q{j}", "rank": RANKS[j % 10], "player_id": f"p{j}", "n_games": m,
                     "game_keys": [str(k) for k in vec["game_key"][rng.choice(len(vec["game_key"]), m, replace=False)]]})
    return pd.DataFrame(rows)


def _packed(Q=30, d=6, seed=0):
    """A hand-made packed set: column 2 is constant (7), column 3 is all 0, padding rows are 0."""
    rng = np.random.RandomState(seed)
    n = rng.randint(1, M + 1, size=Q)
    mask = np.arange(M)[None, :] < n[:, None]
    tokens = rng.randn(Q, M, d).astype(np.float32)
    tokens[..., 2], tokens[..., 3] = 7.0, 0.0
    tokens[~mask] = 0.0
    data = {"tokens": tokens, "mask": mask, "qglob": rng.randn(Q, 6).astype(np.float32), "y": np.arange(Q) % 10}
    return data, rng.randn(Q, 10)


def _toy_max(Q=400, d=4, seed=0):
    """Label 1 iff the max of column 0 over the question's games is > 0.5 (exactly one game above,
    at 0.6..1, the rest at 0..0.4), else 0; the other columns are noise."""
    rng = np.random.RandomState(seed)
    n = rng.randint(2, M + 1, size=Q)
    mask = np.arange(M)[None, :] < n[:, None]
    y = rng.randint(0, 2, size=Q)
    tokens = rng.rand(Q, M, d)
    tokens[..., 0] = rng.uniform(0.0, 0.4, size=(Q, M))
    for q in np.flatnonzero(y == 1):
        tokens[q, rng.randint(n[q]), 0] = rng.uniform(0.6, 1.0)
    tokens = (tokens * mask[..., None]).astype(np.float32)
    qglob = np.column_stack([np.log(n), np.zeros((Q, 5))]).astype(np.float32)
    assert np.array_equal((tokens[..., 0].max(axis=1) > 0.5).astype(int), y)
    return {"tokens": tokens, "mask": mask, "qglob": qglob, "y": y.astype(np.int64)}


def test_standardisation_uses_valid_training_tokens_only():
    data, base = _packed()
    mask = data["mask"]
    data["tokens"][~mask] = 1e6  # padding content must not enter the statistics
    before = data["tokens"].copy()
    fitted = fit_set_model(partial(DeepSets, d_tok=8, hidden=8), data, base, TrainCfg(epochs=1, batch=8, device="cpu"))
    assert np.array_equal(data["tokens"], before)  # the caller's tokens are not modified
    mean, std = _standardise(before[mask].astype(np.float64))
    np.testing.assert_allclose(fitted.tok_mean, mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.tok_std, std, rtol=1e-12, atol=1e-12)
    assert fitted.tok_std[2] == 1.0 and fitted.tok_std[3] == 1.0
    for chunk in (1, 3, 7):
        m2, s2 = settrain._token_stats(before, mask, chunk=chunk)
        np.testing.assert_allclose(m2, mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s2, std, rtol=1e-12, atol=1e-12)
    gmean, gstd = _standardise(data["qglob"].astype(np.float64))
    np.testing.assert_allclose(fitted.glob_mean, gmean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.glob_std, gstd, rtol=1e-12, atol=1e-12)

    tok, m, glob = settrain._Inputs(data, fitted, "cpu")(torch.arange(len(mask)))
    assert torch.equal(m, torch.from_numpy(mask))
    assert torch.all(tok[~m] == 0)  # padding is 0 again after standardisation
    np.testing.assert_allclose(tok[m].numpy(), (before[mask] - mean) / std, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(glob.numpy(), (data["qglob"] - gmean) / gstd, rtol=1e-5, atol=1e-5)
    clean = dict(data, tokens=np.where(mask[..., None], before, 0.0).astype(np.float32))
    np.testing.assert_array_equal(predict_set_model(fitted, data, base), predict_set_model(fitted, clean, base))


def test_constant_columns_standardise_finite():
    vec = _vec()
    data = pack_questions(_questions(vec), vec, counts=False)
    valid = data["tokens"][data["mask"]]
    const = np.flatnonzero(valid.min(axis=0) == valid.max(axis=0))
    assert set(count_cols("staged")) | {STAGED.index("s1_mean_12k")} <= set(const.tolist())
    assert np.all(data["qglob"][:, 1:] == 0)
    base = np.log(np.full((len(data["y"]), 10), 0.1))
    for make in (partial(PooledDelta, hidden=16), partial(DeepSets, d_tok=16, hidden=16),
                 partial(SetTransformer, d_tok=16, hidden=16)):
        fitted = fit_set_model(make, data, base, TrainCfg(epochs=1, batch=16, lr=1e-2, device="cpu"))
        assert len(fitted.losses) == 1 and np.all(np.isfinite(fitted.losses))
        assert np.all(fitted.tok_std[const] == 1.0) and np.all(fitted.glob_std[1:] == 1.0)
        P = predict_set_model(fitted, data, base)
        assert np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)


@pytest.mark.skipif(not (VECTORS / "fold1.npz").exists(), reason="results/ladder/vectors/fold1.npz not built")
def test_real_vectors_train_finite():
    vec = dict(np.load(VECTORS / "fold1.npz"))
    rank = np.asarray([k.split(":")[0] for k in vec["game_key"]])
    rng = np.random.RandomState(0)
    rows = []
    for j in range(60):
        pool, m = np.flatnonzero(rank == RANKS[j % 10]), (5, 10, 20)[j % 3]
        rows.append({"question_id": f"q{j}", "rank": RANKS[j % 10], "player_id": f"p{j}", "n_games": m,
                     "game_keys": [str(k) for k in vec["game_key"][rng.choice(pool, m, replace=False)]]})
    qs = pd.DataFrame(rows)
    base = np.log(np.full((len(qs), 10), 0.1))
    for counts in (True, False):
        data = pack_questions(qs, vec, counts=counts)
        fitted = fit_set_model(partial(SetTransformer, d_tok=16, hidden=16), data, base,
                               TrainCfg(epochs=1, batch=16, device="cpu"))
        assert np.all(np.isfinite(fitted.losses)) and np.all(np.isfinite(fitted.tok_std))
        assert fitted.tok_std[STAGED.index("s1_present")] == 1.0
        P = predict_set_model(fitted, data, base)
        assert np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_learns_set_interaction():
    data = _toy_max()
    base = np.zeros((len(data["y"]), 10))
    make = partial(DeepSets, d_tok=32, hidden=64)
    cfg = TrainCfg(epochs=30, batch=32, lr=3e-3, device="cpu")
    f1 = fit_set_model(make, data, base, cfg)
    P1 = predict_set_model(f1, data, base)
    assert np.mean(P1.argmax(axis=1) == data["y"]) > 0.9
    assert f1.losses[-1] < f1.losses[0]
    f2 = fit_set_model(make, data, base, cfg)
    assert f2.losses == f1.losses
    assert np.array_equal(predict_set_model(f2, data, base), P1)
    # the no-encoder control (mean of the raw tokens) does not clear the same bar (0.80-0.84 here)
    fp = fit_set_model(partial(PooledDelta, hidden=64), data, base, cfg)
    assert np.mean(predict_set_model(fp, data, base).argmax(axis=1) == data["y"]) < 0.9


def test_final_loss_value():
    g = torch.Generator().manual_seed(0)
    deltas = torch.randn(4, 3, 10, generator=g)
    logits = torch.randn(4, 10, generator=g)[:, None] + deltas
    y, w = torch.tensor([0, 3, 3, 9]), torch.tensor([0.5, 1.0, 1.0, 1.5])
    got = final_loss(logits, deltas, y, w, 0.1)
    ce = -logits[:, -1].log_softmax(dim=-1)[torch.arange(4), y]
    want = (w * ce).sum() / w.sum() + 0.1 * (deltas[:, -1] ** 2).sum(dim=-1).mean()
    torch.testing.assert_close(got, want)
    logits2, deltas2 = logits.clone(), deltas.clone()
    logits2[:, :2] += 5.0
    deltas2[:, :2] *= 3.0
    torch.testing.assert_close(final_loss(logits2, deltas2, y, w, 0.1), got)  # only the last tick counts


def test_crossfit_never_sees_own_fold():
    made = []

    class Recorder:
        """Fake static model: column 0 of X is the fold id. It records the folds it is fitted on and
        predicts, and its proba encodes the folds it saw (2 for a seen fold id, 1 otherwise, 0 for class 9)."""

        def __init__(self):
            self.seen, self.n_seen, self.predicted = None, 0, []
            made.append(self)

        def fit(self, X, y):
            assert len(X) == len(y)
            self.seen, self.n_seen = set(X[:, 0].astype(int).tolist()), len(X)
            return self

        def predict_proba(self, X):
            self.predicted.append(set(X[:, 0].astype(int).tolist()))
            p = np.array([2.0 if k in self.seen else 1.0 for k in range(9)] + [0.0])
            return np.tile(p / p.sum(), (len(X), 1))

    rng = np.random.RandomState(0)
    sizes = {3: 11, 1: 7, 4: 9, 2: 5}
    X_by_fold = {f: np.column_stack([np.full(n, f), rng.randn(n, 3)]) for f, n in sizes.items()}
    y_by_fold = {f: rng.randint(0, 10, size=n) for f, n in sizes.items()}
    X_eval = np.column_stack([np.zeros(6), rng.randn(6, 3)])  # fold id 0
    base, base_eval = crossfit_base_logits(Recorder, X_by_fold, y_by_fold, X_eval)
    assert sorted(base) == [1, 2, 3, 4] and len(made) == 5
    for m in made:
        assert len(m.predicted) == 1 and not (m.predicted[0] & m.seen)  # never predicts rows it was fitted on
    for f in (1, 2, 3, 4):
        (m,) = [m for m in made if m.predicted[0] == {f}]
        assert m.seen == {1, 2, 3, 4} - {f} and m.n_seen == sum(n for g, n in sizes.items() if g != f)
        assert base[f].shape == (sizes[f], 10)
        assert set(np.flatnonzero(base[f][0] > np.log(1.5 / 13)).tolist()) == {1, 2, 3, 4} - {f}
        assert np.all(base[f][:, 9] == np.log(1e-9))  # log(clip(proba, 1e-9))
    (m,) = [m for m in made if m.predicted[0] == {0}]
    assert m.seen == {1, 2, 3, 4} and m.n_seen == sum(sizes.values()) and base_eval.shape == (6, 10)


def test_crossfit_eval_base_matches_static_fit():
    make = partial(MLPModel, hidden=4, epochs=30)
    rng = np.random.RandomState(0)
    X_by_fold = {f: rng.randn(30 + f, 5) for f in (2, 4, 1)}
    y_by_fold = {f: rng.randint(0, 10, size=len(X)) for f, X in X_by_fold.items()}
    X_eval = rng.randn(12, 5)
    base, base_eval = crossfit_base_logits(make, X_by_fold, y_by_fold, X_eval)
    stack = lambda fs, d: np.concatenate([d[f] for f in fs])
    ref = make().fit(stack((1, 2, 4), X_by_fold), stack((1, 2, 4), y_by_fold)).predict_proba(X_eval)
    assert np.array_equal(base_eval, np.log(np.clip(ref, 1e-9, None)))  # the static rung's fit, row for row
    np.testing.assert_allclose(_softmax(base_eval), ref, rtol=0, atol=1e-12)
    ref2 = make().fit(stack((1, 4), X_by_fold), stack((1, 4), y_by_fold)).predict_proba(X_by_fold[2])
    assert np.array_equal(base[2], np.log(np.clip(ref2, 1e-9, None)))


def test_cfg_and_input_checks():
    assert dataclasses.astuple(TrainCfg()) == (20, 512, 1e-3, 1e-4, 1e-2, 0, "final", "cuda")
    data, base = _packed()
    make, cpu = partial(PooledDelta, hidden=4), TrainCfg(epochs=0, device="cpu")
    with pytest.raises(ValueError, match="loss"):
        fit_set_model(make, data, base, TrainCfg(loss="nope", device="cpu"))
    y = data["y"].copy()
    y[0] = -1
    with pytest.raises(ValueError, match="rank"):
        fit_set_model(make, dict(data, y=y), base, cpu)
    with pytest.raises(ValueError, match="base"):
        fit_set_model(make, data, base[:-1], cpu)
    empty = data["mask"].copy()
    empty[0] = False
    with pytest.raises(ValueError, match="game"):
        fit_set_model(make, dict(data, mask=empty), base, cpu)
    fitted = fit_set_model(make, data, base, cpu)
    with pytest.raises(ValueError, match="pick"):
        predict_set_model(fitted, data, base, pick="nope")
    with pytest.raises(ValueError, match="d_in|columns"):
        predict_set_model(fitted, dict(data, tokens=data["tokens"][..., :-1]), base)
    X, yy = {1: np.zeros((3, 2))}, {1: np.zeros(3, dtype=int)}
    with pytest.raises(ValueError, match="fold"):
        crossfit_base_logits(MLPModel, X, yy, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="fold"):
        crossfit_base_logits(MLPModel, {1: X[1], 2: X[1]}, {1: yy[1], 3: yy[1]}, np.zeros((2, 2)))
