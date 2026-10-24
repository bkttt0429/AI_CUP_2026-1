import dataclasses
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import settrain
from aicup.aggregators import CTM, DeepSets, PooledDelta, SetTransformer, XAttnGRU
from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.ladder_data import count_cols, pack_questions
from aicup.metric import RANK_TO_IDX
from aicup.rankmodel import MLPModel, _softmax, _standardise, class_weights
from aicup.settrain import (TrainCfg, crossfit_base_logits, ctm_minmax_loss, final_loss, fit_set_model,
                            minmax_ticks, predict_set_model)

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
    fitted = fit_set_model(partial(DeepSets, d_tok=8, hidden=8), data, base,
                           TrainCfg(epochs=1, batch=8, device="cpu", inner_frac=0))
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
        fitted = fit_set_model(make, data, base, TrainCfg(epochs=1, batch=16, lr=1e-2, device="cpu", inner_frac=0))
        assert len(fitted.losses) == 1 and np.all(np.isfinite(fitted.losses))
        assert np.all(fitted.tok_std[const] == 1.0) and np.all(fitted.glob_std[1:] == 1.0)
        P = predict_set_model(fitted, data, base)
        assert np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_non_finite_standardisation_is_refused():
    data, base = _packed()
    make, cpu = partial(PooledDelta, hidden=4), TrainCfg(epochs=0, device="cpu", inner_frac=0)
    for value in (np.nan, np.inf):
        tokens = data["tokens"].copy()
        tokens[0, 0, 1] = value  # a valid game: its statistics (mean, std) are not finite
        with pytest.raises(ValueError, match=r"token standardisation statistics are not finite in columns \[1\]"):
            fit_set_model(make, dict(data, tokens=tokens), base, cpu)
        qglob = data["qglob"].copy()
        qglob[3, 4] = value
        with pytest.raises(ValueError, match=r"qglob standardisation statistics are not finite in columns \[4\]"):
            fit_set_model(make, dict(data, qglob=qglob), base, cpu)
    tokens = data["tokens"].copy()
    tokens[~data["mask"]] = np.nan  # padding never enters the statistics
    fitted = fit_set_model(make, dict(data, tokens=tokens), base, cpu)
    assert np.all(np.isfinite(fitted.tok_mean)) and np.all(np.isfinite(fitted.tok_std))


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
    cfg = TrainCfg(epochs=30, batch=32, lr=3e-3, device="cpu", inner_frac=0)
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


def _minmax_case():
    """Hand-made logits [4, 4, 10] whose min-CE tick t1 and max-certainty tick t2 are known:
    q0 distinct ticks (t1 = 1, t2 = 2); q1 exact ties, ticks 0 = 2 at the lowest CE and 1 = 3 at the
    highest certainty (t1 = 2, t2 = 3: the later tick); q2 all ticks equal (t1 = t2 = 3); q3 one
    tick both most certain and best (t1 = t2 = 0). The deltas are not base-consistent on purpose:
    tied ticks get different ‖Δ‖², so the loss value shows which of the tied ticks was taken."""
    L = torch.zeros(4, 4, 10, dtype=torch.float64)
    y = torch.tensor([2, 0, 5, 9])
    L[0, 1, 2], L[0, 2, 5], L[0, 3, 2] = 3.0, 6.0, 1.0  # tick 0 uniform; 1 good; 2 confident but wrong; 3 weak
    L[1, 0, 0] = L[1, 2, 0] = 2.0  # lowest CE, tie
    L[1, 1, 7] = L[1, 3, 7] = 6.0  # highest certainty, tie
    L[2] = torch.randn(10, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    L[3, 0, 9], L[3, 1, 9], L[3, 2, 4], L[3, 3, 9] = 8.0, 1.0, 0.5, 2.0
    deltas = torch.arange(1, 4 * 4 * 10 + 1, dtype=torch.float64).reshape(4, 4, 10) / 100.0
    return L, deltas, y, torch.tensor([1, 2, 3, 0]), torch.tensor([2, 3, 3, 0])  # t1, t2


def test_minmax_loss_picks_ticks():
    L, deltas, y, t1_want, t2_want = _minmax_case()
    w = torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float64)
    ce = -L.log_softmax(dim=-1)[torch.arange(4), :, y]  # [4, T]
    p = L.softmax(dim=-1)
    cert = 1.0 + (p * p.log()).sum(dim=-1) / np.log(10)  # 1 − H / log 10
    assert torch.equal(ce[1, 0], ce[1, 2]) and torch.equal(cert[1, 1], cert[1, 3])  # the ties are exact
    assert torch.all(ce[[0, 1, 2, 3], t1_want] == ce.min(dim=1).values)
    assert torch.all(cert[[0, 1, 2, 3], t2_want] == cert.max(dim=1).values)
    t1, t2 = minmax_ticks(L, y)
    assert t1.tolist() == t1_want.tolist() and t2.tolist() == t2_want.tolist()
    rows = torch.arange(4)
    pen = (deltas[rows, t1_want].pow(2).sum(-1) + deltas[rows, t2_want].pow(2).sum(-1)) / 2
    want = (w * (ce[rows, t1_want] + ce[rows, t2_want]) / 2).sum() / w.sum() + 0.1 * pen.mean()
    got = ctm_minmax_loss(L, deltas, y, w, 0.1)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    early = (deltas[rows, [1, 0, 0, 0]].pow(2).sum(-1) + deltas[rows, [2, 1, 0, 0]].pow(2).sum(-1)) / 2
    assert abs(float(early.mean() - pen.mean())) > 1e-3  # earlier-tick tie-breaking would give another value
    # T = 1, or all ticks equal (Δ ≡ 0 at initialisation): the same as the final-tick loss
    torch.testing.assert_close(ctm_minmax_loss(L[:, -1:], deltas[:, -1:], y, w, 0.1),
                               final_loss(L[:, -1:], deltas[:, -1:], y, w, 0.1), rtol=1e-12, atol=1e-12)
    same = L[:, -1:].expand(-1, 4, -1)
    torch.testing.assert_close(ctm_minmax_loss(same, deltas, y, w, 0.1), final_loss(same, deltas, y, w, 0.1),
                               rtol=1e-12, atol=1e-12)


def test_pick_certain_per_question():
    L, _, _, _, t2_want = _minmax_case()
    base = np.random.RandomState(0).randn(4, 10)
    logits = base[:, None, :] + L.numpy()
    picked = settrain.PICKS["certain"](logits)
    np.testing.assert_array_equal(picked, logits[np.arange(4), t2_want.numpy()])  # label-free, ties -> later tick
    np.testing.assert_array_equal(settrain.PICKS["final"](logits), logits[:, -1])
    # a visible tie: ticks 0 and 1 differ by an exact integer shift, same softmax and the same certainty to the bit
    shifted = np.zeros((1, 3, 10))
    shifted[0, 0, 3], shifted[0, 1], shifted[0, 2, 3] = 4.0, 2.0, 1.0
    shifted[0, 1, 3] = 6.0
    cert = settrain.certainty(torch.from_numpy(shifted))[0]
    assert cert[0] == cert[1] > cert[2]
    np.testing.assert_array_equal(settrain.PICKS["certain"](shifted), shifted[:, 1])

    class Fixed(torch.nn.Module):
        """Returns the Δ of question qglob[:, 0] (passed through standardisation unchanged)."""

        def __init__(self, table):
            super().__init__()
            self.table = torch.nn.Parameter(table.float(), requires_grad=False)

        def forward(self, tokens, mask, qglob, ticks=None):
            return self.table[qglob[:, 0].round().long()]

    data, _ = _packed(Q=4)
    data["qglob"][:, 0] = np.arange(4)
    fitted = settrain.Fitted(Fixed(L), np.zeros(6), np.ones(6), np.zeros(6), np.ones(6), TrainCfg(batch=3, device="cpu"))
    np.testing.assert_allclose(predict_set_model(fitted, data, base, pick="certain"),
                               _softmax(base + L.numpy()[np.arange(4), t2_want.numpy()]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(predict_set_model(fitted, data, base), _softmax(base + L.numpy()[:, -1]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("loss", ["final", "ctm_minmax"])
def test_fit_applies_class_weights(loss):
    """epochs = 1 and batch >= Q: one forward pass with Δ ≡ 0 (zero-initialised output), so the
    recorded loss is the class-weighted CE of the base logits."""
    data, base = _packed(Q=30)
    y = np.array([0] * 18 + [1] * 6 + [2] * 3 + [3] * 2 + [4])
    data["y"] = y
    cfg = TrainCfg(epochs=1, batch=64, loss=loss, device="cpu", inner_frac=0)
    fitted = fit_set_model(partial(XAttnGRU, d_tok=8, d_hidden=8, heads=2, ticks=3), data, base, cfg)
    ce = -torch.tensor(base).log_softmax(dim=-1)[torch.arange(30), torch.tensor(y)].numpy()
    w = class_weights(y)
    assert len(fitted.losses) == 1
    np.testing.assert_allclose(fitted.losses[0], (w * ce).sum() / w.sum(), rtol=1e-5)
    assert abs((w * ce).sum() / w.sum() - ce.mean()) > 1e-2  # the weights matter on this y


def test_ctm_minmax_training_and_certain_pick():
    data, base = _packed(Q=24)
    make = partial(CTM, d_tok=8, d_latent=16, history=4, nlm_hidden=4, heads=2, ticks=3, n_sync_action=12, n_sync_out=12)
    cfg = TrainCfg(epochs=2, batch=8, lr=1e-2, loss="ctm_minmax", pick="certain", device="cpu", inner_frac=0)
    fitted = fit_set_model(make, data, base, cfg)
    assert len(fitted.losses) == 2 and np.all(np.isfinite(fitted.losses))
    P_by_pick = {}
    for pick in ("final", "certain"):
        P = P_by_pick[pick] = predict_set_model(fitted, data, base, pick=pick)
        assert P.shape == (24, 10) and np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert not np.allclose(P, _softmax(base))
    # the default pick is the TrainCfg's: the tick the model is selected at is the tick it is scored at
    np.testing.assert_array_equal(predict_set_model(fitted, data, base), P_by_pick["certain"])
    assert not np.allclose(P_by_pick["certain"], P_by_pick["final"])
    r = fitted.model.r_act.detach()
    assert float(r.min()) >= 0.0 and 0.0 < float(r.max()) <= 15.0  # trained off 0, projected again by predict's forward


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
    assert dataclasses.astuple(TrainCfg()) == (10, 512, 1e-3, 1e-4, 1e-2, 0, "final", "final", "cuda", 0.1, 4, 8)
    assert settrain.INNER_SEED == 2026
    data, base = _packed()
    make, cpu = partial(PooledDelta, hidden=4), TrainCfg(epochs=0, device="cpu", inner_frac=0)
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


# ---------------------------------------------------------------- inner early stopping

def _player_sets(ranks, q_per_player=8, n_games=20, d=8, seed=0):
    """Packed questions of synthetic players, one per entry of `ranks` (its rank), player_id p000, p001, ...:
    every player has n_games fixed random tokens and each of its questions is a random subset of 5..20 of
    them, so the same games recur across a player's questions with one label, as in the real training
    questions. qglob holds log m, then zeros."""
    rng = np.random.RandomState(seed)
    Q = len(ranks) * q_per_player
    tokens, mask = np.zeros((Q, M, d), dtype=np.float32), np.zeros((Q, M), dtype=bool)
    qglob, pid, y = np.zeros((Q, 6), dtype=np.float32), [], []
    for i, r in enumerate(ranks):
        games = rng.randn(n_games, d)
        for q in range(i * q_per_player, (i + 1) * q_per_player):
            m = rng.randint(5, M + 1)
            tokens[q, :m], mask[q, :m], qglob[q, 0] = games[rng.choice(n_games, m, replace=False)], True, np.log(m)
            pid.append(f"p{i:03d}")
            y.append(int(r))
    return {"tokens": tokens, "mask": mask, "qglob": qglob, "y": np.asarray(y, dtype=np.int64), "player_id": np.asarray(pid)}


def _eval_epochs(n_train, cfg):
    """The completed epochs at each inner evaluation of a run to cfg.epochs on n_train training questions:
    0, then after every ceil(n_batches / evals_per_epoch) batches and at each epoch's end (once per step)."""
    n_batches = -(-n_train // cfg.batch)
    every = -(-n_batches // cfg.evals_per_epoch)
    at = [b for b in range(1, n_batches + 1) if b % every == 0 or b == n_batches]
    return [0.0] + [e + b / n_batches for e in range(cfg.epochs) for b in at]


def _weighted_ce(logits, y):
    """By hand: sum w CE / sum w of log_softmax(logits) with class_weights(y)."""
    w = class_weights(y)
    ce = -torch.tensor(logits, dtype=torch.float64).log_softmax(dim=-1).numpy()[np.arange(len(y)), y]
    return float((w * ce).sum() / w.sum())


def _inner_ce(fitted, data, base):
    """The inner CE of a fitted model, recomputed through predict_set_model on its inner questions."""
    held = settrain.inner_players(data["player_id"], data["y"], fitted.cfg.inner_frac)
    y = data["y"][held]
    P = predict_set_model(fitted, {k: data[k][held] for k in ("tokens", "mask", "qglob")}, base[held])
    w = class_weights(y)
    return float((w * -np.log(P[np.arange(len(y)), y])).sum() / w.sum())


def test_inner_players_grouped_stratified_seed_free():
    rng = np.random.RandomState(0)
    n_players = {0: 1, 1: 2, 2: 3, 3: 10, 4: 11, 5: 30, 7: 45}  # rank -> players
    # ceil(0.1 n), none of a lone player; 0.1 * 30 is 3.0000000000000004 in floating point, still 3 players
    want = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 7: 5}
    pid, y = [], []
    for r, n in n_players.items():
        for j in range(n):
            k = rng.randint(1, 6)  # the player's questions
            pid += [f"r{r}p{j:02d}"] * k
            y += [r] * k
    order = rng.permutation(len(pid))
    pid, y = np.asarray(pid)[order], np.asarray(y)[order]
    inner = settrain.inner_players(pid, y, 0.1)
    assert inner.dtype == bool and inner.shape == pid.shape
    assert not set(pid[inner]) & set(pid[~inner])  # grouped by player: nobody on both sides
    for r, n in n_players.items():
        assert len(set(pid[inner & (y == r)])) == want[r], r
        assert len(set(pid[~inner & (y == r)])) == n - want[r] >= 1, r  # never all of a rank's players
    # only the players and ranks matter: not the order of the questions; the default seed is INNER_SEED
    perm = rng.permutation(len(pid))
    np.testing.assert_array_equal(settrain.inner_players(pid[perm], y[perm], 0.1), inner[perm])
    np.testing.assert_array_equal(settrain.inner_players(pid, y, 0.1, seed=settrain.INNER_SEED), inner)
    assert not np.array_equal(settrain.inner_players(pid, y, 0.1, seed=1), inner)
    assert not settrain.inner_players(pid, y, 0.0).any()
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError, match="inner_frac"):
            settrain.inner_players(pid, y, bad)
    with pytest.raises(ValueError, match="more than one rank"):
        settrain.inner_players(np.array(["a", "a", "b", "b"]), np.array([0, 1, 1, 1]), 0.5)

    # fit_set_model holds out the same players whatever cfg.seed: evaluation 0 (Δ ≡ 0) is the class-weighted
    # CE of the base on exactly those questions for every seed, and the seeds train differently after it
    data = _player_sets(np.arange(40) % 4)
    base = rng.randn(len(data["y"]), 10)
    held = settrain.inner_players(data["player_id"], data["y"], 0.1)
    fits = [fit_set_model(partial(DeepSets, d_tok=8, hidden=8), data, base,
                          TrainCfg(epochs=1, batch=32, lr=1e-2, seed=s, device="cpu")) for s in (0, 1)]
    for f in fits:
        assert f.inner_ce[0] == pytest.approx(_weighted_ce(base[held], data["y"][held]), abs=1e-12)
        assert f.n_inner_players == 4 and f.n_inner_questions == held.sum() == 32
    assert fits[0].inner_ce[0] == fits[1].inner_ce[0] and fits[0].inner_ce[1:] != fits[1].inner_ce[1:]


def test_early_stop_restores_best_checkpoint():
    """Memorisable players (fixed random tokens per player, random ranks 0 / 1, base 0): the inner CE first
    falls -- the shared class prior (only ranks 0 and 1 occur) is learnt -- then rises as the training
    players' games are memorised."""
    data = _player_sets(np.random.RandomState(1).randint(0, 2, size=60))
    base = np.zeros((len(data["y"]), 10))
    make = partial(DeepSets, d_tok=32, hidden=64)
    cfg = TrainCfg(epochs=40, batch=32, lr=2e-3, device="cpu", patience=4)
    f = fit_set_model(make, data, base, cfg)
    held = settrain.inner_players(data["player_id"], data["y"], cfg.inner_frac)
    best, n_eval = f.best_eval, len(f.inner_ce)
    assert f.stopped_early and 0 < best < n_eval - 1 and n_eval - 1 - best == cfg.patience
    assert f.inner_ce[best] < f.inner_ce[0] - 1.0 and f.inner_ce[-1] > f.inner_ce[best]  # fell, then rose
    assert f.inner_ce.index(min(f.inner_ce)) == best  # the lowest (the earliest of equals)
    at = _eval_epochs(int((~held).sum()), cfg)
    assert f.best_epoch == at[best] and at[n_eval - 1] < cfg.epochs
    assert len(f.losses) == math.ceil(at[n_eval - 1])  # one mean per epoch begun, the last one partial
    assert np.all(np.isfinite(f.losses))
    # the returned model is the best checkpoint, not the last weights
    ce = _inner_ce(f, data, base)
    assert abs(ce - f.inner_ce[best]) < 1e-6 and abs(ce - f.inner_ce[-1]) > 1e-3
    assert f.n_inner_players == len(set(data["player_id"][held])) and f.n_inner_questions == held.sum()
    # deterministic on the CPU, checkpoint and all
    g = fit_set_model(make, data, base, cfg)
    assert (g.inner_ce, g.losses, g.best_eval, g.best_epoch) == (f.inner_ce, f.losses, f.best_eval, f.best_epoch)
    assert np.array_equal(predict_set_model(g, data, base), predict_set_model(f, data, base))
    # a patience longer than the run: every epoch and every evaluation of the schedule, still the best checkpoint
    long_cfg = dataclasses.replace(cfg, epochs=6, patience=1000)
    long = fit_set_model(make, data, base, long_cfg)
    assert not long.stopped_early and len(long.losses) == 6
    assert len(long.inner_ce) == len(_eval_epochs(int((~held).sum()), long_cfg))
    assert abs(_inner_ce(long, data, base) - min(long.inner_ce)) < 1e-6


def test_eval0_fallback_returns_base():
    """Training that can only make the inner CE worse: every question shows the same games, so the tokens
    say nothing about the rank and Δ is one vector c for all questions. The training questions' base is
    mis-calibrated (b = linspace(-1, 1)), so training moves c towards -b; the inner questions' base is
    uniform and their 10 ranks balanced, so their CE is logsumexp(c) - mean(c) >= log 10 -- the CE of
    evaluation 0 (Δ ≡ 0: the zero-initialised output layer) -- with equality only for a constant c."""
    data = _player_sets(np.arange(50) % 10)
    for k in ("tokens", "mask", "qglob"):
        data[k][:] = data[k][0]
    held = settrain.inner_players(data["player_id"], data["y"], 0.2)
    base = np.where(held[:, None], 0.0, np.linspace(-1.0, 1.0, 10)[None, :])
    cfg = TrainCfg(epochs=10, batch=32, lr=1e-2, device="cpu", inner_frac=0.2, patience=4)
    f = fit_set_model(partial(DeepSets, d_tok=32, hidden=64), data, base, cfg)
    assert f.inner_ce[0] == pytest.approx(np.log(10), abs=1e-12)
    assert f.best_eval == 0 and f.best_epoch == 0.0 and f.stopped_early
    assert len(f.inner_ce) == cfg.patience + 1 and min(f.inner_ce[1:]) > f.inner_ce[0] + 1e-3  # trained, only worse
    np.testing.assert_allclose(predict_set_model(f, data, base), _softmax(base), rtol=0, atol=1e-12)


def test_inner_frac_zero_is_fixed_epochs():
    rng = np.random.RandomState(3)
    data = _player_sets(np.array([0] * 12 + [1] * 8 + [2] * 5))
    base = rng.randn(len(data["y"]), 10)
    make = partial(DeepSets, d_tok=8, hidden=16)
    cfg = TrainCfg(epochs=3, batch=32, lr=1e-2, device="cpu", inner_frac=0)
    f = fit_set_model(make, data, base, cfg)
    assert len(f.losses) == cfg.epochs and f.inner_ce == []
    assert (f.best_eval, f.best_epoch, f.stopped_early, f.n_inner_players, f.n_inner_questions) == (None,) * 5
    P = predict_set_model(f, data, base)
    assert not np.allclose(P, _softmax(base), atol=1e-3)  # the last weights, trained
    # no checkpoint is chosen: patience and evals_per_epoch play no part; every question trains and enters the statistics
    g = fit_set_model(make, data, base, dataclasses.replace(cfg, patience=1, evals_per_epoch=1))
    assert g.losses == f.losses and np.array_equal(predict_set_model(g, data, base), P)
    np.testing.assert_allclose(f.tok_mean, _standardise(data["tokens"][data["mask"]].astype(np.float64))[0],
                               rtol=1e-12, atol=1e-12)
    # data without player_id (hand-made sets) trains only with inner_frac = 0
    nopid = {k: v for k, v in data.items() if k != "player_id"}
    assert fit_set_model(make, nopid, base, cfg).losses == f.losses
    with pytest.raises(ValueError, match="player_id"):
        fit_set_model(make, nopid, base, dataclasses.replace(cfg, inner_frac=0.1))
    # inner_frac > 0: statistics, class weights and training steps use the training players only -- one step
    # over them all (batch >= their questions) records the class-weighted CE of their base
    h = fit_set_model(make, data, base, dataclasses.replace(cfg, inner_frac=0.1, epochs=1, batch=1024))
    out = ~settrain.inner_players(data["player_id"], data["y"], 0.1)
    assert h.n_inner_players == 2 + 1 + 1 and h.n_inner_questions == (~out).sum()
    tok_mean, tok_std = _standardise(data["tokens"][out][data["mask"][out]].astype(np.float64))
    glob_mean, glob_std = _standardise(data["qglob"][out].astype(np.float64))
    for got, want in ((h.tok_mean, tok_mean), (h.tok_std, tok_std), (h.glob_mean, glob_mean), (h.glob_std, glob_std)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert len(h.losses) == 1
    np.testing.assert_allclose(h.losses[0], _weighted_ce(base[out], data["y"][out]), rtol=1e-5)
    # no rank with two players: nothing to hold out
    lone = _player_sets(np.arange(10))
    with pytest.raises(ValueError, match="inner validation players"):
        fit_set_model(make, lone, np.zeros((len(lone["y"]), 10)), dataclasses.replace(cfg, inner_frac=0.1))


def test_non_finite_inner_ce_is_refused():
    class NaNEval(torch.nn.Module):
        """Δ = 0 in training mode, NaN in eval mode (where the inner CE is computed)."""

        def __init__(self, d_in):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, tokens, mask, qglob, ticks=None):
            return self.w.expand(len(tokens), 1, 10) + (0.0 if self.training else float("nan"))

    data = _player_sets(np.arange(20) % 2)
    with pytest.raises(FloatingPointError, match="non-finite inner validation CE"):
        fit_set_model(NaNEval, data, np.zeros((len(data["y"]), 10)), TrainCfg(epochs=1, batch=32, device="cpu"))


class _TableDelta(torch.nn.Module):
    """Returns the Δ [T, 10] of question qglob[:, 0] from a table (qglob passes standardisation unchanged
    with mean 0 and std 1), so base + Δ is known per question and tick."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.nn.Parameter(table.float(), requires_grad=False)

    def forward(self, tokens, mask, qglob, ticks=None):
        return self.table[qglob[:, 0].round().long()]


def test_early_stop_ce_follows_cfg_pick():
    """The inner CE is read at the tick of cfg.pick -- the last tick, or each question's most certain tick
    (label-free) -- the tick predict_set_model then uses by default."""
    rng = np.random.RandomState(5)
    Q, T = 6, 3
    data, base = _packed(Q=Q)
    data["qglob"][:, 0] = np.arange(Q)
    y = data["y"] = np.array([0, 1, 2, 0, 1, 2])
    table = torch.tensor(rng.randn(Q, T, 10) * 3.0).float()
    logits = base[:, None, :] + table.double().numpy()  # [Q, T, 10], what the model and the base give
    w = class_weights(y)
    want = {}
    for pick in ("final", "certain"):
        cfg = TrainCfg(batch=4, device="cpu", pick=pick)
        fitted = settrain.Fitted(_TableDelta(table), np.zeros(6), np.ones(6), np.zeros(6), np.ones(6), cfg)
        early = settrain._EarlyStop(fitted.model, settrain._Inputs(data, fitted, "cpu"), base, y, np.arange(Q), cfg, fitted)
        picked = settrain.PICKS[pick](logits)
        shifted = picked - picked.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        want[pick] = float((w * -logp[np.arange(Q), y]).sum() / w.sum())
        assert early.ce() == pytest.approx(want[pick], abs=1e-9), pick
        np.testing.assert_allclose(predict_set_model(fitted, data, base), _softmax(picked), rtol=0, atol=1e-6)
    assert abs(want["final"] - want["certain"]) > 1e-3  # the table makes the two ticks differ
    assert settrain._pick_torch(torch.from_numpy(logits), "final").numpy().tolist() == logits[:, -1].tolist()
    with pytest.raises(ValueError, match="pick"):
        settrain._pick_torch(torch.from_numpy(logits), "middle")


def test_cfg_pick_defaults_and_validation():
    """Evaluation 0 (Δ ≡ 0: every tick equal) gives the base's CE whatever the pick; the fitted model predicts
    at its cfg.pick by default; an unknown pick is refused by fit_set_model and predict_set_model."""
    rng = np.random.RandomState(2)
    data = _player_sets(np.arange(40) % 4)
    base = rng.randn(len(data["y"]), 10)
    held = settrain.inner_players(data["player_id"], data["y"], 0.1)
    make = partial(XAttnGRU, d_tok=8, d_hidden=8, heads=2, ticks=3)
    for pick in ("final", "certain"):
        f = fit_set_model(make, data, base, TrainCfg(epochs=1, batch=32, lr=1e-2, device="cpu", pick=pick))
        assert f.cfg.pick == pick and dataclasses.asdict(f.cfg)["pick"] == pick  # recorded in the provenance
        assert f.inner_ce[0] == pytest.approx(_weighted_ce(base[held], data["y"][held]), abs=1e-12)
        np.testing.assert_array_equal(predict_set_model(f, data, base), predict_set_model(f, data, base, pick=pick))
    assert TrainCfg().pick == "final"
    with pytest.raises(ValueError, match="pick"):
        fit_set_model(make, data, base, TrainCfg(epochs=1, device="cpu", pick="middle"))
    with pytest.raises(ValueError, match="pick"):
        predict_set_model(f, data, base, pick="middle")
