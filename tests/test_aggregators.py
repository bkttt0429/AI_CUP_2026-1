from functools import partial

import numpy as np
import pytest
import torch

from aicup.aggregators import DeepSets, GameEncoder, PooledDelta, QglobDelta, SetTransformer, XAttnGRU, n_params
from aicup.rankmodel import _softmax
from aicup.settrain import TrainCfg, fit_set_model, predict_set_model

D_IN, M, B = 9, 20, 6

# every set model at test size, constructed as make(d_in); Task 6 adds CTM here
MODELS = [
    pytest.param(partial(PooledDelta, hidden=16), id="PooledDelta"),
    pytest.param(partial(QglobDelta, hidden=8), id="QglobDelta"),
    pytest.param(partial(DeepSets, d_tok=16, hidden=32), id="DeepSets"),
    pytest.param(partial(SetTransformer, d_tok=16, heads=4, layers=2, hidden=32), id="SetTransformer"),
    pytest.param(partial(XAttnGRU, d_tok=16, d_hidden=32, heads=4, ticks=3), id="XAttnGRU"),
]


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


def _inputs(seed=0):
    """B questions with 1..20 games (prefix mask), padding rows 0."""
    g = torch.Generator().manual_seed(seed)
    m = torch.tensor([1, 3, 5, 12, 19, 20])
    mask = torch.arange(M)[None, :] < m[:, None]
    tokens = torch.randn(B, M, D_IN, generator=g) * mask[..., None]
    return tokens, mask, torch.randn(B, 6, generator=g)


def _randomised(make):
    """make(D_IN) with every all-zero parameter (the output layer, biases) redrawn, so that its
    output is not identically 0 and the invariance checks are not vacuous."""
    torch.manual_seed(0)
    model = make(D_IN)
    with torch.no_grad():
        for p in model.parameters():
            if torch.all(p == 0):
                p.normal_(0.0, 0.5)
    return model.eval()


def _packed(Q=40, seed=0):
    rng = np.random.RandomState(seed)
    n = rng.randint(1, M + 1, size=Q)
    mask = np.arange(M)[None, :] < n[:, None]
    tokens = (rng.randn(Q, M, D_IN) * mask[..., None]).astype(np.float32)
    data = {"tokens": tokens, "mask": mask, "qglob": rng.randn(Q, 6).astype(np.float32),
            "y": np.arange(Q, dtype=np.int64) % 10}
    return data, rng.randn(Q, 10)


@pytest.mark.parametrize("make", MODELS)
def test_permutation_invariance(make):
    model = _randomised(make)
    tokens, mask, qglob = _inputs()
    g = torch.Generator().manual_seed(1)
    perm = torch.arange(M).repeat(B, 1)
    for b in range(B):  # shuffle each question's valid games among themselves
        m = int(mask[b].sum())
        perm[b, :m] = torch.randperm(m, generator=g)
    rows = torch.arange(B)[:, None]
    with torch.no_grad():
        out = model(tokens, mask, qglob)
        shuffled = model(tokens[rows, perm], mask[rows, perm], qglob)
    assert out.shape[0] == B and out.shape[2] == 10
    assert not torch.allclose(out[3], out[4])
    torch.testing.assert_close(shuffled, out, atol=1e-5, rtol=0)


@pytest.mark.parametrize("make", MODELS)
def test_padding_invariance(make):
    model = _randomised(make)
    tokens, mask, qglob = _inputs()
    g = torch.Generator().manual_seed(2)
    noisy = torch.where(mask[..., None], tokens, 100.0 * torch.randn(tokens.shape, generator=g))
    longer = torch.cat([noisy, 50.0 * torch.randn(B, 5, D_IN, generator=g)], dim=1)  # 5 more padding slots
    longer_mask = torch.cat([mask, torch.zeros(B, 5, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out = model(tokens, mask, qglob)
        torch.testing.assert_close(model(noisy, mask, qglob), out, atol=1e-6, rtol=0)
        torch.testing.assert_close(model(longer, longer_mask, qglob), out, atol=1e-6, rtol=0)
    # non-prefix padding: the same games scattered among the padding slots, in a different order
    perm = torch.stack([torch.randperm(M, generator=g) for _ in range(B)])
    rows = torch.arange(B)[:, None]
    scattered_mask = mask[rows, perm]
    assert not torch.equal(scattered_mask, mask) and not scattered_mask[:, :1].all()  # padding comes first somewhere
    scattered = torch.where(scattered_mask[..., None], tokens[rows, perm], 100.0 * torch.randn(tokens.shape, generator=g))
    with torch.no_grad():
        torch.testing.assert_close(model(scattered, scattered_mask, qglob), out, atol=1e-5, rtol=0)


@pytest.mark.parametrize("make", MODELS)
def test_zero_init_delta(make):
    torch.manual_seed(0)
    model = make(D_IN)
    tokens, mask, qglob = _inputs()
    with torch.no_grad():
        out = model(tokens, mask, qglob)
    assert out.shape[0] == B and out.shape[2] == 10
    assert torch.equal(out, torch.zeros_like(out))
    data, base = _packed()
    fitted = fit_set_model(make, data, base, TrainCfg(epochs=0, device="cpu"))
    np.testing.assert_allclose(predict_set_model(fitted, data, base), _softmax(base), rtol=0, atol=1e-12)


@pytest.mark.parametrize("make", MODELS)
def test_proba_valid(make):
    data, base = _packed()
    fitted = fit_set_model(make, data, base, TrainCfg(epochs=2, batch=16, lr=1e-2, device="cpu"))
    P = predict_set_model(fitted, data, base)
    assert P.shape == (40, 10) and P.dtype == np.float64
    assert np.all(np.isfinite(P)) and np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert not np.allclose(P, _softmax(base))  # training moved Δ off 0


def test_single_tick_shapes_and_sizes():
    tokens, mask, qglob = _inputs()
    enc = GameEncoder(D_IN, d_tok=16)
    assert enc(tokens).shape == (B, M, 16)
    for make in (PooledDelta, QglobDelta, DeepSets, SetTransformer):
        model = make(D_IN)
        with torch.no_grad():
            assert model(tokens, mask, qglob).shape == (B, 1, 10)
            assert model(tokens, mask, qglob, ticks=16).shape == (B, 1, 10)  # T = 1 models ignore ticks
    assert n_params(QglobDelta(D_IN)) == (6 * 64 + 64) + (64 * 10 + 10)
    assert n_params(PooledDelta(D_IN)) == ((D_IN + 6) * 256 + 256) + (256 * 10 + 10)
    enc128 = (D_IN * 128 + 128) + (128 * 128 + 128) + 2 * 128
    assert n_params(GameEncoder(D_IN)) == enc128
    head = lambda n_in: (n_in * 256 + 256) + (256 * 256 + 256) + (256 * 10 + 10)
    assert n_params(DeepSets(D_IN)) == enc128 + head(2 * 128 + 6)
    mab = (4 * 128 * 128 + 4 * 128) + 2 * 2 * 128 + 2 * (128 * 128 + 128)  # MHA, 2 LayerNorms, rFF
    assert n_params(SetTransformer(D_IN)) == enc128 + 3 * mab + 128 + head(128 + 6)  # 2 SABs, PMA, seed


def test_xattn_shapes_and_tick_override():
    tokens, mask, qglob = _inputs()
    model = _randomised(partial(XAttnGRU, d_tok=16, d_hidden=32, heads=4))  # default 8 ticks
    with torch.no_grad():
        out = model(tokens, mask, qglob)
        assert out.shape == (B, 8, 10)
        out16 = model(tokens, mask, qglob, ticks=16)
        assert out16.shape == (B, 16, 10)
        assert model(tokens, mask, qglob, ticks=1).shape == (B, 1, 10)
    torch.testing.assert_close(out16[:, :8], out, atol=1e-6, rtol=0)  # the ticks are one recurrence, T only truncates it
    assert not torch.allclose(out[:, 0], out[:, 1])  # the state moves from tick to tick
    assert _randomised(partial(XAttnGRU, d_tok=16, d_hidden=32, heads=4, ticks=5))(tokens, mask, qglob).shape == (B, 5, 10)
    gru = 3 * 256 * (128 + 6) + 3 * 256 * 256 + 2 * 3 * 256  # GRUCell(o ++ qglob, h): W_ih, W_hh, both biases
    enc128 = (D_IN * 128 + 128) + (128 * 128 + 128) + 2 * 128
    assert n_params(XAttnGRU(D_IN)) == enc128 + 256 + (256 * 128 + 128) + (4 * 128 * 128 + 4 * 128) + gru + (256 * 10 + 10)
