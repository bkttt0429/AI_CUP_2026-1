from functools import partial

import numpy as np
import pytest
import torch

from aicup.aggregators import CTM, DeepSets, GameEncoder, PooledDelta, QglobDelta, SetTransformer, XAttnGRU, n_params
from aicup.rankmodel import _softmax
from aicup.settrain import TrainCfg, fit_set_model, predict_set_model

D_IN, M, B = 9, 20, 6

# every set model at test size, constructed as make(d_in)
SMALL_CTM = partial(CTM, d_tok=16, d_latent=32, history=4, nlm_hidden=8, heads=4, ticks=3, n_sync_action=24, n_sync_out=24)
MODELS = [
    pytest.param(partial(PooledDelta, hidden=16), id="PooledDelta"),
    pytest.param(partial(QglobDelta, hidden=8), id="QglobDelta"),
    pytest.param(partial(DeepSets, d_tok=16, hidden=32), id="DeepSets"),
    pytest.param(partial(SetTransformer, d_tok=16, heads=4, layers=2, hidden=32), id="SetTransformer"),
    pytest.param(partial(XAttnGRU, d_tok=16, d_hidden=32, heads=4, ticks=3), id="XAttnGRU"),
    pytest.param(SMALL_CTM, id="CTM"),
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


class _Tap:
    """Records the input of `w_q` (the action sync), the output of `nlm` (z) and the input of
    `w_out` (the output sync) at every tick of a CTM forward."""

    def __init__(self, model):
        self.s_act, self.z, self.s_out = [], [], []
        model.w_q.register_forward_pre_hook(lambda m, args: self.s_act.append(args[0].detach().clone()))
        model.nlm.register_forward_hook(lambda m, args, out: self.z.append(out.detach().clone()))
        model.w_out.register_forward_pre_hook(lambda m, args: self.s_out.append(args[0].detach().clone()))


def test_ctm_shapes_history_and_pairs():
    tokens, mask, qglob = _inputs()
    model = _randomised(CTM)
    assert model.idx_act.shape == (512, 2) and model.idx_out.shape == (512, 2)
    pairs = {}
    for name in ("idx_act", "idx_out"):
        idx = getattr(model, name)
        assert name in dict(model.named_buffers())  # moves with .to(device), not trained
        assert idx.dtype == torch.long and int(idx.min()) >= 0 and int(idx.max()) < 256
        assert torch.all(idx[:, 0] <= idx[:, 1])  # i <= j: one of the 256 * 257 / 2 = 32,896 pairs
        pairs[name] = {tuple(p) for p in idx.tolist()}
        assert len(pairs[name]) == 512  # drawn without replacement
    assert not pairs["idx_act"] & pairs["idx_out"]  # first 512 of the permutation for action, next 512 for output
    with torch.no_grad():
        out = model(tokens, mask, qglob)
        assert out.shape == (B, 8, 10)
        out16 = model(tokens, mask, qglob, ticks=16)
        assert out16.shape == (B, 16, 10) and torch.all(torch.isfinite(out16))
    torch.testing.assert_close(out16[:, :8], out, atol=1e-6, rtol=0)  # one recurrence, T only truncates it
    assert not torch.allclose(out[:, 0], out[:, 1])  # the state moves from tick to tick
    for seed in (0, 7):  # the pairs depend on pair_seed only, not on the global RNG
        torch.manual_seed(seed)
        a = CTM(D_IN, pair_seed=3)
        torch.manual_seed(seed + 100)
        b = CTM(D_IN, pair_seed=3)
        assert torch.equal(a.idx_act, b.idx_act) and torch.equal(a.idx_out, b.idx_out)
    c = CTM(D_IN, pair_seed=4)
    assert not torch.equal(a.idx_act, c.idx_act)
    small = SMALL_CTM(D_IN)
    assert small.idx_act.shape == (24, 2) and small.idx_out.shape == (24, 2)
    assert int(small.idx_act.max()) < 32 and _randomised(SMALL_CTM)(tokens, mask, qglob).shape == (B, 3, 10)
    with pytest.raises(ValueError, match="pairs"):  # 30 + 10 > 8 * 9 / 2 = 36
        CTM(D_IN, d_latent=8, n_sync_action=30, n_sync_out=10)
    # learned start values A0 [256, 8], z0 [256]; decay r_p starts at 0 (no decay)
    assert model.A0.shape == (256, 8) and model.z0.shape == (256,)
    fresh = CTM(D_IN)
    assert torch.equal(fresh.r_act, torch.zeros(512)) and torch.equal(fresh.r_out, torch.zeros(512))
    assert torch.equal(fresh.w_out.weight, torch.zeros(10, 512)) and torch.equal(fresh.w_out.bias, torch.zeros(10))
    enc128 = (D_IN * 128 + 128) + (128 * 128 + 128) + 2 * 128
    synapse = (256 + 128 + 6) * 512 + 512 + 2 * 256  # Linear -> GLU -> LayerNorm(256)
    nlm = 256 * ((8 * 16 + 16) + (16 + 1))  # one 8 -> 16 -> 1 MLP per neuron
    assert n_params(fresh) == (enc128 + 256 * 8 + 256 + 2 * 512 + (512 * 128 + 128) + (4 * 128 * 128 + 4 * 128)
                               + synapse + nlm + (512 * 10 + 10))


def _ctm_sync(Z, idx, r):
    """S_p = sum_tau w_tau z_i z_j / sqrt(sum_tau w_tau), w_tau = exp(-clamp(r, 0, 15) (tau_now - tau)), by hand."""
    r = r.clamp(0.0, 15.0)
    num, den = 0.0, 0.0
    for k, z in enumerate(Z):
        w = torch.exp(-r * (len(Z) - 1 - k))
        num = num + w * z[:, idx[:, 0]] * z[:, idx[:, 1]]
        den = den + w
    return num / den.sqrt()


def test_ctm_action_sync_includes_z0():
    tokens, mask, qglob = _inputs()
    model = _randomised(SMALL_CTM)
    with torch.no_grad():
        model.r_act.zero_()
        model.r_out.zero_()
    tap = _Tap(model)
    with torch.no_grad():
        out = model(tokens, mask, qglob, ticks=2)
    z0 = model.z0.detach().expand(B, -1)
    (z1, z2), (a1, a2), (o1, o2) = tap.z, tap.s_act, tap.s_out
    ia, io = model.idx_act, model.idx_out
    # action: the history starts with z0, so tick 2 reads [z0, z1]
    torch.testing.assert_close(a1, z0[:, ia[:, 0]] * z0[:, ia[:, 1]], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a2, (z0[:, ia[:, 0]] * z0[:, ia[:, 1]] + z1[:, ia[:, 0]] * z1[:, ia[:, 1]]) / 2 ** 0.5,
                               atol=1e-6, rtol=1e-5)
    # output (pinned here too): the history also starts with z0 (the original seeds the 'out'
    # accumulators with the start state before the loop), so tick 1 reads [z0, z1], tick 2 [z0, z1, z2]
    pp = lambda z: z[:, io[:, 0]] * z[:, io[:, 1]]
    torch.testing.assert_close(o1, (pp(z0) + pp(z1)) / 2 ** 0.5, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(o2, (pp(z0) + pp(z1) + pp(z2)) / 3 ** 0.5, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(out, torch.stack([model.w_out(o1), model.w_out(o2)], dim=1), atol=1e-6, rtol=1e-5)

    # decay: per-pair r_p, clamped to [0, 15] on every forward (both ends exercised)
    with torch.no_grad():
        model.r_act.copy_(torch.linspace(-3.0, 20.0, len(ia)))
        model.r_out.copy_(torch.linspace(20.0, -3.0, len(io)))
    r_act, r_out = model.r_act.detach().clone(), model.r_out.detach().clone()
    tap = _Tap(model)
    with torch.no_grad():
        model(tokens, mask, qglob, ticks=3)
    Z = [z0] + tap.z
    for t in range(3):
        torch.testing.assert_close(tap.s_act[t], _ctm_sync(Z[:t + 1], ia, r_act), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(tap.s_out[t], _ctm_sync(Z[:t + 2], io, r_out), atol=1e-6, rtol=1e-5)
    assert float(model.r_act.detach().min()) == 0.0 and float(model.r_act.detach().max()) == 15.0  # projected in place


def test_ctm_decay_gets_gradient_at_init():
    tokens, mask, qglob = _inputs()
    torch.manual_seed(0)
    model = CTM(D_IN, d_tok=16, d_latent=32, n_sync_action=24, n_sync_out=24)
    with torch.no_grad():
        model.w_out.weight.normal_()
    model(tokens, mask, qglob).pow(2).sum().backward()
    assert model.r_act.grad.abs().sum() > 0 and model.r_out.grad.abs().sum() > 0  # r = 0 is not a dead end


def test_ctm_norms_bounded_at_16_ticks():
    tokens, mask, qglob = _inputs()
    for zero_decay in (False, True):  # r as randomised, and r = 0 (no decay: the sync sums grow fastest)
        model = _randomised(CTM)
        if zero_decay:
            with torch.no_grad():
                model.r_act.zero_()
                model.r_out.zero_()
        tap = _Tap(model)
        with torch.no_grad():
            out = model(10.0 * tokens, mask, 10.0 * qglob, ticks=16)
        assert len(tap.z) == len(tap.s_act) == len(tap.s_out) == 16
        for x in tap.z + tap.s_act + tap.s_out + [out]:
            assert torch.all(torch.isfinite(x))
            assert float(x.norm(dim=-1).max()) < 1e3
