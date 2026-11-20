"""Section 4M M30: the two-tower (dual) and CLIP-aligned (clip) token encoders of scripts/trunk_seq_check.py; the default stays ct7's single Linear."""
import json

import numpy as np
import pytest
import torch

import trunk_seq_check as ts


def _batch(G=(5, 7), L=30, seed=0):
    """Random tokens of two questions with G games each (flags 0/1, varying lengths), as batch_tokens lays them out (as tests/test_trunk_seq_4m.py)."""
    rng = np.random.RandomState(seed)
    n_games = sum(G)
    lens = rng.randint(10, L + 1, size=n_games)
    tokens = np.zeros((n_games, L, ts.D_TOKEN), np.float32)
    mask = np.zeros((n_games, L, 1), np.float32)
    for g, n in enumerate(lens):
        tokens[g, :n, :ts.D_TOKEN - 4] = rng.randn(n, ts.D_TOKEN - 4)
        tokens[g, :n, ts.D_TOKEN - 4] = rng.randint(0, 2, n)
        tokens[g, :n, ts.D_TOKEN - 3] = rng.randint(0, 2, n)
        tokens[g, :n, ts.D_TOKEN - 2] = np.arange(n) / n
        mask[g, :n, 0] = 1.0
    owner = np.repeat(np.arange(len(G)), G)
    return torch.as_tensor(tokens), torch.as_tensor(mask), torch.as_tensor(owner)[:, None], torch.as_tensor(rng.randn(len(G), 64).astype(np.float32))


def test_default_is_ct7s_single_linear():
    for arm in ("C", "T"):
        m = ts.Model(arm, 64)
        assert m.token_enc == "linear" and m.proj.weight.shape == (ts.HIDDEN[arm], ts.D_TOKEN) and m.aux_loss is None
        assert not hasattr(m, "board") and not hasattr(m, "clip_board") and not hasattr(m, "logit_scale")
    assert ts.Model("R0", 64, token_enc="clip").token_enc == "linear"
    with pytest.raises(ValueError):
        ts.Model("C", 64, token_enc="towers")


@pytest.mark.parametrize("arm", ["C", "T"])
def test_ct7_members_still_load_as_linear(arm):
    ck = ts.ROOT / f"results/models/ct_cv/val1_{arm}-f0-x40-oh_s0.pt"
    if not ck.exists():
        pytest.skip("ct7 member not on disk")
    side = json.loads(ck.with_suffix(".json").read_text())
    assert "token_enc" not in side
    model, stats, _, _ = ts.load_checkpoint(ck)
    assert model.token_enc == "linear" and sum(p.numel() for p in model.parameters()) == side["n_params"]
    assert sum(p.numel() for p in ts.Model(arm, side["n_r"]).parameters()) == side["n_params"]


@pytest.mark.parametrize("arm", ["C", "T"])
@pytest.mark.parametrize("enc", ["dual", "clip"])
def test_two_towers_forward_aux_loss_and_checkpoint(arm, enc, tmp_path):
    torch.manual_seed(0)
    tokens, mask, game_of, r = _batch()
    m, base = ts.Model(arm, 64, token_enc=enc), ts.Model(arm, 64)
    extra = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in base.parameters())
    assert extra == {"C": 572, "T": 4528}[arm] + (2369 if enc == "clip" else 0)  # the numbers in the M30 pre-registration
    # each tower reads its own block only
    _, (b1, s1) = m.embed(tokens)
    t2 = tokens.clone(); t2[..., 130] += 1.0
    _, (b2, s2) = m.embed(t2)
    assert torch.allclose(b1, b2) and not torch.allclose(s1, s2)
    t3 = tokens.clone(); t3[..., 5] += 1.0
    _, (b3, s3) = m.embed(t3)
    assert torch.allclose(s1, s3) and not torch.allclose(b1, b3)
    m.train()
    out = m(r, tokens, mask, game_of)
    assert out.shape == (2, 10) and torch.isfinite(out).all()
    if enc == "clip":
        assert m.aux_loss is not None and torch.isfinite(m.aux_loss) and m.aux_loss.item() > 0
        (out.sum() + m.aux_loss).backward()
        assert m.logit_scale.grad is not None and m.clip_score.weight.grad.abs().sum() > 0
        assert m.board.net[0].weight.grad.abs().sum() > 0 and m.score.net[0].weight.grad.abs().sum() > 0
    else:
        assert m.aux_loss is None
    m.eval()
    out = m(r, tokens, mask, game_of)
    assert m.aux_loss is None  # evaluation never computes the alignment loss
    tokens_p, mask_p = torch.cat([tokens, torch.zeros(12, 10, ts.D_TOKEN)], 1), torch.cat([mask, torch.zeros(12, 10, 1)], 1)
    assert torch.allclose(out, m(r, tokens_p, mask_p, game_of), atol=1e-5)  # padding does not leak through the towers
    side = ts.save_checkpoint(tmp_path / "m.pt", m, (np.zeros(ts.D_TOKEN - 4, np.float32), np.ones(ts.D_TOKEN - 4, np.float32)), (np.zeros(64), np.ones(64)), {"tag": f"{arm}-{enc}"})
    assert side["token_enc"] == enc
    m2, _, _, side2 = ts.load_checkpoint(tmp_path / "m.pt")
    assert m2.token_enc == enc and side2["token_enc"] == enc
    assert torch.allclose(m2(r, tokens, mask, game_of), out, atol=1e-6)


def test_clip_loss_samples_at_most_clip_plies_and_rewards_alignment(monkeypatch):
    torch.manual_seed(0)
    tokens, mask, game_of, r = _batch()
    m = ts.Model("C", 64, token_enc="clip")
    m.train()
    _, views = m.embed(tokens)
    assert int(mask.sum()) > 50
    seen = []
    hook = m.clip_board.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].shape[0]))
    monkeypatch.setattr(ts, "CLIP_PLIES", 50)
    small = m.clip_loss(views, mask)
    monkeypatch.setattr(ts, "CLIP_PLIES", 2048)
    full = m.clip_loss(views, mask)
    hook.remove()
    assert torch.isfinite(small) and torch.isfinite(full) and seen == [50, int(mask.sum())]  # capped sample, else every real ply
    # identity heads on the first 24 dims: matching board / score views score lower than shuffled ones
    hb, hs = ts.DUAL_H
    with torch.no_grad():
        for head in (m.clip_board, m.clip_score):
            head.weight.zero_(); head.bias.zero_()
        m.clip_board.weight[:hs, :hs] = torch.eye(hs); m.clip_score.weight[:hs, :hs] = torch.eye(hs)
    s = views[1].detach()
    pad = torch.zeros(*s.shape[:2], hb - hs)
    aligned = m.clip_loss((torch.cat([s, pad], -1), s), mask)
    shuffled = m.clip_loss((torch.cat([s[torch.randperm(s.shape[0])], pad], -1), s), mask)
    assert aligned.item() < shuffled.item()


def test_cli_token_enc():
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "R0", "--token-enc", "clip"])
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--token-enc", "towers"])
