"""Section 4M M33: same-player question pairing and the player-contrastive InfoNCE of scripts/trunk_seq_check.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import trunk_seq_check as ts

sys.path.insert(0, str(Path(__file__).parent))  # python -I leaves the tests directory off the path


def test_paired_perm_pairs_same_player_and_covers_even_groups():
    pid = np.array(["a"] * 6 + ["b"] * 4 + ["c"] * 3 + ["d"] * 1)
    perm = ts.paired_perm(pid, np.random.RandomState(0))
    assert len(perm) == 6 + 4 + 2  # c's odd one and d's single question are dropped
    assert all(pid[perm[2 * k]] == pid[perm[2 * k + 1]] for k in range(len(perm) // 2))
    assert len(set(perm.tolist())) == len(perm)
    assert set(perm[np.isin(pid[perm], ["a", "b"])].tolist()) == set(range(10))
    assert np.array_equal(perm, ts.paired_perm(pid, np.random.RandomState(0)))
    assert not np.array_equal(perm, ts.paired_perm(pid, np.random.RandomState(1)))


def test_contrast_loss_rewards_matching_pairs():
    torch.manual_seed(0)
    head = torch.nn.Linear(ts.Q_DIM, 32)
    base = torch.randn(8, ts.Q_DIM)
    z_match = torch.repeat_interleave(base, 2, dim=0)  # rows 2k and 2k + 1 identical
    z_rand = torch.randn(16, ts.Q_DIM)
    assert ts.player_contrast_loss(z_match, head).item() < ts.player_contrast_loss(z_rand, head).item()


def test_question_embedding_only_in_training():
    torch.manual_seed(0)
    from test_trunk_seq_m30 import _batch  # noqa: E402
    tokens, mask, game_of, r = _batch()
    m = ts.Model("C", 64)
    m.train(); m(r, tokens, mask, game_of)
    assert m.q_embed is not None and m.q_embed.shape == (2, ts.Q_DIM)
    m.eval(); out = m(r, tokens, mask, game_of)
    assert m.q_embed is None and out.shape == (2, 10)


def test_cli_player_contrast_guards():
    for argv in (["train", "--arm", "R0", "--player-contrast", "0.1"], ["train", "--arm", "C", "--player-contrast", "0.1", "--batch", "31"],
                 ["train", "--arm", "C", "--player-contrast", "0.1", "--loss", "es-ft"], ["train", "--arm", "C", "--player-contrast", "0.1", "--pseudo", "x.npz"]):
        with pytest.raises(SystemExit):
            ts.main(argv)


def test_cli_pair_batches_guard():
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "R0", "--pair-batches"])
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--pair-batches", "--batch", "31"])


def test_grouped_perm_and_supcon():
    pid = np.array(["a"] * 8 + ["b"] * 6 + ["c"] * 3)
    perm = ts.grouped_perm(pid, np.random.RandomState(0), 4)
    assert len(perm) == 8 + 4 and len(set(perm.tolist())) == len(perm)  # b's 2 leftovers and c's 3 are dropped
    assert all(len(set(pid[perm[4 * g:4 * g + 4]])) == 1 for g in range(len(perm) // 4))
    torch.manual_seed(0)
    head = torch.nn.Linear(ts.Q_DIM, 32)
    base = torch.randn(4, ts.Q_DIM)
    z_match = torch.repeat_interleave(base, 4, dim=0)
    lm, lr = ts.supcon_loss(z_match, head, 4, 0.1), ts.supcon_loss(torch.randn(16, ts.Q_DIM), head, 4, 0.1)
    assert torch.isfinite(lm) and torch.isfinite(lr) and lm.item() < lr.item()
    # the default (2 views, temperature 0.1) is M33's loss exactly
    z = torch.randn(16, ts.Q_DIM)
    assert torch.equal(ts.player_contrast_loss(z, head), ts.player_contrast_loss(z, head, 0.1))


def test_cli_pc_views_guard():
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--player-contrast", "0.1", "--pc-views", "3"])
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--player-contrast", "0.1", "--pc-views", "1"])
