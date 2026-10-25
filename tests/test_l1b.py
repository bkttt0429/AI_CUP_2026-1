"""aicup.l1b: rank-unknown metadata, the 10 rank heads copied from the policy head, trunk features, scoring,
save/load. A random b2c16 KataGo model (no metadata encoder) stands in for humanv0."""
import numpy as np
import pytest
import torch
from katago.game.features import Features
from katago.train import modelconfigs
from katago.train.model_pytorch import Model

from aicup import l1b
from aicup.pilot import game_positions

SGF = "(;SZ[19];B[pd];W[dp];B[pq];W[];B[dd];W[cc];B[])"  # both colours pass once


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    model.eval()
    return model


def _positions(color="B"):
    pos = game_positions(SGF, color, Features(modelconfigs.config_of_name["b2c16"], 19))
    return pos


def _model_logp(model, pos):
    """The model's own masked policy log p of every target move, fp32 (what every head must equal at step 0)."""
    with torch.no_grad():
        bins, globs = torch.tensor(pos["bins"]).float(), torch.tensor(pos["globs"])
        logits = model.postprocess_output(model(bins, globs))[0][0][:, 0, :].float()
        logits = logits.masked_fill(~torch.tensor(pos["legal"]), float("-inf"))
        logp = torch.log_softmax(logits, dim=1)
    return logp.gather(1, torch.tensor(pos["target"]).long()[:, None])[:, 0].numpy()


def test_neutral_meta_row_marks_rank_unknown_and_empty_thermometer():
    rows = l1b.neutral_rows()
    assert rows.shape == (2, 192) and rows.dtype == np.float32 and np.array_equal(rows[0], rows[1])
    assert rows[0, 0] == rows[0, 1] == 1 and rows[0, 4] == rows[0, 5] == 1 and rows[0, 6:74].sum() == 0
    meta = l1b.neutral_meta()
    assert meta.bRankIsUnknown and meta.wRankIsUnknown and meta.inverseBRank == 0 and meta.inverseWRank == 0
    assert meta.bIsHuman and meta.wIsHuman and meta.tcIsByoYomi


def test_heads_start_equal_to_the_policy_head(tiny_model):
    pos = _positions("B")
    heads = l1b.RankHeads(tiny_model.policy_head, n_ranks=10)
    assert len(heads.heads) == 10
    with torch.no_grad():
        trunk, mask, mask_sum_hw, mask_sum = l1b.trunk_features(tiny_model, torch.tensor(pos["bins"]).float(),
                                                                torch.tensor(pos["globs"]), torch.tensor(pos["color"]))
        n = len(pos["target"])
        assert n == 4 and pos["target"].tolist()[-1] == 361  # pd, pq, dd and the final pass
        assert trunk.shape == (n, 16, 19, 19) and trunk.dtype == torch.float32
        logits = heads(trunk, mask, mask_sum_hw, mask_sum)
        assert logits.shape == (n, 10, 362)
        logp = l1b.masked_logp(logits, torch.tensor(pos["legal"]))
    assert torch.isinf(logp[:, :, :361][~torch.tensor(pos["legal"])[:, None, :361].expand(-1, 10, -1)]).all()
    want = _model_logp(tiny_model, pos)
    got = logp.gather(2, torch.tensor(pos["target"]).long()[:, None, None].expand(-1, 10, 1))[:, :, 0].numpy()
    np.testing.assert_allclose(got, want[:, None].repeat(10, axis=1), rtol=0, atol=1e-6)
    scored = l1b.score_positions(tiny_model, heads, pos["bins"], pos["globs"], pos["legal"], pos["target"], pos["color"], batch=3)
    assert scored.shape == (n, 10) and scored.dtype == np.float32
    np.testing.assert_allclose(scored, want[:, None].repeat(10, axis=1), rtol=0, atol=1e-6)


def test_heads_are_independent_copies(tiny_model):
    heads = l1b.RankHeads(tiny_model.policy_head, n_ranks=10)
    before = tiny_model.policy_head.conv1p.weight.clone()
    with torch.no_grad():
        heads.heads[3].conv1p.weight.add_(1.0)
    assert torch.equal(tiny_model.policy_head.conv1p.weight, before)
    assert torch.equal(heads.heads[0].conv1p.weight, before)
    assert not torch.equal(heads.heads[3].conv1p.weight, before)
    assert sum(p.numel() for p in heads.parameters()) == 10 * sum(p.numel() for p in tiny_model.policy_head.parameters())


def test_pass_move_has_finite_logp(tiny_model):
    pos = _positions("W")
    assert pos["target"].tolist()[1] == 361
    heads = l1b.RankHeads(tiny_model.policy_head)
    scored = l1b.score_positions(tiny_model, heads, pos["bins"], pos["globs"], pos["legal"], pos["target"], pos["color"])
    assert scored.shape == (3, 10) and np.isfinite(scored).all()
    np.testing.assert_allclose(scored, _model_logp(tiny_model, pos)[:, None].repeat(10, axis=1), rtol=0, atol=1e-6)


def test_save_load_heads_round_trip(tmp_path, tiny_model):
    pos = _positions("B")
    heads = l1b.RankHeads(tiny_model.policy_head)
    with torch.no_grad():
        for k, h in enumerate(heads.heads):
            h.conv1p.weight.add_(0.01 * (k + 1))
    pt = tmp_path / "heads_A.pt"
    meta = l1b.save_heads(heads, pt, {"train_folds": [1, 2], "max_games": 40})
    assert pt.exists() and pt.with_suffix(".json").exists()
    assert meta["format"] == 1 and meta["n_ranks"] == 10 and meta["levels"][0] == "D" and meta["train_folds"] == [1, 2]
    assert meta["state_sha256"] == l1b.file_sha256(pt) and "torch_version" in meta and "code_sha256" in meta
    loaded, meta2, warnings = l1b.load_heads(pt, tiny_model)
    assert warnings == [] and meta2 == meta
    want = l1b.score_positions(tiny_model, heads, pos["bins"], pos["globs"], pos["legal"], pos["target"], pos["color"])
    got = l1b.score_positions(tiny_model, loaded, pos["bins"], pos["globs"], pos["legal"], pos["target"], pos["color"])
    np.testing.assert_array_equal(got, want)
    assert not np.allclose(want[:, 0], want[:, 9])  # the perturbed heads differ, so the test is not vacuous
    pt.write_bytes(pt.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        l1b.load_heads(pt, tiny_model)
    side = pt.with_suffix(".json")
    side.write_text(side.read_text().replace('"n_ranks": 10', '"n_ranks": 7'))
    with pytest.raises(ValueError, match="n_ranks|sha256"):
        l1b.load_heads(pt, tiny_model)
