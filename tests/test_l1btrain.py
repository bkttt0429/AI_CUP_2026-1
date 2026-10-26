"""aicup.l1btrain: class-balanced position weights, the per-player inner split, inner NLL, head training with
early stopping and resume, and the step-0 equality check -- on a random b2c16 model and a tiny real cache."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from katago.game.features import Features
from katago.train import modelconfigs
from katago.train.model_pytorch import Model

from aicup import featcache as fc
from aicup import l1b, l1btrain
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.pilot import game_positions
from tests.test_l1b_scripts import _sgf


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    model.eval()
    return model


def _tiny_cache_dir(tmp_path, ranks=("D", "A", "1D", "6D"), players_per_rank=3, games_per_player=2):
    """A real cache: synthetic games (3-8 moves) for players_per_rank players of each rank; returns the directory."""
    features = Features(modelconfigs.config_of_name["b2c16"], 19)
    rows, parts, g = [], [], 0
    for rank in ranks:
        for p in range(players_per_rank):
            pid = f"{RANK_TO_IDX[rank]:04x}{p:04x}"
            for _ in range(games_per_player):
                color = "BW"[g % 2]
                sgf = _sgf(3 + g % 6)
                pos = game_positions(sgf, color, features)
                rows.append({"game_key": f"{rank}:g{g}", "player_id": pid, "rank": rank, "color": color, "n_moves": len(pos["target"])})
                parts.append(fc.pack_positions(pos))
                g += 1
    out = tmp_path / "cache"
    fc.write_part(out, 0, 0, pd.DataFrame(rows), parts)
    (out / "meta.json").write_text(json.dumps({"folds": [1, 2], "max_games": 40, "min_games": 1, "games": len(rows)}))
    return out


@pytest.fixture
def tiny_cache(tmp_path):
    return _tiny_cache_dir(tmp_path)


def test_position_weights_balance_ranks():
    rank_idx = np.array([0, 0, 0, 1, 5, 5], dtype=np.int8)
    w = l1btrain.position_weights(rank_idx, n_ranks=10)
    assert w.shape == (6,) and np.isfinite(w).all() and abs(w.mean() - 1.0) < 1e-12
    assert abs(w[:3].sum() - w[3]) < 1e-12 and abs(w[3] - w[4:].sum()) < 1e-12  # every present rank the same total weight
    assert l1btrain.position_weights(np.zeros(0, dtype=np.int8)).shape == (0,)


def test_inner_split_is_by_player_and_stratified():
    rows = [{"game_key": f"{lv}:{p}:{g}", "player_id": f"{i:02d}{p:02d}0000", "rank": lv}
            for i, lv in enumerate(LEVELS) for p in range(4) for g in range(2)]
    games = pd.DataFrame(rows)
    train_rows, inner_rows = l1btrain.inner_split(games, 0.1)
    assert set(train_rows) | set(inner_rows) == set(range(len(games))) and not set(train_rows) & set(inner_rows)
    inner_players = set(games.loc[inner_rows, "player_id"])
    assert not inner_players & set(games.loc[train_rows, "player_id"])
    assert games.loc[inner_rows].groupby("rank")["player_id"].nunique().tolist() == [1] * 10  # ceil(0.1 x 4) per rank
    again = l1btrain.inner_split(games, 0.1)
    assert np.array_equal(again[1], inner_rows)  # deterministic, independent of the model seed
    assert l1btrain.inner_split(games, 0.0)[1].size == 0


def test_inner_eval_with_missing_rank(tiny_model, tiny_cache):
    games, arrays = fc.read_cache(tiny_cache)
    rows = np.flatnonzero(games["rank"].isin(["D", "A", "1D"]).to_numpy())
    idx = l1btrain.positions_of(arrays, rows)
    heads = l1b.RankHeads(tiny_model.policy_head)
    cfg = l1btrain.HeadsCfg(batch=8, device="cpu")
    res = l1btrain.evaluate_nll(tiny_model, heads, arrays, idx, cfg)
    assert np.isfinite(res["nll"]) and res["n"] == len(idx) and sorted(res["ranks_present"]) == sorted(RANK_TO_IDX[r] for r in ["D", "A", "1D"])
    assert len(res["nll_by_rank"]) == 10 and all(res["nll_by_rank"][k] is None for k in range(10) if k not in res["ranks_present"])
    present = [res["nll_by_rank"][k] for k in res["ranks_present"]]
    assert abs(res["nll"] - np.mean(present)) < 1e-9  # class-balanced: the mean over the ranks present


def test_train_heads_improves_inner_nll_and_resumes(tmp_path, tiny_model, tiny_cache):
    cfg = l1btrain.HeadsCfg(epochs=1, batch=8, lr=1e-3, eval_every=0.5, patience=10, inner_frac=0.34, seed=0, device="cpu")
    out = tmp_path / "heads_A.pt"
    heads, meta = l1btrain.train_heads(tiny_model, tiny_cache, cfg, out, log=lambda *a, **k: None)
    assert out.exists() and out.with_suffix(".json").exists() and (tmp_path / "heads_A.pt.ckpt").exists()
    assert meta["inner_nll_best"] <= meta["inner_nll_init"] and meta["best_epoch"] <= 1.0 and meta["stopped_early"] is False
    assert meta["train_folds"] == [1, 2] and meta["max_games"] == 40 and meta["head_init"] == "humanv0_policy_head"
    assert meta["n_train_positions"] + meta["n_inner_positions"] == len(fc.read_cache(tiny_cache)[1]["target"])
    assert sum(meta["per_rank_positions"].values()) == meta["n_train_positions"] and set(meta["per_rank_positions"]) == set(LEVELS)
    assert len(meta["inner_ranks_present"]) >= 1 and meta["wall_seconds"] > 0 and meta["cache_meta"]["games"] == 24
    eps = [e for e, _ in meta["history"]]  # evaluations at whole steps: 0, about half an epoch, the end
    assert len(eps) == 3 and eps[0] == 0.0 and 0.0 < eps[1] < 1.0 and eps[2] == 1.0 and meta["resumed_from_step"] == 0
    assert json.loads(out.with_suffix(".json").read_text())["train_cfg"]["lr"] == 1e-3
    init = l1b.RankHeads(tiny_model.policy_head)
    changed = any(not torch.equal(a, b) for a, b in zip(heads.state_dict().values(), init.state_dict().values()))
    assert changed or meta["best_epoch"] == 0.0
    # more epochs with the same cache and cfg otherwise: the run resumes from the checkpoint, history continues
    heads2, meta2 = l1btrain.train_heads(tiny_model, tiny_cache, replace(cfg, epochs=2), out, log=lambda *a, **k: None)
    assert meta2["resumed_from_step"] == meta["steps_per_epoch"] and meta2["history"][:3] == meta["history"]
    eps2 = [e for e, _ in meta2["history"]]
    assert len(eps2) == 5 and eps2 == sorted(eps2) and eps2[-1] == 2.0 and meta2["steps_run"] == 2 * meta["steps_per_epoch"]
    with pytest.raises(ValueError, match="checkpoint"):  # another batch size is another run
        l1btrain.train_heads(tiny_model, tiny_cache, replace(cfg, epochs=2, batch=4), out, log=lambda *a, **k: None)


def test_check_init_is_zero_at_step_zero(tiny_model, tiny_cache):
    _, arrays = fc.read_cache(tiny_cache)
    heads = l1b.RankHeads(tiny_model.policy_head)
    assert l1btrain.check_init(tiny_model, heads, arrays, np.arange(len(arrays["target"])), n=16) < 1e-6
    with torch.no_grad():
        heads.heads[2].conv1p.weight.add_(0.5)
    assert l1btrain.check_init(tiny_model, heads, arrays, np.arange(len(arrays["target"])), n=16) > 1e-3


def test_inner_frac_zero_trains_to_the_end_and_keeps_the_last_weights(tmp_path, tiny_model, tiny_cache):
    cfg = l1btrain.HeadsCfg(epochs=1, batch=8, lr=1e-3, eval_every=0.5, patience=2, inner_frac=0.0, seed=0, device="cpu")
    out = tmp_path / "heads_noinner.pt"
    heads, meta = l1btrain.train_heads(tiny_model, tiny_cache, cfg, out, log=lambda *a, **k: None)
    assert meta["n_inner_positions"] == 0 and meta["inner_nll_init"] is None and meta["inner_nll_best"] is None
    assert meta["stopped_early"] is False and meta["best_epoch"] == 1.0 and meta["steps_run"] == meta["steps_per_epoch"]
    assert meta["history"] == [] and meta["inner_ranks_present"] == []
    init = l1b.RankHeads(tiny_model.policy_head)
    assert any(not torch.equal(a, b) for a, b in zip(heads.state_dict().values(), init.state_dict().values()))  # trained, not reset


def test_checkpoint_write_is_atomic(tmp_path, tiny_model, tiny_cache, monkeypatch):
    """A crash while the checkpoint is being written leaves the previous checkpoint intact."""
    real_save, calls = torch.save, []

    def crashing_save(obj, path, *a, **k):
        calls.append(str(path))
        if len(calls) == 2:  # the second checkpoint of the run dies half-way through
            with open(path, "wb") as fh:
                fh.write(b"garbage")
            raise RuntimeError("disk full")
        return real_save(obj, path, *a, **k)

    monkeypatch.setattr(l1btrain.torch, "save", crashing_save)
    cfg = l1btrain.HeadsCfg(epochs=1, batch=8, lr=1e-3, eval_every=0.5, patience=10, inner_frac=0.34, seed=0, device="cpu")
    out = tmp_path / "heads_A.pt"
    with pytest.raises(RuntimeError, match="disk full"):
        l1btrain.train_heads(tiny_model, tiny_cache, cfg, out, log=lambda *a, **k: None)
    ckpt = tmp_path / "heads_A.pt.ckpt"
    assert ckpt.exists() and not list(tmp_path.glob("*.tmp"))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)  # the step-0 checkpoint, still readable
    assert state["state"]["step"] == 0


def test_cache_without_meta_is_refused(tmp_path, tiny_model, tiny_cache):
    (tiny_cache / "meta.json").unlink()
    cfg = l1btrain.HeadsCfg(epochs=1, batch=8, device="cpu")
    with pytest.raises(ValueError, match="meta.json"):
        l1btrain.train_heads(tiny_model, tiny_cache, cfg, tmp_path / "x.pt", log=lambda *a, **k: None)


# ---------------------------------------------------------------- stage 2: the trunk fine-tuned, several processes

def _stage2_cfg(**kw):
    base = dict(epochs=1, batch=8, lr=1e-3, eval_every=0.5, patience=10, inner_frac=0.34, seed=0, device="cpu", trunk_lr=1e-3)
    return l1btrain.HeadsCfg(**{**base, **kw})


def test_trunk_parameters_exclude_the_output_heads(tiny_model):
    names = l1btrain.trunk_parameters(tiny_model, "cpu")
    assert names and not any("policy_head" in n or "value_head" in n for n in names)
    grad = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert grad == set(names)


def test_stage2_fine_tunes_the_trunk_and_saves_it(tmp_path, tiny_model, tiny_cache):
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    out = tmp_path / "heads2_A.pt"
    heads, meta = l1btrain.train_heads(tiny_model, tiny_cache, _stage2_cfg(inner_frac=0.0), out, log=lambda *a, **k: None)
    side = json.loads(out.with_suffix(".json").read_text())
    assert meta["trunk_trained"] and meta["n_trunk_params"] > 0 and meta["world_size"] == 1 and meta["global_batch"] == 8
    assert side["trunk_file"] == "heads2_A.trunk.pt" and (tmp_path / "heads2_A.trunk.pt").exists()
    after = tiny_model.state_dict()
    trunk = set(l1btrain.trunk_parameters(tiny_model, "cpu"))
    assert any(not torch.equal(before[k], after[k]) for k in trunk)  # the trunk moved
    assert all(torch.equal(before[k], after[k]) for k in after if "policy_head" in k or "value_head" in k)  # the output heads did not
    # a fresh model gets the fine-tuned trunk back from load_heads, and scores the same
    torch.manual_seed(0)
    fresh = Model(modelconfigs.config_of_name["b2c16"], 19)
    fresh.initialize()
    fresh.eval()
    loaded, _, _ = l1b.load_heads(out, fresh)
    assert all(torch.equal(fresh.state_dict()[k], after[k]) for k in after)
    _, arrays = fc.read_cache(tiny_cache)
    sel = np.arange(min(16, len(arrays["target"])))
    args = (fc.unpack_bins(arrays["bins_packed"][sel]).astype(np.float32), arrays["globs"][sel], fc.unpack_legal(arrays["legal_packed"][sel]),
            arrays["target"][sel], arrays["color"][sel])
    assert np.array_equal(l1b.score_positions(fresh, loaded, *args), l1b.score_positions(tiny_model, heads, *args))
    # a damaged trunk file is refused
    tp = tmp_path / "heads2_A.trunk.pt"
    tp.write_bytes(tp.read_bytes()[:-1] + bytes([tp.read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match="sha256"):
        l1b.load_heads(out, fresh)


def test_stage1_writes_no_trunk_file(tmp_path, tiny_model, tiny_cache):
    out = tmp_path / "heads_A.pt"
    l1btrain.train_heads(tiny_model, tiny_cache, _stage2_cfg(trunk_lr=0.0, inner_frac=0.0), out, log=lambda *a, **k: None)
    assert not (tmp_path / "heads_A.trunk.pt").exists() and "trunk_file" not in json.loads(out.with_suffix(".json").read_text())


def test_init_heads_is_the_starting_point(tmp_path, tiny_model, tiny_cache):
    start = l1b.RankHeads(tiny_model.policy_head)
    with torch.no_grad():
        start.heads[4].conv1p.weight.mul_(0.5)
    heads, meta = l1btrain.train_heads(tiny_model, tiny_cache, _stage2_cfg(epochs=0, trunk_lr=0.0), tmp_path / "h.pt",
                                       log=lambda *a, **k: None, init_heads=start.state_dict())
    assert meta["best_epoch"] == 0.0 and all(torch.equal(a, b) for a, b in zip(heads.state_dict().values(), start.state_dict().values()))


def _two_process_worker(rank, world, init_file, cache_dir, out_dir):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        model = Model(modelconfigs.config_of_name["b2c16"], 19)
        model.initialize()
        model.eval()
        heads, meta = l1btrain.train_heads(model, cache_dir, _stage2_cfg(batch=4), f"{out_dir}/dist.pt", log=lambda *a, **k: None)
        torch.save({"heads": heads.state_dict(), "model": model.state_dict(), "history": meta["history"]}, f"{out_dir}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_processes_equal_one_process_on_the_same_global_batch(tmp_path, tiny_model, tiny_cache):
    """batch 4 x 2 processes (gloo) = batch 8 in one process: same steps, inner NLL and weights up to float error."""
    import torch.multiprocessing as tmp_mp
    one_heads, one_meta = l1btrain.train_heads(tiny_model, tiny_cache, _stage2_cfg(), tmp_path / "one.pt", log=lambda *a, **k: None)
    tmp_mp.spawn(_two_process_worker, args=(2, str(tmp_path / "pg_init"), str(tiny_cache), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in (0, 1))
    for k in r0["heads"]:  # every rank holds the same weights
        assert torch.equal(r0["heads"][k], r1["heads"][k])
    side = json.loads((tmp_path / "dist.json").read_text())
    assert side["world_size"] == 2 and side["global_batch"] == 8 and side["steps_per_epoch"] == one_meta["steps_per_epoch"]
    assert np.allclose([v for _, v in r0["history"]], [v for _, v in one_meta["history"]], atol=1e-5)
    for k, v in one_heads.state_dict().items():
        assert torch.allclose(r0["heads"][k], v, atol=1e-5), k
    for k, v in tiny_model.state_dict().items():
        assert torch.allclose(r0["model"][k], v, atol=1e-5), k
