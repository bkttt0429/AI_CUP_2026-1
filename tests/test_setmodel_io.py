"""aicup.setmodel_io: a trained set model saved as .pt + sidecar predicts bit for bit the same when loaded."""
import dataclasses
import json
from functools import partial

import numpy as np
import pytest
import torch

from aicup import ladder, setmodel_io
from aicup.aggregators import CTM, SetTransformer, n_params
from aicup.humansl import PROFILES
from aicup.qfeatures import FEATURE_NAMES
from aicup.settrain import TrainCfg, fit_set_model, predict_set_model

M = 20


def _player_sets(ranks, q_per_player=6, n_games=20, d=301, seed=0):
    """Packed questions of synthetic players (one per rank entry), each question a random subset of the player's games."""
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


@pytest.fixture
def tiny_rungs(monkeypatch):
    monkeypatch.setitem(ladder.SET_RUNGS, "r3_pma", ("staged", lambda d: SetTransformer(d, d_tok=16, heads=4, layers=0, hidden=32), {}))
    monkeypatch.setitem(ladder.SET_RUNGS, "r5_ctm", ("staged", partial(CTM, d_tok=16, d_latent=32, history=4, nlm_hidden=8, heads=4,
                                                                   ticks=3, n_sync_action=24, n_sync_out=24), {}))


@pytest.mark.parametrize("rung", ["r3_pma", "r5_ctm"])
def test_round_trip_predicts_bit_for_bit(tmp_path, tiny_rungs, rung):
    data = _player_sets(np.arange(40) % 4)
    base = np.random.RandomState(1).randn(len(data["y"]), 10)
    cfg = TrainCfg(epochs=2, batch=32, lr=1e-2, device="cpu", seed=3)
    fitted = fit_set_model(ladder.SET_RUNGS[rung][1], data, base, cfg)
    P = predict_set_model(fitted, data, base)
    pt = tmp_path / "set" / f"{rung}_s3.pt"
    meta = setmodel_io.save_set_model(fitted, rung, 3, pt, extra={"train_folds": [0, 1, 2, 3, 4], "per_player": 32})
    assert pt.exists() and setmodel_io.sidecar_path(pt) == pt.with_suffix(".json") and pt.with_suffix(".json").exists()
    assert json.loads(pt.with_suffix(".json").read_text()) == meta  # strict JSON, round-trips
    assert (meta["rung"], meta["rep"], meta["seed"], meta["d_in"], meta["d_glob"]) == (rung, "staged", 3, 301, 6)
    assert meta["train_cfg"] == dataclasses.asdict(cfg) and meta["format"] == setmodel_io.FORMAT
    for k in setmodel_io.STATS:
        np.testing.assert_array_equal(np.asarray(meta[k]), getattr(fitted, k))
    assert meta["best_eval"] == fitted.best_eval and meta["best_epoch"] == fitted.best_epoch and meta["inner_ce"] == fitted.inner_ce
    assert meta["n_params"] == n_params(fitted.model) and meta["losses"] == fitted.losses and meta["train_folds"] == [0, 1, 2, 3, 4]
    assert meta["profiles"] == list(PROFILES) and meta["features"] == FEATURE_NAMES
    assert meta["state_sha256"] == setmodel_io.file_sha256(pt) and meta["code_sha256"] == ladder.code_version()["code_sha256"]
    assert {"git_head", "git_dirty", "torch_version", "device_name", "written"} <= set(meta)

    loaded, meta2, warnings = setmodel_io.load_set_model(pt, device="cpu")
    assert meta2 == meta and warnings == []
    assert loaded.cfg == cfg and type(loaded.model) is type(fitted.model) and not loaded.model.training
    for (ka, va), (kb, vb) in zip(fitted.model.state_dict().items(), loaded.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka
    for k in setmodel_io.STATS:
        np.testing.assert_array_equal(getattr(loaded, k), getattr(fitted, k))
    np.testing.assert_array_equal(predict_set_model(loaded, data, base), P)  # bit for bit
    assert (loaded.best_eval, loaded.best_epoch, loaded.stopped_early, loaded.n_inner_players) == (
        fitted.best_eval, fitted.best_epoch, fitted.stopped_early, fitted.n_inner_players)
    s = setmodel_io.summary(meta)
    assert s["rung"] == rung and s["seed"] == 3 and s["train_folds"] == [0, 1, 2, 3, 4] and s["base_model"] is None


def test_load_refusals_and_code_warning(tmp_path, tiny_rungs, monkeypatch):
    data = _player_sets(np.arange(20) % 2)
    base = np.zeros((len(data["y"]), 10))
    fitted = fit_set_model(ladder.SET_RUNGS["r3_pma"][1], data, base, TrainCfg(epochs=1, batch=32, device="cpu"))
    pt = tmp_path / "r3_pma_s0.pt"
    setmodel_io.save_set_model(fitted, "r3_pma", 0, pt)
    with pytest.raises(ValueError, match="unknown rung"):
        setmodel_io.save_set_model(fitted, "r9_nope", 0, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="shadow"):
        setmodel_io.save_set_model(fitted, "r3_pma", 0, tmp_path / "y.pt", extra={"rung": "other"})
    side = setmodel_io.sidecar_path(pt)
    meta = json.loads(side.read_text())

    def write(**changes):
        side.write_text(json.dumps({**meta, **changes}))

    write(format=99)
    with pytest.raises(ValueError, match="format"):
        setmodel_io.load_set_model(pt)
    write(rung="r9_nope")
    with pytest.raises(ValueError, match="unknown rung"):
        setmodel_io.load_set_model(pt)
    write(features=FEATURE_NAMES[:-1])
    with pytest.raises(ValueError, match="other profiles or features"):
        setmodel_io.load_set_model(pt)
    write(rep="whole")
    with pytest.raises(ValueError, match="rep"):
        setmodel_io.load_set_model(pt)
    write(code_sha256="0" * 64)
    _, _, warnings = setmodel_io.load_set_model(pt)
    assert len(warnings) == 1 and "other training code" in warnings[0] and "000000000000" in warnings[0]
    write()
    assert setmodel_io.load_set_model(pt)[2] == []
    pt.write_bytes(pt.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        setmodel_io.load_set_model(pt)
    side.unlink()
    with pytest.raises(FileNotFoundError, match="both needed"):
        setmodel_io.load_set_model(pt)
