"""experiments/train_set_model.py on synthetic folds (CPU): the training set, the cross-fitted base and the paired
base document, the saved models and their sidecars, the CLI."""
import json

import numpy as np
import pytest

import predict_submission as ps
import train_set_model as tsm
from aicup import ladder, setmodel_io
from aicup.aggregators import SetTransformer
from aicup.ladder_data import pack_questions
from aicup.rankmodel import _softmax
from aicup.settrain import crossfit_base_logits, predict_set_model
from tests.test_ladder import _synthetic


@pytest.fixture
def tiny_pma(monkeypatch):
    monkeypatch.setitem(ladder.SET_RUNGS, "r3_pma", ("staged", lambda d: SetTransformer(d, d_tok=16, heads=4, layers=0, hidden=32),
                                                     {"epochs": 1}))


def test_training_set_and_base(tmp_path):
    folds = (0, 1, 2)
    vec_dir, players = _synthetic(tmp_path, folds, per_rank=2)
    vec, fold_of = ladder.load_vectors(vec_dir, folds)
    ladder._check_players(players, vec, fold_of, folds)
    q = tsm.training_set(players, folds, 3)
    assert list(q.columns) == ["question_id", "rank", "player_id", "n_games", "game_keys", "fold"]
    assert q["fold"].tolist() == sorted(q["fold"].tolist()) and set(q["fold"]) == set(folds)
    assert len(q) == 3 * sum(len(players[f]) for f in folds)
    # one RandomState over the folds in ascending order, independent of the order --folds was given in
    assert tsm.training_set(players, (2, 0, 1), 3).equals(q)
    assert tsm.questions_sha(q) == tsm.questions_sha(q.copy()) and tsm.questions_sha(q) != tsm.questions_sha(q.iloc[::-1])
    base, paired, info = tsm.fit_base(q, vec)
    assert base.shape == (len(q), 10) and np.isfinite(base).all() and info["folds"] == [0, 1, 2] and info["n_train"] == len(q)
    assert np.isfinite(info["ce_train_crossfit"])
    # the cross-fit: fold f's rows come from a model that never saw fold f; the paired base saw every row
    kind, rep, make = ladder.STATIC_RUNGS[ladder.BASE_RUNG]
    X, y = ladder._features(q, vec, kind, rep, True), ladder._labels(q)
    fold = q["fold"].to_numpy()
    by_fold, _ = crossfit_base_logits(make, {f: X[fold == f] for f in folds}, {f: y[fold == f] for f in folds}, X[:1])
    np.testing.assert_array_equal(base, np.concatenate([by_fold[f] for f in folds]))
    np.testing.assert_array_equal(paired.predict_proba(X), make().fit(X, y).predict_proba(X))
    assert not np.allclose(_softmax(base), paired.predict_proba(X))
    doc = tsm.base_document(paired, folds, 3, len(q), tsm.questions_sha(q))
    assert (doc["name"], doc["train_folds"], doc["per_player"], doc["train_questions"]) == ("mlp_h32", [0, 1, 2], 3, len(q))
    assert doc["features"] == ps.FEATURE_NAMES and doc["profiles"] == list(ps.PROFILES) and doc["state"]["type"] == "mlp"


def test_train_saves_models_base_and_sidecars(tmp_path, tiny_pma, capsys):
    folds = (0, 1, 2)
    vec_dir, players = _synthetic(tmp_path, folds, per_rank=2)
    vec, _ = ladder.load_vectors(vec_dir, folds)
    out, base_out = tmp_path / "set", tmp_path / "models" / "rank_model_f012.json"
    saved = tsm.train("r3_pma", vec, players, folds, [0, 1], 2, out, base_out, device="cpu", log=print)
    printed = capsys.readouterr().out
    assert [p.name for p, _ in saved] == ["r3_pma_s0.pt", "r3_pma_s1.pt"] and all(p.exists() for p, _ in saved)
    assert "wrote the paired base" in printed and "best_epoch" in printed
    # the paired base is a rank_model.json the pipeline can load, trained on the same questions
    doc = json.loads(base_out.read_text())
    saved_doc, base_model = ps.load_rank_model(base_out)
    assert saved_doc["name"] == "mlp_h32" and doc["train_folds"] == [0, 1, 2] and doc["per_player"] == 2
    q = tsm.training_set(players, folds, 2)
    assert doc["train_questions"] == len(q) and doc["questions_sha256"] == tsm.questions_sha(q)
    assert set(ladder.PROVENANCE[:6]) <= set(doc)
    # the sidecars
    for (pt, meta), seed in zip(saved, (0, 1)):
        assert meta == json.loads(setmodel_io.sidecar_path(pt).read_text())
        assert (meta["rung"], meta["seed"], meta["train_folds"], meta["per_player"]) == ("r3_pma", seed, [0, 1, 2], 2)
        assert meta["n_train_questions"] == len(q) and meta["questions_sha256"] == doc["questions_sha256"]
        assert meta["base_model"] == base_out.name and meta["base_model_sha256"] == setmodel_io.file_sha256(base_out)
        assert meta["train_cfg"]["epochs"] == 1 and meta["train_cfg"]["seed"] == seed and meta["d_in"] == 301
        assert np.isfinite(meta["base_ce_train_crossfit"]) and meta["wall_seconds"] > 0
        fitted, _, warnings = setmodel_io.load_set_model(pt)
        assert warnings == [] and fitted.cfg.seed == seed
    # the two seeds differ; a loaded model predicts on new packed questions
    data = pack_questions(q.head(5), vec, "staged", True)
    base = np.zeros((5, 10))
    P0, P1 = (predict_set_model(setmodel_io.load_set_model(p)[0], data, base) for p, _ in saved)
    assert P0.shape == (5, 10) and np.isfinite(P0).all() and not np.array_equal(P0, P1)
    # existing outputs are refused unless overwrite
    with pytest.raises(FileExistsError, match="overwrite"):
        tsm.train("r3_pma", vec, players, folds, [0], 2, out, base_out, device="cpu", log=lambda *a: None)
    again = tsm.train("r3_pma", vec, players, folds, [0], 2, out, base_out, device="cpu", overwrite=True, log=lambda *a: None)
    np.testing.assert_array_equal(predict_set_model(setmodel_io.load_set_model(again[0][0])[0], data, base), P0)  # deterministic
    # another model paired with the same base (here another seed; the other rungs of an ensemble likewise) keeps
    # the base file untouched: the same training questions and parameters
    before = base_out.read_bytes()
    more = tsm.train("r3_pma", vec, players, folds, [2], 2, out, base_out, device="cpu", log=print)
    assert [p.name for p, _ in more] == ["r3_pma_s2.pt"] and base_out.read_bytes() == before
    assert "kept the existing paired base" in capsys.readouterr().out
    assert more[0][1]["base_model_sha256"] == again[0][1]["base_model_sha256"] == setmodel_io.file_sha256(base_out)
    # a base fitted on other training questions (per player 3) or with other parameters is refused, nothing written
    with pytest.raises(FileExistsError, match="other training questions"):
        tsm.train("r3_pma", vec, players, folds, [3], 3, out, base_out, device="cpu", log=lambda *a: None)
    assert not (out / "r3_pma_s3.pt").exists() and base_out.read_bytes() == before
    doc = json.loads(base_out.read_text())
    doc["state"]["params"]["0.bias"][0] += 0.5
    tampered = tmp_path / "models" / "tampered.json"
    tampered.write_text(json.dumps(doc))
    with pytest.raises(FileExistsError, match="other parameters"):
        tsm.train("r3_pma", vec, players, folds, [3], 2, out, tampered, device="cpu", log=lambda *a: None)
    assert not (out / "r3_pma_s3.pt").exists()
    with pytest.raises(ValueError, match="2 distinct folds"):
        tsm.train("r3_pma", vec, players, (1,), [0], 2, tmp_path / "x", tmp_path / "x.json", device="cpu")
    with pytest.raises(KeyError, match="unknown set rung"):
        tsm.train("r2_mlp24", vec, players, folds, [0], 2, tmp_path / "x", tmp_path / "x.json", device="cpu")


def test_cli_trains_on_fold0_too(tmp_path, tiny_pma, monkeypatch, capsys):
    folds = (0, 1, 2)
    vec_dir, players = _synthetic(tmp_path, folds, per_rank=2)
    monkeypatch.setattr(tsm, "load_players_all", lambda f: {k: players[k] for k in f})
    out, base_out = tmp_path / "set", tmp_path / "rank_model_f012.json"
    rc = tsm.main(["--rung", "r3_pma", "--folds", "2", "0", "1", "--seeds", "1", "--per-player", "2", "--vectors", str(vec_dir),
                   "--out", str(out), "--base-out", str(base_out), "--device", "cpu"])
    assert rc == 0 and (out / "r3_pma_s1.pt").exists() and (out / "r3_pma_s1.json").exists() and base_out.exists()
    assert json.loads(base_out.read_text())["train_folds"] == [0, 1, 2]  # fold 0 is training data here
    # the same command again: the model file exists, exit 2
    assert tsm.main(["--rung", "r3_pma", "--folds", "0", "1", "2", "--seeds", "1", "--per-player", "2", "--vectors", str(vec_dir),
                     "--out", str(out), "--base-out", str(base_out), "--device", "cpu"]) == 2
    assert "overwrite" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="fold3.npz"):
        tsm.main(["--rung", "r3_pma", "--folds", "0", "3", "--vectors", str(vec_dir), "--out", str(out), "--base-out", str(base_out)])
