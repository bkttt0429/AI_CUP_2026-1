"""aicup.fused: the fused (Human SL + L1b) question features, the fused model document and the L1b pairing check."""
import json

import numpy as np
import pandas as pd
import pytest

import l1b_cv
from aicup import fused
from aicup.humansl import PROFILES
from aicup.metric import LEVELS
from aicup.qfeatures import FEATURE_NAMES, feature_names
from aicup.rankmodel import MLPModel


def _tables(rng, n_games=12):
    keys = [f"g{i}" for i in range(n_games)]
    n = rng.randint(40, 200, size=n_games)
    hsl = pd.DataFrame({"game_key": keys, "n_moves": n, **{f"sum_{p}": -rng.rand(n_games) * 300 for p in PROFILES}})
    l1b = pd.DataFrame({"game_key": keys, "n_moves": n, **{f"sum_{lv}": -rng.rand(n_games) * 300 for lv in LEVELS}})
    questions = pd.DataFrame({"question_id": ["q1", "q2", "q3"], "rank": ["D", "3D", "6D"],
                              "game_keys": [keys[:5], keys[5:10], keys[10:]]})
    return hsl, l1b, questions


def test_fused_features_match_l1b_cv_feature_sets():
    rng = np.random.RandomState(0)
    hsl, l1b, questions = _tables(rng)
    assert len(fused.FUSED_FEATURE_NAMES) == 40 and fused.FUSED_FEATURE_NAMES[:24] == FEATURE_NAMES
    assert fused.FUSED_FEATURE_NAMES[24:] == [f"l1b_{n}" for n in feature_names(LEVELS)]
    want = l1b_cv.feature_sets(hsl, l1b, questions)[0]["fused"]
    h, l = hsl.set_index("game_key"), l1b.set_index("game_key")
    got = np.stack([fused.fused_features(h.loc[ks, [f"sum_{p}" for p in PROFILES]].to_numpy(), l.loc[ks, [f"sum_{lv}" for lv in LEVELS]].to_numpy(),
                                         h.loc[ks, "n_moves"].to_numpy()) for ks in questions["game_keys"]])
    np.testing.assert_array_equal(got, want)
    assert got.shape == (3, 40)


def test_fused_opp_features_match_l1b_opp_check():
    import l1b_opp_check as oc

    rng = np.random.RandomState(3)
    hsl, l1b, questions = _tables(rng)
    # the opponent's table: the same games, the opponent's own move counts (differ by up to one) and sums
    opp = pd.DataFrame({"game_key": hsl["game_key"], "n_moves": hsl["n_moves"].to_numpy() + rng.randint(-1, 2, size=len(hsl)),
                        **{f"sum_{lv}": -rng.rand(len(hsl)) * 300 for lv in LEVELS}})
    assert len(fused.FUSED_OPP_FEATURE_NAMES) == 56 and fused.FUSED_OPP_FEATURE_NAMES[:40] == fused.FUSED_FEATURE_NAMES
    assert fused.FUSED_OPP_FEATURE_NAMES[40:] == [f"opp_{n}" for n in feature_names(LEVELS)] == fused.OPP_FEATURE_NAMES
    assert fused.feature_names_for(True) == fused.FUSED_OPP_FEATURE_NAMES and fused.feature_names_for(False) == fused.FUSED_FEATURE_NAMES
    want = oc.features(hsl, l1b, opp, questions)[0]["fused_opp"]
    h, l, o = hsl.set_index("game_key"), l1b.set_index("game_key"), opp.set_index("game_key")
    lv_cols = [f"sum_{lv}" for lv in LEVELS]
    got = np.stack([fused.fused_opp_features(h.loc[ks, [f"sum_{p}" for p in PROFILES]].to_numpy(), l.loc[ks, lv_cols].to_numpy(), h.loc[ks, "n_moves"].to_numpy(),
                                             o.loc[ks, lv_cols].to_numpy(), o.loc[ks, "n_moves"].to_numpy()) for ks in questions["game_keys"]])
    np.testing.assert_array_equal(got, want)
    assert got.shape == (3, 56)
    # the first 40 columns are the fused features; the opponent's move counts matter for the last 16
    np.testing.assert_array_equal(got[:, :40], oc.features(hsl, l1b, opp, questions)[0]["fused"])
    other = np.stack([fused.fused_opp_features(h.loc[ks, [f"sum_{p}" for p in PROFILES]].to_numpy(), l.loc[ks, lv_cols].to_numpy(), h.loc[ks, "n_moves"].to_numpy(),
                                               o.loc[ks, lv_cols].to_numpy(), h.loc[ks, "n_moves"].to_numpy()) for ks in questions["game_keys"]])
    assert not np.array_equal(other[:, 40:], got[:, 40:])


def _models():
    return [{"name": "A", "heads_file": "heads3_A.pt", "state_sha256": "a" * 64, "trunk_file": "heads3_A.trunk.pt", "trunk_sha256": "1" * 64,
             "train_folds": [1, 2], "feature_folds": [3, 4]},
            {"name": "B", "heads_file": "heads3_B.pt", "state_sha256": "b" * 64, "trunk_file": "heads3_B.trunk.pt", "trunk_sha256": "2" * 64,
             "train_folds": [3, 4], "feature_folds": [1, 2]}]


def test_document_round_trip_and_checks(tmp_path):
    rng = np.random.RandomState(1)
    X, y = rng.randn(80, 40), np.arange(80) % 10
    mlp = MLPModel(hidden=4, epochs=5).fit(X, y)
    doc = fused.fused_document(mlp, _models(), train_folds=[1, 2, 3, 4], per_player=32, n_train=80, questions_sha256="c" * 64,
                               extra={"fold0_check": {"fused_avg": 0.6535}})
    assert doc["name"] == fused.NAME and doc["features"] == fused.FUSED_FEATURE_NAMES and doc["profiles"] == list(PROFILES)
    assert doc["levels"] == list(LEVELS) and doc["l1b_models"] == _models() and doc["fold0_check"]["fused_avg"] == 0.6535
    assert doc["train_folds"] == [1, 2, 3, 4] and doc["per_player"] == 32 and doc["train_questions"] == 80 and "code_sha256" in doc
    path = tmp_path / "rank_model_fused.json"
    fused.save_fused_model(doc, path)
    doc2, mlp2 = fused.load_fused_model(path)
    assert doc2["l1b_models"] == _models() and doc2["file_sha256"] == fused.file_sha256(path)
    np.testing.assert_array_equal(mlp2.predict_proba(X[:7]), mlp.predict_proba(X[:7]))
    broken = json.loads(path.read_text())
    broken["features"] = broken["features"][:-1]
    (tmp_path / "bad.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="features"):
        fused.load_fused_model(tmp_path / "bad.json")
    broken = json.loads(path.read_text())
    broken["l1b_models"] = []
    (tmp_path / "bad2.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="l1b_models"):
        fused.load_fused_model(tmp_path / "bad2.json")
    assert doc["opponent"] is False and doc2["opponent"] is False


def test_opponent_document_round_trip_and_checks(tmp_path):
    rng = np.random.RandomState(4)
    X, y = rng.randn(80, 56), np.arange(80) % 10
    mlp = MLPModel(hidden=4, epochs=5).fit(X, y)
    doc = fused.fused_document(mlp, _models(), train_folds=[1, 2, 3, 4], per_player=32, n_train=80, questions_sha256="c" * 64, opponent=True)
    assert doc["name"] == fused.NAME_OPP and doc["features"] == fused.FUSED_OPP_FEATURE_NAMES and doc["opponent"] is True
    assert "opponent" in doc["role"] and doc["l1b_models"] == _models()
    path = tmp_path / "rank_model_fused_opp.json"
    fused.save_fused_model(doc, path)
    doc2, mlp2 = fused.load_fused_model(path)
    assert doc2["opponent"] is True and doc2["name"] == fused.NAME_OPP
    np.testing.assert_array_equal(mlp2.predict_proba(X[:7]), mlp.predict_proba(X[:7]))
    # a document whose own flag disagrees with its features is refused, as is one with neither feature list
    broken = json.loads(path.read_text())
    broken["opponent"] = False
    (tmp_path / "bad.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="opponent flag"):
        fused.load_fused_model(tmp_path / "bad.json")
    broken = json.loads(path.read_text())
    broken["features"] = broken["features"][:50]
    (tmp_path / "bad2.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="neither"):
        fused.load_fused_model(tmp_path / "bad2.json")
    # an old 40-feature document without the flag loads as opponent False
    old = json.loads(path.read_text())
    old.pop("opponent")
    old.pop("feature_set")  # written before the key existed
    old["features"], old["name"] = fused.FUSED_FEATURE_NAMES, fused.NAME
    old["state"] = MLPModel(hidden=4, epochs=5).fit(X[:, :40], y).state_dict()
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert fused.load_fused_model(tmp_path / "old.json")[0]["opponent"] is False


def test_pairing_order_and_mismatch():
    doc = {"l1b_models": _models()}
    shard = [{"name": "heads3_B", "state_sha256": "b" * 64, "trunk_sha256": "2" * 64}, {"name": "heads3_A", "state_sha256": "a" * 64, "trunk_sha256": "1" * 64}]
    assert fused.pairing_order(doc, shard) == [1, 0]  # doc's A is shard column 1, doc's B is shard column 0
    with pytest.raises(ValueError, match="B"):
        fused.pairing_order(doc, shard[1:])
    with pytest.raises(ValueError, match="not in the fused model"):
        fused.pairing_order(doc, shard + [{"name": "heads3_C", "state_sha256": "c" * 64, "trunk_sha256": "3" * 64}])
    with pytest.raises(ValueError, match="trunk"):
        fused.pairing_order(doc, [{**shard[0], "trunk_sha256": "9" * 64}, shard[1]])
    with pytest.raises(ValueError, match="L1b scores"):
        fused.pairing_order(doc, [])


def test_fused_predict_averages_the_models():
    rng = np.random.RandomState(2)
    mlp = MLPModel(hidden=4, epochs=5).fit(rng.randn(60, 40), np.arange(60) % 10)
    m = 6
    hsl = -rng.rand(m, 18) * 200
    l1b = -rng.rand(m, 2, 10) * 200  # shard order: column 0 = B, column 1 = A
    n = rng.randint(50, 150, size=m)
    p = fused.fused_predict(mlp, hsl, l1b, n, order=[1, 0])
    want = np.mean([mlp.predict_proba(fused.fused_features(hsl, l1b[:, k], n)[None])[0] for k in (1, 0)], axis=0)
    np.testing.assert_allclose(p, want, rtol=0, atol=1e-15)
    assert p.shape == (10,) and abs(p.sum() - 1) < 1e-9
    assert not np.allclose(p, mlp.predict_proba(fused.fused_features(hsl, l1b[:, 0], n)[None])[0])


def test_fused_predict_with_the_opponent_features():
    rng = np.random.RandomState(6)
    mlp = MLPModel(hidden=4, epochs=5).fit(rng.randn(60, 56), np.arange(60) % 10)
    m = 5
    hsl, l1b, opp = -rng.rand(m, 18) * 200, -rng.rand(m, 2, 10) * 200, -rng.rand(m, 2, 10) * 200
    n, n_opp = rng.randint(50, 150, size=m), rng.randint(50, 150, size=m)
    p = fused.fused_predict(mlp, hsl, l1b, n, order=[1, 0], opp_sums=opp, opp_n_moves=n_opp)
    want = np.mean([mlp.predict_proba(fused.fused_opp_features(hsl, l1b[:, k], n, opp[:, k], n_opp)[None])[0] for k in (1, 0)], axis=0)
    np.testing.assert_allclose(p, want, rtol=0, atol=1e-15)
    assert p.shape == (10,) and abs(p.sum() - 1) < 1e-9
    # each model's opponent column goes with its own L1b column: swapping the opponent columns changes the result
    swapped = fused.fused_predict(mlp, hsl, l1b, n, order=[1, 0], opp_sums=opp[:, ::-1], opp_n_moves=n_opp)
    assert not np.allclose(swapped, p)


def test_fused_opp_diff_features_match_error_analysis():
    import error_analysis as ea
    from aicup.evidence import posterior_mean_index

    rng = np.random.RandomState(7)
    hsl, l1b, questions = _tables(rng)
    opp = pd.DataFrame({"game_key": hsl["game_key"], "n_moves": hsl["n_moves"].to_numpy() + rng.randint(-1, 2, size=len(hsl)),
                        **{f"sum_{lv}": -rng.rand(len(hsl)) * 300 for lv in LEVELS}})
    assert len(fused.FUSED_OPP_DIFF_FEATURE_NAMES) == 59 and fused.FUSED_OPP_DIFF_FEATURE_NAMES[:56] == fused.FUSED_OPP_FEATURE_NAMES
    assert fused.FUSED_OPP_DIFF_FEATURE_NAMES[56:] == fused.DIFF_FEATURE_NAMES == ["opp_minus_own_pmi_mean", "opp_pmi_mean", "opp_pmi_std"]
    assert fused.feature_names_for("fused_opp_diff") == fused.FUSED_OPP_DIFF_FEATURE_NAMES and fused.feature_set_of(True) == "fused_opp"
    assert fused.feature_set_of(False, "fused_opp_diff") == "fused_opp_diff"
    with pytest.raises(ValueError, match="unknown feature set"):
        fused.feature_set_of(feature_set="fused_x")
    # the error-analysis script's mlp_diff columns, column for column
    want = ea.feature_sets(hsl, l1b, opp, questions)[0]["fused_opp_diff"]
    h, l, o = hsl.set_index("game_key"), l1b.set_index("game_key"), opp.set_index("game_key")
    lv_cols = [f"sum_{lv}" for lv in LEVELS]
    got = np.stack([fused.fused_opp_diff_features(h.loc[ks, [f"sum_{p}" for p in PROFILES]].to_numpy(), l.loc[ks, lv_cols].to_numpy(), h.loc[ks, "n_moves"].to_numpy(),
                                                  o.loc[ks, lv_cols].to_numpy(), o.loc[ks, "n_moves"].to_numpy()) for ks in questions["game_keys"]])
    np.testing.assert_array_equal(got[:, :56], want[:, :56])
    np.testing.assert_allclose(got[:, 56:], want[:, 56:], rtol=0, atol=1e-12)  # the same formula on differently laid-out arrays
    assert got.shape == (3, 59)
    # the three differences by hand: posterior-mean index of each game's ten sums
    ks = questions["game_keys"][0]
    own, op = posterior_mean_index(l.loc[ks, lv_cols].to_numpy()), posterior_mean_index(o.loc[ks, lv_cols].to_numpy())
    np.testing.assert_allclose(got[0, 56:], [(op - own).mean(), op.mean(), op.std()], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fused.question_diff_features(l1b, opp, questions), got[:, 56:], rtol=0, atol=1e-12)
    assert fused.question_diff_features(l1b, opp, questions.iloc[:0]).shape == (0, 3)


def test_diff_document_round_trip_and_predict(tmp_path):
    rng = np.random.RandomState(8)
    X, y = rng.randn(80, 59), np.arange(80) % 10
    mlp = MLPModel(hidden=4, epochs=5).fit(X, y)
    doc = fused.fused_document(mlp, _models(), train_folds=[1, 2, 3, 4], per_player=32, n_train=80, questions_sha256="c" * 64, feature_set="fused_opp_diff")
    assert doc["name"] == fused.NAME_OPP_DIFF and doc["features"] == fused.FUSED_OPP_DIFF_FEATURE_NAMES and doc["opponent"] is True
    assert doc["feature_set"] == "fused_opp_diff" and "opponent-minus-own" in doc["role"]
    path = tmp_path / "rank_model_fused_opp_diff.json"
    fused.save_fused_model(doc, path)
    doc2, mlp2 = fused.load_fused_model(path)
    assert doc2["feature_set"] == "fused_opp_diff" and doc2["opponent"] is True and doc2["name"] == fused.NAME_OPP_DIFF
    np.testing.assert_array_equal(mlp2.predict_proba(X[:7]), mlp.predict_proba(X[:7]))
    # a document whose feature_set disagrees with its features is refused; an older fused_opp document without the key loads
    broken = json.loads(path.read_text())
    broken["feature_set"] = "fused_opp"
    (tmp_path / "bad.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="feature_set"):
        fused.load_fused_model(tmp_path / "bad.json")
    old = fused.fused_document(MLPModel(hidden=4, epochs=5).fit(X[:, :56], y), _models(), [1, 2, 3, 4], 32, 80, "c" * 64, opponent=True)
    old.pop("feature_set")
    (tmp_path / "old.json").write_text(json.dumps(old))
    d_old = fused.load_fused_model(tmp_path / "old.json")[0]
    assert d_old["feature_set"] == "fused_opp" and d_old["opponent"] is True
    # prediction: the diff features per L1b model, averaged; the set must be asked for, and it needs the opponent's sums
    m = 5
    hsl, l1b, opp = -rng.rand(m, 18) * 200, -rng.rand(m, 2, 10) * 200, -rng.rand(m, 2, 10) * 200
    n, n_opp = rng.randint(50, 150, size=m), rng.randint(50, 150, size=m)
    p = fused.fused_predict(mlp, hsl, l1b, n, order=[1, 0], opp_sums=opp, opp_n_moves=n_opp, feature_set="fused_opp_diff")
    want = np.mean([mlp.predict_proba(fused.fused_opp_diff_features(hsl, l1b[:, k], n, opp[:, k], n_opp)[None])[0] for k in (1, 0)], axis=0)
    np.testing.assert_allclose(p, want, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="opponent"):
        fused.fused_predict(mlp, hsl, l1b, n, order=[1, 0], feature_set="fused_opp_diff")
    # without feature_set the opponent sums still mean fused_opp (the old call)
    mlp56 = MLPModel(hidden=4, epochs=5).fit(X[:, :56], y)
    np.testing.assert_array_equal(fused.fused_predict(mlp56, hsl, l1b, n, [0, 1], opp_sums=opp, opp_n_moves=n_opp),
                                  fused.fused_predict(mlp56, hsl, l1b, n, [0, 1], opp_sums=opp, opp_n_moves=n_opp, feature_set="fused_opp"))

