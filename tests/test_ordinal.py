"""aicup.ordinal (CORAL and the pairwise Bradley-Terry score) and experiments/ordinal_check.py."""
import json

import numpy as np
import pandas as pd
import pytest

from aicup import ordinal
from aicup.metric import LEVELS
from aicup.rankmodel import MLPModel


def _ordinal_data(rng, n=600, d=5):
    """A latent strength z = x . beta plus noise, cut into 10 classes at its deciles."""
    X = rng.randn(n, d)
    beta = np.linspace(1.0, 2.0, d)
    z = X @ beta + 0.3 * rng.randn(n)
    y = np.digitize(z, np.quantile(z, np.arange(1, 10) / 10))
    return X, y, z


def _valid_probs(P):
    assert P.shape[1] == 10 and (P >= 0).all() and np.allclose(P.sum(axis=1), 1.0, atol=1e-9)


def test_coral_fits_ordinal_data_and_round_trips():
    rng = np.random.RandomState(0)
    X, y, z = _ordinal_data(rng)
    m = ordinal.CoralMLP(hidden=8, epochs=200).fit(X, y)
    P = m.predict_proba(X)
    _valid_probs(P)
    c = m.exceed_proba(X).numpy()
    assert (np.diff(c, axis=1) <= 1e-12).all()  # P(y > k) decreasing in k by construction
    pred = P.argmax(axis=1)
    assert (pred == y).mean() > 0.4 and np.abs(pred - y).mean() < 1.0  # ordinal data: close to the true class
    assert np.corrcoef(P @ np.arange(10), z)[0, 1] > 0.9
    m2 = ordinal.from_state(json.loads(json.dumps(m.state_dict())))
    np.testing.assert_array_equal(m2.predict_proba(X[:9]), P[:9])
    assert m.state_dict()["type"] == "coral"
    # deterministic per seed, different seeds differ
    np.testing.assert_array_equal(ordinal.CoralMLP(hidden=8, epochs=200).fit(X, y).predict_proba(X[:5]), P[:5])
    assert not np.allclose(ordinal.CoralMLP(hidden=8, epochs=200, seed=1).fit(X, y).predict_proba(X[:5]), P[:5])


def test_pairwise_score_orders_the_ranks_and_round_trips():
    rng = np.random.RandomState(1)
    X, y, z = _ordinal_data(rng)
    m = ordinal.PairwiseScoreMLP(hidden=8, epochs=200, link_epochs=200).fit(X, y)
    s = m.score(X)
    assert np.corrcoef(s, z)[0, 1] > 0.9  # the Bradley-Terry score recovers the latent order
    means = [s[y == k].mean() for k in range(10)]
    assert all(a < b for a, b in zip(means, means[1:]))
    P = m.predict_proba(X)
    _valid_probs(P)
    pred = P.argmax(axis=1)
    assert np.abs(pred - y).mean() < 1.0
    th = m.link.thresholds().detach().numpy()
    assert (np.diff(th) > 0).all() and float(m.link.scale().detach()) > 0
    m2 = ordinal.from_state(json.loads(json.dumps(m.state_dict())))
    np.testing.assert_allclose(m2.predict_proba(X[:9]), P[:9], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(m2.score(X[:9]), s[:9])
    # one rank only cannot be paired
    with pytest.raises(ValueError, match="two different ranks"):
        ordinal.PairwiseScoreMLP(hidden=8, epochs=5).fit(X[:20], np.zeros(20, dtype=int))
    # the delegate: an MLP state still restores
    mlp = MLPModel(hidden=4, epochs=5).fit(X, y)
    np.testing.assert_array_equal(ordinal.from_state(mlp.state_dict()).predict_proba(X[:3]), mlp.predict_proba(X[:3]))


def test_ordinal_check_cli(tmp_path, capsys):
    import l1b_cv
    import l1b_opp_check as oc
    import ordinal_check as ocheck
    from aicup.qeval import evaluate
    from aicup.scoring import read_scored_games
    from tests.test_fused_scripts import _folds_csv, _opponent_scores
    from tests.test_l1b_scripts import _synthetic_scores

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(1, 2, 3, 4), l1b_name="l1b_all")
    opp_dir = _opponent_scores(tmp_path)
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "ordinal_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "200",
            "--folds-csv", str(folds_csv), "--out", str(out)]
    assert ocheck.main(args) == 0
    printed = capsys.readouterr().out
    assert "coral vs mlp" in printed and "bt vs mlp" in printed and "wrote" in printed
    res = json.loads(out.read_text())
    assert set(res["scores"]) == {"mlp", "coral", "bt"} and res["feature_set"] == "fused_opp" and res["folds"] == [1, 2, 3, 4]
    assert set(res["comparisons"]) == {"coral vs mlp", "bt vs mlp"}
    for c in res["comparisons"].values():
        assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and "ci95" in c
    assert all(0.0 <= v["balanced_mean_over_n"] <= 1.0 and len(v["per_fold"]) == 4 for v in res["scores"].values())
    # the mlp row is l1b_opp_check's fused_opp fit: fold 1 recomputed with the same protocol gives the same number
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    data = {}
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(hsl_dir, l1b_dir, f)
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": read_scored_games(opp_dir / f"fold{f}"), "players": players, "val": l1b_cv.pilot_questions(players)}
    v = 1
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
    train_q = pd.concat([l1b_cv.sample_questions(data[f]["players"], 2, rng) for f in (2, 3, 4)], ignore_index=True)
    cat = lambda key: pd.concat([data[f][key] for f in (2, 3, 4)], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _ = oc.features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
    X_va, y_va, _ = oc.features(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], data[v]["val"])
    want = evaluate(MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr).predict_proba(X_va["fused_opp"]), y_va, data[v]["val"])[0]["balanced_mean_over_n"]
    assert res["scores"]["mlp"]["per_fold"][0] == want
    # the baseline is required; fold 0 is refused; a subset of models with another feature set runs
    with pytest.raises(SystemExit):
        ocheck.main([*args, "--models", "coral"])
    with pytest.raises(SystemExit):
        ocheck.main([*args, "--folds", "0", "1"])
    out2 = tmp_path / "hsl_check.json"
    assert ocheck.main([*args[:-2], "--out", str(out2), "--set", "hsl", "--models", "mlp", "bt"]) == 0
    res2 = json.loads(out2.read_text())
    assert set(res2["scores"]) == {"mlp", "bt"} and set(res2["comparisons"]) == {"bt vs mlp"} and res2["feature_set"] == "hsl"
    assert LEVELS[0] in res2["scores"]["mlp"]["per_class"]
