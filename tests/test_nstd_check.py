"""aicup.nstd and scripts/nstd_check.py: standardisation per question size n on fused_opp_diff (issue #53)."""
import json

import numpy as np
import pytest

import nstd_check
from aicup.nstd import N_MAX, N_MIN, NStdMLP, group_of
from aicup.rankmodel import MLPModel, _standardise
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def test_nstd_mlp_groups_keep_global_and_nearest_group():
    rng = np.random.RandomState(3)
    n = rng.randint(5, 21, size=400)
    X = rng.randn(400, 6) * (1 + n[:, None] / 10.0) + n[:, None]  # spread and level depend on n
    X[:, 4] = np.log(n)  # the log n column: constant within a group
    X[:, 5] = 7.0  # a constant column
    y = rng.randint(0, 10, size=400)
    m = NStdMLP(keep_global=[4], hidden=4, epochs=3).fit(X, y, n)
    Z = m.transform(X, n)
    for k in range(N_MIN, N_MAX + 1):
        sel = n == k
        if sel.sum() >= 2:
            assert np.allclose(Z[sel][:, :4].mean(axis=0), 0, atol=1e-9) and np.allclose(Z[sel][:, :4].std(axis=0), 1, atol=1e-9)
            assert np.allclose(Z[sel][:, 5], 0)  # constant column: centred, scale 1
    gmean, gstd = _standardise(X)
    np.testing.assert_allclose(Z[:, 4], (X[:, 4] - gmean[4]) / gstd[4])  # kept on the global statistics
    assert group_of([3, 5, 20, 27]).tolist() == [5, 5, 20, 20]
    np.testing.assert_array_equal(m.transform(X[:3], [27, 27, 27]), m.transform(X[:3], [20, 20, 20]))
    table = m.group_table([f"c{i}" for i in range(6)])
    assert set(table) == {str(k) for k in range(N_MIN, N_MAX + 1)} and sum(g["questions"] for g in table.values()) == 400
    assert table["10"]["n_constant"] >= 2 and set(table["10"]["std"]) == {f"c{i}" for i in range(6)} and table["10"]["from_global"] is False
    p = m.predict_proba(X[:5], n[:5])
    assert p.shape == (5, 10) and np.allclose(p.sum(axis=1), 1)
    # MLPModel(standardise=False) on globally standardised inputs is the ordinary MLPModel, bit for bit
    a = MLPModel(hidden=4, epochs=5).fit(X, y)
    b = MLPModel(hidden=4, epochs=5, standardise=False).fit((X - gmean) / gstd, y)
    np.testing.assert_array_equal(a.predict_proba(X[:7]), b.predict_proba(((X - gmean) / gstd)[:7]))
    from aicup.rankmodel import from_state
    assert from_state(b.state_dict()).standardise is False and from_state(a.state_dict()).standardise is True


def test_nstd_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "nstd_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert nstd_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert list(res["scores"]) == list(nstd_check.SETS) and all(h["n_features"] == 59 for h in res["scores"].values())
    assert res["log_games_cols"] == [22, 38, 54] and res["n_groups"] == [5, 20] and len(res["feature_names"]) == 59
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_diff_nstd vs fused_opp_diff"}
    c = res["comparisons"]["pre_specified"]["fused_opp_diff_nstd vs fused_opp_diff"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and set(c["by_n"]) == {"5", "10", "20"} and "ci95" in c["by_n"]["5"]
    assert set(res["groups"]) == {"1", "2", "3", "4"} and set(res["groups"]["1"]) == {str(k) for k in range(5, 21)}
    g = res["groups"]["1"]["5"]
    assert g["questions"] > 0 and len(g["std"]) == 59 and "l1b_log_games" in g["std"] and isinstance(g["n_constant"], int)
    assert sum(v["questions"] for v in res["groups"]["1"].values()) == 2 * 30 * 3  # 2 questions per player, 30 players, 3 training folds
    assert np.isfinite([h["balanced_mean_over_n"] for h in res["scores"].values()]).all()
    assert "[pre_specified] fused_opp_diff_nstd vs fused_opp_diff" in printed and "median column std" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        nstd_check.main(args + ["--folds", "0", "1"])
    assert nstd_check.main(args[:2] + ["--opp", str(tmp_path / "missing")] + args[4:]) == 2
