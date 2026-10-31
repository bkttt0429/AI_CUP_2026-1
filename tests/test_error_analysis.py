"""scripts/error_analysis.py: OOF residual analysis and the two pre-specified model-side checks (issue #23)."""
import json

import numpy as np
import pandas as pd
import pytest

import error_analysis as ea
from aicup.metric import LEVELS
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def test_diff_features_and_entropy():
    l1b = pd.DataFrame({"game_key": ["a", "b"], **{f"sum_{lv}": [0.0, 0.0] for lv in LEVELS}})
    l1b.loc[0, "sum_6D"] = 50.0  # game a: posterior mass on index 9; game b: uniform -> 4.5
    opp = l1b.copy()
    opp.loc[0, "sum_6D"], opp.loc[0, "sum_D"] = 0.0, 50.0  # opponent of a at index 0
    pmi_t, pmi_o = ea.game_pmi(l1b), ea.game_pmi(opp)
    assert np.allclose(pmi_t[["a", "b"]], [9.0, 4.5]) and np.allclose(pmi_o[["a", "b"]], [0.0, 4.5])
    D = ea.diff_features(pmi_t, pmi_o, pd.DataFrame({"game_keys": [["a", "b"], ["b"]]}))
    assert np.allclose(D, [[(-9.0 + 0.0) / 2, 2.25, 2.25], [0.0, 4.5, 0.0]])
    ent = ea.game_entropy(l1b, profiles=LEVELS)
    assert ent["a"] < 1e-6 and abs(ent["b"] - np.log(10)) < 1e-9


@pytest.mark.parametrize("gbm", [False, True])
def test_error_analysis_cli(tmp_path, capsys, gbm):
    if gbm:
        pytest.importorskip("xgboost")
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    out, proba = tmp_path / "error_analysis.json", tmp_path / "oof.csv"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv),
            "--out", str(out), "--proba-out", str(proba)] + ([] if gbm else ["--no-gbm"])
    assert ea.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    models = ["mlp", "mlp_diff", "gbm", "mlp_h128"] if gbm else ["mlp", "mlp_diff", "mlp_h128"]
    assert res["models"] == models and set(res["scores"]) == set(models) and res["folds"] == [1, 2, 3, 4]
    assert set(res["comparisons"]["pre_specified"]) == ({"mlp_diff vs mlp", "gbm vs mlp"} if gbm else {"mlp_diff vs mlp"})
    assert set(res["comparisons"]["report_only"]) == {"mlp_h128 vs mlp"}
    for h in res["scores"].values():
        assert len(h["per_fold"]) == 4 and 0 <= h["balanced_mean_over_n"] <= 1
    r = res["residuals"]
    assert r["questions"] == res["questions"] and r["players"] == 120
    assert r["between_player_share"] is None or 0 <= r["between_player_share"] <= 1  # None when the synthetic predictions are all exact
    assert set(r["spearman_with_abs_mean_e"]) == set(ea.COVARIATES) and set(r["covariate_means"]) == {"all_wrong", "all_right", "other"}
    assert sum(g["players"] for g in r["covariate_means"].values()) == 120 and set(r["by_rank"]) == set(LEVELS)
    assert 0 <= r["all_right"]["observed"] <= 1 and 0 <= r["all_right"]["independent"] <= 1 and len(r["all_wrong_by_train_games_quintile"]) >= 1
    o = res["opponent_channel"]
    assert set(o) == {"spearman_label_own_pmi", "spearman_label_opp_pmi", "spearman_label_resid_on_own_vs_opp_minus_own", "spearman_oof_mean_e_vs_opp_minus_own", "spearman_own_opp"}
    assert o["spearman_label_own_pmi"] > 0.9, "the synthetic per-move scores peak at the player's rank"
    t = pd.read_csv(proba, dtype={"player_id": str, "question_id": str})
    assert len(t) == res["questions"] and set(t["fold"]) == {1, 2, 3, 4} and all(f"fused_opp_p_{lv}" in t.columns for lv in LEVELS)
    assert np.allclose(t[[f"fused_opp_p_{lv}" for lv in LEVELS]].sum(axis=1), 1.0, atol=1e-6) and t["pred"].between(0, 9).all()
    assert "[pre_specified] mlp_diff vs mlp" in printed and "between-player share" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        ea.main(args + ["--folds", "0", "1"])
