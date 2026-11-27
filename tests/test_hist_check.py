"""aicup.seqshape.game_pmi_hist and experiments/hist_check.py: the per-move pmi distribution channel on top of fused_opp_diff (issue #29)."""
import json

import numpy as np
import pandas as pd
import pytest

import hist_check
from aicup import seqshape as ss
from aicup.evidence import posterior_mean_index
from aicup.metric import LEVELS
from aicup.scoring import read_scored_games
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def test_game_pmi_hist_bins_fractions_and_edges():
    lp = np.zeros((3, 10))
    lp[0, 0], lp[1, 9] = 60.0, 60.0  # move 0 at index 0, move 1 at index 9, move 2 uniform -> 4.5
    h = ss.game_pmi_hist(lp)
    assert ss.N_BINS == 10 and len(ss.HIST) == 10 and np.allclose(ss.HIST_EDGES, np.linspace(0, 9, 11))
    assert h.shape == (10,) and abs(h.sum() - 1) < 1e-12
    assert np.allclose(h, [1 / 3, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 1 / 3])  # 4.5 is the lower edge of bin 5; 9.0 falls in the last bin
    assert np.array_equal(ss.game_pmi_hist(np.zeros((0, 10))), np.zeros(10))
    rng = np.random.RandomState(3)
    lp = rng.randn(40, 10) * 3
    assert np.allclose(ss.move_pmi(lp), posterior_mean_index(lp)) and abs(ss.game_pmi_hist(lp).sum() - 1) < 1e-12


def test_game_shape_is_unchanged_by_the_pmi_helper():
    rng = np.random.RandomState(7)
    lp = rng.randn(25, 10).astype(np.float32)
    lp64 = lp.astype(np.float64)
    z = lp64 - lp64.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pmi = p @ np.arange(10)
    s = -lp64.max(axis=1)
    t = np.linspace(0.0, 1.0, 25)
    ref = np.array([pmi.mean(), np.log(25), pmi.std(), ss._ac(pmi, 1), ss._ac(pmi, 2), float(pmi[12:].mean() - pmi[:12].mean()),
                    float(np.polyfit(t, pmi, 1)[0]), s.mean(), ss._ac(s, 1)])
    assert np.array_equal(ss.game_shape(lp), ref)


def test_fold_hist_table_rows_are_games_and_sum_to_one(tmp_path):
    l1b_dir, _, _ = _synthetic_scores(tmp_path, folds=(2,))
    table = ss.fold_hist_table(l1b_dir / "fold2")
    games = read_scored_games(l1b_dir / "fold2")
    assert list(table.columns) == ["game_key", "player_id", "rank", "n_moves"] + ss.HIST and len(table) == len(games)
    assert set(table["game_key"]) == set(games["game_key"]) and np.allclose(table[ss.HIST].sum(axis=1), 1.0)
    # peaked synthetic scores: a rank's moves sit in the bin of its index, so the pooled histograms of adjacent ranks barely overlap
    dist = hist_check.rank_distribution([table])
    assert set(dist["by_rank"]) == set(LEVELS) and all(abs(sum(r["hist"]) - 1) < 1e-9 for r in dist["by_rank"].values())
    assert set(dist["adjacent_overlap"]) == {f"{a}|{b}" for a, b in zip(LEVELS[:-1], LEVELS[1:])} and all(0 <= v <= 1 for v in dist["adjacent_overlap"].values())
    assert np.argmax(dist["by_rank"]["D"]["hist"]) < np.argmax(dist["by_rank"]["A"]["hist"]) < np.argmax(dist["by_rank"]["6D"]["hist"])
    shape = ss.fold_shape_table(l1b_dir / "fold2")
    assert shape[ss.FEATURES].to_numpy().shape == (len(games), len(ss.FEATURES))
    assert np.array_equal(shape.to_numpy(), ss.fold_table(l1b_dir / "fold2", ss.game_shape, ss.FEATURES).to_numpy())


def test_hist_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "hist_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert hist_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert set(res["scores"]) == set(hist_check.SETS) and res["folds"] == [1, 2, 3, 4] and res["hist_features"] == ss.HIST and len(res["bin_edges"]) == 11
    assert {s: h["n_features"] for s, h in res["scores"].items()} == {"fused_opp": 56, "fused_opp_diff": 59, "fused_opp_diff_hist": 69, "fused_opp_diff_hist2": 79}
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_diff_hist vs fused_opp_diff"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_diff_hist2 vs fused_opp_diff", "fused_opp_diff_hist2 vs fused_opp_diff_hist", "fused_opp_diff vs fused_opp"}
    for h in res["scores"].values():
        assert len(h["per_fold"]) == 4 and 0 <= h["balanced_mean_over_n"] <= 1 and set(h["balanced_by_n"]) == {"5", "10", "20"}
    c = res["comparisons"]["pre_specified"]["fused_opp_diff_hist vs fused_opp_diff"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    dist = res["rank_distribution"]
    assert set(dist["by_rank"]) == set(LEVELS) and len(dist["adjacent_overlap"]) == len(LEVELS) - 1
    assert sum(r["games"] for r in dist["by_rank"].values()) == sum(len(read_scored_games(l1b_dir / f"fold{f}")) for f in (1, 2, 3, 4))
    rd = res["residual_diagnostics"]
    assert set(rd) == {"fused_opp", "fused_opp_diff"}
    for r in rd.values():
        assert set(r["opponent_channel"]) == {"spearman_label_own_pmi", "spearman_label_opp_pmi", "spearman_label_resid_on_own_vs_opp_minus_own",
                                              "spearman_oof_mean_e_vs_opp_minus_own", "spearman_own_opp"}
        assert set(r["spearman_mean_e"]) == {"own_pmi", "opp_pmi", "opp_minus_own"} and 0 <= r["exact"] <= 1 and r["exact"] <= r["within1"] <= 1
        q = r["question_level"]
        assert set(q["spearman_e"]) == set(q["spearman_mean_e"]) == set(hist_check.Q_COVARIATES) == {"q_own_pmi", "q_opp_pmi", "q_opp_minus_own"}
        assert set(q["spearman_e_by_n"]) <= {"5", "10", "20"} and all(set(v) == set(hist_check.Q_COVARIATES) for v in q["spearman_e_by_n"].values())
    assert "[pre_specified] fused_opp_diff_hist vs fused_opp_diff" in printed and "adjacent-rank overlap" in printed and "residuals of fused_opp_diff" in printed and "wrote" in printed
    assert "question-level (what the model saw)" in printed
    with pytest.raises(SystemExit):
        hist_check.main(args + ["--folds", "0", "1"])
    assert hist_check.main(args[:2] + ["--opp", str(tmp_path / "missing")] + args[4:]) == 2


def test_hist_features_are_the_question_means_of_the_game_histograms():
    table = pd.DataFrame({"game_key": ["a", "b"], **{h: [v, w] for h, v, w in zip(ss.HIST, np.eye(10)[0], np.eye(10)[5])}})
    qs = pd.DataFrame({"game_keys": [["a", "b"], ["b"]]})
    X = ss.question_shape(table, qs, cols=ss.HIST)
    assert X.shape == (2, 10) and np.allclose(X[0], (np.eye(10)[0] + np.eye(10)[5]) / 2) and np.allclose(X[1], np.eye(10)[5])


def test_question_level_spearmans_follow_the_covariate_the_model_saw():
    # residual e = pred - y rises with q_opp_minus_own strictly (no ties) -> Spearman +1 per question, per n and per player; q_own_pmi = q_opp_pmi - q_opp_minus_own
    oof = pd.DataFrame({"player_id": ["p1", "p1", "p2", "p2", "p3", "p3"], "n_games": [5, 20, 5, 20, 5, 20], "y": [4] * 6, "pred": [3, 4, 5, 6, 7, 8],
                        "q_opp_minus_own": [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0], "q_opp_pmi": [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]})
    oof["q_own_pmi"] = oof["q_opp_pmi"] - oof["q_opp_minus_own"]
    q = hist_check.question_level(oof)
    assert abs(q["spearman_e"]["q_opp_minus_own"] - 1) < 1e-12 and abs(q["spearman_e"]["q_opp_pmi"] - 1) < 1e-12
    assert set(q["spearman_e_by_n"]) == {"5", "20"} and all(abs(v["q_opp_minus_own"] - 1) < 1e-12 for v in q["spearman_e_by_n"].values())
    assert abs(q["spearman_mean_e"]["q_opp_minus_own"] - 1) < 1e-12
    assert hist_check.COL_DIFF == 56 and hist_check.COL_OPP == 57
