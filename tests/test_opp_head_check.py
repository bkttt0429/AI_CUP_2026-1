"""scripts/opp_head_check.py: route C's CV of the opponent-side heads' features on top of fused_opp_diff (issue #33)."""
import json

import numpy as np
import pandas as pd
import pytest

import opp_head_check as ohc
from aicup.metric import LEVELS
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def _c_scores(tmp_path, name="copp_ab", side="opponent", heads_side="opponent"):
    """An l1b_score.py --opponent layout under route-C heads: the opponent games of _opponent_scores with other sums, and
    per-fold metas saying which side was scored with which heads."""
    c_dir = _opponent_scores(tmp_path, name=name)
    for csv in sorted(c_dir.glob("fold*/games_s*_p*.csv")):
        t = pd.read_csv(csv, dtype=str, keep_default_na=False)
        cols = [f"sum_{lv}" for lv in LEVELS]
        t[cols] = t[cols].astype(float) * 1.1 + 0.2
        t.to_csv(csv, index=False)
    for fold_dir in sorted(p for p in c_dir.glob("fold*") if p.is_dir()):
        (fold_dir / "meta_s0.json").write_text(json.dumps({"side": side, "heads": {"side": heads_side, "train_folds": [9]}}))
    return c_dir


def test_features_dimensions_and_the_c_block(tmp_path):
    from aicup.qfeatures import build_matrix
    if True:
        tmp = tmp_path
        l1b_dir, hsl_dir, _ = _synthetic_scores(tmp, folds=(2,), l1b_name="l1b")
        opp_dir = _opponent_scores(tmp, name="opp")
        c_dir = _c_scores(tmp, name="copp")
        import l1b_cv
        from aicup.scoring import read_scored_games
        hsl, l1b = l1b_cv.load_fold(str(hsl_dir), str(l1b_dir), 2)
        opp, copp = read_scored_games(opp_dir / "fold2"), read_scored_games(c_dir / "fold2")
        folds_df = pd.read_csv(_folds_csv(tmp), dtype={"player_id": str})
        players = l1b_cv.players_from_scores(hsl, folds_df)
        qs = l1b_cv.pilot_questions(players)
        X, y, n = ohc.features(hsl, l1b, opp, copp, qs)
        assert {s: x.shape[1] for s, x in X.items()} == {"fused_opp_diff": 59, "fused_opp_diff_c": 75, "fused_opp_diff_c2": 78, "fused_c_diff": 59}
        C, _, _ = build_matrix(copp, qs, LEVELS)
        assert np.array_equal(X["fused_opp_diff_c"][:, 59:], C) and np.array_equal(X["fused_opp_diff_c2"][:, :75], X["fused_opp_diff_c"])
        assert np.array_equal(X["fused_c_diff"][:, :40], X["fused_opp_diff"][:, :40]) and np.array_equal(X["fused_c_diff"][:, 40:56], C)
        assert np.array_equal(X["fused_c_diff"][:, 56:], X["fused_opp_diff_c2"][:, 75:])
        assert len(y) == len(n) == len(qs)


def test_opp_head_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    c_dir = _c_scores(tmp_path)
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "opp_head_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--copp", str(c_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out)]
    assert ohc.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert set(res["scores"]) == set(ohc.SETS) and res["folds"] == [1, 2, 3, 4] and res["copp_dirs"] == [str(c_dir)]
    assert {s: h["n_features"] for s, h in res["scores"].items()} == {"fused_opp_diff": 59, "fused_opp_diff_c": 75, "fused_opp_diff_c2": 78, "fused_c_diff": 59}
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_diff_c vs fused_opp_diff"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_diff_c2 vs fused_opp_diff", "fused_c_diff vs fused_opp_diff", "fused_opp_diff_c2 vs fused_opp_diff_c"}
    c = res["comparisons"]["pre_specified"]["fused_opp_diff_c vs fused_opp_diff"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    st = res["channel_strength"]
    assert set(st) == {"players", "spearman_label_own_pmi", "spearman_label_opp_pmi", "spearman_label_c_pmi", "spearman_label_resid_on_own_vs_opp_minus_own",
                       "spearman_label_resid_on_own_vs_c_minus_own", "spearman_opp_c"}
    assert st["players"] == 4 * 10 * 3 and st["spearman_label_own_pmi"] > 0.9 and st["spearman_label_c_pmi"] > 0.9  # peaked synthetic scores
    assert "[pre_specified] fused_opp_diff_c vs fused_opp_diff" in printed and "channel strength" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        ohc.main(args + ["--folds", "0", "1"])
    assert ohc.main(args[:4] + ["--copp", str(tmp_path / "missing")] + args[6:]) == 2
    # --copp scored with target-side heads, or the target's moves, is refused
    bad = _c_scores(tmp_path, name="copp_bad", heads_side=None)
    assert ohc.main(args[:4] + ["--copp", str(bad)] + args[6:]) == 2 and "heads' side" in capsys.readouterr().err
    bad2 = _c_scores(tmp_path, name="copp_bad2", side="target")
    assert ohc.main(args[:4] + ["--copp", str(bad2)] + args[6:]) == 2 and "side" in capsys.readouterr().err
