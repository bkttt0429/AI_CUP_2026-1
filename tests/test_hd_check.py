"""scripts/hd_check.py: the handicap features on top of fused_opp / fused_opp_diff (issue #42, proposal (a))."""
import json

import numpy as np
import pandas as pd
import pytest

import hd_check
from aicup.handicap import HD_COLUMNS
from aicup.scoring import read_scored_games
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def _hd_table(hsl_dir, folds, path, seed=0):
    """An hd_table over the synthetic games: HD 0 for most, 1 for some, 2-9 for a few (the stronger rank is W)."""
    rng = np.random.RandomState(seed)
    frames = []
    for f in folds:
        g = read_scored_games(hsl_dir / f"fold{f}")[["game_key", "player_id", "rank", "color"]].copy()
        hd = rng.choice([0, 0, 0, 0, 1, 2, 3, 9], size=len(g)).astype(float)
        hd[:3] = np.nan  # a few records without HD
        g["hd"] = hd
        frames.append(g)
    t = pd.concat(frames, ignore_index=True)[HD_COLUMNS]
    t.to_csv(path, index=False)
    return t


def test_hd_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    hd_path = tmp_path / "hd_table.csv"
    _hd_table(hsl_dir, (0, 1, 2, 3, 4), hd_path)
    out = tmp_path / "hd_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--hd-table", str(hd_path), "--per-player", "2", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out)]
    assert hd_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert list(res["scores"]) == list(hd_check.SETS) and res["folds"] == [1, 2, 3, 4] and res["hd_feature_names"][0] == "hd_even"
    dims = {s: h["n_features"] for s, h in res["scores"].items()}
    assert dims == {"fused_opp": 56, "fused_opp_diff": 59, "fused_opp_hd": 61, "fused_opp_diff_hd": 64}
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_diff_hd vs fused_opp_diff"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_hd vs fused_opp", "fused_opp_diff_hd vs fused_opp_hd", "fused_opp_diff vs fused_opp"}
    c = res["comparisons"]["pre_specified"]["fused_opp_diff_hd vs fused_opp_diff"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    for f in ("1", "2", "3", "4"):
        cov = res["coverage"][f]
        assert cov["games"] == 600 and 0 < cov["share_hd2"] < cov["share_hd1"] < 1
    # 30 players per fold, 7 pilot questions each (1 x 20, 2 x 10, 4 x 5), four validation folds
    assert set(res["buckets"]) == set(hd_check.BUCKETS) and sum(b["questions"] for b in res["buckets"].values()) == 4 * 30 * 7
    assert set(res["residual"]) == {"spearman_residual_vs_gap", "by_n", "mean_residual_by_gap_sign"} and set(res["residual"]["by_n"]) == {"5", "10", "20"}
    assert np.isfinite([h["balanced_mean_over_n"] for h in res["scores"].values()]).all()
    assert "[pre_specified] fused_opp_diff_hd vs fused_opp_diff" in printed and "bucket no_hd" in printed and "residual vs signed gap" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        hd_check.main(args + ["--folds", "0", "1"])
    # a table that lacks scored games is refused
    short = tmp_path / "short.csv"
    pd.read_csv(hd_path, dtype=str, keep_default_na=False).iloc[:-5].to_csv(short, index=False)
    assert hd_check.main(args[:7] + [str(short)] + args[8:]) == 2 and "are not in" in capsys.readouterr().err
    assert hd_check.main(args[:2] + ["--opp", str(tmp_path / "missing")] + args[4:]) == 2
