"""experiments/xfit_check.py: the five-fold cross-fit vs A/B comparison and the label-smoothing variants (issue #16)."""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import l1b_cv
import l1b_opp_check as oc
import xfit_check as xc
from aicup.metric import LEVELS
from aicup.qeval import evaluate
from aicup.qfeatures import sample_questions
from aicup.rankmodel import MLPModel
from aicup.scoring import read_scored_games
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def _scaled_copy(tmp_path, name, scale):
    """An l1b_score.py layout of the synthetic games with the sums scaled (another model's scores of the same games)."""
    d, _, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name=name)
    for csv in sorted(d.glob("fold*/games_s*_p*.csv")):
        t = pd.read_csv(csv, dtype=str, keep_default_na=False)
        cols = [f"sum_{lv}" for lv in LEVELS]
        t[cols] = t[cols].astype(float) * scale
        t.to_csv(csv, index=False)
    return d


def test_xfit_check_cli(tmp_path, capsys):
    l1b_ab, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_ab = _opponent_scores(tmp_path, name="opp_ab")
    l1b_x5, opp_x5 = _scaled_copy(tmp_path, "l1b_x5", 1.1), _scaled_copy(tmp_path, "opp_x5", 0.8)
    for d in (l1b_x5, opp_x5):
        for f in range(5):  # one model per fold, as results/l1b/scores5
            (d / f"fold{f}" / "meta_s0.json").write_text(json.dumps({"heads": {"state_sha256": f"{f:064x}"}, "side": "target"}))
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "xfit_check.json"
    args = ["--l1b", str(l1b_ab), "--opp", str(opp_ab), "--xfit-l1b", str(l1b_x5), "--xfit-opp", str(opp_x5), "--hsl", str(hsl_dir),
            "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert xc.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert set(res["scores"]) == set(xc.SETS) and res["folds"] == [1, 2, 3, 4]
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_x5c vs fused_opp"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_c vs fused_opp", "fused_opp_x5c vs fused_opp_c", "fused_opp_x5 vs fused_opp",
                                                      "fused_opp_ls vs fused_opp", "fused_opp_ls2 vs fused_opp", "fused_opp_x5_ls vs fused_opp_x5",
                                                      "fused_opp_x5_ls vs fused_opp"}
    # centring: the synthetic A/B directories have no meta, so they are one model per directory (here one for all four folds);
    # the cross-fit directories carry one fake meta per fold, so four models, one bias vector each, per side
    assert [b["folds"] for b in res["head_bias"]["base"]["l1b"]] == [[1, 2, 3, 4]] and len(res["head_bias"]["x5"]["l1b"]) == 4
    assert sorted(f for b in res["head_bias"]["x5"]["opp"] for f in b["folds"]) == [1, 2, 3, 4] and all(len(b["bias"]) == 10 for b in res["head_bias"]["x5"]["l1b"])
    assert res["scores"]["fused_opp_x5c"]["per_fold"] != res["scores"]["fused_opp_x5"]["per_fold"] or res["scores"]["fused_opp_x5c"]["nll_balanced"] != res["scores"]["fused_opp_x5"]["nll_balanced"]
    for s, h in res["scores"].items():
        assert len(h["per_fold"]) == 4 and 0 <= h["balanced_mean_over_n"] <= 1 and set(h["balanced_by_n"]) == {"5", "10", "20"}
        assert (h["source"], h["label_smoothing"]) == xc.SETS[s]
    c = res["comparisons"]["pre_specified"]["fused_opp_x5c vs fused_opp"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool)
    assert res["dirs"]["base"]["1"]["l1b"] == str(l1b_ab) and res["dirs"]["x5"]["3"]["opp"] == str(opp_x5)
    assert "[pre_specified] fused_opp_x5c vs fused_opp" in printed and "bias x5 l1b folds [1]" in printed and "wrote" in printed
    # the fused_opp row is l1b_opp_check's fit: recompute validation fold 1 independently
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    data = {}
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(hsl_dir, l1b_ab, f)
        opp = read_scored_games(opp_ab / f"fold{f}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = (hsl, l1b, opp, players, l1b_cv.pilot_questions(players))
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + 1)
    train_q = pd.concat([sample_questions(data[f][3], 2, rng) for f in (2, 3, 4)], ignore_index=True)
    cat = lambda i: pd.concat([data[f][i] for f in (2, 3, 4)], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _ = oc.features(cat(0), cat(1), cat(2), train_q)
    X_va, y_va, _ = oc.features(data[1][0], data[1][1], data[1][2], data[1][4])
    r, _ = evaluate(MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr).predict_proba(X_va["fused_opp"]), y_va, data[1][4])
    assert res["scores"]["fused_opp"]["per_fold"][0] == r["balanced_mean_over_n"]
    # the same features with label smoothing differ from the plain fit; x5 reads the other directories
    assert res["scores"]["fused_opp_ls"]["per_fold"] != res["scores"]["fused_opp"]["per_fold"] or res["scores"]["fused_opp_ls"]["nll_balanced"] != res["scores"]["fused_opp"]["nll_balanced"]
    # fold 0 is refused; a cross-fit directory missing a fold is exit 2
    with pytest.raises(SystemExit):
        xc.main([*args, "--folds", "0", "1"])
    bad = _scaled_copy(tmp_path, "l1b_x5_short", 1.0)
    for p in (bad / "fold3").glob("*"):
        p.unlink()
    (bad / "fold3").rmdir()
    assert xc.main([*args[:4], "--xfit-l1b", str(bad), *args[6:]]) == 2
    assert "fold" in capsys.readouterr().err


def test_head_bias_and_centering():
    t = pd.DataFrame({"game_key": ["a", "b"], "n_moves": [10, 30], **{f"sum_{lv}": [-10.0 * (i + 1), -30.0 * (i + 1) - 3.0] for i, lv in enumerate(LEVELS)}})
    b = xc.head_bias([t])
    assert b.shape == (10,) and abs(b[0] - (-10.0 - 33.0) / 40) < 1e-12 and abs(b[1] - (-20.0 - 63.0) / 40) < 1e-12
    c = xc.center_table(t, b)
    assert np.allclose(xc.head_bias([c]), 0.0) and c["game_key"].tolist() == ["a", "b"] and c["n_moves"].tolist() == [10, 30]
    assert np.allclose(c["sum_D"], t["sum_D"] - t["n_moves"] * b[0])
    # two tables of one model share one bias; the key falls back to the directory when there is no meta
    assert np.allclose(xc.head_bias([t, t]), b)
    assert xc.model_key(pathlib.Path("/nonexistent/scores/fold3")).startswith("dir:")
