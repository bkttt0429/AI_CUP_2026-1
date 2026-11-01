"""aicup.seqtsfel and scripts/tsfel_check.py: TSFEL descriptors of the per-move score series on top of fused_opp_diff (issue #36)."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tsfel")

import tsfel_check  # noqa: E402
from aicup import seqtsfel as st  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from tests.test_fused_scripts import _folds_csv  # noqa: E402
from tests.test_l1b_scripts import _synthetic_scores  # noqa: E402


def _opponent_scores_long(tmp_path, name, moves, players_per_rank):
    """tests.test_fused_scripts._opponent_scores for synthetic games with `moves` moves (TSFEL needs >= MIN_MOVES)."""
    opp_dir, _, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name=name, moves=moves, players_per_rank=players_per_rank)
    for csv in sorted(opp_dir.glob("fold*/games_s*_p*.csv")):
        t = pd.read_csv(csv, dtype=str, keep_default_na=False)
        cols = [f"sum_{lv}" for lv in LEVELS]
        t[cols] = t[cols].astype(float) * 0.9 - 0.3
        t["color"] = t["color"].map({"B": "W", "W": "B"})
        t.to_csv(csv, index=False)
    return opp_dir


def test_game_tsfel_names_values_and_short_games():
    rng = np.random.RandomState(0)
    lp = rng.randn(60, 10) * 2
    f = st.game_tsfel(lp)
    assert f and all(k.startswith(("pmi_", "s_")) for k in f) and all(np.isfinite(v) for v in f.values())
    assert "pmi_Mean" in f and abs(f["pmi_Mean"] - st.game_series(lp)[0].mean()) < 1e-9
    assert any(k.startswith("pmi_Spectral") for k in f) and not any(w in k for k in f for w in st.WAVELET_BROKEN)
    assert st.game_tsfel(rng.randn(st.MIN_MOVES - 1, 10)) == {} and st.game_tsfel(np.zeros((0, 10))) == {}
    # the number of outputs depends on the length: the union is handled downstream
    g = st.game_tsfel(rng.randn(300, 10) * 2)
    assert set(f) <= set(g) or set(g) <= set(f) or len(set(f) ^ set(g)) > 0


def test_fold_table_cache_and_question_means(tmp_path):
    l1b_dir, _, _ = _synthetic_scores(tmp_path, folds=(2,), moves=20, players_per_rank=1, games_per_player=3)
    table = st.fold_tsfel_table(l1b_dir / "fold2", workers=2)
    games = read_scored_games(l1b_dir / "fold2")
    cols = st.feature_columns(table)
    assert list(table.columns[:4]) == ["game_key", "player_id", "rank", "n_moves"] and len(table) == len(games) and len(cols) > 50
    assert cols == sorted(cols) and set(table["game_key"]) == set(games["game_key"])
    path = tmp_path / "fold2.npz"
    st.save_tsfel_table(path, table)
    back = st.load_tsfel_table(path)
    assert list(back.columns) == list(table.columns) and np.allclose(back[cols].to_numpy(), table[cols].to_numpy(), equal_nan=True, atol=1e-6)
    keys = table["game_key"].tolist()
    qs = pd.DataFrame({"game_keys": [keys[:2], keys[1:3]]})
    X = st.question_tsfel(table, qs, cols)
    want = np.nanmean(table.set_index("game_key").loc[keys[:2], cols].to_numpy(dtype=float), axis=0)
    assert X.shape == (2, len(cols)) and np.allclose(X[0], want, equal_nan=True)
    # a column no game has gives NaN, and standardise_impute drops it and imputes the rest
    X2 = st.question_tsfel(table, qs, cols + ["pmi_Nonexistent"])
    assert np.isnan(X2[:, -1]).all()
    Z_tr, Z_va, keep = st.standardise_impute(np.vstack([X2, X2 + 1.0]), X2)
    assert len(keep) < X2.shape[1] and np.isfinite(Z_tr).all() and np.isfinite(Z_va).all() and abs(Z_tr.mean()) < 1e-9


def test_standardise_impute_and_pca():
    rng = np.random.RandomState(1)
    X_tr = rng.randn(50, 6)
    X_tr[:, 2] = 3.0  # constant -> dropped
    X_tr[::7, 4] = np.nan  # imputed by the training mean (0 after standardisation)
    X_va = rng.randn(5, 6)
    X_va[0, 0] = np.nan
    Z_tr, Z_va, keep = st.standardise_impute(X_tr, X_va)
    assert list(keep) == [0, 1, 3, 4, 5] and Z_tr.shape == (50, 5) and Z_va.shape == (5, 5)
    assert np.allclose(np.nanmean(Z_tr, axis=0), 0, atol=1e-9) and Z_va[0, 0] == 0.0 and np.isfinite(Z_tr).all()
    pca = st.pca_fit(Z_tr, 3)
    P = st.pca_apply(pca, Z_tr)
    assert pca["components"].shape == (5, 3) and P.shape == (50, 3) and abs(P.mean()) < 1e-9 and 0 < sum(pca["explained"]) <= 1 + 1e-9
    assert np.allclose(P.T @ P / 50, np.diag(np.diag(P.T @ P / 50)), atol=1e-9)  # uncorrelated components
    assert st.pca_fit(Z_tr, 99)["components"].shape[1] == 5  # capped at the rank
    assert st.pca_fit(Z_tr, 0)["components"].shape == (5, 0)


def test_tsfel_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab", moves=20, players_per_rank=2)
    opp_dir = _opponent_scores_long(tmp_path, "opp_ab", moves=20, players_per_rank=2)
    folds_csv = _folds_csv(tmp_path, players_per_rank=2)
    out, cache = tmp_path / "tsfel_check.json", tmp_path / "tsfel_cache"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv),
            "--pca", "2", "16", "--cache-dir", str(cache), "--workers", "2", "--out", str(out)]
    assert tsfel_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    sets = ["fused_opp_diff", "fused_opp_diff_tsfel2", "fused_opp_diff_tsfel16", "fused_opp_diff_tsfel_all"]
    assert list(res["scores"]) == sets and res["folds"] == [1, 2, 3, 4] and res["pca"] == [2, 16] and res["k_pre"] == 16
    dims = {s: h["n_features"] for s, h in res["scores"].items()}
    assert dims["fused_opp_diff"] == 59 and dims["fused_opp_diff_tsfel2"] == 61 and 59 < dims["fused_opp_diff_tsfel16"] <= 75
    assert dims["fused_opp_diff_tsfel_all"] > dims["fused_opp_diff_tsfel16"] and len(res["tsfel_feature_names"]) > 50
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_diff_tsfel16 vs fused_opp_diff"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_diff_tsfel2 vs fused_opp_diff", "fused_opp_diff_tsfel_all vs fused_opp_diff"}
    c = res["comparisons"]["pre_specified"]["fused_opp_diff_tsfel16 vs fused_opp_diff"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    for v in ("1", "2", "3", "4"):
        b = res["blocks"][v]
        assert b["kept_columns"] > 0 and set(b["pca"]) == {"2", "16"} and b["pca"]["2"]["components"] == 2 and 0 < b["pca"]["16"]["explained"] <= 1 + 1e-9
    gl = res["game_level"]
    assert gl["k"] == 16 and len(gl["per_game_logistic"]["mean_only"]) == 4 and len(gl["gain"]["balanced_acc"]) == 4
    assert sorted(p.name for p in cache.glob("fold*.npz")) == ["fold1.npz", "fold2.npz", "fold3.npz", "fold4.npz"]
    assert "[pre_specified] fused_opp_diff_tsfel16 vs fused_opp_diff" in printed and "game-level softmax" in printed and "TSFEL table built" in printed
    # the second run reads the cache
    assert tsfel_check.main(args) == 0 and "TSFEL table cached" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        tsfel_check.main(args + ["--folds", "0", "1"])
    with pytest.raises(SystemExit):
        tsfel_check.main([*args[:-5], "--pca", "4", "--cache-dir", str(cache), "--out", str(out)])
    assert tsfel_check.main(args[:2] + ["--opp", str(tmp_path / "missing")] + args[4:]) == 2
