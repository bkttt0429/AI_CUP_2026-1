"""The fused-model scripts on synthetic scores: fit_fused_model.py (training as l1b_fold0_check, fold-0 reproduction,
per-question fold-0 probabilities)."""
import json

import numpy as np
import pandas as pd

import fit_fused_model as ffm
import l1b_cv
from aicup import fused
from aicup.metric import LEVELS
from aicup.rankmodel import from_state
from tests.test_l1b_scripts import _synthetic_scores


def _sidecar(path, state_sha, trunk_sha, train_folds):
    path.write_text(json.dumps({"format": 1, "n_ranks": 10, "levels": LEVELS, "state_sha256": state_sha, "trunk_file": path.stem + ".trunk.pt",
                                "trunk_sha256": trunk_sha, "train_folds": train_folds, "code_sha256": "x" * 64}))
    return path


def _folds_csv(tmp_path, folds=(0, 1, 2, 3, 4), players_per_rank=3):
    rows = [{"rank": lv, "player_id": f"{r:04x}{fold:02x}{p:02x}", "n_games": 20, "fold": fold}
            for fold in folds for r, lv in enumerate(LEVELS) for p in range(players_per_rank)]
    path = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fit_fused_model_cli(tmp_path, capsys):
    # one synthetic L1b scoring of folds 0-4 serves both "models"; the sidecars' train_folds decide which model feeds which fold
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_all")
    side_a = _sidecar(tmp_path / "heads3_A.json", "a" * 64, "1" * 64, [1, 2])
    side_b = _sidecar(tmp_path / "heads3_B.json", "b" * 64, "2" * 64, [3, 4])
    folds_csv = _folds_csv(tmp_path)
    out, proba = tmp_path / "models" / "rank_model_fused.json", tmp_path / "fold0_fused_proba.csv"
    args = ["--l1b", f"A:{l1b_dir}:{side_a}", f"B:{l1b_dir}:{side_b}", "--hsl", str(hsl_dir), "--per-player", "2",
            "--folds-csv", str(folds_csv), "--out", str(out), "--fold0-proba", str(proba)]
    assert ffm.main(args) == 0
    printed = capsys.readouterr().out
    assert "fused_avg" in printed and "hsl" in printed
    doc, mlp = fused.load_fused_model(out)
    assert doc["name"] == fused.NAME and doc["train_folds"] == [1, 2, 3, 4] and doc["per_player"] == 2
    assert [m["name"] for m in doc["l1b_models"]] == ["A", "B"]
    assert doc["l1b_models"][0] == {"name": "A", "heads_file": "heads3_A.pt", "state_sha256": "a" * 64, "trunk_file": "heads3_A.trunk.pt",
                                    "trunk_sha256": "1" * 64, "train_folds": [1, 2], "feature_folds": [3, 4]}
    assert doc["l1b_models"][1]["feature_folds"] == [1, 2] and doc["feature_source_by_fold"] == {"1": "B", "2": "B", "3": "A", "4": "A"}
    assert doc["train_questions"] == 4 * 30 * 2 and len(doc["questions_sha256"]) == 64
    f0 = doc["fold0_check"]
    assert set(f0["scores"]) == {"hsl", "fused_A", "fused_B", "fused_avg"} and f0["n_questions"] == 30 * 7 and f0["n_players"] == 30
    assert all(0.0 <= v["balanced_mean_over_n"] <= 1.0 for v in f0["scores"].values()) and "ci95" in f0["vs_hsl"]["fused_avg"]
    # the training reproduces l1b_fold0_check.py's fit: the same questions, features and MLP give the same fold-0 numbers
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    src = {1: l1b_dir, 2: l1b_dir, 3: l1b_dir, 4: l1b_dir}
    data = {f: dict(zip(("hsl", "l1b"), l1b_cv.load_fold(hsl_dir, src[f], f))) for f in (1, 2, 3, 4)}
    for f in data:
        data[f]["players"] = l1b_cv.players_from_scores(data[f]["hsl"], folds_df)
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([l1b_cv.sample_questions(data[f]["players"], 2, rng) for f in (1, 2, 3, 4)], ignore_index=True)
    X_tr, y_tr, _ = l1b_cv.feature_sets(pd.concat([data[f]["hsl"] for f in data]), pd.concat([data[f]["l1b"] for f in data]), train_q)
    ref = ffm.MLPModel(hidden=32).fit(X_tr["fused"], y_tr)
    np.testing.assert_array_equal(from_state(doc["state"]).predict_proba(X_tr["fused"][:5]), ref.predict_proba(X_tr["fused"][:5]))
    # the fold-0 per-question probabilities: one row per pilot question, fused_avg = mean of the two models' probabilities
    table = pd.read_csv(proba)
    assert len(table) == 30 * 7 and list(table.columns[:4]) == ["question_id", "rank", "n_games", "player_id"]
    cols = lambda s: [f"{s}_p_{lv}" for lv in LEVELS]  # noqa: E731
    assert all(c in table.columns for s in ("hsl", "fused_A", "fused_B", "fused_avg") for c in cols(s))
    np.testing.assert_allclose(table[cols("fused_avg")].to_numpy(), (table[cols("fused_A")].to_numpy() + table[cols("fused_B")].to_numpy()) / 2, atol=1e-12)
    np.testing.assert_allclose(table[cols("fused_avg")].sum(axis=1), 1.0, atol=1e-9)
    # a fold with no out-of-fold model, or with two, is refused
    side_c = _sidecar(tmp_path / "heads3_C.json", "c" * 64, "3" * 64, [1, 2, 3])
    assert ffm.main([*args[:1], f"A:{l1b_dir}:{side_a}", f"C:{l1b_dir}:{side_c}", *args[3:], "--overwrite"]) == 2
    assert "exactly one must supply" in capsys.readouterr().err  # fold 1: none (A and C both saw it); fold 4: two
    assert ffm.main(args[:-2] + ["--out", str(out)]) == 2 and "exists" in capsys.readouterr().err  # no --overwrite
