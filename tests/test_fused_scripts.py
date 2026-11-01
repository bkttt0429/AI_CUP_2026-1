"""The fused-model scripts on synthetic scores: fit_fused_model.py (training as l1b_fold0_check, fold-0 reproduction,
per-question fold-0 probabilities)."""
import json

import numpy as np
import pandas as pd
import pytest

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


def _opponent_scores(tmp_path, name="opp_all"):
    """An l1b_score.py --opponent layout of the synthetic games: the same game keys, other sums and move counts."""
    opp_dir, _, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name=name)
    for csv in sorted(opp_dir.glob("fold*/games_s*_p*.csv")):
        t = pd.read_csv(csv, dtype=str, keep_default_na=False)
        cols = [f"sum_{lv}" for lv in LEVELS]
        t[cols] = t[cols].astype(float) * 0.9 - 0.3
        t["n_moves"] = t["n_moves"].astype(int) + 1
        t["color"] = t["color"].map({"B": "W", "W": "B"})
        t.to_csv(csv, index=False)
    return opp_dir


def test_fit_fused_model_cli_with_opponent(tmp_path, capsys):
    import l1b_opp_check as oc
    from aicup.scoring import read_scored_games

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_all")
    opp_dir = _opponent_scores(tmp_path)
    side_a = _sidecar(tmp_path / "heads3_A.json", "a" * 64, "1" * 64, [1, 2])
    side_b = _sidecar(tmp_path / "heads3_B.json", "b" * 64, "2" * 64, [3, 4])
    folds_csv = _folds_csv(tmp_path)
    out, proba = tmp_path / "models" / "rank_model_fused_opp.json", tmp_path / "fold0_fused_opp_proba.csv"
    l1b_args = ["--l1b", f"A:{l1b_dir}:{side_a}", f"B:{l1b_dir}:{side_b}"]
    rest = ["--hsl", str(hsl_dir), "--per-player", "2", "--folds-csv", str(folds_csv), "--out", str(out), "--fold0-proba", str(proba)]
    args = [*l1b_args, "--opp", f"A:{opp_dir}", f"B:{opp_dir}", *rest]
    assert ffm.main(args) == 0
    printed = capsys.readouterr().out
    assert "fused_opp_avg" in printed and "fused_opp_avg vs fused_avg" in printed and "opponent moves" in printed
    doc, mlp = fused.load_fused_model(out)
    assert doc["name"] == fused.NAME_OPP and doc["opponent"] is True and len(doc["features"]) == 56 and doc["per_player"] == 2
    assert doc["opp_scores_dirs"] == {"A": str(opp_dir), "B": str(opp_dir)} and [m["name"] for m in doc["l1b_models"]] == ["A", "B"]
    assert doc["l1b_models"][0]["feature_folds"] == [3, 4] and doc["feature_source_by_fold"] == {"1": "B", "2": "B", "3": "A", "4": "A"}
    assert len(doc["fused_state"]["params"]["0.weight"][0]) == 40 and len(doc["state"]["params"]["0.weight"][0]) == 56
    f0 = doc["fold0_check"]
    assert set(f0["scores"]) == {"hsl", "fused_A", "fused_B", "fused_avg", "fused_opp_A", "fused_opp_B", "fused_opp_avg"}
    assert "ci95" in f0["vs_hsl"]["fused_opp_avg"] and "ci95" in f0["fused_opp_avg_vs_fused_avg"]
    # the training reproduces l1b_opp_fold0_check.py's fit: the same questions, oc.features and MLP give the same model
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    data = {}
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(hsl_dir, l1b_dir, f)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": read_scored_games(opp_dir / f"fold{f}"), "players": l1b_cv.players_from_scores(hsl, folds_df)}
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([l1b_cv.sample_questions(data[f]["players"], 2, rng) for f in (1, 2, 3, 4)], ignore_index=True)
    cat = lambda key: pd.concat([data[f][key] for f in (1, 2, 3, 4)], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _ = oc.features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
    ref = ffm.MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr)
    np.testing.assert_array_equal(from_state(doc["state"]).predict_proba(X_tr["fused_opp"][:5]), ref.predict_proba(X_tr["fused_opp"][:5]))
    ref_f = ffm.MLPModel(hidden=32).fit(X_tr["fused"], y_tr)
    np.testing.assert_array_equal(from_state(doc["fused_state"]).predict_proba(X_tr["fused"][:5]), ref_f.predict_proba(X_tr["fused"][:5]))
    # the fold-0 table carries all seven sets; the averages are the means of the two models' probabilities
    table = pd.read_csv(proba)
    cols = lambda s: [f"{s}_p_{lv}" for lv in LEVELS]  # noqa: E731
    assert all(c in table.columns for s in ("hsl", "fused_avg", "fused_opp_A", "fused_opp_B", "fused_opp_avg") for c in cols(s))
    np.testing.assert_allclose(table[cols("fused_opp_avg")].to_numpy(), (table[cols("fused_opp_A")].to_numpy() + table[cols("fused_opp_B")].to_numpy()) / 2, atol=1e-12)
    # --reference in opp_fold0_check.json's layout (scores at the top level): the numbers agree with themselves
    ref_json = tmp_path / "opp_fold0_check.json"
    ref_json.write_text(json.dumps({"scores": {k: {"balanced_mean_over_n": v["balanced_mean_over_n"]} for k, v in f0["scores"].items()}}))
    assert ffm.main([*args, "--reference", str(ref_json), "--overwrite"]) == 0
    printed = capsys.readouterr().out
    assert "fused_opp_avg: +0.00e+00" in printed and "WARNING" not in printed
    # --opp must name every --l1b model exactly once, and only models that exist
    assert ffm.main([*l1b_args, "--opp", f"A:{opp_dir}", *rest, "--overwrite"]) == 2 and "every --l1b model" in capsys.readouterr().err
    assert ffm.main([*l1b_args, "--opp", f"A:{opp_dir}", f"C:{opp_dir}", *rest, "--overwrite"]) == 2 and "no --l1b model" in capsys.readouterr().err
    # an opponent table covering other games than the Human SL scores is refused
    bad = tmp_path / "opp_bad"
    import shutil
    shutil.copytree(opp_dir, bad)
    csv = sorted(bad.glob("fold1/games_s*_p*.csv"))[0]
    t = pd.read_csv(csv, dtype=str, keep_default_na=False)
    t.iloc[1:].to_csv(csv, index=False)
    assert ffm.main([*l1b_args, "--opp", f"A:{opp_dir}", f"B:{bad}", *rest, "--overwrite"]) == 2 and "other games" in capsys.readouterr().err
    # without --opp the plain fused document is still written (the default output name differs)
    plain = tmp_path / "models" / "plain.json"
    assert ffm.main([*l1b_args, *rest[:-4], "--out", str(plain)]) == 0
    assert fused.load_fused_model(plain)[0]["opponent"] is False


def test_fit_fused_model_cli_with_diff(tmp_path, capsys):
    import error_analysis as ea
    from aicup.scoring import read_scored_games

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_all")
    opp_dir = _opponent_scores(tmp_path)
    side_a = _sidecar(tmp_path / "heads3_A.json", "a" * 64, "1" * 64, [1, 2])
    side_b = _sidecar(tmp_path / "heads3_B.json", "b" * 64, "2" * 64, [3, 4])
    folds_csv = _folds_csv(tmp_path)
    out, proba = tmp_path / "models" / "rank_model_fused_opp_diff.json", tmp_path / "fold0_fused_opp_diff_proba.csv"
    l1b_args = ["--l1b", f"A:{l1b_dir}:{side_a}", f"B:{l1b_dir}:{side_b}"]
    rest = ["--hsl", str(hsl_dir), "--per-player", "2", "--folds-csv", str(folds_csv), "--out", str(out), "--fold0-proba", str(proba)]
    args = [*l1b_args, "--opp", f"A:{opp_dir}", f"B:{opp_dir}", "--diff", *rest]
    assert ffm.main(args) == 0
    printed = capsys.readouterr().out
    assert "fused_opp_diff_avg vs fused_opp_avg" in printed and "fused_opp_avg vs fused_avg" in printed
    doc, mlp = fused.load_fused_model(out)
    assert doc["name"] == fused.NAME_OPP_DIFF and doc["feature_set"] == "fused_opp_diff" and doc["opponent"] is True and len(doc["features"]) == 59
    assert len(doc["state"]["params"]["0.weight"][0]) == 59 and len(doc["fused_opp_state"]["params"]["0.weight"][0]) == 56 and len(doc["fused_state"]["params"]["0.weight"][0]) == 40
    f0 = doc["fold0_check"]
    assert set(f0["scores"]) == {"hsl", "fused_A", "fused_B", "fused_avg", "fused_opp_A", "fused_opp_B", "fused_opp_avg", "fused_opp_diff_A", "fused_opp_diff_B", "fused_opp_diff_avg"}
    assert "ci95" in f0["fused_opp_diff_avg_vs_fused_opp_avg"] and "ci95" in f0["vs_hsl"]["fused_opp_diff_avg"]
    # the training reproduces error_analysis.py's mlp_diff fit on the same questions
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    data = {}
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(hsl_dir, l1b_dir, f)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": read_scored_games(opp_dir / f"fold{f}"), "players": l1b_cv.players_from_scores(hsl, folds_df)}
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([l1b_cv.sample_questions(data[f]["players"], 2, rng) for f in (1, 2, 3, 4)], ignore_index=True)
    cat = lambda key: pd.concat([data[f][key] for f in (1, 2, 3, 4)], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _ = ea.feature_sets(cat("hsl"), cat("l1b"), cat("opp"), train_q)
    ref = ffm.MLPModel(hidden=32).fit(X_tr["fused_opp_diff"], y_tr)
    np.testing.assert_array_equal(from_state(doc["state"]).predict_proba(X_tr["fused_opp_diff"][:5]), ref.predict_proba(X_tr["fused_opp_diff"][:5]))
    np.testing.assert_array_equal(from_state(doc["fused_opp_state"]).predict_proba(X_tr["fused_opp"][:5]), ffm.MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr).predict_proba(X_tr["fused_opp"][:5]))
    table = pd.read_csv(proba)
    cols = lambda s: [f"{s}_p_{lv}" for lv in LEVELS]  # noqa: E731
    assert all(c in table.columns for s in ("fused_opp_avg", "fused_opp_diff_A", "fused_opp_diff_B", "fused_opp_diff_avg") for c in cols(s))
    np.testing.assert_allclose(table[cols("fused_opp_diff_avg")].to_numpy(), (table[cols("fused_opp_diff_A")].to_numpy() + table[cols("fused_opp_diff_B")].to_numpy()) / 2, atol=1e-12)
    # --diff without --opp is refused; the default output name follows the kind
    with pytest.raises(SystemExit) as e:
        ffm.main([*l1b_args, "--diff", *rest])
    assert e.value.code == 2
    assert ffm.DEFAULT_OUT["fused_opp_diff"].name == "rank_model_fused_opp_diff.json"


def _proba_frame(ids, P, prefix="p"):
    return pd.DataFrame({**{f"{prefix}_{lv}": P[:, j] for j, lv in enumerate(LEVELS)}, "question_id": ids})


def test_check_fused_submission(tmp_path, capsys):
    import check_fused_submission as cfs
    from aicup.metric import expected_score_decision

    rng = np.random.RandomState(0)
    n = 6
    P = rng.dirichlet(np.ones(10), size=n)
    mock_ids = [f"pub_q_{i:04d}" for i in range(1, n + 1)]
    src_ids = [f"1D_{i:08x}_n5_{i % 4}" for i in range(n)]
    run = tmp_path / "run" / "submission_rank.csv"
    run.parent.mkdir()
    pd.DataFrame({"question_id": mock_ids, "pred_rank": [LEVELS[expected_score_decision(p)] for p in P]}).to_csv(run, index=False)
    _proba_frame(mock_ids, P).to_csv(run.with_name("submission_rank_proba.csv"), index=False)
    answers = tmp_path / "answers.csv"
    pd.DataFrame({"question_id": mock_ids, "rank": ["1D"] * n, "source_question": src_ids, "n_games": [5] * n}).to_csv(answers, index=False)
    offline = tmp_path / "fold0_fused_proba.csv"
    off = pd.DataFrame({"question_id": src_ids[::-1], "rank": ["1D"] * n, "n_games": [5] * n, "player_id": ["x"] * n})
    for j, lv in enumerate(LEVELS):
        off[f"fused_avg_p_{lv}"] = P[::-1, j] + 2e-4 * (j % 2)  # the offline side, in another order, within tolerance
        off[f"hsl_p_{lv}"] = 1.0 / 10
    off.to_csv(offline, index=False)
    report = tmp_path / "check.json"
    args = ["--submission", str(run), "--offline", str(offline), "--answers", str(answers), "--report", str(report)]
    assert cfs.main(args) == 0
    res = json.loads(report.read_text())
    assert res["pass"] and res["n_questions"] == n and abs(res["max_abs_diff"] - 2e-4) < 1e-12 and res["decisions_differ"] == []
    assert "PASS" in capsys.readouterr().out
    assert cfs.main([*args, "--tol", "1e-5"]) == 1 and "FAIL" in capsys.readouterr().out
    # a decision that differs fails whatever the tolerance; a question missing offline is an error
    sub = pd.read_csv(run, dtype=str)
    sub.loc[0, "pred_rank"] = "6D" if sub.loc[0, "pred_rank"] != "6D" else "D"
    sub.to_csv(run, index=False)
    assert cfs.main([*args, "--tol", "1"]) == 1 and "1 decisions differ" in capsys.readouterr().out
    off.iloc[1:].to_csv(offline, index=False)
    assert cfs.main(args) == 2 and "offline" in capsys.readouterr().err
    # two pipeline runs against each other (single GPU vs sharded): same ids, probabilities within tolerance, same decisions
    other = tmp_path / "other" / "submission_rank.csv"
    other.parent.mkdir()
    pd.DataFrame({"question_id": mock_ids, "pred_rank": [LEVELS[expected_score_decision(p)] for p in P]}).to_csv(other, index=False)
    _proba_frame(mock_ids, P + 1e-5).to_csv(other.with_name("submission_rank_proba.csv"), index=False)
    pd.DataFrame({"question_id": mock_ids, "pred_rank": [LEVELS[expected_score_decision(p)] for p in P]}).to_csv(run, index=False)
    assert cfs.main(["--submission", str(run), "--other", str(other), "--report", str(report)]) == 0
    assert abs(json.loads(report.read_text())["max_abs_diff"] - 1e-5) < 1e-12


def test_pack_weights(tmp_path, capsys):
    import tarfile

    import pack_weights as pw
    from tests.test_predict_submission import _tiny_l1b_heads

    paths, sides = _tiny_l1b_heads(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "rank_model.json").write_text("{}")
    fused_doc = {"name": "mlp_h32_fused", "l1b_models": [{"name": "A", "heads_file": paths[0].name, "state_sha256": sides[0]["state_sha256"],
                                                            "trunk_file": sides[0]["trunk_file"], "trunk_sha256": sides[0]["trunk_sha256"]},
                                                           {"name": "B", "heads_file": paths[1].name, "state_sha256": sides[1]["state_sha256"],
                                                            "trunk_file": None, "trunk_sha256": None}]}
    (models / "rank_model_fused.json").write_text(json.dumps(fused_doc))
    out = tmp_path / "release" / "weights.tar"
    args = ["--l1b-heads", *map(str, paths), "--models", str(models / "rank_model.json"), str(models / "rank_model_fused.json"), "--out", str(out)]
    assert pw.main(args) == 0
    with tarfile.open(out) as tar:
        names = tar.getnames()
    assert "SHA256SUMS" in names and paths[0].name in names and sides[0]["trunk_file"] in names and paths[1].name in names
    assert paths[0].with_suffix(".json").name in names and "rank_model_fused.json" in names and "rank_model.json" in names
    sums = out.with_name("SHA256SUMS").read_text()
    assert sides[0]["state_sha256"] in sums and sides[0]["trunk_sha256"] in sums and "trunk.pt" not in [n for n in names if n.endswith("B.trunk.pt")]
    assert f"{out.stat().st_size}" in capsys.readouterr().out or out.stat().st_size > 0
    # a heads file that no longer matches its sidecar is refused
    paths[1].write_bytes(paths[1].read_bytes() + b"\0")
    assert pw.main([*args, "--overwrite"]) == 2 and "sha256" in capsys.readouterr().err


def test_fit_fused_model_cli_with_five_cross_fit_models(tmp_path, capsys):
    """Issue #16: f0..f4 each trained on the other four folds share one scores directory (fold k scored by f_k); fold 0's
    check uses f0 alone and the document says which models were left aside."""
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="scores5")
    opp_dir = _opponent_scores(tmp_path, name="scores5_opp")
    sides = [_sidecar(tmp_path / f"heads5_f{k}.json", chr(ord("a") + k) * 64, str(k) * 64, [f for f in range(5) if f != k]) for k in range(5)]
    folds_csv = _folds_csv(tmp_path)
    out, proba = tmp_path / "models" / "rank_model_fused_opp_x5.json", tmp_path / "fold0_x5_proba.csv"
    args = ["--l1b", *[f"f{k}:{l1b_dir}:{sides[k]}" for k in range(5)], "--opp", *[f"f{k}:{opp_dir}" for k in range(5)], "--hsl", str(hsl_dir),
            "--per-player", "2", "--folds-csv", str(folds_csv), "--out", str(out), "--fold0-proba", str(proba)]
    assert ffm.main(args) == 0
    printed = capsys.readouterr().out
    assert "left out of the check" in printed and "['f0']" in printed
    doc, _ = fused.load_fused_model(out)
    assert doc["feature_source_by_fold"] == {"1": "f1", "2": "f2", "3": "f3", "4": "f4"}
    assert [m["name"] for m in doc["l1b_models"]] == [f"f{k}" for k in range(5)]
    assert [m["feature_folds"] for m in doc["l1b_models"]] == [[], [1], [2], [3], [4]] and doc["l1b_models"][2]["train_folds"] == [0, 1, 3, 4]
    check = doc["fold0_check"]
    assert check["fold0_models"] == ["f0"] and check["models_seen_fold0"] == ["f1", "f2", "f3", "f4"]
    assert set(check["scores"]) == {"hsl", "fused_f0", "fused_avg", "fused_opp_f0", "fused_opp_avg"}
    assert check["scores"]["fused_opp_avg"] == check["scores"]["fused_opp_f0"]  # the average over the OOF models is f0 alone
    table = pd.read_csv(proba)
    assert np.allclose(table[[f"fused_opp_avg_p_{lv}" for lv in LEVELS]].to_numpy(), table[[f"fused_opp_f0_p_{lv}" for lv in LEVELS]].to_numpy())
    # every model trained on fold 0 (the 60% A'/B' design: folds 0,1,2 and 0,3,4): the fit works, the fold-0 check cannot be made
    all_seen = [_sidecar(tmp_path / "seenA.json", "c" * 64, "9" * 64, [0, 1, 2]), _sidecar(tmp_path / "seenB.json", "d" * 64, "8" * 64, [0, 3, 4])]
    args2 = ["--l1b", f"a:{l1b_dir}:{all_seen[0]}", f"b:{l1b_dir}:{all_seen[1]}", "--hsl", str(hsl_dir), "--per-player", "2",
             "--folds-csv", str(folds_csv), "--out", str(tmp_path / "x.json")]
    assert ffm.main(args2) == 2
    assert "out-of-fold for fold 0" in capsys.readouterr().err


def test_fit_fused_model_cli_with_hd_table(tmp_path, capsys):
    """--hd-table (issue #47): the fused_opp_diff_hd MLP on top of the diff fit, the fold-0 table and comparison, the HD provenance."""
    from tests.test_hd_check import _hd_table

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_all")
    opp_dir = _opponent_scores(tmp_path)
    side_a = _sidecar(tmp_path / "heads3_A.json", "a" * 64, "1" * 64, [1, 2])
    side_b = _sidecar(tmp_path / "heads3_B.json", "b" * 64, "2" * 64, [3, 4])
    folds_csv = _folds_csv(tmp_path)
    hd_path = tmp_path / "hd_table.csv"
    _hd_table(hsl_dir, (0, 1, 2, 3, 4), hd_path)
    out, proba = tmp_path / "models" / "rank_model_fused_opp_diff_hd.json", tmp_path / "fold0_hd_proba.csv"
    l1b_args = ["--l1b", f"A:{l1b_dir}:{side_a}", f"B:{l1b_dir}:{side_b}", "--opp", f"A:{opp_dir}", f"B:{opp_dir}"]
    rest = ["--hsl", str(hsl_dir), "--per-player", "2", "--folds-csv", str(folds_csv), "--out", str(out), "--fold0-proba", str(proba)]
    assert ffm.main([*l1b_args, "--diff", "--hd-table", str(hd_path), *rest]) == 0
    printed = capsys.readouterr().out
    assert "HD table" in printed and "fused_opp_diff_hd_avg vs fused_opp_diff_avg" in printed and "fused_opp_diff_avg vs fused_opp_avg" in printed
    doc, mlp = fused.load_fused_model(out)
    assert doc["name"] == fused.NAME_OPP_DIFF_HD and doc["feature_set"] == "fused_opp_diff_hd" and len(doc["features"]) == 64 and doc["hd_table"] == str(hd_path)
    assert doc["hd_coverage"]["games"] == 5 * 600 and 0 < doc["hd_coverage"]["share_hd2"] < doc["hd_coverage"]["share_hd1"] < 1
    assert len(doc["state"]["params"]["0.weight"][0]) == 64 and len(doc["fused_opp_diff_state"]["params"]["0.weight"][0]) == 59
    assert len(doc["fused_opp_state"]["params"]["0.weight"][0]) == 56 and len(doc["fused_state"]["params"]["0.weight"][0]) == 40
    f0 = doc["fold0_check"]
    assert {"fused_opp_diff_hd_A", "fused_opp_diff_hd_B", "fused_opp_diff_hd_avg", "fused_opp_diff_avg"} <= set(f0["scores"])
    assert "ci95" in f0["fused_opp_diff_hd_avg_vs_fused_opp_diff_avg"] and "ci95" in f0["vs_hsl"]["fused_opp_diff_hd_avg"]
    table = pd.read_csv(proba)
    cols = lambda s: [f"{s}_p_{lv}" for lv in LEVELS]  # noqa: E731
    np.testing.assert_allclose(table[cols("fused_opp_diff_hd_avg")].to_numpy(), (table[cols("fused_opp_diff_hd_A")].to_numpy() + table[cols("fused_opp_diff_hd_B")].to_numpy()) / 2, atol=1e-12)
    assert not np.allclose(table[cols("fused_opp_diff_hd_avg")].to_numpy(), table[cols("fused_opp_diff_avg")].to_numpy())
    # the hd MLP's inputs are the diff features plus the HD block of the same training questions (the diff fit is the kept reference)
    from aicup.handicap import question_hd_features, read_hd_table
    hd = read_hd_table(hd_path)
    hsl0 = l1b_cv.read_scored_games(hsl_dir / "fold0")
    folds_df = pd.read_csv(folds_csv, dtype={"player_id": str})
    val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))
    _, l1b0 = l1b_cv.load_fold(hsl_dir, l1b_dir, 0)
    X, _, _ = ffm.feature_sets(hsl0, l1b0, val, ffm.opponent_table(opp_dir, 0, hsl0), True, hd)
    np.testing.assert_allclose(X["fused_opp_diff_hd"], np.hstack([X["fused_opp_diff"], question_hd_features(hd, val)]), atol=0)
    np.testing.assert_allclose(table[cols("fused_opp_diff_hd_A")].to_numpy(), mlp.predict_proba(X["fused_opp_diff_hd"]), atol=1e-12)
    # refusals: --hd-table without --diff; a table that lacks the games
    with pytest.raises(SystemExit):
        ffm.main([*l1b_args, "--hd-table", str(hd_path), *rest, "--overwrite"])
    short = tmp_path / "short.csv"
    pd.read_csv(hd_path, dtype=str, keep_default_na=False).iloc[:100].to_csv(short, index=False)
    assert ffm.main([*l1b_args, "--diff", "--hd-table", str(short), *rest, "--overwrite"]) == 2
