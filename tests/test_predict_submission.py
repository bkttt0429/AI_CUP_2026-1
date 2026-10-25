"""scripts/predict_submission.py with a stand-in for the GPU scorer: the single-process run, shards + merge,
their equality, the shard-file checks, the exit codes and the official-count expectations."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predict_submission as ps
from aicup.humansl import PROFILES
from aicup.metric import LEVELS
from aicup.sgf import parse_moves
from aicup.submission import read_test

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mock_fold0_40"
MODEL = ROOT / "results" / "models" / "rank_model.json"
PUBLIC, PRIVATE = FIXTURE / "rank_prediction_test_public.csv", FIXTURE / "rank_prediction_test_private.csv"


def _fake_score(tasks, workers, log=print):
    """Stand-in for score_tasks: n_moves = the target colour's moves; the per-move log p curve depends on
    the game only (a peak profile hashed from the SGF), so any sharding gives the same per-task values."""
    per_task, logps, mt, tt = {}, [], [], []
    for k, sgf, color in tasks:
        n = sum(1 for c, _, _ in parse_moves(sgf) if c == color)
        seed = int(hashlib.sha256(sgf.encode()).hexdigest()[:8], 16)
        curve = -2.0 - 0.05 * (np.arange(len(PROFILES)) - seed % len(PROFILES)) ** 2 + 0.01 * (seed % 7 - 3)
        logp = np.tile(curve, (n, 1)).astype(np.float32)
        per_task[k] = (n, logp.sum(axis=0))
        logps.append(logp)
        mt.append(np.full(n, k, dtype=np.int32))
        tt.append(np.arange(n, dtype=np.int32))
    moves = (np.concatenate(logps), np.concatenate(mt), np.concatenate(tt)) if logps else None
    return per_task, moves, int(sum(len(x) for x in logps)), 0.5


@pytest.fixture
def fake_scorer(monkeypatch):
    monkeypatch.setattr(ps, "score_tasks", _fake_score)


def _small_inputs(tmp_path, n_pub=3, n_priv=2, name="small"):
    """The first n_pub public and n_priv private questions of the fixture, as two CSVs."""
    d = tmp_path / name
    d.mkdir()
    paths = []
    for src, n, stem in ((PUBLIC, n_pub, "rank_prediction_test_public"), (PRIVATE, n_priv, "rank_prediction_test_private")):
        df = pd.read_csv(src, dtype=str, keep_default_na=False).head(n)
        path = d / f"{stem}.csv"
        df.to_csv(path, index=False)
        paths.append(str(path))
    return paths


def _args(inputs, out, n_rows, n_pub, n_priv, *extra):
    return ["--inputs", *inputs, "--model", str(MODEL), "--out", str(out), "--expect-rows", str(n_rows),
            "--expect-public", str(n_pub), "--expect-private", str(n_priv), *extra]


def test_fixture_reads_and_replays_like_the_recorded_run():
    """The 40-question mock test: 375 games declared, all usable, no passes -- as the recorded report says."""
    questions, warnings = ps.load_questions([PUBLIC, PRIVATE])
    assert warnings == [] and ps.load_questions([PUBLIC, PRIVATE], lenient=True) == (questions, [])
    assert len(questions) == 40 and [q["question_id"][:4] for q in questions] == ["pub_"] * 20 + ["priv"] * 20
    assert all(len(q["games"]) == q["declared_games"] for q in questions)
    tasks, owner, dropped, passes = ps.build_tasks(questions)
    recorded = json.loads((FIXTURE / "submission_rank_report.json").read_text())
    assert (len(tasks), dropped, passes) == (recorded["games_used"], recorded["games_dropped"], recorded["passes"]) == (375, [], 0)
    assert [k for k, _, _ in tasks] == list(range(375)) and len(owner) == 375 and owner == sorted(owner)
    assert ps.questions_sha([PUBLIC, PRIVATE], questions) == ps.questions_sha(["x/" + PUBLIC.name, "y/" + PRIVATE.name], questions)
    assert ps.questions_sha([PUBLIC, PRIVATE], questions) != ps.questions_sha([PRIVATE, PUBLIC], questions)


def test_single_process_run_writes_submission_report_and_shard_file(tmp_path, fake_scorer):
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2)) == ps.EXIT_OK
    questions, _ = ps.load_questions(inputs)
    ids = [q["question_id"] for q in questions]
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(sub.columns) == ["question_id", "pred_rank"] and sub["question_id"].tolist() == ids
    assert set(sub["pred_rank"]) <= set(LEVELS)
    proba = pd.read_csv(out.with_name("submission_rank_proba.csv"))
    assert list(proba.columns) == [f"p_{lv}" for lv in LEVELS] + ["question_id"] and proba["question_id"].tolist() == ids
    np.testing.assert_allclose(proba[[f"p_{lv}" for lv in LEVELS]].sum(axis=1), 1.0, rtol=0, atol=1e-9)
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    n_games = sum(len(q["games"]) for q in questions)
    assert (report["mode"], report["nshards"], len(report["shards"]), report["exit_code"], report["problems"]) == ("single", 1, 1, 0, [])
    assert (report["questions"], report["games_declared"], report["games_present"], report["games_used"]) == (5, n_games, n_games, n_games)
    assert report["fallback_questions"] == [] and report["model_name"] == "mlp_h32" and report["positions"] > 0
    assert (report["lenient_inputs"], report["input_warnings"], report["allow_input_warnings"]) == (False, [], False)
    assert report["expectations"] == {"rows": 5, "pub_": 3, "priv_": 2} and sum(report["predicted_counts"].values()) == 5
    shard = ps.shard_path(out, 0, 1)
    assert shard == out.with_name("submission_rank_shard0of1.npz") and shard.exists()
    with np.load(shard, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        assert z["task"].tolist() == list(range(n_games)) and z["sums"].shape == (n_games, 18) and z["sums"].dtype == np.float64
        assert z["n_moves"].min() >= 1 and len(z["move_logp"]) == z["n_moves"].sum() == len(z["move_task"]) == len(z["move_t"])
        assert z["move_logp"].shape[1] == 18 and set(z["move_task"].tolist()) == set(range(n_games))
    assert (meta["shard"], meta["nshards"], meta["n_tasks_total"], meta["n_tasks"]) == (0, 1, n_games, n_games)
    assert meta["questions_sha256"] == ps.questions_sha(inputs, questions) and meta["profiles"] == list(PROFILES)
    assert report["shards"][0] == meta


def test_shards_then_merge_equal_the_single_process_run(tmp_path, fake_scorer):
    inputs = [str(PUBLIC), str(PRIVATE)]
    single = tmp_path / "single" / "submission_rank.csv"
    assert ps.main(_args(inputs, single, 40, 20, 20)) == ps.EXIT_OK
    merged = tmp_path / "sharded" / "submission_rank.csv"
    for i in range(3):  # what scripts/nano4/shards.sh appends per GPU
        assert ps.main(_args(inputs, merged, 40, 20, 20, "--shard", str(i), "--nshards", "3")) == ps.EXIT_OK
        assert ps.shard_path(merged, i, 3).exists()
    assert not merged.exists()  # a shard predicts nothing
    assert ps.main(_args(inputs, merged, 40, 20, 20, "--merge", "--nshards", "3")) == ps.EXIT_OK
    for name in ("submission_rank.csv", "submission_rank_proba.csv"):
        assert (merged.with_name(name).read_text() == single.with_name(name).read_text()), name
    rs, rm = (json.loads(p.with_name("submission_rank_report.json").read_text()) for p in (single, merged))
    assert (rm["mode"], rm["nshards"], len(rm["shards"])) == ("merge", 3, 3) and (rs["mode"], rs["nshards"]) == ("single", 1)
    assert rm["positions"] == rs["positions"] and rm["predicted_counts"] == rs["predicted_counts"]
    assert [m["shard"] for m in rm["shards"]] == [0, 1, 2] and sum(m["n_tasks"] for m in rm["shards"]) == 375
    assert sorted(k for m in range(3) for k in np.load(ps.shard_path(merged, m, 3))["task"].tolist()) == list(range(375))
    assert np.load(ps.shard_path(merged, 1, 3))["task"].tolist() == list(range(1, 375, 3))
    for key in ("questions", "games_used", "games_dropped", "passes", "fallback_questions", "problems", "exit_code"):
        assert rm[key] == rs[key], key


def test_merge_refuses_missing_inconsistent_or_duplicated_shards(tmp_path, fake_scorer, capsys):
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2, "--shard", "0", "--nshards", "2")) == ps.EXIT_OK
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert "submission_rank_shard1of2.npz is missing" in capsys.readouterr().err
    assert not out.exists()
    # a file of another sharding in the slot
    assert ps.main(_args(inputs, out, 5, 3, 2, "--shard", "0", "--nshards", "3")) == ps.EXIT_OK
    shutil.copy(ps.shard_path(out, 0, 3), ps.shard_path(out, 1, 2))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert "another sharding or other inputs (shard 0 of 3" in capsys.readouterr().err
    # a shard made from other inputs (one question fewer): the hash differs
    other = _small_inputs(tmp_path, n_pub=2, n_priv=2, name="other")
    other_out = tmp_path / "other_run" / "submission_rank.csv"
    assert ps.main(_args(other, other_out, 4, 2, 2, "--shard", "1", "--nshards", "2")) == ps.EXIT_OK
    shutil.copy(ps.shard_path(other_out, 1, 2), ps.shard_path(out, 1, 2))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert "other inputs" in capsys.readouterr().err
    # a shard that scored the wrong slice: tasks missing from the union
    questions, _ = ps.load_questions(inputs)
    tasks, *_ = ps.build_tasks(questions)
    sha = ps.questions_sha(inputs, questions)
    per_task, moves, n_pos, wall = _fake_score(tasks[0::2], 1)
    ps.save_shard(ps.shard_path(out, 1, 2), 1, 2, tasks[0::2], per_task, moves, n_pos, wall, inputs, sha, len(tasks))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert "was scored by two shards" in capsys.readouterr().err
    per_task, moves, n_pos, wall = _fake_score(tasks[1::2][:-1], 1)
    ps.save_shard(ps.shard_path(out, 1, 2), 1, 2, tasks[1::2][:-1], per_task, moves, n_pos, wall, inputs, sha, len(tasks))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert f"cover {len(tasks) - 1} of {len(tasks)} tasks: 1 missing" in capsys.readouterr().err
    # the right file completes the run
    assert ps.main(_args(inputs, out, 5, 3, 2, "--shard", "1", "--nshards", "2")) == ps.EXIT_OK
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_OK and out.exists()


def test_fallback_question_exits_3_unless_allowed(tmp_path, fake_scorer, capsys):
    inputs = _small_inputs(tmp_path)
    df = pd.read_csv(inputs[1], dtype=str, keep_default_na=False)
    df.loc[0, "num_games"], df.loc[0, "sgf_1"], df.loc[0, "color_1"] = "1", "(;SZ[13];B[dd];W[jj])", "B"  # a 13x13 game: unusable
    for i in range(2, 21):
        df.loc[0, f"sgf_{i}"], df.loc[0, f"color_{i}"] = "", ""
    df.to_csv(inputs[1], index=False)
    qid = df.loc[0, "question_id"]
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2)) == ps.EXIT_FALLBACK
    err = capsys.readouterr().err
    assert f"1 questions had no usable game and were set to {ps.FALLBACK}: ['{qid}']" in err and "--allow-fallback" in err
    sub = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("question_id")
    assert sub.loc[qid, "pred_rank"] == ps.FALLBACK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["fallback_questions"] == [qid] and report["exit_code"] == 3 and report["problems"] == []
    assert report["games_dropped"] == [{"question_id": qid, "game": 1, "error": "board size 13"}]
    proba = pd.read_csv(out.with_name("submission_rank_proba.csv")).set_index("question_id")
    assert proba.loc[qid].isna().all() and proba.drop(index=qid).notna().all().all()
    assert ps.main(_args(inputs, out, 5, 3, 2, "--allow-fallback")) == ps.EXIT_OK
    assert "--allow-fallback" not in capsys.readouterr().err
    assert json.loads(out.with_name("submission_rank_report.json").read_text())["exit_code"] == 0


def test_expectations_malformed_inputs_and_argument_errors(tmp_path, fake_scorer, capsys):
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    # the official counts against a 5-question mock: the file is written, but validation fails
    assert ps.main(["--inputs", *inputs, "--model", str(MODEL), "--out", str(out)]) == ps.EXIT_INVALID
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert "5 questions read from the inputs, expected 800" in report["problems"] and report["exit_code"] == 1
    assert report["expectations"] == {"rows": 800, "pub_": 400, "priv_": 400}
    # 0 disables a check
    assert ps.main(_args(inputs, out, 0, 0, 0)) == ps.EXIT_OK
    # inputs given private first: the submission would put priv_ before pub_
    assert ps.main(_args([inputs[1], inputs[0]], out, 5, 3, 2)) == ps.EXIT_INVALID
    assert any("public must come first" in p for p in json.loads(out.with_name("submission_rank_report.json").read_text())["problems"])
    # a malformed input stops before any scoring
    df = pd.read_csv(inputs[0], dtype=str, keep_default_na=False)
    df.loc[1, "num_games"] = str(int(df.loc[1, "num_games"]) + 1)
    df.to_csv(inputs[0], index=False)
    qid = df.loc[1, "question_id"]
    assert ps.main(_args(inputs, out, 5, 3, 2)) == ps.EXIT_INPUTS
    assert qid in capsys.readouterr().err
    # ... unless --lenient-inputs: the question is read from its filled columns, predicted, and reported (exit 4)
    out = tmp_path / "lenient" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2, "--lenient-inputs")) == ps.EXIT_INPUT_WARNINGS
    err = capsys.readouterr().err
    assert "read leniently" in err and qid in err and "--allow-input-warnings" in err
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["exit_code"] == 4 and report["problems"] == [] and report["lenient_inputs"] is True
    assert len(report["input_warnings"]) == 1 and qid in report["input_warnings"][0] and "num_games" in report["input_warnings"][0]
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(sub) == 5 and set(sub["pred_rank"]) <= set(LEVELS) and report["games_declared"] == report["games_present"] + 1
    assert ps.main(_args(inputs, out, 5, 3, 2, "--lenient-inputs", "--allow-input-warnings")) == ps.EXIT_OK
    assert json.loads(out.with_name("submission_rank_report.json").read_text())["exit_code"] == 0
    # shards and the merge must agree on the flag: the questions hash counts the games read
    out = tmp_path / "lenient_shards" / "submission_rank.csv"
    for i in range(2):
        assert ps.main(_args(inputs, out, 5, 3, 2, "--lenient-inputs", "--shard", str(i), "--nshards", "2")) == ps.EXIT_OK
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS  # the strict read refuses
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2", "--lenient-inputs")) == ps.EXIT_INPUT_WARNINGS
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2", "--lenient-inputs", "--allow-input-warnings")) == ps.EXIT_OK
    assert out.read_text() == (tmp_path / "lenient" / "submission_rank.csv").read_text()
    # argument errors
    for bad in (["--nshards", "3"], ["--shard", "3", "--nshards", "3"], ["--shard", "0", "--merge"], ["--nshards", "0"]):
        with pytest.raises(SystemExit) as e:
            ps.main(_args(inputs, out, 5, 3, 2, *bad))
        assert e.value.code == 2


def test_model_with_other_features_is_refused(tmp_path, fake_scorer, capsys):
    inputs = _small_inputs(tmp_path)
    saved = json.loads(MODEL.read_text())
    saved["features"] = saved["features"][:-1]
    other = tmp_path / "other_model.json"
    other.write_text(json.dumps(saved))
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(["--inputs", *inputs, "--model", str(other), "--out", str(out), "--expect-rows", "5",
                    "--expect-public", "3", "--expect-private", "2"]) == ps.EXIT_INPUTS
    assert "different profiles or features" in capsys.readouterr().err and not out.exists()
    assert read_test(inputs[0])[0]["question_id"] == "pub_q_0001"
