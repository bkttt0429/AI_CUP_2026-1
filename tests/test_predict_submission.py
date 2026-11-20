"""scripts/predict_submission.py with a stand-in for the GPU scorer: the single-process run, shards + merge,
their equality, the shard-file checks, the exit codes and the official-count expectations."""
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import predict_submission as ps
from aicup import ladder, setmodel_io
from aicup.aggregators import CTM, SetTransformer
from aicup.humansl import PROFILES
from aicup.ladder_data import pack_questions
from aicup.metric import LEVELS, expected_score_decision
from aicup.qfeatures import question_features
from aicup.rankmodel import _softmax
from aicup.settrain import BASE_CLIP, TrainCfg, fit_set_model, predict_set_model
from aicup.sgf import parse_moves
from aicup.submission import read_test

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mock_fold0_40"
MODEL = ROOT / "results" / "models" / "rank_model.json"
PUBLIC, PRIVATE = FIXTURE / "rank_prediction_test_public.csv", FIXTURE / "rank_prediction_test_private.csv"


def _fake_score(tasks, workers, log=print, l1b_heads=(), opponent=True, opponent_hsl=True):
    """Stand-in for score_tasks: n_moves = the target colour's moves; the per-move log p curve depends on
    the game only (a peak profile hashed from the SGF), so any sharding gives the same per-task values.
    With l1b_heads, 10 more columns per heads file follow the 18 profiles (a peak at a rank that depends on the
    game and on the model's index, so the models differ); with opponent too, the opponent pass gives each task the
    other colour's move count and its own L1b curves (another hash slice), as score_tasks' fifth value; with opponent_hsl
    (section 4M M14) 18 profile columns precede them, as the real pass's do."""
    per_task, logps, mt, tt = {}, [], [], []
    for k, sgf, color in tasks:
        n = sum(1 for c, _, _ in parse_moves(sgf) if c == color)
        seed = int(hashlib.sha256(sgf.encode()).hexdigest()[:8], 16)
        curve = -2.0 - 0.05 * (np.arange(len(PROFILES)) - seed % len(PROFILES)) ** 2 + 0.01 * (seed % 7 - 3)
        for j in range(len(l1b_heads)):
            l1b = -1.5 - 0.1 * (np.arange(len(LEVELS)) - (seed + 3 * j) % len(LEVELS)) ** 2 + 0.02 * ((seed >> 4) % 5 - 2)
            curve = np.concatenate([curve, l1b])
        logp = np.tile(curve, (n, 1)).astype(np.float32)
        per_task[k] = (n, logp.sum(axis=0))
        logps.append(logp)
        mt.append(np.full(n, k, dtype=np.int32))
        tt.append(np.arange(n, dtype=np.int32))
    moves = (np.concatenate(logps), np.concatenate(mt), np.concatenate(tt)) if logps else None
    opp = None
    if l1b_heads and opponent:
        o_per_task, o_logps, o_mt, o_tt = {}, [], [], []
        for k, sgf, color in tasks:
            other = "W" if color == "B" else "B"
            n_o = sum(1 for c, _, _ in parse_moves(sgf) if c == other)
            seed = int(hashlib.sha256(sgf.encode()).hexdigest()[8:16], 16)
            curve = np.concatenate(([-1.8 - 0.05 * (np.arange(len(PROFILES)) - seed % len(PROFILES)) ** 2] if opponent_hsl else [])
                                   + [-1.2 - 0.1 * (np.arange(len(LEVELS)) - (seed + 5 * j) % len(LEVELS)) ** 2 for j in range(len(l1b_heads))])
            o_per_task[k] = (n_o, curve * n_o)
            o_logps.append(np.tile(curve, (n_o, 1)).astype(np.float32))
            o_mt.append(np.full(n_o, k, dtype=np.int32))
            o_tt.append(1 + 2 * np.arange(n_o, dtype=np.int32))  # the other colour's plies
        o_moves = (np.concatenate(o_logps), np.concatenate(o_mt), np.concatenate(o_tt)) if o_logps else None
        opp = (o_per_task, int(sum(v[0] for v in o_per_task.values())), 0.25, o_moves)
    return per_task, moves, int(sum(len(x) for x in logps)), 0.5, opp


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
    for key in ("questions", "games_used", "games_dropped", "passes", "handicap", "fallback_questions", "problems", "exit_code"):
        assert rm[key] == rs[key], key
    assert rs["handicap"] == {"absent": 375}  # the mock test is v1.0-style: no HD[]


def test_handicap_counts_orders_absent_first_then_by_handicap():
    qs = [{"games": [("(;SZ[19];B[pd])", "B"), ("(;SZ[19]HD[2];B[qf])", "W"), ("(;SZ[19]HD[0];B[pd])", "B"), ("(;SZ[19]HD[2];B[qf])", "B")]},
          {"games": [("(;SZ[19]HD[1];B[pd])", "W")]}]
    assert ps.handicap_counts(qs) == {"absent": 1, "0": 1, "1": 1, "2": 2}


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
    per_task, moves, n_pos, wall, _ = _fake_score(tasks[0::2], 1)
    ps.save_shard(ps.shard_path(out, 1, 2), 1, 2, tasks[0::2], per_task, moves, n_pos, wall, inputs, sha, len(tasks))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--nshards", "2")) == ps.EXIT_INPUTS
    assert "was scored by two shards" in capsys.readouterr().err
    per_task, moves, n_pos, wall, _ = _fake_score(tasks[1::2][:-1], 1)
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


# ---------------------------------------------------------------- set models (item 2)

def _tiny_set_models(tmp_path, monkeypatch, rungs=(("r3_pma", 2), ("r5_ctm", 1))):
    """Tiny set models trained on random 301-d tokens and saved as .pt + sidecar, n seeds per rung; the rungs are
    monkeypatched to small modules (load_set_model rebuilds them through SET_RUNGS)."""
    monkeypatch.setitem(ladder.SET_RUNGS, "r3_pma", ("staged", lambda d: SetTransformer(d, d_tok=16, heads=4, layers=0, hidden=32), {}))
    monkeypatch.setitem(ladder.SET_RUNGS, "r5_ctm", ("staged", partial(CTM, d_tok=16, d_latent=32, history=4, nlm_hidden=8, heads=4,
                                                                   ticks=3, n_sync_action=24, n_sync_out=24), {}))
    rng = np.random.RandomState(0)
    Q, M, d = 48, 20, 301
    n = rng.randint(5, M + 1, size=Q)
    mask = np.arange(M)[None, :] < n[:, None]
    data = {"tokens": (rng.randn(Q, M, d) * mask[..., None]).astype(np.float32), "mask": mask,
            "qglob": rng.randn(Q, 6).astype(np.float32), "y": np.arange(Q) % 10, "player_id": np.array([f"p{i // 4}" for i in range(Q)])}
    base = rng.randn(Q, 10)
    paths = []
    for rung, n_seeds in rungs:
        for seed in range(n_seeds):
            fitted = fit_set_model(ladder.SET_RUNGS[rung][1], data, base,
                                   TrainCfg(epochs=1, batch=16, lr=1e-2, device="cpu", seed=seed, inner_frac=0))
            pt = tmp_path / "set" / f"{rung}_s{seed}.pt"
            setmodel_io.save_set_model(fitted, rung, seed, pt, extra={"train_folds": [0, 1, 2, 3, 4], "per_player": 32})
            paths.append(pt)
    return paths


def _manual_set_probs(inputs, out, paths):
    """What the pipeline should compute: shards -> per-game vectors -> packed questions -> base logits -> every
    model -> mean over a rung's seeds, then over the rungs."""
    questions, _ = ps.load_questions(inputs)
    tasks, owner, _, _ = ps.build_tasks(questions)
    per_task, _, moves = ps.load_shards(out, 1, len(tasks), ps.questions_sha(inputs, questions), with_moves=True)
    assert set(moves) == set(range(len(tasks))) and all(len(moves[k][0]) == per_task[k][0] for k in moves)
    vec = ps.game_vectors(per_task, moves)
    assert vec["staged"].shape == (len(tasks), 300) and vec["whole"].shape == (len(tasks), 75)
    assert vec["game_key"].tolist() == [str(k) for k in range(len(tasks))] and vec["n_moves"].tolist() == [per_task[k][0] for k in range(len(tasks))]
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    qis = sorted(by_q)
    frame = pd.DataFrame({"question_id": [questions[qi]["question_id"] for qi in qis], "player_id": [questions[qi]["question_id"] for qi in qis],
                          "n_games": [len(by_q[qi]) for qi in qis], "game_keys": [[str(k) for k in by_q[qi]] for qi in qis]})
    packed = pack_questions(frame, vec, "staged", True)
    _, base_model = ps.load_rank_model(MODEL)
    X = np.stack([question_features(np.stack([per_task[k][1] for k in by_q[qi]]), np.array([per_task[k][0] for k in by_q[qi]])) for qi in qis])
    base = np.log(np.clip(base_model.predict_proba(X), BASE_CLIP, None))
    probs = {}
    for pt in paths:
        fitted, meta, _ = setmodel_io.load_set_model(pt)
        probs.setdefault(meta["rung"], []).append(predict_set_model(fitted, packed, base))
    want = np.mean([np.mean(v, axis=0) for v in probs.values()], axis=0) if probs else None
    return want, probs, base


def test_set_models_decide_with_seed_then_rung_averaging(tmp_path, fake_scorer, monkeypatch, capsys):
    paths = _tiny_set_models(tmp_path, monkeypatch)
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    with pytest.raises(SystemExit) as e:  # the paired base is required
        ps.main(_args(inputs, out, 5, 3, 2, "--set-models", *map(str, paths)))
    assert e.value.code == 2
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", *map(str, paths), "--set-base", str(MODEL))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["decision_model"] == "set_models" and report["set_rungs"] == {"r3_pma": 2, "r5_ctm": 1} and report["set_base"] == str(MODEL)
    assert [(m["rung"], m["seed"]) for m in report["set_models"]] == [("r3_pma", 0), ("r3_pma", 1), ("r5_ctm", 0)]
    assert report["set_models"][0]["path"] == str(paths[0]) and report["set_models"][0]["train_folds"] == [0, 1, 2, 3, 4]
    assert report["set_model_fallback_questions"] == [] and report["set_model_warnings"] == [] and report["exit_code"] == 0
    want, probs, base = _manual_set_probs(inputs, out, paths)
    got = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    naive = np.mean([p for v in probs.values() for p in v], axis=0)  # one mean over the 3 models would weight r3_pma twice
    assert not np.allclose(want, naive) and not np.allclose(want, _softmax(base))
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert sub["pred_rank"].tolist() == [LEVELS[expected_score_decision(p)] for p in want]
    # without set models the saved L3/L4 model decides, with other probabilities
    plain = tmp_path / "plain" / "submission_rank.csv"
    assert ps.main(_args(inputs, plain, 5, 3, 2)) == ps.EXIT_OK
    assert json.loads(plain.with_name("submission_rank_report.json").read_text())["decision_model"] == "rank_model"
    assert not np.allclose(pd.read_csv(plain.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy(), got)
    # sharded scoring + merge with set models gives the same file
    merged = tmp_path / "sharded" / "submission_rank.csv"
    for i in range(2):
        assert ps.main(_args(inputs, merged, 5, 3, 2, "--shard", str(i), "--nshards", "2")) == ps.EXIT_OK
    assert ps.main(_args(inputs, merged, 5, 3, 2, "--merge", "--nshards", "2", "--set-models", *map(str, paths), "--set-base", str(MODEL))) == ps.EXIT_OK
    assert merged.read_text() == out.read_text()
    assert merged.with_name("submission_rank_proba.csv").read_text() == out.with_name("submission_rank_proba.csv").read_text()


def test_set_models_must_pair_with_their_base(tmp_path, fake_scorer, monkeypatch, capsys):
    paths = _tiny_set_models(tmp_path, monkeypatch, rungs=(("r3_pma", 1),))
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    side = setmodel_io.sidecar_path(paths[0])
    meta, sha = json.loads(side.read_text()), "a" * 64
    side.write_text(json.dumps({**meta, "questions_sha256": sha, "base_model": "rank_model_f01234.json", "base_model_sha256": "b" * 64}))
    saved = json.loads(Path(MODEL).read_text())
    # rank_model.json (no training-question hash) is not the paired base: refused before anything is written
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(MODEL))) == ps.EXIT_INPUTS
    assert "paired base" in capsys.readouterr().err and not out.exists()
    # a base fitted on other questions: refused too
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**saved, "questions_sha256": "c" * 64}))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(other))) == ps.EXIT_INPUTS
    assert "other questions" in capsys.readouterr().err and not out.exists()
    # the paired base (the same questions) in a rewritten file: a warning, the run goes on
    paired = tmp_path / "paired.json"
    paired.write_text(json.dumps({**saved, "questions_sha256": sha}))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(paired))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert len(report["set_model_warnings"]) == 1 and "sha256" in report["set_model_warnings"][0] and report["exit_code"] == 0
    assert "WARNING" in capsys.readouterr().err
    side.write_text(json.dumps({**meta, "questions_sha256": sha, "base_model_sha256": setmodel_io.file_sha256(paired)}))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(paired))) == ps.EXIT_OK
    assert json.loads(out.with_name("submission_rank_report.json").read_text())["set_model_warnings"] == []


def test_set_model_fallback_and_warnings(tmp_path, fake_scorer, monkeypatch, capsys):
    paths = _tiny_set_models(tmp_path, monkeypatch, rungs=(("r3_pma", 1),))
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    # a model trained by other code: a warning in the report and on stderr, the run goes on
    side = setmodel_io.sidecar_path(paths[0])
    meta = json.loads(side.read_text())
    side.write_text(json.dumps({**meta, "code_sha256": "0" * 64}))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(MODEL))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert len(report["set_model_warnings"]) == 1 and "other training code" in report["set_model_warnings"][0]
    assert "WARNING" in capsys.readouterr().err and report["set_model_fallback_questions"] == []
    side.write_text(json.dumps(meta))
    # a model whose output is not finite: every question falls back to the base's probabilities
    fitted, _, _ = setmodel_io.load_set_model(paths[0])
    with torch.no_grad():
        next(fitted.model.parameters()).fill_(float("nan"))
    setmodel_io.save_set_model(fitted, "r3_pma", 0, paths[0], extra={"train_folds": [0, 1, 2, 3, 4], "per_player": 32})
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(MODEL))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    ids = [q["question_id"] for q in ps.load_questions(inputs)[0]]
    assert report["set_model_fallback_questions"] == ids and report["exit_code"] == 0
    assert "base's probabilities" in capsys.readouterr().err
    _, _, base = _manual_set_probs(inputs, out, [])
    got = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got, _softmax(base), rtol=0, atol=1e-12)
    # a broken model file is refused before anything is written
    paths[0].write_bytes(paths[0].read_bytes() + b"\0")
    bad = tmp_path / "bad" / "submission_rank.csv"
    assert ps.main(_args(inputs, bad, 5, 3, 2, "--set-models", str(paths[0]), "--set-base", str(MODEL))) == ps.EXIT_INPUTS
    assert "sha256" in capsys.readouterr().err and not bad.exists()


def _tiny_l1b_heads(tmp_path):
    """Two L1b heads files (A with a fine-tuned trunk file, B without) of a random b2c16 model; returns (paths, sidecars)."""
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    from aicup import l1b

    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    paths, sides = [], []
    for name, train_folds, with_trunk in (("heads3_A", [1, 2], True), ("heads3_B", [3, 4], False)):
        heads = l1b.RankHeads(model.policy_head)
        with torch.no_grad():
            heads.heads[0].conv1p.weight.add_(0.1 if name.endswith("A") else 0.2)
        pt = tmp_path / "l1b" / f"{name}.pt"
        sides.append(l1b.save_heads(heads, pt, {"train_folds": train_folds, "max_games": 40}, model_state=model.state_dict() if with_trunk else None))
        paths.append(pt)
    return paths, sides


def _tiny_fused_doc(tmp_path, sides, names=("A", "B"), shuffle_models=False, opponent=False, feature_set=None):
    """A fused (or, with opponent, fused_opp; or the named feature_set) model document whose L1b models are the tiny heads'
    sidecars (feature folds as in the real setup)."""
    from aicup import fused
    from aicup.rankmodel import MLPModel

    feature_set = fused.feature_set_of(opponent, feature_set)
    rng = np.random.RandomState(5)
    mlp = MLPModel(hidden=4, epochs=20).fit(rng.randn(120, len(fused.FEATURE_SETS[feature_set])), np.arange(120) % 10)
    records = [fused.l1b_model_record(n, f"heads3_{n}.pt", s, [3, 4] if n == "A" else [1, 2]) for n, s in zip(names, sides)]
    if shuffle_models:
        records = records[::-1]
    doc = fused.fused_document(mlp, records, [1, 2, 3, 4], 32, 120, "d" * 64, feature_set=feature_set)
    path = tmp_path / "models" / f"rank_model_{feature_set}.json"
    fused.save_fused_model(doc, path)
    return path, mlp, records


def test_l1b_heads_in_shards_and_fused_decision(tmp_path, fake_scorer, capsys):
    from aicup import fused

    paths, sides = _tiny_l1b_heads(tmp_path)
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    heads_args = ["--l1b-heads", *map(str, paths)]
    # shard step with L1b heads: the shard file carries the L1b sums and per-move log p of both models and names them
    assert ps.main(_args(inputs, out, 5, 3, 2, *heads_args)) == ps.EXIT_OK  # single run: decides with rank_model (no --fused-model)
    with np.load(ps.shard_path(out, 0, 1), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        assert z["sums"].shape[1] == 18 and z["move_logp"].shape[1] == 18  # the existing arrays are unchanged
        assert z["l1b_sums"].shape == (len(z["task"]), 2, 10) and z["l1b_sums"].dtype == np.float64
        assert z["l1b_move_logp"].shape == (len(z["move_task"]), 2, 10) and z["l1b_move_logp"].dtype == np.float32
        np.testing.assert_allclose(z["l1b_sums"][0], z["l1b_move_logp"][z["move_task"] == z["task"][0]].sum(axis=0), rtol=0, atol=1e-3)
        # the opponent pass (issue #12): per game the opponent's move count and L1b sums of both models; since 2026-11-04 also
        # the opponent plies' per-move log p (read by scripts/trunk_predict.py only)
        assert z["l1b_opp_sums"].shape == (len(z["task"]), 2, 10) and z["l1b_opp_sums"].dtype == np.float64
        assert z["l1b_opp_n_moves"].shape == (len(z["task"]),) and z["l1b_opp_n_moves"].dtype == np.int64
        assert (np.abs(z["l1b_opp_n_moves"] - z["n_moves"]) <= 1).all() and not np.array_equal(z["l1b_opp_sums"], z["l1b_sums"])
        opp_total = int(z["l1b_opp_n_moves"].sum())
        assert z["l1b_opp_move_logp"].shape == (opp_total, 2, 10) and z["l1b_opp_move_logp"].dtype == np.float32
        assert z["opp_move_task"].shape == z["opp_move_t"].shape == (opp_total,) and set(z["opp_move_task"].tolist()) == set(z["task"].tolist())
        k0 = z["task"][0]
        np.testing.assert_allclose(z["l1b_opp_sums"][0], z["l1b_opp_move_logp"][z["opp_move_task"] == k0].sum(axis=0), rtol=1e-5, atol=1e-3)  # float32 per-move sums
        assert len(z["opp_move_t"][z["opp_move_task"] == k0]) == z["l1b_opp_n_moves"][0] and meta["l1b_opp_moves"] is True
    assert [m["name"] for m in meta["l1b_models"]] == ["heads3_A", "heads3_B"]
    assert meta["l1b_models"][0]["state_sha256"] == sides[0]["state_sha256"] and meta["l1b_models"][0]["trunk_sha256"] == sides[0]["trunk_sha256"]
    assert meta["l1b_models"][1]["trunk_sha256"] is None and meta["l1b_models"][0]["train_folds"] == [1, 2]
    assert meta["l1b_opponent"] is True and meta["opp_positions"] == opp_total and meta["opp_scoring_wall_s"] == 0.25
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["decision_model"] == "rank_model" and [m["name"] for m in report["l1b_models"]] == ["heads3_A", "heads3_B"]
    assert report["l1b_opponent"] is True and report["opp_positions"] == meta["opp_positions"]
    # merge with the fused model: pairing by sha256, MLP per model, probabilities averaged, expected-score decision
    fused_path, mlp, records = _tiny_fused_doc(tmp_path, sides)
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(fused_path))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["decision_model"] == "fused" and report["fused_model"]["path"] == str(fused_path) and report["fused_model"]["opponent"] is False
    assert report["fused_model"]["file_sha256"] == fused.file_sha256(fused_path) and report["fused_model"]["n_models"] == 2
    assert report["fused_fallback_questions"] == [] and report["exit_code"] == 0
    questions, _ = ps.load_questions(inputs)
    tasks, owner, _, _ = ps.build_tasks(questions)
    per_task, _, _ = ps.load_shards(out, 1, len(tasks), ps.questions_sha(inputs, questions))
    l1b_models, l1b_per_task, _, opp_per_task = ps.load_shards_l1b(out, 1)
    assert set(opp_per_task) == set(per_task) and all(isinstance(v[0], int) and v[1].shape == (2, 10) for v in opp_per_task.values())
    order = fused.pairing_order({"l1b_models": records}, l1b_models)
    assert order == [0, 1]
    opp_path, opp_mlp, _ = _tiny_fused_doc(tmp_path / "opp", sides, opponent=True)
    want, want_o = [], []
    for qi in range(len(questions)):
        ks = [k for k, q in enumerate(owner) if q == qi]
        hsl = np.stack([per_task[k][1] for k in ks])
        l1b = np.stack([l1b_per_task[k] for k in ks])
        n = np.array([per_task[k][0] for k in ks])
        want.append(fused.fused_predict(mlp, hsl, l1b, n, order))
        want_o.append(fused.fused_predict(opp_mlp, hsl, l1b, n, order, opp_sums=np.stack([opp_per_task[k][1] for k in ks]),
                                          opp_n_moves=np.array([opp_per_task[k][0] for k in ks])))
    got = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got, np.stack(want), rtol=0, atol=1e-12)
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert sub["pred_rank"].tolist() == [LEVELS[expected_score_decision(p)] for p in want]
    # the models listed in another order in the document: the same decision (pairing is by sha256, not by position)
    fused_path2, _, _ = _tiny_fused_doc(tmp_path / "other", sides, shuffle_models=True)
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(fused_path2))) == ps.EXIT_OK
    np.testing.assert_allclose(pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy(), got, atol=1e-12)
    # the fused_opp document on the same shard files: the opponent's sums and move counts join the features
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(opp_path))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["decision_model"] == "fused_opp" and report["fused_model"]["opponent"] is True and report["fused_model"]["name"] == fused.NAME_OPP
    assert report["fused_fallback_questions"] == [] and report["exit_code"] == 0
    got_o = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got_o, np.stack(want_o), rtol=0, atol=1e-12)
    assert not np.allclose(got_o, got)
    assert pd.read_csv(out, dtype=str, keep_default_na=False)["pred_rank"].tolist() == [LEVELS[expected_score_decision(p)] for p in want_o]
    # sharded scoring + merge gives the same files, for both documents
    sharded = tmp_path / "sharded" / "submission_rank.csv"
    for i in range(2):
        assert ps.main(_args(inputs, sharded, 5, 3, 2, "--shard", str(i), "--nshards", "2", *heads_args)) == ps.EXIT_OK
    assert ps.main(_args(inputs, sharded, 5, 3, 2, "--merge", "--nshards", "2", "--fused-model", str(fused_path))) == ps.EXIT_OK
    np.testing.assert_allclose(pd.read_csv(sharded.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy(), got, atol=1e-12)
    assert ps.main(_args(inputs, sharded, 5, 3, 2, "--merge", "--nshards", "2", "--fused-model", str(opp_path))) == ps.EXIT_OK
    np.testing.assert_allclose(pd.read_csv(sharded.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy(), got_o, atol=1e-12)
    assert json.loads(sharded.with_name("submission_rank_report.json").read_text())["opp_positions"] == meta["opp_positions"]
    # the fused_opp_diff document (issue #25) on the same shards: the three differences come from the per-game sums
    diff_path, diff_mlp, _ = _tiny_fused_doc(tmp_path / "diff", sides, feature_set="fused_opp_diff")
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(diff_path))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["decision_model"] == "fused_opp_diff" and report["fused_model"]["feature_set"] == "fused_opp_diff" and report["fused_model"]["name"] == fused.NAME_OPP_DIFF
    want_d = []
    for qi in range(len(questions)):
        ks = [k for k, q in enumerate(owner) if q == qi]
        want_d.append(fused.fused_predict(diff_mlp, np.stack([per_task[k][1] for k in ks]), np.stack([l1b_per_task[k] for k in ks]), np.array([per_task[k][0] for k in ks]), order,
                                          opp_sums=np.stack([opp_per_task[k][1] for k in ks]), opp_n_moves=np.array([opp_per_task[k][0] for k in ks]), feature_set="fused_opp_diff"))
    got_d = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got_d, np.stack(want_d), rtol=0, atol=1e-12)
    assert not np.allclose(got_d, got_o) and report["fused_fallback_questions"] == []
    # the fused_opp_diff_hd document (issue #47): the handicap features come from the records' HD[]; the mock inputs carry none -> refused
    from aicup.handicap import question_hd_features_from_games
    hd_path, hd_mlp, _ = _tiny_fused_doc(tmp_path / "hd", sides, feature_set="fused_opp_diff_hd")
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(hd_path))) == ps.EXIT_INPUTS
    assert "carry no HD[]" in capsys.readouterr().err
    # --allow-missing-hd: unknown handicap, zeros
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(hd_path), "--allow-missing-hd")) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    n_games = sum(len(q["games"]) for q in questions)
    assert report["decision_model"] == "fused_opp_diff_hd" and report["fused_model"]["name"] == fused.NAME_OPP_DIFF_HD and report["handicap"] == {"absent": n_games}

    def want_hd(task_games):
        rows = []
        for qi in range(len(questions)):
            ks = [k for k, q in enumerate(owner) if q == qi]
            rows.append(fused.fused_predict(hd_mlp, np.stack([per_task[k][1] for k in ks]), np.stack([l1b_per_task[k] for k in ks]), np.array([per_task[k][0] for k in ks]), order,
                                            opp_sums=np.stack([opp_per_task[k][1] for k in ks]), opp_n_moves=np.array([opp_per_task[k][0] for k in ks]),
                                            feature_set="fused_opp_diff_hd", hd_feats=question_hd_features_from_games([task_games[k] for k in ks])))
        return np.stack(rows)
    got_h0 = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got_h0, want_hd({k: (sgf, c) for k, sgf, c in tasks}), rtol=0, atol=1e-12)
    # inputs whose records carry HD[1] (讓先: boards and colours as recorded, so the same shards serve; the file names and the
    # question ids are the same, so the shards' hash matches): the features come from the records, the report counts them
    hd_dir = tmp_path / "hd_inputs"
    hd_dir.mkdir()
    hd_inputs = []
    for path in inputs:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        sgf_cols = [c for c in df.columns if c.startswith("sgf_")]
        df[sgf_cols] = df[sgf_cols].map(lambda v: v.replace("(;SZ[19]", "(;SZ[19]HD[1]", 1) if v else v)
        df.to_csv(hd_dir / Path(path).name, index=False)
        hd_inputs.append(str(hd_dir / Path(path).name))
    assert ps.main(_args(hd_inputs, out, 5, 3, 2, "--merge", "--fused-model", str(hd_path))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    assert report["handicap"] == {"1": n_games} and report["decision_model"] == "fused_opp_diff_hd" and report["exit_code"] == 0
    questions_hd, _ = ps.load_questions(hd_inputs)
    tasks_hd, owner_hd, _, _ = ps.build_tasks(questions_hd)
    assert owner_hd == owner and [(k, c) for k, _, c in tasks_hd] == [(k, c) for k, _, c in tasks]
    got_h1 = pd.read_csv(out.with_name("submission_rank_proba.csv"))[[f"p_{lv}" for lv in LEVELS]].to_numpy()
    np.testing.assert_allclose(got_h1, want_hd({k: (sgf, c) for k, sgf, c in tasks_hd}), rtol=0, atol=1e-12)
    assert not np.allclose(got_h1, got_h0)


def test_fused_decision_refusals_and_fallback(tmp_path, fake_scorer, capsys):
    paths, sides = _tiny_l1b_heads(tmp_path)
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    fused_path, mlp, records = _tiny_fused_doc(tmp_path, sides)
    # shards scored without --l1b-heads cannot feed a fused decision
    assert ps.main(_args(inputs, out, 5, 3, 2)) == ps.EXIT_OK
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(fused_path))) == ps.EXIT_INPUTS
    assert "L1b scores" in capsys.readouterr().err
    # shards scored with one model only: the document's other model is missing -> refused
    assert ps.main(_args(inputs, out, 5, 3, 2, "--l1b-heads", str(paths[0]))) == ps.EXIT_OK
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(fused_path))) == ps.EXIT_INPUTS
    assert "B" in capsys.readouterr().err
    # a document naming other heads (another sha256) is refused
    assert ps.main(_args(inputs, out, 5, 3, 2, "--l1b-heads", *map(str, paths))) == ps.EXIT_OK
    doc = json.loads(fused_path.read_text())
    doc["l1b_models"][0]["state_sha256"] = "0" * 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(bad))) == ps.EXIT_INPUTS
    assert "heads3_A" in capsys.readouterr().err
    # --fused-model and --set-models exclude each other
    with pytest.raises(SystemExit) as e:
        ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(fused_path), "--set-models", str(paths[0]), "--set-base", str(MODEL)))
    assert e.value.code == 2
    # shards scored with --no-l1b-opponent carry no opponent sums: a fused_opp document is refused, the fused one still decides
    opp_path, _, _ = _tiny_fused_doc(tmp_path / "opp", sides, opponent=True)
    no_opp = tmp_path / "noopp" / "submission_rank.csv"
    assert ps.main(_args(inputs, no_opp, 5, 3, 2, "--l1b-heads", *map(str, paths), "--no-l1b-opponent")) == ps.EXIT_OK
    with np.load(ps.shard_path(no_opp, 0, 1), allow_pickle=False) as z:
        assert "l1b_opp_sums" not in z.files and "l1b_opp_n_moves" not in z.files and json.loads(str(z["__meta__"]))["l1b_opponent"] is False
    assert ps.load_shards_l1b(no_opp, 1)[3] is None
    assert ps.main(_args(inputs, no_opp, 5, 3, 2, "--merge", "--fused-model", str(opp_path))) == ps.EXIT_INPUTS
    assert "opponent" in capsys.readouterr().err
    diff_path, _, _ = _tiny_fused_doc(tmp_path / "diff", sides, feature_set="fused_opp_diff")
    assert ps.main(_args(inputs, no_opp, 5, 3, 2, "--merge", "--fused-model", str(diff_path))) == ps.EXIT_INPUTS
    assert "opponent" in capsys.readouterr().err
    assert ps.main(_args(inputs, no_opp, 5, 3, 2, "--merge", "--fused-model", str(fused_path))) == ps.EXIT_OK
    assert json.loads(no_opp.with_name("submission_rank_report.json").read_text())["l1b_opponent"] is False
    # shards scored differently (one with the opponent pass, one without) are refused at the merge
    mixed = tmp_path / "mixed" / "submission_rank.csv"
    assert ps.main(_args(inputs, mixed, 5, 3, 2, "--shard", "0", "--nshards", "2", "--l1b-heads", *map(str, paths))) == ps.EXIT_OK
    assert ps.main(_args(inputs, mixed, 5, 3, 2, "--shard", "1", "--nshards", "2", "--l1b-heads", *map(str, paths), "--no-l1b-opponent")) == ps.EXIT_OK
    assert ps.main(_args(inputs, mixed, 5, 3, 2, "--merge", "--nshards", "2", "--fused-model", str(fused_path))) == ps.EXIT_INPUTS
    assert "scored alike" in capsys.readouterr().err
    # non-finite fused probabilities fall back to rank_model's for that question and are reported
    doc = json.loads(fused_path.read_text())
    doc["state"]["params"]["0.weight"] = [[float("nan")] * 40 for _ in doc["state"]["params"]["0.weight"]]
    nan_doc = tmp_path / "nan.json"
    nan_doc.write_text(json.dumps(doc))
    assert ps.main(_args(inputs, out, 5, 3, 2, "--merge", "--fused-model", str(nan_doc))) == ps.EXIT_OK
    report = json.loads(out.with_name("submission_rank_report.json").read_text())
    ids = [q["question_id"] for q in ps.load_questions(inputs)[0]]
    assert report["fused_fallback_questions"] == ids and report["decision_model"] == "fused" and report["exit_code"] == 0
    assert "fused" in capsys.readouterr().err
    plain = tmp_path / "plain" / "submission_rank.csv"
    assert ps.main(_args(inputs, plain, 5, 3, 2)) == ps.EXIT_OK
    assert plain.with_name("submission_rank_proba.csv").read_text() == out.with_name("submission_rank_proba.csv").read_text()
