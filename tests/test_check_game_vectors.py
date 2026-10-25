"""scripts/check_game_vectors.py: the pipeline's per-game vectors against a fold vectors file."""
import json

import hashlib

import numpy as np
import pandas as pd
import pytest

import check_game_vectors as cgv
import predict_submission as ps
from aicup.metric import LEVELS
from tests.test_predict_submission import MODEL, _args, _fake_score, _small_inputs, _tiny_set_models


def _distinct_score(tasks, workers, log=print):
    """_fake_score plus a per-game constant from the SGF hash: its 126 curves collide on games of equal
    length (two of the 30 small-input games), which real log p never does; vectors must stay consistent."""
    per_task, (logp, mt, tt), positions, secs = _fake_score(tasks, workers, log)
    logp = logp.astype(np.float64)
    for k, sgf, _ in tasks:
        rows = mt == k
        logp[rows] -= 1e-3 * (int(hashlib.sha256(sgf.encode()).hexdigest()[8:16], 16) % 1009)
        per_task[k] = (per_task[k][0], logp[rows].sum(axis=0))
    return per_task, (logp.astype(np.float32), mt, tt), positions, secs


@pytest.fixture
def distinct_scorer(monkeypatch):
    monkeypatch.setattr(ps, "score_tasks", _distinct_score)


def _run_and_vectors(tmp_path, inputs):
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2)) == ps.EXIT_OK
    questions, _ = ps.load_questions(inputs)
    tasks, _, _, _ = ps.build_tasks(questions)
    per_task, _, moves = ps.load_shards(out, 1, len(tasks), ps.questions_sha(inputs, questions), with_moves=True)
    return out, ps.game_vectors(per_task, moves)


def _ref_from(vec, rng, extra=4):
    """A fold vectors file holding the run's games (shuffled, other keys) plus `extra` unrelated games."""
    T = len(vec["n_moves"])
    perm = rng.permutation(T)
    ref = {k: vec[k][perm].copy() for k in ("staged", "whole", "sums", "n_moves")}
    ref["game_key"] = np.asarray([f"3D:g{i:05d}" for i in perm], dtype=str)
    pad = {"staged": rng.randn(extra, 300).astype(np.float32), "whole": rng.randn(extra, 75).astype(np.float32),
           "sums": -rng.rand(extra, 18) * 300, "n_moves": rng.randint(30, 200, size=extra), "game_key": np.asarray([f"1D:x{i}" for i in range(extra)], dtype=str)}
    return {k: np.concatenate([ref[k], pad[k]]) for k in ref}


def test_match_compare_and_cli(tmp_path, distinct_scorer, capsys):
    rng = np.random.RandomState(0)
    inputs = _small_inputs(tmp_path)
    out, vec = _run_and_vectors(tmp_path, inputs)
    T = len(vec["n_moves"])
    ref = _ref_from(vec, rng)
    match, unmatched, ambiguous = cgv.match_tasks(vec, ref, sum_tol=0.5)
    assert (match >= 0).all() and not unmatched and not ambiguous
    np.testing.assert_array_equal(ref["sums"][match], vec["sums"])  # the right rows
    rows = cgv.compare(vec, ref, match)
    assert set(rows) == {f"{rep}.{stat}" for rep in ("staged", "whole") for stat in ("mean", "std", "q10", "q90", "n_eff", "mean_surprisal", "present")} | {"log1p_n_moves"}
    assert all(r["max_abs_diff"] == 0.0 for r in rows.values())
    assert rows["staged.n_eff"]["count"] and rows["staged.present"]["count"] and not rows["staged.mean"]["count"]
    vectors = tmp_path / "fold0.npz"
    np.savez_compressed(vectors, **ref)
    report = tmp_path / "check.json"
    rc = cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors), "--report", str(report)])
    printed = capsys.readouterr().out
    assert rc == 0 and f"{T} of {T} tasks matched" in printed and printed.rstrip().endswith("PASS")
    res = json.loads(report.read_text())
    assert res["pass"] and res["matched"] == T and res["failed_groups"] == [] and res["unmatched"] == [] and res["ambiguous"] == []
    # the sums of a sharded run differ slightly (TF32): still matched, still equal vectors
    noisy = {**ref, "sums": ref["sums"] + rng.uniform(-0.01, 0.01, size=ref["sums"].shape)}
    match2, un2, amb2 = cgv.match_tasks(vec, noisy, sum_tol=0.5)
    np.testing.assert_array_equal(match2, match) and not un2 and not amb2
    # a perturbed field group fails and is named; count fields must agree exactly
    bad = {**ref, "staged": ref["staged"].copy()}
    bad["staged"][:, 0] += 0.2  # s1_mean_<first profile>
    np.savez_compressed(vectors, **bad)
    rc = cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors), "--tol", "0.05"])
    printed = capsys.readouterr().out
    assert rc == 1 and "FAIL" in printed and "staged.mean" in printed.split("FAIL:")[-1]
    assert cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors), "--tol", "0.5"]) == 0
    tiny = {**ref, "staged": ref["staged"].copy()}
    present_col = [i for i, n in enumerate(cgv.vec_names(cgv.SEGMENTS)) if n == "s1_present"][0]
    tiny["staged"][0, present_col] = 0.0  # a count field off by one on one game
    np.savez_compressed(vectors, **tiny)
    assert cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors), "--tol", "10"]) == 1
    assert "staged.present" in capsys.readouterr().out.split("FAIL:")[-1]
    # unmatched (a game missing from the file) and ambiguous (a near duplicate) tasks fail the check
    missing = {k: v[1:] for k, v in ref.items()}
    np.savez_compressed(vectors, **missing)
    assert cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors)]) == 1
    assert "1 unmatched" in capsys.readouterr().out
    dup = {k: np.concatenate([v, v[:1]]) for k, v in ref.items()}
    np.savez_compressed(vectors, **dup)
    assert cgv.main(["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors)]) == 1
    assert "1 ambiguous" in capsys.readouterr().out


def test_set_model_probabilities_offline_vs_pipeline(tmp_path, distinct_scorer, monkeypatch, capsys):
    paths = _tiny_set_models(tmp_path, monkeypatch)
    inputs = _small_inputs(tmp_path)
    out = tmp_path / "run" / "submission_rank.csv"
    assert ps.main(_args(inputs, out, 5, 3, 2, "--set-models", *map(str, paths), "--set-base", str(MODEL))) == ps.EXIT_OK
    questions, _ = ps.load_questions(inputs)
    tasks, _, _, _ = ps.build_tasks(questions)
    per_task, _, moves = ps.load_shards(out, 1, len(tasks), ps.questions_sha(inputs, questions), with_moves=True)
    ref = _ref_from(ps.game_vectors(per_task, moves), np.random.RandomState(3))
    vectors = tmp_path / "fold0.npz"
    np.savez_compressed(vectors, **ref)
    args = ["--inputs", *inputs, "--out", str(out), "--nshards", "1", "--vectors", str(vectors),
            "--set-models", *map(str, paths), "--set-base", str(MODEL)]
    report = tmp_path / "check.json"
    assert cgv.main([*args, "--report", str(report)]) == 0
    res = json.loads(report.read_text())
    pr = res["proba"]
    assert res["pass"] and pr["pass"] and pr["n_questions"] == 5 and pr["max_abs_diff"] <= 1e-12 and pr["decisions_differ"] == []
    assert pr["models"] == [str(p) for p in paths] and pr["warnings"] == [] and "set models: 5 questions" in capsys.readouterr().out
    # the run's probabilities moved by 2e-3 on one question: over the default tolerance, within --proba-tol 0.01
    proba_csv = out.with_name("submission_rank_proba.csv")
    df = pd.read_csv(proba_csv)
    cols, orig = [f"p_{lv}" for lv in LEVELS], df[[f"p_{lv}" for lv in LEVELS]].to_numpy().copy()
    moved = orig.copy()
    moved[0, 0] += 2e-3
    moved[0, 1] -= 2e-3
    df[cols] = moved
    df.to_csv(proba_csv, index=False)
    assert cgv.main([*args, "--report", str(report)]) == 1
    pr = json.loads(report.read_text())["proba"]
    assert abs(pr["max_abs_diff"] - 2e-3) < 1e-9 and pr["question"] == df["question_id"][0] and not pr["pass"]
    assert "proba" in capsys.readouterr().out.split("FAIL:")[-1]
    assert cgv.main([*args, "--proba-tol", "0.01"]) == 0
    # a decision that differs from the run's submission fails whatever the tolerance
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    sub.loc[0, "pred_rank"] = LEVELS[0] if sub.loc[0, "pred_rank"] != LEVELS[0] else LEVELS[-1]
    sub.to_csv(out, index=False)
    assert cgv.main([*args, "--proba-tol", "1", "--report", str(report)]) == 1
    assert json.loads(report.read_text())["proba"]["decisions_differ"] == [sub.loc[0, "question_id"]]
    assert "1 decisions differ" in capsys.readouterr().out
    # with an unmatched task the probabilities are not checked, the check fails anyway
    np.savez_compressed(vectors, **{k: v[1:] for k, v in ref.items()})
    assert cgv.main(args) == 1 and "not checked" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cgv.main([*args[:-2]])  # --set-models without --set-base


def test_vectors_match_the_ladder_builder():
    """game_vectors uses the same gamevec functions as ladder_build_vectors: a hand-made game agrees."""
    from aicup.gamevec import SEGMENTS, WHOLE, game_vector
    rng = np.random.RandomState(1)
    logp, t = rng.randn(57, 18), np.sort(rng.choice(200, 57, replace=False))
    per_task = {0: (57, logp.sum(axis=0))}
    vec = ps.game_vectors(per_task, {0: (logp, t)})
    np.testing.assert_array_equal(vec["staged"][0], game_vector(logp, t, SEGMENTS))
    np.testing.assert_array_equal(vec["whole"][0], game_vector(logp, t, WHOLE))
    assert vec["n_moves"].tolist() == [57] and vec["game_key"].tolist() == ["0"]
    np.testing.assert_array_equal(vec["sums"][0], logp.sum(axis=0))
