"""scripts/trunk_predict.py and the checkpoint side of scripts/trunk_seq_check.py: the weights round trip, the token and score
assembly against trunk_seq_check.build's layout, and the predict command end to end on synthetic shard files and trunk-vector
parts (no GPU, no KataGo model: the vectors and log p are random but laid out exactly as the real steps write them)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import predict_submission as ps
import trunk_predict as tp
import trunk_seq_check as ts
from aicup.metric import LEVELS, expected_score_decision
from aicup.scoring import part_paths
from aicup.setmodel_io import file_sha256, sidecar_path
from aicup.sgf import setup_and_moves

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mock_fold0_40"
PUBLIC, PRIVATE = FIXTURE / "rank_prediction_test_public.csv", FIXTURE / "rank_prediction_test_private.csv"
K = 2  # L1b models in the synthetic shards (heads3_A, heads3_B)


def _small_inputs(tmp_path, n_pub=3, n_priv=2, break_question=None):
    """The first questions of the fixture as two CSVs; break_question: that public question's games become 13x13 records (unusable)."""
    d = tmp_path / "inputs"
    d.mkdir(exist_ok=True)
    paths = []
    for src, n, stem in ((PUBLIC, n_pub, "rank_prediction_test_public"), (PRIVATE, n_priv, "rank_prediction_test_private")):
        df = pd.read_csv(src, dtype=str, keep_default_na=False).head(n)
        if break_question is not None and stem.endswith("public"):
            for i in range(1, 21):
                if df.at[break_question, f"sgf_{i}"]:
                    df.at[break_question, f"sgf_{i}"] = "(;GM[1]FF[4]SZ[13];B[aa];W[bb])"
        path = d / f"{stem}.csv"
        df.to_csv(path, index=False)
        paths.append(str(path))
    return paths


def _checkpoint(tmp_path, arm, seed, n_r=64):
    torch.manual_seed(seed)
    model = ts.Model(arm, n_r)
    rng = np.random.RandomState(seed)
    stats = (rng.randn(ts.D_TOKEN - 4).astype(np.float32), (0.5 + rng.rand(ts.D_TOKEN - 4)).astype(np.float32))
    r_stats = (rng.randn(n_r), 0.5 + rng.rand(n_r))
    path = tmp_path / "ct" / f"all_{arm}_s{seed}.pt"
    side = ts.save_checkpoint(path, model, stats, r_stats, {"tag": arm, "seed": seed, "val_fold": "all", "train_folds": [1, 2, 3, 4], "best_inner": 0.7})
    return path, model.eval(), stats, r_stats, side


def _synthetic_run(tmp_path, inputs, out_main):
    """One shard file (as predict_submission.save_shard writes it, with the opponent plies' log p) and two vec parts (as
    trunk_extract.write_part writes them) for the usable games of `inputs`: every game's plies from setup_and_moves, random trunk
    vectors and log p seeded by the task. Returns (questions, tasks, owner, {task: (hsl [n_t, 18], l1b_t [n_t, K, 10], l1b_o [n_o, K, 10], g, l)})."""
    questions, _ = ps.load_questions(inputs)
    tasks, owner, _, _ = ps.build_tasks(questions)
    sha = ps.questions_sha(inputs, questions)
    per_task, logps, mt, tt, o_per_task, o_logps, o_mt, o_tt, truth = {}, [], [], [], {}, [], [], [], {}
    parts = [[], []]
    for gi, (k, sgf, color) in enumerate(tasks):
        _, moves = setup_and_moves(sgf)
        n = len(moves)
        rng = np.random.RandomState(1000 + k)
        mover = np.array([0 if c == "B" else 1 for c, _, _ in moves], dtype=np.int8)
        target = np.array([c == color for c, _, _ in moves])
        t = np.arange(n)
        hsl = (rng.randn(int(target.sum()), 18) - 2).astype(np.float32)
        l1b_t = (rng.randn(int(target.sum()), K * 10) - 1.5).astype(np.float32)
        l1b_o = (rng.randn(int((~target).sum()), K * 10) - 1.2).astype(np.float32)
        per_task[k] = (int(target.sum()), np.hstack([hsl, l1b_t]).sum(axis=0).astype(np.float64))
        logps.append(np.hstack([hsl, l1b_t]))
        mt.append(np.full(int(target.sum()), k, dtype=np.int32))
        tt.append(t[target].astype(np.int32))
        o_per_task[k] = (int((~target).sum()), l1b_o.sum(axis=0).astype(np.float64))
        o_logps.append(l1b_o)
        o_mt.append(np.full(int((~target).sum()), k, dtype=np.int32))
        o_tt.append(t[~target].astype(np.int32))
        g, l = rng.randn(n, 64).astype(np.float16), rng.randn(n, 64).astype(np.float16)
        truth[k] = (hsl, l1b_t.reshape(-1, K, 10), l1b_o.reshape(-1, K, 10), g, l, target, mover)
        parts[gi % 2].append((k, n, g, l, rng.randint(0, 362, n).astype(np.int16), mover, target, t.astype(np.int16), color))
    vec = tmp_path / "vec"
    vec.mkdir(exist_ok=True)
    for p, rows in enumerate(parts):
        csv_path, npz_path = part_paths(vec, 0, p)
        np.savez(npz_path, g=np.concatenate([r[2] for r in rows]), l=np.concatenate([r[3] for r in rows]), pos=np.concatenate([r[4] for r in rows]),
                 mover=np.concatenate([r[5] for r in rows]), target=np.concatenate([r[6] for r in rows]),
                 game_row=np.repeat(np.arange(len(rows), dtype=np.int32), [r[1] for r in rows]), t=np.concatenate([r[7] for r in rows]))
        pd.DataFrame([{"game_key": tp.task_key(r[0]), "task": r[0], "question_id": questions[owner[r[0]]]["question_id"], "color": r[8], "n_plies": r[1]}
                      for r in rows]).to_csv(csv_path, index=False)
    l1b_models = [{"name": f"heads3_{x}", "path": f"results/l1b/heads3_{x}.pt", "state_sha256": x * 64, "trunk_file": None, "trunk_sha256": None,
                   "train_folds": [1, 2] if x == "A" else [3, 4], "levels": list(LEVELS)} for x in "AB"]
    out_main.parent.mkdir(parents=True, exist_ok=True)
    ps.save_shard(ps.shard_path(out_main, 0, 1), 0, 1, tasks, per_task, (np.concatenate(logps), np.concatenate(mt), np.concatenate(tt)),
                  int(sum(len(x) for x in logps)), 1.0, inputs, sha, len(tasks), l1b_models,
                  (o_per_task, int(sum(len(x) for x in o_logps)), 0.5, (np.concatenate(o_logps), np.concatenate(o_mt), np.concatenate(o_tt))))
    return questions, tasks, owner, truth


def test_checkpoint_round_trip_and_checks(tmp_path):
    path, model, stats, r_stats, side = _checkpoint(tmp_path, "T", 0)
    assert side["state_sha256"] == file_sha256(path) and (side["arm"], side["n_r"], side["d_token"], side["n_scores"]) == ("T", 64, ts.D_TOKEN, ts.N_SCORES)
    assert json.loads(sidecar_path(path).read_text()) == side and side["val_fold"] == "all" and side["hidden"] == ts.HIDDEN["T"]
    loaded, stats2, r2, side2 = ts.load_checkpoint(path, "cpu")
    assert side2 == side and not loaded.training
    np.testing.assert_array_equal(stats2[0], stats[0]) and np.testing.assert_array_equal(stats2[1], stats[1])
    np.testing.assert_allclose(r2[0], r_stats[0], rtol=1e-6) and np.testing.assert_allclose(r2[1], r_stats[1], rtol=1e-6)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)
    rng = np.random.RandomState(1)
    X = rng.randn(50, ts.D_TOKEN).astype(np.float16)
    X[:, ts.D_TOKEN - 4:] = np.column_stack([rng.randint(0, 2, 50), rng.randint(0, 2, 50), rng.rand(50), np.zeros(50)])  # the flags are 0/1, t / T in [0, 1]
    X = torch.as_tensor(X)
    offsets = {"a": (0, 20), "b": (20, 10), "c": (30, 20)}
    R = rng.randn(2, 64).astype(np.float32)
    P1, P2 = (ts.predict(m, R, [["a", "b"], ["c"]], X, offsets, "cpu") for m in (model, loaded))
    np.testing.assert_allclose(P1, P2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(P1.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    # an R0 checkpoint has no token statistics
    r0 = ts.save_checkpoint(tmp_path / "ct" / "r0.pt", ts.Model("R0", 64), None, r_stats, {})
    assert r0["arm"] == "R0" and ts.load_checkpoint(tmp_path / "ct" / "r0.pt")[1] is None
    # tampering and wrong sidecars are refused
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        ts.load_checkpoint(path)
    other = tmp_path / "ct" / "no_side.pt"
    other.write_bytes(b"x")
    with pytest.raises(ValueError, match="sidecar"):
        ts.load_checkpoint(other)
    bad = json.loads(sidecar_path(path).read_text())
    bad["state_sha256"], bad["q_dim"] = file_sha256(path), ts.Q_DIM + 1
    sidecar_path(path).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="layout"):
        ts.load_checkpoint(path)


def test_predict_end_to_end_on_synthetic_shards_and_parts(tmp_path):
    inputs = _small_inputs(tmp_path)
    out_main = tmp_path / "run" / "submission_rank.csv"
    questions, tasks, owner, truth = _synthetic_run(tmp_path, inputs, out_main)
    ids = [q["question_id"] for q in questions]
    sha = ps.questions_sha(inputs, questions)
    ck_c, ck_t = _checkpoint(tmp_path, "C", 0), _checkpoint(tmp_path, "T", 1)
    fb = tmp_path / "run" / "main.csv"
    pd.DataFrame({"question_id": ids, "pred_rank": ["3D"] * len(ids)}).to_csv(fb, index=False)
    out = tmp_path / "run" / "submission_rank_ct.csv"
    base = ["predict", "--inputs", *inputs, "--shards", str(out_main), "--nshards", "1", "--vec", str(tmp_path / "vec"), "--out", str(out),
            "--expect-rows", "5", "--expect-public", "3", "--expect-private", "2", "--fallback-submission", str(fb)]
    # the fixture games carry no HD[]: refused unless --allow-missing-hd
    assert tp.main(base + ["--models", str(ck_c[0]), str(ck_t[0])]) == tp.EXIT_INPUTS
    assert tp.main(base + ["--models", str(ck_c[0]), str(ck_t[0]), "--allow-missing-hd"]) == tp.EXIT_OK
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert sub["question_id"].tolist() == ids and set(sub["pred_rank"]) <= set(LEVELS)
    proba = pd.read_csv(out.with_name("submission_rank_ct_proba.csv"))
    assert proba["question_id"].tolist() == ids and list(proba.columns) == tp.P_COLS + ["question_id"]
    np.testing.assert_allclose(proba[tp.P_COLS].sum(axis=1), 1.0, rtol=0, atol=1e-6)
    report = json.loads(out.with_name("submission_rank_ct_report.json").read_text())
    assert (report["passes"], report["exit_code"], report["problems"], report["fallback_questions"]) == (4, 0, [], [])
    assert report["l1b_models_used"] == ["heads3_A", "heads3_B"] and [c["arm"] for c in report["checkpoints"]] == ["C", "T"]
    assert report["vec_plies"] == sum(len(v[5]) for v in truth.values()) and report["games_used"] == len(tasks) == len(truth)
    assert report["vs_fallback_submission"]["questions_compared"] == 5 and set(report["arm_agreement_with_ensemble"]) == {"C", "T"}
    assert sub["pred_rank"].tolist() == [LEVELS[expected_score_decision(p)] for p in proba[tp.P_COLS].to_numpy()]
    # the tokens of the first game, model B's columns: trunk vectors, the centred log p on the right plies, the flags -- build's layout
    vec = tmp_path / "vec"
    games, n_plies = tp.vec_index(vec)
    per_task, _, moves_hsl = ps.load_shards(out_main, 1, len(tasks), sha, with_moves=True)
    l1b_models, l1b_per_task, moves_l1b, opp_per_task = ps.load_shards_l1b(out_main, 1, with_moves=True)
    moves_opp = tp.load_opp_moves(out_main, 1)
    assert set(moves_opp) == set(range(len(tasks))) and all(moves_opp[k][0].shape == (truth[k][2].shape[0], K, 10) for k in moves_opp)
    sc = tp.score_arrays(games, moves_hsl, moves_l1b, moves_opp, K)
    assert sc.shape == (K, n_plies, ts.N_SCORES) and sc.dtype == np.float16
    sc_b = vec / "scores_heads3_B.npy"
    assert sc_b.exists()  # written by predict
    np.testing.assert_array_equal(np.load(sc_b), sc[1])
    X_raw, offsets = ts.raw_tokens([(vec, sc_b)])
    k0 = games[0]["task"]
    hsl, l1b_t, l1b_o, g, l, target, mover = truth[k0]
    s, n = offsets[tp.task_key(k0)]
    rows = X_raw[s:s + n].astype(np.float32)
    np.testing.assert_array_equal(rows[:, :64], g.astype(np.float32)) and np.testing.assert_array_equal(rows[:, 64:128], l.astype(np.float32))
    np.testing.assert_allclose(rows[target, 128:139], ts.centred(l1b_t[:, 1, :].astype(np.float64)), rtol=0, atol=2e-3)
    np.testing.assert_allclose(rows[~target, 128:139], ts.centred(l1b_o[:, 1, :].astype(np.float64)), rtol=0, atol=2e-3)
    np.testing.assert_allclose(rows[target, 139:158], ts.centred(hsl.astype(np.float64)), rtol=0, atol=2e-3)
    assert (rows[~target, 139:158] == 0).all()
    np.testing.assert_array_equal(rows[:, 158], target.astype(np.float32)) and np.testing.assert_array_equal(rows[:, 159], mover.astype(np.float32))
    np.testing.assert_allclose(rows[:, 160], np.arange(n) / n, rtol=0, atol=1e-3)
    # one checkpoint, model B only: the command's probabilities equal the hand computation through the same tokens and R features
    assert tp.main(base + ["--models", str(ck_t[0]), "--allow-missing-hd", "--l1b-names", "heads3_B"]) == tp.EXIT_OK
    model, stats, (r_mean, r_std), _ = ts.load_checkpoint(ck_t[0], "cpu")
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    qis = sorted(by_q)
    R = tp.question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, {k: (sgf, c) for k, sgf, c in tasks}, 1)
    assert R.shape == (len(qis), 64)
    want = ts.predict(model, ((R - r_mean) / r_std).astype(np.float32), [[tp.task_key(k) for k in by_q[qi]] for qi in qis],
                      torch.as_tensor(ts.standardise_tokens(X_raw.copy(), stats)), offsets, "cpu")
    got = pd.read_csv(out.with_name("submission_rank_ct_proba.csv"))[tp.P_COLS].to_numpy()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
    assert json.loads(out.with_name("submission_rank_ct_report.json").read_text())["passes"] == 1
    # a vec part whose plies do not line up with the shards' moves is refused (exit 2), as is a shard without the opponent moves
    csv_path, npz_path = part_paths(vec, 0, 1)
    with np.load(npz_path) as z:
        arrays = dict(z)
    arrays["target"] = ~arrays["target"]
    np.savez(npz_path, **arrays)
    assert tp.main(base + ["--models", str(ck_t[0]), "--allow-missing-hd"]) == tp.EXIT_INPUTS
    with np.load(npz_path) as z:
        arrays = dict(z)
    arrays["target"] = ~arrays["target"]
    np.savez(npz_path, **arrays)
    shard = ps.shard_path(out_main, 0, 1)
    with np.load(shard, allow_pickle=False) as z:
        keep = {k: z[k] for k in z.files if k not in ("l1b_opp_move_logp", "opp_move_task", "opp_move_t")}
    meta = json.loads(str(keep["__meta__"]))
    meta["l1b_opp_moves"] = False
    keep["__meta__"] = np.asarray(json.dumps(meta))
    np.savez(shard, **keep)
    assert tp.main(base + ["--models", str(ck_t[0]), "--allow-missing-hd"]) == tp.EXIT_INPUTS


def test_question_without_a_usable_game_takes_the_fallback_row(tmp_path):
    inputs = _small_inputs(tmp_path, break_question=1)
    out_main = tmp_path / "run" / "submission_rank.csv"
    questions, tasks, owner, truth = _synthetic_run(tmp_path, inputs, out_main)
    ids = [q["question_id"] for q in questions]
    assert 1 not in set(owner) and len(tasks) == len(truth)
    ck = _checkpoint(tmp_path, "C", 3)[0]
    fb = tmp_path / "run" / "main.csv"
    pd.DataFrame({"question_id": ids, "pred_rank": ["2D", "5D", "A", "1D", "B"]}).to_csv(fb, index=False)
    out = tmp_path / "run" / "submission_rank_ct.csv"
    base = ["predict", "--inputs", *inputs, "--shards", str(out_main), "--nshards", "1", "--vec", str(tmp_path / "vec"), "--out", str(out),
            "--expect-rows", "5", "--expect-public", "3", "--expect-private", "2", "--models", str(ck), "--allow-missing-hd"]
    assert tp.main(base + ["--fallback-submission", str(fb)]) == tp.EXIT_FALLBACK
    sub = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert sub["pred_rank"].tolist()[1] == "5D" and sub["question_id"].tolist() == ids
    report = json.loads(out.with_name("submission_rank_ct_report.json").read_text())
    assert report["fallback_questions"] == [ids[1]] and report["exit_code"] == tp.EXIT_FALLBACK and report["vs_fallback_submission"]["questions_compared"] == 4
    proba = pd.read_csv(out.with_name("submission_rank_ct_proba.csv"))
    assert proba[tp.P_COLS].isna().all(axis=1).tolist() == [False, True, False, False, False]
    assert tp.main(base + ["--fallback-submission", str(fb), "--allow-fallback"]) == tp.EXIT_OK
    assert tp.main(base + ["--allow-fallback"]) == tp.EXIT_OK  # without a fallback submission: FALLBACK
    assert pd.read_csv(out, dtype=str)["pred_rank"].tolist()[1] == tp.FALLBACK


def test_compare_reports_differences_and_scores(tmp_path):
    rng = np.random.RandomState(0)
    P = rng.dirichlet(np.ones(10), size=6)
    ids = [f"pub_q_{i:04d}" for i in range(6)]
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame(P, columns=tp.P_COLS).assign(question_id=ids).to_csv(a, index=False)
    Q = P.copy()
    Q[0] = np.roll(Q[0], 3)  # another decision for the first question
    pd.DataFrame(Q, columns=tp.P_COLS).assign(question_id=ids).to_csv(b, index=False)
    ans = tmp_path / "answers.csv"
    pd.DataFrame({"question_id": ids, "rank": [LEVELS[expected_score_decision(p)] for p in P], "n_games": [5, 5, 10, 10, 20, 20]}).to_csv(ans, index=False)
    rep_path = tmp_path / "cmp.json"
    assert tp.main(["compare", "--a", str(a), "--b", str(b), "--answers", str(ans), "--report", str(rep_path)]) == tp.EXIT_INVALID
    rep = json.loads(rep_path.read_text())
    assert rep["compared"] == 6 and rep["decisions_differ"] >= 1 and rep["pass"] is False and rep["score_a"] == 1.0 and rep["score_b"] < 1.0
    assert tp.main(["compare", "--a", str(a), "--b", str(a), "--answers", str(ans)]) == tp.EXIT_OK


def _banded_checkpoint(tmp_path, arm, seed, sizes, name):
    """A checkpoint trained on one question-size band (trunk_seq_check.py --train-sizes): its sidecar records train_sizes."""
    torch.manual_seed(seed)
    model = ts.Model(arm, 64)
    rng = np.random.RandomState(seed)
    stats = (rng.randn(ts.D_TOKEN - 4).astype(np.float32), (0.5 + rng.rand(ts.D_TOKEN - 4)).astype(np.float32))
    path = tmp_path / "ct" / f"{name}.pt"
    ts.save_checkpoint(path, model, stats, (rng.randn(64), 0.5 + rng.rand(64)),
                       {"tag": f"{arm}-s{sizes[0]}-{sizes[1]}", "seed": seed, "val_fold": "all", "train_folds": [1, 2, 3, 4], "train_sizes": list(sizes)})
    return path


def test_predict_routes_checkpoints_by_question_size(tmp_path):
    """Section 4M M7: a checkpoint with train_sizes votes only on questions of that many games; the ensemble is the mean of the votes."""
    inputs = _small_inputs(tmp_path)  # fixture questions of 5, 10, 5 | 5, 5 games
    out_main = tmp_path / "run" / "submission_rank.csv"
    questions, tasks, owner, truth = _synthetic_run(tmp_path, inputs, out_main)
    ids, sizes = [q["question_id"] for q in questions], [len(q["games"]) for q in questions]
    assert sorted(set(sizes)) == [5, 10] and len(tasks) == sum(sizes)
    ck_all = _checkpoint(tmp_path, "C", 0)[0]
    ck_low = _banded_checkpoint(tmp_path, "T", 1, (5, 5), "all_T-s5-5_s1")
    ck_mid = _banded_checkpoint(tmp_path, "C", 2, (10, 10), "all_C-s10-10_s2")
    ck_none = _banded_checkpoint(tmp_path, "T", 3, (15, 20), "all_T-s15-20_s3")
    fb = tmp_path / "run" / "main.csv"
    pd.DataFrame({"question_id": ids, "pred_rank": ["3D"] * len(ids)}).to_csv(fb, index=False)
    out = tmp_path / "run" / "routed.csv"
    base = ["predict", "--inputs", *inputs, "--shards", str(out_main), "--nshards", "1", "--vec", str(tmp_path / "vec"), "--out", str(out),
            "--expect-rows", "5", "--expect-public", "3", "--expect-private", "2", "--fallback-submission", str(fb), "--allow-missing-hd"]

    def run(models):
        assert tp.main(base + ["--allow-fallback", "--models", *map(str, models)]) == tp.EXIT_OK
        return pd.read_csv(out.with_name("routed_proba.csv"))[tp.P_COLS].to_numpy(), json.loads(out.with_name("routed_report.json").read_text())

    P_all, rep_all = run([ck_all])
    P_low, rep_low = run([ck_low])
    P_mid, rep_mid = run([ck_mid])
    assert rep_all["size_routing"] == {"all": 5} and rep_low["size_routing"] == {"5-5": 4} and rep_mid["size_routing"] == {"10-10": 1}
    big = [i for i in range(5) if sizes[i] != 5]
    assert rep_low["fallback_questions"] == [ids[i] for i in big] and np.isnan(P_low[big]).all() and np.isfinite(np.delete(P_low, big, axis=0)).all()
    sub_low = pd.read_csv(out.with_name("routed.csv"), dtype=str, keep_default_na=False)  # the last run was ck_mid: 4 fallback rows
    assert (sub_low["pred_rank"].to_numpy()[[i for i in range(5) if sizes[i] != 10]] == "3D").all()
    P, rep = run([ck_all, ck_low, ck_mid, ck_none])
    assert rep["size_routing"] == {"all": 5, "5-5": 4, "10-10": 1, "15-20": 0} and rep["fallback_questions"] == [] and rep["passes"] == 8
    for i, n in enumerate(sizes):
        np.testing.assert_allclose(P[i], (P_all[i] + (P_low[i] if n == 5 else P_mid[i])) / 2, rtol=0, atol=1e-6)
    assert rep["arm_questions_voted"] == {"C": 5, "T-s5-5": 4, "C-s10-10": 1, "T-s15-20": 0}
    assert rep["arm_agreement_with_ensemble"]["T-s15-20"] is None and rep["arm_agreement_with_ensemble"]["C"] is not None
    assert [c["train_sizes"] for c in rep["checkpoints"]] == [None, [5, 5], [10, 10], [15, 20]]
    # only out-of-band checkpoints: every question takes the fallback row, and without --allow-fallback the exit code says so
    assert tp.main(base + ["--models", str(ck_none)]) == tp.EXIT_FALLBACK
    assert json.loads(out.with_name("routed_report.json").read_text())["fallback_questions"] == ids


def test_predict_restricts_checkpoints_to_their_l1b_heads(tmp_path):
    """Section 4M M9: a checkpoint whose sidecar lists l1b_heads only runs on those shard models; without the key it runs on all."""
    inputs = _small_inputs(tmp_path)
    out_main = tmp_path / "run" / "submission_rank.csv"
    questions, tasks, owner, truth = _synthetic_run(tmp_path, inputs, out_main)
    ids = [q["question_id"] for q in questions]
    ck_all = _checkpoint(tmp_path, "C", 0)[0]
    torch.manual_seed(1)
    model = ts.Model("T", 64)
    rng = np.random.RandomState(1)
    ck_b = tmp_path / "ct" / "all_T-h4_s1.pt"
    ts.save_checkpoint(ck_b, model, (rng.randn(ts.D_TOKEN - 4).astype(np.float32), (0.5 + rng.rand(ts.D_TOKEN - 4)).astype(np.float32)), (rng.randn(64), 0.5 + rng.rand(64)),
                       {"tag": "T-h4", "seed": 1, "val_fold": "all", "train_folds": [1, 2, 3, 4], "l1b_set": 4, "l1b_heads": ["heads3_B"]})
    fb = tmp_path / "run" / "main.csv"
    pd.DataFrame({"question_id": ids, "pred_rank": ["3D"] * len(ids)}).to_csv(fb, index=False)
    out = tmp_path / "run" / "sets.csv"
    base = ["predict", "--inputs", *inputs, "--shards", str(out_main), "--nshards", "1", "--vec", str(tmp_path / "vec"), "--out", str(out),
            "--expect-rows", "5", "--expect-public", "3", "--expect-private", "2", "--fallback-submission", str(fb), "--allow-missing-hd"]

    def run(extra):
        code = tp.main(base + extra)
        return code, pd.read_csv(out.with_name("sets_proba.csv"))[tp.P_COLS].to_numpy(), json.loads(out.with_name("sets_report.json").read_text())

    code, P_b_only, rep = run(["--models", str(ck_b)])
    assert code == tp.EXIT_OK and rep["passes"] == 1 and rep["checkpoints"][0]["l1b_heads"] == ["heads3_B"]  # one shard model accepted
    code, P_b_named, rep = run(["--models", str(ck_b), "--l1b-names", "heads3_B"])
    assert code == tp.EXIT_OK and rep["passes"] == 1
    np.testing.assert_allclose(P_b_only, P_b_named, rtol=0, atol=1e-6)  # the filter equals restricting the shard models by hand
    code, P_all, rep = run(["--models", str(ck_all)])
    assert rep["passes"] == 2  # no l1b_heads in the sidecar: both shard models
    code, P_both, rep = run(["--models", str(ck_all), str(ck_b)])
    assert rep["passes"] == 3
    np.testing.assert_allclose(P_both, (2 * P_all + P_b_only) / 3, rtol=0, atol=1e-6)  # vote-weighted mean over (checkpoint, model) passes
    assert tp.main(base + ["--models", str(ck_b), "--l1b-names", "heads3_A"]) == tp.EXIT_INPUTS  # nothing accepts heads3_A alone
