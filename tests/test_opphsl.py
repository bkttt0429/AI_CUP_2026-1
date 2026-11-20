"""Section 4M M14, the opponent-HSL channel: the question features (aicup.opphsl), the shard arrays of the 18-profile opponent
pass and their loader (predict_submission.py), the token score arrays with the opponent plies' Human SL block (trunk_predict.py)
and trunk_seq_check.py's tags and paths."""
import argparse

import numpy as np
import pandas as pd
import pytest

from aicup import opphsl
from aicup.evidence import posterior_mean_index
from aicup.humansl import PROFILES
from aicup.metric import LEVELS
import predict_submission as ps
import trunk_predict as tp
import trunk_seq_check as ts

P, L = len(PROFILES), len(LEVELS)


def test_question_features_and_the_table_form_agree():
    rng = np.random.default_rng(0)
    G = 6
    own, opp, n = -rng.uniform(100, 400, (G, P)), -rng.uniform(100, 400, (G, P)), rng.integers(80, 160, G)
    f = opphsl.question_features(own, opp, n)
    assert f.shape == (opphsl.N_OPP_HSL,) == (39,) and len(opphsl.OPP_HSL_FEATURE_NAMES) == 39
    C = opp / n[:, None]
    C = C - C.max(axis=1, keepdims=True)
    np.testing.assert_allclose(f[:P], C.mean(axis=0))
    np.testing.assert_allclose(f[P:2 * P], C.std(axis=0))
    pm = posterior_mean_index(opp)
    np.testing.assert_allclose(f[2 * P:], [pm.mean(), pm.std(), posterior_mean_index(own).mean() - pm.mean()])
    assert (f[:P] <= 1e-12).all()  # centred by each game's best profile
    keys = [f"g{i}" for i in range(G)]
    hsl = pd.DataFrame({"game_key": keys, "n_moves": n + 1, **{f"sum_{p}": own[:, j] for j, p in enumerate(PROFILES)}})
    ohsl = pd.DataFrame({"game_key": keys[::-1], "n_moves": n[::-1], **{f"sum_{p}": opp[::-1, j] for j, p in enumerate(PROFILES)}})  # another row order
    X = opphsl.opp_hsl_features(hsl, ohsl, pd.DataFrame({"game_keys": [keys, keys[:3]]}))
    assert X.shape == (2, 39)
    np.testing.assert_allclose(X[0], f)
    np.testing.assert_allclose(X[1], opphsl.question_features(own[:3], opp[:3], n[:3]))
    with pytest.raises(ValueError, match="not in both"):
        opphsl.opp_hsl_features(hsl, ohsl, pd.DataFrame({"game_keys": [["g0", "nope"]]}))
    with pytest.raises(ValueError):
        opphsl.question_features(own, opp[:, :5], n)


def _target_pass(tasks):
    per, lp, mt, tt = {}, [], [], []
    for k, _, _ in tasks:
        n = 6 + k
        curve = np.concatenate([-1.5 - 0.1 * np.arange(P)] + [-0.8 - 0.05 * np.arange(L) - j for j in range(2)])
        per[k] = (n, curve * n)
        lp.append(np.tile(curve, (n, 1)).astype(np.float32))
        mt.append(np.full(n, k, np.int32))
        tt.append(2 * np.arange(n, dtype=np.int32))
    return per, (np.concatenate(lp), np.concatenate(mt), np.concatenate(tt))


def _opp_pass(tasks, with_hsl):
    """A fake opponent pass over two L1b models: per task 5 + k moves, columns [18 +] 20."""
    per, lp, mt, tt = {}, [], [], []
    for k, _, _ in tasks:
        n_o = 5 + k
        curve = np.concatenate(([-2.0 - 0.1 * np.arange(P)] if with_hsl else []) + [-1.0 - 0.05 * np.arange(L) - j for j in range(2)])
        per[k] = (n_o, curve * n_o)
        lp.append(np.tile(curve, (n_o, 1)).astype(np.float32))
        mt.append(np.full(n_o, k, np.int32))
        tt.append(1 + 2 * np.arange(n_o, dtype=np.int32))
    return per, int(sum(v[0] for v in per.values())), 0.5, (np.concatenate(lp), np.concatenate(mt), np.concatenate(tt))


@pytest.mark.parametrize("with_hsl", [True, False])
def test_save_shard_splits_the_opponent_pass_and_the_loaders_read_it(tmp_path, with_hsl):
    tasks = [(k, "(;GM[1])", "B") for k in range(3)]
    per_task, moves = _target_pass(tasks)
    models = [{"name": "A"}, {"name": "B"}]
    out = tmp_path / "submission_rank.csv"
    path = ps.shard_path(out, 0, 1)
    meta = ps.save_shard(path, 0, 1, tasks, per_task, moves, 100, 1.0, ["pub.csv"], "sha", 3, models, _opp_pass(tasks, with_hsl))
    assert meta["hsl_opponent"] is with_hsl and meta["l1b_opponent"] is True and meta["l1b_opp_moves"] is True
    with np.load(path, allow_pickle=False) as z:
        assert z["l1b_opp_n_moves"].tolist() == [5, 6, 7]
        assert z["l1b_opp_sums"].shape == (3, 2, L) and z["l1b_opp_move_logp"].shape == (18, 2, L)
        np.testing.assert_allclose(z["l1b_opp_sums"][1, 1], (-1.0 - 0.05 * np.arange(L) - 1) * 6)
        assert z["sums"].shape == (3, P) and z["l1b_sums"].shape == (3, 2, L)  # the target pass is untouched
        if with_hsl:
            assert z["hsl_opp_sums"].shape == (3, P) and z["hsl_opp_sums"].dtype == np.float64
            assert z["hsl_opp_move_logp"].shape == (18, P) and z["hsl_opp_move_logp"].dtype == np.float32
            np.testing.assert_allclose(z["hsl_opp_sums"][2], (-2.0 - 0.1 * np.arange(P)) * 7)
            np.testing.assert_allclose(z["hsl_opp_move_logp"][z["opp_move_task"] == 2].sum(axis=0), z["hsl_opp_sums"][2], rtol=1e-5)
        else:
            assert "hsl_opp_sums" not in z.files and "hsl_opp_move_logp" not in z.files
    if with_hsl:
        per, mv = ps.load_shards_opp_hsl(out, 1, with_moves=True)
        assert set(per) == {0, 1, 2} and per[0][0] == 5 and per[0][1].shape == (P,)
        assert mv[2][0].shape == (7, P) and mv[2][1].tolist() == (1 + 2 * np.arange(7)).tolist()
        np.testing.assert_allclose(mv[2][0].sum(axis=0), per[2][1], rtol=1e-5)
    else:
        with pytest.raises(ValueError, match="hsl_opponent"):
            ps.load_shards_opp_hsl(out, 1)
    l1b_models, _, _, opp_per_task = ps.load_shards_l1b(out, 1, with_moves=True)  # the L1b side is the same in both layouts
    assert [m["name"] for m in l1b_models] == ["A", "B"] and opp_per_task[2][0] == 7 and opp_per_task[2][1].shape == (2, L)
    np.testing.assert_allclose(tp.load_opp_moves(out, 1)[1][0][:, 0, :], np.tile(-1.0 - 0.05 * np.arange(L), (6, 1)), rtol=1e-6)
    per, pos, wall, (lp, mt, tt) = _opp_pass(tasks, True)
    bad = ({k: (n, sums[1:]) for k, (n, sums) in per.items()}, pos, wall, (lp[:, 1:], mt, tt))  # 37 columns: neither 20 nor 38
    with pytest.raises(ValueError, match="expected"):
        ps.save_shard(tmp_path / "bad_shard0of1.npz", 0, 1, tasks, per_task, moves, 100, 1.0, ["pub.csv"], "sha", 3, models, bad)


def test_score_arrays_fill_the_opponent_hsl_block():
    games = [{"task": 0, "start": 0, "n": 5, "t": np.arange(5), "target": np.array([1, 0, 1, 0, 1], bool)},   # target B
             {"task": 1, "start": 5, "n": 4, "t": np.arange(4), "target": np.array([0, 1, 0, 1], bool)}]      # target W
    K = 2
    rng = np.random.default_rng(1)

    def mv(n, width):
        return -rng.uniform(0.5, 5, (n, width))

    moves_hsl = {0: (mv(3, P), np.array([0, 2, 4])), 1: (mv(2, P), np.array([1, 3]))}
    moves_l1b = {0: (mv(3, K * L).reshape(3, K, L), np.array([0, 2, 4])), 1: (mv(2, K * L).reshape(2, K, L), np.array([1, 3]))}
    moves_opp = {0: (mv(2, K * L).reshape(2, K, L), np.array([1, 3])), 1: (mv(2, K * L).reshape(2, K, L), np.array([0, 2]))}
    moves_oh = {0: (mv(2, P), np.array([1, 3])), 1: (mv(2, P), np.array([0, 2]))}
    plain = tp.score_arrays(games, moves_hsl, moves_l1b, moves_opp, K)
    filled = tp.score_arrays(games, moves_hsl, moves_l1b, moves_opp, K, moves_opp_hsl=moves_oh)
    assert plain.shape == filled.shape == (K, 9, ts.N_SCORES) and plain.dtype == filled.dtype == np.float16
    tgt_rows, opp_rows = [0, 2, 4, 6, 8], [1, 3, 5, 7]
    np.testing.assert_array_equal(plain[:, opp_rows, L + 1:], 0)
    np.testing.assert_array_equal(plain[:, tgt_rows, :], filled[:, tgt_rows, :])
    np.testing.assert_array_equal(plain[:, opp_rows, :L + 1], filled[:, opp_rows, :L + 1])
    want = np.vstack([ts.centred(moves_oh[0][0]), ts.centred(moves_oh[1][0])]).astype(np.float16)
    np.testing.assert_array_equal(filled[0, opp_rows, L + 1:], want)
    np.testing.assert_array_equal(filled[1, opp_rows, L + 1:], want)
    with pytest.raises(ValueError, match="line up"):
        tp.score_arrays(games, moves_hsl, moves_l1b, moves_opp, K, moves_opp_hsl={**moves_oh, 1: (moves_oh[1][0], np.array([1, 3]))})


def test_question_r_appends_the_opponent_hsl_features():
    rng = np.random.default_rng(2)
    ks = [0, 1, 2, 3, 4]
    per_task = {k: (100 + k, -rng.uniform(100, 300, P)) for k in ks}
    l1b_per_task = {k: -rng.uniform(100, 300, (2, L)) for k in ks}
    opp_per_task = {k: (101 + k, -rng.uniform(100, 300, (2, L))) for k in ks}
    opp_hsl_per_task = {k: (101 + k, -rng.uniform(100, 300, P)) for k in ks}
    sgf = "(;GM[1]FF[4]SZ[19]HA[2]AB[dd][pp];W[dp];B[pd])"
    task_games = {k: (sgf, "B") for k in ks}
    plain = tp.question_r([0], {0: ks}, per_task, l1b_per_task, opp_per_task, task_games, 1)
    full = tp.question_r([0], {0: ks}, per_task, l1b_per_task, opp_per_task, task_games, 1, opp_hsl_per_task)
    assert plain.shape == (1, 64) and full.shape == (1, 64 + opphsl.N_OPP_HSL)
    np.testing.assert_array_equal(full[:, :64], plain)
    np.testing.assert_allclose(full[0, 64:], opphsl.question_features(np.stack([per_task[k][1] for k in ks]), np.stack([opp_hsl_per_task[k][1] for k in ks]),
                                                                     np.array([opp_hsl_per_task[k][0] for k in ks])))


def test_trunk_seq_check_opp_hsl_tags_and_paths():
    a = argparse.Namespace(extra_games=40, opp_hsl=True, opp_hsl_r=False)
    assert ts.opp_hsl_level(a) == 1 and ts.data_segments(a) == ["x40", "oh"]
    a.opp_hsl_r = True
    assert ts.opp_hsl_level(a) == 2 and ts.data_segments(a) == ["x40", "ohr"]
    assert ts.opp_hsl_level(argparse.Namespace()) == 0 and ts.data_segments(argparse.Namespace(extra_games=40)) == ["x40"]
    assert ts.scores_path("c", 1).name == "scores_fold1.npy" and ts.scores_path("c", 1, opp_hsl=True).name == "scores_oh_fold1.npy"
    assert ts.scores_path("c", 2, ts.EXTRAS[40]["vec"], opp_hsl=True).name == "scores_x40_oh_fold2.npy"
    assert ts.opp_hsl_dir(0) == ts.OPP_HSL and ts.opp_hsl_dir(40) == ts.OPP_HSL_X[40]
    with pytest.raises(SystemExit):
        ts.opp_hsl_dir(80)
