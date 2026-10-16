import numpy as np
import pandas as pd
import pytest
import torch
import webdataset as wds

from aicup.baseline import TUTORIAL_DIR, game_features, load_tutorial, split_players
from aicup.metric import RANK_TO_IDX
from baseline_extract import extract_split

needs_tutorial = pytest.mark.skipif(not (TUTORIAL_DIR / "utils.py").exists(), reason="Tutorial checkout missing")
# Earlier CPU forward passes leave torch threads in this test process; the forked workers only
# run numpy/sgfmill code. The extraction script itself forks before torch is ever imported.
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

# D and 6D: 8 players each in folds 1-4 with 1..8 games (36 per rank), plus one fold-0 player each
FOLDS = pd.DataFrame(
    [("D", f"d{i}", i + 1, 1 + i % 4) for i in range(8)]
    + [("6D", f"s{i}", i + 1, 1 + i % 4) for i in range(8)]
    + [("D", "d_fold0", 50, 0), ("6D", "s_fold0", 50, 0)],
    columns=["rank", "player_id", "n_games", "fold"])

GAME = "(;SZ[19];B[pd];W[dp];B[];W[dd])"   # root + 4 moves (one a pass) = 5 board states
G1 = "(;SZ[19];B[pd];W[dp];B[pq];W[dd])"
G2 = "(;SZ[19];B[qd];W[dc];B[];W[pp];B[cq])"
DF5 = pd.DataFrame({"player_id": ["p1", "p1", "p2", "p2", "p3"],
                    "game_id": ["g_1", "g_2", "g_1", "g_3", "g_4"],
                    "rank": ["D", "D", "6D", "6D", "A"],
                    "sgf_content": [G1, G2, G2, G1, "not an sgf"]})
SGF_OF = {"D_g_1": G1, "D_g_2": G2, "6D_g_1": G2, "6D_g_3": G1}
DF6 = pd.DataFrame({"player_id": [f"p{i}" for i in range(6)], "game_id": [f"g_{i}" for i in range(6)],
                    "rank": ["D", "C", "B", "A", "1D", "6D"], "sgf_content": [G1, G2] * 3})


def test_split_players_excludes_other_folds_and_is_player_disjoint():
    train, val = split_players(FOLDS, val_games_per_rank=10)
    pool = set(FOLDS.loc[FOLDS["fold"].isin([1, 2, 3, 4]), "player_id"])
    assert set(train) | set(val) == pool and not set(train) & set(val)
    assert train == sorted(train) and val == sorted(val)


def test_split_players_takes_just_enough_players_per_rank():
    train, val = split_players(FOLDS, val_games_per_rank=10)
    g, r = FOLDS.set_index("player_id")["n_games"], FOLDS.set_index("player_id")["rank"]
    for rank in ("D", "6D"):
        v = [p for p in val if r[p] == rank]
        assert g[v].sum() >= 10 and g[v].sum() - g[v].max() < 10
        assert any(r[p] == rank for p in train)


def test_split_players_is_deterministic_and_ignores_row_order():
    assert split_players(FOLDS.sample(frac=1.0, random_state=3), val_games_per_rank=10) == \
        split_players(FOLDS, val_games_per_rank=10)


def test_split_players_refuses_to_take_a_whole_rank():
    with pytest.raises(ValueError):
        split_players(FOLDS, val_games_per_rank=36)


@needs_tutorial
def test_game_features_is_the_official_parser_output():
    f = game_features(GAME)
    ref, err = load_tutorial()[0]().features_from_sgf_content(GAME)
    assert err is None and np.array_equal(f, ref) and f.shape == (500, 19, 19) and f.dtype == np.uint8
    # 245 padding boards + the empty root, then B pd / W dp / B pass / W dd as (black, white) planes
    assert f[:492].sum() == 0 and f[492].sum() == 1 and f[498].sum() == 1 and f[499].sum() == 2
    assert np.array_equal(f[494:496], f[496:498])   # the pass leaves the board unchanged
    assert game_features("not an sgf") is None


@needs_tutorial
def test_tutorial_network_outputs_ten_logits():
    net = load_tutorial()[1](num_classes=10).eval()
    with torch.no_grad():
        assert tuple(net(torch.zeros(2, 500, 19, 19)).shape) == (2, 10)


@needs_tutorial
def test_extract_split_round_trip(tmp_path):
    stats = extract_split(DF5, tmp_path, "val", nproc=2, per_shard=1, min_shard_bytes=0)
    assert (stats["written"], stats["bad"], stats["shards"]) == (4, 1, 4)
    samples = list(wds.WebDataset(sorted(str(p) for p in (tmp_path / "val").glob("*.tar.gz")), shardshuffle=0).decode())
    assert sorted(s["__key__"] for s in samples) == ["6D_g_1", "6D_g_3", "D_g_1", "D_g_2"]
    for s in samples:
        assert s["rank_idx.txt"] == str(RANK_TO_IDX[s["label.txt"]])
        assert np.array_equal(s["features.npy"], game_features(SGF_OF[s["__key__"]]))


@needs_tutorial
def test_extract_split_refuses_old_shards_and_flags_empty_ones(tmp_path):
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "old.tar.gz").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        extract_split(DF5, tmp_path, "val", nproc=1, per_shard=1)
    with pytest.raises(SystemExit):
        extract_split(DF5, tmp_path, "train", nproc=1, per_shard=1, min_shard_bytes=10**9)


@needs_tutorial
def test_make_loader_reads_every_sample_once_and_pick_workers_falls_back(tmp_path):
    from baseline_train import make_loader, pick_workers

    extract_split(DF6, tmp_path, "val", nproc=2, per_shard=1, min_shard_bytes=0)   # 6 shards
    shards = sorted(str(p) for p in (tmp_path / "val").glob("*.tar.gz"))
    assert sum(len(y) for _x, y, _n in make_loader(shards, 2, False, batch_size=4)) == 6
    assert pick_workers(shards, expected=6, batch_size=4, candidate=2) == 2
    assert pick_workers(shards, expected=7, batch_size=4, candidate=2) == 0
    x, y, _ = next(iter(make_loader(shards, 0, True, batch_size=4)))
    assert x.dtype == torch.float32 and tuple(x.shape) == (4, 500, 19, 19) and y.dtype == torch.long


def test_mean_probs_averages_probabilities_and_marks_empty_questions():
    from aicup.baseline import mean_probs

    a, b = np.eye(10)[0] * 0.8 + np.eye(10)[1] * 0.2, np.eye(10)[0] * 0.2 + np.eye(10)[1] * 0.8
    P = mean_probs({"a": a, "b": b}, [["a", "b"], ["missing"], ["a", "missing"]])
    assert np.allclose(P[0], (a + b) / 2) and np.isnan(P[1]).all() and np.allclose(P[2], a)


def test_decide_rules_differ_on_neighbour_mass_and_fall_back_to_class_0():
    from aicup.baseline import decide

    P = np.array([[0.30, 0.30, 0.40] + [0.0] * 7, [np.nan] * 10])
    assert decide(P, "argmax").tolist() == [2, 0]
    # index 1: 0.30 + e^-1 (0.30 + 0.40) = 0.557 beats index 2: 0.40 + e^-1 * 0.30 = 0.510
    assert decide(P, "expected_score").tolist() == [1, 0]


def test_report_separates_class_balanced_from_question_mean():
    from aicup.baseline import report

    qs = pd.DataFrame({"player_id": ["p1", "p2", "p3", "p4"], "n_games": [5, 5, 5, 5]})
    r = report(np.array([0, 0, 0, 3]), np.array([0, 0, 0, 1]), qs, sizes=(5,))
    assert r["question_mean"] == pytest.approx(0.75) and r["balanced_by_n"][5] == pytest.approx(0.5)
    assert r["balanced_mean_over_n"] == pytest.approx(0.5) and r["by_n"][5]["n_questions"] == 4
    # every resample keeps three class-0 players scoring 1 and one class-1 player scoring 0
    assert r["by_n"][5]["balanced_ci95"] == pytest.approx([0.5, 0.5]) and r["balanced_by_n_ci95"][5] == pytest.approx([0.5, 0.5])
    assert r["by_n"][5]["question_mean_ci95"] == pytest.approx([0.75, 0.75])
    perfect = report(np.array([0, 1]), np.array([0, 1]), qs.iloc[:2], sizes=(5,))
    assert perfect["by_n"][5]["balanced_ci95"] == [1.0, 1.0] and perfect["exact"] == 1.0


def test_report_headline_ci_is_the_mean_over_sizes_statistic():
    from aicup.baseline import report

    # one player per class: every resample is the sample itself, so each CI collapses onto its statistic
    qs = pd.DataFrame({"player_id": ["p1", "p1", "p1", "p2", "p2"], "n_games": [5, 5, 10, 5, 10]})
    r = report(np.array([0, 0, 5, 5, 1]), np.array([0, 0, 0, 1, 1]), qs, sizes=(5, 10))
    # class means are (1, 0) at n=5 and (0, 1) at n=10, so 0.5 per size; pooling sizes would give (2/3 + 1/2) / 2
    assert r["balanced_mean_over_n"] == pytest.approx(0.5)
    assert r["balanced_mean_over_n_ci95"] == pytest.approx([0.5, 0.5])
