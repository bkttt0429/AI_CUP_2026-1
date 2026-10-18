import numpy as np
import pandas as pd
import pytest

from aicup.gamevec import SEG_DIM, SEGMENTS, WHOLE, fold_vectors, game_vector, vec_names
from aicup.humansl import PROFILES
from aicup.qfeatures import SUM_COLS
from aicup.scoring import part_paths, read_scored_games
from ladder_build_vectors import build_fold


def _seg_slice(i):
    return slice(i * SEG_DIM, (i + 1) * SEG_DIM)


def _write_part(out_dir, keys, shard=0, k=0, seed=0, shuffle_rows=False):
    """One scored part in the score_folds.py format: games CSV + per-move npz."""
    rng = np.random.RandomState(seed)
    recs, logps, rows, ts = [], [], [], []
    for i, key in enumerate(keys):
        n = int(rng.randint(3, 130))
        lp = (-rng.rand(n, 18) * 5).astype(np.float32)
        recs.append({"game_key": key, "player_id": "aaaa0000aaaa0000", "rank": key.split(":")[0], "color": "B",
                     "n_moves": n, **{c: float(s) for c, s in zip(SUM_COLS, lp.sum(axis=0))}})
        logps.append(lp)
        rows.append(np.full(n, i, dtype=np.int32))
        ts.append((2 * np.arange(n)).astype(np.int16))  # the target colour's moves: every other move of the game
    logp, row, t = np.concatenate(logps), np.concatenate(rows), np.concatenate(ts)
    if shuffle_rows:
        p = np.random.RandomState(seed + 1).permutation(len(row))
        logp, row, t = logp[p], row[p], t[p]
    csv_path, npz_path = part_paths(out_dir, shard, k)
    pd.DataFrame(recs).to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, logp=logp, game_row=row, t=t)
    return recs


def test_segment_boundaries():
    t = np.array([0, 39, 40, 99, 100, 179, 180, 250])
    logp = -np.random.RandomState(0).rand(len(t), 18).astype(np.float32)
    v = game_vector(logp, t)
    assert v.dtype == np.float32 and v.shape == (SEG_DIM * len(SEGMENTS),) == (300,)
    names = vec_names(SEGMENTS)
    for s in range(1, 5):
        assert v[names.index(f"s{s}_n_eff")] == 2
        assert v[names.index(f"s{s}_present")] == 1
    # the first segment holds exactly moves 1 and 40 (t = 0, 39)
    d = logp[:2].astype(np.float64) - logp[:2].astype(np.float64).mean(axis=1, keepdims=True)
    np.testing.assert_allclose(v[:18], d.mean(axis=0), atol=1e-6)


def test_relative_scores_and_surprisal():
    rng = np.random.RandomState(1)
    t = np.array([0, 5, 38, 40, 70, 120, 150, 179, 200, 260, 300])
    logp = (-rng.rand(len(t), 18) * 4).astype(np.float32)
    c = 3.7
    a, b = game_vector(logp, t), game_vector(logp + np.float32(c), t)
    names = vec_names(SEGMENTS)
    surprisal = [i for i, n in enumerate(names) if n.endswith("_mean_surprisal")]
    rest = [i for i in range(len(names)) if i not in surprisal]
    np.testing.assert_allclose(b[rest], a[rest], atol=1e-5)
    np.testing.assert_allclose(b[surprisal], a[surprisal] - c, atol=1e-5)
    for s in range(4):
        assert abs(a[_seg_slice(s)][:18].sum()) < 1e-5  # d_mean sums to zero over the 18 profiles


def test_empty_segments_are_zero_and_flagged():
    t = np.array([0, 3, 10, 20, 39])
    logp = -np.random.RandomState(2).rand(5, 18).astype(np.float32)
    v = game_vector(logp, t)
    assert np.all(np.isfinite(v))
    names = vec_names(SEGMENTS)
    assert v[names.index("s1_present")] == 1 and v[names.index("s1_n_eff")] == 5
    for s in (1, 2, 3):
        assert np.all(v[_seg_slice(s)] == 0)
        assert v[names.index(f"s{s + 1}_present")] == 0
    # a segment with a single move has zero spread
    one = game_vector(logp[:2], np.array([0, 50]))
    std = [i for i, n in enumerate(names) if "_std_" in n]
    assert np.all(one[std[:18]] == 0) and np.all(one[std[18:36]] == 0)
    assert one[names.index("s2_n_eff")] == 1


def test_whole_matches_single_segment():
    rng = np.random.RandomState(3)
    n = 57
    logp = (-rng.rand(n, 18) * 6).astype(np.float32)
    t = 2 * np.arange(n)
    v = game_vector(logp, t, WHOLE)
    assert v.shape == (SEG_DIM,)
    ll = logp.astype(np.float64)
    d = ll - ll.mean(axis=1, keepdims=True)
    want = np.concatenate([d.mean(axis=0), d.std(axis=0, ddof=0),
                           np.quantile(d, 0.1, axis=0, method="linear"), np.quantile(d, 0.9, axis=0, method="linear"),
                           [n, -ll.mean(), 1.0]])
    np.testing.assert_allclose(v, want, rtol=1e-5, atol=1e-6)
    assert vec_names(WHOLE)[0] == f"w_mean_{PROFILES[0]}" and vec_names(WHOLE)[-1] == "w_present"
    assert len(vec_names(WHOLE)) == SEG_DIM


def test_vec_names_layout():
    names = vec_names(SEGMENTS)
    assert len(names) == len(set(names)) == 300
    assert names[0] == "s1_mean_12k" and names[17] == "s1_mean_6d" and names[18] == "s1_std_12k"
    assert names[36] == "s1_q10_12k" and names[54] == "s1_q90_12k"
    assert names[72:75] == ["s1_n_eff", "s1_mean_surprisal", "s1_present"]
    assert names[-1] == "s4_present"


def test_fold_vectors_checks_csv_sums(tmp_path):
    recs = _write_part(tmp_path, ["1D:a", "1D:b"])
    vec = fold_vectors(tmp_path)
    assert vec["game_key"].dtype.kind == "U" and list(vec["game_key"]) == ["1D:a", "1D:b"]
    assert vec["staged"].dtype == np.float32 and vec["staged"].shape == (2, 300)
    assert vec["whole"].dtype == np.float32 and vec["whole"].shape == (2, 75)
    assert vec["sums"].dtype == np.float64 and vec["n_moves"].dtype == np.int64
    assert list(vec["n_moves"]) == [r["n_moves"] for r in recs]
    # sums are the CSV values, bit for bit what build_matrix reads
    assert np.array_equal(vec["sums"], read_scored_games(tmp_path)[SUM_COLS].to_numpy(dtype=float))

    csv_path = part_paths(tmp_path, 0, 0)[0]
    df = pd.read_csv(csv_path)
    df.loc[1, "sum_3k"] += 0.005  # inside the float32 summation tolerance
    df.to_csv(csv_path, index=False)
    fold_vectors(tmp_path)
    df.loc[1, "sum_3k"] += 1.0
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="sum_3k"):
        fold_vectors(tmp_path)


def test_fold_vectors_rejects_wrong_move_count_and_nonfinite(tmp_path):
    _write_part(tmp_path, ["1D:a", "1D:b"])
    csv_path = part_paths(tmp_path, 0, 0)[0]
    df = pd.read_csv(csv_path)
    df.loc[0, "n_moves"] += 1
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="n_moves"):
        fold_vectors(tmp_path)
    _write_part(tmp_path, ["1D:a", "1D:b"])
    npz_path = part_paths(tmp_path, 0, 0)[1]
    moves = dict(np.load(npz_path))
    moves["logp"][0, 0] = -np.inf
    np.savez_compressed(npz_path, **moves)
    with pytest.raises(ValueError, match="finite"):
        fold_vectors(tmp_path)


def test_fold_vectors_concatenates_parts_and_ignores_row_order(tmp_path):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _write_part(ref_dir, ["1D:a", "1D:b", "1D:c"], seed=5)
    shuf_dir = tmp_path / "shuf"
    shuf_dir.mkdir()
    _write_part(shuf_dir, ["1D:a", "1D:b", "1D:c"], seed=5, shuffle_rows=True)
    ref, shuf = fold_vectors(ref_dir), fold_vectors(shuf_dir)
    for key in ("staged", "whole", "sums", "n_moves"):
        np.testing.assert_allclose(shuf[key], ref[key], atol=1e-5)
    # two parts of two shards come back in completed-part order, one game per key
    two = tmp_path / "two"
    two.mkdir()
    _write_part(two, ["2D:x", "2D:y"], shard=1, k=0, seed=6)
    _write_part(two, ["2D:z"], shard=0, k=1, seed=7)
    _write_part(two, ["2D:w"], shard=0, k=0, seed=8)
    csv_only = part_paths(two, 0, 2)[0]
    csv_only.write_text("game_key\n")  # a part without its npz is not finished and is skipped
    assert list(fold_vectors(two)["game_key"]) == ["2D:w", "2D:z", "2D:x", "2D:y"]
    _write_part(two, ["2D:x"], shard=1, k=1, seed=9)  # the same game scored twice
    with pytest.raises(ValueError, match="more than once"):
        fold_vectors(two)


def test_vectors_roundtrip_without_pickle(tmp_path):
    _write_part(tmp_path, ["1D:g_0045777", "6D:g_1"])
    vec = fold_vectors(tmp_path)
    path = tmp_path / "fold0.npz"
    np.savez_compressed(path, **vec)
    back = dict(np.load(path, allow_pickle=False))
    assert back["game_key"].dtype.kind == "U"
    assert list(back["game_key"]) == list(vec["game_key"])
    for key in ("staged", "whole", "sums", "n_moves"):
        assert np.array_equal(back[key], vec[key])


def _selection_frames(keys):
    games = pd.DataFrame({"player_id": "aaaa0000aaaa0000", "rank": "1D", "game_key": keys})
    folds = pd.DataFrame([("1D", "aaaa0000aaaa0000", len(keys), 0)], columns=["rank", "player_id", "n_games", "fold"])
    return games, folds


def test_build_fold_writes_npz_and_checks_selection(tmp_path, capsys):
    keys = [f"1D:g{i}" for i in range(5)]
    games, folds = _selection_frames(keys)
    scores = tmp_path / "scores"
    (scores / "fold0").mkdir(parents=True)
    _write_part(scores / "fold0", keys)
    out = tmp_path / "vectors"
    build_fold(0, games, folds, scores, out)
    assert capsys.readouterr().out.strip() == "fold 0: 5 games, staged 300 / whole 75"
    back = dict(np.load(out / "fold0.npz", allow_pickle=False))
    assert sorted(back["game_key"]) == keys and back["staged"].shape == (5, 300)
    assert sorted(p.name for p in out.iterdir()) == ["fold0.npz"]

    short = tmp_path / "short"
    (short / "fold0").mkdir(parents=True)
    _write_part(short / "fold0", keys[:4])
    with pytest.raises(ValueError, match=r"fold 0: 4 games scored but 5 selected \(missing 1"):
        build_fold(0, games, folds, short, tmp_path / "vectors_short")
    assert not (tmp_path / "vectors_short" / "fold0.npz").exists()
