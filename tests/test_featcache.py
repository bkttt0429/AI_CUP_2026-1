"""aicup.featcache: bit-packed position cache -- pack/unpack, atomic part files, resume, concatenated reads."""
import numpy as np
import pandas as pd
import pytest

from aicup import featcache as fc
from aicup.metric import RANK_TO_IDX


def _fake_pos(n, rng, color=0):
    """What pilot.game_positions returns for a game with n target moves (random binary planes)."""
    return {"bins": rng.randint(0, 2, size=(n, 22, 19, 19)).astype(np.uint8),
            "globs": rng.randn(n, 19).astype(np.float32),
            "legal": rng.rand(n, 362) < 0.8,
            "target": rng.randint(0, 362, size=n).astype(np.int16),
            "t": np.arange(n, dtype=np.int16) * 2 + color,
            "color": np.full(n, color, dtype=np.uint8)}


def _games(rows):
    return pd.DataFrame(rows, columns=["game_key", "player_id", "rank", "color", "n_moves"])


def test_pack_unpack_round_trip():
    rng = np.random.RandomState(0)
    pos = _fake_pos(9, rng, color=1)
    assert (fc.N_PLANES, fc.N_GLOBAL, fc.N_MOVES, fc.BIN_BYTES, fc.LEGAL_BYTES) == (22, 19, 362, 993, 46)
    packed = fc.pack_positions(pos)
    assert packed["bins_packed"].shape == (9, 993) and packed["bins_packed"].dtype == np.uint8
    assert packed["legal_packed"].shape == (9, 46) and packed["legal_packed"].dtype == np.uint8
    assert packed["globs"].dtype == np.float32 and packed["target"].dtype == np.int16 and packed["t"].dtype == np.int16
    assert packed["color"].tolist() == [1] * 9
    np.testing.assert_array_equal(fc.unpack_bins(packed["bins_packed"]), pos["bins"])
    np.testing.assert_array_equal(fc.unpack_legal(packed["legal_packed"]), pos["legal"])
    empty = fc.pack_positions(_fake_pos(0, rng))
    assert empty["bins_packed"].shape == (0, 993) and fc.unpack_bins(empty["bins_packed"]).shape == (0, 22, 19, 19)


def test_write_read_part_aligns_games_and_positions(tmp_path):
    rng = np.random.RandomState(1)
    pos = [_fake_pos(5, rng, 0), _fake_pos(0, rng, 1), _fake_pos(7, rng, 1)]
    games = _games([("1D:g1", "p1", "1D", "B", 5), ("3D:g2", "p2", "3D", "W", 0), ("D:g3", "p3", "D", "W", 7)])
    fc.write_part(tmp_path, 0, 0, games, [fc.pack_positions(p) for p in pos])
    npz, csv = fc.part_paths(tmp_path, 0, 0)
    assert npz.name == "cache_s0_p000.npz" and csv.name == "games_s0_p000.csv" and npz.exists() and csv.exists()
    assert not list(tmp_path.glob("*.tmp"))
    got_games, arrays = fc.read_cache(tmp_path)
    assert got_games["game_key"].tolist() == ["1D:g1", "3D:g2", "D:g3"] and got_games["n_moves"].tolist() == [5, 0, 7]
    assert len(arrays["target"]) == 12 and arrays["game_row"].tolist() == [0] * 5 + [2] * 7
    assert arrays["rank_idx"].dtype == np.int8 and arrays["rank_idx"].tolist() == [RANK_TO_IDX["1D"]] * 5 + [RANK_TO_IDX["D"]] * 7
    assert arrays["color"].tolist() == [0] * 5 + [1] * 7
    np.testing.assert_array_equal(fc.unpack_bins(arrays["bins_packed"][5:]), pos[2]["bins"])
    np.testing.assert_array_equal(arrays["globs"][:5], pos[0]["globs"])
    np.testing.assert_array_equal(arrays["t"][5:], pos[2]["t"])


def test_game_without_target_moves(tmp_path):
    rng = np.random.RandomState(2)
    games = _games([("A:g9", "p9", "A", "W", 0)])
    fc.write_part(tmp_path, 0, 0, games, [fc.pack_positions(_fake_pos(0, rng, 1))])
    got_games, arrays = fc.read_cache(tmp_path)
    assert got_games["n_moves"].tolist() == [0] and len(arrays["target"]) == 0
    assert arrays["bins_packed"].shape == (0, 993) and arrays["game_row"].shape == (0,) and arrays["rank_idx"].shape == (0,)
    with pytest.raises(ValueError, match="n_moves"):  # the table must agree with the arrays
        fc.write_part(tmp_path, 0, 1, _games([("A:g8", "p8", "A", "B", 3)]), [fc.pack_positions(_fake_pos(2, rng))])


def test_resume_ignores_tmp_parts(tmp_path):
    rng = np.random.RandomState(3)
    keys = [f"D:g{i}" for i in range(7)]
    games = _games([(k, "p", "D", "B", 1) for k in keys[:3]])
    fc.write_part(tmp_path, 0, 0, games, [fc.pack_positions(_fake_pos(1, rng)) for _ in keys[:3]])
    (tmp_path / "cache_s0_p001.npz.tmp").write_bytes(b"partial")
    (tmp_path / "games_s0_p002.csv").write_text("game_key\n")  # csv without its npz: not complete either
    assert fc.completed_parts(tmp_path) == [(0, 0)]
    assert fc.plan_parts(keys, 3, tmp_path, 0) == [(1, keys[3:6]), (2, keys[6:])]
    assert fc.plan_parts(keys, 3, tmp_path, 1) == [(0, keys[:3]), (1, keys[3:6]), (2, keys[6:])]
    got_games, arrays = fc.read_cache(tmp_path)
    assert got_games["game_key"].tolist() == keys[:3] and len(arrays["target"]) == 3


def test_read_cache_concatenates_parts_with_global_rows(tmp_path):
    rng = np.random.RandomState(4)
    fc.write_part(tmp_path, 1, 0, _games([("D:a", "p1", "D", "B", 2)]), [fc.pack_positions(_fake_pos(2, rng))])
    fc.write_part(tmp_path, 0, 0, _games([("C:b", "p2", "C", "W", 1), ("B:c", "p3", "B", "B", 3)]),
                  [fc.pack_positions(_fake_pos(1, rng, 1)), fc.pack_positions(_fake_pos(3, rng))])
    fc.write_part(tmp_path, 0, 1, _games([("A:d", "p4", "A", "B", 1)]), [fc.pack_positions(_fake_pos(1, rng))])
    assert fc.completed_parts(tmp_path) == [(0, 0), (0, 1), (1, 0)]
    games, arrays = fc.read_cache(tmp_path)
    assert games["game_key"].tolist() == ["C:b", "B:c", "A:d", "D:a"]
    assert arrays["game_row"].tolist() == [0] + [1] * 3 + [2] + [3] * 2
    assert arrays["rank_idx"].tolist() == [RANK_TO_IDX[r] for r in ["C"] + ["B"] * 3 + ["A"] + ["D"] * 2]
    assert len(arrays["bins_packed"]) == 7 and arrays["globs"].shape == (7, 19)
    with pytest.raises(FileNotFoundError):
        fc.read_cache(tmp_path / "nothing")
