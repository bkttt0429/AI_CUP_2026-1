import json

import numpy as np
import pandas as pd
import pytest

from aicup.replay import (MAX_BOARDS, build_payload, check_board_count, class_bands, evidence_series, load_move_logp,
                          select_games)


def test_evidence_series_tracks_the_cumulative_best_profile():
    logp = np.full((4, 18), -3.0)
    logp[:, 5] = -1.0                       # profile 5 explains every move best
    logp[0, 9] = -0.5                       # but the first move alone prefers profile 9
    ev = evidence_series(logp)
    assert ev["best"] == [9, 5, 5, 5]
    assert ev["pmi"][0] > 7.5 and abs(ev["pmi"][-1] - 5.0) < 0.05
    assert len(ev["pmi"]) == 4


def test_class_bands_cover_the_middle_80_percent():
    counts = {"D": [0] * 4 + [1, 3, 5, 3, 1] + [0] * 9, "6D": [0] * 13 + [20] + [0] * 4}
    bands = class_bands(counts)
    assert bands["D"] == [5, 7]
    assert bands["6D"] == [13, 13]


def test_select_games_one_game_per_player_k_players_per_rank():
    games = pd.DataFrame({"game_key": [f"{r}:{p}:{g}" for r in ("A", "6D") for p in range(4) for g in range(3)],
                          "player_id": [f"{r}{p}" for r in ("A", "6D") for p in range(4) for g in range(3)],
                          "rank": [r for r in ("A", "6D") for p in range(4) for g in range(3)]})
    sel = select_games(games, per_rank=2, seed=1)
    assert sel["rank"].value_counts().to_dict() == {"A": 2, "6D": 2}
    assert sel["player_id"].is_unique
    assert sel["game_key"].tolist() == select_games(games, per_rank=2, seed=1)["game_key"].tolist()
    assert list(sel["rank"]) == ["A", "A", "6D", "6D"]      # competition order, weak to strong


def test_check_board_count_caps_a_page_at_100():
    assert MAX_BOARDS == 100
    check_board_count(100)
    with pytest.raises(ValueError):
        check_board_count(101)
    check_board_count(150, force=True)


def _write_shard(d, csv_name, npz_name, keys, moves_per_game):
    pd.DataFrame({"game_key": keys}).to_csv(d / csv_name, index=False)
    rows, logp, ts = [], [], []
    for r, n in enumerate(moves_per_game):
        rows += [r] * n
        ts += list(range(0, 2 * n, 2))
        logp.append(np.full((n, 18), -float(r + 1)))
    np.savez(d / npz_name, logp=np.vstack(logp), game_row=np.array(rows), t=np.array(ts))


def test_load_move_logp_reads_pilot_and_scores_layouts(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    _write_shard(pilot, "games_fold0_shard0.csv", "moves_fold0_shard0.npz", ["a", "b"], [3, 2])
    got = load_move_logp(pilot, fold=0, layout="pilot")
    assert sorted(got) == ["a", "b"] and got["a"]["logp"].shape == (3, 18) and got["b"]["t"].tolist() == [0, 2]
    scores = tmp_path / "scores" / "fold1"
    scores.mkdir(parents=True)
    _write_shard(scores, "games_s1_p004.csv", "moves_s1_p004.npz", ["c"], [4])
    got = load_move_logp(tmp_path / "scores", fold=1, layout="scores")
    assert list(got) == ["c"] and float(got["c"]["logp"][0, 0]) == -1.0


def test_build_payload_is_json_and_aligns_evidence_with_moves():
    rows = pd.DataFrame({"game_key": ["A:g1"], "player_id": ["abcdef0123456789"], "rank": ["A"]})
    sgf = {"A:g1": ("(;SZ[19];B[pd];W[dp];B[pq];W[dd])", "W")}
    logp = {"A:g1": {"logp": np.full((2, 18), -2.0), "t": np.array([1, 3])}}
    payload = build_payload(rows, sgf, logp, {"A": [8, 10]}, title="t", source="test")
    json.dumps(payload)
    g = payload["games"][0]
    assert g["color"] == "W" and g["rank"] == "A" and g["band"] == [8, 10]
    assert g["moves"] == [[15, 3], [3, 15], [15, 16], [3, 3]] and g["stones"] == [] and g["first"] == "B" and g["handicap"] is None
    assert g["evidence"]["t"] == [1, 3] and len(g["evidence"]["pmi"]) == 2
    # a handicap record (dataset v1.1): the stones are listed, the moves keep their coordinates and White moves first
    h = build_payload(rows, {"A:g1": ("(;SZ[19]HD[3];B[qf];W[cf])", "W")}, logp, {}, title="t", source="test")["games"][0]
    assert h["stones"] == [[15, 3], [3, 15], [3, 3]] and h["first"] == "W" and h["handicap"] == 3 and h["moves"] == [[16, 5], [2, 5]]
    assert payload["profiles"][0] == "12k" and payload["levels"][-1] == "6D"
