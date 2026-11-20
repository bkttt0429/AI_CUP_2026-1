"""aicup.sgf handicap / result parsing and scripts/hd_census.py (dataset v1.1, issue #42)."""
import json

import pandas as pd
import pytest

import hd_census
from aicup import sgf


def test_sgf_handicap_result_and_properties():
    s = "(;CA[UTF-8]HD[2]RE[W+R];B[pp];W[dd];B[tt])"
    assert sgf.handicap(s) == 2 and sgf.result(s) == "W+R" and sgf.properties(s) == ["CA", "HD", "RE", "B", "W"]
    assert sgf.parse_moves(s) == [("B", 15, 15), ("W", 3, 3), ("B", None, None)]
    v10 = "(;SZ[19];B[pd];W[dp])"
    assert sgf.handicap(v10) is None and sgf.result(v10) is None and sgf.properties(v10) == ["SZ", "B", "W"]
    assert sgf.sgf_to_xy("pd") == (15, 3) and sgf.sgf_to_xy("aa") == (0, 0)
    assert set(sgf.HANDICAP_STONES) == set(range(2, 10)) and all(len(v) == k for k, v in sgf.HANDICAP_STONES.items())
    assert sgf.HANDICAP_STONES[9] == ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"]


def _train_dir(tmp_path, with_hd=True):
    d = tmp_path / "training"
    d.mkdir(parents=True)
    rows = []
    for r, lv in enumerate(["D", "A", "6D"]):
        for g in range(4):
            hd = [0, 0, 1, 2][g] if with_hd else None
            color = ("W" if lv == "6D" else "B") if hd == 2 else "BW"[g % 2]  # in the handicap rows 6D (the strong side) is W, D / A are B
            head = f"(;CA[UTF-8]HD[{hd}]RE[W+R]" if with_hd else "(;SZ[19]"
            first = "B[qf]" if hd == 2 else "B[pd]"
            rows.append({"player_id": f"p{r}{g % 2}", "game_id": f"g{r}{g}", "rank": lv, "color": color, "sgf_content": f"{head};{first};W[dd];B[qq])"})
        pd.DataFrame(rows).to_csv(d / f"train_{lv}.csv", index=False)
        rows = []
    return d


def test_hd_census_cli_v11_and_v10(tmp_path, capsys):
    d = _train_dir(tmp_path)
    out = tmp_path / "hd.json"
    assert hd_census.main(["--train-dir", str(d), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["rows"] == 12 and res["hd"] == {"0": 6, "1": 3, "2": 3} and res["re_present"] == 12
    assert set(res["hd_by_rank"]) == {"D", "A", "6D"} and res["hd_by_rank"]["6D"]["2"] == 1
    h = res["handicap_games"]
    assert h["rows"] == 3 and h["first_move_mark"] == {"B": 3} and h["first_move_on_star_point_share"] == 0.0  # qf is not a star point
    assert set(h["labelled_rank_by_color"]) == {"B", "W"} and h["labelled_rank_by_color"]["W"]["rows"] == 1 and h["labelled_rank_by_color"]["B"]["rows"] == 2
    assert h["labelled_rank_by_color"]["W"]["mean_rank_idx"] > h["labelled_rank_by_color"]["B"]["mean_rank_idx"]
    assert res["players_with_any_hd2_share"] > 0 and "CA,HD,RE,B,W" in res["properties"]
    assert "handicap games (HD >= 2): 3" in capsys.readouterr().out
    d10 = _train_dir(tmp_path / "v10", with_hd=False)
    assert hd_census.main(["--train-dir", str(d10), "--out", str(tmp_path / "hd10.json")]) == 0
    res10 = json.loads((tmp_path / "hd10.json").read_text())
    assert res10["hd"] == {"absent": 12} and res10["re_present"] == 0 and "handicap_games" not in res10
    assert hd_census.main(["--train-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "x.json")]) == 2
    with pytest.raises(SystemExit):
        hd_census.main([])
