"""scripts/s01_check.py: r3_pma on per-game tokens (staged Human SL + L1b / opponent sums + pmi difference + HD) vs the 64-dim MLP (issue #52)."""
import json

import numpy as np
import pytest

import s01_check
from aicup.gamevec import fold_vectors
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_hd_check import _hd_table
from tests.test_l1b_scripts import _synthetic_scores


def test_s01_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab", moves=6)
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    hd_path = tmp_path / "hd_table.csv"
    _hd_table(hsl_dir, (0, 1, 2, 3, 4), hd_path)
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    for f in (1, 2, 3, 4):
        np.savez_compressed(vectors / f"fold{f}.npz", **fold_vectors(hsl_dir / f"fold{f}"))
    out, models = tmp_path / "s01_check.json", tmp_path / "models"
    args = ["--vectors", str(vectors), "--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--hd-table", str(hd_path), "--per-player", "2",
            "--n-boot", "50", "--seeds", "0", "1", "--device", "cpu", "--epochs", "1", "--batch", "64", "--models-dir", str(models), "--folds-csv", str(folds_csv), "--out", str(out)]
    assert s01_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert res["rung"] == "r3_pma" and res["seeds"] == [0, 1] and res["train_cfg"]["epochs"] == 1 and res["train_cfg"]["batch"] == 64 and res["train_cfg"]["inner_frac"] == 0.1
    assert set(res["scores"]) == {"S00", "S01", "S01_s0", "S01_s1"} and len(res["control_features"]) == 64 and len(res["token_names"]["extra"]) == 26
    for v in ("1", "2", "3", "4"):
        sp = res["per_split"][v]
        assert sp["token_dim"] == 301 + 26 and set(sp["seeds"]) == {"0", "1"} and sp["seeds"]["0"]["n_params"] > 0 and sp["seeds"]["0"]["inner_ce_init"] is not None
    c = res["comparisons"]["pre_specified"]["S01 vs S00"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and set(c["by_n"]) == {"5", "10", "20"} and isinstance(c["pass"], bool)
    assert set(res["comparisons"]["per_seed"]) == {"S01_s0 vs S00", "S01_s1 vs S00"}
    assert set(res["buckets"]) == set(s01_check.BUCKETS) and sum(b["questions"] for b in res["buckets"].values()) == 4 * 30 * 7
    # S01's per-question score is the mean over seeds: its pooled score lies between the seeds' when they differ, and equals them when not
    s = res["scores"]
    lo, hi = sorted([s["S01_s0"]["balanced_mean_over_n"], s["S01_s1"]["balanced_mean_over_n"]])
    assert lo - 1e-9 <= s["S01"]["balanced_mean_over_n"] <= hi + 1e-9
    assert sorted(p.name for p in models.glob("*.pt")) == [f"r3_pma_v{v}_s{seed}.pt" for v in (1, 2, 3, 4) for seed in (0, 1)]
    side = json.loads((models / "r3_pma_v1_s0.json").read_text())
    assert side["d_in"] == 327 and side["experiment"] == "s01_check" and side["split"] == 1 and side["token_extra"][-1] == "hd_gap_down"
    assert "[pre_specified] S01 vs S00" in printed and "bucket no_hd" in printed and "wrote" in printed
    with pytest.raises(SystemExit):
        s01_check.main(args + ["--folds", "0", "1"])
    assert s01_check.main(args[:1] + [str(tmp_path / "missing")] + args[2:]) == 2


def test_game_extra_and_pack_tokens(tmp_path):
    import pandas as pd
    from aicup.scoring import read_scored_games

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(1,), moves=6)
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")  # folds 0-4 of the same synthetic keys: fold 1 matches
    hd_path = tmp_path / "hd.csv"
    _hd_table(hsl_dir, (1,), hd_path)
    from aicup.handicap import read_hd_table
    vec = fold_vectors(hsl_dir / "fold1")
    l1b, opp, hd = read_scored_games(l1b_dir / "fold1"), read_scored_games(opp_dir / "fold1"), read_hd_table(hd_path)
    E = s01_check.game_extra(vec, l1b, opp, hd)
    assert E.shape == (len(vec["game_key"]), 26) and E.dtype == np.float32
    k0 = vec["game_key"][0]
    np.testing.assert_allclose(E[0, :10], l1b.set_index("game_key").loc[k0, [f"sum_{lv}" for lv in s01_check.LEVELS]].to_numpy(dtype=float), rtol=1e-6)
    keys = vec["game_key"][:3].tolist()
    qs = pd.DataFrame({"question_id": ["q0"], "player_id": ["p"], "rank": ["A"], "n_games": [3], "game_keys": [keys]})
    packed = s01_check.pack_tokens(qs, vec, E)
    assert packed["tokens"].shape == (1, 20, 327) and packed["mask"][0].sum() == 3
    np.testing.assert_allclose(packed["tokens"][0, :3, 301:], E[:3])
    assert (packed["tokens"][0, 3:] == 0).all()
