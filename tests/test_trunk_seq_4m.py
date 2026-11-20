"""Section 4M (2026-11-04) data-axis flags of scripts/trunk_seq_check.py and the --trunk guard of scripts/trunk_extract.py."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest

import trunk_extract as te
import trunk_seq_check as ts


def _players():
    return pd.DataFrame({"rank": ["D", "C", "B"], "player_id": ["0123456789abcdef", "fedcba9876543210", "00000000deadbeef"],
                         "game_keys": [[f"a{i}" for i in range(20)], [f"b{i}" for i in range(7)], [f"c{i}" for i in range(10)]]})


def test_training_folds_drop_each_fold_once():
    assert ts.training_folds(1) == [2, 3, 4] and ts.training_folds("1") == [2, 3, 4]
    assert ts.training_folds("all") == [1, 2, 3, 4]
    assert [ts.training_folds(v, 2) for v in (1, 2, 3, 4)] == [[3, 4], [1, 4], [1, 2], [2, 3]]
    dropped = [(set(ts.training_folds(v)) - set(ts.training_folds(v, 2))).pop() for v in (1, 2, 3, 4)]
    assert sorted(dropped) == [1, 2, 3, 4]
    assert ts.training_folds(3, 3) == [1, 2, 4] and ts.training_folds(3, 0) == [1, 2, 4]
    assert ts.training_folds(2, 1) == [1]
    with pytest.raises(SystemExit):
        ts.training_folds("all", 2)


def test_cap_games_deterministic_subset_input_untouched():
    players = _players()
    assert ts.cap_games(players, 0) is players
    out = ts.cap_games(players, 10)
    assert [len(k) for k in out["game_keys"]] == [10, 7, 10]
    assert all(set(k) <= set(o) for k, o in zip(out["game_keys"], players["game_keys"]))
    assert out["game_keys"][1] == players["game_keys"][1] and out["game_keys"][2] == players["game_keys"][2]  # too few: unchanged
    assert out["game_keys"][0] != players["game_keys"][0][:10]  # a seeded permutation, not the head
    assert ts.cap_games(players.iloc[::-1].reset_index(drop=True), 10)["game_keys"][2] == out["game_keys"][0]  # row order irrelevant
    assert players["game_keys"][0] == [f"a{i}" for i in range(20)]
    q = ts.sample_questions(ts.cap_games(players, 10), 32, np.random.RandomState(0), sizes=range(5, 11))
    assert set(q["n_games"]) <= set(range(5, 11)) and all(len(k) == n for k, n in zip(q["game_keys"], q["n_games"]))
    assert all(set(k) <= set(out["game_keys"][0]) for k, p in zip(q["game_keys"], q["player_id"]) if p == players["player_id"][0])


def test_tags_and_paths():
    def ns(**kw):
        return argparse.Namespace(**{"train_folds": 0, "train_games": 0, "train_sizes": None, "mix": 0.0, **kw})
    assert ts.data_segments(ns()) == []
    assert ts.data_segments(ns(train_folds=2)) == ["f2"]
    assert ts.data_segments(ns(train_games=10, train_sizes=[5, 10])) == ["g10s5-10"]
    assert ts.data_segments(ns(train_sizes=[5, 10])) == ["s5-10"]
    assert ts.data_segments(ns(mix=0.5)) == ["mix50"]
    assert ts.data_segments(ns(train_folds=2, train_games=10, train_sizes=[5, 10], mix=0.5)) == ["f2", "g10s5-10", "mix50"]
    assert ts.data_segments(argparse.Namespace()) == []  # older callers without the attributes
    assert ts.vec_tag(ts.VEC) == "" and ts.vec_tag(str(ts.VEC)) == ""
    assert ts.vec_tag(ts.ROOT / "results/trunk_vec_ft") == "-ft" and ts.vec_tag(ts.ROOT / "results/other") == "-other"
    assert ts.scores_path("cache", 3).name == "scores_fold3.npy"
    assert ts.scores_path("cache", 3, ts.ROOT / "results/trunk_vec_ft").name == "scores_ft_fold3.npy"


def test_cli_guards():
    with pytest.raises(SystemExit):
        ts.main(["rfeat", "--val-fold", "1", "--train-games", "10"])  # 10 games cannot fill 20-game questions
    with pytest.raises(SystemExit):
        ts.main(["rfeat", "--val-fold", "1", "--train-sizes", "4", "10"])
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--val-fold", "all", "--train-folds", "2"])


def test_trunk_meta_guard(tmp_path):
    pristine = te.trunk_info(None)
    assert pristine["trunk"] == "pristine"
    d = tmp_path / "fold1"
    d.mkdir()
    te.check_trunk_meta(d, pristine)  # an empty directory takes the first trunk
    assert json.loads((d / "trunk.json").read_text()) == pristine
    te.check_trunk_meta(d, pristine)
    ft = {"trunk": "fine-tuned", "heads": "heads3_B.pt", "trunk_file": "heads3_B.trunk.pt", "trunk_sha256": "ab"}
    with pytest.raises(SystemExit):
        te.check_trunk_meta(d, ft)
    d2 = tmp_path / "fold2"  # parts but no trunk.json: the pristine trunk's (results/trunk_vec as extracted before 4M)
    d2.mkdir()
    (d2 / "games_s0_p000.csv").write_text("game_key\n")
    with pytest.raises(SystemExit):
        te.check_trunk_meta(d2, ft)
    te.check_trunk_meta(d2, pristine)
    assert (d2 / "trunk.json").exists()
    d3 = tmp_path / "fold3"
    d3.mkdir()
    te.check_trunk_meta(d3, ft)
    assert json.loads((d3 / "trunk.json").read_text()) == ft
    side = tmp_path / "h.json"
    side.write_text(json.dumps({"trunk_file": "h.trunk.pt", "trunk_sha256": "cd"}))
    assert te.trunk_info(tmp_path / "h.pt") == {"trunk": "fine-tuned", "heads": "h.pt", "trunk_file": "h.trunk.pt", "trunk_sha256": "cd"}
    side.write_text(json.dumps({"state_sha256": "x"}))
    with pytest.raises(SystemExit):
        te.trunk_info(tmp_path / "h.pt")  # a stage-1 run has no fine-tuned trunk


def _batch(G=(5, 7), L=30, seed=0):
    """Random tokens of two questions with G games each (flags 0/1, varying lengths), as batch_tokens lays them out."""
    rng = np.random.RandomState(seed)
    n_games = sum(G)
    lens = rng.randint(10, L + 1, size=n_games)
    tokens = np.zeros((n_games, L, ts.D_TOKEN), np.float32)
    mask = np.zeros((n_games, L, 1), np.float32)
    for g, n in enumerate(lens):
        tokens[g, :n, :ts.D_TOKEN - 4] = rng.randn(n, ts.D_TOKEN - 4)
        tokens[g, :n, ts.D_TOKEN - 4] = rng.randint(0, 2, n)  # is_target
        tokens[g, :n, ts.D_TOKEN - 3] = rng.randint(0, 2, n)  # mover
        tokens[g, :n, ts.D_TOKEN - 2] = np.arange(n) / n
        mask[g, :n, 0] = 1.0
    owner = np.repeat(np.arange(len(G)), G)
    import torch
    return torch.as_tensor(tokens), torch.as_tensor(mask), torch.as_tensor(owner)[:, None], torch.as_tensor(rng.randn(len(G), 64).astype(np.float32))


@pytest.mark.parametrize("arm", ["C", "T"])
def test_pma_model_forward_and_checkpoint(arm, tmp_path):
    import torch
    torch.manual_seed(0)
    tokens, mask, game_of, r = _batch()
    base, pma = ts.Model(arm, 64), ts.Model(arm, 64, pool="pma")
    assert base.pool == "meanstd" and pma.pool == "pma" and not hasattr(base, "game_pool")
    with pytest.raises(ValueError):
        ts.Model(arm, 64, pool="max")
    for m in (base, pma):
        m.eval()
        out = m(r, tokens, mask, game_of)
        assert out.shape == (2, 10) and torch.isfinite(out).all()
    # the linear layers after the poolings keep their shapes; PMA adds its own parameters
    assert pma.game.weight.shape == base.game.weight.shape and pma.question.weight.shape == base.question.weight.shape
    assert sum(p.numel() for p in pma.parameters()) > sum(p.numel() for p in base.parameters())
    # a question's games are pooled by their owner only: shuffling the other question's games leaves its logits unchanged
    pma.eval()
    out = pma(r, tokens, mask, game_of)
    perm = np.r_[np.random.RandomState(1).permutation(5), 5 + np.arange(7)]
    out2 = pma(r, tokens[perm], mask[perm], game_of[perm])
    assert torch.allclose(out[1], out2[1], atol=1e-5) and torch.allclose(out[0], out2[0], atol=1e-5)  # order within a question does not matter either
    # padding does not leak: a longer padded block gives the same logits
    tokens_p, mask_p = torch.cat([tokens, torch.zeros(12, 10, ts.D_TOKEN)], 1), torch.cat([mask, torch.zeros(12, 10, 1)], 1)
    assert torch.allclose(out, pma(r, tokens_p, mask_p, game_of), atol=1e-5)
    # checkpoint round trip keeps the pooling
    side = ts.save_checkpoint(tmp_path / f"{arm}.pt", pma, (np.zeros(ts.D_TOKEN - 4, np.float32), np.ones(ts.D_TOKEN - 4, np.float32)), (np.zeros(64), np.ones(64)), {"tag": f"{arm}-pma"})
    assert side["pool"] == "pma"
    model, _, _, side2 = ts.load_checkpoint(tmp_path / f"{arm}.pt")
    assert model.pool == "pma" and side2["pool"] == "pma"
    assert torch.allclose(model(r, tokens, mask, game_of), out, atol=1e-6)
    r0 = ts.save_checkpoint(tmp_path / "r0.pt", ts.Model("R0", 64), None, (np.zeros(64), np.ones(64)), {})
    assert r0["pool"] == "meanstd" and ts.load_checkpoint(tmp_path / "r0.pt")[0].pool == "meanstd"


def test_fold0_and_extra_games_flags():
    assert ts.training_folds("all", with_fold0=True) == [0, 1, 2, 3, 4] and ts.training_folds("all") == [1, 2, 3, 4]
    with pytest.raises(SystemExit):
        ts.training_folds(1, with_fold0=True)
    ns = lambda **kw: argparse.Namespace(**{"train_folds": 0, "train_games": 0, "train_sizes": None, "mix": 0.0, "with_fold0": False, "extra_games": 0, **kw})  # noqa: E731
    assert ts.data_segments(ns(with_fold0=True)) == ["f0"] and ts.data_segments(ns(extra_games=40)) == ["x40"] and ts.data_segments(ns(extra_games=80)) == ["x80"]
    assert ts.data_segments(ns(with_fold0=True, extra_games=40, train_sizes=[5, 10])) == ["f0", "s5-10", "x40"]
    assert ts.extra_sets(0) == [] and ts.extra_sets(40) == [40] and ts.extra_sets(80) == [40, 80]
    assert ts.extra_level(ns(extra_games=80)) == 80 and ts.extra_level(argparse.Namespace()) == 0
    with pytest.raises(SystemExit):
        ts.extra_level(ns(extra_games=60))
    assert ts.set_dirs_for(ts.L1B_SETS[3], 80)[0][1] == "scores3_B_hd_x80" and ts.set_dirs_for(ts.L1B_SETS[4], 40)[1][0] == "scores4_opp_hd_x40"
    assert ts.EXTRAS[80]["vec"].name == "trunk_vec_x80" and ts.scores_path("c", 2, ts.EXTRAS[80]["vec"]).name == "scores_x80_fold2.npy"
    assert ts.L1B[0] == "scores3_hd" and ts.OPP[0] == "scores3_opp_hd" and ts.EXTRA["l1b"][1] == "scores3_B_hd_x40" and ts.EXTRA["l1b"][3] == "scores3_hd_x40"
    assert ts.scores_path("c", 2, ts.EXTRA["vec"]).name == "scores_x40_fold2.npy"
    with pytest.raises(SystemExit):
        ts.main(["train", "--arm", "C", "--val-fold", "1", "--with-fold0"])


def test_extra_tables_merge_and_selection_order(tmp_path, monkeypatch):
    """extra_tables appends the x40 tables and restores fold_players' order over 40 games; duplicates and mismatches are refused."""
    import pandas as pd
    from aicup.scoring import fold_players
    pid = "0123456789abcdef"
    rng = np.random.RandomState(0)
    keys = sorted(f"D:g{i:03d}" for i in range(45))
    games = pd.DataFrame({"player_id": pid, "rank": "D", "game_key": keys})
    folds_df = pd.DataFrame({"player_id": [pid], "rank": ["D"], "fold": [1], "n_games": [45]})
    base_sel = fold_players(games, folds_df, 1, max_games=20)["game_keys"][0]
    wide_sel = fold_players(games, folds_df, 1, max_games=40)["game_keys"][0]
    assert wide_sel[:20] == base_sel
    def table(ks):
        return pd.DataFrame({"game_key": ks, "player_id": pid, "rank": "D", "color": "B", "n_moves": 100})
    base = {"hsl": table(base_sel), "l1b": table(base_sel), "opp": table(base_sel)}
    monkeypatch.setattr(ts.l1b_cv, "load_fold", lambda hsl_dir, l1b_dir, f: (table(wide_sel[20:]), table(wide_sel[20:])))
    monkeypatch.setattr(ts, "read_scored_games", lambda d: table(wide_sel[20:]))
    hd_all = pd.DataFrame({"game_key": keys, "hd": 0})
    out = ts.extra_tables(1, base, folds_df, hd_all)
    assert len(out["hsl"]) == 40 and out["players"]["game_keys"][0] == wide_sel and len(out["hd"]) == 40
    monkeypatch.setattr(ts, "read_scored_games", lambda d: table(wide_sel[20:30]))  # an incomplete opponent directory
    with pytest.raises(ValueError, match="different games"):
        ts.extra_tables(1, base, folds_df, hd_all)
    monkeypatch.setattr(ts, "read_scored_games", lambda d: table(wide_sel[20:]))
    monkeypatch.setattr(ts.l1b_cv, "load_fold", lambda hsl_dir, l1b_dir, f: (table(wide_sel[10:]), table(wide_sel[10:])))  # overlaps the base sample
    with pytest.raises(ValueError, match="both"):
        ts.extra_tables(1, base, folds_df, hd_all)


def test_l1b_set_4_is_out_of_fold_and_tagged():
    s4 = ts.L1B_SETS[4]
    assert {f for f, d in s4["l1b"].items() if d == "scores4_hd"} == {0, 2, 4} and {f for f, d in s4["l1b"].items() if d == "scores4_B_hd"} == {1, 3}
    assert s4["l1b_x"][1] == "scores4_B_hd_x40" and s4["opp_x"][0] == "scores4_opp_hd_x40" and s4["heads"] == ["heads4_A", "heads4_B"]
    assert ts.L1B_SETS[3]["l1b"] is ts.L1B and ts.L1B_SETS[3]["l1b_x"][2] == "scores3_B_hd_x40"
    ns = lambda **kw: argparse.Namespace(**{"train_folds": 0, "train_games": 0, "train_sizes": None, "mix": 0.0, "with_fold0": False, "extra_games": 0, "l1b_set": 3, **kw})  # noqa: E731
    assert ts.data_segments(ns(l1b_set=4)) == ["h4"] and ts.data_segments(ns(l1b_set=4, with_fold0=True, extra_games=40)) == ["f0", "h4", "x40"]
    assert ts.l1b_set_of(ns()) == 3 and ts.l1b_set_of(ns(l1b_set=4)) == 4 and ts.l1b_set_of(argparse.Namespace()) == 3
    with pytest.raises(SystemExit):
        ts.l1b_set_of(ns(l1b_set=5))
    assert ts.scores_path("c", 1, ts.VEC, 4).name == "scores_h4_fold1.npy" and ts.scores_path("c", 1, ts.EXTRA["vec"], 4).name == "scores_x40_h4_fold1.npy"
    assert ts.scores_path("c", 1).name == "scores_fold1.npy"


def test_swa_average_window():
    import torch
    states = [(s, {"w": torch.full((2,), float(s)), "b": torch.tensor([s], dtype=torch.int64)}) for s in (10, 20, 30, 40, 50)]
    avg = ts.swa_average(states, 30, 3)  # centred on step 30 -> 20, 30, 40
    assert torch.allclose(avg["w"], torch.full((2,), 30.0)) and avg["b"].dtype == torch.int64
    avg = ts.swa_average(states, 50, 3)  # best at the end -> clamped to the last three
    assert torch.allclose(avg["w"], torch.full((2,), 40.0))
    avg = ts.swa_average(states, 10, 5)  # best at the start -> the first five
    assert torch.allclose(avg["w"], torch.full((2,), 30.0))
    avg = ts.swa_average(states, 999, 2)  # unknown step -> the last two
    assert torch.allclose(avg["w"], torch.full((2,), 45.0))
