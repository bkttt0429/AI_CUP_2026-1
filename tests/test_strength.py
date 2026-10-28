"""aicup.strength (the Bradley-Terry strength head), aicup.sefeat, scripts/se_train.py, se_score.py and se_check.py -- on a
random b2c16 model, a tiny real cache and synthetic score tables."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from katago.train import modelconfigs
from katago.train.model_pytorch import Model

from aicup import featcache as fc
from aicup import l1b, l1btrain, sefeat, strength
from aicup.metric import LEVELS
from tests.test_l1btrain import _tiny_cache_dir


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    model.eval()
    return model


@pytest.fixture
def tiny_cache(tmp_path):
    return _tiny_cache_dir(tmp_path)


def test_strength_head_starts_at_the_policy_logit_and_scores_in_batches(tiny_model, tiny_cache):
    _, arrays = fc.read_cache(tiny_cache)
    head = strength.StrengthHead(tiny_model.policy_head)
    idx = np.arange(min(40, len(arrays["target"])))
    bins, globs, legal, target, rank, color = l1btrain._to_tensors(arrays, idx, "cpu")
    rows = l1b.neutral_rows()
    with torch.no_grad():
        s = strength.move_scores(tiny_model, head, bins, globs, target, color, rows, autocast=False)
        meta = torch.as_tensor(rows)[color] if tiny_model.get_has_metadata_encoder() else None
        own = tiny_model.postprocess_output(tiny_model(bins, globs, input_meta=meta))[0][0][:, 0, :].float()
        want = own.gather(1, target[:, None])[:, 0]
    assert s.shape == (len(idx),) and torch.allclose(s, want, atol=1e-4)  # step 0: humanv0's policy logit of the move
    scores = strength.strength_scores(tiny_model, head, fc.unpack_bins(arrays["bins_packed"][idx]), arrays["globs"][idx], arrays["target"][idx],
                                      arrays["color"][idx], batch=16)
    assert scores.shape == (len(idx),) and scores.dtype == np.float32 and np.allclose(scores, s.numpy(), atol=1e-4)


def test_pairwise_bt_loss_and_accuracy():
    rank = torch.tensor([0, 0, 3, 3, 9])
    w = torch.ones(5, dtype=torch.float64)
    good = torch.tensor([-2.0, -1.9, 0.0, 0.1, 3.0], dtype=torch.float64)  # scores ordered like the ranks
    num, den, correct, pairs = strength.pairwise_bt(good, rank, w)
    assert pairs == 2 * (2 * 2 + 2 * 1 + 2 * 1) and correct == pairs and float(num / den) < float(-torch.nn.functional.logsigmoid(torch.tensor(0.0)))
    num_r, den_r, correct_r, pairs_r = strength.pairwise_bt(-good, rank, w)
    assert pairs_r == pairs and correct_r == 0 and float(num_r / den_r) > float(num / den)
    num0, den0, c0, p0 = strength.pairwise_bt(good[:2], rank[:2], w[:2])  # one rank: no pair
    assert p0 == 0 and float(den0) == 0.0 and float(num0) == 0.0
    # the weights: a heavier pair counts more
    w2 = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    num_w, den_w, _, _ = strength.pairwise_bt(good, rank, w2)
    assert float(den_w) > float(den) and np.isfinite(float(num_w))


def test_train_strength_head_runs_resumes_and_round_trips(tmp_path, tiny_model, tiny_cache):
    cfg = l1btrain.HeadsCfg(epochs=1, batch=8, lr=1e-3, eval_every=0.5, patience=10, inner_frac=0.34, seed=0, device="cpu")
    out = tmp_path / "se_A.pt"
    head, meta = strength.train_strength_head(tiny_model, tiny_cache, cfg, out, log=lambda *a, **k: None)
    assert out.exists() and out.with_suffix(".json").exists() and (tmp_path / "se_A.pt.ckpt").exists()
    assert meta["kind"] == "strength" and meta["train_folds"] == [1, 2] and meta["head_init"] == "humanv0_policy_head"
    assert np.isfinite(meta["inner_bt_init"]) and meta["inner_bt_best"] <= meta["inner_bt_init"] and 0.0 <= meta["inner_pair_accuracy_best"] <= 1.0
    assert meta["n_train_positions"] + meta["n_inner_positions"] == len(fc.read_cache(tiny_cache)[1]["target"])
    assert len(meta["history"]) == 3 and meta["history"][0][0] == 0.0 and meta["history"][-1][0] == 1.0 and meta["stopped_early"] is False
    assert len(meta["inner_mean_by_rank_best"]) == 10 and meta["selection"] == "inner Bradley-Terry loss"
    # resume with more epochs; another batch size is another run
    head2, meta2 = strength.train_strength_head(tiny_model, tiny_cache, replace(cfg, epochs=2), out, log=lambda *a, **k: None)
    assert meta2["resumed_from_step"] == meta["steps_per_epoch"] and meta2["history"][:3] == meta["history"] and meta2["steps_run"] == 2 * meta["steps_per_epoch"]
    with pytest.raises(ValueError, match="checkpoint"):
        strength.train_strength_head(tiny_model, tiny_cache, replace(cfg, epochs=2, batch=4), out, log=lambda *a, **k: None)
    # load: the same scores; sha and kind checks
    loaded, side, warnings = strength.load_strength_head(out, tiny_model, "cpu")
    assert side["state_sha256"] == meta2["state_sha256"] != meta["state_sha256"] and warnings == []  # the resumed run rewrote the file
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), head2.state_dict().values()))
    rank_heads = tmp_path / "heads.pt"
    l1b.save_heads(l1b.RankHeads(tiny_model.policy_head), rank_heads, {"train_folds": [1, 2]})
    with pytest.raises(ValueError, match="kind"):
        strength.load_strength_head(rank_heads, tiny_model, "cpu")
    out.write_bytes(out.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        strength.load_strength_head(out, tiny_model, "cpu")
    # a cache with one rank cannot be paired
    _, arrays = fc.read_cache(tiny_cache)
    games = pd.read_csv(sorted(tiny_cache.glob("cache_s*_p*.csv"))[0]) if list(tiny_cache.glob("cache_s*_p*.csv")) else None
    assert games is None or len(games) > 0  # layout detail: the ValueError path is covered by the trainer's rank check below
    one = _tiny_cache_dir(tmp_path / "one", ranks=("D",), players_per_rank=3, games_per_player=2)
    with pytest.raises(ValueError, match="two ranks"):
        strength.train_strength_head(tiny_model, one, replace(cfg, inner_frac=0.0), tmp_path / "one.pt", log=lambda *a, **k: None)


def test_sefeat_game_stats_and_question_features():
    s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    t = np.array([0, 10, 39, 40, 120])
    st = sefeat.game_stats(s, t)
    assert st["se_mean"] == 3.0 and abs(st["se_std"] - s.std()) < 1e-12 and st["se_q50"] == 3.0
    assert st["se_s1"] == 2.0 and st["se_s2"] == 5.0 and st["se_s3"] == 4.0  # stage means by t
    st2 = sefeat.game_stats(s[:3], t[:3])
    assert st2["se_s2"] == st2["se_mean"] and st2["se_s3"] == st2["se_mean"]  # an empty stage takes the game's mean
    assert all(v == 0.0 for v in sefeat.game_stats([], []).values())
    stats = np.array([[1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0], [3.0, 1.5, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0]])
    n = np.array([10, 30])
    f = sefeat.se_features(stats, n)
    assert len(f) == len(sefeat.SE_FEATURE_NAMES) == 9
    assert abs(f[0] - 2.5) < 1e-12  # move-weighted mean of the means
    assert abs(f[1] - np.sqrt(0.25 * 1.5 ** 2 + 0.75 * 0.5 ** 2)) < 1e-12 and abs(f[2] - 1.25) < 1e-12
    assert abs(f[3] - 1.5) < 1e-12 and abs(f[5] - 3.5) < 1e-12
    games = pd.DataFrame({"game_key": ["a", "b", "c"], "n_moves": [10, 30, 5], **{k: [1.0, 3.0, 2.0] for k in sefeat.SE_STATS}})
    questions = pd.DataFrame({"question_id": ["q1", "q2"], "rank": ["D", "6D"], "game_keys": [["a", "b"], ["c"]]})
    X, y, nq = sefeat.build_se_matrix(games, questions)
    assert X.shape == (2, 9) and y.tolist() == [0, 9] and nq.tolist() == [2, 1] and abs(X[0, 0] - 2.5) < 1e-12 and X[1, 1] == 0.0


def test_se_train_and_score_cli(tmp_path, tiny_model, tiny_cache, monkeypatch, capsys):
    import se_score as ss
    import se_train as st
    from aicup.pilot import game_positions
    from aicup.scoring import read_scored_games
    from katago.game.features import Features
    from tests.test_l1b_scripts import _fake_data

    out = tmp_path / "se_B.pt"
    args = ["--cache", str(tiny_cache), "--out", str(out), "--model-config", "b2c16", "--device", "cpu", "--epochs", "1", "--batch", "8",
            "--inner-frac", "0.34", "--eval-every", "0.5"]
    assert st.main(args) == 0 and "inner BT loss" in capsys.readouterr().out
    assert st.main(args) == 2 and "exists" in capsys.readouterr().err
    # the head's train_folds are [1, 2]: scoring folds 1, 2 is refused as not out-of-fold; folds 3, 4 of the fake data are scored
    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    games_df["rank"] = games_df["rank"]  # 6 players over folds 1 and 2
    folds = pd.read_csv(folds_csv)
    folds["fold"] = folds["fold"].map({1: 3, 2: 4})  # move the fake players to folds 3 and 4
    folds.to_csv(folds_csv, index=False)
    monkeypatch.setattr(ss, "player_games", lambda: games_df)
    monkeypatch.setattr(ss, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    base = ["--head", str(out), "--model-config", "b2c16", "--device", "cpu", "--workers", "1", "--part-size", "5", "--min-games", "1",
            "--folds-csv", str(folds_csv)]
    assert ss.main([*base, "--folds", "1", "--out", str(tmp_path / "bad")]) == 2 and "train_folds" in capsys.readouterr().err
    se_dir = tmp_path / "se"
    assert ss.main([*base, "--folds", "3", "4", "--out", str(se_dir)]) == 0
    printed = capsys.readouterr().out
    assert "fold 3" in printed and "fold 4" in printed
    features = Features(modelconfigs.config_of_name["b2c16"], 19)
    head, _, _ = strength.load_strength_head(out, tiny_model, "cpu")
    for fold in (3, 4):
        table = read_scored_games(se_dir / f"fold{fold}")
        assert list(table.columns) == ss.COLUMNS and len(table) == 6
        npz = sorted((se_dir / f"fold{fold}").glob("moves_s0_p*.npz"))
        assert npz
        with np.load(npz[0]) as z:
            assert z["s"].dtype == np.float32 and len(z["s"]) == len(z["game_row"]) == len(z["t"])
        for row in table.itertuples():
            pos = game_positions(*sgfs[row.game_key], features)
            assert row.n_moves == len(pos["target"])
            if row.n_moves:
                s = strength.strength_scores(tiny_model, head, pos["bins"], pos["globs"], pos["target"], pos["color"])
                assert abs(row.se_mean - s.mean()) < 1e-4 and abs(row.se_q50 - np.quantile(s, 0.5)) < 1e-4
    meta = json.loads((se_dir / "fold3" / "meta_s0.json").read_text())
    assert meta["side"] == "target" and meta["head"]["train_folds"] == [1, 2] and meta["head"]["trunk_sha256"] is None and meta["stats"] == list(sefeat.SE_STATS)
    # the same command again scores nothing new; --opponent scores the other colour and says so
    assert ss.main([*base, "--folds", "3", "--out", str(se_dir)]) == 0 and "0 to score" in capsys.readouterr().out
    opp_dir = tmp_path / "se_opp"
    assert ss.main([*base, "--folds", "3", "--out", str(opp_dir), "--opponent"]) == 0
    opp = read_scored_games(opp_dir / "fold3").set_index("game_key")
    tgt = read_scored_games(se_dir / "fold3").set_index("game_key")
    for key in tgt.index:
        sgf, color = sgfs[key]
        other = "W" if color == "B" else "B"
        assert opp.loc[key, "color"] == other and int(opp.loc[key, "n_moves"]) == sgf.count(f";{other}[")
    assert json.loads((opp_dir / "fold3" / "meta_s0.json").read_text())["side"] == "opponent"


def _synthetic_se_dir(tmp_path, like_dir, name, shift=0.0):
    """se_score.py tables for the games of an l1b_score.py directory (every fold there): statistics from the rank."""
    from aicup.scoring import part_paths, read_scored_games

    out = tmp_path / name
    for fold_dir in sorted(Path(like_dir).glob("fold*")):
        games = read_scored_games(fold_dir)
        r = games["rank"].map({lv: i for i, lv in enumerate(LEVELS)}).to_numpy(dtype=float)
        rng = np.random.RandomState(len(fold_dir.name) + int(shift * 10))
        table = games[["game_key", "player_id", "rank", "color", "n_moves"]].copy()
        mean = r + shift + rng.randn(len(r)) * 0.3
        for k, col in enumerate(sefeat.SE_STATS):
            table[col] = mean + 0.1 * k if col != "se_std" else 0.5
        d = out / fold_dir.name
        d.mkdir(parents=True)
        csv_path, npz_path = part_paths(d, 0, 0)
        table.to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, s=np.zeros(0, np.float32), game_row=np.zeros(0, np.int32), t=np.zeros(0, np.int16))
    return out


def test_se_check_cli(tmp_path, capsys):
    import se_check as sc
    from tests.test_fused_scripts import _folds_csv, _opponent_scores
    from tests.test_l1b_scripts import _synthetic_scores

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(1, 2, 3, 4), l1b_name="l1b_all")
    opp_dir = _opponent_scores(tmp_path)
    se_dir, seo_dir = _synthetic_se_dir(tmp_path, l1b_dir, "se_all"), _synthetic_se_dir(tmp_path, l1b_dir, "se_opp_all", shift=0.5)
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "se_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--se", str(se_dir), "--se-opp", str(seo_dir), "--hsl", str(hsl_dir),
            "--per-player", "2", "--n-boot", "200", "--folds-csv", str(folds_csv), "--out", str(out)]
    assert sc.main(args) == 0
    printed = capsys.readouterr().out
    assert "fused_opp_se vs fused_opp" in printed and "wrote" in printed
    res = json.loads(out.read_text())
    assert set(res["scores"]) == set(sc.SETS) and set(res["comparisons"]["pre_specified"]) == {"fused_opp_se vs fused_opp", "fused_opp_se2 vs fused_opp"}
    assert set(res["comparisons"]["report_only"]) == {"se vs hsl", "hsl_se vs hsl", "fused_opp vs hsl"} and res["se_features"] == sefeat.SE_FEATURE_NAMES
    assert all(0.0 <= v["balanced_mean_over_n"] <= 1.0 and len(v["per_fold"]) == 4 for v in res["scores"].values())
    assert res["scores"]["se"]["balanced_mean_over_n"] > 0.5  # the synthetic statistics carry the rank
    for c in res["comparisons"]["pre_specified"].values():
        assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4
    # a strength directory covering other games is refused
    bad = _synthetic_se_dir(tmp_path, l1b_dir, "se_bad")
    csv = sorted((bad / "fold1").glob("games_s0_p000.csv"))[0]
    t = pd.read_csv(csv, dtype=str, keep_default_na=False)
    t.iloc[1:].to_csv(csv, index=False)
    assert sc.main([*args[:5], str(bad), *args[6:]]) == 2 and "other games" in capsys.readouterr().err
