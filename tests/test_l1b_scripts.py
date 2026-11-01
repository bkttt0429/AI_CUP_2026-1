"""The L1b scripts on synthetic games and a random b2c16 model: cache builder, trainer, scorer, early signal, CV."""
import json

import pandas as pd

import l1b_build_cache as bc
import l1b_train as lt
from aicup import featcache as fc

POINTS = ["pd", "dp", "pq", "dd", "qf", "cf", "jj", "fc", "nc", "cn", "pn", "gq"]


def _sgf(n_moves):
    return "(;SZ[19]" + "".join(f";{'BW'[i % 2]}[{POINTS[i]}]" for i in range(n_moves)) + ")"


def _fake_data(tmp_path):
    """6 players over folds 1 and 2, 12 games; game 'D:g0' has one White move and target Black: no positions."""
    # player ids are hex (fold_players seeds each player's permutation from player_id[:8])
    players = [("0000a001", "D", 1), ("0000a002", "D", 2), ("0000a003", "1D", 1), ("0000a004", "3D", 2), ("0000a005", "6D", 1),
               ("0000a006", "A", 2)]
    rows, sgfs = [], {}
    g = 0
    for pid, rank, _ in players:
        for _ in range(2):
            key = f"{rank}:g{g}"
            rows.append({"player_id": pid, "rank": rank, "game_key": key})
            sgfs[key] = ("(;SZ[19];W[dp])", "B") if g == 0 else (_sgf(3 + g % 5), "BW"[g % 2])
            g += 1
    folds = pd.DataFrame([{"rank": r, "player_id": p, "n_games": 2, "fold": f} for p, r, f in players])
    folds_csv = tmp_path / "folds.csv"
    folds.to_csv(folds_csv, index=False)
    return pd.DataFrame(rows), sgfs, folds_csv


def _target_moves(sgf, color):
    return sgf.count(f";{color}[")


def test_build_cache_cli(tmp_path, monkeypatch, capsys):
    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    monkeypatch.setattr(bc, "player_games", lambda: games_df)
    monkeypatch.setattr(bc, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    out = tmp_path / "cache"
    args = ["--folds", "1", "2", "--out", str(out), "--folds-csv", str(folds_csv), "--model-config", "b2c16",
            "--workers", "2", "--part-size", "4", "--max-games", "40", "--min-games", "1"]
    assert bc.main(args) == 0
    printed = capsys.readouterr().out
    assert fc.completed_parts(out) == [(0, 0), (0, 1), (0, 2)]
    games, arrays = fc.read_cache(out)
    assert len(games) == 12 and set(games["game_key"]) == set(sgfs) and (games["n_moves"] == 0).sum() == 1
    assert games.set_index("game_key").loc["D:g0", "n_moves"] == 0
    want = {k: _target_moves(sgf, c) for k, (sgf, c) in sgfs.items()}
    assert games.set_index("game_key")["n_moves"].to_dict() == want
    assert len(arrays["target"]) == sum(want.values()) and arrays["bins_packed"].shape[1] == fc.BIN_BYTES
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["folds"], meta["max_games"], meta["min_games"], meta["players"], meta["games"]) == ([1, 2], 40, 1, 6, 12)
    assert meta["positions"] == len(arrays["target"]) and sum(meta["per_rank_positions"].values()) == meta["positions"]
    assert meta["per_rank_positions"]["D"] == want["D:g1"] + want["D:g2"] + want["D:g3"]
    assert len(meta["games_sha256"]) == 64 and "code_sha256" in meta and meta["nshards"] == 1
    assert "part 0" in printed and "3 parts" in printed
    # the same command again: every part is finished, nothing is rewritten
    mtimes = {p.name: p.stat().st_mtime_ns for p in out.glob("cache_s*")}
    assert bc.main(args) == 0 and {p.name: p.stat().st_mtime_ns for p in out.glob("cache_s*")} == mtimes
    assert "nothing to do" in capsys.readouterr().out
    # two shards write disjoint parts and their own meta; --merge-meta writes meta.json over both
    out2 = tmp_path / "cache2"
    args2 = ["--folds", "1", "2", "--out", str(out2), "--folds-csv", str(folds_csv), "--model-config", "b2c16", "--workers", "1",
             "--part-size", "4", "--nshards", "2"]
    assert bc.main([*args2, "--shard", "0"]) == 0 and bc.main([*args2, "--shard", "1"]) == 0
    assert not (out2 / "meta.json").exists() and (out2 / "meta_s0.json").exists() and (out2 / "meta_s1.json").exists()
    assert bc.main(["--out", str(out2), "--merge-meta"]) == 0
    merged = json.loads((out2 / "meta.json").read_text())
    assert (merged["games"], merged["positions"], merged["nshards"]) == (12, meta["positions"], 2)
    games2, _ = fc.read_cache(out2)
    assert set(games2["game_key"]) == set(sgfs)


def test_train_cli_and_check_init(tmp_path, capsys):
    from tests.test_l1btrain import _tiny_cache_dir
    cache = _tiny_cache_dir(tmp_path)
    out = tmp_path / "heads_A.pt"
    base = ["--cache", str(cache), "--out", str(out), "--model-config", "b2c16", "--device", "cpu"]
    assert lt.main([*base, "--check-init"]) == 0
    printed = capsys.readouterr().out
    assert "max |Δ log p|" in printed and not out.exists()
    assert lt.main([*base, "--epochs", "1", "--batch", "8", "--lr", "1e-3"]) == 0
    assert out.exists() and out.with_suffix(".json").exists()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["train_cfg"]["epochs"] == 1 and meta["model_source"] == "config:b2c16" and meta["humanv0_sha256"] is None
    assert lt.main([*base, "--epochs", "1"]) == 2
    assert "overwrite" in capsys.readouterr().err
    # --extend keeps the checkpoint and continues the finished run with more epochs
    assert lt.main([*base, "--epochs", "2", "--batch", "8", "--lr", "1e-3", "--extend"]) == 0
    meta2 = json.loads(out.with_suffix(".json").read_text())
    assert meta2["resumed_from_step"] == meta["steps_per_epoch"] and meta2["train_cfg"]["epochs"] == 2
    assert lt.main([*base, "--epochs", "1", "--batch", "8", "--lr", "1e-3", "--overwrite"]) == 0
    assert json.loads(out.with_suffix(".json").read_text())["resumed_from_step"] == 0
    # no inner split: the CLI still reports the run
    assert lt.main([*base, "--epochs", "1", "--batch", "8", "--inner-frac", "0", "--out", str(tmp_path / "noinner.pt")]) == 0
    assert "inner NLL -" in capsys.readouterr().out or (tmp_path / "noinner.pt").exists()


def test_score_cli_matches_score_folds_layout(tmp_path, monkeypatch, capsys):
    import numpy as np
    import torch
    from katago.game.features import Features
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    import l1b_score as ls
    from aicup import l1b
    from aicup.metric import LEVELS
    from aicup.pilot import game_positions
    from aicup.scoring import read_scored_games

    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    monkeypatch.setattr(ls, "player_games", lambda: games_df)
    monkeypatch.setattr(ls, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)  # the CLI builds the same model from --model-config b2c16
    model.initialize()
    model.eval()
    heads = l1b.RankHeads(model.policy_head)
    pt = tmp_path / "heads_B.pt"
    l1b.save_heads(heads, pt, {"train_folds": [3, 4], "max_games": 40, "best_epoch": 0.5, "inner_nll_best": 1.0})
    out = tmp_path / "scores"
    args = ["--heads", str(pt), "--folds", "1", "2", "--out", str(out), "--model-config", "b2c16", "--device", "cpu", "--workers", "1",
            "--part-size", "5", "--min-games", "1", "--folds-csv", str(folds_csv)]
    assert ls.main(args) == 0
    printed = capsys.readouterr().out
    assert "fold 1" in printed and "fold 2" in printed
    features = Features(modelconfigs.config_of_name["b2c16"], 19)
    for fold in (1, 2):
        games = read_scored_games(out / f"fold{fold}")
        assert list(games.columns) == ["game_key", "player_id", "rank", "color", "n_moves"] + [f"sum_{lv}" for lv in LEVELS]
        assert len(games) == 6 and (out / f"fold{fold}" / "meta_s0.json").exists()
        sums = games[[f"sum_{lv}" for lv in LEVELS]].to_numpy()
        assert np.allclose(sums, sums[:, :1])  # ten heads at init: identical columns
        for row in games.itertuples():
            pos = game_positions(*sgfs[row.game_key], features)
            want = l1b.score_positions(model, heads, pos["bins"], pos["globs"], pos["legal"], pos["target"], pos["color"])[:, 0].sum()
            assert row.n_moves == len(pos["target"]) and abs(row.sum_D - want) < 1e-4
        npz = sorted((out / f"fold{fold}").glob("moves_s0_p*.npz"))
        assert npz
        with np.load(npz[0]) as z:
            assert z["logp"].shape[1] == 10 and z["logp"].dtype == np.float32 and set(z.files) >= {"logp", "game_row", "t"}
            assert len(z["logp"]) == len(z["game_row"]) == len(z["t"])
    meta = json.loads((out / "fold1" / "meta_s0.json").read_text())
    assert meta["heads"]["train_folds"] == [3, 4] and meta["heads"]["best_epoch"] == 0.5 and meta["fold"] == 1 and "code_sha256" in meta
    # the trunk keys are always written (None for stage-1 heads without a trunk file): check_game_vectors.py --l1b-scores matches on them
    assert "trunk_sha256" in meta["heads"] and meta["heads"]["trunk_sha256"] is None and meta["heads"]["trunk_file"] is None
    assert meta["games"] == 6 and meta["positions"] > 0 and meta["levels"] == LEVELS
    # the same command again scores nothing new
    assert ls.main(args) == 0 and "0 to score" in capsys.readouterr().out
    # heads that saw a fold are refused for it unless --allow-seen, and then only into a *_seen directory
    seen = tmp_path / "heads_seen.pt"
    l1b.save_heads(heads, seen, {"train_folds": [1, 2], "max_games": 40})
    base = ["--heads", str(seen), "--folds", "1", "--model-config", "b2c16", "--device", "cpu", "--workers", "1", "--min-games", "1",
            "--folds-csv", str(folds_csv)]
    assert ls.main([*base, "--out", str(tmp_path / "s2")]) == 2 and "train_folds" in capsys.readouterr().err
    assert ls.main([*base, "--out", str(tmp_path / "s2"), "--allow-seen"]) == 2 and "_seen" in capsys.readouterr().err
    assert ls.main([*base, "--out", str(tmp_path / "s2_seen"), "--allow-seen"]) == 0
    # a sidecar without train_folds (a cache built without meta.json) cannot prove it is out-of-fold
    import pytest
    with pytest.raises(ValueError, match="train_folds"):
        ls.check_unseen({"train_folds": None}, [1], False, str(tmp_path / "s3"))


def test_score_cli_builds_the_pool_before_the_model_touches_the_device(tmp_path, monkeypatch):
    """Forking the CPU pool after CUDA is initialised is unsupported: the trunk is loaded on the CPU, the pool
    is created, and only then is the model moved to --device."""
    import torch
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    import l1b_score as ls
    from aicup import l1b

    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    monkeypatch.setattr(ls, "player_games", lambda: games_df)
    monkeypatch.setattr(ls, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    events = []
    real_load, real_pool = ls.load_trunk, ls.make_pool

    def load_trunk(ckpt, model_config, device):
        events.append(f"load:{device}")
        model = real_load(ckpt, model_config, device)
        original_to = model.to

        def to(*a, **k):
            events.append("to")
            return original_to(*a, **k)

        model.to = to
        return model

    def make_pool(workers, config):
        events.append("pool")
        return real_pool(workers, config)

    monkeypatch.setattr(ls, "load_trunk", load_trunk)
    monkeypatch.setattr(ls, "make_pool", make_pool)
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    pt = tmp_path / "heads.pt"
    l1b.save_heads(l1b.RankHeads(model.policy_head), pt, {"train_folds": [3, 4]})
    assert ls.main(["--heads", str(pt), "--folds", "1", "--out", str(tmp_path / "s"), "--model-config", "b2c16", "--device", "cpu",
                    "--workers", "1", "--min-games", "1", "--folds-csv", str(folds_csv)]) == 0
    assert events[:3] == ["load:cpu", "pool", "to"]


def _write_scored(fold_dir, shard, k, games, logp_by_game, t_by_game, levels):
    """One part in score_folds' layout: games table with sum_<level> columns and the per-move npz."""
    import numpy as np
    from aicup.scoring import part_paths

    fold_dir.mkdir(parents=True, exist_ok=True)
    recs, logps, rows, ts = [], [], [], []
    for i, (g, lp, t) in enumerate(zip(games, logp_by_game, t_by_game)):
        recs.append({**g, "n_moves": len(t), **{f"sum_{lv}": float(s) for lv, s in zip(levels, lp.sum(axis=0))}})
        logps.append(lp)
        rows.append(np.full(len(t), i, dtype=np.int32))
        ts.append(np.asarray(t, dtype=np.int16))
    csv_path, npz_path = part_paths(fold_dir, shard, k)
    np.savez_compressed(npz_path, logp=np.concatenate(logps).astype(np.float32), game_row=np.concatenate(rows), t=np.concatenate(ts))
    pd.DataFrame(recs).to_csv(csv_path, index=False)


def _synthetic_scores(tmp_path, fold=3, players_per_rank=3, games_per_player=20, moves=5, folds=None, l1b_name="l1b"):
    """L1b (10 heads) and Human SL (18 profiles) scores of the same games: peaked at the player's rank, so the
    posterior-mean index separates adjacent ranks perfectly; returns (l1b_dir, hsl_dir, pilot peak profiles)."""
    import numpy as np
    from aicup.humansl import PROFILES
    from aicup.metric import LEVELS

    rng = np.random.RandomState(0)
    peak = {"D": "6k", "C": "5k", "B": "4k", "A": "3k", "1D": "2k", "2D": "2k", "3D": "2k", "4D": "1k", "5D": "1d", "6D": "2d"}
    l1b_dir, hsl_dir = tmp_path / l1b_name, tmp_path / "hsl"
    for fold in (folds or (fold,)):
        games, l1b_lp, hsl_lp, ts = [], [], [], []
        for r, lv in enumerate(LEVELS):
            for p in range(players_per_rank):
                pid = f"{r:04x}{fold:02x}{p:02x}"
                for g in range(games_per_player):
                    games.append({"game_key": f"{lv}:g{fold}_{r}_{p}_{g}", "player_id": pid, "rank": lv, "color": "BW"[g % 2]})
                    t = np.arange(moves) * 2 + (g % 2)
                    l1b_lp.append(-1.0 - np.abs(np.arange(10) - r)[None, :] * 0.5 + rng.randn(moves, 1) * 0.01)
                    j = PROFILES.index(peak[lv])
                    hsl_lp.append(-1.5 - np.abs(np.arange(18) - j)[None, :] * 0.3 + rng.randn(moves, 1) * 0.01)
                    ts.append(t)
        half = len(games) // 2
        _write_scored(l1b_dir / f"fold{fold}", 0, 0, games[:half], l1b_lp[:half], ts[:half], LEVELS)
        _write_scored(l1b_dir / f"fold{fold}", 1, 0, games[half:], l1b_lp[half:], ts[half:], LEVELS)
        _write_scored(hsl_dir / f"fold{fold}", 0, 0, games[half:], hsl_lp[half:], ts[half:], PROFILES)  # other shard order
        _write_scored(hsl_dir / f"fold{fold}", 0, 1, games[:half], hsl_lp[:half], ts[:half], PROFILES)
    return l1b_dir, hsl_dir, peak


def test_early_signal(tmp_path, capsys):
    import numpy as np
    import l1b_early_signal as es
    from aicup import l1beval
    from aicup.metric import LEVELS

    l1b_dir, hsl_dir, peak = _synthetic_scores(tmp_path)
    games, lp_l1b, lp_hsl, game_row, t = l1beval.align_moves(l1b_dir, hsl_dir, 3)
    assert len(games) == 600 and lp_l1b.shape == (3000, 10) and lp_hsl.shape == (3000, 18) and len(game_row) == len(t) == 3000
    assert games["game_key"].is_unique and (np.diff(game_row) >= 0).all()
    rank_idx = games["rank"].map({lv: i for i, lv in enumerate(LEVELS)}).to_numpy()[game_row]
    nll = l1beval.per_rank_nll(lp_l1b, lp_hsl, rank_idx)
    assert len(nll["l1b"]) == 10 and all(abs(v - 1.0) < 0.05 for v in nll["l1b"])  # the true head is the peak: -log p = 1.0
    assert nll["hsl_best_profile"] == [peak[lv] for lv in LEVELS] and all(abs(v - 1.5) < 0.05 for v in nll["hsl_best"])
    assert nll["hsl_pilot_peak"] == nll["hsl_best"] and nll["n_moves"] == [300] * 10
    sums_q, ranks_q, n_q = l1beval.question_sums(games, lp_l1b, game_row, n_games=20)
    assert sums_q.shape == (30, 10) and len(ranks_q) == 30 and n_q.tolist() == [100] * 30
    auc = l1beval.adjacent_auc(sums_q, ranks_q)
    assert list(auc) == [f"{a}|{b}" for a, b in zip(LEVELS, LEVELS[1:])] and all(v == 1.0 for v in auc.values())
    out = tmp_path / "early.json"
    assert es.main(["--l1b", str(l1b_dir), "--hsl", str(hsl_dir), "--folds", "3", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "NLL" in printed and "AUC" in printed and "1D|2D" in printed and "0.62" in printed
    res = json.loads(out.read_text())
    assert res["folds"] == [3] and res["nll"]["l1b"] == nll["l1b"] and res["auc_question20"]["l1b"]["3D|4D"] == 1.0
    assert res["auc_question20"]["hsl"]["1D|2D"] == 0.5  # the pilot peaks of 1D and 2D are the same profile: indistinguishable
    assert res["auc_game"]["l1b"]["D|C"] == 1.0 and res["pilot_reference"]["3D|4D"] == 0.65 and res["n_players_by_rank"]["D"] == 3


def test_misaligned_scores_are_refused(tmp_path):
    import numpy as np
    import pytest
    from aicup import l1beval
    from aicup.scoring import part_paths

    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, players_per_rank=1, games_per_player=2)
    # a game missing on one side
    csv_path, npz_path = part_paths(hsl_dir / "fold3", 0, 1)
    games = pd.read_csv(csv_path)
    games.iloc[:-1].to_csv(csv_path, index=False)
    with np.load(npz_path) as z:
        keep = z["game_row"] < len(games) - 1
        np.savez_compressed(npz_path, logp=z["logp"][keep], game_row=z["game_row"][keep], t=z["t"][keep])
    with pytest.raises(ValueError, match="game"):
        l1beval.align_moves(l1b_dir, hsl_dir, 3)
    # the same games but one move index differs
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path / "b", players_per_rank=1, games_per_player=2)
    _, npz_path = part_paths(hsl_dir / "fold3", 0, 0)
    with np.load(npz_path) as z:
        t = z["t"].copy()
        t[0] += 1
        np.savez_compressed(npz_path, logp=z["logp"], game_row=z["game_row"], t=t)
    with pytest.raises(ValueError, match="move"):
        l1beval.align_moves(l1b_dir, hsl_dir, 3)


def test_cv_cli(tmp_path, capsys):
    import l1b_cv as cv
    from aicup.metric import LEVELS

    l1b_a, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(1, 2), l1b_name="l1b_A")
    l1b_b, _, _ = _synthetic_scores(tmp_path, folds=(3, 4), l1b_name="l1b_B")
    players = []
    for fold in (1, 2, 3, 4):
        for r, lv in enumerate(LEVELS):
            for p in range(3):
                players.append({"rank": lv, "player_id": f"{r:04x}{fold:02x}{p:02x}", "n_games": 20, "fold": fold})
    folds_csv = tmp_path / "folds.csv"
    pd.DataFrame(players).to_csv(folds_csv, index=False)
    out = tmp_path / "cv_summary.json"
    base = ["--hsl", str(hsl_dir), "--folds", "1", "2", "3", "4", "--per-player", "2", "--n-boot", "200", "--folds-csv", str(folds_csv),
            "--out", str(out)]
    assert cv.main(["--l1b", str(l1b_a), str(l1b_b), *base]) == 0
    printed = capsys.readouterr().out
    assert "l1b vs hsl" in printed and "fused vs hsl" in printed
    res = json.loads(out.read_text())
    assert res["folds"] == [1, 2, 3, 4] and res["per_player"] == 2 and res["n_boot"] == 200 and res["questions"] == 4 * 30 * 7
    for name in ("hsl", "l1b", "fused"):
        fs = res["feature_sets"][name]
        assert 0.0 <= fs["balanced_mean_over_n"] <= 1.0 and len(fs["per_fold"]) == 4 and set(fs["balanced_by_n"]) == {"5", "10", "20"}
        assert fs["n_features"] == {"hsl": 24, "l1b": 16, "fused": 40}[name] and fs["nll_balanced"] > 0
    for key in ("l1b_vs_hsl", "fused_vs_hsl"):
        c = res["comparisons"][key]
        assert len(c["ci95"]) == 2 and c["ci95"][0] <= c["diff"] <= c["ci95"][1] and len(c["per_fold_diff"]) == 4
        assert set(c["conditions"]) == {"1", "2", "3", "4"} and c["conditions"]["2"] == "n/a" and isinstance(c["pass"], bool)
    assert res["comparisons"]["l1b_vs_hsl"]["diff"] > 0.05  # synthetic L1b separates 1D-3D, the Human SL peaks do not
    # a fold in two --l1b directories, or in none, is refused
    assert cv.main(["--l1b", str(l1b_a), str(l1b_a), *base]) == 2 and "more than one" in capsys.readouterr().err
    assert cv.main(["--l1b", str(l1b_a), *base]) == 2 and "3" in capsys.readouterr().err


def test_train_cli_with_several_caches_and_an_excluded_fold(tmp_path, capsys):
    import l1b_score as ls
    import pytest
    from tests.test_l1btrain import _two_caches
    c1, c2, csv = _two_caches(tmp_path)
    out = tmp_path / "heads5_f2.pt"
    base = ["--cache", str(c1), str(c2), "--folds-csv", str(csv), "--model-config", "b2c16", "--device", "cpu", "--epochs", "1", "--batch", "8"]
    assert lt.main([*base, "--exclude-folds", "2", "--out", str(out), "--check-init"]) == 0
    assert lt.main([*base, "--exclude-folds", "2", "--out", str(out)]) == 0
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["train_folds"] == [0, 1] and meta["excluded_folds"] == [2] and meta["cache_dirs"] == [str(c1), str(c2)]
    assert "train_folds [0, 1]" in capsys.readouterr().out
    # l1b_score's out-of-fold rule reads the sidecar: fold 2 may be scored, fold 1 may not
    ls.check_unseen(meta, [2], False, tmp_path / "scores5")
    with pytest.raises(ValueError, match="include"):
        ls.check_unseen(meta, [1], False, tmp_path / "scores5")
    # an excluded fold outside the caches, or two caches sharing a fold: exit 2 with the reason
    assert lt.main([*base, "--exclude-folds", "7", "--out", str(tmp_path / "x.pt")]) == 2
    assert "not in the caches" in capsys.readouterr().err
    assert lt.main(["--cache", str(c1), str(c1), *base[3:], "--out", str(tmp_path / "y.pt")]) == 2
    assert "more than one" in capsys.readouterr().err


def test_build_cache_cli_opponent_side(tmp_path, monkeypatch, capsys):
    """--opponent: the other colour's positions, labelled with the target player's rank; meta side opponent (issue #33)."""
    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    monkeypatch.setattr(bc, "player_games", lambda: games_df)
    monkeypatch.setattr(bc, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    out = tmp_path / "cache_opp"
    args = ["--folds", "1", "2", "--out", str(out), "--folds-csv", str(folds_csv), "--model-config", "b2c16", "--workers", "2",
            "--part-size", "4", "--max-games", "40", "--min-games", "1", "--opponent"]
    assert bc.main(args) == 0
    assert "OPPONENT" in capsys.readouterr().out
    games, arrays = fc.read_cache(out)
    other = {"B": "W", "W": "B"}
    want = {k: _target_moves(sgf, other[c]) for k, (sgf, c) in sgfs.items()}
    g = games.set_index("game_key")
    assert g["n_moves"].to_dict() == want and len(arrays["target"]) == sum(want.values())
    assert g.loc["D:g0", "n_moves"] == 1 and g.loc["D:g0", "color"] == "W"  # the target was Black with no move; the opponent's one White move
    assert all(g.loc[k, "color"] == other[c] for k, (_, c) in sgfs.items())
    assert g["rank"].to_dict() == games_df.set_index("game_key")["rank"].to_dict()  # labels stay the target player's
    meta = json.loads((out / "meta.json").read_text())
    assert meta["side"] == "opponent" and meta["positions"] == sum(want.values())
    # the default build says target, and metas of the two sides do not merge
    out_t = tmp_path / "cache_t"
    assert bc.main([*args[:3], "--out", str(out_t), *args[5:-1]]) == 0  # the same build without --opponent
    assert json.loads((out_t / "meta.json").read_text())["side"] == "target"
    import shutil
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    shutil.copy(out / "meta_s0.json", mixed / "meta_s0.json")
    m1 = json.loads((out_t / "meta_s0.json").read_text())
    m1["shards"] = [1]
    (mixed / "meta_s1.json").write_text(json.dumps(m1))
    import pytest
    with pytest.raises(ValueError, match="side"):
        bc.merge_meta(mixed)


def test_train_sidecar_carries_the_cache_side(tmp_path, capsys):
    from aicup import l1btrain
    from tests.test_l1btrain import _tiny_cache_dir
    cache = _tiny_cache_dir(tmp_path, name="cache_opp")
    meta_path = cache / "meta.json"
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "side": "opponent"}))
    out = tmp_path / "heads_opp.pt"
    assert lt.main(["--cache", str(cache), "--out", str(out), "--model-config", "b2c16", "--device", "cpu", "--epochs", "1", "--batch", "8"]) == 0
    assert json.loads(out.with_suffix(".json").read_text())["side"] == "opponent"
    plain = _tiny_cache_dir(tmp_path, name="cache_plain")
    out2 = tmp_path / "heads_plain.pt"
    assert lt.main(["--cache", str(plain), "--out", str(out2), "--model-config", "b2c16", "--device", "cpu", "--epochs", "1", "--batch", "8"]) == 0
    assert json.loads(out2.with_suffix(".json").read_text())["side"] == "target"
    # caches of the two sides are refused together; two of the same side carry it
    other = _tiny_cache_dir(tmp_path, name="cache_opp0", folds=(0,), ranks=("C", "3D"), players_per_rank=2, player_offset=100)
    import pytest
    with pytest.raises(ValueError, match="side"):
        l1btrain.caches_meta([cache, other])
    (other / "meta.json").write_text(json.dumps({**json.loads((other / "meta.json").read_text()), "side": "opponent"}))
    assert l1btrain.caches_meta([cache, other])["side"] == "opponent" and l1btrain.caches_meta([plain])["side" if "side" in json.loads((plain / "meta.json").read_text()) else "folds"]


def test_score_cli_refuses_opponent_side_heads_without_the_flag(tmp_path, monkeypatch, capsys):
    import torch
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    import l1b_score as ls
    from aicup import l1b

    games_df, sgfs, folds_csv = _fake_data(tmp_path)
    monkeypatch.setattr(ls, "player_games", lambda: games_df)
    monkeypatch.setattr(ls, "load_games", lambda keys: {k: sgfs[k] for k in keys})
    torch.manual_seed(0)
    model = Model(modelconfigs.config_of_name["b2c16"], 19)
    model.initialize()
    model.eval()
    pt = tmp_path / "heads_c.pt"
    l1b.save_heads(l1b.RankHeads(model.policy_head), pt, {"train_folds": [3, 4], "max_games": 40, "side": "opponent"})
    base = ["--heads", str(pt), "--folds", "1", "--model-config", "b2c16", "--device", "cpu", "--workers", "1", "--part-size", "5",
            "--min-games", "1", "--folds-csv", str(folds_csv)]
    assert ls.main([*base, "--out", str(tmp_path / "sc")]) == 2 and "--opponent" in capsys.readouterr().err
    assert ls.main([*base, "--out", str(tmp_path / "sc_opp"), "--opponent"]) == 0
    meta = json.loads((tmp_path / "sc_opp" / "fold1" / "meta_s0.json").read_text())
    assert meta["side"] == "opponent" and meta["heads"]["side"] == "opponent"
    # target-side heads (and sidecars from before the side field) score either side
    pt2 = tmp_path / "heads_t.pt"
    l1b.save_heads(l1b.RankHeads(model.policy_head), pt2, {"train_folds": [3, 4], "max_games": 40})
    assert ls.main(["--heads", str(pt2), *base[2:], "--out", str(tmp_path / "st"), "--opponent"]) == 0
    assert json.loads((tmp_path / "st" / "fold1" / "meta_s0.json").read_text())["heads"]["side"] is None
    import pytest
    with pytest.raises(ValueError, match="opponent"):
        ls.check_side({"side": "opponent"}, False)
    ls.check_side({"side": "opponent"}, True)
    ls.check_side({}, False)
