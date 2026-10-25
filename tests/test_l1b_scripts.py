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
    assert lt.main([*base, "--epochs", "1", "--batch", "8", "--lr", "1e-3", "--overwrite"]) == 0
