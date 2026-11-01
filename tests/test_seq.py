"""aicup.seqdata, aicup.seqmodel and scripts/seq_diag.py, seq_crossfit.py, seq_check.py (issue #19)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

import seq_check
import seq_crossfit
import seq_diag
from aicup import seqdata as sd
from aicup import seqmodel as sm
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.scoring import part_paths, read_scored_games
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores

TINY = dict(device="cpu", epochs=2, batch=16, hidden=8, d_latent=12, history=3, nlm_hidden=4, n_sync=16, evals_per_epoch=2, patience=3)


def _dirs(tmp_path):
    """Synthetic A/B-style L1b scores of folds 0-4, the Human SL scores of the same games and opponent scores whose moves sit
    between the target's (t + 1), as l1b_score.py --opponent produces."""
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    for npz in sorted(opp_dir.glob("fold*/moves_s*_p*.npz")):
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["t"] = (arrays["t"] + 1).astype(np.int16)
        np.savez_compressed(npz, **arrays)
    return l1b_dir, hsl_dir, opp_dir


def _random_seq(rng, n_games, max_steps, players=4):
    """A fold_sequences-shaped dict of random steps with varying lengths (encoder tests)."""
    n_steps = rng.randint(2, max_steps + 1, size=n_games)
    offsets = np.concatenate([[0], np.cumsum(n_steps)])
    X = rng.randn(offsets[-1], sd.STEP_DIM).astype(np.float32)
    X[:, sd.COL_TARGET] = 1.0
    X[:, sd.COL_OPP] = 0.0
    y = rng.randint(0, len(LEVELS), size=n_games)
    return {"game_key": np.array([f"g{i}" for i in range(n_games)]), "player_id": np.array([f"p{i % players}_{y[i]}" for i in range(n_games)]),
            "rank": np.array([LEVELS[c] for c in y]), "rank_idx": y.astype(np.int64), "color": np.array(["B"] * n_games),
            "n_moves": n_steps.astype(np.int64), "n_steps": n_steps.astype(np.int64), "offsets": offsets, "X": X,
            "base": rng.randn(n_games, len(LEVELS)) * 3 - 10, "games": pd.DataFrame({"game_key": [f"g{i}" for i in range(n_games)]})}


def test_fold_sequences_merges_target_and_opponent_moves_in_order(tmp_path):
    l1b_dir, hsl_dir, opp_dir = _dirs(tmp_path)
    seq = sd.fold_sequences(hsl_dir, l1b_dir, opp_dir, 3)
    games = read_scored_games(l1b_dir / "fold3")
    assert list(seq["game_key"]) == list(games["game_key"]) and seq["X"].shape == (seq["offsets"][-1], sd.STEP_DIM)
    assert np.all(seq["n_moves"] == 5) and np.all(seq["n_steps"] == 10) and np.array_equal(seq["rank_idx"], games["rank"].map(RANK_TO_IDX))
    _, l1b_moves = sd.read_moves(l1b_dir / "fold3")
    _, opp_moves = sd.read_moves(opp_dir / "fold3")
    _, hsl_moves = sd.read_moves(hsl_dir / "fold3")
    csv_sums = games[[f"sum_{lv}" for lv in LEVELS]].to_numpy(dtype=float)
    for g, key in enumerate(seq["game_key"]):
        s = seq["X"][seq["offsets"][g]:seq["offsets"][g + 1]]
        assert np.all(np.diff(s[:, sd.COL_T]) > 0), "steps in move order"
        tgt, opp = s[:, sd.COL_TARGET] == 1, s[:, sd.COL_OPP] == 1
        assert np.array_equal(tgt, ~opp) and tgt.sum() == 5 and opp.sum() == 5
        assert np.array_equal(s[tgt][:, sd.N_HSL:sd.N_HSL + sd.N_L1B], l1b_moves[key][1]) and np.array_equal(s[tgt][:, :sd.N_HSL], hsl_moves[key][1])
        assert np.array_equal(s[opp][:, sd.N_HSL:sd.N_HSL + sd.N_L1B], opp_moves[key][1]) and np.all(s[opp][:, :sd.N_HSL] == 0)
        assert np.allclose(s[tgt][:, sd.COL_T] * sd.T_SCALE, l1b_moves[key][0]) and np.allclose(s[opp][:, sd.COL_T] * sd.T_SCALE, opp_moves[key][0])
        assert np.array_equal(seq["base"][g], l1b_moves[key][1].astype(np.float64).sum(axis=0))
    assert np.abs(seq["base"] - csv_sums).max() < 1e-4
    alone = sd.fold_sequences(hsl_dir, l1b_dir, None, 3)
    assert np.all(alone["n_steps"] == alone["n_moves"]) and np.all(alone["X"][:, sd.COL_TARGET] == 1) and np.array_equal(alone["base"], seq["base"])


def test_fold_sequences_refuses_other_games_misaligned_moves_and_wrong_sums(tmp_path):
    l1b_dir, hsl_dir, opp_dir = _dirs(tmp_path)
    with pytest.raises(ValueError, match="different games"):
        _other_games(tmp_path, hsl_dir, l1b_dir, opp_dir)
    csv_path, npz_path = part_paths(l1b_dir / "fold2", 0, 0)
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["t"][0] += 1
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ValueError, match="different moves"):
        sd.fold_sequences(hsl_dir, l1b_dir, opp_dir, 2)
    t = pd.read_csv(part_paths(l1b_dir / "fold1", 0, 0)[0], dtype=str, keep_default_na=False)
    t.loc[0, "sum_1D"] = str(float(t.loc[0, "sum_1D"]) + 1.0)
    t.to_csv(part_paths(l1b_dir / "fold1", 0, 0)[0], index=False)
    with pytest.raises(ValueError, match="csv sums"):
        sd.fold_sequences(hsl_dir, l1b_dir, opp_dir, 1)
    with pytest.raises(FileNotFoundError):
        sd.read_moves(tmp_path / "empty")


def _other_games(tmp_path, hsl_dir, l1b_dir, opp_dir):
    other = tmp_path / "other_hsl"
    shutil.copytree(hsl_dir, other)
    csv = part_paths(other / "fold3", 0, 0)[0]
    t = pd.read_csv(csv, dtype=str, keep_default_na=False)
    t.loc[0, "game_key"] = "renamed"
    t.to_csv(csv, index=False)
    return sd.fold_sequences(other, l1b_dir, opp_dir, 3)


def test_concat_and_pad_batch(tmp_path):
    rng = np.random.RandomState(1)
    a, b = _random_seq(rng, 5, 7), _random_seq(rng, 3, 4)
    both = sd.concat_sequences([a, b])
    assert len(both["rank_idx"]) == 8 and np.array_equal(both["offsets"], np.concatenate([[0], np.cumsum(both["n_steps"])]))
    assert np.array_equal(both["X"][both["offsets"][5]:both["offsets"][6]], b["X"][:b["n_steps"][0]])
    X, mask, base, y = sd.pad_batch(both, [1, 6])
    assert X.shape == (2, max(both["n_steps"][[1, 6]]), sd.STEP_DIM) and mask.shape == X.shape[:2]
    assert mask.sum(axis=1).tolist() == both["n_steps"][[1, 6]].tolist() and np.all(X[~mask] == 0)
    assert np.array_equal(X[0, :both["n_steps"][1]], both["X"][both["offsets"][1]:both["offsets"][2]])
    assert np.array_equal(base, both["base"][[1, 6]]) and np.array_equal(y, both["rank_idx"][[1, 6]])


@pytest.mark.parametrize("arch", sm.ARCHS)
def test_encoders_fit_predict_round_trip_and_ignore_padding(tmp_path, arch):
    rng = np.random.RandomState(2)
    seq = _random_seq(rng, 40, 9, players=8)
    cfg = sm.SeqCfg(arch=arch, **TINY)
    model, meta = sm.fit_seq_encoder(seq, cfg, log=lambda *a, **k: None)
    pred = sm.predict_seq(model, seq, cfg)
    assert pred.shape == (40, len(LEVELS)) and np.isfinite(pred).all()
    assert meta["inner_ce_init"] is not None and meta["n_inner_players"] >= 1 and meta["n_params"] == sum(p.numel() for p in model.parameters())
    assert len(meta["history"]) >= 1 and meta["step_names_mean"][sd.COL_TARGET] == 1.0
    if arch == "sum":
        assert np.array_equal(pred, seq["base"]), "the sum encoder is the plain L1b sums"
    else:
        assert not np.allclose(pred, seq["base"]), "trained encoders move away from Δ = 0"
    path = tmp_path / f"{arch}.pt"
    sm.save_encoder(model, cfg, meta, path)
    loaded, cfg2, meta2 = sm.load_encoder(path, "cpu")
    assert cfg2 == cfg and meta2["n_params"] == meta["n_params"] and sm.state_sha256(loaded) == sm.state_sha256(model)
    assert np.array_equal(sm.predict_seq(loaded, seq, cfg2), pred)
    # a game's output does not depend on how much padding its batch carries
    short = int(np.argmin(seq["n_steps"]))
    X1, m1, b1, _ = sd.pad_batch(seq, [short])
    X2, m2, b2, _ = sd.pad_batch(seq, [short, int(np.argmax(seq["n_steps"]))])
    with torch.no_grad():
        alone = model(torch.from_numpy(X1), torch.from_numpy(m1), torch.from_numpy(b1))[0]
        padded = model(torch.from_numpy(X2), torch.from_numpy(m2), torch.from_numpy(b2))[0]
    assert torch.allclose(alone, padded, atol=1e-5), f"{arch}: padding changes the output by {(alone - padded).abs().max():.2e}"


def test_mamba3_scan_is_the_kernels_closed_form_and_rotation_keeps_the_direct_term():
    torch.manual_seed(0)
    B, T, H, P, N = 2, 7, 3, 4, 6
    q, k, v = torch.randn(B, T, H, N), torch.randn(B, T, H, N), torch.randn(B, T, H, P)
    dt, trap, D = torch.rand(B, T, H), torch.randn(B, T, H), torch.rand(H)
    adt = -torch.rand(B, T, H) * dt
    y = sm.mamba3_scan(q, k, v, adt, dt, trap, D)
    lam = torch.sigmoid(trap)
    want = torch.zeros_like(y)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                # mamba3_siso_fwd: Q K^T scaled and decayed below the diagonal, the diagonal through gamma and D
                want[b, t, h] = (D[h] + dt[b, t, h] * lam[b, t, h] * (q[b, t, h] @ k[b, t, h])) * v[b, t, h]
                for j in range(t):
                    scale = dt[b, j, h] * lam[b, j, h] + dt[b, j + 1, h] * (1 - lam[b, j + 1, h])
                    want[b, t, h] += torch.exp(adt[b, j + 1:t + 1, h].sum()) * scale * (q[b, t, h] @ k[b, j, h]) * v[b, j, h]
    assert torch.allclose(y, want, atol=1e-5)
    theta = torch.rand(B, T, H, 2)
    x = torch.randn(B, T, H, N)
    r = sm._rotate(x, torch.cos(theta), torch.sin(theta))
    assert torch.equal(r[..., 4:], x[..., 4:]) and torch.allclose(r.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.allclose(r[..., 0], x[..., 0] * torch.cos(theta[..., 0]) - x[..., 1] * torch.sin(theta[..., 0]))


def test_make_encoder_rejects_unknown_arch_and_zero_delta_at_init():
    with pytest.raises(ValueError, match="unknown arch"):
        sm.make_encoder(sm.SeqCfg(arch="lstm", device="cpu"))
    rng = np.random.RandomState(3)
    seq = _random_seq(rng, 6, 5)
    X, mask, base, _ = sd.pad_batch(seq, np.arange(6))
    for arch in sm.ARCHS:
        model = sm.make_encoder(sm.SeqCfg(arch=arch, **TINY))
        with torch.no_grad():
            out = model(torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(base))
        assert out.dtype == torch.float64 and torch.equal(out, torch.from_numpy(base)), f"{arch}: Δ is not 0 at initialisation"


def test_training_steps_stay_in_train_mode_after_inner_evaluations(monkeypatch):
    modes, make = [], sm.make_encoder

    def spied(cfg, *a):
        model = make(cfg, *a)
        model.register_forward_hook(lambda m, _in, _out: modes.append(m.training) if torch.is_grad_enabled() else None)
        return model

    monkeypatch.setattr(sm, "make_encoder", spied)
    seq = _random_seq(np.random.RandomState(0), 64, 5)
    model, meta = sm.fit_seq_encoder(seq, sm.SeqCfg(arch="gru", **TINY), log=lambda *a, **k: None)
    assert len(modes) == meta["steps_run"] > meta["steps_run"] / 2 >= 1 and all(modes), "a training step ran in eval mode"
    assert not model.training


def test_seq_crossfit_cli_writes_l1b_score_layout(tmp_path, capsys):
    l1b_dir, hsl_dir, opp_dir = _dirs(tmp_path)
    out = tmp_path / "seq_sum"
    common = ["--hsl", str(hsl_dir), "--l1b", str(l1b_dir), "--opp", str(opp_dir), "--device", "cpu", "--epochs", "1", "--batch", "64"]
    assert seq_crossfit.main(common + ["--arch", "sum", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "fold 1: 600 games, 6000 steps" in printed and "wrote" in printed
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arch"] == "sum" and sorted(summary["folds"]) == ["1", "2", "3", "4"]
    for k in (1, 2, 3, 4):
        table, ref = read_scored_games(out / f"fold{k}"), read_scored_games(l1b_dir / f"fold{k}")
        assert list(table["game_key"]) == list(ref["game_key"]) and list(table.columns) == list(ref.columns)
        cols = [f"sum_{lv}" for lv in LEVELS]
        assert np.abs(table[cols].to_numpy(dtype=float) - ref[cols].to_numpy(dtype=float)).max() < 1e-4, "the sum encoder reproduces the L1b sums"
        meta = json.loads((out / f"fold{k}" / "meta_s0.json").read_text())
        assert meta["arch"] == "sum" and meta["train_folds"] == [f for f in (1, 2, 3, 4) if f != k] and meta["side"] == "target"
        assert meta["heads"]["state_sha256"] == sm.load_encoder(out / f"encoder_f{k}.pt", "cpu")[2]["state_sha256"] and meta["mean_abs_delta"] == 0.0
        with np.load(out / f"fold{k}" / "moves_s0_p000.npz") as z:
            assert np.all(z["delta"] == 0) and z["base"].shape == (len(table), len(LEVELS))
    assert seq_crossfit.main(common + ["--arch", "sum", "--out", str(out)]) == 2, "existing outputs need --overwrite"
    # a learned encoder through the same CLI (cfg flags, --max-games) and its fold tables differ from the sums
    out_g = tmp_path / "seq_gru"
    assert seq_crossfit.main(common + ["--arch", "gru", "--out", str(out_g), "--max-games", "60", "--hidden", "8", "--folds", "1", "2", "3"]) == 0
    tab = read_scored_games(out_g / "fold2")
    assert len(tab) == 60 and json.loads((out_g / "fold2" / "meta_s0.json").read_text())["train_folds"] == [1, 3]
    assert json.loads((out_g / "summary.json").read_text())["cfg"]["hidden"] == 8
    with pytest.raises(SystemExit):
        seq_crossfit.main(common + ["--arch", "sum", "--out", str(tmp_path / "x"), "--folds", "0", "1"])
    assert seq_crossfit.main(common + ["--arch", "sum", "--out", str(tmp_path / "y"), "--l1b", str(tmp_path / "missing")]) == 2
    # one job per fold (--only-folds) gives the same tables and training sets as one job for all folds
    args_g = ["--arch", "gru", "--max-games", "60", "--hidden", "8", "--folds", "1", "2", "3"]
    out_p = tmp_path / "seq_gru_parts"
    for k in ("1", "2", "3"):
        assert seq_crossfit.main(common + args_g + ["--out", str(out_p), "--only-folds", k]) == 0
    assert seq_crossfit.main(common + args_g + ["--out", str(out_p), "--only-folds", "2"]) == 2, "existing outputs need --overwrite"
    assert sorted(json.loads((out_p / "summary.json").read_text())["folds"]) == ["1", "2", "3"]
    for k in (1, 2, 3):
        assert json.loads((out_p / f"fold{k}" / "meta_s0.json").read_text())["train_folds"] == [f for f in (1, 2, 3) if f != k]
        assert read_scored_games(out_p / f"fold{k}").equals(read_scored_games(out_g / f"fold{k}"))
    with pytest.raises(SystemExit):
        seq_crossfit.main(common + args_g + ["--out", str(tmp_path / "z"), "--only-folds", "4"])


def test_seq_check_cli(tmp_path, capsys):
    l1b_dir, hsl_dir, opp_dir = _dirs(tmp_path)
    folds_csv = _folds_csv(tmp_path)
    seq_sum = tmp_path / "seq_sum"
    assert seq_crossfit.main(["--hsl", str(hsl_dir), "--l1b", str(l1b_dir), "--opp", str(opp_dir), "--device", "cpu", "--epochs", "1",
                              "--arch", "sum", "--out", str(seq_sum)]) == 0
    for name in ("gru", "ctm_seq"):  # plumbing: copies of the sum tables stand in for trained encoders
        shutil.copytree(seq_sum, tmp_path / f"seq_{name}")
    out = tmp_path / "seq_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--n-boot", "50", "--folds-csv", str(folds_csv),
            "--seq", f"sum:{seq_sum}", f"gru:{tmp_path / 'seq_gru'}", f"ctm_seq:{tmp_path / 'seq_ctm_seq'}", "--out", str(out)]
    assert seq_check.main(args) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert set(res["scores"]) == {"fused_opp", "fused_opp_seq_sum", "fused_opp_seq_gru", "fused_opp_seq_ctm_seq"} and res["pre"] == "gru"
    assert set(res["comparisons"]["pre_specified"]) == {"fused_opp_seq_gru vs fused_opp"}
    assert set(res["comparisons"]["report_only"]) == {"fused_opp_seq_sum vs fused_opp", "fused_opp_seq_ctm_seq vs fused_opp", "fused_opp_seq_ctm_seq vs fused_opp_seq_gru"}
    for s, h in res["scores"].items():
        assert len(h["per_fold"]) == 4 and 0 <= h["balanced_mean_over_n"] <= 1 and set(h["balanced_by_n"]) == {"5", "10", "20"}
    # the sum encoder's tables are the L1b sums up to float32 rounding, so its feature set scores as fused_opp
    assert np.allclose(res["scores"]["fused_opp_seq_sum"]["per_fold"], res["scores"]["fused_opp"]["per_fold"], atol=1e-6)
    c = res["comparisons"]["pre_specified"]["fused_opp_seq_gru vs fused_opp"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool)
    assert "[pre_specified] fused_opp_seq_gru vs fused_opp" in printed and res["min_diff"] == 0.005
    assert seq_check.main(args[:-2] + ["--pre", "conv", "--out", str(out)]) == 2
    with pytest.raises(SystemExit):
        seq_check.main(args + ["--folds", "0", "1"])


def test_seq_diag_cli_and_shape_features(tmp_path, capsys):
    l1b_dir, _, _ = _dirs(tmp_path)
    flat = seq_diag.game_shape(np.tile(np.linspace(-3, -1, 10), (6, 1)))
    assert flat[seq_diag.FEATURES.index("std_pmi")] == 0 and flat[seq_diag.FEATURES.index("ac1_pmi")] == 0 and flat[seq_diag.FEATURES.index("log_moves")] == np.log(6)
    assert seq_diag.game_shape(np.zeros((0, 10))).shape == (len(seq_diag.FEATURES),)
    out = tmp_path / "seq_diag.json"
    assert seq_diag.main(["--l1b", str(l1b_dir), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert res["games"] == 2400 and res["features"] == seq_diag.FEATURES and set(res["per_rank_means"]) == set(LEVELS)
    assert len(res["per_game_logistic"]["mean_only"]) == 4 and len(res["shape_gain"]["balanced_acc"]) == 4
    assert all(r["balanced_acc"] > 0.9 for r in res["per_game_logistic"]["mean_only"]), "the synthetic per-move scores separate the ranks"
    assert "shape gain per fold" in printed
    with pytest.raises(SystemExit):
        seq_diag.main(["--l1b", str(l1b_dir), "--out", str(out), "--folds", "0"])
    assert seq_diag.main(["--l1b", str(tmp_path / "missing"), "--out", str(out)]) == 2
