import copy
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import ladder, ladder_fold0
from aicup.aggregators import CTM, XAttnGRU
from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.ladder_data import pack_questions, static_features
from aicup.metric import ADJACENT, LEVELS, RANK_TO_IDX
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff
from aicup.qfeatures import FEATURE_NAMES, SUM_COLS, build_matrix, sample_questions
from aicup.rankmodel import MLPModel, _softmax

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "results" / "ladder" / "vectors"
SCRIPT = ROOT / "scripts" / "ladder_cv.py"
FOLD0_SCRIPT = ROOT / "scripts" / "ladder_eval_fold0.py"
ALL_FOLDS = (0, 1, 2, 3, 4)
STAGED, WHOLE_NAMES = vec_names(SEGMENTS), vec_names(WHOLE)
RANKS = list(RANK_TO_IDX)
KEYS = ["question_id", "fold", "player_id", "y", "n_games"]
PCOLS = [f"p{k}" for k in range(10)]


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


# ---------------------------------------------------------------- synthetic folds

def _fold_vec(keys, ranks, rng):
    """Vectors of one synthetic fold: real-data layout, rank signal in the per-game sums."""
    n = len(keys)
    staged = rng.randn(n, len(STAGED)).astype(np.float32)
    s3, s4 = rng.randint(1, 40, size=n), rng.randint(0, 60, size=n)
    for s, k in ((1, 20), (2, 30), (3, s3), (4, s4)):
        staged[:, STAGED.index(f"s{s}_n_eff")] = k
        staged[:, STAGED.index(f"s{s}_present")] = np.asarray(k) > 0
    staged[np.ix_(s4 == 0, [i for i, name in enumerate(STAGED) if name.startswith("s4_")])] = 0.0
    n_moves = (50 + s3 + s4).astype(np.int64)
    peak = 1.8 * np.asarray(ranks) + rng.randn(n)
    curve = -2.0 - 0.02 * (np.arange(18)[None, :] - peak[:, None]) ** 2 + 0.1 * rng.randn(n, 18)
    return {"game_key": np.asarray(keys), "staged": staged, "whole": rng.randn(n, len(WHOLE_NAMES)).astype(np.float32),
            "sums": curve * n_moves[:, None], "n_moves": n_moves}


def _synthetic(tmp_path, folds=(1, 2, 3), per_rank=2, n_games=20, seed=0):
    """vectors_dir with fold{F}.npz and the matching players {fold: DataFrame[rank, player_id, game_keys]}."""
    rng = np.random.RandomState(seed)
    vec_dir = tmp_path / "vectors"
    vec_dir.mkdir()
    players = {}
    for f in folds:
        rows, keys, ranks = [], [], []
        for k, rank in enumerate(RANKS):
            for j in range(per_rank):
                gk = [f"{rank}:f{f}r{k}p{j}g{g:02d}" for g in range(n_games)]
                rows.append({"rank": rank, "player_id": f"f{f}r{k:02d}p{j}_{'ab' * 8}", "game_keys": gk})
                keys += gk
                ranks += [k] * n_games
        players[f] = pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])
        np.savez_compressed(vec_dir / f"fold{f}.npz", **_fold_vec(keys, ranks, rng))
    return vec_dir, players


def _check_frame(df, folds, n_questions):
    assert list(df.columns[:len(KEYS) + 11]) == KEYS + PCOLS + ["score"]
    P = df[PCOLS].to_numpy()
    assert np.all(np.isfinite(P)) and np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert df["score"].notna().all()
    assert set(np.round(df["score"], 12)) <= {0.0, round(ADJACENT, 12), 1.0}
    assert sorted(df["fold"].unique()) == list(folds) and len(df) == n_questions
    assert df["y"].between(0, 9).all()


def test_cv_static_and_deepsets_run_end_to_end(tmp_path, monkeypatch):
    # 3 folds: a set rung needs >= 2 training folds per split to cross-fit its base
    folds = (1, 2, 3)
    vec_dir, players = _synthetic(tmp_path, folds)
    seen = []
    orig = ladder.fit_clip_k

    def spy(packed, vec):
        seen.append({str(vec["game_key"][i]) for ix in packed["idx"] for i in ix})
        return orig(packed, vec)

    monkeypatch.setattr(ladder, "fit_clip_k", spy)
    kw = dict(folds=folds, per_player=4, device="cpu", players=players)
    out = {r: ladder.run_rung_cv(r, vec_dir, **kw) for r in ("r2_mlp24", "r1_clip", "r3_deepsets")}
    n_q = len(folds) * 20 * 7  # 20 players per fold, each 1x20 + 2x10 + 4x5 questions
    for df in out.values():
        _check_frame(df, folds, n_q)
    for v in folds:  # identical validation questions for every rung, in the same order
        ids = [df.loc[df["fold"] == v, "question_id"].tolist() for df in out.values()]
        assert ids[0] == ids[1] == ids[2]
    for r in ("r1_clip", "r3_deepsets"):
        pd.testing.assert_frame_equal(out["r2_mlp24"][KEYS], out[r][KEYS])
    # clip_k: estimated once per split, on its training folds only, and recorded
    assert len(seen) == len(folds)
    for v, keys in zip(folds, seen):
        own = {k for ks in players[v]["game_keys"] for k in ks}
        assert keys and not keys & own
    assert sorted(out["r1_clip"].attrs["clip_k"]) == ["1", "2", "3"]
    base = out["r3_deepsets"].attrs["base"]
    assert sorted(base) == ["1", "2", "3"]
    assert all(b["max_dev_unclipped"] <= 1e-12 and np.isfinite(b["ce_train"]) for b in base.values())
    assert out["r3_deepsets"].attrs["kind"] == "set" and out["r2_mlp24"].attrs["kind"] == "static"


def test_split_base_rows_follow_the_training_questions(tmp_path):
    folds, v, pp = (1, 2, 3), 2, 3
    vec_dir, players = _synthetic(tmp_path, folds, per_rank=1)
    vec, _ = ladder.load_vectors(vec_dir, folds)
    tq, vq = ladder.training_questions(players, v, folds, pp), ladder.validation_questions(players, v)
    rng = np.random.RandomState(42 + v)  # fit_rank_model.train_set: one rng, training folds ascending
    ref = {f: sample_questions(players[f], pp, rng) for f in (1, 3)}
    assert tq["game_keys"].tolist() == ref[1]["game_keys"].tolist() + ref[3]["game_keys"].tolist()
    assert tq["fold"].tolist() == [1] * len(ref[1]) + [3] * len(ref[3])
    base, base_val, info = ladder.split_base(v, vec, tq, vq, per_player=pp)
    X = {f: static_features(pack_questions(q, vec), vec, "q24") for f, q in ref.items()}
    y = {f: q["rank"].map(RANK_TO_IDX).to_numpy() for f, q in ref.items()}
    make = ladder.STATIC_RUNGS["r2_mlp24"][2]
    for f, other in ((1, 3), (3, 1)):  # each training fold's rows: the model of the other training fold
        expect = np.log(np.clip(make().fit(X[other], y[other]).predict_proba(X[f]), 1e-9, None))
        assert np.array_equal(base[tq["fold"].to_numpy() == f], expect)
    p_static = make().fit(np.vstack([X[1], X[3]]), np.concatenate([y[1], y[3]])).predict_proba(
        static_features(pack_questions(vq, vec), vec, "q24"))
    assert np.array_equal(base_val, np.log(np.clip(p_static, 1e-9, None)))  # the static rung of the split
    clipped = p_static.min(axis=1) < 1e-9
    assert info["n_clipped_rows"] == clipped.sum() > 0  # this toy data is separable enough to hit the clip
    assert info["max_dev"] == pytest.approx(np.abs(_softmax(base_val) - p_static).max(), abs=1e-15)
    assert info["max_dev_unclipped"] <= 1e-12 and info["n_train"] == len(tq) and info["n_val"] == len(vq)


def test_cv_base_cache_and_tick_extras(tmp_path, monkeypatch):
    folds = (1, 2, 3)
    vec_dir, players = _synthetic(tmp_path, folds, per_rank=1)
    tiny_ctm = partial(CTM, d_tok=8, d_latent=8, nlm_hidden=4, n_sync_action=6, n_sync_out=6, ticks=3)
    monkeypatch.setitem(ladder.SET_RUNGS, "r4_xattn", ("staged", partial(XAttnGRU, d_tok=8, d_hidden=8, ticks=3),
                                                       {"epochs": 1}))
    monkeypatch.setitem(ladder.SET_RUNGS, "r5_ctm_minmax", ("staged", tiny_ctm, {"loss": "ctm_minmax", "epochs": 1}))
    cache = tmp_path / "cv"
    kw = dict(folds=folds, per_player=2, device="cpu", players=players, base_cache=cache)
    a = ladder.run_rung_cv("r3_null", vec_dir, **kw)
    assert sorted(p.name for p in cache.glob("base_*.npz")) == [f"base_v{v}_pp2.npz" for v in folds]
    b = ladder.run_rung_cv("r3_null", vec_dir, **kw)  # from the cache
    pd.testing.assert_frame_equal(a, b)
    b2 = ladder.run_rung_cv("r3_null", vec_dir, **{**kw, "base_cache": None})
    pd.testing.assert_frame_equal(a, b2)

    x = ladder.run_rung_cv("r4_xattn", vec_dir, **kw)
    c = ladder.run_rung_cv("r5_ctm_minmax", vec_dir, **kw)
    assert {"score@T16", "p0@T16"} <= set(x.columns) and "score@certain" not in x.columns
    assert {"score@T16", "score@certain", "p9@certain"} <= set(c.columns)
    for df, extra in ((x, "T16"), (c, "T16"), (c, "certain")):
        P = df[[f"p{k}@{extra}" for k in range(10)]].to_numpy()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert df[f"score@{extra}"].notna().all()
    assert "score@T16" not in a.columns

    # a cache made for other questions is refused
    with np.load(cache / "base_v1_pp2.npz") as z:
        stale = dict(z)
    stale["questions_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(cache / "base_v1_pp2.npz", **stale)
    with pytest.raises(ValueError, match="stale"):
        ladder.run_rung_cv("r3_null", vec_dir, **kw)


def test_cv_refuses_fold0(tmp_path):
    with pytest.raises(ValueError, match="fold 0"):
        ladder.run_rung_cv("r2_mlp24", tmp_path, folds=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="fold 0"):
        ladder.load_players((0, 1))
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--rungs", "r2_mlp24", "--folds", "0", "1", "2", "--out",
                        str(tmp_path / "cv")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stderr
    assert "fold 0" in r.stdout + r.stderr
    assert not (tmp_path / "cv").exists() and not (tmp_path / "cv_summary.json").exists()


# ---------------------------------------------------------------- hand-made CV files

def _keys_table(folds=(1, 2), per_class=2, classes=(0, 1, 2)):
    """Validation questions laid out like pilot_questions: per player 1x20, 2x10, 4x5."""
    rows = []
    for f in folds:
        for k in classes:
            for j in range(per_class):
                pid = f"f{f}c{k}p{j}"
                for n in (20, 10, 5):
                    for i in range(20 // n):
                        rows.append({"question_id": f"{LEVELS[k]}_{pid}_n{n}_{i}", "fold": f, "player_id": pid, "y": k,
                                     "n_games": n})
    return pd.DataFrame(rows)


def _frame(keys, rng, rung, seed, per_player=32, counts=True, shift=0.0):
    df = keys.copy()
    P = rng.dirichlet(np.ones(10), size=len(df))
    for k in range(10):
        df[f"p{k}"] = P[:, k]
    df["score"] = np.clip(rng.choice([0.0, ADJACENT, 1.0], size=len(df)) + shift, 0, 1)
    df.attrs = {"rung": rung, "seed": seed, "per_player": per_player, "counts": counts,
                "folds": sorted(keys["fold"].unique().tolist()), "kind": "set" if rung in ladder.SET_RUNGS else "static"}
    return df


def _bal(s, y, n):
    """By hand: per size the mean over classes of the class mean, then the mean over sizes 5/10/20."""
    return np.mean([np.mean([s[(y == k) & (n == m)].mean() for k in np.unique(y[n == m])]) for m in (5, 10, 20)])


@pytest.mark.filterwarnings("error::RuntimeWarning")  # the summary of a fold set missing classes stays quiet
def test_summarize_pools_folds_and_seed_mean(tmp_path):
    rng = np.random.RandomState(3)
    keys = _keys_table()
    cv = tmp_path / "cv"
    cv.mkdir()
    frames = {("r2_mlp24", 0, 32): _frame(keys, rng, "r2_mlp24", 0),
              ("r2_mlp24", 0, 8): _frame(keys, rng, "r2_mlp24", 0, per_player=8)}
    for s in (0, 1, 2):
        frames[("r3_settf", s, 32)] = _frame(keys, rng, "r3_settf", s)
    frames[("r3_deepsets", 0, 32)] = _frame(keys, rng, "r3_deepsets", 0)  # a set rung with only seed 0 so far
    for (rung, s, pp), df in frames.items():
        ladder.save_cv(df, cv / ladder.cv_filename(rung, s, per_player=pp))
    back, meta = ladder.load_cv(cv / ladder.cv_filename("r3_settf", 1))
    pd.testing.assert_frame_equal(back, frames[("r3_settf", 1, 32)], check_dtype=False)
    assert meta["rung"] == "r3_settf" and meta["seed"] == 1
    with np.load(cv / ladder.cv_filename("r3_settf", 1), allow_pickle=False) as z:
        assert z["question_id"].dtype.kind == "U" and z["player_id"].dtype.kind == "U"

    S = ladder.summarize(cv, n_boot=200)
    y, n, fold = keys["y"].to_numpy(), keys["n_games"].to_numpy(), keys["fold"].to_numpy()
    st = np.stack([frames[("r3_settf", s, 32)]["score"].to_numpy() for s in (0, 1, 2)])
    s_mean, s_ref = st.mean(axis=0), frames[("r2_mlp24", 0, 32)]["score"].to_numpy()

    r = S["rungs"]["r3_settf"]
    assert r["seeds"] == [0, 1, 2] and r["kind"] == "set"
    assert r["balanced_mean_over_n"] == pytest.approx(_bal(s_mean, y, n), abs=1e-12)
    for f in (1, 2):
        assert r["per_fold"][str(f)] == pytest.approx(_bal(s_mean[fold == f], y[fold == f], n[fold == f]), abs=1e-12)
    per_seed = [_bal(st[s], y, n) for s in range(3)]
    assert [r["per_seed"][str(s)] for s in range(3)] == pytest.approx(per_seed, abs=1e-12)
    assert r["seed_sd"] == pytest.approx(np.std(per_seed, ddof=1), abs=1e-12)
    assert r["per_class"]["D"] == pytest.approx(s_mean[y == 0].mean(), abs=1e-12) and r["per_class"]["6D"] is None
    nll = np.mean([-np.log(np.maximum(frames[("r3_settf", s, 32)][PCOLS].to_numpy()[np.arange(len(y)), y], 1e-12))
                   for s in range(3)], axis=0)
    assert r["nll_balanced"] == pytest.approx(np.mean([nll[y == k].mean() for k in range(3)]), abs=1e-12)
    brier = np.mean([((frames[("r3_settf", s, 32)][PCOLS].to_numpy() - np.eye(10)[y]) ** 2).sum(axis=1) for s in range(3)],
                    axis=0)
    assert r["brier"] == pytest.approx(np.mean([brier[y == k].mean() for k in range(3)]), abs=1e-12)
    assert r["balanced_by_n"]["5"] == pytest.approx(np.mean([s_mean[(y == k) & (n == 5)].mean() for k in range(3)]), abs=1e-12)
    assert S["rungs"]["r2_mlp24"]["seed_sd"] is None and S["rungs"]["r2_mlp24"]["kind"] == "static"
    assert "r2_mlp24@pp8" in S["rungs"]

    c = S["comparisons"]["r3_settf vs r2_mlp24"]
    assert c["diff"] == pytest.approx(_bal(s_mean, y, n) - _bal(s_ref, y, n), abs=1e-12)
    for f in (1, 2):
        m = fold == f
        assert c["per_fold"][str(f)] == pytest.approx(_bal(s_mean[m], y[m], n[m]) - _bal(s_ref[m], y[m], n[m]), abs=1e-12)
    assert [c["per_seed"][str(s)] for s in range(3)] == pytest.approx([p - _bal(s_ref, y, n) for p in per_seed], abs=1e-12)
    lo, hi = c["ci95"]
    k = math.sqrt(2.33)
    assert c["nb_ci95"] == pytest.approx([c["diff"] - k * (c["diff"] - lo), c["diff"] + k * (hi - c["diff"])], abs=1e-12)
    assert c["nb_ci95"][0] <= lo <= hi <= c["nb_ci95"][1]
    assert (c["kind_a"], c["kind_b"], c["seeds_a"], c["seeds_b"]) == ("set", "static", [0, 1, 2], [0])
    pp8 = S["comparisons"]["r2_mlp24@pp8 vs r2_mlp24"]
    assert list(pp8["per_seed"]) == ["0"] and (pp8["kind_a"], pp8["kind_b"]) == ("static", "static")
    # set vs set with one side at seed 0 only: one matched seed, and the per-seed condition fails
    d = S["comparisons"]["r3_settf vs r3_deepsets"]
    assert list(d["per_seed"]) == ["0"] and d["seeds_b"] == [0]
    assert d["per_seed"]["0"] == pytest.approx(per_seed[0] - _bal(frames[("r3_deepsets", 0, 32)]["score"].to_numpy(), y, n),
                                               abs=1e-12)
    assert ladder.check_comparison({**d, "n_boot": 5000})["conditions"]["every_seed_gt_0"] is False
    # a 200-replicate summary cannot decide a gate
    assert c["n_boot"] == 200 and not ladder.check_comparison(c)["pass"]
    assert any("n_boot" in why for why in ladder.check_comparison(c)["reasons"])
    json.loads(ladder.to_json({**S, "gates": ladder.gates(S)}))  # strict JSON: no NaN / Infinity

    bad = frames[("r3_settf", 2, 32)].iloc[::-1].reset_index(drop=True)
    bad.attrs = frames[("r3_settf", 2, 32)].attrs
    ladder.save_cv(bad, cv / ladder.cv_filename("r3_settf", 2))
    with pytest.raises(ValueError, match="r3_settf"):
        ladder.summarize(cv, n_boot=50)


# ---------------------------------------------------------------- gates

def _cmp(diff=0.01, lo=0.002, folds=(0.01, 0.02, 0.005, -0.001), seeds=(0.01, 0.012, 0.008), kinds=("set", "set"),
         seeds_a=None, seeds_b=None, n_boot=5000):
    """A comparison dict as summarize writes it; a static side has seed [0], a set side one seed per per-seed value."""
    def side(kind, given):
        return given if given is not None else [0] if kind == "static" else list(range(len(seeds)))

    return {"diff": diff, "ci95": [lo, diff + 0.01], "per_fold": {str(f): d for f, d in zip((1, 2, 3, 4), folds)},
            "per_seed": {str(s): d for s, d in enumerate(seeds)}, "kind_a": kinds[0], "kind_b": kinds[1],
            "seeds_a": side(kinds[0], seeds_a), "seeds_b": side(kinds[1], seeds_b), "n_boot": n_boot}


def test_gates_logic():
    pairs = [p for gate in ladder.GATES.values() for p in gate]
    assert set(ladder.GATES) == {"G_rep", "G_set", "G_iter", "G_sync"}
    assert ladder.GATES["G_set"] == [("r3_settf", "r3_deepsets"), ("r3_settf", "r2_pooled_delta")]
    summary = {"comparisons": {f"{a} vs {b}": _cmp() for a, b in pairs}}
    summary["comparisons"]["r2_staged vs r2_whole"] = _cmp(seeds=(0.01,), kinds=("static", "static"))  # one "seed"
    g = ladder.gates(summary)
    assert all(g[name]["pass"] for name in ladder.GATES)

    cases = {"ci_lo_gt_0": dict(lo=0.0), "every_seed_gt_0": dict(seeds=(0.01, -0.001, 0.02)),
             "folds_gt_0_3of4": dict(folds=(0.01, -0.01, 0.02, -0.001)), "diff_ge_0.005": dict(diff=0.0049)}
    for cond, kw in cases.items():
        s = copy.deepcopy(summary)
        s["comparisons"]["r3_settf vs r2_pooled_delta"] = _cmp(**kw)
        g = ladder.gates(s)
        c = g["G_set"]["comparisons"]["r3_settf vs r2_pooled_delta"]
        assert not c["pass"] and not g["G_set"]["pass"], cond
        assert c["conditions"] == {k: k != cond for k in cases}, cond
        assert g["G_set"]["comparisons"]["r3_settf vs r3_deepsets"]["pass"]
        assert g["G_rep"]["pass"] and g["G_iter"]["pass"] and g["G_sync"]["pass"]

    def seeds_ok(c):
        return ladder.check_comparison(c)["conditions"]["every_seed_gt_0"]

    # set vs set: each side needs its own 3 seeds; a 1-seed side must not be reused against the other's 3
    assert not seeds_ok(_cmp(seeds=(0.01, 0.02, 0.03), seeds_b=[0]))  # three per-seed values, but b ran s0 only
    assert not seeds_ok(_cmp(seeds=(0.01,), seeds_b=[0]))
    assert not seeds_ok(_cmp(seeds=(0.01, 0.02), seeds_b=[0, 1]))
    assert not seeds_ok(_cmp(seeds=(0.01, 0.02, 0.03), seeds_a=[0, 1]))
    r = ladder.check_comparison(_cmp(seeds=(0.01, 0.02, 0.03), seeds_b=[0]))
    assert any("1 seed" in why for why in r["reasons"]), r["reasons"]
    # a static side (deterministic, seed 0) is compared with each of the set side's 3 seeds
    assert seeds_ok(_cmp(kinds=("static", "set"))) and seeds_ok(_cmp(kinds=("set", "static")))
    assert not seeds_ok(_cmp(seeds=(0.01, 0.02), kinds=("set", "static")))
    assert not seeds_ok(_cmp(seeds=(0.01, -0.02, 0.01), kinds=("static", "set")))

    s = copy.deepcopy(summary)  # 3 of 3 folds is not 3 of 4
    s["comparisons"]["r5_ctm vs r4_xattn"]["per_fold"] = {"1": 0.01, "2": 0.01, "3": 0.01}
    assert not ladder.gates(s)["G_sync"]["pass"]
    s = copy.deepcopy(summary)  # missing results fail, and say so
    del s["comparisons"]["r4_xattn vs r3_settf"]
    g = ladder.gates(s)
    assert not g["G_iter"]["pass"] and g["G_iter"]["comparisons"]["r4_xattn vs r3_settf"]["missing"]
    assert g["G_iter"]["comparisons"]["r5_ctm vs r3_settf"]["pass"]
    # an interval from fewer than 5000 bootstrap replicates cannot decide a gate
    r = ladder.check_comparison(_cmp(n_boot=200))
    assert not r["pass"] and all(r["conditions"].values()) and any("n_boot" in why for why in r["reasons"])


def test_set_rungs_registry():
    from aicup.aggregators import DeepSets, PooledDelta, QglobDelta, SetTransformer, n_params
    from aicup.settrain import LOSSES, TrainCfg

    expect = {"r2_pooled_delta": (PooledDelta, "staged", "final"), "r3_null": (QglobDelta, "staged", "final"),
              "r3_deepsets": (DeepSets, "staged", "final"), "r3_settf": (SetTransformer, "staged", "final"),
              "r3_deepsets_whole": (DeepSets, "whole", "final"), "r4_xattn": (XAttnGRU, "staged", "final"),
              "r5_ctm": (CTM, "staged", "final"), "r5_ctm_minmax": (CTM, "staged", "ctm_minmax")}
    assert set(ladder.SET_RUNGS) == set(expect) and ladder.RUNGS == ladder.STATIC_RUNGS | ladder.SET_RUNGS
    assert set(ladder.EXTRAS) <= set(ladder.SET_RUNGS)
    d_in = {"staged": len(STAGED) + 1, "whole": len(WHOLE_NAMES) + 1}  # token width: the vector + log1p(n_moves)
    models = {}
    for name, (rep, make, overrides) in ladder.SET_RUNGS.items():
        cls, want_rep, want_loss = expect[name]
        models[name] = make(d_in[rep])
        assert type(models[name]) is cls and rep == want_rep, name
        loss = dataclasses.replace(TrainCfg(), **overrides).loss
        assert loss == want_loss and loss in LOSSES, name
    ratio = n_params(models["r4_xattn"]) / n_params(models["r5_ctm"])
    assert abs(ratio - 1) <= 0.1, ratio


# ---------------------------------------------------------------- CLI gating

def _run_cli(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=600)


def test_force_required_for_r4_without_g_set(tmp_path):
    cv = tmp_path / "cv"
    r = _run_cli("--rungs", "r4_xattn", "--out", cv, "--vectors", tmp_path / "none")
    assert r.returncode == 2, r.stderr
    assert "r3_settf" in r.stdout + r.stderr and "--force" in r.stdout + r.stderr

    rng = np.random.RandomState(0)
    keys = _keys_table(folds=(1, 2, 3, 4))
    cv.mkdir()
    for rung, shift in (("r3_settf", 0.0), ("r3_deepsets", 0.3), ("r2_pooled_delta", 0.3)):
        for s in (0, 1, 2):
            ladder.save_cv(_frame(keys, rng, rung, s, shift=shift), cv / ladder.cv_filename(rung, s))
    r = _run_cli("--rungs", "r2_mlp24", "r5_ctm", "--out", cv, "--vectors", tmp_path / "none")
    assert r.returncode == 2, r.stderr
    assert "G_set" in r.stdout + r.stderr
    assert not list(cv.glob("r2_mlp24*")) and not list(cv.glob("r5_ctm*"))

    r = _run_cli("--rungs", "r5_ctm", "--force", "--out", cv, "--vectors", tmp_path / "none")
    assert r.returncode not in (0, 2) and "fold1.npz" in r.stderr  # past the gate; stops at the missing vectors

    r = _run_cli("--summarize-only", "--out", cv)
    assert r.returncode == 0, r.stderr
    summary = json.loads((tmp_path / "cv_summary.json").read_text(),
                         parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert summary["gates"]["G_set"]["pass"] is False and set(summary["rungs"]) == {"r3_settf", "r3_deepsets", "r2_pooled_delta"}


def test_summary_path_follows_out(tmp_path):
    out = tmp_path / "cv_pp8"  # a sibling --out must not overwrite the main cv_summary.json
    df = _frame(_keys_table(folds=(1, 2, 3, 4)), np.random.RandomState(1), "r2_mlp24", 0, per_player=8)
    ladder.save_cv(df, out / ladder.cv_filename("r2_mlp24", 0, per_player=8))
    r = _run_cli("--summarize-only", "--out", out)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "cv_pp8_summary.json").exists() and not (tmp_path / "cv_summary.json").exists()
    assert set(json.loads((tmp_path / "cv_pp8_summary.json").read_text())["rungs"]) == {"r2_mlp24@pp8"}
    assert "<out.name>_summary.json" in _run_cli("--help").stdout


# ---------------------------------------------------------------- real data

def _real_inputs():
    return (all((VECTORS / f"fold{f}.npz").exists() for f in (1, 2, 3, 4)) and (ROOT / "results" / "rank_model_eval.json").exists()
            and (ROOT / "data" / "training").is_dir() and (ROOT / "splits" / "player_folds.csv").exists())


@pytest.mark.skipif(not _real_inputs(), reason="real fold 1-4 vectors, training CSVs or rank_model_eval.json absent")
def test_reproduces_rank_model_eval_per_fold():
    ref = json.loads((ROOT / "results" / "rank_model_eval.json").read_text())
    assert ref["per_player"] == 8 and ref["seed"] == 42 and ref["train_folds"] == [1, 2, 3, 4]
    df = ladder.run_rung_cv("r2_mlp24", VECTORS, per_player=8, device="cpu")
    got = [balanced_mean_over_n(d["score"], d["y"], d["n_games"]) for _, d in df.groupby("fold", sort=True)]
    assert sorted(df["fold"].unique()) == [1, 2, 3, 4]
    assert got == pytest.approx(ref["cv"]["mlp_h32"]["per_fold"], abs=1e-9)


# ---------------------------------------------------------------- fold 0 (one shot)

def _games_frame(vec):
    """qfeatures.build_matrix's games frame from the vectors (their sums and n_moves are the score CSVs' values)."""
    return pd.DataFrame({"game_key": vec["game_key"], **{c: vec["sums"][:, i] for i, c in enumerate(SUM_COLS)},
                         "n_moves": vec["n_moves"]})


def _production_doc(players, vec, pp):
    """A production model made as fit_rank_model makes its final model -- one RandomState(42), folds 1-4
    ascending, pp questions per player, MLPModel(hidden=32) on build_matrix features -- after the JSON
    round trip of rank_model.json."""
    rng = np.random.RandomState(42)
    qs = pd.concat([sample_questions(players[f], pp, rng) for f in (1, 2, 3, 4)], ignore_index=True)
    X, y, _ = build_matrix(_games_frame(vec), qs)
    doc = {"name": "mlp_h32", "features": FEATURE_NAMES, "train_folds": [1, 2, 3, 4],
           "state": MLPModel(hidden=32).fit(X, y).state_dict()}
    return json.loads(json.dumps(doc))


def _fold0_case(tmp_path, labels, pp=4, per_rank=2):
    """Synthetic folds 0-4 with their players, a CV summary holding `labels` and a production model file."""
    vec_dir, players = _synthetic(tmp_path, ALL_FOLDS, per_rank=per_rank)
    vec, _ = ladder.load_vectors(vec_dir, ALL_FOLDS)
    summary = tmp_path / "cv_summary.json"
    summary.write_text(json.dumps({"rungs": {lab: {"balanced_mean_over_n": 0.5, "seeds": [0]} for lab in labels},
                                   "comparisons": {}}))
    model = tmp_path / "rank_model.json"
    model.write_text(json.dumps(_production_doc(players, vec, pp)))
    return vec_dir, players, vec, summary, model


def _run_fold0_cli(*args):
    return subprocess.run([sys.executable, "-I", str(FOLD0_SCRIPT), *map(str, args)], capture_output=True, text=True,
                          timeout=600)


def _strict(text):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_refuses_second_fold0_eval(tmp_path, monkeypatch):
    vec_dir, players, _, summary, model = _fold0_case(tmp_path, ["r2_mlp24@pp4"])
    out = tmp_path / "ladder" / "fold0_eval.json"
    kw = dict(out=out, chosen_from=summary, vectors_dir=vec_dir, rank_model=model, per_player=4, device="cpu", n_boot=50)
    res = ladder_fold0.run("r2_mlp24", players=players, **kw)
    first = out.read_bytes()
    saved = _strict(first)
    assert saved == _strict(ladder.to_json(res))
    assert [p.name for p in out.parent.iterdir()] == ["fold0_eval.json"]  # written atomically, no temp file left
    # provenance: what was evaluated, chosen from which summary (path and sha256), at which commit
    assert (saved["rung"], saved["kind"], saved["seeds"], saved["per_player"], saved["counts"]) == ("r2_mlp24", "static", None,
                                                                                                    4, True)
    assert saved["chosen_from"] == str(summary.resolve()) and saved["cv_label"] == "r2_mlp24@pp4"
    assert saved["cv_summary_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert saved["production_model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert saved["eval_fold"] == 0 and saved["train_folds"] == [1, 2, 3, 4] and saved["n_questions"] == 20 * 7
    assert "git_head" in saved and saved["headline"]["n_boot"] == 50 and len(saved["headline"]["ci95"]) == 2
    assert "r2_mlp24@pp8" in saved["note_production"] and saved["cv_pp8_vs_base"] is None
    assert ladder_fold0.repo_path(ROOT / "results" / "ladder" / "cv_summary.json") == "results/ladder/cv_summary.json"

    def no_loading(*a, **k):
        raise AssertionError("data loaded although fold 0 was evaluated already")

    monkeypatch.setattr(ladder_fold0, "load_inputs", no_loading)
    with pytest.raises(ladder_fold0.Refused, match="fold 0"):
        ladder_fold0.run("r2_mlp24", players=players, **kw)
    # the CLI exits with 2 before reading anything (its vectors directory does not even exist)
    r = _run_fold0_cli("--rung", "r2_mlp24", "--per-player", 4, "--out", out, "--chosen-from", summary,
                       "--vectors", tmp_path / "none", "--rank-model", model)
    assert r.returncode == 2, r.stderr
    assert "fold 0" in r.stderr and str(out) in r.stderr
    # a run that reaches the end after another one wrote the file does not overwrite it either
    with pytest.raises(FileExistsError):
        ladder_fold0.write_once(out, {"rung": "other"})
    assert out.read_bytes() == first and [p.name for p in out.parent.iterdir()] == ["fold0_eval.json"]


def test_fold0_paired_alignment():
    rng = np.random.RandomState(5)
    keys = _keys_table(folds=(0,), per_class=3)
    a = keys.assign(score=rng.choice([0.0, ADJACENT, 1.0], size=len(keys)))
    b = keys.assign(score=rng.choice([0.0, ADJACENT, 1.0], size=len(keys)))
    d = ladder_fold0.paired(a, b, "a vs b", n_boot=200)
    ref = paired_diff(a["score"], b["score"], keys["y"], keys["player_id"], keys["n_games"], n=200)
    assert (d["diff"], d["ci95"], d["by_n"], d["n_boot"]) == (ref["diff"], ref["ci95"], ref["by_n"], 200)
    assert ladder_fold0.paired(a, a, "a vs a", n_boot=50)["diff"] == 0.0
    shuffled = b.sample(frac=1.0, random_state=0).reset_index(drop=True)  # same questions, another order
    dropped = b.iloc[1:].reset_index(drop=True)
    renamed = b.assign(question_id=b["question_id"].where(b.index != 3, "other_question"))
    for bad in (shuffled, dropped, renamed):
        with pytest.raises(ValueError, match="question_id"):
            ladder_fold0.paired(a, bad, "a vs bad", n_boot=50)
        with pytest.raises(ValueError, match="question_id"):
            ladder_fold0.paired(bad, a, "bad vs a", n_boot=50)


def test_fold0_static_rung_is_fitted_once(tmp_path, monkeypatch):
    _, players, vec, _, model = _fold0_case(tmp_path, [])
    doc = _strict(model.read_text())
    calls = []
    orig = ladder_fold0._static_split

    def spy(rung, vec, train_q, val_q, counts):
        calls.append(rung)
        assert sorted(train_q["fold"].unique()) == [1, 2, 3, 4] and set(val_q["fold"]) == {0}
        return orig(rung, vec, train_q, val_q, counts)

    monkeypatch.setattr(ladder_fold0, "_static_split", spy)
    kw = dict(production=doc, seeds=(0, 1, 2), per_player=4, device="cpu", n_boot=50)
    res = ladder_fold0.evaluate_fold0("r2_mlp24", vec, players, **kw)
    assert calls == ["r2_mlp24"]  # deterministic: fitted once, whatever the seeds, and it is its own base
    assert (res["kind"], res["seeds"], res["per_player"], res["counts"]) == ("static", None, 4, True)
    assert res["n_questions"] == 20 * 7 and res["n_players"] == 20 and res["n_train_questions"] == 4 * 20 * 4
    h = res["headline"]
    assert h["ci95"][0] <= h["balanced_mean_over_n"] <= h["ci95"][1] and h["n_boot"] == 50 and "per_seed" not in h
    assert res["vs_base"]["diff"] == 0.0 and res["vs_base"]["ci95"] == [0.0, 0.0]
    # retrained on fold 0's split exactly as fit_rank_model trains its final model: the production model itself
    assert res["vs_production"]["diff"] == 0.0 and res["vs_production"]["ci95"] == [0.0, 0.0]
    assert h["balanced_mean_over_n"] == res["production"]["balanced_mean_over_n"] == res["base"]["balanced_mean_over_n"]

    calls.clear()
    res = ladder_fold0.evaluate_fold0("r0_sum", vec, players, **kw)
    assert calls == ["r0_sum", "r2_mlp24"] and res["seeds"] is None  # the rung once, then its base: r2_mlp24 refitted
    assert res["vs_base"]["diff"] == pytest.approx(res["headline"]["balanced_mean_over_n"] - res["base"]["balanced_mean_over_n"],
                                                   abs=1e-12)
    assert res["base"]["balanced_mean_over_n"] == res["production"]["balanced_mean_over_n"]


def test_fold0_set_rung_seed_mean(tmp_path, monkeypatch):
    _, players, vec, _, model = _fold0_case(tmp_path, [], pp=2)
    doc = _strict(model.read_text())
    runs, crossfits = {}, []
    orig_split, orig_crossfit = ladder_fold0._set_split, ladder.crossfit_base_logits

    def spy_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, cache):
        out, info = orig_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, cache)
        assert v == 0 and sorted(train_q["fold"].unique()) == [1, 2, 3, 4] and set(val_q["fold"]) == {0}
        runs[seed] = (val_q, out[""])
        return out, info

    def spy_crossfit(*a, **k):
        crossfits.append(1)
        return orig_crossfit(*a, **k)

    monkeypatch.setattr(ladder_fold0, "_set_split", spy_split)
    monkeypatch.setattr(ladder, "crossfit_base_logits", spy_crossfit)
    res = ladder_fold0.evaluate_fold0("r3_null", vec, players, production=doc, seeds=(0, 1), per_player=2, device="cpu",
                                      n_boot=50)
    assert sorted(runs) == [0, 1] and len(crossfits) == 1  # one training per seed; one 4-fold base shared by the seeds
    assert (res["kind"], res["seeds"]) == ("set", [0, 1])
    val_q = runs[0][0]
    y, n = val_q["rank"].map(RANK_TO_IDX).to_numpy(), val_q["n_games"].to_numpy()
    scores = {s: evaluate(P, y, val_q)[1] for s, (_, P) in runs.items()}
    s_mean = (scores[0] + scores[1]) / 2  # per question the mean over seeds, as the CV summary
    h = res["headline"]
    assert h["balanced_mean_over_n"] == pytest.approx(balanced_mean_over_n(s_mean, y, n), abs=1e-12)
    assert h["per_seed"] == pytest.approx({str(s): balanced_mean_over_n(scores[s], y, n) for s in (0, 1)}, abs=1e-12)
    # the base is the 4-fold cross-fit's validation base (Δ ≡ 0) of the same training questions
    tq, vq = ladder.training_questions(players, 0, (1, 2, 3, 4), 2), ladder.validation_questions(players, 0)
    _, base_val, _ = ladder.split_base(0, vec, tq, vq, per_player=2)
    s_base = evaluate(_softmax(base_val), y, vq)[1]
    assert res["base"]["balanced_mean_over_n"] == pytest.approx(balanced_mean_over_n(s_base, y, n), abs=1e-12)
    for key, ref in (("vs_base", res["base"]), ("vs_production", res["production"])):
        c = res[key]
        assert c["diff"] == pytest.approx(h["balanced_mean_over_n"] - ref["balanced_mean_over_n"], abs=1e-12)
        assert c["per_seed"] == pytest.approx({s: h["per_seed"][s] - ref["balanced_mean_over_n"] for s in ("0", "1")},
                                              abs=1e-12)
    json.loads(ladder.to_json(res))


def test_fold0_cli_refusals_before_loading(tmp_path):
    out, summary = tmp_path / "fold0_eval.json", tmp_path / "cv_summary.json"
    common = ["--out", out, "--chosen-from", summary, "--vectors", tmp_path / "none", "--rank-model", tmp_path / "none.json"]
    r = _run_fold0_cli("--rung", "r3_null", *common)  # the summary the rung was chosen from is missing
    assert r.returncode == 2, r.stderr
    assert str(summary) in r.stderr
    summary.write_text(json.dumps({"rungs": {"r3_null@nocounts": {}, "r2_mlp24@pp8": {}}}))
    for args, label in ((["--rung", "r3_null"], "r3_null"), (["--rung", "r2_mlp24"], "r2_mlp24"),
                        (["--rung", "r2_mlp24", "--per-player", 16], "r2_mlp24@pp16")):
        r = _run_fold0_cli(*args, *common)  # no CV results for this configuration
        assert r.returncode == 2, r.stderr
        assert f"'{label}'" in r.stderr
    for args in (["--rung", "r3_null", "--no-counts"], ["--rung", "r2_mlp24", "--per-player", 8]):
        r = _run_fold0_cli(*args, *common)  # the label is in the summary: past the refusals, stops at the missing inputs
        assert r.returncode not in (0, 2), r.stderr
        assert "fold0.npz" in r.stderr and "none.json" in r.stderr
    out.write_text("{}")
    summary.unlink()
    r = _run_fold0_cli("--rung", "r3_null", *common)  # an existing output is refused first
    assert r.returncode == 2, r.stderr
    assert "fold 0" in r.stderr and out.read_text() == "{}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fold0_eval.json"]


def _fold0_skip_reason():
    need = {"results/ladder/vectors/fold0.npz (scripts/ladder_build_vectors.py --folds 0)": VECTORS / "fold0.npz",
            "results/rank_model_eval.json": ROOT / "results" / "rank_model_eval.json",
            "results/models/rank_model.json": ROOT / "results" / "models" / "rank_model.json",
            "data/training (the training CSVs)": ROOT / "data" / "training",
            "splits/player_folds.csv": ROOT / "splits" / "player_folds.csv"}
    missing = [name for name, path in need.items() if not path.exists()]
    return f"absent: {', '.join(missing)}" if missing else ""


@pytest.mark.skipif(bool(_fold0_skip_reason()), reason=_fold0_skip_reason())
def test_fold0_production_reproduces_rank_model_eval():
    # the production model's fold-0 score was reported already (rank_model_eval.json): this trains no rung and
    # spends nothing; it pins the fold-0 loading path of ladder_fold0 (players, vectors, pilot questions, features)
    ref = _strict((ROOT / "results" / "rank_model_eval.json").read_text())
    assert ref["eval_fold"] == 0 and ref["choice"] == "mlp_h32"
    want = ref["eval"]["mlp_h32"]["balanced_mean_over_n"]
    assert want == 0.6130870708449819
    doc = _strict(ladder_fold0.RANK_MODEL.read_text())
    vec, fold_of, players = ladder_fold0.load_inputs(VECTORS, folds=(0,))
    assert set(fold_of) == {0} and list(players) == [0]
    df = ladder_fold0.production_frame(doc, vec, ladder.validation_questions(players, 0))
    got = balanced_mean_over_n(df["score"], df["y"], df["n_games"])
    assert abs(got - want) <= 1e-12, got - want
    assert len(df) == np.sum(ref["eval"]["mlp_h32"]["confusion_rows_truth_cols_pred"]) and set(df["fold"]) == {0}
