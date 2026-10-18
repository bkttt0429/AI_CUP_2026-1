import copy
import dataclasses
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

from aicup import ladder
from aicup.aggregators import CTM, XAttnGRU
from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.ladder_data import pack_questions, static_features
from aicup.metric import ADJACENT, LEVELS, RANK_TO_IDX
from aicup.qeval import balanced_mean_over_n
from aicup.qfeatures import sample_questions
from aicup.rankmodel import _softmax

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "results" / "ladder" / "vectors"
SCRIPT = ROOT / "scripts" / "ladder_cv.py"
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
