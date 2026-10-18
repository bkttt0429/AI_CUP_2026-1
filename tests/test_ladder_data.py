from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.ladder_data import QGLOB_NAMES, STATIC_RUNGS, fit_clip_k, pack_questions, static_features
from aicup.metric import RANK_TO_IDX
from aicup.qfeatures import FEATURE_NAMES, SUM_COLS, build_matrix
from aicup.rankmodel import MLPModel, SoftmaxModel

VECTORS = Path(__file__).resolve().parents[1] / "results" / "ladder" / "vectors"
STAGED, WHOLE_NAMES = vec_names(SEGMENTS), vec_names(WHOLE)
KINDS = ("sum", "clip", "trim", "q24", "pooled", "length")


def _vec(n_games=60, seed=0):
    """Synthetic vector dict in the Task 1 layout; per-segment n_eff add up to n_moves."""
    rng = np.random.RandomState(seed)
    n_moves = rng.randint(15, 160, size=n_games).astype(np.int64)
    staged = rng.randn(n_games, len(STAGED)).astype(np.float32)
    whole = rng.randn(n_games, len(WHOLE_NAMES)).astype(np.float32)
    caps = (20, 30, 40, None)  # target moves per stage when the target plays every other move
    for g, n in enumerate(n_moves):
        left = int(n)
        for s, cap in enumerate(caps, start=1):
            k = left if cap is None else min(left, cap)
            left -= k
            block = [i for i, name in enumerate(STAGED) if name.startswith(f"s{s}_")]
            if k == 0:
                staged[g, block] = 0.0
            else:
                staged[g, STAGED.index(f"s{s}_n_eff")] = k
                staged[g, STAGED.index(f"s{s}_present")] = 1.0
        whole[g, WHOLE_NAMES.index("w_n_eff")] = n
        whole[g, WHOLE_NAMES.index("w_present")] = 1.0
    ranks = list(RANK_TO_IDX)
    keys = np.asarray([f"{ranks[g % 10]}:{g:05d}" for g in range(n_games)], dtype=str)
    sums = -(rng.rand(n_games, 18) * 2.0 + 0.5) * n_moves[:, None]
    return {"game_key": keys, "staged": staged, "whole": whole, "sums": sums, "n_moves": n_moves}


def _questions(vec, sizes=(5, 7, 12, 20, 9), seed=1, with_rank=True):
    rng = np.random.RandomState(seed)
    rows = []
    for j, m in enumerate(sizes):
        pick = rng.choice(len(vec["game_key"]), size=m, replace=False)
        rows.append({"question_id": f"q{j}", "rank": list(RANK_TO_IDX)[j % 10], "player_id": f"p{j:015d}",
                     "n_games": m, "game_keys": [str(vec["game_key"][i]) for i in pick]})
    qs = pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])
    return qs if with_rank else qs.drop(columns="rank")


def _games_df(vec):
    return pd.DataFrame({"game_key": vec["game_key"], "n_moves": vec["n_moves"],
                         **{c: vec["sums"][:, i] for i, c in enumerate(SUM_COLS)}})


def _one_question(sums, n_moves):
    """A vec holding exactly one question's games, and the packed question."""
    m = len(n_moves)
    vec = _vec(n_games=m, seed=5)
    vec["sums"] = np.asarray(sums, dtype=np.float64)
    vec["n_moves"] = np.asarray(n_moves, dtype=np.int64)
    qs = pd.DataFrame([{"question_id": "q", "rank": "1D", "player_id": "p", "n_games": m,
                        "game_keys": [str(k) for k in vec["game_key"]]}])
    return vec, pack_questions(qs, vec)


def _r(sums, n_moves):
    c = np.asarray(sums, dtype=np.float64) / np.asarray(n_moves, dtype=np.float64)[:, None]
    return c - c.mean(axis=1, keepdims=True)


def test_pack_layout_and_padding():
    vec = _vec()
    qs = _questions(vec, sizes=(5, 20, 8))
    p = pack_questions(qs, vec)
    assert p["tokens"].dtype == np.float32 and p["tokens"].shape == (3, 20, 301)
    assert p["mask"].dtype == bool and p["mask"][0].tolist() == [True] * 5 + [False] * 15
    assert p["mask"][1].all() and p["mask"][2].sum() == 8
    assert np.all(p["tokens"][0, 5:] == 0) and np.all(p["tokens"][2, 8:] == 0)
    row = {k: i for i, k in enumerate(vec["game_key"])}
    idx0 = np.array([row[k] for k in qs["game_keys"][0]])
    assert np.array_equal(p["idx"][0], idx0)
    assert np.array_equal(p["tokens"][0, :5, :300], vec["staged"][idx0])
    assert np.array_equal(p["tokens"][0, :5, 300], np.log1p(vec["n_moves"][idx0]).astype(np.float32))
    assert QGLOB_NAMES == ["log_games", "log_target_moves", "log1p_seg1", "log1p_seg2", "log1p_seg3", "log1p_seg4"]
    seg = [vec["staged"][idx0, STAGED.index(f"s{s}_n_eff")].astype(np.float64).sum() for s in range(1, 5)]
    want = np.array([np.log(5), np.log(vec["n_moves"][idx0].sum()), *np.log1p(seg)], dtype=np.float32)
    assert p["qglob"].dtype == np.float32 and p["qglob"].shape == (3, 6) and np.array_equal(p["qglob"][0], want)
    assert p["y"].dtype == np.int64 and p["y"].tolist() == [RANK_TO_IDX[r] for r in qs["rank"]]
    assert p["n_games"].tolist() == [5, 20, 8]
    assert p["question_id"].tolist() == ["q0", "q1", "q2"] and p["player_id"].tolist() == qs["player_id"].tolist()
    w = pack_questions(qs, vec, rep="whole")
    assert w["tokens"].shape == (3, 20, 76) and np.array_equal(w["tokens"][0, :5, :75], vec["whole"][idx0])
    assert np.array_equal(w["qglob"], p["qglob"])  # stage counts come from the staged vectors for either rep
    assert pack_questions(_questions(vec, with_rank=False), vec)["y"].tolist() == [-1] * 5
    with pytest.raises(ValueError):
        pack_questions(_questions(vec, sizes=(21,)), vec)


def test_counts_off_zeroes_length_columns():
    vec = _vec()
    qs = _questions(vec)
    for rep, names in (("staged", STAGED), ("whole", WHOLE_NAMES)):
        on, off = pack_questions(qs, vec, rep=rep), pack_questions(qs, vec, rep=rep, counts=False)
        cols = [i for i, n in enumerate(names) if n.endswith(("_n_eff", "_present"))] + [len(names)]
        assert len(cols) == (9 if rep == "staged" else 3)
        assert np.all(off["tokens"][:, :, cols] == 0)
        keep = np.setdiff1d(np.arange(len(names) + 1), cols)
        assert np.array_equal(off["tokens"][:, :, keep], on["tokens"][:, :, keep])
        assert np.array_equal(off["qglob"][:, 0], on["qglob"][:, 0]) and np.all(off["qglob"][:, 1:] == 0)
        assert np.array_equal(off["mask"], on["mask"])
    on, off = pack_questions(qs, vec), pack_questions(qs, vec, counts=False)
    q_on, q_off = static_features(on, vec, "q24"), static_features(off, vec, "q24", counts=False)
    assert q_off.shape == (len(qs), len(FEATURE_NAMES) - 1)
    assert np.array_equal(q_off, np.delete(q_on, FEATURE_NAMES.index("log_moves"), axis=1))
    s_on, s_off = static_features(on, vec, "sum"), static_features(off, vec, "sum", counts=False)
    assert s_on.shape == (len(qs), 38) and np.array_equal(s_off, s_on[:, :-1])
    for kind, kw in (("clip", {"clip_k": 0.3}), ("trim", {})):
        a, b = static_features(on, vec, kind, **kw), static_features(off, vec, kind, counts=False, **kw)
        assert a.shape == (len(qs), 20) and np.array_equal(b, a[:, :-1])
    pooled = static_features(off, vec, "pooled", counts=False)
    assert pooled.shape == (len(qs), 301 + 6 + 23)
    assert np.array_equal(pooled[:, 301:307], off["qglob"].astype(np.float64)) and np.array_equal(pooled[:, 307:], q_off)
    assert np.all(pooled[:, [STAGED.index(f"s{s}_n_eff") for s in range(1, 5)] + [300]] == 0)
    assert np.array_equal(static_features(on, vec, "pooled", counts=False), pooled)  # strips counts whatever the packing
    with pytest.raises(ValueError):
        static_features(off, vec, "pooled")  # counts were already stripped from the packing
    with pytest.raises(ValueError):
        static_features(off, vec, "length", counts=False)


@pytest.mark.parametrize("counts", [True, False])
def test_static_features_order_free(counts):
    vec = _vec()
    qs = _questions(vec)
    rng = np.random.RandomState(7)
    shuffled = qs.copy()
    shuffled["game_keys"] = [[str(k) for k in rng.permutation(keys)] for keys in qs["game_keys"]]
    assert any(a != b for a, b in zip(qs["game_keys"], shuffled["game_keys"]))
    for rep in ("staged", "whole"):
        a, b = pack_questions(qs, vec, rep=rep, counts=counts), pack_questions(shuffled, vec, rep=rep, counts=counts)
        for kind in KINDS:
            if kind == "length" and not counts:
                continue
            fa = static_features(a, vec, kind, counts=counts, clip_k=0.4)
            fb = static_features(b, vec, kind, counts=counts, clip_k=0.4)
            assert fa.dtype == np.float64 and np.all(np.isfinite(fa)), kind
            np.testing.assert_allclose(fa, fb, rtol=0, atol=1e-9, err_msg=f"{rep} {kind}")
        assert fit_clip_k(a, vec) == pytest.approx(fit_clip_k(b, vec), abs=1e-12)


def test_clip_equals_consensus_without_outliers():
    rng = np.random.RandomState(3)
    c = -(rng.rand(18) * 2 + 1)
    n = np.array([40, 77, 103, 64, 150, 90, 55])
    vec, p = _one_question(c[None, :] * n[:, None], n)
    r = c - c.mean()
    for kind, kw in (("clip", {"clip_k": 0.05}), ("trim", {})):
        np.testing.assert_allclose(static_features(p, vec, kind, **kw)[0, :18], r, rtol=0, atol=1e-12)


def test_clip_bounds_outlier():
    rng = np.random.RandomState(4)
    c = -(rng.rand(18) * 2 + 1)
    n = np.array([60, 80, 100, 120, 90])
    k = 0.1

    def clip_feat(scale):
        sums = c[None, :] * n[:, None]
        sums[2] = (c + scale * np.where(np.arange(18) % 2 == 0, 1.0, -1.0)) * n[2]  # one game far off in every profile
        vec, p = _one_question(sums, n)
        return static_features(p, vec, "clip", clip_k=k)[0], _r(sums, n)

    f, r = clip_feat(5.0)
    med = r[[0, 1, 3, 4]].mean(axis=0)  # the four agreeing games are the median
    np.testing.assert_allclose(f[:18], med + np.clip(r[2] - med, -k, k) / 5, rtol=0, atol=1e-12)
    assert np.all(np.abs(f[:18] - med) <= k / 5 + 1e-12)
    assert np.abs(r[2] - med).min() > 10 * k  # the outlier is clipped in every profile
    np.testing.assert_allclose(clip_feat(50.0)[0], f, rtol=0, atol=1e-12)  # a wilder outlier changes nothing
    assert f[18] == np.log(5) and f[19] == np.log(n.sum())


def test_trim_rule():
    w = np.linspace(-1.0, 1.0, 18)  # mean 0 (to rounding), so r_g = a_g * w
    a = np.array([6.0, 0.0, 100.0, 2.0, 1.0])
    n = np.array([64, 32, 128, 16, 256])
    vec, p = _one_question(a[:, None] * w[None, :] * n[:, None], n)
    f = static_features(p, vec, "trim")[0]
    np.testing.assert_allclose(f[:18], 3.0 * w, rtol=0, atol=1e-12)  # drop 0 and 100, mean of 1, 2, 6
    a4, n4 = a[:4], n[:4]
    vec4, p4 = _one_question(a4[:, None] * w[None, :] * n4[:, None], n4)
    np.testing.assert_allclose(static_features(p4, vec4, "trim")[0, :18], a4.mean() * w, rtol=0, atol=1e-12)  # floor(0.8) = 0


def test_q24_matches_build_matrix():
    vec = _vec()
    for qs in (_questions(vec), _questions(vec, with_rank=False, seed=2)):
        X, y, n_games = build_matrix(_games_df(vec), qs)
        p = pack_questions(qs, vec)
        assert np.array_equal(static_features(p, vec, "q24"), X)
        assert np.array_equal(p["y"], y) and np.array_equal(p["n_games"], n_games)


def test_sum_and_length_definitions():
    vec = _vec()
    qs = _questions(vec)
    p = pack_questions(qs, vec)
    s, ln = static_features(p, vec, "sum"), static_features(p, vec, "length")
    assert ln.shape == (len(qs), 12)
    for q, idx in enumerate(p["idx"]):
        L, n, m = vec["sums"][idx].sum(axis=0), vec["n_moves"][idx].sum(), len(idx)
        np.testing.assert_allclose(s[q], np.concatenate([(L - L.mean()) / np.sqrt(n), (L - L.mean()) / n, [np.log(m), np.log(n)]]),
                                   rtol=1e-12, atol=1e-12)
        st = vec["staged"][idx].astype(np.float64)
        present = [st[:, STAGED.index(f"s{k}_present")].mean() for k in range(1, 5)]
        n_eff = [st[:, STAGED.index(f"s{k}_n_eff")].mean() for k in range(1, 5)]
        nm = vec["n_moves"][idx].astype(np.float64)
        np.testing.assert_allclose(ln[q], [np.log(m), np.log(n), nm.mean(), nm.std(), *present, *n_eff], rtol=1e-12, atol=1e-12)
    pooled = static_features(pack_questions(qs, vec, rep="whole"), vec, "pooled")
    assert pooled.shape == (len(qs), 76 + 6 + 24)
    np.testing.assert_allclose(pooled[1, :75], vec["whole"][p["idx"][1]].astype(np.float64).mean(axis=0), rtol=0, atol=1e-12)


def test_fit_clip_k_and_clip_requires_k():
    vec = _vec()
    p = pack_questions(_questions(vec), vec)
    dev = []
    for idx in p["idx"]:
        r = _r(vec["sums"][idx], vec["n_moves"][idx])
        dev.append(np.abs(r - np.median(r, axis=0)).ravel())
    assert fit_clip_k(p, vec) == pytest.approx(np.quantile(np.concatenate(dev), 0.95), abs=1e-15)
    with pytest.raises(ValueError):
        static_features(p, vec, "clip")
    with pytest.raises(ValueError):
        static_features(p, vec, "nope")


def test_static_rungs_table():
    assert list(STATIC_RUNGS) == ["r0_sum", "r1_clip", "r1_trim", "r2_mlp24", "r2_staged", "r2_whole", "r2_length_only"]
    want = {"r0_sum": ("sum", "staged", SoftmaxModel, "l2", 1e-3), "r1_clip": ("clip", "staged", SoftmaxModel, "l2", 1e-3),
            "r1_trim": ("trim", "staged", SoftmaxModel, "l2", 1e-3), "r2_mlp24": ("q24", "staged", MLPModel, "hidden", 32),
            "r2_staged": ("pooled", "staged", MLPModel, "hidden", 64), "r2_whole": ("pooled", "whole", MLPModel, "hidden", 64),
            "r2_length_only": ("length", "staged", MLPModel, "hidden", 32)}
    for name, (kind, rep, cls, attr, value) in want.items():
        k, r, make = STATIC_RUNGS[name]
        model = make()
        assert (k, r) == (kind, rep) and type(model) is cls and getattr(model, attr) == value and make() is not model


@pytest.mark.skipif(not (VECTORS / "fold1.npz").exists(), reason="real fold-1 vectors not built")
def test_real_fold1_vectors_pack_finite():
    vec = dict(np.load(VECTORS / "fold1.npz"))
    st = vec["staged"]
    n_eff = st[:, [STAGED.index(f"s{k}_n_eff") for k in range(1, 5)]].astype(np.int64)
    assert np.array_equal(n_eff.sum(axis=1), vec["n_moves"])  # the stage columns are the ones named
    assert np.array_equal(st[:, [STAGED.index(f"s{k}_present") for k in range(1, 5)]] == 1, n_eff > 0)
    rng = np.random.RandomState(0)
    rows = [{"question_id": f"q{j}", "player_id": "p", "n_games": m,
             "game_keys": [str(k) for k in vec["game_key"][rng.choice(len(st), size=m, replace=False)]]}
            for j, m in enumerate([5, 10, 20] * 10)]
    qs = pd.DataFrame(rows)
    for counts in (True, False):
        p = pack_questions(qs, vec, counts=counts)
        k = fit_clip_k(p, vec)
        assert np.isfinite(k) and k > 0
        for kind in KINDS:
            if kind == "length" and not counts:
                continue
            assert np.all(np.isfinite(static_features(p, vec, kind, counts=counts, clip_k=k))), kind
