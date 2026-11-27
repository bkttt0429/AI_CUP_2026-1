"""Fit fixed ordinal corrections on folds 1/2, then score specific tactical choices on 3/4."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr

from extract_games import HERE, R1, RV, questions, save_json, sha

os.chdir(R1)
sys.path[:0] = [str(R1 / "logs/nano4"), str(R1 / "scripts")]
from m137f_build2 import guard2
from aicup.metric import ADJACENT, expected_score_decision
from aicup.qeval import balanced_mean_over_n

SIZES = [5, 10, 20]
RANKS = np.arange(10)
VARIANTS = ["SQ4", "opportunity", "marginal", "full", "shuffle"]
PAIRS = [("full", "SQ4"), ("full", "marginal"), ("full", "shuffle"),
         ("opportunity", "SQ4"), ("marginal", "SQ4"), ("shuffle", "marginal")]


def slope(total, favorable, played, joint):
    denom = np.maximum(total, 1)
    variance = favorable * (total - favorable) / denom
    return (joint - favorable * played / denom).sum(axis=1) / (4.0 + variance.sum(axis=1))


def features(cells, sizes, task):
    cols = np.moveaxis(cells, -1, 0)
    if task == "defense":
        seen, good, chose, joint = cols
        total, favorable = seen, good
        action = [(chose + .5) / (seen + 1)]
    else:
        seen, good, total, favorable, chose, joint = cols
        action = [(total + .5) / (seen + 1), (favorable + .5) / (total + 1), (chose + .5) / (total + 1)]
    assert np.all((joint <= np.minimum(favorable, chose)) & (total >= favorable + chose - joint))
    rng = np.random.default_rng(20261118 if task == "defense" else 20261119)
    fake = np.zeros_like(joint)
    nonempty = total > 0
    fake[nonempty] = rng.hypergeometric(favorable[nonempty], (total-favorable)[nonempty], chose[nonempty])
    b_real = slope(total, favorable, chose, joint)
    b_null = slope(total, favorable, chose, fake)
    opp = np.column_stack([np.log1p(seen / sizes[:, None]), (good + .5) / (seen + 1)])
    marginal = np.column_stack([opp] + action)
    informative = ((favorable > 0) & (favorable < total) & (chose > 0) & (chose < total)).any(axis=1)
    return {"opportunity": opp, "marginal": marginal, "full": np.column_stack([marginal, b_real]),
            "shuffle": np.column_stack([marginal, b_null])}, b_real, b_null, informative


def balanced_weights(y, sizes):
    code = y + 10 * np.searchsorted(SIZES, sizes)
    count = np.bincount(code, minlength=30)
    assert (count > 0).all()
    return 1 / (30 * count[code])


def decide(p):
    u = p.copy()
    u[:, 1:] += ADJACENT * p[:, :-1]
    u[:, :-1] += ADJACENT * p[:, 1:]
    return u.argmax(axis=1)


def tilt(x, dev, baseline, sizes, ydev, w):
    exp_rank = baseline @ RANKS
    adjust = np.column_stack([np.ones(len(baseline)), exp_rank, exp_rank**2, np.log(sizes)])
    sw = np.sqrt(w)[:, None]
    regress = np.linalg.lstsq(adjust[dev] * sw, x[dev] * sw, rcond=None)[0]
    residual = x - adjust @ regress
    center = w @ residual[dev]
    scale = np.sqrt(w @ (residual[dev] - center)**2)
    scale[scale < 1e-12] = 1
    design = np.clip((residual - center) / scale, -4, 4)
    f = design[dev]
    lp = np.log(baseline)
    pdev = lp[dev]
    idx = np.arange(len(ydev))

    def objective(beta):
        logits = pdev + (f @ beta)[:, None] * RANKS
        norm = logsumexp(logits, axis=1)
        probs = np.exp(logits - norm[:, None])
        loss = w @ (norm - logits[idx, ydev]) + .05 * np.dot(beta, beta)
        return loss, f.T @ (w * (probs @ RANKS - ydev)) + .1 * beta

    fit = minimize(objective, np.zeros(f.shape[1]), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 400})
    if not fit.success or np.abs(fit.jac).max() > 1e-5:
        raise RuntimeError(str(fit.message))
    delta = design @ fit.x
    logits = lp + delta[:, None] * RANKS
    probs = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    model = {"beta": fit.x.tolist(), "residualizer": regress.tolist(), "scale": scale.tolist(),
             "center": center.tolist(), "iterations": int(fit.nit), "gradient_max": float(np.max(np.abs(fit.jac)))}
    return probs, delta, model


def score(pred, y):
    distance = np.abs(pred - y)
    return np.where(distance == 0, 1., np.where(distance == 1, ADJACENT, 0.))


def cluster_intervals(diffs, y, n, pid, repeats=5000):
    """Stratified paired bootstrap; one owner multiplicity across all sizes and methods."""
    rng = np.random.default_rng(42)
    sample = np.zeros((repeats, diffs.shape[1]))
    point = np.zeros(diffs.shape[1])
    for k in RANKS:
        mask = y == k
        owners, inv = np.unique(pid[mask], return_inverse=True)
        cnt = np.zeros((len(owners), len(SIZES)))
        sums = np.zeros((*cnt.shape, diffs.shape[1]))
        s = np.searchsorted(SIZES, n[mask])
        np.add.at(cnt, (inv, s), 1)
        np.add.at(sums, (inv, s), diffs[mask])
        weights = rng.multinomial(len(owners), np.full(len(owners), 1 / len(owners)), size=repeats)
        denominators = weights @ cnt
        assert (denominators > 0).all()
        sample += ((weights @ sums.reshape(len(owners), -1)).reshape(repeats, len(SIZES), -1)
                   / denominators[:, :, None]).mean(axis=1) / 10
        point += (sums.sum(axis=0) / cnt.sum(axis=0)[:, None]).mean(axis=0) / 10
    ref = [balanced_mean_over_n(diffs[:, j], y, n) for j in range(diffs.shape[1])]
    np.testing.assert_allclose(point, ref, atol=1e-13)
    return np.quantile(sample, [.025, .975], axis=0).T


def unique_game_stats(cells, task):
    c = cells.sum(axis=0, dtype=np.int64)
    out = {"cells": c.tolist(), "games": int(len(cells)), "opportunities": int(c[:, 0].sum()),
           "games_with_opportunity": int((cells[:, :, 0].sum(axis=1) > 0).sum())}
    if task == "defense":
        rows = []
        for label, pair in (("local_caught", c[:2]), ("local_no_capture", c[2:])):
            total, safe, act, joint = map(int, pair.sum(axis=0))
            rows.append({"local": label, "events": total, "full_safe": safe, "full_caught": total - safe,
                         "extend_if_full_safe": [joint, safe], "extend_if_full_caught": [act-joint, total-safe]})
        out["by_local"] = rows
        out["local_full_disagree"] = int(c[:2, 1].sum() + (c[2:, 0] - c[2:, 1]).sum())
    else:
        def detail(arr):
            seen, good, total, favorable, chose, joint = map(int, arr.sum(axis=0))
            return {"opportunities": seen, "attempted": total, "correct_direction": total-favorable-chose+2*joint,
                    "reference_wrong_events": seen-good, "reference_wrong_attempted": total-favorable,
                    "correct_when_reference_wrong": total-favorable-chose+joint}
        out["all"] = detail(c)
        out["one_local_winner"] = detail(c[2:4])
    return out


def main(task):
    file = HERE / ("game_counts.npz" if task == "defense" else "attack_counts.npz")
    extract_metadata = HERE / ("extraction.json" if task == "defense" else "attack_extraction.json")
    assert sha(file) == json.loads(extract_metadata.read_text())["counts_sha256"]
    qs, _ = questions()
    with np.load(file) as z:
        keys, valid, per_game = z["keys"], z["valid"], z["cells"]
    positions = {k: i for i, k in enumerate(keys)}
    indices = [np.asarray([positions[k] for k in q[4]]) for q in qs]
    keep = np.asarray([valid[ix].all() for ix in indices])
    with np.load(RV / "m183_dump.npz") as z:
        fold, pid, y, n, ref, saved, plain = [z[k][keep] for k in ("fold", "pid", "y", "n", "ct106_rule", "SQ4", "anat_plain")]
        p = z["anat_post"][keep].astype(float)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
    assert np.all(p > 0)
    p /= p.sum(axis=1, keepdims=True)
    raw = decide(p)
    np.testing.assert_array_equal(raw, plain)
    np.testing.assert_array_equal(raw[::211], [expected_score_decision(row) for row in p[::211]])
    np.testing.assert_array_equal(guard2(raw, ref), saved)
    indices = [ix for ix, ok in zip(indices, keep) if ok]
    cells = np.stack([per_game[ix].sum(axis=0, dtype=np.int64) for ix in indices])
    dev, test = fold <= 2, fold >= 3
    assert not (set(pid[dev]) & set(pid[test]))
    w = balanced_weights(y[dev], n[dev])
    xs, real, null, informative = features(cells, n, task)
    scores = {"SQ4": score(saved, y)}
    pred = {"SQ4": saved}
    nll = {"SQ4": -np.log(p[np.arange(len(y)), y])}
    deltas, fits = {}, {}
    for name, x in xs.items():
        probs, deltas[name], fits[name] = tilt(x, dev, p, n, y[dev], w)
        pred[name] = guard2(decide(probs), ref)
        scores[name] = score(pred[name], y)
        nll[name] = -np.log(probs[np.arange(len(y)), y])

    def metrics(v, mask):
        return balanced_mean_over_n(v[mask], y[mask], n[mask])

    summary = {}
    for name in VARIANTS:
        summary[name] = {"score": metrics(scores[name], test), "nll": metrics(nll[name], test),
                         "scores_by_n": {str(s): balanced_mean_over_n(scores[name][test], y[test], n[test], sizes=[s]) for s in SIZES}}
    differences = np.column_stack([(scores[a] - scores[b])[test] for a, b in PAIRS])
    cis = cluster_intervals(differences, y[test], n[test], pid[test])
    comparisons = {}
    for (a, b), ci in zip(PAIRS, cis):
        d = scores[a] - scores[b]
        comparisons[a + "_minus_" + b] = {"difference": metrics(d, test), "ci95": ci.tolist(),
             "folds": {str(f): metrics(d, fold == f) for f in (3, 4)},
             "nll_difference": metrics(nll[a] - nll[b], test), "changed": int((pred[a][test] != pred[b][test]).sum()),
             "improved": int((d[test] > 0).sum()), "worsened": int((d[test] < 0).sum()),
             "by_n": {str(s): balanced_mean_over_n(d[test], y[test], n[test], sizes=[s]) for s in SIZES}}
    main_diff = comparisons["full_minus_SQ4"]
    advance = (main_diff["difference"] >= .001 and min(main_diff["folds"].values()) > 0 and main_diff["ci95"][0] > 0
               and comparisons["full_minus_marginal"]["ci95"][0] > 0)
    unique_test = np.unique(np.concatenate([ix for ix, ok in zip(indices, test) if ok]))
    diagnostic = {"unique_eval_games": unique_game_stats(per_game[unique_test], task),
                  "all_games": unique_game_stats(per_game[valid], task),
                  "both_vary_within_cell_by_n": {str(s): [int(informative[test & (n==s)].sum()), int((test & (n==s)).sum())] for s in SIZES},
                  "median_filtered_opportunities_per_question": {str(s): float(np.median(cells[test & (n==s), :, 0].sum(axis=1))) for s in SIZES}}
    twenty = test & (n == 20)
    if len(set(pid[twenty])) == twenty.sum() and np.std(real[twenty]) > 0:
        diagnostic["twenty_game_rank_spearman"] = float(spearmanr(real[twenty], y[twenty]).statistic)
    out = HERE / (task + "_predictions.npz")
    np.savez_compressed(out, kept_indices=np.flatnonzero(keep), pid=pid, fold=fold, n=n, y=y, real=real, null=null,
                        informative=informative, **{"pred_"+k:v for k,v in pred.items()}, **{"tilt_"+k:v for k,v in deltas.items()})
    report = {"task": task, "development_questions": int(dev.sum()), "evaluation_questions": int(test.sum()),
              "evaluation_players": len(set(pid[test])), "development_players": len(set(pid[dev])),
              "by_fold": {str(f): [int((fold==f).sum()),len(set(pid[fold==f]))] for f in (1,2,3,4)},
              "excluded_questions": int((~keep).sum()), "advance": bool(advance), "diagnostics": diagnostic,
              "evaluation": summary, "paired": comparisons, "models": fits, "bootstrap": "5000 rank-stratified paired player draws, seed 42",
              "sha256": {p.name:sha(p) for p in (file, extract_metadata, HERE/"protocol.json", Path(__file__),
                       RV/"m183_dump.npz", RV/"m221_SC_dump.npz", out, *([HERE/"attack_protocol.json"] if task=="attack" else []))}}
    save_json(HERE / (task + "_results.json"), report)
    print(task, "advance", advance, "evaluated", test.sum(), "questions", len(set(pid[test])), "players", flush=True)
    print(json.dumps({k:v for k,v in comparisons.items() if k in ("full_minus_SQ4", "full_minus_marginal", "opportunity_minus_SQ4")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["defense", "attack"])
    main(parser.parse_args().task)
