"""Calibrate on fold 1 and measure the frozen conditional rank signal on fold 2."""
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from score_games import HERE, R1, digest, immutable_json, read

sys.path.insert(0, str(R1 / "src"))
from aicup.metric import LEVELS, expected_score_decision, score


def question_rows(selection):
    exclusions = read(HERE / "exclusions.json")
    with np.load(HERE / "game_scores.npz", allow_pickle=False) as z:
        assert z["selection_sha"].item() == digest(HERE / "selection.json")
        index = {key: i for i, key in enumerate(z["keys"])}
        bank_scores = z["ranks"].astype(float)
        neutral = z["neutral"].astype(float)
    assert bank_scores.shape == (len(index), 2, 2, len(LEVELS))
    assert set(index).isdisjoint(exclusions)
    sets = {k for q in selection["queries"] for k in q["keys"]}
    assert set(index) | set(exclusions) == sets
    labels = {v: read(R1 / f"results/trunk_seq/r_val{v}.json")["val"] for v in (1, 2)}
    rows, dropped = [], []
    for q in selection["queries"]:
        if any(k in exclusions for k in q["keys"]):
            dropped.append(q["owner"])
            continue
        idx, lower = [index[k] for k in q["keys"]], q["low"]
        evidence = bank_scores[idx].mean(axis=(0, 1))
        d = evidence[:, lower + 1] - evidence[:, lower]
        source, i = labels[q["fold"]], q["index"]
        assert source["player_id"][i] == q["owner"] and source["game_keys"][i] == q["keys"]
        y = LEVELS.index(source["rank"][i])
        assert all(k.split(":")[0] == LEVELS[y] for k in q["keys"])
        rows.append({**q, "y": y, "intact": d[0], "displaced": d[1], "binding_only": d[0]-d[1],
                     "neutral": neutral[idx].mean()})
    return rows, dropped


def inputs(rows, fold):
    selected = [q for q in rows if q["fold"] == fold]
    keys = ("p", "low", "y", "intact", "displaced", "binding_only", "neutral")
    return {k: np.array([q[k] for q in selected]) for k in keys}


def update(base, lower, z, slope):
    logits = np.log(base)
    ix = np.arange(len(logits))
    logits[ix, lower + 1] += .5 * slope * z
    logits[ix, lower] -= .5 * slope * z
    return softmax(logits, axis=1)


def nll(probs, y):
    return -np.log(probs[np.arange(len(y)), y])


def grid_free_fit(dev, test, name):
    means = np.array([dev[name][dev["low"] == b].mean() for b in range(9)])
    centered = dev[name] - means[dev["low"]]
    scale = centered.std()
    assert scale > 0 and np.isfinite(scale)
    zs = centered / scale
    ztest = (test[name] - means[test["low"]]) / scale
    def objective(beta):
        return nll(update(dev["p"], dev["low"], zs, beta), dev["y"]).mean() + .05 * beta ** 2
    interior = minimize_scalar(objective, bounds=(0, 2), method="bounded").x
    beta = min((0., float(interior), 2.), key=objective)
    return beta, scale, means.tolist(), ztest, update(test["p"], test["low"], ztest, beta)


def measurements(probs, y):
    pred = np.array([expected_score_decision(row) for row in probs])
    return {"nll": nll(probs, y), "score": np.array([score(p, t) for p, t in zip(pred, y)]),
            "acc": (pred == y).astype(float), "pred": pred}


def summarize(a, draws):
    return {"mean": float(np.mean(a)), "ci95": np.quantile(np.mean(a[draws], axis=1), [.025, .975]).tolist()}


def paired(base, alt, draws):
    delta = alt["score"] - base["score"]
    return {"nll_gain": summarize(base["nll"] - alt["nll"], draws),
            "score_gain": summarize(delta, draws), "accuracy_gain": summarize(alt["acc"]-base["acc"], draws),
            "decisions_changed": int(sum(base["pred"] != alt["pred"])),
            "score_improved": int(sum(delta > 0)), "score_worsened": int(sum(delta < 0))}


def main():
    config, selection = read(HERE / "protocol.json"), read(HERE / "selection.json")
    assert config["levels"] == list(LEVELS)
    rows, dropped = question_rows(selection)
    dev, test = inputs(rows, config["development_fold"]), inputs(rows, config["evaluation_fold"])
    assert len({q["owner"] for q in rows}) == len(rows)
    for item in (dev, test):
        item["p"] = item["p"] / item["p"].sum(1, keepdims=True)
        assert (item["p"] > 0).all()
    rng = np.random.default_rng(20261118)
    groups = [np.flatnonzero(test["low"] == b) for b in range(9)]
    draws = np.column_stack([rng.choice(idx, (10000, len(idx))) for idx in groups])
    measures = {"baseline": measurements(test["p"], test["y"])}
    fits, standardized = {}, {}
    for name in ("intact", "displaced", "binding_only", "neutral"):
        beta, scale, means, z, probs = grid_free_fit(dev, test, name)
        measures[name] = measurements(probs, test["y"])
        standardized[name] = z
        fits[name] = {"slope_dev": beta, "pooled_sd_dev": scale, "means_dev_by_boundary": means}
    vs_base = {k: paired(measures["baseline"], m, draws) for k, m in measures.items() if k != "baseline"}
    vs_displaced = paired(measures["displaced"], measures["intact"], draws)
    raw = {k: {metric: float(m[metric].mean()) for metric in ("nll", "score", "acc")} for k, m in measures.items()}
    support = min(vs_base["intact"]["nll_gain"]["ci95"][0], vs_displaced["nll_gain"]["ci95"][0]) > 0
    result = {"scope": config["scope"], "verdict": "supports_incremental_context_binding" if support else "incremental_context_binding_not_established",
              "selection_sha256": digest(HERE / "selection.json"), "scores_sha256": digest(HERE / "game_scores.npz"),
              "analysis_sha256": digest(HERE / "evaluate.py"), "selected": len(selection["queries"]),
              "unique_query_players": len(rows), "games_scored_in_used_questions": len(rows)*5,
              "excluded_query_owners": dropped, "n_dev": len(dev["y"]), "n_test": len(test["y"]),
              "per_boundary_test": [len(g) for g in groups],
              "test_truth_in_candidate_pair": int(np.isin(test["y"] - test["low"], [0, 1]).sum()),
              "fits": fits, "fold2": raw, "vs_baseline_fold2": vs_base, "intact_vs_displaced_fold2": vs_displaced,
              "exploratory_fold2_intact_displaced_feature_r": float(np.corrcoef(standardized["intact"], standardized["displaced"])[0, 1])}
    immutable_json(HERE / "rank_result.json", result)
    print(json.dumps({k: result[k] for k in ("verdict", "n_dev", "n_test", "fold2", "intact_vs_displaced_fold2")}, indent=2))
    print("vs baseline", {k: x["nll_gain"] for k, x in vs_base.items()}, "slopes", {k: x["slope_dev"] for k,x in fits.items()})


if __name__ == "__main__":
    main()
