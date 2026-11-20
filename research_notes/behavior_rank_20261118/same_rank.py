"""Reconstruct the previous pilot and test within-rank reference identity by exact randomization."""
import itertools
import json
from pathlib import Path

import numpy as np
from score_games import HERE, PILOT, digest, immutable_json, read


def contrast(matrix, groups):
    """Average the own-reference minus other-same-grade log-score, with equal owner weights."""
    denominator = sum(len(g) for g in groups)
    observed = 0.0
    null = np.array([0.0])
    for g in groups:
        block = matrix[np.ix_(g, g)]
        n = len(g)
        centered = (block - block.mean(1, keepdims=True)) * n / (n - 1)
        observed += centered.trace()
        values = np.array([centered[np.arange(n), perm].sum()
                           for perm in itertools.permutations(range(n))])
        null = (null[:, None] + values).reshape(-1)
    null /= denominator
    observed /= denominator
    return {"nats_per_move": float(observed), "permutations": len(null),
            "one_sided_p": float(np.mean(null >= observed - 1e-12)),
            "null_mean": float(null.mean())}


def main():
    pilot, prior = read(PILOT / "protocol.json"), read(PILOT / "result.json")
    assert prior["protocol_sha256"] == digest(PILOT / "protocol.json")
    assert prior["training_protocol_sha256"] == digest(PILOT / "training_protocol.json")
    owners = pilot["split"]["test"]
    labels = []
    for owner in owners:
        ranks = {g["original_key"].split(":")[0] for g in pilot["rows"] if g["owner"] == owner}
        assert len(ranks) == 1
        labels.append(ranks.pop())
    labels = np.array(labels)
    groups = [np.flatnonzero(labels == rank) for rank in np.unique(labels) if (labels == rank).sum() >= 2]
    mats, old_nll, old_top1 = {}, {}, {}
    for seed in pilot["seeds"]:
        for arm in ("contextual", "global", "background", "contextual_permute", "contextual_no_actions"):
            tag = f"{arm}_{seed}"
            with np.load(PILOT / f"test_{tag}.npz", allow_pickle=False) as z:
                assert z["owners"].tolist() == owners
                scores, truth = z["scores"].astype(float), z["truth"]
            assert scores.shape == (60, 20) and np.array_equal(np.bincount(truth), np.full(20, 3))
            matrix = np.stack([scores[truth == i].mean(0) for i in range(len(owners))])
            want = prior["per_player"][tag]
            nll, margin = -matrix.diagonal(), (matrix.diagonal() - matrix.mean(1)) * len(owners)/(len(owners)-1)
            top1 = np.array([(scores[truth == i].argmax(1) == i).mean() for i in range(len(owners))])
            for actual, name in [(nll, "nll"), (margin, "margin"), (top1, "top1")]:
                np.testing.assert_allclose(actual, want[name], atol=1e-6, rtol=0)
            mats.setdefault(arm, []).append(matrix)
            old_nll.setdefault(arm, []).append(nll)
            old_top1[tag] = top1.mean()
    for arm in prior["nll_improvement"]:
        gain = np.mean(np.array(old_nll[arm]) - old_nll["contextual"], axis=0)
        rng = np.random.default_rng(20261116)
        draws = gain[rng.integers(0, len(gain), size=(2000, len(gain)))].mean(1)
        np.testing.assert_allclose([gain.mean(), *np.quantile(draws, [.025, .975])],
                                   [prior["nll_improvement"][arm]["mean"], *prior["nll_improvement"][arm]["ci95"]],
                                   atol=1e-6, rtol=0)
    mats = {k: np.stack(v) for k, v in mats.items()}
    mats["intact_minus_permuted"] = mats["contextual"] - mats["contextual_permute"]
    result = {}
    for k, matrix in mats.items():
        result[k] = contrast(matrix.mean(0), groups)
        result[k]["per_seed_nats"] = [contrast(m, groups)["nats_per_move"] for m in matrix]
    # A reference-column main effect is unrelated to identity and cancels in the exact statistic.
    np.testing.assert_allclose(contrast(np.tile(np.arange(20), (20, 1)), groups)["nats_per_move"], 0, atol=1e-12)
    assert result["background"]["one_sided_p"] == 1 and abs(result["contextual"]["null_mean"]) < 1e-12
    report = dict(pilot_verdict=prior["verdict"], verified_score_matrices=10, eligible=sum(map(len, groups)),
                  rank_counts={r: int(sum(labels == r)) for r in sorted(set(labels))},
                  within_rank=result, pilot_all_ranks_nll=prior["nll_improvement"],
                  pilot_all_ranks_margin=prior["correct_reference_margin"],
                  pilot_game_top1=old_top1, source_result_sha256=digest(PILOT / "result.json"),
                  scope="Post-hoc within the 20 original model-heldout players. Within-rank shuffled owner assignments preserve the reference bank and all games of a player.")
    immutable_json(HERE / "same_rank.json", report)
    print(json.dumps({"eligible": report["eligible"], "within_rank": result}, indent=2))


if __name__ == "__main__":
    main()
