"""Exploratory decomposition after reading the frozen shortlist test."""
import json
import numpy as np
from score_games import HERE, digest, immutable_json, read
from verify_evaluate import ids, load
from verify_scores import split


def by_player(owners, values, mask):
    ix = np.flatnonzero(mask)
    players = np.unique(owners[ix])
    if not len(players):
        return {"games": 0, "players": 0}
    a = np.array([values[ix[owners[ix] == p]].mean() for p in players])
    b = np.random.default_rng(20261118).choice(a, size=(10000, len(a))).mean(1)
    return {"games": int(len(ix)), "players": int(len(a)), "nats_per_move": float(a.mean()),
            "ci95": np.quantile(b,[.025,.975]).tolist()}


def main():
    result = read(HERE / "verifier_result.json")
    with np.load(HERE / "verifier_scores.npz", allow_pickle=False) as z:
        t = load(z, 2)
    group, _, _ = split(read(HERE / "selection.json")["queries"], 2)
    grade = [q["keys"][0].split(":")[0] for q in group]
    truth = np.repeat(np.arange(72), 2)
    top = t["top"][:len(truth)]
    in_pool = (top == truth[:,None]).any(1)
    first = top[:,0] == truth
    which_wrong = np.where(first, top[:,1], top[:,0])
    same_grade = np.array([grade[a] == grade[b] for a,b in zip(truth,which_wrong)])
    raw = t["raw"][np.arange(len(truth)),truth] - t["raw"][np.arange(len(truth)),which_wrong]
    w = result["chosen_weight"]
    full, gap = ids(t, w)
    base, gap0 = ids(t, 0.)
    found = full[:len(truth)] == truth
    accepted = gap > result["fold1"][str(w)]["threshold"]
    accepted0 = gap0 > result["fold1"]["0.0"]["threshold"]
    good = found & accepted[:len(truth)]
    old = first & accepted0[:len(truth)]
    known = len(truth)
    assert np.array_equal(base[:known],top[:,0]) and not np.any(found & ~in_pool)
    breakdown = {"scope":"Exploratory split conditioned on M218 rank on fold 2 after the main readout; intervals conditional on the fixed candidate bank.",
                 "verifier_result_sha256":digest(HERE / "verifier_result.json"), "known_games":known,
                 "outside_shortlist":int(sum(~in_pool)), "baseline_first":by_player(truth,raw,first),
                 "baseline_rank2_or_3":by_player(truth,raw,in_pool & ~first),
                 "same_grade_rival":by_player(truth,raw,in_pool & same_grade),
                 "different_grade_rival":by_player(truth,raw,in_pool & ~same_grade),
                 "top1_errors_rescued":int(sum(found & ~first)), "top1_correct_lost":int(sum(first & ~found)),
                 "correct_accepts_gained":int(sum(good & ~old)), "correct_accepts_lost":int(sum(old & ~good)),
                 "unknown_new_accepts":int(sum(accepted[known:] & ~accepted0[known:])),
                 "unknown_accepts_removed":int(sum(accepted0[known:] & ~accepted[known:]))}
    immutable_json(HERE / "error_breakdown.json",breakdown)
    print(json.dumps(breakdown,indent=2))


if __name__ == "__main__":
    main()
