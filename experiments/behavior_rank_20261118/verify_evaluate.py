"""Choose the identity verification rule on fold 1 and freeze it for fold 2."""
import json

import numpy as np
from score_games import HERE, digest, immutable_json, read
from verify_scores import split

GRID = (0., .125, .25, .5, 1.)


def by_row(a):
    sd = a.std(axis=1, keepdims=True)
    assert (sd > 1e-12).all()
    return (a - a.mean(1, keepdims=True)) / sd


def load(z, fold):
    rows, _, keys = split(read(HERE / "selection.json")["queries"], fold)
    assert z[f"owners{fold}"].tolist() == [r["owner"] for r in rows] and z[f"keys{fold}"].tolist() == keys
    cos = z[f"cos{fold}"].astype(float)
    policy = z[f"behavior{fold}"].astype(float)
    top = np.argsort(-cos, axis=1, kind="stable")[:, :3]
    delta = policy[:,0,:] - policy[:,1,:]
    return {"top": top, "cos": np.take_along_axis(by_row(cos), top, axis=1),
            "bind": np.take_along_axis(by_row(delta), top, axis=1), "raw": delta}


def ids(data, w):
    s = data["cos"] + w * data["bind"]
    order = np.argsort(-s, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(s, order, axis=1)
    who = np.take_along_axis(data["top"], order[:,:1], axis=1)[:,0]
    return who, sorted_scores[:,0] - sorted_scores[:,1]


def observe(data, w, threshold=None):
    who, gap = ids(data, w)
    truth = np.repeat(np.arange(108), 2)
    known = truth < 72
    if threshold is None:
        threshold = float(np.sort(gap[~known])[::-1][3])
    accepted = gap > threshold
    correct = who[known] == truth[known]
    by_owner = lambda x: x.astype(float).reshape(-1,2).mean(1)
    return threshold, {"known_correct_accept": by_owner(correct & accepted[known]),
                       "known_closed_top1": by_owner(correct), "unknown_accept": by_owner(accepted[~known]),
                       "known_wrong_accept": by_owner(~correct & accepted[known])}


def raw_rival(test):
    records = []
    for owner in range(72):
        differences = []
        for q in range(2*owner, 2*owner+2):
            top = test["top"][q]
            if owner in top:
                rival = next(int(i) for i in top if i != owner)
                differences.append(test["raw"][q,owner] - test["raw"][q,rival])
        if differences:
            records.append(differences)
    means = np.array([np.mean(x) for x in records])
    rng = np.random.default_rng(20261118)
    b = rng.choice(means, size=(10000,len(means))).mean(1)
    return {"eligible_games": sum(map(len, records)), "eligible_owners": len(records),
            "mean_nats_per_move_equal_owner": float(means.mean()), "ci95": np.quantile(b,[.025,.975]).tolist()}


def main():
    with np.load(HERE / "verifier_scores.npz", allow_pickle=False) as z:
        assert z["protocol_sha"].item() == digest(HERE / "verifier_protocol.json")
        assert z["selection_sha"].item() == digest(HERE / "selection.json")
        assert z["code_sha"].item() == digest(HERE / "verify_scores.py")
        dev, test = load(z, 1), load(z, 2)
    weights = {w: observe(dev, w) for w in GRID}
    assert all(t["unknown_accept"].sum() * 2 <= 3 for _, t in weights.values())
    best = max(GRID, key=lambda w: weights[w][1]["known_correct_accept"].mean())
    grids = {str(w): {"threshold": cut, "known_correct_accepted_games": int(stats["known_correct_accept"].sum()*2),
                      "unknown_accepted_games": int(stats["unknown_accept"].sum()*2)} for w,(cut,stats) in weights.items()}
    def at(w):
        return observe(test, w, weights[w][0])[1]
    baseline, chosen, fixed = at(0), at(best), at(.25)
    def counts(measures):
        return {k: {"games": int(v.sum()*2), "total_games": len(v)*2, "rate": float(v.mean())} for k,v in measures.items()}
    rng = np.random.default_rng(20261118)
    ix = {n:rng.integers(0,n,size=(10000,n)) for n in (72,36)}
    def diffs(measures):
        out = {}
        for k,v in measures.items():
            d = v - baseline[k]
            out[k] = {"mean": float(d.mean()), "ci95": np.quantile(d[ix[len(v)]].mean(1), [.025,.975]).tolist()}
        return out
    delta = diffs(chosen)
    ok = delta["known_correct_accept"]["ci95"][0] > 0 and delta["unknown_accept"]["mean"] <= 0
    result = {"verdict": "supports_shortlist_verifier" if ok else "shortlist_gain_not_established",
              "scope": read(HERE / "verifier_protocol.json")["scope"], "chosen_weight": best,
              "fold1": grids, "fold2_baseline": counts(baseline), "fold2_chosen": counts(chosen),
              "fold2_delta": delta, "fold2_fixed_025": counts(fixed), "fold2_fixed_025_delta": diffs(fixed),
              "fold2_shortlisted_hard_rival": raw_rival(test), "scores_sha256": digest(HERE / "verifier_scores.npz"),
              "analysis_sha256": digest(HERE / "verify_evaluate.py")}
    immutable_json(HERE / "verifier_result.json", result)
    print(json.dumps({k:result[k] for k in ("chosen_weight","fold2_baseline","fold2_chosen","fold2_delta","fold2_shortlisted_hard_rival")}, indent=2))


if __name__ == "__main__":
    main()
