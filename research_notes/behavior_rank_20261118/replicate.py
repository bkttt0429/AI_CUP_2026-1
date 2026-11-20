"""Lock the observed hard-rival contrast and repeat on two disjoint, previously uninspected pools."""
import argparse
import json

import numpy as np
import verify_evaluate
import verify_scores
from score_games import HERE, R1, PILOT, digest, immutable_json, load_queries, read

REPEAT = HERE / "replication"


def select():
    old = read(HERE / "selection.json")
    params = read(HERE / "protocol.json")
    block = {q["owner"] for q in old["queries"]} | {g["owner"] for g in read(PILOT / "protocol.json")["rows"]}
    frozen = read(HERE / "verifier_result.json")
    old_keys = {k for q in old["queries"] for k in q["keys"]}
    group = []
    for fold in (1,2):
        folder = R1 / "results/trunk_seq"
        native = read(folder / f"r_val{fold}.json")["val"]
        members = []
        for arm in params["baseline"]:
            for seed in params["baseline_seeds"]:
                path = folder / "es" / f"val{fold}" / f"{arm}_s{seed}.npz"
                assert digest(path) == old["inputs"][str(path)]
                with np.load(path, allow_pickle=False) as z:
                    members.append(z["proba"].astype(float))
        probs = np.mean(members,axis=0)
        counts = np.zeros(9,dtype=int)
        indexes = sorted(range(len(probs)),key=lambda i: verify_scores.order("verify-repeat-q-v1",f"{fold}|{i}|{native['player_id'][i]}"))
        for i in indexes:
            p,owner = probs[i],native["player_id"][i]
            if native["n_games"][i] != 5 or owner in block:
                continue
            a,b = np.argsort(-p,kind="stable")[:2]
            low = int(min(a,b))
            if abs(a-b) != 1 or p[a] > 2*p[b] or counts[low] == 12:
                continue
            assert len(set(native["game_keys"][i])) == 5 and old_keys.isdisjoint(native["game_keys"][i])
            group.append(dict(fold=fold,index=i,owner=owner,low=low,keys=native["game_keys"][i]))
            counts[low] += 1
            block.add(owner)
        assert (counts==12).all(),counts
    w = frozen["chosen_weight"]
    protocol = dict(read(HERE / "verifier_protocol.json"))
    protocol.update(followup="Independent players, separate 72-enrolled / 36-unknown pools in folds 1 and 2. Select by a new fixed hash among original rank-ambiguous questions, exclude all original participants. No new fitting. Main replication outcome: the player-averaged intact-minus-displaced margin against the highest cosine rival on games in which the genuine player is initially #2 or #3. Stratify owner bootstrap by pool. Report deployed identity outcomes with the original fold-1 thresholds.",
                    frozen_weight=w, baseline_threshold=frozen["fold1"]["0.0"]["threshold"],
                    verifier_threshold=frozen["fold1"][str(w)]["threshold"], previous_result_sha=digest(HERE / "verifier_result.json"))
    immutable_json(REPEAT / "verifier_protocol.json",protocol)
    immutable_json(REPEAT / "selection.json",{"queries":group,"parent_selection_sha":digest(HERE / "selection.json"),
                   "protocol_sha":digest(REPEAT / "verifier_protocol.json")})
    print("frozen replication",len(group),"new players",flush=True)


def inference():
    previous = read(HERE / "selection.json")["queries"]
    new = read(REPEAT / "selection.json")["queries"]
    games, duplicates = load_queries(previous + new, read(PILOT / "protocol.json"))
    assert len(games) == 2160 and not duplicates, duplicates
    # The archived scorer takes its input/output location from its module-level HERE.
    # Only point it at this new selection. Every network, split function and score stays frozen.
    verify_scores.HERE = REPEAT
    verify_scores.main()


def ci(pool_values):
    assert all(len(v) > 0 for v in pool_values)
    rng = np.random.default_rng(20261118)
    means = np.concatenate(pool_values)
    draws = sum(rng.choice(v, (10000, len(v))).sum(1) for v in pool_values) / len(means)
    return {"mean": float(means.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "players": len(means)}


def report():
    verify_evaluate.HERE = REPEAT
    cfg = read(REPEAT / "verifier_protocol.json")
    original = read(HERE / "selection.json")["queries"]
    repeated = read(REPEAT / "selection.json")["queries"]
    assert len({q["owner"] for q in original+repeated}) == len(original)+len(repeated)
    assert len({k for q in original+repeated for k in q["keys"]}) == (len(original)+len(repeated))*5
    all_margins, all_baseline, all_adjusted, per_pool = [], [], [], {}
    with np.load(REPEAT / "verifier_scores.npz", allow_pickle=False) as z:
        assert z["protocol_sha"].item()==digest(REPEAT / "verifier_protocol.json")
        assert z["selection_sha"].item()==digest(REPEAT / "selection.json")
        assert z["code_sha"].item()==digest(HERE / "verify_scores.py")
        with np.load(HERE / "verifier_scores.npz",allow_pickle=False) as old:
            assert all(z[f"m218sha{f}"].item()==old[f"m218sha{f}"].item() for f in (1,2))
        for fold in (1, 2):
            t = verify_evaluate.load(z, fold)
            _, base = verify_evaluate.observe(t, 0., cfg["baseline_threshold"])
            _, adjusted = verify_evaluate.observe(t, cfg["frozen_weight"], cfg["verifier_threshold"])
            observed = []
            n = 0
            for owner in range(72):
                margins = []
                for i in range(2 * owner, 2 * owner + 2):
                    top = t["top"][i]
                    if owner in top[1:]:
                        margins.append(t["raw"][i, owner] - t["raw"][i, top[0]])
                if margins:
                    observed.append(np.mean(margins))
                    n += len(margins)
            margin = np.array(observed)
            per_pool[str(fold)]={"hard_games":n,"hard_owners":len(margin),"hard_margin":float(margin.mean()),
                                 "known_shortlisted_games":int(sum(i//2 in t["top"][i] for i in range(144)))}
            all_margins.append(margin)
            all_baseline.append(base)
            all_adjusted.append(adjusted)
    operational = {}
    for k in all_baseline[0]:
        delta = ci([a[k] - b[k] for a, b in zip(all_adjusted, all_baseline)])
        b = np.concatenate([t[k] for t in all_baseline])
        a = np.concatenate([t[k] for t in all_adjusted])
        operational[k] = {"baseline_games": int(2*b.sum()), "adjusted_games": int(2*a.sum()),
                          "total_games": len(b)*2, "delta": delta}
    hard_margin = ci(all_margins)
    result={"players":len(repeated),"known_query_games":288,"unknown_query_games":144,"per_pool":per_pool,
            "hard_rival_margin":hard_margin,"hard_margin_replicated":hard_margin["ci95"][0]>0,"operational":operational,
            "operational_gate":operational["known_correct_accept"]["delta"]["ci95"][0]>0 and operational["unknown_accept"]["delta"]["mean"]<=0,
            "scores_sha256":digest(REPEAT / "verifier_scores.npz"),"code_sha256":digest(HERE / "replicate.py")}
    immutable_json(REPEAT / "result.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("select","score","report"))
    {"select":select,"score":inference,"report":report}[parser.parse_args().mode]()
