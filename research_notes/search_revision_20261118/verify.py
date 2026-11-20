"""Independently reproduce frozen competition decisions and check the game and engine manifests."""
import gzip
import hashlib
import json
from collections import Counter

import numpy as np
from scipy.special import softmax

from corpus import HERE, RV, SHARDS, save_json, sha
from aicup.metric import LEVELS, expected_score_decision, score
from aicup.qeval import paired_diff
from search import BIN, CFG, MODEL


def canonical_replays():
    # Discard SGF metadata and target-player identity before comparing full moves and setup.
    with np.load(HERE/"games.npz") as z:
        partition = np.where(z["fresh"],2,np.where(z["fold"]>=3,1,0))
        assert np.array_equal(np.unique(partition), np.arange(3))
    sets=[set() for _ in range(3)]
    metadata=[set() for _ in range(3)]
    rows=Counter()
    seen=set()
    for s in range(SHARDS):
        with gzip.open(HERE/f"corpus_{s}.tsv.gz","rb") as file:
            for line in file:
                index,digest,_,handicap,setup,moves=line.rstrip().split(b"\t")
                g=int(index);assert g not in seen;seen.add(g)
                k=partition[g];rows[int(k)]+=1
                sets[k].add(hashlib.sha256(b"\0".join([handicap,setup,moves])).digest())
                metadata[k].add(digest)
    assert seen==set(range(len(partition)))
    pairs={(0,1):"fit_vs_eval",(0,2):"fit_vs_new",(1,2):"eval_vs_new"}
    return {"game_records_fit_eval_new":[rows[i] for i in range(3)],
            "distinct_full_replays":[len(a) for a in sets],"distinct_raw_SGF":[len(a) for a in metadata],
            "full_replay_overlap":{key:len(sets[i]&sets[j]) for (i,j),key in pairs.items()},
            "raw_SGF_overlap":{key:len(metadata[i]&metadata[j]) for (i,j),key in pairs.items()}}


def artifacts():
    check={}
    external={p.name:p for p in (MODEL,BIN.parent/"usr/bin/katago",CFG,RV/"m183_dump.npz")}
    for f in [HERE/"corpus.json",*(HERE/f"root_{s}.json" for s in range(SHARDS)),
              *(HERE/f"search_{s}.json" for s in range(SHARDS)),HERE/"rank_results.json",HERE/"destination.json"]:
        manifest=json.loads(f.read_text())
        for name,digest in manifest["sha256"].items():
            if name not in check:
                local=HERE/"parts"/name if name.startswith(("children_","fixed_children_")) else HERE/name
                check[name]=sha(external.get(name,local))
            assert check[name]==digest,(f,name)
    return len(check)


def original_guard(new, old):
    keep=(np.isin(old,[LEVELS.index("A"),LEVELS.index("D")]) |
          ((old==LEVELS.index("5D"))&(new==LEVELS.index("6D"))) | (abs(new-old)>=2))
    return np.where(keep,old,new)


def main():
    checked=artifacts()
    results=json.loads((HERE/"rank_results.json").read_text())
    a,b=results["temperatures"]
    with np.load(RV/"m183_dump.npz") as baseline, np.load(HERE/"predictions.npz") as saved:
        for k in ("y","pid","n","fold"):np.testing.assert_array_equal(baseline[k],saved[k])
        y,pid,n,fold=(saved[k] for k in ("y","pid","n","fold"))
        test=fold>=3
        p=baseline["anat_post"].astype(float)
        lp=np.log(p)
        ref=baseline["ct106_rule"]
        probabilities={"SQ4":p,"marginal":softmax(lp+a*saved["original"],axis=1),
            "depth":softmax(lp+a*saved["original"]+b*saved["inc"],axis=1),
            "fixed_depth":softmax(lp+a*saved["original"]+saved["inc"],axis=1)}
        scores={};decisions={}
        for name,prob in probabilities.items():
            raw=np.array([expected_score_decision(row) for row in prob])
            pred=original_guard(raw,ref)
            np.testing.assert_array_equal(pred,saved["d_"+name])
            if name=="SQ4":np.testing.assert_array_equal(pred,baseline["SQ4"])
            decisions[name]=pred[test].copy()
            scores[name]=np.fromiter((score(x,t) for x,t in zip(pred[test],y[test])),dtype=float)
    official={}
    for first,second in (("depth","SQ4"),("depth","marginal")):
        comparison=first+"-"+second
        actual=paired_diff(scores[first],scores[second],y[test],pid[test],n[test],n=1500,seed=283)
        registered=results["pair_results"][comparison]
        np.testing.assert_allclose(actual["diff"],registered["diff"],atol=1e-13)
        for size,v in actual["by_n"].items():
            np.testing.assert_allclose(v,registered["by_n"][str(size)],atol=1e-13)
        official[comparison]={"difference":actual["diff"],"ci95_independent_seed":list(actual["ci95"])}
    spatial=json.loads((HERE/"destination.json").read_text());al,be=spatial["temperatures"]
    with np.load(HERE/"destination_predictions.npz") as d:
        for key,expected in (("y",y),("pid",pid),("n",n),("fold",fold)):
            np.testing.assert_array_equal(d[key],expected)
        centered=lp+al*d["exact_log_evidence"]
        for name,logits in (("exact",centered),("spatial",centered+be*d["where_log_evidence"])):
            pred=original_guard(np.array([expected_score_decision(row) for row in softmax(logits,axis=1)]),ref)
            np.testing.assert_array_equal(pred,d["d_"+name]);scores[name]=np.array([score(i,j) for i,j in zip(pred[test],y[test])])
            decisions[name]=pred[test].copy()
    for refname in ("SQ4","exact"):
        name="spatial-"+refname
        check=paired_diff(scores["spatial"],scores[refname],y[test],pid[test],n[test],n=1500,seed=284)
        np.testing.assert_allclose(check["diff"],spatial["competition"][name]["difference"],atol=1e-13)
        official[name]={"difference":check["diff"],"ci95_independent_seed":list(check["ci95"])}
    for sample in spatial["samples"].values():
        assert sample["menus"]==sum(sample[k] for k in ("exact","near","far","pass"))
        assert sample["near"]==sum(c["wins"] for c in sample["observed_by_grade"].values())
    interaction=json.loads((HERE/"interaction.json").read_text());extra=HERE/"confirmation"
    for name,digest in interaction["sha256"].items():
        assert sha((extra if name=="corpus.json" else HERE)/name)==digest
    for s in range(SHARDS):
        pth=extra/f"root_{s}.npz"
        assert sha(pth)==json.loads((extra/f"root_{s}.json").read_text())["sha256"][pth.name]
    with np.load(HERE/"interaction_predictions.npz") as z,np.load(HERE/"destination_predictions.npz") as dst:
        trial=lp+al*dst["exact_log_evidence"]+be*dst["where_log_evidence"]+interaction["temperature"]*z["increment"]
        pred=original_guard(np.array([expected_score_decision(row) for row in softmax(trial,axis=1)]),ref)
        np.testing.assert_array_equal(pred,z["decision"]);decisions["interaction"]=pred[test];scores["interaction"]=np.array([score(i,j) for i,j in zip(pred[test],y[test])])
    for refname,reported in (("spatial","previous"),("SQ4","SQ4")):
        v=paired_diff(scores["interaction"],scores[refname],y[test],pid[test],n[test],n=1500,seed=285)
        np.testing.assert_allclose(v["diff"],interaction["competition"]["interaction-"+reported]["difference"],atol=1e-13)
        official["interaction-"+refname]={"difference":v["diff"],"ci95_independent_seed":list(v["ci95"])}
    corpus=canonical_replays()
    assert not any(corpus["raw_SGF_overlap"].values())
    m=json.loads((HERE/"mechanism.json").read_text())
    for tag in m["samples"].values():
        count=tag["choices"]["strong_reversal"]
        grades=tag["observed_by_grade"]
        assert sum(v["opportunities"] for v in grades.values())==count["named_untied"]
        assert sum(v["wins"] for v in grades.values())==count["searched_winner"]
        for ix in range(3):
            assert sum(tag["choices"][k]["choices_A_B_other"][ix] for k in ("raw_A_to_search_B","raw_B_to_search_A"))==count["choices_A_B_other"][ix]
    bridge=json.loads((HERE/"escape_overlap.json").read_text())
    assert sum(v["opportunities"] for v in bridge["strong_by_direction"].values())==bridge["strong_reversals"]["opportunities"]
    transitions={};truth=y[test]
    for a,b in (("depth","SQ4"),("depth","marginal"),("spatial","SQ4"),("spatial","exact"),("interaction","spatial")):
        diff=scores[a]-scores[b];before,after=decisions[b],decisions[a]
        transitions[a+"-"+b]={"newly_exact":int(np.sum((after==truth)&(before!=truth))),
            "lost_exact":int(np.sum((after!=truth)&(before==truth))),"changed_equal_score":int(np.sum((before!=after)&(diff==0))),
            "by_true_rank_better_worse":{grade:[int(np.sum((diff>0)&(truth==k))),int(np.sum((diff<0)&(truth==k)))]
                                       for k,grade in enumerate(LEVELS)}}
    audit={"checked_hashed_sources":checked,"question_rows_reconstructed":len(fold),"held_out_rows":int(test.sum()),
           "methods_with_independently_recomputed_decisions":list(scores),"official_metric":official,
           "corpus":corpus,"choice_counts_reconciled":True,"competition_changes":transitions}
    save_json(HERE/"verification.json",audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":
    main()
