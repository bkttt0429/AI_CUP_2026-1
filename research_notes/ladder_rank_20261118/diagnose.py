"""Cross-check rank results and measure repeatability on disjoint games of the same player."""
import json
from pathlib import Path

import numpy as np

from extract_games import HERE, RV, questions, save_json, sha
from evaluate import decide, guard2, slope
from aicup.metric import score
from aicup.qeval import paired_diff


def correlation(a,b):
    a, b = a-a.mean(axis=0), b-b.mean(axis=0)
    return (a*b).sum(axis=0) / np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0))


def center_by_grade(x,y):
    out = x.copy()
    for grade in range(10):
        out[y==grade] -= x[y==grade].mean(axis=0)
    return out


def half_features(c, role):
    v = np.moveaxis(c,-1,0)
    if role == "defense":
        total, favorable, chose, joint = v
        exposure, acted = total, chose
    else:
        exposure, _, total, favorable, chose, joint = v
        acted = total
    count = exposure.sum(axis=1)
    rate = (acted.sum(axis=1)+.5)/(count+1)
    return np.column_stack([slope(total,favorable,chose,joint), count/10, rate])


def reliability(qs, role):
    selected = sorted((q for q in qs if q[0]>=3 and q[3]==20),key=lambda q:q[1])
    assert len({q[1] for q in selected})==len(selected)
    y = np.asarray([q[2] for q in selected])
    fp = "game_counts.npz" if role=="defense" else "attack_counts.npz"
    with np.load(HERE/fp) as z:
        ix = {k:i for i,k in enumerate(z["keys"])}
        games = np.array([[ix[k] for k in sorted(q[4])] for q in selected])
        assert z["valid"][games].all()
        per_game = z["cells"][games]
    assert games.size == len(np.unique(games))
    rng = np.random.default_rng(20261121)
    repeats, within, boot = [], [], []
    for split in range(20):
        order = np.argsort(rng.random(games.shape),axis=1)
        rows = np.arange(len(y))[:,None]
        left = half_features(per_game[rows,order[:,:10]].sum(axis=1,dtype=np.int64),role)
        right = half_features(per_game[rows,order[:,10:]].sum(axis=1,dtype=np.int64),role)
        repeats.append(correlation(left,right))
        within.append(correlation(center_by_grade(left,y),center_by_grade(right,y)))
        if split==0:
            br = np.random.default_rng(20261122)
            groups = [np.flatnonzero(y==k) for k in range(10)]
            for _ in range(1000):
                sample = np.concatenate([br.choice(g,len(g),replace=True) for g in groups])
                boot.append(correlation(left[sample],right[sample]))
    ci = np.quantile(boot,[.025,.975],axis=0)
    out = {}
    for col, name in enumerate(("coupling","opportunities_per_game","tendency_to_extend_or_attack")):
        out[name] = {"first_split_r":float(repeats[0][col]),"first_split_ci95":ci[:,col].tolist(),
                     "twenty_splits_median_r":float(np.median(np.asarray(repeats)[:,col])),
                     "within_true_grade_twenty_splits_median_r":float(np.median(np.asarray(within)[:,col]))}
    return {"players":len(selected),"distinct_games_per_player":20,"measurements":out}


def verify(role):
    with np.load(HERE/(role+"_predictions.npz")) as z:
        report=json.loads((HERE/(role+"_results.json")).read_text())
        with np.load(RV/"m183_dump.npz") as ref:
            idx = np.flatnonzero(ref["fold"]<=2) if role=="pooled" else z["kept_indices"]
            for key in ("fold","n","pid","y"):
                np.testing.assert_array_equal(z[key],ref[key][idx])
            np.testing.assert_array_equal(z["pred_SQ4"],ref["SQ4"][idx])
            if role!="pooled":
                lp = np.log(ref["anat_post"][idx].astype(float))
                for mod in ("full","opportunity","marginal","shuffle"):
                    logits=lp + z["tilt_"+mod][:,None]*np.arange(10)
                    p=np.exp(logits-logits.max(axis=1,keepdims=True))
                    np.testing.assert_array_equal(guard2(decide(p),ref["ct106_rule"][idx]),z["pred_"+mod])
        mask = z["fold"]<=2 if role=="pooled" else z["fold"]>=3
        y,pid,n=[z[key][mask] for key in ("y","pid","n")]
        a,b=[np.asarray([score(v,t) for v,t in zip(z[k][mask],y)]) for k in ("pred_full","pred_SQ4")]
        ci=paired_diff(a,b,y,pid,n,n=1000,seed=98)
        primary=report["paired"]["full_minus_SQ4"]
        np.testing.assert_allclose(ci["diff"],primary["difference"],atol=1e-13)
        ci_delta=float(np.max(np.abs(np.asarray(ci["ci95"])-primary["ci95"])))
        assert ci_delta<.0003
        for s, value in ci["by_n"].items():
            np.testing.assert_allclose(value,primary["by_n"][str(int(s))],atol=1e-13)
        assert report["sha256"][role+"_predictions.npz"]==sha(HERE/(role+"_predictions.npz"))
        return {"official_implementation_point":float(ci["diff"]),"independent_1000_draw_ci95":ci["ci95"],
                "maximum_endpoint_difference":ci_delta}


if __name__=="__main__":
    q, _ = questions()
    test_n = np.array([[4,0]])
    assert np.isclose(slope(test_n,np.array([[2,0]]),np.array([[2,0]]),np.array([[2,0]]))[0],.2)
    assert np.isclose(slope(test_n,np.array([[2,0]]),np.array([[2,0]]),np.array([[0,0]]))[0],-.2)
    assert slope(test_n,test_n,test_n,test_n)[0]==0
    stability={r:reliability(q,r) for r in ("defense","attack")}
    checks={r:verify(r) for r in ("defense","attack","pooled")}
    result={"split_half":stability,"independent_metric_check":checks,"source_sha256":sha(Path(__file__))}
    save_json(HERE/"diagnostics.json",result)
    print(json.dumps(result,indent=2))
