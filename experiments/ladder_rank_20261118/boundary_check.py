"""Descriptive follow-up on neighboring grades already named by R1 research."""
import numpy as np
from scipy.stats import rankdata

from extract_games import HERE, RV, questions, save_json, sha
from aicup.metric import LEVELS, score

PAIRS=[("C","B"),("B","A"),("1D","2D"),("3D","4D")]


def auc(positive, value):
    # Mann–Whitney statistic, including half credit for tied scores.
    positives = int(positive.sum())
    negatives = len(positive) - positives
    return (rankdata(value)[positive].sum() - positives*(positives+1)/2) / (positives*negatives)


def main():
    with np.load(RV/"m183_dump.npz") as z:
        p, y, n, f, sq4 = [z[k] for k in ("anat_post","y","n","fold","SQ4")]
    top_two = np.sort(np.argsort(p,axis=1)[:,-2:],axis=1)
    base = np.array([score(pr,t) for pr,t in zip(sq4,y)])
    rows={}
    for lower,upper in PAIRS:
        lo,hi=LEVELS.index(lower),LEVELS.index(upper)
        mask=(f>=3) & (top_two[:,0]==lo) & (top_two[:,1]==hi) & np.isin(y,[lo,hi])
        target=y[mask]==hi
        odds=np.log(p[mask,hi]/p[mask,lo])
        baseline=auc(target,odds)
        result={"questions":int(mask.sum()),"truth_by_level":[int((~target).sum()),int(target.sum())],
                "base_auc_conditional":float(baseline),"roles":{}}
        for role in ("defense","attack"):
            with np.load(HERE/(role+"_predictions.npz")) as pred:
                np.testing.assert_array_equal(pred["kept_indices"],np.arange(len(y)))
                score_new=np.array([score(pr,t) for pr,t in zip(pred["pred_full"][mask],y[mask])])
                difference=score_new-base[mask]
                updated=odds+(hi-lo)*pred["tilt_full"][mask]
                result["roles"][role]={"improved":int((difference>0).sum()),"worsened":int((difference<0).sum()),
                                      "auc_change":float(auc(target,updated)-baseline)}
        rows[lower+"|"+upper]=result
    save_json(HERE/"boundaries.json",{"selection":"Folds 3/4; two leading SQ4 posterior probabilities are the indicated pair and truth belongs to the pair. Overlapping questions. Retrospective description only.","pairs":rows,
                                    "sources":{p.name:sha(p) for p in (RV/"m183_dump.npz",HERE/"defense_predictions.npz",HERE/"attack_predictions.npz")}})
    for k,v in rows.items():
        print(k,v,flush=True)
    # Check how often the unclassified alternative to a straight escape is observed.
    qs,_=questions()
    with np.load(HERE/"game_counts.npz") as z:
        key={k:i for i,k in enumerate(z["keys"])}
        cell=z["cells"].astype(np.int64)
        alternatives=(cell[:,:,1]-cell[:,:,3]).sum(axis=1)
        selected=[q for q in qs if q[0]>=3]
        games=[[key[k] for k in q[4]] for q in selected]
        unique=np.unique(np.concatenate(games))
        assert z["valid"][unique].all()
        counts=np.asarray([alternatives[g].sum() for g in games])
        sizes=np.asarray([q[3] for q in selected])
        coverage={"unique_events":int(alternatives[unique].sum()),"games":int(len(unique)),"by_n":{}}
        for size in (5,10,20):
            v=counts[sizes==size]
            coverage["by_n"][str(size)]={"questions":int(len(v)),"one_or_more":int((v>=1).sum()),
                                      "two_or_more":int((v>=2).sum()),"median":float(np.median(v))}
        save_json(HERE/"alternative_coverage.json",coverage)
        print("Favorable straight escape available but not played",coverage,flush=True)


if __name__=="__main__":
    main()
