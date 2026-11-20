"""A held-out, unregularized comparison of real responses in overlapping observed opportunity strata."""
import json
from pathlib import Path
import numpy as np

from extract import HERE,PREV,questions,save_json,sha
from aicup.metric import LEVELS


def summarize(tables):
    # columns within the last axis are observed successes and exposures.
    left,right=tables
    nl,nr=left[...,1],right[...,1]
    pl=np.divide(left[...,0],nl,out=np.zeros_like(nl,dtype=float),where=nl>0)
    pr=np.divide(right[...,0],nr,out=np.zeros_like(nr,dtype=float),where=nr>0)
    w=np.divide(nl*nr,nl+nr,out=np.zeros_like(nr,dtype=float),where=nl+nr>0)
    return (w*(pr-pl)).sum(axis=-1)/w.sum(axis=-1)


def main():
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z: keys=z["keys"]
    ix={k:i for i,k in enumerate(keys)}
    lookup={q[1]:i for i,q in enumerate(qs)}
    owner=np.full(len(keys),-1); grade=np.full(len(keys),-1);fold=np.zeros(len(keys),int)
    for f,p,y,_,games in qs:
        ids=[ix[g] for g in games];owner[ids]=lookup[p];grade[ids]=y;fold[ids]=f
    with np.load(HERE/"offer_events.npz") as z:e={k:z[k] for k in z.files}
    delta=np.full(len(e["game"]),np.nan)
    for k in range(4):
        with np.load(HERE/f"offer_values_{k}.npz") as z: delta[z["ix"]]=z["delta"]
    pick=(e["number"]==1)&(e["choice"]!=2)&(fold[e["game"]]>=3)
    dev=np.digitize(delta[pick],[-5,-1,1,5])
    strata=(dev*2+(e["size"][pick]>1))*3+np.digitize(e["ply"][pick],[80,160])
    grades=grade[e["game"][pick]];pids=owner[e["game"][pick]]
    outcome=(e["choice"][pick]==1).astype(int)
    out={};rng=np.random.default_rng(20261119)
    for lo,hi in [(4,5),(6,7)]:
        tabs=[];boots=[];details={}
        for y in (lo,hi):
            m=grades==y; owners,inv=np.unique(pids[m],return_inverse=True)
            raw=np.zeros((len(owners),30,2),dtype=float)
            np.add.at(raw[:,:,0],(inv,strata[m]),outcome[m]);np.add.at(raw[:,:,1],(inv,strata[m]),1)
            tabs.append(raw.sum(axis=0))
            draws=rng.multinomial(len(owners),np.full(len(owners),1./len(owners)),size=5000)
            boots.append((draws@raw.reshape(len(owners),-1)).reshape(5000,30,2))
            totals=raw.sum(axis=(0,1))
            details[LEVELS[y]]={"players":len(owners),"rescues":int(totals[0]),"choices":int(totals[1]),
                "rates_by_value":{lab:float(outcome[m&v].mean()) for lab,v in [("below_minus_1",delta[pick]< -1),("within_1",abs(delta[pick])<=1),("above_1",delta[pick]>1)]}}
        common=(tabs[0][:,1]>0)&(tabs[1][:,1]>0)
        point=float(summarize(tabs));sample=summarize(boots)
        out[LEVELS[lo]+"/"+LEVELS[hi]]={"data":details,"available_strata":int(common.sum()),"total_strata":30,
            "overlap_choices":[int(tab[common,1].sum()) for tab in tabs],"adjusted_high_minus_low":point,
            "player_cluster_ci95":np.quantile(sample,[.025,.975]).tolist()}
    save_json(HERE/"observed_grades.json",{"method":"Exploratory grade-standardized observed take-up, folds 3/4. Five value bands times original chain size times three phases. Weight each common cell by n1*n2/(n1+n2), recalculated in each of 5000 within-grade player bootstrap draws. No regression, shrinkage, or extrapolation to absent offers.","pairs":out,"source_sha256":sha(Path(__file__))})
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()
