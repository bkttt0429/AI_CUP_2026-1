"""Test whether the rank contrast reverses when the opponent changes areas."""
import json
import numpy as np
from scipy.special import softmax
from destination import data, distance, features
from extra import EXTRA
from mechanism import matched_difference
from response import (HERE,RV,SHARDS,LEVELS,aggregate,balanced_mean_over_n,calibrated,cluster_intervals,
                      decide,fit_bernoulli,get_evidence,groups_of_questions,guard2,logits,owner_weights,
                      questions,save_json,score,sha)

PAIRS=[(4,5),(6,7),(0,9)]


def new_data():
    with np.load(EXTRA/"games.npz") as z:games={k:z[k] for k in z.files}
    chunks=[];fields=["game","A","B","last","actual","pA","pB","ply","rootlead"]
    for s in range(SHARDS):
        p=EXTRA/f"root_{s}.npz";m=json.loads((EXTRA/f"root_{s}.json").read_text())
        assert m["sha256"][p.name]==sha(p)
        with np.load(p) as z:
            mask=(z["eligible"]>0)&(distance(z["A"],z["B"])<=4)
            chunks.append({k:z[k][mask] for k in fields})
    return games,{k:np.concatenate([c[k] for c in chunks]) for k in fields}


def observed(g,e,game_mask,seed):
    x,_,miss,near,cell=features(e)
    gid=e["game"];grade=g["y"][gid];m=game_mask[gid]&miss
    # The final coordinate in the 54-cell encoding is the opponent's previous distance.
    rng=np.random.default_rng(seed);draws={};point={};margins={}
    for level in np.unique(PAIRS):
        owners=np.unique(g["pid"][game_mask&(g["y"]==level)])
        use=m&(grade==level)
        own=np.searchsorted(owners,g["pid"][gid[use]])
        index=own*54+cell[use]
        n=np.bincount(index,minlength=len(owners)*54).reshape(len(owners),54)
        wins=np.bincount(index,weights=near[use],minlength=n.size).reshape(n.shape)
        def separate(v):
            return np.swapaxes(v.reshape(*v.shape[:-1],18,3),-1,-2)
        point[level]=(separate(n.sum(0)),separate(wins.sum(0)))
        wt=rng.multinomial(len(owners),np.full(len(owners),1/len(owners)),size=4000)
        bn,bs=np.split(wt@np.concatenate([n,wins],axis=1),2,axis=1)
        draws[level]=separate(bn),separate(bs)
        totals=point[level][0].sum(1);s=point[level][1].sum(1)
        margins[LEVELS[level]]={"denominators":totals.tolist(),"rates":(s/totals).tolist()}
    results={}
    for lo,hi in PAIRS:
        effect=matched_difference(*point[lo],*point[hi])
        sample=matched_difference(*draws[lo],*draws[hi])
        interaction=sample[:,2]-sample[:,0]
        results[LEVELS[lo]+"/"+LEVELS[hi]]={"matched_differences_near_mid_far":effect.tolist(),
                "far_minus_near":float(effect[2]-effect[0]),"ci95":np.quantile(interaction,[.025,.975]).tolist()}
    return {"misses":int(m.sum()),"per_grade":margins,"contrasts":results}


def extended(x,cell):
    return np.column_stack([x,cell%3==0,cell%3==2])


def main():
    g,e=data();g2,e2=new_data()
    x,_,miss,near,cell=features(e);x=extended(x[miss],cell[miss]);target=near[miss].astype(float)
    gid=e["game"][miss];gr=g["y"][gid];fold=g["fold"][gid];new=g["fresh"][gid];owner=g["pid"][gid]
    nx,_,nm,nt,cell2=features(e2);nx=extended(nx[nm],cell2[nm]);ng=e2["game"][nm]
    old=json.loads((HERE/"destination.json").read_text())
    with np.load(HERE/"destination_predictions.npz") as p:
        prev=p["where_log_evidence"];exact=p["exact_log_evidence"];dy,dn,dpid,qf=(p[k] for k in ("y","n","pid","fold"));prev_answer=p["d_spatial"]
    with np.load(RV/"m183_dump.npz") as za:p=za["anat_post"].astype(float);ref=za["ct106_rule"];sq=za["SQ4"]
    p/=p.sum(axis=1,keepdims=True)
    qs,_=questions();groups=groups_of_questions(g,qs)
    norm=np.sqrt(np.maximum(1,aggregate(np.bincount(gid,minlength=len(g["keys"])),groups)))[:,None]
    q_tr=qf<=2;test=qf>=3
    signal=np.zeros_like(p);nll={};fitted={}
    for part in (1,2,0):
        tr=((fold==3-part) if part else fold<=2)&~new;val=(fold==part) if part else (fold>=3)&~new
        qval=(qf==part) if part else test
        assert not set(owner[tr])&set(owner[val])
        w=owner_weights(owner[tr],gr[tr]); cols=[0,2,3,5,8,9]
        model=fit_bernoulli(x[tr],target[tr],gr[tr],w,cols)
        fitted[str(part)]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}
        out=get_evidence(model,x[val],target[val],gid[val],len(g["keys"]))
        signal[qval]=(aggregate(out,groups)/norm)[qval]-prev[qval]
        if part==0:
            common=fit_bernoulli(x[tr],target[tr],gr[tr],w,cols[:-2])
            for tag,xx,yy,tc,ow in (("old",x[val],target[val],gr[val],owner[val]),("new2",nx,nt[nm].astype(float),g2["y"][ng],g2["pid"][ng])):
                w=owner_weights(ow,tc);nll[tag]={}
                for name,m in (("shared_location_terms",common),("rank_by_location",model)):
                    s=logits(m,xx,tc);nll[tag][name]=float(w@(np.logaddexp(0,s)-yy*s))
    a,b=old["temperatures"];base=np.log(p)+a*exact+b*prev
    np.testing.assert_array_equal(prev_answer,guard2(decide(softmax(base,axis=1)),ref))
    c=calibrated(base[q_tr],signal[q_tr],dy[q_tr],dn[q_tr])
    predicted=guard2(decide(softmax(base+c*signal,axis=1)),ref)
    pred={"interaction":predicted,"previous":prev_answer,"SQ4":sq}
    sc={k:score(v,dy) for k,v in pred.items()};refs=("previous","SQ4")
    ds=np.column_stack([(sc["interaction"]-sc[k])[test] for k in refs])
    ci=cluster_intervals(ds,dy[test],dn[test],dpid[test]);pairs={}
    for j,k in enumerate(refs):
        v=sc["interaction"]-sc[k]
        metric=lambda m:balanced_mean_over_n(v[m],dy[m],dn[m])
        pairs["interaction-"+k]={"difference":metric(test),"ci95":ci[j].tolist(),"folds":[metric(qf==f) for f in (3,4)],
            "changed":int(np.sum(predicted[test]!=pred[k][test])),"improved":int(np.sum(v[test]>0)),"worsened":int(np.sum(v[test]<0))}
    behavior={tag:observed(gg,ee,m,seed) for tag,gg,ee,m,seed in (("historical",g,e,(g["fold"]>=3)&~g["fresh"],20261128),
        ("first_extra",g,e,g["fresh"],20261129),("reserved_extra",g2,e2,np.ones(len(g2["keys"]),bool),20261130))}
    dst=HERE/"interaction_predictions.npz";np.savez_compressed(dst,increment=signal,decision=predicted)
    src=[HERE/"interaction_protocol.json",HERE/"interaction.py",HERE/"destination.json",EXTRA/"corpus.json",dst]
    result={"status":"exploratory","temperature":c,"competition":pairs,
        "score":balanced_mean_over_n(sc["interaction"][test],dy[test],dn[test]),"fits":fitted,"choice_nll":nll,
        "behavior":behavior,"independent_games_same_people":json.loads((EXTRA/"corpus.json").read_text())["games"],
        "sha256":{f.name:sha(f) for f in src}}
    save_json(HERE/"interaction.json",result)
    print(json.dumps({"temp":c,"competition":pairs,"effects":{k:v["contrasts"] for k,v in behavior.items()}},indent=2),flush=True)


if __name__=="__main__":
    main()
