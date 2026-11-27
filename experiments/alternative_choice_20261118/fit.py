"""Test alternative actions as corrections to the frozen 1D/2D and 3D/4D odds."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

from extract import COLS, HERE, NAMES, PREV, R1, questions, save_json, sha
from evaluate import SIZES, balanced_mean_over_n, cluster_intervals, decide, guard2, score
from boundary_check import auc
from aicup.metric import LEVELS

PAIRS=[(4,5),(6,7)]
assert [(LEVELS[l],LEVELS[h]) for l,h in PAIRS]==[("1D","2D"),("3D","4D")]
NVAR=20
COMPARISONS=[("value","SQ4"),("value","choice"),("choice","control"),("choice","SQ4"),("control","SQ4"),
             ("outcome","value"),("outcome","SQ4"),("value","shuffle"),("shuffle","choice")]


def judges(e):
    delta=np.full(len(e["game"]),np.nan); deeper=np.full_like(delta,np.nan)
    maxdiff=[]
    for shard in range(4):
        path=HERE/f"judge_{shard}.npz"
        m=json.loads((HERE/f"judge_{shard}.json").read_text())
        assert sha(path)==m["sha256"][path.name]
        maxdiff.append(m["duplicate_max_points"])
        with np.load(path) as z:
            ix=z["ix"]; deep=z["deep_ids"]
            assert not np.isfinite(delta[ix]).any() and np.all(e["game"][ix]%4==shard)
            delta[ix]=z["opp_escape"]-z["opp_actual"]
            deeper[deep]=z["deep_opp_escape"]-z["deep_opp_actual"]
    assert np.isfinite(delta).all()
    audit=np.isfinite(deeper); a,b=delta[audit],deeper[audit]
    agree=np.sign(a)==np.sign(b); decisive=np.abs(a)>1.
    meta={"events":int(len(a)), "spearman":float(spearmanr(a,b).statistic),
          "median_abs_difference":float(np.median(np.abs(a-b))),"same_sign":float(agree.mean()),
          "raw_outside_one":int(decisive.sum()),"same_sign_raw_outside_one":float(agree[decisive].mean()),
          "maximum_identical_branch_difference":float(max(maxdiff))}
    return delta,meta


def shuffled_values(e,delta,fold_game):
    strata=(fold_game[e["game"]]*6+e["kind"])*3+np.digitize(e["ply"],[80,160])
    strata=(strata*2+(e["size"]>1))*2+e["local"]
    shuffled=delta.copy()
    rng=np.random.default_rng(20261118)
    for s in np.unique(strata):
        ix=np.flatnonzero(strata==s); shuffled[ix]=delta[rng.permutation(ix)]
    return shuffled


def summarize_games(e,delta,n_games):
    s=np.zeros((n_games,NVAR))
    ids=e["game"]; group=e["kind"]
    np.add.at(s,(ids,group),1.)
    normalized=np.clip(delta,-10,10)/10.
    def add(col,values):
        np.add.at(s[:,col],ids,values)
    add(6,normalized);add(7,delta>1);add(8,delta< -1)
    add(9,normalized*(group==0));add(10,normalized*np.isin(group,[1,2]));add(11,normalized*(group>=3))
    complete=e["replies"]==2
    add(12,complete); add(13,complete & (e["lost_second"]==1))
    add(14,complete & (e["lost_second"]==1) & (e["taken_window"]<e["size"]))
    add(15,np.where(complete,np.minimum(e["taken_window"],5),0))
    add(16,complete & (e["lost_second"]==1) & (e["taken_window"]>=e["size"]))
    add(17,complete & (e["lost_first"]==1));add(18,complete & (e["taken_window"]>0))
    add(19,complete & (e["final_lib"]>=2))
    np.testing.assert_array_equal(s[:,13],s[:,14]+s[:,16])
    return s


def feature_set(old, q, shuffled, n):
    seen, favorable, chose, joint=np.moveaxis(old,-1,0)
    alts=q[:,:6].sum(axis=1)
    np.testing.assert_array_equal(alts,(favorable-joint).sum(axis=1))
    control=np.column_stack([np.log1p(seen/n[:,None]),(favorable+.5)/(seen+1),(chose+.5)/(seen+1),np.log1p(alts/n)])
    choice=np.column_stack([control,(q[:,:6]+.5)/(alts[:,None]+3.)])
    def values(q):
        den=np.column_stack([alts,alts,alts,q[:,0],q[:,1]+q[:,2],q[:,3:6].sum(axis=1)])
        return q[:,6:12]/(den+4)
    value=np.column_stack([choice,values(q)])
    outcomes=np.column_stack([q[:,12]/(alts+4),q[:,13:16]/(q[:,12,None]+4)])
    return {"control":control,"choice":choice,"value":value,"outcome":np.column_stack([value,outcomes]),
            "shuffle":np.column_stack([choice,values(shuffled)])}


def calibrate(x, dev, p, n, y):
    out=p.copy()
    fits={}
    for lo,hi in PAIRS:
        tr=dev & np.isin(y,[lo,hi])
        odds=np.log(p[:,hi]/p[:,lo])
        z=np.column_stack([np.ones(len(n)),odds,odds**2,np.log(n)])
        target=(y[tr]==hi).astype(float)
        cell=np.searchsorted(SIZES,n[tr])*2+target.astype(int)
        den=np.bincount(cell,minlength=6)
        assert (den>0).all()
        w=1/(6*den[cell]); sw=np.sqrt(w)[:,None]
        proj=np.linalg.lstsq(z[tr]*sw,x[tr]*sw,rcond=None)[0]
        residual=x-z@proj
        center=w@residual[tr]
        scale=np.sqrt(w@((residual[tr]-center)**2)); scale[scale<1e-10]=1
        design=np.clip((residual-center)/scale,-4,4); xv=design[tr]; base=odds[tr]
        def objective(beta):
            pred=base+xv@beta
            return w@(np.logaddexp(0,pred)-target*pred)+.05*(beta@beta), xv.T@(w*(expit(pred)-target))+.1*beta
        fit=minimize(objective,np.zeros(x.shape[1]),method="L-BFGS-B",jac=True,options={"gtol":1e-9,"ftol":1e-14,"maxiter":400})
        assert fit.success and np.max(np.abs(fit.jac))<1e-5,fit.message
        v=odds+design@fit.x; mass=p[:,lo]+p[:,hi]
        out[:,hi]=mass*expit(v);out[:,lo]=mass*expit(-v)
        fits[LEVELS[lo]+"/"+LEVELS[hi]]={"beta":fit.x.tolist(),"residualizer":proj.tolist(),"center":center.tolist(),"scale":scale.tolist()}
        np.testing.assert_allclose(out[:,hi]+out[:,lo],mass,atol=1e-15)
    assert np.isfinite(out).all() and (out>0).all()
    np.testing.assert_allclose(out.sum(axis=1),p.sum(axis=1),atol=1e-14)
    return out,fits


def mechanism(e,d,mask):
    complete=e["replies"]==2; lost=(e["lost_second"]==1)
    out={}
    for kind,name in [(None,"all")]+list(enumerate(NAMES)):
        sel=mask if kind is None else mask & (e["kind"]==kind)
        if not np.any(sel): continue
        c=sel&complete; v=d[sel]
        out[name]={"events":int(sel.sum()),"share":float(sel.sum()/mask.sum()),"complete":int(c.sum()),
                   "loss_first_complete":int(np.sum(c & (e["lost_first"]==1))),"loss_second":int(np.sum(c&lost)),
                   "loss_with_material_recovered":int(np.sum(c&lost&(e["taken_window"]>=e["size"]))),
                   "captured_enemy_in_window":int(np.sum(c&(e["taken_window"]>0))),
                   "two_liberties_or_more_at_horizon":int(np.sum(c&(e["final_lib"]>=2))),
                   "values_quartiles":np.quantile(v,[.25,.5,.75]).tolist(),
                   "better_than_one":int((v>1).sum()),"worse_than_one":int((v< -1).sum()),
                   "near":int(e["near"][sel].sum()),"immediate_captures":int((e["immediate_taken"][sel]>0).sum()),
                   "new_ataris":int((e["ataris"][sel]>0).sum())}
    return out


def reliability(g,qs,indices):
    mask=[i for i,q in enumerate(qs) if q[0]>=3 and q[3]==20]
    mask.sort(key=lambda i:qs[i][1])
    assert len({qs[i][1] for i in mask})==len(mask)
    games=np.array([sorted(indices[i]) for i in mask])
    assert games.size==len(np.unique(games))
    per_game=g[games]
    rng=np.random.default_rng(20261121); cors=[]
    def features(q):
        n=q[:,:6].sum(axis=1)
        return np.column_stack([n/10,(q[:,0]+.5)/(n+1),(q[:,4]+.5)/(n+1),q[:,6]/(n+4),
                                q[:,8]/(n+4),q[:,13]/(q[:,12]+4),q[:,14]/(q[:,12]+4)])
    for _ in range(20):
        order=np.argsort(rng.random(games.shape),axis=1)
        r=np.arange(len(mask))[:,None]
        l=features(per_game[r,order[:,:10]].sum(axis=1)); h=features(per_game[r,order[:,10:]].sum(axis=1))
        l-=l.mean(axis=0);h-=h.mean(axis=0)
        cors.append((l*h).sum(axis=0)/np.sqrt((l*l).sum(axis=0)*(h*h).sum(axis=0)))
    labels="frequency rescue_share far_share mean_value below_minus_one loss uncompensated_loss".split()
    return {"players":len(mask),"median_pearson":dict(zip(labels,np.median(cors,axis=0).tolist()))}


def main():
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z:
        keys,old=z["keys"],z["cells"].astype(int)
        assert z["valid"].all()
    pos={k:i for i,k in enumerate(keys)}
    indices=[np.array([pos[k] for k in q[4]]) for q in qs]
    gamefold=np.zeros(len(keys),int)
    for ix,q in zip(indices,qs):
        assert np.all(np.isin(gamefold[ix],[0,q[0]]))
        gamefold[ix]=q[0]
    assert np.all(gamefold>0)
    with np.load(HERE/"events.npz") as z:
        e={key:z[key] for key in COLS}
    delta,judges_meta=judges(e)
    actual=summarize_games(e,delta,len(keys))
    fake=summarize_games(e,shuffled_values(e,delta,gamefold),len(keys))
    np.testing.assert_array_equal(actual[:,:6],fake[:,:6]);np.testing.assert_array_equal(actual[:,12:],fake[:,12:])
    with np.load(R1/"results/r_variants/m183_dump.npz") as z:
        y,n,pid,fold,guard,saved,raw=[z[key] for key in ("y","n","pid","fold","ct106_rule","SQ4","anat_plain")]
        p=z["anat_post"].astype(float)
    p/=p.sum(axis=1,keepdims=True)
    np.testing.assert_array_equal(decide(p),raw)
    np.testing.assert_array_equal(guard2(raw,guard),saved)
    dev,test=fold<=2,fold>=3
    assert not (set(pid[dev])&set(pid[test]))
    aggregate=lambda arr:np.stack([arr[ix].sum(axis=0) for ix in indices])
    all_x=feature_set(aggregate(old),aggregate(actual),aggregate(fake),n)
    probs={"SQ4":p}; models={}
    for name,x in all_x.items():
        probs[name],models[name]=calibrate(x,dev,p,n,y)
    pr={k:guard2(decide(v),guard) for k,v in probs.items()}
    sc={k:score(v,y) for k,v in pr.items()}
    nll={k:-np.log(v[np.arange(len(y)),y]) for k,v in probs.items()}
    def metric(v,mask=test):
        return balanced_mean_over_n(v[mask],y[mask],n[mask])
    differences=np.column_stack([(sc[a]-sc[b])[test] for a,b in COMPARISONS])
    intervals=cluster_intervals(differences,y[test],n[test],pid[test])
    pairs={}
    for (a,b),ci in zip(COMPARISONS,intervals):
        d=sc[a]-sc[b]
        pairs[a+"-"+b]={"diff":metric(d),"ci95":ci.tolist(),"folds":[metric(d,fold==f) for f in (3,4)],
             "by_n":{str(s):balanced_mean_over_n(d[test],y[test],n[test],sizes=[s]) for s in SIZES},
             "changed":int(np.sum(pr[a][test]!=pr[b][test])),"gained":int(np.sum(d[test]>0)),"lost":int(np.sum(d[test]<0)),
             "nll_change":metric(nll[a]-nll[b])}
    boundaries={}
    top=np.sort(np.argsort(p,axis=1)[:,-2:],axis=1)
    for lo,hi in PAIRS:
        label=LEVELS[lo]+"/"+LEVELS[hi]
        for label2,keep in [("all",np.ones(len(y),bool)),("top_two",(top[:,0]==lo)&(top[:,1]==hi))]:
            m=test & np.isin(y,[lo,hi]) & keep
            target=y[m]==hi
            results={}
            for name,post in probs.items():
                odds=np.log(post[m,hi]/post[m,lo])
                loss=np.logaddexp(0,odds)-target*odds
                diff=(sc[name]-sc["SQ4"])[m]
                results[name]={"auc":float(auc(target,odds)),"conditional_nll":float(loss.mean()),
                               "gained":int((diff>0).sum()),"lost":int((diff<0).sum())}
            boundaries[label+"_"+label2]={"questions":int(m.sum()),"players":len(set(pid[m])),"models":results}
    v=pairs["value-SQ4"]; c=pairs["value-choice"]
    passed=(v["diff"]>=.001 and min(v["folds"])>0 and v["ci95"][0]>0 and c["ci95"][0]>0)
    dest=HERE/"predictions.npz"
    np.savez_compressed(dest,fold=fold,pid=pid,y=y,n=n,**{"p_"+k:v for k,v in probs.items()},**{"d_"+k:v for k,v in pr.items()})
    mtest=gamefold[e["game"]]>=3
    out={"primary_pass":bool(passed),"development":[int(dev.sum()),len(set(pid[dev]))],"evaluation":[int(test.sum()),len(set(pid[test]))],
         "unique_evaluation_games":int((gamefold>=3).sum()),"evaluation_mechanism":mechanism(e,delta,mtest),
         "evaluation_split_half":reliability(actual,qs,indices),"searched_audit":judges_meta,"models":models,"paired":pairs,"boundaries":boundaries,
         "metrics":{k:{"score":metric(sc[k]),"nll":metric(nll[k])} for k in probs},
         "sha256":{q.name:sha(q) for q in (Path(__file__),HERE/"protocol.json",HERE/"events.npz",dest,
                      R1/"results/r_variants/m183_dump.npz",*(HERE/f"judge_{i}.npz" for i in range(4)))}}
    save_json(HERE/"results.json",out)
    print(json.dumps({"primary_pass":passed,"models":out["metrics"],"differences":pairs,"stability":out["evaluation_split_half"]},indent=2),flush=True)


if __name__=="__main__":
    main()
