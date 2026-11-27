"""Model take-up of a physically available alternative before the action is observed."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from extract import HERE,PREV,R1,questions,save_json,sha
from evaluate import balanced_mean_over_n,cluster_intervals,decide,guard2,score
from fit import PAIRS


def weights(owner,grade):
    owners,idx,num=np.unique(owner,return_inverse=True,return_counts=True)
    ranks=np.zeros(len(owners),int);ranks[idx]=grade
    np.testing.assert_array_equal(ranks[idx],grade)
    bygrade=np.bincount(ranks,minlength=10)
    assert len(bygrade)==10 and min(bygrade)>0
    w=1/(10*bygrade[grade]*num[idx])
    np.testing.assert_allclose(w.sum(),1.)
    return w


def covariance(e,m):
    ply,stones=e["ply"][m],e["stones"][m]
    return np.column_stack([np.ones(m.sum()),np.log1p(e["size"][m]),stones==2,stones>=3,
                            e["local"][m],(ply>=80)&(ply<160),ply>=160])


def matrix(context,grade,price,model):
    if model=="common": return context
    indicator=np.eye(10)[grade]
    if model=="intercept": return np.column_stack([context,indicator])
    return np.column_stack([context,indicator,indicator*price[:,None]])


def fit_model(x,target,w):
    def fun(beta):
        logit=x@beta
        penalty=beta.copy();penalty[0]=0
        return w@(np.logaddexp(0,logit)-target*logit)+.01*(penalty@penalty), x.T@(w*(expit(logit)-target))+.02*penalty
    sol=minimize(fun,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"ftol":1e-14,"gtol":1e-10,"maxiter":400})
    assert sol.success and abs(sol.jac).max()<1e-5
    return sol.x


def main():
    file=HERE/"offer_events.npz";assert sha(file)==json.loads((HERE/"offer_extraction.json").read_text())["sha256"][file.name]
    with np.load(file) as z: e={k:z[k] for k in z.files}
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z: keys=z["keys"]
    ix={k:i for i,k in enumerate(keys)}
    games=[np.array([ix[k] for k in q[4]]) for q in qs]
    with np.load(R1/"results/r_variants/m183_dump.npz") as z:
        fold,n,pid,y,guard,saved=[z[k] for k in ("fold","n","pid","y","ct106_rule","SQ4")]
        p=z["anat_post"].astype(float)
    p/=p.sum(axis=1,keepdims=True)
    owner,owner_idx=np.unique(pid,return_inverse=True); game_owner=np.full(len(keys),-1)
    game_fold=np.zeros(len(keys),int);game_rank=np.full(len(keys),-1)
    for i,ixs in enumerate(games):
        assert np.all((game_owner[ixs]==-1)|(game_owner[ixs]==owner_idx[i]))
        game_fold[ixs]=fold[i];game_rank[ixs]=y[i];game_owner[ixs]=owner_idx[i]
    assert min(game_owner)>=0
    binary=(e["number"]==1)&(e["choice"]!=2)
    e_test=game_fold[e["game"]]>=3; target=(e["choice"][binary]==1).astype(float)
    bt=e_test[binary]; br=game_rank[e["game"][binary]]; bo=game_owner[e["game"][binary]]
    assert not set(bo[bt])&set(bo[~bt])
    train_weights=weights(bo[~bt],br[~bt]);test_weights=weights(bo[bt],br[bt])
    context=covariance(e,binary);price=np.log(np.minimum(e["stones"][binary],6))
    touched_games=e["game"][binary]
    counts=np.bincount(touched_games,minlength=len(keys)); k=np.array([counts[ixs].sum() for ixs in games])
    preds={"SQ4":saved}; nll={}; tables={}
    for model in ("common","intercept","slope"):
        m=matrix(context,br,price,model)
        beta=fit_model(m[~bt],target[~bt],train_weights)
        true_logit=m[bt]@beta
        nll[model]=float(test_weights@(np.logaddexp(0,true_logit)-target[bt]*true_logit))
        ll=np.empty((len(br),10))
        for r in range(10):
            logit=matrix(context,np.full_like(br,r),price,model)@beta
            ll[:,r]=target*logit-np.logaddexp(0,logit)
        gg=np.zeros((len(keys),10));np.add.at(gg,touched_games,ll)
        qll=np.stack([gg[ixs].sum(axis=0) for ixs in games])/np.sqrt(np.maximum(k,1))[:,None]
        pp=p.copy()
        for lo,hi in PAIRS:
            mass=p[:,lo]+p[:,hi]; z=np.log(p[:,hi]/p[:,lo])+qll[:,hi]-qll[:,lo]
            pp[:,hi]=mass*expit(z);pp[:,lo]=mass*expit(-z)
        if model=="common": np.testing.assert_allclose(pp,p,atol=2e-14)
        preds[model]=guard2(decide(pp),guard);tables[model]=beta.tolist()
    np.testing.assert_array_equal(preds["common"],saved)
    masks={"all":e_test,"one":e_test&(e["number"]==1),"multiple":e_test&(e["number"]>=2),"none":e_test&(e["number"]==0)}
    for lbl,condition in (("one_stone",e["stones"]==1),("two_stones",e["stones"]==2),("three_plus",e["stones"]>=3)):
        masks[lbl]=masks["one"]&condition
    summary={}
    for lbl,m in masks.items():
        counts=np.bincount(e["choice"][m],minlength=3)
        summary[lbl]={"events":int(m.sum()),"direct":int(counts[0]),"offered_rescue":int(counts[1]),"other":int(counts[2]),
                      "rescue_among_all":float(counts[1]/m.sum()),
                      "rescue_among_two":float(counts[1]/(counts[0]+counts[1])) if counts[0]+counts[1] else None}
    t=fold>=3;comparison=[("intercept","SQ4"),("slope","SQ4"),("slope","intercept")]
    sc={name:score(arg,y) for name,arg in preds.items()}
    d=np.column_stack([(sc[a]-sc[b])[t] for a,b in comparison])
    intervals=cluster_intervals(d,y[t],n[t],pid[t])
    comparisons={}
    for (a,b),ci in zip(comparison,intervals):
        diff=sc[a]-sc[b]
        comparisons[a+"-"+b]={"diff":balanced_mean_over_n(diff[t],y[t],n[t]),"ci95":ci.tolist(),
                  "folds":[balanced_mean_over_n(diff[fold==f],y[fold==f],n[fold==f]) for f in (3,4)],
                  "gained":int((diff[t]>0).sum()),"lost":int((diff[t]<0).sum())}
    dest=HERE/"offer_predictions.npz"
    np.savez_compressed(dest,fold=fold,pid=pid,y=y,n=n,**{"d_"+k:v for k,v in preds.items()})
    report={"train_binary_events":int((~bt).sum()),"eval_binary_events":int(bt.sum()),"menu":summary,"choice_nll":nll,
            "paired":comparisons,"beta":tables,"sha256":{p.name:sha(p) for p in (file,dest,Path(__file__),HERE/"menu_protocol.json")}}
    save_json(HERE/"offer_results.json",report); print(json.dumps({k:v for k,v in report.items() if k not in ("beta","sha256")},indent=2))


if __name__=="__main__":
    main()
