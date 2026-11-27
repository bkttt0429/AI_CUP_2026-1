"""Independently check competition scores and quantify grade information on a common two-move menu."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, xlogy

from extract import HERE,PREV,R1,questions,save_json,sha
from fit_offers import covariance,matrix
from aicup.metric import LEVELS,score,expected_score_decision
from aicup.qeval import paired_diff
from boundary_check import auc


def auc_ci(ref,updated,rank,player):
    """Paired rank-stratified player bootstrap of AUC, with average credit for ties."""
    uu,owner=np.unique(player,return_inverse=True); label=np.zeros(len(uu),bool);label[owner]=rank
    np.testing.assert_array_equal(label[owner],rank)
    def group_scores(value):
        order=np.argsort(value,kind="stable")
        starts=np.r_[0,1+np.flatnonzero(np.diff(value[order])!=0)]
        def eval_weights(w):
            sorted_w=w[:,owner[order]]
            pos=np.add.reduceat(sorted_w*rank[order],starts,axis=1)
            neg=np.add.reduceat(sorted_w*(~rank[order]),starts,axis=1)
            return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))
        return eval_weights
    a,b=group_scores(ref),group_scores(updated)
    allone=np.ones((1,len(uu)))
    point=float((b(allone)-a(allone))[0])
    np.testing.assert_allclose(point,auc(rank,updated)-auc(rank,ref),atol=1e-12)
    rng=np.random.default_rng(1118);hist=[]
    for _ in range(40):
        w=np.zeros((100,len(uu)))
        for r in (False,True):
            idx=np.flatnonzero(label==r)
            w[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=len(w))
        hist.extend(b(w)-a(w))
    return {"difference":point,"player_ci95":np.quantile(hist,[.025,.975]).tolist()}


def verify_scores():
    outputs={}
    with np.load(R1/"results/r_variants/m183_dump.npz") as m:
        for fname,report,key,variant in [("predictions.npz","results.json","paired","value"),
                            ("offer_predictions.npz","offer_results.json","paired","slope"),
                            ("value_response_predictions.npz","value_response.json","rank_paired","slope")]:
            with np.load(HERE/fname) as z:
                for col in ("fold","pid","n","y"):np.testing.assert_array_equal(z[col],m[col])
                np.testing.assert_array_equal(z["d_SQ4"],m["SQ4"])
                t=z["fold"]>=3;y,n,player=z["y"][t],z["n"][t],z["pid"][t]
                scores=[np.asarray([score(pred,truth) for pred,truth in zip(z[k][t],y)]) for k in ("d_"+variant,"d_SQ4")]
                calc=paired_diff(*scores,y,player,n,n=900,seed=98)
                other=json.loads((HERE/report).read_text())[key][variant+"-SQ4"]
                np.testing.assert_allclose(calc["diff"],other["diff"],atol=1e-13)
                assert np.max(abs(np.asarray(calc["ci95"])-other["ci95"]))<.00035
                outputs[fname]={"check_diff":float(calc["diff"]),"independent_900_draw_ci95":calc["ci95"]}
                if fname=="predictions.npz":
                    p0=z["p_SQ4"][t];p1=z["p_value"][t];top=np.sort(np.argsort(p0,axis=1)[:,-2:],axis=1)
                    for lo,hi in ((4,5),(6,7)):
                        ix=(top[:,0]==lo)&(top[:,1]==hi)&np.isin(y,[lo,hi])
                        outputs[LEVELS[lo]+"/"+LEVELS[hi]]=auc_ci(np.log(p0[ix,hi]/p0[ix,lo]),
                                    np.log(p1[ix,hi]/p1[ix,lo]),y[ix]==hi,player[ix])
                    np.testing.assert_array_equal([expected_score_decision(v) for v in m["anat_post"][::79]],m["anat_plain"][::79])
    return outputs


def entropy(p):
    return -(xlogy(p,p)+xlogy(1-p,1-p))/np.log(2.)


def info():
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z:keys=z["keys"]
    idx={k:i for i,k in enumerate(keys)}
    n_game=len(keys);fold=np.zeros(n_game,int);grades=np.full(n_game,-1);owners=np.full(n_game,-1);q_indices=[]
    player={q[1]:i for i,q in enumerate(qs)}
    for f,who,y,_,games in qs:
        inds=np.array([idx[k] for k in games]);q_indices.append(inds)
        fold[inds]=f;grades[inds]=y;owners[inds]=player[who]
    with np.load(HERE/"offer_events.npz") as z:e={k:z[k] for k in z.files}
    delta=np.full(len(e["game"]),np.nan)
    for s in range(4):
        with np.load(HERE/f"offer_values_{s}.npz") as z: delta[z["ix"]]=z["delta"]
    b=(e["number"]==1)&(e["choice"]!=2)&(fold[e["game"]]>=3)
    beta=np.array(json.loads((HERE/"value_response.json").read_text())["beta"]["slope"])
    payoff=np.clip(delta[b],-10,10)/5;ctx=np.column_stack([covariance(e,b),payoff]);truth=grades[e["game"][b]]
    real=(e["choice"][b]==1); pid=owners[e["game"][b]]; results={}
    for lo,hi in ((4,5),(6,7)):
        keep=np.isin(truth,[lo,hi]);g=truth[keep];y=real[keep];persons=pid[keep]
        p0,p1=[expit(matrix(ctx[keep],np.full_like(g,k),payoff[keep],"slope")@beta) for k in (lo,hi)]
        mid=(p0+p1)/2
        implied=entropy(mid)-.5*(entropy(p0)+entropy(p1))
        pk=np.where(g==lo,p0,p1)
        logscore=np.log2(np.where(y,pk,1-pk)/np.where(y,mid,1-mid))
        group_means=[];rng=np.random.default_rng(1118);boot=np.zeros(5000)
        for k in (lo,hi):
            idx_g=g==k;unique,inv=np.unique(persons[idx_g],return_inverse=True)
            avg=np.bincount(inv,logscore[idx_g])/np.bincount(inv)
            group_means.append(avg.mean())
            mult=rng.multinomial(len(avg),np.full(len(avg),1./len(avg)),size=len(boot))
            boot+=.5*(mult@avg)/len(avg)
        results[LEVELS[lo]+"/"+LEVELS[hi]]={"binary_choices":int(keep.sum()),"players":len(set(persons.tolist())),
            "mean_absolute_probability_gap":float(np.mean(abs(p1-p0))),"mean_fitted_js_bits_per_event":float(implied.mean()),
            "observed_bits_per_choice_grade_balanced_player_mean":float(np.mean(group_means)),"player_ci95_bits":np.quantile(boot,[.025,.975]).tolist()}
    cnt=np.bincount(e["game"][b],minlength=n_game); info={}
    for size in (5,10,20):
        k=np.array([cnt[inds].sum() for q,inds in zip(qs,q_indices) if q[0]>=3 and q[3]==size])
        info[str(size)]={"questions":int(len(k)),"median":float(np.median(k)),"zero":int((k==0).sum()),"mean":float(k.mean())}
    return {"grade_resolution":results,"binary_choice_coverage":info}


if __name__=="__main__":
    result={"competition_and_boundary_checks":verify_scores(),**info(),"source_sha256":sha(Path(__file__))}
    save_json(HERE/"verification.json",result); print(json.dumps(result,indent=2))
