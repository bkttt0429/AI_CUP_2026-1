"""Cross-fit the grade-conditional response to equal-budget search and test frozen SQ4 corrections."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit,logsumexp
from scipy.stats import spearmanr

from corpus import HERE,PREV,R1,RV,SHARDS,questions,save_json,sha
from evaluate import SIZES,balanced_weights,balanced_mean_over_n,cluster_intervals,decide,guard2,score
from aicup.metric import LEVELS
from boundary_check import auc

GRADES=np.arange(10)
COMMON=["intercept","log_prior_ratio","policy_mass","phase","root_lead_magnitude","raw_B_minus_A","searched_direction",
        "search_revision","signed_strong_reversal"]
MARG=[0,1,5,6];DEPTH=MARG+[7,8]
PAIR_METRICS=[("depth","SQ4"),("depth","marginal"),("marginal","SQ4"),("depth","shuffle"),("fixed_depth","SQ4")]


def events():
    shards=[];audits=[]; roots_by_game=[]
    with np.load(HERE/"games.npz") as z: games={k:z[k] for k in z.files}
    for s in range(SHARDS):
        a,b=HERE/f"root_{s}.npz",HERE/f"search_{s}.npz"
        meta=json.loads((HERE/f"search_{s}.json").read_text())
        assert sha(a)==meta["sha256"][a.name] and sha(b)==meta["sha256"][b.name]
        with np.load(a) as root,np.load(b) as child:
            roots_by_game.append(np.bincount(root["game"],minlength=len(games["keys"])))
            take=root["eligible"]>0
            e={k:root[k][take] for k in ("event","game","ply","actual","A","B","pA","pB","rootlead")}
            np.testing.assert_array_equal(e["event"],child["event"])
            e["d0"]=child["rawA"]-child["rawB"];e["d24"]=child["s24A"]-child["s24B"]
            check=child["audit"]
            audits.append(np.column_stack([e["d0"][check],e["d24"][check],child["s128A"]-child["s128B"],
                                          abs(child["rawA"][check]-child["raw128A"]),abs(child["rawB"][check]-child["raw128B"])]))
            e["response"]=np.where(e["actual"]==e["A"],0,np.where(e["actual"]==e["B"],1,2)).astype(np.int8)
            shards.append(e)
    e={k:np.concatenate([row[k] for row in shards]) for k in shards[0]}
    order=np.argsort(e["event"])
    e={k:v[order] for k,v in e.items()};assert (np.diff(e["event"])>0).all()
    return games,e,np.concatenate(audits),np.sum(roots_by_game,axis=0)


def strong(d0,d):
    return (d0*d<0)&(np.minimum(abs(d0),abs(d))>=.5)


def design(e):
    raw=np.clip(e["d0"],-10,10)/5.
    searched=np.clip(e["d24"],-10,10)/5.
    sign=np.sign(e["d24"])
    return np.column_stack([np.ones(len(raw)),np.log(e["pB"]/e["pA"]),e["pA"]+e["pB"]-.6,
                            (e["ply"]-111.)/96.,np.log1p(np.minimum(abs(e["rootlead"]),65.))/4,
                            raw,sign,searched-raw,sign*strong(e["d0"],e["d24"])])


def owner_weights(pid,y):
    ids,inv,n=np.unique(pid,return_inverse=True,return_counts=True)
    grade=np.zeros(len(ids),int);grade[inv]=y
    np.testing.assert_array_equal(grade[inv],y)
    ng=np.bincount(grade,minlength=10);assert (ng>0).all()
    w=1./(10*ng[y]*n[inv]);np.testing.assert_allclose(w.sum(),1.)
    return w


def fit_bernoulli(x,target,grade,w,cols):
    d=x[:,cols] if cols else None
    nc=x.shape[1];nz=len(cols)
    def objective(beta):
        theta=beta[:nc];u=beta[nc:].reshape(10,nz)
        lp=x@theta
        if nz:lp+=np.einsum("ij,ij->i",d,u[grade])
        residual=w*(expit(lp)-target)
        val=w@(np.logaddexp(0,lp)-target*lp)+.002*(theta[1:]@theta[1:])+.001*(u*u).sum()
        gc=x.T@residual;gc[1:]+=.004*theta[1:]
        if not nz:return val,gc
        gd=np.column_stack([np.bincount(grade,weights=residual*d[:,k],minlength=10) for k in range(nz)])+.002*u
        return val,np.r_[gc,gd.ravel()]
    opt=minimize(objective,np.zeros(nc+10*nz),method="L-BFGS-B",jac=True,
                 options={"gtol":1e-8,"ftol":1e-13,"maxiter":480})
    if not opt.success or np.abs(opt.jac).max()>1e-5:raise RuntimeError((opt.message,abs(opt.jac).max()))
    return {"shared":opt.x[:nc],"deviations":opt.x[nc:].reshape(10,nz),"columns":cols,
            "max_gradient":float(np.max(abs(opt.jac))),"iterations":int(opt.nit)}


def logits(model,x,grade):
    shared=x@model["shared"];cols=model["columns"]
    if cols:shared+=np.einsum("ij,ij->i",x[:,cols],model["deviations"][grade])
    return shared


def get_evidence(model,x,target,gid,num_games):
    common=x@model["shared"]; d=x[:,model["columns"]]; table=np.empty((num_games,10))
    for g in GRADES:
        lp=common+d@model["deviations"][g]
        ll=target*lp-np.logaddexp(0,lp)
        table[:,g]=np.bincount(gid,weights=ll,minlength=num_games)
    return table-table.mean(axis=1,keepdims=True)


def groups_of_questions(games,qs):
    ix={k:i for i,k in enumerate(games["keys"])}
    groups={}
    for size in SIZES:
        rows=np.array([i for i,q in enumerate(qs) if q[3]==size])
        ids=np.array([[ix[k] for k in qs[i][4]] for i in rows])
        assert not games["fresh"][ids].any()
        groups[size]=(rows,ids)
    return groups


def aggregate(per_game,groups):
    n=sum(len(q) for q,_ in groups.values())
    out=np.zeros((n,)+per_game.shape[1:],float)
    for q,ix in groups.values():out[q]=per_game[ix].sum(axis=1)
    return out


def calibrated(lp,signal,y,n):
    w=balanced_weights(y,n);idx=np.arange(len(y))
    def loss(a):
        z=lp+a*signal
        return float(w@(logsumexp(z,axis=1)-z[idx,y]))
    opt=minimize_scalar(loss,bounds=(0.,2.),method="bounded",options={"xatol":1e-9})
    assert opt.success
    return min([0.,float(opt.x),2.],key=loss)


def scrambled(x,e,fold):
    raw=e["d0"];s=np.sign(e["d24"]).astype(int)+1
    group=((((fold*3+np.digitize(e["ply"],[84,148]))*3+np.digitize(e["pA"]/e["pB"],[1.4,2.2]))*3+
            np.digitize(raw,[-.5,.5]))*3+s)
    out=x.copy();rng=np.random.default_rng(20261118)
    for g in np.unique(group):
        ix=np.flatnonzero(group==g);out[ix,7:9]=x[rng.permutation(ix),7:9]
    return out


def serialization(model):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}


def main():
    games,e,audit,roots=events()
    qs,_=questions();groups=groups_of_questions(games,qs)
    with np.load(RV/"m183_dump.npz") as z:
        fold,y,n,pid,ref,saved,raw=[z[k] for k in ("fold","y","n","pid","ct106_rule","SQ4","anat_plain")]
        p=z["anat_post"].astype(float)
    p/=p.sum(axis=1,keepdims=True)
    np.testing.assert_array_equal(decide(p),raw);np.testing.assert_array_equal(guard2(raw,ref),saved)
    fraction=e["response"]!=2
    be={k:v[fraction] for k,v in e.items()};gid=be["game"];target=be["response"].astype(float)
    _,owner_i=np.unique(games["pid"],return_inverse=True)
    e_owner=owner_i[gid];ef=games["fold"][gid];ey=games["y"][gid]
    fresh=games["fresh"][gid]
    assert not np.any(fresh & (ef<=2))
    train_q=fold<=2;test_q=fold>=3
    assert not set(pid[train_q])&set(pid[test_q])
    x=design(be); sx=scrambled(x,be,ef+5*fresh)
    counts=np.bincount(gid,minlength=len(games["keys"]))
    shrink=np.sqrt(np.maximum(1,aggregate(counts,groups)))[:,None]
    marginal=np.full_like(p,np.nan); increment=marginal.copy(); fake=np.zeros_like(p);fit_log={}; test_models={}
    for part in (1,2,0):
        tr=((ef==3-part) if part else ef<=2) & ~fresh
        ev=((ef==part) if part else ef>=3)
        qval=(fold==part) if part else test_q
        assert not (set(e_owner[tr])&set(e_owner[ev]))
        train_x=x[tr];train_y=target[tr];gr=ey[tr];w=owner_weights(e_owner[tr],gr)
        models={k:fit_bernoulli(train_x,train_y,gr,w,c) for k,c in (("marginal",MARG),("depth",DEPTH))}
        if part==0:
            models["common"]=fit_bernoulli(train_x,train_y,gr,w,[]);test_models=models
        fit_log[str(part)]={k:serialization(m) for k,m in models.items()}
        qe={}
        for name in ("marginal","depth"):
            qq=get_evidence(models[name],x[ev],target[ev],gid[ev],len(games["keys"]))
            qe[name]=aggregate(qq,groups)/shrink
        marginal[qval]=qe["marginal"][qval];increment[qval]=(qe["depth"]-qe["marginal"])[qval]
        if not part:
            gm=get_evidence(models["marginal"],sx[ev],target[ev],gid[ev],len(games["keys"]))
            gd=get_evidence(models["depth"],sx[ev],target[ev],gid[ev],len(games["keys"]))
            fake[qval]=(aggregate(gd-gm,groups)/shrink)[qval]
    assert np.isfinite(marginal).all() and np.isfinite(increment).all()
    lp=np.log(p)
    alpha=calibrated(lp[train_q],marginal[train_q],y[train_q],n[train_q])
    base=lp+alpha*marginal
    beta=calibrated(base[train_q],increment[train_q],y[train_q],n[train_q])
    def softmax(v):return np.exp(v-logsumexp(v,axis=1)[:,None])
    probs={"SQ4":p,"marginal":softmax(base),"depth":softmax(base+beta*increment),"shuffle":softmax(base+beta*fake),
           "fixed_depth":softmax(base+increment)}
    pred={k:guard2(decide(v),ref) for k,v in probs.items()};sc={k:score(v,y) for k,v in pred.items()}
    nll={k:-np.log(v[np.arange(len(y)),y]) for k,v in probs.items()}
    def metric(values,mask=test_q):return balanced_mean_over_n(values[mask],y[mask],n[mask])
    diffs=np.column_stack([(sc[a]-sc[b])[test_q] for a,b in PAIR_METRICS])
    ci=cluster_intervals(diffs,y[test_q],n[test_q],pid[test_q]); comparisons={}
    for (a,b),ic in zip(PAIR_METRICS,ci):
        d=sc[a]-sc[b]
        comparisons[a+"-"+b]={"diff":metric(d),"ci95":ic.tolist(),"folds":[metric(d,fold==f) for f in (3,4)],
             "by_n":{str(s):balanced_mean_over_n(d[test_q],y[test_q],n[test_q],sizes=[s]) for s in SIZES},
             "changed":int((pred[a][test_q]!=pred[b][test_q]).sum()),"improved":int((d[test_q]>0).sum()),"worsened":int((d[test_q]<0).sum()),
             "nll_diff":metric(nll[a]-nll[b])}
    bounds={};top=np.sort(np.argsort(p,axis=1)[:,-2:],axis=1)
    for lo,hi in ((4,5),(6,7)):
        mask=test_q&(top[:,0]==lo)&(top[:,1]==hi)&np.isin(y,[lo,hi])
        bounds[LEVELS[lo]+"/"+LEVELS[hi]]={"questions":int(mask.sum()),"methods":{}}
        for name,post in probs.items():
            odds=np.log(post[mask,hi]/post[mask,lo]);t=y[mask]==hi
            bounds[LEVELS[lo]+"/"+LEVELS[hi]]["methods"][name]={"auc":float(auc(t,odds)),
                    "cond_nll":float((np.logaddexp(0,odds)-t*odds).mean())}
    behavior={}
    for tag,mask in (("eval",(ef>=3)&~fresh),("new_games",fresh)):
        w=owner_weights(e_owner[mask],ey[mask])
        behavior[tag]={}
        for name,m in test_models.items():
            l=logits(m,x[mask],ey[mask])
            behavior[tag][name]=float(w@(np.logaddexp(0,l)-target[mask]*l))
    v=comparisons["depth-SQ4"];c=comparisons["depth-marginal"]
    passed=v["diff"]>=.001 and min(v["folds"])>0 and v["ci95"][0]>0 and c["ci95"][0]>0
    a0,a24,a128,raw_a,raw_b=audit.T;rev24=strong(a0,a24); rev128=strong(a0,a128)
    numeric={"audits":len(audit),"sign_agreement":float(np.mean(np.sign(a24)==np.sign(a128))),
          "spearman24_128":float(spearmanr(a24,a128).statistic),"median_abs_point_difference":float(np.median(abs(a24-a128))),
          "strong_revisions24":int(rev24.sum()),"same_winner128_on_revisions24":float(np.mean(np.sign(a24[rev24])==np.sign(a128[rev24]))),
          "still_strong_at128":float(rev128[rev24].mean()),"same_child_raw_repeat_p95":float(np.quantile(np.r_[raw_a,raw_b],.95))}
    dest=HERE/"predictions.npz"
    np.savez_compressed(dest,y=y,n=n,pid=pid,fold=fold,inc=increment,original=marginal,**{"d_"+k:v for k,v in pred.items()})
    report={"pass":bool(passed),"protocol_sha256":sha(HERE/"protocol.json"),"sampled_positions":int(roots.sum()),"eligible":len(e["game"]),
       "responses_A_or_B":len(gid),"eval_questions":int(test_q.sum()),"eval_players":len(set(pid[test_q])),
       "temperatures":[alpha,beta],"pair_results":comparisons,"metrics":{k:{"score":metric(sc[k]),"nll":metric(nll[k])} for k in sc},
       "target_boundaries":bounds,"choice_nll_player_grade_balanced":behavior,"depth_audit":numeric,"fits":fit_log,
       "sha256":{p.name:sha(p) for p in (Path(__file__),dest,HERE/"corpus.json",RV/"m183_dump.npz",*(HERE/f"search_{s}.npz" for s in range(SHARDS)))}}
    save_json(HERE/"rank_results.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("fits","sha256","target_boundaries")},indent=2),flush=True)


if __name__=="__main__":main()
