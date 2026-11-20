"""Test selection of a known two-option menu as a function of its ex-ante relative value."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

from extract import HERE,PREV,R1,questions,save_json,sha
from fit_offers import covariance,fit_model,matrix,weights
from fit import PAIRS
from evaluate import balanced_mean_over_n,cluster_intervals,decide,guard2,score


def clustered_rate_ci(rank,owner,accept,delta):
    # A player's repeated choices travel together; group rates use raw event denominators.
    groups,inv=np.unique(owner,return_inverse=True); grade=np.zeros(len(groups),int); grade[inv]=rank
    s=np.zeros((len(groups),4),int)
    for j,m in enumerate((delta< -1,delta>1)):
        np.add.at(s[:,2*j],inv[m],accept[m]);np.add.at(s[:,2*j+1],inv[m],1)
    rng=np.random.default_rng(20261118); sums=np.zeros((5000,4))
    for k in range(10):
        block=s[grade==k]
        mult=rng.multinomial(len(block),np.full(len(block),1/len(block)),size=5000)
        sums+=mult@block
    diff=sums[:,2]/sums[:,3]-sums[:,0]/sums[:,1]
    return np.quantile(diff,[.025,.975]).tolist()


def main():
    with np.load(HERE/"offer_events.npz") as z:e={k:z[k] for k in z.files}
    total=len(e["game"]);delta=np.full(total,np.nan); repeat=[]
    for shard in range(4):
        file=HERE/f"offer_values_{shard}.npz"
        assert sha(file)==json.loads((HERE/f"offer_values_{shard}.json").read_text())["sha256"][file.name]
        with np.load(file) as z:
            ids=z["ix"];assert not np.isfinite(delta[ids]).any()
            delta[ids]=z["delta"];repeat.extend(z["repeat_abs"].tolist())
    unique=e["number"]==1
    np.testing.assert_array_equal(unique,np.isfinite(delta))
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z: keys=z["keys"]
    pos={k:i for i,k in enumerate(keys)}; indices=[np.array([pos[k] for k in q[4]]) for q in qs]
    with np.load(R1/"results/r_variants/m183_dump.npz") as z:
        y,n,pid,fold,ref,saved=[z[k] for k in ("y","n","pid","fold","ct106_rule","SQ4")]
        p=z["anat_post"].astype(float)
    p/=p.sum(axis=1,keepdims=True)
    _,oi=np.unique(pid,return_inverse=True)
    gf=np.zeros(len(keys),int);gy=np.full(len(keys),-1);go=np.full(len(keys),-1)
    for i,v in enumerate(indices):gf[v]=fold[i];gy[v]=y[i];go[v]=oi[i]
    gid=e["game"]; val=gf[gid]>=3;binary=unique&(e["choice"]!=2)
    test=val[binary]; g=gy[gid[binary]];o=go[gid[binary]];target=(e["choice"][binary]==1).astype(float)
    sx=np.clip(delta[binary],-10,10)/5.
    context=np.column_stack([covariance(e,binary),sx])
    train_w=weights(o[~test],g[~test]);valid_w=weights(o[test],g[test])
    cnt=np.bincount(gid[binary],minlength=len(keys))
    k=np.sqrt(np.maximum(1,np.array([cnt[v].sum() for v in indices])))[:,None]
    action_nll={};params={};pred={"SQ4":saved}
    for model in ("common","intercept","slope"):
        x=matrix(context,g,sx,model); beta=fit_model(x[~test],target[~test],train_w)
        params[model]=beta.tolist();l=x[test]@beta
        action_nll[model]=float(valid_w@(np.logaddexp(0,l)-target[test]*l))
        gl=np.zeros((len(keys),10))
        for grade in range(10):
            l=matrix(context,np.full_like(g,grade),sx,model)@beta
            np.add.at(gl[:,grade],gid[binary],target*l-np.logaddexp(0,l))
        evid=np.stack([gl[v].sum(axis=0) for v in indices])/k
        pp=p.copy()
        for lo,hi in PAIRS:
            v=np.log(p[:,hi]/p[:,lo])+(evid[:,hi]-evid[:,lo]);mass=p[:,hi]+p[:,lo]
            pp[:,lo]=mass*expit(-v);pp[:,hi]=mass*expit(v)
        pred[model]=guard2(decide(pp),ref)
    np.testing.assert_array_equal(pred["common"],saved)
    stats={}
    for label,mask in (("below_minus_1",delta< -1),("within_1",abs(delta)<=1),("above_1",delta>1)):
        m=val&unique&mask; a=np.bincount(e["choice"][m],minlength=3);binary_n=int(a[0]+a[1])
        stats[label]={"events":int(m.sum()),"direct":int(a[0]),"rescue":int(a[1]),"other":int(a[2]),
                      "rescue_among_all":float(a[1]/a.sum()),"rescue_among_two":float(a[1]/binary_n)}
    sub=binary&val
    ci=clustered_rate_ci(gy[gid[sub]],go[gid[sub]],(e["choice"][sub]==1).astype(int),delta[sub])
    t=fold>=3; sc={k:score(v,y) for k,v in pred.items()}
    comparisons=[("slope","SQ4"),("intercept","SQ4"),("slope","intercept")]
    d=np.column_stack([(sc[a]-sc[b])[t] for a,b in comparisons])
    pci=cluster_intervals(d,y[t],n[t],pid[t]); paired={}
    for (a,b),c in zip(comparisons,pci):
        diff=sc[a]-sc[b]
        paired[a+"-"+b]={"diff":balanced_mean_over_n(diff[t],y[t],n[t]),"ci95":c.tolist(),
              "folds":[balanced_mean_over_n(diff[fold==f],y[fold==f],n[fold==f]) for f in (3,4)],
              "gained":int((diff[t]>0).sum()),"lost":int((diff[t]<0).sum())}
    dest=HERE/"value_response_predictions.npz"
    np.savez_compressed(dest,n=n,y=y,pid=pid,fold=fold,**{"d_"+k:v for k,v in pred.items()})
    out={"bins":stats,"two_way_rate_high_minus_low":stats["above_1"]["rescue_among_two"]-stats["below_minus_1"]["rescue_among_two"],
         "rate_diff_ci95":ci,"action_nll":action_nll,"rank_paired":paired,"beta":params,
         "duplicate_comparisons":len(repeat),"duplicate_abs_diff_p95":float(np.quantile(repeat,.95)),
         "duplicate_abs_diff_max":float(max(repeat)),
         "sha256":{f.name:sha(f) for f in (dest,Path(__file__),HERE/"offered_value_protocol.json",*(HERE/f"offer_values_{s}.npz" for s in range(4)))}}
    save_json(HERE/"value_response.json",out)
    print(json.dumps({k:v for k,v in out.items() if k not in ("beta","sha256")},indent=2))


if __name__=="__main__": main()
