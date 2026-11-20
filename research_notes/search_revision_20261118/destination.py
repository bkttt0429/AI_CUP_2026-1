"""Separate an exact policy hit from a nearby alternative and a move elsewhere on the board."""
import json
import numpy as np
from scipy.special import softmax

from response import (HERE, RV, SHARDS, SIZES, aggregate, balanced_mean_over_n, calibrated, cluster_intervals,
                      decide, fit_bernoulli, get_evidence, groups_of_questions, guard2, logits, owner_weights,
                      questions, save_json, score, sha)
from mechanism import matched_proportions


def distance(a,b):
    return abs(a%19-b%19)+abs(a//19-b//19)


def data():
    fields="event game A B pA pB ply rootlead actual last".split()
    chunks=[]
    with np.load(HERE/"games.npz") as z:games={k:z[k] for k in z.files}
    assert json.loads((HERE/"corpus.json").read_text())["sha256"]["games.npz"]==sha(HERE/"games.npz")
    for s in range(SHARDS):
        m=json.loads((HERE/f"root_{s}.json").read_text());path=HERE/f"root_{s}.npz"
        assert m["sha256"][path.name]==sha(path)
        with np.load(path) as z:
            use=(z["eligible"]>0)&(distance(z["A"],z["B"])<=4)
            chunks.append({k:z[k][use] for k in fields})
    col={k:np.concatenate([z[k] for z in chunks]) for k in fields};order=np.argsort(col["event"])
    e={k:v[order] for k,v in col.items()}
    assert np.all(np.diff(e["event"])>0)
    return games,e


def features(e):
    a,b=e["A"],e["B"];last=e["last"];played=e["actual"]
    gap=distance(a,b);prev=np.minimum(distance(last,a),distance(last,b));prev=np.where(last==361,37,prev)
    actual_distance=np.minimum(distance(played,a),distance(played,b))
    hit=(played==a)|(played==b);miss=~hit&(played!=361);near=(actual_distance<=3)&miss
    assert (gap>=1).all() and (gap<=4).all() and not (near&hit).any()
    mass=e["pA"]+e["pB"]
    x=np.column_stack([np.ones(len(a)),np.log(e["pB"]/e["pA"]),mass-.6,(e["ply"]-111.)/96.,
        np.log1p(np.minimum(abs(e["rootlead"]),65.))/4,(gap-2.5)/2.,np.minimum(prev,15)/10.,last==361])
    phase_mass=np.digitize(e["ply"],[84,148])*3+np.digitize(mass,[.5,.7])
    group=(phase_mass*2+(gap>=3))*3+np.digitize(prev,[3.5,8.5])
    return x,hit,miss,near,group


def shuffle_choices(target,group):
    rng=np.random.default_rng(20261124)
    out=target.copy()
    for k in np.unique(group):
        idx=np.flatnonzero(group==k);out[idx]=target[rng.permutation(idx)]
    return out


def repeatability(per_game,ix,grade):
    """Correlate two disjoint ten-game estimates of both decision stages."""
    rng=np.random.default_rng(20261121)
    all_values=per_game[ix]
    rs={k:{"all":[],"within_rank":[]} for k in ("exact","near_given_miss")}
    for _ in range(20):
        order=rng.random(ix.shape).argsort(axis=1)
        halves=[np.take_along_axis(all_values,part[:,:,None],axis=1).sum(axis=1) for part in (order[:,:10],order[:,10:])]
        for k,j in (("exact",0),("near_given_miss",2)):
            a,b=[(m[:,j+1]+.5)/(m[:,j]+1) for m in halves]
            rs[k]["all"].append(float(np.corrcoef(a,b)[0,1]))
            centered=[u-(np.bincount(grade,weights=u)/np.bincount(grade))[grade] for u in (a,b)]
            rs[k]["within_rank"].append(float(np.corrcoef(*centered)[0,1]))
    return {k:{j:float(np.median(v)) for j,v in s.items()} for k,s in rs.items()}


def main():
    games,e=data()
    gid=e["game"];x,hit,miss,near,cell=features(e)
    _,owner=np.unique(games["pid"],return_inverse=True)
    yr=games["y"][gid];ef=games["fold"][gid];fresh=games["fresh"][gid];own=owner[gid]
    qs,_=questions();groups=groups_of_questions(games,qs)
    with np.load(RV/"m183_dump.npz") as z:
        p=z["anat_post"].astype(float);y,n,pid,fold,ref,sq=[z[k] for k in ("y","n","pid","fold","ct106_rule","SQ4")]
    p/=p.sum(axis=1,keepdims=True)
    np.testing.assert_array_equal(guard2(decide(p),ref),sq)
    per_game=np.column_stack([np.bincount(gid[m],minlength=len(games["keys"])) for m in
                              (np.ones(len(gid),bool),hit,miss,near)])
    cntq=aggregate(per_game,groups)
    qsdev=fold<=2;qstest=fold>=3
    ev_exact,ev_where,ev_shuffled=(np.zeros_like(p) for _ in range(3))
    sample_stats={};fits={}
    names=[("exact",np.ones(len(gid),bool),hit,0,ev_exact),("near_given_miss",miss,near,2,ev_where)]
    for name,take,target,ci,q_evidence in names:
        selected=np.flatnonzero(take);ft=ef[selected];new=fresh[selected];g=gid[selected];owners=own[selected];rank=yr[selected]
        z=x[selected];t=target[selected].astype(float)
        t_fake=shuffle_choices(t,54*(ft+5*new)+cell[selected]) if name=="near_given_miss" else None
        denom=np.sqrt(np.maximum(cntq[:,ci],1))[:,None];fits[name]={}
        for part in (1,2,0):
            train=((ft==3-part) if part else ft<=2)&~new
            val=(ft==part) if part else ft>=3
            qval=(fold==part) if part else qstest
            assert not set(owners[train])&set(owners[val])
            w=owner_weights(owners[train],rank[train])
            model=fit_bernoulli(z[train],t[train],rank[train],w,[0,2,3,5])
            batch_rows=z[val];ng=len(games["keys"])
            out=get_evidence(model,batch_rows,t[val],g[val],ng)
            q_evidence[qval]=(aggregate(out,groups)/denom)[qval]
            fits[name][str(part)]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}
            if not part:
                common=fit_bernoulli(z[train],t[train],rank[train],w,[])
                for tag,ev in (("old",(ft>=3)&~new),("new",new)):
                    score_w=owner_weights(owners[ev],rank[ev])
                    for method,m in (("common",common),("grade",model)):
                        lp=logits(m,z[ev],rank[ev])
                        sample_stats.setdefault(tag,{}).setdefault(name,{})[method]=float(score_w@(np.logaddexp(0,lp)-t[ev]*lp))
                if t_fake is not None:
                    out=get_evidence(model,batch_rows,t_fake[val],g[val],ng)
                    ev_shuffled[qval]=(aggregate(out,groups)/denom)[qval]
    lp=np.log(p)
    al=calibrated(lp[qsdev],ev_exact[qsdev],y[qsdev],n[qsdev]);base=lp+al*ev_exact
    be=calibrated(base[qsdev],ev_where[qsdev],y[qsdev],n[qsdev])
    probs={"SQ4":p, "exact":softmax(base,axis=1),"spatial":softmax(base+be*ev_where,axis=1),
           "shuffle":softmax(base+be*ev_shuffled,axis=1),"fixed":softmax(base+ev_where,axis=1)}
    pred={k:guard2(decide(prob),ref) for k,prob in probs.items()}
    scores={k:score(v,y) for k,v in pred.items()};ll={k:-np.log(v[np.arange(len(y)),y]) for k,v in probs.items()}
    pairs=[("spatial","SQ4"),("spatial","exact"),("exact","SQ4"),("spatial","shuffle"),("fixed","SQ4")]
    ci=cluster_intervals(np.column_stack([(scores[a]-scores[b])[qstest] for a,b in pairs]),y[qstest],n[qstest],pid[qstest])
    def metric(s,m=qstest):
        return balanced_mean_over_n(s[m],y[m],n[m])
    summary={}
    for (a,b),ic in zip(pairs,ci):
        d=scores[a]-scores[b]
        summary[a+"-"+b]={"difference":metric(d),"ci95":ic.tolist(),"folds":[metric(d,fold==f) for f in (3,4)],
              "by_size":{str(size):balanced_mean_over_n(d[qstest],y[qstest],n[qstest],sizes=[size]) for size in SIZES},
              "nll_difference":metric(ll[a]-ll[b]),"changed":int((pred[a][qstest]!=pred[b][qstest]).sum()),
              "improved":int((d[qstest]>0).sum()),"worsened":int((d[qstest]<0).sum())}
    by_sample={}
    for tag,m,seed in (("question_games",(games["fold"]>=3)&~games["fresh"],20261126),("new_games",games["fresh"],20261127)):
        sub=m[gid]
        grade,contrasts=matched_proportions(games,gid,sub&miss,near,cell,m,54,np.random.default_rng(seed))
        by_sample[tag]={"menus":int(sub.sum()),"exact":int((sub&hit).sum()),"near":int((sub&near).sum()),
                "far":int((sub&miss&~near).sum()),"pass":int((sub&(e["actual"]==361)).sum()),
                "observed_by_grade":grade,"matched_near_given_miss":contrasts}
    by_size={}
    for size in SIZES:
        m=qstest&(n==size)
        by_size[str(size)]={"questions":int(m.sum()),"median_menus_exact_miss_near":np.median(cntq[m],axis=0).tolist(),
                    "any_nonpass_miss":float((cntq[m,2]>0).mean())}
    row,ix=groups[20];m=qstest[row]
    assert len(set(pid[row][m]))==m.sum()
    full=summary["spatial-SQ4"]
    passed=full["difference"]>=.001 and min(full["folds"])>0 and full["ci95"][0]>0 and summary["spatial-exact"]["ci95"][0]>0
    dest=HERE/"destination_predictions.npz"
    np.savez_compressed(dest,y=y,fold=fold,n=n,pid=pid,exact_log_evidence=ev_exact,where_log_evidence=ev_where,
                        **{"d_"+k:v for k,v in pred.items()})
    result={"temperatures":[al,be],"pass":bool(passed),"competition":summary,"scores":{k:metric(v) for k,v in scores.items()},
            "behavior_nll":sample_stats,"samples":by_sample,"five_ten_twenty":by_size,
            "split_half_correlations":repeatability(per_game,ix[m],y[row][m]),"fits":fits,
            "sha256":{p.name:sha(p) for p in (HERE/"destination_protocol.json",HERE/"destination.py",HERE/"mechanism.py",dest,
                *(HERE/f"root_{s}.npz" for s in range(SHARDS)))}}
    save_json(HERE/"destination.json",result)
    print(json.dumps({"pass":passed,"temperatures":[al,be],"scores":result["scores"],"pairs":summary,
                "contrasts":{k:v["matched_near_given_miss"] for k,v in by_sample.items()}},indent=2),flush=True)


if __name__=="__main__":
    main()
