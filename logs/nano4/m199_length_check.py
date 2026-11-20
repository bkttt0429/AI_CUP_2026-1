#!/usr/bin/env python
"""M199: fixed-weight game-length likelihood, label-free open-set check. Report only."""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path.cwd();sys.path[:0]=[str(ROOT/"src"),str(ROOT/"logs/nano4")]
import m198_tail_style as pilot
from aicup.sgf import parse_moves


def length(sgf):
    return len(parse_moves(sgf))


def margin(F):
    top=np.partition(F,-5,axis=1)[:,-5:];best=top.max(1)
    return best-(top.sum(1)-best)/4


def main():
    start=time.time();cache=ROOT/"results/m84_cache";q=np.load(cache/"queries.npz")
    players=q["players"].astype(str);Q,H,T=int(q["Q"]),int(q["H"]),int(q["T"])
    ids=np.arange(2*Q+H,2*Q+H+2*T);assert (q["net"][ids]==4).all()
    ng=pd.read_csv(ROOT/"splits/player_folds.csv",dtype={"player_id":str}).set_index("player_id").n_games.reindex(players).to_numpy(float)
    rows=pilot.m81.load_rows();held=set(json.loads((ROOT/"results/m82/held_keys.json").read_text()))
    with mp.get_context("fork").Pool(10) as pool:
        lengths=np.fromiter(pool.imap(length,rows.sgf_content,chunksize=512),dtype=np.int32,count=len(rows))
    pidx={p:i for i,p in enumerate(players)};owner=rows.player_id.map(pidx).to_numpy(int);use=~rows.game_key.isin(held).to_numpy()
    bins=np.searchsorted(pilot.BOUNDS,lengths,side="right");counts=np.ones((len(players),8),np.float64)
    np.add.at(counts,(owner[use],bins[use]),1)
    probs=counts/counts.sum(1)[:,None]
    hk=q["qkey"][2*Q:2*Q+H].astype(str);hbin=pd.Series(bins,index=rows.game_key).reindex(hk).to_numpy(int)
    test=pilot.m81.load_tests();tl=np.array([length(g[0]) for _,g in test]);tbin=np.searchsorted(pilot.BOUNDS,tl,side="right")[q["qtest"][ids]]
    del rows
    Fh,ev=pilot.baseline(q,ng);Lh=pilot.std(np.log(probs[:,hbin].T).astype(np.float32));Nh=Fh+.5*Lh
    F=np.zeros((len(ids),len(players)),np.float32);seen=np.zeros(len(ids),bool)
    for k in range(4):
        idx=np.load(cache/f"idx_{k}of4.npy");mask=(idx>=ids[0])&(idx<=ids[-1]);loc=idx[mask]-ids[0]
        block=np.load(cache/f"block_{k}of4.npy",mmap_mode="r")
        F[loc]=ev["weights"][0]*block[mask];seen[loc]=True
    assert seen.all()
    for k,name in enumerate(ev["sources"]):
        with np.load(cache/f"var_{name}.npz") as v:
            proto,query=v["protos"],v["qvec"];has=v["has"] if "has" in v.files else None
            for m in range(len(proto)):
                P=torch.as_tensor(proto[m,4].astype(np.float32),device="cuda").T
                for i in range(0,len(ids),512):
                    z=(torch.as_tensor(query[m,ids[i:i+512]].astype(np.float32),device="cuda")@P).cpu().numpy()
                    F[i:i+512]+=ev["weights"][k+1]/len(proto)*pilot.std(z,has[m,4] if has is not None else None)
        print(f"source {name} done ({time.time()-start:.0f}s)",flush=True)
    F+=ev["beta"]*np.log(ng)[None].astype(np.float32)
    N=F+.5*pilot.std(np.log(probs[:,tbin].T).astype(np.float32))
    truth=q["held_truth"].astype(int);w=ng[truth]/ng[truth].sum();result={"experiment":"M199","fixed_weight":.5,"report_only":True,"readouts":{}}
    for name,Ftest,Fheld in [("SQ4",F,Fh),("SQ4_plus_length",N,Nh)]:
        mt=margin(Ftest);mh=margin(Fheld);threshold=float(np.quantile(mt[T:],.95));correct=Fheld.argmax(1)==truth
        levels=np.r_[np.linspace(.5,.95,10),.97,.98,.99,.995,.999]
        thresholds=np.quantile(mt[T:],levels)
        excess=[float((mt[:T]>=x).mean()-(mt[T:]>=x).mean()) for x in thresholds]
        result["readouts"][name]=dict(held_top1=float(w@correct),held_recognised=float(w@(correct&(mh>=threshold))),
            held_wrong_confident=float(w@(~correct&(mh>=threshold))),null_q95=threshold,
            test_recognised_net_q95=float((mt[:T]>=threshold).mean()-(mt[T:]>=threshold).mean()),
            max_excess=float(max(excess)),excess_by_quantile=dict(zip(map(str,levels),excess)))
    b,n=result["readouts"].values();result["delta"]={k:n[k]-b[k] for k in ["held_top1","held_recognised","held_wrong_confident","test_recognised_net_q95","max_excess"]}
    d=result["delta"];result["screen_pass"]=bool(d["test_recognised_net_q95"]>=.005 and d["held_recognised"]>=.005 and d["held_wrong_confident"]<=.002)
    result["decision"]="register_crossfit_downstream_test" if result["screen_pass"] else "close_length_as_added_identifier_signal"
    result["seconds"]=time.time()-start
    (ROOT/"results/r_variants/m199_length_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
