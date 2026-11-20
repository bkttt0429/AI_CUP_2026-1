"""Checkpoint the two-child search on disjoint contiguous game blocks using the frozen requests."""
import json
import os
from pathlib import Path
import sys

import numpy as np

from search import BIN, CFG, HERE, MODEL, SHARDS, child_tasks, is_audit, run_engine, save_json, sha, write_npz

PARTS = 2
FOLDER = HERE / "parts"


def indices(shard):
    path=HERE/f"root_{shard}.npz"
    assert json.loads((HERE/f"root_{shard}.json").read_text())["sha256"][path.name]==sha(path)
    with np.load(path) as z:
        valid=z["eligible"]>0
        return z["event"][valid], z["A"][valid], z["B"][valid]


def worker(rank):
    shard,part=rank%SHARDS,rank//SHARDS
    FOLDER.mkdir(exist_ok=True)
    labels,a,b=indices(shard)
    # An event is never split between workers; both candidates and both budgets stay together.
    pieces=np.array_split(np.arange(len(labels)),PARTS)
    keep=pieces[part];ids=labels[keep]
    slots={int(e):i for i,e in enumerate(ids)}
    menu={int(e):(int(x),int(y)) for e,x,y in zip(ids,a[keep],b[keep])}
    target=FOLDER/f"children_{shard}_{part}.npy"
    marker=target.with_suffix(".json")
    if marker.exists():
        assert sha(target)==json.loads(marker.read_text())["sha256"][target.name]
        return
    shape=(len(ids),2,2,2)
    if target.exists():
        values=np.load(target,mmap_mode="r+");assert values.shape==shape and values.dtype==np.float64
    else:
        values=np.lib.format.open_memmap(target,mode="w+",dtype=np.float64,shape=shape)
        values[:]=np.nan; values.flush()
    expected=np.array([is_audit(e) for e in ids])
    seen=np.isfinite(values).all(axis=3)
    assert not np.any(seen[~expected,1])
    tasks=child_tasks(shard,menu)
    count=0
    def missing():
        for key,query,meta in tasks:
            e,depth,branch,_,_=meta
            if not seen[slots[e],int(depth==128),branch]:
                yield key,query,meta
    def receive(ans,meta):
        nonlocal count
        e,depth,branch,t,who=meta; info=ans["rootInfo"]
        assert ans["turnNumber"]==t and info["currentPlayer"]==who and int(info["visits"])>=depth
        i=slots[e];di=int(depth==128)
        assert not seen[i,di,branch]
        pair=[float(info["rawLead"]),float(info["scoreLead"])]
        assert np.isfinite(pair).all()
        values[i,di,branch]=pair;seen[i,di,branch]=True;count+=1
        if count%4000==0:
            values.flush();print("persisted",shard,part,count,flush=True)
    remaining=int(2*(len(ids)+expected.sum())-seen.sum())
    print("start",shard,part,remaining,"queries",flush=True)
    if remaining:run_engine(missing(),receive,"paired")
    assert remaining==count and seen[:,0].all() and seen[expected,1].all()
    assert np.isnan(values[~expected,1]).all()
    values.flush()
    save_json(marker,{"shard":shard,"part":part,"events":len(ids),"first":int(ids[0]),"last":int(ids[-1]),
        "sha256":{p.name:sha(p) for p in (target,Path(__file__),HERE/"search.py",HERE/f"root_{shard}.npz",HERE/"protocol.json")}})


def combine():
    for shard in range(SHARDS):
        src=HERE/f"root_{shard}.npz";dest=HERE/f"search_{shard}.npz"
        marker=dest.with_suffix(".json")
        if marker.exists():
            assert sha(dest)==json.loads(marker.read_text())["sha256"][dest.name]
            continue
        ids,_,_=indices(shard)
        pieces=[];sources=[]
        for part,ix in enumerate(np.array_split(np.arange(len(ids)),PARTS)):
            p=FOLDER/f"children_{shard}_{part}.npy";m=p.with_suffix(".json");meta=json.loads(m.read_text())
            assert meta["first"]==ids[ix[0]] and meta["last"]==ids[ix[-1]] and meta["events"]==len(ix)
            assert meta["sha256"][p.name]==sha(p) and meta["sha256"]["search.py"]==sha(HERE/"search.py")
            pieces.append(np.load(p));sources.append(m)
        v=np.concatenate(pieces);audit=np.array([is_audit(e) for e in ids])
        assert np.isfinite(v[:,0]).all() and np.isfinite(v[audit,1]).all() and np.isnan(v[~audit,1]).all()
        write_npz(dest,event=ids,rawA=v[:,0,0,0],rawB=v[:,0,1,0],s24A=v[:,0,0,1],s24B=v[:,0,1,1],
            audit=audit,raw128A=v[audit,1,0,0],raw128B=v[audit,1,1,0],s128A=v[audit,1,0,1],s128B=v[audit,1,1,1])
        sources.extend([src,dest,MODEL,CFG,BIN.parent/"usr/bin/katago",Path(__file__),HERE/"search.py",HERE/"protocol.json"])
        save_json(marker,{"menus":len(ids),"depth_audits":int(audit.sum()),"sha256":{p.name:sha(p) for p in sources}})
        print("joined",shard,len(ids),int(audit.sum()),flush=True)


if __name__=="__main__":
    if sys.argv[1]=="combine":combine()
    else:worker(int(os.environ["SLURM_PROCID"]))
