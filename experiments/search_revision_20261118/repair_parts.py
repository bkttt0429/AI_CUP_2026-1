"""Verify persisted checkpoint bytes and fill only absent engine responses in a separate immutable file."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from child_parts import FOLDER,PARTS,indices
from search import BIN,CFG,HERE,MODEL,SHARDS,child_tasks,is_audit,run_engine,save_json,sha,write_npz


def read_part(s,j,ids):
    original=FOLDER/f"children_{s}_{j}.npy";fixed=FOLDER/f"fixed_children_{s}_{j}.npy"
    p=fixed if fixed.with_suffix(".json").exists() else original
    meta=json.loads(p.with_suffix(".json").read_text())
    assert meta["sha256"][p.name]==sha(p) and meta["events"]==len(ids)
    a=np.load(p)
    assert a.shape==(len(ids),2,2,2)
    audit=np.fromiter((is_audit(e) for e in ids),bool,count=len(ids))
    expected=np.ones(a.shape[:-1],bool);expected[~audit,1]=False
    assert np.isnan(a[~expected]).all()
    needs=expected & ~np.isfinite(a).all(-1)
    return p,a,audit,needs


def repair(s,j):
    full,aa,bb=indices(s);ix=np.array_split(np.arange(len(full)),PARTS)[j]; ids=full[ix]
    origin,a,audit,needs=read_part(s,j,ids)
    if not needs.any():return
    assert not origin.name.startswith("fixed")
    take=np.flatnonzero(needs.any(axis=(1,2)))
    slots={int(ids[k]):int(k) for k in take}
    menu={int(ids[k]):(int(aa[ix[k]]),int(bb[ix[k]])) for k in take}
    missing_count=int(needs.sum())
    def requests():
        for key,q,meta in child_tasks(s,menu):
            e,depth,b,t,who=meta
            if needs[slots[e],int(depth==128),b]:yield key,q,meta
    def consume(ans,meta):
        e,depth,b,t,who=meta;i=slots[e];d=int(depth==128);v=ans["rootInfo"]
        assert needs[i,d,b] and v["currentPlayer"]==who and ans["turnNumber"]==t and int(v["visits"])>=depth
        a[i,d,b]=[v["rawLead"],v["scoreLead"]];assert np.isfinite(a[i,d,b]).all()
        needs[i,d,b]=False
    print("recover",s,j,missing_count,"of",2*(len(ids)+audit.sum()),flush=True)
    run_engine(requests(),consume,"recovered")
    assert not needs.any() and np.isfinite(a[:,0]).all() and np.isfinite(a[audit,1]).all()
    dest=FOLDER/f"fixed_children_{s}_{j}.npy";tmp=dest.with_suffix(".part")
    with tmp.open("wb") as handle:
        np.save(handle,a);handle.flush();os.fsync(handle.fileno())
    assert np.array_equal(np.load(tmp),a,equal_nan=True)
    tmp.replace(dest);assert np.array_equal(np.load(dest),a,equal_nan=True)
    save_json(dest.with_suffix(".json"),{"events":len(ids),"filled_missing_queries":missing_count,"first":int(ids[0]),"last":int(ids[-1]),
        "sha256":{p.name:sha(p) for p in (dest,origin,origin.with_suffix(".json"),Path(__file__),HERE/"search.py",HERE/"protocol.json")}})


def combine():
    for s in range(SHARDS):
        root=HERE/f"root_{s}.npz";dest=HERE/f"search_{s}.npz";marker=dest.with_suffix(".json")
        assert not marker.exists() or sha(dest)==json.loads(marker.read_text())["sha256"][dest.name]
        if marker.exists():continue
        ids,_,_=indices(s);sources=[];parts=[];audits=[]
        for j,ix in enumerate(np.array_split(np.arange(len(ids)),PARTS)):
            p,a,audit,needs=read_part(s,j,ids[ix])
            assert not needs.any() and np.isfinite(a[:,0]).all() and np.isfinite(a[audit,1]).all()
            parts.append(a);audits.append(audit);sources.append(p.with_suffix(".json"))
        v=np.concatenate(parts);audit=np.concatenate(audits)
        write_npz(dest,event=ids,rawA=v[:,0,0,0],rawB=v[:,0,1,0],s24A=v[:,0,0,1],s24B=v[:,0,1,1],
            audit=audit,raw128A=v[audit,1,0,0],raw128B=v[audit,1,1,0],s128A=v[audit,1,0,1],s128B=v[audit,1,1,1])
        sources.extend([root,dest,MODEL,CFG,BIN.parent/"usr/bin/katago",Path(__file__),HERE/"child_parts.py",HERE/"search.py",HERE/"protocol.json"])
        save_json(marker,{"menus":len(ids),"depth_audits":int(audit.sum()),"sha256":{p.name:sha(p) for p in sources}})
        print("joined",s,len(ids),int(audit.sum()),flush=True)


if __name__=="__main__":
    if sys.argv[1]=="combine":combine()
    else:
        rank=int(os.environ["SLURM_PROCID"])
        for k in range(rank,SHARDS*PARTS,4):repair(k%SHARDS,k//SHARDS)
