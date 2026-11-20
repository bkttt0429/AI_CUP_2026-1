"""Generate two plausible moves before seeing the reply, then search both children equally."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np

from corpus import HERE, R1, SHARDS, save_json, sha
from aicup.sgf import to_gtp

KD=R1/"tools/katago_v1182"
MODEL=KD/"dl/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
BIN=KD/"bin/squashfs-root/AppRun"
CFG=KD/"analysis_m25.cfg"
COORD=tuple(to_gtp(i%19,i//19) for i in range(361))+("pass",)
ROOT_FIELDS="event game ply actual last A B pA pB rootlead eligible".split()


def packed(a,b):
    return 361 if a=="-" else (ord(b)-97)*19+(ord(a)-97)


def positions(shard):
    with gzip.open(HERE/f"corpus_{shard}.tsv.gz","rt") as src:
        for line in src:
            num,digest,who,handicap,stones,seq=line.rstrip("\n").split("\t")
            gid=int(num); h=int(handicap)
            assert gid%SHARDS==shard
            moves=[[seq[t],COORD[packed(seq[t+1],seq[t+2])]] for t in range(0,len(seq),3)]
            init=[] if stones=="-" else [["B",COORD[packed(stones[i],stones[i+1])]] for i in range(0,len(stones),2)]
            base={"boardXSize":19,"boardYSize":19,"rules":"japanese","initialPlayer":moves[0][0],"komi":6.5 if h==0 else .5,"initialStones":init}
            offsets=hashlib.blake2s(bytes.fromhex(digest)+b"root-schedule-v1",digest_size=16).digest()
            parity=int(who!=moves[0][0])
            for window in range(12):
                t=20+16*window+2*(offsets[window]%8)+parity
                if t>=len(moves):continue
                assert moves[t][0]==who
                here=3*t
                actual=packed(seq[here+1],seq[here+2])
                prev=packed(seq[here-2],seq[here-1])
                yield gid*12+window,gid,t,who,actual,prev,moves,base


def root_tasks(shard):
    for event,game,t,who,actual,last,moves,base in positions(shard):
        yield str(event),dict(base,moves=moves[:t],maxVisits=1,includePolicy=True),(event,game,t,who,actual,last)


def is_audit(event):
    return int.from_bytes(hashlib.blake2s(f"revision-deep-{event}".encode(),digest_size=4).digest(),"big")%32==0


def child_tasks(shard,menus):
    for event,_,t,who,_,_,moves,base in positions(shard):
        menu=menus.get(event)
        if menu is None:continue
        prefix=moves[:t]
        for branch,loc in enumerate(menu):
            assert loc!=361
            child=prefix+[[who,COORD[loc]]]
            for depth in ((24,128) if is_audit(event) else (24,)):
                key=f"{event}.{depth}.{branch}"
                yield key,dict(base,moves=child,maxVisits=depth),(event,depth,branch,t+1,"W" if who=="B" else "B")


def run_engine(tasks,consume,label):
    capacity=threading.Semaphore(256); stop=threading.Event(); pending={};errors=[]
    nv=R1/".venv/lib/python3.12/site-packages/nvidia";env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=":".join(["/work/envstack/apps/gcc/13.2/lib64"]+
         [str(nv/d/"lib") for d in ("cuda_runtime","cublas","cudnn","cuda_nvrtc","nvjitlink")]+[env.get("LD_LIBRARY_PATH","")])
    sent=[0];received=0; start=time.monotonic()
    with TemporaryDirectory(prefix="rev_engine_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as td:
        config=f"logDir={td},nnForcedSymmetry=0,nnRandomize=false,nnCacheSizePowerOfTwo=17,nnMutexPoolSizePowerOfTwo=12,numAnalysisThreads=64,nnMaxBatchSize=96"
        proc=subprocess.Popen([BIN,"analysis","-config",CFG,"-model",MODEL,"-override-config",config],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,text=True,bufsize=1,env=env)
        def feed():
            try:
                for key,query,meta in tasks:
                    while not capacity.acquire(timeout=.5):
                        if stop.is_set():return
                    if stop.is_set():return
                    assert key not in pending
                    pending[key]=meta;sent[0]+=1
                    query["id"]=key
                    proc.stdin.write(json.dumps(query,separators=(",",":"))+"\n")
                proc.stdin.close()
            except BaseException as e:
                if not stop.is_set():
                    errors.append(repr(e));proc.terminate()
        worker=threading.Thread(target=feed,daemon=True);worker.start()
        try:
            for line in proc.stdout:
                ans=json.loads(line)
                if "error" in ans or "warning" in ans:raise RuntimeError(str(ans)[:1000])
                assert not ans.get("isDuringSearch",False)
                meta=pending.pop(ans["id"]);capacity.release();consume(ans,meta)
                received+=1
                if received%30000==0: print(label,received,"positions",round(time.monotonic()-start,1),"s",flush=True)
            worker.join(timeout=10)
            code=proc.wait(timeout=20)
            assert not errors and not pending and not worker.is_alive() and code==0 and sent[0]==received and received>0,(errors,code,received,sent)
        finally:
            stop.set()
            if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
    print(label,received,"total",round(time.monotonic()-start,1),"s",flush=True)


def write_npz(path,**arrays):
    tmp=path.with_suffix(".part")
    with tmp.open("wb") as out:np.savez_compressed(out,**arrays)
    tmp.replace(path)


def first_stage(shard):
    dest=HERE/f"root_{shard}.npz"
    rows=[]
    def consume(ans,meta):
        event,game,t,who,actual,last=meta; root=ans["rootInfo"]
        assert ans["turnNumber"]==t and root["currentPlayer"]==who
        policy=np.array(ans["policy"],dtype=float)
        assert policy.shape==(362,) and policy[actual]>=0 and abs(policy[policy>=0].sum()-1)<.002
        if last!=361:assert policy[last]<0,(last,policy[last])
        a=int(policy.argmax());pA=policy[a];policy[a]=-2
        b=int(policy.argmax());pB=policy[b]
        assert a!=b and pA>=pB>0
        eligible=a!=361 and b!=361 and pB>=.05 and pA<=4*pB and pA+pB>=.30
        rows.append((event,game,t,actual,last,a,b,pA,pB,float(root["rawLead"]),int(eligible)))
    run_engine(root_tasks(shard),consume,"root")
    data=np.array(sorted(rows),float)
    expected=np.array(sorted(e for e,*_ in positions(shard)))
    np.testing.assert_array_equal(data[:,0],expected)
    result={name:data[:,i].astype(np.float64 if name in ("pA","pB","rootlead") else (np.int32 if name in ("event","game") else np.int16)) for i,name in enumerate(ROOT_FIELDS)}
    write_npz(dest,**result)
    manifest={"roots":len(data),"menus":int(result["eligible"].sum()),"sha256":{p.name:sha(p) for p in (dest,HERE/f"corpus_{shard}.tsv.gz",Path(__file__),HERE/"protocol.json")}}
    save_json(HERE/f"root_{shard}.json",manifest)


def second_stage(shard):
    src=HERE/f"root_{shard}.npz"
    assert sha(src)==json.loads((HERE/f"root_{shard}.json").read_text())["sha256"][src.name]
    with np.load(src) as z:
        keep=z["eligible"]==1
        ids=z["event"][keep]
        menus={int(e):(int(a),int(b)) for e,a,b in zip(ids,z["A"][keep],z["B"][keep])}
    slot={int(e):i for i,e in enumerate(ids)}
    data=np.full((len(ids),2,2,2),np.nan,dtype=np.float64)
    def consume(ans,meta):
        e,depth,b,t,who=meta; root=ans["rootInfo"]
        assert ans["turnNumber"]==t and root["currentPlayer"]==who
        cell=data[slot[e],int(depth==128),b]
        assert np.isnan(cell).all() and int(root["visits"])>=depth
        cell[:]=[root["rawLead"],root["scoreLead"]]
        assert np.isfinite(cell).all()
    run_engine(child_tasks(shard,menus),consume,"children")
    expected=np.array([is_audit(e) for e in ids])
    assert np.isfinite(data[:,0]).all()
    np.testing.assert_array_equal(np.isfinite(data[:,1]).all(axis=(1,2)),expected)
    assert np.isnan(data[~expected,1]).all()
    dest=HERE/f"search_{shard}.npz"
    # This is the opponent's perspective in both children: A minus B is the mover's gain from choosing B.
    write_npz(dest,event=ids,rawA=data[:,0,0,0],rawB=data[:,0,1,0],s24A=data[:,0,0,1],s24B=data[:,0,1,1],
              audit=expected,raw128A=data[expected,1,0,0],raw128B=data[expected,1,1,0],
              s128A=data[expected,1,0,1],s128B=data[expected,1,1,1])
    manifest={"menus":len(ids),"depth_audits":int(expected.sum()),"sha256":{p.name:sha(p) for p in
              (src,dest,MODEL,CFG,BIN.parent/"usr/bin/katago",Path(__file__),HERE/"protocol.json")}}
    save_json(HERE/f"search_{shard}.json",manifest)


if __name__=="__main__":
    shard=int(sys.argv[1]); assert 0<=shard<SHARDS
    if not (HERE/f"root_{shard}.json").exists(): first_stage(shard)
    if not (HERE/f"search_{shard}.json").exists():second_stage(shard)
