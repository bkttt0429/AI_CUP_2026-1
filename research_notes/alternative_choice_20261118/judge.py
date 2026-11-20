"""Value each played alternative and the straight escape after exactly the same history."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np

from extract import HERE, PREV, R1, save_json, sha
from aicup.sgf import handicap, setup_and_moves, to_gtp

KDIR=R1/"tools/katago_v1182"
BIN=KDIR/"bin/squashfs-root/AppRun"
MODEL=KDIR/"dl/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
CONFIG=KDIR/"analysis_m25.cfg"
NSHARDS=4


def coordinate(p):
    if p==361:
        return "pass"
    return to_gtp(int(p)%19,int(p)//19)


def searched(gid,ply):
    raw=f"alternative-20261118-{gid}-{ply}".encode()
    return int.from_bytes(hashlib.blake2s(raw,digest_size=4).digest(),"big")%32==0


def main(shard):
    kept=HERE/"events.npz"
    meta=json.loads((HERE/"extraction.json").read_text())
    assert sha(kept)==meta["sha256"][kept.name]
    with np.load(kept) as data:
        game,ply,white,escape,actual=[data[k] for k in ("game","ply","white","escape","actual")]
    with np.load(PREV/"game_counts.npz") as z:
        keys=z["keys"]
    ix=np.flatnonzero(game%NSHARDS==shard)
    deep={int(i) for i in ix if searched(game[i],ply[i])}
    duplicates=set(map(int,ix[:8]))
    jobs=defaultdict(list)
    for i in ix: jobs[str(keys[game[i]])].append(int(i))
    expected=2*len(ix)+2*len(deep)+len(duplicates)
    results={}
    completed=set()
    failures=[]
    semaphore=threading.Semaphore(192)
    stop=threading.Event()
    env=dict(os.environ)
    nv=R1/".venv/lib/python3.12/site-packages/nvidia"
    env["LD_LIBRARY_PATH"]=":".join(["/work/envstack/apps/gcc/13.2/lib64"]+
                 [str(nv/d/"lib") for d in ("cuda_runtime","cublas","cudnn","cuda_nvrtc","nvjitlink")]+[env.get("LD_LIBRARY_PATH","")])
    with TemporaryDirectory(prefix="alt_katago_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as td:
        cfg=f"logDir={td},nnForcedSymmetry=0,nnRandomize=false,nnCacheSizePowerOfTwo=16,nnMutexPoolSizePowerOfTwo=12,numAnalysisThreads=48,nnMaxBatchSize=96"
        cmd=[str(BIN),"analysis","-config",str(CONFIG),"-model",str(MODEL),"-override-config",cfg]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,text=True,bufsize=1,env=env)
        def launch(q):
            while not semaphore.acquire(timeout=1.):
                if stop.is_set(): raise RuntimeError("engine stopped")
            if stop.is_set(): raise RuntimeError("engine stopped")
            proc.stdin.write(json.dumps(q,separators=(",",":"))+"\n")
            proc.stdin.flush()
        def feed():
            try:
                for path in sorted((R1/"data/training").glob("train_*.csv")):
                    with path.open() as f:
                        for row in csv.DictReader(f):
                            if stop.is_set(): return
                            key=row["rank"]+":"+row["game_id"]
                            if key not in jobs: continue
                            assert key not in completed
                            stones,moves=setup_and_moves(row["sgf_content"])
                            mv=[[c,"pass" if x is None else to_gtp(x,y)] for c,x,y in moves]
                            q={"boardXSize":19,"boardYSize":19,"rules":"japanese",
                               "komi":6.5 if not handicap(row["sgf_content"]) else .5,
                               "initialStones":[["B",to_gtp(x,y)] for x,y in stones],"initialPlayer":mv[0][0]}
                            for i in jobs[key]:
                                t=int(ply[i]); color="W" if white[i] else "B"
                                assert mv[t]==[color,coordinate(actual[i])] and color==row["color"]
                                prefix=mv[:t]
                                for visits in ([1,16] if i in deep else [1]):
                                    for branch in ([0,1,2] if visits==1 and i in duplicates else [0,1]):
                                        loc=escape[i] if branch==1 else actual[i]
                                        q.update(id=f"{i}.{visits}.{branch}",moves=prefix+[[color,coordinate(loc)]],
                                                 maxVisits=visits,analyzeTurns=[t+1])
                                        launch(q)
                            completed.add(key)
                assert len(completed)==len(jobs)
            except BaseException as exc:
                if not stop.is_set():
                    failures.append(repr(exc))
                    proc.terminate()
        th=threading.Thread(target=feed,daemon=True); th.start()
        start=time.monotonic()
        try:
            for line in proc.stdout:
                ans=json.loads(line)
                if "error" in ans or "warning" in ans:
                    raise RuntimeError(str(ans)[:900])
                i,visits,branch=map(int,ans["id"].split("."))
                key=(i,visits,branch)
                assert i in range(len(game)) and game[i]%NSHARDS==shard and key not in results
                root=ans["rootInfo"]
                assert ans["turnNumber"]==ply[i]+1 and root["currentPlayer"]==("B" if white[i] else "W")
                value=root["rawLead"] if visits==1 else root["scoreLead"]
                assert np.isfinite(value) and not ans.get("isDuringSearch",False)
                results[key]=float(value)
                semaphore.release()
                if len(results)%5000==0:
                    print(f"{len(results)}/{expected} positions in {time.monotonic()-start:.0f}s",flush=True)
                if len(results)==expected: break
            if len(results)!=expected: raise RuntimeError(f"received {len(results)}/{expected}; feeder errors {failures}")
            th.join(timeout=20)
            assert not th.is_alive() and not failures and len(completed)==len(jobs)
        finally:
            stop.set()
            proc.stdin.close()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait(timeout=15)
    ident=float(max(abs(results[i,1,0]-results[i,1,2]) for i in duplicates))
    assert ident<.15,ident
    deep_ids=np.array(sorted(deep),int)
    dest=HERE/f"judge_{shard}.npz"
    # Both children have the opponent on move. Escape lead for them minus actual lead for them is actual gain for the subject.
    raw_actual=np.array([results[i,1,0] for i in ix]); raw_escape=np.array([results[i,1,1] for i in ix])
    deep_actual=np.array([results[i,16,0] for i in deep_ids]); deep_escape=np.array([results[i,16,1] for i in deep_ids])
    np.savez_compressed(dest,ix=ix,opp_actual=raw_actual,opp_escape=raw_escape,deep_ids=deep_ids,
                        deep_opp_actual=deep_actual,deep_opp_escape=deep_escape)
    out={"events":len(ix),"searched":len(deep_ids),"requests":expected,"duplicate_max_points":ident,"elapsed_seconds":round(time.monotonic()-start,2),
         "sha256":{p.name:sha(p) for p in (MODEL,CONFIG,BIN.parent/"usr/bin/katago",kept,Path(__file__),dest)}}
    print(json.dumps(out),flush=True)
    if not (HERE/f"judge_{shard}.json").exists(): save_json(HERE/f"judge_{shard}.json",out)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("shard",type=int,choices=range(NSHARDS))
    main(ap.parse_args().shard)
