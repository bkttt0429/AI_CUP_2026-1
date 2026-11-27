"""Compare request delivery on the same 2,048 fixed child queries with unchanged evaluation settings."""
import itertools
import json
import sys
import tempfile
import time
import os
import subprocess
from pathlib import Path
import numpy as np
from search import HERE, R1, BIN, CFG, MODEL, child_tasks, run_engine

def main():
    with np.load(HERE/"root_0.npz") as r:
        v=r["eligible"]>0;menu=dict(zip(r["event"][v],zip(r["A"][v],r["B"][v])))
    tasks=list(itertools.islice(child_tasks(0,menu),2048))
    out={}
    t=time.monotonic()
    def handle(ans,key):
        out[ans["id"]]=float(ans["rootInfo"]["scoreLead"])
    run_engine(iter(tasks),handle,"pilot_stream")
    print("stream took",time.monotonic()-t,"seconds",flush=True)
    nv=R1/".venv/lib/python3.12/site-packages/nvidia";env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=":".join(["/work/envstack/apps/gcc/13.2/lib64"]+[str(nv/d/"lib") for d in
        ("cuda_runtime","cublas","cudnn","cuda_nvrtc","nvjitlink")]+[env.get("LD_LIBRARY_PATH","")])
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);q=root/"q";a=root/"a"
        with q.open("w") as f:
            for key,req,_ in tasks:
                req["id"]=key;f.write(json.dumps(req,separators=(",",":"))+"\n")
        for threads in (64,128):
            cfg=f"logDir={td},nnForcedSymmetry=0,nnRandomize=false,nnCacheSizePowerOfTwo=17,nnMutexPoolSizePowerOfTwo=12,numAnalysisThreads={threads},nnMaxBatchSize=96"
            t=time.monotonic()
            with q.open("rb") as src,a.open("wb") as dst:
                subprocess.run([BIN,"analysis","-config",CFG,"-model",MODEL,"-override-config",cfg],stdin=src,stdout=dst,env=env,check=True)
            elapsed=time.monotonic()-t
            with a.open() as f: payload={ans["id"]:float(ans["rootInfo"]["scoreLead"]) for ans in map(json.loads,f)}
            assert out.keys()==payload.keys()
            dx=[abs(out[k]-payload[k]) for k in out]
            print("file",threads,elapsed,"seconds",len(payload),"results; median, p95 delta",np.quantile(dx,[.5,.95]),flush=True)
if __name__=="__main__":main()
