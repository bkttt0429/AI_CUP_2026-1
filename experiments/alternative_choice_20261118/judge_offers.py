"""Complete ex-ante value comparisons for the alternatives actually declined."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from extract import HERE,PREV,R1,save_json,sha
from judge import BIN,MODEL,CONFIG,coordinate
from aicup.sgf import handicap,setup_and_moves,to_gtp


def repeat(game,ply):
    import hashlib
    return int.from_bytes(hashlib.blake2s(f"offer-repeat-{game}/{ply}".encode(),digest_size=4).digest(),"big")%64==0


def main(shard):
    with np.load(HERE/"offer_events.npz") as z: e={k:z[k] for k in z.files}
    with np.load(HERE/"events.npz") as z: orig={k:z[k] for k in ("game","ply","actual","escape")}
    with np.load(PREV/"game_counts.npz") as z: keys=z["keys"]
    with np.load(HERE/f"judge_{shard}.npz") as z:
        old={ (int(orig["game"][i]),int(orig["ply"][i])):(int(orig["actual"][i]),int(orig["escape"][i]),float(b-a))
              for i,a,b in zip(z["ix"],z["opp_actual"],z["opp_escape"])}
    ix=np.flatnonzero((e["number"]==1)&(e["game"]%4==shard))
    query_by_game=defaultdict(list); values={}; audits={}
    for i in ix:
        game,ply=int(e["game"][i]),int(e["ply"][i])
        if e["choice"][i]==1:
            a,x,d=old[game,ply]
            assert (a,x)==(e["offer"][i],e["escape"][i])
            values[i]=d
            if not repeat(game,ply): continue
            audits[i]=d
        query_by_game[str(keys[game])].append(i)
    nv=R1/".venv/lib/python3.12/site-packages/nvidia"
    env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=":".join(["/work/envstack/apps/gcc/13.2/lib64"]+[str(nv/d/"lib") for d in ("cuda_runtime","cublas","cudnn","cuda_nvrtc","nvjitlink")]+[env.get("LD_LIBRARY_PATH","")])
    branch={}
    with TemporaryDirectory(prefix="rank_menu_judge_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as tmp:
        td=Path(tmp); request=td/"requests.jsonl"; response=td/"responses.jsonl"; seen=set()
        with request.open("w") as qfile:
            for path in sorted((R1/"data/training").glob("train_*.csv")):
                with path.open() as inp:
                    for row in csv.DictReader(inp):
                        key=row["rank"]+":"+row["game_id"]
                        if key not in query_by_game: continue
                        assert key not in seen; seen.add(key)
                        stones,moves=setup_and_moves(row["sgf_content"]); mv=[[c,"pass" if x is None else to_gtp(x,y)] for c,x,y in moves]
                        q={"boardXSize":19,"boardYSize":19,"rules":"japanese","maxVisits":1,"initialPlayer":mv[0][0],
                           "komi":6.5 if not handicap(row["sgf_content"]) else .5,"initialStones":[["B",to_gtp(x,y)] for x,y in stones]}
                        for i in query_by_game[key]:
                            t=int(e["ply"][i]); who="W" if e["white"][i] else "B"
                            assert mv[t]==[who,coordinate(e["actual"][i])] and e["offer"][i]!=e["escape"][i]
                            for b,loc in enumerate([e["offer"][i],e["escape"][i]]):
                                q.update(id=f"{i}.{b}",moves=mv[:t]+[[who,coordinate(loc)]],analyzeTurns=[t+1])
                                qfile.write(json.dumps(q,separators=(",",":"))+"\n")
        assert len(seen)==len(query_by_game)
        cfg=f"logDir={td},nnForcedSymmetry=0,nnRandomize=false,nnCacheSizePowerOfTwo=16,nnMutexPoolSizePowerOfTwo=12,numAnalysisThreads=48,nnMaxBatchSize=96"
        with request.open() as src, response.open("w") as dst:
            subprocess.run([BIN,"analysis","-config",CONFIG,"-model",MODEL,"-override-config",cfg],stdin=src,stdout=dst,env=env,check=True)
        with response.open() as src:
            for line in src:
                packet=json.loads(line)
                if "error" in packet or "warning" in packet: raise RuntimeError(str(packet)[:400])
                i,b=map(int,packet["id"].split("."))
                assert (i,b) not in branch and packet["turnNumber"]==e["ply"][i]+1 and not packet.get("isDuringSearch",False)
                assert packet["rootInfo"]["currentPlayer"]==("B" if e["white"][i] else "W")
                branch[i,b]=float(packet["rootInfo"]["rawLead"])
    differences=[]
    for indices in query_by_game.values():
        for i in indices:
            delta=branch[i,1]-branch[i,0]
            if i in audits: differences.append(abs(values[i]-delta))
            else: values[i]=delta
    assert len(differences)==len(audits)>0 and set(values)==set(ix)
    assert len(branch)==2*sum(map(len,query_by_game.values()))
    dest=HERE/f"offer_values_{shard}.npz"
    np.savez_compressed(dest,ix=ix,delta=np.array([values[i] for i in ix]),repeat_abs=np.array(differences))
    out={"all":len(ix),"new":len(ix)-int(np.sum(e["choice"][ix]==1)),"audited_accepts":len(differences),
         "max_duplicate_delta":float(max(differences)),"delta_duplicate_p95":float(np.quantile(differences,.95)),
         "sha256":{p.name:sha(p) for p in (dest,Path(__file__),HERE/"offered_value_protocol.json",HERE/"offer_events.npz",HERE/f"judge_{shard}.npz")}}
    save_json(HERE/f"offer_values_{shard}.json",out);print(out)


if __name__=="__main__": main(int(sys.argv[1]))
