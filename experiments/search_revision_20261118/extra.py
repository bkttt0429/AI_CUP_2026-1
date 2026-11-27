"""Reserve new full games before inspecting the opponent-location interaction."""
from collections import Counter,defaultdict
from contextlib import ExitStack
import gzip
import hashlib
import os
import sys
import numpy as np
import search
from corpus import HERE,SHARDS,encoded_game,handicap,save_json,sha,training_rows

EXTRA=HERE/"confirmation"


def build():
    EXTRA.mkdir(exist_ok=True)
    with np.load(HERE/"games.npz") as z:
        known=set(z["keys"]); take=z["fold"]>=3
        owners={p:(int(f),int(y)) for p,f,y in zip(z["pid"][take],z["fold"][take],z["y"][take])}
    raw=set();matches=set()
    def canonical(h,stone,moves):
        return hashlib.sha256(b"\0".join((h,stone,moves))).digest()
    for s in range(SHARDS):
        with gzip.open(HERE/f"corpus_{s}.tsv.gz","rb") as f:
            for row in f:
                _,d,_,h,stone,moves=row.rstrip().split(b"\t");raw.add(bytes.fromhex(d.decode()));matches.add(canonical(h,stone,moves))
    assert len(raw)==len(known)==len(matches)
    choices=defaultdict(list);counts=Counter()
    for row in training_rows():
        p=row["player_id"];k=row["rank"]+":"+row["game_id"]
        if p not in owners or k in known:continue
        d=hashlib.sha256(row["sgf_content"].encode()).digest()
        if d in raw:continue
        choices[p].append((d,k));counts[d]+=1
    selected={k:(p,d) for p,rows in choices.items() for d,k in sorted((d,k) for d,k in rows if counts[d]==1)[:4]}
    keys=sorted(selected);ix={k:i for i,k in enumerate(keys)}
    seen=set();new_replays=set();pids=[selected[k][0] for k in keys]
    with ExitStack() as stack:
        writers=[stack.enter_context(gzip.open(EXTRA/f"corpus_{s}.tsv.gz","wt",compresslevel=3)) for s in range(SHARDS)]
        for row in training_rows():
            k=row["rank"]+":"+row["game_id"]
            if k not in ix:continue
            p,d=selected[k];assert row["player_id"]==p and k not in seen
            sgf=row["sgf_content"];assert hashlib.sha256(sgf.encode()).digest()==d
            st,moves=encoded_game(sgf);h=handicap(sgf) or 0
            dig=canonical(str(h).encode(),st.encode(),moves.encode());assert dig not in matches and dig not in new_replays
            new_replays.add(dig);seen.add(k);g=ix[k]
            writers[g%SHARDS].write(f"{g}\t{d.hex()}\t{row['color']}\t{h}\t{st}\t{moves}\n")
    assert len(seen)==len(keys)
    np.savez_compressed(EXTRA/"games.npz",keys=keys,pid=pids,
                        fold=[owners[p][0] for p in pids],y=[owners[p][1] for p in pids],fresh=np.ones(len(keys),bool))
    save_json(EXTRA/"protocol.json",{"frozen_anchor_sha256":sha(HERE/"protocol.json"),"interaction_sha256":sha(HERE/"interaction_protocol.json")})
    save_json(EXTRA/"corpus.json",{"games":len(keys),"owners":len(set(pids)),"overlap_raw":0,"overlap_full_replays":0,
              "sha256":{p.name:sha(p) for p in (EXTRA/"games.npz",HERE/"extra.py",EXTRA/"protocol.json",*(EXTRA/f"corpus_{s}.tsv.gz" for s in range(SHARDS)))}})
    print("reserved",len(keys),"games",len(set(pids)),"owners",flush=True)


if __name__=="__main__":
    if sys.argv[1]=="build":build()
    else:
        rank=int(os.environ["SLURM_PROCID"]);search.HERE=EXTRA
        for s in range(rank,SHARDS,4):
            if not (EXTRA/f"root_{s}.json").exists():search.first_stage(s)
