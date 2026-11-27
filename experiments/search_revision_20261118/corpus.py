"""Join frozen rank questions and choose independent additional games by SGF digest."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"ladder_rank_20261118"
sys.path.insert(0,str(PREV))
from extract_games import R1, RV, encoded_game, questions, save_json, sha
from aicup.sgf import handicap

SHARDS=8


def training_rows():
    for file in sorted((R1/"data/training").glob("train_*.csv")):
        with file.open() as src:
            yield from csv.DictReader(src)


def main():
    qs,_=questions()
    with np.load(PREV/"game_counts.npz") as z:
        oldkeys=z["keys"].tolist()
        assert z["valid"].all()
    old=set(oldkeys)
    owners={q[1]:(q[0],q[2]) for q in qs}
    assert all(owners[q[1]]==(q[0],q[2]) for q in qs)
    selected={k:q[1] for q in qs for k in q[4]}; assert set(selected)==old
    history_hash={}
    for row in training_rows():
        key=row["rank"]+":"+row["game_id"]
        if key in old:
            assert selected[key]==row["player_id"] and key not in history_hash
            history_hash[key]=hashlib.sha256(row["sgf_content"].encode()).digest()
    assert len(history_hash)==len(old)
    historical=set(history_hash.values()); candidates=defaultdict(list); frequencies=Counter()
    for row in training_rows():
        owner=row["player_id"]
        if owner not in owners or owners[owner][0]<3: continue
        key=row["rank"]+":"+row["game_id"]
        if key in old: continue
        digest=hashlib.sha256(row["sgf_content"].encode()).digest()
        if digest in historical: continue
        candidates[owner].append((digest,key));frequencies[digest]+=1
    fresh={}
    for owner,rows in candidates.items():
        good=sorted((d,k) for d,k in rows if frequencies[d]==1)
        for d,key in good[:4]:
            fresh[key]=owner;history_hash[key]=d
    keys=oldkeys+sorted(fresh);selected.update(fresh)
    index={k:i for i,k in enumerate(keys)}; seen=set()
    plies=np.zeros(len(keys),int);colors=np.zeros(len(keys),int)
    with ExitStack() as stack:
        streams=[stack.enter_context(gzip.open(HERE/f"corpus_{s}.tsv.gz","wt",compresslevel=3)) for s in range(SHARDS)]
        for row in training_rows():
            key=row["rank"]+":"+row["game_id"]
            if key not in index:continue
            game=index[key];owner=selected[key];sgf=row["sgf_content"]
            assert owner==row["player_id"] and game not in seen
            assert history_hash[key]==hashlib.sha256(sgf.encode()).digest()
            stones,seq=encoded_game(sgf);hd=handicap(sgf)
            colors[game]=int(row["color"]=="W");plies[game]=len(seq)//3
            streams[game%SHARDS].write(f"{game}\t{history_hash[key].hex()}\t{row['color']}\t{hd or 0}\t{stones}\t{seq}\n")
            seen.add(game)
    assert len(seen)==len(keys)
    isfresh=np.arange(len(keys))>=len(oldkeys)
    owners_arr=np.array([selected[k] for k in keys])
    folds=np.array([owners[o][0] for o in owners_arr],np.int8);rank=np.array([owners[o][1] for o in owners_arr],np.int8)
    set12={history_hash[k] for k in keys if owners[selected[k]][0]<=2}
    set34={history_hash[k] for k in keys if owners[selected[k]][0]>=3}
    assert not set12&set34 and not set(history_hash[k] for k in fresh)&historical
    dest=HERE/"games.npz"
    np.savez_compressed(dest,keys=keys,pid=owners_arr,fold=folds,y=rank,fresh=isfresh,plies=plies,white=colors)
    report={"question_games":len(old),"fresh_games":len(fresh),"fresh_owners":len(set(fresh.values())),"questions":len(qs),
            "fresh_digest_intersection":0,"dev_eval_digest_intersection":0,"sha256":{p.name:sha(p) for p in
             (HERE/"protocol.json",Path(__file__),dest,RV/"m183_dump.npz",*(HERE/f"corpus_{s}.tsv.gz" for s in range(SHARDS)))}}
    save_json(HERE/"corpus.json",report)
    print({k:v for k,v in report.items() if k!="sha256"},flush=True)


if __name__=="__main__": main()
