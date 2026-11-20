"""Replay alternatives on validated training SGFs; only integer moves reach the tactical reader."""
import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"ladder_rank_20261118"
sys.path.insert(0,str(PREV))
from extract_games import R1, encoded_game, questions, save_json, sha

COLS = "game ply white target escape actual size local kind immediate_taken ataris near immediate_lib replies lost_first lost_second taken_window final_lib".split()
NAMES = ["immediate_rescue","capture","new_atari","other_near","other_far","pass"]


def native(args):
    binary, path = args
    with path.open("rb") as inp, path.with_suffix(".ev").open("wb") as out:
        subprocess.run([binary,path.with_suffix(".status")], stdin=inp, stdout=out, check=True)
    return (np.loadtxt(path.with_suffix(".ev"), dtype=np.int32, ndmin=2),
            np.loadtxt(path.with_suffix(".status"),dtype=np.int32,ndmin=2))


def main():
    qs,_=questions()
    owner={key:q[1] for q in qs for key in q[4]}
    previous=PREV/"game_counts.npz"
    with np.load(previous) as z:
        keys,valid,own_turns,base=z["keys"],z["valid"],z["own_turns"],z["cells"].astype(np.int64)
    assert valid.all()
    idx={key:i for i,key in enumerate(keys)}
    shards=int(os.getenv("SLURM_CPUS_PER_TASK","1"))
    kat=R1/"repo/KataGo/cpp"
    sources=[kat/f for f in ("game/board.cpp","core/rand.cpp","core/rand_helpers.cpp","core/hash.cpp","core/sha2.cpp","core/md5.cpp","core/global.cpp")]
    seen=set()
    with TemporaryDirectory(prefix="alt_choice_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as d:
        tmp=Path(d)
        binary=tmp/"read"
        subprocess.run(["g++","-std=c++17","-O2","-ffunction-sections","-fdata-sections","-Wl,--gc-sections","-pthread","-I",kat,
                        HERE/"read_alternatives.cpp",*sources,"-o",binary],check=True)
        subprocess.run([binary,"--self-test"],check=True)
        paths=[tmp/f"games{i}.tsv" for i in range(shards)]
        streams=[path.open("w") for path in paths]
        try:
            for source in sorted((R1/"data/training").glob("train_*.csv")):
                with source.open() as f:
                    for row in csv.DictReader(f):
                        key=row["rank"]+":"+row["game_id"]
                        if key in idx:
                            i=idx[key]
                            assert i not in seen and owner[key]==row["player_id"]
                            seen.add(i)
                            stones,seq=encoded_game(row["sgf_content"])
                            streams[i%shards].write(f"{i} {row['color']} {stones} {seq}\n")
        finally:
            for f in streams: f.close()
        assert len(seen)==len(keys)
        with ThreadPoolExecutor(max_workers=shards) as pool:
            chunks=list(pool.map(native,[(binary,path) for path in paths]))
    events=np.concatenate([part[0] for part in chunks])
    status=np.concatenate([part[1] for part in chunks])
    status=status[np.argsort(status[:,0])]
    np.testing.assert_array_equal(status[:,0],np.arange(len(keys)))
    np.testing.assert_array_equal(status[:,1],own_turns)
    events=events[np.lexsort((events[:,1],events[:,0]))]
    assert np.unique(events[:,:2],axis=0).shape[0]==len(events)
    e=dict(zip(COLS,events.T))
    assert (e["actual"]!=e["escape"]).all()
    counts=np.zeros((len(keys),4),int)
    np.add.at(counts,(e["game"],2*e["local"]+(e["size"]>1)),1)
    np.testing.assert_array_equal(counts,base[:,:,1]-base[:,:,3])
    complete=e["replies"]==2
    assert (e["lost_second"][~complete]==-1).all() and (e["taken_window"][~complete]==-1).all()
    assert np.all(e["lost_second"][complete]>=e["lost_first"][complete])
    assert np.all((e["final_lib"][complete]==0)==(e["lost_second"][complete]==1))
    assert np.all(e["immediate_taken"][complete]<=e["taken_window"][complete])
    dest=HERE/"events.npz"
    np.savez_compressed(dest,**e)
    report={"games":len(keys),"events":len(events),"complete":int(complete.sum()),"actions":dict(zip(NAMES,map(int,np.bincount(e["kind"],minlength=6)))),
            "per_game_old_cells_identical":True,"sha256":{f.name:sha(f) for f in (dest,HERE/"protocol.json",Path(__file__),HERE/"read_alternatives.cpp",previous)}}
    save_json(HERE/"extraction.json",report)
    print(report,flush=True)


if __name__=="__main__":
    main()
