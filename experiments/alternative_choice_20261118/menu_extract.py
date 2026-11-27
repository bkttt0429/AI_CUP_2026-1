from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from extract import HERE, PREV, R1, encoded_game, native, save_json, sha

FIELDS="game ply white size local number stones choice actual offer escape".split()


def main():
    with np.load(PREV/"game_counts.npz") as z:
        keys,old,turns=z["keys"],z["cells"],z["own_turns"]
    lookup={k:i for i,k in enumerate(keys)}
    kat=R1/"repo/KataGo/cpp"
    sources=[kat/f for f in ("game/board.cpp","core/rand.cpp","core/rand_helpers.cpp","core/hash.cpp","core/sha2.cpp","core/md5.cpp","core/global.cpp")]
    workers=int(os.getenv("SLURM_CPUS_PER_TASK","1"))
    with TemporaryDirectory(prefix="alt_menu_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as tmp:
        d=Path(tmp); binary=d/"menu"
        subprocess.run(["g++","-std=c++17","-O2","-ffunction-sections","-fdata-sections","-Wl,--gc-sections","-pthread","-I",kat,
                        HERE/"available_moves.cpp",*sources,"-o",binary],check=True)
        paths=[d/f"part{i}.tsv" for i in range(workers)]
        outs=[p.open("w") for p in paths]
        try:
            for p in sorted((R1/"data/training").glob("train_*.csv")):
                with p.open() as f:
                    for row in csv.DictReader(f):
                        idx=lookup.get(row["rank"]+":"+row["game_id"])
                        if idx is None: continue
                        setup,seq=encoded_game(row["sgf_content"])
                        outs[idx%workers].write(f"{idx} {row['color']} {setup} {seq}\n")
        finally:
            for f in outs: f.close()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks=list(pool.map(native,[(binary,path) for path in paths]))
    events=np.concatenate([c[0] for c in chunks]); status=np.concatenate([c[1] for c in chunks])
    events=events[np.lexsort((events[:,1],events[:,0]))]; status=status[np.argsort(status[:,0])]
    np.testing.assert_array_equal(status[:,0],np.arange(len(keys))); np.testing.assert_array_equal(status[:,1],turns)
    e=dict(zip(FIELDS,events.T)); counts=np.zeros((len(keys),4,2),int)
    cell=2*e["local"]+(e["size"]>1)
    np.add.at(counts[:,:,0],(e["game"],cell),1)
    np.add.at(counts[:,:,1],(e["game"],cell),e["choice"]==0)
    np.testing.assert_array_equal(counts[:,:,0],old[:,:,1]);np.testing.assert_array_equal(counts[:,:,1],old[:,:,3])
    alt=e["choice"]!=0
    with np.load(HERE/"events.npz") as primary:
        for field in ["game","ply","actual","escape","size","white","local"]:
            np.testing.assert_array_equal(e[field][alt],primary[field])
        np.testing.assert_array_equal(e["choice"][alt]==1,primary["kind"]==0)
    assert np.all((e["choice"]!=1)|(e["number"]>0))
    assert np.all((e["offer"]>=0)==(e["number"]==1))
    dst=HERE/"offer_events.npz"; np.savez_compressed(dst,**e)
    result={"events":len(events),"exhaustive_361_point_checks":int(status[:,2].sum()),
            "sha256":{p.name:sha(p) for p in (dst,Path(__file__),HERE/"read_alternatives.cpp",HERE/"available_moves.cpp",HERE/"menu_protocol.json")}}
    assert result["exhaustive_361_point_checks"]>0
    save_json(HERE/"offer_extraction.json",result); print(result,flush=True)


if __name__=="__main__":
    main()
