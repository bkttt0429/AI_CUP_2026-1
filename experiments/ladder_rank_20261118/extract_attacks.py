"""Replay the already-audited SGF selection for the alternative attacking decision."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from extract_games import HERE, R1, encoded_game, native_shard, questions, save_json, sha


def main():
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("use a Slurm allocation")
    qs, hashes = questions()
    previous = json.loads((HERE/"extraction.json").read_text())
    assert sha(HERE/"game_counts.npz") == previous["counts_sha256"]
    with np.load(HERE/"game_counts.npz") as data:
        keys, good, own_turns = data["keys"], data["valid"], data["own_turns"]
    assert good.all(), "unvalidated or duplicated games require a joint exclusion"
    idx = {k:i for i,k in enumerate(keys)}
    owner = {k:q[1] for q in qs for k in q[4]}
    rows = np.zeros((len(keys),38), dtype=np.int32)
    shards = min(6,int(os.getenv("SLURM_CPUS_PER_TASK",1)))
    src = R1/"repo/KataGo/cpp"
    with TemporaryDirectory(prefix="rank_attack_",dir=os.getenv("SLURM_TMPDIR") or "/tmp") as tmp:
        tmp = Path(tmp)
        binary = tmp/"read_attacks"
        sources = [src/p for p in ("game/board.cpp","core/rand.cpp","core/rand_helpers.cpp","core/hash.cpp",
                                    "core/sha2.cpp","core/md5.cpp","core/global.cpp")]
        subprocess.run(["g++","-std=c++17","-O2","-ffunction-sections","-fdata-sections","-Wl,--gc-sections",
                        "-pthread","-I",src,HERE/"read_attacks.cpp",*sources,"-o",binary],check=True)
        subprocess.run([binary,"--self-check"],check=True)
        paths = [tmp/f"shard_{i}.txt" for i in range(shards)]
        streams = [p.open("w") for p in paths]
        seen = set()
        try:
            for source in sorted((R1/"data/training").glob("train_*.csv")):
                with source.open() as fin:
                    for row in csv.DictReader(fin):
                        k = row["rank"]+":"+row["game_id"]
                        if k not in idx:
                            continue
                        i = idx[k]
                        assert i not in seen and owner[k] == row["player_id"]
                        seen.add(i)
                        stones, moves = encoded_game(row["sgf_content"])
                        streams[i%shards].write(f"{i} {row['color']} {stones} {moves}\n")
        finally:
            for stream in streams:
                stream.close()
        assert len(seen) == len(keys)
        with ThreadPoolExecutor(max_workers=shards) as pool:
            pieces = list(pool.map(native_shard,[(binary,p) for p in paths]))
        ids = np.concatenate([piece[:,0] for piece in pieces])
        np.testing.assert_array_equal(np.sort(ids),np.arange(len(keys)))
        for piece in pieces:
            rows[piece[:,0]] = piece[:,1:]
    np.testing.assert_array_equal(rows[:,0],good)
    np.testing.assert_array_equal(rows[:,1],own_turns)
    cells = rows[:,2:].reshape(len(keys),6,6)
    n,g,k,f,a,j = np.moveaxis(cells,-1,0)
    assert np.all((k<=n) & (g<=n) & (f<=g) & (k-f<=n-g) & (j<=np.minimum(f,a)) & (f+a-j<=k))
    dest=HERE/"attack_counts.npz"
    with dest.with_suffix(".tmp").open("wb") as fout:
        np.savez_compressed(fout,keys=keys,valid=good,cells=cells.astype(np.uint16))
    dest.with_suffix(".tmp").replace(dest)
    hashes.update({str(p):sha(p) for p in (HERE/"extract_games.py",HERE/"extraction.json",HERE/"attack_protocol.json",
                                          HERE/"read_attacks.cpp",Path(__file__))})
    report={"games":len(keys), "eligible":int(n.sum()), "attempted":int(k.sum()),
            "same_turns_as_defense":True,"counts_sha256":sha(dest),"sha256":hashes}
    save_json(HERE/"attack_extraction.json",report)
    print("games",len(keys),"eligible",int(n.sum()),"attempted",int(k.sum()),flush=True)


if __name__=="__main__":
    main()
