"""Replay the existing R1 validation games without exposing grade or owner to the tactical extractor."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
R1 = HERE.parents[1]
RV = R1 / "results/r_variants"
sys.path.insert(0, str(R1 / "src"))
from aicup.metric import LEVELS
from aicup.sgf import board_size, setup_and_moves


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_hash(f)


def stream_hash(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError(f"different result already present at {path}")
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)


def questions():
    qs = []
    hashes = {}
    for fold in (1, 2, 3, 4):
        path = R1 / f"results/trunk_seq/r_val{fold}.json"
        hashes[str(path)] = sha(path)
        val = json.loads(path.read_text())["val"]
        for owner, grade, size, games in zip(val["player_id"], val["rank"], val["n_games"], val["game_keys"]):
            assert len(games) == len(set(games)) == size
            qs.append((fold, owner, LEVELS.index(grade), size, games))
    with np.load(RV / "m183_dump.npz") as z, np.load(RV / "m221_SC_dump.npz") as m221:
        for column, key in enumerate(("fold", "pid", "y", "n")):
            np.testing.assert_array_equal([q[column] for q in qs], z[key])
            np.testing.assert_array_equal(z[key], m221[key])
        for key in ("SQ4", "ct106_rule"):
            np.testing.assert_array_equal(z[key], m221[key])
    return qs, hashes


def encoded_game(sgf):
    if board_size(sgf) != 19:
        raise ValueError("size")
    setup, moves = setup_and_moves(sgf)
    if not moves:
        raise ValueError("empty")
    coords = "abcdefghijklmnopqrs"
    stones = "".join(coords[x] + coords[y] for x, y in setup) or "-"
    seq = "".join(c + ("--" if x is None else coords[x] + coords[y]) for c, x, y in moves)
    return stones, seq


def native_shard(args):
    binary, path = args
    with path.open("rb") as f, path.with_suffix(".out").open("wb") as out:
        subprocess.run([binary], stdin=f, stdout=out, check=True)
    return np.loadtxt(path.with_suffix(".out"), dtype=np.int32, ndmin=2)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run the full corpus in a Slurm allocation")
    qs, hashes = questions()
    needed = {k: (q[0], q[1]) for q in qs for k in q[4]}
    assert all(needed[k] == q[:2] for q in qs for k in q[4])
    keys = sorted(needed)
    indexes = {k: i for i, k in enumerate(keys)}
    seen, corrupt = set(), set()
    by_hash = defaultdict(list)
    n_shards = min(int(os.getenv("SLURM_CPUS_PER_TASK", 1)), 6)
    with TemporaryDirectory(prefix="r1_ladder_", dir=os.getenv("SLURM_TMPDIR") or "/tmp") as tmp:
        work = Path(tmp)
        binary = work / "read_ladders"
        subprocess.run(["bash", HERE / "build.sh", binary], check=True)
        subprocess.run([binary, "--self-check"], check=True)
        shards = [work / f"games_{i}.txt" for i in range(n_shards)]
        streams = [path.open("w") for path in shards]
        try:
            for source in sorted((R1 / "data/training").glob("train_*.csv")):
                with source.open() as f:
                    for row in csv.DictReader(f):
                        key = row["rank"] + ":" + row["game_id"]
                        if key not in indexes:
                            continue
                        i = indexes[key]
                        assert i not in seen and needed[key][1] == row["player_id"]
                        seen.add(i)
                        sgf = row["sgf_content"]
                        by_hash[hashlib.sha256(sgf.encode()).digest()].append(i)
                        try:
                            setup, seq = encoded_game(sgf)
                        except ValueError:
                            corrupt.add(i)
                            continue
                        color = row["color"]
                        assert color in ("B", "W")
                        streams[i % n_shards].write(f"{i} {color} {setup} {seq}\n")
                print("parsed", source.name, len(seen), flush=True)
        finally:
            for stream in streams:
                stream.close()
        assert len(seen) == len(keys)
        with ThreadPoolExecutor(max_workers=n_shards) as pool:
            pieces = list(pool.map(native_shard, [(binary, path) for path in shards]))
    rows = np.zeros((len(keys), 20), dtype=np.int32)
    ids = np.concatenate([a[:, 0] for a in pieces])
    assert set(ids) == seen - corrupt and len(ids) == len(set(ids))
    for a in pieces:
        assert a.shape[1] == 21
        rows[a[:, 0]] = a[:, 1:]
    bad_replay = int((rows[:, 0] == 0).sum())
    mixed = [i for group in by_hash.values() if len({needed[keys[i]][0] for i in group}) > 1 for i in group]
    rows[mixed, 0] = 0
    cells = rows[:, 4:].reshape(len(keys), 4, 4)
    count, safe, chosen, safe_chosen = np.moveaxis(cells, -1, 0)
    assert ((safe_chosen <= np.minimum(safe, chosen)) & (count >= safe + chosen - safe_chosen)).all()
    assert np.all(count.sum(1) <= rows[:, 2])
    dest = HERE / "game_counts.npz"
    with dest.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, keys=keys, valid=rows[:, 0].astype(bool), own_turns=rows[:, 1],
                            single_threats=rows[:, 2], repeated=rows[:, 3], cells=cells.astype(np.uint16))
    dest.with_suffix(".tmp").replace(dest)
    hashes.update({str(p): sha(p) for p in (HERE / "protocol.json", HERE / "read_ladders.cpp",
                  HERE / "build.sh", Path(__file__), R1 / "src/aicup/sgf.py", RV / "m183_dump.npz", RV / "m221_SC_dump.npz")})
    good = rows[:, 0].astype(bool)
    report = {"games": len(keys), "valid": int(good.sum()), "invalid_replay_or_parse": bad_replay,
              "invalid_parse": len(corrupt), "cross_fold_duplicate_game_rows": len(mixed),
              "valid_own_turns": int(rows[good, 1].sum()), "unique_events": int(cells[good, :, 0].sum()),
              "games_with_event": int((count[good].sum(1) > 0).sum()), "sha256": hashes,
              "counts_sha256": sha(dest), "katago_revision": subprocess.check_output(
                  ["git", "-C", str(R1 / "repo/KataGo"), "rev-parse", "HEAD"], text=True).strip()}
    save_json(HERE / "extraction.json", report)
    print({k: v for k, v in report.items() if k != "sha256"}, flush=True)


if __name__ == "__main__":
    main()
