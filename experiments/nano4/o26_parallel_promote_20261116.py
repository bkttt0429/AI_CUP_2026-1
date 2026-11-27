"""Validate a complete o26 run before installing its checkpoints for the existing chain.

Run on Slurm after ALL training groups have succeeded, with the old writer stopped.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "logs/nano4"))
import m88_seqid as M

mode = sys.argv[1]
assert mode in ("all", "split", "split-r1")
root = Path("/work/u2507115/pid_v2/o26_parallel_run_20261116")
dest = Path("/work/u2507115/pid_v2/seqnet")
copies = []
for i in range(8):
    group = "all" if mode == "all" else str(int(i in (2, 3, 6, 7)))
    tag = f"o26Ss{i}" if i < 4 else f"o26SMs{i-4}"
    source = root / f"g{group}"
    if mode == "split-r1" and group == "0":
        source = root / "g0r1"
    history = source / f"train_-1_{tag}.json"
    meta = json.loads(history.read_text())
    assert meta["step"] == 40000 and meta["horizon"] == 60000, (tag, meta)
    assert meta["seed"] == 180 + i and meta["all_folds"] is False, tag
    assert meta["games"] == 1858245 and meta["players_trained"] == 121067, tag
    assert meta["mask"] == 0.3 and meta["sym_aug"] is True, tag
    assert meta["drop_open"] == (0.0 if i < 4 else 0.3), tag
    assert max(row["step"] for row in meta["history"]) == 40000, tag
    checkpoint = source / f"ckpt_-1_{tag}e.pt"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 40000 and state["best"]["seed"] == 180 + i, tag
    assert state["best"]["state"] == "ema" and state["net"], tag
    assert state["arch"] == [384, 8, 8, 1536], tag
    M.D, M.LAYERS, M.HEADS, M.FFN = state["arch"]
    model = M.build_model()
    model.load_state_dict(state["best_state"], strict=True)
    assert all(torch.isfinite(t).all().item() for t in state["best_state"].values()), tag
    assert state["net"].keys() == state["best_state"].keys(), tag
    assert all(torch.equal(t, state["best_state"][k]) for k, t in state["net"].items()), tag
    del model
    del state
    copies.extend((checkpoint, history))

# Validate every source and destination before the first copy; never overwrite another writer.
for source in copies:
    assert not (dest / source.name).exists(), f"destination exists: {dest/source.name}"
manifest = []
for source in copies:
    target = dest / source.name
    temporary = dest / (source.name + ".o26-promoting")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    src_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(temporary.read_bytes()).hexdigest() == src_hash, temporary
    target.hardlink_to(temporary)
    temporary.unlink()
    manifest.append({"source": str(source), "destination": str(target), "sha256": src_hash})
(root / "promotion.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("PROMOTED: eight validated 40000-step EMA checkpoints and histories", flush=True)
