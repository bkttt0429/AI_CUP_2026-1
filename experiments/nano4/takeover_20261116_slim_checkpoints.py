"""Approved one-time cleanup: drop only optimizer state from 30 audited checkpoints.

Run through Slurm from the rank project. Each temporary replacement must preserve
every other field exactly, load through the actual pool-adaptation loader, and
produce identical best/last embeddings before it can replace its source.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import torch

ROOT = Path.cwd()
AUDIT = ROOT / "logs/nano4/takeover_20261116_checkpoint_storage_audit.json"
REPORT = ROOT / "logs/nano4/takeover_20261116_slim_result.json"
sys.path.insert(0, str(Path.home() / "BK_AICUP_2026-2/scripts"))
import pid_v2_seq_finetune as ft


def same(a, b, path="root"):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            same(a[key], b[key], f"{path}/{key}")
        if hasattr(a, "_metadata"):
            same(a._metadata, b._metadata, path + "/_metadata")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}/{i}")
    else:
        assert a == b, (path, a, b)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    audit = json.loads(AUDIT.read_text())
    assert len(audit["files"]) == 30
    assert not REPORT.exists(), "Do not silently repeat a completed/partial cleanup"
    report = {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
              "removed_key": "opt", "files": []}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    torch.manual_seed(20261116)
    gl = torch.randn(2, 16, 128, device="cuda")
    pos = torch.arange(32, device="cuda").reshape(2, 16)
    pos[0, 4] = 361  # pass
    pos[1, 11:] = 362  # padding
    mov = torch.arange(16, device="cuda").remainder(2).expand(2, -1)
    lens = torch.tensor([16, 11], device="cuda")
    subjects = torch.tensor([0, 1], device="cuda")
    for item in audit["files"]:
        path = Path(item["path"])
        assert path.parent in [ROOT / "results" / x for x in ("m88", "m90", "m101", "m101c")]
        assert path.resolve() == path and not path.is_symlink()
        before = path.stat()
        assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1
        assert before.st_size == item["bytes"], str(path)
        original_hash = digest(path)
        source = torch.load(path, map_location="cpu", weights_only=True)
        assert "opt" in source and source["step"] == item["step"]
        keep = {k: v for k, v in source.items() if k != "opt"}
        assert "net" in keep and "best_state" in keep
        tmp = path.with_name(path.name + ".takeover-slim.tmp")
        try:
            with tmp.open("xb") as f:
                torch.save(keep, f)
                f.flush()
                os.fsync(f.fileno())
            reloaded = torch.load(tmp, map_location="cpu", weights_only=True)
            same(keep, reloaded)
            assert "opt" not in reloaded and tmp.stat().st_size < before.st_size
            for state in ("best", "last"):
                original_net = ft.load_net(path, state=state).eval()
                with torch.inference_mode():
                    expected = original_net(gl, pos, mov, lens, subjects)
                del original_net
                new_net = ft.load_net(tmp, state=state).eval()
                with torch.inference_mode():
                    actual = new_net(gl, pos, mov, lens, subjects)
                assert torch.isfinite(actual).all() and actual.shape == (2, 128)
                assert torch.equal(expected, actual), (path, state, (expected - actual).abs().max())
                del new_net, actual, expected
            now = path.stat()
            assert (now.st_ino, now.st_mtime_ns, now.st_size) == (before.st_ino, before.st_mtime_ns, before.st_size)
            assert digest(path) == original_hash, "Source changed during verification"
            new_hash = digest(tmp)
            os.chmod(tmp, stat.S_IMODE(before.st_mode))
            os.replace(tmp, path)
            assert digest(path) == new_hash
            row = {"path": str(path), "before_bytes": before.st_size,
                   "after_bytes": path.stat().st_size, "before_sha256": original_hash,
                   "after_sha256": new_hash, "retained_fields_exact": True,
                   "best_last_embeddings_exact": True, "retained_keys": list(keep)}
            report["files"].append(row)
            REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{len(report['files'])}/30 verified and slimmed: {path}", flush=True)
        finally:
            if tmp.exists():
                tmp.unlink()
        del source, keep, reloaded
        gc.collect()
    report["status"] = "completed_verified"
    report["saved_bytes"] = sum(x["before_bytes"] - x["after_bytes"] for x in report["files"])
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"COMPLETE: {len(report['files'])} checkpoints, saved {report['saved_bytes']} bytes", flush=True)


if __name__ == "__main__":
    run()
