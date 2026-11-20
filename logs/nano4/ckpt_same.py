#!/usr/bin/env python
"""Are the slim checkpoints of one folder present in another with the same weights? (The step-2 pass re-saves a checkpoint,
so the bytes differ; the tensors of best_state / best_wc must be equal.)  Exit code 1 on any difference.

  python -I logs/nano4/ckpt_same.py <source folder> <destination folder> [name pattern, default ckpt_*.pt]
"""
import sys
from pathlib import Path

import torch


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    return True


src, dst = Path(sys.argv[1]), Path(sys.argv[2]); pat = sys.argv[3] if len(sys.argv) > 3 else "ckpt_*.pt"; bad = 0; fs = sorted(src.glob(pat))
for f in fs:
    g = dst / f.name
    if not g.exists():
        print(f"  {f.name}: MISSING in {dst}"); bad += 1; continue
    x, y = torch.load(f, map_location="cpu", weights_only=False), torch.load(g, map_location="cpu", weights_only=False)
    ok = x.get("best_state") is not None and same(x["best_state"], y.get("best_state")) and same(x.get("best_wc"), y.get("best_wc")) and x.get("best") == y.get("best")
    print(f"  {f.name}: {'same weights' if ok else 'DIFFERENT'} ({len(x['best_state'])} tensors, best step {x['best'].get('step')})"); bad += not ok
print(f"== {len(fs)} file(s) of {src} checked against {dst}: {'all the same' if fs and not bad else str(bad) + ' problem(s)' if fs else 'NO FILES'}"); sys.exit(0 if fs and not bad else 1)
