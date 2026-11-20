#!/usr/bin/env python
"""M216 check (CPU): the network of m216_prod.build_model() starts from exactly the production network's initial values.

Same torch seed -> every parameter of base.build_model() with 256 plies equals the 384-ply network's, the position table
on its first 256 rows; the CPU random stream after the build is the same in both (the new rows do not consume it).
"""
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m216_prod as m  # noqa: E402

base = m.base; base.D, base.FFN = 512, 2048
for ck in (0, 4):
    m.CKPT_LAYERS = ck; base.T_MAX = m.T_OLD; torch.manual_seed(1640); a = base.build_model(); ra = torch.rand(3)
    base.T_MAX = m.T_LONG; torch.manual_seed(1640); b = m.build_model(); rb = torch.rand(3); sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), set(sa) ^ set(sb); diff = [k for k in sa if k != "tpos.weight" and not torch.equal(sa[k], sb[k])]
    assert not diff, diff; assert torch.equal(sa["tpos.weight"], sb["tpos.weight"][:m.T_OLD]) and sb["tpos.weight"].shape[0] == m.T_LONG; assert torch.equal(ra, rb), "the random stream after the build differs"
    assert float(sb["tpos.weight"][m.T_OLD:].abs().sum()) > 0
    print(f"checkpointed layers {ck}: {len(sa)} tensors identical, position table {tuple(sb['tpos.weight'].shape)} with the first {m.T_OLD} rows identical, random stream after the build identical")
print("M216 initial-value check passed")
