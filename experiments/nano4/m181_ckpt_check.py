#!/usr/bin/env python
"""Section 4M M181 check: the layer-checkpointed encoder gives the same loss and gradients as the plain forward
(same weights, same batch, same dropout masks), and what each number of checkpointed layers costs in GPU memory and time.
  python -I logs/nano4/m181_ckpt_check.py WIDTH FFN BATCH K [K ...]      (K = number of checkpointed layers; 0 = plain)
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161r_prod as mr  # noqa: E402

base = mr.base; W, Fz, B = (int(x) for x in sys.argv[1:4]); KS = [int(x) for x in sys.argv[4:]]; base.D, base.FFN = W, Fz; dev = "cuda"; g = torch.Generator().manual_seed(0); T = base.T_MAX
gl = torch.randn(B, T, 128, generator=g).to(dev); pos = torch.randint(0, 361, (B, T), generator=g).to(dev); mov = torch.randint(0, 2, (B, T), generator=g).to(dev); ln = torch.randint(80, T, (B,), generator=g).to(dev); subj = torch.randint(0, 2, (B,), generator=g).to(dev); tgt = torch.randint(0, 16, (B,), generator=g).to(dev)
ref = None
for K in KS:
    mr.CKPT_LAYERS = K; torch.manual_seed(1); net = mr.build_model().to(dev); net.train(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        ts = []
        for it in range(4):
            net.zero_grad(set_to_none=True); torch.manual_seed(7); torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(gl, pos, mov, ln, subj)
            loss = torch.nn.functional.cross_entropy(10.0 * z.float() @ z.float()[:16].detach().T, tgt); loss.backward(); torch.cuda.synchronize(); ts.append(time.time() - t0)
        gv = torch.cat([p.grad.flatten().double() for p in net.parameters() if p.grad is not None]).cpu(); line = f"loss {float(loss.detach()):.6f}; peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB; {min(ts[1:]):.3f} s per forward + backward"
        if ref is None:
            ref = gv
        else:
            line += f"; gradient against the first K: cosine {float((ref @ gv) / (ref.norm() * gv.norm())):.6f}, relative difference {float((ref - gv).norm() / ref.norm()):.5f}"
        print(f"width {W}, batch {B}, T {T}, checkpointed layers {K}: {line}", flush=True)
    except torch.OutOfMemoryError:
        print(f"width {W}, batch {B}, T {T}, checkpointed layers {K}: OUT OF MEMORY ({torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f} GB card)", flush=True)
    del net; torch.cuda.empty_cache()
