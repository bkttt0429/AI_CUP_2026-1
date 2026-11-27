#!/usr/bin/env python
"""Section 4M M191 (2026-11-16): the identity network with a board-geometry bias in its attention.

m88_seqid.SeqID gives every ply its absolute point, its displacement from the previous ply and from the mover's previous
move; everything else about "where this move is relative to the earlier stones" has to be learned through absolute point
embeddings. Here every pair of plies (i, j) adds a learned bias to the attention of ply i to ply j, one value per head,
indexed by the displacement between the two points (37 x 37 offsets, one more index for passes / padding; the 8 board
symmetries are applied to the points before, as in training). Zero at the start, so the network begins as the plain one.
Nothing else changes: same inputs, same sizes, same trainer.

  build() -> the patched network (base.build_model is replaced by it in m191_pilot.py)
  python -I logs/nano4/m191_relbias.py check        (forward + backward at the production batch: memory, time, dtype)
"""
import sys
import time
import types
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

PLAIN = base.build_model


def build():
    import torch
    from torch import nn
    net = PLAIN(); H = net.enc.layers[0].self_attn.num_heads; net.rel = nn.Embedding(base.NONE + 1, H); nn.init.zeros_(net.rel.weight)

    def forward(self, gl, pos, mov, length, subj):
        B, T = pos.shape; x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
        x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]
        ok = pos < 361; px, py = pos % 19, pos // 19; idx = (px[:, :, None] - px[:, None, :] + 18) * 37 + (py[:, :, None] - py[:, None, :] + 18)
        idx = torch.where(ok[:, :, None] & ok[:, None, :], idx, torch.full_like(idx, base.NONE)); bias = torch.zeros(B, H, T + 1, T + 1, device=pos.device, dtype=torch.float32); bias[:, :, 1:, 1:] = self.rel(idx).permute(0, 3, 1, 2).float()
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf")).reshape(B * H, T + 1, T + 1)
        h = self.enc(x, mask=bias)[:, 0]
        return torch.nn.functional.normalize(self.head(h).float(), dim=-1)
    net.forward = types.MethodType(forward, net)
    return net


def check():
    import torch
    base.D, base.FFN = 512, 2048; dev = "cuda"; B, T = 2048, base.T_MAX; g = torch.Generator().manual_seed(0)
    gl = torch.randn(B, T, 128, generator=g).to(dev); pos = torch.randint(0, 361, (B, T), generator=g).to(dev); mov = torch.randint(0, 2, (B, T), generator=g).to(dev); ln = torch.randint(80, T, (B,), generator=g).to(dev); subj = torch.randint(0, 2, (B,), generator=g).to(dev); tgt = torch.randint(0, 16, (B,), generator=g).to(dev)
    for name, fn in (("plain", PLAIN), ("with the geometry bias", build)):
        torch.manual_seed(1); net = fn().to(dev); net.train(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); ts = []
        for it in range(3):
            net.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(gl, pos, mov, ln, subj)
            loss = torch.nn.functional.cross_entropy(10.0 * z.float() @ z.float()[:16].detach().T, tgt); loss.backward(); torch.cuda.synchronize(); ts.append(time.time() - t0)
        extra = "" if name == "plain" else f"; gradient reaches the bias table: {bool(net.rel.weight.grad.abs().sum() > 0)}"
        print(f"{name}: loss {float(loss.detach()):.6f}; peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB; {min(ts[1:]):.3f} s per forward + backward; parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M{extra}", flush=True); del net


if __name__ == "__main__":
    check()
