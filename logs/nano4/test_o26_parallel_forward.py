"""GPU regression: full-episode contrastive gradients must survive multi-GPU scatter/gather."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "logs/nano4"))
import torch
import torch.nn.functional as F
import m88_seqid as M
from o26_parallel_forward import make_forward

assert torch.cuda.device_count() >= 2, "test requires two allocated GPUs"
class DtypeEcho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        actual = self.linear(x)
        correct = (torch.is_autocast_enabled("cuda")
                   and torch.get_autocast_dtype("cuda") == torch.bfloat16
                   and actual.dtype == torch.bfloat16)
        return x.new_full((len(x), 1), float(correct))

with torch.autocast("cuda", dtype=torch.bfloat16):
    dtype_flags = make_forward(DtypeEcho().cuda())(torch.zeros(8, 1, device="cuda"))
assert bool((dtype_flags == 1).all()), "replica threads did not preserve bfloat16 autocast"

M.D, M.LAYERS, M.HEADS, M.FFN, M.DROP = 64, 1, 4, 128, 0.0
torch.manual_seed(123)
reference = M.build_model().cuda().train()
parallel = copy.deepcopy(reference).train()
P, G, T = 4, 3, 16
args = (torch.randn(P*G, T, 128, device="cuda"),
        torch.randint(0, 361, (P*G, T), device="cuda"),
        torch.randint(0, 2, (P*G, T), device="cuda"),
        torch.arange(P*G, device="cuda") % 8 + 8,
        torch.randint(0, 2, (P*G,), device="cuda"))

def loss(net, forward):
    z = forward(*args).float().reshape(P, G, M.D_OUT)
    tot, scale = z.sum(1), net.log_scale.exp()
    logits = scale * z.reshape(-1, M.D_OUT) @ F.normalize(tot, dim=-1).T
    own = scale * (z * F.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)
    target = torch.arange(P, device="cuda").repeat_interleave(G)
    return z, F.cross_entropy(logits.scatter(1, target[:, None], own[:, None]), target)

z_ref, l_ref = loss(reference, reference)
z_dp, l_dp = loss(parallel, make_forward(parallel))
torch.testing.assert_close(z_ref, z_dp, rtol=2e-4, atol=2e-5)
torch.testing.assert_close(l_ref, l_dp, rtol=2e-4, atol=2e-5)
l_ref.backward()
l_dp.backward()
for (name, p), (other, q) in zip(reference.named_parameters(), parallel.named_parameters()):
    assert name == other
    assert (p.grad is None) == (q.grad is None), name
    if p.grad is not None:
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-3, atol=3e-5, msg=name)
assert set(reference.state_dict()) == set(parallel.state_dict())
assert parallel.log_scale.grad is not None
# An optimizer step and EMA-style parameter correspondence still operate on the base model.
torch.optim.SGD(reference.parameters(), lr=0.001).step()
torch.optim.SGD(parallel.parameters(), lr=0.001).step()
for name, p in reference.state_dict().items():
    torch.testing.assert_close(p, parallel.state_dict()[name], rtol=2e-4, atol=2e-5, msg=name)
print("PASS: per-replica BF16 linear dtype, outputs, full-episode loss, gradients, optimizer update and checkpoint keys", flush=True)
