"""Parallelize only embedding computation; keep the full-episode loss and base state intact."""

def make_forward(net):
    import torch
    if torch.cuda.device_count() < 2:
        return net

    class BF16Replica(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, *args, **kwargs):
            # parallel_apply propagates enabled state but loses dtype in fresh threads.
            # o26's recipe is BF16; leave FP32 calls uncast for numerical regression.
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.is_autocast_enabled("cuda")):
                return self.base(*args, **kwargs)

    print("parallel helper v2: explicit BF16 in replica threads", flush=True)
    return torch.nn.DataParallel(BF16Replica(net), device_ids=list(range(torch.cuda.device_count())))
