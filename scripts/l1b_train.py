#!/usr/bin/env python
"""Train the L1b rank heads on a featcache (plan 2026-10-25-l1b-rank-heads, Task 4).

The trunk is humanv0 (--ckpt, frozen) or, for tests, a randomly initialised KataGo config (--model-config).
`--check-init` only measures max |Δ log p| between the freshly copied heads and the model's own policy
under the rank-unknown metadata on 256 cached positions (fp32) and exits 0 when it is <= 1e-4.
Otherwise it trains with aicup.l1btrain.train_heads and writes <out> (.pt) and its .json sidecar; an
existing output is refused (exit 2) unless --overwrite, which also discards the checkpoint; an existing
<out>.ckpt without <out> resumes the interrupted run.

  python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt --check-init
  python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt

Stage 2 (issue #6): --trunk-lr > 0 fine-tunes the trunk too (its state_dict is saved as <out stem>.trunk.pt and
named in the sidecar, so l1b_score.py loads it with the heads); --init-heads starts the heads from a saved run
(the stage-1 heads). Several GPUs: run under torchrun; every rank takes its share of each global batch of
--batch x world positions and rank 0 writes the outputs.

  torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A \
      --out results/l1b/heads2_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads_A.pt --batch 128
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.featcache import read_cache  # noqa: E402
from aicup.l1b import RankHeads, load_heads  # noqa: E402
from aicup.l1btrain import HeadsCfg, check_init, train_heads  # noqa: E402
from aicup.setmodel_io import file_sha256  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
INIT_TOL = 1e-4


def load_trunk(ckpt, model_config, device):
    """(model in eval mode on device, source string, checkpoint sha256 or None)."""
    import torch
    if model_config:
        from katago.train import modelconfigs
        from katago.train.model_pytorch import Model
        torch.manual_seed(0)
        model = Model(modelconfigs.config_of_name[model_config], 19)
        model.initialize()
        return model.to(device).eval(), f"config:{model_config}", None
    from aicup.humansl import load_humansl
    model, _ = load_humansl(str(ckpt), device)
    return model, str(ckpt), file_sha256(ckpt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="heads .pt (the .json sidecar goes next to it)")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--model-config", default=None, help="random KataGo config instead of --ckpt (tests)")
    ap.add_argument("--epochs", type=int, default=HeadsCfg.epochs)
    ap.add_argument("--batch", type=int, default=HeadsCfg.batch)
    ap.add_argument("--lr", type=float, default=HeadsCfg.lr)
    ap.add_argument("--eval-every", type=float, default=HeadsCfg.eval_every)
    ap.add_argument("--patience", type=int, default=HeadsCfg.patience)
    ap.add_argument("--inner-frac", type=float, default=HeadsCfg.inner_frac)
    ap.add_argument("--inner-max-positions", type=int, default=HeadsCfg.inner_max_positions)
    ap.add_argument("--seed", type=int, default=HeadsCfg.seed)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trunk-lr", type=float, default=HeadsCfg.trunk_lr, help="> 0: fine-tune the trunk at this lr (stage 2)")
    ap.add_argument("--init-heads", default=None, help="start the heads from this saved run's .pt (e.g. the stage-1 heads)")
    ap.add_argument("--check-init", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output and its checkpoint")
    ap.add_argument("--extend", action="store_true", help="continue a finished run from its kept checkpoint (e.g. more --epochs)")
    a = ap.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:  # torchrun: one process per GPU
        import torch
        import torch.distributed as dist
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        a.device = f"cuda:{local}"
    me = int(os.environ.get("RANK", "0"))
    try:
        return _run(a, me)
    finally:
        if world > 1:
            import torch.distributed as dist
            dist.destroy_process_group()


def _run(a, me):
    out = Path(a.out)
    cfg = HeadsCfg(epochs=a.epochs, batch=a.batch, lr=a.lr, eval_every=a.eval_every, patience=a.patience, inner_frac=a.inner_frac,
                   inner_max_positions=a.inner_max_positions, seed=a.seed, device=a.device, trunk_lr=a.trunk_lr)
    model, source, sha = load_trunk(a.ckpt, a.model_config, a.device)
    if a.check_init:
        _, arrays = read_cache(a.cache)
        heads = RankHeads(model.policy_head).to(a.device)
        worst = check_init(model, heads, arrays, np.arange(len(arrays["target"])), n=256)
        print(f"max |Δ log p| between the copied heads and the model's rank-unknown policy on 256 positions: {worst:.3e} "
              f"({'ok' if worst <= INIT_TOL else 'FAIL'}, tolerance {INIT_TOL:g})")
        return 0 if worst <= INIT_TOL else 1
    if out.exists() and not (a.overwrite or a.extend):
        if me == 0:
            print(f"{out} exists; pass --overwrite to retrain (this also discards {out.name}.ckpt) or --extend to continue it", file=sys.stderr)
        return 2
    ckpt_path = out.with_name(out.name + ".ckpt")
    if a.overwrite and not a.extend and ckpt_path.exists() and me == 0:
        ckpt_path.unlink()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # no rank may look for a checkpoint before rank 0 removed it
        import torch.distributed as dist
        dist.barrier()
    extra = {"model_source": source, "humanv0_sha256": sha}
    init_state = None
    if a.init_heads:
        init, init_meta, _ = load_heads(a.init_heads, model, "cpu")
        if init_meta.get("trunk_file"):
            print(f"{a.init_heads} carries a fine-tuned trunk; start a stage-2 run from stage-1 heads only", file=sys.stderr)
            return 2
        init_state = init.state_dict()
        extra.update({"head_init": str(a.init_heads), "head_init_sha256": init_meta.get("state_sha256"),
                      "head_init_inner_nll_best": init_meta.get("inner_nll_best")})
    heads, meta = train_heads(model, a.cache, cfg, out, extra_meta=extra, init_heads=init_state)
    if me == 0:
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"wrote {out}: inner NLL {fmt(meta['inner_nll_init'])} -> {fmt(meta['inner_nll_best'])} at epoch {meta['best_epoch']:.2f}, "
              f"{'stopped early' if meta['stopped_early'] else 'ran to the end'}, {meta['steps_run']} steps, {meta['wall_seconds']:.0f} s"
              + (f", trunk fine-tuned ({meta['n_trunk_params']} parameters, lr {cfg.trunk_lr:g}), world {meta['world_size']}"
                 if meta["trunk_trained"] else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
