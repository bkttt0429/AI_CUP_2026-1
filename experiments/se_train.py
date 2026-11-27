#!/usr/bin/env python
"""Train the L1b strength head on a featcache (issue #14, option b): a Bradley-Terry score of each move.

As scripts/l1b_train.py, with aicup.strength.train_strength_head: the trunk is humanv0 (--ckpt, frozen unless
--trunk-lr > 0) or, for tests, a random KataGo config (--model-config); the output is <out> (.pt) with a .json sidecar
of kind "strength"; an existing output is refused (exit 2) unless --overwrite (which also drops the checkpoint) or
--extend (continue a finished run from its kept checkpoint); an existing <out>.ckpt without <out> resumes. --init-head
starts from a saved strength head (a stage-2 file also loads its trunk). Several GPUs: torchrun, one process per GPU.

  python -I scripts/se_train.py --cache results/l1b/cache_A --out results/l1b/se_A.pt
  torchrun --standalone --nproc_per_node 8 scripts/se_train.py --cache results/l1b/cache_A --out results/l1b/se_A.pt --batch 128
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.l1btrain import HeadsCfg  # noqa: E402
from aicup.strength import load_strength_head, train_strength_head  # noqa: E402
import l1b_train as lt  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="strength head .pt (the .json sidecar goes next to it)")
    ap.add_argument("--ckpt", default=str(lt.CKPT))
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
    ap.add_argument("--init-head", default=None, help="start from this saved strength head (.pt); a stage-2 file also loads its trunk")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output and its checkpoint")
    ap.add_argument("--extend", action="store_true", help="continue a finished run from its kept checkpoint (e.g. more --epochs)")
    a = ap.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
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
    model, source, sha = lt.load_trunk(a.ckpt, a.model_config, a.device)
    if out.exists() and not (a.overwrite or a.extend):
        if me == 0:
            print(f"{out} exists; pass --overwrite to retrain (this also discards {out.name}.ckpt) or --extend to continue it", file=sys.stderr)
        return 2
    ckpt_path = out.with_name(out.name + ".ckpt")
    if a.overwrite and not a.extend and ckpt_path.exists() and me == 0:
        ckpt_path.unlink()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        dist.barrier()
    extra = {"model_source": source, "humanv0_sha256": sha}
    init_state = None
    if a.init_head:
        init, init_meta, _ = load_strength_head(a.init_head, model, "cpu")
        init_state = init.state_dict()
        extra.update({"head_init": str(a.init_head), "head_init_sha256": init_meta.get("state_sha256"),
                      "head_init_trunk_sha256": init_meta.get("trunk_sha256"), "head_init_inner_bt_best": init_meta.get("inner_bt_best")})
    try:
        head, meta = train_strength_head(model, a.cache, cfg, out, extra_meta=extra, init_head=init_state)
    except ValueError as e:
        if me == 0:
            print(e, file=sys.stderr)
        return 2
    if me == 0:
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"wrote {out}: inner BT loss {fmt(meta['inner_bt_init'])} -> {fmt(meta['inner_bt_best'])}, pair accuracy "
              f"{fmt(meta['inner_pair_accuracy_init'])} -> {fmt(meta['inner_pair_accuracy_best'])} at epoch {meta['best_epoch']:.2f}, "
              f"{'stopped early' if meta['stopped_early'] else 'ran to the end'}, {meta['steps_run']} steps, {meta['wall_seconds']:.0f} s"
              + (f", trunk fine-tuned ({meta['n_trunk_params']} parameters, lr {cfg.trunk_lr:g}), world {meta['world_size']}" if meta["trunk_trained"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
