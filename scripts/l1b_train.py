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
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.featcache import read_cache  # noqa: E402
from aicup.l1b import RankHeads  # noqa: E402
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
    ap.add_argument("--check-init", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    cfg = HeadsCfg(epochs=a.epochs, batch=a.batch, lr=a.lr, eval_every=a.eval_every, patience=a.patience, inner_frac=a.inner_frac,
                   inner_max_positions=a.inner_max_positions, seed=a.seed, device=a.device)
    model, source, sha = load_trunk(a.ckpt, a.model_config, a.device)
    if a.check_init:
        _, arrays = read_cache(a.cache)
        heads = RankHeads(model.policy_head).to(a.device)
        worst = check_init(model, heads, arrays, np.arange(len(arrays["target"])), n=256)
        print(f"max |Δ log p| between the copied heads and the model's rank-unknown policy on 256 positions: {worst:.3e} "
              f"({'ok' if worst <= INIT_TOL else 'FAIL'}, tolerance {INIT_TOL:g})")
        return 0 if worst <= INIT_TOL else 1
    if out.exists() and not a.overwrite:
        print(f"{out} exists; pass --overwrite to retrain (this also discards {out.name}.ckpt)", file=sys.stderr)
        return 2
    ckpt_path = out.with_name(out.name + ".ckpt")
    if a.overwrite and ckpt_path.exists():
        ckpt_path.unlink()
    heads, meta = train_heads(model, a.cache, cfg, out, extra_meta={"model_source": source, "humanv0_sha256": sha})
    print(f"wrote {out}: inner NLL {meta['inner_nll_init']:.4f} -> {meta['inner_nll_best']:.4f} at epoch {meta['best_epoch']:.2f}, "
          f"{'stopped early' if meta['stopped_early'] else 'ran to the end'}, {meta['wall_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
