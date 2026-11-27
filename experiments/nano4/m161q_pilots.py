#!/usr/bin/env python
"""Section 4M M161q pilots (2026-11-15): one change each on top of the second production recipe, all-players network, seed 30
(design doc M161q). The reference is the production set's own all-players network of seed 30 (results/m161q/out_-1_s30.npz).

Production recipe (m161q_prod.py): by games, 128 players x 16 games, width 512 / feed-forward 2,048, 8 layers, mask 0.3, learned
centres, 40,000 steps. Variants:
  w768   width 768 / feed-forward 3,072          L12    12 layers             s80    80,000 steps
  d256   identity vector of 256 dimensions       b4096  256 players x 16 games per episode (batch 4,096)
  g125   players drawn by games ** 1.25          nomask no ply mask           nocen  no learned centres (does the centre loss cost the prototypes anything)
  w768b / L12b / ctl96   width 768 / 12 layers / the production recipe, all three with 96 players x 16 games (batch 1,536)
Every variant writes to results/m161q/<variant>/ and /work/u2507115/m161q_ckpt/<variant>/ (resumable).

  sbatch -p 8gpus --gres=gpu:6 -c 72 ... run.sbatch python -I logs/nano4/m161q_pilots.py --jobs w768 L12 s80 d256 b4096 g125
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161q_prod as mq  # noqa: E402

base = mq.base
VARIANTS = {"w768": dict(size=(768, 3072)), "L12": dict(layers=12), "s80": dict(steps=80000), "d256": dict(d_out=256), "b4096": dict(ep=(256, 16)), "g125": dict(gamma=1.25),
            "nomask": dict(mask=0.0), "nocen": dict(centres=False),
            # second batch (2026-11-15 afternoon): w768 / L12 ran out of GPU memory at 128 x 16 (the production recipe already uses 99 of 144 GB), so 96 x 16 with its own control
            "w768b": dict(size=(768, 3072), ep=(96, 16)), "L12b": dict(layers=12, ep=(96, 16)), "ctl96": dict(ep=(96, 16))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--seed", type=int, default=30); ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs), a_.jobs
    base.FULL, base.EXT, base.VAR = True, None, "S7"; root_out, root_ck = (ROOT / "results/m161q_smoke", Path("/work/u2507115/m161q_ckpt_smoke")) if a_.smoke else (mq.OUT, mq.CK)
    if a_.smoke:
        base.EVAL_EVERY = 40
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); spec = VARIANTS[name]
                out = open(ROOT / f"logs/nano4/m161q_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                base.P_EP, base.M_EP = (32, 16) if a_.smoke else spec.get("ep", (128, 16)); base.MIN_GAMES = max(base.MIN_GAMES, base.M_EP); base.D, base.FFN = spec.get("size", (512, 2048)); base.LAYERS = spec.get("layers", 8); base.D_OUT = spec.get("d_out", base.D_OUT)
                kw = dict(gamma=spec.get("gamma", 1.0), centres=spec.get("centres", True), mask=spec.get("mask", 0.3)); steps = 80 if a_.smoke else spec.get("steps", a_.steps)
                if a_.smoke:
                    kw.update(warm_b=20, ramp_b=20)
                mq.OUT, mq.CK = root_out / name, root_ck / name; base.OUT = mq.OUT
                print(f"variant {name}: episode {base.P_EP} x {base.M_EP}, width {base.D} / feed-forward {base.FFN}, {base.LAYERS} layers, identity vector {base.D_OUT}, {steps} steps, {kw}", flush=True)
                t1 = time.time(); mq.train_one(-1, a_.seed, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = name; log(f"started variant {name} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); name = kids.pop(pid); bad += status != 0; log(f"variant {name} finished with status {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
