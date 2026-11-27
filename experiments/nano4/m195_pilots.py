#!/usr/bin/env python
"""Section 4M M195 pilots: anti-memorisation variants of the production recipe (128 x 16, width 512, 8 layers, by games,
learned centres, 40,000 steps), all-players network, seed 30 - read against the production network of the same seed at the
same steps (results/m161q/train_-1_s30.json; its mask is 0.3 on the engine features only, dropout 0.1).
  mp30  mask 0.3 hiding the engine features AND the point       mp60  the same with mask 0.6
  m60   mask 0.6 on the engine features only (rate control)      dr20  dropout 0.2, mask as in production
Outputs results/m161q/<variant>, checkpoints /work/u2507115/m161q_ckpt/<variant> (resumable).

  sbatch -p 8gpus --gres=gpu:4 -c 48 ... run.sbatch python -I logs/nano4/m195_pilots.py --jobs mp30 mp60 m60 dr20
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m195_prod as mq  # noqa: E402

base = mq.base
VARIANTS = {"mp30": dict(mask=0.3, mask_pos=1), "mp60": dict(mask=0.6, mask_pos=1), "m60": dict(mask=0.6), "dr20": dict(drop=0.2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--steps", type=int, default=40000); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs), a_.jobs
    base.FULL, base.EXT, base.VAR = True, None, "S7"; root_out, root_ck = mq.OUT, mq.CK
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), gpus
    jid = os.environ.get("SLURM_JOB_ID", "local"); G, Tt = base.load_games(False, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); spec = VARIANTS[name]
                out = open(ROOT / f"logs/nano4/m195_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                base.P_EP, base.M_EP = 128, 16; base.MIN_GAMES = max(base.MIN_GAMES, 16); base.D, base.FFN = 512, 2048; base.LAYERS = 8; base.DROP = spec.get("drop", base.DROP); mq.MASK_POS = spec.get("mask_pos", 0)
                mq.OUT, mq.CK = root_out / name, root_ck / name; base.OUT = mq.OUT
                print(f"variant {name}: {spec}; dropout {base.DROP}, mask {spec.get('mask', 0.3)}, mask hides the point: {bool(mq.MASK_POS)}", flush=True)
                t1 = time.time(); mq.train_one(-1, 30, G, Tt, a_.steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), gamma=1.0, centres=True, mask=spec.get("mask", 0.3))
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
