#!/usr/bin/env python
"""Section 4M M191 pilot: the production recipe of the second set (m161q_prod.py: by games, 128 x 16, width 512, 8 layers,
mask 0.3, learned centres, 40,000 steps) with the geometry bias of m191_relbias.py; all-players network, seed 30 - to be read
against the production network of the same seed (results/m161q/train_-1_s30.json) at the same steps.
Outputs results/m161q/relb, checkpoints /work/u2507115/m161q_ckpt/relb (resumable).

  sbatch -p 8gpus --gres=gpu:1 -c 12 ... run.sbatch python -I logs/nano4/m191_pilot.py
"""
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161q_prod as mq  # noqa: E402
import m191_relbias as rb  # noqa: E402

base = mq.base


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    base.FULL, base.EXT, base.VAR = True, None, "S7"; base.P_EP, base.M_EP = 128, 16; base.MIN_GAMES = max(base.MIN_GAMES, 16); base.D, base.FFN = 512, 2048; base.LAYERS = 8
    mq.OUT, mq.CK = mq.OUT / "relb", mq.CK / "relb"; base.OUT = mq.OUT; base.build_model = rb.build
    G, Tt = base.load_games(False, log); log("variant relb: production recipe + board-geometry attention bias, all-players network, seed 30")
    mq.train_one(-1, 30, G, Tt, steps, log, gamma=1.0, centres=True, mask=0.3)


if __name__ == "__main__":
    main()
