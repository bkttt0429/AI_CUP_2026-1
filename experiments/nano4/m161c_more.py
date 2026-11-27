#!/usr/bin/env python
"""Section 4M M161c (2026-11-15): add-on ingredients for the by-games identity trainer (design doc M161c).

All-players network, seed 0, 12,000 steps, players drawn by their games (M161's variant a) plus one change each:
  j  128 players x 16 games per episode (players with >= 16 games)      k  j + the all-player prototype loss (M161b)
  n  a wider network (384 -> 512, feed-forward 1,536 -> 2,048)          o  ply mask 0.3 (the production setting)
  p  j + n      q  players drawn by games ** 0.75      r  n + the all-player prototype loss      s  a with seed 1
The trainer is m161p_prod.train_one; every variant writes to its own folder results/m161/<variant>/ (out_-1_s0.npz as seed 0
shares the pilot's random streams) and /work/u2507115/m161_ckpt/<variant>/.

  sbatch -p 8gpus --gres=gpu:4 -c 48 ... run.sbatch python -I logs/nano4/m161c_more.py --jobs j k n o --steps 12000
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161p_prod as mp  # noqa: E402

base = mp.base
VARIANTS = {"j": dict(ep=(128, 16), kw=dict(gamma=1.0, bank_kind="none", mask=0.0)), "k": dict(ep=(128, 16), kw=dict(gamma=1.0, bank_kind="proto", mask=0.0)),
            "n": dict(size=(512, 2048), kw=dict(gamma=1.0, bank_kind="none", mask=0.0)), "o": dict(kw=dict(gamma=1.0, bank_kind="none", mask=0.3)),
            "p": dict(ep=(128, 16), size=(512, 2048), kw=dict(gamma=1.0, bank_kind="none", mask=0.0)),  # j + n
            "q": dict(kw=dict(gamma=0.75, bank_kind="none", mask=0.0)),  # between a (1.0) and b (0.5)
            "r": dict(size=(512, 2048), kw=dict(gamma=1.0, bank_kind="proto", mask=0.0)),  # n + the all-player prototype loss
            "s": dict(seed=1, kw=dict(gamma=1.0, bank_kind="none", mask=0.0))}  # a with another seed (how much of a's gain is the seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs), a_.jobs
    base.FULL, base.EXT, base.VAR = True, None, "S6"; root_out, root_ck = (ROOT / "results/m161_smoke", Path("/work/u2507115/m161_ckpt_smoke")) if a_.smoke else (ROOT / "results/m161", Path("/work/u2507115/m161_ckpt"))
    if a_.smoke:
        base.EVAL_EVERY = 40
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps; G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); spec = VARIANTS[name]; kw = dict(spec["kw"])
                out = open(ROOT / f"logs/nano4/m161_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                if "ep" in spec:
                    base.P_EP, base.M_EP = spec["ep"]; base.MIN_GAMES = max(base.MIN_GAMES, spec["ep"][1])
                if "size" in spec:
                    base.D, base.FFN = spec["size"]
                if a_.smoke:
                    kw.update(warm_b=20, ramp_b=20, refresh=20)
                mp.OUT, mp.CK = root_out / name, root_ck / name; print(f"variant {name}: episode {base.P_EP} x {base.M_EP}, players with >= {base.MIN_GAMES} games, width {base.D} / feed-forward {base.FFN}, {kw}", flush=True)
                t1 = time.time(); mp.train_one(-1, int(spec.get("seed", 0)), G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
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
