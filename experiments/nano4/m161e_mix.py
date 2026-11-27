#!/usr/bin/env python
"""Section 4M M161d / M161e (2026-11-15): one node, eight trainers of two kinds (design doc M161d, M161e).

  t1, t0   m161d_fresh.train_one: prototypes from the half of the games the network did not train on
  the rest m161p_prod.train_one (as m161c_more.py): seed replicates of the by-games recipe and its control, and three more
           ingredients: u = control (uniform players) seed 1, a2 = by games seed 2, u2 = control seed 2,
           j2 = 64 players x 32 games, g2 = players drawn by games ** 1.25, n2 = 12 layers instead of 8
Outputs: results/m161/m161d_<t>.json; results/m161/<variant>/out_-1_s<seed>.npz.

  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch python -I logs/nano4/m161e_mix.py --jobs t1 t0 u a2 u2 j2 g2 n2 --steps 12000
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161d_fresh as md  # noqa: E402
import m161p_prod as mp  # noqa: E402

base = mp.base
VARIANTS = {"u": dict(seed=1, kw=dict(gamma=0.0, bank_kind="none", mask=0.0)), "a2": dict(seed=2, kw=dict(gamma=1.0, bank_kind="none", mask=0.0)), "u2": dict(seed=2, kw=dict(gamma=0.0, bank_kind="none", mask=0.0)),
            "j2": dict(ep=(64, 32), kw=dict(gamma=1.0, bank_kind="none", mask=0.0)), "g2": dict(kw=dict(gamma=1.25, bank_kind="none", mask=0.0)), "n2": dict(layers=12, kw=dict(gamma=1.0, bank_kind="none", mask=0.0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS or x in md.VARIANTS for x in a_.jobs), a_.jobs
    base.FULL, base.EXT, base.VAR = True, None, "S6"; root_out, root_ck = (ROOT / "results/m161_smoke", Path("/work/u2507115/m161_ckpt_smoke")) if a_.smoke else (ROOT / "results/m161", Path("/work/u2507115/m161_ckpt"))
    if a_.smoke:
        base.EVAL_EVERY = 40; md.OUT = root_out
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps; G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); out = open(ROOT / f"logs/nano4/m161_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out; t1 = time.time(); lg = lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True)  # noqa: E731
                if name in md.VARIANTS:
                    md.train_one(name, G, steps, lg)
                else:
                    spec = VARIANTS[name]; kw = dict(spec["kw"])
                    if "ep" in spec:
                        base.P_EP, base.M_EP = spec["ep"]; base.MIN_GAMES = max(base.MIN_GAMES, spec["ep"][1])
                    if "layers" in spec:
                        base.LAYERS = spec["layers"]
                    if a_.smoke:
                        kw.update(warm_b=20, ramp_b=20, refresh=20)
                    mp.OUT, mp.CK = root_out / name, root_ck / name; print(f"variant {name}: episode {base.P_EP} x {base.M_EP}, players with >= {base.MIN_GAMES} games, {base.LAYERS} layers, seed {spec.get('seed', 0)}, {kw}", flush=True)
                    mp.train_one(-1, int(spec.get("seed", 0)), G, Tt, steps, lg, **kw)
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
