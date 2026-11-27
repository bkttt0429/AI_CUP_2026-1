#!/usr/bin/env python
"""Section 4M M221 (2026-11-18): the two input / training changes of M218 and M219 in ONE network - the production recipe
with external accounts as strangers in every episode (m218_prod.py, share 0.25, no outlier term) AND the 9 x 9 window input
(pid_win.wrap, as m219_prod.py). A thin wrapper: m218_prod.main() with the model builder replaced (the production network
is built first, the encoder is added afterwards and the CPU random state restored, so the initial values of every other
parameter and the episode streams are those of m218_prod.py / the production run of the same seed).
Outputs: results/m221/<arm> (train_<net>_s<seed>.json); per-network logs keep m218_prod's pattern logs/nano4/m218_<arm>_net*.log.

  python -I logs/nano4/m221_prod.py --out wg25 --jobs 1:30 2:30 3:30 4:30 all:30 --ext-share 0.25 --oe 0 --ck-dir /dev/shm/m221/wg25 --no-periodic
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m216_prod as m216  # noqa: E402
import m218_prod as m218  # noqa: E402

WIN = {"size": 9, "ch": 64}
_plain = m216.build_model


def build_model():
    import torch
    import pid_win
    net = _plain(); st = torch.get_rng_state(); pid_win.wrap(net, dict(WIN)); torch.set_rng_state(st)
    assert float(net.win.out.weight.abs().sum()) == 0.0 and float(net.win.out.bias.abs().sum()) == 0.0  # the new term starts at zero
    return net


if __name__ == "__main__":
    m216.build_model = build_model; m218.OUT, m218.CK = ROOT / "results/m221", Path("/work/u2507115/m221_ckpt")
    print(f"M221: m218_prod with the window input {WIN}", flush=True); m218.main()
