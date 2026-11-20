#!/usr/bin/env python
"""Section 4M M192 step 1 (2026-11-16): per-game vectors of the all-players networks of the second production set, for the
diagnosis of what the recogniser cannot separate (m192_where.py).

For seeds 30 and 31: the own-side vector of every training game the network used (not held out), the own-side vector of the
held-out own games (3,244, the owner known and absent from the prototypes), and the same held-out games cut to their first
40 / 80 / 120 / 160 plies. float16, /work/u2507115/m192/emb_s<seed>.npz; meta.npz once (owner index, subject colour, length,
game key, held flag of every stored game).

  sbatch -p 8gpus --gres=gpu:1 -c 12 ... run.sbatch python -I logs/nano4/m192_embed.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

CK, OUT = Path("/work/u2507115/m161q_ckpt"), Path("/work/u2507115/m192"); CUTS = (40, 80, 120, 160)


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT = True, None; base.D, base.FFN = 512, 2048; OUT.mkdir(parents=True, exist_ok=True); G, _ = base.load_games(False, log); dev = "cuda"
    q = np.load(base.CACHE / "queries.npz"); pidx = {p: i for i, p in enumerate(q["players"])}; held = set(json.load(open(base.M82 / "held_keys.json"))); owner = np.array([pidx[p] for p in G["pid"]], np.int32); is_held = np.array([k in held for k in G["key"]])
    np.savez(OUT / "meta.npz", owner=owner, subj=G["SUBJ"].astype(np.int8), length=G["LEN"].astype(np.int16), key=G["key"].astype("S12"), held=is_held, fold=G["fold"].astype(np.int8)); log(f"meta written: {len(owner)} games, {int(is_held.sum())} held out")
    tb = lambda x: torch.as_tensor(x, device=dev); use = np.flatnonzero(~is_held); hr = np.flatnonzero(is_held)  # noqa: E731

    def embed(net, rows, cut=None):
        out = np.zeros((len(rows), base.D_OUT), np.float16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; ln = G["LEN"][r].astype(np.int64); ln = np.minimum(ln, cut) if cut else ln
                out[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(ln), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy().astype(np.float16)
        return out
    for seed in (30, 31):
        st = torch.load(CK / f"ckpt_-1_s{seed}.pt", map_location=dev, weights_only=False); net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval()
        E = embed(net, use); Eh = embed(net, hr); cuts = {f"held_cut{c}": embed(net, hr, c) for c in CUTS}
        ref = np.load(ROOT / f"results/m161q/out_-1_s{seed}.npz")["protos"]; S = np.zeros((len(pidx), base.D_OUT)); np.add.at(S, owner[use], E.astype(np.float64)); S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12); has = np.bincount(owner[use], minlength=len(pidx)) > 0
        np.savez(OUT / f"emb_s{seed}.npz", own=E, held=Eh, **cuts); log(f"seed {seed}: {len(use)} own-side vectors, {len(hr)} held-out; prototypes against the stored ones: mean cosine {float((S * ref).sum(1)[has].mean()):.5f}")


if __name__ == "__main__":
    main()
