#!/usr/bin/env python
"""Check of the regenerated results/trunk_vec_x80 (2026-11-15; design doc M176, "我的失誤與補救").

(1) file inventory of the three stored-game directories and the keys of the regenerated one against results/data/x80_keys.csv
    (every key exactly once);
(2) a real load (m88_seqid.load_games): the number of training games must be 1,000,000 again, and the order of the keys must
    equal the order recorded before the deletion (results/m150/pilot.npz["key"], written from the loader on 2026-11-15 05:41);
(3) end to end: the pilot network p (trained and prototyped on the ORIGINAL data, results/m161/p/out_-1_s0.npz) re-embeds all
    training games from the regenerated files; the cosine between the recomputed and the stored prototypes, for all players and
    for the players who own games in the regenerated directory, and the largest difference of the stored query vectors.

  python -I logs/nano4/x80_regen_check.py        (one GPU, about 8 minutes)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    want = pd.read_csv(ROOT / "results/data/x80_keys.csv", dtype=str); d = ROOT / "results/trunk_vec_x80"; got = []
    for f in range(5):
        fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv")); nz = sorted((d / f"fold{f}").glob("moves_s*_p*.npz"))
        for p in fs:
            got.append(pd.read_csv(p, usecols=["game_key"], dtype=str))
        log(f"fold {f}: {len(fs)} csv parts, {len(nz)} npz parts, other files {sorted(x.name for x in (d / f'fold{f}').iterdir() if x not in fs and x not in nz)}")
    got = pd.concat(got).game_key; ok1 = (len(got) == len(want)) and (got.nunique() == len(got)) and (set(got) == set(want.game_key))
    log(f"(1) regenerated games {len(got)} (unique {got.nunique()}), expected {len(want)}; same set of keys: {set(got) == set(want.game_key)} -> {'OK' if ok1 else 'FAILED'}")
    base.FULL, base.EXT = True, None; base.D, base.FFN = 512, 2048; G, Tt = base.load_games(False, log)
    ref = np.load(ROOT / "results/m150/pilot.npz", allow_pickle=True)["key"].astype(str); same_order = len(G["key"]) == len(ref) and bool((G["key"].astype(str) == ref).all())
    log(f"(2) training games loaded {len(G['key'])} (expected 1000000); key order equals the order before the deletion: {same_order} -> {'OK' if len(G['key']) == 1000000 and same_order else 'CHECK'}")
    dev = "cuda"; q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); st = torch.load("/work/u2507115/m161_ckpt/p/ckpt_-1_s0.pt", map_location=dev, weights_only=False)
    net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); tb = lambda x: torch.as_tensor(x, device=dev); rows = np.flatnonzero(~is_held); E = torch.zeros(len(rows), base.D_OUT, device=dev)  # noqa: E731
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), 1024):
            r = rows[i:i + 1024]; E[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float()
    P = torch.nn.functional.normalize(torch.zeros(C, base.D_OUT, device=dev).index_add_(0, tb(owner[rows]), E), dim=-1).cpu().numpy(); z = np.load(ROOT / "results/m161/p/out_-1_s0.npz"); cos = (P * z["protos"]).sum(1); has = np.bincount(owner[rows], minlength=C) > 0
    xk = set(want.game_key); inx = np.array([k in xk for k in G["key"]]); px = np.bincount(owner[inx], minlength=C) > 0
    log(f"(3) prototypes recomputed from the regenerated files against the stored ones (network p): all {int(has.sum())} players mean cosine {cos[has].mean():.6f}, minimum {cos[has].min():.6f}; the {int((px & has).sum())} players owning regenerated games: mean {cos[px & has].mean():.6f}, minimum {cos[px & has].min():.6f}; players without such games: minimum {cos[has & ~px].min():.6f}")
    # per-game check on the regenerated games themselves: their vectors must be as close to their owner's stored prototype as the other games are
    Ex = E.cpu().numpy(); own_r = owner[rows]; sim = (Ex * z["protos"][own_r]).sum(1); mx = inx[rows]
    log(f"    cosine of a game to its owner's stored prototype: regenerated games mean {sim[mx].mean():.4f} (sd {sim[mx].std():.4f}), other games of the same players mean {sim[~mx & px[own_r]].mean():.4f} (sd {sim[~mx & px[own_r]].std():.4f})")
    ok3 = cos[has].min() > 0.999
    print(f"RESULT: inventory {'OK' if ok1 else 'FAILED'}; load {'OK' if len(G['key']) == 1000000 else 'FAILED'} (order {'same' if same_order else 'different'}); prototypes {'OK' if ok3 else 'CHECK'}")


if __name__ == "__main__":
    main()
