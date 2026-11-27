#!/usr/bin/env python
"""M260 step 1a (2026-11-20, report only): who is on the OPPONENT side of every training row? (design doc M260)

The all-players S7t networks (m161q, width 512, seeds 30 / 31, one orientation) embed the opponent side of every training row
(subject flag flipped); the row is scored against the 8,007 players' prototypes (owner masked, standardised per row, mean of the
seeds). Stored per row: owner, the three best candidates with their standardised scores, and the margin m15 of the best.
Template: m150_outsiders.py (its output results/m150/pilot.npz no longer exists).

  python -I logs/nano4/m260_rows.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

base.D, base.FFN = 512, 2048; CK, M161Q, OUT, SEEDS = Path("/work/u2507115/m161q_ckpt"), ROOT / "results/m161q", ROOT / "results/m260", (30, 31)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True); q = np.load(base.CACHE / "queries.npz"); players = q["players"].astype(str); pidx = {p: i for i, p in enumerate(players)}; C = len(players); dev = "cuda"
    base.FULL, base.EXT = True, None; G, _ = base.load_games(False, log); N = len(G["key"]) if not a_.limit else min(a_.limit, len(G["key"]))
    owner = np.array([pidx[p] for p in G["pid"][:N]]); tb = lambda x: torch.as_tensor(x, device=dev); E, PR = [], []  # noqa: E731
    log(f"{N} rows of {len(set(owner.tolist()))} owners loaded; candidates {C}")
    for s in SEEDS:
        st = torch.load(CK / f"ckpt_-1_s{s}.pt", map_location=dev, weights_only=False); assert "opt" not in st; net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); del st; e = torch.zeros((N, base.D_OUT), dtype=torch.float16, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, N, 1024):
                r = slice(i, min(i + 1024, N)); e[r] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb((1 - G["SUBJ"][r]).astype(np.int64))).to(torch.float16)
                if (i // 1024) % 200 == 0:
                    log(f"  seed {s}: rows {i} / {N}")
        pr = np.load(M161Q / f"out_-1_s{s}.npz")["protos"]; assert pr.shape[0] == C, pr.shape; E.append(e); PR.append(tb(pr).float()); del net; log(f"seed {s}: opponent side of {N} rows embedded")
    # check: the OWN side of the first rows must find the owner (the prototypes come from these games)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = torch.load(CK / f"ckpt_-1_s{SEEDS[0]}.pt", map_location=dev, weights_only=False); net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); del st; r = slice(0, min(4096, N))
        x = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float(); hit = float(((x @ PR[0].T).argmax(1).cpu().numpy() == owner[r]).mean()); del net
    log(f"CHECK: own side of the first {r.stop} rows -> the owner is the best candidate in {hit:.3f} (seed {SEEDS[0]})")
    own_t = tb(owner); top_i = np.zeros((N, 3), np.int32); top_z = np.zeros((N, 3), np.float32); m15 = np.zeros(N, np.float32); B = 4096
    with torch.no_grad():
        for i in range(0, N, B):
            r = slice(i, min(i + B, N)); z = 0.0
            for e, pr in zip(E, PR):
                sc = e[r].float() @ pr.T; sc[torch.arange(sc.shape[0], device=dev), own_t[r]] = float("nan"); mu = sc.nanmean(1, keepdim=True); sd = ((sc - mu) ** 2).nanmean(1, keepdim=True).sqrt().clamp(min=1e-9); z = z + ((sc - mu) / sd).nan_to_num(nan=-1e9) / len(E)
            top = z.topk(5, dim=1); top_i[r] = top.indices[:, :3].cpu().numpy(); top_z[r] = top.values[:, :3].cpu().numpy(); m15[r] = (top.values[:, 0] - top.values[:, 1:5].mean(1)).cpu().numpy()
    np.savez(OUT / "rows.npz", owner=owner.astype(np.int32), top_i=top_i, top_z=top_z.astype(np.float16), m15=m15.astype(np.float16), n_rows=N, hit_own=hit)
    log(f"margin m15 of the best candidate, 50 / 80 / 90 / 99 %: " + " / ".join(f"{float(np.quantile(m15, q_)):.2f}" for q_ in (0.5, 0.8, 0.9, 0.99)) + "; written results/m260/rows.npz")


if __name__ == "__main__":
    main()
