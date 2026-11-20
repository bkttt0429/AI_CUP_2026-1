#!/usr/bin/env python
"""Section 4M M189 (2026-11-16): give every candidate the games in which it appears as the OPPONENT.

A candidate's prototype is built from the games filed under its own name. About three quarters of the opponents of the
training games are themselves training players (M187), and a game is filed under one owner only - so a candidate has
roughly as many further games in other players' files, where it is the opponent. They are the same kind of object as the
query of a test game (the candidate's moves in a game owned by somebody else). No retraining: for every network of the
second production set (m161q_prod.py; fold networks 1-4 and the all-players network, seeds 30 and 31)

  1. embed both sides of every training game the network may use (as in its training: not held out, owner outside the
     network's fold): own side -> the prototypes P0 (must reproduce the stored ones), opponent side -> the queries;
  2. score the opponent side against P0 (owner excluded; standardised over the candidates; + beta log games), statistic m15;
     the null is the OWN side of the same games against the other players' prototypes (never the right person);
  3. the opponent sides above the null's q99 are added to the prototype of their first candidate: P1 = normalise(sum of the
     own-side vectors + sum of the adopted opponent-side vectors).

Writes results/m161q/asopp_<net>_s<seed>.npz (protos, adopted counts, thresholds) and, with --assemble, the variant file
results/m84_cache/var_S7u.npz (the queries of var_S7t, the new prototypes).

  sbatch -p 8gpus --gres=gpu:5 -c 60 ... run.sbatch python -I logs/nano4/m189_asopp.py --seeds 30 31
  python -I logs/nano4/m189_asopp.py --assemble --seeds 30 31
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

OUT, CK = ROOT / "results/m161q", Path("/work/u2507115/m161q_ckpt"); BETA = 0.9


def one(v, seed, G, log):
    import torch
    dev = "cuda"; tag = f"{v}_s{seed}"; st = torch.load(CK / f"ckpt_{tag}.pt", map_location=dev, weights_only=False); assert "opt" not in st, tag; net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval()
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); rows = np.flatnonzero(~is_held & (G["fold"] != v)); own = owner[rows]; tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def embed(subj):
        out = torch.zeros(len(rows), base.D_OUT, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float()
        return out
    E_own = embed(G["SUBJ"][rows]); E_opp = embed(1 - G["SUBJ"][rows]); own_t = tb(own); S = torch.zeros(C, base.D_OUT, device=dev).index_add_(0, own_t, E_own); cnt = torch.bincount(own_t, minlength=C).float(); has = cnt > 0
    P0 = torch.nn.functional.normalize(S, dim=-1); ref = np.load(OUT / f"out_{tag}.npz")["protos"]; cos = (P0.cpu().numpy() * ref).sum(1)[has.cpu().numpy()]; log(f"net {tag}: {len(rows)} games, {int(has.sum())} candidates; prototypes against the stored ones: mean cosine {cos.mean():.5f}, min {cos.min():.5f}")
    prior = BETA * torch.log(cnt.clamp(min=1.0))

    def score(E):
        stat = torch.zeros(len(rows), device=dev); first = torch.zeros(len(rows), dtype=torch.long, device=dev)
        for i in range(0, len(rows), 8192):
            Z = E[i:i + 8192] @ P0.T; ok = has[None].expand(Z.shape[0], -1).clone(); ok[torch.arange(Z.shape[0], device=dev), own_t[i:i + 8192]] = False; n_ = ok.sum(1, keepdim=True); mu = (Z * ok).sum(1, keepdim=True) / n_; sd = (((Z - mu) ** 2 * ok).sum(1, keepdim=True) / n_).sqrt().clamp(min=1e-12)
            F = ((Z - mu) / sd + prior[None]).masked_fill(~ok, -float("inf")); top = F.topk(5, dim=1); stat[i:i + 8192] = top.values[:, 0] - top.values[:, 1:5].mean(1); first[i:i + 8192] = top.indices[:, 0]
        return stat, first
    s_opp, f_opp = score(E_opp); s_null, _ = score(E_own); res = {}
    for qq in (0.90, 0.95, 0.99, 0.999):
        thr = float(torch.quantile(s_null[torch.randperm(len(rows), device=dev)[:200000]], qq)); res[str(qq)] = {"thr": thr, "share": float((s_opp > thr).float().mean())}
    thr = res["0.99"]["thr"]; m = s_opp > thr; A = torch.zeros(C, base.D_OUT, device=dev).index_add_(0, f_opp[m], E_opp[m]); add = torch.bincount(f_opp[m], minlength=C).float(); P1 = torch.nn.functional.normalize(S + A, dim=-1)
    log(f"net {tag}: opponent sides above the null's q90 / q95 / q99 / q99.9: " + " / ".join(f"{res[k]['share']:.3f}" for k in ("0.9", "0.95", "0.99", "0.999")) + f"; adopted (above q99) {int(m.sum())} = {float(m.float().mean()):.3f} of the games; candidates that gain games {int((add > 0).sum())}; games per candidate before / added: median {float(cnt[has].median()):.0f} / {float(add[has].median()):.0f}, candidates with 30-100 own games: mean added {float(add[(cnt >= 30) & (cnt < 100)].mean()):.1f}")
    np.savez(OUT / f"asopp_{tag}.npz", protos=P1.cpu().numpy().astype(np.float32), protos0=P0.cpu().numpy().astype(np.float32), added=add.cpu().numpy(), own=cnt.cpu().numpy(), thresholds=json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--seeds", type=int, nargs="+", default=[30, 31]); ap.add_argument("--assemble", action="store_true"); ap.add_argument("--name", default="S7u"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT = True, None; base.D, base.FFN = 512, 2048
    if a_.assemble:
        V = np.load(base.CACHE / "var_S7t.npz"); P = np.zeros_like(V["protos"]); assert P.shape[0] == len(a_.seeds)
        for si, s in enumerate(a_.seeds):
            for j, v in enumerate(base.NETS):
                z = np.load(OUT / f"asopp_{v}_s{s}.npz"); P[si, j] = z["protos"]; print(f"seed {s} net {v}: cosine of the new prototypes to the old ones, mean {float((z['protos'] * z['protos0']).sum(1)[z['own'] > 0].mean()):.4f}; added games {int(z['added'].sum())}")
        assert not (base.CACHE / f"var_{a_.name}.npz").exists(), "variant file exists already"; np.savez(base.CACHE / f"var_{a_.name}.npz", protos=P, qvec=V["qvec"]); print(f"var_{a_.name}.npz written"); return
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(base.NETS), gpus; G, _ = base.load_games(False, log); kids = {}
    for k, v in enumerate(base.NETS):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                for s in a_.seeds:
                    one(v, s, G, log)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = v
    bad = 0
    while kids:
        pid, status = os.wait(); v = kids.pop(pid); bad += status != 0; log(f"net {v} done, status {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
