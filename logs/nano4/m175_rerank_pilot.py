#!/usr/bin/env python
"""Section 4M M175 pilot (2026-11-15, report only): re-scoring candidates with their individual game vectors instead of one
mean prototype (design doc M175).

One all-players network (a slim checkpoint with best_state and its out file): every non-held training game is embedded; for
the held-out own games and the test games (opponent side / target side) five scores against every candidate:
  S0 mean prototype (the current matcher)      S1 prototype of the candidate's games of the SAME colour as the query side
  S2 / S3 log-mean-exp of the per-game cosines at temperature 10 / 30      S4 the largest per-game cosine
Report: held-out top-1 (unweighted / by rows / by band) of each score and of S0 fused with each (weights fitted on the
held-out games, standardised scores); the label-free test-side reading (m15 of the opponent side against the target side);
the mean cosine between a player's Black and White prototypes.

  python -I logs/nano4/m175_rerank_pilot.py --ckpt /work/u2507115/m161_ckpt/p/ckpt_-1_s0.pt --out results/m161/p/out_-1_s0.npz --size 512 2048
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m88_seqid as base  # noqa: E402

BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, nargs=2, default=[384, 1536]); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--save", default=None, help="npz for the game vectors (E, owner, colour) for later use"); ap.add_argument("--json", default="m175_rerank_pilot.json"); ap.add_argument("--save-scores", default=None, help="prefix: <prefix>_<score>.npz with the standardised scores of the held-out (ZH) and the test (ZT) queries, for m161_fused_eval.py --addz"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True); dev = "cuda"; F = torch.nn.functional  # noqa: E731
    base.FULL, base.EXT = True, None; base.D, base.FFN = a_.size; base.LAYERS = a_.layers; G, Tt = base.load_games(False, log)
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); truth = q["held_truth"].astype(np.int64)
    held = set(json.load(open(base.M82 / "held_keys.json"))); owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    st = torch.load(a_.ckpt, map_location=dev, weights_only=False); net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def embed(src, rows, subj):
        out = torch.zeros(len(rows), base.D_OUT, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float()
        return out
    rows = np.flatnonzero(~is_held); E = embed(G, rows, G["SUBJ"][rows]); own_t = tb(owner[rows]); col_t = tb(G["SUBJ"][rows].astype(np.int64)); log(f"{len(rows)} training games embedded")
    cnt = torch.bincount(own_t, minlength=C).float(); P0 = F.normalize(torch.zeros(C, base.D_OUT, device=dev).index_add_(0, own_t, E), dim=-1); PC = []; NC = []
    for c in (0, 1):
        m = col_t == c; S = torch.zeros(C, base.D_OUT, device=dev).index_add_(0, own_t[m], E[m]); n_ = torch.bincount(own_t[m], minlength=C).float(); PC.append(torch.where((n_ >= 4)[:, None], F.normalize(S, dim=-1), P0)); NC.append(n_)
    both = (NC[0] >= 4) & (NC[1] >= 4); cbw = (PC[0][both] * PC[1][both]).sum(1); z = np.load(a_.out); chk = float((torch.as_tensor(z["protos"], device=dev) * P0).sum(1)[cnt > 0].mean())
    log(f"prototypes: cosine with the stored ones {chk:.4f}; players with >= 4 games of each colour {int(both.sum())}; cosine between a player's Black and White prototypes: mean {float(cbw.mean()):.3f}, quartiles " + " / ".join(f"{x:.3f}" for x in np.quantile(cbw.cpu().numpy(), [0.25, 0.5, 0.75])))
    if a_.save:
        np.savez(a_.save, E=E.cpu().numpy().astype(np.float16), owner=owner[rows], colour=G["SUBJ"][rows].astype(np.int8), key=G["key"][rows]); log(f"game vectors saved to {a_.save}")
    # ---------------------------------------------------------------- queries: held-out own games and the test games
    pos = {int(g): i for i, g in enumerate(z["qidx"])}; hq = np.arange(2 * Q, 2 * Q + H); tq = np.arange(2 * Q + H, 2 * Q + H + 2 * T); qv = torch.as_tensor(z["qvec"].astype(np.float32), device=dev)
    hk = np.array([gidx[k] for k in q["qkey"][hq]]); hcol = G["SUBJ"][hk].astype(np.int64); trow = q["qtest"][tq].astype(np.int64); tcol = np.where(q["side"][tq] == 1, 1 - Tt["SUBJ"][trow], Tt["SUBJ"][trow]).astype(np.int64)
    ids = np.concatenate([hq, tq]); qcol = torch.as_tensor(np.concatenate([hcol, tcol]), device=dev); Qm = qv[[pos[int(g)] for g in ids]]; N = len(ids); names = ("S0 mean", "S1 same colour", "S2 lme 10", "S3 lme 30", "S4 max"); S = {k: np.zeros((N, C), np.float32) for k in names}; Et = E.T.contiguous(); idx_o = own_t[None]
    for i in range(0, N, 128):
        qb = Qm[i:i + 128]; B = len(qb); S["S0 mean"][i:i + B] = (qb @ P0.T).cpu().numpy(); c_ = qcol[i:i + B]; S["S1 same colour"][i:i + B] = torch.where((c_ == 0)[:, None], qb @ PC[0].T, qb @ PC[1].T).cpu().numpy()
        sim = qb @ Et; io = idx_o.expand(B, -1)
        for nm, tau in (("S2 lme 10", 10.0), ("S3 lme 30", 30.0)):
            acc = torch.zeros(B, C, device=dev).scatter_add_(1, io, torch.exp(tau * (sim - 1.0))); S[nm][i:i + B] = (torch.log(acc / cnt.clamp(min=1)[None] + 1e-30) / tau + 1.0).cpu().numpy()
        S["S4 max"][i:i + B] = torch.full((B, C), -2.0, device=dev).scatter_reduce_(1, io, sim, reduce="amax", include_self=True).cpu().numpy()
    has = (cnt > 0).cpu().numpy(); log("scores computed")
    for k in names:
        S[k][:, ~has] = S[k][:, has].min() - 1.0
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(players.astype(str)).to_numpy(); band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]]); share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum(); wb = share[band] / np.bincount(band, minlength=len(BANDS))[band]; wb /= wb.sum()
    Z = {k: ip.standardise(v).astype(np.float32) for k, v in S.items()}; rep = {}
    if a_.save_scores:
        for k, nm in (("S0 mean", "mean"), ("S4 max", "max"), ("S3 lme 30", "lme30")):
            np.savez(f"{a_.save_scores}_{nm}.npz", ZH=Z[k][:H].astype(np.float16), ZT=Z[k][H:].astype(np.float16))
        log(f"standardised scores saved with prefix {a_.save_scores}")

    def read(name, Fm):
        Fh, Ft = Fm[:H], Fm[H:]; r = (Fh > Fh[np.arange(H), truth][:, None]).sum(1); t1 = [float((r[band == k] == 0).mean()) for k in range(len(BANDS))]
        top = -np.partition(-Ft, 5, axis=1)[:, :5]; top.sort(axis=1); m15 = top[:, 4] - top[:, :4].mean(1); opp, tgt = m15[:T], m15[T:]; grid = np.quantile(tgt, np.linspace(0.5, 0.999, 500)); sup = float(max((opp > t).mean() - (tgt > t).mean() for t in grid)); ex = [float((opp > np.quantile(tgt, qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)]
        rep[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "top1_rows": float((wb * (r == 0)).sum()), "by_band": t1, "test_excess": ex, "test_sup": sup}
        log(f"{name:28s} held-out top-1 / top-5 / top-20 {rep[name]['top1']:.3f} / {rep[name]['top5']:.3f} / {rep[name]['top20']:.3f}; by rows {rep[name]['top1_rows']:.3f}; by band " + " ".join(f"{x:.3f}" for x in t1) + " | test excess q90 / q98 / q99.5 " + " / ".join(f"{x:+.3f}" for x in ex) + f", largest {sup:.3f}")
    for k in names:
        read(k, Z[k])
    for k in names[1:]:
        w = ip.fit_weights([Z["S0 mean"][:H], Z[k][:H]], truth); read(f"S0 + {k} (weights {w[0]:.2f}, {w[1]:.2f})", w[0] * Z["S0 mean"] + w[1] * Z[k])
    w = ip.fit_weights([Z["S0 mean"][:H], Z["S1 same colour"][:H], Z["S3 lme 30"][:H]], truth); read(f"S0 + S1 + S3 (weights {w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})", w[0] * Z["S0 mean"] + w[1] * Z["S1 same colour"] + w[2] * Z["S3 lme 30"])
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1)); log(f"written results/r_variants/{a_.json}")


if __name__ == "__main__":
    main()
