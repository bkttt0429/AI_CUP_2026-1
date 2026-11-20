#!/usr/bin/env python
"""Section 4M M161 (2026-11-15): what a pilot network adds to the FUSED identifier, read without level labels (design doc M161).

A pilot trains the all-players network only, so the validation side (cross-fit networks) is not available; the held-out own
games and the 10,000 test games are. For the base identifier (block + --base sources, all-players members) and for the base
plus each --add network (and each --combo of them):
  fusion weights fitted on the held-out own games (ident_pool.fit_weights, as in production);
  held-out top-1: unweighted / weighted by rows (bands) / weighted by n_c; the same with the activity prior (beta fitted by the
  n_c-weighted likelihood, M163);
  test games: statistic m15 (first minus the mean of the 2nd-5th fused score) of the opponent side against the target side
  (a test player is never a candidate: every target-side match is an impostor) - excess above the target side's q90 / q98 /
  q99.5 and the largest excess over all thresholds = a lower bound of the share of test opponents that are recognised.

  python -I logs/nano4/m161_fused_eval.py --base P,S2y,S4ft,S4Lt --add a=results/m161/out_-1_a.npz --add p=results/m161/p/out_-1_s0.npz \
      --add h2c=results/m161/h2/out_-1_s0.npz:centres --combo a+p
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
from m164_ident import fit_w  # noqa: E402  (ident_pool.fit_weights with a damped refit when Newton diverges)

CACHE, RV = ROOT / "results/m84_cache", ROOT / "results/r_variants"
BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]


def std(S, has=None):
    S = S.astype(np.float32)
    if has is not None:
        out = np.zeros_like(S); out[:, has] = std(S[:, has]); return out
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); sd[sd < 1e-12] = 1.0
    return (S - mu) / sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--base", default="P,S2y,S4ft,S4Lt"); ap.add_argument("--add", action="append", default=[], help="NAME=out file[:key of the matching vectors, default protos]")
    ap.add_argument("--combo", action="append", default=[], help="NAME+NAME[+...]: several added networks together"); ap.add_argument("--group", action="append", default=[], help="NAME=a,b,c: added networks averaged into ONE source (as the seeds of a production set)"); ap.add_argument("--addz", action="append", default=[], help="NAME=npz with ZH [H, C] and ZT [2T, C]: a source given as standardised scores (m175_rerank_pilot.py --save-scores)"); ap.add_argument("--json", default="m161_fused_eval.json"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    q = np.load(CACHE / "queries.npz"); players = q["players"].astype(str); C = len(players); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); truth = q["held_truth"].astype(np.int64)
    hq = np.arange(2 * Q, 2 * Q + H); tq = np.arange(2 * Q + H, 2 * Q + H + 2 * T); assert (q["side"][tq[:T]] == 1).all() and (q["side"][tq[T:]] == 0).all() and (q["excl"][tq] < 0).all() and (q["excl"][hq] < 0).all()
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(players).to_numpy(); logn = np.log(ng.astype(np.float32))
    band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]]); share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum()
    wb = share[band] / np.bincount(band, minlength=len(BANDS))[band]; wb /= wb.sum(); wa = ng[truth].astype(np.float64); wa /= wa.sum()
    # ---------------------------------------------------------------- sources: standardised scores on the held-out and the test queries
    ZH, ZT = {}, {}
    ZH["block"] = np.load(CACHE / "block_held.npy").astype(np.float32); parts = []
    for k in range(4):
        idx = np.load(CACHE / f"idx_{k}of4.npy"); m = np.flatnonzero(idx >= 2 * Q + H)
        if len(m):
            parts.append((idx[m], np.load(CACHE / f"block_{k}of4.npy", mmap_mode="r")[m]))
    ti = np.concatenate([p[0] for p in parts]); tb_ = np.concatenate([np.asarray(p[1], dtype=np.float32) for p in parts]); o = np.argsort(ti); assert np.array_equal(ti[o], tq); ZT["block"] = tb_[o]; del tb_, parts; log("block scores loaded")
    base = a_.base.split(",")
    for v in base:
        V = np.load(CACHE / f"var_{v}.npz"); M = V["protos"].shape[0]; zh = np.zeros((H, C), np.float32); zt = np.zeros((2 * T, C), np.float32)
        for m in range(M):
            P = V["protos"][m, 4].astype(np.float32).T; zh += std(V["qvec"][m, hq].astype(np.float32) @ P); zt += std(V["qvec"][m, tq].astype(np.float32) @ P)
        ZH[v], ZT[v] = zh / M, zt / M; log(f"source {v}: {M} member(s)")
    adds = []
    for spec in a_.add:
        nm, rest = spec.split("=", 1); path, _, key = rest.partition(":"); key = key or "protos"
        if not Path(path).exists():
            log(f"source {nm}: {path} missing, skipped"); continue
        z = np.load(path); pos = {int(g): i for i, g in enumerate(z["qidx"])}; P = z[key].astype(np.float32); P = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)).T; qv = z["qvec"].astype(np.float32)
        has = z["has_centre"] if (key == "centres" and "has_centre" in z.files) else None  # a centre source: standardised over the players with a centre, 0 for the others
        ZH[nm] = std(qv[[pos[int(g)] for g in hq]] @ P, has); ZT[nm] = std(qv[[pos[int(g)] for g in tq]] @ P, has); adds.append(nm); log(f"source {nm}: {path} [{key}]" + ("" if has is None else f", {int(has.sum())} players with a centre"))
    for spec in a_.addz:
        nm, path = spec.split("=", 1); zz = np.load(path); ZH[nm] = zz["ZH"].astype(np.float32); ZT[nm] = zz["ZT"].astype(np.float32); adds.append(nm); log(f"source {nm}: scores from {path}")
    for spec in a_.group:
        nm, members = spec.split("=", 1); members = [m for m in members.split(",") if m in ZH]
        if members:
            ZH[nm] = sum(ZH[m] for m in members) / len(members); ZT[nm] = sum(ZT[m] for m in members) / len(members); adds.append(nm); log(f"source {nm}: mean of {'+'.join(members)}")
    rnk = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1)  # noqa: E731

    def m15(F):
        top = -np.partition(-F, 5, axis=1)[:, :5]; top.sort(axis=1); top = top[:, ::-1]
        return top[:, 0] - top[:, 1:5].mean(1)

    def test_side(F):
        s = m15(F); opp, tgt = s[:T], s[T:]; ex = [float((opp > np.quantile(tgt, qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)]; grid = np.quantile(tgt, np.linspace(0.5, 0.999, 500))
        return ex, float(max((opp > t).mean() - (tgt > t).mean() for t in grid))

    def config(name, names):
        w = fit_w([ZH[k] for k in names], truth, log); Fh = sum(float(w[i]) * ZH[k] for i, k in enumerate(names)); Ft = sum(float(w[i]) * ZT[k] for i, k in enumerate(names)); out = {"sources": names, "weights": w.tolist()}
        import torch
        Fh_t = torch.as_tensor(Fh); ln = torch.as_tensor(logn); wt = torch.as_tensor(wa, dtype=torch.float32); ar = torch.arange(H); tr = torch.as_tensor(truth); best = (-1e18, 0.0)
        for beta in np.arange(0.0, 2.01, 0.05):
            ll = float((wt * torch.log_softmax(Fh_t + float(beta) * ln[None], dim=1)[ar, tr]).sum())
            if ll > best[0]:
                best = (ll, float(beta))
        for tag, beta in (("plain", 0.0), ("prior", best[1])):
            r = rnk(Fh + beta * logn[None]); ex, sup = test_side(Ft + beta * logn[None])
            out[tag] = {"beta": beta, "top1": float((r == 0).mean()), "top1_rows": float((wb * (r == 0)).sum()), "top1_nc": float((wa * (r == 0)).sum()), "by_band": [float((r[band == k] == 0).mean()) for k in range(len(BANDS))], "test_excess": ex, "test_sup": sup}
        a, b = out["plain"], out["prior"]
        log(f"{name:28s} | weights " + " ".join(f"{x:.2f}" for x in w) + f" | held-out top-1 {a['top1']:.3f}, by rows {a['top1_rows']:.3f}, by n_c {a['top1_nc']:.3f} | test excess q90 / q98 / q99.5 " + " / ".join(f"{x:+.3f}" for x in a["test_excess"]) + f", largest {a['test_sup']:.3f}"
            + f" || with the prior (beta {b['beta']:.2f}): {b['top1']:.3f}, {b['top1_rows']:.3f}, {b['top1_nc']:.3f} | " + " / ".join(f"{x:+.3f}" for x in b["test_excess"]) + f", largest {b['test_sup']:.3f}")
        return out
    rep = {}; core = ["block", *base]; log("configuration | fusion weights (block, sources...) | held-out own games | test games, label-free || the same with the activity prior")
    rep["base"] = config("base (block + " + "+".join(base) + ")", core)
    for nm in adds:
        rep[f"+{nm}"] = config(f"base + {nm}", [*core, nm])
    for c in a_.combo:
        names = c.split("+")
        if all(n in adds for n in names):
            rep[f"+{c}"] = config(f"base + {c}", [*core, *names])
    for nm in adds:
        rep[f"{nm} alone"] = config(f"block + {nm} only", ["block", nm])
    (RV / a_.json).write_text(json.dumps(rep, indent=1)); log(f"written results/r_variants/{a_.json}")


if __name__ == "__main__":
    main()
