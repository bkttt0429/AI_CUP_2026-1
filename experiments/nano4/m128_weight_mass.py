#!/usr/bin/env python
"""Section 4M M128 (2026-11-14): fusion weights of the identifier chosen by the label-free share of recognised opponents.

The identifier's fused score is w0 * block + sum_k w_k * source_k with the weights fitted by the likelihood of the true candidate
on the held-out OWN games (target side). The anchors use the opponent side, and M120 / M124 / M125 showed that held-out accuracy
and the opponent side do not move together. Here every non-block weight is multiplied by a factor from a grid (the block's
weight fixed; only the ratios matter), the validation queries are fused once per setting, and each setting gets the label-free
share of recognised opponents under the m15 statistic (m120_diag_mass.py's definition), the reading quality of the top bands
and the share of opponent-side queries whose first candidate differs from the fitted weights'. Report only.
  python -I logs/nano4/m128_weight_mass.py --tag SP2e [--grid 0.5 0.75 1 1.5 2 3]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m84_variant as mv  # noqa: E402

CACHE, RV, s2 = mv.CACHE, mv.RV, mv.s2


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--grid", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0]); a_ = ap.parse_args()
    t0 = time.time(); dev = "cuda"; tag = a_.tag; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q = int(q["Q"]); C = len(players); src = []
    for var, members in mv.MULTI[tag]:
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))
    w = np.array(json.loads((RV / f"m84_{tag}_eval.json").read_text())["weights"]); n_src = len(src) + 1
    settings = [(1.0, *f) for f in itertools.product(a_.grid, repeat=n_src - 1)]; ref = settings.index(tuple([1.0] * n_src)); W = torch.as_tensor(np.array([[w[k] * s[k] for k in range(n_src)] for s in settings], np.float32), device=dev)

    def zs(ids):
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); cnt = valid.sum(1, keepdim=True); out = []
        for protos, qvec in src:
            z = torch.zeros(len(ids), C, device=dev)
            for m in range(protos.shape[0]):
                S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)
                for j in range(5):
                    s = nj == j
                    if s.any():
                        S[s] = qv[s] @ protos[m, j].T
                mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)
            out.append(z / protos.shape[0])
        return out, valid
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; keep = idx < 2 * Q; idx, blk = idx[keep], blk[order][keep]
    B = np.zeros((len(settings), len(idx)), np.int32); M = np.zeros((len(settings), len(idx)), np.float32)
    for i in range(0, len(idx), 1024):
        z, valid = zs(idx[i:i + 1024]); parts = torch.stack([torch.as_tensor(blk[i:i + 1024].astype(np.float32), device=dev), *z])
        for s_ in range(len(settings)):
            F = (W[s_][:, None, None] * parts).sum(0).masked_fill(~valid, -float("inf")); top = F.topk(5, dim=1); B[s_, i:i + 1024] = top.indices[:, 0].cpu().numpy(); M[s_, i:i + 1024] = (top.values[:, 0] - top.values[:, 1:5].mean(1)).cpu().numpy()
    print(f"tag {tag}: {len(settings)} weight settings fused on {len(idx)} validation queries ({time.time() - t0:.0f} s)", flush=True)
    d82 = dict(np.load(RV / "m82_val.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); o, t_ = slice(0, Q), slice(Q, 2 * Q); rows = []
    for s_, st in enumerate(settings):
        Z = {**base, "opp_lab": plab[B[s_, o]], "opp_m": M[s_, o], "imp_m": M[s_, t_], "imp_lab": plab[B[s_, t_]]}; ones = np.ones(Q, bool); edges, cal, band, g = s2.fit(Z, ones)
        mass = {j: float(np.maximum(cal[j]["share"] - nominal, 0).sum()) for j in (0, 1, 2)}; wg = np.array([(g == j).sum() for j in (0, 1, 2)], float); wg /= wg.sum(); top = (g == 0) & (band >= 5); dd = (Z["opp_lab"] - Z["y"])[top]
        rows.append({"factors": list(st[1:]), "mass": float(sum(wg[j] * mass[j] for j in (0, 1, 2))), "top_same": float((dd == 0).mean()), "top_within1": float((np.abs(dd) <= 1).mean()), "first_candidate_changed": float((B[s_, o] != B[ref, o]).mean())})
    names = [v for v, _ in mv.MULTI[tag]]; best = sorted(range(len(rows)), key=lambda i: -rows[i]["mass"])
    print(f"fitted weights: block {w[0]:.2f}, " + ", ".join(f"{n} {w[k + 1]:.2f}" for k, n in enumerate(names)) + f"; at the fitted weights: share {rows[ref]['mass']:.4f}, top bands same / within one {rows[ref]['top_same']:.3f} / {rows[ref]['top_within1']:.3f}")
    print("factors on (" + ", ".join(names) + "): share, top bands same / within one, first candidate changed - the best 12 settings:")
    for i in best[:12]:
        r = rows[i]; print(f"  {tuple(r['factors'])!s:22s} {r['mass']:.4f}   {r['top_same']:.3f} / {r['top_within1']:.3f}   {r['first_candidate_changed']:.3f}")
    print("one factor at a time:")
    for k in range(n_src - 1):
        print(f"  {names[k]:5s} " + "  ".join(f"x{f:g}: {rows[settings.index(tuple([1.0] + [f if j == k else 1.0 for j in range(n_src - 1)]))]['mass']:.4f}" for f in a_.grid))
    (RV / f"m128_weight_mass_{tag}.json").write_text(json.dumps({"weights": w.tolist(), "sources": names, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
