#!/usr/bin/env python
"""Section 4M M125 (2026-11-14): which statistic of the fused candidate scores says best that the opponent was recognised?

The anchor pipeline bands every opponent-side query by the margin of the fused score between the first and the second candidate
(quantiles of the impostor margins); M120 / M124 showed that the anchors follow the share of opponents recognised with a clear
margin, i.e. the stability of that statistic. The fused score is a calibrated logit (weights fitted by the likelihood of the true
candidate on the held-out own games), so other read-outs of the same score vector are available at no cost:
  m12    F1 - F2                                   (current)
  m1k    F1 - mean(F2 .. Fk), k = 5, 10, 20        (the runner-up field instead of one runner-up)
  lo     F1 - logsumexp(F2 ..)                     (log posterior odds of the first candidate against the whole field)
  --stage fuse --tags A B ...   one GPU pass per identifier tag (the sources and weights of m84_variant.py's MULTI / eval JSON):
                                top candidate and the statistics for every query -> results/r_variants/m125_<tag>.npz
  --stage report --tags ...     per tag and statistic: the label-free share of recognised opponents (m120_diag_mass.py's
                                definition), and the anchors' score against the same identifier with m12 (m84_variant.anchors,
                                seeds 0-2 and 6-8, paired) - report only; a statistic is written as identifier files
                                m84_<tag>~<stat>_{val,test}.npz for the bundle / m121 path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m84_variant as mv  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

CACHE, RV, s2 = mv.CACHE, mv.RV, mv.s2
STATS = ("m12", "m15", "m110", "m120", "lo")


def fuse(tag):
    import torch
    t0 = time.time(); dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []
    for var, members in mv.MULTI[tag]:
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))
    w = np.array(json.loads((RV / f"m84_{tag}_eval.json").read_text())["weights"])

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
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]
    b = np.zeros(len(idx), np.int64); st = {k: np.zeros(len(idx), np.float32) for k in STATS}
    for i in range(0, len(idx), 2048):
        z, valid = zs(idx[i:i + 2048]); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)
        for k in range(len(z)):
            F = F + w[k + 1] * z[k]
        F = F.masked_fill(~valid, -float("inf")); top = F.topk(20, dim=1); v = top.values; b[i:i + 2048] = top.indices[:, 0].cpu().numpy()
        rest = F.clone(); rest.scatter_(1, top.indices[:, :1], -float("inf"))
        st["m12"][i:i + 2048] = (v[:, 0] - v[:, 1]).cpu().numpy(); st["lo"][i:i + 2048] = (v[:, 0] - torch.logsumexp(rest, dim=1)).cpu().numpy()
        for name, k in (("m15", 5), ("m110", 10), ("m120", 20)):
            st[name][i:i + 2048] = (v[:, 0] - v[:, 1:k].mean(1)).cpu().numpy()
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)
    ref = dict(np.load(RV / f"m84_{tag}_val.npz")); assert np.array_equal(plab[b[o]], ref["f_opp_lab"]) and np.allclose(st["m12"][o], ref["f_opp_m"], atol=1e-3), "the fused pass does not reproduce the stored identifier"
    np.savez_compressed(RV / f"m125_{tag}.npz", opp_lab=plab[b[o]], imp_lab=plab[b[t_]], t_opp_lab=plab[b[to]], t_tgt_lab=plab[b[tt]], opp_best=b[o], t_opp_best=b[to],
                        **{f"opp_{k}": x[o] for k, x in st.items()}, **{f"imp_{k}": x[t_] for k, x in st.items()}, **{f"t_opp_{k}": x[to] for k, x in st.items()}, **{f"t_tgt_{k}": x[tt] for k, x in st.items()})
    print(f"tag {tag}: fused, the stored identifier reproduced; statistics saved ({time.time() - t0:.0f} s)", flush=True)


def report(tags, write):
    d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); out = {}
    for tag in tags:
        z = dict(np.load(RV / f"m125_{tag}.npz")); A = {}; out[tag] = {}
        print(f"\n=== {tag}: statistic, share of recognised opponents (all / even / HD1 target Black / HD1 target White), top bands same / within one, anchors vs m12 (seeds 0-2, 6-8)")
        for k in STATS:
            Z = {**base, "opp_lab": z["opp_lab"], "opp_m": z[f"opp_{k}"], "imp_m": z[f"imp_{k}"], "imp_lab": z["imp_lab"]}; ones = np.ones(len(Z["y"]), bool); edges, cal, band, g = s2.fit(Z, ones)
            mass = {j: float(np.maximum(cal[j]["share"] - nominal, 0).sum()) for j in (0, 1, 2)}; wg = np.array([(g == j).sum() for j in (0, 1, 2)], float); wg /= wg.sum(); top = (g == 0) & (band >= 5); dd = (Z["opp_lab"] - Z["y"])[top]
            row = {"mass": float(sum(wg[j] * mass[j] for j in (0, 1, 2))), "mass_by_group": mass, "top_same": float((dd == 0).mean()), "top_within1": float((np.abs(dd) <= 1).mean()), "anchors_vs_m12": {}}
            for seeds in ((0, 1, 2), (6, 7, 8)):
                A[(k, seeds)] = mv.anchors(Z, seeds)
                if k != "m12":
                    a, c = A[(k, seeds)], A[("m12", seeds)]; y, n, pid = (np.concatenate([x[j] for x in a]) for j in (2, 3, 4))
                    d = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] for x in c]), y, pid, n, sizes=SIZES, n=3000); pf = [float(balanced_mean_over_n(u[0], u[2], u[3], sizes=SIZES) - balanced_mean_over_n(v_[0], v_[2], v_[3], sizes=SIZES)) for u, v_ in zip(a, c)]
                    row["anchors_vs_m12"][str(seeds[0])] = {**d, "per_fold": pf}
            out[tag][k] = row
            print(f"  {k:5s} {row['mass']:.4f} / {mass[0]:.4f} / {mass[1]:.4f} / {mass[2]:.4f}   {row['top_same']:.3f} / {row['top_within1']:.3f}   " + "   ".join(f"{v_['diff']:+.4f} [{v_['ci95'][0]:+.4f}, {v_['ci95'][1]:+.4f}] folds {' '.join(f'{e:+.4f}' for e in v_['per_fold'])}" for v_ in row["anchors_vs_m12"].values()), flush=True)
            if write and k == write:
                np.savez_compressed(RV / f"m84_{tag}~{k}_val.npz", **base, f_opp_lab=z["opp_lab"], f_opp_m=z[f"opp_{k}"], f_opp_best=np.load(CACHE / "queries.npz")["players"][z["opp_best"]], f_imp_lab=z["imp_lab"], f_imp_m=z[f"imp_{k}"])
                np.savez_compressed(RV / f"m84_{tag}~{k}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=z["t_opp_lab"], f_opp_m=z[f"t_opp_{k}"], f_opp_best=np.load(CACHE / "queries.npz")["players"][z["t_opp_best"]], f_tgt_lab=z["t_tgt_lab"], f_tgt_m=z[f"t_tgt_{k}"])
                print(f"    identifier files m84_{tag}~{k}_{{val,test}}.npz written")
    (RV / "m125_conf_stat.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("fuse", "report")); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--write", default=None); a_ = ap.parse_args()
    if a_.stage == "fuse":
        for t in a_.tags:
            fuse(t)
    else:
        report(a_.tags, a_.write)
