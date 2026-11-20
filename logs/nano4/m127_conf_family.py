#!/usr/bin/env python
"""Section 4M M127 (2026-11-14): a wider family of "was the opponent recognised" statistics, chosen without labels.

M125: replacing first-minus-second by first-minus-mean(2..5) of the fused score raises the label-free share of recognised
opponents from 0.204 to 0.226 on the production identifier; M126: alone that is worth about +0.002 at recipe level (below the
adoption threshold on one seed set). Here the same fused score vector is read in more ways:
  m1k     F1 - mean(F2 .. Fk), k = 3, 4, 5, 7, 10, 20          z10, z20   (F1 - mean(F2..Fk)) / sd(F2..Fk)
  lr      a logistic model of "the first candidate is the true player", fitted on the held-out own games only (their first
          candidate is right or wrong; never on the opponent-side or impostor queries, so it cannot learn the two query
          types apart), on: m12, m15, m110, m120, z20, F2 - F3, and per source (block, matcher, sequence nets) the margin of
          the fused first candidate inside that source and whether the source ranks it first.
  --stage fuse --tags ...     one GPU pass per identifier: top-20 fused values, the first candidate, the per-source read-outs,
                              for the validation / test queries and the held-out own games -> results/r_variants/m127_<tag>.npz
  --stage report --tags ...   per tag and statistic: the label-free share (m120_diag_mass.py's definition), the reading quality
                              of the top bands, anchors against m12 and against m15 (report only); --write STAT writes
                              m84_<tag>~<STAT>_{val,test}.npz for the bundle / m121 path.
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
KS = (3, 4, 5, 7, 10, 20)


def fuse(tag):
    import torch
    t0 = time.time(); dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []
    for var, members in mv.MULTI[tag]:
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))
    w = np.array(json.loads((RV / f"m84_{tag}_eval.json").read_text())["weights"]); n_src = len(src) + 1

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

    def run(ids, block):
        n = len(ids); TV = np.zeros((n, 20), np.float32); b = np.zeros(n, np.int64); SR = np.zeros((n, n_src), np.uint8); SG = np.zeros((n, n_src), np.float32)
        for i in range(0, n, 2048):
            z, valid = zs(ids[i:i + 2048]); parts = [torch.as_tensor(block[i:i + 2048].astype(np.float32), device=dev), *z]; F = sum(w[k] * parts[k] for k in range(n_src))
            F = F.masked_fill(~valid, -float("inf")); top = F.topk(20, dim=1); TV[i:i + 2048] = top.values.cpu().numpy(); t1 = top.indices[:, :1]; b[i:i + 2048] = t1[:, 0].cpu().numpy()
            for k in range(n_src):
                Sk = parts[k].masked_fill(~valid, -float("inf")); own = Sk.gather(1, t1); SR[i:i + 2048, k] = (Sk > own).sum(1).clamp(max=255).cpu().numpy().astype(np.uint8)
                rest = Sk.scatter(1, t1, -float("inf")); SG[i:i + 2048, k] = (own[:, 0] - rest.max(1).values).cpu().numpy()
        return TV, b, SR, SG
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]
    TV, b, SR, SG = run(idx, blk[order]); del blk
    o = slice(0, Q); ref = dict(np.load(RV / f"m84_{tag}_val.npz")); assert np.array_equal(plab[b[o]], ref["f_opp_lab"]) and np.allclose((TV[:, 0] - TV[:, 1])[o], ref["f_opp_m"], atol=1e-3), "the fused pass does not reproduce the stored identifier"
    hq = np.arange(2 * Q, 2 * Q + H); HV, hb, HR, HG = run(hq, np.load(CACHE / "block_held.npy")); hit = hb == q["held_truth"]
    np.savez_compressed(RV / f"m127_{tag}.npz", TV=TV, best=b, lab=plab[b], SR=SR, SG=SG, HV=HV, hit=hit, HR=HR, HG=HG, held_truth=q["held_truth"])
    print(f"tag {tag}: fused ({n_src} sources), stored identifier reproduced; held-out first candidate right {float(hit.mean()):.3f} ({time.time() - t0:.0f} s)", flush=True)


def simple_stats(TV):
    out = {"m12": TV[:, 0] - TV[:, 1]}
    for k in KS:
        out[f"m1{k}"] = TV[:, 0] - TV[:, 1:k].mean(1)
    for k in (10, 20):
        out[f"z{k}"] = (TV[:, 0] - TV[:, 1:k].mean(1)) / np.maximum(TV[:, 1:k].std(1), 1e-6)
    return out


def features(TV, SR, SG):
    s = simple_stats(TV)
    return np.column_stack([s["m12"], s["m15"], s["m110"], s["m120"], s["z20"], TV[:, 1] - TV[:, 2], np.clip(SG, -8, 8), (SR == 0).astype(np.float32)]).astype(np.float64)


def fit_logistic(X, y, l2=1.0, iters=200):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.column_stack([(X - mu) / sd, np.ones(len(X))]); w = np.zeros(Z.shape[1]); R = l2 * np.eye(Z.shape[1]); R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) + R @ w; Hm = (Z * (p * (1 - p))[:, None]).T @ Z + R; step = np.linalg.solve(Hm, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: np.column_stack([(Xn - mu) / sd, np.ones(len(Xn))]) @ w


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def quick_write(tags, k):
    """Identifier files m84_<tag>~<k>_{val,test}.npz for a simple statistic, without the comparisons."""
    q = np.load(CACHE / "queries.npz"); Q, T = int(q["Q"]), int(q["T"]); players = q["players"]; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)
    d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {x: d82[x] for x in ("key", "pid", "fold", "y", "hd", "target_black")}
    for tag in tags:
        z = dict(np.load(RV / f"m127_{tag}.npz")); st = simple_stats(z["TV"])
        np.savez_compressed(RV / f"m84_{tag}~{k}_val.npz", **base, f_opp_lab=z["lab"][o], f_opp_m=st[k][o], f_opp_best=players[z["best"][o]], f_imp_lab=z["lab"][t_], f_imp_m=st[k][t_])
        np.savez_compressed(RV / f"m84_{tag}~{k}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=z["lab"][to], f_opp_m=st[k][to], f_opp_best=players[z["best"][to]], f_tgt_lab=z["lab"][tt], f_tgt_m=st[k][tt])
        print(f"identifier files m84_{tag}~{k}_{{val,test}}.npz written", flush=True)


def report(tags, write):
    q = np.load(CACHE / "queries.npz"); Q, T = int(q["Q"]), int(q["T"]); players = q["players"]; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)
    d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); out = {}
    for tag in tags:
        z = dict(np.load(RV / f"m127_{tag}.npz")); st = simple_stats(z["TV"]); hs = simple_stats(z["HV"]); y = z["hit"].astype(float); out[tag] = {}
        Xh = features(z["HV"], z["HR"], z["HG"]); half = np.array([int(str(players[t])[:8], 16) % 2 for t in z["held_truth"]]); cv = np.zeros(len(y))
        for h in (0, 1):
            cv[half == h] = fit_logistic(Xh[half != h], y[half != h])(Xh[half == h])
        st["lr"] = fit_logistic(Xh, y)(features(z["TV"], z["SR"], z["SG"])).astype(np.float32); hs["lr"] = cv
        print(f"\n=== {tag}: held-out own games, AUC of 'first candidate is right': " + "  ".join(f"{k} {auc(v, y):.4f}" for k, v in hs.items()))
        print("  statistic: share of recognised opponents (all / even / HD1-B / HD1-W), top bands same / within one, anchors vs m12 and vs m15 (seeds 0-2 | 6-8)")
        A = {}
        for k in ("m12", "m15", *[x for x in st if x not in ("m12", "m15")]):
            Z = {**base, "opp_lab": z["lab"][o], "opp_m": st[k][o], "imp_m": st[k][t_], "imp_lab": z["lab"][t_]}; ones = np.ones(Q, bool); edges, cal, band, g = s2.fit(Z, ones)
            mass = {j: float(np.maximum(cal[j]["share"] - nominal, 0).sum()) for j in (0, 1, 2)}; wg = np.array([(g == j).sum() for j in (0, 1, 2)], float); wg /= wg.sum(); top = (g == 0) & (band >= 5); dd = (Z["opp_lab"] - Z["y"])[top]
            row = {"mass": float(sum(wg[j] * mass[j] for j in (0, 1, 2))), "mass_by_group": mass, "top_same": float((dd == 0).mean()), "top_within1": float((np.abs(dd) <= 1).mean()), "heldout_auc": auc(hs[k], y), "vs": {}}
            line = f"  {k:5s} {row['mass']:.4f} / {mass[0]:.4f} / {mass[1]:.4f} / {mass[2]:.4f}   {row['top_same']:.3f} / {row['top_within1']:.3f}  "
            for seeds in ((0, 1, 2), (6, 7, 8)):
                A[(k, seeds)] = mv.anchors(Z, seeds)
            for ref in ("m12", "m15"):
                if k == ref or (ref == "m15" and k == "m12"):
                    continue
                for seeds in ((0, 1, 2), (6, 7, 8)):
                    a, c = A[(k, seeds)], A[(ref, seeds)]; yy, n, pid = (np.concatenate([x[j] for x in a]) for j in (2, 3, 4))
                    d = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] for x in c]), yy, pid, n, sizes=SIZES, n=2000); pf = [float(balanced_mean_over_n(u[0], u[2], u[3], sizes=SIZES) - balanced_mean_over_n(v_[0], v_[2], v_[3], sizes=SIZES)) for u, v_ in zip(a, c)]
                    row["vs"][f"{ref}|{seeds[0]}"] = {**d, "per_fold": pf}
                line += f" vs {ref}: " + " | ".join(f"{row['vs'][f'{ref}|{s_}']['diff']:+.4f} [{row['vs'][f'{ref}|{s_}']['ci95'][0]:+.4f}, {row['vs'][f'{ref}|{s_}']['ci95'][1]:+.4f}] ({sum(e > 0 for e in row['vs'][f'{ref}|{s_}']['per_fold'])}/4)" for s_ in (0, 6))
            out[tag][k] = row; print(line, flush=True)
            if write and k == write:
                np.savez_compressed(RV / f"m84_{tag}~{k}_val.npz", **base, f_opp_lab=z["lab"][o], f_opp_m=st[k][o], f_opp_best=players[z["best"][o]], f_imp_lab=z["lab"][t_], f_imp_m=st[k][t_])
                np.savez_compressed(RV / f"m84_{tag}~{k}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=z["lab"][to], f_opp_m=st[k][to], f_opp_best=players[z["best"][to]], f_tgt_lab=z["lab"][tt], f_tgt_m=st[k][tt])
                print(f"    identifier files m84_{tag}~{k}_{{val,test}}.npz written")
    (RV / f"m127_conf_family{'_' + '_'.join(tags) if len(tags) < 3 else ''}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("fuse", "report", "write")); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--write", default=None); a_ = ap.parse_args()
    if a_.stage == "fuse":
        for t in a_.tags:
            fuse(t)
    elif a_.stage == "write":
        quick_write(a_.tags, a_.write or "m15")
    else:
        report(a_.tags, a_.write)
