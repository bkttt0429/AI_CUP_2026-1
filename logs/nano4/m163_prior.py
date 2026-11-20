#!/usr/bin/env python
"""Section 4M M163 (2026-11-15): the candidates' activity as a prior inside the identifier's fused score (design doc M163).

The fused score F of an identifier (block + matcher sources, weights fitted on the held-out own games, nats) treats the 8,007
candidates alike, while opponents turn up in proportion to their activity. Here F'(c) = F(c) + beta * log n_c, n_c = the
candidate's number of training games.
  (1) report: held-out own games, beta grid - top-1 unweighted, by the player's number of games, weighted by rows (bands),
      weighted by n_c; beta (and a temperature of F) fitted by the weighted likelihood. --single NAME=out.npz: the same for
      one network's prototypes (standardised cosine, one fitted weight).
  (2) --out NAME:beta ...: all queries fused again with the prior -> results/r_variants/m127_<NAME>.npz (m127's layout) and
      m84_<NAME>~m15_{val,test}.npz (m127_conf_family.quick_write); beta = 0 must reproduce the stored identifier.

  python -I logs/nano4/m163_prior.py --tag SPAL:P,S2y,S4ft,S4Lt --out SPALn:1.0 --out SPALh:0.5 --single a=results/m161/out_-1_a.npz
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
import m127_conf_family as m127  # noqa: E402
import m84_variant as mv  # noqa: E402

CACHE, RV = mv.CACHE, mv.RV
BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def table(name, F, truth, ng, logn, log):
    """Held-out own games: the beta grid and the weighted-likelihood fits. F [H, C] in nats (numpy)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; Ft = torch.as_tensor(F, device=dev, dtype=torch.float32); ln = torch.as_tensor(logn, device=dev, dtype=torch.float32); tr = torch.as_tensor(truth, device=dev); H = len(truth)
    band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]]); share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum()
    wb = share[band] / np.bincount(band, minlength=len(BANDS))[band]; wb /= wb.sum(); wa = ng[truth].astype(np.float64); wa /= wa.sum(); rep = {"grid": {}}
    log(f"== {name}: held-out own games {H}; share of rows by band " + " / ".join(f"{x:.3f}" for x in share) + "; held-out games by band " + " / ".join(str(int((band == k).sum())) for k in range(len(BANDS))))
    log("   beta | top-1 unweighted | by games " + "  ".join(f"{lo}+" for lo, _ in BANDS) + " | weighted by rows (bands) | weighted by n_c | top-5 weighted by rows | first candidate changed from beta 0")
    b0 = None
    for beta in GRID:
        G = Ft + beta * ln[None]; own = G[torch.arange(H, device=dev), tr]; r = (G > own[:, None]).sum(1).cpu().numpy(); best = G.argmax(1).cpu().numpy(); b0 = best if b0 is None else b0
        t1 = np.array([float((r[band == k] == 0).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))])
        row = {"top1": float((r == 0).mean()), "by_band": t1.tolist(), "top1_rows": float((wb * (r == 0)).sum()), "top1_nc": float((wa * (r == 0)).sum()), "top5_rows": float((wb * (r < 5)).sum()), "changed": float((best != b0).mean())}; rep["grid"][str(beta)] = row
        log(f"   {beta:4.2f} | {row['top1']:.3f} | " + "  ".join(f"{x:.3f}" for x in t1) + f" | {row['top1_rows']:.3f} | {row['top1_nc']:.3f} | {row['top5_rows']:.3f} | {row['changed']:.3f}")
    for wn, w in (("rows (bands)", wb), ("n_c", wa)):
        wt = torch.as_tensor(w, device=dev, dtype=torch.float32); best = (-1e18, 0.0, 1.0); line = {}
        for tau in (0.6, 0.8, 1.0, 1.2, 1.4):
            for beta in np.arange(0.0, 2.01, 0.05):
                ll = float((wt * torch.log_softmax(tau * Ft + float(beta) * ln[None], dim=1)[torch.arange(H, device=dev), tr]).sum())
                if ll > best[0]:
                    best = (ll, float(beta), tau)
                if tau == 1.0 and abs(beta * 4 - round(beta * 4)) < 1e-9 and beta <= 1.5:
                    line[round(float(beta), 2)] = ll
        rep[f"fit_{wn}"] = {"beta": best[1], "tau": best[2], "loglik": best[0], "loglik_tau1": line}
        log(f"   likelihood weighted by {wn}: best beta {best[1]:.2f} at temperature {best[2]:.1f} (mean log-likelihood {best[0]:.4f}); at temperature 1: " + "  ".join(f"beta {k}: {v:.4f}" for k, v in line.items()))
    return rep


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--tag", required=True, help="NAME:var,var,... of an identifier whose m84_<NAME>_eval.json (weights) exists")
    ap.add_argument("--out", action="append", default=[], help="NEW:beta, an identifier to write"); ap.add_argument("--single", action="append", default=[], help="NAME=out file of one network (protos / qidx / qvec)"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tag, vars_ = a_.tag.split(":"); outs = [(x.split(":")[0], float(x.split(":")[1])) for x in a_.out]
    for nm, _ in outs:
        assert nm != tag and not (RV / f"m127_{nm}.npz").exists(), f"{nm}: results exist already, nothing overwritten"
    q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); truth = q["held_truth"].astype(np.int64)
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players.astype(str)).to_numpy(); assert np.isfinite(ng).all() and (ng >= 1).all(); ng = ng.astype(np.int64); logn = np.log(ng.astype(np.float64))
    log(f"device {dev}; candidates {C}; games per candidate: quartiles " + " / ".join(f"{x:.0f}" for x in np.quantile(ng, [0.25, 0.5, 0.75, 0.9, 0.99])) + f"; sd of log n_c {logn.std():.2f}")
    src = []
    for v in vars_.split(","):
        V = np.load(CACHE / f"var_{v}.npz"); src.append((torch.as_tensor(V["protos"], device=dev), V["qvec"]))
    w = np.array(json.loads((RV / f"m84_{tag}_eval.json").read_text())["weights"]); n_src = len(src) + 1; assert len(w) == n_src, (w, n_src); ln_t = torch.as_tensor(logn, device=dev, dtype=torch.float32)
    log(f"identifier {tag}: block + {vars_}; weights " + ", ".join(f"{x:.2f}" for x in w))

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

    def run(ids, block, betas, keep_f=False):
        """Per beta: top-20 fused values, the first candidate, per source its rank of the first candidate and its margin inside the source."""
        n = len(ids); res = {b_: (np.zeros((n, 20), np.float32), np.zeros(n, np.int64), np.zeros((n, n_src), np.uint8), np.zeros((n, n_src), np.float32)) for b_ in betas}; Fs = []
        for i in range(0, n, 2048):
            z, valid = zs(ids[i:i + 2048]); parts = [torch.as_tensor(block[i:i + 2048].astype(np.float32), device=dev), *z]; F0 = sum(float(w[k]) * parts[k] for k in range(n_src))
            if keep_f:
                Fs.append(F0.cpu().numpy())
            for b_ in betas:
                TV, bb, SR, SG = res[b_]; F = (F0 + b_ * ln_t[None]).masked_fill(~valid, -float("inf")); top = F.topk(20, dim=1); TV[i:i + 2048] = top.values.cpu().numpy(); t1 = top.indices[:, :1]; bb[i:i + 2048] = t1[:, 0].cpu().numpy()
                for k in range(n_src):
                    Sk = parts[k].masked_fill(~valid, -float("inf")); own = Sk.gather(1, t1); SR[i:i + 2048, k] = (Sk > own).sum(1).clamp(max=255).cpu().numpy().astype(np.uint8)
                    rest = Sk.scatter(1, t1, -float("inf")); SG[i:i + 2048, k] = (own[:, 0] - rest.max(1).values).cpu().numpy()
        return res, (np.concatenate(Fs) if keep_f else None)
    # ---------------------------------------------------------------- (1) held-out own games
    rep = {"tag": tag, "weights": w.tolist()}; hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy"); betas = sorted({0.0, *[b_ for _, b_ in outs]})
    hres, Fh = run(hq, hb, betas, keep_f=True); assert (excl[hq] < 0).all(), "held-out queries are expected to have every candidate valid"
    rep["fused"] = table(f"identifier {tag} (fused score)", Fh, truth, ng, logn, log); del Fh
    for spec in a_.single:
        nm, path = spec.split("=", 1); z = np.load(path); pos = {int(g): i for i, g in enumerate(z["qidx"])}; S = z["qvec"][[pos[2 * Q + i] for i in range(H)]].astype(np.float64) @ z["protos"].astype(np.float64).T
        Z = ip.standardise(S).astype(np.float32); w1 = float(ip.fit_weights([Z], truth)[0]); rep[f"single_{nm}"] = {"weight": w1, **table(f"single network {nm} (standardised cosine x fitted weight {w1:.2f})", w1 * Z, truth, ng, logn, log)}
    (RV / f"m163_prior_{tag}.json").write_text(json.dumps(rep, indent=1)); log(f"written results/r_variants/m163_prior_{tag}.json")
    if not outs:
        return
    # ---------------------------------------------------------------- (2) all queries with the prior
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)])[order]; log(f"blocks loaded ({len(idx)} queries)")
    res, _ = run(idx, blk, betas); del blk; log("all queries fused")
    ref = np.load(RV / f"m127_{tag}.npz"); TV0, b0 = res[0.0][0], res[0.0][1]; same = float((b0 == ref["best"]).mean()); dev_ = float(np.abs(TV0[:, :5] - ref["TV"][:, :5]).max())
    log(f"check: beta 0 reproduces the stored identifier - same first candidate {same:.5f}, largest difference of the top-5 values {dev_:.2e}"); assert same > 0.9995 and dev_ < 5e-3, "the fused pass does not reproduce the stored identifier"
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); st0 = m127.simple_stats(TV0)["m15"]
    for nm, beta in outs:
        TV, bb, SR, SG = res[beta]; HV, hbest, HR, HG = hres[beta]; hit = hbest == truth
        np.savez_compressed(RV / f"m127_{nm}.npz", TV=TV, best=bb, lab=plab[bb], SR=SR, SG=SG, HV=HV, hit=hit, HR=HR, HG=HG, held_truth=truth, beta=beta, base=tag)
        st = m127.simple_stats(TV)["m15"]; line = []
        for name_, sl, null in (("validation opponent side", o, t_), ("test opponent side", to, tt)):
            ch = float((bb[sl] != b0[sl]).mean()); ex = {qq: float((st[sl] > np.quantile(st[null], qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)}; ex0 = {qq: float((st0[sl] > np.quantile(st0[null], qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)}
            line.append(f"{name_}: first candidate changed {ch:.3f}; excess above the own-side q90 / q98 / q99.5 " + " / ".join(f"{ex[k]:+.3f}" for k in ex) + " (beta 0: " + " / ".join(f"{ex0[k]:+.3f}" for k in ex0) + f"); median games of the first candidate {np.median(ng[bb[sl]]):.0f} (beta 0: {np.median(ng[b0[sl]]):.0f})")
            rep[f"{nm}|{name_}"] = {"changed": ch, "excess": {str(k): v for k, v in ex.items()}, "excess_beta0": {str(k): v for k, v in ex0.items()}}
        log(f"identifier {nm} (beta {beta}): held-out first candidate right {float(hit.mean()):.3f}; " + "; ".join(line))
        m127.quick_write([nm], "m15")
    (RV / f"m163_prior_{tag}.json").write_text(json.dumps(rep, indent=1)); log("done")


if __name__ == "__main__":
    main()
