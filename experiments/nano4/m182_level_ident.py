#!/usr/bin/env python
"""Section 4M M182 (2026-11-16): level-aggregated identifier - m164_ident.py with one change in the per-query output.

m164_ident.py reports, per query, the first candidate, its label and the top-20 fused scores (the margin statistic m15 is read
from them). Here the fused scores F(c) (block + sources + activity prior; fitted exactly as in m164_ident.py, so the same F as
the identifier with the same sources) are first summed within each level: S(l) = logsumexp of F(c) over the candidates c whose
label is l = the log posterior mass of "the opponent is one of the level-l candidates". Reported per query: the level with
the largest mass (lab), the level scores in descending order (TV; 10 levels, padded with the last value), and as "best" the
first candidate inside that level. A game whose opponent cannot be told apart from a few look-alikes of the SAME level then
still gives a confident label; a clearly recognised opponent gives the same label and nearly the same margin as before.
Everything downstream (bands from the impostor quantiles, P0 / K, a_b, the pool) is unchanged: it only reads lab and TV.
hit (held-out own games) = the level is right.

  python -I logs/nano4/m182_level_ident.py --prior fit --tag SL4:S6at,S7t,S7tc
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
BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]


def fit_w(Zs, truth, log, l2=1e-3):
    """ident_pool.fit_weights; when its undamped Newton diverges, a damped Newton on the same objective."""
    w = ip.fit_weights(Zs, truth, l2=l2)
    if np.isfinite(w).all() and np.abs(w).max() < 20:
        return w
    log(f"   fit_weights diverged (largest |w| {np.abs(w).max():.1f}): damped refit")
    Z = np.stack(Zs).astype(np.float32); J, Qn, _ = Z.shape; qi = np.arange(Qn); w = np.ones(J)

    def obj(w_):
        lg = np.einsum("j,jqc->qc", w_, Z); lg -= lg.max(1, keepdims=True); P = np.exp(lg); s = P.sum(1, keepdims=True)
        return float(-(lg[qi, truth] - np.log(s[:, 0])).mean() + l2 * (w_ ** 2).sum()), P / s
    f0, P = obj(w)
    for _ in range(100):
        EZ = np.einsum("jqc,qc->jq", Z, P); grad = (EZ - Z[:, qi, truth]).mean(1) + 2 * l2 * w
        A = (Z * P[None]).reshape(J, -1); Hm = (A @ Z.reshape(J, -1).T) / Qn - (EZ @ EZ.T) / Qn + 2 * l2 * np.eye(J); step = np.linalg.solve(Hm + 1e-6 * np.eye(J), grad); t = 1.0
        while t > 1e-4:
            f1, P1 = obj(w - t * step)
            if f1 <= f0:
                break
            t /= 2
        if t <= 1e-4 or abs(f0 - f1) < 1e-9:
            break
        w, f0, P = w - t * step, f1, P1
    return w


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--tag", action="append", required=True); ap.add_argument("--prior", default="fit"); ap.add_argument("--stat", default="m15"); ap.add_argument("--heldout-only", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); truth = q["held_truth"].astype(np.int64)
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players.astype(str)).to_numpy(); assert np.isfinite(ng).all() and (ng >= 1).all(); logn = np.log(ng); ln_t = torch.as_tensor(logn, device=dev, dtype=torch.float32)
    band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]]); share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum(); wb = share[band] / np.bincount(band, minlength=len(BANDS))[band]; wb /= wb.sum(); wa = ng[truth] / ng[truth].sum()
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy"); assert (excl[hq] < 0).all(); blk = idx = None
    for spec in a_.tag:
        name, vars_ = spec.split(":"); vars_ = vars_.split(","); assert name not in mv.TAGS and name not in mv.MULTI, f"{name} is an existing tag"
        assert a_.heldout_only or not ((RV / f"m127_{name}.npz").exists() or (RV / f"m84_{name}_eval.json").exists()), f"{name}: results exist already, nothing overwritten"
        src = []
        for v in vars_:
            V = np.load(CACHE / f"var_{v}.npz"); src.append((torch.as_tensor(V["protos"], device=dev), V["qvec"], torch.as_tensor(V["has"], device=dev) if "has" in V.files else None))
        n_src = len(src) + 1; log(f"== identifier {name}: block + " + " + ".join(f"{v} ({s[0].shape[0]} member(s){', centre source' if s[2] is not None else ''})" for v, s in zip(vars_, src)) + f"; device {dev}")

        def zs(ids):
            ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); out = []
            for protos, qvec, has in src:
                z = torch.zeros(len(ids), C, device=dev)
                for m in range(protos.shape[0]):
                    S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev); ok = valid.clone() if has is not None else valid
                    for j in range(5):
                        s = nj == j
                        if s.any():
                            S[s] = qv[s] @ protos[m, j].T
                            if has is not None:
                                ok[s] &= has[m, j][None]
                    cnt = ok.sum(1, keepdim=True).clamp(min=1); mu = (S * ok).sum(1, keepdim=True) / cnt; zz = (S - mu) / (((S - mu) ** 2 * ok).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)
                    z += torch.where(ok, zz, torch.zeros_like(zz)) if has is not None else zz
                out.append(z / protos.shape[0])
            return out, valid
        # ---------------------------------------------------------------- weights and the prior on the held-out own games
        zh = [x.cpu().numpy() for x in zs(hq)[0]]; w = fit_w([hb.astype(np.float32), *zh], truth, log); Fh = w[0] * hb.astype(np.float32) + sum(w[k + 1] * zh[k] for k in range(len(zh)))
        rnk = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1); acc = {}  # noqa: E731
        for nm, F in [(f"source{k} ({vars_[k]}) alone", zh[k]) for k in range(len(zh))] + [("fused", Fh)]:
            r_ = rnk(F); acc[nm] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean()), "top1_rows": float((wb * (r_ == 0)).sum()), "top1_nc": float((wa * (r_ == 0)).sum())}
        Ft = torch.as_tensor(Fh, device=dev, dtype=torch.float32); tr = torch.as_tensor(truth, device=dev); wt = torch.as_tensor(wa, device=dev, dtype=torch.float32); lls = {}
        for beta in np.arange(0.0, 2.01, 0.05):
            lls[round(float(beta), 2)] = float((wt * torch.log_softmax(Ft + float(beta) * ln_t[None], dim=1)[torch.arange(H, device=dev), tr]).sum())
        beta = max(lls, key=lls.get) if a_.prior == "fit" else float(a_.prior); r_ = rnk(Fh + beta * logn[None].astype(np.float32)); acc["fused with the prior"] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean()), "top1_rows": float((wb * (r_ == 0)).sum()), "top1_nc": float((wa * (r_ == 0)).sum())}
        log(f"   weights block {w[0]:.2f}, " + ", ".join(f"{vars_[k]} {w[k + 1]:.2f}" for k in range(len(zh))) + f"; beta {beta:.2f} ({'fitted' if a_.prior == 'fit' else 'given'}; likelihood-best {max(lls, key=lls.get):.2f})")
        for nm, x in acc.items():
            log(f"   held-out own games, {nm}: top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; weighted by rows {x['top1_rows']:.3f}, by n_c {x['top1_nc']:.3f}")
        rep = {"sources": vars_, "weights": w.tolist(), "beta": beta, "beta_loglik": lls, "heldout": acc}
        if a_.heldout_only:
            continue
        # ---------------------------------------------------------------- all queries
        if blk is None:
            idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)])[order]; log(f"   blocks loaded ({len(idx)} queries)")

        grp = np.unique(plab, return_inverse=True)[1].astype(np.int64); n_grp = int(grp.max()) + 1; grp_t = torch.as_tensor(grp, device=dev); assert 2 <= n_grp <= 20, n_grp

        def run(ids, block):
            n = len(ids); TV = np.zeros((n, 20), np.float32); bb = np.zeros(n, np.int64); SR = np.zeros((n, n_src), np.uint8); SG = np.zeros((n, n_src), np.float32)
            for i in range(0, n, 2048):
                z, valid = zs(ids[i:i + 2048]); parts = [torch.as_tensor(block[i:i + 2048].astype(np.float32), device=dev), *z]; F = (sum(float(w[k]) * parts[k] for k in range(n_src)) + beta * ln_t[None]).masked_fill(~valid, -float("inf"))
                S = torch.stack([torch.logsumexp(F.masked_fill(grp_t[None] != l, -float("inf")), dim=1) for l in range(n_grp)], 1)  # [B, levels]: log posterior mass of each level
                tl = S.topk(n_grp, dim=1); tv = tl.values.clamp(min=-1e4); TV[i:i + 2048, :n_grp] = tv.cpu().numpy(); TV[i:i + 2048, n_grp:] = tv[:, -1:].cpu().numpy()
                t1 = F.masked_fill(grp_t[None] != tl.indices[:, :1], -float("inf")).argmax(1, keepdim=True); bb[i:i + 2048] = t1[:, 0].cpu().numpy()
                for k in range(n_src):
                    Sk = parts[k].masked_fill(~valid, -float("inf")); own = Sk.gather(1, t1); SR[i:i + 2048, k] = (Sk > own).sum(1).clamp(max=255).cpu().numpy().astype(np.uint8)
                    rest = Sk.scatter(1, t1, -float("inf")); SG[i:i + 2048, k] = (own[:, 0] - rest.max(1).values).cpu().numpy()
            return TV, bb, SR, SG
        TV, bb, SR, SG = run(idx, blk); HV, hbest, HR, HG = run(hq, hb); hit = grp[hbest] == grp[truth]; log(f"   held-out own games: level of the largest mass is the true owner's level {float(hit.mean()):.3f}; first candidate is the owner {float((hbest == truth).mean()):.3f}")
        np.savez_compressed(RV / f"m127_{name}.npz", TV=TV, best=bb, lab=plab[bb], SR=SR, SG=SG, HV=HV, hit=hit, HR=HR, HG=HG, held_truth=truth, beta=beta)
        o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); st = m127.simple_stats(TV)[a_.stat]
        for nm, sl, null in (("validation opponent side", o, t_), ("test opponent side", to, tt)):
            ex = {qq: float((st[sl] > np.quantile(st[null], qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)}; grid = np.quantile(st[null], np.linspace(0.5, 0.999, 500)); sup = float(max((st[sl] > t).mean() - (st[null] > t).mean() for t in grid))
            rep[nm] = {"excess": {str(k): v for k, v in ex.items()}, "largest_excess": sup, "median_games_of_first": float(np.median(ng[bb[sl]]))}
            log(f"   {nm}: excess above the own-side q90 / q98 / q99.5 " + " / ".join(f"{ex[k]:+.3f}" for k in ex) + f", largest {sup:.3f}; median games of the first candidate {np.median(ng[bb[sl]]):.0f}")
        (RV / f"m84_{name}_eval.json").write_text(json.dumps(rep, indent=1)); m127.quick_write([name], a_.stat); log(f"   identifier {name} written (held-out first candidate right {float(hit.mean()):.3f})")


if __name__ == "__main__":
    main()
