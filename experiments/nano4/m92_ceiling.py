"""Section 4M M92 (2026-11-13, report only): the ceiling of the mirror channel. M91's trials found few mirror matches because a test
player is unseen and has a 5-20 game prototype. How much would mirror anchors be worth if the player were identified as well as a
SEEN training player? Here every training player C plays the role of the candidate with his full prototype under the all-players
network (seen): the opponent side of every other player's row is matched against the 8,007 prototypes (owner masked, scores
standardised per candidate, mean over the available seeds); a row whose best match is C is a mirror anchor of C with the owner's
level. Confidence tiers by the margin's quantile over all rows. Calibration by candidate fold (K from the top tier, P0 from the
lowest, a per tier by maximum likelihood on the label differences), one row per (candidate, owner). Then on the validation
questions of folds 1-4: base, base + bundle-v5 forward evidence, + this mirror evidence (temperatures 1 / 0.5 / 0.25 / 0.1).
No label of C enters C's own evidence except through the cross-fitted calibration tables.
  python -I logs/nano4/m92_ceiling.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m86_bundle as b  # noqa: E402
import m91_mirror as mm  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

TQ = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98]); NT = len(TQ) + 1; SM = b.SM


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; D = mm.load_meta(log); N = len(D["key"]); C = len(D["players"]); seeds = [s for s in (0, 1, 2) if (mm.EMB / f"emb_-1{f'_s{s}' if s else ''}.npz").exists()]; log(f"all-players network, seeds {seeds}")
    own_t = torch.as_tensor(D["owner"].copy(), device=dev); cnt = np.bincount(D["owner"], minlength=C); cand = np.flatnonzero(cnt >= 8); col = np.full(C, -1, np.int64); col[cand] = np.arange(len(cand)); col_t = torch.as_tensor(col, device=dev)
    PS, MU, SD, EO = [], [], [], []; chunk = 50_000
    for s in seeds:
        z = np.load(mm.EMB / f"emb_-1{f'_s{s}' if s else ''}.npz"); eo = torch.as_tensor(z["opp"].astype(np.float32), device=dev); ew = torch.as_tensor(z["own"].astype(np.float32), device=dev); del z
        P = torch.zeros((C, 128), device=dev).index_add_(0, own_t, ew); P = torch.nn.functional.normalize(P[torch.as_tensor(cand, device=dev)], dim=1); del ew; s1 = torch.zeros(len(cand), device=dev, dtype=torch.float64); s2_ = torch.zeros_like(s1)
        for i in range(0, N, chunk):
            S = (eo[i:i + chunk] @ P.T).double(); s1 += S.sum(0); s2_ += (S * S).sum(0)
        mu = s1 / N; PS.append(P); MU.append(mu.float()); SD.append((s2_ / N - mu * mu).clamp(min=1e-12).sqrt().float() + 1e-6); EO.append(eo)
    J = np.zeros(N, np.int64); MG = np.zeros(N, np.float32)
    for i in range(0, N, chunk):
        Z = 0.0
        for k in range(len(seeds)):
            Z = Z + (EO[k][i:i + chunk] @ PS[k].T - MU[k]) / SD[k]
        Z = Z / len(seeds); r = torch.arange(i, min(i + chunk, N), device=dev); c = col_t[own_t[r]]; ok = c >= 0; Z[torch.arange(len(r), device=dev)[ok], c[ok]] = -float("inf"); t = Z.topk(2, dim=1)
        J[i:i + chunk] = t.indices[:, 0].cpu().numpy(); MG[i:i + chunk] = (t.values[:, 0] - t.values[:, 1]).cpu().numpy()
    del EO, PS; cq = cand[J]; g = D["g_opp"]; ok = g >= 0; edges = np.quantile(MG[ok], TQ); tier = np.searchsorted(edges, MG); ylab = np.full(C, -1, np.int64); ylab[D["owner"]] = D["lab"]; pf = np.full(C, -9, np.int64); pf[D["owner"]] = D["fold"]
    d = np.clip(D["lab"] - ylab[cq] + 9, 0, 18); log(f"{len(cand)} candidates, {N} queries (the opponent side of every row); margin quantile edges " + " ".join(f"{x:.2f}" for x in edges))
    for k in range(3):
        s = g == k; log(f"group {k}: owner at the matched player's level / within one, by tier (lowest to highest): " + " | ".join(f"{float((d[s & (tier == x)] == 9).mean()):.2f} / {float((np.abs(d[s & (tier == x)] - 9) <= 1).mean()):.2f}" for x in range(NT)))
    # one row per (candidate, owner): the highest tier
    k = ok; qd, od, td, (ld, gd) = mm.dedup(cq[k], D["owner"][k], tier[k], [D["lab"][k], g[k]]); dd = np.clip(ld - ylab[qd] + 9, 0, 18); fq = pf[qd]; log(f"pairs (candidate, owner): {len(qd)}; in the top two tiers {int((td >= NT - 2).sum())}; per candidate in the top two tiers: mean {float(np.bincount(qd[td >= NT - 2], minlength=C)[cand].mean()):.1f}, median {float(np.median(np.bincount(qd[td >= NT - 2], minlength=C)[cand])):.0f}")
    grid = np.linspace(0.0, 0.98, 99); LM = np.zeros((C, 10)); MC = np.bincount(qd[td >= NT - 2], minlength=C)
    for v in sorted(set(pf[cand].tolist())):
        tr = fq != v; te = fq == v; out = np.zeros((int(te.sum()), 10)); lt, gt, tt = ld[te], gd[te], td[te]
        for k_ in range(3):
            h = np.zeros((NT, 19)); m_ = tr & (gd == k_); np.add.at(h, (td[m_], dd[m_]), 1); P0 = (h[0] + 1) / (h[0].sum() + 19); K = (h[NT - 1] + 0.5) / (h[NT - 1].sum() + 9.5); a = np.zeros(NT)
            for x in range(1, NT):
                a[x] = grid[int(np.argmax([(h[x] * np.log(u * K + (1 - u) * P0)).sum() for u in grid]))]
            ratio = np.minimum(K / P0, 30.0); s = gt == k_; aw = a[tt[s]][:, None]; out[s] = np.log((1 - aw) + aw * ratio[lt[s][:, None] - np.arange(10)[None, :] + 9])
            if v == 1:
                log(f"   fold-1 calibration, group {k_}: a by tier " + " ".join(f"{x:.2f}" for x in a) + " | K(-2..2) " + " ".join(f"{K[9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)) + " | P0(-2..2) " + " ".join(f"{P0[9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)))
        np.add.at(LM, qd[te], out)
    pred = LM.argmax(1); has = (np.abs(LM).sum(1) > 0) & (ylab >= 0)
    for lo, hi in ((1, 5), (6, 20), (21, 50), (51, 10 ** 9)):
        s = has & (MC >= lo) & (MC <= hi)
        if s.sum() > 30:
            log(f"mirror anchors alone, players with {lo}-{hi if hi < 10 ** 9 else 'inf'} pairs in the top two tiers: {int(s.sum())} players; exact {float((pred[s] == ylab[s]).mean()):.3f}, within one {float((np.abs(pred[s] - ylab[s]) <= 1).mean()):.3f}, mean score {float(SM[pred[s], ylab[s]].mean()):.4f}")
    V, _, _, used = b.sources(["5e", "SP2e"]); cols = defaultdict(list)
    for seeds_ in ((0, 1, 2), (6, 7, 8)):
        cols = defaultdict(list)
        for v in FOLDS:
            y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; em, ee = b.evidence(V["SP2e"], v, 1, 1), b.evidence(V["enc"], v, 0, 1); LF = np.zeros((nq, 10))
            for i, g_ in enumerate(gk):
                for k_ in g_:
                    if k_ in em:
                        LF[i] += em[k_]
                    if k_ in ee:
                        LF[i] += b.U_ENC * ee[k_]
            LF = LF * Tn; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds_], axis=0); pl = np.array([D["pidx"][p_] for p_ in pid]); Lm = LM[pl]
            sc = lambda L: SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), y]  # noqa: E731
            for name, L in (("base", np.zeros((nq, 10))), ("fwd", LF), ("mir", Lm), ("mir_only", Lm + 1e-9), ("both_T1", LF + Lm), ("both_T0.5", LF + 0.5 * Lm), ("both_T0.25", LF + 0.25 * Lm), ("both_T0.1", LF + 0.1 * Lm)):
                cols[name].append(sc(L) if name != "mir_only" else SM[Lm.argmax(1), y])
            cols["y"].append(y); cols["n"].append(n); cols["pid"].append(pid); cols["fold"].append(np.full(nq, v)); cols["m"].append(MC[pl])
        A = {k_: np.concatenate(x) for k_, x in cols.items()}; log(f"seeds {seeds_}: class-balanced scores: " + ", ".join(f"{name} {balanced_mean_over_n(A[name], A['y'], A['n'], sizes=SIZES):.4f}" for name in ("base", "fwd", "mir", "mir_only", "both_T1", "both_T0.5", "both_T0.25", "both_T0.1")))
        for label, x, w in (("forward_vs_base", "fwd", "base"), ("mirror_vs_base", "mir", "base"), ("both_T1_vs_forward", "both_T1", "fwd"), ("both_T0.5_vs_forward", "both_T0.5", "fwd"), ("both_T0.25_vs_forward", "both_T0.25", "fwd"), ("both_T0.1_vs_forward", "both_T0.1", "fwd")):
            dd_ = paired_diff(A[x], A[w], A["y"], A["pid"], A["n"], sizes=SIZES, n=3000); pf_ = [balanced_mean_over_n(A[x][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) - balanced_mean_over_n(A[w][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) for v in FOLDS]
            log(f"  seeds {seeds_} {label:24s} {dd_['diff']:+.4f} [{dd_['ci95'][0]:+.4f}, {dd_['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf_) + "  by n " + " ".join(f"{m_}: {x_:+.4f}" for m_, x_ in dd_["by_n"].items()))
        for lo, hi in ((0, 5), (6, 20), (21, 50), (51, 10 ** 9)):
            s = (A["m"] >= lo) & (A["m"] <= hi)
            if s.sum() > 100:
                log(f"     questions of players with {lo}-{hi if hi < 10 ** 9 else 'inf'} top-tier pairs: share {float(s.mean()):.3f}; mean score base {A['base'][s].mean():.4f}, forward {A['fwd'][s].mean():.4f}, mirror alone {A['mir_only'][s].mean():.4f}, both (T 0.5) {A['both_T0.5'][s].mean():.4f}")
    log("done")


if __name__ == "__main__":
    main()
