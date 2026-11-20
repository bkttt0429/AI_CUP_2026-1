"""Section 4M M96 (2026-11-13, report only): the fine split of "two players paired as equals share a label only half the time".
The pairing graph inside the training set: the opponent side of every row is matched against the 7,748 training players under the
all-players network (three seeds, owner masked, per-candidate standardisation, as M92); rows with a margin at or above q90 are
identified pairs (owner B, opponent C) with weight a = the tier's true-pair share (label fit, cross-fitted by the owner's fold).
Even games only.
(1) personal offset: consensus of a player's opponents' labels against his own label; between-player share of the variance.
(2) position: by decile of the row in the owner's game order (game_id): mean and agreement of the opponent's label.
(3) label definition: regression of the owner's label on the early / middle / late opponent means; which summary is closest.
(4) opponent profiles: evidence for the owner's label from each identified opponent C, generic table K(L_C - c) against C's own
    profile pi_C(c) = the label distribution of C's other identified opponents (either role), shrunk to K (kappa 8), the predicted
    owner's own pair left out. Same anchors, same weights, same denominators; exact / within-one / score by the number of anchors.
Writes results/r_variants/m96_pairs.npz (owner, opp, tier, weight, pos) and m96_block1.json.
  python -I logs/nano4/m96_block1.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m86_bundle as b  # noqa: E402
import m91_mirror as mm  # noqa: E402

TQ = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98]); NT = len(TQ) + 1; SM = b.SM; KAPPA = 8.0


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; D = mm.load_meta(log); N = len(D["key"]); C = len(D["players"]); own_t = torch.as_tensor(D["owner"].copy(), device=dev); cnt = np.bincount(D["owner"], minlength=C); cand = np.flatnonzero(cnt >= 8); col = np.full(C, -1, np.int64); col[cand] = np.arange(len(cand)); col_t = torch.as_tensor(col, device=dev)
    PS, MU, SD, EO = [], [], [], []; chunk = 50_000
    for s in (0, 1, 2):
        z = np.load(mm.EMB / f"emb_-1{f'_s{s}' if s else ''}.npz"); eo = torch.as_tensor(z["opp"].astype(np.float32), device=dev); ew = torch.as_tensor(z["own"].astype(np.float32), device=dev); del z
        P = torch.nn.functional.normalize(torch.zeros((C, 128), device=dev).index_add_(0, own_t, ew)[torch.as_tensor(cand, device=dev)], dim=1); del ew; s1 = torch.zeros(len(cand), device=dev, dtype=torch.float64); s2_ = torch.zeros_like(s1)
        for i in range(0, N, chunk):
            S = (eo[i:i + chunk] @ P.T).double(); s1 += S.sum(0); s2_ += (S * S).sum(0)
        mu = s1 / N; PS.append(P); MU.append(mu.float()); SD.append((s2_ / N - mu * mu).clamp(min=1e-12).sqrt().float() + 1e-6); EO.append(eo)
    J = np.zeros(N, np.int64); MG = np.zeros(N, np.float32)
    for i in range(0, N, chunk):
        Z = sum((EO[k][i:i + chunk] @ PS[k].T - MU[k]) / SD[k] for k in range(3)) / 3; r = torch.arange(i, min(i + chunk, N), device=dev); c = col_t[own_t[r]]; ok = c >= 0; Z[torch.arange(len(r), device=dev)[ok], c[ok]] = -float("inf"); t = Z.topk(2, dim=1)
        J[i:i + chunk] = t.indices[:, 0].cpu().numpy(); MG[i:i + chunk] = (t.values[:, 0] - t.values[:, 1]).cpu().numpy()
    del EO, PS; opp = cand[J]; ylab = np.full(C, -1, np.int64); ylab[D["owner"]] = D["lab"]; pf = np.full(C, -9, np.int64); pf[D["owner"]] = D["fold"]; g = D["g_opp"]; edges = np.quantile(MG[g >= 0], TQ); tier = np.searchsorted(edges, MG)
    # keys look like "A:g_0036743" (fixed 2026-11-13: the first version did not parse them and fell back to the loader order)
    gid = np.array([int(k.rsplit("_", 1)[1]) for k in D["key"]]); assert (gid > 0).all(); order = np.lexsort((gid, D["owner"])); pos = np.zeros(N); rk = np.arange(N) - np.searchsorted(D["owner"][order], D["owner"][order], side="left"); pos[order] = (rk + 0.5) / cnt[D["owner"][order]]
    ev = g == 0; d = D["lab"] - ylab[opp]  # owner minus identified opponent
    # true-pair share per tier (label fit against the lowest tier), all even rows
    h = np.zeros((NT, 19)); np.add.at(h, (tier[ev], np.clip(d[ev] + 9, 0, 18)), 1); P0 = (h[0] + 1) / (h[0].sum() + 19); K = (h[NT - 1] + 0.5) / (h[NT - 1].sum() + 9.5); grid = np.linspace(0, 0.98, 99)
    a = np.array([0.0] + [grid[int(np.argmax([(h[x] * np.log(u * K + (1 - u) * P0)).sum() for u in grid]))] for x in range(1, NT)]); log("even games: true-pair share by tier " + " ".join(f"{x:.2f}" for x in a) + " | K(-2..2) " + " ".join(f"{K[9 + x]:.3f}" for x in (-2, -1, 0, 1, 2)) + " | P0(-2..2) " + " ".join(f"{P0[9 + x]:.3f}" for x in (-2, -1, 0, 1, 2)))
    sel = ev & (tier >= 4); ow, op, tr_, w, ps = D["owner"][sel], opp[sel], tier[sel], a[tier[sel]], pos[sel]; dd = (ylab[op] - ylab[ow]).astype(float)  # opponent minus owner: the anchor's reading of the owner
    np.savez(mm.RV / "m96_pairs.npz", owner=ow, opp=op, tier=tr_, weight=w, pos=ps); log(f"identified even-game rows at >= q90: {len(ow)} (tiers q90 / q95 / q98: {int((tr_ == 4).sum())} / {int((tr_ == 5).sum())} / {int((tr_ == 6).sum())}); owners {len(np.unique(ow))}, opponents {len(np.unique(op))}")
    rep = {"a": a.tolist(), "K": K.tolist()}
    # (1) personal offset, from the two best tiers
    top = tr_ >= 5; n_b = np.bincount(ow[top], minlength=C); s_b = np.bincount(ow[top], weights=dd[top], minlength=C); q_b = np.bincount(ow[top], weights=dd[top] ** 2, minlength=C)
    for kmin in (5, 10, 20):
        m = n_b >= kmin; off = s_b[m] / n_b[m]; within = (q_b[m] / n_b[m] - off ** 2) * n_b[m] / (n_b[m] - 1); vb = off.var() - (within / n_b[m]).mean(); mode_ok = []
        for p_ in np.flatnonzero(m):
            x = ylab[op[top & (ow == p_)]]; mode_ok.append(np.bincount(x, minlength=10).argmax() - ylab[p_])
        mode_ok = np.array(mode_ok); log(f"(1) players with >= {kmin} top-tier anchors: {int(m.sum())}; mean offset (opponents' labels minus own) {off.mean():+.3f}, sd of the personal offset {np.sqrt(max(vb, 0)):.3f} (raw sd of player means {off.std():.3f}); single-anchor sd within a player {np.sqrt(within.mean()):.3f}; between-player share of the single-anchor variance {max(vb, 0) / (max(vb, 0) + within.mean()):.3f}; rounded mean equals own label {float((np.rint(off) == 0).mean()):.3f}, within one {float((np.abs(np.rint(off)) <= 1).mean()):.3f}; mode of the opponents' labels equals own label {float((mode_ok == 0).mean()):.3f}, within one {float((np.abs(mode_ok) <= 1).mean()):.3f}")
        rep[f"offset_k{kmin}"] = {"n": int(m.sum()), "sd_offset": float(np.sqrt(max(vb, 0))), "sd_within": float(np.sqrt(within.mean())), "mode_exact": float((mode_ok == 0).mean())}
    # (2) position in the owner's game order
    dec = np.minimum((ps * 10).astype(int), 9); log("(2) by decile of the row in the owner's game order (top-tier anchors): mean of opponent minus owner / share equal / share within one: " + " | ".join(f"{float(dd[top & (dec == k)].mean()):+.3f} / {float((dd[top & (dec == k)] == 0).mean()):.3f} / {float((np.abs(dd[top & (dec == k)]) <= 1).mean()):.3f}" for k in range(10)))
    rep["position"] = [[float(dd[top & (dec == k)].mean()), float((dd[top & (dec == k)] == 0).mean())] for k in range(10)]
    # (3) label definition: early / middle / late thirds
    th = np.minimum((ps * 3).astype(int), 2); rows_ = []
    for p_ in np.flatnonzero(n_b >= 15):
        s_ = top & (ow == p_); x = ylab[op[s_]].astype(float); t_ = th[s_]
        if all((t_ == k).sum() >= 3 for k in range(3)):
            rows_.append([ylab[p_]] + [x[t_ == k].mean() for k in range(3)] + [x.mean(), np.bincount(x.astype(int), minlength=10).argmax(), x.max(), x.min(), np.median(x)])
    R = np.array(rows_); A = np.c_[np.ones(len(R)), R[:, 1:4]]; coef = np.linalg.lstsq(A, R[:, 0], rcond=None)[0]
    log(f"(3) {len(R)} players with >= 3 top-tier anchors in each third of their games: regression of the label on the early / middle / late means: intercept {coef[0]:+.3f}, weights {coef[1]:.3f} / {coef[2]:.3f} / {coef[3]:.3f}; exact match of the rounded summary with the label: early {float((np.rint(R[:, 1]) == R[:, 0]).mean()):.3f}, middle {float((np.rint(R[:, 2]) == R[:, 0]).mean()):.3f}, late {float((np.rint(R[:, 3]) == R[:, 0]).mean()):.3f}, mean {float((np.rint(R[:, 4]) == R[:, 0]).mean()):.3f}, mode {float((R[:, 5] == R[:, 0]).mean()):.3f}, max {float((R[:, 6] == R[:, 0]).mean()):.3f}, min {float((R[:, 7] == R[:, 0]).mean()):.3f}, median {float((np.rint(R[:, 8]) == R[:, 0]).mean()):.3f}; mean of late minus early {float((R[:, 3] - R[:, 1]).mean()):+.3f}")
    rep["definition"] = {"n": int(len(R)), "weights": coef.tolist()}
    # (4) opponent profiles against the generic table. Every identified row links two players; from a player's point of view
    # the "other" player's label is one observation of whom he is paired with.
    u = np.concatenate([ow, op]); v_ = np.concatenate([op, ow]); ww = np.concatenate([w, w]); Hc = np.zeros((C, 10)); np.add.at(Hc, (u, ylab[v_]), ww)  # Hc[C, c] = weighted count of C's partners with label c
    Ks = (h[4:].sum(0) + 0.5); Ks = (Ks + Ks[::-1]) / 2; Ks = Ks / Ks.sum(); P0s = (P0 + P0[::-1]) / 2  # symmetric tables over the label difference
    res = {k: [] for k in ("y", "n", "gen", "pro", "pro_w")}; owners = np.unique(ow[top]); o_sort = np.argsort(ow, kind="stable"); starts = np.searchsorted(ow[o_sort], owners); ends = np.searchsorted(ow[o_sort], owners, side="right"); lv = np.arange(10)
    for p_, s0, s1_ in zip(owners, starts, ends):
        idx = o_sort[s0:s1_]; Lg = np.zeros(10); Lp = np.zeros(10); y = ylab[p_]; k_top = 0
        pair_w = np.zeros(C); np.add.at(pair_w, op[idx], w[idx])  # this owner's own weight inside each opponent's profile (to leave out); his rows as the opponent's partner are found the same way
        back = (op == p_); np.add.at(pair_w, ow[back], w[back])
        for i in idx:
            c_ = op[i]; lc = ylab[c_]; ai = w[i]; kd = Ks[lc - lv + 9]; p0 = P0s[lc - lv + 9]; hc = Hc[c_].copy(); hc[y] -= pair_w[c_]; hc = np.maximum(hc, 0.0); pi = (hc + KAPPA * kd / kd.sum()) / (hc.sum() + KAPPA); kd_n = kd / kd.sum()
            Lg += np.log((1 - ai) + ai * np.minimum(kd_n / p0, 30)); Lp += np.log((1 - ai) + ai * np.minimum(pi / p0, 30)); k_top += tr_[i] >= 5
        res["y"].append(y); res["n"].append(k_top); res["gen"].append(int(Lg.argmax())); res["pro"].append(int(Lp.argmax()))
    R4 = {k: np.array(x) for k, x in res.items() if x}; y, n_, gen, pro = R4["y"], R4["n"], R4["gen"], R4["pro"]
    for lo, hi in ((1, 1), (2, 3), (4, 9), (10, 19), (20, 10 ** 9), (1, 10 ** 9)):
        s = (n_ >= lo) & (n_ <= hi)
        if s.sum() > 30:
            f = lambda x: f"exact {float((x[s] == y[s]).mean()):.3f}, within one {float((np.abs(x[s] - y[s]) <= 1).mean()):.3f}, score {float(SM[x[s], y[s]].mean()):.4f}"  # noqa: E731
            se = float(np.std(SM[pro[s], y[s]] - SM[gen[s], y[s]]) / np.sqrt(s.sum())); log(f"(4) owners with {lo}-{hi if hi < 10 ** 9 else 'inf'} top-tier anchors ({int(s.sum())}): generic table: {f(gen)} | opponent profiles: {f(pro)} | score difference {float((SM[pro[s], y[s]] - SM[gen[s], y[s]]).mean()):+.4f} (se {se:.4f}); answers differ {float((pro[s] != gen[s]).mean()):.3f}")
            rep[f"profiles_{lo}_{hi}"] = {"n": int(s.sum()), "generic_exact": float((gen[s] == y[s]).mean()), "profile_exact": float((pro[s] == y[s]).mean()), "score_diff": float((SM[pro[s], y[s]] - SM[gen[s], y[s]]).mean()), "se": se}
    (mm.RV / "m96_block1.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
