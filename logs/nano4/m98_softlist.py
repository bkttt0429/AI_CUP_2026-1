"""Section 4M M98 (2026-11-13): candidate-list evidence. The current anchor evidence of a game uses only the best candidate:
e(c) = (1 - a) + a * K(l_1 - c) / P0(l_1 - c). The right candidate is first in 40% of the held-out games but among the first 20 in
73%. Generalisation (record linkage with several candidate links / N-best rescoring): every candidate r of the list is the true
opponent with its own probability q_r, e(c) = 1 + sum_r q_r * (K(l_r - c) / P0_r(l_r - c) - 1). q_1, K and the rank-1 tables are
the current ones (pi0-corrected, m87). For the ranks 2..20, in rank buckets (2, 3-5, 6-10, 11-20): P0_r = the label-difference
distribution of that rank in the confident bands (there the first candidate is the opponent, the others are look-alikes); q_r per
group and band of the first candidate's margin by maximum likelihood on the label differences a * K + (1 - a) * P0_r; the sum of the
q's capped at 0.98. All cross-fitted over folds as the current evidence.
Stage "lists": top-20 candidates of every query under the SP2e fusion (weights of m84_SP2e_eval.json) -> m98_top_SP2e.npz.
Stage "eval" (pre-registered): bundle v5 with the list terms against bundle v5 (class-conditional rank-1 tables, encoder, tempering,
pi0), M37 on both seed sets; reported: list against top-1 in the plain framework, and the buckets added one by one.
  python -I logs/nano4/m98_softlist.py --stage lists | eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m84_variant as mv  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

RV, CACHE, TAG, M = mv.RV, mv.CACHE, "SP2e", 20
RB = ((1, 2), (2, 5), (5, 10), (10, 20)); RBN = ("rank 2", "ranks 3-5", "ranks 6-10", "ranks 11-20"); GRID = np.linspace(0.0, 0.4, 161); SM = b.SM


def stage_lists():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []
    for var, members in mv.MULTI[TAG]:
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))
    w = json.loads((RV / f"m84_{TAG}_eval.json").read_text())["weights"]; log(f"fusion weights {w}")

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
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; cand = np.zeros((len(idx), M), np.int16); sc = np.zeros((len(idx), M), np.float32)
    for i in range(0, len(idx), 2048):
        z, valid = zs(idx[i:i + 2048]); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)
        for k in range(len(z)):
            F = F + w[k + 1] * z[k]
        top = F.masked_fill(~valid, -float("inf")).topk(M, dim=1); cand[i:i + 2048] = top.indices.cpu().numpy(); sc[i:i + 2048] = top.values.cpu().numpy()
    o, to = slice(0, Q), slice(2 * Q, 2 * Q + T); old = np.load(RV / f"m84_{TAG}_val.npz"); same = float((plab[cand[o, 0]] == old["f_opp_lab"]).mean()); dm = float(np.abs((sc[o, 0] - sc[o, 1]) - old["f_opp_m"]).max())
    log(f"lists of {len(idx)} queries; check against the stored identifier output: first candidate's label equal {same:.6f}, largest margin difference {dm:.2e}"); assert same > 0.9999 and dm < 1e-2
    np.savez_compressed(RV / f"m98_top_{TAG}.npz", val_lab=plab[cand[o]].astype(np.int8), val_score=sc[o], val_cand=cand[o], test_lab=plab[cand[to]].astype(np.int8), test_score=sc[to], test_cand=cand[to]); log("written")


def list_terms(z, LAB, tr, cal, band_all, g_all, te, band, g, upto=len(RB)):
    """Extra evidence X [n_te, 10] = sum over the ranks 2.. of q_r * (K / P0_r - 1) and the summed q's, calibrated on `tr`."""
    y = z["y"]; nb = len(s2.QS) + 1; X = np.zeros((len(te), 10)); qs = np.zeros(len(te)); info = {}; lv = np.arange(10)
    for k in (0, 1, 2):
        K = cal[k]["K"]; s_te = g == k
        for rb, (lo, hi) in enumerate(RB[:upto]):
            Hh = np.zeros((nb, 19)); s_tr = tr & (g_all == k)
            for r in range(lo, hi):
                np.add.at(Hh, (band_all[s_tr], np.clip(LAB[s_tr, r] - y[s_tr] + 9, 0, 18)), 1)
            P0 = (Hh[5:].sum(0) + 1.0) / (Hh[5:].sum() + 19); a = np.zeros(nb)
            for bb in range(nb):
                if Hh[bb].sum() >= 50:
                    a[bb] = GRID[int(np.argmax([(Hh[bb] * np.log(u * K + (1 - u) * P0)).sum() for u in GRID]))]
            ratio = np.minimum(K / P0, 30.0); info[(k, rb)] = a
            for r in range(lo, hi):
                ar = a[band[s_te]][:, None]; X[s_te] += ar * (ratio[LAB[te][s_te, r][:, None] - lv[None, :] + 9] - 1.0); qs[s_te] += ar[:, 0]
    return X, qs, info


def stage_eval():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    V, _, key_pid, used = b.sources(["5e", TAG]); z = V[TAG]; LAB = np.load(RV / f"m98_top_{TAG}.npz")["val_lab"].astype(np.int64); assert len(LAB) == len(z["y"]) and (LAB[:, 0] == z["opp_lab"]).mean() > 0.9999
    F = {}; shown = False
    for v in FOLDS:
        tr = z["fold"] != v; edges, cal0, band_all, g_all = s2.fit(z, tr); _, cal, _ = m87.corrected(z, tr); R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
        band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; Ep = np.ones((len(te), 10)); Ec = np.ones((len(te), 10)); a1 = np.zeros(len(te))
        for k in (0, 1, 2):
            s_ = g == k; a = cal[k]["a"][band[s_]][:, None]; Ep[s_] = (1 - a) + a * cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; Ec[s_] = (1 - a) + a * R[k][:, lab[s_]].T; a1[s_] = a[:, 0]
        Xs = {}
        for upto in range(1, len(RB) + 1):
            X, qs, info = list_terms(z, LAB, tr, cal, band_all, g_all, te, band, g, upto); scale = np.where(a1 + qs > 0.98, np.maximum(0.98 - a1, 0) / np.maximum(qs, 1e-12), 1.0)[:, None]; Xs[upto] = X * scale
        if not shown:
            shown = True
            for rb in range(len(RB)):
                log(f"fold {v} calibration, even games: probability that one candidate of {RBN[rb]} is the opponent, by band of the first candidate's margin (low to high): " + " ".join(f"{x:.3f}" for x in info[(0, rb)]) + f" | first candidate: " + " ".join(f"{x:.2f}" for x in cal[0]["a"]))
            sh = np.bincount(band[g == 0], minlength=len(s2.QS) + 1) / max((g == 0).sum(), 1); tot1 = float((sh * cal[0]["a"]).sum()); totr = float(sum((sh * info[(0, rb)]).sum() * (hi - lo) for rb, (lo, hi) in enumerate(RB)))
            log(f"   even games of fold {v}: expected true-opponent mass per game: first candidate {tot1:.3f}, ranks 2-20 {totr:.3f}")
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; row = {k_: i for i, k_ in enumerate(z["key"][te])}; ee = b.evidence(V["enc"], v, 0, 1)
        variants = {"top1": np.log(Ep), "v5": np.log(Ec), **{f"list{u}": np.log(np.maximum(Ep + Xs[u], 1e-6)) for u in Xs}, "v5list": np.log(np.maximum(Ec + Xs[len(RB)], 1e-6)), "v5list2": np.log(np.maximum(Ec + Xs[2], 1e-6))}; L = {k_: np.zeros((nq, 10)) for k_ in variants}; Le = np.zeros((nq, 10))
        for i, g_ in enumerate(gk):
            rr = [row[k_] for k_ in g_ if k_ in row]
            for k_, lg in variants.items():
                if rr:
                    L[k_][i] = lg[rr].sum(0)
            for k_ in g_:
                if k_ in ee:
                    Le[i] += b.U_ENC * ee[k_]
        F[v] = dict(y=np.asarray(y), n=np.asarray(n), pid=np.asarray(pid), L={**{k_: L[k_] * Tn for k_ in ("top1", "list1", "list2", "list3", "list4")}, **{k_: (L[k_] + Le) * Tn for k_ in ("v5", "v5list", "v5list2")}})
    rep = {}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {}
        for v in FOLDS:
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); S[("base", v)] = SM[(P @ SM).argmax(1), F[v]["y"]]
            for k_, L in F[v]["L"].items():
                S[(k_, v)] = SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), F[v]["y"]]
        y, n, pid = (np.concatenate([F[v][k_] for v in FOLDS]) for k_ in ("y", "n", "pid")); res = {}
        for label, x, w_ in (("v5 with the list (ranks 2-20) vs v5", "v5list", "v5"), ("v5 with ranks 2-5 vs v5", "v5list2", "v5"), ("list (2-20) vs first only, plain", "list4", "top1"), ("  rank 2 only vs first only", "list1", "top1"), ("  ranks 2-5 vs first only", "list2", "top1"), ("  ranks 2-10 vs first only", "list3", "top1"), ("v5 vs base", "v5", "base"), ("first only (plain) vs base", "top1", "base")):
            A_, B_ = np.concatenate([S[(x, v)] for v in FOLDS]), np.concatenate([S[(w_, v)] for v in FOLDS]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(S[(x, v)], F[v]["y"], F[v]["n"], sizes=SIZES) - balanced_mean_over_n(S[(w_, v)], F[v]["y"], F[v]["n"], sizes=SIZES) for v in FOLDS]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label.strip()] = {**dd, "per_fold": pf, "pass": bool(good)}
            log(f"  seeds {seeds} {label:40s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x_:+.4f}" for m_, x_ in dd["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} -> {'PASS' if good else 'no'}")
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["v5 with the list (ranks 2-20) vs v5"]["pass"])
    rep["passed_cv"] = bool(all(ok)); log(f"PRE-REGISTERED: bundle v5 with the candidate list against bundle v5: {'PASS' if rep['passed_cv'] else 'NO'}"); (RV / "m98_softlist.json").write_text(json.dumps(rep, indent=1, default=float)); log("written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("lists", "eval")); a_ = ap.parse_args()
    stage_lists() if a_.stage == "lists" else stage_eval()
