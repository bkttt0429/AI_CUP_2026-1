"""Section 4M M103b (2026-11-13, report only): the same five-recipe vote on the two kinds of decisions the current composition never
touches. (1) REVERSALS: questions where ct44's recipe moved the base decision; the five later recipes (SP, SP2e, S2, S3c, S3; test
rules) may go back to the base decision or elsewhere. By the number of recipes that leave ct44's answer for the same other answer:
how often that answer is better than ct44's, mean gain. (2) OUTSIDE THE BOUNDARY REGION: questions the region rule freezes at the
base decision; the recipes' unrestricted decisions (all levels) and the number that agree on the same change: precision, mean gain.
Composition rules built from them, against the ct75s rule (shared with SP, no reversal, region only).
  python -I logs/nano4/m103b_votes.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; FIVE = ("v3", "v5", "s2", "s3c", "s3")


def main():
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); Fd = {}
    for v in FOLDS:
        y, n, pid, q = m3.load_fold(v); gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)
    for seeds in ((0, 1, 2), (6, 7, 8)):
        rv = {k: [] for k in ("cnt", "gain")}; ro = {k: [] for k in ("cnt", "gain")}; S = {k: [] for k in ("ct75s rule", "+ reversal when 5 agree", "+ reversal when >= 4 agree", "+ outside the region when 5 agree", "+ outside the region when >= 4 agree", "+ both when 5 agree")}; ys, ns, ps = [], [], []
        for v in FOLDS:
            y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR, dA = {}, {}
            for name in REC:
                d0, dA[name], dR[name], _, region = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
            d44 = dR["v44"]; keep = d44 != d0; base_rule = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44))  # ct75s rule
            # (1) reversals: among the questions ct44's recipe changed, the modal answer of the five that differs from ct44's
            votes = np.stack([dR[k] for k in FIVE]); alt = d44.copy(); cnt = np.zeros(len(y), int)
            for c in range(10):
                k_ = (votes == c).sum(0); better = (c != d44) & (k_ > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k_, cnt)
            s = keep & (cnt >= 1); rv["cnt"].append(cnt[s]); rv["gain"].append(SM[alt[s], y[s]] - SM[d44[s], y[s]]); rev5 = np.where(keep & (cnt >= 5), alt, base_rule); rev4 = np.where(keep & (cnt >= 4), alt, base_rule)
            # (2) outside the region: unrestricted decisions of the five
            va = np.stack([dA[k] for k in FIVE]); alt2 = d0.copy(); cnt2 = np.zeros(len(y), int)
            for c in range(10):
                k_ = (va == c).sum(0); better = (c != d0) & (k_ > cnt2); alt2 = np.where(better, c, alt2); cnt2 = np.where(better, k_, cnt2)
            s2_ = ~region & ~keep & (cnt2 >= 1); ro["cnt"].append(cnt2[s2_]); ro["gain"].append(SM[alt2[s2_], y[s2_]] - SM[d0[s2_], y[s2_]]); out5 = np.where(~region & ~keep & (cnt2 >= 5), alt2, base_rule); out4 = np.where(~region & ~keep & (cnt2 >= 4), alt2, base_rule)
            both5 = np.where(keep & (cnt >= 5), alt, out5)
            for k, d in (("ct75s rule", base_rule), ("+ reversal when 5 agree", rev5), ("+ reversal when >= 4 agree", rev4), ("+ outside the region when 5 agree", out5), ("+ outside the region when >= 4 agree", out4), ("+ both when 5 agree", both5)):
                S[k].append(SM[d, y])
            ys.append(y); ns.append(n); ps.append(pid)
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps); print(f"seeds {seeds}: {len(y)} questions")
        for title, r in (("(1) reversals of ct44-recipe changes, by the number of the five recipes that prefer the same other answer", rv), ("(2) questions outside the boundary region, by the number of the five recipes whose unrestricted decision is the same change", ro)):
            c_, g_ = np.concatenate(r["cnt"]), np.concatenate(r["gain"]); print(f"  {title}:")
            for k in range(1, 6):
                s = c_ == k
                if s.sum():
                    print(f"      {k} of 5: {int(s.sum()):5d} questions ({s.sum() / len(y):.4f}); better {float((g_[s] > 0).mean()):.3f}, worse {float((g_[s] < 0).mean()):.3f}; mean gain {float(g_[s].mean()):+.3f}; total {float(g_[s].sum()):+.1f}")
        print("  composition rules against the ct75s rule:")
        for k in S:
            if k == "ct75s rule":
                continue
            A_ = np.concatenate(S[k]); B_ = np.concatenate(S["ct75s rule"]); d = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(a_, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(c_, Fd[v][0], Fd[v][1], sizes=SIZES) for a_, c_, v in zip(S[k], S["ct75s rule"], FOLDS)]
            print(f"      {k:38s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" | decisions changed {float((A_ != B_).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
