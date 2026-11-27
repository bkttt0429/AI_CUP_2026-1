"""Section 4M M109 (2026-11-14, report only): where the current pipeline loses its points, in CV under the test rules.
Decisions: base ensemble, the ct44 rule, the ct75s rule, the ct75u rule (as m103b_votes.py). Per question the official score is
1 (exact), e^-1 (one level off), 0 (two or more). Tables: the ladder; by question size with a 1/n extrapolation (what the same
method would reach with unlimited games per question); error types and their share of the lost points; by true level; the model's
own belief (posterior with the v5 recipe's evidence): calibration by confidence band, and how often the truth is its first / second
/ third choice (what a perfect tie-breaker could gain); what the anchors touch.
  python -I logs/nano4/m109_error_budget.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; FIVE = ("v3", "v5", "s2", "s3c", "s3"); E1 = math.exp(-1.0)


def main():
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); Fd = {}
    for v in FOLDS:
        y, n, pid, q = m3.load_fold(v); gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)
    rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        A = {k: [] for k in ("y", "n", "d0", "d44", "d75s", "d75u", "P", "Pa", "region")}
        for v in FOLDS:
            y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR, dA = {}, {}
            for name in REC:
                d0, dA[name], dR[name], _, region = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
            d44 = dR["v44"]; keep = d44 != d0; d75s = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); d75u = d75s.copy()
            for votes, ref, mask in ((np.stack([dR[k] for k in FIVE]), d44, keep), (np.stack([dA[k] for k in FIVE]), d0, ~region & ~keep)):
                alt = ref.copy(); cnt = np.zeros(len(y), int)
                for c in range(10):
                    k_ = (votes == c).sum(0); better = (c != ref) & (k_ > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k_, cnt)
                s = mask & (cnt >= 5); d75u[s] = alt[s]
            L = Ls["v5"]; Pa = P * np.exp(L - L.max(1, keepdims=True)); Pa /= Pa.sum(1, keepdims=True)
            for k, x in (("y", y), ("n", n), ("d0", d0), ("d44", d44), ("d75s", d75s), ("d75u", d75u), ("P", P), ("Pa", Pa), ("region", region)):
                A[k].append(x)
        y, n, d0, d44, d75s, d75u, P, Pa, region = (np.concatenate(A[k]) for k in ("y", "n", "d0", "d44", "d75s", "d75u", "P", "Pa", "region")); bal = lambda d, sz=SIZES: float(balanced_mean_over_n(SM[d, y], y, n, sizes=sz))  # noqa: E731
        print(f"\n=== seeds {seeds}: {len(y)} validation questions ===")
        print("ladder (official score): " + "  ".join(f"{k} {bal(d):.4f}" for k, d in (("base", d0), ("ct44 rule", d44), ("ct75s rule", d75s), ("ct75u rule", d75u))))
        sc = {m: bal(d75u, (m,)) for m in SIZES}; sb = {m: bal(d0, (m,)) for m in SIZES}; x = np.array([1 / m for m in SIZES]); fit = np.polyfit(x, [sc[m] for m in SIZES], 1); fitb = np.polyfit(x, [sb[m] for m in SIZES], 1)
        print("by question size: " + "  ".join(f"n={m}: final {sc[m]:.4f} (base {sb[m]:.4f})" for m in SIZES) + f" | straight line in 1/n: unlimited games -> final {fit[1]:.4f}, base {fitb[1]:.4f}")
        err = np.abs(d75u - y); ex, o1, o2 = float((err == 0).mean()), float((err == 1).mean()), float((err >= 2).mean()); lost = o1 * (1 - E1) + o2
        print(f"error types (final): exact {ex:.3f}, one level off {o1:.3f}, two or more {o2:.3f} | lost points per question {lost:.3f}: {o1 * (1 - E1) / lost:.0%} from one-level misses, {o2 / lost:.0%} from misses of two or more")
        for m in SIZES:
            s = n == m; print(f"    n={m}: exact {float((err[s] == 0).mean()):.3f}, one off {float((err[s] == 1).mean()):.3f}, two or more {float((err[s] >= 2).mean()):.3f}")
        e = P @ np.arange(10) - y; var = {m: float(np.mean([e[(n == m) & (y == c)].var() for c in range(10)])) for m in SIZES}; fv = np.polyfit(x, [var[m] for m in SIZES], 1)
        print("base model, spread of (expected level - label) within a label: " + "  ".join(f"n={m}: sd {math.sqrt(var[m]):.3f}" for m in SIZES) + f" | variance = {fv[1]:.3f} + {fv[0]:.2f}/n -> with unlimited games sd {math.sqrt(max(fv[1], 0)):.3f} (the part more games cannot remove)")
        print("by true level (final): " + "  ".join(f"{LEVELS[c]} {float(SM[d75u[y == c], y[y == c]].mean()):.3f}/{float((d75u[y == c] == c).mean()):.2f}/{float((d75u[y == c] - c).mean()):+.2f}" for c in range(10)) + "   (score / exact rate / mean signed error)")
        E = Pa @ mb.S.T; order = np.argsort(-E, 1); r = np.arange(len(y)); c1 = E[r, order[:, 0]]; s1 = SM[order[:, 0], y]; s2_ = np.maximum(s1, SM[order[:, 1], y]); s3_ = np.maximum(s2_, SM[order[:, 2], y])
        print(f"the model's own belief (posterior with the v5 evidence): truth is its first choice {float((order[:, 0] == y).mean()):.3f}, first or second {float(((order[:, 0] == y) | (order[:, 1] == y)).mean()):.3f}, among the first three {float((order[:, :3] == y[:, None]).any(1).mean()):.3f}")
        print(f"    score if a perfect tie-breaker chose among its first two choices {float(s2_.mean()):.4f}, among three {float(s3_.mean()):.4f} (its own first choice {float(s1.mean()):.4f}; unweighted means)")
        gap = c1 - E[r, order[:, 1]]; qs = np.quantile(gap, [0.25, 0.5, 0.75]); bands = np.searchsorted(qs, gap)
        print("    by the gap between its first and second choice (quartiles, smallest gap first): " + "  ".join(f"Q{k + 1}: expects {float(c1[bands == k].mean()):.3f}, gets {float(s1[bands == k].mean()):.3f}, share of lost points {float((1 - s1[bands == k]).sum() / (1 - s1).sum()):.0%}" for k in range(4)))
        ch = d75u != d0; g = SM[d75u, y] - SM[d0, y]
        print(f"anchors: questions whose answer differs from the base {float(ch.mean()):.3f}; of those better {float((g[ch] > 0).mean()):.3f}, worse {float((g[ch] < 0).mean()):.3f}, mean gain {float(g[ch].mean()):+.3f}; inside the boundary region {float(region.mean()):.3f} of all questions; by size: " + "  ".join(f"n={m} changed {float(ch[n == m].mean()):.3f}" for m in SIZES), flush=True)
        rep[f"seeds_{seeds[0]}"] = {"ladder": {"base": bal(d0), "ct44": bal(d44), "ct75s": bal(d75s), "ct75u": bal(d75u)}, "by_n": sc, "by_n_base": sb, "limit_final": float(fit[1]), "limit_base": float(fitb[1]), "exact": ex, "off1": o1, "off2": o2, "var_fit": [float(fv[1]), float(fv[0])]}
    (ROOT / "results/r_variants/m109_error_budget.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
