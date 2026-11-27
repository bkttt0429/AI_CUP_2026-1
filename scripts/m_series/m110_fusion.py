"""Section 4M M110 (2026-11-14): how much of the external-rank gain of M107 is left at the level of the base ensemble.
The probe's question probabilities (m107 --save-proba) are pooled into the base ensemble, P_f ~ P_base x P_probe^w, w cross-fitted
on fold halves (seeds 0-2; seeds 6-8 reuse the picks). Registered test: base + B (with external players) against base + A (same
architecture, competition players only), M37. Report: each against the base alone; the larger-body pair and lambda 3; and the
same differences after the ct75u rule (anchors unchanged) is applied on top of the pooled probabilities.
  python -I scripts/m_series/m110_fusion.py [--proba results/r_variants/m107_proba.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m103_stability as m3  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0); ARMS = ("A", "B", "A_big", "B_big", "B_lambda_3")


def rule75u(P, Ls):
    """Base decision and the ct75u rule's decision for base probabilities P (as m109_error_budget.py)."""
    dR, dA = {}, {}
    for name in REC:
        d0, dA[name], dR[name], _, region = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
    d44 = dR["v44"]; keep = d44 != d0; d = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44))
    for votes, ref, mask in ((np.stack([dR[k] for k in FIVE]), d44, keep), (np.stack([dA[k] for k in FIVE]), d0, ~region & ~keep)):
        alt = ref.copy(); cnt = np.zeros(len(P), int)
        for c in range(10):
            k_ = (votes == c).sum(0); better = (c != ref) & (k_ > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k_, cnt)
        s = mask & (cnt >= 5); d[s] = alt[s]
    return d0, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--proba", default="results/r_variants/m107_proba.npz"); a_ = ap.parse_args(); Z = np.load(ROOT / a_.proba)
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); Fd = {}
    for v in FOLDS:
        y, n, pid, q = m3.load_fold(v); gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)
    half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; picks = None; rep = {}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S0, SU = {}, {}  # base-level and after-the-ct75u-rule scores, keyed (arm, w) -> list over folds
        for v in FOLDS:
            y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); d0, du = rule75u(P, Ls); S0.setdefault(("base", 0), []).append(SM[d0, y]); SU.setdefault(("base", 0), []).append(SM[du, y])
            for arm in ARMS:
                Q = np.log(np.clip(Z[f"{arm}__{v}"].astype(np.float64), 1e-9, None)); assert len(Q) == len(P)
                for w in WS:
                    Pf = P * np.exp(w * (Q - Q.max(1, keepdims=True))); Pf /= Pf.sum(1, keepdims=True); d0, du = rule75u(Pf, Ls); S0.setdefault((arm, w), []).append(SM[d0, y]); SU.setdefault((arm, w), []).append(SM[du, y])
        ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); fs = lambda S, k: np.array([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731
        if picks is None:
            picks = {}
            for arm in ARMS:
                best = lambda fl: max(WS, key=lambda w: fs(S0, (arm, w))[[fi[v] for v in fl]].mean())  # noqa: E731
                picks[arm] = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}
            print("weights picked on the other half of the folds (seeds 0-2): " + "  ".join(f"{a}: {[picks[a][v] for v in FOLDS]}" for a in ARMS))
            for arm in ARMS:
                print(f"    {arm:11s} base-level score by weight: " + "  ".join(f"w={w}: {fs(S0, (arm, w)).mean():.4f}" for w in WS))
        cf = lambda S, arm: [S[(arm, picks[arm][v])][fi[v]] for v in FOLDS] if arm != "base" else S[("base", 0)]  # noqa: E731
        print(f"\n=== seeds {seeds}: base ensemble {fs(S0, ('base', 0)).mean():.4f}; after the ct75u rule {fs(SU, ('base', 0)).mean():.4f} ===")

        def cmp(S, new, old, tag):
            A_, B_ = cf(S, new), cf(S, old); d = paired_diff(np.concatenate(A_), np.concatenate(B_), y, pid, n, sizes=SIZES, n=3000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(A_, B_, ys, ns)]
            print(f"  [{tag}] base+{new:10s} vs {('base+' + old) if old != 'base' else 'base':15s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True); rep[f"{seeds[0]}|{tag}|{new} vs {old}"] = {**d, "per_fold": pf}; return d, pf
        d, pf = cmp(S0, "B", "A", "base level"); ok.append((d["ci95"][0] > 0) and (d["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3))
        for new, old in (("B", "base"), ("A", "base"), ("B_big", "A_big"), ("B_big", "base"), ("A_big", "base"), ("B_lambda_3", "A"), ("B_lambda_3", "base")):
            cmp(S0, new, old, "base level")
        for new, old in (("B", "A"), ("B", "base"), ("B_big", "A_big"), ("B_big", "base"), ("B_lambda_3", "base")):
            cmp(SU, new, old, "after ct75u rule")
    passed = bool(all(ok)); print(f"\nREGISTERED TEST (base + B vs base + A at the base level, M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep["passed"] = passed; rep["picks"] = {a: {str(v): w for v, w in picks[a].items()} for a in ARMS}
    (ROOT / "results/r_variants/m110_fusion.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
