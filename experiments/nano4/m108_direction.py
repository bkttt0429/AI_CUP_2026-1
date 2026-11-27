"""Section 4M M108 (2026-11-14, CV): the unanimous reversals of M104 (the five recipes' region decisions all equal the same answer
other than the ct44 recipe's) split by direction: UP = the ct44 recipe's answer is ABOVE the agreed answer (it had moved the
question up; the reversal takes it down), DOWN = below. Per group and seed set: questions, better / worse, mean gain per question
with its standard error, change of the exact and of the within-one count per question. Report only: the ct44 recipe's own changes
by direction (against the base decision), all and those not reversed.
  python -I logs/nano4/m108_direction.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; FIVE = ("v3", "v5", "s2", "s3c", "s3")


def line(name, new, old, y, s):
    g = SM[new[s], y[s]] - SM[old[s], y[s]]; k = int(s.sum())
    if not k:
        return f"      {name:44s} none", None
    ex = (new[s] == y[s]).astype(int) - (old[s] == y[s]).astype(int); w1 = (np.abs(new[s] - y[s]) <= 1).astype(int) - (np.abs(old[s] - y[s]) <= 1).astype(int)
    return f"      {name:44s} {k:5d} questions; better {float((g > 0).mean()):.3f}, worse {float((g < 0).mean()):.3f}; gain per question {float(g.mean()):+.3f} (se {float(g.std() / np.sqrt(k)):.3f}); exact {float(ex.mean()):+.3f}, within one {float(w1.mean()):+.3f} per question", float(g.mean())


def main():
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); Fd = {}; rep = {}
    for v in FOLDS:
        y, n, pid, q = m3.load_fold(v); gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Fd[v] = (np.asarray(y), np.asarray(n), Ls)
    for seeds in ((0, 1, 2), (6, 7, 8)):
        Y, D0, D44, ALT, CNT = [], [], [], [], []
        for v in FOLDS:
            y, n, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR = {}
            for name in REC:
                d0, _, dR[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
            d44 = dR["v44"]; votes = np.stack([dR[k] for k in FIVE]); alt = d44.copy(); cnt = np.zeros(len(y), int)
            for c in range(10):
                k_ = (votes == c).sum(0); better = (c != d44) & (k_ > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k_, cnt)
            Y.append(y); D0.append(d0); D44.append(d44); ALT.append(alt); CNT.append(cnt)
        y, d0, d44, alt, cnt = (np.concatenate(x) for x in (Y, D0, D44, ALT, CNT)); keep = d44 != d0; una = keep & (cnt >= 5); print(f"seeds {seeds}: {len(y)} questions; changed by the ct44 recipe {int(keep.sum())}; unanimous reversals {int(una.sum())} (back to the base decision: {float((alt[una] == d0[una]).mean()):.3f})")
        r = {}
        for name, s in (("unanimous reversals, all", una), ("  UP: ct44 recipe above the agreed answer", una & (d44 > alt)), ("  DOWN: ct44 recipe below the agreed answer", una & (d44 < alt))):
            t, g = line(name, alt, d44, y, s); print(t); r[name.strip()] = g
        print("    report only - the ct44 recipe's own changes against the base decision:")
        for name, s in (("all, moved up", keep & (d44 > d0)), ("all, moved down", keep & (d44 < d0)), ("not unanimously reversed, moved up", keep & ~una & (d44 > d0)), ("not unanimously reversed, moved down", keep & ~una & (d44 < d0))):
            print(line(name, d44, d0, y, s)[0])
        rep[f"seeds_{seeds[0]}"] = r; sys.stdout.flush()
    up = [rep[k]["UP: ct44 recipe above the agreed answer"] for k in rep]; rule = "(i) UP <= 0 in both seed sets -> candidate ct75d (ct75u without the UP reversals); the user decides" if all(u <= 0 for u in up) else ("(ii) UP >= +0.05 in both seed sets -> the three Public questions are read as noise; ct75u stays" if all(u >= 0.05 for u in up) else "(iii) neither -> ct75u stays; reported as is")
    print(f"PRE-WRITTEN HANDLING: UP group gain per question {up[0]:+.3f} / {up[1]:+.3f} -> {rule}"); (ROOT / "results/r_variants/m108_direction.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
