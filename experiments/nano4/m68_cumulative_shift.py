"""Section 4M M68 (2026-11-10; user: 再分析一次怎麼提升). Label-free cumulative composition of the test set at each of the 9 class cuts.
For cut k (classes <= k "lower"): observed share of test questions whose ct9 expected-score decision is <= k, against the share a
class-balanced set would give, from the CV members' out-of-fold decisions (class-balanced, n mix 5 / 10 / 20 = 25 / 25 / 50% as in M63).
Two-class BBSE per cut: a = P(decide lower | true lower), b = P(decide lower | true upper) (balanced within each side), so
pi_lower = (obs - b) / (a - b); uniform would be (k + 1) / 10. Question bootstrap (2000) for the CI; z of obs against the balanced
expectation with the binomial SE. Public, Private and all 800. The decisions are ct9's (before any bias push). Report only.
POST-HOC NOTE: the kyu | dan cut (A | 1D) was looked at because the M63 printout showed a kyu-heavy predicted share; the other 8 cuts
are exploratory. Read z with 9 cuts x 3 sets in mind.
  python -I logs/nano4/m68_cumulative_shift.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

MEMBERS = [(f"{t}-pc-f0-x40-oh", s) for t in ("C", "T") for s in (0, 1, 2)]
N_SHARE = {5: 0.25, 10: 0.25, 20: 0.5}
Ps, ys, ns = [], [], []
for v in FOLDS:
    y, n, _, _ = load_fold(v)
    Ps.append(np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t, s in MEMBERS], axis=0)); ys.append(y); ns.append(n)
P, y, n = np.concatenate(Ps), np.concatenate(ys), np.concatenate(ns)
d = np.asarray(decide(P), int)
w = np.zeros(len(y))
for c in range(10):
    for m, share in N_SHARE.items():
        s = (y == c) & (n == m); w[s] = share / max(s.sum(), 1)
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id")
Pt = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); dt = np.asarray(decide(Pt), int); ids = pr.index.to_numpy().astype(str)
halves = {"public": np.char.startswith(ids, "pub_"), "private": np.char.startswith(ids, "priv_"), "all": np.ones(len(ids), bool)}
rng = np.random.default_rng(20261110); out = {}
print("cut            | balanced exp |" + "".join(f" {h:>7s}: obs    z   pi_lower [90% CI]  uniform |" for h in halves))
for k in range(9):
    lo, up = y <= k, y > k
    a = float((w[lo] * (d[lo] <= k)).sum() / w[lo].sum()); b = float((w[up] * (d[up] <= k)).sum() / w[up].sum())
    u = (k + 1) / 10; m0 = u * a + (1 - u) * b
    row = {"a": a, "b": b, "balanced_expected": m0, "uniform_pi_lower": u}; line = f"{LEVELS[k]:>3s} | {LEVELS[k+1]:<3s}      |    {m0:.3f}     |"
    for h, mask in halves.items():
        x = (dt[mask] <= k).astype(float); N = len(x); obs = float(x.mean()); z = (obs - m0) / np.sqrt(m0 * (1 - m0) / N)
        boot = np.array([x[rng.integers(0, N, N)].mean() for _ in range(2000)]); pi = (obs - b) / (a - b); ci = ((np.percentile(boot, [5, 95]) - b) / (a - b)).tolist()
        row[h] = {"n": N, "obs": obs, "z": float(z), "pi_lower": float(pi), "ci90": ci}
        line += f"         {obs:.3f} {z:+5.1f}   {pi:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   {u:.1f}   |"
    out[f"{LEVELS[k]}|{LEVELS[k+1]}"] = row; print(line)
# what a kyu-vs-dan label shift implies for the A | 1D prior odds
r = out["A|1D"]["all"]; pk = r["pi_lower"]; rho = (pk / 4) / ((1 - pk) / 6)
print(f"kyu share of the test (all 800): {pk:.3f} [{r['ci90'][0]:.3f}, {r['ci90'][1]:.3f}] vs 0.400 uniform; implied per-class prior ratio kyu:dan = {rho:.2f} "
      f"[{(r['ci90'][0]/4)/((1-r['ci90'][0])/6):.2f}, {(r['ci90'][1]/4)/((1-r['ci90'][1])/6):.2f}]")
# per-class decided counts by half, against the balanced expectation of decided shares
exp = np.array([sum(w[(y == c)] @ (d[y == c] == j) for c in range(10)) / 10 for j in range(10)]); exp = exp / exp.sum()
for h, mask in halves.items():
    obs = np.bincount(dt[mask], minlength=10) / mask.sum(); zz = (obs - exp) / np.sqrt(exp * (1 - exp) / mask.sum())
    print(f"decided share {h:7s} " + " ".join(f"{lv} {o:.3f}({z_:+.1f})" for lv, o, z_ in zip(LEVELS, obs, zz)))
print("balanced expectation  " + " ".join(f"{lv} {e:.3f}" for lv, e in zip(LEVELS, exp)))
out["kyu_share_all"] = {"pi": pk, "ci90": r["ci90"], "rho": rho}
Path(ES / "m68_cumulative_shift.json").write_text(json.dumps(out, indent=1)); print("written")
