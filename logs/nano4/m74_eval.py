"""Section 4M M74 part A evaluation (2026-11-10): members whose tokens carry the strong net's state columns (-ai) against the standard ones.
Primary (pre-registered): 3C-ai + 3T-ai vs 3C + 3T, seeds 0-2; replication seeds 3-5 vs standard seeds 3-5; M37 rule in both.
Report only: single arms; the state-evolution contrast (T-ai - C-ai) vs (T - C).
  python -I logs/nano4/m74_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

BASE = "-pc-f0-x40-oh"


def scores(arms, seeds):
    out = []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds], axis=0)
        out.append((evaluate(P, y, q, sizes=SIZES)[1], y, n, pid))
    return out


def diff(a, b, label, rule=True):
    y, n, pid = (np.concatenate([x[k] for x in a]) for k in (1, 2, 3))
    d = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] for x in b]), y, pid, n, sizes=SIZES, n=5000)
    pf = [balanced_mean_over_n(x[0], x[1], x[2], sizes=SIZES) - balanced_mean_over_n(w[0], x[1], x[2], sizes=SIZES) for x, w in zip(a, b)]
    ok = (d["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (d["diff"] >= 0.0025)
    print(f"  {label:34s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + (f"  -> {'PASS' if ok else 'no'}" if rule else ""))
    return {**d, "per_fold": pf, "pass": bool(ok)}


rep, ok = {}, []
for seeds in ((0, 1, 2), (3, 4, 5)):
    have = all((ES / f"val{v}/{t}{BASE}-ai_s{s}.npz").exists() for v in FOLDS for t in ("C", "T") for s in seeds)
    if not have:
        print(f"seeds {seeds}: members missing, skipped"); ok.append(False); continue
    print(f"seeds {seeds}:")
    C, T, Ca, Ta = scores([f"C{BASE}"], seeds), scores([f"T{BASE}"], seeds), scores([f"C{BASE}-ai"], seeds), scores([f"T{BASE}-ai"], seeds)
    E, Ea = scores([f"C{BASE}", f"T{BASE}"], seeds), scores([f"C{BASE}-ai", f"T{BASE}-ai"], seeds)
    r = {"ensemble": diff(Ea, E, "3C-ai + 3T-ai vs 3C + 3T (primary)"), "C": diff(Ca, C, "C-ai vs C", False), "T": diff(Ta, T, "T-ai vs T", False),
         "T_minus_C_plain": diff(T, C, "T vs C (no state)", False), "T_minus_C_ai": diff(Ta, Ca, "T-ai vs C-ai (with state)", False)}
    # the state-evolution contrast: does the sequence arm gain more from the state columns than the per-move arm?
    y, n, pid = (np.concatenate([x[k] for x in C]) for k in (1, 2, 3))
    dd = paired_diff(np.concatenate([a[0] - b[0] for a, b in zip(Ta, Ca)]), np.concatenate([a[0] - b[0] for a, b in zip(T, C)]), y, pid, n, sizes=SIZES, n=5000)
    print(f"  (T-ai - C-ai) - (T - C)            {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  (positive = the state makes cross-move dynamics learnable)")
    r["contrast"] = dd; rep[f"seeds_{seeds[0]}"] = r; ok.append(r["ensemble"]["pass"])
rep["adopt"] = bool(all(ok)); print(f"PRE-REGISTERED: adopt the state columns: {'YES' if all(ok) else 'NO'}")
Path(ES / "m74_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
