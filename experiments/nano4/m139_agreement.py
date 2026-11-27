#!/usr/bin/env python
"""Section 4M M139 (2026-11-14, report only): label-free reliability of the opponent identification on the test side.

Two independent identifiers picking the same candidate among ~7,900 is almost surely a correct identification. For bands of one
identifier's margin (cut at quantiles of the validation distribution): the share of queries in the band and the share whose
top candidate equals the other identifier's top candidate - validation games against test games.

  python -I logs/nano4/m139_agreement.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"
QS = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)


def load(name):
    if name in ("5e", "5", "6", "7"):
        v, t = np.load(RV / "m82_val.npz"), np.load(RV / "m82_test.npz"); k = f"f{name}"
        return (v[f"{k}_opp_best"].astype(str), v[f"{k}_opp_m"], v[f"{k}_imp_m"]), (t[f"{k}_opp_best"].astype(str), t[f"{k}_opp_m"])
    v, t = np.load(RV / f"m84_{name}_val.npz"), np.load(RV / f"m84_{name}_test.npz")
    return (v["f_opp_best"].astype(str), v["f_opp_m"], v["f_imp_m"]), (t["f_opp_best"].astype(str), t["f_opp_m"])


def main():
    rep = {}
    for a, b_ in (("S2", "5e"), ("5e", "S2"), ("SP2e", "5e"), ("S3", "5e"), ("S2~m15", "5e")):
        try:
            (va, ma, ia), (ta, mta) = load(a); (vb, _, _), (tb, _) = load(b_)
        except FileNotFoundError as e:
            print(f"{a} / {b_}: missing ({e})"); continue
        cuts = np.quantile(ma, QS); edges = np.r_[-np.inf, cuts, np.inf]; rows = []
        print(f"\nbands of {a}'s margin (validation quantiles), agreement of its top candidate with {b_}'s; impostor share = validation impostor margins in the band")
        print("   band              | validation: share, agreement | test: share, agreement | share ratio, agreement ratio | impostor share")
        for i in range(len(edges) - 1):
            mv, mt = (ma > edges[i]) & (ma <= edges[i + 1]), (mta > edges[i]) & (mta <= edges[i + 1]); mi = (ia > edges[i]) & (ia <= edges[i + 1])
            sv, st = float(mv.mean()), float(mt.mean()); av = float((va[mv] == vb[mv]).mean()) if mv.any() else float("nan"); at = float((ta[mt] == tb[mt]).mean()) if mt.any() else float("nan")
            lab = f"q{QS[i - 1] if i else 0:g}-{QS[i] if i < len(QS) else 1:g}"
            print(f"   {lab:17s} | {sv:.4f}  {av:.3f} | {st:.4f}  {at:.3f} | {st / max(sv, 1e-9):.2f}  {at / max(av, 1e-9):.2f} | {float(mi.mean()):.4f}")
            rows.append(dict(band=lab, val_share=sv, val_agree=av, test_share=st, test_agree=at, impostor_share=float(mi.mean())))
        top = ma > cuts[2]; topt = mta > cuts[2]
        print(f"   above q0.9 overall: validation share {top.mean():.4f}, agreement {float((va[top] == vb[top]).mean()):.3f}; test share {topt.mean():.4f}, agreement {float((ta[topt] == tb[topt]).mean()):.3f}; all queries: agreement validation {float((va == vb).mean()):.3f}, test {float((ta == tb).mean()):.3f}")
        rep[f"{a} vs {b_}"] = rows
    (RV / "m139_agreement.json").write_text(json.dumps(rep, indent=1)); print("\nwritten results/r_variants/m139_agreement.json")


if __name__ == "__main__":
    main()
