#!/usr/bin/env python
"""M262 check (2026-11-20, report only): the clean win share of the Public questions whose truth the ledger pins down, by the
sign of (truth - model answer). The hypothesis (accounts with a self-declared rank) predicts: questions the model answers too HIGH
(truth below the answer) are winners, questions it answers too LOW are losers. Also the eight 1D -> A flips of ct13 (all true A).

  python -I logs/nano4/m262_checks.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; RV = ROOT / "results/r_variants"


def main():
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    oc = np.load(RV / "m262_outcomes.npz"); qid = oc["qid"].astype(str); ne = oc["n_even"].astype(float); sh = np.where(ne > 0, oc["wins_even"] / np.maximum(ne, 1), np.nan); n = oc["n"].astype(int); pos = {q: i for i, q in enumerate(qid)}; rng = np.random.default_rng(2620)
    prof = json.load(open(RV / "m210_profiles.json")); p211 = {q: d["with_composition"] for q, d in json.load(open(RV / "m211_ledger_ct116.json")).items() if d.get("with_composition")}; p254 = json.load(open(RV / "m254_update.json"))
    a9, a13, a112 = rd("submission_rank_ct9.csv"), rd("submission_rank_ct13.csv"), rd("submission_rank_ct112.csv"); pub = [q for q in qid if q.startswith("pub")]
    ok = np.isfinite(sh) & (ne >= 3); allp = np.array([pos[q] for q in pub]); base_mean = float(np.nanmean(sh[allp][ok[allp]]))
    print(f"Public questions with >= 3 even games: {int(ok[allp].sum())}; mean clean win share {base_mean:.3f}")
    flips = [q for q in pub if a9[q] == "1D" and a13[q] == "A"]; oth = [q for q in pub if a9[q] == "1D" and a13[q] == "1D"]
    f_ = np.array([sh[pos[q]] for q in flips]); o_ = np.array([sh[pos[q]] for q in oth]); print(f"ct13's 1D -> A flips (Public, all true A by the reading): {len(flips)} questions, clean win shares " + " ".join(f"{x:.2f}" for x in f_) + f" (even games {[int(ne[pos[q]]) for q in flips]}); mean {np.nanmean(f_):.3f}; the other Public questions the base answers 1D: {len(oth)}, mean {np.nanmean(o_):.3f}")
    rows = []
    for q in pub:
        w = p254.get(q) or p211.get(q) or prof.get(q)
        if w is None:
            continue
        lv = max(w, key=w.get); tot = sum(w.values())
        if w[lv] / tot >= 0.9 and ok[pos[q]]:
            rows.append((q, LEV.index(lv), LEV.index(a112[q]), LEV.index(a9[q]), float(sh[pos[q]]), int(n[pos[q]])))
    print(f"Public questions whose truth the ledger pins down (weight >= 0.9) and with >= 3 even games: {len(rows)}")
    for nm, col in (("the ct112 rule's answer", 2), ("the base answer (ct9)", 3)):
        d = np.array([r[1] - r[col] for r in rows]); s = np.array([r[4] for r in rows])
        print(f"   against {nm}: truth BELOW the answer {int((d < 0).sum())} questions, mean win share {s[d < 0].mean() if (d < 0).any() else float('nan'):.3f}; truth == answer {int((d == 0).sum())}, {s[d == 0].mean():.3f}; truth ABOVE the answer {int((d > 0).sum())}, {s[d > 0].mean() if (d > 0).any() else float('nan'):.3f}")
        if (d != 0).sum() >= 6:
            x, y = s[d != 0], np.sign(d[d != 0]); r = float(np.corrcoef(x, y)[0, 1]); null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(20000)]); p = float((null <= r).mean())
            print(f"      among the {int((d != 0).sum())} wrong answers: correlation of the win share with the sign of (truth - answer) {r:+.3f}, one-sided permutation p (negative = as predicted) {p:.4f}")
            for q_, t_, a_, b_, s_, n_ in sorted(rows, key=lambda r_: r_[4]):
                if (t_ - (a_ if col == 2 else b_)) != 0:
                    print(f"         {q_} ({n_} games): win share {s_:.2f}, answer {LEV[a_ if col == 2 else b_]}, truth {LEV[t_]}")


if __name__ == "__main__":
    main()
