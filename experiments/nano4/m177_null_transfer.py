#!/usr/bin/env python
"""Section 4M M177 diagnostic (2026-11-15, report only): does the impostor null of an identifier transfer from validation to test?

The band edges of the anchor evidence are quantiles of the VALIDATION impostor margins (the target's own side); on the test
side the same statistic exists for the target side of the test games (a test player is never a candidate). If the null
transfers, the share of test target-side margins above the validation q80 / q90 / q95 / q98 / q99 edges is 0.20 / 0.10 / 0.05 /
0.02 / 0.01. Per identifier and per colour of the edges; and the share of the test OPPONENT side above the same edges.
  python -I logs/nano4/m177_null_transfer.py SQ1 SQ2 SQ3 SQ4 SQa1 SPALm SPAL
"""
import sys
from pathlib import Path

import numpy as np

RV = Path.cwd() / "results/r_variants"; QS = (0.80, 0.90, 0.95, 0.98, 0.99)
print("identifier | test TARGET side above the validation impostor q80 / q90 / q95 / q98 / q99 (nominal 0.200 / 0.100 / 0.050 / 0.020 / 0.010) | test OPPONENT side above the same edges | validation opponent side")
for tag in sys.argv[1:]:
    v = dict(np.load(RV / f"m84_{tag}~m15_val.npz", allow_pickle=True)); t = dict(np.load(RV / f"m84_{tag}~m15_test.npz", allow_pickle=True)); rows = {"tgt": [], "opp": [], "val": []}
    for q in QS:
        e = {True: np.quantile(v["f_imp_m"][v["target_black"]], q), False: np.quantile(v["f_imp_m"][~v["target_black"]], q)}
        tb = t["target_black"]; rows["tgt"].append(float(np.where(tb, t["f_tgt_m"] > e[True], t["f_tgt_m"] > e[False]).mean()))  # the target side plays the target's colour
        rows["opp"].append(float(np.where(tb, t["f_opp_m"] > e[False], t["f_opp_m"] > e[True]).mean())); vb = v["target_black"]; rows["val"].append(float(np.where(vb, v["f_opp_m"] > e[False], v["f_opp_m"] > e[True]).mean()))
    print(f"{tag:6s} | " + " / ".join(f"{x:.3f}" for x in rows["tgt"]) + " | " + " / ".join(f"{x:.3f}" for x in rows["opp"]) + " | " + " / ".join(f"{x:.3f}" for x in rows["val"]))
