#!/bin/bash
set -e
X=/work/u2507115/ext_fox; P=/home/u2507115/BK_AICUP_2026-2/data/ext_ply
python -I logs/nano4/m107_multitask_probe.py --src $X/pilot/select.csv $X/pilot/ply --src $X/scale1/select.csv $X/scale1/ply_a,$X/scale1/ply_b,$P/scale1_c 2k,2d,4d --src $X/scale2_5d/select.csv $P/scale2_5d --src $X/midrank/select.csv $X/midrank/ply --save-proba results/r_variants/m107_proba.npz > logs/nano4/m107_rerun.log 2>&1
grep -E "^B  |^REGISTERED" logs/nano4/m107_rerun.log
python -I logs/nano4/m110_fusion.py
