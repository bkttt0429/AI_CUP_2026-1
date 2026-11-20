#!/bin/bash
set -e
cd $HOME/BK_AICUP_2026-1; export PATH=$PWD/.venv/bin:$PATH PYTHONUNBUFFERED=1
for arm in C T; do python -I scripts/trunk_seq_check.py train --arm $arm --val-fold 1 --seed 0 --epochs 1 --limit 3200 --out results/trunk_seq/smoke 2>&1 | grep -v -E 'UserWarning|as_tensor'; python3 -I -c "
import json;d=json.load(open('results/trunk_seq/smoke/val1/${arm}_s0.json'));print('$arm','params',d['n_params'],'block',d['block_params'],'wall',round(d['wall_s']),'epoch_s',round(d['history'][0]['s']),'val',round(d['final'],4))"; done
