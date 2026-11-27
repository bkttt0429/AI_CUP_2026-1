#!/usr/bin/env bash
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
A=/home/u2507115/.cache/report_audit/m36smoke; rm -rf $A; mkdir -p $A
DATA=/dev/shm/e2es_${SLURM_JOB_ID:-local}; rm -rf $DATA
python -I logs/nano4/e2e_pilot.py games --out results/r_variants/attn_games.csv || exit 1
python -I logs/nano4/e2e_pilot.py prepare --data $DATA --smoke --no-eval --train-folds 1 2 --workers 20 || exit 1
python -I logs/nano4/e2e_pilot.py train --data $DATA --arm attn --seed 0 --epochs 1 --batch 8 --log-every 5 --save-head $A/headA.pt || exit 1
head -201 results/r_variants/attn_games.csv > $A/games.csv
python -I logs/nano4/e2e_pilot.py score --games $A/games.csv --heads $A/headA.pt $A/headA.pt --out $A/scores --workers 20 --part-size 100 || exit 1
python -I logs/nano4/e2e_pilot.py score --games $A/games.csv --heads $A/headA.pt $A/headA.pt --out $A/scores --workers 20 --part-size 100 || exit 1  # resume: nothing left
python -I -c "
import glob, numpy as np
fs = sorted(glob.glob('$A/scores/part_*.npz')); z = [np.load(f) for f in fs]
k = np.concatenate([x['keys'] for x in z]); a = np.concatenate([x['logp0'] for x in z]); b = np.concatenate([x['logp1'] for x in z])
print(len(fs), 'parts', len(k), 'games', len(set(k)), 'unique; logp shape', a.shape, 'finite', bool(np.isfinite(a).all()), 'heads agree', float(np.abs(a - b).max()), 'prob sums', float(np.abs(np.exp(a).sum(1) - 1).max()))
assert len(k) == 200 and len(set(k)) == 200 and np.isfinite(a).all() and np.abs(a - b).max() < 1e-5"
rm -rf $DATA
