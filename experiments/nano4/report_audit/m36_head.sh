#!/usr/bin/env bash
# M36 step 1: one cross-fit attention head (attn, frozen trunk, 8 games a player, 2 epochs, seed 0). $1 = A (folds 1 2) | B (folds 3 4).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
case "$1" in A) F="1 2";; B) F="3 4";; *) echo "A or B"; exit 2;; esac
DATA=/dev/shm/m36_$1_${SLURM_JOB_ID:-local}; rm -rf $DATA
python -I logs/nano4/e2e_pilot.py prepare --data $DATA --no-eval --train-folds $F --games-per-player 8 --workers 88 || exit 1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python -I logs/nano4/e2e_pilot.py train --data $DATA --arm attn --seed 0 --save-head results/r_variants/attn_heads/attn_$1.pt || exit 1
rm -rf $DATA
