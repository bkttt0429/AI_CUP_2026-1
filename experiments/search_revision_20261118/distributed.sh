#!/usr/bin/env bash
set -eu
ulimit -Su "$(ulimit -Hu)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 --wait=0 --kill-on-bad-exit=1 --cpu-bind=cores --output=part_%j_%t.log "$py" -u child_parts.py worker
"$py" -u child_parts.py combine
"$py" -u response.py
bash measure.sh
