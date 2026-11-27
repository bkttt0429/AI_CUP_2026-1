#!/usr/bin/env bash
set -eu
ulimit -Su "$(ulimit -Hu)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
if [[ ! -f confirmation/corpus.json ]]; then
  "$py" -u extra.py build
fi
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --wait=0 --kill-on-bad-exit=1 --cpu-bind=cores --output=extra_%j_%t.log "$py" -u extra.py root
"$py" -u interaction.py
