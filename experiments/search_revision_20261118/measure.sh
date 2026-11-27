#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
"$py" -u mechanism.py
"$py" -u escape_overlap.py
"$py" -u verify.py
"$py" -u figure.py
