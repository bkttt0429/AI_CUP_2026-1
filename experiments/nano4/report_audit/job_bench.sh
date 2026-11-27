#!/bin/bash
cd $HOME/BK_AICUP_2026-1; export PATH=$PWD/.venv/bin:$PATH PYTHONUNBUFFERED=1
echo "== default threads"; python -I logs/nano4/replay_bench.py
echo "== one thread"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I logs/nano4/replay_bench.py
