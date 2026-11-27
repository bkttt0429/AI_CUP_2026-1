#!/usr/bin/env bash
# M105: M90 seeds 1 and 2 as single identifier sources (voters), outputs m84_S2b_*.npz / m84_S2c_*.npz
set -uo pipefail
for t in S2b S2c; do python -I logs/nano4/m84_variant.py --stage eval --tag "$t"; done
