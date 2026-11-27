#!/usr/bin/env bash
# M24 on nano4 for the record (the user's PC already ran it): full ident_pool + plurality-null check + ct8 rule. One job.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
python -I logs/nano4/ident_pool.py --out results/r_variants/ident_pool.json --workers 10
python -I logs/nano4/plurality_null_check.py --out results/r_variants/plurality_null_check.json --workers 10
python -I logs/nano4/ident_ct8.py --ident results/r_variants/ident_pool.json --out results/r_variants/ident_ct8.json --submission results/r_variants/submission_rank_ct8_nano4.csv
