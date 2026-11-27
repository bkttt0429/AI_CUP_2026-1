#!/usr/bin/env bash
# Registered M181 final check after m161r_step2; report only, no upload.
set -euo pipefail
candidate=submission_rank_ct113.csv
posterior=results/r_variants/m163_test_posterior_ct113.npz
if [[ ! -f "$candidate" ]]; then
  echo "M181 produced no ct113 candidate; ledger comparison skipped. Inspect m181_eval log for gates/failures."
  exit 0
fi
test -f "$posterior"
test -f results/r_variants/m181_pool.json
python -I logs/nano4/m129_ledger.py "$candidate" submission_rank_ct112.csv \
  --prior=learned --post="$posterior"
echo "Ledger report ready. Adoption still requires M181 paired >= +0.001 on both seed sets, nonnegative ledger result, and review."
