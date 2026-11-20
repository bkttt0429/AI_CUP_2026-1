#!/usr/bin/env bash
# Section 4M M178b (report only; the A guard by direction): guard variants in cross-validation and on the test side, then the ledger back-test of every relaxation against ct112.
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m178b_guards.py --new SQ4 --name ct112 --out-dir results/m178b_scratch --json m178b_guards.json || return 1
  local f
  for f in results/m178b_scratch/submission_rank_ct112_*.csv; do
    echo "== ledger back-test of $(basename "$f") against ct112"
    python -I logs/nano4/m129_ledger.py "$f" submission_rank_ct112.csv --prior=learned --post=results/r_variants/m163_test_posterior_ct112.npz || echo "(back-test failed for $f)"
  done
}
main "$@"; exit $?
