#!/usr/bin/env bash
# M200 step 2 as REPORT ONLY (2026-11-17; the user: "好 跑" after the 40,000-step component gate failed on the 30-100 band).
# The registered procedure unchanged (m200_step2.sh: S9t, SS1..SS3, the label-free choice, the paired pool comparison against
# SQ4 on both seed sets, the registered pool test), then the ledger back-test against ct112 if a file was built.
# Whatever comes out is a result obtained after a failed gate: no file of this run is a candidate unless the user decides so.
#   bash logs/nano4/m200_step2_report.sh
main() {
  bash logs/nano4/m200_step2.sh 30 S9t ct114 || { echo "step 2 failed"; return 1; }
  if [ -f submission_rank_ct114.csv ] && [ -f results/r_variants/m163_test_posterior_ct114.npz ]; then
    python -I logs/nano4/m129_ledger.py submission_rank_ct114.csv submission_rank_ct112.csv --prior=learned --post=results/r_variants/m163_test_posterior_ct114.npz
  else
    echo "no ct114 file was built (the registered pool test did not pass); see the paired comparison above"
  fi
}
main "$@"; exit $?
