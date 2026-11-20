#!/usr/bin/env bash
# M200 after the training (2026-11-17): the paired table of all five networks, the registered component gate, and - only on
# a pass - step 2 (m200_step2.sh: S9t, SS1..SS3, paired against SQ4, candidate name ct114) and the ledger back-test.
main() {
  python -I logs/nano4/m200_read.py | tail -n 8
  python -I logs/nano4/m200_gate.py || { echo "component gate not passed: step 2 is NOT run (registered); report to the user"; return 0; }
  bash logs/nano4/m200_step2.sh 30 S9t ct114 || return 1
  if [ -f submission_rank_ct114.csv ] && [ -f results/r_variants/m163_test_posterior_ct114.npz ]; then
    python -I logs/nano4/m129_ledger.py submission_rank_ct114.csv submission_rank_ct112.csv --prior=learned --post=results/r_variants/m163_test_posterior_ct114.npz
  else
    echo "no ct114 file was built (the registered pool test did not pass); see the paired comparison above"
  fi
}
main "$@"; exit $?
