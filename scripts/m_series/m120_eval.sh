#!/usr/bin/env bash
# M120: merge the symmetry-augmented identity networks into the matcher cache and evaluate (design doc M120).
#   m120_eval.sh one     seed 0 only  -> var_S4;  tags S4, SP4 (preview, report only)
#   m120_eval.sh three   seeds 0,1,2  -> var_S4e; tags S4e, SP4e, SP2y4, SP2e4, then the registered test (m121, --build)
#   m120_eval.sh auto    three if all fifteen networks are there, one if seed 0 is complete; each stage runs once (lock directory)
set -uo pipefail
have() { for t in "$@"; do [ -f "results/m120/out_$t.npz" ] || return 1; done; }
main() {
  local stage="${1:-auto}"
  if [ "$stage" = auto ]; then
    if have 1 2 3 4 -1 1_s1 2_s1 3_s1 4_s1 -1_s1 1_s2 2_s2 3_s2 4_s2 -1_s2; then stage=three
    elif have 1 2 3 4 -1; then stage=one
    else echo "seed 0 is not complete: nothing to evaluate yet"; return 0; fi
    mkdir "results/m120/.eval_$stage.lock" 2>/dev/null || { echo "evaluation '$stage' was already started by another job"; return 0; }
  fi
  echo "== evaluation stage: $stage ($(date '+%F %H:%M'))"
  case "$stage" in
    one)
      python -I scripts/m_series/m120_seqid_sym.py --merge 0 || return 1
      python -I scripts/m_series/m84_variant.py --stage eval --tag S4
      python -I scripts/m_series/m84_variant.py --stage eval --tag SP4 ;;
    three)
      python -I scripts/m_series/m120_seqid_sym.py --merge 0,1,2 || return 1
      for t in S4e SP4e SP2y4 SP2e4; do python -I scripts/m_series/m84_variant.py --stage eval --tag "$t" || return 1; done
      python -I scripts/m_series/m121_sym_anchor_test.py --id SP2y4 --also SP4e SP2e4 --build ;;
    *) echo "usage: $0 one|three|auto" >&2; return 2 ;;
  esac
}
main "$@"; exit $?
