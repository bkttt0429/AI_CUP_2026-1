#!/usr/bin/env bash
# M90: merge the trained networks into the matcher cache and evaluate the identifiers built on them.
#   m90_eval.sh one    seed 0 only          -> S2, SP2
#   m90_eval.sh three  seeds 0,1,2 averaged -> S2e, SP2e (falls back to the two seeds that finished if one seed's job failed)
set -uo pipefail
case "$1" in
  one)
    python -I logs/nano4/m88_seqid.py --stage merge --full --seeds 0 || exit 1
    python -I logs/nano4/m84_variant.py --stage eval --tag S2
    python -I logs/nano4/m84_variant.py --stage eval --tag SP2 ;;
  three)
    ok=0
    for s in 0,1,2 0,1 0,2 1,2; do
      if python -I logs/nano4/m88_seqid.py --stage merge --full --seeds "$s"; then ok=1; echo "merged seeds $s"; break; fi
    done
    [ "$ok" -eq 1 ] || exit 1
    python -I logs/nano4/m84_variant.py --stage eval --tag S2e
    python -I logs/nano4/m84_variant.py --stage eval --tag SP2e ;;
  *) echo "usage: $0 one|three" >&2; exit 2 ;;
esac
