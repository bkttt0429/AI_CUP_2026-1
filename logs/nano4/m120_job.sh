#!/usr/bin/env bash
# M120 job: train the given networks (one per GPU: network[:seed] ...), then evaluate whatever is complete (m120_eval.sh auto).
set -uo pipefail
main() {
  python -I logs/nano4/m120_seqid_sym.py --jobs "$@" || echo "training exited non-zero; the evaluation still looks at what finished"
  bash logs/nano4/m120_eval.sh auto
}
main "$@"; exit $?
