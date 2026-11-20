#!/usr/bin/env bash
# M124: evaluate the orientation-averaged query vectors of seed 0 (tags S4t, SP4t) and print the opponent-side recognition shares.
set -uo pipefail
main() {
  python -I logs/nano4/m84_variant.py --stage eval --tag S4t || return 1
  python -I logs/nano4/m84_variant.py --stage eval --tag SP4t || return 1
  python -I logs/nano4/m120_diag_mass.py 5e S2 SP2 S2e SP2e S4 SP4 S4t SP4t
}
main "$@"; exit $?
