#!/usr/bin/env bash
# M211 (2026-11-17): decode the Public reading of ct116 and solve it jointly with the ledger, the probes and the composition.
main() {
  python -I logs/nano4/m206_decode.py submission_rank_ct116.csv 0.8394955 || return 1
  python -I logs/nano4/m211_ledger_ct116.py
}
main "$@"; exit $?
