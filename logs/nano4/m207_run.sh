#!/usr/bin/env bash
# M207 (2026-11-17): decode the three probe readings of M206 and build the files that follow from them.
main() {
  echo "== composition with all-C"; python -I logs/nano4/m202_decode.py 4D=0.1552986 5D=0.1552986 A=0.2071379 1D=0.1427728 C=0.2019698 || return 1
  echo "== probe_4D5D_locked"; python -I logs/nano4/m206_decode.py upload_probe/probe_4D5D_locked.csv 0.8347546 || return 1
  echo "== probe_A2B"; python -I logs/nano4/m206_decode.py upload_probe/probe_A2B.csv 0.8363349 || return 1
  echo "== build"; python -I logs/nano4/m207_build.py
}
main "$@"; exit $?
