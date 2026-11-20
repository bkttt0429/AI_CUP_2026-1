#!/usr/bin/env bash
# Section 4M M183 (report only): the anatomy of the remaining loss under the ct112 rule, and whether game numbers carry time.
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m183_time.py SQ4 || echo "(part G failed)"
  python -I logs/nano4/m183_dump.py --sets SQ4 --json m183_pool.json --dump m183_dump.npz || return 1
  python -I logs/nano4/m183_anatomy.py m183_dump.npz SQ4
}
main "$@"; exit $?
