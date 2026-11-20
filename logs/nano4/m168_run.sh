#!/usr/bin/env bash
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=16
  python -I logs/nano4/m165_pool.py --sets SX4L --json m168_pool_tmp.json --dump m168_dump.npz || return 1
  python -I logs/nano4/m168_error_by_anchors.py m168_dump.npz SX4L
}
main "$@"; exit $?
