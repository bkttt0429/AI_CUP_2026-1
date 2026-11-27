#!/usr/bin/env bash
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=16
  python -I logs/nano4/m169_pool.py --sets SX4L SX4L@lastmove --json m169_pool_lastmove.json
}
main "$@"; exit $?
