#!/usr/bin/env bash
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m171_cand.py SX4L~m15 || return 1
  python -I logs/nano4/m171_pool.py --sets SX4L SX4L@cand SX4L+V@cand SX4L@cand+V@cand --json m171_pool.json
}
main "$@"; exit $?
