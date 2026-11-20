#!/usr/bin/env bash
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m165_pool.py --sets SQ4 SQ1 SQ3 SQ2 SQ5 SQ7 SQa1 --json m179_pool.json --dump m179_dump.npz || return 1
  python -I logs/nano4/m180_by_class.py m179_dump.npz SQ4
  python -I logs/nano4/m180_by_class.py m179_dump.npz SQ1
}
main "$@"; exit $?
