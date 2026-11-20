#!/usr/bin/env bash
# Section 4M M184 (report only): count-scaled anchor evidence and a separate base weight for handicap-heavy questions.
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m184_pool.py --sets SQ4 --json m184_pool.json
}
main "$@"; exit $?
