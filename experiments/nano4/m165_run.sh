#!/usr/bin/env bash
# Section 4M M165 (2026-11-15), report only: single-source identifiers with the prior, then several of them as separate pool sources.
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=16
  python -I logs/nano4/m164_ident.py --prior fit --tag SX4L:S4Lt --tag SX4f:S4ft --tag SX2y:S2y || return 1
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SX4L~m15 SX4f~m15 SX2y~m15 || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SX4L SX4L+SX4f SX4L+SX4f+SX2y SPALm SPALm+SX4L SPALm+SX4L+SX4f+SX2y
}
main "$@"; exit $?
