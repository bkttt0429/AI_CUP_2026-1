#!/usr/bin/env bash
# Section 4M M189 step 2: the variant file with the enlarged prototypes, the identifiers SU1 (block + S7u) and SU4 (block + S6at
# + S7u + S7tc; ct112's composition with the prototypes exchanged), their label-free readings, and the paired pool comparison
# against SQ4 (report only).
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m189_asopp.py --assemble --seeds 30 31 || return 1
  python -I logs/nano4/m164_ident.py --prior fit --tag SU1:S7u --tag SU4:S6at,S7u,S7tc || return 1
  python -I logs/nano4/m120_diag_mass.py SQ1~m15 SU1~m15 SQ4~m15 SU4~m15 || echo "(the recognition-share report failed; the comparison goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 SU4 SQ1 SU1 --json m189_pool.json
}
main "$@"; exit $?
