#!/usr/bin/env bash
# M225 (2026-11-18): identifiers that put today's trained families side by side (no training), paired pool comparison against SQ4.
main() {
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  python -I logs/nano4/m164_ident.py --prior fit --tag SV1:S6at,S7t,S11u,S12p --tag SV2:S6at,S11u,S12p --tag SV3:S7t,S11u,S12p || return 1
  python -I logs/nano4/m164_select.py SV1 SV2 SV3 || true
  python -I logs/nano4/m120_diag_mass.py SQ4~m15 SZ3~m15 SV1~m15 SV2~m15 SV3~m15 || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 SV1 SV2 SV3 --json m225_pool.json --dump m225_dump.npz || return 1
  python -I logs/nano4/m224_anchor_diag.py SQ4 SV1 | grep -A 1 "^== D\|^== C" | cut -c1-200
  echo "== $(date '+%F %T') M225 finished"
}
main "$@"; exit $?
