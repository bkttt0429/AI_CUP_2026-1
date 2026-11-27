#!/usr/bin/env bash
# Section 4M M182 (report only): the level-aggregated identifier SL4 (same fused scores as SQ4) and its pool comparison.
main() {
  source .venv/bin/activate; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12
  python -I logs/nano4/m182_level_ident.py --prior fit --tag SL4:S6at,S7t,S7tc || return 1
  python -I logs/nano4/m120_diag_mass.py SQ4~m15 SL4~m15 || echo "(the recognition-share report failed; the comparison goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 SL4 SQ4+SL4 --json m182_pool.json
}
main "$@"; exit $?
