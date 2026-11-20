#!/usr/bin/env bash
# M216 smoke (2026-11-17): the list of the truncated games of the first 20 m90 parts, their tail extraction, the initial-value
# check, and 80 training steps of the all-players network on the smoke subset (the loader's overlap checks, memory, step time).
# Everything goes to *_smoke folders; nothing of the production paths is touched.
main() {
  local L=results/m216_smoke/tail_keys.npy T=/work/u2507115/m216_tail_smoke
  python -I logs/nano4/m216_tail_extract.py --list --first-files 20 --list-file $L --workers 10 || return 1
  python -I logs/nano4/m216_tail_extract.py --list-file $L --out $T --workers 10 || return 1
  ls $T | wc -l; du -sh $T | cut -f1
  python -I logs/nano4/m216_check.py || return 1
  python -I logs/nano4/m216_prod.py --jobs all:30 --ckpt-layers 4 --smoke --tail $T --tail-list $L || { echo "smoke training failed"; tail -n 30 logs/nano4/m216_net-1_s30-${SLURM_JOB_ID:-local}.log; return 1; }
  grep -E "tail attached|step [0-9]+/80|held-out own games|best row-weighted|outputs written" logs/nano4/m216_net-1_s30-${SLURM_JOB_ID:-local}.log logs/nano4/m216_smoke-${SLURM_JOB_ID:-local}.out 2>/dev/null | cut -c1-420
}
main "$@"; exit $?
