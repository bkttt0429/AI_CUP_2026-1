#!/usr/bin/env bash
# Section 4M M4 (2026-11-04): pytest of the PMA code, then C-pma / T-pma on folds 1-4, seeds 0-2 (24 jobs), compare + 6-vs-6 ensemble check.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=18
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 50; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
stops() { for j in ${1//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/*-$j.out 2>/dev/null | sed 's/.*stopped/stopped/')"; done; }
echo "== $(date '+%F %T') M4 start"
T=$(submit pmatest "-c 12 -t 00:40:00" python -m pytest -q -x tests/test_trunk_seq_4m.py tests/test_trunk_predict.py tests/test_trunk_seq_mix.py)
echo "pmatest $T"; wait_jobs "$T"
[ "$(state $T)" = "COMPLETED" ] || { echo "tests $(state $T): abort"; tail -30 logs/nano4/pmatest-$T.out; exit 1; }
echo "tests passed $(date '+%T')"
ids=""
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  ids="$ids,$(submit pma$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --pooling pma --out results/trunk_seq/es)"
done; done; done
echo "m4 jobs ${ids#,}" | tee $AUD/m4_pma_jobs.txt
wait_jobs "$ids"; stops "$ids"
K=$(submit m4cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-pma T-pma --seeds 0 1 2 --pairs C-pma:C T-pma:T --tag m4_ --out results/trunk_seq/es)
E=$(submit m4ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-pma T-pma --seeds 0 1 2 --out results/trunk_seq/es/ens_m4.json)
wait_jobs "$K,$E"
echo "== M4 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m4cmp-$K.out
echo "== M4 ensemble $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m4ens-$E.out
echo "== $(date '+%F %T') M4 done"
