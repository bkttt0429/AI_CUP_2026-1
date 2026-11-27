#!/usr/bin/env bash
# Section 4M M7 (2026-11-04): (a) size-routed ensembles from stored members; (b) the -s15-20 arm (4 caches, 24 training jobs), its
# n=20 compare and the routed check with both bands. Cap-limited submissions (QOS 20).
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
echo "== $(date '+%F %T') M7 start"
A=$(submit m7a "-c 12 -t 00:30:00" python -I logs/nano4/n_routed.py --low C-s5-10 T-s5-10 --high C T --base C T --seeds 0 1 2 --out results/trunk_seq/es/routed_m7a.json)
C=""
for v in 1 2 3 4; do C="$C,$(submit rs1520v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --train-sizes 15 20)"; done
echo "m7a $A caches ${C#,}"; wait_jobs "$A,$C"
echo "== M7(a) routed from stored members $A $(state $A)"; grep -v "^==\|^+" logs/nano4/m7a-$A.out
bad=0; for j in ${C//,/ }; do [ -z "$j" ] && continue; [ "$(state $j)" = "COMPLETED" ] || bad=1; done
[ "$bad" = 0 ] || { echo "a cache job failed: abort M7(b)"; exit 1; }
ids=""
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  ids="$ids,$(submit s1520$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --train-sizes 15 20 --out results/trunk_seq/es)"
done; done; done
echo "m7 jobs ${ids#,}" | tee $AUD/m7_jobs.txt
wait_jobs "$ids"; stops "$ids"
K=$(submit m7cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-s15-20 T-s15-20 --seeds 0 1 2 --pairs C-s15-20:C T-s15-20:T --sizes 20 --tag m7_n20_ --out results/trunk_seq/es)
K2=$(submit m7cmpall "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-s15-20 T-s15-20 --seeds 0 1 2 --pairs C-s15-20:C T-s15-20:T --tag m7_all_ --out results/trunk_seq/es)
E=$(submit m7ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-s15-20 T-s15-20 --seeds 0 1 2 --sizes 20 --out results/trunk_seq/es/ens_m7_n20.json)
R=$(submit m7b "-c 12 -t 00:30:00" python -I logs/nano4/n_routed.py --low C-s5-10 T-s5-10 --high C-s15-20 T-s15-20 --base C T --seeds 0 1 2 --out results/trunk_seq/es/routed_m7b.json)
wait_jobs "$K,$K2,$E,$R"
for pair in "M7 compare n=20:m7cmp:$K" "M7 compare all n:m7cmpall:$K2" "M7 ensemble n=20:m7ens:$E" "M7(b) routed two bands:m7b:$R"; do
  IFS=: read -r label name id <<< "$pair"
  echo "== $label $id $(state $id)"; grep -v "^==\|^+" logs/nano4/$name-$id.out
done
echo "== $(date '+%F %T') M7 done"
