#!/usr/bin/env bash
# M135, report-only completion: the external loss at weight 3 (tag -ex3), 24 members (C / T x folds 1-4 x seeds 0-2), then the
# member-level and final-level comparisons against the current members (login node, detached: submits and waits only).
#   setsid nohup bash logs/nano4/m135_l3.sh > logs/nano4/m135_l3.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"
E=/work/u2507115/ext_rank
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 240); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 30; done; return 1; }
main() {
  mkdir $E/.m135_l3.lock 2>/dev/null || { echo "the weight-3 arm was already started: not starting it twice"; return 0; }
  echo "== $(date '+%F %T') M135 weight-3 arm: 24 members"
  rm -f $A/m135l3_cmds_*
  for s in 0 1 2; do for v in 1 2 3 4; do for arm in C T; do
    echo "$TR train --arm $arm --val-fold $v --seed $s $REC --ext $E --ext-weight 3 --out results/trunk_seq/es" >> $A/m135l3_cmds_all.txt
  done; done; done
  split -l 8 -d $A/m135l3_cmds_all.txt $A/m135l3_cmds_
  P=""; k=0
  for f in $A/m135l3_cmds_0*; do
    if [ $k -eq 2 ]; then part="-p 8gpus"; else part="-p 16gpus"; fi
    P="$P,$(submit m135q$k "$part --gres=gpu:8 -c 96 -t 03:00:00" bash $A/pack8.sh $f)"; k=$((k + 1))
  done
  echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
  ok_all "$P" || echo "a pack failed: the comparisons below show what is missing"
  echo "== $(date '+%F %T') comparisons"
  t="pc-f0-x40-oh-ex3"; O="C-pc-f0-x40-oh T-pc-f0-x40-oh"; N="C-$t T-$t"; EC="python -I logs/nano4/ens_compare.py"
  J=$(submit m135d "-p 8gpus,16gpus -c 12 -t 01:30:00" bash -c "$TR compare --val-folds 1 2 3 4 --arms $O $N --seeds 0 1 2 --pairs C-$t:C-pc-f0-x40-oh T-$t:T-pc-f0-x40-oh --tag m135_ex3_ --out results/trunk_seq/es; $EC --old $O --new $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex3.json; $EC --old C-pc-f0-x40-oh-ex T-pc-f0-x40-oh-ex --new $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex3_vs_ex.json; python -I logs/nano4/m135_final_cv.py --new $N --seeds 0 1 2 --json m135_final_cv_ex3.json")
  wait_jobs "$J"; echo "== comparison job $J $(state $J)"; grep -v "^==\|^+" logs/nano4/m135d-$J.out | tail -40 | cut -c1-330
  echo "== $(date '+%F %T') M135 weight-3 arm done"
}
main "$@"; exit $?
