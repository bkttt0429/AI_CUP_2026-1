#!/usr/bin/env bash
# M135 driver, part C (login node, detached: submits and waits only; start it after the score array is built, the per-move score
# files are removed and the trunk vectors are complete): a trainer smoke on the real external set, the 48 members with --ext
# (C / T x folds 1-4 x seeds 0-2 and 6-8), the comparisons against the current members and the final-level check.
#   setsid nohup bash logs/nano4/m135_c.sh > logs/nano4/m135_c.log 2>&1 < /dev/null &
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
  [ -f $E/scores_oh_fold9.npy ] && [ -f $E/scores_index_fold9.npz ] && [ -f $E/r_ext_p0.npz ] && [ -f $E/trunk_vec/fold9/quant8.npz ] || { echo "the external set is not complete: stop"; return 1; }
  echo "== $(date '+%F %T') M135 part C stage 4: trainer smoke on the real set"; du -sh $E; df -h /work/u2507115 | tail -n 1
  SM="--limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m135smoke_real --ext $E"
  T=$(submit m135st "-p 8gpus -c 12 -t 01:00:00" bash -c "$TR train --arm T --val-fold 1 --seed 0 $REC $SM")
  wait_jobs "$T"; echo "smoke $T $(state $T)"; grep -E "external|tokens loaded|stopped at step|Error|error|Traceback" logs/nano4/m135st-$T.out | tail -8 | cut -c1-260
  ok_all "$T" || { echo "trainer smoke failed: stop"; return 1; }
  echo "== $(date '+%F %T') M135 part C stage 5: 48 members with --ext (seeds 0-2 and, at the same time, the replication seeds 6-8)"
  rm -f $A/m135_cmds_*
  for s in 0 1 2 6 7 8; do for v in 1 2 3 4; do for arm in C T; do
    echo "$TR train --arm $arm --val-fold $v --seed $s $REC --ext $E --save-dir results/models/ct_cv_m135 --out results/trunk_seq/es" >> $A/m135_cmds_all.txt
  done; done; done
  split -l 8 -d $A/m135_cmds_all.txt $A/m135_cmds_
  P=""; k=0
  for f in $A/m135_cmds_0*; do
    if [ $((k % 3)) -eq 2 ]; then part="-p 16gpus"; else part="-p 8gpus"; fi
    P="$P,$(submit m135p$k "$part --gres=gpu:8 -c 96 -t 03:00:00" bash $A/pack8.sh $f)"; k=$((k + 1))
  done
  echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
  ok_all "$P" || echo "a pack failed: the comparisons below show what is missing"
  echo "== $(date '+%F %T') M135 part C stage 6: comparisons"
  t="pc-f0-x40-oh-ex"; O="C-pc-f0-x40-oh T-pc-f0-x40-oh"; N="C-$t T-$t"; EC="python -I logs/nano4/ens_compare.py"
  J=$(submit m135c "-p 8gpus -c 12 -t 01:30:00" bash -c "$TR compare --val-folds 1 2 3 4 --arms $O $N --seeds 0 1 2 --pairs C-$t:C-pc-f0-x40-oh T-$t:T-pc-f0-x40-oh --tag m135_ex_ --out results/trunk_seq/es; $EC --old $O --new $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex.json; $EC --old $O --new $N --seeds 6 7 8 --out results/trunk_seq/es/ens_m135_ex_s6.json; $EC --old $O --new $O $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex_added.json; python -I logs/nano4/m135_final_cv.py --new $N --seeds 0 1 2 --seeds2 6 7 8")
  wait_jobs "$J"; echo "== comparison job $J $(state $J)"; grep -v "^==\|^+" logs/nano4/m135c-$J.out | tail -70 | cut -c1-330
  echo "== $(date '+%F %T') M135 part C done"
}
main "$@"; exit $?
