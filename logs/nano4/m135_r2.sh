#!/usr/bin/env bash
# M135 recovery 2 (login node, detached: submits and waits only). After the four Human SL jobs end: the opponent-side games of the
# dead shard 7/16 as eight sub-shards (ids 7 + 16 j of 128), the rest of shard 5/16 on both sides (it stops at the second cut
# record), the trunk vectors (int8, two jobs, only if /work has room); then, when recovery 1 (job 776187: L1b) is done too, the
# score array + index + game tables + questions + checks. It stops there: the per-move score files are removed by hand before
# the members train.
#   setsid nohup bash logs/nano4/m135_r2.sh > logs/nano4/m135_r2.log 2>&1 &
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
E=/work/u2507115/ext_rank
ENVV="AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 120); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 30; done; return 1; }
main() {
  echo "== $(date '+%F %T') waiting for the Human SL jobs 775761-775764"
  wait_jobs "775761,775762,775763,775764"; for j in 775761 775762 775763 775764; do echo "$j $(state $j) $(sacct -n -X -j $j -o JobName,Elapsed | head -1)"; done
  for j in 775761 775762 775763 775764; do for i in 0 1 2 3 4 5 6 7; do f=logs/nano4/pack-$j-$i.log; grep -q "fold 9 done" $f || echo "  $j slot $i did not finish: $(grep -c Traceback $f) traceback(s), last part line: $(grep 'fold 9 part' $f | tail -n 1 | cut -c1-70)"; done; done
  rm -f $A/m135_r2_opp7.txt $A/m135_r3_s5.txt
  for j in 0 1 2 3 4 5 6 7; do echo "$ENVV python -I scripts/score_folds.py --folds 9 --workers 10 --out $E/scores_opp_hd --opponent --shard $((7 + 16 * j)) --nshards 128" >> $A/m135_r2_opp7.txt; done
  grep -q "fold 9 done" logs/nano4/pack-775763-5.log || echo "$ENVV python -I scripts/score_folds.py --folds 9 --workers 10 --out $E/scores_hd --shard 5 --nshards 16" >> $A/m135_r3_s5.txt
  grep -q "fold 9 done" logs/nano4/pack-775761-5.log || echo "$ENVV python -I scripts/score_folds.py --folds 9 --workers 10 --out $E/scores_opp_hd --opponent --shard 5 --nshards 16" >> $A/m135_r3_s5.txt
  R="$(submit m135r2 "-p 16gpus --gres=gpu:8 -c 96 -t 02:00:00" bash $A/pack8.sh $A/m135_r2_opp7.txt)"
  [ -s $A/m135_r3_s5.txt ] && R="$R,$(submit m135r3 "-p 8gpus --gres=gpu:2 -c 24 -t 02:00:00" bash $A/pack8.sh $A/m135_r3_s5.txt)"
  echo "recovery jobs $R"
  echo "== $(date '+%F %T') trunk vectors"; T=$(bash logs/nano4/m135_trunk.sh 12); echo "$T"; V=$(echo "$T" | grep "^trunk jobs" | sed 's/trunk jobs //; s/ /,/g')
  wait_jobs "$R,776187${V:+,$V}"; for j in ${R//,/ } 776187 ${V//,/ }; do echo "$j $(state $j) $(sacct -n -X -j $j -o JobName,Elapsed | head -1)"; done
  ok_all "$R,776187${V:+,$V}" || { echo "a job failed: stop (every pass resumes per game)"; return 1; }
  echo "== $(date '+%F %T') score array, index, game tables, questions, checks"; df -h /work/u2507115 | tail -n 1; du -sh $E
  B=$(submit m135b "-p 8gpus -c 12 -t 02:00:00" bash -c "$TR build --ext $E && python -I logs/nano4/m135_smoke_check.py $E")
  wait_jobs "$B"; echo "build job $B $(state $B)"; grep -v "^==\|^+" logs/nano4/m135b-$B.out | tail -26 | cut -c1-260
  ok_all "$B" || { echo "build failed: stop"; return 1; }
  du -sh $E $E/*; df -h /work/u2507115 | tail -n 1
  echo "== $(date '+%F %T') recovery 2 done: remove the per-move score files, then train the members"
}
main "$@"; exit $?
