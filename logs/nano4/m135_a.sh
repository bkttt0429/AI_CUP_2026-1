#!/usr/bin/env bash
# M135 driver, part A (login node: submits and waits only): the external set's Human SL passes (four 8-GPU jobs, 16 shards a side)
# and L1b passes (one 8-GPU job), then its score array, index, game tables, questions and checks. The trunk vectors come in part B,
# after the per-move score files are removed (disk: /work is shared with the other session).
#   nohup bash logs/nano4/m135_a.sh > logs/nano4/m135_a.log 2>&1 &
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
E=/work/u2507115/ext_rank
ENVV="AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 45; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 240); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 45; done; return 1; }
main() {
  echo "== $(date '+%F %T') M135 part A stage 1: scoring"
  rm -f $A/m135_hsl_*.txt
  for k in $(seq 0 15); do
    f=$((k / 8))
    echo "$ENVV python -I scripts/score_folds.py --folds 9 --workers 10 --out $E/scores_hd --shard $k --nshards 16" >> $A/m135_hsl_tgt_$f.txt
    echo "$ENVV python -I scripts/score_folds.py --folds 9 --workers 10 --out $E/scores_opp_hd --opponent --shard $k --nshards 16" >> $A/m135_hsl_opp_$f.txt
  done
  S="$(submit m135ho0 "-p 16gpus --gres=gpu:8 -c 96 -t 06:00:00" bash $A/pack8.sh $A/m135_hsl_opp_0.txt)"
  S="$S,$(submit m135ho1 "-p 16gpus --gres=gpu:8 -c 96 -t 06:00:00" bash $A/pack8.sh $A/m135_hsl_opp_1.txt)"
  S="$S,$(submit m135ht0 "-p 8gpus --gres=gpu:8 -c 96 -t 06:00:00" bash $A/pack8.sh $A/m135_hsl_tgt_0.txt)"
  S="$S,$(submit m135ht1 "-p 8gpus --gres=gpu:8 -c 96 -t 06:00:00" bash $A/pack8.sh $A/m135_hsl_tgt_1.txt)"
  S="$S,$(submit m135l "-p 8gpus --gres=gpu:8 -c 96 -t 06:00:00" bash logs/nano4/m135_l1b.sh)"
  echo "scoring jobs $S"; wait_jobs "$S"; for j in ${S//,/ }; do echo "$j $(state $j) $(sacct -n -X -j $j -o JobName,Elapsed | head -1)"; done
  ok_all "$S" || { echo "a scoring job failed: stop (the passes resume per game: fix and rerun the failed command)"; return 1; }
  echo "== $(date '+%F %T') M135 part A stage 2: score array, index, game tables, questions, checks"; df -h /work/u2507115 | tail -n 1; du -sh $E
  B=$(submit m135b "-p 8gpus -c 24 -t 02:00:00" bash -c "$TR build --ext $E && python -I logs/nano4/m135_smoke_check.py $E")
  wait_jobs "$B"; echo "build job $B $(state $B)"; grep -v "^==\|^+" logs/nano4/m135b-$B.out | tail -24 | cut -c1-260
  ok_all "$B" || { echo "build failed: stop"; return 1; }
  du -sh $E $E/*; df -h /work/u2507115 | tail -n 1
  echo "== $(date '+%F %T') M135 part A done: remove the per-move score files, then start part B"
}
main "$@"; exit $?
