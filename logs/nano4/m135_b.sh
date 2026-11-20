#!/usr/bin/env bash
# M135 driver, part B (login node: submits and waits only; start it after the per-move score files of part A are removed):
# the external set's trunk vectors (int8, 16 shards on two nodes), a trainer smoke on the real set, the 24 members with --ext
# (C / T x folds 1-4 x seeds 0-2), the comparisons against the current members.
#   nohup bash logs/nano4/m135_b.sh > logs/nano4/m135_b.log 2>&1 &
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"
E=/work/u2507115/ext_rank
ENVV="AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv"
NEED_GB=${NEED_GB:-6}   # free space on /work required before the trunk vectors (about 4.6 GB) are written
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 45; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 240); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 45; done; return 1; }
main() {
  mkdir $E/.m135_b.lock 2>/dev/null || { echo "part B was ALREADY STARTED ($(cat $E/.m135_b.lock/who 2>/dev/null)): not starting it twice; its log is logs/nano4/m135_b.log.lgn02"; return 0; }
  echo "started $(date '+%F %T') on $(hostname), pid $$" > $E/.m135_b.lock/who
  echo "== $(date '+%F %T') M135 part B stage 3: trunk vectors"
  free=$(df --output=avail -BG /work/u2507115 | tail -n 1 | tr -dc '0-9'); echo "/work free: ${free} GB (need ${NEED_GB})"; du -sh $E
  [ -n "${TRUNK_JOBS:-}" ] || [ "$free" -ge "$NEED_GB" ] || { echo "not enough free space on /work: stop (nothing submitted)"; return 1; }
  [ -f $E/scores_oh_fold9.npy ] && [ -f $E/scores_index_fold9.npz ] && [ -f $E/r_ext_p0.npz ] || { echo "part A's outputs are missing: stop"; return 1; }
  if [ -n "${TRUNK_JOBS:-}" ]; then  # the extraction was submitted earlier (logs/nano4/m135_trunk.sh): only wait for it
    V="$TRUNK_JOBS"
  else
    rm -f $A/m135_trunk_*.txt
    for k in $(seq 0 15); do
      echo "$ENVV python -I scripts/trunk_extract.py --folds 9 --folds-csv $E/player_folds.csv --out $E/trunk_vec --workers 10 --quant8 results/m90/glstats.npz --shard $k --nshards 16" >> $A/m135_trunk_$((k / 8)).txt
    done
    V="$(submit m135v0 "-p 8gpus --gres=gpu:8 -c 96 -t 04:00:00" bash $A/pack8.sh $A/m135_trunk_0.txt)"
    V="$V,$(submit m135v1 "-p 16gpus --gres=gpu:8 -c 96 -t 04:00:00" bash $A/pack8.sh $A/m135_trunk_1.txt)"
  fi
  echo "trunk jobs $V"; wait_jobs "$V"; for j in ${V//,/ }; do echo "$j $(state $j) $(sacct -n -X -j $j -o JobName,Elapsed | head -1)"; done
  ok_all "$V" || { echo "a trunk job failed: stop (it resumes per game: rerun the failed command)"; return 1; }
  du -sh $E $E/trunk_vec; df -h /work/u2507115 | tail -n 1
  if [ -z "${SKIP_SMOKE:-}" ]; then  # SKIP_SMOKE=1: the full external set was already loaded and checked by the build job (ext_tokens), so the packs themselves are the test
    echo "== $(date '+%F %T') M135 part B stage 4: trainer smoke on the real set"
    SM="--limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m135smoke_real --ext $E"
    T=$(submit m135st "-p 8gpus -c 12 -t 01:00:00" bash -c "$TR train --arm T --val-fold 1 --seed 0 $REC $SM")
    wait_jobs "$T"; echo "smoke $T $(state $T)"; grep -E "external|tokens loaded|stopped at step|Error|error|Traceback" logs/nano4/m135st-$T.out | tail -8 | cut -c1-260
    ok_all "$T" || { echo "trainer smoke failed: stop"; return 1; }
  fi
  echo "== $(date '+%F %T') M135 part B stage 5: 48 members with --ext (seeds 0-2 and, at the same time, the replication seeds 6-8)"
  rm -f $A/m135_cmds_*
  for s in 0 1 2 6 7 8; do for v in 1 2 3 4; do for arm in C T; do
    echo "$TR train --arm $arm --val-fold $v --seed $s $REC --ext $E --save-dir results/models/ct_cv_m135 --out results/trunk_seq/es" >> $A/m135_cmds_all.txt
  done; done; done
  split -l 8 -d $A/m135_cmds_all.txt $A/m135_cmds_
  P=""; k=0
  for f in $A/m135_cmds_0*; do
    if [ $((k % 3)) -eq 2 ]; then part="-p 8gpus"; else part="-p 16gpus"; fi  # 19:20: the other session has six jobs on 8gpus until about 20:40 and asked for four packs on 16gpus, two on 8gpus
    P="$P,$(submit m135p$k "$part --gres=gpu:8 -c 96 -t 03:00:00" bash $A/pack8.sh $f)"; k=$((k + 1))
  done
  echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
  ok_all "$P" || echo "a pack failed: the comparisons below show what is missing"
  echo "== $(date '+%F %T') M135 part B stage 6: comparisons"
  t="pc-f0-x40-oh-ex"; O="C-pc-f0-x40-oh T-pc-f0-x40-oh"; N="C-$t T-$t"; EC="python -I logs/nano4/ens_compare.py"
  J=$(submit m135c "-p 8gpus -c 12 -t 01:30:00" bash -c "$TR compare --val-folds 1 2 3 4 --arms $O $N --seeds 0 1 2 --pairs C-$t:C-pc-f0-x40-oh T-$t:T-pc-f0-x40-oh --tag m135_ex_ --out results/trunk_seq/es; $EC --old $O --new $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex.json; $EC --old $O --new $N --seeds 6 7 8 --out results/trunk_seq/es/ens_m135_ex_s6.json; $EC --old $O --new $O $N --seeds 0 1 2 --out results/trunk_seq/es/ens_m135_ex_added.json; python -I logs/nano4/m135_final_cv.py --new $N --seeds 0 1 2 --seeds2 6 7 8")
  wait_jobs "$J"; echo "== comparison job $J $(state $J)"; grep -v "^==\|^+" logs/nano4/m135c-$J.out | tail -70 | cut -c1-330
  echo "== $(date '+%F %T') M135 part B done"
}
main "$@"; exit $?
