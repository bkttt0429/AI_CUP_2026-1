#!/usr/bin/env bash
# M74 part A: wait for the two state-extraction jobs, build the -ai score arrays, smoke, 48 members (seeds 0-2 first), evaluate.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M74 waiting for the extraction jobs $1"
wait_jobs "$1"; for j in ${1//,/ }; do echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$1" || { echo "an extraction job failed: stop"; grep -l "Error\|Traceback" logs/nano4/shard*-{${1//,/,}}.log 2>/dev/null | head -3; echo "== M74 stopped"; exit 1; }
du -sh results/ai_state/base results/ai_state/x40 | cut -c1-80
echo "== $(date '+%F %T') M74 stage 1: score arrays with the state columns"
B1=$(submit m74b0 "-c 12 -t 02:00:00" env AICUP_AI=1 $TR build --opp-hsl --build-folds 0 1 2 3 4 --cache results/trunk_seq)
B2=$(submit m74b1 "-c 12 -t 02:00:00" env AICUP_AI=1 $TR build --opp-hsl --vec-dir results/trunk_vec_x40 --extra-games 40 --build-folds 0 1 2 3 4 --cache results/trunk_seq)
wait_jobs "$B1,$B2"; echo "build $B1 $(state $B1), $B2 $(state $B2)"; tail -2 logs/nano4/m74b0-$B1.out | cut -c1-200; tail -2 logs/nano4/m74b1-$B2.out | cut -c1-200
ok_all "$B1,$B2" || { echo "a build failed: stop"; tail -6 logs/nano4/m74b0-$B1.out logs/nano4/m74b1-$B2.out | cut -c1-300; echo "== M74 stopped"; exit 1; }
echo "== $(date '+%F %T') M74 stage 2: smoke"
rm -rf $A/m74smoke; mkdir -p $A/m74smoke
S=$(submit m74smk "-c 12 -t 01:00:00" env AICUP_AI=1 $TR train --arm T --val-fold 1 --seed 0 $REC --limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m74smoke)
wait_jobs "$S"; echo "smoke $S $(state $S)"; grep -E "stopped at step|Error|error|tokens" logs/nano4/m74smk-$S.out | tail -4 | cut -c1-240
ok_all "$S" || { echo "smoke failed: stop"; tail -8 logs/nano4/m74smk-$S.out | cut -c1-300; echo "== M74 stopped"; exit 1; }
echo "== $(date '+%F %T') M74 stage 3: 48 members"
rm -f $A/m74_cmds_*
for s in 0 1 2 3 4 5; do for v in 1 2 3 4; do for arm in C T; do
  echo "AICUP_AI=1 $TR train --arm $arm --val-fold $v --seed $s $REC --save-dir results/models/ct_cv_m74_ai --out results/trunk_seq/es" >> $A/m74_cmds_all.txt
done; done; done
split -l 8 -d $A/m74_cmds_all.txt $A/m74_cmds_
P=""; for f in $A/m74_cmds_0*; do P="$P,$(submit m74p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
echo "== $(date '+%F %T') M74 stage 4: evaluation"
E=$(submit m74eval "-c 12 -t 01:00:00" python -I logs/nano4/m74_eval.py); wait_jobs "$E"
echo "== eval $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m74eval-$E.out | cut -c1-330
echo "== $(date '+%F %T') M74 done"
