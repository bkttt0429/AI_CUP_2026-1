#!/usr/bin/env bash
# M69: short-question members (--train-sizes 5 10) on the current recipe, routing by question size. Stage 0 R caches; stage 1 smoke;
# stage 2 48 members (seeds 0-2 first, 6 x 8-GPU packs); stage 3 evaluation with the pre-registered rule (design doc M69).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 --train-sizes 5 10"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M69 stage 0: R caches"
C=""; for v in 1 2 3 4; do C="$C,$(submit m69rf$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --with-fold0 --extra-games 40 --train-sizes 5 10)"; done
echo "rfeat jobs ${C#,}"; wait_jobs "$C"; for j in ${C//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$C" || { echo "an R cache job failed: stop"; for j in ${C//,/ }; do [ -z "$j" ] && continue; tail -3 logs/nano4/m69rf*-$j.out | cut -c1-300; done; echo "== M69 stopped"; exit 1; }
ls -la results/trunk_seq/ | grep "s5-10_x40" | awk '{print $5, $9}'
echo "== $(date '+%F %T') M69 stage 1: smoke"
rm -rf $A/m69smoke; mkdir -p $A/m69smoke
S=$(submit m69smk "-c 12 -t 01:00:00" $TR train --arm C --val-fold 1 --seed 0 $REC --limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m69smoke)
echo "smoke job $S"; wait_jobs "$S"; echo "smoke $(state $S)"; grep -E "InfoNCE|stopped at step|Error|error|questions" logs/nano4/m69smk-$S.out | tail -5 | cut -c1-240
ok_all "$S" || { echo "smoke failed: stop"; tail -8 logs/nano4/m69smk-$S.out | cut -c1-300; echo "== M69 stopped"; exit 1; }
echo "== $(date '+%F %T') M69 stage 2: 48 members"
rm -f $A/m69_cmds_*
for s in 0 1 2 3 4 5; do for v in 1 2 3 4; do for arm in C T; do
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --save-dir results/models/ct_cv_m69 --out results/trunk_seq/es" >> $A/m69_cmds_all.txt
done; done; done
split -l 8 -d $A/m69_cmds_all.txt $A/m69_cmds_
P=""; for f in $A/m69_cmds_0*; do P="$P,$(submit m69p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$P" || echo "a pack failed (the evaluation's member check will name what is missing)"
echo "== $(date '+%F %T') M69 stage 3: evaluation"
E=$(submit m69eval "-c 12 -t 01:00:00" python -I logs/nano4/m69_route_eval.py); wait_jobs "$E"
echo "== eval $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m69eval-$E.out | cut -c1-330
echo "== $(date '+%F %T') M69 done"
