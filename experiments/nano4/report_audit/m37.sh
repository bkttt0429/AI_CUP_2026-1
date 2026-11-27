#!/usr/bin/env bash
# M37: family stacking on the pc (ct9) members under the new rule. Stage 0 prerequisites (ohr / h4 R caches, h4 + oh score arrays);
# stage 1 216 members (27 x 8-GPU packs, as many at once as the QOS lets through); stage 2 protocol check; stage 3 the two seed sets.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --player-contrast 0.1 --out results/trunk_seq/es"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 600); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M37 stage 0: prerequisites"
B=""
for v in 1 2 3 4; do B="$B,$(submit m37rohr$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --with-fold0 --extra-games 40 --opp-hsl-r)"; done
for v in 1 2 3 4; do B="$B,$(submit m37rh4$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --with-fold0 --extra-games 40 --l1b-set 4)"; done
B="$B,$(submit m37bh4 "-c 12 -t 02:00:00" $TR build --l1b-set 4 --opp-hsl --build-folds 0 1 2 3 4 --cache results/trunk_seq)"
B="$B,$(submit m37bh440 "-c 12 -t 02:00:00" $TR build --l1b-set 4 --opp-hsl --vec-dir results/trunk_vec_x40 --extra-games 40 --build-folds 0 1 2 3 4 --cache results/trunk_seq)"
echo "prerequisite jobs ${B#,}"; wait_jobs "$B"; ok_all "$B" || { echo "a prerequisite failed: stop"; for j in ${B//,/ }; do echo "$j $(state $j)"; done; exit 1; }
ls results/trunk_seq/r_val*_f0_x40_ohr.npz results/trunk_seq/r_val*_f0_h4_x40.npz results/trunk_seq/scores*h4_oh*.npy 2>&1 | tr '\n' ' '; echo
echo "== $(date '+%F %T') M37 stage 1: members"
rm -f $A/m37_cmds_*
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2 3 4 5; do
  [ $s -ge 3 ] && echo "$TR train --arm $arm --val-fold $v --seed $s $REC --opp-hsl --save-dir results/models/ct_cv_m37" >> $A/m37_cmds_all.txt
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --opp-hsl-r --save-dir results/models/ct_cv_m37" >> $A/m37_cmds_all.txt
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --opp-hsl --uniqueness --save-dir results/models/ct_cv_m37" >> $A/m37_cmds_all.txt
  echo "AICUP_HSL=jp $TR train --arm $arm --val-fold $v --seed $s $REC --opp-hsl --save-dir results/models/ct_cv_m37_jp" >> $A/m37_cmds_all.txt
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --opp-hsl --l1b-set 4 --save-dir results/models/ct_cv_m37" >> $A/m37_cmds_all.txt
done; done; done
echo "$(wc -l < $A/m37_cmds_all.txt) member commands"
split -l 8 -d -a 2 $A/m37_cmds_all.txt $A/m37_cmds_
P=""; for f in $A/m37_cmds_[0-9]*; do P="$P,$(submit m37p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; echo "submitted $(basename $f) $(date '+%T')"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"
echo "== $(date '+%F %T') M37 stage 2: protocol check"
C=$(submit m37chk "-c 12 -t 00:20:00" python -I logs/nano4/m37_protocol_check.py); wait_jobs "$C"
grep -v "^==\|^+" logs/nano4/m37chk-$C.out | tail -45
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M37 stage 3: stacking, both seed sets"
S1=$(submit m37s012 "-c 12 -t 01:00:00" python -I logs/nano4/m37_stack.py --seeds 0 1 2 --deploy-seeds 0 1 2 3 4 5 --out results/trunk_seq/es/stack_m37_s012.json)
S2=$(submit m37s345 "-c 12 -t 01:00:00" python -I logs/nano4/m37_stack.py --seeds 3 4 5 --out results/trunk_seq/es/stack_m37_s345.json)
wait_jobs "$S1,$S2"
for j in $S1 $S2; do echo "== $j $(state $j)"; grep -vE "^==|^\+|Warning|warnings.warn" logs/nano4/m37s*-$j.out | tail -12; done
echo "== $(date '+%F %T') M37 done"
