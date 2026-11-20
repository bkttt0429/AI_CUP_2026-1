#!/usr/bin/env bash
# M30: CLIP-style two-tower token encoders (dual, clip) vs ct7's M22 members. Stage 0: GPU smoke of both encoders on 2,000 questions;
# stage 1: 48 members (6 x 8-GPU packs); stage 2: member protocol check (stop on FAIL); stage 3: compare + three 6v6 ensembles.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M30 stage 0: smoke"
rm -rf $A/m30smoke; mkdir -p $A/m30smoke
rm -f $A/m30_smoke_cmds.txt
for enc in dual clip; do echo "$TR train --arm T --val-fold 1 --seed 0 $REC --token-enc $enc --limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m30smoke" >> $A/m30_smoke_cmds.txt; done
S=$(submit m30smk "--gres=gpu:8 -c 96 -t 01:00:00" bash $A/pack8.sh $A/m30_smoke_cmds.txt)
echo "smoke job $S"; wait_jobs "$S"; echo "smoke $(state $S)"; for f in logs/nano4/pack-$S-*.log; do grep -E "InfoNCE|stopped at step|Error|error" $f | tail -3 | cut -c1-200; done
ok_all "$S" || { echo "smoke failed: stop"; exit 1; }
echo "== $(date '+%F %T') M30 stage 1: 48 members"
rm -f $A/m30_cmds_*
for v in 1 2 3 4; do for enc in clip dual; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --token-enc $enc --save-dir results/models/ct_cv_m30 --out results/trunk_seq/es" >> $A/m30_cmds_all.txt
done; done; done; done
split -l 8 -d $A/m30_cmds_all.txt $A/m30_cmds_
P=""; for f in $A/m30_cmds_0*; do P="$P,$(submit m30p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; grep -h "slot .* done\|FAILED" logs/nano4/m30p*-$j.out | cut -c1-140; done
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"
echo "== $(date '+%F %T') M30 stage 2: member protocol check"
C=$(submit m30chk "-c 12 -t 00:20:00" python -I logs/nano4/m30_protocol_check.py); wait_jobs "$C"
grep -v "^==\|^+" logs/nano4/m30chk-$C.out | tail -52
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M30 stage 3: compare"
O="C-f0-x40-oh T-f0-x40-oh"; D="C-dual-f0-x40-oh T-dual-f0-x40-oh"; L="C-clip-f0-x40-oh T-clip-f0-x40-oh"
K=$(submit m30cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms $O $D $L --seeds 0 1 2 --pairs C-clip-f0-x40-oh:C-f0-x40-oh T-clip-f0-x40-oh:T-f0-x40-oh C-dual-f0-x40-oh:C-f0-x40-oh T-dual-f0-x40-oh:T-f0-x40-oh C-clip-f0-x40-oh:C-dual-f0-x40-oh T-clip-f0-x40-oh:T-dual-f0-x40-oh --tag m30_ --out results/trunk_seq/es)
E1=$(submit m30ens1 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $O --new $L --seeds 0 1 2 --out results/trunk_seq/es/ens_m30_clip.json)
E2=$(submit m30ens2 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $O --new $D --seeds 0 1 2 --out results/trunk_seq/es/ens_m30_dual.json)
E3=$(submit m30ens3 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $D --new $L --seeds 0 1 2 --out results/trunk_seq/es/ens_m30_clip_vs_dual.json)
wait_jobs "$K,$E1,$E2,$E3"
echo "== compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m30cmp-$K.out | tail -14
for E in $E1 $E2 $E3; do echo "== ensemble $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m30ens*-$E.out | tail -12; done
echo "== $(date '+%F %T') M30 done"
