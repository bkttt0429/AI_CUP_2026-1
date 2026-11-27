#!/usr/bin/env bash
# M33: player-contrastive auxiliary loss (--player-contrast 0.1, tag -pc) vs ct7's M22 members. Stage 0 smoke; stage 1 24 members
# (3 x 8-GPU packs); stage 2 member protocol check (stop on FAIL); stage 3 single-arm compare + 6v6 ensemble.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M33 stage 0: smoke"
rm -rf $A/m33smoke; mkdir -p $A/m33smoke
S=$(submit m33smk "-c 12 -t 01:00:00" $TR train --arm C --val-fold 1 --seed 0 $REC --limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m33smoke)
echo "smoke job $S"; wait_jobs "$S"; echo "smoke $(state $S)"; grep -E "InfoNCE|stopped at step|Error|error" logs/nano4/m33smk-$S.out | tail -4 | cut -c1-200
ok_all "$S" || { echo "smoke failed: stop"; exit 1; }
echo "== $(date '+%F %T') M33 stage 1: 24 members"
rm -f $A/m33_cmds_*
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --save-dir results/models/ct_cv_m33 --out results/trunk_seq/es" >> $A/m33_cmds_all.txt
done; done; done
split -l 8 -d $A/m33_cmds_all.txt $A/m33_cmds_
P=""; for f in $A/m33_cmds_0*; do P="$P,$(submit m33p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"
echo "== $(date '+%F %T') M33 stage 2: member protocol check"
C=$(submit m33chk "-c 12 -t 00:20:00" python -I scripts/m_series/m33_protocol_check.py); wait_jobs "$C"
grep -v "^==\|^+" logs/nano4/m33chk-$C.out | tail -26
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M33 stage 3: compare"
K=$(submit m33cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-f0-x40-oh T-f0-x40-oh C-pc-f0-x40-oh T-pc-f0-x40-oh --seeds 0 1 2 --pairs C-pc-f0-x40-oh:C-f0-x40-oh T-pc-f0-x40-oh:T-f0-x40-oh --tag m33_ --out results/trunk_seq/es)
E=$(submit m33ens "-c 12 -t 01:00:00" python -I scripts/m_series/ens_compare.py --old C-f0-x40-oh T-f0-x40-oh --new C-pc-f0-x40-oh T-pc-f0-x40-oh --seeds 0 1 2 --out results/trunk_seq/es/ens_m33.json)
wait_jobs "$K,$E"
echo "== compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m33cmp-$K.out | tail -8
echo "== ensemble 6v6 $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m33ens-$E.out | tail -8
echo "== $(date '+%F %T') M33 done"
