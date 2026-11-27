#!/usr/bin/env bash
# M33 diagnostics (report only): -pb (same-player paired batches, no InfoNCE) and -rr (ct7's recipe re-run with the current code),
# 24 members each. Stage 0 tests; stage 1 48 members (6 x 8-GPU packs); stage 2 protocol check (stop on FAIL); stage 3 ensembles.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --save-dir results/models/ct_cv_m33d --out results/trunk_seq/es"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M33d stage 0: tests"
T=$(submit m33dtest "-c 12 -t 00:30:00" python -I -m pytest -q tests/test_trunk_seq_m33.py tests/test_trunk_seq_m30.py tests/test_trunk_predict.py tests/test_seq.py)
wait_jobs "$T"; grep -E "passed|failed" logs/nano4/m33dtest-$T.out | tail -2
ok_all "$T" || { echo "tests failed: stop"; exit 1; }
echo "== $(date '+%F %T') M33d stage 1: 48 members"
rm -f $A/m33d_cmds_*
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --pair-batches" >> $A/m33d_cmds_all.txt
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --tag-extra rr" >> $A/m33d_cmds_all.txt
done; done; done
split -l 8 -d $A/m33d_cmds_all.txt $A/m33d_cmds_
P=""; for f in $A/m33d_cmds_0*; do P="$P,$(submit m33dp$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"
echo "== $(date '+%F %T') M33d stage 2: protocol check"
C=$(submit m33dchk "-c 12 -t 00:20:00" python -I logs/nano4/m33_diag_check.py); wait_jobs "$C"
grep -v "^==\|^+" logs/nano4/m33dchk-$C.out | tail -50
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M33d stage 3: ensembles (report only)"
O="C-f0-x40-oh T-f0-x40-oh"; PC="C-pc-f0-x40-oh T-pc-f0-x40-oh"; PB="C-pb-f0-x40-oh T-pb-f0-x40-oh"; RR="C-f0-x40-oh-rr T-f0-x40-oh-rr"
E=""; i=0
for spec in "$O|$PB|pb_vs_ct7" "$O|$RR|rr_vs_ct7" "$PB|$PC|pc_vs_pb" "$RR|$PC|pc_vs_rr" "$PC|$O $PC|ct7pc_vs_pc"; do
  IFS='|' read -r old new name <<< "$spec"; i=$((i + 1))
  E="$E,$(submit m33dens$i "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $old --new $new --seeds 0 1 2 --out results/trunk_seq/es/ens_m33d_$name.json)"
done
wait_jobs "$E"
for j in ${E//,/ }; do [ -z "$j" ] && continue; echo "== $j $(state $j)"; grep -E " vs " logs/nano4/m33dens*-$j.out | head -1; grep -E "^written" logs/nano4/m33dens*-$j.out; done
echo "== $(date '+%F %T') M33d done"
