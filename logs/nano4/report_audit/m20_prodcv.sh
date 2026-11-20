#!/usr/bin/env bash
# M20 (2026-11-05): the production recipe cross-validated -- fold 0 + three folds train, fold v (1, 2) held out, x40, inner 0.1;
# members C/T x seeds 0-2 x {x40, x40-oh} per fold (24 trainings, 3 packs); then compare + ens_compare on folds 1-2.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19; AUD=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; while [ -z "$j" ]; do while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done; j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }; done; echo "$j"; }
echo "== $(date '+%F %T') M20 start"
C=""; for v in 1 2; do C="$C,$(submit rf0v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --with-fold0 --extra-games 40)"; done
echo "caches ${C#,}"; wait_jobs "$C"; ok_all "$C" || { echo "cache failed"; exit 1; }
rm -f $AUD/m20_cmds_*.txt
for v in 1 2; do for var in "" "--opp-hsl"; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 $var --out results/trunk_seq/es" >> $AUD/m20_cmds_all.txt
done; done; done; done
split -l 8 -d $AUD/m20_cmds_all.txt $AUD/m20_cmds_
ids=""; for f in $AUD/m20_cmds_0*; do ids="$ids,$(submit m20p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $AUD/pack8.sh $f)"; done
echo "packed jobs ${ids#,}" | tee $AUD/m20_pack_jobs.txt
wait_jobs "$ids"; for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/m20p*-$j.out | cut -c1-140; done
K=$(submit m20cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 --arms C-f0-x40 T-f0-x40 C-f0-x40-oh T-f0-x40-oh C-x40 T-x40 C-x40-oh T-x40-oh --seeds 0 1 2 --pairs C-f0-x40-oh:C-f0-x40 T-f0-x40-oh:T-f0-x40 C-f0-x40:C-x40 C-f0-x40-oh:C-x40-oh --tag m20_ --out results/trunk_seq/es)
E1=$(submit m20ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-f0-x40 T-f0-x40 --new C-f0-x40-oh T-f0-x40-oh --seeds 0 1 2 --folds 1 2 --out results/trunk_seq/es/ens_m20_f0.json)
E2=$(submit m20ensb "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40 T-x40 --new C-x40-oh T-x40-oh --seeds 0 1 2 --folds 1 2 --out results/trunk_seq/es/ens_m20_nof0.json)
wait_jobs "$K,$E1,$E2"
echo "== M20 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m20cmp-$K.out
echo "== M20 ensemble f0 recipe $E1 $(state $E1)"; grep -v "^==\|^+" logs/nano4/m20ens-$E1.out
echo "== M20 ensemble 3-fold recipe (same folds) $E2 $(state $E2)"; grep -v "^==\|^+" logs/nano4/m20ensb-$E2.out
echo "== $(date '+%F %T') M20 done"
