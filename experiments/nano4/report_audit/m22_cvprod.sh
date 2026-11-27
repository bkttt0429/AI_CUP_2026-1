#!/usr/bin/env bash
# M22 (2026-11-05): 24 CV-validated production members (fold 0 + 3 folds, held-out fold v = 1..4, x40, oh, inner 0.1), weights saved;
# 4-fold 6v6 vs the 3-fold x40 members; then ct7 from the results/ct6 shards (opponent HSL present).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19; AUD=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; while [ -z "$j" ]; do while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done; j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }; done; echo "$j"; }
echo "== $(date '+%F %T') M22 start"
C=""; for v in 3 4; do C="$C,$(submit rf0v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --with-fold0 --extra-games 40)"; done
echo "caches ${C#,}"; wait_jobs "$C"; ok_all "$C" || { echo "cache failed"; exit 1; }
rm -f $AUD/m22_cmds_*.txt
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --save-dir results/models/ct_cv --out results/trunk_seq/es" >> $AUD/m22_cmds_all.txt
done; done; done
split -l 8 -d $AUD/m22_cmds_all.txt $AUD/m22_cmds_
ids=""; for f in $AUD/m22_cmds_0*; do ids="$ids,$(submit m22p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $AUD/pack8.sh $f)"; done
echo "packed jobs ${ids#,}" | tee $AUD/m22_pack_jobs.txt
wait_jobs "$ids"; for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/m22p*-$j.out | cut -c1-140; done
ok_all "$ids" || { echo "a pack failed"; }
K=$(submit m22cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-x40 T-x40 C-f0-x40-oh T-f0-x40-oh --seeds 0 1 2 --pairs C-f0-x40-oh:C-x40 T-f0-x40-oh:T-x40 --tag m22_ --out results/trunk_seq/es)
E=$(submit m22ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40 T-x40 --new C-f0-x40-oh T-f0-x40-oh --seeds 0 1 2 --out results/trunk_seq/es/ens_m22.json)
wait_jobs "$K,$E"
echo "== M22 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m22cmp-$K.out
echo "== M22 ensemble 6v6 x 4 folds $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m22ens-$E.out
MODELS=""; for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do MODELS="$MODELS results/models/ct_cv/val${v}_$arm-f0-x40-oh_s$s.pt"; done; done; done
echo "checkpoints present: $(ls $MODELS 2>/dev/null | wc -l)/24"
P=$(submit ctpred7 "-t 01:00:00" python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $MODELS --fallback-submission submission_rank_ct3.csv --out submission_rank_ct7.csv)
wait_jobs "$P"; echo "== ctpred7 $P $(state $P)"; grep -v "^+" logs/nano4/ctpred7-$P.out | tail -3 | cut -c1-700; md5sum submission_rank_ct7.csv 2>/dev/null
echo "== $(date '+%F %T') M22 done"
