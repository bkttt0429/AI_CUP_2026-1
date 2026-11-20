#!/usr/bin/env bash
# x80 (2026-11-05, user: 都跑): after the games-41-80 scoring (704395 HSL, 704396-704399 L1b, 704400 trunk): build the x80 score
# arrays, caches, 24 CV members C-x80/T-x80 + 12 production f0-x80 (packed), compare vs the x40 arms (6-vs-6), submission_rank_ct5.csv.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }
  done
  echo "$j"
}
echo "== $(date '+%F %T') x80 pipeline start; waiting for scoring $(cat $AUD/x80_jobs.txt)"
J=$(cat $AUD/x80_jobs.txt | tr ' ' ','); wait_jobs "$J"
for j in ${J//,/ }; do echo "$j $(state $j)"; done
ok_all "$J" || { echo "a scoring job did not complete: abort (parts resume per game)"; exit 1; }
T=$(cat $AUD/x80test_job.txt | awk '{print $2}'); wait_jobs "$T"; [ "$(state $T)" = "COMPLETED" ] || { echo "tests failed: abort"; exit 1; }
du -sh results/scores_hd_x80 results/l1b/*_x80 results/trunk_vec_x80; df -h /home/u2507115 | tail -1
B=$(submit x80b "-c 12 -t 02:00:00" $TR build --vec-dir results/trunk_vec_x80 --extra-games 80 --build-folds 0 1 2 3 4 --cache results/trunk_seq)
wait_jobs "$B"; echo "build x80 $B $(state $B)"; [ "$(state $B)" = "COMPLETED" ] || { tail -20 logs/nano4/x80b-$B.out; exit 1; }
C=""
for v in 1 2 3 4; do C="$C,$(submit rx80v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --extra-games 80)"; done
C="$C,$(submit rallf0x80 "-c 12 -t 02:00:00" $TR rfeat --val-fold all --with-fold0 --extra-games 80)"
echo "caches ${C#,}"; wait_jobs "$C"; ok_all "$C" || { echo "a cache job failed: abort"; exit 1; }
rm -f $AUD/m12_cmds_*.txt
for arm in C T; do for s in 0 1 2 3 4 5; do
  echo "$TR train --arm $arm --val-fold all --seed $s --inner-frac 0.1 --with-fold0 --extra-games 80 --out results/trunk_seq" >> $AUD/m12_cmds_all.txt
done; done
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --extra-games 80 --out results/trunk_seq/es" >> $AUD/m12_cmds_all.txt
done; done; done
split -l 8 -d $AUD/m12_cmds_all.txt $AUD/m12_cmds_
ids=""; for f in $AUD/m12_cmds_0*; do ids="$ids,$(submit x80p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $AUD/pack8.sh $f)"; done
echo "packed jobs ${ids#,}" | tee $AUD/m12_jobs.txt
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/x80p*-$j.out | cut -c1-140; done
K=$(submit x80cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-x40 T-x40 C-x80 T-x80 --seeds 0 1 2 --pairs C-x80:C-x40 T-x80:T-x40 --tag m12_ --out results/trunk_seq/es)
E=$(submit x80ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40 T-x40 --new C-x80 T-x80 --seeds 0 1 2 --out results/trunk_seq/es/ens_m12.json)
P=$(submit ctpred5 "-t 01:00:00" python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models results/models/ct/all_C-f0-x80_s0.pt results/models/ct/all_C-f0-x80_s1.pt results/models/ct/all_C-f0-x80_s2.pt results/models/ct/all_C-f0-x80_s3.pt results/models/ct/all_C-f0-x80_s4.pt results/models/ct/all_C-f0-x80_s5.pt results/models/ct/all_T-f0-x80_s0.pt results/models/ct/all_T-f0-x80_s1.pt results/models/ct/all_T-f0-x80_s2.pt results/models/ct/all_T-f0-x80_s3.pt results/models/ct/all_T-f0-x80_s4.pt results/models/ct/all_T-f0-x80_s5.pt --fallback-submission submission_rank_ct3.csv --out submission_rank_ct5.csv)
wait_jobs "$K,$E,$P"
echo "== x80 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/x80cmp-$K.out
echo "== x80 ensemble 6v6 vs x40 $E $(state $E)"; grep -v "^==\|^+" logs/nano4/x80ens-$E.out
echo "== ctpred5 $P $(state $P)"; grep -v "^+" logs/nano4/ctpred5-$P.out | tail -3 | cut -c1-600; md5sum submission_rank_ct5.csv
echo "== $(date '+%F %T') x80 pipeline done"
