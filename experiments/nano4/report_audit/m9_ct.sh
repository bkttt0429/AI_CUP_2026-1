#!/usr/bin/env bash
# Section 4M M9, second half (2026-11-04): after m9_heads.sh ("M9 heads + scoring done") and M8's variant line, build the set-4 score
# arrays, caches, 24 CV members C-h4/T-h4, the pre-registered 12-vs-6 ensemble test (+ same-source control), the 12 production h4
# models with M8's variant flags, a 4-heads re-shard of the official test, submission_rank_ct4.csv (12 h3 + 12 h4 checkpoints), and a
# fold-1 mock sanity run. Cap-limited submissions.
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
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
stops() { for j in ${1//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/*-$j.out 2>/dev/null | sed 's/.*stopped/stopped/' | cut -c1-90)"; done; }
echo "== $(date '+%F %T') M9 C/T start; waiting for m9_heads.sh and M8's variant"
until grep -q "M9 heads + scoring done\|abort" $AUD/m9_heads.log 2>/dev/null; do sleep 120; done
grep -q "abort" $AUD/m9_heads.log && { echo "m9_heads aborted: stop"; exit 1; }
until grep -q "production variant by the pre-registered rule\|abort" $AUD/m8_pipeline.log 2>/dev/null; do sleep 120; done
VAR=$(grep -o "production variant by the pre-registered rule: [a-z0-9]*" $AUD/m8_pipeline.log | tail -1 | awk '{print $NF}')
echo "M8 variant: ${VAR:-unknown} $(date '+%T')"
XF=""; [ "$VAR" = "x40" ] && XF="--extra-games"
# score arrays for set 4 (base folds 0-4 and the x40 games)
B1=$(submit h4b "-c 12 -t 02:00:00" $TR build --l1b-set 4 --build-folds 0 1 2 3 4 --cache results/trunk_seq)
B2=$(submit h4bx "-c 12 -t 02:00:00" $TR build --l1b-set 4 --vec-dir results/trunk_vec_x40 --extra-games --build-folds 0 1 2 3 4 --cache results/trunk_seq)
wait_jobs "$B1,$B2"; echo "build h4 $B1 $(state $B1); build h4 x40 $B2 $(state $B2)"
ok_all "$B1,$B2" || { tail -20 logs/nano4/h4b-$B1.out logs/nano4/h4bx-$B2.out; echo "build failed: abort"; exit 1; }
C=""
for v in 1 2 3 4; do C="$C,$(submit rh4v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --l1b-set 4)"; done
C="$C,$(submit rh4all "-c 12 -t 02:00:00" $TR rfeat --val-fold all --with-fold0 $XF --l1b-set 4)"
echo "caches ${C#,}"; wait_jobs "$C"
ok_all "$C" || { for j in ${C//,/ }; do [ -z "$j" ] && continue; [ "$(state $j)" = "COMPLETED" ] || tail -15 logs/nano4/*-$j.out; done; echo "a cache job failed: abort"; exit 1; }
# 48 trainings packed 8 per 8-GPU job (user: 可以多調顯卡); production FIRST (packs 00-02) so ct4 does not wait for the CV members
rm -f $AUD/m9_cmds_*.txt
for arm in C T; do for s in 0 1 2 3 4 5; do
  echo "$TR train --arm $arm --val-fold all --seed $s --inner-frac 0.1 --with-fold0 $XF --l1b-set 4 --out results/trunk_seq" >> $AUD/m9_cmds_all.txt
done; done
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --l1b-set 4 --out results/trunk_seq/es" >> $AUD/m9_cmds_all.txt
done; done; done
split -l 8 -d $AUD/m9_cmds_all.txt $AUD/m9_cmds_
prod=""; cv=""
for f in $AUD/m9_cmds_0*; do
  j=$(submit h4pack$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 04:00:00" bash $AUD/pack8.sh $f)
  case "$(basename $f)" in m9_cmds_00|m9_cmds_01|m9_cmds_02) prod="$prod,$j";; *) cv="$cv,$j";; esac
done
echo "packed production jobs ${prod#,}; cv jobs ${cv#,}" | tee $AUD/m9_cv_jobs.txt
S=$(grep -o "reshard [0-9]*" $AUD/m9_scoring_jobs.txt | awk '{print $2}')
echo "reshard $S (from m9_heads.sh)"
wait_jobs "$prod,$S"
for j in ${prod//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/h4pack*-$j.out | cut -c1-140; done
echo "reshard $S $(state $S)"
[ "$(state $S)" = "COMPLETED" ] || { tail -20 logs/nano4/sub4-$S.out; echo "re-shard failed: abort ct4"; exit 1; }
if [ "$VAR" = "x40" ]; then M3="results/models/ct/all_C-f0-x40_s*.pt results/models/ct/all_T-f0-x40_s*.pt"; M4="results/models/ct/all_C-f0-h4-x40_s*.pt results/models/ct/all_T-f0-h4-x40_s*.pt"; else M3="results/models/ct/all_C-f0_s*.pt results/models/ct/all_T-f0_s*.pt"; M4="results/models/ct/all_C-f0-h4_s*.pt results/models/ct/all_T-f0-h4_s*.pt"; fi
echo "models h3: $(ls $M3 | wc -l), h4: $(ls $M4 | wc -l)"
P=$(submit ctpred4 "-t 01:00:00" python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards submission_rank4.csv --nshards 8 --vec results/ct_submission/vec --models $M3 $M4 --fallback-submission submission_rank_ct3.csv --out submission_rank_ct4.csv)
wait_jobs "$P"
echo "== ctpred4 $P $(state $P)"; grep -v "^+" logs/nano4/ctpred4-$P.out | tail -4 | cut -c1-700
md5sum submission_rank_ct4.csv
IN="data/mock_tests/fold1_100/rank_prediction_test_public.csv data/mock_tests/fold1_100/rank_prediction_test_private.csv"
OUT=results/mock_fold1_100_h4
Q=$(submit mock4 "--gres=gpu:4 -c 48 -t 02:00:00" bash -c "bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs $IN --expect-rows 100 --expect-public 50 --expect-private 50 --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt results/l1b/heads4_A.pt results/l1b/heads4_B.pt --workers 10 && python -I scripts/trunk_predict.py predict --inputs $IN --expect-rows 100 --expect-public 50 --expect-private 50 --shards $OUT/submission_rank.csv --nshards 4 --vec results/mock_fold1_100/vec --models $M3 $M4 --fallback-submission results/mock_fold1_100/submission_rank.csv --out $OUT/submission_rank_ct4.csv && python -I scripts/trunk_predict.py compare --a $OUT/submission_rank_ct4_proba.csv --b results/mock_fold1_100/submission_rank_ct3_proba.csv --answers data/mock_tests/fold1_100/answers.csv --report $OUT/ct4_vs_ct3.json")
wait_jobs "$cv"
for j in ${cv//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/h4pack*-$j.out | cut -c1-140; done
K=$(submit h4cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-h4 T-h4 --seeds 0 1 2 --pairs C-h4:C T-h4:T --tag m9_ --out results/trunk_seq/es)
E1=$(submit h4ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C T C-h4 T-h4 --seeds 0 1 2 --out results/trunk_seq/es/ens_m9_12v6.json)
E2=$(submit h4ctl "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C T --seeds 0 1 2 --new-seeds 0 1 2 3 4 5 --out results/trunk_seq/es/ens_m9_control.json)
E3=$(submit h4ens6 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-h4 T-h4 --seeds 0 1 2 --out results/trunk_seq/es/ens_m9_6v6.json)
wait_jobs "$K,$E1,$E2,$E3,$Q"
echo "== M9 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/h4cmp-$K.out
echo "== M9 12 (h3+h4) vs 6 (h3) $E1 $(state $E1)"; grep -v "^==\|^+" logs/nano4/h4ens-$E1.out
echo "== M9 control 12 (h3 seeds 0-5) vs 6 (h3 seeds 0-2) $E2 $(state $E2)"; grep -v "^==\|^+" logs/nano4/h4ctl-$E2.out
echo "== M9 6 h4 vs 6 h3 $E3 $(state $E3)"; grep -v "^==\|^+" logs/nano4/h4ens6-$E3.out
echo "== mock ct4 $Q $(state $Q)"; grep -v "^+" logs/nano4/mock4-$Q.out | grep -i "wrote\|max\|decision\|score\|differ\|passes" | cut -c1-400
echo "== $(date '+%F %T') M9 C/T done"
