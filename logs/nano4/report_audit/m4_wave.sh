#!/usr/bin/env bash
# Section 4M wave (2026-11-04, user: 好, more GPUs allowed): tests -> M3 extraction (8 GPUs, fine-tuned OOF trunks, in the
# background) alongside the M1/M2 caches and 72 training jobs; M3 build + 24 training jobs after the extraction; compares and
# ensemble checks. Submissions capped at CAP queued jobs (QOS limit 20).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=18
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {  # submit <job name> "<sbatch options>" <command...>
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
stops() { for j in ${1//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/*-$j.out 2>/dev/null | sed 's/.*stopped/stopped/')"; done; }
echo "== $(date '+%F %T') wave start"
# 0. tests
T=$(submit m4test "-c 12 -t 01:00:00" python -m pytest -q -x tests/test_trunk_seq_4m.py tests/test_trunk_seq_mix.py tests/test_trunk_predict.py)
echo "tests $T"; wait_jobs "$T"
if [ "$(state $T)" != "COMPLETED" ]; then echo "tests $(state $T): abort"; tail -30 logs/nano4/m4test-$T.out; exit 1; fi
echo "tests passed $(date '+%T')"
# 1. M3 extraction: folds 1-2 with heads3_B's trunk, folds 3-4 with heads3_A's (the OOF rule of the L1b scores), 8 GPUs
X=$(submit m3vec "--gres=gpu:8 -c 96 -t 04:00:00" bash -c "bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 1 2 --trunk results/l1b/heads3_B.pt --out results/trunk_vec_ft --workers 10 && bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 3 4 --trunk results/l1b/heads3_A.pt --out results/trunk_vec_ft --workers 10")
echo "m3vec $X"
m3_chain() {
  wait_jobs "$X"; echo "m3vec $(state $X) $(date '+%T')"
  [ "$(state $X)" = "COMPLETED" ] || { echo "M3 extraction failed: abort M3"; tail -20 logs/nano4/m3vec-$X.out; return 1; }
  du -sh results/trunk_vec_ft; df -h /home/u2507115 | tail -1
  B=$(submit m3build "-c 12 -t 01:00:00" $TR build --vec-dir results/trunk_vec_ft --cache results/trunk_seq)
  wait_jobs "$B"; echo "m3build $(state $B)"
  [ "$(state $B)" = "COMPLETED" ] || { tail -20 logs/nano4/m3build-$B.out; return 1; }
  m3=""
  for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
    m3="$m3,$(submit ft$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --vec-dir results/trunk_vec_ft --out results/trunk_seq/es)"
  done; done; done
  echo "m3 jobs ${m3#,}" | tee $AUD/m4_m3_jobs.txt
  wait_jobs "$m3"; stops "$m3"
  K=$(submit m3cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-ft T-ft --seeds 0 1 2 --pairs C-ft:C T-ft:T --tag m3_ --out results/trunk_seq/es)
  E=$(submit m3ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-ft T-ft --seeds 0 1 2 --out results/trunk_seq/es/ens_m3.json)
  wait_jobs "$K,$E"
  echo "== M3 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m3cmp-$K.out
  echo "== M3 ensemble $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m3ens-$E.out
  echo "M3 done $(date '+%T')"
}
m3_chain > $AUD/m4_m3.log 2>&1 &
# 2. M1 / M2 caches
C=""
for v in 1 2 3 4; do
  C="$C,$(submit rf2v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --train-folds 2)"
  C="$C,$(submit rg10v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --train-games 10 --train-sizes 5 10)"
  C="$C,$(submit rs510v$v "-c 12 -t 02:00:00" $TR rfeat --val-fold $v --train-sizes 5 10)"
done
echo "caches ${C#,}"; wait_jobs "$C"
for j in ${C//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'questions\|cached' logs/nano4/*-$j.out 2>/dev/null | tail -1)"; done
bad=0; for j in ${C//,/ }; do [ -z "$j" ] && continue; [ "$(state $j)" = "COMPLETED" ] || bad=1; done
if [ "$bad" = 1 ]; then echo "a cache job failed: abort M1/M2 (M3 chain continues)"; for j in ${C//,/ }; do [ -z "$j" ] && continue; [ "$(state $j)" = "COMPLETED" ] || tail -15 logs/nano4/*-$j.out; done; wait; exit 1; fi
# 3. M1 (24: -f2) and M2 (48: -g10s5-10 and its control -s5-10) training jobs
ids=""
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  ids="$ids,$(submit f2$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --train-folds 2 --out results/trunk_seq/es)"
  ids="$ids,$(submit g10$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --train-games 10 --train-sizes 5 10 --out results/trunk_seq/es)"
  ids="$ids,$(submit s510$arm$v$s "-t 02:30:00" $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --train-sizes 5 10 --out results/trunk_seq/es)"
done; done; done
echo "m12 jobs ${ids#,}" | tee $AUD/m4_m12_jobs.txt
wait_jobs "$ids"; stops "$ids"
# 4. compares: M1 = P3 (C, T) - P2 (C-f2, T-f2); M2 = G20 (C-s5-10) - G10 (C-g10s5-10) on n = 5 / 10 questions, n = 20 reported too
K1=$(submit m1cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-f2 T-f2 --seeds 0 1 2 --pairs C:C-f2 T:T-f2 --tag m1_ --out results/trunk_seq/es)
K2=$(submit m2cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-s5-10 T-s5-10 C-g10s5-10 T-g10s5-10 --seeds 0 1 2 --pairs C-s5-10:C-g10s5-10 T-s5-10:T-g10s5-10 --sizes 5 10 --tag m2_n510_ --out results/trunk_seq/es)
K3=$(submit m2cmpall "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-s5-10 T-s5-10 C-g10s5-10 T-g10s5-10 --seeds 0 1 2 --pairs C-s5-10:C-g10s5-10 T-s5-10:T-g10s5-10 C-s5-10:C T-s5-10:T --tag m2_all_ --out results/trunk_seq/es)
E1=$(submit m1ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-f2 T-f2 --new C T --seeds 0 1 2 --out results/trunk_seq/es/ens_m1.json)
E2=$(submit m2ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-g10s5-10 T-g10s5-10 --new C-s5-10 T-s5-10 --seeds 0 1 2 --sizes 5 10 --out results/trunk_seq/es/ens_m2.json)
wait_jobs "$K1,$K2,$K3,$E1,$E2"
for pair in "M1 compare:m1cmp:$K1" "M2 compare n5/10:m2cmp:$K2" "M2 compare all n:m2cmpall:$K3" "M1 ensemble:m1ens:$E1" "M2 ensemble n5/10:m2ens:$E2"; do
  IFS=: read -r label name id <<< "$pair"
  echo "== $label $id $(state $id)"; grep -v "^==\|^+" logs/nano4/$name-$id.out
done
wait
echo "== $(date '+%F %T') wave done (M3 log: $AUD/m4_m3.log)"
