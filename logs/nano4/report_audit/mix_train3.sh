#!/usr/bin/env bash
# Section 4L.4 wave 3: seeds 3-5 of C-mix50 / T-mix50 on folds 1-4 (24 jobs, cap-limited), the fold-2 C-mix100 report-only arm with
# a rebuilt cache (held-out players' questions now stay unmixed), then the ensemble adoption check (12 mix vs 12 old).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=10
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 60; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
rm -f results/trunk_seq/r_val2_mix100.npz results/trunk_seq/r_val2_mix100.json
T=$(submit mixtest2 "-c 12 -t 01:00:00" python -m pytest -q -x tests/test_trunk_seq_mix.py)
R=$(submit rmix2f "-c 12 -t 02:00:00" python -I scripts/trunk_seq_check.py rfeat --val-fold 2 --mix 1.0 --inner-frac 0.1)
echo "mixtest2 $T rmix2f $R"
ids=""
for v in 1 2 3 4; do
  for arm in C T; do
    for seed in 3 4 5; do
      j=$(submit mx$arm$v$seed "-t 02:30:00" python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed $seed --inner-frac 0.1 --mix 0.5 --out results/trunk_seq/es)
      echo "mx$arm$v$seed $j"; ids="$ids,$j"
    done
  done
done
wait_jobs "$T,$R"
echo "mixtest2 $(state $T), rmix2f $(state $R)"
for seed in 0 1 2; do
  j=$(submit mxC2f$seed "-t 02:30:00" python -I scripts/trunk_seq_check.py train --arm C --val-fold 2 --seed $seed --inner-frac 0.1 --mix 1.0 --out results/trunk_seq/es)
  echo "mxC2f$seed $j"; ids="$ids,$j"
done
echo "jobs ${ids#,}" | tee /home/u2507115/.cache/report_audit/mix_train3_jobs.txt
wait_jobs "${ids#,}"
for j in ${ids//,/ }; do echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/mx*-$j.out | sed 's/.*stopped/stopped/')"; done
C2=$(submit mixcmp2 "-c 12 -t 01:00:00" python -I scripts/trunk_seq_check.py compare --val-folds 2 --arms C C-mix50 C-mix100 --seeds 0 1 2 --pairs C-mix100:C C-mix100:C-mix50 --tag mix100_ --out results/trunk_seq/es)
E=$(submit mixens "-c 12 -t 01:00:00" python -I logs/nano4/mix_ensemble.py)
wait_jobs "$C2,$E"
echo "== compare fold 2 mix100 $C2 $(state $C2)"; grep -v "^==\|^+" logs/nano4/mixcmp2-$C2.out
echo "== ensemble $E $(state $E)"; grep -v "^==\|^+" logs/nano4/mixens-$E.out
echo "wave3 done"
