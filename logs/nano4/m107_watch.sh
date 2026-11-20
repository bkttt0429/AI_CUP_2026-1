#!/bin/bash
# Partition 16gpus (jobs of 8-16 GPUs; its own per-user limits: 4 running / 9 submitted) so that the test does not wait for one of
# the 13 slots of 8gpus that the other session's chain keeps full.
# M107 auto-start (2026-11-14, GPU version): (0) as soon as a queue slot is free, a smoke run of the GPU path on the sources that are
# already complete (no score printed); (1) once the external import / extraction jobs have left the queue and the mid-rank
# selection exists, the real test on a full node (8 GPUs, 96 CPUs). The test refuses incomplete sources (INCOMPLETE -> wait, retry);
# a crash of the GPU path (or a failed smoke run) -> the same test with --cpu on the same allocation.
cd /home/u2507115/BK_AICUP_2026-1
SB="sbatch --parsable -p 16gpus --gres=gpu:8 -c 96 --exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155,177]"
X=/work/u2507115/ext_fox; P=/home/u2507115/BK_AICUP_2026-2/data/ext_ply; PY="python -I logs/nano4/m107_multitask_probe.py"
OLD="--src $X/pilot/select.csv $X/pilot/ply --src $X/scale1/select.csv $X/scale1/ply_a,$X/scale1/ply_b,$P/scale1_c 2k,2d,4d"
deadline=$(( $(date +%s) + 5*3600 )); attempt=0; MODE=""; smoke=${SMOKE:-todo}; sj=""
num() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ "$smoke" = todo ]; then out=$($SB -J m107smk scripts/nano4/run.sbatch $PY $OLD --smoke 2>&1); if num "$out"; then sj=$out; smoke=running; echo "$(date +%H:%M) GPU smoke run submitted $sj"; fi; fi
  if [ "$smoke" = running ] && [ -z "$(squeue -h -j $sj 2>/dev/null)" ]; then
    if grep -q "SMOKE: every arm trained" logs/nano4/m107smk-$sj.out 2>/dev/null; then smoke=ok; else smoke=failed; MODE="--cpu"; fi
    echo "$(date +%H:%M) GPU smoke run $sj: $smoke ($(sacct -n -X -j $sj -o Elapsed | tr -d ' '))"; grep -E "allocation|fold data ready|trainings done|Error|error" logs/nano4/m107smk-$sj.out | cut -c1-200 | tail -6
  fi
  busy=$(squeue -h -u u2507115 -o %j | grep -c -E '^p_extplyM|^p_extplyD|^p_extfox4')
  if [ "$busy" -eq 0 ] && [ "$smoke" != running ] && [ -s $X/midrank/select.csv ] && ls $X/midrank/ply/ply_x_*.npz >/dev/null 2>&1 && ls $P/scale2_5d/ply_x_*.npz >/dev/null 2>&1; then
    [ "$smoke" = todo ] && smoke=skipped
    jid=""
    for i in $(seq 1 60); do out=$($SB -J m107 scripts/nano4/run.sbatch $PY $OLD --src $X/scale2_5d/select.csv $P/scale2_5d --src $X/midrank/select.csv $X/midrank/ply $MODE 2>&1); if num "$out"; then jid=$out; break; fi; sleep 30; done
    [ -n "$jid" ] || { echo "$(date +%H:%M) SUBMIT_BLOCKED for 30 minutes"; exit 3; }
    echo "$(date +%F\ %H:%M) m107 $jid $MODE" >> logs/nano4/m90_jobs.txt; echo "$(date +%H:%M) real run submitted $jid ${MODE:-(GPU)}"
    until [ -z "$(squeue -h -j $jid 2>/dev/null)" ]; do sleep 10; done
    if grep -q "^INCOMPLETE" logs/nano4/m107-$jid.out 2>/dev/null; then
      attempt=$((attempt + 1)); echo "$(date +%H:%M) sources incomplete (attempt $attempt)"; grep "selected games without" logs/nano4/m107-$jid.out | cut -c1-200
      [ "$attempt" -ge 8 ] && exit 4; sleep 180; continue
    fi
    if grep -q "^REGISTERED GATE" logs/nano4/m107-$jid.out 2>/dev/null; then
      echo "$(date +%H:%M) job $jid $(sacct -n -X -j $jid -o State,Elapsed | tr -s ' ')"; tail -n +2 logs/nano4/m107-$jid.out | cut -c1-420 | tail -45; exit 0
    fi
    echo "$(date +%H:%M) job $jid ended without a result:"; tail -n 12 logs/nano4/m107-$jid.out | cut -c1-300
    [ -z "$MODE" ] && { MODE="--cpu"; echo "retrying on CPUs"; continue; }
    exit 6
  fi
  sleep 30
done
echo "$(date +%H:%M) TIMEOUT: the external data did not arrive within five hours"; exit 5
