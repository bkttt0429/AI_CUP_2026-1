#!/bin/bash
# Resubmit the five 4F C/T fits that hit CUDA OOM on 25a-hgpn155 (a foreign 130 GB process on the GPU); that node excluded.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062,155]'
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
ids=""
for spec in "C 2 kernel k" "C 2 es-ft e" "T 2 kernel k" "T 2 es-ft e" "C 3 kernel k"; do
  set -- $spec
  ids="$ids,$(submit $EX -c 12 -t 03:30:00 -J ml$1$4$2 scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm $1 --val-fold $2 --seed 0 --inner-frac 0.1 --loss $3 --out results/trunk_seq/es)"
done
echo "resubmitted ${ids#,}"
until [ -z "$(squeue -h -j ${ids#,} 2>/dev/null)" ]; do sleep 60; done
echo "states: $(sacct -j ${ids#,} -X --format=JobName%8,State -n | tr -s ' ' | tr '\n' ';')"
