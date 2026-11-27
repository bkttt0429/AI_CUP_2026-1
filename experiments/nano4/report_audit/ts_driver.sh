#!/bin/bash
# Second round: R features of folds 2-4, then the three arms per validation fold, one fold at a time (the 20-job submit cap).
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,007,009,039,062]'
wait_for() { until [ -z "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 30; done; }
ids=""
for v in 2 3 4; do j=$(sbatch --parsable $EX --mem=190G -t 01:00:00 -J tsr$v scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py rfeat --val-fold $v); ids="$ids,$j"; done
wait_for "${ids#,}"; echo "R features: $(sacct -j ${ids#,} -X --format=State -n | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
for v in 2 3 4 1; do
  ids=""
  for arm in R0 C T; do for s in 0 1 2; do
    j=$(sbatch --parsable $EX --mem=190G -t 03:00:00 -J es$arm$v$s scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --out results/trunk_seq/es)
    ids="$ids,$j"
  done; done
  wait_for "${ids#,}"
  echo "val $v: $(sacct -j ${ids#,} -X --format=State -n | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
done
mkdir -p results/trunk_seq/es
for spec in "2 3 4" "1 2 3 4"; do
  j=$(sbatch --parsable $EX -t 00:20:00 -J escmp scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds $spec --out results/trunk_seq/es)
  wait_for $j; echo "=== folds $spec"; grep -v -E '^(==|\+)' logs/nano4/escmp-$j.out
done
