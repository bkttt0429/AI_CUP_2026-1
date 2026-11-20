#!/bin/bash
# Third round: ablations (seeds 0-2) and fresh seeds 3-5 of C and T, folds 1-4; never more than 16 of the user's jobs at once.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
cmds=()
for v in 1 2 3 4; do
  for s in 0 1 2; do for drop in trunk scores; do cmds+=("es${drop:0:2}$v$s|train --arm C --drop $drop --val-fold $v --seed $s --inner-frac 0.1 --out results/trunk_seq/es"); done; done
  for s in 3 4 5; do for arm in C T; do cmds+=("es$arm$v$s|train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --out results/trunk_seq/es"); done; done
done
ids=""
for c in "${cmds[@]}"; do
  while [ "$(squeue -u $USER -h | wc -l)" -ge 16 ]; do sleep 60; done
  j=$(sbatch --parsable $EX --mem=190G -t 03:00:00 -J "${c%%|*}" scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py ${c#*|})
  ids="$ids,$j"; sleep 2
done
until [ -z "$(squeue -h -j ${ids#,} 2>/dev/null)" ]; do sleep 60; done
echo "jobs: $(sacct -j ${ids#,} -X --format=State -n | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
j=$(sbatch --parsable $EX -t 00:30:00 -J esabl scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 1 2 3 4 --arms C C-notrunk C-noscores --pairs C-notrunk:C C-noscores:C --seeds 0 1 2 --tag abl_ --out results/trunk_seq/es)
k=$(sbatch --parsable $EX -t 01:30:00 -J esens scripts/nano4/run.sbatch python -I logs/nano4/ts_vs_mlp.py --seeds 3 4 5 --arms C T --tag _s345)
m=$(sbatch --parsable $EX -t 01:30:00 -J esens6 scripts/nano4/run.sbatch python -I logs/nano4/ts_vs_mlp.py --seeds 0 1 2 3 4 5 --arms C T --tag _s0_5)
until [ -z "$(squeue -h -j $j,$k,$m 2>/dev/null)" ]; do sleep 30; done
echo "=== ablation"; grep -v -E '^(==|\+)' logs/nano4/esabl-$j.out
echo "=== ensemble, fresh seeds 3-5"; grep -v -E '^(\+|== 2026)' logs/nano4/esens-$k.out
echo "=== ensemble, seeds 0-5"; grep -v -E '^(\+|== 2026)' logs/nano4/esens6-$m.out
