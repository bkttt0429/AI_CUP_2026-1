#!/bin/bash
# Third round, robust: submit every missing (arm, fold, seed) whose result file is absent and whose job is not queued; retry when the
# submit cap refuses; then the comparisons.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
ES=results/trunk_seq/es
specs=()
for v in 1 2 3 4; do
  for s in 0 1 2; do for drop in trunk scores; do specs+=("es${drop:0:2}$v$s|C-no$drop|$v|$s|train --arm C --drop $drop --val-fold $v --seed $s --inner-frac 0.1 --out $ES"); done; done
  for s in 3 4 5; do for arm in C T; do specs+=("es$arm$v$s|$arm|$v|$s|train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --out $ES"); done; done
done
submit() {  # name, args...; retries until sbatch accepts
  local name=$1; shift
  until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done
  echo "$j"
}
while :; do
  missing=0
  queued=$(squeue -u $USER -h -o '%j')
  for sp in "${specs[@]}"; do
    IFS='|' read -r name tag v s args <<< "$sp"
    [ -f "$ES/val$v/${tag}_s$s.npz" ] && continue
    missing=$((missing + 1))
    grep -qx "$name" <<< "$queued" && continue
    if [ "$(squeue -u $USER -h | wc -l)" -lt 18 ]; then
      j=$(submit "$name" $EX --mem=190G -t 03:00:00 -J "$name" scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py $args)
      echo "$(date +%T) submitted $name $j"; queued="$queued"$'\n'"$name"; sleep 2
    fi
  done
  [ "$missing" -eq 0 ] && break
  # a job that ended without its file failed: say so once, it is resubmitted on the next pass
  sleep 120
done
echo "$(date +%T) all 48 result files present"
j=$(submit x $EX -t 00:30:00 -J esabl scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 1 2 3 4 --arms C C-notrunk C-noscores --pairs C-notrunk:C C-noscores:C --seeds 0 1 2 --tag abl_ --out $ES)
k=$(submit x $EX -t 01:30:00 -J esens scripts/nano4/run.sbatch python -I logs/nano4/ts_vs_mlp.py --seeds 3 4 5 --arms C T --tag _s345)
m=$(submit x $EX -t 01:30:00 -J esens6 scripts/nano4/run.sbatch python -I logs/nano4/ts_vs_mlp.py --seeds 0 1 2 3 4 5 --arms C T --tag _s0_5)
until [ -z "$(squeue -h -j $j,$k,$m 2>/dev/null)" ]; do sleep 30; done
echo "=== ablation"; grep -v -E '^(==|\+)' logs/nano4/esabl-$j.out
echo "=== ensemble, fresh seeds 3-5 (pre-specified)"; grep -v -E '^(\+|== 2026)' logs/nano4/esens-$k.out
echo "=== ensemble, seeds 0-5 (report only)"; grep -v -E '^(\+|== 2026)' logs/nano4/esens6-$m.out
