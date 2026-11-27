#!/bin/bash
# Step 2 (architecture v2 section 4C): the stacker now; three more question draws per fold; a smoke run of the three flags; the 72 variant runs
# (missing results resubmitted, refused submits retried); then the evaluation.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
ES=results/trunk_seq/es
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
wait_for() { until [ -z "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 60; done; }
st=$(submit $EX -c 12 -t 01:00:00 -J stackgb scripts/nano4/run.sbatch python -I logs/nano4/stack_gbdt.py)
ids=""
for v in 1 2 3 4; do for k in 1 2 3; do
  [ -f results/trunk_seq/r_val${v}_p$k.npz ] && continue
  ids="$ids,$(submit $EX --mem=190G -t 01:30:00 -J rp$v$k scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py rfeat --val-fold $v --pool $k)"
done; done
[ -n "$ids" ] && wait_for "${ids#,}"
echo "$(date +%T) draws: $(ls results/trunk_seq/r_val*_p*.npz | wc -l) of 12"
wait_for $st; echo "=== stacker"; grep -v -E '^(\+|== 2026)' logs/nano4/stackgb-$st.out
sm=$(submit $EX --mem=190G -t 01:00:00 -J tsmk3 scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm T --val-fold 1 --seed 0 --epochs 3 --limit 3200 --inner-frac 0.1 --inner-questions 1000 --resample 4 --uniqueness --bag 0.8 --out results/trunk_seq/smoke)
wait_for $sm; echo "=== smoke"; grep -E 'training draws|stopped at|Error|error' logs/nano4/tsmk3-$sm.out | head -5
grep -q 'stopped at' logs/nano4/tsmk3-$sm.out || { echo "smoke failed, stopping"; exit 1; }
specs=()
for v in 1 2 3 4; do for s in 0 1 2; do for arm in C T; do
  specs+=("rs$arm$v$s|$arm-rs|$v|$s|train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --resample 4 --out $ES")
  specs+=("uq$arm$v$s|$arm-uq|$v|$s|train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --uniqueness --out $ES")
  specs+=("bg$arm$v$s|$arm-bg|$v|$s|train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --bag 0.8 --out $ES")
done; done; done
while :; do
  missing=0; queued=$(squeue -u $USER -h -o '%j')
  for sp in "${specs[@]}"; do
    IFS='|' read -r name tag v s args <<< "$sp"
    [ -f "$ES/val$v/${tag}_s$s.npz" ] && continue
    missing=$((missing + 1))
    grep -qx "$name" <<< "$queued" && continue
    if [ "$(squeue -u $USER -h | wc -l)" -lt 18 ]; then
      j=$(submit $EX --mem=190G -t 03:00:00 -J "$name" scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py $args)
      queued="$queued"$'\n'"$name"; sleep 2
    fi
  done
  [ "$missing" -eq 0 ] && break
  sleep 120
done
echo "$(date +%T) all 72 variant runs present"
e=$(submit $EX -t 01:00:00 -J tsvar scripts/nano4/run.sbatch python -I logs/nano4/ts_variants.py)
wait_for $e; echo "=== variants"; grep -v -E '^(\+|== 2026)' logs/nano4/tsvar-$e.out
