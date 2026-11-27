#!/usr/bin/env bash
# Same o26 recipe, disjoint output folders, resume checkpoints enabled.
set -euo pipefail
group=${1:?expected all, 0 or 1}
case "$group" in
  all) members=(0 1 2 3 4 5 6 7);;
  0) members=(0 1 4 5);;
  1) members=(2 3 6 7);;
  *) exit 2;;
esac
r2=/home/u2507115/BK_AICUP_2026-2
x=/work/u2507115/ext_fox
out=/work/u2507115/pid_v2/o26_parallel_run_20261116/g${group}
data="--ext-ply $x/pilot/ply $x/scale1/ply_a $x/scale1/ply_b $r2/data/ext_ply/scale1_c $r2/data/ext_ply/scale2_top $r2/data/ext_ply/scale2_5d --ext-select $x/pilot/select.csv $x/scale1/select.csv $x/scale2_top/select.csv $x/scale2_5d/select.csv"
common="--steps 40000 --horizon 60000 --ext-share 0.5 --sym-aug --mask 0.3 --ema-only --own-by-games linear --drop-fold 4 --out $out"
specs=()
for i in "${members[@]}"; do
  seed=$((180 + i))
  if (( i < 4 )); then
    specs+=("-1 o26Ss$i $common --seed $seed")
  else
    specs+=("-1 o26SMs$((i-4)) $common --drop-open 0.3 --seed $seed")
  fi
done
if [[ "${2:-}" == --dry-run ]]; then
  printf '%s\n' "${specs[@]}"
  exit 0
fi
[[ ! -e "$out" ]] || { echo "output already exists; inspect before explicit resume: $out"; exit 1; }
mkdir -p "$out"
printf '%s\n' "${specs[@]}" > "$out/specs.txt"
sha256sum "$r2/scripts/pid_seq_train.py" "$r2/scripts/nano4/pid_seq_train_fork.sh" > "$out/source.sha256"
bash "$r2/scripts/nano4/pid_seq_train_fork.sh" "$data" "${specs[@]}"
