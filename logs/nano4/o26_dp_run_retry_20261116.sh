#!/usr/bin/env bash
# Four independent o26 models per node, two GPUs each, unchanged full-episode loss.
set -euo pipefail
group=${1:?expected 0 or 1}
case "$group" in
  0) members=(0 1 4 5);;
  1) members=(2 3 6 7);;
  *) exit 2;;
esac
r2=/home/u2507115/BK_AICUP_2026-2
x=/work/u2507115/ext_fox
out=/work/u2507115/pid_v2/o26_parallel_run_20261116/g${group}r1
IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ ${#cards[@]} == 8 ]] || { echo 'expected eight allocated GPUs'; exit 1; }
common="--steps 40000 --horizon 60000 --ext-share 0.5 --sym-aug --mask 0.3 --ema-only --own-by-games linear --drop-fold 4 --out $out"
lines=()
slot=0
for i in "${members[@]}"; do
  seed=$((180 + i))
  if (( i < 4 )); then tag=o26Ss$i; extra=; else tag=o26SMs$((i-4)); extra='--drop-open 0.3'; fi
  pair=${cards[$((2*slot))]},${cards[$((2*slot+1))]}
  lines+=("$pair logs/nano4/pidseq_-1_${tag}-${SLURM_JOB_ID:-dryrun}.log --net -1 --tag $tag $common $extra --seed $seed")
  slot=$((slot+1))
done
if [[ "${2:-}" == --dry-run ]]; then printf '%s\n' "${lines[@]}"; exit 0; fi
# Check all four GPU pairs before loading 143 GB of data.
for slot in 0 1 2 3; do
  pair=${cards[$((2*slot))]},${cards[$((2*slot+1))]}
  CUDA_VISIBLE_DEVICES="$pair" timeout 45 python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("PAIR PREFLIGHT PASSED", flush=True)'
done
python - <<'PY'
import json, math, re
from pathlib import Path
log = Path('logs/nano4/pidseq_-1_o26probe_dp2_Ss0-785616.log').read_text()
readings = re.findall(r'step (1000|1500)/1500:.*?, ([0-9.]+) s/step', log)
assert {int(s) for s, _ in readings} == {1000, 1500} and len(readings) == 2, readings
times = [float(t) for _, t in readings]
assert all(math.isfinite(t) and t > 0 for t in times) and max(times) <= 0.57 * 0.85, times
assert 'parallel helper v2: explicit BF16' in log
losses = [float(x) for x in re.findall(r'step \d+/1500: loss ([^,]+)', log)]
assert len(losses) == 3 and all(math.isfinite(x) for x in losses), losses
meta = json.loads(Path('/work/u2507115/pid_v2/o26_parallel_probe_20261116/dp2/train_-1_o26probe_dp2_Ss0.json').read_text())
assert meta['step'] == 1500 and meta['horizon'] == 60000 and meta['seed'] == 180
assert meta['games'] == 1858245 and meta['all_folds'] is False
assert 'PASS: per-replica BF16 linear dtype' in Path('logs/nano4/o26_dp_regression_final_20261116.log').read_text()
print(f'ADOPTION GATE PASSED: stable step seconds {times}; BF16 regression passed', flush=True)
PY
[[ ! -e "$out" ]] || { echo "output exists; inspect before explicit resume: $out"; exit 1; }
mkdir -p "$out"
printf '%s\n' "${lines[@]}" > "$out/specs.txt"
sha256sum logs/nano4/o26_dp_train_20261116.py logs/nano4/o26_parallel_forward.py logs/nano4/m88_seqid.py > "$out/source.sha256"
echo 'four models, two GPUs each; BF16; original batch and loss; periodic resume checkpoints enabled'
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I logs/nano4/o26_dp_train_20261116.py --fork "$out/specs.txt" \
  --ext-ply "$x/pilot/ply" "$x/scale1/ply_a" "$x/scale1/ply_b" "$r2/data/ext_ply/scale1_c" "$r2/data/ext_ply/scale2_top" "$r2/data/ext_ply/scale2_5d" \
  --ext-select "$x/pilot/select.csv" "$x/scale1/select.csv" "$x/scale2_top/select.csv" "$x/scale2_5d/select.csv"
