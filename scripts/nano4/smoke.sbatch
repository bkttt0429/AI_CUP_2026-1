#!/usr/bin/env bash
#SBATCH --job-name=aicup-smoke
#SBATCH --account=gov113074
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/nano4/%x-%j.out
# First job on nano4, about 20 minutes: the GPU stack, the test suite, H200 throughput
# (results/bench_nano4.json), then 200 fold-0 games rescored and compared game by game with
# the lab01 scores. Submit from the project root after setup.sh:
#   sbatch scripts/nano4/smoke.sbatch
# The log (logs/nano4/aicup-smoke-<jobid>.out) ends with SMOKE PASSED or names the failed step.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT/scripts/nano4/common.sh"
step() { STEP="$1"; printf '\n== %s %s\n' "$(date +%T)" "$1"; }
trap 'echo "SMOKE FAILED at step: ${STEP:-start}"' ERR

step "hardware"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "nproc $(nproc) | $(free -g | awk '/^Mem:/ {print $2}') GB RAM"
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURMD_NODENAME:-}" ]; then
  scontrol show node "$SLURMD_NODENAME" | grep -o -E "(CPUTot|RealMemory|Gres)=[^ ]+" || true
fi

step "torch on the GPU"
python -I - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
x = torch.randn(4096, 4096, device="cuda")
s = (x @ x).sum().item()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}) on "
      f"{torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}; matmul ok ({s:.3g})")
EOF

step "tests"
python -I -m pytest -q -p no:cacheprovider tests

step "throughput -> results/bench_nano4.json"
python -I scripts/bench_humansl.py --games 200 --out results/bench_nano4.json

OUT="results/nano4_smoke/${SLURM_JOB_ID:-local}"
step "rescore 200 fold-0 games into $OUT"
python -I scripts/score_folds.py --folds 0 --limit-games 200 --workers "$WORKERS" --out "$OUT"

step "compare with the lab01 scores"
python -I scripts/compare_scores.py "$OUT/fold0" results/scores/games_fold0.csv.gz

echo
echo "SMOKE PASSED"
