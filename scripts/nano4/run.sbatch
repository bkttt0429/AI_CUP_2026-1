#!/usr/bin/env bash
#SBATCH --job-name=aicup
#SBATCH --account=gov113074
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/nano4/%x-%j.out
# Run one command inside the project env on a GPU node (`python` is .venv's). NCHC policy: all
# computation -- tests and analyses included -- goes through Slurm on the production partitions
# (8gpus takes 1-GPU jobs), never on the login node. Submit from the project root; sbatch options
# before the script name override the defaults above:
#   sbatch scripts/nano4/run.sbatch python -I scripts/predict_submission.py --workers 10
#   sbatch -p ngs1gpu -t 2-00:00:00 -J score scripts/nano4/run.sbatch python -I scripts/score_folds.py --folds 0 --workers 10
#   sbatch -p 8gpus --gres=gpu:8 -c 96 -t 2-00:00:00 -J score8 scripts/nano4/run.sbatch \
#     bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 0 --workers 10
# The log is logs/nano4/<job name>-<jobid>.out.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT/scripts/nano4/common.sh"
if [ $# -eq 0 ]; then
  echo "usage: sbatch [options] scripts/nano4/run.sbatch <command> [args...]" >&2
  exit 2
fi
echo "+ $*"
"$@"
