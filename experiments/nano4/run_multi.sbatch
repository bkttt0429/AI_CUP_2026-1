#!/usr/bin/env bash
#SBATCH --job-name=multi
#SBATCH --account=gov113074
#SBATCH --partition=128gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --no-kill
#SBATCH --output=logs/nano4/%x-%j.out
# A multi-node job whose command (a driver, logs/nano4/multi_drive.py) runs on the first node and starts full-node steps on
# the job's nodes. Since 2026-11-17 the small partitions take no new jobs (group cap of 64 GPUs for all users) and the larger
# ones need more than one node per job (128gpus: at least 65 GPUs = 9 nodes, 2 running jobs per user). Pattern taken from the
# player-identification session's scripts/nano4/run_multi.sbatch. Submit from the project root:
#   sbatch -N 9 -J m217multi --exclude=<bad nodes> scripts/nano4/run_multi.sbatch python3 -u logs/nano4/multi_drive.py --spool logs/nano4/multi_spool
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT/scripts/nano4/common.sh"
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "+ $*"
"$@"
