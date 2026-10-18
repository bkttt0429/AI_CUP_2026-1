# Sourced by the Slurm job scripts in this folder once they have set ROOT (the directory sbatch
# was run from): enters the project, puts .venv first on PATH and prints a job banner.
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "no .venv in $ROOT: run 'bash scripts/nano4/setup.sh' on the login node first" >&2
  exit 1
fi
export PATH="$ROOT/.venv/bin:$PATH" PYTHONUNBUFFERED=1
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
WORKERS=$(( CPUS > 3 ? CPUS - 2 : 1 ))  # CPU workers for score_folds.py / predict_submission.py
echo "== $(date '+%F %T') job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-no name}) on $(hostname)" \
  "| partition ${SLURM_JOB_PARTITION:-none} | GPUs ${CUDA_VISIBLE_DEVICES:-none} | CPUs $CPUS | $ROOT"
trap 'status=$?; echo "== $(date "+%F %T") exit $status"' EXIT
