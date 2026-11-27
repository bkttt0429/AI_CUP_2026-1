# 由本資料夾的 Slurm 工作腳本在設定好 ROOT（執行 sbatch 時所在的目錄）之後 source 載入：
# 進入專案、把 .venv 放到 PATH 的最前面，並印出工作標頭。
cd "$ROOT"  # 進入專案根目錄，之後的相對路徑都以它為準
if [ ! -x .venv/bin/python ]; then  # 檢查專案環境是否已經建立
  echo "no .venv in $ROOT: run 'bash scripts/nano4/setup.sh' on the login node first" >&2  # 提示先在登入節點建立環境
  exit 1  # 沒有環境就中止工作
fi  # 結束環境檢查
export PATH="$ROOT/.venv/bin:$PATH" PYTHONUNBUFFERED=1  # 指令中的 python 指向 .venv；輸出不緩衝，log 即時更新
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"  # 可用的 CPU 核心數：優先用 Slurm 配給本工作的數量，沒有時用 nproc
WORKERS=$(( CPUS > 3 ? CPUS - 2 : 1 ))  # score_folds.py／predict_submission.py 的 CPU worker 數（多於 3 核時少用 2 核）
echo "== $(date '+%F %T') job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-no name}) on $(hostname)" "| partition ${SLURM_JOB_PARTITION:-none} | GPUs ${CUDA_VISIBLE_DEVICES:-none} | CPUs $CPUS | $ROOT"  # 工作標頭：時間、工作編號與名稱、節點、分區、GPU、CPU 與專案路徑
trap 'status=$?; echo "== $(date "+%F %T") exit $status"' EXIT  # 工作結束時印出時間與結束碼
