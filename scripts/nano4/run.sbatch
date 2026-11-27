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
# 在 GPU 節點的專案環境裡執行一個指令（指令中的 `python` 是 .venv 的 Python）。國網規定：所有
# 計算——包括測試與分析——都要透過 Slurm 在正式分區執行（8gpus 接受只要 1 張 GPU 的工作），
# 絕不在登入節點上執行。請在專案根目錄送出；寫在腳本名稱之前的 sbatch 參數會覆蓋上面的預設值：
#   例一：sbatch scripts/nano4/run.sbatch python -I scripts/predict_submission.py --workers 10
#   例二：sbatch -p ngs1gpu -t 2-00:00:00 -J score scripts/nano4/run.sbatch python -I scripts/score_folds.py --folds 0 --workers 10
#   例三：sbatch -p 8gpus --gres=gpu:8 -c 96 -t 2-00:00:00 -J score8 scripts/nano4/run.sbatch \
#     （例三續）bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 0 --workers 10
# log 檔是 logs/nano4/<工作名稱>-<jobid>.out。
set -euo pipefail  # 任何錯誤、未定義變數或管線失敗都立即中止
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"  # 專案根目錄＝送出 sbatch 時所在的目錄
source "$ROOT/scripts/nano4/common.sh"  # 進入專案、啟用 .venv 並印出工作標頭
if [ $# -eq 0 ]; then  # 沒有給要執行的指令
  echo "usage: sbatch [options] scripts/nano4/run.sbatch <command> [args...]" >&2  # 印出用法
  exit 2  # 以參數錯誤結束
fi  # 結束參數檢查
echo "+ $*"  # 在 log 中記下實際執行的指令
"$@"  # 執行指令，它的結束碼就是工作的結束碼
