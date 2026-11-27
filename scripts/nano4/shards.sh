#!/usr/bin/env bash
# 執行 N 份分片指令，每份使用一張配置到的 GPU，並等待全部結束：
#   例：bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 0 --workers 10
# 第 i 份只看得到配置中的第 i 張 GPU，指令後面會加上 "--shard i --nshards N"。
# 它的輸出寫到 logs/nano4/shard<i>-<jobid>.log。任何一份失敗時，以非零結束碼結束。
# 每份在 CPU 端的 torch/BLAS 執行緒池上限為 SHARD_THREADS（預設：有設定 OMP_NUM_THREADS 就用它，否則為 2）。
# 不設上限時，每份都會為每個核心各開一個 OpenMP 執行緒，N 份加上它們的特徵 worker 會讓節點
# 超額使用（PR #10：48 核上跑 4 個分片，每份 44 個空轉的執行緒，65 秒的評分被拖長到 8 分鐘）。模型運算
# 由 GPU 負責；主程序只搬移批次並加總 log p，所以 2 個執行緒就很夠了。
set -uo pipefail  # 未定義的變數視為錯誤，管線任一段失敗都算失敗
if [ $# -lt 2 ]; then  # 至少要有分片數與要執行的指令
  echo "usage: $0 <n_shards> <command> [args...]" >&2  # 印出用法
  exit 2  # 以參數錯誤結束
fi  # 結束參數檢查
N="$1"  # 分片數
shift  # 其餘參數就是要執行的指令
THREADS="${SHARD_THREADS:-${OMP_NUM_THREADS:-2}}"  # 每份的 CPU 執行緒上限（原因見檔頭說明）
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N - 1)))}"  # 配置到的 GPU 清單；沒有 CUDA_VISIBLE_DEVICES 時假設為 0..N-1
if [ "${#GPUS[@]}" -lt "$N" ]; then  # GPU 不夠每份一張
  echo "need $N GPUs but the job has ${#GPUS[@]} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2  # 說明需要與實有的 GPU 數
  exit 1  # 中止，不啟動任何分片
fi  # 結束 GPU 數檢查
mkdir -p logs/nano4  # 確保分片 log 的資料夾存在
pids=()  # 各分片的程序編號
for ((i = 0; i < N; i++)); do  # 逐一啟動 N 份
  log="logs/nano4/shard$i-${SLURM_JOB_ID:-local}.log"  # 這一份的 log 檔
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" "$@" --shard "$i" --nshards "$N" > "$log" 2>&1 &  # 在背景執行：只看見第 i 張 GPU、限制執行緒數、附加分片參數
  pids+=($!)  # 記下剛啟動的程序編號
  echo "shard $i on GPU ${GPUS[$i]} ($THREADS CPU threads): pid ${pids[$i]}, log $log"  # 回報分片、GPU、執行緒數與 log 位置
done  # 全部分片都已啟動
failed=0  # 有任何一份失敗時設為 1
for i in "${!pids[@]}"; do  # 依序等待每一份
  if wait "${pids[$i]}"; then  # 這一份正常結束
    echo "shard $i done"  # 回報完成
  else  # 這一份失敗
    echo "shard $i FAILED, see logs/nano4/shard$i-${SLURM_JOB_ID:-local}.log"  # 指出要看哪一個 log
    failed=1  # 記為失敗
  fi  # 結束判斷
done  # 全部分片都已結束
exit "$failed"  # 任何一份失敗就以 1 結束
