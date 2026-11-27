#!/usr/bin/env bash
# M90 修補：重跑逐手抽取的其中一個分片，其分段檔分配到本工作的各張 GPU；接著以一般方式再跑一遍，
# 補上任何副本留下的缺漏（副本落在壞掉的 GPU 上時），最後檢查是否完整。
#   sbatch --gres=gpu:8 -c 96 ... scripts/nano4/run.sbatch bash scripts/m_series/m90_fix_shard.sh <job> <shard> <三個資料夾合計應有的分段檔數>
set -uo pipefail  # 未定義變數視為錯誤；管線中任一指令失敗即整體失敗
JOB="$1"; SHARD="$2"; EXPECT="$3"  # 工作編號、分片編號與應有的分段檔數
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"  # 本工作分到的 GPU 清單
N=${#GPUS[@]}; pids=()  # GPU 數與背景行程編號
for ((r = 0; r < N; r++)); do  # 每張 GPU 跑一個副本
  log="logs/nano4/m90fix_r$r-${SLURM_JOB_ID:-local}.log"  # 該副本的紀錄檔
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I scripts/m_series/m90_ply_extract.py --job "$JOB" --njobs 3 --shard "$SHARD" --nshards 8 --workers 10 --sub "$r" --nsub "$N" > "$log" 2>&1 &  # 副本 r 只做序號 k % N == r 的分段檔，背景執行
  pids+=($!); echo "copy $r on GPU ${GPUS[$r]}: pid ${pids[$r]}, log $log"  # 記下行程編號並印出
done  # 結束啟動副本的迴圈
for r in "${!pids[@]}"; do if wait "${pids[$r]}"; then echo "copy $r done"; else echo "copy $r FAILED (GPU ${GPUS[$r]})"; fi; done  # 等每個副本結束並回報成敗
for g in "${GPUS[@]}"; do  # 補掃：已存在的分段檔會略過，所以每個副本都完成時這一步很快
  if CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I scripts/m_series/m90_ply_extract.py --job "$JOB" --njobs 3 --shard "$SHARD" --nshards 8 --workers 40 > "logs/nano4/m90fix_sweep-${SLURM_JOB_ID:-local}.log" 2>&1; then echo "sweep on GPU $g done"; break; else echo "sweep on GPU $g FAILED"; fi  # 依序換 GPU 嘗試，成功一次就停
done  # 結束補掃
have=$(ls results/m90_ply0/ply_*.npz results/m90_ply1/ply_*.npz results/m90_ply2/ply_*.npz | grep -vc '\.tmp\.npz$')  # 已完成的分段檔數（不含暫存檔）
tmp=$(ls results/m90_ply0/ results/m90_ply1/ results/m90_ply2/ | grep -c '\.tmp\.npz$')  # 未寫完的暫存檔數
echo "parts: $have of $EXPECT, unfinished temporary files: $tmp"  # 印出檢查結果
[ "$have" -eq "$EXPECT" ] && [ "$tmp" -eq 0 ]  # 檔數齊全且沒有暫存檔時才以 0 結束
