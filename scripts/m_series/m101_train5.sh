#!/usr/bin/env bash
# M101：M90 的序列身分模型改用較短的訓練排程，加入或不加入外部玩家（Fox Go dataset 的帳號；取自玩家辨識工作階段的
# 逐手分段檔，唯讀），這個工作的每張 GPU 訓練一個交叉擬合網路。啟動腳本的邏輯同 m90_train5.sh。
#   用法：sbatch --gres=gpu:5 -c 60 ... scripts/nano4/run.sbatch bash scripts/m_series/m101_train5.sh <seed> <ext-share|none> <steps> <out folder:VAR> [net ...]
set -uo pipefail  # 未定義變數即報錯；管線中任一段失敗即視為失敗
SEED="$1"; SHARE="$2"; STEPS="$3"; OUT="$4"; shift 4  # 種子、外部帳號占比、訓練步數、輸出位置；其餘參數為網路編號
EXT_PLY="${EXT_PLY:-/work/u2507115/ext_fox/pilot/ply}"; EXT_SEL="${EXT_SEL:-/work/u2507115/ext_fox/pilot/select.csv}"  # 外部帳號的逐手資料與選取清單（可由環境變數覆寫）
if [ "$SHARE" = "none" ]; then EXT=(); else EXT=(--ext-ply "$EXT_PLY" --ext-select "$EXT_SEL" --ext-share "$SHARE"); fi  # none 為對照組，不加外部帳號
if [ $# -gt 0 ]; then NETS=("$@"); else NETS=(1 2 3 4 -1); fi  # 未指定時訓練全部五個網路（折 1-4 與看過全部玩家的 -1）
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"  # 取得分配給這個工作的 GPU 清單
GPUS=()  # 實際可用的 GPU
for g in "${ALL[@]}"; do  # 逐張檢查 GPU
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi  # 能在卡上配置張量才算可用，壞卡跳過
done  # 結束 GPU 檢查
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "${#NETS[@]}" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for ${#NETS[@]} networks" >&2; exit 1; fi  # 每張 GPU 最多兩個網路，不夠就中止
i=0; pids=(); name="${OUT%%:*}"  # 計數器、行程清單與輸出資料夾名稱（冒號前段）
for v in "${NETS[@]}"; do  # 每個網路啟動一個訓練行程
  g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/${name}s${SEED}_net${v}-${SLURM_JOB_ID:-local}.log"  # 輪流分配 GPU 並設定紀錄檔
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I scripts/m_series/m88_seqid.py --stage train --net "$v" --full --seed "$SEED" --steps "$STEPS" --out "$OUT" "${EXT[@]}" > "$log" 2>&1 &  # 在背景訓練此網路
  pids+=($!); echo "seed $SEED net $v on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))  # 記下行程編號並回報
done  # 全部網路都已啟動
failed=0  # 是否有行程失敗
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done  # 等待每個訓練行程並記錄失敗
exit "$failed"  # 任一網路失敗時以 1 結束
