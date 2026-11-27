#!/usr/bin/env bash
# M90：全部對局的序列認人模型，一個種子的各交叉擬合網路，本工作的每張 GPU 一個。
#   sbatch --gres=gpu:5 -c 60 ... scripts/nano4/run.sbatch bash scripts/m_series/m90_train5.sh <seed> [net ...]   （預設網路：1 2 3 4 -1）
# 每張 GPU 先試用一次（2026-11-12：25a-hgpn177 有一張 GPU 對 CUDA 不可見，失敗的工作讓整條
# 相依鏈都被取消）。網路依序輪流分配到能用的 GPU，所以有一張 GPU 壞掉時會有兩個網路共用一張卡
# （較慢，但工作能跑完）。再次執行這個工作時，訓練會從檢查點續跑。
set -uo pipefail  # 未定義變數視為錯誤；管線中任一指令失敗即整體失敗
SEED="$1"; shift  # 第一個參數是種子，其餘是網路編號
if [ $# -gt 0 ]; then NETS=("$@"); else NETS=(1 2 3 4 -1); fi  # 沒指定網路時訓練全部五個
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"  # 本工作分到的 GPU 清單
GPUS=()  # 能用的 GPU
for g in "${ALL[@]}"; do  # 逐張測試
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi  # 能在該卡上建立張量才算能用
done  # 結束測試迴圈
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "${#NETS[@]}" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for ${#NETS[@]} networks" >&2; exit 1; fi  # 每張卡最多兩個網路，卡不夠就停止
i=0; pids=()  # 網路序號與背景行程編號
for v in "${NETS[@]}"; do  # 逐一網路
  g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/m90s${SEED}_net${v}-${SLURM_JOB_ID:-local}.log"  # 輪流分配 GPU；該網路的紀錄檔
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I scripts/m_series/m88_seqid.py --stage train --net "$v" --full --seed "$SEED" > "$log" 2>&1 &  # 背景訓練一個網路（M90 配方）
  pids+=($!); echo "seed $SEED net $v on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))  # 記下行程編號並印出
done  # 結束啟動迴圈
failed=0  # 是否有網路失敗
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done  # 等每個訓練結束並回報成敗
exit "$failed"  # 有任何失敗就以 1 結束
