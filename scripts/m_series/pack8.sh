#!/usr/bin/env bash
# 在同一個 8 張 GPU 的配置內平行執行 $1 列出的指令（每行一個，至多 8 個），各占一張 GPU；任何一個失敗就以非 0 結束。
cd /home/u2507115/BK_AICUP_2026-1  # 切到專案根目錄
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"  # 本工作的 GPU 清單（未設定時用 0–7）
i=0; pids=(); names=()  # 槽位序號、背景行程編號與紀錄檔
while IFS= read -r cmd; do  # 逐行讀入指令
  [ -z "$cmd" ] && continue  # 略過空行
  log="logs/nano4/pack-${SLURM_JOB_ID:-local}-$i.log"  # 該槽位的紀錄檔
  echo "slot $i on GPU ${GPUS[$i]}: $cmd -> $log"  # 印出分配情形
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -c "$cmd" > "$log" 2>&1 &  # 在指定的 GPU 上背景執行
  pids+=($!); names+=("$log"); i=$((i + 1))  # 記下行程編號與紀錄檔
done < "$1"  # 指令清單來自第一個參數
fail=0  # 是否有指令失敗
for k in "${!pids[@]}"; do  # 逐一等待
  if wait "${pids[$k]}"; then echo "slot $k done: $(grep -h 'stopped at step' "${names[$k]}" | tail -1 | cut -c1-120)"; else echo "slot $k FAILED (${names[$k]})"; tail -5 "${names[$k]}"; fail=1; fi  # 成功時印出停止的步數，失敗時印出紀錄的最後 5 行
done  # 結束等待迴圈
exit $fail  # 有任何失敗就以 1 結束
