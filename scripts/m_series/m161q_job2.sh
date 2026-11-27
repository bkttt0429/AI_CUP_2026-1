#!/usr/bin/env bash
# Section 4M M161q（2026-11-15）：正式組的第二個 8 卡工作——前七張 GPU 跑七個訓練程序（都是登錄的配方），
# 第八張跑一個先導試驗（全體玩家網路，每個 episode 64 位玩家 x 32 局，其餘配方相同，輸出在 results/m161q/e64）。
# 工作的結束狀態以正式訓練程序為準。
main() {  # 主函式
  local gpus first rc=0  # 區域變數：GPU 陣列、前七張 GPU、結束碼
  IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-}"  # 把分配到的 GPU 拆成陣列
  [ "${#gpus[@]}" -ge 8 ] || { echo "8 GPUs expected, got '${CUDA_VISIBLE_DEVICES:-}'"; return 2; }  # 必須有 8 張 GPU
  first=$(IFS=,; echo "${gpus[*]:0:7}")  # 前七張 GPU 組成逗號分隔清單
  CUDA_VISIBLE_DEVICES="$first" python -I scripts/m_series/m161q_prod.py --jobs 4:31 all:31 1:32 2:32 3:32 4:32 all:32 --steps 40000 &  # 背景執行七個正式訓練（種子 31 的網路 4 與全體、種子 32 的五個網路）
  local p1=$!  # 記下正式訓練的程序編號
  CUDA_VISIBLE_DEVICES="${gpus[7]}" python -I scripts/m_series/m161q_prod.py --jobs all:30 --ep 64 32 --out e64 --steps 40000 &  # 第八張 GPU 背景執行 64 x 32 的先導試驗
  local p2=$!  # 記下先導試驗的程序編號
  wait "$p1" || rc=1  # 等正式訓練結束，失敗時結束碼為 1
  wait "$p2" || echo "(the 64 x 32 pilot failed; the production trainers are what counts)"  # 先導試驗失敗只提示，不影響結束碼
  return $rc  # 回傳正式訓練的狀態
}  # 結束主函式
main "$@"; exit $?  # 執行主函式後立即以其結束碼結束（不再往下讀腳本）
