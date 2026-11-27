#!/usr/bin/env bash
# M90：把訓練好的網路合併進配對器快取，並評估以它們建立的識別器。
#   m90_eval.sh one    只用種子 0           -> S2、SP2
#   m90_eval.sh three  種子 0,1,2 取平均    -> S2e、SP2e（若有一個種子的工作失敗，退回已完成的兩個種子）
set -uo pipefail  # 未定義變數視為錯誤；管線中任一指令失敗即整體失敗
case "$1" in  # 依第一個參數選擇模式
  one)  # 單一種子
    python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds 0 || exit 1  # 合併種子 0 的網路，失敗就停止
    python -I scripts/m_series/m84_variant.py --stage eval --tag S2  # 評估 S2（逐手模型取代配對器）
    python -I scripts/m_series/m84_variant.py --stage eval --tag SP2 ;;  # 評估 SP2（再加分階段比對）
  three)  # 三個種子平均
    ok=0  # 是否已合併成功
    for s in 0,1,2 0,1 0,2 1,2; do  # 先試三個種子，失敗再退回兩個種子的組合
      if python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds "$s"; then ok=1; echo "merged seeds $s"; break; fi  # 第一個合併成功的組合就採用
    done  # 結束種子組合的迴圈
    [ "$ok" -eq 1 ] || exit 1  # 全部失敗就停止
    python -I scripts/m_series/m84_variant.py --stage eval --tag S2e  # 評估 S2e
    python -I scripts/m_series/m84_variant.py --stage eval --tag SP2e ;;  # 評估 SP2e
  *) echo "usage: $0 one|three" >&2; exit 2 ;;  # 其他參數：印出用法後結束
esac  # 結束模式選擇
