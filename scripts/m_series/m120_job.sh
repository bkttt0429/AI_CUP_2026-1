#!/usr/bin/env bash
# M120 的工作：訓練指定的網路（每張 GPU 一個：network[:seed] ...），再評估已完成的部分（m120_eval.sh auto）。
set -uo pipefail  # 未定義變數即報錯；管線中任一段失敗即視為失敗
main() {  # 主流程包成函式，整支讀完才執行
  python -I scripts/m_series/m120_seqid_sym.py --jobs "$@" || echo "training exited non-zero; the evaluation still looks at what finished"  # 平行訓練各網路；失敗也繼續評估
  bash scripts/m_series/m120_eval.sh auto  # 依已完成的網路自動選擇評估階段
}  # 結束 main
main "$@"; exit $?  # 執行 main 後立即結束，執行中修改本檔也不影響
