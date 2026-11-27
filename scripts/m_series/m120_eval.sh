#!/usr/bin/env bash
# M120：把對稱擴增的身分網路併入配對器快取並評估（設計文件 M120）。
#   m120_eval.sh one     只用種子 0      -> var_S4；標籤 S4、SP4（預覽，只報告）
#   m120_eval.sh three   種子 0、1、2    -> var_S4e；標籤 S4e、SP4e、SP2y4、SP2e4，接著執行事先登記的檢定（m121，--build）
#   m120_eval.sh auto    十五個網路都在時跑 three，種子 0 完整時跑 one；每個階段只執行一次（以鎖定資料夾控制）
set -uo pipefail  # 未定義變數即報錯；管線中任一段失敗即視為失敗
have() { for t in "$@"; do [ -f "results/m120/out_$t.npz" ] || return 1; done; }  # 檢查指定網路的輸出是否都已存在
main() {  # 主流程包成函式，整支讀完才執行
  local stage="${1:-auto}"  # 評估階段，預設 auto
  if [ "$stage" = auto ]; then  # 自動判斷該跑哪個階段
    if have 1 2 3 4 -1 1_s1 2_s1 3_s1 4_s1 -1_s1 1_s2 2_s2 3_s2 4_s2 -1_s2; then stage=three  # 三個種子的十五個網路都在
    elif have 1 2 3 4 -1; then stage=one  # 只有種子 0 的五個網路完整
    else echo "seed 0 is not complete: nothing to evaluate yet"; return 0; fi  # 還不能評估，正常結束
    mkdir "results/m120/.eval_$stage.lock" 2>/dev/null || { echo "evaluation '$stage' was already started by another job"; return 0; }  # 以建立資料夾當鎖，避免多個工作重複評估
  fi  # 結束自動判斷
  echo "== evaluation stage: $stage ($(date '+%F %H:%M'))"  # 記錄階段與時間
  case "$stage" in  # 依階段執行
    one)  # 只有種子 0：預覽
      python -I scripts/m_series/m120_seqid_sym.py --merge 0 || return 1  # 合併種子 0 -> var_S4
      python -I scripts/m_series/m84_variant.py --stage eval --tag S4  # 評估只用 S4 的識別器
      python -I scripts/m_series/m84_variant.py --stage eval --tag SP4 ;;  # 評估 SP4 識別器
    three)  # 三個種子：正式評估
      python -I scripts/m_series/m120_seqid_sym.py --merge 0,1,2 || return 1  # 合併三個種子 -> var_S4e
      for t in S4e SP4e SP2y4 SP2e4; do python -I scripts/m_series/m84_variant.py --stage eval --tag "$t" || return 1; done  # 逐一評估四個識別器
      python -I scripts/m_series/m121_sym_anchor_test.py --id SP2y4 --also SP4e SP2e4 --build ;;  # 事先登記的檢定；兩項都通過才建候選檔
    *) echo "usage: $0 one|three|auto" >&2; return 2 ;;  # 不認得的階段：印出用法
  esac  # 結束階段分支
}  # 結束 main
main "$@"; exit $?  # 執行 main 後立即結束，執行中修改本檔也不影響
