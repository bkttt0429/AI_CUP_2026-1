#!/usr/bin/env bash
# 微調過的 L1b 主幹（各約 103 MB）分段存放，因為 GitHub 拒收超過 100 MB 的檔案。
# 這支腳本把分段接回 results/l1b/heads3_{A,B}.trunk.pt，並以 heads3.sha256 核對每一個 L1b 檔案。
#   用法：bash scripts/join_trunk_parts.sh
main() {  # 主流程包成函式，整段讀入後才執行
  set -euo pipefail  # 任何錯誤或未定義變數就中止
  cd "$(dirname "$0")/../results/l1b"  # 移到等級頭權重所在的資料夾
  for h in A B; do  # 逐一處理 A、B 兩個主幹
    cat heads3_$h.trunk.pt.part? > heads3_$h.trunk.pt  # 依 part0、1、2 的順序串接回原檔
  done  # 結束迴圈
  sha256sum -c heads3.sha256  # 核對等級頭與主幹共 4 個檔的 sha256
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以其結束碼離開
