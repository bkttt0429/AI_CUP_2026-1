#!/usr/bin/env bash
# M129 的準備：其他投票來源之識別器（SP、S2、S3c、S3）的 m15 識別器檔；SP2e 的已存在（M126）。
set -uo pipefail  # 未定義變數即報錯；管線中任一段失敗即視為失敗
main() {  # 主流程包成函式，整支讀完才執行
  python -I scripts/m_series/m127_conf_family.py --stage fuse --tags SP S2 S3c S3 || return 1  # 先算出四個識別器融合分數的讀出（m127_<tag>.npz）
  python -I scripts/m_series/m127_conf_family.py --stage write --tags SP S2 S3c S3 --write m15  # 再以 m15 統計量寫出 ~m15 識別器檔
}  # 結束 main
main "$@"; exit $?  # 執行 main 後立即結束，執行中修改本檔也不影響
