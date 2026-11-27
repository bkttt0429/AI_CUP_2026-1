#!/usr/bin/env bash
# Section 4M M161q 第二步（2026-11-15；設計文件 M161q，在看到這一組的任何錨點／合併結果之前登錄）：
#   第二組正式網路八個方向平均的查詢 -> var_<var>（原型）與 var_<var>c（學到的中心）；
#   識別器組成 SQ1..SQ8（由簡到繁），每個都加上在留出本人局上擬合的活躍度先驗；
#   不看標籤的選擇（m164_select.py：測試端的認出比例，容許差 0.003，取最簡單的）；
#   認出比例的報告；選中的識別器對 ct106 規則的登錄合併檢定
#   （只加新來源，長得像的人讀數不變；兩組種子都要達 M37；通過時建出候選檔 <name>）。
#   用法：bash scripts/m_series/m161q_step2.sh 30,31,32 S7t ct112
main() {  # 主函式
  local seeds="$1" var="$2" name="$3" old="P,S2y,S4ft,S4Lt" byg="" c="results/m84_cache" tags=() specs=() t best also=()  # 參數：種子、變體名、候選檔名；舊來源、依局數抽樣的來源、快取位置與工作變數
  python -I scripts/m_series/m161q_tta.py --seeds "$seeds" --name "$var" || return 1  # 八向平均的查詢與學到的中心 -> var_<var> 與 var_<var>c
  [ -f "$c/var_S6at.npz" ] && byg="S6at"  # M161p 依局數抽樣網路的變體檔存在時納入
  [ -f "$c/var_S6ft.npz" ] && byg="${byg:+$byg,}S6ft"  # f 配方的變體檔存在時也納入
  specs+=("SQ1:$var" "SQ2:$var,${var}c")  # SQ1＝新網路；SQ2＝再加中心來源
  [ -n "$byg" ] && specs+=("SQ3:$byg,$var" "SQ4:$byg,$var,${var}c")  # SQ3／SQ4：新網路再加 M161p 的依局數抽樣網路（SQ4 含中心）
  specs+=("SQ5:$old,$var" "SQ6:$old,$var,${var}c")  # SQ5／SQ6：舊來源加新網路（SQ6 含中心）
  [ -n "$byg" ] && specs+=("SQ7:$old,$byg,$var" "SQ8:$old,$byg,$var,${var}c")  # SQ7／SQ8：舊來源、M161p 網路加新網路（SQ8 再含中心）
  for t in "${specs[@]}"; do tags+=("${t%%:*}"); done  # 取出各組成的名稱
  echo "== identifiers: ${specs[*]}"  # 印出要建的識別器
  python -I scripts/m_series/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1  # 建出各識別器（含擬合的活躍度先驗）
  best=$(python -I scripts/m_series/m164_select.py "${tags[@]}") || return 1  # 不看標籤選出一個識別器
  echo "== chosen identifier (label-free rule): $best"  # 印出選中的識別器
  for t in "${tags[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done  # 其餘組成只報告
  python -I scripts/m_series/m120_diag_mass.py SPAL~m15 SPALm~m15 $(printf -- '%s~m15 ' "${tags[@]}") || echo "(the recognition-share report failed; the test goes on)"  # 認出比例的報告，失敗也繼續
  python -I scripts/m_series/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"  # 登錄的合併檢定，通過時建出候選檔
}  # 結束主函式
main "$@"; exit $?  # 執行主函式後立即以其結束碼結束（不再往下讀腳本）
