#!/usr/bin/env bash
# M33：玩家對比式輔助損失（--player-contrast 0.1，臂標 -pc）對 ct7 的 M22 成員。第 0 階段冒煙測試；第 1 階段 24 個成員
# （3 個 8 卡包）；第 2 階段成員協定檢查（FAIL 就停止）；第 3 階段單臂比較＋6 對 6 集成。
set -uo pipefail  # 未定義變數視為錯誤；管線中任一指令失敗即整體失敗
cd /home/u2507115/BK_AICUP_2026-1  # 切到專案根目錄
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'  # 排除已知有問題的節點
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"  # 指令清單、冒煙輸出與 pack8.sh 所在的資料夾；訓練程式
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"  # ct7 的配方再加玩家對比損失
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }  # 查詢工作的狀態
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }  # 等逗號分隔的每個工作都進入結束狀態
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }  # 全部工作都是 COMPLETED 才回傳成功
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }  # 送出工作並回傳編號：$1 工作名、$2 sbatch 選項、其餘為指令；送不進去時每分鐘重試，最多 300 次
echo "== $(date '+%F %T') M33 stage 0: smoke"  # 第 0 階段：冒煙測試
rm -rf $A/m33smoke; mkdir -p $A/m33smoke  # 清空冒煙測試的輸出資料夾
S=$(submit m33smk "-c 12 -t 01:00:00" $TR train --arm C --val-fold 1 --seed 0 $REC --limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m33smoke)  # 小規模訓練一個 C 臂成員
echo "smoke job $S"; wait_jobs "$S"; echo "smoke $(state $S)"; grep -E "InfoNCE|stopped at step|Error|error" logs/nano4/m33smk-$S.out | tail -4 | cut -c1-200  # 等待後摘錄對比損失、停止步數與錯誤訊息
ok_all "$S" || { echo "smoke failed: stop"; exit 1; }  # 冒煙測試失敗就停止
echo "== $(date '+%F %T') M33 stage 1: 24 members"  # 第 1 階段：24 個成員
rm -f $A/m33_cmds_*  # 清掉舊的指令清單
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do  # 4 個驗證折 × C、T 兩臂 × 種子 0–2
  echo "$TR train --arm $arm --val-fold $v --seed $s $REC --save-dir results/models/ct_cv_m33 --out results/trunk_seq/es" >> $A/m33_cmds_all.txt  # 寫入一個成員的訓練指令
done; done; done  # 結束三層迴圈
split -l 8 -d $A/m33_cmds_all.txt $A/m33_cmds_  # 每 8 個指令切成一包
P=""; for f in $A/m33_cmds_0*; do P="$P,$(submit m33p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done  # 每包送一個 8 卡工作，由 pack8.sh 平行執行；收集工作編號
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done  # 等全部包結束，列出狀態與耗時
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"  # 有包失敗只提示，由協定檢查指出缺了什麼
echo "== $(date '+%F %T') M33 stage 2: member protocol check"  # 第 2 階段：成員協定檢查
C=$(submit m33chk "-c 12 -t 00:20:00" python -I scripts/m_series/m33_protocol_check.py); wait_jobs "$C"  # 檢查 24 個成員的設定是否與 M22 的雙胞胎一致
grep -v "^==\|^+" logs/nano4/m33chk-$C.out | tail -26  # 摘錄檢查結果
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }  # 檢查失敗就不做比較
echo "== $(date '+%F %T') M33 stage 3: compare"  # 第 3 階段：比較
K=$(submit m33cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-f0-x40-oh T-f0-x40-oh C-pc-f0-x40-oh T-pc-f0-x40-oh --seeds 0 1 2 --pairs C-pc-f0-x40-oh:C-f0-x40-oh T-pc-f0-x40-oh:T-f0-x40-oh --tag m33_ --out results/trunk_seq/es)  # 單臂比較：pc 成員對 ct7 的成員
E=$(submit m33ens "-c 12 -t 01:00:00" python -I scripts/m_series/ens_compare.py --old C-f0-x40-oh T-f0-x40-oh --new C-pc-f0-x40-oh T-pc-f0-x40-oh --seeds 0 1 2 --out results/trunk_seq/es/ens_m33.json)  # 集成比較：6 對 6
wait_jobs "$K,$E"  # 等兩個比較都結束
echo "== compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m33cmp-$K.out | tail -8  # 摘錄單臂比較的結果
echo "== ensemble 6v6 $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m33ens-$E.out | tail -8  # 摘錄集成比較的結果
echo "== $(date '+%F %T') M33 done"  # 全部完成
