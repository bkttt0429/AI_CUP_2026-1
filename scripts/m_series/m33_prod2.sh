#!/usr/bin/env bash
# M33 正式產出，修正後的 mock 標準（決策全同、分數相同、max |dp| <= ct7 先例的 0.0040）：診斷既有 mock 執行裡
# |dp| 最大的幾題，接著以 24 個 pc 成員產生 submission_rank_ct9.csv，最後比較 ct9 與 ct7。
set -uo pipefail  # 未定義變數視為錯誤；管線中任一指令失敗即整體失敗
cd /home/u2507115/BK_AICUP_2026-1  # 切到專案根目錄
OUT=results/mock_fold1_100_pc  # 既有 mock 一致性檢查（折 1 的 100 題）的輸出資料夾
python -I -c "
import json, sys, numpy as np, pandas as pd
r = json.load(open('$OUT/ct_check.json'))
a = pd.read_csv('$OUT/submission_rank_ct_proba.csv').set_index('question_id'); b = pd.read_csv('$OUT/offline_ct_proba.csv').set_index('question_id').loc[a.index]
cols = [c for c in a.columns if c.startswith('p_')]
d = (a[cols] - b[cols]).abs().max(axis=1).sort_values(ascending=False)
for q in d.index[:5]: print(f'{q}: max |dp| {d[q]:.4f}, max probability {a.loc[q, cols].max():.3f}')
print(f'questions above 1e-3: {int((d > 1e-3).sum())}/100')
ok = r['decisions_differ'] == 0 and abs(r['score_a'] - r['score_b']) < 1e-12 and r['max_abs_dp'] <= 0.0040
print('AMENDED MOCK CHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)" || exit 1  # 診斷 |dp| 最大的題目並判定修正後的 mock 標準；沒通過就停止
MODELS=""; for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do MODELS="$MODELS results/models/ct_cv_m33/val${v}_$arm-pc-f0-x40-oh_s$s.pt"; done; done; done  # 24 個 pc 成員的權重路徑
echo "checkpoints present: $(ls $MODELS 2>/dev/null | wc -l)/24"  # 印出找到的權重檔數（應為 24；只顯示，不中止）
python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $MODELS --fallback-submission submission_rank_ct3.csv --out submission_rank_ct9.csv || { echo "ct9 predict FAILED"; exit 1; }  # 以 24 個成員預測測試集，產生 ct9
md5sum submission_rank_ct9.csv submission_rank_ct7.csv  # 記錄兩個提交檔的 md5
python -I -c "
import pandas as pd
a, b = pd.read_csv('submission_rank_ct9.csv'), pd.read_csv('submission_rank_ct7.csv')
assert list(a.columns) == list(b.columns) and len(a) == len(b) == 800 and (a.iloc[:, 0] == b.iloc[:, 0]).all()
pub = a.iloc[:, 0].str.startswith('pub')
d = a.iloc[:, 1] != b.iloc[:, 1]
print(f'ct9 vs ct7: {int(d.sum())}/800 decisions differ (public {int(d[pub].sum())}, private {int(d[~pub].sum())})')
print('ct9 predicted counts', a.iloc[:, 1].value_counts().sort_index().to_dict())"  # 比較 ct9 與 ct7 的逐題決策差異，並列出 ct9 各等級的預測數
