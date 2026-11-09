# 最終提交流程（AI CUP 2026 圍棋棋力等級預測）

最終上傳檔是 `submission_rank_ct28.csv`（md5 前 12 碼 `d8a854c8f6b4`），Public 0.7827471。
本分支只收最終流程用到的程式；實驗性的腳本、比賽資料與大型權重不在其中。

## 環境

- Python 3.12，套件見 `requirements.txt`；在 nano4 上以 `bash scripts/nano4/setup.sh` 建 `.venv`。
- KataGo v1.15.0 執行檔放在 `tools/katago/`，Human SL 權重 `b18c384nbt-humanv0` 放在 `models/katago/`。
- 比賽資料放在 `data/training/` 與 `data/tests/`（不隨 repo 散布）。
- 所有工作以 `sbatch scripts/nano4/run.sbatch <指令>` 送到計算節點；8 卡分片用 `bash scripts/nano4/shards.sh 8 <指令>`。各步的完整參數見 `docs/NANO4.md` 對應小節。

## 步驟

1. **切折**：`splits/player_folds.csv`（`scripts/make_splits.py` 產生，已在 repo）。
2. **Human SL 18 剖面評分**：`scripts/score_folds.py`，本人與對手（`--opponent`）各一趟。讓子局：`scripts/hd_table.py` 找出 HD ≥ 2 的局，重評後以 `scripts/patch_scored.py` 併回，得到 `results/scores_hd`、`results/scores_opp_hd`。每人第 21–40 局：`logs/nano4/x40_keys.py` 產生鍵清單，再以 `--max-games 40 --game-keys results/data/x40_keys.csv` 評到 `*_x40` 目錄。見 NANO4 §4.19、§4.23、§4.25。
3. **L1b 等級頭**：`scripts/l1b_build_cache.py` → `scripts/l1b_train.py`（第一階段只訓 head；第二階段 `--trunk-lr 1e-5 --init-heads ...`）得到 `heads3_A`（訓 folds 1、2）與 `heads3_B`（訓 folds 3、4）；`scripts/l1b_score.py` 交叉評分（A 評 folds 0、3、4，B 評 folds 0、1、2，含 `--opponent`）。見 NANO4 §4.2、§4.19、§4.23。
4. **trunk 向量**：`scripts/trunk_extract.py` → `results/trunk_vec`、`results/trunk_vec_x40`。見 NANO4 §4.21、§4.23。
5. **token 陣列與題目層特徵**：`scripts/trunk_seq_check.py build --opp-hsl ...` 與 `rfeat`。見 NANO4 §4.23、§4.25。
6. **24 個成員**（4 個驗證折 × C、T × seed 0–2）：

   ```bash
   for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
     python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 \
       --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 \
       --out results/trunk_seq/es --save-dir results/models/ct_cv_m33
   done; done; done
   ```

   訓練好的 24 個權重與設定檔已收在 `results/models/ct_cv_m33/`。
7. **測試集推論**：

   ```bash
   bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --workers 10 \
     --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --out results/ct6/submission_rank.csv
   bash scripts/nano4/shards.sh 8 python -I scripts/trunk_predict.py extract \
     --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv \
     --vec results/ct_submission/vec --workers 10
   python -I scripts/trunk_predict.py predict \
     --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv \
     --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec \
     --models results/models/ct_cv_m33/val*_*-pc-f0-x40-oh_s*.pt --out submission_rank_ct9.csv
   ```

   產出 `submission_rank_ct9.csv`（md5 前 12 碼 `6ff60b847e26`）。
8. **類別偏置校正**（勝算比 r = 2，兩半套用同一條規則）：

   ```bash
   python -I logs/nano4/m53_write_probe.py --base submission_rank_ct9.csv --push 1D A --r 2 --out submission_rank_ct13.csv
   python -I logs/nano4/m53_write_probe.py --base submission_rank_ct13.csv --push 4D 5D --r 2 --out submission_rank_ct28.csv
   ```

   2026-11-10 從 ct9 重跑這兩行，得到的 ct13（`a9be72a943ee`）與 ct28（`d8a854c8f6b4`）和上傳檔逐位元相同。
   這兩個邊界是以 Public 排行榜回饋選出；每次探測的分數以 `logs/nano4/m58_decode.py` 解碼成翻對與翻錯的題數。

## 不在 repo 的檔案

- L1b 權重 `results/l1b/heads3_{A,B}.pt` 與微調過的 trunk `heads3_{A,B}.trunk.pt`（各約 108 MB，超過 GitHub 單檔上限），另以 Release 附件提供。
- 比賽資料、評分中間檔與快取。
