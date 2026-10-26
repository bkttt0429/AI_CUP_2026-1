# 國網 nano4 使用說明

把專案搬到 NCHC nano4（每個節點 8 張 H200），並在上面跑 GPU 工作。nano4 只接受密碼加 2FA 登入，所以下面每一個 `scp` 和 `ssh` 都要你自己輸入。腳本都放在 `scripts/nano4/`。

nano4 用同一個主機名稱 `nano4.nchc.org.tw` 提供兩種服務（見[官方手冊：登入與傳輸節點](https://man.twcc.ai/@nano4-manual/BydP-_lvZg)、[資料傳輸節點操作流程](https://man.twcc.ai/@nano4-manual/Hkayr3zvWl)）：

| 連接埠 | 節點 | 用途 |
| --- | --- | --- |
| 22 | 登入節點 | `ssh` 登入、`sbatch` 送工作；沒有 SFTP，`scp` 到這裡會在 2FA 之後出現 `Connection closed` |
| 2222 | 資料傳輸節點 | 只能傳檔（`scp -P 2222`、`sftp -P 2222`、WinSCP／FileZilla 選 SFTP、埠號 2222），不能登入 |

## 1. 上傳與解開（只做一次）

在本機把 `D:\AICUP_transfer` 裡的四個檔案傳到國網的家目錄：

```bash
cd D:/AICUP_transfer
```

```bash
scp -P 2222 AICUP_2026-1.tar.xz AICUP_2026-1.tar.xz.sha256 AICUP_2026-1.files.sha256 AICUP_2026-1_nano4.tar.gz nano4:~/
```

傳到一半斷線時，`scp` 會從頭重傳；改用 `sftp -P 2222 nano4`，連上後輸入 `reput AICUP_2026-1.tar.xz` 可以從中斷處接著傳。

登入 nano4 後，解開並核對：

```bash
cd ~ && sha256sum -c AICUP_2026-1.tar.xz.sha256 && tar -xJf AICUP_2026-1.tar.xz && sha256sum -c --quiet AICUP_2026-1.files.sha256 && tar -xzf AICUP_2026-1_nano4.tar.gz && echo ALL OK
```

- 主壓縮檔是 10/19 的完整專案，含 `.git`；`_nano4` 補充包是之後加上的 nano4 腳本和 git 紀錄，必須在主壓縮檔之後解開。
- 最後印出 `ALL OK`，代表主壓縮檔的 7,416 個檔案都核對無誤，補充包也完整解開。
- 需要約 15 GB 空間；之後 Python 環境還要再約 6 GB。

## 2. 建 Python 環境（登入節點，只做一次）

```bash
cd ~/AICUP_2026-1 && bash scripts/nano4/setup.sh
```

- 會下載約 4 GB：uv、Python 3.12、torch 2.14.1，以及 `requirements.txt` 裡和 lab01 相同版本的套件。
- 全部裝在專案資料夾裡（`.venv`、`tools/uv*`），不會動到家目錄的其他地方。
- 中途斷線就重跑，已完成的步驟會略過。
- torch 預設用 CUDA 12.6 版本，驅動 R525 以上都能跑；lab01 用的是 CUDA 13.0 版本。若國網的驅動是 R580 以上，也可以改裝和 lab01 一樣的版本：`TORCH_CUDA=cu130 bash scripts/nano4/setup.sh`。

## 3. 冒煙測試（第一個 GPU 工作，約 20 分鐘）

```bash
cd ~/AICUP_2026-1 && sbatch scripts/nano4/smoke.sbatch
```

用 `squeue -u $USER` 看排隊狀況。log 在 `logs/nano4/aicup-smoke-<jobid>.out`，最後一行是 `SMOKE PASSED`，失敗時會寫出是哪一步。它依序做：

1. 印出 GPU、驅動版本、CPU 核數和記憶體
2. 在 GPU 上實際跑一次 torch
3. 跑完整的測試（KataGo 引擎的測試會自動略過，見第 6 節）
4. 測 H200 的吞吐量，結果寫到 `results/bench_nano4.json`
5. 重算 fold 0 的 200 局，和 lab01 算好的分數逐局比對；每手的 log p 差異要在 0.005 以內

跑完請把 log 的硬體資訊、吞吐量和比對結果貼回來，用來更新 METHOD_NOTES §3.1 的估算。

## 4. 平常跑工作

國網規定：登入節點只用來改檔、管理檔案、查詢和送工作；測試（pytest）、分析腳本、訓練、推論一律用 Slurm 送到計算節點，而且用正式分區（`8gpus`，可以只要 1 張卡），不用 `dev`。

`scripts/nano4/run.sbatch` 會在專案環境裡執行你給的指令，指令裡的 `python` 就是 `.venv` 的 Python。預設是 8gpus 分區、1 張 GPU、12 核、4 小時；在腳本名稱前面加 sbatch 參數就能覆蓋預設值。log 在 `logs/nano4/<工作名稱>-<jobid>.out`。

跑測試也一樣送工作：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -J tests scripts/nano4/run.sbatch python -m pytest -q
```

```bash
sbatch scripts/nano4/run.sbatch python -I scripts/predict_submission.py --workers 10
```

提交管線（2026-10-25 加固，待辦第 1 項）：單卡照上面跑，它會順便寫 `submission_rank_shard0of1.npz`。8 張卡：`shards.sh` 給每張卡加 `--shard i --nshards 8`，各寫一個分片檔，之後用 `--merge` 合併預測（不需要 GPU）：

```bash
sbatch -p 8gpus --gres=gpu:8 -c 96 -t 04:00:00 -J sub8 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --workers 10
sbatch -t 01:00:00 -J submerge scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8
```

正式測試集預設期望 800 列＝400 `pub_` + 400 `priv_`（`--expect-rows`／`--expect-public`／`--expect-private`，0 表示不檢查）；跑 mock test 要給它自己的數字：

```bash
sbatch -t 02:00:00 -J submock scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40/submission_rank.csv
```

exit code：0 正常；1 提交檔驗證失敗（檔案仍會寫出，問題列在報告的 `problems`）；2 輸入格式錯（`num_games` 與非空 `sgf_*` 欄不一致）、模型特徵不符、分片檔缺漏或不一致；3 有題目沒有可用棋局、被填 `1D`（加 `--allow-fallback` 才算正常）；4 輸入是用 `--lenient-inputs` 讀進來、帶有警告（加 `--allow-input-warnings` 才算正常）。報告 JSON 多了 `mode`、`shards`（每個分片的題目雜湊、局數、局面數、秒數）、`expectations`、`input_warnings`、`exit_code`。

截止日的退路：官方檔若真有題目 `num_games` 與非空 `sgf_*` 欄不一致，預設 exit 2、什麼都不產出；加 `--lenient-inputs` 會把那些題目改用實際非空的欄讀進來、照常預測，每題一條警告寫進報告的 `input_warnings`，exit 4。**分片與 merge 都要加同一個旗標**（題目雜湊算的是讀進來的局數），分片程序寫完檔就 exit 0，由 merge 決定最後的 exit code。

```bash
sbatch -p ngs1gpu -t 2-00:00:00 -J score scripts/nano4/run.sbatch python -I scripts/score_folds.py --folds 0 --workers 10
```

一整個節點（8 張卡）分片跑。`shards.sh` 讓每張卡各跑一份，自動加上 `--shard i --nshards 8`，各份的輸出在 `logs/nano4/shard<i>-<jobid>.log`：

```bash
sbatch -p 8gpus --gres=gpu:8 -c 96 -t 2-00:00:00 -J score8 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 0 --workers 10
```

`-c 96` 是暫定值，等冒煙測試量到每個節點的核數再調整。

| 分區 | 單次時限 | 用途 |
| --- | --- | --- |
| dev | 4 小時 | 測試分區；正式工作不要用（國網規定） |
| ngs1gpu、ngs2gpu、ngs4gpu、ngs8gpu | 14 天 | 1–8 張卡的長工作；計畫能不能用還沒確認 |
| 8gpus | 2 天 | 預設；1–8 張 H200，可以只要 1 張 |
| 16gpus 以上 | 1–2 天 | 多節點，目前的腳本用不到 |

常用指令：`squeue -u $USER` 看排隊和執行中的工作，`scancel <jobid>` 取消，`sacct -j <jobid>` 看結束後的狀態。

L3 聚合器梯子（`docs/superpowers/plans/2026-10-19-l3-aggregator-ladder-ctm.md` Task 9）的工作都在 `~/BK_AICUP_2026-1` 下送出，並一律加 `--exclude=25a-hgpn[001,003,004,007,062]` 避開壞 GPU 節點。輸出全部在 `results/ladder/`。

建每局向量（folds 0–4）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J ladvec scripts/nano4/run.sbatch python -I scripts/ladder_build_vectors.py --folds 0 1 2 3 4
```

跑 CV 之前，先做一次 CUDA 冒煙：`r3_settf` 和 `r5_ctm` 各用 GPU 跑一個 seed，確認 CUDA 路徑能跑完（損失或機率出現 NaN／inf 時腳本會停下、不寫檔）。輸出在 `results/ladder/cv_smoke/`，摘要是 `results/ladder/cv_smoke_summary.json`；兩者都只是冒煙，已列在 `.gitignore`，不要 commit：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 02:00:00 -J ladsmoke scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r3_settf r5_ctm --seeds 0 --force --out results/ladder/cv_smoke
```

folds 1–4 的 CV：靜態梯子、(3) 和對照，另加只報告、不作門檻的 `r3_pma`（Set Transformer 拿掉 SAB，只留 PMA 池化）。摘要寫到 `results/ladder/cv_summary.json`：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 12:00:00 -J ladder3 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r0_sum r1_clip r1_trim r2_mlp24 r2_staged r2_whole r2_length_only r2_pooled_delta r3_null r3_deepsets r3_deepsets_whole r3_settf r3_pma
```

再跑一次每人 8 題的 `r2_mlp24`（摘要裡的 `r2_mlp24@pp8`，正式模型的訓練量）。這是約 1 分鐘的靜態工作，用預設分區、1 小時就好：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J ladpp8 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r2_mlp24 --per-player 8
```

(4)(5) 只在摘要的 `G_set` 成立時跑；不成立時腳本 exit 2，要跑就得另加 `--force`。只報告的 `r4_xattn_t1`（cross-attention + GRU 只跑 1 tick）也算 r4，受同樣的限制：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 24:00:00 -J ladder45 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r4_xattn r5_ctm r5_ctm_minmax r4_xattn_t1
```

局長 ablation（Task 9 Step 5）：對 CV 最好的一階和 `r2_mlp24` 各加 `--no-counts` 跑一次。staged 和 whole 兩種表示都要跑（例如最好的是 `r3_deepsets`，就連 `r3_deepsets_whole` 一起列進 `--rungs`）。摘要裡是 `@nocounts` 標籤，和 `r2_length_only` 並列，成為「拿掉局長」和「只有局長」兩端：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 12:00:00 -J ladnocnt scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs <CV 最好的一階> r2_mlp24 --no-counts
```

`counts=False` 只拿掉明確的局長欄位（token 的各段 n_eff、present 和 log1p 手數；qglob 除了 log m 以外的欄；log n；q24 的 log_moves）。缺段的全零區塊，以及總和特徵的 /√n、/n 縮放，仍然帶有局長訊號，解讀時要註明。

每個 CV 檔都記下出處：git HEAD、是否有未 commit 的修改（`git_dirty`）、訓練相關模組的 `code_sha256`、TrainCfg、torch 版本、GPU 名稱、每個 split 的秒數和參數量。從 PC 同步過來的程式若沒有在國網上 commit，`git_dirty` 會是 true，版本以 `code_sha256` 為準。已存在的檔會被略過，所以同一個標籤的各 seed 可能出自不同版本的程式；摘要的 `mixed_versions` 會列出這些標籤，腳本也會印一行 `WARNING`（門檻照算，不會因此失敗）。看到時，把舊版本 seed 的檔刪掉重跑。

集合 rung 用內層提早停止：每個 split 從訓練玩家留出 10%（按段位分層）當內層驗證，最多 10 個 epoch，保留內層 CE 最低的權重（可能就是第 0 次，即 Δ ≡ 0、等於基準）。每個集合 rung 的 CV 檔都記下每個 split 的 `best_epoch`（保留的權重訓練了幾個 epoch，可為小數）、`inner_ce_init`、`inner_ce_best`、`stopped_early`、`n_inner_players`，`ladder_cv.py` 每跑完一個集合 rung 會印一行 `best_epoch per split: …`。2026-10-19 之前用固定 20 epoch 訓練的集合 rung CV 檔（例如 `results/ladder/cv_smoke/` 的 `r3_settf_s0.npz`、`r5_ctm_s0.npz`）已經作廢：腳本會略過已存在的檔，所以要先刪掉再重跑。`base_v*.npz`（基準 logits 快取）和靜態 rung 的檔不受影響，可以保留。

GPU 上的集合模型沒有開 deterministic 模式，不保證逐位元重現（index 和 MHA 的 backward 用 atomic add）；不過實測同一個 seed 在不同 H200 節點重跑（冒煙與正式 CV），分數與 train loss 都逐位元相同。

fold 0 的評估**只跑一次**，而且只在使用者從 `cv_summary.json` 選定 rung 之後才跑。CV 若用了 `--per-player` 或 `--no-counts`，這裡要給一樣的參數，讓設定對得上摘要裡的標籤；set rung 的 `--seeds` 也要和該標籤的 seeds 相同（預設 0 1 2）。對不上就 exit 2。結果寫到 `results/ladder/fold0_eval.json`，這個檔已存在時，不論 `--out` 指到哪裡，腳本都 exit 2，不會重評。fold 0 只評選定的設定（該 rung `TrainCfg.pick` 的 tick，除了 `r5_ctm_minmax_pc` 都是最後一個 tick），CV 裡只報告的 @T16、@certain、@final 不在 fold 0 上算：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 12:00:00 -J ladf0 scripts/nano4/run.sbatch python -I scripts/ladder_eval_fold0.py --rung <選定>
```

2026-10-24 追加（issue #2，待辦第 3 項）。兩個新 rung：`r5_ctm_minmax_pc`（`r5_ctm_minmax` 的模型與 loss，但內層 CE 與預測都取最確定 tick，`TrainCfg.pick = "certain"`；原本的 `r5_ctm_minmax` 與它的 CV 檔原樣保留）和 `r5_ctm_unet`（`r5_ctm` 換成 depth-4 U-Net synapse）。兩者都是 r5，仍要 `--force`。單張 H200 兩個 rung 約 1 小時：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 06:00:00 -J ladder3b scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r5_ctm_minmax_pc r5_ctm_unet --force
```

前三名 ensemble（`ens_top3`）不訓練：讀 `r3_pma`、`r4_xattn_t1`、`r5_ctm` 各 seed 的 CV 檔，機率平均後重算每題分數，寫成 `results/ladder/cv/ens_top3_s{seed}.npz`，再重寫 `cv_summary.json`，並印出事先指定的六組比較（`ens_top3`／`r5_ctm_minmax_pc`／`r5_ctm_unet` 各對 `r5_ctm` 與 `r3_pma`；只報告、不作門檻）。CPU 幾分鐘，但依國網規定仍送 Slurm：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J ladens scripts/nano4/run.sbatch python -I scripts/ladder_ensemble.py
```

`ens_top3` 的檔已存在時要加 `--overwrite`（成員重跑過就該重建）。要 commit 的是 `results/ladder/cv_summary.json`；`.npz` 一律不進 git。

### 4.1 集合模型接進提交管線（2026-10-25，待辦第 2 項）

四個步驟，順序固定：訓練 → 向量一致性 → mock test → seed 平均報告。前三步都讀 `results/ladder/vectors/fold{F}.npz`（`ladder_build_vectors.py` 的輸出，folds 0–4 都要在）。

**1. 訓練**（`scripts/train_set_model.py`）：一個 rung、三個 seed、folds 0–4 的全部訓練題（每人 32 題，和 CV 同一套抽樣），base logits 用 5 折交叉擬合。輸出 `results/models/set/{rung}_s{seed}.pt`（`state_dict`）加同名 `.json` 側檔（rung、seed、TrainCfg、標準化統計、內層提早停止診斷、provenance、訓練折與題目雜湊、base 檔名與 sha256），以及**配對的推論用 base** `results/models/rank_model_f01234.json`（`r2_mlp24` 在同一組訓練題上的全折擬合，`rank_model.json` 格式）。先跑 `r3_pma`；單張 H200 三個 seed 實測 5 分鐘（254,336 題，交叉擬合 base 71 秒，每個 seed 31–67 秒就被內層提早停止）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 04:00:00 -J setpma scripts/nano4/run.sbatch python -I scripts/train_set_model.py --rung r3_pma
```

模型檔已存在時腳本 exit 2、什麼都不寫，要重訓就加 `--overwrite`。`rank_model_f01234.json` 已存在時，腳本仍會重新擬合一次，確認是同一個 base（同一組訓練題的雜湊、folds、每人題數，參數也相同）就沿用、不改檔；不同就 exit 2（`--overwrite` 會連 base 一起重寫，配這個 base 的其他 rung 就得重訓）。所以 `r4_xattn_t1`、`r5_ctm` 之後各跑一次同樣的指令、不加任何旗標即可，三個 rung 共用同一個 base。`.pt`／`.json`／`rank_model_f01234.json` 都進 git（`results/models/` 已在版本控制內）。

**2. 向量一致性**（`scripts/check_game_vectors.py`，CPU）：提交管線從 shard 檔的每手 log p 算的每局向量，必須和訓練用的 `fold0.npz` 一樣。先用現有的 mock test 分片檔（第 1 項的 `submock`，`results/mock_fold0_40/`），以手數與 18 維 log p 總和把 40 題的 375 局對回 fold 0 的局，再逐欄比較 300 維 staged、75 維 whole 和 log1p 手數：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J vecchk scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --report results/mock_fold0_40/game_vectors_check.json
```

浮點欄位的容許差 `--tol` 預設 0.05（單卡／多卡與批次組成的 TF32 差異在每手 4e-3 nats 以內，向量差應該遠小於 0.05；差到 0.1–1 整組一起偏掉就是正規化、分段或濾手規則不同），計數欄位（各段 n_eff、present）必須完全相等；有局對不上（`unmatched`）或對到兩局（`ambiguous`）也算失敗，exit 1。多卡跑的 mock test 要給 `--nshards 8`。報告 JSON 請 commit。

**3. mock test 用集合模型決策**（`predict_submission.py --set-models … --set-base …`）：不必重新評分，`--merge` 會從 shard 檔重建每局向量；所以對已跑過的 mock test 直接 merge 一次即可（CPU 幾分鐘；集合模型預設在 CPU 上跑，`--set-device cuda` 可改）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J mockset scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 1 --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_set/submission_rank.csv --set-models results/models/set/r3_pma_s0.pt results/models/set/r3_pma_s1.pt results/models/set/r3_pma_s2.pt --set-base results/models/rank_model_f01234.json
```

（`--out` 指到新目錄時要先把 `results/mock_fold0_40/submission_rank_shard0of1.npz` 複製或連結過去；或直接用原目錄的 `--out`，提交檔會被覆寫。）平均規則：**同一 rung 的各 seed 先平均機率，再在 rung 之間平均**，最後期望得分決策；三個 `r3_pma` 就是 `r3_pma`，再加 `r4_xattn_t1`、`r5_ctm` 各三個就是 `ens_top3`。報告多了 `decision_model`（`set_models`）、每個 `.pt` 的側檔摘要、`set_rungs`（每個 rung 幾個模型）、`set_model_warnings`（側檔 `code_sha256` 與現行程式不同、或側檔記的 base 檔 sha256 與 `--set-base` 不同時只警告）、`set_model_fallback_questions`（集合模型機率非有限數的題目改用 base 的機率，印一行 WARNING，不改 exit code）。配對檢查：側檔記的訓練題雜湊必須等於 `--set-base` 文件的 `questions_sha256`（拿 `rank_model.json` 當 base 會被拒）；這個、側檔的 `d_in`／profiles／features 與程式不符、rung 不認得、`.pt` 的 sha256 不符 → exit 2。

mock test 來自 fold 0（現在是訓練資料），所以只驗**一致性**不看分數：步驟 2 的腳本加上同一組 `--set-models`／`--set-base`，會另走離線路徑——同 40 題直接用 `fold0.npz` 裡對到的局的向量與總和、同一個 base、同一個平均——逐題比對剛寫出的 `submission_rank_proba.csv` 與 `submission_rank.csv`：每題 max |Δp| 要在 `--proba-tol`（預設 1e-3；步驟 2 的向量差傳到機率上的量級）以內，40 題決策要全部相同，否則 exit 1：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J vecset scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40_set/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --set-models results/models/set/r3_pma_s0.pt results/models/set/r3_pma_s1.pt results/models/set/r3_pma_s2.pt --set-base results/models/rank_model_f01234.json --report results/mock_fold0_40_set/game_vectors_check.json
```

九個模型都在之後（`r3_pma`、`r4_xattn_t1`、`r5_ctm` 各三個 seed；2026-10-25 已訓練、檢查通過、進 git），`ens_top3` 的提交路徑就是第 1 項的 merge 指令加上：

```bash
--set-models results/models/set/{r3_pma,r4_xattn_t1,r5_ctm}_s{0,1,2}.pt --set-base results/models/rank_model_f01234.json
```

（大括號由你送 `sbatch` 的 shell 展開成九個路徑，`run.sbatch` 原樣轉交；分片評分的指令不用加這兩個旗標，只有 merge 要。）

**4. seed 平均的量化**（`ladder_ensemble.py --across-seeds`，CPU，只報告）：CV 統計量是每個 seed 各自決策再平均分數，提交只能出一個答案、是機率平均後決策；用現有 CV 檔算這個差（`r3_pma` 三個 seed 機率平均後的分數 − CV 分數，附玩家 bootstrap CI），順便重寫摘要：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J ladseed scripts/nano4/run.sbatch python -I scripts/ladder_ensemble.py --skip-ensemble --across-seeds r3_pma r4_xattn_t1 r5_ctm ens_top3
```

把印出的四行貼回 issue。

### 4.2 L1b 第一階段（2026-10-25，issue #6；計畫 `docs/superpowers/plans/2026-10-25-l1b-rank-heads.md`）

凍結 humanv0 trunk、「等級未知」metadata、10 個從 humanv0 policy head 複製出來的等級 head，只訓 head。模型 A 用 folds 1、2 的所有玩家（每人 ≤ 40 局）、給 folds 3、4（與 0）出每手 10 維 log p；模型 B 用 folds 3、4、給 folds 1、2（與 0）。輸出全部在 `results/l1b/`；快取、`.pt`、逐手 `.npz` 不進 git，`.json`（側檔、meta、報告）進 git。六步順序固定，11/04 當天 `ens_top3` 正式提交優先。

**1. 快取**（CPU，每半約 1.1×10⁷ 局面、12–13 GB，48 核 15–30 分；只要 KataGo 的 Python 套件，不讀權重）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -c 48 -t 02:00:00 -J cacheA scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A --workers 48
sbatch --exclude=25a-hgpn[001,003,004,007,062] -c 48 -t 02:00:00 -J cacheB scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 3 4 --out results/l1b/cache_B --workers 48
```

每個 part 寫完才改名，中斷重跑會略過已完成的 part；跑完寫 `meta.json`（玩家、局、局面、各級局面數、`games_sha256`）。預期 `games` ≈ 102,019（A）／101,711（B）。

**2. 初始化檢查**（單卡幾分鐘）：10 個 head 在第 0 步必須等於 humanv0 在等級未知 metadata 下的 policy，fp32 下 max |Δ log p| ≤ 1e-4：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J initA scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt --check-init
```

**3. 訓練**（單卡，每半 ≤ 3 小時：trunk 只做 bf16 前向、約 4,000–6,000 局面/秒，1.1×10⁷ 局面一個 epoch 30–45 分，最多 3 epoch）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 08:00:00 -J trainA scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 08:00:00 -J trainB scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_B --out results/l1b/heads_B.pt
```

類別平衡 CE、Adam lr 1e-4、batch 256、內層依玩家分層留出 10%、每 1/4 epoch 評一次內層 NLL、連續 4 次沒改善就停、保留最好（含第 0 步）。每次評估印一行 `epoch x.xx inner NLL …`，並寫 `heads_A.pt.ckpt`；工作被殺就重送同一指令，會從 ckpt 接著跑。輸出已存在要加 `--overwrite`。側檔 `heads_A.json` 進 git（`inner_nll_init`、`inner_nll_best`、`best_epoch`、`per_rank_positions`、provenance）。

**4. OOF 評分**（與 `results/scores` 同一批評估局：每人 ≤ 20 局、≥ 5 局；佈局相同，`sum_D … sum_6D`）。heads 的訓練 fold 與 `--folds` 有交集會被拒（exit 2）。8 卡分片 10–30 分：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scoreA scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 0 --out results/l1b/scores --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scoreB scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_B.pt --folds 1 2 0 --out results/l1b/scores_B --workers 10
```

fold 0 兩邊都出、分開存（之後給 L3 當訓練資料用）。每個 fold 寫 `meta_s{shard}.json`。

對照組（審查建議，便宜）：第 0 步的 head 就是 humanv0 在等級未知 metadata 下的 policy，用它評一次 folds 3、4，和訓練後的結果擺在一起看，可以把「訓練學到的」和「bf16 trunk + 等級未知 metadata」的效應分開。`l1b_train.py --epochs 0 --out results/l1b/heads_init.pt` 會寫出這組初始化 head（側檔 `train_folds` 來自快取，所以也受同樣的 OOF 檢查），再用同樣的 `l1b_score.py` 指令、`--out results/l1b/scores_init` 評分，`l1b_early_signal.py --l1b results/l1b/scores_init` 得到對照表。

**5. 早期訊號**（CPU 幾分鐘；**決策點 1**）：每級每手 NLL（L1b 自己的 head 對 Human SL 該級最好的 profile 與 pilot 峰值 profile）和相鄰等級 AUC（每人一題 20 局，posterior-mean index，與 §2.4.1(d) 同法；另報每局），印兩張表、寫 JSON：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J earlyA scripts/nano4/run.sbatch python -I scripts/l1b_early_signal.py --l1b results/l1b/scores --hsl results/scores --folds 3 4 --out results/l1b/early_signal_A.json
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J earlyB scripts/nano4/run.sbatch python -I scripts/l1b_early_signal.py --l1b results/l1b/scores_B --hsl results/scores --folds 1 2 --out results/l1b/early_signal_B.json
```

兩張表貼回 issue #6。pilot 的 n=20 參考：1D|2D 0.62、2D|3D 0.73、3D|4D 0.65。建議的讀法在計畫裡：L1b 在 ≥ 7 級的 NLL 較低、或兩個弱邊界的 AUC 都升 → 進第 6 步；都沒有 → 直接規劃第二階段。由使用者決定。

**6. CV**（CPU 20–40 分；**決策點 2**）：folds 1–4 留一 fold，驗證題與訓練題和梯子完全相同（從評分表還原 `fold_players` 的順序，不讀訓練資料），`MLPModel(hidden=32)` 各擬合一次；三組特徵 `hsl`（24 維）／`l1b`（16 維）／`fused`（40 維），兩組事先指定的比較 `l1b` vs `hsl`、`fused` vs `hsl`，條件 ①③④（② 不適用）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 02:00:00 -J l1bcv scripts/nano4/run.sbatch python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 --out results/l1b/cv_summary.json
```

把印出的五行（三組分數、兩組比較）貼回 issue #6；`cv_summary.json` 進 git。

### 4.3 提交改用鑑別式 L1b 的 fused MLP（2026-10-26，issue #9）

提交的決策模型從 `ens_top3` 換成 fused MLP：Human SL 24 維 + L1b 16 維題目特徵，`MLPModel(hidden=32)`，A、B 兩個 L1b 模型各出一套特徵、各預測一次、機率平均（CV 0.6545、fold 0 0.6535；`ens_top3` 留當退路）。權重檔：`results/l1b/heads3_{A,B}.pt` 加 `heads3_{A,B}.trunk.pt`（微調後的 trunk，各 108 MB，不進 git，見下面第 5 步）。五步順序固定：

**1. 擬合 fused MLP**（CPU 幾分鐘；訓練完全照 `l1b_fold0_check.py`：folds 1–4、每人 32 題、folds 1、2 用 B 的評分、3、4 用 A 的；A 的評分目錄在 nano4 上是 `results/l1b/scores3`，B 是 `results/l1b/scores3_B`）。順便在 fold 0 重算 `hsl`／`fused_A`／`fused_B`／`fused_avg`，`--reference` 會印出與 `fold0_check.json` 的差（應為 0），並存下 fold 0 每題機率給第 4 步用：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J fitfused scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json B:results/l1b/scores3_B:results/l1b/heads3_B.json --out results/models/rank_model_fused.json --fold0-proba results/l1b/fold0_fused_proba.csv --reference results/l1b/fold0_check.json --reference-run disc
```

`rank_model_fused.json` 與 `fold0_fused_proba.csv` 進 git。文件裡的 `l1b_models` 記了 A、B 的 heads 檔 sha256 與 trunk sha256，merge 時會用來確認分片用的是同一組模型。

**2. mock test 分片評分加 L1b**（`--l1b-heads` 讓每個分片在 18 個 profile 之外多做 A、B 各一次 trunk 前向，分片檔多 `l1b_sums`／`l1b_move_logp`；不加這個旗標的分片檔和以前一模一樣）。單卡一次、4 卡一次，給第 4 步比對：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 02:00:00 -J mockf1 scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused.json
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus --gres=gpu:4 -c 48 -t 02:00:00 -J mockf4 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused4/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J mockf4m scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 4 --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused4/submission_rank.csv --fused-model results/models/rank_model_fused.json
```

單卡那次把評分和 merge 做在同一個工作裡（`--fused-model` 在評分完直接用）。報告的 `decision_model` 是 `fused`，`fused_model` 記 fused 文件的 sha256 與 `l1b_models`，`l1b_models` 記分片用的模型。配對檢查：分片的 (state_sha256, trunk_sha256) 集合必須等於 fused 文件裡的，否則 exit 2；`--fused-model` 和 `--set-models` 互斥。

分片時的 CPU：`shards.sh` 把每個分片的 `OMP_NUM_THREADS`／`MKL_NUM_THREADS` 設成 2（`SHARD_THREADS=k` 可改）。PR #10 實測 4 卡時每個分片主行程開了 44 條 torch 執行緒在 48 核上空轉，worker 搶不到 CPU，65 秒的評分拖成 8 分鐘；模型在 GPU 上，主行程只搬批次、加總 log p，2 條就夠。每個分片 log 的開頭印 `torch CPU threads: 2` 代表設定生效。單卡直接跑的工作不受影響。

**3. 每局 L1b 總和對帳**（CPU；`check_game_vectors.py` 原有的向量檢查加 `--l1b-scores`：局對回 fold 0 後，分片的 A、B 每局 10 維總和逐局比 `l1b_score.py` 的表，目錄靠 `meta_s*.json` 的 sha256 對到模型；容差 `--l1b-tol` 0.05）。PR #10 之前的 `l1b_score.py` 沒把 `trunk_sha256` 寫進 `meta_s*.json`，這種目錄只靠 `state_sha256` 對模型並印 WARNING（state 的 sha256 已能唯一識別一次訓練）；之後評分的目錄兩個都比：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J vecl1b scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40_fused/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --l1b-scores results/l1b/scores3/fold0 results/l1b/scores3_B/fold0 --report results/mock_fold0_40_fused/game_vectors_check.json
```

**4. 機率與決策對帳**（CPU；`answers.csv` 的 `source_question` 把 40 題對回 fold 0 的題；管線 vs 離線 40 題決策全同、max |Δp| ≤ `--tol` 1e-3；再比單卡 vs 4 卡）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J fusedchk scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv --offline results/l1b/fold0_fused_proba.csv --answers tests/fixtures/mock_fold0_40/answers.csv --report results/mock_fold0_40_fused/fused_check.json
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J fusedchk4 scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv --other results/mock_fold0_40_fused4/submission_rank.csv --report results/mock_fold0_40_fused4/fused_check.json
```

兩份 `fused_check.json` 與 `game_vectors_check.json` 進 git。容差 1e-3 是分片間每手 log p 差（≤ 4e-3）傳到機率上的量級，實測若超過，先看第 3 步的每局總和差再調。

**5. 權重封存**（CPU；不進 git）：`pack_weights.py` 把 heads、trunk、側檔、`rank_model*.json`、集合模型打成一個 tar 加 `SHA256SUMS`，打包前逐檔核對側檔與 fused 文件記的 sha256。得獎時由使用者上傳成 GitHub Release 資產（單檔上限 2 GB）；repo 裡的側檔與 `rank_model_fused.json` 已記每個檔的 sha256，對得上就是同一份權重：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J pack scripts/nano4/run.sbatch python -I scripts/pack_weights.py --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --models results/models/rank_model.json results/models/rank_model_f01234.json results/models/rank_model_fused.json --set-dir results/models/set --out release/weights-2026-11.tar
```

**11/04 正式流程**：8 卡 `shards.sh 8 … --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt`（§4 的 `sub8` 指令加這個旗標；`shards.sh` 自動限制每分片 2 條 CPU 執行緒）→ `--merge --nshards 8 --fused-model results/models/rank_model_fused.json`。退路：同一組分片檔 `--merge` 改給 `--set-models …`（`ens_top3`）或什麼都不給（`rank_model.json`），不用重新評分。

## 5. 同步程式碼、取回結果

在本機改完程式後，用 Git Bash 同步（走 2222 埠的傳輸節點）。只會傳程式碼（src、scripts、tests、viz、pyproject.toml、requirements.txt），不會傳資料：

```bash
bash scripts/sync_to_nano4.sh
```

腳本預設傳到 `~/AICUP_2026-1`。這個專案（含 L3 聚合器梯子）在國網上的位置是 `~/BK_AICUP_2026-1`，同步時要指定目標：`NANO4_DIR=BK_AICUP_2026-1 bash scripts/sync_to_nano4.sh`。

從國網把結果拉回本機，例如：

```bash
scp -P 2222 nano4:AICUP_2026-1/results/bench_nano4.json results/
```

## 6. 被拒或出錯時

- `scp` 在 2FA 之後出現 `Connection closed`：連到了 22 埠的登入節點，加上 `-P 2222` 改走傳輸節點。
- 計畫或分區不對：用 `-A <計畫代號>` 或 `-p <分區>` 覆蓋。預設計畫是 gov113074。
- CPU 或 GPU 數量不符合分區規定：用 `-c` 和 `--gres=gpu:N` 調整，錯誤訊息通常會寫出上限。
- 記憶體不足：加上 `--mem=64G` 之類的參數。
- `setup.sh` 說連不到網路：登入節點可能要設 proxy（`export https_proxy=...`）。
- torch 說驅動太舊：用預設的 cu126 版本重裝（`TORCH_CUDA=cu126 bash scripts/nano4/setup.sh`）。
- KataGo 引擎（`scripts/katago.sh`）在國網跑不起來：它需要的 libzip 和 OpenSSL 1.1 只裝在 lab01。只有 0A 一致性檢查會用到引擎，評分和訓練都只用 PyTorch，相關測試會自動略過。
