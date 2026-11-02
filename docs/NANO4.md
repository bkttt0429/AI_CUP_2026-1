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
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -J tests scripts/nano4/run.sbatch python -m pytest -q
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

L3 聚合器梯子（`docs/superpowers/plans/2026-10-19-l3-aggregator-ladder-ctm.md` Task 9）的工作都在 `~/BK_AICUP_2026-1` 下送出，並一律加 `--exclude=25a-hgpn[001,003,004,007,039,062]` 避開壞 GPU 節點。輸出全部在 `results/ladder/`。

建每局向量（folds 0–4）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J ladvec scripts/nano4/run.sbatch python -I scripts/ladder_build_vectors.py --folds 0 1 2 3 4
```

跑 CV 之前，先做一次 CUDA 冒煙：`r3_settf` 和 `r5_ctm` 各用 GPU 跑一個 seed，確認 CUDA 路徑能跑完（損失或機率出現 NaN／inf 時腳本會停下、不寫檔）。輸出在 `results/ladder/cv_smoke/`，摘要是 `results/ladder/cv_smoke_summary.json`；兩者都只是冒煙，已列在 `.gitignore`，不要 commit：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J ladsmoke scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r3_settf r5_ctm --seeds 0 --force --out results/ladder/cv_smoke
```

folds 1–4 的 CV：靜態梯子、(3) 和對照，另加只報告、不作門檻的 `r3_pma`（Set Transformer 拿掉 SAB，只留 PMA 池化）。摘要寫到 `results/ladder/cv_summary.json`：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 12:00:00 -J ladder3 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r0_sum r1_clip r1_trim r2_mlp24 r2_staged r2_whole r2_length_only r2_pooled_delta r3_null r3_deepsets r3_deepsets_whole r3_settf r3_pma
```

再跑一次每人 8 題的 `r2_mlp24`（摘要裡的 `r2_mlp24@pp8`，正式模型的訓練量）。這是約 1 分鐘的靜態工作，用預設分區、1 小時就好：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J ladpp8 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r2_mlp24 --per-player 8
```

(4)(5) 只在摘要的 `G_set` 成立時跑；不成立時腳本 exit 2，要跑就得另加 `--force`。只報告的 `r4_xattn_t1`（cross-attention + GRU 只跑 1 tick）也算 r4，受同樣的限制：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 24:00:00 -J ladder45 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r4_xattn r5_ctm r5_ctm_minmax r4_xattn_t1
```

局長 ablation（Task 9 Step 5）：對 CV 最好的一階和 `r2_mlp24` 各加 `--no-counts` 跑一次。staged 和 whole 兩種表示都要跑（例如最好的是 `r3_deepsets`，就連 `r3_deepsets_whole` 一起列進 `--rungs`）。摘要裡是 `@nocounts` 標籤，和 `r2_length_only` 並列，成為「拿掉局長」和「只有局長」兩端：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 12:00:00 -J ladnocnt scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs <CV 最好的一階> r2_mlp24 --no-counts
```

`counts=False` 只拿掉明確的局長欄位（token 的各段 n_eff、present 和 log1p 手數；qglob 除了 log m 以外的欄；log n；q24 的 log_moves）。缺段的全零區塊，以及總和特徵的 /√n、/n 縮放，仍然帶有局長訊號，解讀時要註明。

每個 CV 檔都記下出處：git HEAD、是否有未 commit 的修改（`git_dirty`）、訓練相關模組的 `code_sha256`、TrainCfg、torch 版本、GPU 名稱、每個 split 的秒數和參數量。從 PC 同步過來的程式若沒有在國網上 commit，`git_dirty` 會是 true，版本以 `code_sha256` 為準。已存在的檔會被略過，所以同一個標籤的各 seed 可能出自不同版本的程式；摘要的 `mixed_versions` 會列出這些標籤，腳本也會印一行 `WARNING`（門檻照算，不會因此失敗）。看到時，把舊版本 seed 的檔刪掉重跑。

集合 rung 用內層提早停止：每個 split 從訓練玩家留出 10%（按段位分層）當內層驗證，最多 10 個 epoch，保留內層 CE 最低的權重（可能就是第 0 次，即 Δ ≡ 0、等於基準）。每個集合 rung 的 CV 檔都記下每個 split 的 `best_epoch`（保留的權重訓練了幾個 epoch，可為小數）、`inner_ce_init`、`inner_ce_best`、`stopped_early`、`n_inner_players`，`ladder_cv.py` 每跑完一個集合 rung 會印一行 `best_epoch per split: …`。2026-10-19 之前用固定 20 epoch 訓練的集合 rung CV 檔（例如 `results/ladder/cv_smoke/` 的 `r3_settf_s0.npz`、`r5_ctm_s0.npz`）已經作廢：腳本會略過已存在的檔，所以要先刪掉再重跑。`base_v*.npz`（基準 logits 快取）和靜態 rung 的檔不受影響，可以保留。

GPU 上的集合模型沒有開 deterministic 模式，不保證逐位元重現（index 和 MHA 的 backward 用 atomic add）；不過實測同一個 seed 在不同 H200 節點重跑（冒煙與正式 CV），分數與 train loss 都逐位元相同。

fold 0 的評估**只跑一次**，而且只在使用者從 `cv_summary.json` 選定 rung 之後才跑。CV 若用了 `--per-player` 或 `--no-counts`，這裡要給一樣的參數，讓設定對得上摘要裡的標籤；set rung 的 `--seeds` 也要和該標籤的 seeds 相同（預設 0 1 2）。對不上就 exit 2。結果寫到 `results/ladder/fold0_eval.json`，這個檔已存在時，不論 `--out` 指到哪裡，腳本都 exit 2，不會重評。fold 0 只評選定的設定（該 rung `TrainCfg.pick` 的 tick，除了 `r5_ctm_minmax_pc` 都是最後一個 tick），CV 裡只報告的 @T16、@certain、@final 不在 fold 0 上算：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 12:00:00 -J ladf0 scripts/nano4/run.sbatch python -I scripts/ladder_eval_fold0.py --rung <選定>
```

2026-10-24 追加（issue #2，待辦第 3 項）。兩個新 rung：`r5_ctm_minmax_pc`（`r5_ctm_minmax` 的模型與 loss，但內層 CE 與預測都取最確定 tick，`TrainCfg.pick = "certain"`；原本的 `r5_ctm_minmax` 與它的 CV 檔原樣保留）和 `r5_ctm_unet`（`r5_ctm` 換成 depth-4 U-Net synapse）。兩者都是 r5，仍要 `--force`。單張 H200 兩個 rung 約 1 小時：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 06:00:00 -J ladder3b scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r5_ctm_minmax_pc r5_ctm_unet --force
```

前三名 ensemble（`ens_top3`）不訓練：讀 `r3_pma`、`r4_xattn_t1`、`r5_ctm` 各 seed 的 CV 檔，機率平均後重算每題分數，寫成 `results/ladder/cv/ens_top3_s{seed}.npz`，再重寫 `cv_summary.json`，並印出事先指定的六組比較（`ens_top3`／`r5_ctm_minmax_pc`／`r5_ctm_unet` 各對 `r5_ctm` 與 `r3_pma`；只報告、不作門檻）。CPU 幾分鐘，但依國網規定仍送 Slurm：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J ladens scripts/nano4/run.sbatch python -I scripts/ladder_ensemble.py
```

`ens_top3` 的檔已存在時要加 `--overwrite`（成員重跑過就該重建）。要 commit 的是 `results/ladder/cv_summary.json`；`.npz` 一律不進 git。

### 4.1 集合模型接進提交管線（2026-10-25，待辦第 2 項）

四個步驟，順序固定：訓練 → 向量一致性 → mock test → seed 平均報告。前三步都讀 `results/ladder/vectors/fold{F}.npz`（`ladder_build_vectors.py` 的輸出，folds 0–4 都要在）。

**1. 訓練**（`scripts/train_set_model.py`）：一個 rung、三個 seed、folds 0–4 的全部訓練題（每人 32 題，和 CV 同一套抽樣），base logits 用 5 折交叉擬合。輸出 `results/models/set/{rung}_s{seed}.pt`（`state_dict`）加同名 `.json` 側檔（rung、seed、TrainCfg、標準化統計、內層提早停止診斷、provenance、訓練折與題目雜湊、base 檔名與 sha256），以及**配對的推論用 base** `results/models/rank_model_f01234.json`（`r2_mlp24` 在同一組訓練題上的全折擬合，`rank_model.json` 格式）。先跑 `r3_pma`；單張 H200 三個 seed 實測 5 分鐘（254,336 題，交叉擬合 base 71 秒，每個 seed 31–67 秒就被內層提早停止）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus -t 04:00:00 -J setpma scripts/nano4/run.sbatch python -I scripts/train_set_model.py --rung r3_pma
```

模型檔已存在時腳本 exit 2、什麼都不寫，要重訓就加 `--overwrite`。`rank_model_f01234.json` 已存在時，腳本仍會重新擬合一次，確認是同一個 base（同一組訓練題的雜湊、folds、每人題數，參數也相同）就沿用、不改檔；不同就 exit 2（`--overwrite` 會連 base 一起重寫，配這個 base 的其他 rung 就得重訓）。所以 `r4_xattn_t1`、`r5_ctm` 之後各跑一次同樣的指令、不加任何旗標即可，三個 rung 共用同一個 base。`.pt`／`.json`／`rank_model_f01234.json` 都進 git（`results/models/` 已在版本控制內）。

**2. 向量一致性**（`scripts/check_game_vectors.py`，CPU）：提交管線從 shard 檔的每手 log p 算的每局向量，必須和訓練用的 `fold0.npz` 一樣。先用現有的 mock test 分片檔（第 1 項的 `submock`，`results/mock_fold0_40/`），以手數與 18 維 log p 總和把 40 題的 375 局對回 fold 0 的局，再逐欄比較 300 維 staged、75 維 whole 和 log1p 手數：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J vecchk scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --report results/mock_fold0_40/game_vectors_check.json
```

浮點欄位的容許差 `--tol` 預設 0.05（單卡／多卡與批次組成的 TF32 差異在每手 4e-3 nats 以內，向量差應該遠小於 0.05；差到 0.1–1 整組一起偏掉就是正規化、分段或濾手規則不同），計數欄位（各段 n_eff、present）必須完全相等；有局對不上（`unmatched`）或對到兩局（`ambiguous`）也算失敗，exit 1。多卡跑的 mock test 要給 `--nshards 8`。報告 JSON 請 commit。

**3. mock test 用集合模型決策**（`predict_submission.py --set-models … --set-base …`）：不必重新評分，`--merge` 會從 shard 檔重建每局向量；所以對已跑過的 mock test 直接 merge 一次即可（CPU 幾分鐘；集合模型預設在 CPU 上跑，`--set-device cuda` 可改）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J mockset scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 1 --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_set/submission_rank.csv --set-models results/models/set/r3_pma_s0.pt results/models/set/r3_pma_s1.pt results/models/set/r3_pma_s2.pt --set-base results/models/rank_model_f01234.json
```

（`--out` 指到新目錄時要先把 `results/mock_fold0_40/submission_rank_shard0of1.npz` 複製或連結過去；或直接用原目錄的 `--out`，提交檔會被覆寫。）平均規則：**同一 rung 的各 seed 先平均機率，再在 rung 之間平均**，最後期望得分決策；三個 `r3_pma` 就是 `r3_pma`，再加 `r4_xattn_t1`、`r5_ctm` 各三個就是 `ens_top3`。報告多了 `decision_model`（`set_models`）、每個 `.pt` 的側檔摘要、`set_rungs`（每個 rung 幾個模型）、`set_model_warnings`（側檔 `code_sha256` 與現行程式不同、或側檔記的 base 檔 sha256 與 `--set-base` 不同時只警告）、`set_model_fallback_questions`（集合模型機率非有限數的題目改用 base 的機率，印一行 WARNING，不改 exit code）。配對檢查：側檔記的訓練題雜湊必須等於 `--set-base` 文件的 `questions_sha256`（拿 `rank_model.json` 當 base 會被拒）；這個、側檔的 `d_in`／profiles／features 與程式不符、rung 不認得、`.pt` 的 sha256 不符 → exit 2。

mock test 來自 fold 0（現在是訓練資料），所以只驗**一致性**不看分數：步驟 2 的腳本加上同一組 `--set-models`／`--set-base`，會另走離線路徑——同 40 題直接用 `fold0.npz` 裡對到的局的向量與總和、同一個 base、同一個平均——逐題比對剛寫出的 `submission_rank_proba.csv` 與 `submission_rank.csv`：每題 max |Δp| 要在 `--proba-tol`（預設 1e-3；步驟 2 的向量差傳到機率上的量級）以內，40 題決策要全部相同，否則 exit 1：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J vecset scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40_set/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --set-models results/models/set/r3_pma_s0.pt results/models/set/r3_pma_s1.pt results/models/set/r3_pma_s2.pt --set-base results/models/rank_model_f01234.json --report results/mock_fold0_40_set/game_vectors_check.json
```

九個模型都在之後（`r3_pma`、`r4_xattn_t1`、`r5_ctm` 各三個 seed；2026-10-25 已訓練、檢查通過、進 git），`ens_top3` 的提交路徑就是第 1 項的 merge 指令加上：

```bash
--set-models results/models/set/{r3_pma,r4_xattn_t1,r5_ctm}_s{0,1,2}.pt --set-base results/models/rank_model_f01234.json
```

（大括號由你送 `sbatch` 的 shell 展開成九個路徑，`run.sbatch` 原樣轉交；分片評分的指令不用加這兩個旗標，只有 merge 要。）

**4. seed 平均的量化**（`ladder_ensemble.py --across-seeds`，CPU，只報告）：CV 統計量是每個 seed 各自決策再平均分數，提交只能出一個答案、是機率平均後決策；用現有 CV 檔算這個差（`r3_pma` 三個 seed 機率平均後的分數 − CV 分數，附玩家 bootstrap CI），順便重寫摘要：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J ladseed scripts/nano4/run.sbatch python -I scripts/ladder_ensemble.py --skip-ensemble --across-seeds r3_pma r4_xattn_t1 r5_ctm ens_top3
```

把印出的四行貼回 issue。

### 4.2 L1b 第一階段（2026-10-25，issue #6；計畫 `docs/superpowers/plans/2026-10-25-l1b-rank-heads.md`）

凍結 humanv0 trunk、「等級未知」metadata、10 個從 humanv0 policy head 複製出來的等級 head，只訓 head。模型 A 用 folds 1、2 的所有玩家（每人 ≤ 40 局）、給 folds 3、4（與 0）出每手 10 維 log p；模型 B 用 folds 3、4、給 folds 1、2（與 0）。輸出全部在 `results/l1b/`；快取、`.pt`、逐手 `.npz` 不進 git，`.json`（側檔、meta、報告）進 git。六步順序固定，11/04 當天 `ens_top3` 正式提交優先。

**1. 快取**（CPU，每半約 1.1×10⁷ 局面、12–13 GB，48 核 15–30 分；只要 KataGo 的 Python 套件，不讀權重）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J cacheA scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A --workers 48
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J cacheB scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 3 4 --out results/l1b/cache_B --workers 48
```

每個 part 寫完才改名，中斷重跑會略過已完成的 part；跑完寫 `meta.json`（玩家、局、局面、各級局面數、`games_sha256`）。預期 `games` ≈ 102,019（A）／101,711（B）。

**2. 初始化檢查**（單卡幾分鐘）：10 個 head 在第 0 步必須等於 humanv0 在等級未知 metadata 下的 policy，fp32 下 max |Δ log p| ≤ 1e-4：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J initA scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt --check-init
```

**3. 訓練**（單卡，每半 ≤ 3 小時：trunk 只做 bf16 前向、約 4,000–6,000 局面/秒，1.1×10⁷ 局面一個 epoch 30–45 分，最多 3 epoch）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 08:00:00 -J trainA scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 08:00:00 -J trainB scripts/nano4/run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_B --out results/l1b/heads_B.pt
```

類別平衡 CE、Adam lr 1e-4、batch 256、內層依玩家分層留出 10%、每 1/4 epoch 評一次內層 NLL、連續 4 次沒改善就停、保留最好（含第 0 步）。每次評估印一行 `epoch x.xx inner NLL …`，並寫 `heads_A.pt.ckpt`；工作被殺就重送同一指令，會從 ckpt 接著跑。輸出已存在要加 `--overwrite`。側檔 `heads_A.json` 進 git（`inner_nll_init`、`inner_nll_best`、`best_epoch`、`per_rank_positions`、provenance）。

**4. OOF 評分**（與 `results/scores` 同一批評估局：每人 ≤ 20 局、≥ 5 局；佈局相同，`sum_D … sum_6D`）。heads 的訓練 fold 與 `--folds` 有交集會被拒（exit 2）。8 卡分片 10–30 分：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scoreA scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 0 --out results/l1b/scores --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scoreB scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_B.pt --folds 1 2 0 --out results/l1b/scores_B --workers 10
```

fold 0 兩邊都出、分開存（之後給 L3 當訓練資料用）。每個 fold 寫 `meta_s{shard}.json`。

對照組（審查建議，便宜）：第 0 步的 head 就是 humanv0 在等級未知 metadata 下的 policy，用它評一次 folds 3、4，和訓練後的結果擺在一起看，可以把「訓練學到的」和「bf16 trunk + 等級未知 metadata」的效應分開。`l1b_train.py --epochs 0 --out results/l1b/heads_init.pt` 會寫出這組初始化 head（側檔 `train_folds` 來自快取，所以也受同樣的 OOF 檢查），再用同樣的 `l1b_score.py` 指令、`--out results/l1b/scores_init` 評分，`l1b_early_signal.py --l1b results/l1b/scores_init` 得到對照表。

**5. 早期訊號**（CPU 幾分鐘；**決策點 1**）：每級每手 NLL（L1b 自己的 head 對 Human SL 該級最好的 profile 與 pilot 峰值 profile）和相鄰等級 AUC（每人一題 20 局，posterior-mean index，與 §2.4.1(d) 同法；另報每局），印兩張表、寫 JSON：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J earlyA scripts/nano4/run.sbatch python -I scripts/l1b_early_signal.py --l1b results/l1b/scores --hsl results/scores --folds 3 4 --out results/l1b/early_signal_A.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J earlyB scripts/nano4/run.sbatch python -I scripts/l1b_early_signal.py --l1b results/l1b/scores_B --hsl results/scores --folds 1 2 --out results/l1b/early_signal_B.json
```

兩張表貼回 issue #6。pilot 的 n=20 參考：1D|2D 0.62、2D|3D 0.73、3D|4D 0.65。建議的讀法在計畫裡：L1b 在 ≥ 7 級的 NLL 較低、或兩個弱邊界的 AUC 都升 → 進第 6 步；都沒有 → 直接規劃第二階段。由使用者決定。

**6. CV**（CPU 20–40 分；**決策點 2**）：folds 1–4 留一 fold，驗證題與訓練題和梯子完全相同（從評分表還原 `fold_players` 的順序，不讀訓練資料），`MLPModel(hidden=32)` 各擬合一次；三組特徵 `hsl`（24 維）／`l1b`（16 維）／`fused`（40 維），兩組事先指定的比較 `l1b` vs `hsl`、`fused` vs `hsl`，條件 ①③④（② 不適用）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J l1bcv scripts/nano4/run.sbatch python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 --out results/l1b/cv_summary.json
```

把印出的五行（三組分數、兩組比較）貼回 issue #6；`cv_summary.json` 進 git。

### 4.3 提交改用鑑別式 L1b 的 fused MLP（2026-10-26，issue #9）

提交的決策模型從 `ens_top3` 換成 fused MLP：Human SL 24 維 + L1b 16 維題目特徵，`MLPModel(hidden=32)`，A、B 兩個 L1b 模型各出一套特徵、各預測一次、機率平均（CV 0.6545、fold 0 0.6535；`ens_top3` 留當退路）。權重檔：`results/l1b/heads3_{A,B}.pt` 加 `heads3_{A,B}.trunk.pt`（微調後的 trunk，各 108 MB，不進 git，見下面第 5 步）。五步順序固定：

**1. 擬合 fused MLP**（CPU 幾分鐘；訓練完全照 `l1b_fold0_check.py`：folds 1–4、每人 32 題、folds 1、2 用 B 的評分、3、4 用 A 的；A 的評分目錄在 nano4 上是 `results/l1b/scores3`，B 是 `results/l1b/scores3_B`）。順便在 fold 0 重算 `hsl`／`fused_A`／`fused_B`／`fused_avg`，`--reference` 會印出與 `fold0_check.json` 的差（應為 0），並存下 fold 0 每題機率給第 4 步用：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J fitfused scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json B:results/l1b/scores3_B:results/l1b/heads3_B.json --out results/models/rank_model_fused.json --fold0-proba results/l1b/fold0_fused_proba.csv --reference results/l1b/fold0_check.json --reference-run disc
```

`rank_model_fused.json` 與 `fold0_fused_proba.csv` 進 git。文件裡的 `l1b_models` 記了 A、B 的 heads 檔 sha256 與 trunk sha256，merge 時會用來確認分片用的是同一組模型。

**2. mock test 分片評分加 L1b**（`--l1b-heads` 讓每個分片在 18 個 profile 之外多做 A、B 各一次 trunk 前向，分片檔多 `l1b_sums`／`l1b_move_logp`；不加這個旗標的分片檔和以前一模一樣）。單卡一次、4 卡一次，給第 4 步比對：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J mockf1 scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:4 -c 48 -t 02:00:00 -J mockf4 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused4/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J mockf4m scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 4 --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_fused4/submission_rank.csv --fused-model results/models/rank_model_fused.json
```

單卡那次把評分和 merge 做在同一個工作裡（`--fused-model` 在評分完直接用）。報告的 `decision_model` 是 `fused`，`fused_model` 記 fused 文件的 sha256 與 `l1b_models`，`l1b_models` 記分片用的模型。配對檢查：分片的 (state_sha256, trunk_sha256) 集合必須等於 fused 文件裡的，否則 exit 2；`--fused-model` 和 `--set-models` 互斥。

分片時的 CPU：`shards.sh` 把每個分片的 `OMP_NUM_THREADS`／`MKL_NUM_THREADS` 設成 2（`SHARD_THREADS=k` 可改）。PR #10 實測 4 卡時每個分片主行程開了 44 條 torch 執行緒在 48 核上空轉，worker 搶不到 CPU，65 秒的評分拖成 8 分鐘；模型在 GPU 上，主行程只搬批次、加總 log p，2 條就夠。每個分片 log 的開頭印 `torch CPU threads: 2` 代表設定生效。單卡直接跑的工作不受影響。

**3. 每局 L1b 總和對帳**（CPU；`check_game_vectors.py` 原有的向量檢查加 `--l1b-scores`：局對回 fold 0 後，分片的 A、B 每局 10 維總和逐局比 `l1b_score.py` 的表，目錄靠 `meta_s*.json` 的 sha256 對到模型）。容差兩條、都按 head：每局 |Δ sum| / n_moves ≤ `--l1b-tol` 0.02，且跨局的平均有號差 |mean(Δ sum / n_moves)| ≤ `--l1b-bias-tol` 1e-3。前者放過 bf16 的批次雜訊（trunk 在 bf16 autocast 下跑，同一局的局面離線和分片落在不同批次：PR #10 實測 375 局大多數逐位元相同，少數每手差到 0.007、每局總和差到 0.57，HSL 分片間每手 4e-3 同量級，對機率的影響就是第 4 步的 ≤ 3.9e-4），後者抓用錯模型或 metadata 那種整體偏移（雜訊的平均只有 1e-5）。PR #10 之前的 `l1b_score.py` 沒把 `trunk_sha256` 寫進 `meta_s*.json`，這種目錄只靠 `state_sha256` 對模型並印 WARNING（state 的 sha256 已能唯一識別一次訓練）；之後評分的目錄兩個都比：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J vecl1b scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40_fused/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --l1b-scores results/l1b/scores3/fold0 results/l1b/scores3_B/fold0 --report results/mock_fold0_40_fused/game_vectors_check.json
```

**4. 機率與決策對帳**（CPU；`answers.csv` 的 `source_question` 把 40 題對回 fold 0 的題；管線 vs 離線 40 題決策全同、max |Δp| ≤ `--tol` 1e-3；再比單卡 vs 4 卡）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J fusedchk scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv --offline results/l1b/fold0_fused_proba.csv --answers tests/fixtures/mock_fold0_40/answers.csv --report results/mock_fold0_40_fused/fused_check.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J fusedchk4 scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv --other results/mock_fold0_40_fused4/submission_rank.csv --report results/mock_fold0_40_fused4/fused_check.json
```

兩份 `fused_check.json` 與 `game_vectors_check.json` 進 git。容差 1e-3 是分片間每手 log p 差（≤ 4e-3）傳到機率上的量級，實測若超過，先看第 3 步的每局總和差再調。

**5. 權重封存**（CPU；不進 git）：`pack_weights.py` 把 heads、trunk、側檔、`rank_model*.json`、集合模型打成一個 tar 加 `SHA256SUMS`，打包前逐檔核對側檔與 fused 文件記的 sha256。得獎時由使用者上傳成 GitHub Release 資產（單檔上限 2 GB）；repo 裡的側檔與 `rank_model_fused.json` 已記每個檔的 sha256，對得上就是同一份權重：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J pack scripts/nano4/run.sbatch python -I scripts/pack_weights.py --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --models results/models/rank_model.json results/models/rank_model_f01234.json results/models/rank_model_fused.json --set-dir results/models/set --out release/weights-2026-11.tar
```

**11/04 正式流程**：提交已再換成 fused_opp，見 §4.4 最後一段；`rank_model_fused.json`（本節的模型）留作退路，同一組分片檔 `--merge` 換旗標即可。

### 4.4 提交改用 fused_opp（加入對手落子的 L1b 特徵；2026-10-27，issue #12）

PR #11（nano4）發現對手的落子是第二份證據：線上對局配對實力相近，所以用同一組 L1b 模型評**對手**的落子、再做 16 維題目特徵，接在 fused 的 40 維之後成 56 維。CV 0.6735（fused 0.6545）、fold 0 0.6716（fused 0.6535，+0.0181 [+0.0094, +0.0266]）。使用者決定提交改用 fused_opp。權重檔不變（`heads3_{A,B}.pt` 加 trunk），多的是對手的評分目錄：A 是 `results/l1b/scores3_opp`（folds 3、4、0），B 是 `results/l1b/scores3_B_opp`（folds 1、2、0），都已在 nano4。五步和 §4.3 相同、順序固定：

**1. 擬合 fused_opp MLP**（CPU；訓練完全照 `l1b_opp_fold0_check.py`：`--opp` 給每個模型的對手評分目錄，特徵由 `l1b_opp_check.features` 算，fold 的目標與對手特徵來自同一個模型）。順便在 fold 0 重算七組（`hsl`、`fused_A/B/avg`、`fused_opp_A/B/avg`），`--reference` 對 `opp_fold0_check.json`（差應全為 0：fused_opp_A 0.6684、fused_opp_B 0.6684、fused_opp_avg 0.6716），並存 fold 0 每題七組機率給第 4 步：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J fitopp scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json B:results/l1b/scores3_B:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp B:results/l1b/scores3_B_opp --out results/models/rank_model_fused_opp.json --fold0-proba results/l1b/fold0_fused_opp_proba.csv --reference results/l1b/opp_fold0_check.json
```

`rank_model_fused_opp.json` 與 `fold0_fused_opp_proba.csv` 進 git。文件的 `opponent: true`、56 個 features；`l1b_models` 同 §4.3（配對檢查不變）；`fused_state` 是同一批題目上 40 維 fused MLP 的參數，當參考。

**2. mock test 分片評分加對手**：`--l1b-heads` 現在**預設**多跑一趟對手的落子（同一個 CPU pool，任務換成另一色，只過 K 個 L1b trunk，不過 18 個 HSL profile；`--no-l1b-opponent` 關掉）。分片檔多 `l1b_opp_sums [n, K, 10]`、`l1b_opp_n_moves [n]`（對手手數，特徵用它），meta 多 `l1b_opponent`、`opp_positions`、`opp_scoring_wall_s`；既有欄位不變。單卡一次（評分 + merge 同工作）、4 卡一次：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J mocko1 scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_opp/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused_opp.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:4 -c 48 -t 02:00:00 -J mocko4 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_opp4/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J mocko4m scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 4 --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_opp4/submission_rank.csv --fused-model results/models/rank_model_fused_opp.json
```

預期：兩次 exit 0，報告 `decision_model: fused_opp`、`fused_fallback_questions: []`、`l1b_opponent: true`、`opp_positions` 與 `positions` 相差不多（對手手數至多差一）。請貼單卡的評分秒數（`scoring_wall_s` 與 `opp_scoring_wall_s`）和 PR #10 的 250 s 比：GPU 多 2 次對手 trunk 前向（約 +10%），CPU 特徵約加倍，哪個是瓶頸就看這個數字。分片沒有對手分數卻給 `rank_model_fused_opp.json` → exit 2；給 `rank_model_fused.json` 則照舊可用。

**3. 每局 L1b 總和對帳，目標與對手兩邊**（CPU；對手那邊多一條：分片的 `l1b_opp_n_moves` 必須逐局等於表的 `n_moves`；目錄放錯邊會被 meta 的 `side` 拒絕）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J vecopp scripts/nano4/run.sbatch python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40_opp/submission_rank.csv --nshards 1 --vectors results/ladder/vectors/fold0.npz --l1b-scores results/l1b/scores3/fold0 results/l1b/scores3_B/fold0 --l1b-opp-scores results/l1b/scores3_opp/fold0 results/l1b/scores3_B_opp/fold0 --report results/mock_fold0_40_opp/game_vectors_check.json
```

預期：375/375、HSL PASS、`l1b.heads3_A/B` 與 `l1b_opp.heads3_A/B` 每手 ≤ 0.02、bias ≤ 1e-3、`0 games with other move counts`；目標那兩個目錄是舊 meta，照舊印 WARNING，對手那兩個是 PR #10 之後評的，有 `trunk_sha256` 和 `side`，不會。

**4. 機率與決策對帳**（`--set fused_opp_avg` 取第 1 步表裡的 fused_opp 平均；再比單卡 vs 4 卡）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J oppchk scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_opp/submission_rank.csv --offline results/l1b/fold0_fused_opp_proba.csv --set fused_opp_avg --answers tests/fixtures/mock_fold0_40/answers.csv --report results/mock_fold0_40_opp/fused_check.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J oppchk4 scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_opp/submission_rank.csv --other results/mock_fold0_40_opp4/submission_rank.csv --report results/mock_fold0_40_opp4/fused_check.json
```

預期：40/40 決策相同、max |Δp| ≤ 1e-3（對手特徵多兩組 bf16 雜訊來源，PR #10 的 3.9e-4 可能略增）。兩份 `fused_check.json` 與 `game_vectors_check.json` 進 git。

**5. 權重封存**（多帶新文件；`--overwrite` 蓋掉 §4.3 打的同名 tar，舊的 fused 版 nano4 備份在 `release/prev_fused/`）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J packopp scripts/nano4/run.sbatch python -I scripts/pack_weights.py --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --models results/models/rank_model.json results/models/rank_model_f01234.json results/models/rank_model_fused.json results/models/rank_model_fused_opp.json --set-dir results/models/set --out release/weights-2026-11.tar --overwrite
```

**11/04 正式流程**：8 卡 `shards.sh 8 … --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt`（§4 的 `sub8` 指令加這個旗標；對手自動評，`shards.sh` 限每分片 2 條 CPU 執行緒）→ `--merge --nshards 8 --fused-model results/models/rank_model_fused_opp.json`。退路依序：同一組分片檔 `--merge` 改給 `--fused-model results/models/rank_model_fused.json`（fused）、`--set-models …`（`ens_top3`）或什麼都不給（`rank_model.json`），都不用重新評分。

### 4.5 棋力標量對照實驗 a）：有序最後一層（2026-10-28，issue #14）

提交管線不動。`scripts/ordinal_check.py` 在既有 56 維 fused_opp 特徵上比三個最後一層：`mlp`（現在的 10 類 MLP，必須重現 `opp_check.json` 的 0.6735）、`coral`（CORAL 有序 MLP：共用標量加 9 個遞減門檻）、`bt`（題目層的成對 Bradley–Terry 強度分數，再用累積 logit 鏈結給機率）。協定同 `l1b_opp_check.py`（folds 1–4、pilot 驗證題、每人 32 題、`RandomState(42+v)`），預先指定 `coral vs mlp`、`bt vs mlp` 三條件。CPU，幾分鐘：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J ordchk scripts/nano4/run.sbatch python -I scripts/ordinal_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/l1b/ordinal_check.json
```

預期：`mlp` 0.6735（差 0）；把三列數字與兩條比較貼回 issue #14，`ordinal_check.json` 進 git。不碰 fold 0。決策：任一有序版本三條件全過才另開 issue 討論換最後一層；都沒過就記進 METHOD_NOTES，b）照做。

**結果（2026-10-29，job 638248，3 分 56 秒）**：`mlp` 0.6735（差 0）、`coral` 0.6472、`bt` 0.6761；coral vs mlp −0.0263 [−0.0309, −0.0217]、bt vs mlp +0.0025 [−0.0012, +0.0063]（四折全正但 CI 含 0、未達 0.005），都沒過，最後一層維持 10 類 MLP（`results/l1b/ordinal_check.json`，METHOD_NOTES §2.12）。

### 4.6 棋力標量對照實驗 b）：humanv0 trunk 上的 BT 強度 head（2026-10-28，issue #14）

a）的結果不影響 b）要不要跑（b 測的是特徵，不是最後一層）。提交管線不動。強度 head 是 humanv0 `PolicyHead` 的一份拷貝，通道 0 在落子處的輸出當每手的棋力分數 s（第 0 步就是中性 metadata 下的 policy logit）；訓練目標是 Bradley–Terry：同一批次內等級不同的位置對，−log σ(sign(r_i − r_j)·(s_i − s_j))，權重 √(w_i w_j)，inner 用同一個損失選模，另印成對準確率。沿用 L1b 的快取與 A／B out-of-fold。四步：

**1. 訓練 A、B 兩個強度 head**（第一階段，凍結 trunk；`cache_A` 是 folds 1、2，`cache_B` 是 3、4。單卡用預設 batch 256；8 卡 torchrun 每卡 128。log 看 `inner BT loss` 與 `pair accuracy` 從第 0 步到最佳的變化）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J seA scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/se_train.py --cache results/l1b/cache_A --out results/l1b/se_A.pt --batch 128
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J seB scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/se_train.py --cache results/l1b/cache_B --out results/l1b/se_B.pt --batch 128
```

側檔 `se_{A,B}.json` 進 git（`kind: strength`，記 `inner_bt_init/best`、`inner_pair_accuracy_*`、`inner_mean_by_rank_best`：十個等級的平均分數應該大致遞增）；`.pt` 不進 git。

**2. 評分**（`se_score.py`，`l1b_score.py` 的佈局與 out-of-fold 檢查；A 評 folds 3、4、0，B 評 1、2、0；目標與對手各一次）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J seScA scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_A.pt --folds 3 4 0 --out results/l1b/se_A_scores --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J seScAo scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_A.pt --folds 3 4 0 --out results/l1b/se_A_opp --opponent --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J seScB scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_B.pt --folds 1 2 0 --out results/l1b/se_B_scores --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J seScBo scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_B.pt --folds 1 2 0 --out results/l1b/se_B_opp --opponent --workers 10
```

每局一列：`n_moves` 與 `se_mean`、`se_std`、`se_q10`、`se_q50`、`se_q90`、三段均值（t < 40、40–99、≥ 100）；每手 s 在 npz（不進 git，`games_s*_p*.csv` 也不進，`meta_s*.json` 進）。

**3. CV**（CPU，幾分鐘）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J sechk scripts/nano4/run.sbatch python -I scripts/se_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --se results/l1b/se_A_scores results/l1b/se_B_scores --se-opp results/l1b/se_A_opp results/l1b/se_B_opp --out results/l1b/se_check.json
```

預期：`fused_opp` 0.6735（重現）；`se` 單獨、`hsl_se`、`fused_opp_se`（+ 目標 9 維）、`fused_opp_se2`（+ 對手 9 維）；預先指定 `fused_opp_se vs fused_opp`、`fused_opp_se2 vs fused_opp` 三條件。六列與五條比較貼回 issue #14，`se_check.json` 進 git。**不碰 fold 0**：通過才在 issue 上談第三次 fold 0 檢查。

**4. 決策**：預先指定的比較通過 → 另開 issue 談接進提交（分片評分多一次強度 head 前向、分片檔多每局統計、fused 文件多 9 或 18 維）；沒過 → 記進 METHOD_NOTES §2.12，結束。

**結果（2026-10-29）**：se_A／se_B 成對準確率 0.52 → 0.74／0.75、十個等級的平均分數嚴格遞增，各約 25 分鐘（`se_{A,B}.json`）；四個評分目錄每 fold 約 8 分鐘；`se_check.py`（job 638316）：`fused_opp` 0.6735（差 0）、`fused_opp_se` 0.6694、`fused_opp_se2` 0.6720，兩條預先指定都沒過（−0.0041 [−0.0073, −0.0009]、−0.0015 [−0.0050, +0.0019]），強度特徵不接進提交、不碰 fold 0（`results/l1b/se_check.json`，METHOD_NOTES §2.12）。

### 4.7 L1b 五折交叉擬合（每個模型看 80% 的玩家）+ 相鄰級 label smoothing（2026-10-29，issue #16）

提交管線不動。五個鑑別式 L1b 模型 f0…f4，f_k 用 folds {0,1,2,3,4} ∖ {k} 訓練、只評 fold k；配方與 heads3 完全相同（第一階段凍結 trunk → 第二階段 `--trunk-lr 1e-5` → 鑑別式 `--disc-weight 1`），只改資料。`l1b_train.py` 現在可以吃多個 `--cache` 並以 `--exclude-folds k` 丟掉 fold k 玩家的局（`splits/player_folds.csv`），側檔 `train_folds` = 三包快取的聯集 − {k}，所以 `l1b_score.py` 的 OOF 檢查只准 f_k 評 fold k。五條鏈互相獨立，可以同時排。

**0. fold 0 的快取**（CPU；和 `cache_A`／`cache_B` 同一支腳本，`--max-games 40` 預設。`run.sbatch` 預設 1 張 GPU 配 `-c 48` 會撞每張 GPU 12 核的上限，所以照 cache_A 當時的做法用 8 卡節點分 8 片、每片 12 個 worker，最後合併 meta）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J cache0 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_build_cache.py --folds 0 --out results/l1b/cache_0 --workers 12
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 00:10:00 -J cache0m scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --out results/l1b/cache_0 --merge-meta
```

預期（nano4 2026-10-29）：1,605 人、5,701,594 個位置。

**1. 五條訓練鏈**（k = 0…4；下面以 k = 0 為例，其餘把 `f0`／`--exclude-folds 0` 換掉）。第一階段用 8 卡 `--batch 32`（全域 batch 256 = `heads_A` 的；使用者決定，2026-10-29），第二、三階段用 8 卡 `--batch 128`（全域 1,024 = `heads2`／`heads3` 的）；epoch 數不改，讓 inner 早停決定。五條鏈同時排（`afterok` 串三段與評分）；每人同時最多送 20 個工作（`QOSMaxSubmitJobPerUserLimit`），整條流程 29 個工作，超過的由 login node 上只做 `squeue`／`sbatch` 的小迴圈等空位補送：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J x5s1f0 scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0 --exclude-folds 0 --out results/l1b/heads5s1_f0.pt --batch 32
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J x5s2f0 scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0 --exclude-folds 0 --out results/l1b/heads5s2_f0.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s1_f0.pt --batch 128
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J x5f0 scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0 --exclude-folds 0 --out results/l1b/heads5_f0.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s2_f0.pt --disc-weight 1 --epochs 2 --batch 128
```

每個 rank 各自把三包快取讀進 RAM（約 2,260 萬個位置 × 1.1 KB ≈ 25 GB；8 個 rank 約 200 GB／節點；8gpus 節點 1.9 TB、`DefMemPerGPU=200G`，夠）。五條鏈同時起跑時 40 個 rank 一起從共享檔案系統讀快取，單一 rank 讀完可能要 6–9 分鐘，先到的 rank 會在第一次集體通訊等人：NCCL 預設 10 分鐘逾時會讓 all-reduce 回傳壞資料（印出 `inner NLL nan` 後中止），所以 `l1b_train.py` 的 `init_process_group` 現在用 60 分鐘逾時（nano4 `95d4382`）。log 第一行會印「caches […]: folds [1, 2, 3, 4] after leaving out [0] (N players)」，確認 `train_folds`；第 0 步的 inner NLL 應在 1.61–1.63（heads_A 1.6164）。側檔 `heads5*_f{k}.json`（15 個）進 git，`.pt` 不進；要貼的是五個 `heads5_f{k}.json` 的 `inner_nll_init/best`、`inner_disc_init/best`、`best_epoch`、`wall_seconds`，和 heads3_A／B（NLL 1.434 → 1.516／1.419 → 1.493，rank CE 見側檔）擺在一起。

**2. 評分**（f_k 只評 fold k，目標與對手各一次，五個模型寫進同一組目錄；OOF 檢查會拒絕評到自己訓練的 fold）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J x5sc0 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f0.pt --folds 0 --out results/l1b/scores5 --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J x5so0 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f0.pt --folds 0 --out results/l1b/scores5_opp --opponent --workers 10
```

（k = 1…4 同樣，共 10 個工作。）每個 fold 的 `meta_s*.json` 進 git（csv／npz 不進）。

**3. CV**（CPU，十幾分鐘）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J xfit scripts/nano4/run.sbatch python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json
```

預期：`fused_opp` 0.6735（重現 `opp_check.json`，差 0）；`fused_opp_x5`、`fused_opp_ls`（ε 0.1）、`fused_opp_ls2`（0.2）、`fused_opp_x5_ls`。預先指定 `fused_opp_x5 vs fused_opp`、`fused_opp_ls vs fused_opp` 三條件；其餘只報告。五列與五條比較貼回 issue #16，`xfit_check.json` 進 git。**不碰 fold 0。**

**4. 決策**：`x5` 通過 → 另開 issue 接進提交：`fit_fused_model.py --l1b f0:results/l1b/scores5:results/l1b/heads5_f0.json … f4:… --opp f0:results/l1b/scores5_opp … --out results/models/rank_model_fused_opp_x5.json`（fold 0 表只用 f0，腳本會印「models [f1…f4] trained on it and are left out」；**5 個模型平均在任何留出資料上都沒有被量過**），分片評分 `--l1b-heads` 給 5 個 `.pt`（多 6 次 trunk 前向），對帳與封存同 §4.4；`ls` 通過 → 同一個 issue 一起談（只是 MLP 的 loss）。都沒過 → 記 METHOD_NOTES §2.13，結束。

**結果（2026-10-30，分支 `db3199f`）**：五條鏈各約 4.5–5.3 小時（第一段 6,400–7,900 s、第二段 6,100–6,400 s、第三段 3,662–5,029 s），inner rank CE 2.100–2.110（heads3 2.137／2.111）；評分每 fold 一側約 8 分鐘；`xfit_check.py`（job 640475）：`fused_opp` 0.6735（差 0）、`fused_opp_x5` 0.6704、`fused_opp_ls` 0.6739、`fused_opp_ls2` 0.6755、`fused_opp_x5_ls` 0.6705；預先指定 x5 vs fused_opp −0.0032 [−0.0071, +0.0007]、ls vs fused_opp +0.0003 [−0.0019, +0.0025]，都沒過，提交維持 A／B 的 fused_opp（`results/l1b/xfit_check.json`，METHOD_NOTES §2.13）。

### 4.8 刻度校正變體：每個 L1b 模型每個 head 的偏移置中（2026-10-30，issue #16 後續）

使用者決定（2026-10-30）跑這個 CPU 變體檢驗 §4.7 結果的刻度假設。`xfit_check.py` 現在多算 `fused_opp_c`、`fused_opp_x5c`：對每個模型（fold 目錄的 `meta_s*.json` 認模型）、每個 head，用它在 folds 1–4 評過的局算每手平均 log p（目標與對手各一組），每局 `sum_<rank>` 減掉 `n_moves ×` 這個偏移後再做 56 維；開頭會印每個模型的偏移表。預先指定 `fused_opp_x5c vs fused_opp`；只報告 `c vs fused_opp`、`x5c vs c` 與第一次跑的五條。同一支指令、換輸出檔名，不碰 fold 0：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J xfitc scripts/nano4/run.sbatch python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_center_check.json
```

預期：`fused_opp` 0.6735、`fused_opp_x5` 0.6704 等五列與 §4.7 完全相同（校正不碰它們）；偏移表（兩側各七個模型 × 10 個 head：A、B 與 f0…f4，fold 0 不在 CV 裡所以 f0 沒有列）、`fused_opp_c`、`fused_opp_x5c` 與八條比較貼回 issue #16，`xfit_center_check.json` 進 git。決策：`x5c` 過三條件 → 另開 issue 談接進提交（五個模型評分 + 校正步驗，分片評分 8 卡約 1 分 48 秒 → 2 分半，11/04 前時間緊，接不接由使用者決定）；沒過 → 記 METHOD_NOTES §2.14，刻度假設不成立。

**結果（2026-10-30，`results/l1b/xfit_center_check.json`）**：五列與 §4.7 逐位元相同；偏移在模型間差到 0.3 nats/手，但 `fused_opp_x5c` 0.6704（= `fused_opp_x5`）、`fused_opp_c` 0.6683；預先指定 x5c vs fused_opp −0.0031 [−0.0070, +0.0007] 沒過，c vs fused_opp −0.0052 [−0.0087, −0.0016] 反而變差。刻度假設不成立，這條線結束（METHOD_NOTES §2.14）。

### 4.9 落子序列實驗：每手 L1b log p 的「順序」有沒有加總看不到的訊號（2026-10-30，issue #19）

提交管線不動。今天所有特徵都把一局每手的 log p 加總（`sum_<rank>`），手與手的順序、局內的起伏、與對手落子的交錯全丟掉；使用者（2026-10-30）問 CTM 為什麼不能一步一步學——§2.7 的 CTM 吃的是整局的向量、tick 是「想」的步數，從沒看過逐手序列。這個實驗把逐手序列餵給四個 encoder（`aicup.seqmodel`：sum 錨點、1-D conv、GRU、tick = 落子的 CTM），輸出都是 `base + Δ`（base 是今天的 L1b 加總，Δ 零初始化，所以 sum encoder 就是今天的特徵），以 folds 1–4 四折交叉擬合寫出 `l1b_score.py` 版面的每局表，再用 `l1b_opp_check` 的協定比 `fused_opp`。每步 31 維：目標手的 18 個 Human SL log p + 10 個 L1b log p、對手手的 10 個 L1b log p（Human SL 欄為 0）、兩個旗標、t/300（`aicup.seqdata`）。只需要現有評分檔的每手 npz（`results/scores`、`scores3`／`scores3_B`、`scores3_opp`／`scores3_B_opp`），不碰 fold 0，不用 trunk、不用快取。設計與決策規則在 METHOD_NOTES §2.15。

**0. 診斷（CPU，幾分鐘；可以先跑）**：每局把每手 L1b 的後驗平均等級與 surprisal 做成兩條標量序列，算 std、lag-1／lag-2 自相關、後半減前半、斜率等「形狀」特徵，按等級印平均，並做每局的 softmax 回歸：只用平均（`mean_pmi`、`log_moves`）vs 平均＋形狀，三折訓一折測，差就是加總看不到的訊號有多大。

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 00:30:00 -J seqdiag scripts/nano4/run.sbatch python -I scripts/seq_diag.py --l1b results/l1b/scores3 results/l1b/scores3_B --out results/seq/seq_diag.json
```

貼回 issue #19：每等級的平均表、四折 mean-only vs mean+shape 的平衡準確率／NLL 與 `shape gain` 那一行；`seq_diag.json` 進 git。

**1. 四個 encoder 各一支工作（1 張 GPU，互相獨立，可同時排）**。每支先讀四個 fold 的序列（每折約 29,500 局、約 656 萬步，RAM 約 8 GB，幾分鐘），再對 k = 1…4 各訓一個模型（訓練集三折約 88,500 局、約 2,000 萬步）：

```bash
for a in sum conv gru ctm_seq; do
  sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 08:00:00 -J seq_$a scripts/nano4/run.sbatch python -I scripts/seq_crossfit.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --arch $a --out results/seq/$a
done
```

預設 `--epochs 10 --batch 256 --lr 1e-3 --hidden 64`（CTM 另有 `--d-latent 128 --history 8 --n-sync 256`），inner 早停（每等級 10% 的玩家留出、每 epoch 評 4 次、8 次沒進步就停）。sum 幾分鐘（只算標準化與評估）；conv／GRU 每 epoch 約 1 分鐘；CTM 每 tick 一圈 Python 迴圈（每局 200–500 步），每 epoch 約 5–10 分鐘，10 epoch 內結束，8 小時夠。輸出 `results/seq/<arch>/fold{k}/games_s0_p000.csv`（`sum_<rank>` = base + Δ）、`moves_s0_p000.npz`（base、delta）、`meta_s0.json`、`encoder_f{k}.pt`、`summary.json`。`sum` 的表必須與 `scores3`／`scores3_B` 的 `sum_<rank>` 只差 float32 捨入（`mean_abs_delta` 0）。CTM 可以用 `--only-folds k` 拆成每折一支工作（4 張 GPU 同時跑，訓練集仍由 `--folds` 決定，結果定義不變；`summary.json` 從各折的 `meta_s0.json` 重建，最後結束的那支寫齊四折）。進 git 的只有每個 arch 的 `summary.json` 與四個 `meta_s0.json`（`.gitignore` 已排除 .pt／.npz／每局 csv）。貼回：每個 arch 每折的 inner CE init → best、best_epoch、mean |Δ|、wall。

**2. CV（CPU；四個 arch 都寫完後。實測 25 分鐘，慢的節點一折就要 48 分鐘，所以時限給 8 小時）**：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 08:00:00 -J seqcheck scripts/nano4/run.sbatch python -I scripts/seq_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --seq sum:results/seq/sum conv:results/seq/conv gru:results/seq/gru ctm_seq:results/seq/ctm_seq --out results/seq/seq_check.json
```

預期：`fused_opp` 0.6735（差 0）；`fused_opp_seq_sum` 與它只差 float32 捨入（管線檢查，應在 ±0.001 內）。**預先指定** `fused_opp_seq_gru vs fused_opp`（條件 (1)(3)(4)）；只報告 conv、ctm_seq vs fused_opp 與 `ctm_seq vs gru`。五列、四條比較貼回 issue #19，`seq_check.json` 進 git。

**3. 決策**：GRU 過 → 序列裡有加總看不到的訊號，另開 issue 談接進提交（`predict_submission.py` 的分片已存每手的 L1b 陣列與對手陣列，只要加 encoder 這一步），這時才看 `ctm_seq vs gru` 決定用哪個 encoder；GRU 沒過、診斷的 shape gain 也 ≈ 0 → 在每手 log p 這個層級順序沒有可用訊號，CTM 逐手也不會有，記 METHOD_NOTES §2.15 結束；GRU 沒過但 ctm_seq 過（沒有預先指定）→ 當線索，`--seed 1` 重跑 ctm_seq 與 gru 再說。

**結果（nano4，2026-10-31，`results/seq/seq_check.json`；PR #18 合併為 `main` `914e4e2`）**：`fused_opp` 0.6735 重現；`fused_opp_seq_sum` 0.6730（−0.0005，float32 捨入內）、conv 0.6719、**gru 0.6732（預先指定 −0.0004 [−0.0029, +0.0022]，三條件全不過）**、ctm_seq 0.6705（vs gru −0.0026 [−0.0050, −0.0003]，只報告）。第 0 步的 shape gain 不是 0（每局準確率 +0.0085、NLL +0.0328，四折同向），所以結論是「形狀訊號存在但 base + Δ 這條通道沒有把它變成題目層級的分數」，不是「順序沒有訊號」。實測資源：conv 每折 20 s、gru 2–3 分、ctm_seq 11–28 分（1 卡），`seq_check` 25–48 分。GRU 在 GPU 上需要 `517a5a4` 的修正（inner 評估後回到 train 模式）。這條線結束，提交維持 A／B 的 fused_opp；後續若做獨立通道的變體，見 METHOD_NOTES §2.15 的結論。

### 4.10 獨立通道 fused_opp_shape：每題局內形狀特徵的平均（2026-10-31，issue #21）

提交管線不動。§4.9 的 base + Δ 沒過，但第 0 步的形狀訊號存在；這裡把每局的 9 個形狀特徵（`aicup.seqshape.game_shape`，就是 `seq_diag.py` 算的那些）在每題的局上取平均、去掉 log_moves 後接在 56 維 `fused_opp` 後面：`fused_opp_shape`（+8，目標手）、`fused_opp_shape2`（+16，再加對手手）。CV 協定與 `l1b_opp_check` 相同，只多算形狀表（讀 `scores3`／`scores3_B`、`scores3_opp`／`scores3_B_opp` 的每手 npz，每折兩側約 1 分鐘）。CPU 一支工作；三個特徵集 × 四折的 MLP 擬合，依 §4.9 `seq_check` 的經驗（五個集合 25–48 分）估 15–40 分，時限給 4 小時：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 04:00:00 -J shapecheck scripts/nano4/run.sbatch python -I scripts/shape_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/seq/shape_check.json
```

預期：`fused_opp` 0.6735（差 0；與 `seq_check.json`、`opp_check.json` 的 `fused_opp` 列相同）。**預先指定** `fused_opp_shape vs fused_opp`（條件 (1)(3)(4)）；只報告 `shape2 vs fused_opp`、`shape2 vs shape`。三列（總分、NLL、by n、每折）與三條比較貼回 issue #21，`shape_check.json` 進 git。決策：過 → 另開 issue 接進提交（分片已有每手 L1b 陣列，形狀特徵在分片後算）；沒過 → METHOD_NOTES §2.16，局內順序這條線結束。

**結果（nano4，2026-10-31，`results/seq/shape_check.json`，commit `a02f1b7`；job 662599，實跑約 4.5 分）**：`fused_opp` 0.6735 重現（每折同 `seq_check.json`）；**`fused_opp_shape` 0.6706（預先指定 −0.0029 [−0.0064, +0.0004]，三條件全不過）**、`fused_opp_shape2` 0.6742（+0.0006，不顯著）；兩者 NLL 都變差（1.2007、1.2037 對 1.1790）。局內順序這條線結束，提交維持 fused_opp（METHOD_NOTES §2.16）。

### 4.11 80 分路線：測試玩家是否在訓練集裡（2026-10-31，issue #22；程式在 AI_CUP_2026-2，從這裡的根目錄送件）

提交管線不動。設計與判準在 METHOD_NOTES §2.17 與 AI_CUP_2026-2 的 METHOD_NOTES §2.3、docs/NANO4.md §4。AI_CUP_2026-2 的 checkout（nano4 上是 `~/BK_AICUP_2026-2`）要在 `claude/pid-rank-overlap`（PR #4）或合併後的 `main`。

**步 0：SGF 標頭確認**（登入節點一行即可）：

```bash
for f in data/training/train_*.csv; do echo "$f RE=$(grep -c 'RE\[' "$f") HA=$(grep -c 'HA\[' "$f") KM=$(grep -c 'KM\[' "$f") AB=$(grep -c 'AB\[' "$f") DT=$(grep -c 'DT\[' "$f")"; done
```

全為 0 → 元資料這條關閉（預期如此，mock 測試檔每局只有 `SZ[19]`）。

**步 1：8,007 路辨識可行性**（CPU 一支；`run.sbatch` 預設配 1 張 GPU、每張卡最多 12 核，所以 `-c 12 --workers 12`；nano4 實測 12 核 627 s、RAM 數 GB）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 02:00:00 -J rankovl scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_rank_overlap.py --scores results/scores --calib-fold 1 --eval-fold 0 --fused-proba results/l1b/fold0_fused_opp_proba.csv --siblings --workers 12 --out ../BK_AICUP_2026-2/results/rank_overlap.json
```

貼回 issue #22：校準折（fold 1）的 Top-1／Top-5 按 n 與 τ；評估折（fold 0）的 Top-1／Top-5 按 n、`id_rank_score`、margin ≥ τ 的比例與精確率、認對／認錯的 margin 分位數；`combined` vs `fused_opp`（差、CI；`fused_opp` 須重現 0.6716）；`siblings` 每個 n 的 `top1_same_player`。`rank_overlap.json` 進 AI_CUP_2026-2 的 git。

**步 2（11/04）**：官方檔案到手後先跑 `--test-csv`（指令在 AI_CUP_2026-2 docs/NANO4.md §4，τ 取 json 的 `calibration.tau`），貼回 `share_above_tau`（按 n）與 margin 分位數；判準見 METHOD_NOTES §2.17。

**結果（nano4，2026-10-31）**：步 0 全 0（元資料關閉）。步 1 以 fold 1 校準、fold 2 評估試跑（fold 0 沒動）：Top-1 0.425、id_rank 0.6513、margin ≥ τ 的題 15.4%（精確率 0.974）；手算上限約 0.74。**使用者決定暫停這條路線**（`REPORT.md` §2.7），步 2 不跑、fold 0 不評，等主辦單位回覆（METHOD_NOTES §2.17）。

### 4.12 判錯的原因：OOF 殘差分解、對手通道、模型類別檢查（2026-10-31，issue #23）

提交管線不動，不碰 fold 0。CPU 一支：四折 × 四個模型（`mlp`、`mlp_diff`、`gbm`、`mlp_h128`），`gbm` 用 `requirements.txt` 既有的 xgboost 2.1.4（沒裝就加 `--no-gbm`）；依 §4.10 的經驗（三個集合 4.5 分）估 15–30 分，`mlp_h128` 與 300 棵樹較慢，時限給 4 小時：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 04:00:00 -J erran scripts/nano4/run.sbatch python -I scripts/error_analysis.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --jobs 12 --out results/l1b/error_analysis.json --proba-out results/l1b/oof_fused_opp_proba.csv
```

預期：`mlp` 0.6735（差 0；每折同 `opp_check.json`）。**預先指定** `mlp_diff vs mlp`、`gbm vs mlp`；只報告 `mlp_h128 vs mlp`。貼回 issue #23：四列（總分、NLL、by n、每折）、三條比較、殘差分析（玩家間比例、全對／全錯對獨立期望、按局數五分位、按等級、Spearman 表、三群平均）、對手通道的五個 Spearman。`error_analysis.json` 與 `oof_fused_opp_proba.csv`（約 39,000 列）進 git；fold 2 的那段機率給 issue #22 的 combined 用（欄名 `fused_opp_p_<rank>`）。

**結果（nano4，2026-10-31，`results/l1b/error_analysis.json`，PR #24；12 核實跑 81 分）**：`mlp` 0.6735 重現；**`mlp_diff` 0.6916（+0.0181 [+0.0140, +0.0220]，四折全正，預先指定過）**；`gbm` 0.6677（−0.0058）、`mlp_h128` 0.6686（−0.0050）都沒過。`gbm` 改用 xgboost 原生 API（`XGBClassifier` 需要 scikit-learn，nano4 的 venv 沒有）。殘差與對手通道的數字在 METHOD_NOTES §2.18。使用者決定接進提交（§4.13）。

### 4.13 提交改用 fused_opp_diff（2026-11-01，issue #25）

與 §4.3／§4.4 當年同一套流程。**步 1：fold 0 評一次 + 擬合提交用的文件**（CPU，`-c 12`；fused_opp 當時約 20 分鐘，這次多一個集合；時限 nano4 放寬到 4 小時：這次 fold 0 只評一次，不想因逾時重跑）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 04:00:00 -J fitdiff scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json B:results/l1b/scores3_B:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp B:results/l1b/scores3_B_opp --diff --out results/models/rank_model_fused_opp_diff.json --fold0-proba results/l1b/fold0_fused_opp_diff_proba.csv --reference results/l1b/opp_fold0_check.json
```

預期：`--reference` 對 hsl／fused_*／fused_opp_* 各列印 `+0.00e+00`（`fused_opp_avg` 0.6716 重現）；`fused_opp_diff_avg` 約 0.69，`fused_opp_diff_avg vs fused_opp_avg` 約 +0.018。貼回 issue #25：fold 0 的全部列（總分、NLL、by n）與那條比較；`rank_model_fused_opp_diff.json`、`fold0_fused_opp_diff_proba.csv` 進 git（結果檔走新分支 + PR）。

**結果（nano4，2026-11-01，PR #27 `75a66c6`；job 665469，約 24 分）**：`--reference` 全部 `+0.00e+00`、`fused_opp_avg` 0.6716 重現；文件 `mlp_h32_fused_opp_diff`／`fused_opp_diff`／59 特徵／203,328 題。**`fused_opp_diff_avg` 0.6888**（NLL 1.1177；n=5／10／20 0.6392／0.6991／0.7280；A 0.6860、B 0.6817），`fused_opp_diff_avg vs fused_opp_avg` **+0.0172 [+0.0098, +0.0245]**。採用檢查通過（METHOD_NOTES §2.19）。

**步 1b：mock 40 題驗提交管線**（與 §4.3／§4.4 採用時相同；nano4 建議，2026-11-01）。文件擬合好後，用同一組 mock 題跑一次提交管線，再與離線的 fold 0 機率比對；`decision_model` 應為 `fused_opp_diff`，`fused_fallback_questions` 為空，`check_fused_submission.py` 的機率差應在浮點誤差內（和 `fused_opp` 當時一樣）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J mockd1 scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_diff/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused_opp_diff.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 01:00:00 -J diffchk scripts/nano4/run.sbatch python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_diff/submission_rank.csv --offline results/l1b/fold0_fused_opp_diff_proba.csv --set fused_opp_diff_avg --answers tests/fixtures/mock_fold0_40/answers.csv --report results/mock_fold0_40_diff/fused_check.json
```

貼回 `fused_check.json` 的摘要（最大機率差、40 題的分數）；`fused_check.json` 與 `submission_rank_report.json` 進 git。

**結果（nano4，2026-11-01，jobs 665537／666475，`results/mock_fold0_40_diff/`）**：`decision_model` `fused_opp_diff`、無退路題、無 problems、375 局 0 丟棄；40 題 max |Δp| 5.12e-4（容差 1e-3）、0 題決策不同，PASS。注意：`heads3_{A,B}.pt: trained by other training code` 是 `rankmodel.py`（`0be8e82`）被算進 `code_sha256` 所致，heads 沒變、無害，11/04 會再出現；這次評分 1785 s（10/28 為 248 s，節點上有其他 GPU 工作），11/04 排程留餘裕；`submission_rank_report.json` 要 `git add -f`。

**步 2（11/04，官方檔案到手後）**：§4.4 的分片／合併指令不變，合併時 `--fused-model results/models/rank_model_fused_opp_diff.json`；報告的 `decision_model` 應為 `fused_opp_diff`、`fused_fallback_questions` 為空。若提交次數允許，另以 `rank_model_fused_opp.json` 合併一次出第二版，兩版都上 Public LB 對比。`pack_weights.py --models` 加上新文件。

### 4.14 每手分數的「分布範圍」：每局 pmi_t 直方圖的題內平均接在 fused_opp_diff 後面（2026-11-01，issue #29）

提交管線不動、不碰 fold 0；可與 §4.13 的 fold 0 檢查並行。§4.10 的作法換兩處：每局特徵換成 pmi_t 在 [0, 9] 等分 10 格的直方圖（`aicup.seqshape.game_pmi_hist`），基準換成 `fused_opp_diff`（59 維）。特徵集 `fused_opp`（56）、`fused_opp_diff`（59）、`fused_opp_diff_hist`（69，目標手）、`fused_opp_diff_hist2`（79，再加對手手）。CV 協定與 `l1b_opp_check`／`error_analysis` 相同；另算各等級的合併直方圖與 `fused_opp`／`fused_opp_diff` 的 OOF 殘差診斷（只報告）。CPU 一支工作：四個特徵集 × 四折的 MLP 加兩側直方圖表，約 §4.10 的 1.5 倍（§4.10 實跑 4.5 分），`-t 02:00:00` 足夠：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J histcheck scripts/nano4/run.sbatch python -I scripts/hist_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/seq/hist_check.json
```

預期：`fused_opp` 0.6735、`fused_opp_diff` 0.6916（每折同 `error_analysis.json` 的 `mlp`／`mlp_diff`）；`fused_opp` 殘差的 `spearman_oof_mean_e_vs_opp_minus_own` 重現 −0.213。**預先指定** `fused_opp_diff_hist vs fused_opp_diff`（條件 (1)(3)(4)）；只報告 `hist2 vs fused_opp_diff`、`hist2 vs hist`、`fused_opp_diff vs fused_opp`（+0.0181）。貼回 issue #29：四列（總分、NLL、by n、每折）、四條比較、各等級的合併直方圖與相鄰重疊係數、兩個模型的殘差 Spearman；`hist_check.json` 進 git（結果檔走新分支 + PR）。決策見 METHOD_NOTES §2.20：過 → 另開 issue 接進提交（分片後算直方圖，不需重新評分）；沒過 → 這把尺上的「分布範圍」結束。C 路線的判讀規則：`fused_opp_diff` 的 |Spearman(mean e, opp − own)| ≥ 0.10 → 開 C 的 issue。

**結果（nano4，2026-11-01，PR #31 `7b5726a`；實跑 297 s）**：`fused_opp` 0.6735、`fused_opp_diff` 0.6916 重現；**`fused_opp_diff_hist` 0.6861（預先指定 −0.0055 [−0.0085, −0.0025]，四折全負，三條件全不過）**、`fused_opp_diff_hist2` 0.6853（−0.0063）；相鄰等級重疊 0.858–0.937（1D|2D 0.937）；`fused_opp` 殘差對 (opp − own) −0.213 重現，`fused_opp_diff` −0.130。分布範圍這條線結束（METHOD_NOTES §2.20）。

**重跑（2026-11-01）**：第一次跑完後 `hist_check.py` 多了題內版的殘差診斷（`question_level`：殘差對模型看到的題內 opp／own／(opp − own) 的 Spearman，分每題、按 n、每人；METHOD_NOTES §2.20）。同一條指令重跑一次（約 7 分），四列與四條比較應逐位元相同，`hist_check.json` 以重跑的為準。

**重跑結果（nano4，2026-11-01，job 667579，342 s；PR #31 `b4e8df3`）**：四列、四條比較、各等級分布逐位元相同；`fused_opp_diff` 的殘差對題內 (opp − own)：每題 +0.011（n=5／10／20：+0.034／−0.014／−0.072）、每人 −0.123；`fused_opp` −0.099／−0.208。判讀見 METHOD_NOTES §2.20。

### 4.15 80 分路線 C：對手當老師的 L1b head（2026-11-01，issue #33；**先不跑**，等 11/04 Public LB 兩版對比）

提交管線不動、不碰 fold 0。與 §4.2／§4.7 同一套配方，只換資料：快取取**對手**落子前的局面、標籤是這位玩家的等級。側檔 `heads3c*_{A,B}.json` 進 git，`.pt` 與快取不進。

**1. 對手快取**（CPU 48 核，每包約 30 分；`--opponent` 讓 games 表的 `color` 是對手那一色、meta `side: opponent`）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J cacheAo scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A_opp --workers 48 --opponent
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J cacheBo scripts/nano4/run.sbatch python -I scripts/l1b_build_cache.py --folds 3 4 --out results/l1b/cache_B_opp --workers 48 --opponent
```

預期 `games` 與 `cache_A`／`cache_B` 相同（102,019／101,711），`positions` 接近（對手手數 ≈ 目標手數 ± 1）。

**2. 三階段訓練**（與 §4.7 的鏈同參數：第一階段 8 卡 `--batch 32`、第二、三階段 `--batch 128`；B 把 `cache_A_opp`／`_A` 換成 `cache_B_opp`／`_B`；三段用 `afterok` 串）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J c1A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A_opp --out results/l1b/heads3cs1_A.pt --batch 32
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J c2A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A_opp --out results/l1b/heads3cs2_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads3cs1_A.pt --batch 128
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J c3A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A_opp --out results/l1b/heads3c_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads3cs2_A.pt --disc-weight 1 --epochs 2 --batch 128
```

log 第一行的 `train_folds` 應為 [1, 2]（A）／[3, 4]（B），側檔 `side` 為 `opponent`。要貼的：三段的 `inner_nll_init/best`、`inner_disc_init/best`（第三段；heads3 為 2.137／2.111，ln 10 = 2.303）、`best_epoch`、`wall_seconds`。

**3. OOF 評分**（8 卡分片；`--opponent` **必帶**，否則 exit 2）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scAo scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3c_A.pt --folds 3 4 0 --out results/l1b/scores3c_A_opp --opponent --workers 10
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scBo scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3c_B.pt --folds 1 2 0 --out results/l1b/scores3c_B_opp --opponent --workers 10
```

**4. CV**（CPU 約 10 分；`--copp` 的 fold meta 須是 `side: opponent` 且 `heads.side: opponent`，否則拒絕）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J ohc scripts/nano4/run.sbatch python -I scripts/opp_head_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --copp results/l1b/scores3c_A_opp results/l1b/scores3c_B_opp --out results/l1b/opp_head_check.json
```

預期：`fused_opp_diff` 0.6916（每折同 `error_analysis.json` 的 `mlp_diff`）。**預先指定** `fused_opp_diff_c vs fused_opp_diff`（條件 (1)(3)(4)）；只報告 `c2 vs fused_opp_diff`、`fused_c_diff vs fused_opp_diff`、`c2 vs c`、通道強度。貼回 issue #33：四列、四條比較、通道強度那一行；`opp_head_check.json` 與側檔進 git（新分支 + PR）。決策見 METHOD_NOTES §2.21。

### 4.16 TSFEL 生成的每局時序特徵接在 fused_opp_diff 後面（2026-11-01，issue #36）

提交管線不動、不碰 fold 0。TSFEL **不裝進提交用的 `.venv`**（它會連帶拉進 scikit-learn、statsmodels、ipython 等十幾個套件，而 `.venv` 要在 11/04 跑正式評分；nano4 的決定，2026-11-01）：另開一個獨立環境（同一個 Python 3.12，numpy／pandas／scipy 用 `requirements.txt` 的版本），只多裝 `requirements-tsfel.txt`，下面的工作在那個環境裡跑；提交 `.venv` 的套件清單雜湊應前後相同。nano4 的做法（2026-11-01）：

```bash
# 一次：用提交 .venv 的同一個直譯器建獨立環境，只裝 tsfel 與其依賴
$(readlink -f .venv/bin/python) -m venv --without-pip ~/BK_AICUP_tsfel_venv
python -m pip install --no-compile --target ~/BK_AICUP_tsfel_venv/lib/python3.12/site-packages -r requirements-tsfel.txt
# 刪掉 target 裡 pip 重抓的 numpy / pandas / scipy，放兩個 .pth：zz_base_venv.pth 指向提交 .venv/lib/python3.12/site-packages
# （torch、numpy 2.5.4、pandas 3.0.6、scipy 1.18.1 從那裡來），katago.pth 複製自提交 .venv
# 切換：另開在 main 的 worktree ~/BK_AICUP_2026-1-tsfelwt，.venv 符號連結指到這個環境，data / logs / repo / models 連回主 checkout；
# run.sbatch 不用改，從 worktree 根目錄送件，--hsl / --l1b / --opp 給主 checkout 的絕對路徑
```

多出來的只有 tsfel、scikit-learn 1.9.1、statsmodels 0.15.0、PyWavelets 1.10.0、ipython 9.18 與其依賴；提交 `.venv` 的 site-packages 清單雜湊前後相同。

然後一支 CPU 工作：第一次會算每折的 TSFEL 表（約 118k 局 × 2 條序列 × 11 ms，`--workers 12` 約 5 分）並快取到 `results/seq/tsfel/fold<F>.npz`（不進 git），接著五個特徵集 × 四折的 MLP 與局層級的 softmax 回歸（約 §4.10 的 2 倍）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 03:00:00 -J tsfelchk scripts/nano4/run.sbatch python -I scripts/tsfel_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --workers 12 --out results/seq/tsfel_check.json
```

**結果（nano4，2026-11-01，PR #39 `1447777`；job 669660，612 s，每折 TSFEL 表約 35 s；pytest 獨立環境 325 passed／5 skipped）**：`fused_opp_diff` 0.6916 重現；**`fused_opp_diff_tsfel16` 0.6788（預先指定 −0.0128 [−0.0167, −0.0090]，四折全負，三條件全不過）**；tsfel4 −0.0014、tsfel64 −0.0375、tsfel_all −0.0610，退步隨維度單調放大；局層級增益 +0.0093。自動特徵庫這條線在這個接法下結束（METHOD_NOTES §2.22，含「模型對多餘欄位沒抵抗力 vs 共變量偏移」的兩個未預先指定的對照提案）。

預期：`fused_opp_diff` 0.6916（每折同 `error_analysis.json` 的 `mlp_diff`）。**預先指定** `fused_opp_diff_tsfel16 vs fused_opp_diff`（條件 (1)(3)(4)）；只報告 tsfel4／tsfel64／tsfel_all vs `fused_opp_diff`、局層級 softmax 的增益（mean-only vs mean + tsfel16）、每折保留欄數與 PCA 解釋變異。貼回 issue #36：五列（總分、NLL、by n、每折）、四條比較、`TSFEL:` 與 `game-level softmax:` 兩行；`tsfel_check.json` 進 git（新分支 + PR），`results/seq/tsfel/*.npz` 不進。決策見 METHOD_NOTES §2.22。

### 4.17 正則化的題目層模型與對多餘欄位的抵抗力（2026-11-01，issue #41）

提交管線不動、不碰 fold 0。CPU 一支，在提交 `.venv` 跑即可（TSFEL 塊只讀 §4.16 的 npz 快取，不需要 tsfel）；七個模型 × 三個特徵集 × 四折，約 §4.14 的 2–3 倍：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 03:00:00 -J qmodel scripts/nano4/run.sbatch python -I scripts/qmodel_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --tsfel-cache ~/BK_AICUP_2026-1-tsfelwt/results/seq/tsfel --out results/l1b/qmodel_check.json
```

`--tsfel-cache` 指向 §4.16 那個 worktree 裡的四個 `fold<F>.npz`；沒有就加 `--no-tsfel`（只跑純雜訊對照）。預期：`mlp@base` 0.6916（每折同 `error_analysis.json` 的 `mlp_diff`）。**預先指定** `mlp_l1@base vs mlp@base`（條件 (1)(3)(4)）；只報告其他模型 vs `mlp@base`、每個模型的 `@base_tsfel16 − @base` 與 `@base_noise16 − @base`。貼回 issue #41：各模型 × 特徵集的總分、NLL、每折、best epochs；預先指定那條；抵抗力的兩條；`qmodel_check.json` 進 git（新分支 + PR）。判讀規則與決策見 METHOD_NOTES §2.23。

**結果（nano4，2026-11-01，PR #43 `426d719`；job 669767，947 s；pytest 327 passed／3 skipped）**：`mlp@base` 0.6916 重現；**`mlp_l1@base` 0.6922（預先指定 +0.0006 [−0.0026, +0.0038]，不過）**；抵抗力：`mlp` +TSFEL-16 −0.0135、+雜訊-16 −0.0014（CI 含 0）→ 模型有抵抗力，三條特徵線的退步是塊自己的偏移；`mlp_es` −0.0043、`mlp_l2` −0.0189、`softmax` −0.0245、`mlp_pp64` −0.0007。這條線結束（METHOD_NOTES §2.23）。

### 4.18 資料集 v1.1 讓子資訊的普查（2026-11-01，issue #42）

v1.1 在 nano4 的 `data/dl_v1.1/training/`（`data/training/` 仍是 v1.0）。`scripts/hd_census.py` 只讀 CSV、不重播（CPU 幾分鐘）；除了 HD 分布，還算 HD≥2 的局「標籤玩家等級 by `color`」（決定 Q13 待確認的 `color` 語義）與第一手落在標準讓子點的比例：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 4 -t 01:00:00 -J hdcensus scripts/nano4/run.sbatch python -I scripts/hd_census.py --train-dir data/dl_v1.1/training --out results/data/hd_census.json
```

貼回 issue #42：`hd`、`handicap_games.labelled_rank_by_color`、`first_move_on_star_point_share`、`players_with_any_hd1_share`／`hd2_share`；`hd_census.json` 進 git。切換 `data/training` 到 v1.1 時：`games_sha256` 不含 `sgf_content`，快取 provenance 不受影響；REPORT.md §134「沒有讓子」那行要改。

### 4.19 讓子資訊：(a) HD 特徵的 CV、(b) 只重評 HD≥2 的局並併回（2026-11-01，issue #42）

前提：`data/training` 是 v1.1。`aicup.sgf.setup_and_moves` 讓每個重播點（評分、快取、提交、`check_game`、replay）對 HD≥2 的局自動補讓子石、B/W 對調；HD 缺／0／1 的局逐位不變，所以現有評分只有那 3.6% 的局（36,421 局）要重算。設計與決策規則在 METHOD_NOTES §2.24。

**步 1（CPU，幾分鐘）：HD 表與 HD≥2 的 key 清單**（兩個檔都不進 git）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 4 -t 01:00:00 -J hdtable scripts/nano4/run.sbatch python -I scripts/hd_table.py --out results/data/hd_table.csv --keys-out results/data/hd2_keys.csv
```

**步 2（CPU，約 1 小時；(a)，用現有分數、不重評）**（已跑：job 670322 步 1+2 共 50 分，`fused_opp_diff_hd` vs `fused_opp_diff` +0.0071 [+0.0034, +0.0107] **通過**，結果在 PR #46、METHOD_NOTES §2.24）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 8 -t 03:00:00 -J hdcheck scripts/nano4/run.sbatch python -I scripts/hd_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --hd-table results/data/hd_table.csv --out results/seq/hd_check.json
```

貼回 issue #42：`scores`（`fused_opp` 須 0.6735、`fused_opp_diff` 0.6916）、`comparisons`（預先指定 `fused_opp_diff_hd vs fused_opp_diff`）、`coverage`、`buckets`、`residual`；`hd_check.json` 進 git。

**步 3（GPU；(b)）：只重評 HD≥2 的局，進補丁目錄。**（已跑：2026-11-02 jobs 674256–674260，Human SL 1 小時 9 分、L1b 四支各 9 分；步 4 674851；步 5 674852／674854，結果見 §2.24） 五個評分目錄各跑一次，`--game-keys` 直接吃 `hd2_keys.csv`（有 `game_key` 欄）；fold 的選局不變，只評清單上的局，所以每折只有幾百到幾千局。A 的 heads（訓練 folds 1、2）評 0、3、4，B 的評 0、1、2（`check_unseen` 會擋錯的）：

```bash
# Human SL 18 profiles（原目錄 results/scores）
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 04:00:00 -J hdhsl scripts/nano4/run.sbatch python -I scripts/score_folds.py --folds 0 1 2 3 4 --game-keys results/data/hd2_keys.csv --out results/scores_hdpatch
# L1b 目標側與對手側，A 與 B
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 02:00:00 -J hdl1bA scripts/nano4/run.sbatch python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt --folds 0 3 4 --game-keys results/data/hd2_keys.csv --out results/l1b/scores3_hdpatch
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 02:00:00 -J hdl1bAo scripts/nano4/run.sbatch python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt --folds 0 3 4 --game-keys results/data/hd2_keys.csv --opponent --out results/l1b/scores3_opp_hdpatch
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 02:00:00 -J hdl1bB scripts/nano4/run.sbatch python -I scripts/l1b_score.py --heads results/l1b/heads3_B.pt --folds 0 1 2 --game-keys results/data/hd2_keys.csv --out results/l1b/scores3_B_hdpatch
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 02:00:00 -J hdl1bBo scripts/nano4/run.sbatch python -I scripts/l1b_score.py --heads results/l1b/heads3_B.pt --folds 0 1 2 --game-keys results/data/hd2_keys.csv --opponent --out results/l1b/scores3_B_opp_hdpatch
```

`--game-keys` 可以用在任何原本的評分命令上（`--shard`／`--nshards` 照舊）；補丁目錄的 part 從 0 編號、可中斷續跑。

**步 4（CPU，幾分鐘）：併回。** 每個目錄的 `fold{F}/patch_meta.json` 與 `--summary` 記錄換了幾局、手數前後、每個 `sum_*` 欄的平均 |Δ| 與平均 Δ（`sum` 的平均 Δ 預期為負——缺子棋盤上評的是對手的棋）：

```bash
python -I scripts/patch_scored.py --base results/scores --patch results/scores_hdpatch --out results/scores_hd --folds 0 1 2 3 4 --summary results/data/patch_scores_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3 --patch results/l1b/scores3_hdpatch --out results/l1b/scores3_hd --folds 0 3 4 --summary results/data/patch_scores3_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_opp --patch results/l1b/scores3_opp_hdpatch --out results/l1b/scores3_opp_hd --folds 0 3 4 --summary results/data/patch_scores3_opp_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_B --patch results/l1b/scores3_B_hdpatch --out results/l1b/scores3_B_hd --folds 0 1 2 --summary results/data/patch_scores3_B_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_B_opp --patch results/l1b/scores3_B_opp_hdpatch --out results/l1b/scores3_B_opp_hd --folds 0 1 2 --summary results/data/patch_scores3_B_opp_hd.json
```

**步 5（CPU）：修正後的 CV 與提交文件。** 用 `_hd` 目錄跑一次 `hist_check.py`（有 `fused_opp`／`fused_opp_diff` 兩組與殘差診斷；比較 `fused_opp_diff` 與修正前的 0.6916，下降不超過 0.002 視為不劣）和 `hd_check.py`（(a) 在修正後的分數上）；不劣就照 §4.13 的 `fitdiff` 命令、把五個目錄換成 `_hd`、`--out results/models/rank_model_fused_opp_diff_hd.json --fold0-proba results/l1b/fold0_fused_opp_diff_hd_proba.csv`，fold 0 的 `--reference` 對照照舊。11/04 用哪個文件由使用者決定；`predict_submission.py` 的報告有 `handicap`（每局 HD 值的計數），測試檔有沒有帶 `HD[` 一跑就知道。

### 4.20 讓子特徵的提交文件 `rank_model_fused_opp_diff_hd.json`（2026-11-01，issue #47；**fold 0 第五次使用（前四次 r3_pma、fused、fused_opp、fused_opp_diff），要使用者本人同意才跑**）

nano4 在完整資料上的全套 pytest：`f431660` 342 passed／3 skipped（job 670321）、`cf0008c` 344 passed／3 skipped（job 670747）。

前提：§4.19 步 1 的 `results/data/hd_table.csv` 已有；(b) 的 `_hd` 目錄 2026-11-02 已做好（步 3–5 完成，§2.24），**五個評分目錄都換成 `_hd` 版**（`results/scores_hd`、`results/l1b/scores3_hd`、`scores3_B_hd`、`scores3_opp_hd`、`scores3_B_opp_hd`）。

**先做（不碰 fold 0）：現行 59 維文件在 `_hd` 上重擬**，讓 11/04 推論端的新重播與訓練特徵一致（`--no-fold0`：文件記 `fold0_check.skipped`，沒有 fold-0 表，mock 檢查只能 run-vs-run）。**已做：job 676071，`results/models/rank_model_fused_opp_diff_r.json`（PR #50），帶 HD 的 mock-40 走過（job 676157，40／40、exit 0）；11/04 主提交用它，舊文件留作退路（§2.24）。**

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 02:00:00 -J fitdiffhd scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3_hd:results/l1b/heads3_A.json B:results/l1b/scores3_B_hd:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp_hd B:results/l1b/scores3_B_opp_hd --hsl results/scores_hd --diff --no-fold0 --out results/models/rank_model_fused_opp_diff_r.json
```

**使用者同意 fold 0 第五次使用後**：HD 版文件的擬合與 fold 0 檢查（CPU，約 10 分；下面的命令已換成 `_hd` 目錄）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 04:00:00 -J fithd scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3_hd:results/l1b/heads3_A.json B:results/l1b/scores3_B_hd:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp_hd B:results/l1b/scores3_B_opp_hd --hsl results/scores_hd --diff --hd-table results/data/hd_table.csv --out results/models/rank_model_fused_opp_diff_hd.json --fold0-proba results/l1b/fold0_fused_opp_diff_hd_proba.csv
```

貼回 issue #47：`fold 0 fused_opp_diff_hd_avg vs fused_opp_diff_avg` 的差與 95% CI（採用條件：下界 > 0）、`hsl`／`fused_opp_avg`／`fused_opp_diff_avg`（在 `_hd` 上與 §4.13 的舊重播數字不會相同，不加 `--reference`）；文件與 fold-0 表進 git。管線檢查用 mock-40（它的 SGF 沒有 `HD[`，所以要加 `--allow-missing-hd`，特徵全 0，只驗管線一致）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 02:00:00 -J mockhd scripts/nano4/run.sbatch python -I scripts/predict_submission.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --expect-rows 40 --expect-public 20 --expect-private 20 --out results/mock_fold0_40_hd/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused_opp_diff_hd.json --allow-missing-hd
```

mock-40 的 fold-0 表對照（`check_fused_submission.py --set fused_opp_diff_hd_avg`）**不會**逐題相等——離線表的 HD 特徵來自 HD 表、mock 的全 0——所以 mock 只看管線不崩、報告 `decision_model` 是 `fused_opp_diff_hd`、`handicap` 計數為 `absent`。真正的逐題對照要用帶 `HD[` 的 mock：nano4 已從 v1.1 的 `data/training` 做了 `data/mock_tests/fold0_40_v11/`（題目與舊 fixture 不同批，SGF 帶 HD；不進 git），`--inputs` 換成它的兩個 CSV、`--answers` 換成它的 `answers.csv`，再跑上面兩條（不加 `--allow-missing-hd`）。11/04：`predict_submission.py ... --fused-model results/models/rank_model_fused_opp_diff_hd.json`（報告 `handicap` 沒有 `absent` 才算成立；有 `absent` 會直接拒絕，換 `rank_model_fused_opp_diff.json`）。

### 4.21 R-full (a)：快取補丁 → heads4 → 全量重評 → 重擬（2026-11-02，issue #51；設計在 METHOD_NOTES §2.26）

前提：`results/data/hd2_keys.csv`（§4.19 步 1）、`results/scores_hd`（§4.19 步 4）都在。**步 1（CPU，幾分鐘；先 `--dry-run` 看數量）：就地補 `cache_A`／`cache_B` 裡 HD≥2 的局**。磁碟只多一個 part 的 tmp（約 25 MB）；中斷重跑會接著做；`patch_meta.json` 進 git（`results/l1b/cache_*/` 的 json 不在 .gitignore 的排除項裡，若被擋就 `-f`）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J patchA scripts/nano4/run.sbatch python -I scripts/l1b_patch_cache.py --cache results/l1b/cache_A --game-keys results/data/hd2_keys.csv --workers 48
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 48 -t 02:00:00 -J patchB scripts/nano4/run.sbatch python -I scripts/l1b_patch_cache.py --cache results/l1b/cache_B --game-keys results/data/hd2_keys.csv --workers 48
```

貼回 issue #51：每個快取的 `summary`（parts、games、positions before → after）與 log 裡每個 part 的「positions a -> b」；`meta.json` 的 `positions` 應等於 after 的總和。

**步 2（GPU）：heads4_A／heads4_B**，與 heads3 完全相同的三段配方（§4.7 x5 鏈的旗標），只換快取（A 在 `cache_A`、B 在 `cache_B`）。側檔 `heads4*_{A,B}.json` 進 git：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J h4s1A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads4s1_A.pt --batch 32
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J h4s2A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads4s2_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads4s1_A.pt --batch 128
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J h4A scripts/nano4/run.sbatch torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads4_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads4s2_A.pt --disc-weight 1 --epochs 2 --batch 128
```

（B：`--cache results/l1b/cache_B`、`heads4s1_B` → `heads4s2_B` → `heads4_B`。）貼回每段側檔的 `inner_nll_init/best`、`inner_disc_init/best`、`best_epoch`、`wall_seconds`，與 heads3 擺在一起。

**步 3（GPU，約 2.2 億局面，8 卡一節點 8–9 小時；可照 §4.4 用 `shards.sh 8` 分片）：heads4 全量重評兩側五折。** A 評 0 3 4、B 評 0 1 2（`check_unseen` 會擋錯的）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 10:00:00 -J s4A scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads4_A.pt --folds 0 3 4 --out results/l1b/scores4 --workers 12
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 10:00:00 -J s4Ao scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads4_A.pt --folds 0 3 4 --opponent --out results/l1b/scores4_opp --workers 12
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 10:00:00 -J s4B scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads4_B.pt --folds 0 1 2 --out results/l1b/scores4_B --workers 12
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -p 8gpus --gres=gpu:8 -c 96 -t 10:00:00 -J s4Bo scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads4_B.pt --folds 0 1 2 --opponent --out results/l1b/scores4_B_opp --workers 12
```

**步 4（CPU，各約 15 分）：重擬**，Human SL 用 `results/scores_hd`，都不碰 fold 0：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 02:00:00 -J fitfull scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores4:results/l1b/heads4_A.json B:results/l1b/scores4_B:results/l1b/heads4_B.json --opp A:results/l1b/scores4_opp B:results/l1b/scores4_B_opp --hsl results/scores_hd --diff --no-fold0 --out results/models/rank_model_fused_opp_diff_full.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 02:00:00 -J fithdfull scripts/nano4/run.sbatch python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores4:results/l1b/heads4_A.json B:results/l1b/scores4_B:results/l1b/heads4_B.json --opp A:results/l1b/scores4_opp B:results/l1b/scores4_B_opp --hsl results/scores_hd --diff --hd-table results/data/hd_table.csv --no-fold0 --out results/models/rank_model_fused_opp_diff_hd_full.json
```

**步 5（CPU，各約 1 小時）：E 四格的 full 版**，與 §2.24 的 `_hd` 版擺在一起：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 8 -t 03:00:00 -J histfull scripts/nano4/run.sbatch python -I scripts/hist_check.py --l1b results/l1b/scores4 results/l1b/scores4_B --opp results/l1b/scores4_opp results/l1b/scores4_B_opp --hsl results/scores_hd --out results/seq/hist_check_full.json
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 8 -t 03:00:00 -J hdfull scripts/nano4/run.sbatch python -I scripts/hd_check.py --l1b results/l1b/scores4 results/l1b/scores4_B --opp results/l1b/scores4_opp results/l1b/scores4_B_opp --hsl results/scores_hd --hd-table results/data/hd_table.csv --out results/seq/hd_check_full.json
```

提交端：`predict_submission.py ... --l1b-heads results/l1b/heads4_A.pt results/l1b/heads4_B.pt --fused-model results/models/rank_model_fused_opp_diff_full.json`（文件的 `l1b_models` 記的是 heads4 的 sha256，配對檢查會擋 heads3 的 shard）。

### 4.22 按 n 條件化標準化（2026-11-02，issue #53；設計在 METHOD_NOTES §2.27）

CPU 約 1 小時，用 R-hdpatch 的 `_hd` 目錄、現有分數、不重評：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 8 -t 03:00:00 -J nstd scripts/nano4/run.sbatch python -I scripts/nstd_check.py --l1b results/l1b/scores3_hd results/l1b/scores3_B_hd --opp results/l1b/scores3_opp_hd results/l1b/scores3_B_opp_hd --hsl results/scores_hd --out results/seq/nstd_check.json
```

貼回 issue #53：`fused_opp_diff` 須重現 `hd_check_hd.json` 的 0.6929；`groups`（split 1 每組 n 的題數、n=5／10／20 的欄標準差中位數）；預先指定 `fused_opp_diff_nstd vs fused_opp_diff` 的差、CI、每折、`by_n`；JSON 進 git。

### 4.23 S00／S01：r3_pma 在 R-hdpatch 每局訊號上重測（2026-11-02，issue #52；設計在 METHOD_NOTES §2.28）

步 1（CPU，約 10 分）：從 `_hd` 的 Human SL 分數建梯子向量（只要 folds 1–4）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 12 -t 01:00:00 -J vechd scripts/nano4/run.sbatch python -I scripts/ladder_build_vectors.py --folds 1 2 3 4 --scores results/scores_hd --out results/ladder/vectors_hd
```

步 2（GPU 1 張；四折 × 3 seed，每個 fit 幾分鐘，訓練 token 約 150k 題 × 20 × 327 float32 ≈ 4 GB RAM）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] --gres=gpu:1 -c 12 -t 04:00:00 -J s01 scripts/nano4/run.sbatch python -I scripts/s01_check.py --vectors results/ladder/vectors_hd --l1b results/l1b/scores3_hd results/l1b/scores3_B_hd --opp results/l1b/scores3_opp_hd results/l1b/scores3_B_opp_hd --hsl results/scores_hd --hd-table results/data/hd_table.csv --models-dir results/models/s01 --out results/seq/s01_check.json
```

貼回 issue #52：`S00` 須重現 `hd_check_hd.json` 的 `fused_opp_diff_hd` 0.7026；預先指定 `S01 vs S00` 的差、CI、每折、`by_n`；`per_seed`；`per_split` 的 `inner_ce_init → inner_ce_best`、`best_epoch`、`n_params`；`buckets`。`s01_check.json` 與 `results/models/s01/*.json` 側檔進 git，`.pt` 不進。

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
