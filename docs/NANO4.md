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

fold 0 的評估**只跑一次**，而且只在使用者從 `cv_summary.json` 選定 rung 之後才跑。CV 若用了 `--per-player` 或 `--no-counts`，這裡要給一樣的參數，讓設定對得上摘要裡的標籤；set rung 的 `--seeds` 也要和該標籤的 seeds 相同（預設 0 1 2）。對不上就 exit 2。結果寫到 `results/ladder/fold0_eval.json`，這個檔已存在時，不論 `--out` 指到哪裡，腳本都 exit 2，不會重評。fold 0 只評選定的設定（該 rung `TrainCfg.pick` 的 tick，除了 `r5_ctm_minmax` 都是最後一個 tick），CV 裡只報告的 @T16、@final 不在 fold 0 上算：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 12:00:00 -J ladf0 scripts/nano4/run.sbatch python -I scripts/ladder_eval_fold0.py --rung <選定>
```

2026-10-24 追加（issue #2，待辦第 3 項）。`r5_ctm_minmax` 的選模準則改了（內層 CE 與預測都取最確定 tick，`TrainCfg.pick = "certain"`），舊的 `results/ladder/cv/r5_ctm_minmax_s{0,1,2}.npz` 作廢，**在登入節點先刪掉**再送工作，否則腳本會略過；`r5_ctm_unet` 是新 rung（`r5_ctm` 換成 depth-4 U-Net synapse）。兩者都是 r5，仍要 `--force`。單張 H200 兩個 rung 約 1 小時：

```bash
rm -f results/ladder/cv/r5_ctm_minmax_s*.npz
sbatch --exclude=25a-hgpn[001,003,004,007,062] -p 8gpus -t 06:00:00 -J ladder3b scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r5_ctm_minmax r5_ctm_unet --force
```

前三名 ensemble（`ens_top3`）不訓練：讀 `r3_pma`、`r4_xattn_t1`、`r5_ctm` 各 seed 的 CV 檔，機率平均後重算每題分數，寫成 `results/ladder/cv/ens_top3_s{seed}.npz`，再重寫 `cv_summary.json`，並印出事先指定的六組比較（`ens_top3`／`r5_ctm_minmax`／`r5_ctm_unet` 各對 `r5_ctm` 與 `r3_pma`；只報告、不作門檻）。CPU 幾分鐘，但依國網規定仍送 Slurm：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,062] -t 01:00:00 -J ladens scripts/nano4/run.sbatch python -I scripts/ladder_ensemble.py
```

`ens_top3` 的檔已存在時要加 `--overwrite`（成員重跑過就該重建）。要 commit 的是 `results/ladder/cv_summary.json`；`.npz` 一律不進 git。

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
