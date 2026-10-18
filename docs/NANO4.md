# 國網 nano4 使用說明

把專案搬到 NCHC nano4（每個節點 8 張 H200），並在上面跑 GPU 工作。nano4 只接受密碼加 OTP 登入，所以下面每一個 `scp` 和 `ssh` 都要你自己輸入。腳本都放在 `scripts/nano4/`。

## 1. 上傳與解開（只做一次）

在本機把 `D:\AICUP_transfer` 裡的四個檔案傳到國網的家目錄：

```bash
cd D:/AICUP_transfer
```

```bash
scp AICUP_2026-1.tar.xz AICUP_2026-1.tar.xz.sha256 AICUP_2026-1.files.sha256 AICUP_2026-1_nano4.tar.gz nano4:~/
```

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

`scripts/nano4/run.sbatch` 會在專案環境裡執行你給的指令，指令裡的 `python` 就是 `.venv` 的 Python。預設是 dev 分區、1 張 GPU、12 核、4 小時；在腳本名稱前面加 sbatch 參數就能覆蓋預設值。log 在 `logs/nano4/<工作名稱>-<jobid>.out`。

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
| dev | 4 小時 | 測試和短工作（預設） |
| ngs1gpu、ngs2gpu、ngs4gpu、ngs8gpu | 14 天 | 1–8 張卡的長工作；計畫能不能用還沒確認 |
| 8gpus | 2 天 | 一整個節點（8 張 H200） |
| 16gpus 以上 | 1–2 天 | 多節點，目前的腳本用不到 |

常用指令：`squeue -u $USER` 看排隊和執行中的工作，`scancel <jobid>` 取消，`sacct -j <jobid>` 看結束後的狀態。

## 5. 同步程式碼、取回結果

在本機改完程式後，用 Git Bash 同步。只會傳程式碼（src、scripts、tests、viz、pyproject.toml、requirements.txt），不會傳資料：

```bash
bash scripts/sync_to_nano4.sh
```

從國網把結果拉回本機，例如：

```bash
scp nano4:AICUP_2026-1/results/bench_nano4.json results/
```

## 6. 被拒或出錯時

- 計畫或分區不對：用 `-A <計畫代號>` 或 `-p <分區>` 覆蓋。預設計畫是 gov113074。
- CPU 或 GPU 數量不符合分區規定：用 `-c` 和 `--gres=gpu:N` 調整，錯誤訊息通常會寫出上限。
- 記憶體不足：加上 `--mem=64G` 之類的參數。
- `setup.sh` 說連不到網路：登入節點可能要設 proxy（`export https_proxy=...`）。
- torch 說驅動太舊：用預設的 cu126 版本重裝（`TORCH_CUDA=cu126 bash scripts/nano4/setup.sh`）。
- KataGo 引擎（`scripts/katago.sh`）在國網跑不起來：它需要的 libzip 和 OpenSSL 1.1 只裝在 lab01。只有 0A 一致性檢查會用到引擎，評分和訓練都只用 PyTorch，相關測試會自動略過。
