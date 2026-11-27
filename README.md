# AI_CUP_2026-1

AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級」，TEAM_11819 的程式與紀錄。

- 任務：每題給同一位玩家的 5 至 20 局棋譜，預測其等級（D、C、B、A、1D–6D，共 10 級）。
- 本儲存庫另提供競賽一（[AI_CUP_2026-2](https://github.com/bkttt0429/AI_CUP_2026-2)）共用的 SGF 解析、特徵抽取與序列網路定義。
- 專案規則（AI 工具與成員都要遵守）見 [AGENTS.md](AGENTS.md)。

<!-- final-status:begin -->
| 項目 | 內容 |
| --- | --- |
| 競賽 | AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級」 |
| 隊伍 | TEAM_11819 |
| 最終提交檔 | `submission_rank_ct118e_a2b4.csv`（不在儲存庫內） |
| 最終提交檔 md5 | `2c2e4ad637a6407a35de7f599c4ac641` |
| Public 成績 | 0.8587182 |
| Private 成績 | 0.7902471（第 5 名） |
<!-- final-status:end -->

## 1. 文件與目錄結構

### 1.1 目錄結構

```text
AI_CUP_2026-1/
├── AGENTS.md          專案規則（AI 工具與成員都要遵守）
├── src/aicup/         共用套件：SGF 解析、評分、等級頭、序列模型（兩個競賽共用）
├── scripts/           第一段「基礎集成」的程式，與核對入口 verify_final_chain.py
│   ├── m_series/      第二到四段（對手辨識、證據合併、決策）的程式與啟動腳本，檔名沿用實驗登記的 M 編號
│   └── nano4/         Slurm 啟動腳本與環境安裝（nano4 是國網的叢集名稱）
├── results/           最終權重（models/ct_cv_m33/、l1b/heads3*）與最終流程讀取的檔案
│   └── archive/       封存的實驗結果
├── splits/            玩家分折
├── tests/             pytest 測試
├── tools/             規則檢查：run_checks.sh、check_code_rules.py、doc_rules.py
├── docs/              現行文件（最終流程、權重清單、分支紀錄、成績來源 final_status.json、新舊路徑對照）
│   ├── records/       封存紀錄：方法紀錄、實驗登記、計畫、備忘、研究筆記
│   └── work/          工作檔（YYYY-MM-DD-主題.md）
├── experiments/       封存的實驗程式（以 tag pre-restructure-20261127 的目錄結構執行）
└── reports/           簡報與報告網頁（go_deck/ 是結案簡報，仍可更新）
```

執行期資料夾不進版控：`data/`、`models/`、`repo/`、`tools/katago/`、`logs/`、`.venv/`。2026-11-27 整理前的結構保存在 tag `pre-restructure-20261127`，每個檔案的新舊路徑見 [docs/layout_migration.tsv](docs/layout_migration.tsv)。

### 1.2 文件

| 文件 | 內容 |
| --- | --- |
| [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md) | 最終提交流程：各階段的程式、指令、權重位置與不在儲存庫內的檔案 |
| [docs/OFFICIAL_WEIGHTS.md](docs/OFFICIAL_WEIGHTS.md) | 最終流程使用的權重清單與 sha256 |
| [docs/records/NANO4.md](docs/records/NANO4.md) | 叢集上的操作步驟與各步驟的完整參數 |
| [docs/records/architecture_v2_2026-11-02.md](docs/records/architecture_v2_2026-11-02.md) | 完整的實驗紀錄：每一項實驗於取得結果前登記的假設、門檻與結果（以 M 加編號稱呼） |
| [docs/records/REPORT.md](docs/records/REPORT.md)、[docs/records/METHOD_NOTES.md](docs/records/METHOD_NOTES.md) | 資料集與規則、早期的方法紀錄 |
| [docs/records/ARCHITECTURE.md](docs/records/ARCHITECTURE.md) | 分層架構（開案時） |

## 2. 環境

```bash
git clone https://github.com/bkttt0429/AI_CUP_2026-1
git clone https://github.com/bkttt0429/AI_CUP_2026-2   # 並排放置：對手辨識程式會匯入競賽一的 aicup_pid
cd AI_CUP_2026-1
# setup.sh 的兩個前置檔案：KataGo 的 Python 套件（d91ea855，只取 python/）與 Human SL 網路（KataGo v1.15.0 釋出）
git clone --filter=blob:none --no-checkout https://github.com/lightvector/KataGo.git repo/KataGo
git -C repo/KataGo sparse-checkout set python
git -C repo/KataGo checkout d91ea855110dae533f0aada947b2b7d78cc8a4e1
mkdir -p models/katago && curl -fL -o models/katago/b18c384nbt-humanv0.ckpt https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt
echo "84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27  models/katago/b18c384nbt-humanv0.ckpt" | sha256sum -c
bash scripts/nano4/setup.sh          # 建立 .venv：Python 3.12、PyTorch 2.14.1（CUDA 12.6）、requirements.txt 與 requirements-dev.txt
.venv/bin/python -I repo/KataGo/python/migrate_defaultdict_to_dict.py -checkpoint models/katago/b18c384nbt-humanv0.ckpt   # 比賽時也做了這一步（見下方說明）
bash scripts/join_trunk_parts.sh     # 接回切成三段的微調主幹權重，並以 results/l1b/heads3.sha256 核對
```

- 作業系統：Red Hat Enterprise Linux 9.6；排程：Slurm；GPU：NVIDIA H200（每節點 8 張）。
- 預訓練模型：KataGo Human SL 網路 `b18c384nbt-humanv0`，置於 `models/katago/`（取自 KataGo 官方釋出，不隨儲存庫散布）。KataGo v1.15.0 執行檔僅用於一致性檢查，置於 `tools/katago/`。
- Human SL 檢查點的轉換：官方釋出檔（sha256 `84b23476…`）的訓練指標含 `defaultdict`，以 `torch.load(weights_only=True)` 讀取會失敗（`scripts/score_folds.py` 就是這樣讀），所以比賽時以 KataGo 的 `python/migrate_defaultdict_to_dict.py` 轉換；它只把這些指標改成一般的 dict，415 個權重張量與設定都不變，原檔留為 `.ckpt.backup`。轉換後的檔與比賽時的檔大小相同（323,749,927 bytes），sha256 則不同（比賽時為 `00e608f3…`，記於 `docs/nano4_large_files.sha256` 與 `results/l1b/heads3_*.json`）：檢查點的 `train_state` 含一個有 422 個檔名的 Python set（`data_files_used`），pickle 出的順序隨每次執行的字串雜湊而變，所以重新存檔不是逐位元固定的（固定 `PYTHONHASHSEED` 時，兩次轉換逐位元相同）。
- KataGo 的 Python 套件：`setup.sh` 需要 `repo/KataGo/python/katago`（KataGo 官方儲存庫 2026-10-15 的 master，即 `d91ea855`，只用 `python/`），並以 `.pth` 把它與並排的 `../AI_CUP_2026-2/src` 加入 `.venv`。
- 官方資料置於 `data/training/`（訓練集 v1.1）與 `data/tests/`（測試集），不隨儲存庫散布。
- 計算一律以 `sbatch scripts/nano4/run.sbatch <指令>` 送至計算節點；8 張 GPU 的分片以 `bash scripts/nano4/shards.sh 8 <指令>` 執行。Slurm 日誌寫在 `logs/nano4/`（`setup.sh` 會建立）。
- `scripts/m_series/` 的啟動腳本與部分程式含比賽時叢集上的絕對路徑（`/home/u2507115/BK_AICUP_2026-1`、`/home/u2507115/.cache/report_audit`、`/work/u2507115/...`；`BK_AICUP_2026-1`、`BK_AICUP_2026-2` 是兩個儲存庫當時的資料夾名稱），於其他環境執行時須改為自己的路徑。
- 規則檢查：`bash tools/run_checks.sh --all`（中文註解、重複函式、封存區、文件規則、ruff、pytest；見 [AGENTS.md](AGENTS.md) 第 7 節）。

## 3. 流程與各模組的輸入、輸出

系統分四段：基礎集成、對手辨識、證據合併、決策。各步驟的完整指令見 [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md)。

### 3.1 資料處理

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| SGF 解析與讓子局修正 | `src/aicup/sgf.py`、`scripts/hd_table.py`、`scripts/patch_scored.py` | 單行 SGF | 著手序列；讓 2 子以上的對局補上讓子石並還原實際顏色 |
| 玩家分折 | `scripts/make_splits.py` | 訓練集 CSV | `splits/player_folds.csv`（5 折，同一玩家僅在一折） |
| Human SL 逐手評分 | `scripts/score_folds.py`（`--opponent` 評對手的著手） | 每位玩家前 20 局（另補第 21 至 40 局） | 每手於 18 個棋力設定下的對數機率 |
| 等級頭逐手評分 | `scripts/l1b_score.py` | 等級頭權重 `heads3_A`、`heads3_B` | 每手於 10 個等級下的對數機率（交叉評分） |
| 主幹向量 | `scripts/trunk_extract.py` | 棋譜、Human SL 權重 | 每手兩個 64 維向量（盤面平均、落點位置） |
| 模型輸入 | `scripts/trunk_seq_check.py build --opp-hsl`、`rfeat` | 上述三項 | 每手 162 維的陣列；每題 64 個題目層特徵 |
| 測試集 | `scripts/predict_submission.py`、`scripts/trunk_predict.py extract` | 測試集 CSV | 與訓練端相同格式的特徵（8 張 GPU 分片） |

### 3.2 訓練

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 等級頭 | `scripts/l1b_build_cache.py` → `scripts/l1b_train.py` | 局面快取、等級標籤 | `results/l1b/heads3_{A,B}.pt` 與微調後的主幹 |
| 基礎集成的 24 個成員 | `scripts/trunk_seq_check.py train` | 每手陣列、題目層特徵 | `results/models/ct_cv_m33/*.pt` 與設定檔 `.json` |
| 統計成分（對手辨識） | `scripts/m_series/ident_pool.py`、`m84_cache.py` | 訓練棋譜 | 各成分的分數快取 |
| 序列辨識網路 | `scripts/m_series/m88_seqid.py`、`m120_seqid_sym.py`、`m161p_prod.py`、`m161q_prod.py` | 每局前 256 手的主幹向量與落點 | 各網路的檢查點；每局對 8,007 位訓練玩家的相似度 |
| 識別器 | `scripts/m_series/m164_ident.py`、`m164_select.py`、`m127_conf_family.py` | 各辨識來源的分數 | `results/r_variants/m84_<識別器>~m15_{val,test}.npz`：每局最相似的訓練玩家、把握程度、其等級 |

24 個成員的訓練指令（`$arm` 為 C 或 T，`$v` 為驗證折 1–4，`$s` 為種子 0–2）：

```bash
python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 \
  --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 \
  --out results/trunk_seq/es --save-dir results/models/ct_cv_m33
```

### 3.3 推論與決策

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 基礎集成的預測 | `scripts/trunk_predict.py predict` | 24 個成員的權重、測試集特徵 | 每題 10 級的機率與基礎答案 |
| 對手證據 | `scripts/m_series/m71_step2.py`、`m83_classcond.py`、`m87_pi0.py`、`m86_bundle.py`、`m129_bundle2.py` | 識別器輸出 | 依把握程度分帶校準的等級證據 |
| 合併層與整批規則 | `scripts/m_series/m141_opp_lookalike.py`、`m163_pool.py --new SQ4`（兩條推移的來源為 `m53_write_probe.py`） | 基礎機率、對手證據、原型比對的等級分布；先前版本的答案（鎖定的推移與防護規則的比較基準） | 合併後的機率；套用兩條推移（1D 改 A、4D 改 5D）與四條防護規則後的答案（`submission_rank_ct112.csv`，Public 0.8331743） |
| 逐題調整 | `scripts/m_series/m58_decode.py`、`m206_decode.py`、`m223_build.py`、`scripts/build_ct118e_a2b4.py` | ct112 的答案、Public 分數讀數 | 最終提交檔（Private 12 題、Public 22 題與 ct112 不同） |

## 4. 重現與核對

- **核對最終提交檔**（僅需標準函式庫）：`python3 -I scripts/verify_final_chain.py submission_rank_ct118e_a2b4.csv`。此程式列出合併層輸出之後每一題的改動（題號、改前、改後、步驟、依據），逐步還原並以 md5 核對每一個中間檔。
- **由保存的輸出重建合併層的答案**（不需 GPU）：`python -I scripts/m_series/m163_pool.py --new SQ4 --control SPALm --name ctverify`，結果與上傳的 `submission_rank_ct112.csv` 逐位元相同。所需的模型輸出（`results/r_variants/`、`results/m77/`、`results/trunk_seq/`、`results/m84_cache/` 等）、四個先前的提交檔（`submission_rank_ct9.csv`、`submission_rank_ct28.csv`、`submission_rank_ct104b.csv`、`submission_rank_ct106.csv`）與 ct9 的機率檔 `submission_rank_ct9_proba.csv`（這五個檔置於儲存庫根目錄）都不在儲存庫內，保存於本隊的本地備份；放回原位置後，從儲存庫根目錄執行。其中 `results/m84_cache/` 的 `queries.npz`、`var_S4e.npz` 在備份的 `AICUP1_MOVE_20261121_work/work/aicup1_moved/m84_cache/`（比賽時 `results/m84_cache` 是指向 `/work` 的符號連結），要放進實際的 `results/m84_cache/` 資料夾（逐檔清單為 `docs/records/backup_20261118/audit_chain_files.txt` 標記 `pool163` 的列；另見 [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md)「不在 repo 的檔案」）。
- **從頭重現**：依 [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md) 的順序執行，需要官方資料與 GPU。

## 5. 權重

| 權重 | 位置 |
| --- | --- |
| 24 個基礎成員 | `results/models/ct_cv_m33/` |
| 等級頭與微調後的主幹 | `results/l1b/`（主幹各切為三段，以 `scripts/join_trunk_parts.sh` 接回） |
| 題目層模型 | `results/models/rank_model_fused_opp_diff_hd_r.json` |
| 序列辨識網路（50 個檔） | 單檔多超過 GitHub 的 100 MB 上限，保存於本隊的本地備份，可依主辦單位要求提供；清單與 sha256 見 [docs/OFFICIAL_WEIGHTS.md](docs/OFFICIAL_WEIGHTS.md) |

三個較早的辨識元件僅保存其輸出、未保存權重（見 docs/FINAL_PIPELINE.md「不在 repo 的檔案」）；合併層與決策層可由這些輸出重算。

2026-11-27 起，`src/` 與 `scripts/` 的每一行程式都加上了中文註解（程式邏輯經語法樹比對確認未變）。權重的附屬檔記錄了當時訓練程式的位元組雜湊（`code_sha256`），因此載入這些權重時會印出「由其他訓練程式訓練」的警告；這只是提醒，不影響預測結果。

## 6. 外部資源與揭露事項

- KataGo 與其 Human SL 網路 `b18c384nbt-humanv0`（<https://github.com/lightvector/KataGo>）：逐手特徵與等級頭的基礎。`kata1-b18c384nbt-s9996604416` 僅用於未採用的實驗。
- Fox Go Dataset（<https://github.com/featurecat/go-dataset>，GPL-3.0）：9,685 個帳號、193,700 局，用於一個辨識網路的訓練；帳號名稱僅以加鹽雜湊使用，未與競賽資料比對身分。
- 本隊於競賽一撰寫的統計成分程式，以本競賽訓練資料重新擬合。
- Public 排行榜分數的使用：兩條整批推移與四條防護規則經 Public 驗證後採用；Private 有 12 題答案由隊員依 Public 讀數決定；Public 有 22 題為從 Public 分數回推生成的標籤。逐題清單見 `scripts/verify_final_chain.py`。
- 生成式 AI：程式撰寫、實驗執行與紀錄整理使用 Anthropic Claude（Claude Code），並曾使用 OpenAI 的 GPT 與 Codex、Google 的 Gemini 審閱設計構想及處理部分實驗；採用與上傳的決定由隊員做成。

官方資料、KataGo 權重與引擎、虛擬環境與提交檔均不進版控（見 `.gitignore`）。競賽資料僅供本競賽使用，不得用於識別真實使用者。
