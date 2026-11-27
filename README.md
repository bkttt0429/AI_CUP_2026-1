# AI_CUP_2026-1

AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級」，TEAM_11819 的程式與紀錄。

- 任務：每題給同一位玩家的 5 至 20 局棋譜，預測其等級（D、C、B、A、1D–6D，共 10 級）。
- 成績：Public 0.8587182、Private 0.7902471（最終提交檔 `submission_rank_ct118e_a2b4.csv`，md5 `2c2e4ad637a6407a35de7f599c4ac641`）。
- 本倉庫另提供競賽一（[AI_CUP_2026-2](https://github.com/bkttt0429/AI_CUP_2026-2)）共用的 SGF 解析、特徵抽取與序列網路定義。

## 1. 文件

| 文件 | 內容 |
| --- | --- |
| [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md) | 最終提交流程：各階段的程式、指令、權重位置與不在倉庫內的檔案 |
| [docs/OFFICIAL_WEIGHTS.md](docs/OFFICIAL_WEIGHTS.md) | 最終流程使用的權重清單與 sha256 |
| [docs/records/NANO4.md](docs/records/NANO4.md) | 叢集上的操作步驟與各步驟的完整參數 |
| [docs/records/architecture_v2_2026-11-02.md](docs/records/architecture_v2_2026-11-02.md) | 完整的實驗紀錄：每一項實驗於取得結果前登記的假設、門檻與結果（以 M 加編號稱呼） |
| [docs/records/REPORT.md](docs/records/REPORT.md)、[docs/records/METHOD_NOTES.md](docs/records/METHOD_NOTES.md) | 資料集與規則、早期的方法紀錄 |
| [docs/records/ARCHITECTURE.md](docs/records/ARCHITECTURE.md) | 分層架構（開案時） |

## 2. 環境

```bash
git clone https://github.com/bkttt0429/AI_CUP_2026-1 && cd AI_CUP_2026-1
bash scripts/nano4/setup.sh          # 建立 .venv：Python 3.12、PyTorch 2.14.1（CUDA 12.6）與 requirements.txt 的套件
bash scripts/join_trunk_parts.sh     # 接回切成三段的微調主幹權重，並以 results/l1b/heads3.sha256 核對
```

- 作業系統：Red Hat Enterprise Linux 9.6；排程：Slurm；GPU：NVIDIA H200（每節點 8 張）。
- 預訓練模型：KataGo Human SL 網路 `b18c384nbt-humanv0`，置於 `models/katago/`（取自 KataGo 官方釋出，不隨倉庫散布）。KataGo v1.15.0 執行檔僅用於一致性檢查，置於 `tools/katago/`。
- 官方資料置於 `data/training/`（訓練集 v1.1）與 `data/tests/`（測試集），不隨倉庫散布。
- 計算一律以 `sbatch scripts/nano4/run.sbatch <指令>` 送至計算節點；8 張 GPU 的分片以 `bash scripts/nano4/shards.sh 8 <指令>` 執行。

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
- **由保存的輸出重建合併層的答案**（不需 GPU）：`python -I scripts/m_series/m163_pool.py --new SQ4 --control SPALm --name ctverify`，結果與上傳的 `submission_rank_ct112.csv` 逐位元相同。
- **從頭重現**：依 [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md) 的順序執行，需要官方資料與 GPU。

## 5. 權重

| 權重 | 位置 |
| --- | --- |
| 24 個基礎成員 | `results/models/ct_cv_m33/` |
| 等級頭與微調後的主幹 | `results/l1b/`（主幹各切為三段，以 `scripts/join_trunk_parts.sh` 接回） |
| 題目層模型 | `results/models/rank_model_fused_opp_diff_hd_r.json` |
| 序列辨識網路（50 個檔） | 單檔多超過 GitHub 的 100 MB 上限，保存於本隊的本地備份，可依主辦單位要求提供；清單與 sha256 見 [docs/OFFICIAL_WEIGHTS.md](docs/OFFICIAL_WEIGHTS.md) |

三個較早的辨識元件僅保存其輸出、未保存權重（見 docs/FINAL_PIPELINE.md「不在 repo 的檔案」）；合併層與決策層可由這些輸出重算。

## 6. 外部資源與揭露事項

- KataGo 與其 Human SL 網路 `b18c384nbt-humanv0`（<https://github.com/lightvector/KataGo>）：逐手特徵與等級頭的基礎。`kata1-b18c384nbt-s9996604416` 僅用於未採用的實驗。
- Fox Go Dataset（<https://github.com/featurecat/go-dataset>，GPL-3.0）：9,685 個帳號、193,700 局，用於一個辨識網路的訓練；帳號名稱僅以加鹽雜湊使用，未與競賽資料比對身分。
- 本隊於競賽一撰寫的統計成分程式，以本競賽訓練資料重新擬合。
- Public 排行榜分數的使用：兩條整批推移與四條防護規則經 Public 驗證後採用；Private 有 12 題答案由隊員依 Public 讀數決定；Public 有 22 題為從 Public 分數回推生成的標籤。逐題清單見 `scripts/verify_final_chain.py`。
- 生成式 AI：程式撰寫、實驗執行與紀錄整理使用 Anthropic Claude（Claude Code），並曾使用 OpenAI 的 GPT 與 Codex、Google 的 Gemini 審閱設計構想及處理部分實驗；採用與上傳的決定由隊員做成。

官方資料、KataGo 權重與引擎、虛擬環境與提交檔均不進版控（見 `.gitignore`）。競賽資料僅供本競賽使用，不得用於識別真實使用者。
