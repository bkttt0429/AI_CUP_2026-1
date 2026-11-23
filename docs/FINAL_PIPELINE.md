# 最終提交流程（AI CUP 2026 圍棋棋力等級預測）

2026-11-23 版。這份文件說明最終提交檔是怎麼來的、每一段用到哪些程式、權重放在哪裡，以及哪些東西不在這個 repo。

## 最終提交檔

- 定案檔是 `submission_rank_ct118e_a2b4.csv`（md5 `2c2e4ad637a6407a35de7f599c4ac641`），Public 0.8587182。
- 它的 Public 400 題與 `submission_rank_ct118e.csv`（md5 `0ef7368fb85cf1eecde2eb18b99367cb`，Public 0.85871825）逐題相同，所以 Public 讀數也相同；ct118e 的 Public 有 16 題是依歷次上傳的 Public 分數回推後修正的。
- 它的 Private 400 題＝`submission_rank_ct116.csv`（md5 `6b053db6672cd31cbb2e6b742de18649`，Public 0.8394955）的 Private，再把 4 題（priv_q_0062、0125、0204、0325）由 A 改成 B（2026-11-23 參賽者決定，見第四段）。
- 備援是 ct118e（Private 與 ct116 逐題相同）。比賽以最後一次上傳計分，定案檔必須是最後一筆上傳。
- Private 的預期（2026-11-21 結案時的計算，M264）：約 0.81（中心 0.80–0.82），八成區間 0.785–0.835，九成區間 0.78–0.84；那 4 題自估只多約 0.001–0.0015。
- 提交檔本身不放在 repo（見 `.gitignore`）。
- **核對**：`python3 -I scripts/verify_final_chain.py submission_rank_ct118e_a2b4.csv`（只用標準函式庫）。ct112 之後每一個答案的改動（題號、改前、改後、步驟、出處）都列在這支程式的 STEPS；它先從定案檔逐步還原到 ct118e、ct118d、ct117、ct116、ct112，每一步都要對上建檔時登記的 md5，再從 ct112 依序套用回到定案檔。2026-11-23 在計算節點上全部通過。要從模型輸出開始核對：先在放回本地備份的 repo 裡跑 `python -I logs/nano4/m163_pool.py --new SQ4 --control SPALm --name ctverify`（2026-11-18 的檔案稽核用同一個指令重建出與 ct112 逐位元相同的檔），再加上 `--ct112 submission_rank_ctverify.csv`。`--ct116`／`--ct118e` 等選項可以把存著的檔一起比對，`--write DIR` 會寫出還原的中間檔。
- 每一個實驗（事前登記與結果）都記在 `reports/architecture_v2_2026-11-02.md`，以 M 加編號稱呼；下面提到的 M### 都指那份文件裡的條目。

## 環境

- Python 3.12，套件見 `requirements.txt`；在 nano4 上以 `bash scripts/nano4/setup.sh` 建 `.venv`。
- KataGo v1.15.0 執行檔放在 `tools/katago/`，Human SL 權重 `b18c384nbt-humanv0` 放在 `models/katago/`。
- 比賽資料放在 `data/training/` 與 `data/tests/`（不隨 repo 散布）。
- 所有計算以 `sbatch scripts/nano4/run.sbatch <指令>` 送到計算節點；8 卡分片用 `bash scripts/nano4/shards.sh 8 <指令>`。前七步的完整參數見 `docs/NANO4.md` 的對應小節。
- 2026-11-05 之後的實驗腳本都在 `logs/nano4/`（一個實驗一支，檔名開頭是 M 編號）；正式流程的啟動腳本在 `logs/nano4/report_audit/`。

## 第一段：基礎集成（只看棋的內容）

1. **切折**：`splits/player_folds.csv`（`scripts/make_splits.py`）。
2. **Human SL 18 個剖面評分**：`scripts/score_folds.py`，本人與對手（`--opponent`）各一趟；讓子局以 `scripts/hd_table.py`、`scripts/patch_scored.py` 重評併回；每人第 21–40 局以 `logs/nano4/x40_keys.py` 產生清單後補評。見 NANO4 §4.19、§4.23、§4.25。
3. **L1b 等級頭**：`scripts/l1b_build_cache.py` → `scripts/l1b_train.py`（先只訓 head，再以 `--trunk-lr 1e-5` 微調 trunk）得到 `heads3_A`、`heads3_B`；`scripts/l1b_score.py` 交叉評分。見 NANO4 §4.2、§4.19、§4.23。
4. **trunk 向量**：`scripts/trunk_extract.py`。見 NANO4 §4.21、§4.23。
5. **token 陣列與題目層特徵**：`scripts/trunk_seq_check.py build --opp-hsl ...` 與 `rfeat`。
6. **24 個成員**（4 個驗證折 × C、T 兩種結構 × 種子 0–2；含玩家對比損失，M33）：

   ```bash
   python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 \
     --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 \
     --out results/trunk_seq/es --save-dir results/models/ct_cv_m33
   ```

7. **測試集推論**：`scripts/predict_submission.py`（分片評分）→ `scripts/trunk_predict.py extract` → `scripts/trunk_predict.py predict --models results/models/ct_cv_m33/val*_*-pc-f0-x40-oh_s*.pt`，產出 `submission_rank_ct9.csv`（md5 前 12 碼 `6ff60b847e26`，Public 0.7622032）與各題的機率檔。啟動腳本：`logs/nano4/report_audit/b1_public.sh`。

## 第二段：認出對手（錨點）

互先局的兩人當局掛牌相同，所以「對手是哪一位有標籤的訓練玩家」就是本人等級的證據。

- **五個手工的習慣成分**：`logs/nano4/ident_pool.py`（開局、落點統計、局部棋形、盤面朝向、讓子類別；改寫自本隊在同屆「玩家辨識」競賽的程式），分數快取 `m84_cache.py`。
- **學習式配對器**：`m82_cond_id.py`（M82），變體與組檔 `m84_variant.py`。
- **逐手序列的認人網路**：`m88_seqid.py`（M88；`--full` 是用全部 100 萬局的 M90；`--ext-ply`／`--ext-select` 是加入外部公開帳號的 M101）、`m120_seqid_sym.py`（M120，對稱擴增）、`m161p_prod.py` 與 `m161q_prod.py`（M161：依局數抽樣、加寬、學習式中心）。查詢向量的八向平均：`m124_tta_queries.py`、`m161q_tta.py`。
- **識別器的組成**：`m164_ident.py`（來源融合、中心來源、活躍度先驗 M163）、`m164_select.py`（不用標籤的選擇規則）、`m127_conf_family.py`（把握統計量 m15）。最後加入合併的來源是 SQ4。
- 輸出是每一局的「最像的訓練玩家、把握、他的等級」：`results/r_variants/m84_<識別器>~m15_{val,test}.npz`。

## 第三段：證據與合併

- **錨點證據**：`m71_step2.py`（依把握分帶校準）、`m83_classcond.py`（依等級的條件表）、`m87_pi0.py`、`m86_bundle.py`、`m103_stability.py`、`m105_voters.py`、`m129_bundle2.py`；編碼器項 `m77_prep.py`、`m77_idbert.py`、`m77_eval.py`。
- **「像誰」讀數與合併層**：`m141_opp_lookalike.py`（把本人的向量對訓練玩家的原型讀成等級分布，並擬合每種題數一組係數的線性合併）；組檔與防護 `m137b_build.py`、`m137f_build2.py`。
- **`m163_pool.py --new SQ4`** 產出 `submission_rank_ct112.csv`（md5 前 12 碼 `7d99d6f2c5b7`，Public 0.8331743）。採用規則是兩組種子的交叉驗證都過門檻（M37）。

## 第四段：決策層的調整

- **兩條由 Public 讀數選出的推移**（1D→A、4D→5D）：`m53_write_probe.py`；每次上傳的精確分數以 `m58_decode.py`、`m206_decode.py` 解成「完全正確、差一級以內」各變動幾題。
- **ct116**＝ct112 加上參賽者決定的 8 題 Private（解除 6 題 4D→5D 的鎖定、2 題 A→B；M209），以及 7 題依 Public 讀數修正的 Public 答案（`m210_profile_all.py`、`m211_ledger_ct116.py`）。
- **ct118e**＝ct116 的 Private，加上依後續讀數修正的 Public 答案（`m220_resolve.py`、`m222_update.py`、`m223_build.py`）；中間經過 ct117（M212，Public 12 題，沒上傳）與 ct118d（M220，再改 Public 6 題）。各步的逐題改動見 `scripts/verify_final_chain.py`。
- **ct118e_a2b4（定案）**＝ct118e 的 Private 再把 4 題由 A 改成 B（建檔程式 `scripts/build_ct118e_a2b4.py`）。這 4 題的合併後驗本來就答 B（B 的機率 0.50–0.64），只是被「不離開 A」的防護擋住。依據：Public 上同一條規則的探測（`probe_A2B`，被擋住的 7 題改 B，M206）淨多對 2 題（+1.26 分），沒達事先登錄的門檻（+3）。參賽者 11-17 先放寬把握最高的 2 題（priv_q_0084、0087，已在 ct116），11-23 決定再放寬其餘 4 題。自估 Private 多約 0.4–0.6 分（400 題總分），實際落在 ±2.5 分之間。這是參賽者依 Public 讀數做的決定，報告要揭露。
- 這一段用到了 Public 排行榜的回饋；只改 Public 的修正不影響 Private 的答案。

## 外部資源（報告需揭露）

- KataGo Human SL 網路 `b18c384nbt-humanv0`（逐手特徵的來源）。
- KataGo `kata1-b18c384nbt-s9996604416`（只用在未採用的實驗：M23、M25、M74、M262）。
- Fox Go Dataset（GPL-3.0）：外部公開帳號，只當作認人網路訓練時的「陌生人」（M101、M218）；帳號名稱只以加鹽雜湊分組，未與比賽資料比對身分。
- 同屆「玩家辨識」競賽的程式（習慣成分）；該競賽的候選棋譜只用在未採用的實驗（M95、M253、M255）。
- Public 分數的使用如第四段所述。

## repo 裡的權重

| 檔案 | 內容 |
| --- | --- |
| `results/models/ct_cv_m33/` | 24 個基礎成員的權重（`.pt`）與設定（`.json`） |
| `results/l1b/heads3_{A,B}.pt`、`.json` | L1b 等級頭 |
| `results/l1b/heads3_{A,B}.trunk.pt.part{0,1,2}` | 微調過的 trunk（各約 103 MB，GitHub 單檔上限 100 MB，所以切成三段）；`bash scripts/join_trunk_parts.sh` 會接回原檔並以 `results/l1b/heads3.sha256` 核對 |
| `results/models/rank_model_fused_opp_diff_hd_r.json` | 題目層 64 個特徵的模型（基礎集成的輸入之一） |

## 不在 repo 的檔案

- **認人網路的權重與中間輸出**（共約十餘 GB，單檔多半超過 100 MB）：2026-11-21 起保存在參賽者的本地備份（nano4 上已刪除）；當時的清單、大小與 sha256 在 `docs/nano4_large_files.sha256`，其中最終流程用到的權重另列在 `docs/OFFICIAL_WEIGHTS.md`，需要驗證時可提供。
- **已刪除的檢查點**：比賽期間為了騰出磁碟配額，一部分較早的認人檢查點（例如 M82 的配對器、M120 的部分網路）已經刪除，紀錄在 `reports/backup_20261118/deleted_*.txt`；它們的輸出（上面的 `m84_*.npz` 與 `results/m84_cache/var_*.npz`）仍在，第三、四段可以由這些輸出重算。最後幾層實際讀到的檔案清單見 `reports/backup_20261118/audit_chain_files.txt`。
- 比賽資料、評分中間檔、快取與提交檔。
