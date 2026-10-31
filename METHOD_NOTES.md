# 方法筆記（收斂結論，持續更新）

最後更新：2026-10-16（pilot 後）。只記「已確認的事實」與「已收斂的決策」，討論過程不留。資料與規則見 [REPORT.md](REPORT.md)；問題定義、分層架構與時程見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 1. 已確認的事實

| 事實 | 來源 / 驗證方式 |
| --- | --- |
| 官方 baseline 不用 `color`、不用 `player_id`；取最後 250 個盤面（含根節點空盤），不足補空盤 | `repo/AICup-2026-Tutorial/utils.py` |
| 手數 ≥ 250 的棋局 33.7%（D 36.8% → 6D 29.7%，隨棋力單調下降）；補盤數 = max(250 − 盤面數, 0)，只對短局攜帶局長訊號 | lab01 `scripts/movecount.awk` 全量 |
| 100 萬局 SGF 全部唯一，對手永不在資料集中 | lab01 全量去重 |
| 抽樣 10,000 局重播：0 解析錯、0 落在已有子、0 簡單打劫違規、0 虛手（未檢查自殺、superko） | lab01 `scripts/check_legal.py` |
| 測試每題 5–20 局同一玩家、附執色；提交 800 列 `question_id,pred_rank` | Tutorial README |
| KataGo Human SL（`b18c384nbt-humanv0`，有 `.ckpt` 可微調）：`rank_3d` = `makeBasicRankProfile(7,7)`，**黑白雙方同為 3d**；並固定 source=KGS、日期 2020-03-01（`preaz_*` 為 2016-09-01）、讀秒制 1200s + 30s×5、雙方 human。雙等級形式 `rank_{BR}_{WR}` 可分別指定 | `cpp/neuralnet/sgfmetadata.cpp` |
| Python `SGFMetadata` 預設值（日期 1970-01-01、source 0、human=False、tcIsUnknown）**不等於**引擎 profile；編碼依下一手執色分「我方／對方」各 34 通道 thermometer | `python/katago/game/sgfmetadata.py` |
| Human SL 的 metadata encoder 輸出在**進入殘差主幹之前**加到輸入嵌入 → 每個 profile 都要重算整個 trunk；`Model.forward()` 永遠同時算 policy 與 value/ownership/score 等 head | `python/katago/train/model_pytorch.py` |
| **0A 通過（2026-10-15）**：PyTorch 直讀 `humanv0.ckpt`（`use_swa=False`、非法步以 `Board.would_be_legal` 遮 −inf 後 FP32 log_softmax）vs. analysis engine v1.15.0 Eigen（`rank_1d`／`rank_5k`、chinese 7.5、`nnRandomize=false`、含歷史）：20 個局面（黑白目標各半，含提子後局面）目標落子 max |Δlog p| = 3.8e-4，整個合法步向量 max |Δ| = 3.6e-3、相關 > 0.9999999，argmax 全部一致。殘差：Python `Board` 把**多子自殺**當合法（3/20 局面各多 1 個合法點，p≈1e-7），引擎視為非法；只影響歸一化的第 7 位小數，目標落子永不受影響 | `scripts/check_consistency.py` → `results/consistency_0A.json` |
| Analysis engine 的 `humanSLProfile` 是單一欄位，一個 profile 一次查詢；人類行為分析建議 `ignorePreRootHistory=false` | `docs/Analysis_Engine.md` |

## 2. 已收斂的決策

### 2.1 驗證協議（所有實驗共用）
- 依 `player_id` 整人留出，5-fold（`splits/player_folds.csv`，`scripts/make_splits.py`，seed 42）：每級內玩家依盤數降冪、同分打亂後 round-robin 分 fold，所以重度玩家平均散在各 fold；每級每 fold 玩家數差 ≤ 1（D 178、C 163–164、B 209–210、A 316–317、1D 86–87、2D 122、3D 123–124、4D 128、5D 131–132、6D 142–143）。
- 假題（`splits/questions_fold{0..4}.csv`）：每位留出玩家的棋局打亂後依 5 → 10 → 20 局輪流切題，每種大小最多 2 題、題間棋局互斥。每 fold 約 6,900 題：n=5 約 2,920、n=10 約 2,300、n=20 約 1,680；每級每種大小 ≥ 122 題（最少是 1D 的 n=20）。另加「總目標手數相近但局數不同」的對照組（待做）。
- 指標：官方 ±1 加權分 + 查詢級 NLL + 校準。以玩家為重抽樣單位報不確定性。
- 校準／映射用 out-of-fold，不與最終本地評估共用同一批玩家。
- 同時維持一組「已見玩家」驗證集，兩者差距 = 測試集玩家重疊與否的敏感度。
- 先在新協議下重跑原始 baseline 取參考分；本地分與 Public 0.4631 不直接比高低。
- **原始 baseline 在本協議下的參考分**（2026-10-16，`results/baseline_official_eval.json`，`scripts/baseline_{extract,train,eval}.py`）：官方 `utils.py`／`network.py` 原封載入、notebook 超參數照抄，只改切分：訓練 folds 1–4 的 5,593 人／691,636 局（notebook 90 萬），驗證 folds 1–4 整人 809 人／102,414 局（每級約 1 萬），fold 0 不碰；最佳 epoch 4，val 每局準確率 21.7%（fold 0 每局 21.5%）。與 L3/L4 同一組 fold 0 題（1,594 人、9,872 題），各級平均官方分取 5／10／20 局平均：notebook 的平均 softmax + argmax **0.399 [0.387, 0.411]**（n=5／10／20 為 0.382／0.399／0.416，題目平均 0.398）；同一機率改用 expected-score 決策 **0.440 [0.427, 0.453]**（0.414／0.436／0.470）。L3/L4 的 0.613 [0.600, 0.627]（§2.8，同樣用 expected-score）高 0.17，各題長高 0.16–0.19。pilot 的 200 人題上 argmax 0.400／0.427／0.441、expected-score 0.431／0.465／0.505，同題 Human SL 門檻 0.586／0.601／0.647（§2.4.1(e)）。單一 seed（CI 不含訓練變異）。Public 0.4631 高於本協議 n=20 的 CI 上界，原因未查（測試玩家與訓練重疊？訓練量多 30%？題目組成不同？）。

### 2.2 主線模型
- 等級條件化的人類落子模型 \(p_\theta(a_t \mid H_t, k)\)：**骨幹不帶等級**，等級在後段注入（或 10 個小 policy head 輸出 `[B,10,362]`）。選擇理由是全等級推論成本（trunk 算一次）與防止等級 head 洩漏，**不是**已證明辨識力優於早注入。
- 直接等級 head 走不含等級的分支，可看實際落子，不可看整局總局長（policy 分支只准用落子前資訊）。
- 選模型同時看目標落子 NLL 與查詢級官方分，不用落子 accuracy。
- 「晚注入」與「從零／預訓練／蒸餾」是獨立決策；不把自訓與 Human SL 微調當互斥。

### 2.3 聚合與決策
- 基本量：只計目標玩家落子的 \(L_k(Q)=\sum\log p(a\mid H,k)\)。
- 校準：\(z_k = b_k + (L_k-\bar L)/(T_0 (n/n_0)^\alpha)\)，第一輪 \(b_k=0\)，比較固定溫度與 \(T_0+\alpha\)。過度自信時應**增大 T**（減小 β）。
- 若做細等級混合 \(\log\sum_r w_{k,r}\exp(\beta(n)[L_r-\bar L])\)，縮放要在 LSE **內部**。
- 對照組：softmax 算術平均（baseline 做法）、先驗修正後 log 相加、逐手等權 vs 每局等權。
- 最終決策：\(\hat k=\arg\max_k[q_k+e^{-1}(q_{k-1}+q_{k+1})]\)，越界項為 0。純量溫度不改變排序，沒有辨識訊號時校準救不回來。

### 2.4 Human SL pilot（步驟 2）
- 每級 20 位留出玩家 × 20 局 × 只計目標落子 × 全部細 profile（`rank_12k`…`rank_1k` 12 個 + `rank_1d`…`rank_6d` 6 個 = 18 個）。
- 算量：200 × 20 × ~110 × 18 ≈ **7.9×10⁶** 次評估。以 0B 實測 bf16 batch 64 的 4,296 次/秒/卡，純前向約 31 分鐘（單卡）；特徵只算一次（18 個 profile 共用盤面特徵，只換 metadata 列），CPU 重播＋特徵 440k 局面 ÷ 226/秒 ≈ 32 分鐘單核，多進程可壓到幾分鐘。
- 記錄假設：每次評估都隱含「對手與我方同級、KGS、2020、讀秒制」。另用小子集固定我方、掃對手等級，量 Δlog p。
- 解讀：測的是「既有模型對本資料的轉移」，不是方法上限；零樣本不好 ≠ 自訓條件模型不可行。

#### 2.4.1 Pilot 實測（2026-10-16，`results/pilot_humansl.json`）
- 規模：fold 0（評估）與 fold 1（校準）各 200 位玩家（每級 20 人，有 20 局題者，seed 42）× 20 局；fold 0 共 447,393 個目標落子、fold 1 共 442,548；每手 18 個 profile。兩張 GPU 各一個 shard，每 fold 約 23 分鐘（每卡 161–165 局面/秒 × 18 profile）。FP32 + cuDNN 預設 TF32 卷積（與單局面路徑的差：最大 9×10⁻⁴、平均 2.6×10⁻⁴ nats/手）。
- 題目：每人 1 題 20 局、2 題 10 局、4 題 5 局（同大小互斥，跨大小巢狀）。fold 0：n=5 800 題、n=10 400 題、n=20 200 題。
- **(a) 似然曲線峰值隨等級單調右移，但 KGS 尺度被壓縮**：D 的峰在 6k、C 5k、B 4k、A 3k、1D 與 2D 2k、3D 2k/1k、4D 1k、5D 1d、6D 2d。10 個類別只落在 KGS 6k–2d 的 8 個 profile 上，1D–4D 擠在 2k–1k。
- **(b) 零樣本硬分箱（D=10–12k…6D=6d）失敗**：官方分 0.25（n=5/10/20 都一樣）。B、A 還行（0.78、0.81），段位幾乎全錯（預測太弱），D 預測太強。尺度必須學。
- **(c) 每局正規化 vs 直接相加**：差別在 CI 內（0.25 vs 0.26），不是瓶頸。
- **(d) 相鄰等級 AUC（不擬合任何參數）**，n=20：D|C 0.87、C|B 0.77、B|A 0.79、A|1D 0.88、1D|2D 0.62、2D|3D 0.73、3D|4D 0.65、4D|5D 0.86、5D|6D 0.89；posterior-mean profile index 與等級的 Spearman 0.944。**n=5 → n=20 的 AUC 只升 ≤ 0.05**：瓶頸是玩家層級（同一級內玩家的「下法強度」分布重疊），不是證據量。
- **(e) fold 1 擬合單調門檻 → fold 0**：n=5 0.586 [0.551, 0.622]、n=10 0.601 [0.558, 0.645]、n=20 0.647 [0.598, 0.694]（玩家層級 bootstrap 95% CI）；fold 1 自身 0.625/0.639/0.677，過擬合約 0.03–0.04。n=20 每級：D 0.81、C 0.73、B 0.67、A 0.59、1D 0.71、2D 0.30、3D 0.52、4D 0.62、5D 0.65、6D 0.89；exact 0.50、±1 內 0.90。這些是 fold 0 pilot 題的分數，**不能直接和 Public 0.4631 比**，但已指出 L1a + 單調映射是可行的第一份提交。
- **(f) 對手等級敏感度**（20 局，我方固定在類別中心，掃對手 12k–6d）：對手**相鄰一級**的改變 ≤ 0.024 nats/手，我方相鄰一級 0.007–0.099；整段 12k→6d 的對手範圍 0.02–0.22（段位較大，極端組合屬分布外）。結論：`rank_k` 的「對手同級」假設在配對接近時無害，不是瓶頸。掃描與主跑在 opp=own 時一致到 4.8×10⁻⁵/手。
- 限制：(1) pilot 只收在該 fold 有 20 局題的玩家（≥ 35 局），比測試題的玩家活躍，分數可能偏樂觀；(2) 5／10 局題是從同一組 20 局切出來的，跨大小不獨立，只能各自看；(3) 門檻建在 posterior-mean index 上，它幾乎是整數（最佳 profile），解析度受 18 個 profile 的格點限制；(4) 對手掃描的「我方」固定在名義中心（D=11k…6D=6d），不是實測峰值，所以敏感度是在偏離峰值的位置量的。
- **H1 判讀：部分成立。** 序數訊號強、對未見玩家有效；但 1D|2D、3D|4D 兩個邊界弱，且多給局數幾乎沒幫助 → 下一步的分數要從「模型」來（L1b 用本資料標籤訓練、或用整條 18 維似然曲線當特徵學 L3/L4），不是從「更多證據」來。

### 2.5 步驟 0（pilot 之前）
- 0A 一致性：同一局面、同 profile（完整重現引擎 metadata，不是只填 rank）、同權重（確認 SWA 與否）、固定對稱，比較 Python 直讀與 analysis engine 的目標落子 log p。涵蓋黑白、各階段、提子、打劫。用 FP32 `log_softmax`，合法步語意（引擎非法步 = −1）要對齊。
- 0B 吞吐：分開記純推論（warm-up、同步）與端到端（SGF→特徵→log p）。只讀 policy 不等於沒算 value head。

### 2.6 Ablation 清單（便宜優先）
執色平面｜不截開局 vs 最後 250 vs 分段取樣｜明確局長特徵（兩種表示都給）｜只用局長統計的 baseline｜只計我方 vs 兩方都算（檢驗「對手等級相關」假設）｜按局取樣 vs 等級→玩家→局分層取樣（不刪資料）｜鄰級 label smoothing（當序數正則化，不當指標對齊）｜聚合方式三選一。

### 2.7 聚合器實驗（含 CTM）— 排在 pilot 之後的可選支線
- 要回答的問題：**給定相同的分局、分階段證據，哪種聚合器最能分辨未見玩家的相鄰等級？** 不是「CTM vs ResNet」。CTM 的定位是查詢級的條件式整合「修正器」，不是棋力模型本體。
- 證據表示：每手 (ll_{g,t,r})（18 或 10 個 profile）→ 相對分數 (d=ll-mathrm{mean}_rll)（或 (ll-mathrm{LSE}_rll)）加保留 (arll)；**不做**單位長度正規化。四個手數區段 1–40／41–100／101–180／181+，每段每 profile 存 mean／std／Q10／Q90、有效目標手數、平均 surprisal；缺段用 mask。每局先編成一個 128 維 token，查詢 = 5–20 個 token；permutation-invariant、不加局索引位置編碼；明確輸入 (og m)、(og n_{target})、各段手數（softmax attention 對重複 token 不變，自己不會知道證據量）。注意 (n_{g,p}) 會帶入局長訊號（見 2.6），要 ablate。
- 梯子（同一輸入、同一固定查詢集、以玩家為單位成對比較）：(0) 校準後似然累加；(1) 穩健統計：每局 LLR 截斷／trimmed sum／Student-t（直接針對「一局撐起全部證據」的誤設）；(2) 分階段統計 + logistic／MLP；(3) DeepSets／小型 Set Transformer；(4) cross-attention + GRU，固定 8 ticks；(5) CTM（latent 256、neuron history 8、NLM hidden 16、4 heads、固定 8 ticks、action／output 各約 512 sync pairs），輸出對基準 logits 的修正 (Delta)，正則化趨零。
- 訓練／選模：固定 (T=8)，(mathrm{CE}(z^T)+gamma|Delta^T|^2)；原始 CTM 的 min-loss／max-certainty 雙 tick loss 當對照。不把各 tick 輸出相乘或對實際落子取 (max_	au) 當似然（不是正規化分布）；不在驗證集上挑 tick；訓練 8 跑 16 另列為「超出訓練長度的泛化」。
- 判讀：分階段特徵讓所有模型都升 → 貢獻在輸入表示，不留 CTM；(4) 與 (5) 都優於靜態聚合 → 反覆讀取有價值但不能歸因同步機制；(5) 穩定優於同輸入、近似成本的 (4) → 才深究。
- 統計效力：200 人 pilot 只能看方向，不能裁決聚合器間的小差距。成對差異 SD 若約 0.3，要偵測 0.02 的得分差需約 900 位玩家 → 用整個 fold（約 1,600 人）或多 fold，玩家為重抽樣單位。
- 成本（評論沒算）：聚合器訓練集需要訓練 fold 玩家**所有**棋局的 Human SL 特徵。全量 1.11 億手 × 18 profile ≈ 2×10⁹ 次，以 4,296/秒/卡約兩卡 65 小時。可行版本：每級 400 位玩家 × 30 局 × 18 profile ≈ 2.4×10⁸ 次 ≈ 單卡 15 小時；或 pilot 後縮到 10 個 profile。
- 自訓 policy 的特徵必須 out-of-fold（例：policy 用 fold 1–3 訓練、聚合器用 fold 4 的特徵、fold 0 評估）；凍結的 Human SL 少一層訓練內外差異，更適合先比聚合器。
- 順序：pilot → (0)(1)(2) 幾乎免費 → (3) → 只在 (3) 顯示集合互動有益時做 (4)(5)。相對主模型（自訓條件落子模型、校準、提交管線），這整節是可選支線；11/24 前時間不夠就砍。
- 2026-10-19 起：聚合器比較用 folds 1–4 CV 的合併成對差值與穩定性門檻；fold 0 只評一次（`scripts/ladder_eval_fold0.py`）
- 歸因對照（只報告、不作門檻，門檻不變）：`r3_pma`（Set Transformer 拿掉 SAB、只留 PMA 池化，局與局之間沒有交互）和 `r4_xattn_t1`（cross-attention + GRU 只跑 1 tick）。`G_set` 成立只能說「注意力集合模型勝過 mean/max 和沒有編碼器的對照」；要說是 SAB（局與局的交互）的功勞，還要 `r3_settf` > `r3_pma`。`G_iter` 也一樣：要說是反覆讀取的功勞，還要 `r4_xattn` > `r4_xattn_t1`。
- 訓練長度（2026-10-19 使用者核定，plan Task 10）：集合 rung 改用內層提早停止。從訓練題依玩家分組、按段位分層留出 10% 玩家當內層驗證（`settrain.inner_players`，`INNER_SEED` 2026，同一個 split 的所有 rung、seed 用同一批），最多 10 個 epoch，每 1/4 epoch 算一次內層的類別平衡 CE（最後一個 tick，不含 γ 項），保留最好那次的權重（含第 0 次，即 Δ ≡ 0，所以內層 CE 不會比基準差），連續 8 次沒改善就停；不在全部訓練玩家上重訓，集合 rung 因此少用約 10% 的訓練玩家，對它們略為保守。原因：每位訓練玩家的 ≤20 局在他的 32 題裡反覆出現、標籤相同，原本固定 20 epoch 時集合模型背熟「這幾局 → 這個段位」——CUDA 冒煙的訓練 CE 1.27→0.22；診斷（split 1）中 `r3_settf` 的驗證分數隨 epoch 1／2／3／5／20 為 0.6245／0.6185／0.6010／0.5979／0.5521（基準 0.6136），看過的訓練玩家則升到 0.89；γ = 1 把 Δ 壓成 0，weight decay 0.1 沒有幫助。驗證 fold 仍然只用來報分；靜態 rung 不變。
- **梯子結果（2026-10-19，folds 1–4 CV）**：四個事先指定的門檻都不成立。7 個讀分段每局 token 的集合 rung（含探索性的 (4)(5)；不含退化的 `r5_ctm_minmax`）對 `r2_mlp24` 都是 +0.0054 到 +0.0087（報告用的比較），彼此最多再差約 0.003；前三名 `r5_ctm` 0.6279、`r4_xattn_t1` 0.6277（兩者探索性）與 `r3_pma` 0.6274 事後兩兩差 ≤ 0.0005、CI 都跨 0。使用者選三者中最簡單的 `r3_pma` 做 fold 0 一次評估：0.6233 [0.6093, 0.6371]，對同次重訓的基準 +0.0121 [+0.0051, +0.0189]，對現行提交模型 +0.0102 [+0.0032, +0.0171]。協議：每個 rung 都跑 folds 1–4 的留一 fold CV（4 個驗證 fold 合併 39,327 題），訓練題每人 32 題；集合 rung 用「訓練長度」那條的內層提早停止、seeds 0／1／2（逐題分數先在 seed 間平均再算指標），靜態 rung 只做一次確定性擬合。程式碼都是 git `7ab952f`（工作樹乾淨；`r2_mlp24@pp8` 例外，見第一張表下）、torch 2.14.1+cu126；集合 rung 在 NVIDIA H200 上訓練，靜態 rung 在 CPU 上擬合；摘要的 `mixed_versions` 為空（沒有 label 混用不同版本的程式碼）。時間（nano4 單張 H200）：靜態梯子＋(3)＋對照 51 分鐘、(4)(5) 24 分鐘、局長 ablation 28 分鐘、fold 0 4.5 分鐘；log 在 nano4 的 `logs/nano4/`（檔名以 `ladder3`、`ladder45`、`ladnocnt`、`ladf0`、`ladpp8` 開頭，不進 git）。
- **門檻（plan 的 Global Constraints；比較對象事先指定、不取 max）**：每組比較要同時滿足 ① seed 平均後合併成對差的 95% CI 下界 > 0（以玩家為單位 bootstrap，n = 5000）；② 3 個 seed 各自的點差都 > 0；③ 4 個驗證 fold 至少 3 個點差 > 0；④ 點差 ≥ 0.005。Nadeau–Bengio（NB）校正區間（變異數 × 2.33）只當保守參考，不作門檻。

| rung | 內容 | 合併分數 | 類別平衡 NLL | seed SD | 對 `r2_mlp24` 差 [95% CI] |
| --- | --- | --- | --- | --- | --- |
| `r0_sum` | (0) 校準後似然累加（§2.3 的線性化形式）→ softmax 迴歸 | 0.6008 | 1.4431 | — | −0.0184 [−0.0228, −0.0140] |
| `r1_clip` | (1) 每局相對曲線以中位數為中心截斷後平均 → softmax 迴歸 | 0.6017 | 1.4564 | — | −0.0176 [−0.0221, −0.0130] |
| `r1_trim` | (1) 每局相對曲線的 20% trimmed mean → softmax 迴歸 | 0.6007 | 1.4603 | — | −0.0186 [−0.0231, −0.0139] |
| `r2_mlp24` | (2) §2.8 的 24 維題目特徵 + MLP h32（現行提交模型的特徵與模型，這裡每人 32 題）；集合 rung 的基準 | 0.6192 | 1.3196 | — | （參考） |
| `r2_staged` | (2) 分段 token 的題內平均 + 題目層級輸入（log m、log n、各段手數）+ 24 維 → MLP h64 | 0.6023 | 1.5652 | — | −0.0170 [−0.0230, −0.0111] |
| `r2_whole` | 同上，改用整局 token | 0.6176 | 1.3562 | — | −0.0016 [−0.0060, +0.0026] |
| `r2_length_only` | 只有局長統計 → MLP h32 | 0.2265 | 2.2805 | — | −0.3927 [−0.4012, −0.3840] |
| `r2_pooled_delta` | 對基準的修正 Δ：標準化 token 的題內平均 + 題目層級輸入 → MLP；沒有每局編碼器、沒有交互 | 0.6247 | 1.3164 | 0.0016 | +0.0055 [+0.0027, +0.0082] |
| `r3_null` | Δ 只看題目層級輸入（純再校準） | 0.6188 | 1.3209 | 0.0003 | −0.0004 [−0.0010, +0.0001] |
| `r3_deepsets` | (3) 每局編碼器 → mean＋max → Δ | 0.6254 | 1.3119 | 0.0010 | +0.0062 [+0.0037, +0.0086] |
| `r3_deepsets_whole` | 同上，改用整局 token | 0.6204 | 1.3179 | 0.0002 | +0.0012 [−0.0002, +0.0026] |
| `r3_settf` | (3) 每局編碼器 → 2 層 SAB → PMA → Δ | 0.6246 | 1.3113 | 0.0015 | +0.0054 [+0.0027, +0.0080] |
| `r3_pma` | 〔歸因對照〕`r3_settf` 拿掉 SAB，只留 PMA | 0.6274 | 1.3066 | 0.0009 | +0.0082 [+0.0050, +0.0113] |
| `r4_xattn` | 〔探索性〕(4) cross-attention + GRU，8 ticks | 0.6253 | 1.3109 | 0.0008 | +0.0061 [+0.0035, +0.0087] |
| `r4_xattn_t1` | 〔歸因對照、探索性〕同上，只跑 1 tick | 0.6277 | 1.3099 | 0.0020 | +0.0085 [+0.0055, +0.0115] |
| `r5_ctm` | 〔探索性〕(5) CTM，8 ticks | 0.6279 | 1.3048 | 0.0013 | +0.0087 [+0.0055, +0.0117] |
| `r5_ctm_minmax` | 〔探索性、退化〕(5) 改用原始 min-loss／max-certainty 損失；12 次訓練（4 個 split × 3 個 seed）都是 Δ ≡ 0 | 0.6192 | 1.3196 | 0.0000 | +0.0000 [+0.0000, +0.0000] |
| `r5_ctm_unet` | 〔2026-10-24 追加〕`r5_ctm` 的 synapse 換成原始實作預設的 depth-4 U-Net（467,898 參數） | 0.6275 | 1.3072 | 0.0006 | +0.0083 [+0.0053, +0.0112] |
| `r5_ctm_minmax_pc` | 〔2026-10-24 追加、退化〕`r5_ctm_minmax` 的模型與 loss，選模與預測改取最確定 tick；12 次訓練仍全部 Δ ≡ 0 | 0.6192 | 1.3196 | 0.0000 | +0.0000 [+0.0000, +0.0000] |
| `ens_top3` | 〔2026-10-24 追加〕`r3_pma`／`r4_xattn_t1`／`r5_ctm` 逐 seed 機率平均後以期望得分決策（不訓練） | 0.6306 | 1.2993 | 0.0005 | +0.0113 [+0.0083, +0.0142] |

- 表外：(1) 只做截斷與 trimmed（規格把三種列成擇一），Student-t 沒做。`r2_mlp24@pp8`（每人 8 題，即現行提交模型的訓練量）0.6200，對 `r2_mlp24` +0.0008 [−0.0011, +0.0026]。它是在 `2854c4e`（內層提早停止之前的 commit，工作樹乾淨）上跑的；那次修改只動集合 rung 的訓練，靜態 rung 的重現測試仍逐位元相同，它 4 個 fold 的分數也與現行提交模型的 CV 逐位元相同（0.6131／0.6100／0.6312／0.6258）。超出訓練長度（訓練 8 ticks、推論 16，不參與選擇）：`r4_xattn@T16` 0.6252（+0.0060 [+0.0033, +0.0085]）、`r5_ctm@T16` 0.6187（−0.0005 [−0.0049, +0.0038]，seed SD 0.0026）；`r5_ctm_minmax` 的 `@T16`、`@certain`（同一個模型逐題改取確定性最高的 tick，不參與選擇）都等於基準。

| 門檻 | 比較 | 點差 [95% CI] | 3 個 seed 的點差 | fold > 0 | 沒過的條件 |
| --- | --- | --- | --- | --- | --- |
| `G_rep` | `r2_staged` vs `r2_whole` | −0.0153 [−0.0211, −0.0097] | −0.0153（單次擬合） | 0/4 | ①②③④ |
| `G_rep` | `r3_deepsets` vs `r3_deepsets_whole` | +0.00498 [+0.0026, +0.0073] | +0.0058／+0.0050／+0.0042 | 4/4 | ④ |
| `G_set` | `r3_settf` vs `r3_deepsets` | −0.0008 [−0.0027, +0.0011] | −0.0019／+0.0007／−0.0013 | 1/4 | ①②③④ |
| `G_set` | `r3_settf` vs `r2_pooled_delta` | −0.0001 [−0.0025, +0.0024] | −0.0012／+0.0033／−0.0024 | 2/4 | ①②③④ |
| `G_iter` | `r4_xattn` vs `r3_settf` | +0.0007 [−0.0013, +0.0027] | +0.0004／−0.0012／+0.0031 | 3/4 | ①②④ |
| `G_iter` | `r5_ctm` vs `r3_settf` | +0.0033 [+0.0013, +0.0053] | +0.0032／+0.0006／+0.0061 | 4/4 | ④ |
| `G_sync` | `r5_ctm` vs `r4_xattn` | +0.0026 [+0.0003, +0.0048] | +0.0029／+0.0018／+0.0030 | 3/4 | ④ |

- 四個門檻都不成立；`G_iter`、`G_sync` 是 `--force` 之後的探索性結果。只差 ④ 的三組，NB 參考區間依序是 [+0.0014, +0.0085]、[+0.0002, +0.0064]、[−0.0008, +0.0059]（最後一組跨 0）；`G_rep` 那組的 +0.00498 四捨五入是 0.0050，但仍在 0.005 之下。
- **輸入表示（`G_rep` 不成立）**：分段證據沒有讓所有模型都升，依判讀規則不能說貢獻在輸入表示。靜態 MLP 改用分段 token 大幅變差（`r2_staged` vs `r2_whole` 4 個 fold 全負；NLL 1.5652 vs 1.3562），可能是固定 epoch、沒有提早停止的 MLP 在高維輸入上過擬合，不一定是表示本身的問題。DeepSets 的分段版比整局版高 +0.00498 [+0.0026, +0.0073]，①②③ 成立、④ 不成立，依門檻不算成立。分段版對基準 +0.0062，整局版 `r3_deepsets_whole` 只有 +0.0012 [−0.0002, +0.0026]；但整局版的 12 次訓練有 4 次保留第 0 次（Δ ≡ 0），分段版的優勢也可能有一部分來自每段明確的手數欄（見局長 ablation）。
- **局與局的交互（`G_set` 不成立）**：`r3_settf` 既沒勝過 `r3_deepsets`，也沒勝過無編碼器、無交互的 `r2_pooled_delta`。歸因對照指向反方向：`r3_settf` vs `r3_pma` −0.0028 [−0.0050, −0.0006]（3 個 seed、4 個 fold 全負，但未達 0.005，NB 參考 [−0.0062, +0.0005] 跨 0），加 2 層 SAB 的點估計較差，SAB 沒有帶來可歸因的增益；`r3_pma` vs `r3_deepsets` +0.0020 [−0.00004, +0.0040]（3 個 seed 正、3/4 fold 正），注意力池化比 mean／max 頂多小幅有利。另兩組只報告的比較：`r3_deepsets` vs `r2_pooled_delta` +0.0007 [−0.0013, +0.0028]，沒有偵測到每局非線性編碼器的貢獻（CI 上界 +0.0028，不能排除與架構間差距同量級的效果）；`r2_staged` vs `r2_pooled_delta` −0.0224 [−0.0276, −0.0174]，同樣的分段 token 平均，當成基準的修正並提早停止，遠勝從頭訓練的靜態 MLP——訓練框架的影響遠大於聚合架構。
- **反覆讀取（`G_iter` 不成立）**：判讀規則的前提「(4) 與 (5) 都優於靜態聚合」事先指定為 `G_iter`（兩者都勝過 `r3_settf`），不成立：`r4_xattn` vs `r3_settf` 只有 +0.0007 [−0.0013, +0.0027]，`r5_ctm` vs `r3_settf` 未達 0.005。兩者對靜態的 `r2_mlp24` 雖有 +0.0061、+0.0087，但那是報告用的比較、不是門檻，而且 (4)(5) 是探索性的。對照也顯示不是反覆讀取的功勞：`r4_xattn` vs `r4_xattn_t1` −0.0024 [−0.0042, −0.0005]（NB 參考 [−0.0052, +0.0005]；seed、fold 方向不一致），8 ticks 沒有比 1 tick 好；對 `r2_mlp24`，不反覆讀取的 `r4_xattn_t1`（+0.0085）、`r3_pma`（+0.0082）和反覆讀取 8 ticks 的 `r5_ctm`（+0.0087）一樣高（事後兩兩比較的 CI 都跨 0）。資料不支持反覆讀取有價值。
- **同步機制（`G_sync` 不成立）**：`r5_ctm` 對 `r4_xattn` 的 ①②③ 成立（CI 下界 +0.0003、3 個 seed 正、3/4 fold 正），④ 不成立（+0.0026 < 0.005），NB 參考區間跨 0；對 1 tick 的 `r4_xattn_t1` 事後只差 +0.0002。推論改跑 16 ticks 時 `r5_ctm` 的增益全部消失，`r4_xattn@T16` 則不變（數字見第一張表下）。依規則「(5) 穩定優於同輸入、近似成本的 (4) 才深究」，資料不支持深究 CTM 的同步機制。
- **再校準（`r3_null` vs `r2_mlp24`，只報告）**：−0.0004 [−0.0010, +0.0001]（3 個 seed 全負；12 次訓練有 7 次保留第 0 次，即 Δ ≡ 0）。只用題目層級輸入再校準基準沒有增益，集合 rung 的增益要靠讀每局 token。
- **事後比較（不是事先指定，只報告；不在 `cv_summary.json` 裡，用同一個 `aicup.ladder._compare`、n_boot = 5000 算；這 6 組數字沒有存進 repo 的任何檔案，重算要用留在 nano4 的逐題 `.npz`）**：前三名分不開：`r5_ctm` vs `r3_pma` +0.0005 [−0.0013, +0.0024]、`r5_ctm` vs `r4_xattn_t1` +0.0002 [−0.0020, +0.0023]、`r4_xattn_t1` vs `r3_pma` +0.0003 [−0.0018, +0.0025]，CI 都跨 0、3 個 seed 方向不一致。另挑的三組前三名對中段：`r3_pma` vs `r2_pooled_delta` +0.0027 [+0.0003, +0.0052]、`r4_xattn_t1` vs `r3_settf` +0.0031 [+0.0010, +0.0053]、`r5_ctm` vs `r3_deepsets` +0.0025 [+0.0004, +0.0045]，CI 在 0 以上、seed 全正，但都不到 0.005（只差 ④）。同樣是前三名對中段，`r3_pma` vs `r3_deepsets` 的 CI 跨 0、`r4_xattn_t1` 對 `r4_xattn` 的 seed 方向不一致（見「局與局的交互」與「反覆讀取」兩條），所以不能說前三名整體與中段分得開。
- **資料支持的**：讀分段每局 token、在基準上加修正 Δ、用內層提早停止的集合 rung，對 `r2_mlp24` 都有增益：事先列入的 `r2_pooled_delta`、`r3_deepsets`、`r3_settf` 與歸因對照 `r3_pma` 是 +0.0054 到 +0.0082，探索性的 `r4_xattn`、`r4_xattn_t1`、`r5_ctm` 是 +0.0061 到 +0.0087；7 組的 CI 下界都 > 0，3 個 seed、4 個 fold 也全正（報告用的比較，不是門檻；事後描述）。架構之間最多再差約 0.003。各成分沒有分開檢定：Δ 框架與內層提早停止只一起比過（`r2_staged` vs `r2_pooled_delta`），分段 token 的檢定是 `G_rep`（不成立）。(1) 與 (0) 的標題分數相差不到 0.001（`r1_clip` 0.6017、`r1_trim` 0.6007 vs `r0_sum` 0.6008；摘要沒有這組成對比較）。**不支持的**：局與局的交互、反覆讀取、CTM 的同步機制帶來額外價值；也不能說貢獻在輸入表示（靜態 MLP 用分段反而變差）。
- **局長 ablation（`--no-counts`）**：跑了 `r5_ctm`、`r3_pma`、`r3_deepsets`／`r3_deepsets_whole`（分段／整局）與靜態的 `r2_mlp24`、`r2_whole`；前三名中的 `r4_xattn_t1` 沒有跑。集合 rung 的 `@nocounts` 用同樣拿掉長度欄的基準。拿掉明確的長度欄後（不算 `@T16`）分數變化在 −0.0026 到 +0.0044 之間。`r3_pma`、`r5_ctm` 對有長度欄的 `r2_mlp24`（0.6192）的領先仍在（`r3_pma@nocounts` +0.0072 [+0.0036, +0.0106]、`r5_ctm@nocounts` +0.0083 [+0.0047, +0.0117]），對同樣拿掉長度欄的 `r2_mlp24@nocounts`（0.6186）差距更大。整局 token 沒有缺段模式，是比較乾淨的一組：`r3_deepsets_whole` 幾乎不變，`r2_whole` 反而升。另一端（只有局長）是第一張表的 `r2_length_only` 0.2265。

| rung | 原本 | `@nocounts` | `@nocounts` − 原本 [95% CI] |
| --- | --- | --- | --- |
| `r2_mlp24` | 0.6192 | 0.6186 | −0.0006 [−0.0029, +0.0018] |
| `r2_whole` | 0.6176 | 0.6220 | +0.0044 [+0.0009, +0.0078] |
| `r3_deepsets` | 0.6254 | 0.6229 | −0.0026 [−0.0048, −0.0002] |
| `r3_deepsets_whole` | 0.6204 | 0.6199 | −0.0006 [−0.0029, +0.0017] |
| `r3_pma` | 0.6274 | 0.6264 | −0.0010 [−0.0035, +0.0014] |
| `r5_ctm` | 0.6279 | 0.6276 | −0.0004 [−0.0027, +0.0019] |
| `r5_ctm@T16` | 0.6187 | 0.6160 | −0.0027 [−0.0058, +0.0002] |

- 所以只能說 `r3_pma`、`r5_ctm` 對 `r2_mlp24` 的領先不靠明確的長度欄（`r4_xattn_t1` 沒跑）。排名並非不變：`r2_whole` +0.0044 [+0.0009, +0.0078]，在這 6 個 rung 中從最低升到第四；`r3_deepsets` −0.0026 [−0.0048, −0.0002]、整局版只有 −0.0006，DeepSets 分段減整局的標題分數差從 +0.0050 縮到約 +0.0030（0.6229 − 0.6199；沒有成對 CI），分段版的優勢可能有一部分來自每段明確的手數欄。也不能說模型沒用到局長：`counts=False` 只拿掉明確的長度輸入（token 的 n_eff／present／log1p(n_g)、題目層級輸入中 log m 以外各欄（log n 與各段手數）、24 維特徵的 `log_moves`），缺段的全零區塊仍帶局長訊號（(0) 的 sum 特徵的 /√n、/n 縮放也是，但這次沒跑 (0)）。
- **fold 0 一次評估（`results/ladder/fold0_eval.json`）**：`r3_pma` 在 folds 1–4 重訓（每人 32 題、203,328 題；seeds 0／1／2，逐題分數先在 seed 間平均），在 §2.8 同一組 fold 0 題（1,594 位玩家、9,872 題）上 **0.6233 [0.6093, 0.6371]**（玩家層級 bootstrap，n = 5000）；5／10／20 局 0.5831／0.6308／0.6561；各 seed 0.6235／0.6196／0.6270（SD 0.0037）；類別平衡 NLL 1.3295、Brier 0.6550。
- 對基準 **+0.0121 [+0.0051, +0.0189]**：基準是同一次重訓中 Δ ≡ 0 的輸出，也就是 `r2_mlp24` 在同一批訓練題上重訓（0.6113；5／10／20 局 0.5765／0.6195／0.6377）；差值 5／10／20 局 +0.0066／+0.0112／+0.0184，3 個 seed +0.0122／+0.0083／+0.0157；比 CV 的 +0.0082 大，但 fold 0 的 CI 涵蓋 CV 值。對現行提交模型（production：`results/models/rank_model.json`，每人 8 題訓練）**+0.0102 [+0.0032, +0.0171]**；5／10／20 局 +0.0062／+0.0092／+0.0153，3 個 seed +0.0104／+0.0065／+0.0139。提交模型在這裡是 0.6131（5／10／20 局 0.5769／0.6216／0.6408），重現 §2.8 的 fold 0 結果（0.613；0.577／0.622／0.641）。對提交模型的差同時含 rung 與訓練題數；訓練題數單獨的效應是 CV 的 `r2_mlp24@pp8` 對 `r2_mlp24` +0.0008 [−0.0011, +0.0026]，也就是每人 32 題相對 8 題 −0.0008 [−0.0026, +0.0011]（CI 跨 0）。fold 0 上每人 32 題的基準也比提交模型低 0.0018（0.6113 vs 0.6131，沒有成對 CI），所以對提交模型的差（+0.0102）小於對基準的差（+0.0121）。
- **來源與 fold 0 的狀態**：git `7ab952f`（乾淨）、NVIDIA H200、torch 2.14.1+cu126，訓練相關模組的 sha256（`c138c070…`）與 CV 相同；`chosen_from` 是 `results/ladder/cv_summary_at_fold0_choice.json`（sha256 `f61051dc…`），即使用者選 rung 時看的摘要，和最終的 `cv_summary.json` 只差後來加的 `@nocounts` rung、比較與對應的基準診斷。fold 0 對這個梯子已經用掉：`scripts/ladder_eval_fold0.py` 在 `results/ladder/fold0_eval.json` 存在時拒絕執行，之後在 rung 之間的任何選擇只能靠 folds 1–4 CV。這些仍是 fold 0 題上的離線分數，§2.8「還不能說的」同樣適用。
- **使用者決定（2026-10-19）**：(1) CUDA 冒煙顯示集合模型背熟訓練玩家後，改用內層提早停止（見「訓練長度」那條）；(2) `G_set` 不成立，依 §2.7「順序」本來不做 (4)(5)（`scripts/ladder_cv.py` 在 `G_set` 不成立時拒絕跑 r4／r5），使用者決定加 `--force` 照跑，結果標為探索性；(3) fold 0 選 `r3_pma`（歸因對照只是不作門檻，摘要裡它的 `report_only` 是 false，可以被選）：CV 前三名打平（事後兩兩差 ≤ 0.0005、CI 都跨 0），而 `r3_pma` 是三者中最簡單的（258,058 個參數，`r4_xattn_t1` 458,250、`r5_ctm` 437,514；沒有局與局的交互，也沒有反覆讀取）；(4) `r5_ctm_minmax` 記為已知限制，不重跑。
- **已知限制：`r5_ctm_minmax`**：在內層提早停止下，12 次訓練全部保留第 0 次（best_epoch 都是 0.00，Δ ≡ 0）。它的損失訓練的是 min-loss 與 max-certainty 那兩個 tick，內層選模卻看最後一個 tick，最後一個 tick 的內層 CE 從未低於第 0 次。所以「原始 CTM 雙 tick 損失」這個對照照目前的跑法無法解讀；摘要裡 `r5_ctm_minmax@certain` vs `r5_ctm` 的 −0.0087 其實只是基準對 `r5_ctm`。2026-10-24 的 `r5_ctm_minmax_pc` 把選模與預測都改到最確定 tick（見追加項），12 次訓練仍全部保留第 0 次、訓練 loss 與 `r5_ctm_minmax` 逐值相同：這個對照現在可以解讀，結論是原始雙 tick 損失學到的東西在這個資料量下推廣不到未見玩家。
- **其他注意事項**：集合 rung 在 GPU 上訓練、沒有開 deterministic 模式，不保證逐位元重現；但實測同一個 seed 在不同 H200 節點重跑（冒煙 `ladsmoke2` 與正式 CV），`r3_settf`、`r3_deepsets`、`r5_ctm` seed 0 的逐 fold 分數與 train loss 都逐位元相同；seed 之間的差異見表的 seed SD。靜態 rung 是單次確定性擬合，所以 `G_rep` 靜態那組的條件 ② 只有一個點差，比集合那組寬鬆；內層留出對集合 rung 的影響見「訓練長度」那條；CV 的 168 次集合訓練（14 個 label × 4 個 split × 3 個 seed）都提早停止，best_epoch 在 0–3.50 之間（135 次在 0.25–1.00、27 次是 0），fold 0 那 3 次沒有記錄 best_epoch；上面「不支持」的結論只限於這個資料量與訓練協議。
- **檔案**：隨這個結果 commit 的是 `results/ladder/cv_summary.json`、`results/ladder/cv_summary_at_fold0_choice.json`、`results/ladder/fold0_eval.json`；逐題的 CV `.npz`（`results/ladder/cv/`）不進 git，留在 nano4。
- **接下來**：`r3_pma` 是 L3 聚合器的候選，**還沒有接進提交管線**，也還沒有決定要不要接。要接需要：評分時另存每局的分段向量（`scripts/predict_submission.py` 仍用 `rank_model.json`，每局只留 18 維總和與手數）、重訓 `r3_pma` 並存權重（fold 0 那次沒有存）、在提交腳本加集合模型的路徑。待補：前三名中 `r4_xattn_t1` 的 `--no-counts` 還沒跑。
- **2026-10-24 追加（issue #2，待辦第 3 項）——設計**：三件事各對 `r5_ctm` 與 `r3_pma` 事先指定比較（`aicup.ladder.FOLLOWUP_COMPARISONS`，只報告、不作門檻，四個條件照算；對照表在 PR）：(i) `ens_top3`——`r3_pma`／`r4_xattn_t1`／`r5_ctm` 逐 seed 機率平均、以期望得分決策重算每題分數（`scripts/ladder_ensemble.py`，不訓練）；(ii) `r5_ctm_minmax_pc`——`r5_ctm_minmax` 的模型與 loss，但選模（內層 CE）與預測都取最確定 tick（`TrainCfg.pick = "certain"`），與原始碼「評估用最確定 tick」一致；上表的 `r5_ctm_minmax` 與「已知限制」那段原樣保留，它的檔案也不動；(iii) `r5_ctm_unet`——`r5_ctm` 的 synapse 換成原始實作預設的 depth-4 U-Net（`CTM(synapse_depth=4)`，467,898 參數 vs 437,514），其餘相同。原始實作比對（純讀程式碼）見 issue #2：同步表示、decay clamp、z0／A0 起始、pair 抽法、query 來源都與原始碼等價；與原始**預設**不同的是 synapse 深度與 tick／history／訓練步數的規模，NLM 用 ReLU 而非 GLU。同一討論的兩點結論：(a) 分 rung 的 best_epoch 表（nano4，168 次訓練）顯示 r4／r5 的 best_epoch 中位數 0.5，與 `r3_pma` 相同，`r3_settf` 反而最晚（0.75），168 次全部由 patience 觸發停止——內層提早停止對多 tick 模型沒有不公平的證據；lr 排程的問題（cosine 跨 10 epoch、權重保留在 ≤ 1 epoch，等於沒有退火、也沒有 warmup）對所有集合 rung 一樣，要修就整個梯子重跑。(b) 1 tick 的 CTM 在結構上與 `r3_pma` 同類：action sync 只含 z0、是常數，query 因此是學出來的固定向量，即單 seed 的注意力池化；輸出是 [z0, z1] 上的二次同步讀出，所以是同類而非相同。`r5_ctm`（8 ticks）對 `r3_pma` +0.0005 [−0.0013, +0.0024]，多 7 個 tick 沒有加東西，比 `r4_xattn` vs `r4_xattn_t1` 更直接。
- **追加項結果（2026-10-24，nano4 Slurm job 610067／610089／610090，H200，commit `b66bb02`（`r5_ctm_unet`）與 `4386185`，兩者的訓練模組 `code_sha256` 相同；`cv_summary.json` 為 `58003a3`，既有 29 個 rung、47 組比較與 4 個門檻逐項不變）**：三件事都跑完。`r5_ctm_unet` 17.5 分鐘、`r5_ctm_minmax_pc` 15.5 分鐘、`ladder_ensemble.py` 10 分鐘（CPU）；完整 pytest 在 nano4 200 passed、1 skipped。

| rung | 合併分數 | 5／10／20 局 | 類別平衡 NLL | Brier | seed SD |
| --- | --- | --- | --- | --- | --- |
| `ens_top3` | 0.6306 | 0.5894／0.6381／0.6643 | 1.2993 | 0.6442 | 0.0005 |
| `r5_ctm` | 0.6279 | 0.5875／0.6346／0.6616 | 1.3048 | 0.6465 | 0.0013 |
| `r4_xattn_t1` | 0.6277 | 0.5878／0.6347／0.6607 | 1.3099 | 0.6483 | 0.0020 |
| `r5_ctm_unet` | 0.6275 | 0.5860／0.6363／0.6603 | 1.3072 | 0.6475 | 0.0006 |
| `r3_pma` | 0.6274 | 0.5868／0.6344／0.6610 | 1.3066 | 0.6471 | 0.0009 |
| `r5_ctm_minmax_pc` | 0.6192 | 0.5822／0.6246／0.6509 | 1.3196 | 0.6512 | 0.0000 |
| `r2_mlp24` | 0.6192 | 0.5822／0.6246／0.6509 | 1.3196 | 0.6512 | — |

| 比較（事先指定） | 點差 [95% CI] | NB 參考 | 3 個 seed | 4 個 fold | ①②③④ |
| --- | --- | --- | --- | --- | --- |
| `ens_top3` vs `r5_ctm` | +0.0027 [+0.0013, +0.0041] | [+0.0005, +0.0048] | +0.0034 +0.0035 +0.0012 | +0.0009 +0.0029 +0.0036 +0.0033 | ✓✓✓✗ |
| `ens_top3` vs `r3_pma` | +0.0032 [+0.0017, +0.0047] | [+0.0009, +0.0056] | +0.0047 +0.0028 +0.0021 | −0.0001 +0.0036 +0.0039 +0.0053 | ✓✓✓✗ |
| `r5_ctm_minmax_pc` vs `r5_ctm` | −0.0087 [−0.0117, −0.0055] | [−0.0133, −0.0039] | −0.0086 −0.0075 −0.0100 | −0.0070 −0.0089 −0.0100 −0.0088 | ✗✗✗✗ |
| `r5_ctm_minmax_pc` vs `r3_pma` | −0.0082 [−0.0113, −0.0050] | [−0.0130, −0.0034] | −0.0073 −0.0081 −0.0091 | −0.0080 −0.0082 −0.0098 −0.0068 | ✗✗✗✗ |
| `r5_ctm_unet` vs `r5_ctm` | −0.0004 [−0.0021, +0.0014] | [−0.0030, +0.0023] | −0.0002 +0.0002 −0.0011 | −0.0014 +0.0017 +0.0005 −0.0022 | ✗✗✗✗ |
| `r5_ctm_unet` vs `r3_pma` | +0.0001 [−0.0018, +0.0020] | [−0.0028, +0.0031] | +0.0010 −0.0005 −0.0002 | −0.0025 +0.0023 +0.0008 −0.0002 | ✗✗✗✗ |

- **判讀**：(i) **U-Net synapse 沒有用**：`r5_ctm_unet` 對 `r5_ctm` −0.0004 [−0.0021, +0.0014]、對 `r3_pma` +0.0001 [−0.0018, +0.0020]，3 個 seed 方向不一致；issue #2 列的最後一個「CTM 可能被低估」的實作疑點排除，`r5_ctm_unet@T16` 0.6162 同樣在推論拉長時變差。best_epoch 0.50–1.50，12 次全由 patience 觸發停止。(ii) **原始雙 tick 損失在選模修正後依然退化**：`r5_ctm_minmax_pc` 12 次訓練全部保留第 0 次（Δ ≡ 0，分數即基準 0.6192），訓練 loss 與 `r5_ctm_minmax` 逐值相同（`pick` 只影響選模）而且確實下降（seed 0 各 split 從 1.10–1.15 降到 0.86–0.93），但最確定 tick 上的內層 CE 從未低於基準——學到的東西推廣不到未見玩家，提案所說「max-certainty tick 容易過度自信」在此應驗。這組對照現在可以解讀，結論是在這個資料量下不能用。(iii) **`ens_top3` 是目前 CV 最高的**：0.6306，NLL 與 Brier 也是全場最好；對 `r5_ctm` +0.0027 [+0.0013, +0.0041]、對 `r3_pma` +0.0032 [+0.0017, +0.0047]，兩組都滿足條件 ①②③、只差 ④（< 0.005）；對 `r2_mlp24` +0.0113 [+0.0083, +0.0142]，四個條件全滿足。1D／2D／3D 各級 0.487／0.425／0.515（`r5_ctm` 0.476／0.423／0.507），最弱的一段也略升。fold 0 已用掉，換不換 ensemble 只能靠 CV 決定；代價是提交管線要訓練並保存 3 個集合模型的權重、Δ 模型的推論成本 ×3（集合模型本身很便宜）。**是否把待辦第 2 項的目標從 `r3_pma` 改為 `ens_top3`，待使用者決定。**

### 2.8 全 fold 評分、L3/L4 與第一份提交管線（2026-10-16 實測）
- **資料**：5 個 fold 中每位 ≥ 5 局的玩家各取 ≤ 20 局（`scoring.fold_players`，每人以 player_id 衍生 seed 打亂），共 7,948 人、147,780 局、16,430,090 個目標落子，每手 18 個 profile。lab01 兩張 GPU 約 14 小時（04:06–20:32，其中暫停 04:16–06:18）。彙總表 `results/scores/games_fold{F}.csv.gz`；逐手陣列留在 lab01。
- **斷點設計**：以「已完成分段裡有哪些局」決定續跑（段大小可變、不重算、不重複），每段 100 局；`scripts/scores_ctl.sh pause|resume|status`；暫停會等當前這段做完。
- **題目特徵（24 維，`qfeatures`）**：18 維相對每手 log p 曲線、最佳 profile 的每手 log p、posterior-mean index、拋物線內插峰值、log 總手數、log 局數、各局最佳 profile 的標準差。
- **選模（只用 folds 1–4，留一 fold CV，各級平均官方分取 5／10／20 局平均）**：門檻（pmi）0.596、門檻（峰值）0.601、邏輯迴歸 l2=1e-4 0.609、1e-3 0.605、1e-2 0.597、**MLP h32 0.620（選定）**、MLP h64 0.618。訓練題為每位玩家 8 題隨機子集（5–20 局），類別平衡權重。
- **fold 0 只評一次（1,594 位玩家、9,872 題）**：MLP h32 **0.613 [0.600, 0.627]**；5 局 0.577 [0.565, 0.590]、10 局 0.622 [0.606, 0.638]、20 局 0.641 [0.621, 0.660]；題目平均 0.612；完全猜中 0.461、±1 內 0.872。各級：D 0.76、C 0.63、B 0.62、A 0.63、1D 0.48、2D 0.45、3D 0.50、4D 0.57、5D 0.63、6D 0.73。同一 fold 0 上門檻法 0.597、邏輯迴歸 0.600。CI 是與標題數字同一統計量的玩家層級 bootstrap（每次重抽樣所有題長共用）。
- **判讀**：學出來的 L3/L4 只比 pilot 的門檻法多約 0.016；1D–3D 仍是最弱的一段。Human SL 特徵大約到 0.61 為止，再往上要靠 L1b（用本資料標籤自訓）。
- **提交管線**：`scripts/predict_submission.py`（讀官方格式、丟棄並記錄不能用的局、評分、模型預測、期望得分決策、寫出並驗證 public→private 檔案，另存每題機率與報告）。從 fold 0 做的 40 題官方格式假測試：與離線路徑 **40/40 一致**、0 局丟棄。單卡 166 局面/秒 → 正式 800 題（約 1 萬局、110 萬局面）約 1.9 小時，11/04 前要把評分分到兩張卡或做加速。**2026-10-25 加固（待辦第 1 項）**：評分可分片——`--shard i --nshards N` 讓每張卡只評分自己那份（任務表只由輸入決定，各進程一致），寫 `submission_rank_shard{i}of{N}.npz`（每局 18 維總和與手數，以及每手 18 維 log p 與其局／手索引，供之後的每局向量），`--merge` 讀 N 個分片、檢查它們來自同一組題目且每局恰好評一次，再預測；單卡就是 shard 0 of 1 接著 merge，同一條路徑。`read_test` 拒絕 `num_games` 與非空 `sgf_*` 欄不一致的檔；`validate_submission` 加檢查：ID 序列等於輸入依序串接、`priv_` 不得在 `pub_` 前、每個 ID 帶 `pub_`／`priv_` 前綴、期望列數與各前綴題數（預設 800／400／400，mock test 自帶）。exit code：fallback 到 `1D` 的題目預設 3（`--allow-fallback` 放行）、驗證失敗 1、輸入格式／模型特徵／分片檔有問題 2。截止日退路 `--lenient-inputs`：格式不一致的題目改用實際非空欄讀入並記進報告的 `input_warnings`，exit 4（`--allow-input-warnings` 放行）。測試用假的評分器驗證分片＋合併與單進程逐位元相同；正式 GPU 上多卡 vs 單卡的逐題比對由 nano4 跑。
- **集合模型接進提交管線（2026-10-25，待辦第 2 項；程式、測試、訓練與驗證都已完成）**：`aicup.setmodel_io` 把 `fit_set_model` 的結果存成 `state_dict`（`.pt`）加 JSON 側檔（rung、seed、TrainCfg、token／qglob 標準化統計、內層提早停止診斷、provenance、訓練折與題目雜湊、配對 base 的檔名與 sha256），讀回的模型預測逐位元相同。`scripts/train_set_model.py` 在 folds 0–4 的全部訓練題（每人 32 題，與 CV 同一抽樣）上訓練一個 rung 的各 seed；base logits 以 5 折交叉擬合（與 CV 的 `split_base` 同一做法），並另存配對的推論用 base `results/models/rank_model_f01234.json`（`r2_mlp24` 在同一組題上的全折擬合；同一個 base 由各 rung 共用）。fold 0 的唯一評估已用掉，依使用者決定併入訓練，所以之後沒有乾淨的 held-out 分數。`predict_submission.py --set-models … --set-base …`：從分片檔的每手 log p 重建每局 300／75 維向量（與 `ladder_build_vectors` 同一函式）與 packed 題目，base = log(clip(p, 1e-9))，各模型機率**先在同一 rung 的 seed 間平均、再在 rung 間平均**，最後期望得分決策（三個 `r3_pma` 就是 `r3_pma`；加 `r4_xattn_t1`、`r5_ctm` 各三個就是 `ens_top3`，與 CV 的 `build_ensemble` 同一規則）；側檔的訓練題雜湊必須等於 base 文件的（配對檢查，否則 exit 2）；機率非有限數的題目退回 base 的機率並記進報告。`scripts/check_game_vectors.py` 以手數與 18 維總和把 mock test 的局對回 `fold0.npz`，逐欄比較向量（浮點欄容許 0.05，計數欄須相等），並可用同一組模型走離線路徑比對提交機率（每題 |Δp| ≤ 1e-3、40 題決策全同）。`ladder_ensemble.py --across-seeds` 量化「每個 seed 各自決策再平均分數」（CV 統計量）與「機率平均後一次決策」（提交規則）之差。**nano4 實測（2026-10-25，H200，程式 `24045c5`）**：`r3_pma` 三個 seed 於 folds 0–4 的 254,336 題共 5 分鐘（交叉擬合 base CE 1.2576；三個 seed 都由 patience 提早停止，best_epoch 0.75／1.00／0.75，內層 CE 1.2621 → 1.2104／1.2136／1.2054；每個 258,058 參數）。mock test 375 局全部對回 fold 0（0 unmatched、0 ambiguous），浮點欄位最大差 4.1e-3（`staged.q10`，其餘 ≤ 2.6e-3），計數欄位全為 0；帶三個 `r3_pma` 的 merge 與離線路徑 40 題機率最大差 5.0e-4、**決策 40/40 相同**。seed 平均方式（機率平均後一次決策 vs CV 統計量，差 [95% CI]）：`r3_pma` 0.6304 vs 0.6274，+0.0030 [+0.0016, +0.0044]；`r4_xattn_t1` 0.6306 vs 0.6277，+0.0029 [+0.0015, +0.0042]；`r5_ctm` 0.6315 vs 0.6279，+0.0036 [+0.0022, +0.0049]；`ens_top3` 0.6327 vs 0.6306，+0.0021 [+0.0010, +0.0032]——提交用的規則在四個 rung 上都略好、CI 都在 0 以上，照設計做。**`ens_top3` 路徑（同日）**：`r4_xattn_t1`（3 個 seed 共 2.5 分鐘，best_epoch 0.75／0.75／0.50，內層 CE → 1.2147／1.2145／1.2206，458,250 參數）與 `r5_ctm`（4 分鐘，best_epoch 0.50 ×3，內層 CE → 1.2144／1.2123／1.2154，437,514 參數）都沿用同一個配對 base（腳本確認同一組訓練題、參數相同後不改檔）。九個模型的 mock test merge exit 0、無警告、無 fallback；帶九個模型的檢查 PASS，40 題機率最大差 4.5e-4、決策 40/40 相同。權重、配對 base 與三份檢查報告在 `results/models/set/`（9 個 `.pt` + `.json`）、`results/models/rank_model_f01234.json`、`results/mock_fold0_40{,_set,_ens}/game_vectors_check.json`。正式提交用 `ens_top3`：第 1 項的 merge 指令加 `--set-models results/models/set/{r3_pma,r4_xattn_t1,r5_ctm}_s{0,1,2}.pt --set-base results/models/rank_model_f01234.json`（`docs/NANO4.md` §4.1）。
- **還不能說的**：0.613 不能直接和 Public 0.4631 比（題目、玩家、題長分布都不同）；同一組 fold 0 題目上的官方 baseline 已重跑：0.440 [0.427, 0.453]（expected-score 決策；argmax 0.399；§2.1）。各級平均的前提是測試集各級題數相近，尚未證實。

### 2.9 L1b：用本資料標籤訓練的每手證據模型（2026-10-25 設計；issue #6；計畫 `docs/superpowers/plans/2026-10-25-l1b-rank-heads.md`）
- **為什麼**：§2.4.1 與 §2.7 的結論是瓶頸在每手證據的解析度（1D|2D、3D|4D 分不開），不在聚合器也不在證據量；Human SL 特徵到 0.63 左右（`ens_top3` CV 0.6306）。H2：用 1.11 億個有等級標籤的目標落子自訓的條件模型 \(p(a\mid H,k)\) 勝過轉移來的 Human SL。
- **與 Human SL 的關係**：微調而非蒸餾、晚注入。trunk 是 humanv0 的（第一階段凍結），metadata 用 humanv0 訓練時見過的「等級未知」列（`bRankIsUnknown`／`wRankIsUnknown` 為 1、thermometer 全 0，其餘與 `rank_*` profile 相同）；10 個等級 head 都是 humanv0 policy head 的深拷貝，第 k 個只用 k 級玩家的落子訓練，輸出就是 \(\log p(a\mid H,k)\)，一次 trunk 出 10 個等級（§2.2 的晚注入）。第 0 步每個 head 等於 humanv0 在等級未知 metadata 下的 policy（`--check-init`，fp32 max |Δ| ≤ 1e-4），所以改善量有已知起點。蒸餾不排進：推論成本不是瓶頸（測試集約 100 萬局面）。L1a 與 L1b 的晚期融合（§2.2）第一版以 `fused` 特徵（18 條 Human SL 曲線 + 10 條 L1b 曲線的題目特徵串接）代表。
- **資料與抽樣**：只用目標玩家的落子（對手等級未知）。訓練用 folds 的所有玩家（不設最少局數），每人 ≤ 40 局（`fold_players` 同一個 seed 42 排列，前 20 局就是評估那 20 局）；每人局數中位數 51、前 1% 超過 850，所以上限是必要的。模型 A：folds 1、2（3,206 人、102,019 局、約 1.1×10⁷ 局面）；模型 B：folds 3、4（3,196 人、101,711 局）。損失是類別平衡 CE（每級總權重相同、平均 1，從完整訓練集算），合法步以外的點在 log-softmax 前設 −inf（與 `humansl._masked_policy_logp` 一致）。
- **out-of-fold**：自訓模型的特徵必須 out-of-fold（§2.7）。兩路交叉：A 給 folds 3、4（與 0）出特徵，B 給 1、2（與 0）；每位訓練玩家的特徵都來自沒看過他的模型，L3 的 folds 1–4 CV 照原協議。測試時 A、B 各出一套特徵、各走一次 L3，機率平均——L3 訓練與測試看到的特徵分布一致。只寫成選項、第一階段不採用：A′ 用 folds 0、1、2、B′ 用 0、3、4（各看 60% 玩家），代價是 fold 0 沒有 OOF 特徵、L3 只能用 folds 1–4 訓練；用早期訊號比一次再決定。
- **訓練**（第一階段）：trunk 凍結、bf16 前向；head fp32、Adam lr 1e-4、batch 256、最多 3 epoch；內層依玩家分層留出 10%（`settrain.inner_players`，同一份快取所有 seed 共用），每 1/4 epoch 算內層類別平衡 NLL，保留最好（含第 0 步），連續 4 次沒改善就停；checkpoint 可續跑。輸入平面壓成 bit 快取（每局面約 1.1 KB，每半 12–13 GB），避開 0B 量到的 CPU 特徵瓶頸（226 局面/秒/核）；trunk 輸出不快取（每局面 277 KB，TB 級）。
- **評估與採用規則（事先指定）**：(1) 早期訊號（決策點 1）：每級每手 NLL，L1b 自己的 head 對 Human SL 該級 18 個 profile 中最好的（對 Human SL 樂觀）與 pilot 峰值 profile；相鄰等級 AUC 照 §2.4.1(d)（每人一題 20 局、posterior-mean index、Mann–Whitney），pilot 參考 1D|2D 0.62、2D|3D 0.73、3D|4D 0.65。(2) CV（決策點 2）：folds 1–4 留一 fold、梯子的同一批驗證題與訓練題（從評分表還原 `fold_players` 的順序）、`MLPModel(hidden=32)` 單次確定性擬合；三組特徵 `hsl`（24 維，現行模型的特徵）／`l1b`（16 維）／`fused`（40 維）；只比 `l1b` vs `hsl`、`fused` vs `hsl`；條件沿用梯子的 ①（成對 CI 下界 > 0）③（4 個 fold 至少 3 個 > 0）④（≥ 0.005），② 不適用。任一組成立 → L1b 特徵進 L3（下一個計畫）；都不成立 → 看 `fused` 的方向決定要不要做第二階段（解凍 trunk）。fold 0 不在任何選擇裡用。
- **風險**：預訓練權重合規未問——不只 L1b，`ens_top3` 整條提交管線建在 humanv0 上，若不准，路線 2（自己的 trunk 從零訓練、同一組 head、訓練程式只換 `--model-config`）是唯一出路；沒有乾淨 held-out（fold 0 已用掉、也是集合模型的訓練資料）；只動 head 可能改不動 1D–3D（早期訊號一天內可知）；凍結 trunk 保留 KGS 2020 表徵，L1b 若只是重學等級尺度的對應，CV 增益會很小。
- **程式**：`aicup.l1b`（metadata、`RankHeads`、trunk 前向、評分、存取）、`aicup.featcache`、`aicup.l1btrain`、`aicup.l1beval`、`aicup.l1bcv`；`scripts/l1b_build_cache.py`、`l1b_train.py`、`l1b_score.py`（`score_folds.py` 佈局）、`l1b_early_signal.py`、`l1b_cv.py`；指令與預期值在 `docs/NANO4.md` §4.2。**結果待 nano4 回報**（issue #6）。

### 2.10 提交改用鑑別式 L1b 的 fused MLP（2026-10-26；issue #9；PR #10，nano4 驗證 2026-10-26）
- **依據（nano4，issue #6，PR #8）**：L1b 三個版本——第一階段（凍結 trunk）、第二階段（trunk 微調，8 卡 torchrun）、鑑別式（第二階段起步，loss 加 1 × 十個 head 對落子 log p 的等級交叉熵，選模看 NLL + CE）。只有鑑別式把題目分數拉開：folds 1–4 CV 的 `fused`（Human SL 24 維 + L1b 16 維，`MLPModel(hidden=32)`，A、B 平均）**0.6545**，對 `ens_top3` 0.6327 高 0.0218、對 `hsl` 0.6192 高 0.035；`l1b` 單獨反而比 `hsl` 差，加上增幅遠大於所有聚合器實驗，所以先查洩漏：A、B 的訓練／出特徵 folds 無交叉，CV 特徵流向正確；再用誰都沒看過的 fold 0 做最乾淨的檢查（`scripts/l1b_fold0_check.py`，使用者核准、不作選擇）：`hsl` 0.6113（重現 §2.7 的基準）、`fused_avg` **0.6535**（+0.0422 [+0.0323, +0.0524]），A、B 各自與各題長都穩定領先。鑑別式的代價是每手 NLL 變差（內層 1.43 → 1.52），鑑別力（等級 CE 2.29 → 2.14）大升；XGBoost 對照沒有勝過 MLP。**使用者決定：提交改用鑑別式 fused、A／B 平均；`ens_top3` 留當退路。**
- **接進管線（PR #10）**：`predict_submission.py --l1b-heads A.pt B.pt` 讓分片在 18 個 profile 之外各做一次 L1b trunk 前向（成本 18 → 20 次），分片檔既有欄位不變、另加 `l1b_sums [n, K, 10]`、`l1b_move_logp [M, K, 10]` 與模型的 sha256；`--fused-model results/models/rank_model_fused.json` 在 merge 時先做配對檢查（分片模型的 state／trunk sha256 集合 = 文件裡的，否則 exit 2），每題 \(p = \tfrac1K\sum_k \mathrm{MLP}(\text{fused}_k)\) 再期望得分決策；機率非有限數退回 `rank_model.json` 並記進報告。fused MLP 由 `scripts/fit_fused_model.py` 擬合，訓練與 fold 0 檢查完全相同（folds 1–4、每人 32 題、folds 1、2 用 B 的評分、3、4 用 A 的），文件記 features、訓練題雜湊、A／B 的 heads／trunk sha256 與各自提供特徵的 folds；同一支腳本重算 fold 0 的四個數字對 `fold0_check.json`，並存下 fold 0 每題機率供 mock test 對帳。檢查：`check_game_vectors.py --l1b-scores`（分片每局 L1b 總和 vs `l1b_score.py` 的表）、`check_fused_submission.py`（40 題決策全同、|Δp| ≤ 1e-3；單卡 vs 4 卡）。權重 > 100 MB：不進 git，`pack_weights.py` 打成帶 `SHA256SUMS` 的 tar，得獎時上傳成 GitHub Release 資產（上限 2 GB），側檔與 fused 文件裡的 sha256 是對帳依據；不改 fp16。指令在 `docs/NANO4.md` §4.3。
- **nano4 驗證（PR #10，2026-10-26）**：pytest 267 passed／1 skipped。`fit_fused_model.py`：fold 0 的 hsl 0.6113、fused_A 0.6479、fused_B 0.6510、fused_avg 0.6535，與 `fold0_check.json` 的差全為 0；訓練題 203,328（`results/models/rank_model_fused.json`、`results/l1b/fold0_fused_proba.csv` 進 git）。mock test 單卡：`decision_model: fused`、無退路題，評分 250 s（原 18 profile 222 s，+13%，預期 +11%）。管線 vs 離線（`results/mock_fold0_40_fused/fused_check.json`）：40 題決策全同、max |Δp| 3.92e-4；單卡 vs 4 卡（`results/mock_fold0_40_fused4/fused_check.json`）：40 題決策全同、max |Δp| 1.69e-4（容差 1e-3）。權重 tar 233,820,160 bytes，sha256 `94f9af64aa70edf2af9bfa9bfe0206eb0f6380e635bfd17429f4e674e41ddfe7`。兩個修正：(1) `check_game_vectors.py --l1b-scores` 以 (state, trunk) sha256 配對，而舊 `l1b_score.py` 沒把 trunk 寫進 `meta_s*.json`，永遠配不到；現在 meta 沒有該鍵時只比 state_sha256 並印 WARNING，`l1b_score.py` 之後也寫 trunk；(2) 4 卡分片時每個主行程 44 條 torch 執行緒在 48 核上空轉，65 s 的評分拖成 8 min，`shards.sh` 改為每分片 `OMP_NUM_THREADS=2`。重跑（`2699540`）：pytest 268 passed／1 skipped；4 卡工作 1 分 23 秒（原 8 分 10 秒），單卡 vs 4 卡仍 PASS；第 3 步配對成功，但 L1b 每局總和差到 A 0.326、B 0.568，超過當時的每局容差 0.05——nano4 量了 375 局：中位數 0（大多數局逐位元相同）、每手最大 0.007、每個 head 每手平均有號差 −2e-5 到 +3e-5，是 bf16 下同一局落在不同批次的雜訊（HSL 分片間每手 4e-3 同量級；鑑別式 heads 對 bf16 誤差較敏感），不是模型或 metadata 用錯（那會整體偏移），對機率的影響即第 4 步的 ≤ 3.9e-4。容差因此改成兩條：每局每 head |Δ sum|/n_moves ≤ 0.02，且每 head 跨局平均有號差 ≤ 1e-3（`--l1b-bias-tol`）。第 3 步重跑（`082f5a6`）PASS：375/375 局，heads3_A 每手最大 3.585e-3、bias 1.60e-5，heads3_B 每手最大 7.015e-3、bias 3.34e-5（`results/mock_fold0_40_fused/game_vectors_check.json`）。**五步全部對上（2026-10-26）**；是否合併由使用者決定。
- **還不能說的**：fold 0 的 0.6535 和 CV 的 0.6545 都是離線題上的分數，不能直接和 Public 比；A、B 兩個模型各只看到 40% 的玩家，全量模型（或 A′／B′）是否更好沒有測；提交的模型和 fold 0 檢查的模型相同，所以 fold 0 之後沒有再做任何選擇。

### 2.11 提交改用 fused_opp：對手落子的 L1b 特徵（2026-10-27；issue #12；PR #13，nano4 驗證 2026-10-27）
- **依據（nano4，PR #11）**：線上對局的配對實力相近，對手的落子因此是目標玩家等級的第二份、近乎獨立的證據。`l1b_score.py --opponent` 用同一組出 fold 的 L1b 模型（A 評 folds 3、4、0，B 評 1、2、0）評同一批評估對局裡**對手**的落子，`question_features` 做 16 維，接在 fused 的 40 維之後成 `fused_opp`（56 維）。folds 1–4 CV（`l1b_opp_check.py`，預先指定 fused_opp vs fused，三條件全過）：**0.6735** 對 fused 0.6545；fold 0（`l1b_opp_fold0_check.py`，使用者核准、不作選擇）：fused_opp_A 0.6684、fused_opp_B 0.6684、**fused_opp_avg 0.6716**，對 fused_avg 0.6535 高 +0.0181 [+0.0094, +0.0266]，對 hsl 0.6113 高 +0.0603。同一個 PR 也試了 humanv0 value／score head 的落子品質特徵，沒有採用。**使用者決定：提交改用 fused_opp，`fused` 留作第一退路。**
- **接進管線（PR #13）**：`predict_submission.py --l1b-heads` 預設多跑一趟對手（同一個 CPU pool，任務換成另一色，只過 K 個 L1b trunk；`--no-l1b-opponent` 關掉），分片檔多 `l1b_opp_sums [n, K, 10]`、`l1b_opp_n_moves [n]`（對手手數，特徵用它），不存對手的每手 log p（沒有決策或檢查用到）。`aicup.fused.fused_opp_features` 與 `l1b_opp_check.features` 的 `fused_opp` 逐欄相同（測試）。`fit_fused_model.py --opp A:scores3_opp B:scores3_B_opp` 直接呼叫 `l1b_opp_check.features`，訓練與 fold 0 檢查完全相同，寫 `rank_model_fused_opp.json`（`opponent: true`、56 features、`l1b_models` 同格式、`fused_state` 當參考），fold 0 七組數字對 `opp_fold0_check.json` 差應為 0。`--fused-model` 依文件的 features 判斷要不要對手特徵；分片沒有對手分數就 exit 2，40 維的 `rank_model_fused.json` 仍可用同一組分片檔。檢查：`check_game_vectors.py --l1b-opp-scores`（同樣的每手 0.02 + bias 1e-3，另加對手手數逐局相等與 meta `side` 的左右檢查）、`check_fused_submission.py --set fused_opp_avg`。成本：GPU 多 K 次對手 trunk 前向，CPU 特徵約加倍；nano4 單卡實測評分 247.6 s + 對手 40.8 s（PR #10 的 250 s → 298 s，+16%），4 卡分片工作 1 分 48 秒。指令在 `docs/NANO4.md` §4.4。
- **還不能說的**：對手特徵的前提是測試集的局也是相近實力的線上配對；若官方測試集的對局來源不同（例如讓子局、指導棋），對手的落子會誤導，mock test 不會暴露這點（它的局來自同一份訓練資料）。fold 0 的 0.6716 仍是離線題上的分數；A、B 各只看到 40% 的玩家。
- **nano4 驗證（PR #13，2026-10-27，分支 `9c12363`）**：pytest 273 passed／1 skipped。`fit_fused_model.py --opp`：fold 0 七組對 `opp_fold0_check.json` 差全為 0（hsl 0.6113；fused_A/B/avg 0.6479/0.6510/0.6535；fused_opp_A/B/avg 0.6684/0.6684/0.6716；fused_opp_avg vs fused_avg +0.0181 [+0.0094, +0.0266]），訓練題 203,328（`results/models/rank_model_fused_opp.json`、`results/l1b/fold0_fused_opp_proba.csv` 進 git）。mock test 單卡與 4 卡：`decision_model: fused_opp`、無退路題、`l1b_opponent: true`，positions 41,620、opp_positions 41,643。每局 L1b 總和對帳 375/375：目標 `heads3_A/B` 每手 3.6e-3／7.0e-3、bias 1.6e-5／3.3e-5，對手 3.4e-3／4.9e-3、bias 2.4e-5／4.1e-5，對手手數逐局相等（`results/mock_fold0_40_opp/game_vectors_check.json`）。管線 vs 離線 `fused_opp_avg`：40/40 決策相同、max |Δp| 2.8e-4；單卡 vs 4 卡：40/40、2.7e-4（兩份 `fused_check.json`）。權重 tar 233,963,520 bytes，sha256 `4b7951f3ecf19c6742f8ec8b718e355c3b9142b4dcb7601bbf7d93b0e30b58e3`（舊 fused 版備份在 nano4 `release/prev_fused/`）。**五步全部對上**；是否合併由使用者決定。

### 2.12 棋力標量（strength estimator）對照（2026-10-28；issue #14；PR #15；nano4 2026-10-29 跑完：a)、b) 都沒過，不採用）
- **依據**：Chen、Shih、吳毅成《Strength Estimation and Human-Like Strength Adjustment in Games》（ICLR 2025）：對局面＋落子打棋力分數（SE），不同等級玩家樣本的成對比較（Bradley–Terry）訓練，聚合成標量再對應等級，圍棋 15 局 80% 以上。Kuboki、Hsueh、Ikeda 等《Policies of Multiple Skill Levels for Better Strength Estimation in Games》（AIIDE 2025）：在 SE 上加 KataGo Human SL 多等級 policy 的落子機率特徵與 meta-model，10 局 80%、20 局 92%（SE 單獨 71%／84%）。我們的 fused_opp 等於後者的「多等級似然特徵 + meta-model」半邊（更深：鑑別式 L1b head、對手、嚴格 OOF），缺 SE 標量；但 L1b 的 `pmi_sum` 已是一個有序棋力標量，SE 與它只差訓練目標（成對比較 vs 交叉熵），所以預期有加分但有限。
- **a) 有序最後一層**（`aicup.ordinal`、`scripts/ordinal_check.py`）：現有 56 維特徵不動，比三個最後一層：`mlp`（現行）、`coral`（CORAL：共用標量 g(x) + 9 個以 softplus 遞減參數化的門檻，每門檻一個 BCE，類別機率恆非負）、`bt`（題目層 Bradley–Terry 成對損失訓出標量 s(x)，每 epoch 每題配 4 個隨機對手；之後在訓練分數上擬合累積 logit 鏈結 P(y ≤ k) = σ(θ_k − a·s) 給類別機率，再期望得分決策）。三者同一套 `MLPModel` 制度（一層 ReLU、全批次 Adam float64、300 epochs、類別平衡權重、seed 固定）。CV 協定同 `l1b_opp_check.py`，`mlp` 列須重現 `opp_check.json` 的 0.6735；預先指定 `coral vs mlp`、`bt vs mlp` 三條件；**不碰 fold 0**。指令 `docs/NANO4.md` §4.5；決策規則：任一有序版本全過才另開 issue 討論換最後一層。
- **a) 結果（nano4，2026-10-29，分支 `52410a8`，`results/l1b/ordinal_check.json`，39,327 題）**：`mlp` 重現 `opp_check.json` 的 fused_opp 0.6735（差 0）。`mlp` 0.6735（NLL 1.1790；n=5/10/20 0.6271/0.6833/0.7102）、`coral` 0.6472（1.2804；0.6138/0.6589/0.6688）、`bt` 0.6761（1.1905；0.6320/0.6832/0.7130）。預先指定：coral vs mlp −0.0263 [−0.0309, −0.0217]，四折全負，三條件全不過；bt vs mlp +0.0025 [−0.0012, +0.0063]，四折全正（+0.0030、+0.0025、+0.0005、+0.0041）但 CI 含 0、未達 0.005，只過條件 (3)。**兩個有序版本都沒過，最後一層維持 10 類 MLP。** 觀察（只報告，不作選擇）：bt 的 1D–3D 每類準確率較高（0.577/0.531/0.560 對 mlp 0.540/0.489/0.536）而 NLL 較差，和「單一標量 + 鏈結」把機率形狀壓平一致；coral 差最多的是 20 局題（−0.041），共用標量加遞減門檻比 10 個獨立 logit 少了可表達的形狀。
- **b) BT 強度 head**（`aicup.strength`、`aicup.sefeat`、`scripts/se_train.py`、`se_score.py`、`se_check.py`；指令 `docs/NANO4.md` §4.6）：humanv0 trunk 上一份 PolicyHead 拷貝，通道 0 在落子處的輸出當每手強度分數 s（初始 = 中性 metadata 下的 policy logit），每個行程批次內跨等級位置對的 BT 損失 −log σ(sign(r_i − r_j)(s_i − s_j))、權重 √(w_i w_j)，inner 同損失選模並記成對準確率與各等級平均分數；訓練沿用 `l1btrain` 的快取、inner 切分、checkpoint、torchrun 慣例，第一階段凍結 trunk、A/B OOF 照舊。`se_score.py` 每局 mean／std／q10／q50／q90／三段均值（t < 40、40–99、≥ 100），`--opponent` 同 `l1b_score.py`；`sefeat` 9 維（手數加權的平均、局間離散、局內離散、三個分位、三段），對手再 9；`se_check.py` 預先指定 `fused_opp_se vs fused_opp`、`fused_opp_se2 vs fused_opp`，報告 `se`、`hsl_se`。通過才談 fold 0——它已經替 fused 和 fused_opp 看過兩次，第三次要先聲明只當檢查、不作選擇。
- **b) 結果（nano4，2026-10-29，分支 `0e51c90`：側檔 `se_{A,B}.json`、四個評分目錄的 96 個 `meta_s*.json`、`results/l1b/se_check.json`）**：訓練（第一階段凍結 trunk，8 卡 torchrun 每卡 batch 128、全域 1,024，lr 1e-4，3 epochs 跑滿、沒有早停，inner 40 萬個位置、十個等級都在）：se_A（folds 1、2，1,015 萬個訓練位置）inner BT loss 1.1286 → 0.5228、成對準確率 0.5186 → 0.7369，1,528 s；se_B（folds 3、4，1,014 萬）1.1148 → 0.5076、0.5218 → 0.7480，1,486 s。兩個 head 十個等級的平均分數都**嚴格遞增**（A：−0.48 … +1.66；B：−1.47 … +0.93），最佳都在最後一步、每 0.25 epoch 還降約 0.001（未完全收斂）。評分：A 評 folds 3、4、0，B 評 1、2、0，目標與對手各一次，每 fold 約 2.95 萬局、328 萬個位置，8 卡分片每 fold 約 8 分鐘。CV（`se_check.py`，39,327 題）：`fused_opp` 重現 0.6735（差 0）；`hsl` 0.6192、`se` 單獨 0.4245、`hsl_se` 0.6329、`fused_opp_se` 0.6694、`fused_opp_se2` 0.6720。預先指定：fused_opp_se vs fused_opp **−0.0041** [−0.0073, −0.0009]（顯著變差；四折 −0.0032、−0.0013、−0.0133、+0.0014）；fused_opp_se2 vs fused_opp −0.0015 [−0.0050, +0.0019]（四折 +0.0029、−0.0059、−0.0017、−0.0013）；三條件全不過。只報告：hsl_se vs hsl +0.0137 [+0.0091, +0.0182]（四折全正，三條件都會過）；se vs hsl −0.1947 [−0.2041, −0.1852]；fused_opp vs hsl +0.0543 [+0.0485, +0.0603]。**兩條都沒過，強度特徵不接進提交，不碰 fold 0。** 解讀：強度標量本身帶等級資訊（只有 Human SL 特徵時 +0.014），但 L1b 的 10 個鑑別式 head 加對手特徵已經涵蓋它，多出的 9／18 維只給 h32 MLP 加雜訊——和本節開頭「SE 與 `pmi_sum` 只差訓練目標」的預期一致。
- **還不能說的**：兩篇論文的準確率用他們自己的資料與等級切法，不能和我們的 0.67 直接比。a) 只說明「最後一層的有序性」在這 56 維上不值錢；bt 四折全正但未達門檻，更多 epoch、更寬隱藏層、或 bt 與 mlp 的機率平均都沒有測。b) 的 head 只訓了 3 epoch 且還在降、trunk 凍結、每局只取 9 個固定統計量；更長的訓練、解凍 trunk、或把每手 s 的分佈直接交給聚合器都沒有測——但 fused_opp_se 是顯著變差而不是持平，瓶頸看來在 MLP 的輸入維度對訓練題數，不在 head 的品質，這些延伸要翻盤的機會不大。後續若再試，仍須先過 folds 1–4 同一套預先指定比較。

### 2.13 L1b 五折交叉擬合（80% 的玩家、5 模型平均）+ 相鄰級 label smoothing（2026-10-29；issue #16；PR #17；nano4 2026-10-30 跑完：兩條預先指定都沒過，不採用）
- **fold 0 的錯誤結構（2026-10-29，`results/l1b/fold0_fused_opp_proba.csv`，fused_opp_avg，9,872 題、1,594 人）**：命中 0.519、±1 以內 0.917、差兩級以上 0.083，分數只能從「差一級變命中」來。錯誤是玩家層級的：同一人（4 或 7 題）全對 16.1%（獨立期望 1.6%）、全錯 13.4%（1.0%）；帶號誤差的變異數玩家間 0.63、玩家內 0.40（61% 在玩家之間）；偷看標籤扣掉每人平均偏差的診斷上界 0.844。多給棋局只壓玩家內的部分：n=5／10／20 命中 0.496／0.544／0.570。1D／2D／3D 命中 0.34／0.32／0.37，1D 有 32% 被判成 A。校準沒有空間：可靠度貼對角線，在 fold 0 上作弊擬合溫度（T 1.14）分數不變；錯題 59% 真值是第二名、83% 與預測相鄰。訓練檔的 `game_id` 在同一人同一色內是隨機排列（id 跨度 0.83 對隨機 0.82；與每局 Human SL pmi 的 Spearman −0.005，打亂後 +0.006；前半／後半局對標籤的 Spearman 0.908／0.902），沒有時間序訊號。結論：標籤（伺服器等級）與「這些局下出來的棋力」之間有每人固定的落差，佔錯誤六成，SGF 裡沒有直接量它的欄位；a)、b)（§2.12）證明在 p(落子 | 局面, 等級) 這條通道上換包裝沒有用。
- **為什麼是資料量**：專案最大的兩次跳升（Human SL → 鑑別式 L1b +0.035、加對手 +0.019）都來自用標籤訓練證據模型，而 heads3_A／B 各只看過 2 個 fold（約 3,200 人、1,015 萬個位置）。使用者決定（2026-10-29）：做五折交叉擬合，不做 60%（A′／B′）也不做 100% 單一模型（後者沒有任何 OOF 特徵可以訓 MLP，特徵分布會比 MLP 看過的更尖），不量資料量斜率；GPU 多開，五條鏈平行。
- **設計**：f_k 用 folds {0,…,4} ∖ {k} 訓練（約 6,400 人、2,030 萬個位置），配方與 heads3 完全相同（三段、同超參數、同 inner 早停；第一階段全域 batch 256、後兩段 1,024），只改資料。不重建快取：新建 `cache_0`，`l1b_train.py --cache cache_A cache_B cache_0 --exclude-folds k`（`featcache.read_caches` 讀多包快取並依 `splits/player_folds.csv` 丟掉該 fold 玩家的局；`l1btrain.caches_meta` 檢查快取間 fold 不重疊、排除的 fold 在聯集內、`max_games` 一致；側檔 `train_folds` = 聯集 − {k}，checkpoint 也記 `train_folds`）。f_k 只評 fold k，目標與對手各一次，五個模型寫進同一組目錄 `scores5`／`scores5_opp`（`l1b_score.py` 的 OOF 檢查保證配對）。`scripts/xfit_check.py`：`l1b_opp_check` 協定（folds 1–4、pilot 驗證題、每人 32 題、`RandomState(42+v)`、`MLPModel(hidden=32)`），`fused_opp`（A／B，須重現 0.6735）、`fused_opp_x5`、`fused_opp_ls`（ε 0.1）、`fused_opp_ls2`（0.2）、`fused_opp_x5_ls`；預先指定 `fused_opp_x5 vs fused_opp`、`fused_opp_ls vs fused_opp`（條件 (1)(3)(4)），其餘只報告。label smoothing（`rankmodel.ordinal_targets`）：(1−ε) 在真類、ε/2 在左右鄰級，邊界類整個 ε 給唯一鄰級；ε = 0 走原本的 `cross_entropy`，逐位元相同；文件 `state` 多 `label_smoothing`，舊文件讀成 0。指令 `docs/NANO4.md` §4.7。
- **要說清楚的限制**：只有 f0 沒看過 fold 0。`fit_fused_model.py` 的 fold 0 表現在只用沒看過 fold 0 的模型（A／B 兩個都是，不變；交叉擬合只有 f0），文件記 `fold0_models`／`models_seen_fold0`。**測試時 5 個模型的機率平均在任何留出資料上都沒有被量過**（f1–f4 都看過 fold 0）；驗證到的是「每個模型的 OOF 特徵 + 這個 MLP」，平均不變差是假設（A／B 平均在 fold 0 上 +0.003 的經驗）。fold 0 若再看，宣告只當檢查、不作選擇（第三次）。
- **決策規則**：`x5` 通過 → 另開 issue 接進提交（5 份 heads + trunk 約 550 MB，分片評分多 6 次 trunk 前向；`--l1b-heads`、fused 文件、`fit_fused_model` 都是 K 模型通用的）；`ls` 通過 → 同一個 issue 一起談。都沒過 → 記在這裡，結束。
- **nano4 實際跑法（2026-10-29，分支 `0be8e82` → `95d4382` → `db3199f`）**：`cache_0` 以 8 卡節點分 8 片建（1,605 人、5,701,594 個位置）；第一段 8 卡 `--batch 32`（全域 256）、後兩段 8 卡 `--batch 128`（全域 1,024），epoch 不改，五條鏈同時排（節點 RAM 1.9 TB）；五條 8 卡鏈同時從共享檔案系統讀快取時最慢的 rank 到第一次集體通訊超過 NCCL 預設 10 分鐘（f2 第一段 `inner NLL nan` 中止），`l1b_train.py` 的 `init_process_group` 改 60 分鐘逾時後重跑。每條鏈：第一段 6,400–7,900 s、第二段 6,100–6,400 s、第三段 3,662–5,029 s；每個 fold 一側的評分約 8 分鐘。
- **訓練面（`heads5*_f{k}.json`）**：五個模型各約 6,400 人、2,260 萬個位置（inner 留出 10% 玩家後訓練 2,028–2,040 萬，2.0× heads3）。第一段 inner NLL 1.615–1.622 → 1.478–1.483（heads_A 1.616 → 1.489）、第二段 → 1.415–1.424（heads2_A 1.434）、鑑別式 rank CE 2.281–2.284 → **2.100–2.110**（heads3_A／B 2.137／2.111），best_epoch 都是 0.75。資料量在每手層級確實有幫助：NLL 低約 0.01、rank CE 低 0.01–0.04。
- **CV 結果（`results/l1b/xfit_check.json`，39,327 題）**：`fused_opp` 重現 0.6735（差 0）。`fused_opp_x5` **0.6704**（NLL 1.1857；n=5/10/20 0.6285/0.6753/0.7073；折 0.6611、0.6722、0.6845、0.6635）、`fused_opp_ls` 0.6739（1.1670）、`fused_opp_ls2` 0.6755（1.1722）、`fused_opp_x5_ls` 0.6705。預先指定：x5 vs fused_opp **−0.0032** [−0.0071, +0.0007]，折 −0.0041、+0.0038、−0.0049、−0.0075，三條件全不過；ls vs fused_opp +0.0003 [−0.0019, +0.0025]，只過 (3)。只報告：ls2 vs fused_opp +0.0019 [−0.0006, +0.0045]（四折全正 +0.0030、+0.0031、+0.0010、+0.0006，CI 含 0）；x5_ls vs x5 +0.0002；x5_ls vs fused_opp −0.0030。**兩條都沒過，不採用；提交維持 A／B 的 fused_opp，不碰 fold 0。**
- **解讀與未驗證的假設**：每手層級更好（rank CE 低 0.01–0.04）、題目層級卻持平偏差，是這次最值得記的事。nano4 的假設：x5 的每個驗證 fold，MLP 的訓練題特徵來自**三個不同模型**（A／B 時是一或兩個），各模型的 log p 刻度有各自的偏移（五個模型的 inner NLL 整體差 0.008，單一等級跨模型差到 0.10：D 級 1.604–1.707，而題目特徵裡有每手絕對 log p 的項），MLP 學到混合刻度、套到第四個模型的特徵上就偏；若成立，對每個模型的特徵分別標準化（或只用相對曲線）應該能收回那 0.003 以上。另一個可能是 A／B 的 CV 本身略為樂觀：驗證 fold 的特徵模型（例如 A 給 fold 3）在訓練題裡也出現（A 也給 fold 4 的訓練題），x5 則完全沒有這種重疊。兩者都只是解讀，是否開新實驗由使用者決定；它只是 CPU 的 CV 變體（分片檔已有五個模型的評分）。
- **label smoothing**：ε 0.1 零效果，ε 0.2 四折全正但未達門檻、NLL 也較好（1.1722 vs 1.1790）；是個一致但很小的方向，不夠接進提交。

### 2.14 刻度校正變體：每個 L1b 模型每個 head 的偏移置中（2026-10-30；issue #16 後續；nano4 同日跑完：沒過，刻度假設不成立）
- **使用者決定（2026-10-30）**：合併 PR #17；跑這個 CPU 變體檢驗 §2.13 的刻度假設（評分檔都在，幾分鐘）。若 `x5c` 過關，接進提交要五個模型評分加校正步驟，11/04 前時間緊，接不接等結果再決定。
- **做法**（`scripts/xfit_check.py` 的 `fused_opp_c`、`fused_opp_x5c`）：對每個 L1b 模型（fold 目錄的 `meta_s*.json` 以 heads 的 state sha256 認模型；沒有 meta 時以目錄為一個模型）和每個 head，用它在 folds 1–4 評過的局算以手數加權的每手平均 log p（目標表與對手表各一組 10 個數），每局 `sum_<rank>` := `sum_<rank>` − `n_moves` × 偏移，再照原本的 `l1b_opp_check.features` 做 56 維。只用無標籤的評分，10 個數就把各模型各 head 的絕對刻度對齊；相對曲線裡 head 間的差也隨之改變（偏移各 head 不同）。A／B 做同樣的校正當對照（A 用 folds 3、4，B 用 1、2）。測試時若採用：每個模型的偏移從它的 OOF fold 算好存進文件，分片評分後套用。
- **預先指定**：`fused_opp_x5c vs fused_opp`（條件 (1)(3)(4)）。只報告：`fused_opp_c vs fused_opp`（校正對 A／B 本身有無影響）、`fused_opp_x5c vs fused_opp_c`、以及 §2.13 的五條（數字應與 `xfit_check.json` 完全相同）。偏移表印出並寫進 `xfit_center_check.json`。
- **結果（nano4，2026-10-30，`results/l1b/xfit_center_check.json`，分支 `17bdc49`，PR #17 合併為 `main` `5a3c1b0`）**：§2.13 的五列與 `xfit_check.json` 逐位元相同。偏移表（每手平均 log p，10 個 head）：A（folds 3、4）與 B（1、2）同一 head 差 0.01–0.31（6D head：−2.906 vs −3.213），五個 f_k 之間差到 0.33（D head：−2.533 vs −2.861），目標側與對手側幾乎相同（差 ≤ 0.02）——刻度偏移確實存在。但校正沒有用：`fused_opp_x5c` **0.6704**（= `fused_opp_x5` 0.6704；NLL 1.1877 vs 1.1857），vs `fused_opp` −0.0031 [−0.0070, +0.0007]，折 −0.0013、+0.0003、−0.0116、0.0000，三條件全不過；`fused_opp_c` **0.6683**，vs `fused_opp` **−0.0052** [−0.0087, −0.0016]，四折全負——把 A、B 各自置中反而顯著變差；`x5c vs c` +0.0021 [−0.0015, +0.0056]。**刻度假設不成立**：偏移存在，但 MLP 本來就能處理（A／B 兩個模型的偏移也沒有拖累 fused_opp），把它減掉還拿走了有用的訊息（每手絕對 log p 的水準本身帶等級訊息，各模型的偏移與它糾纏）。x5 比 A／B 低 0.003 的原因因此更可能是 §2.13 的第二個解讀（A／B 的 CV 略為樂觀：驗證 fold 的特徵模型也給了訓練題）或純雜訊；資料量在每手層級的進步（rank CE 低 0.03）在這組 56 維 + h32 MLP 的題目層級看不到。這條線結束，提交維持 A／B 的 fused_opp。

### 2.15 落子序列：每手 L1b log p 的順序有沒有加總看不到的訊號（2026-10-30；issue #19；nano4 待跑）
- **起因**：使用者質疑「為什麼 CTM 不能一步一步學」，並指出資料理論上是時序訊號。確認：§2.7 的 CTM 是聚合器，輸入是每局 24／40 維的向量、tick 是內部的思考步，從沒看過逐手序列；而今天提交管線的每個特徵都是一局各手 log p 的加總（`sum_<rank>`，§2.9–2.11）。加總是「每手獨立、順序無關」下的充分統計量，只有在那個假設下才無損。可能丟掉的訊號：(a) 局內的起伏與自相關（讀秒、形勢好壞對出手品質的影響是連續的）；(b) 布局／中盤／官子的等級訊號強度不同（加總對每手一視同仁）；(c) 目標手與對手前後手的交互（對手一手好棋後我方的應手）；(d) log p 隨手數的斜率（手數本身已在 n_moves 裡）。§2.13 確認的是**跨局**的 game_id 順序沒有時序；這裡問的是**局內**的順序。
- **設計**（`aicup.seqdata`、`aicup.seqmodel`、`scripts/seq_diag.py`、`seq_crossfit.py`、`seq_check.py`）：
  - 步序列：每局依 t 合併目標手與對手手，每步 31 維 = 目標手 18 個 Human SL log p + 10 個 L1b log p、對手手 10 個 L1b log p（其 Human SL 欄為 0）、`is_target`／`is_opponent`、t/300；全部來自現有評分檔的每手 npz（`results/scores`、`scores3`／`scores3_B`、`scores3_opp`／`scores3_B_opp`）。每局 base = 目標手 L1b 的加總，與 csv 的 `sum_<rank>` 核對（容忍 0.05）。
  - 四個 encoder，輸出都是 **base + Δ**（Δ 的輸出層零初始化，與 §2.7 的 Δ 模型同一招；base 以 float64 傳遞）：`sum`（Δ ≡ 0，就是今天的特徵，管線錨點）、`conv`（兩層 1-D 卷積核 5、masked mean + max 池化；padding 的步在每層卷積前歸零，輸出與 batch 的 padding 無關）、`gru`（GRU 末狀態 + 狀態的 masked mean；**必要的對照**：若順序有訊號，GRU 是最便宜能吃到它的模型）、`ctm_seq`（tick = 落子：每步 synapse 讀 [z, enc(x_t)] 做 Linear → GLU → LayerNorm，NLM 把每個神經元最近 `history` 個 pre-activation 變成活化，輸出讀自 `n_sync` 對神經元同步的遞迴累積 α ← d·α + z_i z_j、β ← d·β + 1、S = α/√β，d = exp(−r) 可學——原版 CTM 的機制，資料改成一手一 tick 餵入而不是 attention；padding 的步不改狀態）。輸入以訓練步的均值／標準差標準化（存進模型）。
  - 訓練：每局等級的類別平衡交叉熵 + γ‖Δ‖²（γ = 1e-3），Adam 1e-3、weight decay 1e-4、batch 256、至多 10 epoch；inner 早停：每等級 10% 的玩家留出（`settrain.inner_players`，`INNER_SEED`），每 epoch 評 4 次，8 次沒進步停，取 inner CE 最低的權重（含第 0 次評估，即 Δ ≡ 0——encoder 學不到東西就退回加總）。
  - 交叉擬合：k = 1…4，encoder_k 用其他三折訓、只對 fold k 輸出，寫成 `l1b_score.py` 版面的每局表（`sum_<rank>` := base + Δ），`seq_check.py` 直接用 `l1b_opp_check.features` 把目標的 L1b 表換掉（對手表仍是 A／B 的加總），其餘協定與 `l1b_opp_check` 相同（含 §2.13 提到的「驗證 fold 的特徵模型也給了訓練題」這點輕微樂觀——所有比較的集合一樣，不影響差）。
  - 第 0 步診斷（`seq_diag.py`，不訓神經網路）：每手 L1b 的後驗平均等級 pmi_t 與 surprisal s_t 兩條標量序列的形狀特徵（std、lag-1／2 自相關、後半−前半、斜率、log 手數），按等級印平均；每局 softmax 回歸「只用平均」vs「平均＋形狀」的平衡準確率／NLL 差 = 加總看不到的訊號有多大。
- **預先指定**：`fused_opp_seq_gru vs fused_opp`（條件 (1)(3)(4)）。只報告：`fused_opp_seq_sum vs fused_opp`（應為 0 ± float32 捨入）、`conv`、`ctm_seq vs fused_opp`、`ctm_seq vs gru`。**CTM vs GRU 只在 GRU 過關時才有意義**：先確認順序裡有訊號，再問哪個 encoder 吃得最好；GRU 沒過而 ctm_seq 過不算證據，要換 seed 重現。
- **限制**：輸入是每手壓縮過的 28 個 log p，不是棋盤；若訊號在局面本身隨時間的變化（例如形勢判斷），這一階看不到，下一階才是在 trunk 特徵上做序列模型（要重新評分、成本高得多）。
- **結果（nano4，2026-10-31；`results/seq/seq_check.json`，`git_head` `517a5a4`；PR #18 合併為 `main` `914e4e2`；folds 1–4，39,327 題；`fused_opp` 重現 0.6735、每折與先前相同）**：

  | 特徵集 | 總分 | NLL | n=5 / 10 / 20 | 每折 1–4 |
  | --- | --- | --- | --- | --- |
  | fused_opp（A／B 加總） | 0.6735 | 1.1790 | 0.6271 / 0.6833 / 0.7102 | 0.6652 0.6684 0.6895 0.6710 |
  | fused_opp_seq_sum（管線錨點） | 0.6730 | 1.1782 | 0.6265 / 0.6820 / 0.7106 | 0.6642 0.6674 0.6887 0.6719 |
  | fused_opp_seq_conv | 0.6719 | 1.1828 | 0.6246 / 0.6816 / 0.7095 | 0.6649 0.6666 0.6816 0.6744 |
  | **fused_opp_seq_gru**（預先指定） | 0.6732 | 1.1790 | 0.6248 / 0.6820 / 0.7127 | 0.6658 0.6659 0.6855 0.6756 |
  | fused_opp_seq_ctm_seq | 0.6705 | 1.1833 | 0.6241 / 0.6799 / 0.7076 | 0.6640 0.6644 0.6837 0.6700 |

  | 比較 | 差 | 95% CI | 每折 | (1) (3) (4) |
  | --- | --- | --- | --- | --- |
  | **預先指定** gru vs fused_opp | −0.0004 | [−0.0029, +0.0022] | +0.0006 −0.0026 −0.0040 +0.0045 | ✗ ✗ ✗ |
  | 只報告 sum vs fused_opp | −0.0005 | [−0.0013, +0.0003] | −0.0010 −0.0011 −0.0007 +0.0008 | ✗ ✗ ✗ |
  | 只報告 conv vs fused_opp | −0.0017 | [−0.0042, +0.0009] | −0.0003 −0.0019 −0.0078 +0.0033 | ✗ ✗ ✗ |
  | 只報告 ctm_seq vs fused_opp | −0.0030 | [−0.0057, −0.0004] | −0.0012 −0.0040 −0.0058 −0.0011 | ✗ ✗ ✗ |
  | 只報告 ctm_seq vs gru | −0.0026 | [−0.0050, −0.0003] | −0.0017 −0.0014 −0.0018 −0.0056 | ✗ ✗ ✗ |

  - 第 0 步診斷（`results/seq/seq_diag.json`，118,131 局）：每局回歸「平均＋形狀」比「只用平均」的平衡準確率 +0.0085（四折 +0.0068 +0.0097 +0.0113 +0.0061）、NLL +0.0328（+0.0310 +0.0308 +0.0354 +0.0339），四折同向；按等級的平均裡 `half_diff_pmi`（後半−前半）從 D 的 −0.508 單調升到 6D 的 +0.285、`slope_pmi` 從 −1.067 到 +0.582、`std_pmi` 從 1.049 單調降到 0.751，`ac1_pmi` 呈 U 形（0.611 → 1D 0.485 → 0.582）。弱者一局之內出手品質往下掉、強者往上走——**局內的趨勢訊號存在，只是小**。
  - 訓練（`results/seq/<arch>/summary.json`）：inner CE 初始 8.56–8.92（Δ ≡ 0）→ conv 5.31–5.71、gru 5.20–5.62、ctm_seq 4.59–5.42；mean |Δ| 5.3–7.0 nats；conv／gru 的 best_epoch 9.2–10（到最後還在降），ctm_seq 2.25–9.0；每折 wall：sum 5–10 s、conv 20–25 s、gru 2–3 分、ctm_seq 11–28 分（比 §4.9 估的省很多；`seq_check` 反而要 25–48 分）。
  - nano4 修了一個 GPU 才會出現的 bug（`517a5a4`）：inner 評估後模型留在 eval 模式，cuDNN 拒絕 GRU 在 eval 模式做 backward；conv／ctm_seq 沒有依模式的層，結果不受影響；`gru` 用修好的版本重跑。另加 `seq_crossfit.py --only-folds k`（`16a67d5`）讓 CTM 四折各開一支工作。
- **解讀**：
  1. 預先指定沒過：把序列 encoder 的輸出當 L1b 加總的修正（base + Δ）接進 `fused_opp`，題目層級的分數沒有提升；`sum` 錨點 −0.0005 在 float32 捨入的預期內（兩表每局差平均 3×10⁻⁵、最大 8.7×10⁻⁴），這也是這組比較的**雜訊底線：單折 ±0.001**。
  2. **不能寫成「順序沒有訊號」**：第 0 步的 shape gain 不是 0、四折同向、趨勢特徵隨等級單調，所以是「單局回歸看得到的形狀訊號，用 base + Δ 這條通道接進 fused_opp 後沒有變成題目層級的分數」。
  3. 設計上的限制（nano4 指出，我同意）：每局 log p 的加總直接當 logits 嚴重過度自信——inner CE 8.6–8.9，而同一批局的單局 softmax 回歸 NLL 約 1.85。encoder 把 CE 從 8.7 壓到約 5.4、mean |Δ| 約 6 nats，這部分很可能主要是在修刻度而不是學順序（推論，沒有拆開量）；下游的 MLP 本來就處理得了刻度（§2.14），所以這些容量多半白花。conv／gru 到第 10 個 epoch 還在降也符合「在補刻度」的讀法。
  4. `ctm_seq` 落後 `gru` 且 CI 不含 0（只報告；依預先指定 GRU 沒過時不解讀）。在這條通道下，容量更大的 CTM 更差；`ctm_seq` 自己也沒過，`--seed 1` 重跑那條不觸發。對使用者的問題「CTM 為什麼不能一步一步學」：可以，而且學得比 GRU 更會壓 inner CE（4.59–5.42），但在這個資料、這條通道上沒有變成題目層級的分數。
- **結論**：提交維持 A／B 的 `fused_opp`。這條線（base + Δ）結束。若要再問「局內順序」，應改走**獨立通道**而不是修正加總（新實驗，要另外事先指定）：(a) 便宜的做法——把第 0 步的每局形狀特徵（`std_pmi`、`ac1_pmi`、`half_diff_pmi`、`slope_pmi` 等）在每題的局上取平均，接在 56 維 `fused_opp` 後面成 `fused_opp_shape`，CPU 幾分鐘，預先指定 vs `fused_opp`；(b) 或把 Δ 限制成只能學形狀（序列輸入先減掉每局的平均，encoder 就修不了刻度）。CTM 逐手這條，只有 (a)／(b) 顯示形狀訊號能進到題目層級後才值得再開。

### 2.16 獨立通道 fused_opp_shape：每題局內形狀特徵的平均接在 fused_opp 後面（2026-10-31；issue #21；nano4 待跑）
- **使用者決定（2026-10-31）**：做 §2.15 結論的 (a)。形狀特徵不去修正 L1b 加總，直接當額外欄位給 MLP；提交管線不動，不碰 fold 0。
- **做法**（`aicup.seqshape`、`scripts/shape_check.py`）：
  - 每局形狀特徵 = 第 0 步診斷的 9 個（`game_shape`，從 `seq_diag.py` 搬到 `aicup.seqshape`，數值不變）：每手 L1b 後驗平均等級 pmi_t 與 surprisal s_t 的 mean_pmi、log_moves、std_pmi、ac1_pmi、ac2_pmi、half_diff_pmi、slope_pmi、mean_s、ac1_s。來源：`scores3`／`scores3_B` 的每手 npz（目標手）、`scores3_opp`／`scores3_B_opp`（對手手），與 `fused_opp` 同一組 OOF 評分。
  - 每題：題內各局的形狀特徵取**平均**（`question_shape`；每局的量都是強度量，不隨局數累加，所以取平均而不是加總），去掉 log_moves（n_moves 已在 16 維 L1b 特徵裡）→ 8 維。`fused_opp_shape` = 56 + 8 = 64 維；`fused_opp_shape2` = 再加對手手的 8 維 = 72。MLP 本來就把輸入標準化，刻度不是問題。
  - CV 與 `l1b_opp_check` 完全相同（pilot 驗證題、`sample_questions` 32／人、`RandomState(42 + v)`、`MLPModel(hidden=32)`、玩家 bootstrap），只多算形狀表；CPU 一支工作。
- **預先指定**：`fused_opp_shape vs fused_opp`（條件 (1)(3)(4)）。只報告：`fused_opp_shape2 vs fused_opp`、`fused_opp_shape2 vs fused_opp_shape`。雜訊底線沿用 §2.15 的單折 ±0.001。
- **預期**：單局層級的 shape gain 是 +0.0085 平衡準確率（§2.15）；題目層級對 5–20 局取平均，形狀特徵的雜訊縮 √n，但 56 維已帶走大部分等級訊息，效應應該小；門檻就是條件 (4) 的 0.005。
- **決策**：過 → 另開 issue 接進提交（`predict_submission.py` 的分片已存每手 L1b 陣列，目標與對手都有，形狀特徵可在分片後算，不需重新評分；`fit_fused_model.py` 加 shape 欄）。沒過 → 單局形狀訊號在題目層級被 56 維吃掉或太小，局內順序這條線（§2.15 + §2.16）結束，提交維持 `fused_opp`。
- **結果**：待 nano4（指令在 NANO4 §4.10）。

### 2.17 80 分的帳與路線：測試玩家是否在訓練集裡（2026-10-31；issue #22；程式在 AI_CUP_2026-2 PR #4；nano4 待跑）
- **帳**：官方分 = 命中 1、差一級 e⁻¹ ≈ 0.368、其餘 0。fold 0 命中 0.519、±1 以內 0.917 → 0.67。要到 0.80：±1 以內維持 0.917 時命中要 0.73，±1 以內做到 1.0 時要 0.68，即命中率要升 16–21 個百分點。落子通道的六個實驗（§2.7、§2.12–2.16）變動都在 ±0.005 內，這條通道已飽和。
- **13 分在哪**（§2.13）：六成誤差在玩家之間（同一人的題全對 16%、全錯 13%）；多給棋局只壓玩家內的四成（n=5／10／20 命中 0.496／0.544／0.570，外推到 100 局也只約 0.62）；偷看標籤扣掉每人固定偏差的上界 0.844。**標籤（伺服器等級）與「這些局下出來的棋力」之間有每人固定的落差，落子裡量不到**；落差只能從兩處來：(A) 測試玩家就是訓練玩家，認出來直接查標籤；(B) SGF 元資料（勝負、讓子、貼目）。
- **(B) 元資料**：本機 mock 測試檔由 `load_games` 原樣複製訓練 `sgf_content`，每局只有 `(;SZ[19];B[..];W[..]…)`，沒有 RE／HA／KM／DT；nano4 以 grep 在訓練 CSV 上確認（NANO4 §4.11 步 0）。預期為 0，則 (B) 關閉。
- **(A) 辨識**（使用者決定 2026-10-31；程式在 AI_CUP_2026-2：`aicup_pid.opening` 加 `color`、`scripts/pid_rank_overlap.py`，設計在那邊的 METHOD_NOTES §2.3）：線索是官方 baseline 的 Public 0.4631 高於本協定 n=20 的 CI 上界（§1）。候選庫 = 8,007 位訓練玩家 × 黑／白兩個開局模型（競賽一的模型在 400 人、100 局候選下 Top-1 0.93）；查詢 = fold 的 pilot 假題（與 `fold0_fused_opp_proba.csv` 同一組 9,872 題），題目裡的局全部從候選庫拿掉；一題 = 各局 log 似然相加，margin = Top-1 − Top-2。**事先指定**：fold 1 選 τ 使 margin ≥ τ 的題辨識精確率 ≥ 0.98；fold 0 只評一次：margin ≥ τ → 認出者的等級，否則 `fused_opp_avg` 的 expected-score 決策；報 combined vs fused_opp（玩家 bootstrap；條件 (1) 與 (4)）、Top-1／Top-5 按 n、全信辨識的 id_rank、認對／認錯的 margin 分布。另測同一人多題合併（同大小其他題中最像的是否同一人）當測試玩家不在訓練集時的退路。
- **11/04 的判準**：對官方測試檔跑 `--test-csv`：margin ≥ τ 的題比例接近 fold 0 的水準 → 測試玩家在訓練集裡，另開 issue 談接進提交（認出者的等級取代 fused_opp 的決策，其餘不動）；margin 全落在 fold 0「認錯」的分布 → 不重疊，只剩多題合併的退路。技術上只用主辦方給的訓練資料，但它靠的是「測試玩家在訓練集裡」而不是棋力模型，**用不用由使用者決定**。
- **沒排的次要路線**：humanv0 value head 的每手勝率損失（落子通道裡唯一沒用過的形式，但量的仍是棋力，預期 +0.00–0.02）；§2.16 的 `fused_opp_shape` 等 nano4。
- **結果**：待 nano4（NANO4 §4.11）。

## 3. 成本：0B 實測（2026-10-16，lab01 單卡 RTX PRO 6000，torch 2.14.1+cu130，`scripts/bench_humansl.py` → `results/bench_0B.json`）

| 量測 | 數字 |
| --- | --- |
| 純推論 humanv0（b18c384），真實局面預先上 GPU | fp32：batch 64 **3,194**、256 2,841、1024 2,136 局面/秒；bf16 autocast：batch 64 **4,296**、256 3,714、1024 2,473 局面/秒 |
| 吞吐隨 batch **下降**（反常） | 推測是 KataGo 模型內 mask／norm 等 elementwise 運算在大 batch 時變成記憶體頻寬瓶頸（activations 溢出 L2）；bf16 只比 fp32 快 35%，遠低於 tensor core 理論值。先用 batch 64；之後可試 channels_last、`cudnn.benchmark`、`torch.compile`、TensorRT |
| 端到端 `target_move_logps`（逐局面單一前向，含重播、特徵、合法步遮罩） | **87.5 目標落子/秒**（200 局、22,572 手） |
| 只做重播＋`get_input_features` | 226 局面/秒／核 ← **端到端的瓶頸是 CPU 特徵，不是 GPU** |
| 只做 361 次 `would_be_legal` 遮罩 | 3,580 局面/秒／核 |

推算（以上述實測）：
- 全資料 1.11 億目標落子 × 10 profile = 11.1 億次：GPU bf16 batch 64 兩卡 ≈ **36 小時**；CPU 特徵 1.11 億 ÷ 226/秒 ≈ 136 核時（16 核約 8.5 小時，可與 GPU 重疊）。
- b18c384 每局面約 34 GFLOP；4,296 局面/秒 ≈ 146 TFLOPS 有效，離 bf16 峰值很遠，有優化空間。
- 晚注入自訓模型約省 K(C_t+C_h)/(C_t+K C_h) 倍：K=10、head=5% trunk 時約 7 倍。
- 1.11 億 × 10 × 362 × float32 ≈ 1.6 TB，所以只存實際落子 log p、熵、value 摘要。

### 3.1 算力分工（2026-10-16）

| | lab01 | 國網 nano4 |
| --- | --- | --- |
| GPU | 2 × RTX PRO 6000 Blackwell 96 GB（nvidia-smi 實測） | 每節點 8 × H200（sinfo，由使用者在登入節點 25a-lgn04 查得） |
| CPU | Core Ultra 9 285K，24 核（8P+16E），251 GB | 計算節點核數未查 |
| 排隊／時限 | 無排隊；帳號共用（13 個 session），GPU 可能被佔 | Slurm，計畫 gov113074、QOS normal。dev 4 小時；8gpus、16gpus 2 天；32/64gpus 1 天；256gpus 12 小時；ngs1–8gpu 14 天（各 8 節點，是否可用未確認）。查詢當下 30 節點閒置、115 mix |
| 存取 | 金鑰 SSH，Claude 可直接跑 | 只接受密碼／OTP 互動登入，Claude 不能直接操作；GPU 小時餘額在 NCHC 計費系統，未查 |

H200 對 RTX PRO 6000 的單卡速度比 r **未實測**。以 0B 的 4,296 評估/秒/卡推算：
- 全量 Human SL 特徵（10 個 profile，1.11×10⁹ 次評估）：lab01 約 36 小時；nano4 一個 8 卡節點 r=1 約 9 小時、r=2 約 4.5 小時，都在 8gpus 的 2 天時限內。CPU 特徵 1.11×10⁸ 局面 ÷ 226/秒/核，需要計算節點核數才能估。
- L1b 一個 epoch（假設訓練吞吐 ≈ 推論的 1/3）：lab01 約 11 小時；nano4 一節點 r=1 約 2.8 小時、r=2 約 1.4 小時。
- 分工：開發、測試、pilot、加速、評估與校準留在 lab01；全量特徵、L1b 全量訓練與掃描丟 nano4。第一步在 dev 分區用 1 張 H200 跑 `scripts/bench_humansl.py` 與 `nproc`，把 r 和核數換成實測值。
- nano4 環境與工作腳本（2026-10-19）：`scripts/nano4/`（setup、smoke、run、shards），操作步驟見 `docs/NANO4.md`。torch 2.14.1 預設裝 cu126 版（lab01 是 cu130），其他套件版本照 lab01 釘在 `requirements.txt`。冒煙測試會跑完上面的第一步，並重算 fold 0 的 200 局、和 lab01 的分數逐局比對。

## 4. 開放問題
- 測試集玩家是否與訓練集重疊 → §2.17（issue #22）：fold 0 模擬的 8,007 路辨識可行性先跑，11/04 以 `--test-csv` 判定。
- ~~本資料集等級尺度 ↔ KGS `rank_*`~~ → pilot 已回答：尺度被壓縮（D≈6k…6D≈2d），硬分箱失敗，必須學映射（§2.4.1）。
- ~~對手等級對 Human SL 似然的影響~~ → pilot 已回答：相鄰一級 ≤ 0.024 nats/手，遠小於我方等級效應（§2.4.1）。
- 1D|2D、3D|4D 為什麼分不開：Human SL 解析度不足，還是本資料這兩對標籤本身重疊？要靠 L1b（用本資料標籤訓練）回答。
- 測試集是否含虛手／殘局（11/04 後重跑 check_legal）。
- 測試集每題的棋局順序是否有意義（若為時間序，順序本身帶訊號；聚合器先設計成順序無關，11/04 後測順序敏感度）。
- ~~局內落子的順序是否帶加總看不到的等級訊號~~ → §2.15 部分回答：單局層級有小而一致的趨勢訊號（弱者越下越差、強者越下越好），但以 base + Δ 修正 L1b 加總接進 fused_opp 沒有提升題目層級的分數；獨立通道（每題的形狀特徵）見 §2.16（issue #21，待 nano4）。
