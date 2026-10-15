# 方法筆記（收斂結論，持續更新）

最後更新：2026-10-15。只記「已確認的事實」與「已收斂的決策」，討論過程不留。資料與規則見 [REPORT.md](REPORT.md)。

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
| Analysis engine 的 `humanSLProfile` 是單一欄位，一個 profile 一次查詢；人類行為分析建議 `ignorePreRootHistory=false` | `docs/Analysis_Engine.md` |

## 2. 已收斂的決策

### 2.1 驗證協議（所有實驗共用）
- 依 `player_id` 整人留出，每級約 10% 玩家；1D 只有 433 人，用 5-fold。
- 留出玩家各組 5／10／20 局假題，每人題數設上限；另加「總目標手數相近但局數不同」的對照組。
- 指標：官方 ±1 加權分 + 查詢級 NLL + 校準。以玩家為重抽樣單位報不確定性。
- 校準／映射用 out-of-fold，不與最終本地評估共用同一批玩家。
- 同時維持一組「已見玩家」驗證集，兩者差距 = 測試集玩家重疊與否的敏感度。
- 先在新協議下重跑原始 baseline 取參考分；本地分與 Public 0.4631 不直接比高低。

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
- 算量：200 × 20 × ~110 × 18 ≈ **7.9×10⁶** 次評估。以未實測的 3,000–7,000 次/秒/卡，純前向約 19–44 分鐘（不含重播、特徵、I/O）。
- 記錄假設：每次評估都隱含「對手與我方同級、KGS、2020、讀秒制」。另用小子集固定我方、掃對手等級，量 Δlog p。
- 解讀：測的是「既有模型對本資料的轉移」，不是方法上限；零樣本不好 ≠ 自訓條件模型不可行。

### 2.5 步驟 0（pilot 之前）
- 0A 一致性：同一局面、同 profile（完整重現引擎 metadata，不是只填 rank）、同權重（確認 SWA 與否）、固定對稱，比較 Python 直讀與 analysis engine 的目標落子 log p。涵蓋黑白、各階段、提子、打劫。用 FP32 `log_softmax`，合法步語意（引擎非法步 = −1）要對齊。
- 0B 吞吐：分開記純推論（warm-up、同步）與端到端（SGF→特徵→log p）。只讀 policy 不等於沒算 value head。

### 2.6 Ablation 清單（便宜優先）
執色平面｜不截開局 vs 最後 250 vs 分段取樣｜明確局長特徵（兩種表示都給）｜只用局長統計的 baseline｜只計我方 vs 兩方都算（檢驗「對手等級相關」假設）｜按局取樣 vs 等級→玩家→局分層取樣（不刪資料）｜鄰級 label smoothing（當序數正則化，不當指標對齊）｜聚合方式三選一。

## 3. 成本估算（全部未實測）
- b18c384 每局面約 34 GFLOP（18 blocks × 2 convs × 384² × 9 × 361 MACs ×2）。
- 全資料 1.11 億目標落子 × 10 profile = 11.1 億次；兩卡各 3–7k/s → 22–51 小時。晚注入自訓模型約省 K(C_t+C_h)/(C_t+K C_h) 倍：K=10、head=5% trunk 時約 7 倍。
- 1.11 億 × 10 × 362 × float32 ≈ 1.6 TB，所以只存實際落子 log p、熵、value 摘要。

## 4. 開放問題
- 測試集玩家是否與訓練集重疊。
- 本資料集的伺服器與等級尺度 ↔ KataGo KGS `rank_*` 的對應（決定 w 是硬分箱還是學習的軟映射）。
- 對手等級對 Human SL 似然的影響大小（2.4 的掃描決定）。
- 測試集是否含虛手／殘局（11/04 後重跑 check_legal）。
