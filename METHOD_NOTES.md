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

## 4. 開放問題
- 測試集玩家是否與訓練集重疊。
- ~~本資料集等級尺度 ↔ KGS `rank_*`~~ → pilot 已回答：尺度被壓縮（D≈6k…6D≈2d），硬分箱失敗，必須學映射（§2.4.1）。
- ~~對手等級對 Human SL 似然的影響~~ → pilot 已回答：相鄰一級 ≤ 0.024 nats/手，遠小於我方等級效應（§2.4.1）。
- 1D|2D、3D|4D 為什麼分不開：Human SL 解析度不足，還是本資料這兩對標籤本身重疊？要靠 L1b（用本資料標籤訓練）回答。
- 測試集是否含虛手／殘局（11/04 後重跑 check_legal）。
- 測試集每題的棋局順序是否有意義（若為時間序，順序本身帶訊號；聚合器先設計成順序無關，11/04 後測順序敏感度）。
