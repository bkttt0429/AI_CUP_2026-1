# 從棋步估計棋力／等級的模型文獻（圍棋與西洋棋，2015–2026）

*範圍：只涵蓋「由棋步序列推估棋手強度」的模型文獻。任務說明中列為「已使用」或「已判死」的特徵不再重提，只在相關處標註「與專案結論一致」。所有數字引自原始論文或官方文件；無法取得原文處列於各節 Gaps。撰寫日期 2026-11-04。*

## KQ1 主要論文的輸入、消融、各特徵家族的貢獻與失敗

### Takeaway
目前最強的公開結果是 Chen–Shih–Wu（ICLR 2025）的 Bradley–Terry 強度分數（SE）與 Kuboki 等人（2025）的「多等級 policy 似然幾何平均 + SE + 失分 → LightGBM」。Kuboki 的消融顯示：在已有 10 個等級 policy 似然時，SE 分數仍帶來 +4.5～+6.1 個百分點（10 局），失分（move loss）幾乎為零；Neubauer（2024）顯示 KataGo trunk 向量遠優於 head 輸出的標量特徵。西洋棋端（Maia-2、ChessMimic）完全沒有做等級推估實驗；Omori–Tadepalli 的 182 Elo MAE 約四分之一來自時鐘。

### Cited Findings

**A. Chen, Shih & Wu, "Strength Estimation and Human-Like Strength Adjustment in Games", ICLR 2025**
- 輸入 18 平面（過去 8 手雙色盤面 16 平面 + 2 平面執棋色），AlphaZero 型 20 殘差塊 × 256 通道；同一網路輸出 policy、value、強度分數 β，λ = e^β（Bradley–Terry）；SE 損失為依等級由強至弱排序的 sequential softmax（Plackett–Luce 型，Eq. 5），使各級分數嚴格遞減 — [arXiv 2502.17109](https://arxiv.org/abs/2502.17109)
- 圍棋資料：Fox 弈城，11 個等級（9d 至 3–5k；1–2k 合為一類；5k 以下因等級變動快而排除）；訓練 495,000 局（每級 45,000），candidate set 1,100 局（每級 100）、query set 9,900 局（每級 900）；每個訓練樣本每級取 m = 7 個 state-action pairs — 同上
- 聚合：對 state-action 的 λ 取幾何平均 = exp(mean β)（Eq. 2–3）；因「開局各級分數相近（開局容易背）」，以手數 d 的級別均值 β̄^d 為比對目標；未知棋手取 |β̄_u − β̄_i| 最小的等級 — 同上
- 基線 SL_sum / SL_vote（單一 state-action 的 11 類分類器，分別以機率加總／多數投票聚合）：100 局後仍只有 49%；SE 與 SE∞：15 局 >80%、100 局 97.5%；±1 級容忍下 1 局近 80%、6 局 >99%（Figure 2a） — 同上
- 每局只取 1 個 action、20 局時約 60%（全部 action 約 80%，Appendix B.2）；只用 2 個等級訓練需 38 局達 80%、3 級 21 局、11 級 15 局（Sec. 4.4，顯示訓練等級越多越好、且能預測未訓練的等級） — 同上
- 各級準確率（Figure 2b）：9d 最高、8d 最低（強 9d 被推入 8d）、3d 與 3–5k 低於平均（Fox 可從 3d 起算、kyu 等級變動快）；只用前 50 手（fuseki）時 kyu 級準確率大降；只用官子時 8d 下降（Appendix B.3–B.4） — 同上
- 西洋棋：Lichess 1000–2600 共 8 個 200 分段、每段 240,000 局；SE∞ 26 局 >80%、100 局 93.38%（Figure 6） — 同上
- 訓練成本：圍棋約 242 GPU-hours、西洋棋 69（RTX A5000）；論文未分析對手棋步、未報告對稱增強；SE 與 SE∞（多一個隨機合法手類別 r∞）在等級推估上「幾乎相同」 — 同上
- 投影片摘要：SE 在圍棋 15 局、西洋棋 26 局內達 80% — [ICLR 2025 slides](https://www.iclr.cc/media/iclr-2025/Slides/30488_2qkfdUL.pdf)

**B. Kuboki, Ogawa, Hsueh, Yen & Ikeda, "Policies of Multiple Skill Levels for Better Strength Estimation in Games", arXiv 2505.00279（2025）**
- 特徵三類：(1) SE 分數 β 的算術平均；(2) 10 個 KataGo HumanSL preaz 等級（10k, 8k, …, 2k, 1d, 3d, …, 9d）下「所下之手機率」的幾何平均 GM_j；(3) 失分 l_i = v'_i − v_i（圍棋用 KataGo scoreLead；西洋棋用 LC0 勝率 logit），統計量 MeanLoss / MedLoss / StdLoss，只取前 N_cut 手（50/100/∞）；圍棋最終選 MeanLoss(N_cut=100) 與 MedLoss(∞)，西洋棋選 MeanLoss 與 StdLoss(N_cut=50) — [arXiv 2505.00279 (HTML)](https://arxiv.org/html/2505.00279v1)
- 模型：LightGBM 預設參數的迴歸，四捨五入為等級；對 n 局的特徵取平均；每級重抽 10,000 次（圍棋）、26,000 次（西洋棋）；圍棋資料 Fox 每級 5,000 局 × 11 級，限雙方同級、≥50 手、以 pass 或投子結束；測試每級 900 局（與訓練不重疊）；準確率以每級 500 次重複計算 — 同上
- Table 1（隨機抽樣，exact(±1)，Proposed vs Chen 重現）：n=1 37.5(82.8) vs 33.7(74.0)；n=5 64.8(98.8) vs 56.3(96.6)；n=10 79.6(99.9) vs 71.4(99.7)；n=15 87.5(100) vs 78.2(99.9)；n=20 91.7(100) vs 83.8(100)。西洋棋：n=10 74.3(99.7) vs 66.5(99.1)；n=20 86.2(100) vs 79.5(99.9) — 同上
- Table 2（圍棋消融，隨機抽樣）：n=10 全部 79.6 / w/o Strength 73.5 / w/o Prior 73.0 / w/o Loss 79.8；n=20 91.7 / 87.7 / 86.4 / 91.9；n=1 37.5 / 34.4 / 32.3 / 37.9 — 同上
- Table 3（西洋棋消融，隨機抽樣）：n=10 全部 74.3 / w/o Strength 63.7 / w/o Prior 69.5 / w/o Loss 72.9 — 同上
- Table 4（圍棋，「同一棋手」抽樣 vs 隨機抽樣）：player-specific n=5 58.1(95.3)、n=10 61.0(96.2)、n=15 63.2(96.6)；Chen 重現 51.6(93.1)、59.4(96.0)、61.1(96.9)；同組實驗的隨機抽樣為 70.4/82.8/89.4 — 同上
- Table 7（圍棋 player-specific 消融）：n=5 全部 58.1 / w/o Strength 53.1 / w/o Prior 52.5 / w/o Loss 56.7；n=10 61.0 / 56.5 / 59.0 / 61.7；n=15 63.2 / 57.8 / 61.1 / 62.8（n≥10 時去掉失分反而略升） — 同上
- Table 8（西洋棋 player-specific 消融）：n=15 全部 60.4 / w/o Strength 60.2 / w/o Prior 56.2 / w/o Loss 59.8（SE 在此設定下幾乎無貢獻） — 同上
- Sec. 5.3 與 Table 6（西洋棋 n=20，單一 Maia 等級 vs 兩個）：Maia-1100 單獨 28.1%、Maia-1200 單獨 24.3%、兩者合用 35%；Maia-1500 單獨 13.8%、Maia-1900 單獨 33.0%、兩者 65.1%；Maia-1500 只對第 3–4 組強、Maia-1900 只對第 0–1 組強；圍棋中 2k/1d prior 可分辨 8d 與 9d，5d/7d 不能（Fig. 2） — 同上
- 失分變異在中盤最大（圍棋約第 100 手，Fig. 3）；Appendix 0.F 指出失分單獨使用時資訊量弱；論文無專門的限制章節、未討論對手效應 — 同上
- 前作（GPW 2024，Kuboki, Hsueh, Ikeda）：以失分估棋力時「跳過連續高失分手、忽略後半盤、改用受局面影響較小的 goodness-of-shape 指標」，少量棋譜下 SD 0.387、RMSE 0.752（單位未確認，關鍵字含 KataGo HumanSL） — [JAIST repository](https://dspace.jaist.ac.jp/dspace/handle/10119/20018)；[IPSJ record](https://ipsj.ixsq.nii.ac.jp/records/240735)

**C. Neubauer, "Strength Estimation in the Game of Go", TU Wien diploma thesis, 2024-10**
- 資料 OGS 2021 collection；過濾：19 路、無讓子、每手 >5 秒、≥20 手、ranked、只留數子／投子／超時結束且以 KataGo 覆核勝者（<60% 確信者丟棄）；訓練 10,000 局、驗證 5,000、測試 5,000、exhibition 5,000（雙方近史 ≤4 局）；每位棋手取最近 N = 500 手（約 4 局） — [thesis PDF](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)；[reposiTUm](https://repositum.tuwien.at/handle/20.500.12708/212251)；[code](https://github.com/Animiral/go-strength-model)
- 輸入：KataGo kata1-b18c384nbt-s9131461376-d4087399203 的 trunk 輸出（19×19×384）在落子點的 384 維向量（落子前局面），不使用任何 head；Basic model（消融）只用 6 個標量：落子後勝率、落子後目差、該手 policy 值、該局面最大 policy 值、勝率損失、目數損失；Stochastic model 只用目數損失的常態模型 — 同上（Sec. 4.2, 4.3, 4.5, 4.6）
- 架構 Set Transformer（ISAB），最佳超參數 l=5、d=254、d_q=154、inducing points m=61，共 2,469,388 參數；目標 = 未來 Glicko-2 評分（A=10 局後，標準化）MSE + Bradley–Terry 勝負 NLL + 正則；N 在初步搜尋後固定為 500 — 同上（Table 5.1, Sec. 4.7, 5.2）
- Table 5.2（測試集 accuracy α / log-likelihood λ）：50:50 61.9%/−0.69；Glicko-2 69.1%/−0.58；Future Glicko-2 75.4%/−0.51；Stochastic 58.5%/−1.18；Basic 65.2%/−0.61；Full 68.8%/−0.58。Exhibition set：Glicko-2 64.4%/−0.64、Basic 70.9%/−0.57、Full 73.7%/−0.55 — 同上
- 結論：trunk 含有比 head 輸出更多的強度資訊（假說獲證實）；僅目數損失遠不足以預測（甚至低於 50:50）；模型「迴避給出最高等級」、低段預測散佈大、對 KataGo 棋風有偏（其他強 bot 被判為較低）；資料有 Glicko-2 配對（matchmaking）偏差 — 同上（Sec. 6.1–6.3）
- 相關工作節引用 KataGo 作者建議：可將 humanSL 在各等級下對實際落子點的 policy 值取最大者作為棋手等級（即本專案現行主線） — 同上（Sec. 3.4）

**D. Omori & Tadepalli, "Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM", arXiv 2409.11506（2024）**
- 1.2M Lichess 局（2021-04～2024-07，每月 30,000 局），12 個棋子平面 + 標準化剩餘時間；4 層 CNN + BiLSTM + 2 層全連接，逐手輸出雙方 rating；評分 mean 1514、SD 366；無手工特徵、未用 Stockfish/Maia — [arXiv 2409.11506](https://arxiv.org/abs/2409.11506)
- MAE：RatingNet 182、NoClock 239、均值基線 346（Bullet 182/274/340；Rapid 182/207/305；Classical 151/185/246）；MSE 56,618 / 91,619 / 183,538；去掉時鐘平均 +57 分（約 24%），Bullet +92 分（約 34%）；無「每位棋手局數」消融、無 Maia／引擎基線；高分段（2922 vs 3163）被預測為約 1460，歸因於高分樣本稀少 — 同上

**E. Maia（KDD 2020）、Maia-2（NeurIPS 2024）、ChessMimic（2026）**
- Maia：每個模型以約 12M 局訓練，測試集每段 10,000 局（雙方同段）；Table 1 峰值：Maia-1100 在 1200 段 50.8%、1300→1400 51.8%、1500→1700 52.2%、1700→1800 52.7%、1900→1900 52.9%；最低 46%（Maia-1900 預測 1100 棋手）；Stockfish 深度 1–15 為 33–41%、Leela 最高 46% 且對等級平坦 — [ar5iv 2006.01855](https://ar5iv.labs.arxiv.org/html/2006.01855)
- Maia 失誤預測（勝率降 ≥10 個百分點）：僅盤面 ResCNN 67.7%，加 rating／剩餘時間／centipawn 71.7% — 同上
- Maia-2：skill-aware attention（雙方等級的 128 維類別嵌入加到注意力 Q），Lichess Rapid 169M 局；平均吻合率 53.25%；消融（subset 52.74）w/o Att 51.75、w/o Aux 52.09；改變自身等級輸入可改變預測著法至多 22% 的局面、改變對手等級至多 6%；單調一致（coherence）27.61/28.51/26.38% vs Maia-1 的 1.61/1.42/1.14%；**論文沒有任何由棋步推估等級的實驗** — [arXiv 2409.20553](https://arxiv.org/abs/2409.20553)
- ChessMimic（Johnson, 2026）：14 個 Elo 區段各自一組 9M 參數 encoder-only transformer（著法／時鐘／勝負三模型）；rating 僅作為輸入（elo_self）；全段 top-1 0.5623 vs Maia-2 0.5260，各段 +3.0 到 +6.1 pp；**無等級推估**；論文自述沒有任何隔離消融 — [arXiv 2606.04473](https://arxiv.org/html/2606.04473)

**F. 其他「由棋步推 Elo／等級」工作**
- Moudřík, Baudiš & Neruda 2015（KGS，6d–20k 共 26 級，每級 120 個樣本、每樣本 10–50 局；bagged NN）Table I RMSE（級）：均值迴歸 7.503；Patterns（N=400 最常見局部空間模式 + 叫吃旗標）2.788；ω-local 先後手序列 5.765；邊距直方圖 5.818；提子統計 5.904；勝負統計 6.792；勝負目數 5.116；全部合併 2.659 — [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)
- Moudřík & Neruda 2016：188,700 局 OGS，約 7M 參數 CNN，3 類（弱／中／強），單一局面 71.5%、一局聚合 77.9%（摘要） — [Springer](https://link.springer.com/10.1007/978-3-319-49001-4_15)
- Kaggle Finding Elo（2015，單局推雙方 Elo）第 2 名 MAE ≈160：Stockfish 特徵「最佳手更換次數」、「Stockfish 與棋手一致的最深深度」、Guid–Bratko 複雜度、最差 25% 手的 cp 損失（q_error_one）等分位數、24 ply 開局庫的 Elo 統計、逐手 RandomForest；每手分析 16 秒（時間加倍只增 0.3–0.4 分）；Regan 式序列貝氏模型單獨有效但與既有特徵冗餘；加入外部 50k 局反而變差；blitz/rapid 旗標值約 5 分但測試集不可用 — [dsjoerg/blundercheck](https://github.com/dsjoerg/blundercheck)
- Regan–Biswas（ICMLA 2015）：以 u_i = exp(−(δ(v_1,v_i)/s)^c) 對所有合法手建模，s = 靈敏度、c = 一致性；「satisficing 深度」（最佳手被更深搜尋揭露為錯誤的深度 d−1）隨等級穩定上升：初學者近 0、頂尖「勉強超過深度 10」；該指標只由出錯的手推得；資料 1.08M 手／17,500 局 — [Lipton–Regan blog 2015](https://rjlipton.com/2015/10/06/depth-of-satisficing/)；[ICMLA 2015 abstract](https://researchconnect.buffalo.edu/en/publications/measuring-level-k-reasoning-satisficing-and-human-error-in-game-p/)
- Regan, Biswas & Zhou（AAAI-14 workshop）：每手平均誤差隨任一方優勢幅度上升、人類遠較電腦明顯；人類偏好把局面導向「合理選項較少、較早解決」的局面 — [AAAI-14 WS PDF](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- Chessable Research Award 2025「Rating Your Moves」（Jim Yang）：僅盤面 64 層編碼、逐手貝氏更新；未公布任何數字 — [chess.com blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)

**G. KataGo Human SL（b18c384nbt-humanv0）官方文件（沒有論文）**
- 2024-07 隨 v1.15.0 發布，「以大量人類棋譜監督學習，預測各等級、各時代人類的著法」，可模擬讓子局，官方列出的開發者用途之一為「棋手等級估計」 — [katagotraining extra networks](https://katagotraining.org/extra_networks/)
- Profiles：preaz_20k…preaz_9d（2016 前開局風格）、rank_20k…rank_9d（AlphaZero 後）、**rank_{BR}_{WR} / preaz_{BR}_{WR}（黑 BR 級對白 WR 級，模型「知道對手較強或較弱」）**、proyear_1800…proyear_2023；等級差 >9 級或與讓子不符者為分佈外 — [gtp_human5k_example.cfg](https://github.com/lightvector/KataGo/blob/master/cpp/configs/gtp_human5k_example.cfg)
- 分析引擎輸出欄位：humanPolicy、moveInfos[].humanPrior、rootInfo.humanWinrate、humanScoreMean、humanScoreStdev、humanStWrError、humanStScoreError（最後兩者為該 profile 下的短期勝率／目數不確定度）；建議 ignorePreRootHistory=false、rootNumSymmetriesToSample 2 或 8「以得到較不嘈雜的 policy」；警告訓練資料的等級可能登錄不準；讓子局白方通常較強且部分伺服器讓子不足，故讓子局的勝率／目數可能不準 — [Analysis_Engine.md Human SL guide](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide)

### Inferences
- Kuboki 的消融是目前唯一直接回答「多等級似然之外還有什麼」的量化證據：在 10 級 policy GM 已存在時，去掉 SE 分數掉 6.1 點（n=10 隨機）／4.5 點（n=10 同一棋手），去掉失分幾乎不變（甚至略升）。SE 分數由等級排序的 Plackett–Luce 損失學到，與任何單一 profile 的似然都不是單調關係。注意：專案記憶顯示 2026-10-29 的 SE/ordinal 實驗未通過，故此處不重提 SE，只列出可核對的實作差異（見 KQ2 第 7 項）。
- Neubauer 的 Basic（6 個 head 標量）vs Full（trunk 384 維）差距（exhibition λ −0.57 vs −0.55；test α 65.2 vs 68.8）支持專案已採用 trunk 向量的路線，也暗示「該手 policy 值、最大 policy 值、勝率損失」等標量在 trunk 面前多屬冗餘。
- Omori 的 182 vs 239 MAE 表示時鐘約佔單局 Elo 訊號的四分之一；本專案無時鐘，是硬上限的一部分。
- Chen 的 SL_sum（49%@100 局）vs SE（97.5%）差距主要來自聚合方式：每手分類機率加總會被大量「各級都一樣」的手淹沒；專案以 whole-question loss 訓練的 TCN/MLP 已避開此問題。
- Finding Elo 第 2 名與 Regan 的經驗一致：在已有失分統計時，再加一個「序列貝氏技能模型」是冗餘的；這與專案「value/score 失分為死」的結論同向。

### Gaps
- Chen 論文 Figure 2a 在 1/5/10/20 局的精確數字僅見於圖；此處以 Kuboki Table 1 的「Chen et al.」重現值（Fox 資料）代替。
- Neubauer 論文無「局數 vs 準確率」曲線（N 固定 500 手），亦無對稱增強或對手消融。
- Maia-2 全文最後約 4.8k 字元未讀；ChessMimic 無等級推估內容可補。
- Kaggle Finding Elo 第 1 名寫法僅見論壇片段（GBM + Stockfish deltas/means/stds），討論頁無法抓取（[thread](https://www.kaggle.com/c/finding-elo/discussion/13008)）；評分指標與第 1 名 MAE 未能確認。
- IPSJ 日文論文「囲碁における人間の着手を模倣するには KataGo の HumanSL だけで十分か?」（[record](https://ipsj.ixsq.nii.ac.jp/records/240730)）兩次 HTTP 503，無法取得其對 HumanSL 失配的分析。
- Kuboki GPW 2024 的 SD 0.387 / RMSE 0.752 的單位（等級？）未能確認。

## KQ2 哪些輸入不是「所下之手 policy 似然」的單調函數，且有證據帶額外棋力資訊？

### Takeaway
文獻中有直接消融證據的「超越單一似然」訊號只有三類：(1) 跨等級似然的**形狀**（哪個等級的 prior 最能解釋，而非任一等級似然的高低；Kuboki Table 6：單一 Maia 13.8–33.0% → 兩個 65.1%）；(2) 以等級排序損失學到的每手強度分數（SE）；(3) 搜尋深度／最佳手變更等「全分佈」特徵（Regan 的 (s,c) 與 satisficing 深度、Finding Elo 的 agreement depth 與 Guid–Bratko 複雜度）。Policy 熵、top-k 名次、相對最佳手機率、等級條件 value 等雖非似然的單調函數，但**沒有任何論文對其做過棋力推估的消融**。

### Cited Findings
- 同一等級 prior 的似然對不同區段的鑑別力不同：Maia-1500 單獨只對第 3–4 組強（13.8% 整體）、Maia-1900 單獨只對第 0–1 組強（33.0%），兩者合用 65.1%（n=20）；圍棋 2k/1d prior 可分 8d 與 9d，5d/7d 不能 — [Kuboki 2025](https://arxiv.org/html/2505.00279v1)
- 每個 Maia 的吻合率在其訓練等級附近達峰（50.8–52.9%），離峰最低 46%；Leela 對等級平坦（最高 46%） — [Maia KDD 2020](https://ar5iv.labs.arxiv.org/html/2006.01855)
- 改變自身等級輸入可改變 Maia-2 的預測著法至多 22% 的局面（對手等級 6%），即約八成局面的預測著法與等級無關 — [Maia-2](https://arxiv.org/abs/2409.20553)
- SE 的強度分數在開局各級相近，故需按手數對齊；只看前 50 手時 kyu 級準確率大降；只看官子時 8d 下降 — [Chen et al. 2025](https://arxiv.org/abs/2502.17109)
- (s,c) 模型使用所有合法手的評估差分佈；satisficing 深度（只由出錯手推得）隨等級穩定上升 — [Lipton–Regan 2015](https://rjlipton.com/2015/10/06/depth-of-satisficing/)
- Finding Elo 第 2 名特徵：「Stockfish 與棋手一致的最深深度」、「最佳手更換次數」、Guid–Bratko 複雜度（加入逐手模型後值 1–2 分／約 160 MAE）；Regan 式序列貝氏模型與既有特徵冗餘 — [blundercheck](https://github.com/dsjoerg/blundercheck)
- 人類偏好把局面導向合理選項較少的局面；每手誤差隨優勢幅度上升且人類更明顯 — [Regan, Biswas, Zhou 2014](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- Neubauer 的 Basic model 特徵含「該局面最大 policy 值」（局面尖銳度代理）與「該手 policy 值」，整體仍劣於 trunk — [Neubauer 2024](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- humanv0 提供每 profile 的 humanWinrate / humanScoreMean / humanStWrError / humanStScoreError，以及對手等級條件 profile rank_{BR}_{WR} — [Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide)
- 2015 年的手工特徵中，局部 pattern 統計 RMSE 2.788 級，其餘（先後手序列 5.765、邊距 5.818、提子 5.904）接近均值基線 7.503 — [Moudřík 2015](https://arxiv.org/abs/1512.08969)
- 身分嵌入的注意力隨手數單調遞減（早期手最具身分資訊），而開局基線本身就能以 0.929 的 P@1 認出棋手 — [arXiv 2208.01366](https://arxiv.org/abs/2208.01366)

### Inferences（候選特徵家族逐一評估；「單調」= 很可能是所下之手似然的單調函數，於本專案冗餘）

1. **跨 profile 的鑑別位置加權／反事實分歧（counterfactual disagreement）**
   - 量什麼：每個局面上 18 個 rank profile 的 argmax 是否一致、各 profile 對所下之手 log p 的離散度（max−min、方差）、以及「只在 profile 分歧的局面上」聚合似然向量；等價於問「弱／強棋手在這裡會下不一樣的棋嗎？」
   - 超越似然的證據：Maia-2 的 22% 局面對等級敏感（其餘約 78% 無資訊）、Kuboki Fig. 2 的等級特異鑑別、Chen 的「開局各級相近」。平均所有手會被無資訊的手稀釋。
   - 效果量：無直接消融（gap）。最接近的數字是 Kuboki 以兩個等級 prior 取代一個：13.8/33.0% → 65.1%（n=20，西洋棋），顯示跨級「形狀」而非「水平」在起作用；Chen 以手數對齊處理同一問題但未單獨消融。
   - 成本：零額外推論（18 個 profile 的 policy 已算好）。
   - 風險：專案的 TCN/MLP 以 whole-question loss 訓練，可能已隱式學到；建議先做遮罩消融（移除 profile 全體一致的局面、或只保留分歧局面）確認是否有增量。**非單調。**

2. **等級條件 policy 的分佈形狀：熵、最大值、top-k 名次、p_played/p_max**
   - 量什麼：同一似然值在「尖銳局面」（只有一手好棋）與「平坦局面」（多手等價）意義不同；熵／最大值衡量局面尖銳度，名次與相對機率衡量「偏離該等級共識的程度」。
   - 證據：僅間接——Regan 的 (s,c) 需全分佈；Neubauer 把 max-policy 當特徵但未單獨消融；Finding Elo 的複雜度特徵值 1–2 分／160；Regan 2014 指出人類偏好選項少的局面。
   - 效果量：文獻無任何數字（gap）。
   - 成本：若已存 full humanPolicy 為零；若只存 log p 需重跑 humanv0（1M 局 × 約 250 手 × 18 profile）。
   - 風險：trunk 向量已含局面資訊，熵／最大值可能冗餘；**p_played/p_max 與 top-k 名次在固定局面下是似然的單調函數**，只有跨局面比較時才非單調 → 標為「半單調、低優先」；熵為「非單調但疑冗餘」。

3. **搜尋深度一致性（satisficing depth 的圍棋版）**
   - 量什麼：所下之手是否等於 KataGo 原始 policy top-1（深度 0）、是否等於 N 次訪問 MCTS 的 top-1（深度 N）、最佳手隨訪問數變更的次數（Guid–Bratko 複雜度類比）、首次「被搜尋判為錯誤」的訪問數。
   - 證據：Regan 的 crossover depth 隨 Elo 穩定上升且只需出錯的手；Finding Elo 第 2 名以「一致的最深深度」與「最佳手變更次數」取得約 160 MAE。
   - 效果量：圍棋無報告；西洋棋中複雜度特徵約值 1–2 Elo 分（約 1%）。
   - 成本：高（每局面需數十至數百次訪問；可只對「原始 policy top-1 ≠ 所下之手」的局面子集執行，約降為三到四成）。
   - 風險：專案的 value/score 失分特徵已判死，而深度一致性與失分高度相關；Finding Elo 也指出 Regan 式模型與既有特徵冗餘。**非單調，但預期增量小。**

4. **對手等級條件似然 rank_{r}_{r'}**
   - 量什麼：以 humanv0 的雙等級 profile 計算 p(move | 自身 r, 對手 r')，對目標棋手與對手的著法同時評估，形成 (r, r') 格點上的聯合似然曲面；取最大者即雙方等級的聯合估計。
   - 證據：KataGo 文件說明該 profile 讓模型「知道對手較強或較弱」；Maia-2 顯示對手等級只改變至多 6% 局面的預測（自身 22%）。
   - 效果量：預期小（約為自身效應的四分之一量級，無直接數字）。
   - 成本：每多一個 (r, r') 組合需一次完整 humanv0 推論；全格點 18×18 不可行，可只取對角附近 3–5 個差距。
   - 風險：等級差 >9 或與讓子不符為分佈外；專案已有對手著法與差分特徵，增量不確定。**非單調。**

5. **讓子／貼目條件化 profile**
   - 量什麼：HD≥2 的局改用 rank_{BR}_{WR}（黑較弱、白較強，差距 = 讓子數）而非單一等級 profile；HD1（無貼目）為訓練分佈的小偏移。
   - 證據：文件指出讓子局白方通常較強、部分伺服器讓子不足，單一 profile 下的勝率／目數會失準；humanv0 可模擬讓子局。
   - 效果量：無報告（gap）。
   - 成本：只需對 HD≥2 的局重跑。
   - 風險：讓子局比例若小則增量有限。**非單調（改變的是條件分佈，不是似然的變換）。**

6. **等級條件 value／不確定度（humanWinrate、humanStWrError、humanStScoreError）**
   - 量什麼：rank-r 棋手「主觀」勝率與 KataGo 客觀勝率之差（感知落差）、短期不確定度（局面對該等級的難度）。
   - 證據：僅文件；專案已判 humanv0 value head 終局代理與 KataGo 失分為死。
   - 效果量：無。成本：與 policy 同次推論。
   - 風險：高度重疊於已死特徵 → 低優先。**非單調但疑冗餘。**

7. **以等級排序損失學到的每手強度分數（SE）** — 證據最強（Kuboki：去掉後 −4.5～−6.1 點），但專案 2026-10-29 已試 SE/ordinal 未通過，**不重提**。僅列出 Chen 實作中可核對的差異點：Plackett–Luce 由強至弱的 sequential softmax、每級 m=7 個 pairs 同批、按手數 d 的級別均值 β̄^d 對齊、幾何平均聚合、r∞ 隨機合法手類別、單一網路同時輸出 policy/value/β（多任務）。

8. **局部 pattern 統計（Moudřík）** — 2015 年最佳手工特徵（2.788 級），但 KataGo trunk 直接看到局部形狀；Neubauer 顯示 trunk 優於一切標量；**判為冗餘**。先後手序列、邊距、提子統計與專案已死的 response geometry 同類（RMSE 5.7–5.9 vs 均值 7.5），**判死**。

9. **失分分位數（最差 25% 手、中位數、前 N_cut 手）** — Kuboki 的 MedLoss 幾乎無增量；Finding Elo 用 q_error_one；專案的 value/score 失分、blunder clustering 已死。**非似然但判死。**

10. **手數對齊（per-move-index normalisation）** — 不是新特徵而是前處理：對每個 profile 的 log p 以「手數 d 的訓練集等級均值／標準差」標準化（Chen 的 β̄^d 作法）。效果量：Chen 未單獨消融，但其 fuseki-only 實驗顯示開局手對 kyu 級幾乎無鑑別力。成本：零。風險：TCN 若已輸入手數位置則冗餘。

### Gaps
- 沒有任何已發表工作對「policy 熵／top-k 名次／相對最佳手機率」作為棋力特徵做過消融（Go 或 chess 皆無）。
- 沒有工作比較 rank_{BR}_{WR} 聯合 profile 與單一 profile 在等級推估上的差異。
- Regan–Biswas ICMLA 2015 全文（深度與 Elo 的數值表）未取得，僅有部落格摘要。
- 沒有圍棋文獻把「MCTS 訪問數下的最佳手變更」當作特徵。

## KQ3 對手聯合建模、對稱增強、更長上下文、以玩家身分為目標的訓練

### Takeaway
對手聯合建模的直接證據只有 Maia-2 的「對手等級改變至多 6% 局面的預測」（自身 22%）與 Omori 以一個 BiLSTM 同時輸出雙方 rating（無消融）；對稱增強只有 KataGo 文件的降噪建議（無消融）；更長上下文在「同一棋手」抽樣下收益極小（圍棋 5→15 局僅 +5.1 點），在「同級不同棋手」抽樣下收益極大（+19 點）；玩家身分目標（GE2E）能從 100 局在 2,844 人中以 86% 認出棋手，但開局佔了大部分身分訊號，且無人證明其對等級預測有增量。

### Cited Findings
- Maia-2：改變自身等級可改變預測著法至多 22% 的局面，改變對手等級至多 6%；訓練時對「雙方不同等級」的對局過抽樣 — [Maia-2](https://arxiv.org/abs/2409.20553)
- Omori–Tadepalli：單一模型以雙方著法與時鐘同時輸出雙方 rating（逐手），無單方／雙方消融 — [arXiv 2409.11506](https://arxiv.org/abs/2409.11506)
- Kuboki 與 Chen 均只用「雙方同級」的對局（對手著法與目標同標籤）；Neubauer 指出 OGS 以 Glicko-2 配對，資料帶有「對手等級≈自身等級」的配對偏差 — [Kuboki](https://arxiv.org/html/2505.00279v1)；[Neubauer Sec. 6.3](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- KataGo 文件：rootNumSymmetriesToSample 設 2 或 8 可得「較不嘈雜的 policy」；humanv0 提供 rank_{BR}_{WR} 對手條件 profile — [Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide)
- Kuboki Table 4（圍棋）：隨機抽樣 5/10/15 局 70.4/82.8/89.4%，同一棋手抽樣 58.1/61.0/63.2%（±1 95.3/96.2/96.6）；Table 5（西洋棋）同一棋手 52.2/57.0/60.4 — [Kuboki](https://arxiv.org/html/2505.00279v1)
- Kuboki 以兩個高度相關的 prior（Maia-1100 + 1200）取代一個：28.1/24.3 → 35%（n=20），論文歸因於「每手 prior 波動大，合併相關等級仍能降低變異」 — 同上
- Chen：SE 15 局 >80%，100 局 97.5%（query set 以同級不同棋手隨機抽樣） — [Chen](https://arxiv.org/abs/2502.17109)
- Neubauer：固定 N=500 手（約 4 局）後即與 Glicko-2 持平；初步搜尋後不再把 N 當超參數 — [Neubauer Sec. 5.2, 6](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- 行為風格計量（McIlroy-Young et al., NeurIPS 2021）：Maia 式 6 殘差塊 × 64 通道 → 全局平均池化 → transformer（32 手序列）→ 1024 → 512 維局向量；GE2E 損失（每批 N=40 人 × M=20 局）；訓練 63.7M 局／16,181 人；資料為 Lichess Blitz、平均 rating 1000–2000、≥1,000 局的棋手 — [arXiv 2208.01366](https://arxiv.org/abs/2208.01366)
- 同上 Table 2：k=15（略過前 15 手）、100 參考局 + 100 查詢局，在 2,844 候選中 P@1 0.860（未見棋手）、0.853（已見）；k=0 時 0.979，但純開局（前 5 手 5-hot）基線亦達 0.929；Table 3：vs 個人化 Maia（400 人池）0.953 vs 0.552；高段棋手 408/2,157 僅 0.308；41,184 人全池 0.54；1900 附近窄等級帶 92.6%；性能約在 50 局飽和，10 局「尚可」；注意力隨手數單調遞減 — 同上
- 個人化 Maia（KDD 2022）：對單一棋手微調可「顯著提升」預測並可用於 stylometry；摘要未給數字 — [arXiv 2008.10086](https://arxiv.org/abs/2008.10086)
- Chen 的 r∞（隨機合法手）類別對等級推估「幾乎無差」，只對強度調整有用 — [Chen](https://arxiv.org/abs/2502.17109)

### Inferences
- 「同一棋手抽樣」下局數收益小（5→10 局 +2.9 點、10→15 局 +2.2 點）說明瓶頸是棋手層級的變異與標籤噪音，而非每局訊號量；本專案的 5/10/15/20 局設定應預期類似飽和，且 20 局相對 15 局的增量恐 <2 點。
- 對稱增強在推論端（8 對稱平均 humanPolicy）等同於降低每手 log p 的噪音；Kuboki 以兩個高度相關 prior 合併仍能提升（28.1/24.3 → 35）支持「降噪本身有價值」，但代價是 ×2～×8 推論。訓練端對稱增強在 Chen／Kuboki／Neubauer 均未報告。
- 身分目標（GE2E）在 8,007 人 × 平均約 125 局的資料上可行，且比賽問題「5–20 局同一人」正是 few-shot 設定；但 (a) 身分訊號大半來自開局（0.929 基線），而專案的開局一致性特徵已判死；(b) 無任何論文證明身分嵌入對等級預測有增量；(c) 高段棋手識別率只有 0.308，顯示嵌入對強棋手區分力弱。可視為中成本的探索方向，不是文獻支持的主線。
- 對手聯合建模最便宜的實作是 humanv0 的 rank_{r}_{r'}，但 Maia-2 的 6% 暗示效應量小；在無對手標籤下，可用「對手著法在各 rank_r 下的似然」當作對手等級的弱代理（專案已有對手特徵），再與 rank_{r}_{r'} 比較。
- 專案資料中「對手不在資料集」但 Kuboki／Chen／Neubauer 都顯示線上配對使對手等級接近自身，故對手著法的等級似然本質上是目標標籤的噪音副本；這解釋了專案「對手著法特徵有效」的既有結果，也限制了再從對手端挖訊號的空間。

### Gaps
- 無任何文獻報告「單方 vs 雙方著法」的等級推估消融。
- 無任何文獻報告對稱增強（推論平均或訓練增強）對等級推估的消融。
- 無文獻報告以身分嵌入作為等級預測輸入的增量；stylometry Figure 3a 在 10/20/50 局的精確 P@1 僅見於圖。
- 個人化 Maia（KDD 2022）的量化增益未取得。

## KQ4 依局數的準確率／MAE 校準

### Takeaway
最可比的設定（圍棋 11 級、同一棋手抽樣）在 10 局為 61.0% exact／96.2% ±1，15 局 63.2%／96.6%（Kuboki 2025）；專案目前 whole-question CV 0.726（exact 1／鄰級 e^-1）若拆成 exact ≈ 0.60、鄰級 ≈ 0.35，即 0.60 + 0.35 × 0.368 ≈ 0.73，與該前沿相當。隨機抽樣（同級不同棋手）的 80–92% 不應作為本專案的目標。

### Cited Findings
- 圍棋 11 級（Fox，9d–3-5k），同一棋手抽樣：n=5 58.1 (±1 95.3)、n=10 61.0 (96.2)、n=15 63.2 (96.6)；Chen 重現 51.6 (93.1)、59.4 (96.0)、61.1 (96.9) — [Kuboki Table 4](https://arxiv.org/html/2505.00279v1)
- 圍棋 11 級，隨機抽樣：n=1 37.5 (82.8)；n=5 64.8 (98.8)；n=10 79.6 (99.9)；n=15 87.5 (100)；n=20 91.7 (100)；Chen 重現 33.7/56.3/71.4/78.2/83.8 — [Kuboki Table 1](https://arxiv.org/html/2505.00279v1)
- Chen 原文：SE 15 局 >80%、100 局 97.5%；±1 容忍 1 局近 80%、6 局 >99%；SL 分類器 100 局 49%；每局 1 手 × 20 局約 60% — [Chen](https://arxiv.org/abs/2502.17109)
- 西洋棋 8 級（200 分段）：隨機 n=10 74.3 (99.7)、n=20 86.2 (100)；同一棋手 n=5 52.2 (94.5)、n=10 57.0 (97.2)、n=15 60.4 (97.7) — [Kuboki Tables 1, 5](https://arxiv.org/html/2505.00279v1)
- 單局 Elo 迴歸：Omori MAE 182（有時鐘）／239（無時鐘）／346（均值），rating SD 366 — [arXiv 2409.11506](https://arxiv.org/abs/2409.11506)；Finding Elo 第 2 名 MAE ≈160 — [blundercheck](https://github.com/dsjoerg/blundercheck)
- 10–50 局、26 級（KGS）：RMSE 2.66 級 — [Moudřík 2015](https://arxiv.org/abs/1512.08969)；3 類：單局面 71.5%、單局 77.9% — [Moudřík & Neruda 2016](https://link.springer.com/10.1007/978-3-319-49001-4_15)
- 約 4 局（500 手）即與 Glicko-2 的勝負預測持平（test λ −0.58 vs −0.58；α 68.8% vs 69.1%） — [Neubauer Table 5.2](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- 身分識別：100 局在 2,844 人中 P@1 0.86；41,184 人 0.54；約 50 局飽和 — [arXiv 2208.01366](https://arxiv.org/abs/2208.01366)

### Inferences
- 專案的「10 局精確等級 55–65%」與 Kuboki 同一棋手設定的 59.4–61.0% 一致；文獻中沒有在同一棋手設定下超過約 63%（15 局）的圍棋結果，故 exact 的可達上限可能在 65–70%，而非隨機抽樣的 90%。
- ±1 級準確率在 5 局即達 95% 以上且幾乎飽和，表示剩餘誤差幾乎全是鄰級混淆；在 exact-1／鄰級 e^-1 的計分下，每把 1% 的鄰級轉為精確只增加 0.0063，與專案近期 +0.005～+0.018 的增量尺度相符。
- AI CUP 的 kyu 段為 3 kyu 一帶（D/C/B/A），Chen 指出 kyu 等級變動快、開局鑑別力弱、Fox 從 3d 起算且 3d 準確率偏低；可預期 D–A 與 1D–3D 的鄰級混淆是主要損失來源。

### Gaps
- 沒有文獻在「同一棋手、20 局」的圍棋設定上給數字（Kuboki player-specific 只到 15 局）。
- 沒有文獻使用與 AI CUP 相同的 10 類（3 kyu 一帶 + 1D–6D）切分，等級邊界不可直接比對。
- 沒有文獻報告「無時鐘、無結果、無日期」的西洋棋 Elo 推估基準可供直接比對（Omori 的 NoClock 239 是最接近的）。
