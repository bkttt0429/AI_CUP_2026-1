# 西洋棋 Elo 預測競賽與論文中的實用特徵工程 → 可移植到圍棋等級預測的特徵家族

研究日期:2026-11-04。範圍:從「幾局棋的著手」預測棋手 Elo 的西洋棋競賽/論文,找出**不是**「等級條件策略下著手機率的單調變換」的特徵家族,並逐一翻譯成圍棋(AI CUP 2026:每題 5–20 局匿名網路棋譜、雙方著手、目標顏色、讓子標記;無時鐘、無勝負、無日期;10 個等級類別)。

專案已用:KataGo humanv0 等級條件策略(28 個 profile)之逐手 log p(本方與對手)、微調等級頭、KataGo trunk 向量 + 序列模型、讓子與對手特徵。已證無效:逐手品質時間序列統計(自相關、軌跡形狀、失誤叢聚)、應手幾何、風格一致性、引擎價值式著手損失、attention pooling、校準。

---

## KQ1:Kaggle「Finding Elo」(2014–2015)頂尖解法用了哪些特徵?哪些最重要?有沒有非引擎評估的特徵?

### Takeaway
找到的唯一第一手頂尖寫法是第 2 名 dsjoerg(最終 MAE ≈160):特徵幾乎全來自 Stockfish 輸出(評估差、最佳著變動次數、引擎在多深時同意棋手著手、Guid–Bratko 複雜度、第 25 百分位最差著手的 centipawn loss),再加「材料/非材料 equity」與「以訓練集 Elo 建立的 24 步開局書(匹配局的 Elo 平均/最小/最大/標準差)」。非引擎特徵中,開局書 Elo 統計與材料特徵確有貢獻,但「時間控制(blitz/rapid)」才是最大缺失變數(估值 ≈5 分,但競賽不可用)。第 1 名寫法與官方 benchmark 數字未能找到(Kaggle 頁面為 JS 動態載入,Wayback 被封鎖)。

### Cited Findings
- 競賽資料:25,000 局訓練(含雙方 Elo)+ 25,000 局測試(Elo 隱藏),PGN 格式;主辦方已用 Stockfish 對每手位置給 centipawn 評估(正值利白、負值利黑)— [jwnorman/Chess_ELO README](https://github.com/jwnorman/Chess_ELO)
- 競賽於 2015-03-24 結束;dsjoerg 得第 2 名,「final MAE of ~160」— [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- dsjoerg 的 Stockfish 特徵:「the number of times the best move changed, and the deepest depth at which Stockfish agreed with the player's move」;Stockfish 以固定時間(最終 16 秒/手)而非固定深度跑;每次把時間加倍(2→4→8→16 s)只各得約 0.3、0.4、0.4 分,「made surprisingly little difference」— [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- dsjoerg 的「單手 Elo 模型」(RandomForestRegressor 從**一手**預測 Elo)「worth a point or two, but only once I had the Guid-Bratko complexity measure as one of its features」— [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- dsjoerg 的誤差形狀特徵 `q_error_one`:「the centipawn loss of the player's 25% worst move」(對截斷後的逐手分數取百分位)— [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- dsjoerg 的非引擎特徵:材料與非材料 equity 彙總「worth a point」;以 Elo 建的開局書至 24 ply、匹配局少於 10 局即停止,輸出匹配局 Elo 的 average/min/max/stdev;知道 blitz 或 rapid「worth ~5 score points」但競賽資料無此欄位 — [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- dsjoerg 無效嘗試:Sequential Bayesian skill model 的特徵「were predictive in isolation」但疊在既有特徵上增益太小;多加 50k 外部局「actually made the model worse」,重新加權到 Kaggle Elo 分布後約 +0.5 分,不值得;把線性模型預測值餵進 RandomForest 的堆疊反而重要(移除損失約 3–4 分)— [dsjoerg/blundercheck README](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- 論壇參賽者描述兩組特徵:Stockfish 評估差(mean absolute value、standard deviation)與「positional facts」(剩餘子力、皇后首次移動時機);並稱 Stockfish 檔案含大部分訊號:「sudden change in score -> blunder, smooth score curve -> good players」— [Kaggle Finding Elo discussion 13008(搜尋摘要)](https://www.kaggle.com/c/finding-elo/discussion/13008)
- rozim 的 pipeline 欄位名顯示典型特徵集:`raw_score_mean/median/stddev`、`delta_avg/median/stddev/max`、以 `_op/_mg/_eg` 分階段的變體、`first_loss_100/200/300`(首次損失 ≥100/200/300 cp 的時點)、`pct_best/pct_best2/pct_best3`(著手落在引擎前 1/2/3 的比例)、`alt_raw_0–4`(替代著手評估)、`complexity`、`game_ply`、`result`、`i_was_mated`;模型為 Vowpal Wabbit 回歸(按顏色×結果分組)與 500/5000 樹 RandomForest — [rozim/KaggleFindingElo PIPELINE.md](https://github.com/rozim/KaggleFindingElo/blob/master/PIPELINE.md)
- Guid–Bratko 複雜度(二手描述):引擎隨深度迭代、每次根節點最佳著改變時,累加「最佳著與次佳著分差」,總和越大越難 — [talkchess 討論](https://www.talkchess.com/forum/viewtopic.php?p=601677);Biswas & Regan (2015) 另定義跨深度的複雜度並討論加權深度 — [Biswas & Regan 2015](https://www.scitepress.org/papers/2015/52883/52883.pdf);另有以「達固定深度所需節點數」與「最佳/次佳差」為複雜度代理的實證研究 — [arXiv 2201.10808](https://arxiv.org/pdf/2201.10808)

### Inferences
- Finding Elo 的核心訊號分三類:(a) 評估曲線的平滑度/跳變(=失誤幅度分布),(b) 著手與引擎前 k 著的一致性及「在多深才一致」,(c) 位置難度(Guid–Bratko)作為單手模型的正規化子。這三類與「等級條件策略 log p」並不等價:(b) 是對強引擎的 top-k 一致性,(c) 是位置難度,(a) 是分布形狀而非平均。
- 真正的非引擎特徵中,「開局書 Elo 統計」本質是**以訓練集標籤做的 kNN 經驗先驗**(同位置在訓練集中出現於哪些等級),這是策略模型(在其他資料上訓練)不具備的資訊來源。
- 時間控制/時鐘是西洋棋最強的非著手特徵(見 KQ2 Omori:去掉時鐘 MAE +24%),本專案無時鐘,應把期望值校準在「無時鐘」文獻數字上。

### Gaps
- 第 1 名解法與 benchmark MAE:Kaggle 頁面只回傳標題,web.archive.org 無法存取;未找到 1 名寫法或 Elliot Noma / Nicolas Ramírez 的文章。
- dsjoerg 的「單手 Elo 模型」如何彙總到整局(平均?分位數?)未在 README 寫明。
- Finding Elo 的特徵重要度數值(哪個特徵貢獻多少 MAE)只找到 dsjoerg 的零散「worth ~1 / ~5 points」敘述。

---

## KQ2:後續工作 — Maia/Maia-2、Omori & Tadepalli、Lichess accuracy、chess.com 估計、失誤率對難度正規化、損失分布形狀、與引擎 top-k 一致性、脫離開局理論的步數

### Takeaway
2020–2026 的文獻裡,**沒有**一篇把 Maia/Maia-2/ChessMimic 直接拿來做等級估計並報告精度(它們是等級條件著手預測器);等級估計的 SOTA 方向是 Chen et al. (ICLR 2025) 的 Bradley–Terry 強度分數與 Kuboki et al. (2505.00279) 的多等級策略幾何平均。可轉移的「非策略單調變換」特徵的最強證據來自:Anderson et al. (KDD 2016) 的「位置難度(blunder potential)解釋失誤率遠勝等級(0.73 vs 0.55)」、Maia 的「最佳/次佳差越大、預測越準」、Maia-2 的「約 22–28% 位置才是等級單調/過渡位置」、Regan–Haworth 的雙參數 (s 敏感度、c 一致性) 誤差模型、Lichess 的「調和平均 + 波動加權」聚合,以及 Tijhuis et al. (CoG 2023) 的「理論相關 + 引擎相關特徵最重要」。

### Cited Findings
**等級條件模型與等級估計**
- Maia (KDD 2020):Maia-1900 預測 1100 棋手著手的準確率 46%(最低),最高逾 52%;各模型峰值:Maia 1100 在 1200 區 50.8%、1300→1400 51.8%、1500→1700 52.2%、1700→1800 52.7%、1900→1900 52.9%;論文**未**宣稱可用於估計棋手等級,只說「each model best captures how players at a specific skill level play」— [Maia KDD 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.01855)
- Maia 失誤預測(失誤 = 勝率下降 ≥10 個百分點):RF 56.4%(僅棋盤)/63.0%(加等級、剩餘時間比、cp);全連接 62.9/66.0;殘差 CNN 67.7/71.7(平衡測試集);「集體失誤」(同位置 >10% 棋手失誤)預測達 75.3–76.9%;Maia 預測準確率隨「最佳著與次佳著品質差」增大而上升(Fig. 7),前 10 ply 被捨棄以避開背譜 — [Maia KDD 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.01855)
- Maia-2 (NeurIPS 2024):在 100K 位置測試集,Maia-2 對 27.61%/28.51%/26.38%(Skilled/Advanced/Master)位置呈「隨等級單調」,Maia-1 僅 1.61/1.42/1.14%;「過渡位置」約 22.59/23.39/21.72%;改變己方等級輸入可改變決策「up to 22% of the time」,改變對手等級「up to 6%」;mate-in-1 案例中最佳著機率隨等級 1100→1900 由 0.22 升到 0.45;線性探針:Stockfish 整體評估等「等級相依概念」隨等級變化,「雙象」等概念不隨等級變;**未**做等級估計實驗 — [Maia-2 arXiv 2409.20553v2](https://arxiv.org/html/2409.20553v2)
- ChessMimic (2026-06):14 個 Elo 區段各訓一組 9M 參數 encoder transformer(著手/時鐘/勝負三模型);top-1 著手預測 0.5623 vs Maia-2 0.5260(n=4,775,033),各區段皆勝(+3.0 至 +6.1 pp);**未**用跨區段似然做等級估計,僅引用 Omori & Tadepalli 的 ~182 MAE;提出「per-position softmax entropy」可作服務端信心分數;Lichess 謎題位置的正規化熵在 12/14 區段低於同區段 blitz 位置 — [ChessMimic arXiv 2606.04473](https://arxiv.org/html/2606.04473)
- Omori & Tadepalli (CG 2024):CNN(4 層)+ 雙向 LSTM,輸入 12 棋盤平面 + 每手剩餘時鐘,逐手輸出雙方等級;~1.2M Lichess 局;MAE 182(有時鐘)/239(無時鐘)/346(平均值基線);去掉時鐘 MAE +57(≈24%),Bullet +92(≈34%);Classical 最佳 151;無任何手工特徵;高分段失效(2922/3163 的局預測約 1460);未報告 MAE 隨已見手數的曲線 — [arXiv 2409.11506v1](https://arxiv.org/html/2409.11506v1)
- Tijhuis, Mavromoustakos Blom & Spronck (IEEE CoG 2023)「Predicting Chess Player Rating Based on a Single Game」:從單局抽 30 個特徵預測等級區段;極端區段(最低 vs 最高)79.3% 準確率,區段變寬(類別變多)則下降;「the most important features are theory-related and engine-related」— [Tilburg University 頁](https://research.tilburguniversity.edu/en/publications/predicting-chess-player-rating-based-on-a-single-game);Omori 引述其為「30 hand-crafted features … players binned by rating into 10 different splits」— [arXiv 2409.11506v1](https://arxiv.org/html/2409.11506v1)
- 「Rating Your Moves」(Jim Yang, 2025 Chessable Research Award):逐手更新對棋手等級的信念,用 5+0 blitz、400–2400 Elo;無已發表精度 — [chess.com/Chessable blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)
- 2026 Elo-disentangled style embeddings 預印本:線性探針只能從風格向量 z 回復等級 R²=0.06,顯示風格與強度可分離 — [emergentmind 2606.25176](https://www.emergentmind.com/papers/2606.25176)

**平台公式**
- Lichess:Win% = 50 + 50·(2/(1+exp(−0.00368208·cp)) − 1);Accuracy% = 103.1668·exp(−0.04354·(Win%before − Win%after)) − 3.1669;整局 accuracy = 「滑動視窗內 Win% 標準差為權重的加權平均」與「逐手 accuracy 的調和平均」兩者再取平均;cp→Win% 以 2300 分棋手的對局校準;明言 accuracy 與等級「有些相關但不直接」:高分者常選理論複雜開局、雙方失誤更多;低分者不願認輸,長的一面倒殘局會拉高 accuracy — [Lichess accuracy 頁](https://lichess.org/page/accuracy)
- chess.com Game Review「game rating」:官方只說是把著手品質與「你的等級段的典型表現」比對的單局表現估計,公式未公開;社群回報估值同時受雙方等級影響 — [chess.com 支援文](https://support.chess.com/en/articles/10773754-how-is-game-rating-calculated-in-game-review);[論壇](https://www.chess.com/forum/view/general/how-does-chess-com-calculate-your-rating-in-game-review)
- Lichess Kaladin(作弊偵測):以 Keras/TensorFlow CNN 吃「player insights」(含平均 centipawn loss、手數等)— [lichess-org/kaladin](https://github.com/lichess-org/kaladin);[arXiv 2409.11506](https://arxiv.org/pdf/2409.11506)

**失誤率 vs 位置難度(正規化)**
- Anderson, Kleinberg & Mullainathan (KDD 2016 / TKDD 2017):定義 blunder potential β(P)=b(P)/n(P)(合法著中失誤著比例 = 「隨機下子會失誤的機率」);平衡資料集上預測失誤:β 單獨 0.73、難度特徵合計 0.75、技能(己方/對手 Elo)0.55、剩餘時間 0.53;難度特徵之上再加技能或時間**無增益**(仍 0.75);固定位置時等級單獨 0.62;各 β 下的等級曲線在對數尺度「almost completely parallel」;β 每增 0.2 對失誤率的影響大於 600 Elo 差;存在「skill-anomalous」位置(等級越高越易失誤);quantal-response 曲線 γ_c(β)=β/(c−(c−1)β),FICS c≈15、GM c≈100 — [arXiv 1606.04956 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1606.04956)
- Regan & Haworth「Intrinsic Chess Ratings」(AAAI 2011):以高深度引擎評估建立雙參數模型,s=sensitivity(s 小會放大小的評估差),c=consistency(c 大會急遽壓低壞著機率);對每組 (s,c) 定義期望平均誤差並線性映射到 Elo;發現與 Elo 百分段的平滑對應且「little or no rating inflation」— [AAAI 2011](https://ojs.aaai.org/index.php/AAAI/article/view/7951);Regan 課程投影片指出不直接用棋手自身對局的實際 ASD,因「raw averages can be skewed by unusually easy or unusually hard games」,改用模型投影值並附標準差 — [CSE702 Week 9 IPRs](https://cse.buffalo.edu/~regan/ChessSeminar/CSE702Week9IPRs.pdf)(搜尋摘要)
- 「Chess Signatures of Play」(Turk & Polson, 2026):每手通道 c_t=引擎 top-k 線評估的離散度(複雜度)、q_t=−centipawn loss、m_t=累計與引擎一致次數;以 path signature(深度 2,21 座標,其中 6 個 Lévy area 捕捉「難度與準確度的 lead-lag」)做作弊監測;僅合成實驗;估計期望簽名的收斂率 N^(−1/2),建議「few dozen to a few hundred games」— [arXiv 2606.18544](https://arxiv.org/html/2606.18544)

**損失分布形狀(重尾)**
- 「ACPL 50 可能是幾乎每手小不精確,也可能是整局很好卻一次大失誤」— [chessenginelab Substack: Centipawn Loss Distribution](https://chessenginelab.substack.com/p/centipawn-loss-distribution);Lichess 論壇引述 Tord Romstad:「playing strength depends much more on your weakest moves than on your average moves」;Lichess 在評估超過門檻後不再累加損失,導致對手早早送后後 ACPL 失真;有人建議看逐手損失分布而非單一數字 — [Lichess 論壇](https://lichess.org/forum/general-chess-discussion/very-low-centipawn-loss?page=3)
- Kuboki et al. 附錄 F:相鄰等級組之間,損失(score loss)分布的重疊程度高於強度分數的重疊 — [arXiv 2505.00279v1](https://arxiv.org/html/2505.00279v1)

**圍棋原生的非策略特徵(先行研究)**
- Moudřík, Baudiš & Neruda (2015):KGS 26 級(6d–20k),每級 120 位棋手、每人 10–50 局,袋裝 NN 回歸;RMSE(級):平均值基線 7.503;**Patterns(1000 個最常見局部樣式計數)2.788**;ω-local sente/gote 序列 5.765;邊線距離×手數區段直方圖 5.818;提子直方圖 5.904;勝負統計 6.792;勝負點數 5.116;全部 2.659;「the Pattern feature works by far the best」— [arXiv 1512.08969 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1512.08969)
- Chen et al. (ICLR 2025, Fox 495,000 局、11 級):只用前 50 手會降低準確率、尤以級位為甚,「opening moves look similar across ranks」;只用最後 50 手整體也降,但 9 段幾乎不降、8 段下降明顯(作者歸因於殘局能力相近)— [arXiv 2502.17109v2](https://arxiv.org/html/2502.17109v2)

### Inferences
- 文獻裡「位置難度」有三種可操作定義:(i) 合法著中壞著的比例(Anderson β);(ii) 最佳/次佳差或 top-k 評估離散度(Maia Fig. 7、Guid–Bratko、Turk & Polson);(iii) 策略熵(ChessMimic)。三者在圍棋都可由 KataGo 全策略分布算出,(i)(ii) 若要用「分數損失」版本則需子節點價值(昂貵),用「策略機率」代理版本只需一次前向。
- Lichess 的「調和平均 + 波動加權」與 dsjoerg 的「第 25 百分位最差著」、Regan 的 c、Romstad 的「最弱著決定強度」是同一個主張:**聚合算子應強調最差著而非平均**;幾何平均(= log p 算術平均)是 Kuboki/Chen 的預設,但不是唯一選擇。
- Maia-2 的 22–28% 與 Anderson 的「難度主導」共同暗示:多數位置不帶等級資訊,等級訊號集中在少數「過渡位置」;逐手等權的幾何平均會被大量無資訊位置稀釋,但在 n 局聚合後稀釋效果可能已被平均掉(Kuboki 等權仍達 20 局 91.7%)。

### Gaps
- Regan–Haworth 的精確機率公式與「幾手可達 ±X Elo」的標準差表:PDF 無法在本機抽文字(無 pdftotext/pypdf),Reading eprint 被 Anubis 阻擋;只取得參數的定性意義。
- Tijhuis et al. 的 30 個特徵完整清單與 10 類精度:PDF 回 403;只有摘要層級的敘述。
- Maia/Maia-2/ChessMimic 皆無「跨等級似然→等級估計」的精度數字;chess.com 公式未公開。
- 「脫離開局理論的手數」作為單獨特徵的量化效果:只找到 Tijhuis「理論相關最重要」與 dsjoerg 的 Elo 開局書敘述,無逐特徵數字。

---

## KQ3:哪些特徵家族不是策略似然的單調變換、可移植到圍棋?(逐一給圍棋對應、計算成本、證據、轉移價值、風險)

### Takeaway
六個家族中,「(2) 等級條件策略間的反事實分歧」與「(3) 誤差幅度分布形狀」可**純 CPU 從既有 28 個 profile 的逐手 log p** 算出,成本最低;「(1) 策略分布形狀/難度正規化」與「(5) 局部戰鬥 vs 全局錯誤拆分(策略版)」需重新評分取得全分布(~250k 局一次前向,無需搜尋);「(4) 訓練集位置頻率/定式書的經驗等級先驗」只需 SGF 與雜湊,是唯一引入策略模型所沒有之資訊(訓練標籤)的家族;「(6) 分階段表現」可由既有 log p 直接分段。文獻對「疊在強策略似然上還有多少增益」的直接證據是 Kuboki 的消融:在圍棋,加入引擎損失特徵的增益為 0(91.7 vs 91.9),加入 Chen 強度分數 +3–6 pp(n=1–10)— 所以期望值應設在「每個新家族 0–3 pp,且彼此高度重疊」。

### Cited Findings(支撐各家族的證據,含成本標記)

| # | 特徵家族(西洋棋) | 西洋棋證據 | 圍棋對應(可由著手+顏色+讓子+KataGo 算) | 成本 |
|---|---|---|---|---|
| 1 | 位置難度正規化:策略熵、近似最優著數、blunder potential、最佳/次佳差 | Anderson:β 單獨 0.73 vs 等級 0.55,加等級無增益 — [1606.04956](https://ar5iv.labs.arxiv.org/html/1606.04956);Maia 準確率隨最佳/次佳差上升 — [2006.01855](https://ar5iv.labs.arxiv.org/html/2006.01855);dsjoerg 單手模型僅在加入 Guid–Bratko 複雜度後才有用 — [blundercheck](https://github.com/dsjoerg/blundercheck/blob/master/README.md);ChessMimic 建議 per-position 熵作信心 — [2606.04473](https://arxiv.org/html/2606.04473) | 每手:強策略(或目標等級 profile)的熵 H、`#{a: p(a) ≥ τ·max p}`、top-k 質量;以之加權或分桶 log p(例如只在「低熵/一邊倒」位置計算失誤率 = 「簡單題做錯」比例,類 Anderson 的固定難度下等級 0.62) | 需全分布 → 重新評分 ~250k 局(每位置一次前向,無搜尋);若只存了 28 個 log p 則**不可算** |
| 2 | 等級條件策略間的反事實分歧(「1500 與 2500 在此會走不同」的位置子集) | Maia-2:僅 ~27% 位置單調、~22% 過渡;改己方等級輸入改變決策至多 22% — [2409.20553](https://arxiv.org/html/2409.20553v2);Chen:開局手「look similar across ranks」、前 50 手資訊少 — [2502.17109](https://arxiv.org/html/2502.17109v2);Kuboki 多等級幾何平均是核心特徵且無任何位置加權 — [2505.00279](https://arxiv.org/html/2505.00279v1) | 每手由 28 個 profile 的 log p(played) 算跨 profile 離散度 D_t(range/std/「argmax profile」);特徵:高 D_t 手的比例、僅在 D_t>τ 子集上的各 profile 幾何平均、D_t 加權似然、argmax-profile 直方圖;若有全分布則可改用 profile 間 JS 散度(與所下之著無關的「位置可鑑別度」) | 離散度版:**純 CPU、現有儲存即可**;JS 版需全分布 |
| 3 | 誤差幅度分布形狀(尾指數、大錯比例、分位數)而非平均 | dsjoerg `q_error_one` = 第 25 百分位最差著 — [blundercheck](https://github.com/dsjoerg/blundercheck/blob/master/README.md);Regan 的 c(一致性)獨立於 s — [AAAI 2011](https://ojs.aaai.org/index.php/AAAI/article/view/7951);Lichess 以調和平均 + 波動加權取代算術平均 — [Lichess](https://lichess.org/page/accuracy);Romstad「最弱著決定強度」— [論壇](https://lichess.org/forum/general-chess-discussion/very-low-centipawn-loss?page=3);rozim 的 `first_loss_100/200/300` — [PIPELINE.md](https://github.com/rozim/KaggleFindingElo/blob/master/PIPELINE.md) | 對每個 profile j 的 −log p_j 序列取分位數(q10/q25/q50/q90)、超過門檻(如 −log p>5)的比例、調和平均版 exp(log p)、首次「大驚訝」出現手數;以及 Regan 式兩參數擬合(s,c)對「本方著手在強策略下的機率排名」 | **純 CPU、現有儲存即可** |
| 4 | 脫離開局理論/定式的手數;開局書 Elo 統計 | Tijhuis:理論相關特徵最重要 — [Tilburg](https://research.tilburguniversity.edu/en/publications/predicting-chess-player-rating-based-on-a-single-game);dsjoerg 24-ply Elo 開局書(avg/min/max/std;<10 局即停)— [blundercheck](https://github.com/dsjoerg/blundercheck/blob/master/README.md);Moudřík 局部樣式計數是圍棋最強單一家族(RMSE 2.79 vs 基線 7.50)— [1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969) | (a) 以訓練集建全盤位置雜湊(8 對稱正規化)之「等級分布書」至 N 手,輸出首次 <k 局匹配的手數 + 匹配局等級均值/標準差/熵(需 out-of-fold 建書);(b) 定式層級:每個角的局部樣式序列與訓練集樣式→等級表比對;(c) 讓子局用相同機制但分開建書 | **純 CPU、無需 KataGo**;需 SGF 重放與雜湊 |
| 5 | 戰術 vs 位置性錯誤拆分 | 西洋棋以引擎戰術性定義;圍棋對應證據:Moudřík ω-local sente/gote 序列單獨 RMSE 5.77(弱但非零)— [1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969) | 以座標幾何(gridcular 距離 ω 內 = 局部接觸戰;否則 tenuki)把每手分為「局部/全局」,分別計算 log p 統計;策略版:以強策略 top-1 是否在局部來定義「該不該脫先」,統計「該局部卻脫先」「該脫先卻局部」比例 | 幾何版純 CPU;策略版需全分布(或至少 top-1 座標) |
| 6 | 分階段表現(開/中/殘) | rozim `_op/_mg/_eg` 變體 — [PIPELINE.md](https://github.com/rozim/KaggleFindingElo/blob/master/PIPELINE.md);Chen:前 50 手級位資訊少、末 50 手 9 段仍準 — [2502.17109](https://arxiv.org/html/2502.17109v2);Kuboki:損失隨手數呈山形、峰約第 100 手,故用 N_cut — [2505.00279](https://arxiv.org/html/2505.00279v1);Lichess:殘局/不認輸影響 accuracy — [Lichess](https://lichess.org/page/accuracy) | 按手數區段(≤50 / 51–150 / >150)或盤面填滿比例,對各 profile 分別取幾何平均與分位數;另加「終局方式」(雙 pass vs 中途結束)與局長 | **純 CPU、現有儲存即可** |

其他低成本圍棋原生家族(非策略):
- Moudřík 的邊線距離×手數直方圖(RMSE 5.82)、提子直方圖(5.90)— [1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969);純 CPU。
- 時鐘不可得,但文獻顯示其價值極大(Omori 去時鐘 MAE +24%;dsjoerg blitz/rapid ≈5 分)— [2409.11506](https://arxiv.org/html/2409.11506v1);[blundercheck](https://github.com/dsjoerg/blundercheck/blob/master/README.md)。

### Inferences(轉移價值與風險,逐家族)
- **(1) 難度正規化**:價值中等。Anderson 顯示難度主導失誤率,在固定位置下等級訊號才清楚(0.62)。圍棋分枝更大、熵更高,但「一邊倒位置做錯」(低熵處失誤)在圍棋同樣存在(例如簡單的打吃/逃子)。風險:等級條件 log p 本身在低熵位置對各 profile 幾乎相同,序列模型可能已隱式學到此權重;且 Kuboki 無加權仍達高精度,顯示 n 局聚合後稀釋不嚴重。需重新評分是主要成本。
- **(2) 反事實分歧**:價值中等偏高且成本最低。它直接對應 Maia-2 的「過渡位置」概念,把等級訊號集中到 ~20–30% 的位置;對本專案,離散度特徵還能作為「每手置信權重」餵進既有序列模型。風險:Chen 的 SLvote(逐位置投票)只到 49%@100 局,說明離散的 argmax 投票遠劣於連續似然加總;應用連續加權,不用投票。
- **(3) 誤差分布形狀**:價值中等。注意專案已判定「逐手品質時間序列統計、失誤叢聚」無效,但那是以**引擎價值損失**或時序結構為基礎;此處是以 **−log p 的分位數/尾部**(與幾何平均非單調相關)。Kuboki 附錄 F 指出圍棋相鄰等級的損失分布重疊大,是重尾特徵在圍棋可能弱於西洋棋的直接警訊;Regan 式 (s,c) 擬合在圍棋的對應需要「以強策略機率排名代替 centipawn 差」。
- **(4) 位置頻率/定式等級先驗**:價值可能最高,因為它是**唯一引入訓練標籤資訊**的家族(策略模型不知道「這個佔角順序在本資料集中多半是 5 級」);Moudřík 的樣式計數在圍棋是最強單一家族。風險:(i) 必須 out-of-fold 建書,否則嚴重洩漏;(ii) 10 類 ~250k 局,全盤書大概十餘手內即稀疏,角部定式書較深;(iii) 讓子局開局結構不同,須分開;(iv) 本資料集的匿名化若打亂對稱(旋轉/鏡射)需正規化後再雜湊;(v) KataGo trunk 向量可能已部分編碼開局選擇,增益可能與既有特徵重疊。
- **(5) 局部/全局拆分**:價值低到中。圍棋的局部戰鬥是強特色(西洋棋無直接對應),Moudřík 顯示局部序列單獨訊號弱;但「該脫先卻續戰/該續戰卻脫先」的比例是對「大局觀」的直接量化,可與 (2) 結合(局部位置與全局位置的 profile 分歧不同)。
- **(6) 分階段**:價值低但近零成本;Chen 的證據說明開局對級位幾乎無鑑別力、殘局對高段仍有,等於建議對不同手數段使用不同權重或分別輸入。風險:序列模型已見手數位置資訊,增益可能很小。
- 整體:所有家族都與「28 profile 幾何平均」高度相關;Kuboki 證明在圍棋加上引擎損失特徵得 0 增益,是「疊加新家族邊際價值趨近 0」的最直接先例;只有 (4) 的資訊來源不同於策略模型,期望邊際最大。

### Gaps
- 沒有任何文獻直接測試「跨等級策略分歧加權」或「策略熵正規化」對**等級估計**精度的影響(Maia-2/ChessMimic 的熵與單調性分析未連到等級估計)。
- 沒有圍棋文獻量化「開局書/定式偏離手數」對等級預測的貢獻;Moudřík 的樣式特徵雖強,但未與 KataGo 式策略似然並列比較。
- Regan (s,c) 模型在圍棋的實作與效果:無文獻。

---

## KQ4:要幾局才能達到某等級精度?疊加特徵在強模型似然之上的邊際報酬?

### Takeaway
圍棋(Fox,11 級):Chen 的強度估計器 15 局 >80%、100 局 97.5%;Kuboki 隨機抽同級對局 1/5/10/20 局 = 37.5/64.8/79.6/91.7%(±1 級 82.8/98.8/99.9/100%),但**改為「同一位棋手的對局」抽樣**,5/10/15 局只有 58.1/61.0/63.2%(±1 級約 96.6%),顯示真實題目設定下的天花板受棋手自身等級噪音限制,約在 60–65% top-1。邊際報酬:在圍棋,引擎損失特徵疊加於多等級策略 + 強度分數上 **+0.0 pp**;強度分數疊加於策略先驅 +3–6 pp(n=1–10);西洋棋同樣結構但損失特徵 +0.7–1.4 pp。西洋棋單局 MAE:有時鐘 182、無時鐘 239、基線 346(Omori)。

### Cited Findings
- Chen et al. (ICLR 2025):圍棋 SE/SE∞ 15 局超過 80%、100 局 97.5%;±1 級容忍下 1 局約 80%、6 局 >99%;監督式基線 SLsum/SLvote 100 局僅 49%;西洋棋 SE∞ 26 局 >80%、100 局 93.38%;僅用單一位置約 60%(對比 20 局全用約 80%)— [arXiv 2502.17109v2](https://arxiv.org/html/2502.17109v2)
- Kuboki et al. (2505.00279) Table 1 隨機抽樣(圍棋 11 級):n=1/5/10/15/20 → 37.5(82.8)/64.8(98.8)/79.6(99.9)/87.5(100)/91.7(100);西洋棋 8 級:32.2/58.7/74.3/81.5/86.2;Chen 基線同設定圍棋 33.7/56.3/71.4/78.2/83.8 — [arXiv 2505.00279v1](https://arxiv.org/html/2505.00279v1)
- Kuboki 玩家特定抽樣(每級 100 位棋手各 20 局):圍棋 n=5/10/15 → 58.1/61.0/63.2(±1 級 96.6 @ n=15);西洋棋 52.2/57.0/60.4;Chen 基線圍棋 51.6/59.4/61.1 — [arXiv 2505.00279v1](https://arxiv.org/html/2505.00279v1)
- Kuboki 消融(圍棋,隨機抽樣,All vs w/o Loss):37.5/37.9、64.8/64.8、79.6/79.8、87.5/87.4、91.7/91.9 → 損失特徵 0 增益;w/o Strength 34.4/60.0/73.5/82.0/87.7;w/o Prior 32.3/58.1/73.0/81.1/86.4;西洋棋 All vs w/o Loss:32.2/31.5、58.7/58.6、74.3/72.9、81.5/81.1、86.2/85.0;玩家特定抽樣圍棋 All 58.1/61.0/63.2 vs w/o Loss 56.7/61.7/62.8 — [arXiv 2505.00279v1](https://arxiv.org/html/2505.00279v1)
- Kuboki 的特徵全是等權幾何平均/算術平均,未用熵、難度或位置加權;僅以 N_cut(前 50/100/∞ 手)截斷 — [arXiv 2505.00279v1](https://arxiv.org/html/2505.00279v1)
- Omori & Tadepalli 單局:MAE 182(時鐘)/239(無時鐘)/346(平均值);去時鐘 +57(24%)— [arXiv 2409.11506v1](https://arxiv.org/html/2409.11506v1)
- dsjoerg:Stockfish 時間每加倍只進步 0.3–0.4 分;Bayesian skill 模型特徵孤立有效、疊加無增益;材料特徵 ≈1 分;時間控制 ≈5 分 — [blundercheck](https://github.com/dsjoerg/blundercheck/blob/master/README.md)
- Turk & Polson:期望簽名收斂率 N^(−1/2),「few dozen to a few hundred games」;N=100/400/800 局的校準均值 5.6/6.0/6.1 — [arXiv 2606.18544](https://arxiv.org/html/2606.18544)
- Moudřík:每位棋手 10–50 局,26 級尺度 RMSE 2.66 級 — [arXiv 1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969)

### Inferences
- AI CUP 每題 5–20 局、10 類:若題目來自同一棋手(與 Kuboki 玩家特定抽樣相同),文獻天花板約 top-1 60–65%、±1 級 ~97%;若題目是同級隨機對局,則 20 局可達 ~90%。兩者差距 ~30 pp 全部來自棋手自身等級噪音,**任何著手特徵都無法消除**,新特徵的可觀測增益應以玩家特定設定的 +1–2 pp 為合理期望(Kuboki 的 All vs Chen 基線在該設定只差 +1.6–6.5 pp)。
- 在強策略似然已存在時,新增「與策略似然同源」的特徵(損失、品質時序)邊際接近 0;只有來源不同的資訊(訓練標籤的位置頻率先驅;若有的話時鐘)有較大期望。

### Gaps
- 無任何文獻給出「10 類、5–20 局、匿名同棋手」設定下的精度表;以上為最接近之代理。
- Chen 在 1/5/10/20 局的精確數字只在圖中,文字未給。
- 無文獻量化「對手著手」對本方等級估計的增益(專案已用)。
