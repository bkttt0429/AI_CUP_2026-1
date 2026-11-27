# 難度正規化與資訊論式的著手品質量度：如何把 rank-conditioned policy 的「完整分佈」變成棋力特徵

研究日期 2026-11-04。範圍：圍棋／西洋棋／將棋與一般「從決策推估技能」文獻中，(a) 局面熵與難度條件化的準確率、(b) 實際著手在 p_r 下的名次（top-k 命中率）、(c) 強弱 profile 之間的分歧（Jensen–Shannon，「診斷局面」）、(d) 逐手更新的 rank 後驗及其軌跡、(e) 相對於自身 profile 的 surprisal（自我一致性）、(f) 校準式特徵（各 profile 的期望 vs 實際 top-1 吻合）。專案背景：AI CUP 2026 圍棋等級預測（10 類、每題 5–20 局、雙方著手、無時鐘／勝負），現有管線已對 18 個 KataGo humanv0 rank profile 與 10 個微調 rank head 累加 log p(實際著手)，並把 trunk 向量送入序列模型。

---

## 關鍵問題 1：(a)–(f) 哪些已被研究過（圍棋、西洋棋、將棋、IRT／Bradley–Terry 類的決策技能估計）？結果如何？

### Takeaway
(a)、(b)、(d)、(f) 在西洋棋文獻（Regan–Haworth 的 Intrinsic Performance Rating 系統、Haworth–Di Fatta 的 Bayesian 參考代理、Anderson–Kleinberg–Mullainathan 的難度主導結論、JAIST 池田研的「期待損失」難度調整）都有直接或近似的對應，且有量化結果；(c)「診斷局面」只有間接證據（Kuboki et al. 2025 的多等級 prior 斜率論證、Maia-2 的 22% 決策隨技能改變），沒有任何論文直接用 JS divergence 加權做 rank 估計；(e) 只有 Regan 的 c（consistency）參數與 Kuboki 的 StdLoss 這類「離散度」特徵可類比，沒有「相對於自身 profile 的 surprisal」的直接研究。把局面當 IRT 題目的框架明確出現在 Regan–Biswas 2013（自稱 Rasch 多分題模型的延伸）與 ChessBase Elo-Meter（2PL IRT、Amsterdam Chess Test 的 Choose-A-Move 題），但兩者都沒有把「profile 分歧 = discrimination、熵 = difficulty」當成特徵來估計對局中的棋力。

### Cited Findings

**(a) 局面難度／熵與難度條件化準確率**

- Anderson, Kleinberg & Mullainathan（KDD 2016，西洋棋 tablebase 殘局）：預測一手是否犯錯時，「難度」特徵（blunder potential β(P)=b(P)/n(P)，即合法著手中會惡化 minimax 值的比例）單獨達到 0.73 準確率，深度 1+2 難度特徵合計 0.75；而棋手與對手 Elo 單獨只有 0.55，時間特徵 0.53；「neither adds predictive value once position difficulty features are in the model」。他們用 quantal-response 模型 γ_c(P)=β/(c−(c−1)β)，最佳 c≈15（FICS）、c≈100（GM）；2700 棋手在 β=0.9 局面的錯誤率高於 1200 棋手在 β=0.3 局面的錯誤率；不同 β 的錯誤率–Elo 曲線在對數尺度上「almost completely parallel」— [arXiv 1606.04956](https://arxiv.org/abs/1606.04956)
- Nishikawa（JAIST 碩士論文 2025-03，指導池田心，圍棋）：以「期待損失」定義局面難度 —「The expected loss is calculated as the weighted sum of the loss values of possible moves, where the weights are their predicted selection probabilities」（選擇機率取自 KataGo 的 10k,8k,…,2k,1d,…,7d,9d 各等級，棋譜來自野狐）；依難度上下調整每手 loss；用每位棋手 10 局估 rank（10k–9d），原始 loss 的 RMSE 3.975 rank → Kuboki 等人的前 150 手濾波法 1.935 → 加上期待損失調整 1.511 rank — [JAIST 10119/19809](http://hdl.handle.net/10119/19809)（PDF：https://dspace.jaist.ac.jp/dspace/bitstream/10119/19809/5/paper.pdf）
- Kuboki, Hsueh & Ikeda（GPW 2024，圍棋，KataGo HumanSL）：loss 受棋局進程影響大（200 手投降 vs 300 手終局的平均 loss 不可比），提出「連續高 loss 只計部分、不用後半盤、改看不受進程影響的『形的好壞』」等調整，用少量棋譜估棋力，達到估計值標準差 0.387、RMSE 0.752（單位未在摘要中明示，推測為 rank）— [JAIST 10119/20018](http://hdl.handle.net/10119/20018)
- Maia（KDD 2020）：「as the difference in quality between the two best moves increases, all models increase in accuracy」— top-2 著手價值差是預測性（亦即難度）的單調調節變數 — [arXiv 2006.01855](https://arxiv.org/abs/2006.01855)
- Regan 系列把難度處理成「局面內所有候選手價值分佈」而非外加權重：Regan–Biswas 2013 明言「The weight w_t may represent some measure of the value or difficulty of the decision. Our experiments in the basic model reported here use unit weights.」，並提出以「各深度評估值的變異（swing）」當內在難度量度 — [ReBiNF13.pdf](https://cse.buffalo.edu/faculty/regan/papers/pdf/ReBiNF13.pdf)；Regan 的部落格以 swing 分組：2700 級棋手找到最佳手的比例從 swing 0–1 區的 70% 掉到 swing 4–5 區的 30%；初學者在「clear-cut」局面找到最佳手超過 40% — [Depth of Satisficing (GLL 2015)](https://rjlipton.com/2015/10/06/depth-of-satisficing/)
- Guid & Bratko 的「complexity」調整：Regan–Haworth 2011 指 Guid–Bratko 的 AD 統計「subject to some adjustments for 'complexity' that we did not employ」— [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)；Guid–Bratko 2011 ICGA 文章只說「we designed a method for assessing the complexity of a position」而無公式，主結論是四種引擎、各深度下棋手排名一致（「all the programs unanimously rank one player to be clearly stronger than the other」）— [ChessBase 2011](https://en.chessbase.com/post/using-che-engines-to-estimate-human-skill)
- Turk & Polson（arXiv 預印本 2026，西洋棋反作弊，合成實驗）：指出 Regan 的聚合統計「conditions on each position's difficulty but then sums, discarding the joint pattern of which positions the player got right」；他們用路徑簽名的 Lévy area 捕捉「準確率是否恰在局面變難時上升」，並展示能讓 Regan 的 best-move z-score 完全不變、ACPL 不變（AUC 0.500）但簽名可分（AUC≈0.75）的作弊策略 — [arXiv 2606.18544](https://arxiv.org/abs/2606.18544)

**(b) 實際著手的名次（ordinal rank）、top-1／top-k 命中**

- Regan–Haworth 2011 的三個聚合統計：MM（move-match，與引擎第一選一致的頻率）、AD/AE（平均誤差）、以及「the frequency M_i of playing the i-th best move, for all i」；三種擬合法中「Calculate s,c to maximize the projected likelihood ∏_t p_t* … failed to yield even remotely accurate projections for mmp and adp」，因此改用 percentile fitting — [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)
- Regan–Biswas 2013：主要獨立擬合度量是「ordinal rank fit」ORF=Σ_k(f̂_k−f_k)²，即「第 k 好的手被下出的頻率」之投影 vs 實際 — [ReBiNF13.pdf](https://cse.buffalo.edu/faculty/regan/papers/pdf/ReBiNF13.pdf)
- Maia 2020：每個 Maia-r 的 top-1 命中率在其訓練 rating 附近達峰（Maia-1100 峰 50.8%、Maia-1900 峰 52.9%；Maia-1900 預測 1100 棋手最低 46%），峰值「usually one rating range higher than their training set」；Stockfish 33–41%、Leela 最高 46% 且對棋手等級近乎平坦（Leela 2700「matches human moves 40% of the time, no matter whether」對手是 1100/1500/1900）— [arXiv 2006.01855](https://arxiv.org/abs/2006.01855)
- Kuboki et al. 2025 使用的 policy 特徵只有「geometric mean of these priors, i.e., likelihood」（每等級一個 GM_j），全文沒有 entropy、top-k 或名次特徵（grep 0 次）— [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)

**(c) profile 間分歧／診斷局面**

- Kuboki et al. 2025：「when distinguishing strong players within a narrow skill range (e.g., 9 dan and 8 dan players in Go), selection probabilities from models at similarly high skill levels (e.g., 7 dan) often fail to provide sufficient separation. In contrast, models at relatively low skill levels (e.g., 1 dan) can clearly distinguish those strong players by how unlikely the moves are made.」並以 Fig. 2 各等級 GM 曲線的「斜率」論證哪個等級對哪段 rank 有鑑別力；附錄 Table 6（西洋棋，n=20，只用 Maia 特徵）：Maia-1500 單獨總準確率 13.8%、Maia-1900 單獨 33.0%、兩者合用 65.1% — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- Maia-2（NeurIPS 2024）：改變自身技能輸入「can change the decision up to 22% of the time」、改變對手技能最多 6%；Maia-2 在 100K 局面中 27% 為「monotonic」（正解機率隨技能單調上升）vs Maia-1 的 1%；22% 為「transitional」；論文**沒有**報告 skill-conditioned policies 之間的 KL/JS — [arXiv 2409.20553](https://arxiv.org/abs/2409.20553)
- Chen, Shih & Wu（ICLR 2025）：strength estimator 對每個 (state, action) 輸出 β，以 Bradley–Terry 式 softmax 損失 L=−Σ_i log(e^{β̄_i}/Σ_{j≥i}e^{β̄_j}) 訓練（每 rank 取 m=7 個 state-action 平均），預測時用 β 的算術平均；論文「does not analyze which positions receive high or low scores, and it does not describe any weighting of 'informative' positions」— [arXiv 2502.17109](https://arxiv.org/abs/2502.17109)

**(d) 逐手 Bayesian 後驗**

- Haworth, Regan & Di Fatta（ACG 2009/LNCS 2010）：參考代理 E(c) 以 L(j,c)=w_j^{−c}、w_j=κ+|v_1−v_j|（κ=0.1）給每個候選手機率，逐手 Bayes 更新：「p_{k,l} ≡ (p_{k,l−1} × q_k) / Σ_α(p_{α,l−1} × q_α)」，最終 Prob[E≡E(c_k)]=r_k 即棋手在代理空間的 profile；實作提醒「probabilities need to be held in log-form」、「care is required in setting/adapting the range/granularity of the hypotheses … otherwise, the prior probabilities will overly affect the posterior」；展示 Korchnoi 近 100 局滾動 c、Keres 1948 每局 c、Allwermann 逐手 c-profile（作弊偵測）— [HRdF10.pdf](https://cse.buffalo.edu/~regan/papers/pdf/HRdF10.pdf)
- Di Fatta, Haworth & Regan（CIDM 2009）「Skill Rating by Bayesian Inference」：從著手而非勝負推棋力分佈，摘要稱 c 與 FIDE Elo 強相關（全文被 Anubis 擋，未讀）— [CentAUR 4489](https://centaur.reading.ac.uk/4489/)
- Chess.com/Chessable 的「Guess the Elo, move by move」：每手輸出 rating 分佈，「simple Bayesian updates」逐手累積，但只用盤面特徵、無精度數字 — [chess.com blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)

**(e) 自我一致性／不穩定度**

- Regan–Haworth 的 c 參數：「c when large sharply cuts down the probability of poor moves」，被稱為 consistency；實測 s 由 0.07 到 0.16 隨 Elo 變動，但 c 只在 0.430–0.545 之間，因此最終改以線性內插固定 c、只擬合 s — [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)
- Kuboki et al. 2025 的 StdLoss（loss 標準差）在西洋棋被選為特徵（MeanLoss+StdLoss, N_cut=50），圍棋則選 MeanLoss(100)+MedLoss(∞)；但整體「the impact of losses was quite limited」— [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)

**(f) 期望 vs 實際 top-1 吻合（校準式）**

- Regan 的 MM_e=Σ_t p_{0,t}、σ_MMe=√Σ_t p_{0,t}(1−p_{0,t})，AE_e=(1/T)Σ_tΣ_{i≥1}p_{i,t}δ_{i,t}；用 Solitaire Set 的 AE_e 對應 Elo：IPR = 3571 − 15413·AE_e（2006–09 訓練；Elo 2700→AE_e .0572、2200→.0883）；實測偏差需乘 1.15（MM）／1.4（AE）修正獨立性假設 — [RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)
- Turk & Polson 轉述 FIDE 用的 Regan 統計：z=(Σ_i m_i − Σ_i r_i)/√Σ_i r_i(1−r_i)，門檻約 4 — [arXiv 2606.18544](https://arxiv.org/abs/2606.18544)

**IRT 框架**

- Regan–Biswas 2013：「our work is an extension of Rasch modeling … for polytomous items」，人參數 s（sensitivity）、c（consistency），局面=題目、引擎值=答案鍵，「does not need a prior difficulty rating for each position」— [ReBiNF13.pdf](https://cse.buffalo.edu/faculty/regan/papers/pdf/ReBiNF13.pdf)
- ChessBase Elo-Meter：76 題 Choose-A-Move（取自 Amsterdam Chess Test），2PL IRT「allows items to differ a) in difficulty and b) in discriminatory power」，以 test information function 給 95% CI，校準樣本 259 人、Elo 1169–2629；讀者回報的個人誤差動輒 200–600 Elo — [ChessBase Elo-Meter](https://en.chessbase.com/post/elo-meter-the-test-that-calculates-your-elo)；ACT 資料（40 題、259 人、含 Elo）見 [LNIRT::AmsterdamChess](https://rdrr.io/cran/LNIRT/man/AmsterdamChess.html)（van der Maas & Wagenmakers 2005, Am. J. Psychol. 118:29–60）
- IRT 用於遊戲的其他例子：2PL 的 difficulty/discrimination 用在 Atari/GVGAI 基準題上 — [arXiv 1811.08186](https://arxiv.org/pdf/1811.08186)；Elo/Glicko/IRT/Bayesian-IRT 比較，少量題目時 Bayesian 1PL 最準 — [core.ac.uk 215464524](https://core.ac.uk/download/215464524.pdf)

### Inferences
- 文獻中最接近「把完整分佈變成特徵」的成熟系統是 Regan 的 IPR：它對每個局面用「所有候選手的價值分佈」投影出 p_i，再聚合 MM_e、AE_e、ordinal frequencies；這些正是候選 (b)、(f) 的直接前身。Regan 明確放棄純 likelihood 擬合，理由是其對 MM/AD 的投影「wide of the mark」——這是「likelihood 與分佈型統計並不等價」在真實資料上的證據（在模型錯設下）。
- (a) 的證據全部來自「loss 為基底」的體系（Anderson、Nishikawa、Kuboki GPW2024）：難度調整對 loss 很重要，是因為 loss 本身沒有條件化於局面；而 rank-conditioned 的 log p_r(move|position) 已隱含局面條件。因此 (a) 在本專案的邊際效益應明顯小於文獻中對 loss 的改善幅度。
- (c) 在文獻裡是以「用多個等級的 profile」間接實現（Kuboki Table 6：單一 Maia 幾乎無法分辨遠離其等級的玩家，兩個合用才 65%），而非顯式 JS 加權；專案已有 18+10 個 profile，故 (c) 的主要功能（多等級鑑別）已經存在，剩下的是「顯式局面權重」這一層。
- (d) 與 (e) 本質上是現有 per-move log p 向量的確定函數（見關鍵問題 2），文獻中的 Bayesian 版本（Haworth 等）只是把 18 個 profile 換成連續 c 格點。

### Gaps
- 沒找到任何論文直接把「兩個 skill-conditioned policy 的 JS/KL divergence」當成局面權重或特徵用於 rank 估計；Maia-2 只報告 monotonic/transitional 比例。
- Biswas & Regan 2015（ICAART，"Quantifying Depth and Complexity"）全文 503 無法取得；其難度定義（swing）與各 Elo 在難易局面的表現數字只能從部落格間接引用。
- Guid & Bratko 2006 的 complexity 公式與「expected error given complexity」正規化細節未能取得全文（core.ac.uk 403、ailab.si DNS 失敗）；僅能確認 Regan 未採用該調整。
- Di Fatta et al. 2009 的「後驗需要多少手才集中」數字未取得（Anubis 擋）。

---

## 關鍵問題 2：數學上，哪些函數不由 18 個「實際著手 log-prob」決定（需要完整分佈重新評分）？需儲存什麼精簡統計？

### Takeaway
在「棋手著手確實由某個 p_{r*} 產生」的模型內，18（或 28）個 per-move log p_r(played) 的向量是 r* 的充分統計量（likelihood principle），因此後驗軌跡 (d) 與自我一致性 surprisal (e) 完全由現有資料決定、無需重新評分；熵 (a)、名次／top-k (b)、JS 分歧 (c)、期望 top-1 吻合 (f) 則需要每個 profile 在每個局面的完整（或 top-K）policy，必須改評分程式；但它們能帶來的增益只能來自模型錯設（profile 不是真實條件分佈），這是文獻（Regan 捨棄 ML 擬合、Kuboki 的 loss 幾乎無增益）一致指向的事實。精簡儲存方案：每局面每 profile 5 個標量（float16 ≈ 280 B/局面），或跨 profile 聚合成每局面約 10 個標量（≈ 40 B/局面），遠低於 362×28 float32 的 40.5 KB/局面。

### Cited Findings
- Bayes 逐手更新式 p_{k,l} ∝ p_{k,l−1}·q_k，其中 q_k 是代理 k 給實際著手的機率 — [HRdF10.pdf](https://cse.buffalo.edu/~regan/papers/pdf/HRdF10.pdf)；亦即後驗只依賴各 profile 對「實際著手」的機率乘積。
- Kuboki et al. 2025 的 policy 特徵 GM_j=(∏_i p_{j,i})^{1/k}，即「geometric mean of these priors, i.e., likelihood」，等價於 exp(平均 log p)；全部 L 個等級各一個特徵 — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- Regan 的 MM_e、σ_MMe、AE_e 公式都需要每個局面「所有候選手」的 p_{i,t} 與 δ_{i,t}，不是只有實際著手 — [RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)
- KataGo analysis engine：`includePolicy` 為 true 且有 human model 時回傳 `humanPolicy`（與 `policy` 同格式，19×19 為長度 362 的陣列，非法手 −1、pass 在最後），依 `humanSLProfile` 條件化；`moveInfos[].humanPrior`；`rootInfo` 另有 `humanWinrate`、`humanScoreMean`、`humanScoreStdev`、`humanStWrError`、`humanStScoreError`；profile 有 `preaz_20k`…`preaz_9d`、`rank_20k`…`rank_9d`、`rank_{BR}_{WR}`、`proyear_1800`…`proyear_2023`；`includePolicy`「will not significantly affect performance」— [KataGo Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
- humanv0 是「raw neural net with no search」，預測「most analogous to a top player's 'first instinct with no reading'」，高段「won't be as accurate in guessing what such players, with the ability to read sharply, would likely play」— [KataGo v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)
- 專案規模（本地 METHOD_NOTES.md，2026-10-27 PR #13 紀錄）：mock test 40 題為 positions 41,620、opp_positions 41,643（≈1,040 己方局面／題）；訓練題 203,328 — /home/u2507115/BK_AICUP_2026-1/METHOD_NOTES.md 第 204 行

### Inferences
- **由 28 個 per-move log p 決定（不需重評分）**：(d) 任何先驗下的後驗軌跡、後驗集中度（熵、最大後驗機率、SPRT 式的 log-likelihood-ratio 隨手數的路徑、首次超過門檻的手數）；(e) log p_{r*}(move) 其中 r*=後驗模式，及其變異數／尾部比例（例如「在 r* 下 log p < −5 的手佔比」）；(c) 的**弱代理**：各局面 28 個 log p_r(played) 的跨 profile 變異數／range（實際著手本身的「鑑別力」），以及限制在高變異數局面上的 likelihood ratio。前提是管線保留 per-move 向量而非只存總和——若序列模型已吃 per-move 28 維向量，這些是模型可學到的確定函數，新增價值取決於模型容量與資料量，而非資訊量。
- **需要完整分佈（要改評分碼）**：(a) H(p_r(·|s_t))；(b) rank_r(played)、top-k 指示；(c) 真正的 JS(p_weak‖p_strong) 或各相鄰 profile 對的 JS；(f) max_m p_r(m|s_t)（期望 top-1）與 1[argmax=played]（實際 top-1）、Regan 式 z_r；Regan 式 AE_e 類比（以最強 profile 的 log-policy 當「價值」時的期望損失 Σ_i p_{r,i}·(log p_top − log p_i)）。
- **充分統計量論證**：若真實資料生成確為 p_{r*}，則任何上述完整分佈統計對 r* 的資訊都已在 likelihood 內，增益 = 0；實際增益全來自錯設：humanv0 的訓練母體（不同伺服器／年代、無讀棋的 first-instinct）與 AI CUP 母體不同、profile 間的校準不一致、以及玩家本身非 stationary。Regan 在 Rybka 資料上發現 ML 擬合的 MM/AD 投影「wide of the mark」即此類錯設的實證；Kuboki 在 GM 特徵之外加入 loss 幾乎無增益（Go n=20：91.7 vs 91.9），則顯示在良好 profile 下 likelihood 已接近充分。
- **計算成本**：profile 是網路的 metadata 輸入，每個 profile 一次前向傳遞，管線既已對 28 個 profile 評分，policy 向量在記憶體中即可取得（KataGo analysis engine 用 `includePolicy`，或自寫 loader 的 log-softmax 輸出）；熵／名次／JS 每局面每 profile 僅 O(362) 次運算，相對網路可忽略。零額外 GPU 時數，只有 I/O 與儲存。
- **儲存估算**（以每題 ≈1,040 己方 + 1,040 對手局面、203,328 訓練題估）：完整 362×28 float32 = 40.5 KB/局面 → ≈ 84 MB/題 → ≈ 17 TB，不可行。方案 A「每局面每 profile 5 標量」（H_r、log p_r(played)、rank_r(played)、max p_r、JS(p_r‖p_9d)）：140 標量 → 560 B float32／280 B float16 → ≈ 1.2 MB／0.6 MB 每題 → ≈ 240 GB／120 GB。方案 B「跨 profile 聚合，每局面約 10 標量」（平均熵、最弱／最強 profile 的熵、JS(p_20k‖p_9d)、在最強與最弱 profile 下的名次、最強 profile 的 top-1 機率與是否命中、跨 profile log p 變異數、後驗模式 profile 的 log p）：≈ 40 B/局面 → ≈ 85 KB/題 → ≈ 17 GB，建議先做方案 B 以 pilot 檢驗增益，再決定是否擴到方案 A。方案 C（Regan 式 s,c 擬合）：只需最強 profile 的 top-20 log-prob + 實際著手 log-prob ≈ 21 標量/局面（float16 42 B）。
- 若對手著手也要同樣統計，容量加倍；JS 若要所有 C(28,2)=378 對則不可行，取「相鄰 profile 對」（27 個）或「對最強 profile」（28 個）即可。

### Gaps
- 專案現有評分碼是否保留 per-move 28 維 log p 向量（而非只存總和）需從程式確認；本筆記未檢視 `humansl.target_move_logps` 的輸出格式。
- 10 個微調 rank head 是否共享 trunk、是否能同時輸出 10 個完整 policy（一次前向）未確認；若是 10-way 共享輸出則儲存成本更低。
- 203,328「訓練題」是否即題數（而非局數）未從原始資料檔核對，上述 TB/GB 估算依此假設。

---

## 關鍵問題 3：「診斷局面」加權的證據；Regan 的 IPR 模型（s、c、位置相依縮放）及其以 KataGo profile 的實作方式

### Takeaway
Regan–Haworth 的 IPR 用兩個人參數——s（sensitivity，把價值差 δ 換成無量綱 δ/s，小 s 放大小差距）與 c（consistency，指數項 (δ/s)^c，大 c 把劣手機率急遽壓低）——配合經驗發現的對數縮放 δ=∫1/(1+|z|)dz（接近平手時每 centipawn 的影響約為 +100 時的 5 倍），對每個局面由「所有候選手」的價值產生 p_i=p_0^{α_i}、α_i=e^{−(δ_i/s)^c}，再以 percentile fitting 擬合；難度不是外加權重，而是透過每個局面的價值分佈自然進入（unit weights）。文獻中沒有「顯式以 profile 分歧加權」的 rank 估計實驗；最接近的是 Kuboki 2025 的多等級 prior（隱式加權）與 Chen 2025 的學習式 β。以 KataGo profile 實作 IPR 的最自然做法是把「價值」換成最強 profile（或主網路）的 log-policy，對每題擬合 (s,c) 兩個連續參數作為特徵。

### Cited Findings
- 模型定義：「log(1/p_i)/log(1/p_0) = e^{−(δ/s)^c}, where δ_i = ∫_{v_0}^{v_i} 1/(1+|z|) dz」；同號時「simplifies to |log(1+v_0) − log(1+v_i)|」；「s when small can enlarge small differences in value, while c when large sharply cuts down the probability of poor moves」；p_i=p_0^α、Σp_i=1 定出全部 — [RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)
- 縮放的經驗來源：從 15 萬局 Single-PV 資料的 AD-vs-e(m_0) 直方圖發現「a uniform scaling effect all the way down to zero advantage」，dμ=1/(1+x)dx「was found to level the AD histogram very well」；落後方的 raw AD 顯著高於領先方；縮放也壓縮 2 pawn 與 5 pawn 失誤的差距（「both blunders may be equally fatal」）；同值並列手中第一列者被下出的頻率顯著較高，施以 −0.03 修正 — [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)
- 「a 'marginal centipawn' seems to have 5x impact when the engine's evaluation is +20 to one side as when it is +100」— Regan, Intrinsic Ratings Compendium（2012 工作草稿，cse.buffalo.edu；同公式見 [RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)）
- 排除規則：「we omit: game turns 1–8, turns involved in repetitions of the position, and turns where the program judges an advantage greater than 300 centipawns for either side」— [RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)
- 擬合結果（2006–09）：Elo 2700 s=.078 c=.503；2200 s=.123 c=.490；1600 s=.165 c=.431；mmp/mma 由 56.2/56.3（2700）降到 44.0/42.9（1600）；c 幾乎常數故固定 c_fit 只擬合 s；三個年代的 s_fit 幾乎相同（無通膨）— [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)
- IPR 信賴區間（2006–09 訓練集）：Elo 2700 組 7,032 手 → IPR 2690、2σ_e 範圍 2648–2731（實測修正後 2632–2748）；1700 組 8,429 手 → IPR 1707、2σ 1642–1772；2000 組 9,471 手 → 1909、1853–1966 — Regan Compendium 表（同 RMH11b 表 2）
- 擴充方向：以各深度評估的非負加權和 y_i=Σ_d w_d·g_{s,c}(δ_{i,d}) 引入「depth」參數，「we would expect stronger players to have higher weights for higher d」— [ReHa11c.pdf](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)；但部落格說加權深度版「was dropped because its regression had unstable local minima」— [GLL 2015](https://rjlipton.com/2015/10/06/depth-of-satisficing/)
- 診斷局面的間接證據：Kuboki 2025 Fig. 2「A steeper slope suggests that the imitation model at that skill level」對該段 rank 更有鑑別力；Table 6 單一 Maia-1500 只有 13.8%、Maia-1900 33.0%、兩者 65.1% — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- Chen 2025 的 β 是學習出的 (state, action) 強度分數，用 BT 式排序損失訓練、算術平均預測；單局面預測 20 局約 60% vs 全部著手約 80%；只用前 50 手準確率下降（kyu 段尤甚），9 dan 用後 50 手仍穩 — [arXiv 2502.17109](https://arxiv.org/abs/2502.17109)
- Haworth 參考代理：L(j,c)=w_j^{−c}、w_j=κ+|v_1−v_j|、κ=0.1；指出早期 C_1(v_j)=1+|v_1|+|v_1−v_j| 的缺陷（最佳手機率不隨優勢 v 增加）；並計畫「a correlation of (various Elo) players' apparent errors with v_1」— [HRdF10.pdf](https://cse.buffalo.edu/~regan/papers/pdf/HRdF10.pdf)

### Inferences
- **IPR 的「難度加權」機制**：每個局面的 p_i 由該局面所有候選手的 δ 分佈決定；單一明顯最佳手的局面會給 p_0≈1（幾乎不提供資訊），多手近似的局面給平坦分佈（高 MM_e 變異）。這正好是「局面熵由模型內生」的實作——與本專案用 18 個 profile 的 p_r(·|s) 直接提供每局面分佈是同構的，差別是 IPR 以 2 個連續參數跨局面共享，而 profile 是離散的 18 點。
- **以 KataGo profile 實作 IPR 的三種途徑**：(i) 價值版：用 KataGo 主網路對每個候選手的 scoreLead/winrate 當 v_i（需搜尋或至少 top-K 候選手的子局面評估；專案已判定 value-based move loss 無增益，且 Kuboki 的 loss 幾乎無貢獻，故不建議）；(ii) policy 版：δ_i = log p_top(m_0|s) − log p_top(m_i|s)，p_top 取 rank_9d／proyear 或主網路 policy，對每題以格點搜尋擬合 (s,c)（Regan 用 s 步 0.02、c 步 0.2，2 維格點），把 (s,c) 或投影出的 AE_e、MM_e、ordinal fit 當特徵；每題約 1,000–2,000 手，計算量是 O(格點數 × 手數 × K)，CPU 即可；(iii) 校準版：用 Regan 的 percentile fitting 檢驗各 profile 的「percentile 均勻性」R^q_r，當成 (f) 類的 goodness-of-fit 特徵。
- (ii) 與 (iii) 的理論價值在於它們是 likelihood 的**替代估計量**（percentile／矩法），在模型錯設下可能比 ML 更穩健（Regan 的經驗）；但若專案的 10 個微調 head 已在 AI CUP 資料上校準，錯設較小，增益可能有限。
- 「診斷局面加權」在現有管線中最便宜的近似是 (c) 的弱代理（跨 profile log p 變異數），不需重評分；顯式 JS 版本需方案 B 儲存。

### Gaps
- 沒有任何已發表實驗量化「顯式診斷局面加權」對 rank 估計的增益；Chen 2025 的 β 是否集中在高分歧局面也未分析。
- Regan 的 Rybka 深度 13 以 centipawn 為「價值」；把 log-policy 當價值的 IPR 變體沒有文獻先例，其 s,c 與 rank 的對應需自行校準。
- Regan 2011 的 (s,c) 對 Elo 2σ 區間表（本筆記引用 Compendium/RMH11b 版）是對「同 Elo 群體」合併資料的投影，個人層級的變異（Kuboki 2025 的 player-specific 下準確率由 89.4% 掉到 63.2%）會更大。

---

## 關鍵問題 4：這些精煉相較於純 ACPL／純 likelihood 的改善幅度（引用數字）

### Takeaway
最可比的圍棋數字來自 JAIST 池田研與 Kuboki 2025：對 loss 基底加入難度調整，10 局 rank RMSE 由 1.935 降到 1.511（−22%；相對原始 loss 3.975 則 −62%）；對 Chen 2025 的 strength score 加入「多等級 policy 的平均 log-likelihood」特徵，10 局準確率由 71.4% 升到 79.6%、20 局 83.8%→91.7%，而再加 loss 只有 ≤0.2 點變化。Regan 的 IPR 以約 7,000 手給出 ±40 Elo（2σ）的區間；Anderson 顯示難度特徵（0.73–0.75）對預測單手錯誤遠勝技能特徵（0.55）。沒有任何文獻直接比較「熵／名次／JS 特徵」加在「完整多 profile likelihood」之上的增益——這是本專案要自己量測的空白。

### Cited Findings
- Kuboki et al. 2025（Go，11 rank 群、隨機抽樣）：Proposed vs Chen et al.：n=1 37.5 vs 33.7；n=5 64.8 vs 56.3；n=10 79.6 vs 71.4；n=15 87.5 vs 78.2；n=20 91.7 vs 83.8。消融（Go）：w/o Strength n=20 87.7；w/o Prior 86.4；w/o Loss 91.9。西洋棋 n=20：Proposed 86.2、w/o Loss 85.0、Chen 79.5。Player-specific（Go）：n=5 58.1 vs 51.6；n=10 61.0 vs 59.4；n=15 63.2 vs 61.1 — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- Chen, Shih & Wu 2025（Go，FoxWeiqi，11 rank）：「over 80% accuracy in predicting ranks by observing 15 games only, whereas the previous method reached 49% accuracy for 100 games」；100 局 97.5%；±1 rank 容忍下 1 局近 80%、6 局 >99%；西洋棋 26 局 >80%、100 局 93.38% — [arXiv 2502.17109](https://arxiv.org/abs/2502.17109)
- Nishikawa 2025（Go，10 局／人，10k–9d）：raw loss RMSE 3.975 rank → 前 150 手濾波 1.935 → 期待損失難度調整 1.511 — [JAIST 10119/19809](http://hdl.handle.net/10119/19809)
- Anderson et al. 2016：難度特徵 0.73（blunder potential 單獨）、0.75（深度 1+2）；Elo 0.55；時間 0.53；固定局面（Task 3）下 rating 單獨 0.62 — [arXiv 1606.04956](https://arxiv.org/abs/1606.04956)
- Regan 2006–09：Elo 2700 群 7,032 手 → 2σ_e 2648–2731；2300 群 17,632 手 → 2257–2328；1600 群 9,050 手 → 1496–1625 — Regan Compendium／[RMH11b.pdf](https://cse.buffalo.edu/~regan/papers/pdf/RMH11b.pdf)
- Moudřík, Baudiš & Neruda 2015（Go，KGS 6d–20k，手工特徵）：強度 RMSE 2.659 rank（Patterns 單獨 2.788；mean regression 7.503）— [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)；Moudřík & Neruda 2016 CNN 三類分類：單一局面 71.5%、單局聚合 77.9% — [Springer](https://link.springer.com/10.1007/978-3-319-49001-4_15)
- Omori & Tadepalli 2024（西洋棋，盤面+時鐘 CNN-LSTM）：MAE 182 含時鐘、239 不含（時鐘 −24%），僅與平均預測（346）比較，無 Maia/ACPL 基線 — [arXiv 2409.11506](https://arxiv.org/abs/2409.11506)
- Björkqvist 2025（西洋棋謎題難度，FedCSIS 第一名）：微調 Maia-2 是關鍵（移除後 MSE +49%），引擎特徵 +5.4%、手工特徵 +4.1%；微調 Maia-2 單獨比最終集成差 15% — [annals-csis 6497](https://annals-csis.org/Volume_43/drp/6497.html)
- Turk & Polson 2026（合成反作弊）：Regan z-score 不變的作弊策略下聚合偵測 AUC 0.500、簽名 AUC≈0.75；40 局比賽中每局 12 手協助時 e-process 檢出力 0.79、20 手 1.00 — [arXiv 2606.18544](https://arxiv.org/abs/2606.18544)

### Inferences
- 把 Regan 的區間換算：±40 Elo（2σ）需約 7,000 手；資訊量與手數平方根成比例，1,000 手約 ±105 Elo、2,000 手約 ±75 Elo。若 1 個圍棋 rank ≈ 100 Elo 級距，則 5–20 局（約 500–2,000 己方手）的理論極限約 ±1 rank（2σ），與 Chen/Kuboki 在 10–20 局達 80–92%（±1 內 ≈100%）一致；這意味著 10 類問題的天花板主要受樣本量限制，分佈型特徵的可期增益是「幾個百分點」級，而非 Kuboki 加入多 profile 那種 +8 點。
- Kuboki 的消融給出一個上界參考：在「多等級 likelihood + 學習式 β」之上，再加三個 loss 統計只動 ≤1.2 點（西洋棋）／≤0.2 點（Go）。熵／名次／JS 與 loss 同樣是「likelihood 之外的分佈資訊」，除非它們糾正了特定錯設（例如高段 first-instinct 不準），期望值應在同一量級。
- Nishikawa 的 −22% RMSE 是「對 loss 加難度」而非「對 likelihood 加難度」，不可直接移植；但它證實圍棋局面難度分佈差異足以在 10 局尺度造成 rank 偏差，支持至少把「局面熵」當控制變數餵給序列模型。

### Gaps
- 無任何研究同時擁有多 profile likelihood 特徵並報告加入熵／top-k／JS 的增益；所有「分佈型特徵」的效果大小都是間接推估。
- Chen 2025 各局數的準確率只在圖中，文字僅給 15 局 >80%、100 局 97.5%。
- Kuboki GPW 2024 的 SD 0.387／RMSE 0.752 單位未在可讀部分明示。

---

## 關鍵問題 5：各候選特徵的逐項評估（預期效果、與累加 log-likelihood 的非冗餘性、成本、風險）

### Takeaway
依「需不需要重評分」與「文獻支持度」排序：最便宜且有文獻背書的是 (b) 名次／top-1 與 (f) Regan 式期望-實際 top-1 z 分數（需方案 B 儲存、零 GPU）；(a) 熵做為序列模型的每手控制變數有 Anderson／Nishikawa 的方向性支持；(c) 顯式 JS 加權沒有文獻先例、但有多 profile 隱式版本的強證據，宜先用免重評分的弱代理試；(d)(e) 不需重評分、理論上冗餘，只在序列模型吃不到 per-move 向量時才值得做。整體預期增益為「數個百分點以內」，主要風險是與 likelihood 高度共線、以及與專案先前失敗的「每手品質時間序列」「校準」重疊。

### Cited Findings
- 「Our experiments in the basic model reported here use unit weights.」（Regan 不用顯式難度權重）— [ReBiNF13.pdf](https://cse.buffalo.edu/faculty/regan/papers/pdf/ReBiNF13.pdf)
- 「the impact of losses was quite limited」；player-specific 下「losses contributed to evaluating players' reading ability」— [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- humanv0 高段「won't be as accurate … players, with the ability to read sharply」— [KataGo v1.15.0](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)
- Kuboki 使用的是 `preaz_xx` profile（2016 前 AlphaZero 開局風格），非 `rank_xx` — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)；兩類 profile 定義見 [KataGo Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
- Maia 單一等級模型在遠離其等級的玩家上只有 46% 命中且「cannot distinguish whether the unlikeness was because the player was stronger or weaker」（Kuboki 對 Maia-1500 單獨使用的解讀）— [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)、[arXiv 2006.01855](https://arxiv.org/abs/2006.01855)

### Inferences
| 候選 | 文獻中的對應與效果 | 相對累加 log-likelihood 的非冗餘性 | 成本 | 風險 |
|---|---|---|---|---|
| (a) 熵 H(p_r) 與難度條件化準確率 | Anderson：難度特徵 0.73–0.75 vs 技能 0.55（預測單手錯誤）；Nishikawa：loss 加期待損失調整 RMSE 1.935→1.511；Maia：top-2 差距愈大命中愈高 | 中低：likelihood 已條件化於局面；熵只在「同 log p 但不同集中度」情境提供額外資訊，增益來自 profile 錯設。可作序列模型的每手控制變數（例如「低熵局面的 surprisal」） | 需重評分；方案 B 每局面 2–3 標量（平均熵、最強／最弱 profile 熵）| 與 likelihood 共線；與「每手品質時間序列」死胡同重疊，若只做聚合（難度分箱後的平均 log p）風險較低 |
| (b) 實際著手的名次／top-k | Regan 的 MM 與 ordinal frequencies f_k 是其主要擬合度量；ML 擬合給出「wide of the mark」的 MM 投影 | 中：序統計量對 profile 校準誤差穩健（名次不受溫度影響），這是 likelihood 沒有的性質 | 需重評分；方案 B 存最強／最弱／後驗模式 profile 下的名次（3 標量，int8 即可）| 名次對高熵局面雜訊大；增益可能僅在 Kuboki 式 LightGBM 元模型上可見 |
| (c) JS(p_weak‖p_strong) 診斷局面加權 | 無直接研究；Kuboki Table 6（單 profile 13.8/33.0% vs 合用 65.1%）與 Maia-2「22% 決策隨技能改變」是隱式版本的證據；Turk–Polson 指出聚合丟棄「哪些局面答對」 | 中：IRT discrimination 類比成立，但多 profile likelihood 已隱式加權（高分歧局面的 log-likelihood ratio 自然較大）；顯式權重的增量主要是去除低分歧局面的雜訊 | 弱代理免重評分（跨 profile log p 變異數）；顯式版需方案 B 的 JS(20k‖9d) 與相鄰對 JS | 「診斷」局面在 humanv0 中可能集中於開局定式／風格差而非棋力，加權可能放大年代／伺服器偏差 |
| (d) 逐手後驗軌跡／SPRT 統計 | Haworth–Regan–Di Fatta 的 Bayesian 參考代理即此；用於作弊偵測與歷史比較，無 rank 準確率數字 | 低：由 per-move log p 向量完全決定；若序列模型吃 per-move 向量則理論上冗餘 | 零重評分；每題 O(手數×28) | 與現有序列模型重複；只在模型只見總和時有價值 |
| (e) 自身 profile 下的 surprisal／不穩定度 | Regan 的 c（consistency）；Kuboki 的 StdLoss（西洋棋被選入、Go 未被選入） | 低至中：由 per-move 向量決定；但 c 類「尾部行為」參數是 likelihood 總和的非線性函數，對「偶爾大失誤」敏感 | 零重評分 | Regan 發現 c 幾乎不隨 Elo 變（0.43–0.55），鑑別力可能弱；與「每手品質時間序列」死胡同同源 |
| (f) 各 profile 期望 vs 實際 top-1 吻合（z_r） | Regan 的 MM_e/σ_MMe z-score 是 FIDE 反作弊的核心統計；IPR 2σ ±40 Elo（7,000 手） | 中：是 goodness-of-fit 向量而非 likelihood；對校準誤差敏感的方向與 likelihood 不同（profile 可高 likelihood 卻 MM 不符） | 需重評分；方案 B 每局面每 profile 1 bit（命中）+ 1 float（max p），或只存 3 個 profile | 專案已判「校準」為死胡同；需釐清先前是輸出校準還是這類 per-profile 吻合統計 |
| IPR 式 (s,c) 擬合（policy 版 δ） | Regan：s 隨 Elo 平滑變化（.078→.165），percentile 擬合優於 ML | 中：矩法／percentile 估計量在錯設下可與 ML 互補 | 需存最強 profile top-20 log p（≈21 標量/局面）；格點擬合 CPU | 無先例；s,c 與 rank 對應需自校；c 可能近常數 |

- 建議的最小 pilot：在評分碼加方案 B（約 10 標量／局面、≈17 GB 全訓練集），先用 fold-0 之外的資料檢驗三個最便宜的假設——(i) 加入「最強 profile 名次 ≤3 的比例」與「後驗模式 profile 的 MM z」是否改善 Kuboki 式元模型；(ii) 以 JS(20k‖9d) 分箱後的 likelihood ratio 是否比未分箱更集中；(iii) 熵分箱後的平均 log p 是否在同 rank 內更穩定（Nishikawa 的三指標：stochastic dominance、分佈重疊率、三類分類）。
- 專案的特殊性：試題同時包含雙方著手且無時鐘，Kuboki 2025 的 player-specific 結果（Go 15 局 63.2%）比隨機抽樣（89.4%）低很多，顯示個人風格造成的錯設是主要誤差源；這正是分佈型統計（名次、z_r）可能比純 likelihood 更穩健的情境，但也可能同樣被風格污染。

### Gaps
- 專案「校準」死胡同的具體內容（輸出機率校準 vs per-profile 吻合統計）未在本研究中查證，影響 (f) 的可行性判斷。
- 專案序列模型是否以 per-move 28 維 log p 為輸入未查證，影響 (d)(e) 是否冗餘的結論。
- 沒有任何已發表實驗把 (a)(b)(c)(f) 疊加在多 profile likelihood 之上，表中「預期效果」為推論，非引用數字。
