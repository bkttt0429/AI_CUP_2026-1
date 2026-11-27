# 對手如何形塑可觀察的棋力訊號：對手感知（opponent-aware）特徵家族研究筆記

研究日期：2026-11-04。範圍：對手強度對棋手「可觀察著手品質」的影響證據、評分系統理論中的對手建模、KataGo humanv0 的雙方等級條件化、以及尚未在本專案測試的互動特徵。已排除題目列出的「已使用」與「已死」路線（own/opp 似然和、pmi、own−opp 差、讓子方向、response geometry、blunder 後 keep/giveback、勝率軌跡、終局 value 代理）。

本專案現況（供冗餘判斷，來自原始碼）：
- 每手已儲存：目標方著手在 18 個 Human SL profile（12k…6d）下的 log p（`moves_*.npz`: `logp [M,18]`, `game_row`, `t`），以及 10 個 L1b rank head 的 log p（`[M,10]`），對手方的 10-head log p 亦有（`l1b_score.py --opponent`）。完整 policy 向量 / 熵**未**儲存（`scripts/pilot_score.py`、`src/aicup/scoring.py`）。
- `src/aicup/humansl.py` 的 `make_profile_meta(black_rank, white_rank=None)` 已支援 `rank_<b>_<w>` 雙等級 profile，但目前只用對稱 profile 評分。
- `fused_opp_diff`（對手 pmi − 自己 pmi 的均值、對手 pmi 的均值/標準差）是目前最大的增益 +0.018（`src/aicup/fused.py` docstring）。

---

## Q1. 棋手的著手品質是否依對手強度而變？（鬆懈 vs 專注 vs 不變）

### Takeaway
證據指向「有影響，但量級小且方向依情境而異」：線上快棋中，面對較強對手棋手變慢、變準（Fernandez Slezak & Sigman 2012）；菁英古典賽中，對上超級巨星（≈+100 Elo）的 ACPL 反而上升約 11–17%（Bilen & Matros）；Maia-2 顯示對手等級只在 ≤6% 的局面改變人類的選擇，自身等級則 ≤22%。對本專案（配對 ±1 段、無時鐘）而言，對手效應應視為一個「小而方向未知」的偏差來源，適合讓模型學 own-quality × 估計差距的交互作用，而非預設方向。

### Cited Findings
- Maia-2（NeurIPS 2024）把活動方與對手的 Elo 各自查表為獨立嵌入 e_a、e_o，注入自注意力的 query；動機是先前模型「ignore opponent skill level, meaning that predictions cannot vary as a function of opponent strength」，而棋手「may adjust their strategy based on their perception of the opponent's skill」。固定局面、改變等級設定（Fig. 3B）：改活動方等級「can change the decision up to 22% of the time」，只改對手等級「will only change the decision up to 6% of the time」；作者稱後者直覺上「one's opponent shouldn't affect one's decision」但人類「sometimes take their opponent's skill level into account」。訓練時「games between players of different skill levels are over-sampled」。— [Maia-2 arXiv HTML](https://arxiv.org/html/2409.20553)；[NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/250190819ff1dda47cd23cecc0c5a69b-Abstract.html)
- Maia-2 在同等級測試集的著手預測準確率：Skilled 51.72%、Advanced 54.15%、Master 53.87%、平均 53.25%（最佳 Maia-1 平均 51.39%）；去掉 skill-aware attention 的消融降到 51.75%（完整子集模型 52.74%）。— [Maia-2 arXiv HTML](https://arxiv.org/html/2409.20553)
- Fernandez Slezak & Sigman，*J. Exp. Psychol.: General* 141(3):527–538, 2012（PMID 22004170, DOI 10.1037/a0025761）：180 秒快棋資料；面對較強對手時棋手「adopt a more conservative strategy … with slower and more accurate moves」；作者判定此轉變是次優的——若沿用面對同級對手的策略勝率會更高（prevention-mode 理論）。— [Europe PMC 記錄](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Do%20not%20fear%20your%20opponent%22%20AND%20AUTH:%22Sigman%22&resultType=core&format=json)；[摘要轉述](https://tartajubow.blogspot.com/2012/03/show-no-fear.html?m=1)
- Sigman 等 2010（*Frontiers in Neuroscience*，同團隊的 FICS 資料庫：>2.8M 局、44,069 名玩家、主分析為 3 分鐘無加秒棋局）：高分者（>1900）開局快、中盤慢；低分者首次 blunder 平均在第 17 手前、高分者約第 40 手；但「blunder counts only modestly separated high and low rated players」。該文**未**以對手分數或分差分析。— [Frontiers 2010](https://www.frontiersin.org/articles/10.3389/fnins.2010.00060/full)
- Bilen & Matros，「The Queen's Gambit: Explaining the Superstar Effect Using Evidence from Chess」（JEBO 接受稿）：ChessBase Mega Database 2020、1962–2019 年 >300 場邀請賽、約 25,000 局、>2M 手，Stockfish 11 深度 19（每手 0.5 秒）；超級巨星與同代前十的 Elo 差「about 100 points」（≈64%–36% 期望勝率）。Table 3（控制棋手/年份/輪次/場地固定效應、ANN 局面複雜度、**對手 ACPL**、顏色、手數；按賽事叢集 SE）：對上 Carlsen 的 ACPL +1.868\*\*\*（SE 0.449；該期樣本均值 ACPL 16.58）、blunder +0.080\*\*\*、勝率 −0.069\*\*\*、敗率 +0.169\*\*\*、複雜度 +0.172（n.s.）；對上 Kasparov ACPL +2.667\*\*\*、複雜度 +1.575\*\*\*；對上 Fischer ACPL +4.410\*\*\*、複雜度 +2.635\*\*\*；合併「Against Superstar」ACPL +2.887\*\*\*（0.263）、複雜度 +0.666\*\*\*（23,614 局、1,825,871 手）。作者註：對巨星的棋局更複雜「could be due to the superstar's willingness to reach more complex positions in order to make the ability-gap more salient」。— [Lancaster eprint PDF](https://eprints.lancs.ac.uk/211153/14/Superstar_Effect_JEBO.pdf)
- Adhikari, Anatolyev & Dagaev，「Do Mistakes Provoke New Mistakes? Evidence from Chess」（IEEE Trans. Games）：FIDE 2603–2872 的職業棋手、Stockfish 12 每手 7 秒（深度 ≥17 半手）、有序 probit；結論「small inaccuracies lead to less accurate play in future … blunders surprisingly tend to discipline players」。協變數中自身 Elo 的係數 −0.0726×10⁻²（SE 0.0096×10⁻²，Elo SD 54.4）；局面評估值 ev 的係數 +3.71×10⁻²（局面越好越容易隨便）。我摘出的協變數清單（elo、ev、taken、white、自身過去失誤）**不含對手的失誤變數**。— [NES PDF](https://www.nes.ru/files/faculty/Do-mistakes-provoke-new-mistakes-Evidence.pdf)
- Anderson, Kleinberg & Mullainathan 2016（KDD；arXiv 1606.04956）：以 tablebase 為真值，「features describing the inherent difficulty of an instance are significantly more powerful than features based on skill or time」，且存在 skill-anomalous 局面。— [arXiv](https://arxiv.org/abs/1606.04956v1)
- 競賽理論（運動經濟學）：Berger & Nieken 以德國手球聯賽資料發現「a highly significant negative impact of the matchup's heterogeneity on joint team efforts」，且「this overall decrease in efforts is almost entirely driven by the reaction of the ex-ante favorite team」。— [IDEAS/RePEc](https://ideas.repec.org/p/trf/wpaper/325.html)
- 大型「準確度 vs 對手分差」的公開分析**未找到**。找到的只有：個人資料（Lichess Insights）顯示對 100+ 分較弱者敗率 31%、對 100+ 分較強者勝率 32%，但無準確度數據 — [Stepping Up vs. Stepping Down](https://chessforlife.substack.com/p/stepping-up-vs-stepping-down)；約 2,500 局 Lichess blitz 的 ACPL 只按自身分數分桶，未分析對手 — [kwojcicki blog](https://kwojcicki.github.io/blog/CHESS-BLUNDERS)；Lichess 論壇的說法（對強者 CPL 低、對弱者 CPL 高）皆為軼事 — [Lichess forum](https://lichess.org/forum/general-chess-discussion/correlation-of-accuracy-to-rating)。
- 圍棋讓子：Mori 2016 以線上平台資料與讓子規則的斷點做 RDD，「an additional handicap typically changes the game odds by about 30 percent points」，效應隨讓子數變化；**未**分析著手風格。— [arXiv 1606.05778](https://arxiv.org/abs/1606.05778)

### Inferences
- 兩類實證的「方向」相反但可相容：線上快棋的鬆懈/專注是**時間分配**效應（對強者放慢→更準），古典菁英賽的巨星效應是**壓力/被動複雜化**效應（對手製造複雜局面→ACPL 上升）。本專案無時鐘，Human SL 的 log p 不含時間，故第一類效應不會直接進入資料；第二類（強者把局面變難）會以「對手的 log p 偏低」出現在資料中，且會被現有 opp 特徵**誤讀為對手較弱**——這是 Q4-E「製造複雜度」特徵的動機。
- 量級估計：Maia-2 的 6% vs 22% 說明對手條件化的訊息量約為自身等級的四分之一；Bilen–Matros 的 ACPL +11%（1.868/16.58）對應 ~100 Elo 的差距，約等於本賽制 1 段差距的尺度。配對 ±1 段的線上棋局中，對手造成的著手品質偏移應在「等級似然」尺度上為 0.1–0.3 段量級，不會改寫 own 似然和，但足以作為交互作用修正。
- Anderson et al. 的結論（難度 > 技能）意味任何「對手剛犯錯／對手表現」的條件化特徵都必須對局面難度做 difference-in-difference，否則只是量到雙方共享的局面複雜度。

### Gaps
- 沒有找到任何以 Lichess/chess.com 大樣本量化「準確度 vs 分差」的研究或資料集；Lichess Insights 雖有 opponentStrength 維度（−200/−100/+100/+200 分桶），但無公開彙整。
- 圍棋讓子棋對白方「過分手」風格的影響只有論壇軼事（Life in 19x19、Nordic Go Dojo），無量化研究；且測試集沒有 HD≥2，相關性低。
- Hsueh & Ikeda（IPSJ 2024，「KataGo HumanSL だけで十分か」）的全文無法取得（IPSJ 503 / CiNii 空頁），無法引用其每段預測準確率。

---

## Q2. 評分系統理論：沒有勝負結果時，能否把雙方的逐手證據當成「無結果的成對比較」？聯合估計是否優於獨立估計？

### Takeaway
Elo/Glicko/TrueSkill 系列全部以勝負為觀測；TrueSkill 2 顯示把個人表現統計當作額外證據能大幅加速/提高估計（Halo 5：68% vs 52%）。圍棋棋力估計的最新文獻（Chen et al. ICLR 2025；Kuboki et al. 2025）只用目標方著手、不建模對手，且以 Bradley–Terry 跨等級比較著手強度分數。「以雙方逐手證據做無結果成對比較並加入配對先驗」在文獻中沒有直接前例；本專案的 fused_opp_diff 是一個經驗版本，可升級為每局聯合後驗 + 精度加權，但相對於已看到兩方特徵的 MLP，增益可能有限。

### Cited Findings
- TrueSkill Through Time（NIPS 2007，Dangauthier, Herbrich, Minka, Graepel）：以每年一個潛在技能變數串成時間序列、EP 平滑；分析約 3.5M 局西洋棋，發現模擬「逼和能力」顯著提升預測力。觀測仍是勝負。— [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2007/hash/9f53d83ec0691550f7d2507d57f4f5a2-Abstract.html)
- TrueSkill 2（MSR-TR-2018-8，Minka, Cleven, Zaykov）：在勝負外加入個人表現（擊殺數、經驗、小隊等），「uses the individual scores of players to weight the contribution of each player」，在 Halo 5 預測歷史結果 68%（原版 52%）。— [MSR TrueSkill publications](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/publications/)
- Chen, Shih & Wu，「Strength Estimation and Human-Like Strength Adjustment in Games」（ICLR 2025）：SE 網路對每個 (s,a) 輸出強度分數 β，訓練用跨等級群組的 Bradley–Terry 式 softmax（負對數似然），一局以幾何平均（β 平均的指數）聚合；資料 FoxWeiqi 5k–9d 共 11 級、每級 45,000 局訓練；「over 80% of accuracy with only 15 games and reaching 97.5% after evaluating 100 games」，監督式基線約 49%。文中**未**說明是否只用目標方著手，也**未**使用對手等級。— [arXiv HTML](https://arxiv.org/html/2502.17109v2)
- Kuboki, Ogawa, Hsueh, Yen & Ikeda，「Policies of Multiple Skill Levels for Better Strength Estimation in Games」（arXiv 2505.00279）：以多等級 policy 下已下著手先驗的幾何平均（即似然）加上 SE 分數做特徵；「Given a match, we separate the state-move pairs by the players」→**只用目標方著手**，不用對手。圍棋 Table 1（隨機抽局）準確率 n=1/5/10/15/20：37.5/64.8/79.6/87.5/91.7%（Chen et al. 33.7/56.3/71.4/78.2/83.8%）；**按玩家抽樣**（Table 4）掉到 58.1/61.0/63.2%（n=5/10/15；Chen 51.6/59.4/61.1%）。— [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- QuickSkill（arXiv 2208.07704）：以新手前幾局的序列表現特徵用神經網路預測未來評分，解決冷啟動；摘要未給數值增益。— [arXiv](https://arxiv.org/abs/2208.07704)
- 遊戲業專利（EA，US12364929B1）：Siamese 網路由兩隊的賽中統計預測技能，約束「the difference between two team's skill ratings should be equivalent」到結果——仍以勝負訓練。— [Google Patents](https://patents.google.com/patent/US12364929)
- Regan & Haworth 2011「Intrinsic Chess Ratings」：以著手品質（而非勝負）定義棋力，參數 s（sensitivity）、c（consistency），對大量錦標賽棋局回歸對應 Elo；摘要未討論對手強度。— [PDF](https://www.cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)
- 行為計量學（stylometry）：McIlroy-Young 等（NeurIPS 2021）用 100 局在 2,844 名候選中辨識棋手達約 98.2%（未見玩家 97.9%），在 41,184 名 Lichess 玩家上 95.3%。— [NeurIPS 2021 paper](https://papers.nips.cc/paper_files/paper/2021/file/ccf8111910291ba472b385e9c5f59099-Paper.pdf)；[slides](https://dev.neurips.cc/media/neurips-2021/Slides/33030.pdf)

### Inferences
- 「無結果成對比較」的正確形式：對每局 g，兩方的逐手似然給出 p(moves_t | r_t) 與 p(moves_o | r_o)，配對機制給出先驗 π(r_o − r_t)（可由訓練集的對手 pmi − 自己 pmi 經驗分佈估計，題目述 ±1 段為典型）。聯合後驗 p(r_t | g) ∝ Σ_{r_o} p(moves_t|r_t) p(moves_o|r_o) π(r_o−r_t) 在 10×10 網格上可閉式計算（CPU，毫秒級）。相對獨立估計，聯合估計把對手證據**按先驗寬度折減**後併入，而不是像現在的 MLP 以學到的權重併入——理論上更穩，但在 10 萬級題目的訓練集上 MLP 已能學到近似權重，所以期待增益 ≤ fused_opp_diff 的一小部分（推測 +0.000 ~ +0.004）。
- 真正可能新增訊息的地方是**精度加權**：每局的證據量不同（手數、對手 pmi 的後驗寬度），而目前是各局 pmi 的簡單平均/標準差。以每局聯合後驗的精度加權（長局、對手證據明確的局權重高）屬「尚未使用」。
- Kuboki 等的「按玩家抽樣」掉 25 個百分點，提示玩家內的相關性（風格、同一對手池）會讓隨機抽局評估高估增益；本專案的 fold 0 是按玩家切分，符合保守評估。stylometry 的 98% 也提醒：對手其實是一組人（同一伺服器同一時期的同級玩家池），對手特徵部分在量「玩家池」而非「等級」，在測試集的玩家池不同時有偏移風險（見 Q5）。

### Gaps
- 沒有找到「兩人同時從行為軌跡聯合估計技能且不使用勝負」的文獻；也沒有「聯合 vs 獨立估計」的對照實驗數字。
- TrueSkill 2 的全文公式未取得（只有專案頁與摘要）。

---

## Q3. KataGo humanv0 的對手等級條件化：文件怎麼說、能否用「對手估計等級」條件化的 profile 來銳化似然？

### Takeaway
humanv0 的 metadata 編碼器**同時**輸入雙方等級（各 34 位 thermometer，自身 6–39、對手 40–73 位），分析引擎文件明確提供 `rank_{BR}_{WR}` 用於「predict how black with the rank BR and white with the rank WR would play against each other」，並警告 >9 段差或與讓子不符時為分佈外。官方沒有量化對手等級改變 policy 的幅度；最接近的旁證是 Maia-2 的 ≤6%。本專案的 `humansl.py` 已能產生雙等級 metadata，所需只是重新評分。

### Cited Findings
- `sgfmetadata.cpp`：`fillMetadataRow` 依 `nextPlayer` 取 `invPlaRank`/`invOppRank`，各以 1.0 填入 `min(invRank, RANK_LEN_PER_PLA=34)` 個欄位（自身 RANK_START_IDX 6–39、對手 40–73）；`getProfile` 對 `rank_5d` 呼叫 `makeBasicRankProfile(inverseRank, inverseRank, preAZ)`（雙方同級），對 `rank_5d_3k` 以 `_` 拆成 `inverseRankBlack`/`inverseRankWhite`；rank profile 同時設 `bIsHuman=wIsHuman=true`、`gameRatednessIsUnknown`、`tcIsByoYomi`、`source=SOURCE_KGS`；`preaz_` 把日期設為 2016-09-01（否則 2020-03-01）。— [KataGo sgfmetadata.cpp](https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/neuralnet/sgfmetadata.cpp)
- `Analysis_Engine.md`：`humanSLProfile` 允許 `preaz_20k`…`preaz_9d`（2016 前 AlphaZero 開局風格）、`rank_20k`…`rank_9d`（現代開局）、`preaz_{BR}_{WR}` / `rank_{BR}_{WR}`（「predict how black with the rank BR and white with the rank WR would play against each other」，模型知道對方比自己強或弱）、`proyear_1800`…`proyear_2023`；「for rank differences > 9 ranks, or drastically mis-matched to the handicap used in the game」可能分佈外；多 visit 時「KataGo will NOT match the strength of a player of the given humanSLProfile」；一 visit、全溫度最貼近該段棋手，「up to mid-high dan level」。— [KataGo Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
- v1.15.0 發行說明：humanv0 「trained on a large number of human games to predict human moves across players of different ranks and time periods」；無搜尋的原始網路「most analogous to a top player's 'first instinct with no reading'」，高段的預測較不準。未給訓練伺服器/局數，也未說明對手等級的效應。— [KataGo v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)
- 本專案 `src/aicup/humansl.py`：`make_profile_meta(black_rank, white_rank=None, preaz=False)` 已鏡射 `makeBasicRankProfile`，`white_rank` 預設等於 `black_rank`（即目前 18 個 profile 全為對稱）。— `/home/u2507115/BK_AICUP_2026-1/src/aicup/humansl.py`

### Inferences
- 對稱 profile `rank_r_r` 等於假設對手與自己同級；在對手實際強/弱 1–2 段時，模型的「自身等級」解讀會與「對手等級」耦合，造成系統性偏差。以對手估計等級條件化（`rank_{r}_{r_o*}`，r_o* = 對手 pmi 四捨五入）或對 r_o 邊際化（Σ_{r_o} p(r_o|opp evidence)·p(moves|rank_{r}_{r_o})）理論上給出更乾淨的 p(moves_t | r_t)。
- 期待的增益：Maia-2 的 ≤6% 局面改變上限、且訓練時異級棋局被過採樣，說明 humanv0（KGS 等伺服器也多為近級配對）學到的對手效應應為小量；推測對 R 的增益在 +0.001 ~ +0.006 之間，主要來自對手 pmi 與自己 pmi 差 ≥1 段的題目（約佔題目一部分，可由訓練集 fused_opp_diff 分佈查）。
- 成本：GPU 重評分。只對「題目的局」評分：每局多算 k 個條件化 profile（k=1 用對手 pmi；k=3 用 r_o*±1 做邊際化）。相較一次 18-profile 評分，成本是原來的 k/18 倍（humanv0 的 metadata 不同就需重算整個網路前向，無法重用 trunk；但 L1b 的 10 head 共享 trunk 不受影響）。
- 風險：分佈外（>9 段差）在本資料不會發生（10 級內）；HD≥2 的局用對稱 profile 與讓子不符本就有警告，用雙等級 profile 反而更一致，但測試集沒有 HD≥2。

### Gaps
- 沒有官方或第三方的數字說明 `rank_{BR}_{WR}` 相對 `rank_r` 的 policy 變動幅度（KL 或 top-1 改變率）；需在本資料上自測（建議先在 2,000 局抽樣上量 KL(rank_r_r ‖ rank_r_{r±2})）。
- humanv0 的訓練資料來源與各等級對局數量未公開。

---

## Q4. 尚未測試的互動特徵家族：定義、證據、預期增益、成本、風險、與現有 own/opp 似然和的冗餘

### Takeaway
最值得做的三個家族依序為：(E) 「製造複雜度 / 誘錯」把對手低似然中「被目標方製造的部分」分離出來（與現有 opp 特徵的錯誤歸因直接對沖）；(A) 對手條件化似然（Q3，重評分）；(B) 每局聯合後驗的精度加權聚合（CPU、零風險、增益小）。(C) 品質共變/領先滯後與 (D) 懲罰率可從已儲存的逐手 log p 計算，但受局面難度共因支配，預期增益最小。

### Cited Findings
- 誘錯可被量化且非零：Hamade 等（ICLR 2024，「Designing Skill-Compatible AI」）的 EXP 代理用 Maia 的著手分佈選「最大化期望結果」的著手，在 STT 框架對對方 junior 的誘導損失（Tricking）為 1.38（±0.05），Leela 為 −0.27；TREE 0.54、ATT 0.25；HB 框架 4.8% vs 3.8%。— [arXiv HTML](https://arxiv.org/html/2405.05066v1)
- 強者會把局面變複雜：Bilen & Matros 合併樣本中對巨星的局面複雜度 +0.666\*\*\*（對 Fischer +2.635\*\*\*、Kasparov +1.575\*\*\*），且他們的局面複雜度（ANN 預測的期望 ACPL）與實際 ACPL 的斜率 1.14。— [Lancaster eprint PDF](https://eprints.lancs.ac.uk/211153/14/Superstar_Effect_JEBO.pdf)
- 難度是失誤主因：Anderson et al. 2016「features describing the inherent difficulty of an instance are significantly more powerful than features based on skill or time」。— [arXiv](https://arxiv.org/abs/1606.04956v1)
- 失誤的時序相依：Adhikari 等——小不準確使後續更不準，blunder 反而使棋手收斂；局面越好越易隨便（ev 係數 +3.71×10⁻²）。— [NES PDF](https://www.nes.ru/files/faculty/Do-mistakes-provoke-new-mistakes-Evidence.pdf)
- 對手條件化只改變 ≤6% 的人類決策（Maia-2）。— [arXiv HTML](https://arxiv.org/html/2409.20553)
- 錯位對局中弱方不調整、強方放鬆：Berger & Nieken（努力下降「almost entirely driven by the reaction of the ex-ante favorite」）。— [IDEAS/RePEc](https://ideas.repec.org/p/trf/wpaper/325.html)
- 評分器的對手統計已被控制：Bilen & Matros 的所有回歸含「opponent ACPL」為控制變數（Table 3 註），即棋藝計量學已把對手品質當作自身品質的混淆因子處理。— [Lancaster eprint PDF](https://eprints.lancs.ac.uk/211153/14/Superstar_Effect_JEBO.pdf)

### Inferences（各候選家族）

**A. 對手條件化似然（opponent-conditioned likelihood）**
- 定義：以對手 pmi（或對手 10-head 後驗）選 r_o*，用 `rank_{r}_{r_o*}`（目標為黑）或 `rank_{r_o*}_{r}`（目標為白）對 18 個 r 重算目標方著手 log p；或對 r_o 邊際化。特徵 = 條件化 18 維和（可取代或並列現有對稱和）。
- 證據：Q3；Maia-2 ≤6%。
- 預期增益：+0.001 ~ +0.006 R（推測），集中在 |對手 pmi − 自己 pmi| ≥ 1 的題目。
- 成本：GPU 重評分（題目局 × k profile）；`humansl.py` 已支援。
- 風險：對手 pmi 本身有誤（5–20 局的對手各不相同，必須**每局**用該局對手的估計，不能用題目平均）；如果 humanv0 的對手條件化幾乎無效，則與現有特徵近乎重複。
- 冗餘：高度與現有 own 似然和相關（同一著手、相近 profile）；要以「條件化和 − 對稱和」的差作為增量特徵進模型，否則 MLP 只看到幾乎共線的 18 維。

**B. 每局聯合後驗 + 精度加權聚合（paired-comparison without outcome）**
- 定義：每局在 10×10 (r_t, r_o) 網格上計算 p(moves_t|r_t) p(moves_o|r_o) π(r_o−r_t)，π 由訓練集經驗估計（HD1/HD0 分開估，因為 HD1 意味有段差）；題目層以每局後驗精度（後驗變異數的倒數，與手數成正比）加權合併。新增特徵：加權 pmi、後驗熵、各局 gap 後驗的精度加權均值/標準差。
- 證據：TrueSkill 2 把個人表現當證據的成功；Chen et al. 的 BT 跨等級框架；本專案 fused_opp_diff +0.018 證明 gap 資訊有效。
- 預期增益：+0.000 ~ +0.004（推測；MLP 已見兩方 pmi 均值/標準差，新資訊主要是精度加權與 HD 分層先驗）。
- 成本：純 CPU，分鐘級。
- 風險：若測試集的配對分佈（π）與訓練集不同，固定先驗會偏；可改用寬先驗或由模型學。
- 冗餘：中等；與 fused_opp_diff 三個特徵高度相關。

**C. 雙方品質共變 / 領先滯後（quality covariance, lead–lag, 「誰主導」）**
- 定義：以目標方與對手各自的逐手 surprisal（−log p 在各自 pmi profile 下，減去該局均值，必要時以 5 手移動平均平滑）為兩條時間序列 x_t、y_t；特徵 = 同期相關 ρ₀、交叉相關 ρ₊₁（對手錯後我是否也錯）、ρ₋₁（我錯後對手是否錯）、不對稱指標 ρ₊₁−ρ₋₁（「誰帶節奏」）、分階段（開局/中盤/官子）的 ρ₀。
- 證據：Anderson et al. 的難度主導預測雙方共變為正且主要反映局面難度；Adhikari 等的 tilt 顯示單方序列自相關；Bilen–Matros 顯示強方製造複雜度 → 不對稱指標可能反映等級差。
- 預期增益：+0.000 ~ +0.003（推測；文獻無任何「共變反映等級」的直接證據）。
- 成本：CPU；需要對手方在 18-profile 或 10-head 下的逐手 log p（10-head 版已有 `--opponent` 輸出）。
- 風險：5–20 局 × ~100 手的相關係數雜訊大；與「對手 pmi」的交互作用非線性。
- 冗餘：低（和的統計量不含交叉項），但訊號本身可能微弱。

**D. 懲罰率 / 對手剛犯錯後的品質（punishment rate, difference-in-difference）**
- 定義：定義對手「失誤手」= 對手著手在對手 pmi profile 下的 surprisal 高於該局對手 90 百分位（或在 10-head 下 argmax rank 遠低於 pmi）；特徵 = 目標方在對手失誤後 1–2 手的 surprisal 均值 − 目標方在對手「好手」（surprisal 低於 10 百分位）後的 surprisal 均值（DiD，去除局面共享難度），以及目標方在對手失誤後的「高段 profile 似然增益」（即是否下出更像高段的手）。
- 證據：Adhikari 等「ev 越高越隨便」暗示拿到優勢後品質下降（這會使懲罰率特徵有雙向效應）；Bilen–Matros 以對手 ACPL 作控制變數表明對手品質與自身 ACPL 相關。
- 預期增益：+0.001 ~ +0.004（推測）。注意「blunder 後 keep/giveback」已死，差別在這裡用的是 Human SL 似然而非 value 軌跡、且用 DiD 對難度去混淆；若死因是 value 代理噪音，此版本可能存活，若死因是訊號本身不存在，則同樣會死。
- 成本：CPU，從已儲存逐手 log p 直接算（需對手逐手 log p）。
- 風險：Human SL 的低似然不一定是失誤（可能是罕見但好的手）；用 10-head 的 rank 後驗偏移代替更穩。
- 冗餘：中低；與 own 似然和共享成分（目標方的所有手），但條件化子集的平均與全局平均的差是新資訊。

**E. 製造複雜度 / 誘錯（complexity sought, induced error）**
- 定義（兩版）：(E1) 需重評分：在目標方每手之後的局面，取對手 pmi profile 的 policy 熵 H_o（或 top-1 機率、有效選點數），特徵 = 目標方製造的 H_o 均值 − 對手製造的 H_t 均值（「我讓你的局面更難 vs 你讓我的更難」），分階段；(E2) 不重評分：對手在「目標方高段似然手」之後的 surprisal 增量（對手 surprisal 對目標方前一手 log p 的回歸斜率），即「我下得像高段時你是否更常出錯」。
- 證據：Hamade 等 EXP 誘導損失 1.38 vs Leela −0.27（誘錯是可選擇、可量化的能力）；Bilen–Matros 巨星局複雜度 +0.666\*\*\*；Maia-2 的「製造對手難題」解讀；Anderson et al. 的 blunder potential（可犯錯的合法手比例）本質上就是局面熵的度量。
- 預期增益：+0.002 ~ +0.006（推測），理由是它直接修正現有 opp 特徵的錯誤歸因——對手 log p 低目前被解讀為「對手弱 → 配對先驗說我也弱」，但若低似然是目標方製造的，應解讀為「目標方強」。這個反向的歸因偏差正是 fused_opp_diff 大增益背後可能殘留的系統性誤差。
- 成本：E1 需儲存完整 policy（或至少熵/top-k）的重評分，一次 forward 即可同時得到對手 pmi profile 的熵；E2 純 CPU。
- 風險：熟悉定式/佈局階段的熵低與等級無關 → 需按階段並以對手 pmi profile 條件化；高段對高段的熵天然較高（雙方選點都廣），需以 own−opp 差而非絕對值。
- 冗餘：E2 與 opp 似然和相關（都用對手 surprisal），但斜率/條件化差是新量；E1 與任何現有特徵都不同源（熵非已下著手的似然）。

**F. 跨局的對手強度反應斜率（gap-response slope）**
- 定義：在題目的 5–20 局內，以每局對手 pmi − 自己整體 pmi 為 x、該局目標方品質（own 似然和的 pmi 或 surprisal）為 y，取斜率與相關；表徵「對強者更準（Sigman）」或「對強者更差（Bilen–Matros）」的個人傾向。
- 證據：Q1 兩種相反方向的實證。
- 預期增益：+0.000 ~ +0.002（推測；5–20 點的斜率極噪，且與 HD 方向特徵部分重複）。
- 成本：CPU。
- 風險：題目局數少；方向在人群內可能異質而相消。
- 冗餘：中（own−opp 差的均值已用；斜率是新量）。

### Gaps
- 沒有任何文獻直接檢驗「雙方逐手品質的共變或領先滯後與等級的關係」；C/D 的預期增益純屬推測。
- 無公開數字量化圍棋中「段位差 vs 著手似然」的系統偏移；A 的增益上限需由自測 KL 決定。
- 「製造複雜度」在圍棋的實證（例如高段是否刻意走複雜變化）未找到量化研究；Bilen–Matros 的複雜度是西洋棋 ANN 期望 ACPL，不能直接換算。

---

## Q5. 風險：測試題目同色、只有 HD0/HD1 對手、配對結構可能不同

### Takeaway
所有對手感知特徵都在量「配對機制」而非純棋力；若測試集的玩家池、時期或配對規則不同，依賴 π(gap) 的特徵（B、fused_opp_diff）首先失效，而以**局內相對量**定義的特徵（C、D、E 的 own−opp 差）較穩健。Kuboki 等「按玩家抽樣準確率掉 25 點」與 stylometry 98% 的結果共同提示：對手池攜帶身份/時期訊息，是增益也是偏移來源。

### Cited Findings
- Kuboki 等：隨機抽局 vs 按玩家抽樣，n=15 的準確率 87.5% → 63.2%（Chen et al. 78.2% → 61.1%）。— [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- McIlroy-Young 等：100 局可在 2,844 人中以 98.2% 辨識個人；41,184 人時 95.3%。— [slides](https://dev.neurips.cc/media/neurips-2021/Slides/33030.pdf)
- KataGo 文件：profile 與讓子「drastically mis-matched」時分佔外；rank profile 固定 `source=SOURCE_KGS`、`tcIsByoYomi`（與目標資料的伺服器/時制未必一致）。— [Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)；[sgfmetadata.cpp](https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/neuralnet/sgfmetadata.cpp)
- Mori：讓子規則依段差斷點決定，讓子數本身是段差的函數。— [arXiv 1606.05778](https://arxiv.org/abs/1606.05778)
- Maia-2 訓練時對異級對局過採樣，顯示線上伺服器的自然配對以同級為主。— [arXiv HTML](https://arxiv.org/html/2409.20553)

### Inferences
- 同色題目：A 的 `rank_{BR}_{WR}` 需按目標色正確排列（黑在前），同色反而簡化；但若 humanv0 對黑/白有系統偏差（貼目 6.5 vs HD1 無貼目），HD1 局以 `rank_r_r` 假設同級會與「HD1 意味有段差」矛盾——HD1 局用雙等級 profile（弱方執黑）最一致，也是 A 最可能有增益的子集（HD1 佔 17%）。
- 只有 HD0/HD1：所有與 HD≥2 相關的訓練訊號在測試無用；建議訓練時把 HD≥2 局降權或排除於對手感知特徵的擬合。
- 配對結構診斷：在提交前比較測試集與訓練集的「對手 pmi − 自己 pmi」分佈（均值、標準差、|gap|≥1 比例）與「對手 pmi 的題內標準差」，若偏移明顯，則對 B/fused_opp_diff 做先驗寬化或改用不依賴 π 的 own−opp 相對量。

### Gaps
- 無法從公開資料得知主辦方測試題的配對/時期來源；此處只能給診斷程序。
- 線上圍棋伺服器的配對演算法（Fox/野狐、OGS、KGS）對段差的容忍度沒有可引用的官方數據。
