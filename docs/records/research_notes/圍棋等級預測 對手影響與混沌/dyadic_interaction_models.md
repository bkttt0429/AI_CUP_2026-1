# 雙人互動（dyadic）與賽局論模型：能否從「雙方聯合行為」萃取圍棋等級特徵

研究日期：2026-11-04。對象：AI CUP 2026 圍棋等級預測（每人 5–20 局、雙方著手、有讓子標記、無時鐘/勝負、對手 ±1 級內配對但身分未知）。已用：rank-conditioned policy 對自己與對手的逐手 log-likelihood、兩人後驗平均等級、自減對手差、讓子方向、雙方 ply 的序列模型。已死：response geometry（與對手上一手距離、跟著走率、脫先率）、對手失誤後 punish/keep/giveback、勝率軌跡統計、終局結果代理（分半信度 0.07–0.40 @ 20 局）。

總結一句：文獻裡「互動/協調」量測（CRQA、Granger、transfer entropy、complexity matching）在**棋類**完全沒有被用於技能估計；在認知科學裡它們預測「集體表現」的證據有，但樣本都是連續、數千點的時間序列，且 dyad 多為合作任務。賽局論「可利用性/懲罰效率」在撲克只用於評估**程式策略**，在西洋棋以 conversion/meltdown 形式存在、且**群體層級**強烈隨等級單調（62.5%→75.1%），但本專案的 per-player 試點已證明 5–20 局下的個人層級信度不足。對「聯合推論勝過獨立推論多少」：**沒有任何文獻直接量測**；最接近的 Maia-2 顯示對手等級只改變 ≤6% 的著手決策（自身等級 22%），把互動特徵的理論上限壓得很低。下方給出可從既有 per-move log p / score lead 便宜計算、且有機會通過信度門檻的少數候選定義，與其風險評估。

---

## 問題一：人際協調/適應模型（CRQA、Granger、TE、互資訊、lead–lag、complexity matching）對專長與技能差距說了什麼？有沒有用在棋類？

### Takeaway
這些方法在認知科學中確實能區分「領導者/跟隨者」與預測集體表現，而且有「專家能進入共同領導、新手退化為領導–跟隨」的證據（mirror game），但全部來自連續運動/語音/眼動序列（數千個樣本點、合作任務），**沒有任何研究把它們用於棋類或零和序貫對局的技能估計**；在圍棋這種「輪流落子由規則固定」的互動裡，時序 lead–lag 本身是退化的，只有「品質/驚訝序列」之間的方向性依賴還有定義空間。

### Cited Findings
- Complexity matching 是指兩人行為**變異性結構**（冪律尺度）對齊，而非行為本身對齊；Abney 等人在對話語音計時中發現其與 behavioral matching 不同，並主張它是更一般的人際協調動力學 — [Abney et al. / eScholarship](https://escholarship.org/content/qt24j6f27n/qt24j6f27n.pdf)；[Fusaroli et al. 2013 CogSci "Dialog as interpersonal synergy"](https://drewabney.github.io/files/Fusarolietal_2013_CogSciProc.pdf)
- 擺錘協調研究指出 global complexity matching 與局部動作調整緊密相連，「不是一個獨立的協調機制」 — [Emergent complexity matching in interpersonal coordination (ASU Pure)](https://asu.elsevierpure.com/en/publications/emergent-complexity-matching-in-interpersonal-coordination-local-)
- 個人層面：瞄準任務中動作時間變異性隨專長發展變得更分形化（這是個人運動學習，不是 dyad） — [Nature Sci. Rep. 2020](https://www.nature.com/articles/s41598-020-63056-x)
- CRQA 的 diagonal cross-recurrence profile 可用來辨識 leader–follower 型態；crqa R 套件實作此法與 windowed recurrence — [crqa CRAN vignette](https://cran.r-project.org/web/packages/crqa/vignettes/crqa_tutorial.html)
- 組織行為：Meinecke et al. 2020 用 (C)RQA 分析主管–部屬考核面談中的 leader–follower 溝通 — [FAU CRIS record](https://cris.fau.de/publications/310524300)
- Lee et al. 2025 提出「context matrix」（線性動力系統中各人當前行為依賴自己過去 vs 他人過去的程度），主張 CRQA/相關/Granger/TE 各只抓到同步或方向性影響其中一維；用序貫貝氏推論在眼動資料上，其摘要特徵可預測任務正確率並與 CRQA、Granger 部分對應 — [arXiv 2509.08811](https://arxiv.org/abs/2509.08811)
- Pair programming 眼動 CRQA：成功配對的 cross-recurrence 較高；引用文獻指出專長混合時「較專家的一方傾向主導互動」 — [Villamor & Rodrigo 2021, SIP](https://nowpublishers.com/article/OpenAccessDownload/SIP-2021-0031)
- 同一研究群依先備知識分組的 lag：兩人皆高 2.33 s、皆低 1.96 s、混合 1.51 s（混合組 lag **最短**，與「專家領先、新手跟隨」的簡單預測相反；為 diagonal recurrence 分析） — [ICCE paper (APSCE library)](https://library.apsce.net/index.php/ICCE/article/download/2251/2127/3215)（僅見搜尋摘要，PDF 未能解析，數字請謹慎）
- Transfer entropy：Trendafilov et al. 2020 在繩球節律任務用 TE 描繪方向性影響，發現 leader–follower 角色「從 dyad 關係的趨勢中浮現」 — [Frontiers Hum. Neurosci. 2020](https://www.frontiersin.org/articles/10.3389/fnhum.2020.00185/text)
- Kostrubiec et al. 2015（虛擬教師）：當 VT–人協調失穩時，一方到另一方的資訊流增加；作者認為可變的聯合表現反而促進互動與學習 — [PLOS One / PMC4646495](https://pmc.ncbi.nlm.nih.gov/articles/PMC4646495)
- Mirror game（Noy, Dekel, Alon 2011 PNAS）：兩人可在無指定領導者下共同創造新穎複雜運動、同步 <40 ms；**專家即興者在無領導者條件下表現更好**（速度與停止時間誤差更低、速度範圍更廣） — [PMC3248496](https://pmc.ncbi.nlm.nih.gov/articles/PMC3248496)；新手無法進入共同領導狀態 — [Weizmann blog](https://scienceblogs.com/weizmann/2012/01/09/going-into-the-unknown-togethe)
- Hart, Noy et al. 2014：約 15% 時間專家 dyad 產生「co-confident motion」（雙方皆無 jitter），被視為 togetherness 的範例 — [PMC3922750](https://pmc.ncbi.nlm.nih.gov/articles/PMC3922750)
- 另一即興研究：新手無法達到速度同步，因為「落入 leader–follower 型態」，專家則能同時達到間隔與速度同步 — [Sci. Rep. 2020 (Springer link)](https://link.springer.com/article/10.1038/s41598-020-72729-6)
- 對話：Fusaroli & Tylén 2016 — synergy（CRQA 類）量測是集體表現最強的統計預測子，加入 alignment 量測不改善模型 — [Cognitive Science 10.1111/cogs.12251](https://zendy.io/title/10.1111/cogs.12251)；[2015 CogSci 版 PDF](https://interactingminds.au.dk/fileadmin/interacting_minds/2015_event_files/2015_cogsci_Investigating_conversational_dynamics_Interactive_alignment_Interpersonal_synergy_and_collective_task_performance.pdf)
- Fusaroli et al. 2012：一般語言對齊對表現無正效果，**任務相關詞彙**的對齊才與集體表現強相關 — [Psychological Science](https://journals.sagepub.com/doi/10.1177/0956797612436816)
- 聯合敲擊（Konvalinka et al. 2010）：雙方相互預測與適應；跟隨者 vs 領導者的差異在 lag −1 inter-tap interval 相關與維持自身節奏的程度 — [DTU Orbit](https://orbit.dtu.dk/en/publications/follow-you-follow-me-continuous-mutual-prediction-and-adaptation-/)
- 棋類時間序列文獻中最接近的是「西洋棋走法序列的長程記憶」：高水準玩家的長時間尺度相關更強、長程記憶與水準相關（DFA/R-S，非因果或資訊量測） — [Schaigorodsky et al., arXiv 1307.0729](https://arxiv.org/pdf/1307.0729)
- 搜尋「Granger causality / mutual information + chess move quality」沒有任何命中；唯一的 Granger 應用是棋手 fMRI 腦區分析（28 位大師 vs 27 位新手） — [Frontiers Neurosci. 2020](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.605986/pdf)

### Inferences
- 文獻中的「專家 vs 新手」互動訊號（co-leadership、對稱 vs 不對稱影響）全部是**合作/模仿任務**；圍棋是零和、輪流、離散決策，領導–跟隨在「時序」上是由規則決定的（每人一手），所以 CRQA/lead–lag 若直接套在落子座標上，只會重新發明已死的 response geometry（距離、跟著走、脫先）。
- 唯一有定義空間的是「品質序列」間的方向性依賴：令 x_t = 對手第 t 手的 point loss（或 −log p 驚訝度），y_t = 自己下一手的 point loss；opp→self 的 Granger 係數/TE 量的是「對手失誤是否牽動我的失誤」；self→opp 則反向。兩者差（不對稱性）理論上對應文獻中的「較強者主導」，但棋類裡**完全沒有實證**。
- Fusaroli 2012 的教訓可類比：一般性的「對齊/同步」不帶訊號，**任務相關**的對齊才有；在圍棋，「任務相關」就是局面價值——這把我們推回 KataGo score lead 條件下的統計，而非幾何或時序同步。
- Lee et al. 2025 的 context matrix 等價於在 (x_t, y_t) 上擬合 2×2 VAR(1)：四個係數（自身持續、opp→self、self→opp、對手持續）。這是最便宜、最可解釋的 dyadic 模型，可直接用專案已存的 per-move 數列擬合（見問題五的候選 D1）。

### Gaps
- 沒有找到任何把 CRQA、Granger、TE、complexity matching 用於棋類（圍棋、西洋棋、將棋）技能估計的研究。
- 沒有找到「同步/不對稱性隨技能差距單調變化」的量化曲線（即便在運動協調文獻中也只有有/無專家的二元比較）。
- 無法取得 Villamor/ICCE 論文全文（403/PDF 無法解析），lag 數字只來自搜尋摘要。

---

## 問題二：賽局論框架——「可利用性 / 懲罰效率」作為技能量測，在撲克、西洋棋、圍棋的現況；如何用 KataGo 輸出定義；為何不是著手似然的單調函數

### Takeaway
撲克的 exploitability/LBR 是評估**整體策略**對最壞對手的損失，需要對手私有狀態分佈與完整策略，對人類手牌記錄不可計算；西洋棋有群體層級的 conversion/meltdown（800 分 62.5%/29.5% → 2800 分 75.1%/11.8%）與「優勢時失誤率翻倍」的證據，證明「對局面狀態（雙方共同產物）條件化的失誤率」確實隨等級變化；圍棋沒有任何公開的 KataGo「懲罰率 vs 等級」統計。KataGo 可定義連續版「保留對手贈分比例」，它與 policy likelihood 不是單調關係，因為兩者是 (局面, 著手) 的不同函數——一個量「典型性」、一個量「兌現的價值」。

### Cited Findings
- Exploitability = 策略對最壞情況對手的表現；在大型遊戲中「難以計算，因為需要完整遍歷遊戲樹算 best response」；HUNL 無法精確計算 — [Lisý & Bowling, Equilibrium Approximation Quality (arXiv 1612.07547)](https://arxiv.org/pdf/1612.07547)
- Local Best Response (LBR) 用淺層搜尋 + 手工價值函數 + **對手私有狀態的明確推論**求 exploitability 的下界；2016 ACPC 的多個 bot（含第二、三名）比「每手棋都 fold」還更可被利用；頂尖程式間差距只有幾分之一盲注，但 exploitability 達數個盲注/手 — [arXiv 1612.07547](https://arxiv.org/pdf/1612.07547)；[Approximate exploitability, Timbers et al. (arXiv 2004.09677)](https://arxiv.org/pdf/2004.09677)
- Johanson & Waugh 2011 的 accelerated best response 讓 heads-up limit 的精確最壞情況評估可行 — [IJCAI 2011](https://www.cs.ualberta.ca/~games/poker/publications/ijcai2011_accelerated_best_response.pdf)；Davis, Burch & Bowling 2014 提出以一類「response functions」取代最壞對手來量測策略強度 — [AAAI 2014](https://www.cs.ualberta.ca/~games/poker/publications/14aaai-pgr.pdf)
- 有一件美國專利聲稱以賽局論方法量測撲克玩家相對技能，未經驗證 — [USPTO 8734219](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8734219)
- Chess.com 3+0 blitz，336,199,232 手、10,448,065 玩家側序列、599,094 玩家：conversion（取得優勢後以優勢結束）800–899 分 62.5%、2400–2499 分 66.2%、2800–2899 分 75.1%；meltdown（取得優勢後落敗）29.5%→19.8%→11.8%；blunder（≥100 cp）率 24.6%→7.3%；**局面狀態效應**：均勢 vs 優勢時 blunder 率 1200 分 14.7% vs 24.5%、2000 分 9.3% vs 19.6%、2800 分 4.4% vs 12.7%；該文**未**量測對對手 blunder 的懲罰、也未分析對手強度對失誤率的影響 — [Hunting for Context in 336 Million Blitz Moves](https://www.chess.com/blog/Jordi641/hunting-for-context-in-336-million-blitz-moves)
- Lichess 社群的 conversion rate 定義（達 +2 評估後獲勝的比例）會把「對手一手崩盤直接贏」也算進去，「不太算是轉換，更像對手失誤」 — [Lichess blog: Calculating your conversion rate](https://lichess.org/@/TotalNoob69/blog/calculating-your-conversion-rate/wPW57bju)
- 落後後翻盤機率也隨對手等級變化；面對較弱對手時落後後更可能贏 — [Coulombe, When should you resign](https://sites.google.com/view/patrick-coulombe-phd/articles/when-should-you-resign)
- Regan, Biswas & Zhou 2014（>3.5M 手）：(1) 每手平均誤差隨任一方優勢增大而一致上升，人類比電腦陡得多；(2) 在幾乎任何優劣程度下，**輪到人類走時其得分期望比輪到對手走時低 2–3%**，電腦幾乎無此效應；(3) 人類偏好把局面導向可行選項較少、較早決定的方向 — [Human and Computer Preferences at Chess (AAAI-14 workshop PDF)](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- 同文的統計方法：視同一局所有手為相關，目標最小樣本 41,000 手（≈1,200 局）給頻率 ±3% 的 2σ；兩獨立樣本差約 2% 才顯著 — [同上](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- Adhikari, Anatolyev & Dagaev（IEEE Trans. Games vol.16 no.2, 2024）：以頂尖棋手資料、Stockfish 評估、ordered probit，「小的不精確導致之後走法更不準，較嚴重的失誤有同方向但**較弱**的效應」；摘要未提對手失誤 — [作者頁](https://pages.nes.ru/sanatoly/Papers/TiltChess.htm)；[Crossref 10.1109/TG.2023.3275710](https://api.crossref.org/works/10.1109%2FTG.2023.3275710)；資料為 Wijk aan Zee 2011–2020 共 885 局（二手報導） — [Naked Science](https://naked-science.ru/article/column/sereznye-promahi-v-igre-zastavlyayut)
- Anderson, Kleinberg & Mullainathan 2016：數百萬局 + tablebase 真值，「局面本身難度」的特徵比技能或時間特徵「顯著更強」地預測失誤 — [arXiv 1606.04956](https://arxiv.org/abs/1606.04956)
- KataGo 生態：KaTrain 1.3 的 rank estimation 面板「僅依著手與 KataGo 形狀感的吻合程度估計，可能相當不準」；每 80 手一段估兩位玩家強度 — [KaTrain on PyPI](https://www.pypi.org/project/KaTrain/1.3.1/)
- OGS 論壇有人提議「rank vs loss per move」分析、也有人建議微調 KataGo 網路預測等級；另一討論串指出 KataGo 能給失誤點數但「連定義什麼是失誤都不容易」、無法分類成人類可理解的類別 — [OGS: Rank vs Loss per Move](https://forums.online-go.com/t/data-analysis-proposal-rank-vs-loss-per-move/28837)；[OGS: objectively measure differences between ranks](https://forums.online-go.com/t/has-anyone-tried-to-objectively-measure-differences-between-ranks-on-ogs/51889)
- JAIST 論文以前 150 手平均 loss 估等級，誤差約 1.5 級（僅見搜尋摘要，PDF 無法解析） — [JAIST paper](https://dspace.jaist.ac.jp/dspace/bitstream/10119/19809/5/paper.pdf)
- 「The cost of passing」：以 KataGo 量化每手價值，報告職業對殺型對局約 −0.6 分/手、業餘中級平和局約 −0.4 分/手的平均效應（搜尋摘要，脈絡未核實） — [arXiv 2208.12643](https://arxiv.org/pdf/2208.12643)
- 搜尋「KataGo 懲罰對手失誤頻率 vs 等級」沒有任何公開資料集或部落格統計。

### Inferences
- **撲克 exploitability 不可移植**：它定義在「策略」而非「觀察到的少量著手」上，且需要對手私有資訊分佈；圍棋是完全資訊，對應物就是 KataGo 的 point loss，本專案已用。
- **KataGo 版「懲罰效率」的連續定義**（避免 punish/keep/giveback 三分類丟資訊）：
  - 令 S_t 為第 t 手後、由自己視角的 score lead；對手第 t−1 手的贈分 g = max(0, S_{t-1} − S_{t-2})（對手失誤讓我多得的分；更嚴格可用 KataGo 對該局面最佳著手與實際著手的 score 差）。
  - 自己第 t 手的 point loss ℓ_t = S_t^{best} − S_t。
  - 懲罰效率 E_t = 1 − ℓ_t / (g + c)（c 為平滑常數），只在 g ≥ τ（例如 3 分）時計算；玩家特徵 = Σ_t (g_t − ℓ_t) / Σ_t g_t（以機會大小加權的「保留比例」），或 ℓ_t 對 g_t 的迴歸斜率（「贈分越大失誤越多」的 complacency 斜率，對應 chess.com 與 Regan 的「優勢時失誤率上升」）。
- **為何不是 policy likelihood 的單調函數**：rank-conditioned policy 給 p_r(a|s)，量的是「a 對 r 級棋手有多典型」；ℓ_t 量「a 兌現了多少局面價值」。(i) 兩者是 (s,a) 的不同泛函，低似然但零損失（非典型的正解）與高似然但大損失（典型的壞形）都常見；(ii) 對手 blunder 後的局面對人類棋譜分佈而言是尾端事件，各級 policy 在那裡的校準最差，likelihood 的訊號最弱，而 value/score head 不依賴人類分佈；(iii) likelihood sum 是對所有手的平均，懲罰效率是對「機會子集」的條件統計——chess.com 的 conversion 跨 2000 分僅差 12.6 個百分點，卻與 blunder 率（24.6%→7.3%）明顯不是同一條曲線，顯示條件統計有獨立變異。
- 但本專案試點（punish/keep/giveback 信度 0.07–0.40 @ 20 局）已說明：**群體層級單調 ≠ 個人層級可靠**。chess.com 之所以看到乾淨的曲線，是因為 10M 序列的群體平均；每位棋手每局只有少數幾次「對手大失誤」事件（事件數本身未見圍棋文獻量化，是本專案可從已存資料直接數出的量）。
- Regan 的「輪到人類走時得分期望低 2–3%」是一個真正的互動/tempo 現象（被迫出手是負擔），但它是群體效應，個人層級也會被 ℓ 的平均吸收。
- Anderson et al. 的結論（難度 > 技能）暗示：任何懲罰/失誤統計都必須先對「局面難度」殘差化，否則個人特徵大半在量「我這 20 局碰到了多難的局面」——而局面是雙方共同產物，這恰是雜訊來源而非訊號。

### Gaps
- 沒有找到任何圍棋（KataGo）層級的「對手失誤後反應 / 懲罰率 vs 等級」公開統計。
- 沒有找到西洋棋「對手 blunder 後的 conversion」按等級分層的研究（chess.com 文章的 conversion 不限於對手 blunder）。
- Adhikari et al. 全文未取得，不確定是否建模對手失誤對自己後續品質的影響。
- 「cost of passing」與 JAIST 論文的數字只來自搜尋摘要，脈絡未核實。

---

## 問題三：隱狀態 dyadic 模型——從一局的著手序列同時估兩個潛在技能；聯合推論勝過獨立推論嗎？差多少？

### Takeaway
沒有任何研究直接比較「聯合 vs 獨立」從著手推論雙方等級；現有最強的「從著手估等級」系統（Chen–Shih–Wu ICLR 2025 的 Bradley-Terry strength estimator、Neubauer 2024 的 KataGo trunk Set Transformer、Regan 的 IPR）都是**單邊**特徵，聯合只出現在以勝負為標籤的 TrueSkill/Glicko 類系統裡（本專案無勝負）。Maia-2 給出的間接上界：對手等級只改變 ≤6% 局面的預測著手（自身等級 22%），所以「對手條件化」能新增的資訊很有限；本專案用對手後驗平均等級 + 配對先驗，已經是這個聯合模型的一階近似。

### Cited Findings
- Regan & Haworth 2011 Intrinsic Performance Ratings：以著手品質（而非勝負）評技能，兩參數 s（敏感度）、c（一致性），對照 Elo 校準；用途包含整體與各類局面的技能評估、作弊案件 — [AAAI 2011](https://ojs.aaai.org/index.php/AAAI/article/view/7951)；[Regan 研究總覽](https://cse.buffalo.edu/~regan/chess/ChessResearchOverview.pdf)
- IPR 的 2σ 區間 vs 手數（Regan et al. 2014 表）：3,065–3,352 手 → ±48–51 Elo；4,212 手 → ±47；5,277 手 → ±43；9,474–9,489 手 → ±28–31；11,024 手 → ±21.5 — [AAAI-14 workshop PDF](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- Chen, Shih & Wu (ICLR 2025) Strength Estimator：神經網路對每個 (state, action) 輸出強度分數 β，λ = e^β（Bradley-Terry），一個等級的複合強度為 m=7 個樣本的幾何平均；預測時取查詢局的平均 β 與各已知等級的平均 β 最接近者。Go（FoxWeiqi，11 級：9d 到 3–5k，每級 45,000 局訓練、候選 100、查詢 900）：15 局達 ~80% 精確等級；±1 級容許下 1 局 ~80%、6 局 >99%；100 局 97.5%。西洋棋（Lichess 8 個 200 分段）26 局 >80%、100 局 93.38%。監督式基線 SLsum 100 局僅 ~49%。**論文未分析對手強度對估計的影響**，並把 opponent-specific strength scores 列為未來工作（BT 無法表達非傳遞性）；是否使用雙方著手不明 — [arXiv 2502.17109v2](https://arxiv.org/abs/2502.17109v2)；[Lab page + code](https://rlg.iis.sinica.edu.tw/papers/strength-estimator/)
- Neubauer 2024（TU Wien 碩論）：Set Transformer 吃 KataGo trunk 特徵，**只取「被評估玩家視角的每隔一手」**（即自己的著手），但 KataGo 輸入含前 5 手（含對手）資訊；訓練目標之一是 Bradley-Terry：模型對兩位對手各預測一個等級，再餵入 BT 預測勝負；500 手近期著手（約 4 局）即與 Glicko-2 相當（test set log-likelihood −0.58 vs −0.58）；雙方歷史 ≤4 局的展示集上 −0.55 vs Glicko-2 −0.64；只用最終輸出（winrate、points lead、policy、policy max、winrate loss、points loss）的 basic model 明顯較差（−0.61）；「每手 points loss 與強度相關，但單靠它遠不足以做出合格預測」 — [neubauer_24.pdf](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- Moudřík et al. 2015：KGS 10–50 局/人、26 級（20k–6d）各 120 人，bagged NN 10-fold CV RMSE：mean regression 7.503、patterns 2.788、ω-local（sente/gote）序列 5.765、border distance 5.818、captured stones 5.904、win/loss points 5.116、全部合併 2.659 — [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)
- Maia-2（NeurIPS 2024）：自身與對手等級各一個 embedding 串接、注入 attention query；改變自身等級最多改變 22% 局面的預測決策，改變對手等級最多 6%；移除 skill-aware attention 準確率 51.75% vs 52.74%（無單獨拆對手的 ablation）；作者寫道「理論上對手不應影響決策」但人類「有時會把對手等級考慮進去」 — [arXiv 2409.20553](https://arxiv.org/abs/2409.20553)
- Rating Your Moves（Chessable/Maia 團隊）：每 ply 轉成 64 層棋盤表示，標籤為**兩位玩家的平均等級**，逐手輸出等級分佈後貝氏更新；未描述分別估兩人 — [chess.com blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)
- CNN-LSTM 逐手估等級（含時鐘），自稱首個無手工特徵的逐手等級預測 — [ar5iv 2409.11506](https://ar5iv.labs.arxiv.org/html/2409.11506)
- 勝負式聯合技能模型：Score-based Bayesian Skill Learning 把 TrueSkill 延伸到比分，拆攻/守技能 — [Guo et al. ECML 2012](https://microsoft.com/en-us/research/wp-content/uploads/2012/01/sbsl_ecml2012.pdf)；技能估計的 state-space（filtering/smoothing）綜述 — [arXiv 2308.02414](https://arxiv.org/abs/2308.02414)
- 電競：PandaSkill 用個人統計估表現再餵 OpenSkill，無互動特徵 — [arXiv 2501.10049](https://arxiv.org/abs/2501.10049)；搜尋「skill estimation from interaction」無命中。

### Inferences
- **聯合推論的正確形式**：p(r_s, r_o | 棋譜) ∝ p(moves_s | r_s) p(moves_o | r_o) p(r_o | r_s)。對手著手只透過配對先驗 p(r_o | r_s) 影響 r_s 的邊際後驗。若配對均勻在 {−1, 0, +1} 級，則「對手等級」是自己等級的一個雜訊量測，SD = √(2/3) ≈ 0.82 級；n 局獨立對手平均後 SD ≈ 0.82/√n（5 局 0.37 級、20 局 0.18 級）——這是**可觀**的資訊，但本專案的「對手後驗平均等級」特徵已經線性地吃掉它；完整 dyadic 後驗只在 (a) 對手估計不確定性差異很大（需要用對手估計的方差加權）與 (b) 讓子局（r_o − r_s 由讓子數近乎已知）時才有二階增益，而讓子方向已是特徵。
- Maia-2 的 6% 是互動特徵的**自然上限**：如果連完整的對手等級資訊都只改動 6% 的著手，那麼任何從「雙方聯合行為」萃取、且與個人似然正交的訊號，其變異量級不會超過幾個百分點的著手——再除以 5–20 局的抽樣雜訊，期望增益落在本專案多數「不過關」實驗的區間。
- Regan 的 2σ 表給出一個尺度：IPR 式估計 2σ ≈ 2,300–2,900/√n Elo（n = 手數）。一位玩家 20 局 × ~120 手 ≈ 2,400 手 → ±50–60 Elo（西洋棋尺度）。互動特徵只用事件子集（每局數次），有效 n 小 10–20 倍，標準誤大 3–4.5 倍——這就是信度 0.07–0.40 的結構性原因。
- Neubauer 的 ablation 間接說明：KataGo **trunk** 比最終 heads 多出很多強度資訊；若本專案要加「對手相關」訊息，從 KataGo 內部表示（含前 5 手含對手著手的脈絡）學比手工 dyadic 統計更有望——但那是模型路線，不是特徵路線。

### Gaps
- 沒有找到任何「聯合 vs 獨立推論雙方等級」的量化比較（棋類或其他）。
- Chen et al. 是否用雙方著手、對手強度對其估計的敏感度未知。
- Neubauer 的 Bradley-Terry 目標是否實質上讓兩人估計互相修正，論文未單獨 ablation。

---

## 問題四：節奏/主動權特徵——先手/後手（sente/gote）記帳、劫爭、交換（trades）有無按等級量化的文獻？會不會已被 policy likelihood 吸收？

### Takeaway
唯一的實證是 Moudřík et al. 2015 的 ω-local sente/gote 序列統計：單獨用它 RMSE 7.50→5.77（1.30×），遠弱於 pattern 特徵（2.79），且其定義（局部性 + 誰結束序列）本質上就是本專案已死的 response geometry；劫爭、交換頻率 vs 等級沒有任何文獻。sente/gote 在理論上（miai counting）是局面價值的函數，KataGo score lead 已包含，policy likelihood 也會吸收「典型的先手/後手選擇」，剩下的只有「以 KataGo 價值加權的脫先/應手決策品質」，那又回到 point loss。

### Cited Findings
- Moudřík et al. 2015 的 ω-local sente/gote 序列：若一手與前一手的 gridcular 距離 < ω（強度資料集 ω=10），視為局部；把每局切成不相交的局部序列，若開始者 ≠ 結束者則對開始者為 sente（他先去別處/tenuki），否則為 gote；特徵 = 每局平均 sente、gote 序列數；「判定哪些手是 sente/gote 很難，所以我們限制在 ω-local」；該特徵單獨 RMSE 5.765 ± 0.003（mean regression 7.503，patterns 2.788，全部 2.659） — [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)
- 同文術語：sente = 需要對方立即回應的一手；gote = 不需要；tenuki = 不理會對方上一手（gote）而取得主動權 — [同上](https://arxiv.org/abs/1512.08969)
- 官子計數理論：deiri counting 比較先走的差值再標 sente/gote，需要「sente 值雙倍」之類規則；miai counting 把所有手放在同一尺度；sente 的 miai 值為 0，因基準假設雙方都走自己的 sente；double sente 的 miai 值無定義（除以零） — [dfan.org: area endgame counting part 2, sente](https://dfan.org/blog/posts/2025/08/13/area-endgame-counting-2/)；[Sensei's Library 討論](https://senseis.xmp.net/?topic=366)；[L19 miai values thread](https://lifein19x19.com/viewtopic.php?p=144390)
- 劫爭：Wikipedia 僅描述劫的結束方式，無頻率；L19 論壇有「通常較強者從劫中獲利」的個人觀察，非統計 — [Wikipedia Ko fight](https://en.wikipedia.org/wiki/Ko_fight)；[L19 thread](https://www.lifein19x19.com/viewtopic.php?p=128494)
- Regan et al. 2014 的「輪到走的一方得分期望低 2–3%」是跨棋類最接近「tempo 作為負擔」的量化證據（西洋棋，人類獨有） — [AAAI-14 workshop PDF](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)

### Inferences
- Moudřík 的 patterns（3×3 到更大的局部形狀頻率）是 policy likelihood 的粗糙祖先；在它存在下，sente/gote 等所有非 pattern 特徵合計只再降 RMSE 約 4.6%（2.788→2.659）。本專案的 rank-conditioned policy 遠強於 pattern 計數，吸收力更大；加上 ω-local 定義 ≡ 本專案已死的距離/脫先統計，**局部性版 sente/gote 可直接判死**。
- 若要做「價值版」主動權記帳：對每個局部序列結束點，用 KataGo score lead 比較「脫先 vs 繼續應」的分差（即脫先那一手的 point loss 是否為負——真正的 sente 判定）；玩家特徵 = 錯誤脫先率（脫先導致 ≥τ 分損失）與錯誤應手率（該脫先卻應了，損失 ≥τ）。這是 point loss 的**子集條件版**，與平均 loss 的差異在於「哪一類決策出錯」；可用性取決於每局此類決策點數（估計每局 10–30 個局部序列邊界，比 blunder 事件多），信度應優於 punish 三分類，但仍需試點。
- 劫爭/交換：圍棋規則下劫可由 SGF 直接偵測（提一子後對方不可立刻回提）；但劫爭「頻率」是風格與局面產物，與等級的關係沒有證據，且 KataGo 已把劫的價值算進 score lead。預期：若有訊號會是「劫爭中的 point loss」而非次數，又回到 loss。

### Gaps
- 沒有任何按等級分層的 sente/gote、劫爭、交換頻率統計（KGS/OGS/Fox 皆無公開分析）。
- 沒有文獻檢驗「sente/gote 統計在 policy likelihood 存在下的增量」。

---

## 問題五：風險評估——互動量測需要多大樣本？哪些定義在 100–150 手/局、5–20 局下可能可靠？

### Takeaway
以 Spearman–Brown 推算，本專案試點的分半信度 0.07–0.40（@20 局）要達到 0.7 需 70–620 局；在 5 局時掉到 0.02–0.14，與等級的最大可能相關（√信度）只剩 0.13–0.37，再扣掉與似然特徵的重疊，增量接近零。Granger/TE 在單局 100–150 ply 下嚴重欠檢定力（一個 VAR 設計 T=100 檢定力 21%、T=1000 才 98%），必須跨局 pooling 到 ~1,000–2,400 ply 並做 effective-TE 偏差校正。唯一有機會的設計是：**用到每一手**（而非事件子集）、**連續值**（而非三分類）、**對局面難度/優勢殘差化**、**以 KataGo 期望值做 shrinkage** 的 2×2 VAR 係數或 complacency 斜率；它們可以從已存的 per-move score lead 以 CPU 直接算，預期增益小（LB +0.00x 級），風險高。

### Cited Findings
- Spearman–Brown：r' = k r / (1 + (k−1) r)；反解 k = r*(1−r)/(r(1−r*))；假設新增項目與原項目平行，否則高估 — [Wikipedia](https://en.wikipedia.org/wiki/Spearman%E2%80%93Brown_prediction_formula)
- Granger 檢定力：某高維 VAR 設計 T=100 時檢定力約 21%、T=1000 約 98%，size 接近名目；非線性因果檢定在 n=500 檢定力才接近 1 — [arXiv 2302.01434](https://arxiv.org/pdf/2302.01434)；[arXiv 1703.01102](https://arxiv.org/pdf/1703.01102)
- Transfer entropy 的小樣本偏差「通常存在」、依估計器而異；標準校正為 effective TE（減去 shuffled source 的 TE，多次 shuffle 平均）；高階 embedding 下整段 shuffle 無效，需 Markov block bootstrap；建議 bins 與歷史長度都小以避免零計數 — [RTransferEntropy vignette](https://archive.linux.duke.edu/cran/web/packages/RTransferEntropy/vignettes/transfer-entropy.html)；[infomeasure docs](https://infomeasure.readthedocs.io/en/0.6.0/guide/transfer_entropy/)
- Regan et al. 2014：同一局內各手視為相關；41,000 手 ≈ 1,200 局給頻率 ±3% 2σ；IPR 2σ 在 3,000–5,000 手為 ±43–51 Elo、11,000 手為 ±21.5 — [AAAI-14 workshop PDF](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)
- Chess.com：優勢 vs 均勢 blunder 率比值隨等級擴大（1200：24.5/14.7 = 1.67；2000：19.6/9.3 = 2.1；2800：12.7/4.4 = 2.9），顯示「狀態條件化的失誤比」本身帶等級訊號（群體層級） — [chess.com 336M](https://www.chess.com/blog/Jordi641/hunting-for-context-in-336-million-blitz-moves)
- Anderson et al.：局面難度對失誤的預測力 > 技能、時間 — [arXiv 1606.04956](https://arxiv.org/abs/1606.04956)
- Adhikari et al.：自身小失誤 → 後續品質下降（own→own 持續性存在） — [TiltChess](https://pages.nes.ru/sanatoly/Papers/TiltChess.htm)
- Chen et al.：±1 級容許下**單局**即 ~80%、6 局 >99%（FoxWeiqi 11 級）；表示單邊著手強度訊號本來就很強，互動特徵要在這之上增量 — [arXiv 2502.17109v2](https://arxiv.org/abs/2502.17109v2)

### Inferences
- **Spearman–Brown 推算（本專案數字）**：
  - r₂₀ = 0.07 → 0.7 信度需 k = 31 → 620 局；r₂₀ = 0.40 → k = 3.5 → 70 局；達 0.5 需 30 局。
  - 5 局（k = 0.25）：r = 0.40 → 0.14；r = 0.07 → 0.018。最大可能與等級的相關 √r：0.37 / 0.13。
  - 平均每人約 12 局時，0.40 的特徵信度約 0.29——意味著該特徵至少 70% 是雜訊，進入樹模型後主要貢獻過擬合。
- **有效樣本的結構性差距**：似然特徵每人用 ~600–2,400 手；事件型互動特徵（對手 blunder 後反應）每局僅數個事件，每人 15–120 個。標準誤比例 √(2,400/120) ≈ 4.5 倍。要可靠，互動量測必須設計成「每一手都貢獻」。
- **候選定義（全部可從已存的逐手 score lead / point loss 以 CPU 計算；若只存 log p 則需以 value/score head 重新評分，GPU 成本 ≈ 一次完整 KataGo 分析）**：
  - **D1 Dyadic VAR(1) / context matrix**：對每位玩家 pooling 全部局，令 x_t = 對手上一手 point loss（或 −log p 驚訝度）、y_t = 自己這手 point loss；擬合 y_t = a + b·y_{t−2}（自身持續，Adhikari 效應）+ c·x_{t−1}（opp→self）+ 局面控制（|score lead|、move number、KataGo 局面熵或 top-1 policy 做難度）。特徵 = c（契約：對手失誤是否牽動我，包含「失誤後我更會錯」與「我把對手贈分吐回去」兩種符號）與 b。非冗餘性：是條件斜率而非水準，與 Σ log p 不同泛函。信度：未知；以每人 ~1,000–2,400 手、兩個係數估計，比事件型特徵多 10× 樣本，但係數本身是二階量，期望信度仍低（猜測 0.2–0.4 @20 局）。成本：CPU 分鐘級。風險：高；Maia-2 的 6% 上限壓低期望增益。
  - **D2 Complacency 斜率**：自己 point loss 對「自己當前 score lead」的斜率（優勢越大越鬆懈），控制難度。chess.com 顯示此比值隨等級由 1.67→2.9（群體）。非冗餘性：平均 loss 已在特徵中，但「loss 對狀態的斜率」不是；狀態是雙方共同產物（互動的最小形式）。信度：用到每一手，應優於事件型；但局內 score lead 自相關高，有效 n 遠小於手數。成本：CPU。風險：中高；注意與「勝率軌跡統計」（已死）部分重疊。
  - **D3 以 KataGo 期望做 shrinkage 的連續懲罰效率**：Σ(g_t − ℓ_t)/Σ g_t（保留對手贈分比例，g ≥ τ），再對 KataGo 在同局面下的「期望保留率」（例如以該局面 top-k policy 的加權 loss 作為難度基準）殘差化，並對事件數做 empirical-Bayes 收縮。非冗餘性：見問題二。信度：事件型，預期與已死的 punish 三分類同量級（0.1–0.4），連續化與收縮可能略升但不改量級。成本：CPU（需逐手 best-move score；若只存 played-move loss 則需重評分）。風險：高——本專案已有負結果。
  - **D4 價值版 sente/gote 決策錯誤率**：在局部序列邊界上，脫先造成 ≥τ 損失的比例、該脫先卻應的比例（以 KataGo loss 判定）。非冗餘性：是 loss 的「決策類型」分解；信度：決策點每局 10–30 個，優於 blunder 事件；風險：中高，與 response geometry 的局部性定義共用偵測器。
  - **不建議**：CRQA/complexity matching 於座標序列（≡ 已死的 geometry）、單局 Granger/TE（T=100–150 無檢定力）、劫爭/交換次數（無文獻、KataGo 已計價）、exploitability（不可計算）。
- **先驗期望**：所有候選的「群體層級」訊號在西洋棋文獻中都存在；個人層級在 5–20 局下，最樂觀的 D1/D2 預期增益為 CV +0.00x（與本專案多數失敗實驗同級），且會與 fused_opp_diff 的「自減對手」特徵部分共線。建議若要做，先只做 D1 與 D2 的**分半信度與 incremental R²（對現有特徵殘差）** 試點，門檻：信度 ≥ 0.5 @ 20 局且殘差相關 ≥ 0.1，否則不進 pipeline。

### Gaps
- 沒有任何研究報告「互動量測 vs 技能」的 per-subject 信度或所需樣本量；上面的推算全部是 Spearman–Brown 外推加本專案數字。
- 圍棋每局「對手 ≥τ 分失誤」的事件數分佈未見文獻（可直接從專案資料統計）。
- Lee et al. 2025 context matrix 在短序列下的恢復精度未知（摘要只說「在有雜訊的模擬中準確恢復」，未給 T）。
- 沒有找到西洋棋/圍棋上「opp→self 失誤傳染」的實證（Adhikari 只做 own→own）。
