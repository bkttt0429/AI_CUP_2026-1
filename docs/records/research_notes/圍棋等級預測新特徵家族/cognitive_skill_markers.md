# 認知科學與專長文獻中的「可觀察棋力標記」：從棋譜座標推導新特徵家族

研究日期：2026-11-04。範圍：圍棋（及作為對照的西洋棋）不同強度業餘棋手在**可觀察著手行為**上的差異，並篩出僅靠 SGF 著手 + 顏色 + 讓子標記 + 規則引擎即可計算、且不只是「rank-conditioned policy 下該著手的機率」的候選特徵。專案中已使用（rank-conditioned policy log-likelihood、fine-tuned 10-rank heads、KataGo trunk 向量序列模型、讓子/對手特徵）與已測死（每手品質時間序列統計、應手幾何、跨局風格一致性、終局價值代理、強引擎目損）的項目，下文一律標為「已用／已死」不再重提，只在判斷冗餘性時引用。

> 證據等級標示：**[實證]** 有數據的研究或大樣本分析；**[理論]** 認知理論推論；**[教學假說]** 強者教學／論壇經驗談，只能當假說。

---

## 關鍵問題 1：專長文獻說 kyu、低段、高段在錯誤「類型／位置／階段／應對威脅／劫／形／收官／認輸／棋局長度」上有何可觀察差異？

### Takeaway
認知科學的圍棋專長文獻（Reitman、Burmeister、Masunaga & Horn、眼動研究）只確立了「高手以重疊的空間 chunk 感知全盤、演繹推理（讀棋）與專長工作記憶是與一般流體智力分離的因子」這類**機制層**結論，幾乎沒有以段級分層、計數「錯誤類型」的實證；真正有數字的錯誤分層幾乎全來自西洋棋大數據（Anderson et al. 2016、Maia/Maia-2、chess.com 3.36 億手分析），而圍棋端最接近的是 Moudřík/Baudiš 的 KGS 特徵工程。「kyu 錯在讀棋與連接、SDK/低段錯在收官型著手與活棋判斷、高段錯在方向與細微收官」是教學界的共識假說，但尚無人用引擎對業餘棋譜按段級統計過。

### Cited Findings

**圍棋認知／專長文獻（機制層）**
- Reitman (1976) 以一位初學者與一位 Go Master 做記憶與知覺任務，發現無法用單一 inter-response time 切出一致的 chunk；大師描述的模式「不是線性 chunk 也不是嚴格巢狀階層，而是重疊的叢集（overlapping clusters）」 — [Reitman 1976, Cognitive Psychology 8(3)](https://deepblue.lib.umich.edu/items/0e4cd16a-0741-43a1-b1d4-55a10b6cda00)
- Burmeister, Saito, Yoshikawa & Wiles (2000) 以日本棋手重做記憶實驗，目標是研究「棋手如何使用資訊、其圍棋知識如何結構化」，並指出圍棋心理學研究遠少於西洋棋 — [Memory Performance of Master Go Players (UQ)](https://staff.itee.uq.edu.au/janetw/Computer%20Go/go-master.pdf)；Burmeister 博士論文 *Studies in Human and Computer Go* (2000) — [dblp](https://dblp.dagstuhl.de/rec/phd/au/Burmeister00.html)
- Masunaga & Horn (2001, *Psychology and Aging*)：263 名男性圍棋棋手、48 個專長等級、18–78 歲；「專長演繹推理（expertise deductive reasoning）」與「專長工作記憶」是與一般流體推理、短期記憶分離的因子；年齡衰退隨專長提高而縮小至零 — [期刊索引紀錄](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/84524905)；理論版 Masunaga & Horn (2000) — [索引](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/81601295)
- Gobet 的 template theory 將 Chase & Simon 的 chunk 擴充為帶插槽的 template，用以解釋專家快速編碼；圍棋（Reitman 1976）與五子棋（Eisenstadt & Kareev 1977）被列為「有意義材料的專家記憶優勢」已複製的領域之一 — [Gobet, A pattern-recognition theory of search in expert problem solving](https://www.bohrium.com/paper-details/a-pattern-recognition-theory-of-search-in-expert-problem-solving/812750792124530688-10156)；[Gobet 相關全文 (core.ac.uk，未逐頁核對)](https://core.ac.uk/download/333533.pdf)
- 眼動：台師大碩士論文（2017）22 位專家 vs 22 位新手，專家注視時間較短、更集中於任務相關區域、更早到達關鍵區 — [NTNU 論文](https://etds.lib.ntnu.edu.tw/thesis/detail/2bfc4fcb32b5de2210471073a4685dc3)；日本 IPSJ 研究：新手只看局部，專家掃視全盤以辨識棋塊位置，歸因於圍棋專家仰賴「空間 chunk」（將棋專家則仰賴手順的「時間 chunk」） — [IPSJ 紀錄](https://ipsj.ixsq.nii.ac.jp/records/58517/export/bibtex)
- 日本 fMRI 研究比較業餘與職業棋手判斷歷程（Ouchi et al. 2005, "Neural substrates in judgment process while playing go: a comparison of amateurs with professionals"）— [PubMed 15820625](https://pubmed.ncbi.nlm.nih.gov/15820625/)（本次只取得標題，摘要未能讀取）

**西洋棋大數據：錯誤率 vs 等級、位置難度、階段**
- Anderson, Kleinberg & Mullainathan (2016)：GM 與 FICS 資料中 blunder rate 隨 rating 平滑下降，高分段趨平；單用局面難度（blunder potential）預測是否失誤準確率 0.73（全套難度特徵 0.75），單用棋力 0.55、單用時間 0.53（隨機 0.50）；固定局面後 rating 單獨可達 0.62；blunder potential 增加 0.2 的效應大於 600 分 rating 差；固定難度下不同 rating 的曲線在 log 尺度近乎平行；存在「skill-neutral」局面（各級失誤率相同）與「skill-anomalous」局面（高手反而失誤更多） — [arXiv 1606.04956 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1606.04956)
- Maia (McIlroy-Young et al. 2020)：各 rating 專屬模型在自身段位預測準確率最高（1100 模型 50.8%、1500 模型 52.2%、1900 模型 52.9%），離目標段位越遠越低（1900 模型預測 1100 棋手僅 46%）；預測準確率隨著手品質上升，blunder 最難預測；Maia 預測「占據一個獨特的子空間」而非只是較弱的 Stockfish；集體 blunder 位置幾乎都在開局與殘局（中局局面不重複） — [arXiv 2006.01855 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.01855)
- Maia-2 (Tang et al., NeurIPS 2024)：線性探針顯示「全盤評估、子力總值」等概念是 skill-dependent，「是否擁有雙象、能否吃后」等靠近規則的概念是 skill-independent；27% 的局面為 monotonic（正解機率隨棋力單調上升；Maia-1 僅 1%）；改變「自己」的棋力輸入會改變 22% 的預測著手，改變「對手」棋力只影響 6%；Mate-in-1 例子中正解機率由 0.22 (1100) 升至 0.45 (1900) — [arXiv 2409.20553v2](https://arxiv.org/html/2409.20553v2)
- chess.com 3.36 億手 blitz 分析：blunder（≥100cp）率 800–899 分 24.6% → 2800–2899 分 7.3%；平均 cp 損 85.6 → 29.2；第 1–10 手 blunder 率僅 6.6%（平均損 28.8）；**領先時 blunder 率顯著高於均勢**（1200 分：均勢 14.7% vs 領先 24.5%；2800 分：4.4% vs 12.7%）；「轉換率 / 崩盤率」800 分 62.5% / 29.5%，2800 分 75.1% / 11.8%；每方著手數隨等級上升（800 分 24.9 手 → 2800 分 42.9 手） — [Hunting for Context in 336 Million Blitz Moves](https://www.chess.com/blog/Jordi641/hunting-for-context-in-336-million-blitz-moves)
- ChessBase 2015 GM blunder 統計與 lichess 論壇皆報告 blunder 數隨等級急降（質性） — [ChessBase](https://en.chessbase.com/newsroom/post/grandmaster-blunders-a-statistical-analysis?page=0)

**圍棋教學／論壇對「各段級典型錯誤」的描述（教學假說，非證據）**
- Life in 19x19「Which are the worst mistakes?」：RobertJasiek（職業棋士級作者）— 純初學者「無法視覺化著手」；DDK「不走簡明、不連接重要棋子」；SDK 與低段「失誤太多、因收官型著手輸掉的目數超乎想像」；業餘高段「戰術讀棋速度與半目精度的收官計算速度都慢於職業」。jlt（2k）— 誤判棋塊強弱、不讀棋就下形狀手；「小錯累積超過大錯」。John Fairbairn — 業餘棋手因下得太快太不專注至少損失 2–3 級；「業餘 suji 差、職業 suji 好」。kvasir（5d）— 強者多因「邏輯錯誤、盲點」輸棋；「錯過大官子在細棋就是終局」；SDK 自認的三大錯：gote、沒走 AI 著手、不夠積極（他懷疑是感知偏差）。Knotwilg（2d）— 錯誤分四層：注意、讀棋、策略、技術；「不讀棋是最難治的壞習慣」 — [lifein19x19 p=276919](https://lifein19x19.com/viewtopic.php?p=276919)
- Life in 19x19「What separates 10 kyu from 5 kyu and 5 kyu from 1 dan?」：Kirby — 「讀棋能力」；MinjaeKim — 5k 有形感但讀棋問題多，1d 會走「看起來夠好」的手但漏細節；Solomon — 關鍵是效率，弱者每子得點少、4k 的做活方式「沒有段位棋手會這樣做」、10k 太早走一線收官而失分；Uberdude — 5k→1d 的關鍵是「理解弱棋塊（weak groups）的負債與如何攻擊它」；Jarmo — 新手攻得太急想殺光一切，之後學會先守；10k→5k 學會「不再跟著對手走、搶先手」，5k 以上才學「何時該後手」；Knotwilg — 10k 以下會「在自己地裡下無意義的手」，段位棋手仍會「弄錯方向」；bayu — 5k 之後對手錯誤更細微、大逆轉變少；Bonobo 轉述 — 10k「真的很笨的錯」、5k「有意義的錯」、段位「很漂亮的錯」 — [lifein19x19 p=203141](https://www.lifein19x19.com/viewtopic.php?p=203141)
- Sensei's Library 討論串：有人觀察 kyu 棋手「在有疑慮時脫先到別處，約 99% 的情況導致災難」 — [senseis.xmp.net topic 3452](https://senseis.xmp.net/?topic=3452)
- 教學課程分層：GoMagic《Typical Mistakes》針對 17–10k，《Typical Mistakes 2.0》針對 10–1k（「秘密困擾大多數 kyu 棋手」） — [GoMagic 1](https://gomagic.org/courses/typical-mistakes)、[GoMagic 2](https://gomagic.org/courses/typical-mistakes-2)；Moffatt《Go by Example: Correcting common mistakes in DDK play》(20k–10k) — [AbeBooks](https://www.abebooks.com/9781453851258/Example-Correcting-Common-Mistakes-Double-1453851259/plp)

### Inferences
- 認知文獻給的可操作線索是**「讀棋（演繹推理）」與「全盤 vs 局部的注意分配」**是區分等級的主軸，而非具體錯誤類型。對應到棋譜，可觀察的「讀棋失敗」是**有結果可驗證的戰術事件**（征子逃不掉、對殺氣數佔優卻輸、被叫吃後棋子仍被提、死活該活卻死），而「全盤 vs 局部」對應**脫先／應手幾何**（專案已測死）。
- Anderson et al. 的核心發現（局面難度 >> 棋力）直接解釋為何「每手品質時間序列」在專案中死掉：每手損失主要由局面決定，玩家只貢獻 log 尺度上的平行位移。這意味著**任何不先條件化局面難度的「品質統計」都注定高噪聲**；rank-conditioned policy LL 本質上已做了這個條件化，所以新特徵要麼是**不同維度的事件計數**（類別型），要麼是**分布形狀**（Q3）。
- chess.com 的「領先時 blunder 率翻倍、低分段崩盤率 2.5 倍」是「overplay/鬆懈」在大數據中的實證形式；Maia-2 「全盤評估是 skill-dependent、規則近端概念是 skill-independent」暗示**全盤判斷（厚薄、弱棋塊數）比局部手段更能分段**，與 Uberdude「理解弱棋塊」假說一致。
- 教學假說的分層版本（供設計特徵用，無實證）：DDK：自殺手、征子誤讀、不接、在自己地內落子；SDK：過早一線收官、不讀棋的形狀手、追著對手走（後手過多）、做活效率差；低段：收官型著手於中盤、方向錯；高段：半目收官計算、tedomari。

### Gaps
- 未找到任何以引擎對業餘棋譜按 kyu/dan 分層統計「錯誤類型」的研究（搜尋 KataGo/Leela 業餘分層分析、中文「業餘棋手 棋力 認知 錯誤類型」皆無結果；中文學術庫未能存取）。
- Masunaga & Horn、Burmeister 全文未取得，只有摘要與二手引用；Ouchi 2005 僅標題。
- 「劫的處理」「tedomari / 先後手排序精度」「填自己眼位」「過早收官」等皆無任何分層數據，只有教學假說。

---

## 關鍵問題 2：哪些差異可由著手座標 + 規則引擎操作化，且不等於「rank-conditioned policy 下該手的機率」？

### Takeaway
圍棋端唯一可直接對照的實證是 Moudřík/Baudiš/Neruda 的 KGS 特徵工程：在 26 個等級（20k–6d）、每人 10–50 局（與本題 5–20 局極相近）的設定下，**模式（pattern）特徵壓倒性最強（RMSE 2.79 級 vs 基準 7.50），而「先手/後手局部序列」「邊線距離×階段」「提子數×階段」各只貢獻 1.27–1.30 倍改善，且全部加起來只再多 0.13 級**；這是「類別型著手統計相對 policy 型特徵只有小幅增量」的最強先驗。可由規則引擎計算且文獻未測、理論上與 policy LL 正交的家族是：自殺手／被叫吃不應而被提、征子與對殺的「有結果驗證」誤讀、劫爭事件、弱棋塊同時存在數、終局型態（是否虛手、填眼/地內落子），以及分布形狀（見 Q3）。

### Cited Findings

**圍棋：以著手衍生特徵預測等級的實證**
- Moudřík, Baudiš & Neruda (2015, IEEE CIG)：KGS 檔案 >100,000 局，6d–20k 共 26 級、每級 120 人，只用 19 路、無讓子、對手強度相當；每人棋局 <10 局丟棄、>50 局隨機降抽至 10–50 局。特徵：(a) Patterns：最頻繁 1000 個空間模式（gridcular size 2–6）的計數，含雙方 atari 旗標；(b) ω-local sente/gote sequences：與前一手 gridcular 距離 <ω(=10) 視為局部，將棋局切成連續局部序列，由對手結束者為發起者的先手、自己最後應者為後手，特徵=每局平均先手數與後手數（2 維）；(c) Border distance：4×4 直方圖（1–2 線/3 線/4 線/5 線以上 × 第 1–10/11–64/65–200/201+ 手）；(d) Captured stones：3×3 直方圖（第 1–60/61–240/241+ 手 × 自己提子/對手提子/差）；(e) Win/Loss：勝負×計點/認輸計數 + 計點局平均目差。等級 RMSE（基準 mean regression 7.503）：Patterns 2.788（2.69×）、ω-local 5.765（1.30×）、Border 5.818（1.29×）、Captured 5.904（1.27×）、Win/Loss 計數 6.792（1.10×）、Win/Loss 目差 5.116（1.47×）、全部 2.659（2.82×） — [arXiv 1512.08969 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1512.08969)
- 同文風格預測（GoGoD 25 位職業、5 位專家評 1–10 分）：Patterns 1.691（1.28×）、Border 1.723（1.26×）、ω-local 2.115（1.03×）、Captured 2.223（0.98×，無效）、全部 1.615；分項：領地性 1.52×、正統性 1.37×、**攻擊性 1.41×**、厚度 1.06×（專家意見變異太低） — 同上
- Baudiš & Moudřík (2012)：每手特徵 = 提子旗標、叫吃旗標、逃叫吃旗標、與前一手／前兩手的相鄰性、邊線距離（≤4）、gridcular 半徑 2–9 的空間模式；Go Teaching Ladder 7,700 局（30k–4d）上 PCA 第一主成分與等級相關 r=0.979；k-NN 以 85 局誤差 σ=2.35 級、9 局 σ=4.62、2 局 σ=7.23（隨機 14.44） — [arXiv 1209.5251 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1209.5251)
- Moudřík & Neruda (2016, TPNC)：OGS 188,700 局訓練 CNN 分三個棋力類別，單一局面 71.5%、整局聚合 77.9% — [Springer LNCS 10071](https://link.springer.com/10.1007/978-3-319-49001-4_15)
- Maddison et al. (2014) 的 CNN 輸入含 9 個等級平面（one-hot of player rank）、征子可提旗標與提子數特徵 — [arXiv 1412.6564](https://arxiv.org/pdf/1412.6564)
- Egri-Nagy & Törmänen (2020)：定義每手效果 δ(a)=μ(s_{i+1})−μ(s_i)（引擎分數估計差）作為逐手表現剖面；KGS 樣本 915 局**全為 4d 以上**，未做等級掃描 — [arXiv 2009.01606 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2009.01606)
- OGS 論壇「Rank vs Loss per Move」提案：KaTrain 貢獻者 sanderl 稱以 AI 評審估等級「平均合理但誤差很大，不建議做評級依據」；提案者預期「戰鬥型風格 LpM 較高、領先者會簡化局面」 — [OGS forums 28837](https://forums.online-go.com/t/data-analysis-proposal-rank-vs-loss-per-move/28837)
- 商用工具已採「依等級調整錯誤門檻」：SHAPE 專案稱 AI 回饋依使用者等級標記「該等級相關的錯誤」 — [goshape PyPI](https://pypi.org/project/goshape/)；OGS Move-Type chart 只分 excellent→blunder 品質類別，無按等級統計 — [OGS forums 59148](https://forums.online-go.com/t/move-type-chart/59148)

**終局行為（認輸 / 計點 / 棋局長度）的伺服器數據**
- OGS 排位 19 路棋局（Lys 分析，bulk SGF）：kyu 區內**等級越高認輸比例越高**，認輸占比約 40%–<60%；段位區維持約 50%，「計點似乎逐漸消失」；高段樣本過小；全站認輸占比逐年上升 — [OGS forums 52977 p.2](https://forums.online-go.com/t/when-do-people-usually-resign-some-statistics/52977?page=2)
- OGS 236 位線上使用者各取最近一局認輸敗局（2024-09）：平均認輸手數 168.19、平均落後 63.17 目（KataGo）、5.93% 為領先時誤認輸；5k 平均約第 175 手認輸；樣本多為 SDK — [OGS forums 52977](https://forums.online-go.com/t/when-do-people-usually-resign-some-statistics/52977)
- OGS 10 萬局樣本：19 路 47,602 局中僅 11,803 局（≈25%）走到計點；平均 255 手；有人推測「強者較早認輸所以較少走到計點」但未檢驗 — [OGS forums 35042](https://forums.online-go.com/t/average-game-length-on-different-board-sizes/35042)；2,700 萬局樣本中 2,632,713 局無讓子計點 19 路平均 253.5 手（去除最後兩手虛手），AGA 規則「白最後虛手」造成偶數手尖峰 — [OGS forums 39850](https://forums.online-go.com/t/statistics-from-a-27m-game-sample/39850)
- KGS 約 43,000 局分手合（2005 起）：中國規則貼目下認輸率 79%，6.5 目貼目 68%（規則代理，非等級） — [lifein19x19 p=210911](https://www.lifein19x19.com/viewtopic.php?p=210911)
- Kombilo 棋局長度直方圖呈雙峰，疑為早期認輸造成，未按等級拆分 — [lifein19x19 p=142719](https://lifein19x19.com/viewtopic.php?p=142719)
- 職業棋局有單局 64 次劫提（1998 年 364 手棋局）的紀錄；日韓棋譜常省略末段著手，影響提子計數與平均長度 — [lifein19x19 p=179039](https://www.lifein19x19.com/viewtopic.php?p=179039)

### Inferences — 候選特徵家族評估表

每項：定義 / 規則引擎算法 / 等級效應（文獻或假說）/ 相對 policy LL 的增量 / 成本 / 風險 / 冗餘旗標。「policy LL」指專案已有的 humanv0 18 檔 + 10-rank heads 的 log-likelihood。

**F1. 自殺手與「被叫吃不應而被提」計數（attention / reading failure）**
- 定義：(a) self-atari：己方著手使所含棋塊只剩 1 氣且未提子、非劫、棋塊 ≥2 子（排除撲、倒撲等刻意送吃）；(b) atari-ignored-loss：對手叫吃後，己方下一手未逃/未反提，且該棋塊隨後被提；記錄被提子數。
- 算法：棋塊/氣數追蹤即可；排除法：若送吃後 3 手內己方提回更多子則視為手段。
- 等級效應：教學假說 DDK 最高、SDK 快速下降、段位趨零（Jasiek「不連接重要棋子」、Fairbairn「不專注」）。無分層數據。
- 增量：中低。policy LL 對這類手已給極低機率，但 LL **取平均**會把罕見極端事件稀釋；類別計數保留尾部事件。chess 證據（Maia：blunder 最難預測）支持「極端錯誤是 policy 最弱的部分」。
- 成本低；風險：刻意送吃誤判；每題 5–20 局計數為 Poisson 小樣本，需以每百手率 + 收縮估計。冗餘：部分。

**F2. 有結果驗證的戰術誤讀：征子 / 對殺 / 死活**
- 定義：(a) ladder-fail：己方 1 氣棋塊連續延氣 ≥3 手後被提（逃失敗的征子）；(b) ladder-chase-fail：己方連續叫吃對手 ≥3 手後對手逃出（追失敗的征子）；(c) semeai-lost-with-liberty-edge：兩相鄰異色不活棋塊對殺，氣數（含共氣修正）多者卻被提；(d) 死活：己方棋塊在終局被判死（需所有權估計；專案已有 KataGo，可用 ownership，或 Benson 無條件活 + 簡易眼形判定）且其最大尺寸。
- 算法：標準征子讀取器（GNU Go/Pachi 皆有）、氣數計數、Benson 演算法。
- 等級效應：Masunaga & Horn 的「專長演繹推理」與論壇共識「讀棋是 5k→1d 分水嶺」對應此處；bayu「5k 後大逆轉變少」暗示大塊死亡率隨等級下降。無數字。
- 增量：中。policy LL 評的是「此手是否典型」，不評「此手是否被後續結果證實錯誤」；結果驗證是另一個維度。但事件稀少。
- 成本中（征子讀取需實作）；風險：樣本小、對手強度混淆（對手也在同等級，已有對手特徵可調）。冗餘：低。

**F3. 劫爭事件統計**
- 定義：每局劫形出現次數、劫提次數、劫材數（劫中不解消劫且被應的手）、無視劫材直接消劫的次數、劫爭「勝方」（最終提/粘者）、劫爭平均長度。
- 算法：規則引擎的劫禁點追蹤 + 局面重複偵測即可。
- 等級效應：純假說——kyu 避劫或誤算劫材、段位劫爭較頻繁且較長。**本次未找到任何等級分層數據**，只有職業紀錄（單局 64 劫提）。
- 增量：未知但與 policy LL 正交（LL 不計數「事件」）。成本低。風險：訊號可能極弱；劫頻率受對手影響。冗餘：低。

**F4. 弱棋塊負債：同時存在的未活棋塊數與其壽命**
- 定義：每手計算己方「未確定活且 ≤k 氣或與對手棋塊接觸」的棋塊數，取全局最大值、平均值、以及「同時 ≥2 塊弱棋」的手數比例；弱棋塊存活到終局的比例。
- 算法：Benson 無條件活 + 氣數 + 眼位簡判；或用專案已有 KataGo ownership 當「活」代理（已在用，不算新成本）。
- 等級效應：Uberdude「理解弱棋塊是 5k→1d 關鍵」、Jarmo「新手攻太急」；Maia-2「全盤評估是 skill-dependent」支持全盤狀態量比局部手段更能分段。
- 增量：中。這是**全局狀態軌跡**，不是單手典型性；trunk-vector 序列模型可能隱含學到，但顯式計數是便宜的正則化特徵。成本中。風險：定義「弱」的門檻敏感。冗餘：中（與 trunk 序列模型）。

**F5. 階段錯置：過早收官 / 中盤的收官型著手 / 虛手與地內落子**
- 定義：(a) 前 N 手（如 ≤60）的一線、二線著手數；(b) 中盤（棋盤未定邊界 ≥m 處）時下在已確定邊界附近的 1–2 線手；(c) 終局：下在己方已確定領地內（ownership > 0.9）的手數、填自己眼位、單官先於有價值收官的次數；(d) 終局型態：是否以雙方虛手結束（計點）、最後 10 手是否皆為單官。
- 算法：邊線距離 × 階段直方圖（Moudřík 的 Border feature，1.29×）純規則；「已確定邊界/領地」需 ownership（已有）。
- 等級效應：Solomon「10k 太早走一線收官」、Jasiek「SDK/低段輸在收官型著手」、Knotwilg「10k 以下在自己地裡落子」；OGS 認輸數據顯示低 kyu 更常走到計點（Lys）。
- 增量：**低—中**。線距×階段分布幾乎就是 rank-conditioned policy 已學會的東西（冗餘高）；但「終局型態」（計點 vs 突然結束、地內落子數）屬棋局層級類別，policy LL 無此資訊，且專案列為死的是「長度變異」而非型態。成本低。風險：資料集伺服器的終局記錄方式未知（是否記錄虛手、是否自動計點）。冗餘：線距部分高、終局型態低。

**F6. 先手/後手局部序列（Moudřík ω-local）與「跟著對手走」**
- 定義：每局先手序列數、後手序列數、平均局部序列長度、己方結束序列的比例。
- 等級效應：Moudřík 1.30×；Jarmo「10k→5k 學會不跟對手走」。
- 增量：**低**。專案已測死「tenuki rate、與對手前一手距離」，此家族只是其聚合版本。冗餘：高。列出僅為完整性，**不建議優先**。

**F7. 提子數 × 階段直方圖（Moudřík Captured）**
- 定義：3×3（階段 × 己提/被提/差）。等級 1.27×；風格無效（0.98×）。
- 增量：低—中；與 F1/F2 高度重疊（提子事件的粗略總量）。成本極低，可作為 F1/F2 的廉價替身先試。冗餘：中。

**F8. 形的缺陷計數（空三角、留斷點、被扭頭）**
- 定義：己方形成空三角（排除邊線、提子、必要連接）次數；己方留下可被切斷的斜線連接且對手子已貼近的次數；被 hane at head of two/three 的次數。
- 等級效應：教學界公認初學者標記，**無任何頻率數據**（搜尋無果）。
- 增量：**低，判定冗餘**。形狀正是 policy 網路最擅長學習的局部典型性，rank-conditioned LL 已涵蓋。不建議。

**F9. 攻擊性指標（見 Q4）**：接觸手率（開局）、切斷數、打入數、殺棋嘗試（進入對手眼位）、對手弱棋塊數。增量中、冗餘中。

**F10. 錯誤分布形狀（見 Q3）**：Regan 式 (s, c) 擬合或 LL / δ 損的分位數與尾部比例。這是唯一有跨領域實證（西洋棋 c 隨 Elo 單調變化）支持、且專案尚未明確測過的「非平均」特徵。

### Gaps
- 沒有任何研究報告自殺手、征子誤讀、劫爭、空三角、過早收官、填眼的**按等級頻率**；上述效應方向全為假說。
- Moudřík 2015 未報告各特徵的逐級趨勢（只有整體 RMSE）。
- 資料集的終局記錄慣例（虛手、認輸標記被移除後是否仍可辨識）未知，影響 F5(d) 可行性。

---

## 關鍵問題 3：表現「變異性」（局內／跨局）作為棋力標記的證據，以及在中段棋手身上的表現

### Takeaway
西洋棋有實證：Regan 的雙參數模型把棋力分解為「敏感度 s」與「一致性 c」，擬合值顯示 **c（尾部薄厚）隨 Elo 單調上升（1600: 0.436 → 2700: 0.513，擬合線），s 單調下降（0.168 → 0.080）**，即弱者不只平均差、**大錯尾部更厚**；但 Anderson et al. 顯示變異的主要來源是局面難度而非棋手，chess.com 大數據也指出同分段內個別棋手的誤差變異差異很大、「強度與一致性是不同的東西」。圍棋端沒有任何等級分層的變異性量測，只有論壇的理論推導（總損失 SD ∝ 平均損失／√手數，故隨強度縮小）。「中段棋手變異是否特別大」在兩個領域都**沒有證據**。

### Cited Findings
- Regan & Haworth (AAAI 2011)：以引擎 multi-PV 值擬合人類選著頻率，模型 p_i = p_0^α、α = exp(−(δ_i/s)^c)；s=sensitivity（小則放大微小價值差）、c=consistency（大則急遽壓低壞手機率） — [AAAI 7951](https://ojs.aaai.org/index.php/AAAI/article/view/7951)
- Regan 《Intrinsic Ratings Compendium》(2012 working draft) Table 2，2006–09 資料：Elo 2700: s=.078, c=.502；2600: .092/.523；2500: .092/.491；2400: .098/.483；2300: .108/.475；2200: .123/.490；2100: .134/.486；2000: .139/.454；1900: .159/.474；1800: .146/.442；1700: .153/.439；1600: .165/.431。「中央動脈」擬合線：c_fit 由 2700 的 .513 線性降至 1600 的 .436，s_fit 由 .080 升至 .168。預期平均誤差 AE_e：2700 .0572、2600 .0624、2500 .0689、2400 .0749、2300 .0843、2200 .0883；IPR = 3571 − 15413·AE_e。作者註明 1600–2100 的某些點「異常，可能與棋局與棋譜品質有關」，只用 2200–2700 做線性擬合；每個 Elo 組使用 7,000–21,000 手。2011 加拿大公開賽 149 人：「IPR 相對自身均值的離散度高於 TPR」 — [Reg12IPRs.pdf](https://cse.buffalo.edu/~regan/papers/pdf/Reg12IPRs.pdf)
- Regan 另指出人類行為「價值差按局面總值比例縮放」：局面接近均勢時 1 個 marginal centipawn 的影響約為 +100 時的 5 倍 — 同上
- Anderson et al. (2016)：局面難度主導失誤（0.73 vs 棋力 0.55）；固定局面後棋力才有 0.62 的預測力；高手從額外時間得到的相對收益更大 — [arXiv 1606.04956](https://ar5iv.labs.arxiv.org/html/1606.04956)
- chess.com 3.36 億手：「平均著手品質隨 Elo 穩定提升，但局與局之間的離散即使在高分段仍很寬」；「同一分段內某些棋手的誤差變異遠大於他人（≥100 局者）」；風格聚類 Elite Grinders ≈27.7%、Chaotic Improvers ≈10.5%；未報告標準差或分位數 — [chess.com blog](https://www.chess.com/blog/Jordi641/hunting-for-context-in-336-million-blitz-moves)
- 自行發表（非審查）：Leite 聲稱 ACPL 與 STDCPL（cp 損標準差）與 rating「完美相關」 — [Medium](https://medium.com/@rafaelvleite82/how-i-found-perfect-correlation-between-chess-player-rating-and-acpl-and-stdcpl-bea9485055de)；相反地 Wojcicki (2020) 發現 ACPL 與 rating 相關低、只能估範圍 — [kwojcicki blog](https://kwojcicki.github.io/blog/CHESS-BLUNDERS)；兩者衝突且皆非同儕審查。
- Glickman (Glicko-boost)：模型假設「較強棋手的強度平均而言可能較穩定」，為建模假設非實證 — [glicko-boost.pdf](https://www.glicko.net/glicko/glicko-boost.pdf)
- 圍棋論壇「Strength as error distribution」：moha 提出整局總損失 SD ≈ 平均離完美距離／√手數，故 SD 應隨強度大約每段線性縮小；但強調「誤差獨立」假設不成立——棋手在特定類型棋局（模樣、大戰鬥）或特定對手面前會系統性變弱；以 KataGo 貼目位移估一設定下每局 SD ≈ 4.4 目（粗估）；**無人報告中段棋手的實測變異** — [lifein19x19 p=255718](https://lifein19x19.com/viewtopic.php?p=255718)
- Moudřík 2015：計點局的平均目差（Win/Loss points）單獨即有 1.47× 改善（RMSE 5.116），優於先手/後手、邊線、提子特徵 — [arXiv 1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969)（註：本專案無勝負資料，僅供「結果離散度載有等級資訊」之參考）

### Inferences
- Regan 的 c 參數是「分布形狀載有獨立棋力資訊」最乾淨的證據：在固定 s 下，c 決定壞手機率的尾部衰減速度。可直接移植：對每題，以 KataGo（已有）multi-PV 分數差 δ_i 與玩家實際選擇，擬合 (s, c)，或更簡單地取 **LL／δ 損的 10/50/90 分位、超過門檻的尾部比例、Gini**。Regan 用 ~1,500 手即可擬合；本題 5–20 局 ≈ 1,000–4,000 手量級相當。
- 專案已死的「每手品質時間序列統計（自相關、軌跡形狀、blunder 聚集）」是**時間結構**，而 (s, c) 或分位數是**邊際分布形狀**，兩者不同；Regan 的 δ 還做了「依局面總值比例縮放」，這是專案「強引擎目損」死掉時可能沒做的正規化（目損未按局面緊張度縮放時，難度雜訊主導——正是 Anderson 的結論）。
- 風險：rank-conditioned LL 的均值已隱含部分尾部資訊（弱者的低 LL 手更多）；增量可能只有小幅。另外 Anderson 的「skill-anomalous 局面」意味著尾部事件中有一部分是強者特有的（過度自信型錯誤），所以尾部比例對等級的關係未必單調。
- 「中段棋手變異更大」沒有任何支持證據；若要測，直接看 10 類中 B/A/1D 的 LL 分位數跨度即可，成本極低。

### Gaps
- 無任何圍棋研究量測逐手損失分布形狀隨等級的變化（Egri-Nagy 樣本全為 4d+）。
- Regan 的擬合只在 1600–2700（相當於強業餘至頂尖），沒有 kyu 級對照；1600–2100 區段作者自認異常。
- 「同分段內變異差異大」（chess.com）是否代表變異是獨立於等級的「個人特質」而非等級標記，未有檢驗。

---

## 關鍵問題 4：「過度攻擊 vs 消極」（打入、接觸手、切斷等攻擊指數）按等級的證據

### Takeaway
沒有任何圍棋伺服器資料按等級報告過接觸手率、打入數或切斷數；唯一的實證是 Moudřík 等人以模式特徵能把職業棋手的「攻擊性」專家評分預測到 1.41× 基準（但提子數特徵對攻擊性完全無效），以及西洋棋大數據中「領先時失誤率倍增、低分段崩盤率 2.5 倍」。教學界的主張是倒 U 型：新手攻太急想殺光一切，SDK 轉為消極跟著對手走，段位才取得平衡——這是可檢驗的假說而非結論。

### Cited Findings
- Moudřík et al. (2015)：攻擊性（aggressivity）以全部特徵預測 RMSE 1.545 vs 基準 2.183（1.41×）；Patterns 單獨對四種風格平均 1.28×，Captured stones 0.98×（無效），ω-local 1.03× — [arXiv 1512.08969](https://ar5iv.labs.arxiv.org/html/1512.08969)
- Baudiš & Moudřík (2012)：專家評分間相關——領地性 vs 攻擊性 r=−0.581、正統性 vs 攻擊性 r=0.682、領地性 vs 厚度 r=0.721；領地型模式為低位掛角與締角，模樣型為高位掛角/高位遠夾 — [arXiv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- 2013 年論壇對 GoStyle 風格估計器的評價：「結果感覺含糊」，領地↔模樣以 5.5 為中點 — [lifein19x19 p=135041](https://www.lifein19x19.com/viewtopic.php?p=135041)
- chess.com 3.36 億手：領先時 blunder 率高於均勢（1200 分 24.5% vs 14.7%；2000 分 19.6% vs 9.3%；2800 分 12.7% vs 4.4%）；崩盤率 800 分 29.5%、2400 分 19.8%、2800 分 11.8% — [chess.com blog](https://www.chess.com/blog/Jordi641/hunting-for-context-in-336-million-blitz-moves)
- 教學假說：Jarmo「新手攻得太急、想殺光一切，之後學會先守；10k→5k 不再跟著對手走」；Knotwilg「10k 以下在自己地裡下無意義的手」；Polama「1k 時焦點轉為選項：棄子、做活、逃出或脫先」；DJLLAP「有些人善戰不善靜，棋局以大殺或大死收場」 — [lifein19x19 p=203141](https://www.lifein19x19.com/viewtopic.php?p=203141)；kvasir 的 SDK 三大錯含「不夠積極」但他懷疑是感知偏差；Fairbairn「業餘 suji 差、下法斷斷續續」 — [lifein19x19 p=276919](https://lifein19x19.com/viewtopic.php?p=276919)
- Sensei's Library：kyu 棋手「有疑慮時脫先約 99% 導致災難」 — [senseis topic 3452](https://senseis.xmp.net/?topic=3452)
- OGS 提案者預期「戰鬥型風格 LpM 較高」 — [OGS forums 28837](https://forums.online-go.com/t/data-analysis-proposal-rank-vs-loss-per-move/28837)

### Inferences
- 可由規則引擎計算的攻擊性指數（皆為每百手率，分階段）：(a) 接觸手率：落子與對手子相鄰（開局 1–40 手尤其有鑑別假說）；(b) 切斷數：落子後對手棋塊連通分量數增加（或對手兩塊間的斜線/一間連接被破壞）；(c) 打入數：落子時周圍 5×5 內對手子數 ≥ 己方子數 + 3 且距邊 ≥3（純幾何代理，或用 ownership 定義為落在對手所有權 >0.7 區）；(d) 殺棋嘗試：落在對手未活棋塊的潛在眼位內；(e) 棋塊接觸比：盤上與對手相鄰的己方棋塊比例。
- 等級效應假說：DDK 高接觸、高切斷但低完成率（切了不提）；SDK 接觸率下降、跟手率上升；段位切斷「有結果」（切斷後對手棋塊死亡或被迫做活）比例上升。因此**原始攻擊性計數可能是非單調的，而「攻擊成功率」（切斷/打入後 20 手內的提子或對手補活）才可能單調**——這呼應 chess.com「崩盤率」隨等級單調下降。
- 冗餘：policy LL 已知道「kyu 棋手比較愛貼」，所以原始接觸率增量低；**攻擊結果（成功率）**與 F2 一樣是結果驗證維度，LL 不含。對手特徵已在用，可部分去除對手驅動的混淆。
- Moudřík「提子數對攻擊性無效」提醒：攻擊性不是提子多寡，而是模式分布；因此直接以提子數當攻擊指數大概會失敗。

### Gaps
- 無任何線上伺服器（KGS/OGS/Fox/Tygem）按等級的接觸手/打入/切斷統計；GoKibitz、Go Ratings 部落格搜尋未得相關分析。
- Moudřík 的攻擊性是職業棋手風格評分，非等級；業餘等級與攻擊性的關係完全無數據。
- 中文平台（野狐、弈客「吻合度」）的等級統計搜尋無結果 — 僅確認弈客鷹眼提供吻合度等指標 — [App Store](https://apps.apple.com/app/id988682022)。

---

## 總結建議（優先順序，依「與 policy LL 正交 × 有任何證據 × 成本」）

1. **分布形狀（F10）**：對已算好的 rank-LL 與 KataGo δ 損做分位數/尾部比例/Regan (s,c) 擬合，δ 先按局面緊張度縮放。唯一有跨領域實證、幾乎零成本。
2. **結果驗證型戰術事件（F1、F2）**：自殺手、被叫吃不應而被提、征子失敗、對殺氣數佔優卻輸、大塊死亡數與尺寸。與 LL 正交；稀少事件需收縮估計。
3. **弱棋塊負債軌跡（F4）** 與 **攻擊成功率（F9）**：全局狀態量與結果維度；中等成本。
4. **終局型態（F5d）**：計點 vs 突然結束、地內落子/填眼數、單官先於有價值手——先確認資料集終局記錄慣例。
5. **劫爭事件（F3）**：便宜但訊號未知，可順手一起算。
6. **判定冗餘、不建議**：形的缺陷計數（F8）、先手/後手序列與跟手率（F6，近似已死特徵）、線距×階段直方圖（F5a–b 的大部分）、原始接觸手率。

先驗提醒：Moudřík 2015 在幾乎相同的設定（26 級、每人 10–50 局）中，所有類別型特徵合計只比模式特徵多改善 0.13 級（2.788→2.659）。對本專案而言，所有新家族加總的合理預期是**小幅、非零**的增量，應以 fold-0 之外的 CV 嚴格驗證，避免再次消耗 fold 0。
