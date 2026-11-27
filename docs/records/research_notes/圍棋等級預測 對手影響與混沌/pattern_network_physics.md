# 圍棋局部定式（pattern）使用網路 — 統計物理 / 複雜網路文獻與等級預測可行性

研究日期：2026-11-04。範圍：Georgeot–Giraud 系列（2012 / 2014 / 2017）、Baudiš–Moudřík 系列（2012 / 2015 / 2016）、Zipf / Heaps 類文獻（圍棋、西洋棋、語言）、以及把「canonical local pattern 頻率 / 轉移網路」特徵接進現有 policy-likelihood + trunk-vector 流程的實務評估。所有數字均附來源；找不到來源的放在 Gaps。

---

## KQ1. Georgeot & Giraud 及後續：pattern 定義、網路建構、統計量，以及業餘 vs 職業（與 1d/6d/9d）的差異

### Takeaway
三篇論文全部以「落子點為中心的 3×3 plaquette（或 3×3+四個距離 2 點的 diamond）、8 個對稱 + 黑白互換歸一、空中心」為節點（3×3 → 1107 個節點；diamond → 193,995 個），邊 = 同一局中「下一手落在 Chebyshev 距離 ≤ 4 內」的 pattern→pattern 有向加權連結。Zipf 指數 ≈ 1（1.06 / 0.94–1.02）在職業與業餘資料庫中幾乎相同、最常見 10 手完全相同，**頻率分佈的形狀不是等級鑑別子**；差異出現在（a）連續手距離分佈 P(d)（業餘更多短距離手）、（b）Google matrix 主特徵向量 / 社群結構、（c）1d vs 9d 的 out-link 差異量 r ≈ 1.33（相對 6d-6d 樣本變異「有一些統計顯著性，但差異相對小」）。所有這些網路量都需要數千局才能穩定，無法直接套用到每位玩家 5–20 局的設定；但「節點 = canonical pattern、邊 = 局部後續手」這套定義可以直接搬來做 per-player 的 bag-of-patterns / bigram 特徵。

### Cited Findings

**Georgeot & Giraud, EPL 97, 68002 (2012), arXiv 1105.2470**
- 資料：職業 = 四個日本頭銜戰 1941 年起全部棋譜（棋聖 143、名人 259、本因坊 305、十段 158，共 865 局）；業餘 = 4000 局，同樣來自 u-go.net。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 節點定義：以落子點為中心的 3×3 方格，8 個鄰點各為空/黑/白；棋盤邊界以「兩條額外虛線」處理；旋轉/鏡射等價、黑白互換等價；空中心的非等價 plaquette 共 **1107** 個 = 網路節點數。5×5 被提及為假設情境（約 3×10^10 個頂點），未使用。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 邊定義：若手 b 在手 a 之後、且 max(|Δh|,|Δv|) ≤ d（主網路 d = 4），則 a→b 有向連結，權重 = 資料庫中出現次數。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 統計量：pattern 頻率的積分分佈 Zipf 指數約 **1.06**（各資料庫斜率 −1.058 至 −1.081）；若單純以 361 個座標為單位則看不到 Zipf 律。連續手序列的頻率斜率從 2 手序列 −1.01 遞減到 7 手序列 −0.77（作者歸因於長序列中的個人策略）。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 入度/出度分佈尾端 ~1/k^γ，γ ≈ 1.0 且入出對稱（WWW：out 1.7、in 1.1）；clustering coefficient > 0.7（WWW ≈ 0.11）；Google matrix 第一特徵值 λ=1 之後有「巨大 gap」，把同一局內手順打亂後 gap 消失；PageRank / CheiRank 衰減斜率 ≈ −1，HITS ≈ −1.5；PageRank 與 CheiRank 在五個資料庫都有「極強相關」。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 業餘 vs 職業：pattern 頻率積分分佈「非常相似」，**最常見 10 手在所有資料庫中相同**（順序略異）；距離分佈 P(d) 中業餘資料庫「顯著偏離」職業資料庫，**短距離手更多**（作者推測職業同時進行多處戰鬥或局部戰鬥較短，屬推測）；四個職業頭銜戰在序列距離分析上「幾乎無法區分」；主特徵向量在頭銜戰之間與職業/業餘之間都不同（業餘的特徵向量分佈與本因坊「非常不同」）。論文**沒有**宣稱可以分類玩家等級，只說職業/業餘差異可見。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)

**Kandiah, Georgeot & Giraud, EPJ B (2014), arXiv 1405.6077**（注意：任務說明中的 arXiv 1405.7731 是另一篇數學物理論文，正確編號為 1405.6077 — [arXiv 1405.7731 實為 Einstein constraint equations](https://arxiv.org/abs/1405.7731)）
- 資料：u-go.net 資料庫 **135,663 局**，棋手 1–9 段（線上對局，段位由成績評定）。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- 三種網路：I = 3×3 方格（1107 節點）；II = 3×3 + 四個最近鄰的 atari 狀態（2051 個合法 plaquette）；III = diamond（3×3 + 四個距離 2 的點），**193,995** 個 plaquette，其中 **96,771 個在資料庫中從未出現**。對稱：平移、正方形對稱群、黑白互換。連結：a→「距離 d=4 內的第一個後續手」b。去重後連結數 558,190（I）/ 852,578（II）/ 7,405,395（III）。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- Zipf 斜率：網路 II ≈ −1.02、網路 III ≈ −0.94；連結（邊權）分佈指數入 ≈ −1.47、出 ≈ −1.69。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- 譜分析：Google matrix α=1；網路 I、II 第一特徵值後有大 gap，網路 III 的 gap 較小，作者解讀為**更多社群結構**；G 的特徵向量 = 「難以離開的社群」，G* 的特徵向量 = 「外部少有連入的群」，例子有 ko 爭奪、即將提子、沿邊連接、邊上吃鏈等。社群抽取為經驗法（截斷特徵向量 + 共同祖先生長，閾值 0.3<ε<0.7），**沒有 modularity 或社群數**。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- **等級差異（唯一的 graded-amateur 網路比較）**：定義 r_j = Σ_{i←j}|k_i − k'_i| / Σ_i k_i（節點 j 在兩個網路中出鏈數之差），在 PageRank 前 1500 名節點上平均；每個樣本 **2731 局**（= 資料庫中 1d/1d 對局數）；1d/1d vs 9d/9d 的 ⟨r_j⟩ ≈ **1.33**（Fig. 18），與多組 6d/6d 樣本之間的 r 分佈比較（Fig. 19 給 6d 均值與變異水平線，數值未在文字中給出）。結論：「1d 與 9d 的差異有一些統計顯著性」但「相對小」，9d「可能有整體結構上不同的風格」；兩個 6d/6d 網路之間也有差異。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)；[搜尋摘要](https://arxiv.org/pdf/1405.6077)
- 分階段網路（前 50 / 中 50 / 後 50 手）：連結總數 6,155,936 / 6,460,771 / 5,947,467，去重後 **613,953 / 2,070,305 / 3,182,771**（pattern-bigram「詞彙量」從佈局到官子增加約 5 倍），譜在各階段不同，佈局特徵向量偏好較空的 plaquette。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)

**Coquidé, Georgeot & Giraud, EPL 119, 48001 (2017), arXiv 1707.04044（人類 vs 電腦）**
- 資料：8000 局業餘人類（u-go.net）、8000 局 GNU Go 自戰（不同種子，決定性程式）、8000 局 Fuego（Monte-Carlo）、50 局 AlphaGo（只用於 Fig. 6）。無職業棋譜。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- 網路：3×3、空中心、N=1107，d_s=4（下一手必須是該距離內的第一手）。總連結 1,589,729 / 2,046,260 / 1,527,421。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- 度分佈三者皆 γ≈1.0，電腦網路在較小 K 截斷且震盪強，hub「略少」，作者說度分佈本身**不是可靠指標**。人類 PageRank 前 20 節點中 GnuGo 有 12 個、Fuego 有 18 個重疊。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- 鑑別量 σ（Eq. 1，PageRank 向量的離散度）：隨機置換 ≈ 450；4000 局一組時 人–人 43.66、機–機 24.04、人–機 **192.58**；1000 局一組時 66.71 / 44.14 / 199.27；全 8000 局人–機 193.48。1000 局子集已可分離兩群；沒有正式檢定。GnuGo 有許多距原點 0.2–0.6 的特徵值（人類沒有），解讀為內部緊密、對外弱連的手群 = 「相對獨立的策略」；Fuego 局較長可能造成偏差。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- 引用的先前結果：「職業的手比業餘更不可預測，且可預測性隨專長改變」(ref [21])；「業餘等級之間有差異」(ref [20] = Kandiah 2014)。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- 新聞稿說法：人類與電腦網路「統計特徵不同，而且在相對少量的局數上就能統計顯著」。 — [phys.org 2017](https://phys.org/news/2017-11-distinguishing-humans-game.amp)

**其他**
- computer-go 郵件列表：從 40,000 局職業棋譜收集 3×3 pattern（對稱視為同一），兩位獨立開發者都得到 **1093** 個（與 1107 理論上限相符）；Yamashita 加上「到前一手的距離（17 種）」後 move-prediction 從 38.8% 升到 40.2%。 — [computer-go 2015-11](https://www.eugeneweb.com/pipermail/computer-go/2015-November/008117.html)
- Liu, Dou & Lu, Computers and Games 2008, LNCS 5131 pp.125–134：9,447 局職業棋譜，contextual pattern 頻率呈 Mandelbrot 修正的 Zipf 律；拒絕「隨機對局也會呈 Zipf」的虛無假設。 — [dblp](https://dblp1.uni-trier.de/rec/conf/cg/LiuDL08.html)；[doi](https://doi.org/10.1007/978-3-540-87608-3_12)

### Inferences
- 「網路 = 玩家在局面中製造出來的 pattern 轉移統計」的差異量（r_j、PageRank 向量離散度 σ、主特徵向量）在**數千局**規模才顯著；1d vs 9d 的 r≈1.33 需要 2731 局/組。對 5–20 局的玩家，網路譜量（特徵值 gap、社群數）**無法逐人計算**；可行的是把玩家的 pattern 直方圖 / bigram 直方圖與「用訓練集各等級建好的參考網路」比較（KL / r_j 型距離），或讓模型直接學直方圖 → 等級。
- Zipf 指數在職業/業餘之間幾乎相同、top-10 pattern 相同 ⇒ 等級訊號在**尾部與條件結構**（哪些 pattern 之後接哪些、距離分佈），不在整體頻率形狀；單純的「Zipf 斜率」或「熵」特徵預期弱。
- Georgeot 的 P(d)（連續手距離）差異與 Moudřík 的 ω-local sente/gote 特徵是同一件事的兩種包裝；本專案「response geometry 已死」顯示這類距離特徵在強基線下已無增益，不應再投入。
- Kandiah 的 diamond 網路有 96,771/193,995 個從未出現的節點，說明 >3×3 的 pattern 空間極稀疏；per-player 使用需 hash 或 top-N。

### Gaps
- 三篇論文都沒有逐玩家（per-player）或少量局數的實驗；沒有任何一篇把 kyu 級（10k–1k）納入網路比較。
- Kandiah 2014 的 6d/6d 基準 r 值、以及 r 的分佈/p 值只在圖中（Fig. 19），文字無數字。
- 無法取得 Liu–Dou–Lu 2008 全文（Springer 需登入），contextual pattern 的尺寸與 Mandelbrot 參數未知。
- Coquidé 2017 ref [21]（「職業手更不可預測」）的出處未確認（可能是 Harré 等人，見 KQ3）。

---

## KQ2. Moudřík–Baudiš–Neruda：pattern 特徵建構、各特徵族結果、為何 pattern 帶有等級資訊；與 rank-conditioned policy likelihood 的非冗餘性論證

### Takeaway
Moudřík et al. (CIG 2015) 用 Pachi 匹配「gridcular 半徑 2–6 的空間 pattern + atari 旗標、歸一到黑先、旋轉/鏡射不變」，取全庫最常見 N=1000 個 pattern 的 per-player 計數 /|games| 作特徵；KGS 26 級（20k–6d）、每級 120 位玩家、每人 10–50 局、僅無讓子互先局；bagged NN 的 RMSE：平均預測 7.503 → Patterns **2.788**（×2.69）、全部特徵合併 2.659（其他族只再貢獻 0.13）。2012 年前作在 GTL 7700 局上 k-NN 9 局 MSE 21.3（σ≈4.6 kyu）、85 局 MSE 5.5。作者對「為什麼 pattern 帶等級」只給直覺（弱者「因為可以」就提子、邊線一線之差影響巨大、佈局形狀帶策略目標），沒有與任何 policy likelihood 比較。非冗餘性要靠論證：policy likelihood 評的是「這手在完整局面下對該等級有多典型」（contextual appropriateness），pattern 頻率是「玩家在自己的對局中製造 / 選擇哪類局部形狀」的邊際統計（habit / repertoire），兩者在數學上互不決定，但 trunk vector + sequence model 原則上能學到後者，所以預期增益小且必須實測。

### Cited Findings

**Baudiš & Moudřík 2012, "On Move Pattern Trends in a Large Go Games Corpus", arXiv 1209.5251**
- 每手的 pattern = 以下特徵組合：capture 旗標、atari 旗標、atari-escape 旗標、與前一手的 contiguity、與前前手的 contiguity、邊界距離（最多 4）、空間 pattern；空間 pattern 歸一到黑先、平移與旋轉對稱；匹配 gridcular 度量 d(x,y)=|δx|+|δy|+max(|δx|,|δy|) 下**半徑 2–9** 的配置，來自 Pachi 的 pattern-matching 程式碼（Elo-rated pattern selection）。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- 特徵向量 = 全庫最常見 n=500 個 pattern 的頻率，線性縮放到 [−1,1]；log+logistic 的「extended normalization」提升 PCA 可解釋性但降低分類精度；**強度分類時移除 contiguity 特徵以減少小樣本上的 pattern 多樣性**。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- 強度資料：Go Teaching Ladder 審局檔 7700 局、30k–4d、僅互先且等級明確；每個等級一個聚合向量；PCA 第一維與等級 Pearson **r = 0.979**，第二特徵值小四倍。（注意：這是**按等級聚合**後的相關，不是逐玩家。） — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- 逐玩家精度 vs 局數（Table I，MSE，單位 kyu²）：k-NN 85 局 5.514、43 局 8.449、17 局 10.096、**9 局 21.343（σ=4.62）**、2 局 52.212；NN 17 局 110.633、9 局 44.512、2 局 43.682；PCA 法 85 局 24.070、2 局 128.660；隨機基準 208.549。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- 風格：PCA 第一維與 territoriality 相關 −0.530；風格聯合分類 MSE 3.960 vs 隨機 11.776。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)

**Moudřík, Baudiš & Neruda, IEEE CIG 2015, arXiv 1512.08969**（已從 PDF 全文擷取）
- Pachi 對每手輸出：atari 旗標、atari-escape 旗標、capture 數、與前一手的 gridcular 距離、邊界距離、空間 pattern；**空間 pattern 取 size 2–6**。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Pattern 特徵族 = 全庫最常見 **N 個空間 pattern（含兩個 atari 旗標）**的統計，N=1000（強度）/ 600（風格）；歸一為黑先、旋轉/鏡射不變；對一組 colored games GC 計數每個 pattern 的出現次數 c，再除以 |GC|。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- ω-local sente/gote：手與前一手 gridcular 距離 < ω（強度 ω=10、風格 ω=5）為 local；把每局切成 ω-local 序列，看起手者是否為結束者 → 每局平均 sente 數、gote 數（2 維）。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Border distance：2D 直方圖，邊界距離 {1–2, 3, 4, ≥5} × 手數 {1–10, 11–64, 65–200, ≥201}（強度）→ 16 維，除以 |GC|。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Captured stones：手數 {1–60, 61–240, ≥241} × {己方提子、對方提子、差}（9 維）；動機：「初學者傾向因為『可以提』就提子而不是因為那是最佳手」。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Win/Loss：以點數/投降贏/輸各自頻率 + 平均勝負點數（6 維）。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- 模型：bagged（N=20）前饋 NN，1 隱藏層 20 單元，RPROP ≤100 迭代；目標縮放到 [−1,1]；10-fold CV 重複 5 次。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- 強度資料：KGS 公開檔 >100,000 局；6d–20k 每個等級 r 取玩家集 Pr；**僅 19×19、實力相近、無讓子**；每位玩家需 ≥10 局，>50 局則隨機取 [10,50] 均勻隨機大小的子集；**26 個等級各 120 組 GC**；y = 20（20k）…1（1k）、0（1d）…−5（6d）。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- **Table I（強度 RMSE，單位：等級）**：Mean regression 7.503±0.001；**Patterns 2.788±0.013（×2.69）**；ω-local Seq. 5.765；Border Distance 5.818；Captured Stones 5.904；Win/Loss Statistic 6.792；Win/Loss Points 5.116；**All combined 2.659±0.002（×2.82）**。作者：「Pattern 特徵效果遠遠最好，其他特徵帶來較小但非平凡的貢獻」。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- 風格（GoGoD 25 位職業、每人 192 局切成 12 組 16 局，專家 1–10 分問卷）RMSE：Mean 2.168、Patterns 1.691、Border 1.723、All 1.615；Captured stones 2.223（比均值差）；per-style：territoriality 2.396→1.575、orthodoxity 2.423→1.773、aggressivity 2.183→1.545、thickness 1.672→1.567（thickness 被判定定義不良）。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- 作者對「為何有效」的說明僅限直覺：可以找出「與弱棋力最強相關的 pattern」警告使用者不要下；弱者可能沒有一致風格；可能的其他目標還有年代與國籍。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- 部落格二手整理：pattern 特徵約 2.79 級誤差 vs 其他族 5–7 級。 — [搜尋摘要（jmoudrik.github.io，頁面現已 404）](https://jmoudrik.github.io/post/2016/01/15/convolutional_neural_net_for_Go_strength_prediction.html)
- 後續 CNN 法（Moudřík & Neruda, TPNC 2016, LNCS 10071）：188,700 局訓練 CNN，3 個棋力類別，單一局面 71.5%、一局聚合 77.9%。 — [搜尋摘要 / Springer](https://link.springer.com/10.1007/978-3-319-49001-4_15)

**Pachi pattern 基礎設施（canonical pattern 的工程定義）**
- 特徵列舉：FEAT_CAPTURE（ATARIDEF/PEEP/LADDER/NOLADDER/TAKE_KO/END_KO）、FEAT_CAPTURE2、FEAT_AESCAPE、FEAT_SELFATARI（BAD/GOOD/2LIBS）、FEAT_ATARI（SNAPBACK/LADDER_*/DOUBLE/KO/SOME…）、FEAT_NET、FEAT_DEFENCE、FEAT_CUT、FEAT_WEDGE、FEAT_L1_BLUNDER_PUNISH、FEAT_DOUBLE_SNAPBACK、FEAT_EYEFILL、FEAT_BORDER（線數 ≤4）、FEAT_DISTANCE（到前一手 ≤17）、FEAT_DISTANCE2、FEAT_MCOWNER、FEAT_SPATIAL3…FEAT_SPATIAL10（spatial_dict 索引，黑先）。 — [pachi pattern.h](https://raw.githubusercontent.com/pasky/pachi/master/pattern/pattern.h)
- spatial.h：MAX_PATTERN_DIST = 10、MAX_PATTERN_AREA = 73 點；累計點數 d=2:5、3:9、4:13、5:21、6:29；Zobrist 雜湊 pthashes[8 rotations][73][S_MAX]，同時雜湊黑白互換；字典雜湊 20 bits（「4Mb array」），產生字典時 26 bits（~256 MB），鏈結解決碰撞；每筆紀錄 = 距離位元組 + 2-bit 打包點。 — [pachi spatial.h](https://raw.githubusercontent.com/pasky/pachi/master/pattern/spatial.h)
- 來源方法：Stern, Herbrich & Graepel, ICML 2006 "Bayesian pattern ranking"，181,000 局專家棋譜，精確預測職業手約 34%（演講版 33%，前 10 內 80%）。 — [Microsoft Research](https://www.microsoft.com/en-us/research/?p=150689)；[talks.cam](https://talks.cam.ac.uk/talk/vcal/4461/)

### Inferences（非冗餘性論證）
- **數學上互不決定**：rank-conditioned policy likelihood 的 per-move 量是 ℓ_r(t) = log p_r(a_t | s_t)，再對手數平均；pattern 頻率是 f(p) = (1/M) Σ_t 1[π(a_t, s_t) = p]（π = canonical 3×3/diamond 投影）。給定 ℓ_r 的分佈完全不能還原 f，反之亦然：兩位玩家可以在各自製造的局面中都下「對該等級典型」的手（ℓ 相同），但一位總是把局面導向貼、扭斷、打吃（接觸形 pattern 多），另一位總是避戰（空曠形 pattern 多）。f 反映的是「玩家把棋局帶到哪類局部狀況 + 在那裡選了什麼形」，是 habit / repertoire；ℓ 反映「在已給定的狀況下選手是否典型」，是 contextual appropriateness。
- **但實務上高度相關**：Moudřík Table I 顯示其他特徵族（border distance、captures、sente/gote）加進 pattern 後只把 RMSE 從 2.788 降到 2.659，表示 pattern 直方圖已吸收大部分「邊界距離、提子、局部應對」訊號；反過來，humanv0 的 18 個 profile 是以整盤局面訓練的強模型，其 likelihood 必然已涵蓋「弱者在這個局面下常提子 / 常下空三角」這類與 pattern 強相關的訊號。pattern 直方圖的獨立貢獻只剩「跨局的邊際組成」部分。
- **trunk vector 的冗餘**：trunk 特徵圖在落子點附近就是局部形狀的分散式表示；sequence model 對 trunk vector 做平均池化在原理上能逼近 pattern 直方圖。顯式 pattern id 的優勢只在（1）硬編碼 16 重對稱不變性與精確計數（小樣本下方差較小）、（2）可做跨局 bigram（pattern→下一個局部 pattern）這種 sequence model 跨局無法看到的統計、（3）可做與「對手 pattern 直方圖」的對比（資料含雙方手順）。
- **預期增益**：沒有任何文獻把 pattern 頻率與 policy likelihood 併用，增益未知；依上述冗餘分析，對已有 likelihood + trunk + opponent + handicap 的強基線，合理預期為「零到微小」（本專案已死的 style-consistency / response-geometry 特徵就是同類邊際統計失敗的先例）。最有機會非冗餘的是 (a) 等級參考分佈下的 bag-of-patterns log-likelihood（KQ4 候選 C）與 (b) pattern bigram（候選 D），因為兩者是跨局邊際量而非逐手條件量。
- **局數門檻**：Moudřík 要求 ≥10 局（實際 10–50）；2012 年結果 9 局 σ≈4.6 kyu、85 局 σ≈2.3 kyu。本專案 5–20 局落在其「可用但雜訊大」區間；10 類（約 16 級壓成 10 類，每類約 1.6 級）下 pattern-only 的 σ≈2.8–4.6 級 ≈ ±2–3 類，遠弱於現有基線（見 KQ3 的 Chen et al. 10 局約 80%、20 局約 92%）。

### Gaps
- Moudřík 2015 全文無 MAE、無逐等級誤差、無 per-pattern 重要度表；Moudřík 碩士論文（2013, j2m.cz）與 gostyle 網站無法取得。
- 沒有文獻報告「pattern 頻率特徵對 policy-likelihood（或任何 move-prediction 模型）基線的增量」；Moudřík 2016 CNN（3 類 77.9%）與 2015 pattern 法（26 級 RMSE 2.66）用不同指標，無法直接比較。
- Coulom 2007（ICGA J. 30(4) 198–208）全文為二進位 PDF 無法讀取，其 shape 半徑設定未確認。

---

## KQ3. Zipf / Heaps / 詞彙量 / 熵與棋力：圍棋與類比領域有哪些數字

### Takeaway
圍棋：Zipf 指數在職業與業餘之間一致（≈1.06；1.02/0.94 依網路定義），Liu–Dou–Lu 在 9,447 局職業棋譜確認 Mandelbrot–Zipf；Kandiah 的分階段網路顯示 pattern-bigram 詞彙從佈局 613,953 → 官子 3,182,771。**沒有任何圍棋文獻報告「每千手不同 pattern 數 / 熵 vs kyu-dan」的數字**。Harré 等（2011, Minds & Machines + EPJ B）在 7×7 角部定式區用 Shannon 熵量化手的可預測性 vs 棋力，二手來源說「手變得越來越可預測，直到接近職業水準」後反轉，但原文數字無法取得。西洋棋：Perotti 2013 的 Heaps 指數 λ_d = 1 − 0.854^d；Chowdhary 等（Lichess 1.23 億局）：初學者第一手更多樣、專家更常下「最愛的開局」但算進變化後反而遭遇最多多樣性；另有海報說高分玩家開局早期變化少、後期變化多。語言：MTLD 等詞彙多樣性指標與寫作能力顯著正相關，但受文本長度與主題影響。整體：「詞彙量」與實力**不是單調關係**（早期收斂、後期發散），且強烈依賴樣本長度，須做長度匹配。

### Cited Findings
- 圍棋 Zipf：pattern 頻率積分分佈指數 ≈1.06，職業四個頭銜戰與業餘資料庫斜率 −1.058 至 −1.081；top-10 pattern 相同。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- 三種網路定義下斜率約 −0.94 至 −1.02。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- 分階段去重連結數（pattern-bigram 詞彙量）：前 50 手 613,953、中 50 手 2,070,305、後 50 手 3,182,771。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- 9,447 局職業棋譜 contextual pattern 呈 Mandelbrot–Zipf；隨機對局不呈此分佈。 — [dblp LiuDL08](https://dblp1.uni-trier.de/rec/conf/cg/LiuDL08.html)
- Harré, Bossomaier & Snyder, "The Development of Human Expertise in a Complex Environment", Minds & Machines 21, 449–464 (2011)：以 Shannon 熵量化對手下一手的不確定性如何隨棋力變化；報告「從業餘到世界冠軍的進程中有兩個未預期的轉變」，其一是「全局意識的穩定上升，開始得出奇地晚」。 — [搜尋摘要 / DOI](https://unpaywall.org/10.1007%2FS11023-011-9247-X)
- Harré, Bossomaier, Gillett & Snyder, "The aggregate complexity of decisions in the game of Go", Eur. Phys. J. B 80, 555–563 (2011)：「在目前圍棋 AI 尚未達到的水準上，量化專家手的不可預測性以及它如何隨棋力改變」；摘要無方向與數字。 — [ideas.repec.org](https://ideas.repec.org/a/spr/eurphb/v80y2011i4p555-563.html)
- 二手摘述（Wired 引文經論壇轉述）：分析方法是 7×7 角部區域（定式區）的手序樹；「手變得越來越可預測，直到玩家接近職業水準」，之後反轉，Harré 解釋為全盤資訊開始影響局部決策。 — [lifein19x19 討論串](https://lifein19x19.com/viewtopic.php?p=164936)
- 新聞稿：Bossomaier 估計專家的 pattern「chunks」可能「超過 100,000」（Simon 估 50,000）；從業餘到職業時知識「經歷劇烈重組」，包括「知覺層級的根本重組」。 — [phys.org 2012](https://phys.org/news/2012-07-patterns.html)
- 以 Darkforest 預測率衡量：職業平均 89.09%、業餘 82.02%（受邀樣本小，6 段業餘到 9 段職業）。 — [arXiv 1707.04828 搜尋摘要](https://arxiv.org/pdf/1707.04828)
- 以多等級 policy 估棋力（Chen, Shih, Wu 等）：圍棋 10 局約 80%、20 局約 92%，優於先前方法（約 71% 與 84%）。（類別數在摘要中未給出。） — [aitopics 摘要](https://aitopics.org/doc/arxivorg:5F7705A2)
- 西洋棋 Heaps：ChessDB 約 140 萬局（1998–2007），手序樹頂點 = 「詞」；Heaps 指數隨深度 λ_d = 1 − B^d，B ≈ 0.854±0.002（R²=0.998）；創新機率 p(n) ~ n^−ν，ν ≈ 0.881±0.009；**未分析等級**。 — [ar5iv 1309.0336](https://ar5iv.labs.arxiv.org/html/1309.0336)（EPL 104, 48005）
- 西洋棋開局多樣性 vs 等級（Chowdhary, Iacopini & Battiston）：Lichess 2013–2016，1.23 億局、98 萬玩家、僅 Blitz，四等分技能類；初學者白方第一手更多樣；專家「下最愛開局的次數顯著多於第二愛」，初學者兩者相近；主線層級專家多樣性較低，但「計入眾多變化後，專家遭遇最多多樣性」；多樣性隨生涯階段下降。無數值。 — [ar5iv 2207.07780](https://ar5iv.labs.arxiv.org/html/2207.07780)
- 另一分析：高分玩家開局早期變化略少、後期變化較多，作者歸因於記憶背誦。 — [Northwestern 海報](https://sites.northwestern.edu/kovacslab/files/2023/06/mckenziesmiththomas_222967_16743406_McKenzie-Smith_Thomas_Poster_Presentation.pdf)
- 開局複雜度 vs 等級：等級上升時低複雜度開局使用減少、高複雜度開局增加。 — [arXiv 2206.14312 搜尋摘要](https://arxiv.org/pdf/2206.14312)
- AlphaZero：遊戲狀態頻率呈 Zipf 律（源於樹結構）；agent 依頻率遞減順序學會狀態；模型放大失效與異常 Zipf 曲線（終局狀態最常見）相關。無 Elo 數字。 — [arXiv 2412.11979](https://arxiv.org/abs/2412.11979)
- 語言：匈牙利與約旦 EFL 大學生的寫作能力分數與詞彙多樣性（VocD、MTLD）顯著正相關。 — [IJOLTL](https://ijotl-tl.soloclcs.org/index.php/ijoltl/article/view/769)；法語 L2 的 MTLD/HD-D/D 解釋 C-test 分數最多 62% 變異，但文本長度仍影響指標（搜尋摘要，出處連結未逐一確認）。 — [搜尋結果彙整](https://core.ac.uk/works/6791257)

### Inferences
- 圍棋的 Zipf 形狀對等級不敏感 ⇒ 「Zipf 斜率」作為特徵預期無效；可能有效的是**尾部內容**（哪些稀有 pattern 出現）而非形狀。
- 詞彙量（distinct canonical patterns per N moves）與實力的關係在類比領域呈**非單調**（初學者亂下 → 多樣；中段收斂到定式 → 少樣；高段脫離定式 → 再多樣，Harré 的「近職業反轉」亦同），對 10k–6d 的區間可能單調（收斂段）也可能跨過轉折點，必須實測；且 Heaps 律使詞彙量隨樣本長度次線性增長，5 局與 20 局的玩家不可直接比較，需固定手數抽樣（例如每人 bootstrap 固定 500 手）或以「每 100 手新 pattern 率」衡量。
- Harré 的「7×7 角部定式區熵」是比 3×3 更接近人類「定式知識」的單位；對本專案可轉譜為「角部前 N 手的 canonical 序列 id」，但這就是已死的 opening-repertoire 特徵的近親，不建議重做。

### Gaps
- 完全找不到圍棋「每千手不同 pattern 數 / pattern 熵 / 重複率 vs 等級」的任何數字（含 KGS/OGS/Fox 資料）。
- Harré 2011 兩篇全文（Minds & Machines、EPJ B）為付費牆，熵隨等級的曲線、資料集大小（傳聞 160,000 玩家，未證實）與轉折等級均未取得。
- Chowdhary et al. 的各技能類熵值只在圖中；是否已刊於 Scientific Reports（s41598-023-27735-9）未能確認（nature.com 需登入）。
- 語言領域「詞彙量 vs 能力」的 62% 數字出處連結未核實。

---

## KQ4. 實務：以規則引擎計算 canonical local-pattern 特徵（1M 局、CPU）的成本、維度、接進 MLP / sequence model 的方式；候選特徵清單

### Takeaway
3×3 canonical pattern 是 O(1) 查表（4^8 = 65,536 條目的 neighbour-code → canonical-id 表，涵蓋邊界為第 4 種狀態），diamond（12 點）可用 16 個對稱變換取最小 hash，gridcular d≤6（29 點）用 Pachi 式 Zobrist（8 旋轉 × 黑白互換）；成本瓶頸是棋盤規則維護（提子）而非 pattern 計算。以本 repo 的 Go 語言引擎估計 1M 局（≈2.2×10^8 手）單核數分鐘級、Python 純實作數 CPU-小時級（可平行）。維度：3×3 精確 1107；diamond 用過（≈97k）需 top-N（1000–2000）或 feature hashing 到 2^12–2^14。接法有四種：per-game 歸一直方圖 → question-level 平均（MLP）；per-move pattern id 當額外 token embedding（sequence model）；等級參考分佈下的 bag-of-patterns log-likelihood（10 維，與 policy likelihood 同形）；pattern bigram（Georgeot 邊）直方圖。

### Cited Findings
- 3×3 空中心 canonical pattern 共 1107 個（理論）、40,000 局職業棋譜觀測到 1093 個。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)；[computer-go 2015-11](https://www.eugeneweb.com/pipermail/computer-go/2015-November/008117.html)
- diamond（3×3 + 4 個距離 2 點）canonical 共 193,995，其中 96,771 在 135,663 局中從未出現（即實際用到約 97k）。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- Pachi：gridcular 半徑 d 的累計點數 d=2:5、3:9、4:13、5:21、6:29、最大 d=10:73；Zobrist 雜湊表 pthashes[8][73][S_MAX]，同時雜湊黑白互換；字典 20-bit 雜湊（4 MB）、鏈結解碰撞。 — [pachi spatial.h](https://raw.githubusercontent.com/pasky/pachi/master/pattern/spatial.h)
- Moudřík 的 per-player 歸一：count / |GC|（每局平均次數）；N=1000 top patterns + atari 旗標；強度資料需每人 ≥10 局。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Baudiš & Moudřík：小樣本時移除 contiguity 特徵以降低 pattern 多樣性；log+logistic 歸一化幫助 PCA 但傷分類。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- Georgeot 邊定義（bigram 特徵的依據）：a→b 若 b 為 a 之後、Chebyshev 距離 ≤4 內的第一手。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- Kandiah 等級差異量 r_j = Σ|k_i − k'_i| / Σ k_i 在 PageRank 前 1500 節點平均（可改成玩家 vs 等級參考網路的距離）。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- Coquidé 的 PageRank 向量離散度 σ 在 1000 局子集已可分群（人–機 199 vs 人–人 67）。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)

### Inferences（成本與設計估算 — 以下為本人推估，非文獻數字）
- **手數規模**：1M 局 × 約 220 手 ≈ 2.2×10^8 個局面更新（假設平均手數 200–250；本資料集若含讓子局與中盤認輸局會較短）。
- **3×3 計算**：每手讀 8 鄰點（空/黑/白/界外 4 態）→ 16-bit code → 查預建 65,536 表得 canonical id（表以 8 對稱 × 黑白互換取最小 code 建立，一次性）。附加成本 ≈ 10 ns/手；全部 ≈ 2–3 CPU-秒。棋盤維護（提子、氣）在 Go/C 實作約 1–5 M 手/秒 ⇒ 全量 **1–4 CPU-分**；純 Python 棋盤約 20–50 k 手/秒 ⇒ **1.2–3 CPU-小時**，可用多進程切局平行，於 compute node 以 sbatch 跑數分鐘。
- **diamond（12 點）**：4^12 = 16.7M 表太大（或用 3 態 + 邊界旗標），改為每手做 16 次變換取最小 24-bit code（≈ 200 ns/手，全量 ≈ 1 CPU-分）。**gridcular d≤6（29 點）**：16 組 Zobrist XOR（≈ 500 ops/手）仍 < 1 µs/手。
- **儲存**：每手 1 個 uint16（3×3）或 uint32（hash）⇒ 0.4–0.9 GB；與現有 per-move likelihood 表對齊即可。
- **維度與稀疏度**：每位玩家己方手數 ≈ 5–20 局 × 110 手 = 550–2200 手；3×3 的 1107 格平均每格 0.5–2 次 ⇒ 直方圖極雜訊；必須（i）log1p 或平方根壓縮、（ii）Laplace / Dirichlet 平滑、（iii）以訓練集全域頻率排序取 top-N（N=200–1000）並把其餘併入「其他」桶、或（iv）feature hashing 到 2^12 並接 PCA/線性瓶頸。
- **接法 A（MLP）**：per-game 直方圖 h_g（除以該局己方手數）→ log1p → question-level 平均（不做跨局 std：style-consistency 已死）→ 與現有特徵串接。維度 +1107（或 +N）。
- **接法 B（sequence model）**：per-move canonical id → embedding（1107×d，或 hashed 2^14×d）與 trunk vector 相加/串接；讓模型自行決定是否利用。幾乎零額外成本，最不會傷基線，但也最可能被 trunk 吸收。
- **接法 C（rank-reference bag-of-patterns）**：用訓練集估 P_r(p)（r = 10 類，Dirichlet 平滑），每位玩家計算 L_r = (1/M) Σ_t [log P_r(p_t) − log P̄(p_t)]，得 10 維「邊際 pattern likelihood profile」，形式與 humanv0 的 18-profile likelihood 平行，直接可比、可做消融。也可擴充為 P_r(p | colour, HD) 以處理顏色與讓子。
- **接法 D（bigram / Georgeot 邊）**：對每手找「其後 Chebyshev ≤4 內第一手」的 pattern，計數 (p_t → p_next) 兩種版本：己方→對方（應對）與己方→己方（連貫）；hash 到 2^12。這是 sequence model 看不到的跨局邊際量；也可做 Kandiah 式 r_j 距離到各等級參考網路（10 維）。
- **接法 E（詞彙 / 熵）**：固定抽樣 M₀ 手（如 500）後計 distinct 3×3 數、Shannon 熵、top-50 覆蓋率、每 100 手新 pattern 率（Heaps）；3–5 維、幾乎免費，但依 KQ3 預期弱且非單調。
- **對手側**：資料含雙方手順 ⇒ 同樣算對手的 A/C 特徵，與既有 opponent features 並列（對手等級與我方等級相關）。

### Gaps
- 沒有文獻給出 pattern 抽取的 wall-clock 數字；以上為估算。
- 本專案 sequence model 的具體輸入格式（每手 token 維度、是否有位置/顏色 embedding）未查閱 repo，接法 B 的細節需對照現有程式。

---

## KQ5. 風險：與 trunk vector / likelihood 的冗餘、稀疏詞彙過擬合、測試集顏色限制

### Takeaway
最大風險是冗餘：trunk vector 已是局部形狀的分散式表示、humanv0 likelihood 已含「該等級在此局面下的典型選擇」，Moudřík 的其他特徵族在 pattern 之後只剩 0.13 RMSE 的增量正好說明「邊際統計彼此高度重疊」。第二是稀疏：Baudiš–Moudřík 在 9 局時 MSE 從 5.5 升到 21.3、且必須移除 contiguity 特徵才能在小樣本上穩定。第三是顏色/讓子：pattern 經黑白互換歸一後 id 本身是顏色中性的，但**pattern 的邊際頻率強烈依賴執黑/執白與讓子數**（讓子局白方侵入多、黑方多星位石），Moudřík 刻意只用無讓子互先局；本專案的測試集若有顏色限制，必須以顏色與 HD 分層計算或當條件變數，否則模型會把顏色/讓子分佈差異誤學成等級。

### Cited Findings
- Moudřík：pattern 單獨 RMSE 2.788，全部特徵 2.659（其他五族合計僅 −0.129）；資料僅 19×19、實力相近、**排除讓子局**。 — [arXiv 1512.08969 PDF](https://arxiv.org/pdf/1512.08969)
- Baudiš & Moudřík：9 局 k-NN MSE 21.343（σ=4.62）vs 85 局 5.514；為減少小樣本 pattern 多樣性而移除 contiguity 特徵；用戶自報等級可能不一致。 — [ar5iv 1209.5251](https://ar5iv.labs.arxiv.org/html/1209.5251)
- Kandiah：diamond 空間 193,995 個 canonical pattern 中 96,771 從未出現；兩個同為 6d/6d 的網路也有差異（同等級樣本變異不可忽略）。 — [ar5iv 1405.6077](https://ar5iv.labs.arxiv.org/html/1405.6077)
- Georgeot：pattern 定義把「黑下在某 plaquette」與「白下在顏色互換的 plaquette」視為同一節點（顏色中性 id）。 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- Coquidé：Fuego 局數較長造成連結總數偏多，是比較網路時的已知偏差來源（手數長度混淆）。 — [ar5iv 1707.04044](https://ar5iv.labs.arxiv.org/html/1707.04044)
- Moudřík 2016 CNN 直接從局面學 3 類棋力達單局 77.9%，顯示局面→棋力的訊號可被卷積表示吸收。 — [Springer LNCS 10071](https://link.springer.com/10.1007/978-3-319-49001-4_15)
- 本專案記憶：dataset v1.1 加入 HD[n] 讓子標記，HD≥2 的棋盤缺讓子石（需在規則引擎中補星位石，否則 3×3 pattern 在角部全錯）。 — 專案記憶 aicup-2026-1-go-project.md（非公開來源）

### Inferences
- **冗餘檢驗設計**：先做接法 C（10 維）與 E（3–5 維）加到 MLP 的消融 — 若 fold-0 無增益，就不必投入 A/B/D；接法 B 幾乎免費，可直接加進 sequence model 當 sanity check（若 trunk 已含該資訊，B 的 embedding 梯度會接近零）。
- **稀疏過擬合緩解**：top-N 截斷 + Dirichlet 平滑 + log1p；以「每手」而非「每局」歸一（本資料集局長差異大、含認輸）；對 5 局與 20 局玩家做手數匹配（固定抽樣或加入 log(手數) 作共變量）。
- **顏色 / 讓子**：pattern 直方圖分別以（colour, HD）條件化後再比較；或把顏色、HD 當模型輸入讓它自行校正；讓子石必須在引擎中實際擺上（否則角部 pattern 與氣數全錯）。測試集顏色限制若使某顏色樣本稀少，C 法的 P_r(p | colour) 估計會不穩，宜回退到合併顏色 + 顏色 one-hot。
- **手數長度混淆**：弱者對局常早認輸或下到終局填子，pattern 組成（官子形 vs 中盤形）隨局長改變；需以手數分段（如 Moudřík 的 1–10 / 11–64 / 65–200 / 201+）或以每局手數歸一。
- **整體判斷**：相對強 policy-likelihood 基線，pattern-usage 特徵的期望增益為零到微小、風險低（主要是浪費算力與增加過擬合維度）；若要測，優先 C（最低維、最可解釋、與 likelihood 形式對齊）與 D（唯一真正跨局、sequence model 看不到的量），其餘不建議。

### Gaps
- 不知道本專案測試集顏色限制的確切形式（只知「test-set colour constraint」），無法精確評估分層後各格樣本數。
- 沒有任何已發表實驗把 pattern 頻率加到深度 policy / trunk 基線上做消融，上述冗餘預期無法引用數字。
