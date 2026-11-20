# 其他遊戲／運動中「從少量對局記錄預測實力等級」的他人解法

研究日期：2026-11-05。對照的我方任務：圍棋，每位玩家 5–20 局（只有著手，無時鐘／結果／日期），10 個有序等級，計分 exact=1、相鄰=e^-1；現行系統 = 強棋力預訓練網路的「段位條件化」每手策略似然 + 以整題訓練的小型序列模型，12 模型 ensemble，CV 0.726、Public 0.7625；殘餘誤差約 2/3 為每位玩家持續性（標籤 vs 表現不符）、1/3 為抽樣誤差。

說明：Kaggle 頁面為 JS 渲染，直接抓取只得標題；下方 Kaggle 討論串內容是透過 reader proxy 取得的全文，數字以該全文為準。凡是只在搜尋摘要層級看到、未能打開全文核實的敘述，皆標註「(snippet)」。

---

## 1. Kaggle「Finding Elo」(2014–15)：引擎特徵 → 梯度提升／堆疊；冠亞軍差異；雙玩家處理；MAE

### Takeaway
冠軍 elyase 的關鍵不是模型而是「把目標拆成兩位玩家的平均 Elo 與黑白差」再用 Stockfish 評分差統計量 + 棋局位置特徵，最終是 ExtraTrees／ElasticNet／autoencoder+線性回歸的平均；亞軍 dsjoerg (David Joerg) 的獨家武器是「開局資料庫查表」(同一開局序列的歷史對弈者 Elo 統計) 加上「較差一半著手的中位數損失」特徵與更深的 Stockfish 搜尋。排行榜 MAE：1st 155.78、2nd 160.68、3rd 167.83、4th 178.76；比賽是「單局估 Elo」，沒有跨局聚合。

### Cited Findings
- 比賽定義：以「單一棋局的著手」估計雙方「對局當時的 FIDE Elo」，資料為「50,000 局精英、有等級的棋手對局」，Kaggle 事先用引擎對每手評分 — [Kaggle Finding Elo overview](https://www.kaggle.com/competitions/finding-elo/overview)
- 資料切分：25,000 局訓練(含雙方 Elo)、25,000 局測試(Elo 隱藏)；官方附 Stockfish 逐手 centipawn 評分檔 — [jwnorman/Chess_ELO (參賽者 README)](https://github.com/jwnorman/Chess_ELO)
- 最終排行榜前十 (分數 = 誤差，Elo 點數；頁面未標示指標名稱，討論串中選手以「score points」稱之)：1 elyase 155.77762；2 David Joerg 160.68057；3 Peter Hendrix 167.82660；4 Bats & Robots 178.76241；5 Marger 178.89668；6 Yury Ustinovskiy 179.70910；7 Mihai Kondort 181.96470；8 Dave Spencer 183.15446；9 JakeM 183.71169；10 DataCanary 184.70473 — [Kaggle Finding Elo leaderboard](https://www.kaggle.com/c/finding-elo/leaderboard)
- **1st (elyase)**：兩組特徵——(a) Stockfish 評分差的統計量 (mean absolute、std 等)，(b) 棋局位置特徵 (終局剩餘子、皇后首次移動、首次將軍的手數等)；**目標改為分別預測「該局平均 Elo」與「黑白差」**(此想法來自其他貼文者，稱「讓演算法的工作更容易」)；快速測試中預測平均 Elo 的 MAE「約 60」；最終模型 = ExtraTreesRegressor、ElasticNet、H2O autoencoder(深度特徵萃取)+線性回歸 三者平均，皆用同一組特徵；自稱「特徵選擇大概是運氣」、只在截止前兩天認真做 — [Kaggle 討論串 13008 (全文)](https://www.kaggle.com/c/finding-elo/discussion/13008)
- **2nd (David Joerg, dsjoerg)**：用 OpeningMaster OM GOLEM 開局資料庫，對每局找「最深、且至少 10 筆完全相符」的開局節點 (至多 24 ply)，取這些歷史對弈者 Elo 的平均/最小/最大/標準差作特徵；Stockfish 每手 16 秒 (他說 2 秒只會差「1.1 分」)；centipawn 特徵 = 該玩家「較差一半著手」損失的中位數；把 OLS 預測值當特徵餵進 RandomForestRegressor (拿掉會差「3–4 分」) — 同上討論串
- **3rd (Peter Hendrix)**：R GBM 白、黑各一個，10-fold CV，最終 7,500 棵樹；Stockfish 重跑 2.5 秒/手 (LB 進步「約 1.5–2 分」)；開局 1–20 ply 精確比對外部資料庫 (PGN Mentor、Million Base 2.2，170 萬局) 的 Elo 統計；加 FICS 線上對局進步「不到 1 分」 — 同上
- **4th (Snow Dog)**：不用外部資料、不重跑 Stockfish；特徵依開局 (ply 1–8)/中盤 (8–28)/殘局 (28–) 分段；前 10 ply 做 hash；評分裁剪 ±20000；R gbm quantile loss (alpha=0.5)、depth 5、shrinkage 0.001、13,000 棵樹、5-fold CV；「雙方約 177」 — 同上
- **8th (Dave Spencer)**：Stockfish 深度 19/13/7 三種，RandomForestRegressor；六個模型 (白/黑 × 勝/負/和)；棋局分 5 段各 20 ply；特徵擴充 x²、√x、log(1+x)；約 183–184；xgboost 與 Vowpal Wabbit 對他無效 — 同上
- 14th (simonmab)：SPADE 序列探勘找三個 Elo 區間各 8,000 條鑑別性著手序列，GBM+ExtraTrees，白黑分開；「預測平均與差」對他無效。37th (Triskelion)：每局 mistake/inaccuracy/blunder 計數 + 開局特徵 — 同上
- 搜尋摘要層級的另一版本把「depth 19/13/7」「2.5 秒/手」「GBM+ExtraTrees」歸給 elyase，與 reader proxy 全文不一致 (全文把這些分別歸給 8th 與 3rd)；以全文為準 (snippet 衝突) — [搜尋摘要來源同一討論串](https://www.kaggle.com/c/finding-elo/discussion/13008)

### Inferences
- 名次差距的主要來源是「特徵設計」而非模型：1st 的目標重參數化 (mean, diff)、2nd 的開局查表、3rd 的外部資料庫比對，都是把「對手強度／對局脈絡」顯式建模；單純加深引擎搜尋只值 1–2 分 (約 1%)。
- 兩位玩家的處理有兩派：(a) 白黑各訓一個模型 (3rd、8th、14th)；(b) 預測 (平均, 差) 再還原 (1st)。1st 的做法等於承認「單局中兩位玩家的 Elo 高度相關 (配對效應)」，先估共同成分再估差異。
- 所有解法都是單局 → 單一 Elo 回歸 (MAE)，沒有「同一玩家多局」的聚合問題；最佳 MAE ≈ 156 Elo 相對於 FIDE 精英棋手的 Elo 分布，顯示單局資訊的天花板不高。
- **對圍棋任務的可移轉性**：
  - 已有的：我們的「段位條件化策略似然」本質上就是 Finding Elo 的「引擎評分差統計量」的更強版本 (以人類風格而非最佳著手為尺度)。
  - 未試而值得試：(1) 「(雙方平均段位, 本方−對方差)」的目標重參數化——若題目檔含對手著手，可同時估對手段位當 nuisance，1st 名的經驗是「讓演算法更容易」；(2) 「較差一半著手的中位數損失」這類穩健統計 (2nd)，對應我方可用「每局似然最差 50% 著手的中位數」等 robust aggregate，而非只用均值；(3) 分段 (開局/中盤/官子) 各自統計 (4th、8th)，若尚未做。
  - 不可移轉：開局資料庫查表 (圍棋題目是匿名、無外部段位資料庫可對)。

### Gaps
- Kaggle 的 Evaluation 頁面未能渲染出指標公式 (白黑誤差如何合併)；只能從選手用語與 4th「雙方約 177」推斷為兩位玩家 Elo 的 MAE。
- 未找到 2014–15 後對 Finding Elo 的系統性重分析論文；後續同題研究 (見 §2 的 Tijhuis 2023、RatingNet 2024) 皆用不同資料集，無法直接對比 MAE。
- elyase 的最終 CV/LB 數字只有「平均 Elo 快速測試 MAE 約 60」一句，無完整表。

---

## 2. Maia / Maia-2 與「從著手估 Elo」：人們如何用移動預測器反推等級；數字

### Takeaway
Maia 原論文已顯示「每個 Maia-k 的著手吻合率在 k 附近達峰 (約 46%→52%+)」，這是用「段位條件化策略似然」當等級估計器的根據；公開的實作 (fastelo) 正是以 12 個 Elo 條件下的每手 log-likelihood (逐手截斷) 取峰值+寬度再做單調偏差校正，20 手約 ±300 Elo、40 手約 ±200 Elo，並明確區分「統計誤差 (多局會縮)」與「系統誤差 (多局不縮)」。直接回歸法 (RatingNet, CNN-BiLSTM) 單局 MAE 182 (含時鐘)／239 (無時鐘)，基準 346；Chess.com 的 accuracy→Elo 線性關係需約 50 局才對個人穩定。

### Cited Findings
- Maia (KDD 2020)：每個 rating bin (1100–1999，每 100 一個) 各訓一個模型，各用 1,200 萬局；測試每 bin 10,000 局 (排除 Bullet/HyperBullet，丟前 10 ply 與剩餘時間 <30 秒的著手)；**每個 Maia 的著手吻合率在自己的 rating band 達峰**：最低約 46% (Maia 1900 預測 1100 棋手)，最高 >52%；Maia 1100 對 1100 棋手 >50%。衰減版 Stockfish 吻合率 35–40% 且隨等級單調上升、Leela 亦單調，故無法「捕捉特定等級」 — [CSSLab Maia KDD blog](https://csslab.cs.toronto.edu/blog/2020/08/24/maia_chess_kdd/)
- Maia 部落格並未宣稱 Maia 可估等級或識別個人 — 同上
- **fastelo (HuggingFace 模型卡)**：policy-only transformer (64 格 token + 3 context token，d=256、8 層，約 6.93M 參數)，輸入含棋局、走子者 Elo、對手 Elo；先從 Leela 網路蒸餾 3 億個 Lichess 真實局面，再以 5 億個人類 rapid/classical 局面 (2025/8–11，過濾後 4,270 萬局) 訓練；Top-1 吻合 53.9%。**等級估計法**：對每位玩家的每一手計算 p(move | position, rating) 於 400–2600 共 12 個 rating 輸入；每手 log-prob 貢獻設上限 (一手大錯不能主導)；取曲線均值與寬度；套用以「手數」為參數的單調偏差校正表；輸出含不對稱統計誤差與系統誤差，**系統誤差在同一玩家所有局間共享，局數增加不會縮小**；多局合併時依各局長度的典型誤差加權。精度：單局 20 手後約 ±300 Elo，40 手後約 ±200；20 手閉合測試 (獨立對局) 校正後偏差：400–600 帶 +28、1000–1200 帶 +15、1600–1800 帶 +15、2400–2600 帶 −59 Elo；統計誤差約 −137/+127 至 −326/+270；5 局合併誤差 59–138 Elo；1σ 覆蓋率 49%–81%。曾做過「整局網路」單局誤差約 190 Elo，但因「依賴對手強度」而棄用 — [irsotarriva/fastelo](https://huggingface.co/irsotarriva/fastelo)
- **RatingNet (Omori & Tadepalli, arXiv 2409.11506)**：12 平面棋盤 → 4 層 CNN (BN、mean pooling) → 串接時鐘特徵 → BiLSTM → 2 層 FC 同時輸出雙方 Elo；約 120 萬局 Lichess (2021/4–2024/7，每月 3 萬局)，80/20 切分；Elo 以 mean 1514、sd 366 標準化。測試 MAE：含時鐘 182、無時鐘 239、均值基準 346；Classical 151/185/246；Bullet 182/274/340；時鐘使平均 MAE 降 57 點 (24%)，Bullet 降 92 點 (34%)；MSE 56,618 / 91,619 / 183,538。每手都輸出預測但標籤只有每局一個；高分例 (2922/3163) 被預測為 1459/1460，歸因於高分樣本少。只有 53% 的訓練局由高分方勝出 — [arXiv 2409.11506v2](https://arxiv.org/html/2409.11506v2)
- RatingNet 文中描述 Tijhuis et al. (2023) 為「30 個手工特徵 + random forest 與 SVM，玩家分成 10 個 rating 區間」(資料不公開，未直接比較) — 同上
- **Tijhuis, Mavromoustakos Blom, Spronck, IEEE CoG 2023「Predicting Chess Player Rating Based on a Single Game」**：從單局著手萃取 30 個特徵；「比較資料集兩極端時，預測 rating bracket 準確率 79.3%」，bracket 變化時準確率下降；最重要特徵「同時是理論相關與引擎相關」 — [Tilburg University record](https://research.tilburguniversity.edu/en/publications/predicting-chess-player-rating-based-on-a-single-game/)
- **Chessable「Rating Your Moves」(Jim Yang, 2025 Chessable Research Award)**：資料為數百萬局線上 5+0 blitz，約 400–2400 Elo，標籤 = 該局雙方平均 rating；每個 ply 一筆資料，棋盤+著手編成 64 層 (子位、控制格、起訖格、前後控制樣式)；模型對每手輸出「rating level 的機率分布」，**逐手以簡單貝氏更新合併**；訓練/驗證各 rating band 數量約略均衡，另有保留自然偏斜分布的 real-world 測試集；目前不用時鐘與引擎評分 (列為未來工作)；文中無準確率數字。引述先前工作：棋理特徵、時間使用、引擎著手品質三類資訊各有獨立預測價值 — [Chess.com/Chessable blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)
- 搜尋摘要稱該專案「模型看到每位玩家更多局時表現急升、之後趨平」，但打開全文未見此句或數字 (snippet，未核實) — 同上
- **Chess.com accuracy→rating (hissha blog)**：Rapid accuracy ≈ Rapid Elo/100 + 64 (一個 accuracy 點 ≈ 100 rapid Elo；blitz 3+0 約 128、rapid 約 93–102)；1600+ 的中位誤差 0.39 accuracy (≈39 Elo)，最大 1.2 (≈120 Elo)；20 位樣本棋手預測中位誤差 48.5 Elo、無人差 ≥140；**個人估計需「約 50 局競技對局」，30 局「已不錯」**，分組研究每組要 80 局；對手遠強時低估約 100 Elo、遠弱時高估約 100；高和棋率會高估；白方 accuracy 平均比黑方高約 1.12 — [Accuracy And Ratings On Chess.com](https://chess.com/blog/hissha/accuracy-and-ratings-on-chess-com)
- 原始 Maia 為 1100–1900 每個 band 一個模型；Maia-2 為跨等級的單一模型，含「skill-aware attention」 — [Chessable blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)

### Inferences
- 我方的「段位條件化策略似然」就是 fastelo 路線；fastelo 的三個工程細節值得對照是否已做：(1) **每手 log-likelihood 截斷** (cap) 防單手大錯主導；(2) **以手數為參數的單調偏差校正** (曲線峰值對長短局有系統偏移)；(3) **把誤差拆成統計 (隨局數縮) 與系統 (不縮) 兩塊**——fastelo 的「系統誤差」概念正好對應我方「2/3 持續性誤差」；多局聚合只能消抽樣誤差，這與我方觀察一致。
- fastelo 棄用「整局網路」的理由——「依賴對手強度」——與 Chess.com 的 ±100 Elo 對手偏差、Finding Elo 的 (mean, diff) 重參數化三者互相印證：**對手強度是等級估計最大的混淆變數**。若我方題目檔含對手著手，應檢查模型是否在用對手棋力代替本方棋力；可試「同時估對手段位並當條件」或「本方似然減對手似然」的差分特徵。
- 時鐘在西洋棋貢獻 24% MAE 下降；我方無時鐘，這部分天花板無法突破，不必在此找增益。
- Chessable 做法 (逐手機率分布 + 貝氏逐手更新) 是「天真貝氏」聚合，我方的序列模型是其可學習版；可把天真貝氏乘積 (加溫度) 當 ensemble 的一個成員或 sanity baseline。

### Gaps
- 未找到 Lichess 官方「estimated rating」實驗或「Elo from Maia agreement」的部落格數字；Maia 系列論文本身不做等級估計。
- Tijhuis 2023 的 10 類整體準確率、鄰近準確率、各 bracket 數字只在 PDF 內 (未取得)。
- Chessable 專案無任何準確率數字，「局數 vs 準確率」曲線無法核實。
- 未見把 Maia-2 的 skill-aware 條件化直接反轉為 rating 似然估計並報告 MAE 的公開論文。

---

## 3. StarCraft II SkillCraft (Thompson et al. 2013) 與後續：聯盟預測的特徵、分類器、各聯盟準確率、個人一致性

### Takeaway
SkillCraft 以 3,360 名玩家「各一局」的動作/注意力特徵 (Action Latency 最強，APM 在最高段最強) 用 conditional inference forest 做兩兩聯盟分類，且**只比較相隔兩級的聯盟，因相鄰聯盟重疊太大**；原作者承認無個人內變異估計。後續單局 7 聯盟多類預測：Avontuur 2013 加權準確率 47.3%、Chen et al. 2020 加入巨觀經濟特徵 (Spending Quotient) 達 **61.7% 精確／94.5% ±1 級**、GIST 論文以 46,398 局達 75.3% (觀察越長準確率越高)。

### Cited Findings
- SkillCraft 資料：3,360 名玩家、7 聯盟 (Bronze 167、Silver 347、Gold 553、Platinum 811、Diamond 806、Masters 621、Professional 55；Grandmaster 排除)；**每人一局** replay (職業為公開賽事對局)；限 1v1 排位、>5 分鐘 — [Thompson et al. 2013, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075129)
- 特徵約 15 個：Perception-Action Cycle (Action Latency、每 PAC 動作數、PAC/分、PAC 間隔)、熱鍵 (Hotkey Selects/Assignments/Unique Hotkeys)、複雜單位/技能使用、注意力控制 (小地圖攻擊/右鍵、探索面積)、工人/分、Unique Units、APM — 同上
- 方法：R `cforest` 1,000 棵樹，**兩兩聯盟分類且只比相隔兩級者 (Bronze–Gold、Silver–Platinum、Gold–Diamond、Platinum–Masters、Diamond–Pro) 加 Bronze–Pro；理由是相鄰聯盟重疊太大無法分開**；25 次 70% 隨機子樣本；重要度以隨機雜訊變數第 95 百分位為門檻 — 同上
- 準確率 (括號為多數類基準)：Bronze–Gold 82.32% (77.81)；Silver–Platinum 78.27% (70.03)；Gold–Diamond 79.01% (59.31)；Platinum–Masters 80.70% (56.63)；Diamond–Pro 96.75% (93.61)；Bronze–Pro 98.59% (75.23) — 同上
- 特徵重要度：Action Latency 在 Gold–Diamond、Platinum–Masters 最強；Diamond–Pro 中 APM 25 次皆勝過 Action Latency (後者第 4)；Hotkey Selects 在 Bronze–Pro 第 2；工人/分在低段重要、高段不重要；12 個變數至少在一個分類器有用，6 個在全部五個有用 — 同上
- **個人一致性**：「作者沒有個人表現變異的估計」，設計無法捕捉個人層次變化；單局未必代表典型表現；聯盟是序數測量、限制連續描述；篩選法未控制 family-wise error — 同上
- Avontuur, Spronck, van Zaanen (AIIDE 2013)：1,297 局 replay；SMO 分類器加權準確率 47.3%，加權基準 25.5%；目標是從遊戲內行為預測玩家所在聯盟 — [AIIDE 2013 (DOI 10.1609/aiide.v9i1.12682)](https://ojs.aaai.org/index.php/AIIDE/article/view/12682)；結論「模型可在對局早期就預測實力」(snippet) — [scite 作者頁](https://scite.ai/authors/tetske-avontuur-6nd6AX)
- Chen, Aitchison, Sweetser (AI 2020, LNCS 12576 pp. 256–268)：先前研究過度依賴 APM 等微操特徵、效果有限；加入巨觀經濟指標 Spending Quotient 後，**從單局 replay 預測精確聯盟 61.7%、±1 聯盟 94.5%**，先前 SOTA 47.3% — [ANU research portal](https://researchportalplus.anu.edu.au/en/publications/improving-starcraft-ii-player-league-prediction-with-macro-level-/)
- GIST 碩士論文 (Feature Extraction for StarCraft II League Prediction)：收集 46,398 局 replay，比較「從 replay 六個區段萃取的特徵」，k-NN 與 Random Forest，準確率 75.3% (對照先前 47.3%、61.3%)；主張可取代「五場定位賽」 — [GIST 11553 PDF (僅 metadata/摘要可讀)](https://scholar.gist.ac.kr/bitstream/local/11553/2/38537.pdf)；另一 GIST 紀錄摘要稱約 67%、「較長的 replay 區段準確率較高」(snippet) — [GIST 33192](https://scholar.gist.ac.kr/handle/local/33192)
- 業餘專案 (UCI SkillCraft1 表格資料，8 個 LeagueIndex)：直接多類分類「score 0.45，因 level 8 資料嚴重不足」 — [StarCraftOracle](https://github.com/VictorGoubet/StarCraftOracle)；二分化 LOW(1–5)/HIGH(6+) 後 RF 準確率 0.8422 vs 無資訊率 0.8032 — [RStudio Pubs 分析](https://rstudio-pubs-static.s3.amazonaws.com/225972_f647c1110d884587921cbb3b51a76f50.html)

### Inferences
- SC2 文獻的一致訊息：**相鄰聯盟本質上重疊**，精確率 ~60% 而 ±1 級 ~95% (Chen 2020)，與我方「exact 1／adjacent e^-1」的計分邏輯及 CV 0.726 的量級相近；提升空間主要在「把相鄰誤判拉回正確格」而非消滅遠距錯誤。
- 歷次進步皆來自新增「另一層次」的特徵 (微操 → 巨觀經濟 → 多區段)，而非換分類器 (SMO→RF/kNN 都是基本模型)；對我方啟示是增加「不同時間尺度／不同語意」的特徵 (開局選擇、官子精度、棋形風格) 而非再加同類模型。
- 「觀察越長準確率越高」(GIST) 與 Chess.com「50 局才穩」一致：我方 5–20 局的題目本身是資訊受限的；題內局數應作為模型輸入 (校正信心)，fastelo 亦以手數參數化校正。
- **可移轉性**：SkillCraft 的「個人一致性未估」是明確缺口；我方已量化 (2/3 持續性) 比這些文獻更進一步，沒有可借的直接解法。

### Gaps
- Avontuur 2013 全文被 AAAI 站台 bot 防護擋下 (Tilburg repository 403)，無法取得特徵表、相鄰準確率與「早期預測」的具體手數。
- Chen 2020 全文 (ANU repository DNS 失敗) 無法取得資料集大小、分類器與各聯盟混淆矩陣、Spending Quotient 定義。
- GIST 論文 PDF 內文為壓縮串流，本機無 pdftotext，無法讀出「六區段 vs 準確率」表。
- 無人在 SC2 研究「同一玩家多局聚合」的效果 (SkillCraft 每人一局)。

---

## 4. Dota 2 / LoL / CS:GO / FPS：從對局資料預測等級；特徵、模型、準確率；多局聚合；標籤雜訊

### Takeaway
MOBA/FPS 的等級預測文獻多為「單局→二分類或回歸」：Dota 2 (OpenDota, 13,416 局) 以 LR/Ridge 達 ROC AUC ~0.79–0.81 (Ancient+ 二分類)，另一 Dota 研究以策略/物品/移動特徵與「分階段切割」用 ANN 達 MAE 0.60 等級、R² 0.85，並指出**預測誤差與玩家真實等級波動一致**；FPS 以鍵鼠輸入統計在幾秒到一分鐘內預測技能 (76–77%)。多局聚合的研究極少 (LoL 以 10 局平均 MMR 特徵達 AUC 0.923 預測季末等級)，「比賽當時等級 vs 目前等級」的標籤雜訊幾乎無人正式討論；Dota 本身的 MMR 有「Rank Confidence」機制，等級標籤天生帶噪。

### Cited Findings
- **Dota 2 ML pipeline (Springer, Discover AI 2024)**：OpenDota API，2020 年 8 月 (patch 7.27c) 排位 All Draft、10 真人、有等級勳章、>15 分鐘；13,416 局 (134,160 筆玩家觀測)，角色唯一性過濾後 12,412 局；標籤二元：Ancient 以上 = skillful (38.66%)，Legend 占 30.08%；特徵：基本/終局/死亡 (存活時間、買活)/戰術 (動作數、TP 卷軸、團戰參與、兵線效率)、英雄 synergy/counter 勝率、隊伍資源分配、前 15 分鐘騷擾傷害、生存效率、英雄熟練度 (與職業平均比較)、時空特徵 (早期覆蓋面積、隊伍散度)；依角色×勝負切 10 子集、min-max；模型 LR、Ridge、SVM、NB、RF、XGBoost；最佳 AUC 0.7852–0.8013，LR 0.7932、Ridge 0.7927，線性模型與 SVM 最佳；合併測試集 precision 0.71、recall 0.585、F1 0.641、AUC 0.8123；宣稱「單局即可預測等級」，對照 Aung et al. 的 LoL 研究需 10 局 (以平均 MMR 為特徵、LR/RF 預測季末等級 AUC 0.923)；**無標籤雜訊 (比賽時 vs 目前等級) 討論、無單局 vs 多局直接實驗**；評估描述前後不一致 (10% 測試+5-fold vs 10-fold) — [Springer s44163-024-00139-y](https://link.springer.com/article/10.1007/s44163-024-00139-y)
- **Dota 2 等級預測 (IEEE 2024 / Chulalongkorn 論文)**：加入玩家策略、購買物品、遊戲內移動特徵，並提出依遊戲階段「match splitting」；最佳模型 ANN (等長切割) **MAE 0.60、R² 0.85**；「預測誤差與真實等級波動一致」；動機之一是代打 (boosting) 使帳號等級難以判斷，故單局預測有價值 (snippet) — [IEEE Xplore 10770753](https://ieeexplore.ieee.org/document/10770753)；[Chula ETD 11904](https://digital.car.chula.ac.th/chulaetd/11904)
- Dota 2 公開 API 不提供 MMR，研究者以 TrueSkill 重建代理 MMR — [arXiv 1805.03285](https://arxiv.org/pdf/1805.03285)
- Dota 2 等級機制 (玩家指南，非學術)：隱藏 MMR + Rank Confidence，信心低時每局 MMR 波動大 — [SteelSeries Dota 2 ranks](https://steelseries.com/en-ee/blog/dota-2-ranks-guide)；勳章在信心達 30% 才顯示；7.33 起改為 Glicko 基礎 — [esports.net](https://www.esports.net/wiki/guides/dota-2-ranks/)
- **LoL**：Stanford CS229 2016 專案以 Riot Stats API 聚合統計 (kills、wins、damage) 用 AdaBoost/SVM、10-fold CV 預測季末等級 (25 類) 與升降方向；準確率未在搜尋結果出現 — [CS229 poster](https://cs229.stanford.edu/proj2016/poster/Jang-PredictingSeasonalRankChangesInLeagueOfLegends-poster.pdf)；LoL 勝負預測文獻 (非等級) 55–75% — [arXiv 2108.02799](https://arxiv.org/pdf/2108.02799)
- **CS:GO 從 Twitch 直播推等級** (arXiv 1907.01615)：425 位有公開等級 + 9,928 位無等級的直播主 (2019/1–4)；影像/音訊/聊天文字的神經特徵；各模態分別估「該期間為頂級等級」的機率，再以**階層貝氏模型跨模態匹配出每位玩家的潛在技能**；以 2019/5 公開等級驗證；摘要無準確率 — [arXiv 1907.01615](https://arxiv.org/abs/1907.01615)
- **FPS 鍵鼠輸入預測技能**：Buckley, Chen, Knowles (IEEE CIG 2013)，Red Eclipse，476 局 log、40+ 玩家，鍵盤與滑鼠輸入可以 76% 準確率建模技能，不需遊戲特定特徵 — [Manchester research portal](https://research.manchester.ac.uk/en/publications/predicting-skill-from-gameplay-input-to-a-first-person-shooter/)；期刊版「Rapid Skill Capture in a First-Person Shooter」(IEEE TCIAIG 9(1) 63–75, 2017)：部分技能指標「只需幾秒鐘的鍵鼠輸入」即可預測，主張比 TrueSkill 這類表現平均系統更快適應 — [arXiv 1411.1316](https://arxiv.org/abs/1411.1316)；RF 5-fold 以鍵盤特徵平均 77.1% (snippet) — 同上 PDF
- CS:GO 以 demo 檔直接預測等級的研究未找到；demo 解析器可從 RankUpdate 事件取得等級 — [demoinfocs-golang discussion](https://www.github.com/markus-wa/demoinfocs-golang/discussions/292)

### Inferences
- MOBA 文獻的等級預測多半把等級當成「可由一局決定」的目標，但 Dota 2024 論文自己的結論 (誤差隨真實等級波動) 與 Rank Confidence 機制說明：**標籤是帶噪的動態量**，單局模型的天花板受標籤雜訊限制——與我方「2/3 持續性誤差」同源，文獻並無解法，只有承認。
- 多局聚合在 MOBA 文獻中是以「跨局平均特徵」(Aung et al. 的 10 局平均 MMR) 進行，仍是 feature-level 平均；沒有人比較 feature 平均 vs 機率乘積 vs 序列模型。我方已有序列模型，領先此文獻。
- FPS 結果提示「低階動作統計」(輸入節律) 在幾秒內就能分級；圍棋的對應物是「著手風格統計」(例如與策略分布的 KL、熵、著手在 top-k 的頻率分布) 而非只用似然；若尚未用「每手 top-k rank 直方圖」類特徵，可補。
- **可移轉性**：階層貝氏跨證據池化 (CS:GO Twitch 論文) 的結構——各來源給「屬於頂級」的機率，再以潛在技能變數合併——對我方就是「每局給 10 類 posterior，再以潛在段位變數合併」，等價於天真貝氏 + 先驗；可當 ensemble 基線。

### Gaps
- 未找到 Overwatch SR 預測、LoL 等級 (非勝負) 預測的有準確率之同儕審查研究。
- 「比賽當時等級 vs 目前等級」的標籤雜訊在任何 MOBA 論文中都未被正式分析；只有遊戲機制說明可佐證。
- IEEE 10770753 全文未能取得 (頁面空白)，MAE 0.60/R² 0.85 僅 snippet 層級。
- 撲克：僅有論壇層級的「以 bb/100 做功效分析，需數萬手」討論，無可靠學術數字 — [PokerStrategy forum](https://www.pokerstrategy.com/forum/postid/346397/)；球拍運動：有以 IMU 分辨職業/次菁英/業餘三級與 ITN 等級 (p<0.001) 的研究，但樣本小、多為實驗室條件，與「從對局記錄估序數等級」的任務距離遠 — [CityU wearable sensor](https://www.cityu.edu.hk/kto/industry/find-new-opportunities-our-ip-portal-0/s/systems-and-methods-using-a-wearable-sensor-for-sports-action-recognition-and-assessment)、[tennis IMU/ITN (UGR)](https://produccioncientifica.ugr.es/documentos/6421d4e87eed003a9ca74fbb?lang=en)

---

## 5. 稀疏觀測下的技能估計：階層貝氏 / 經驗貝氏收縮、TrueSkill 式潛在技能；收縮是否勝過直接分類

### Takeaway
階層／部分池化 (partial pooling) 對「少觀測個體」的好處在文獻中是共識 (少樣本者被拉向母體均值、極端 ML 估計被修正)，TrueSkill 官方估計 2 人對戰需約 12 局 (實務可達 3 倍) 才收斂，追蹤真實實力變化需 50–100 局；但**沒有找到任何在「等級分類」任務上以對照實驗證明階層收縮勝過直接分類的研究**。fastelo 的「系統誤差不隨局數縮小」與我方「持續性誤差」提示收縮能處理的只有抽樣誤差那 1/3。

### Cited Findings
- TrueSkill 官方：各模式所需最少局數 3–20 (16 人 FFA 約 3 局、2 人 FFA 12 局、4 隊×4 人 20 局)，實際可能高達 3 倍，取決於表現變異、對手匹配程度與和局機率；不確定度隨局數縮小，早期大幅調整、後期小幅 — [Microsoft Research TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- TrueSkill 設計者：系統需 50–100 局才能追蹤「實質的實力升降」 — [Smogon 討論 (設計者回覆)](https://wbf.www.smogon.com/forums/threads/tournament-formats-and-rating-systems.29319/latest)
- Ludometrics (arXiv 1811.00673)：階層貝氏模型假設技能在母體中常態分布；稀疏資料下純最大似然會把全勝/全敗者推到極端，小的 ridge 懲罰 (即收縮) 可修正；頻率派演算法對應固定技能、貝氏演算法對應階層模型 — [arXiv 1811.00673](https://arxiv.org/pdf/1811.00673)
- 階層貝氏為團體競賽評個人 (BYU，Microsoft Research 演講)：MCMC 太慢，改用遞迴 Newton-Raphson 近似 — [Microsoft Research video](https://www.microsoft.com/en-us/research/video/hierarchical-bayesian-models-for-rating-individual-players-from-group-competitions)
- 經驗貝氏 Dirichlet-multinomial 估職業飛鏢選手技能模型 — [arXiv 2302.10750](https://arxiv.org/html/2302.10750v3)
- 部分池化教學 (棒球打擊率)：少打數者的獨立估計不合理，池化後被拉向母體；完全池化與不池化都不切實際 — [PyMC hierarchical partial pooling](https://pymc3-testing.readthedocs.io/en/rtd-docs/notebooks/hierarchical_partial_pooling.html)；[Win-Vector Stan 範例](https://winvector.github.io/WVExamples/blog/batting-estimation-stan/baseball_stats/)
- LoL 階層貝氏技能 (arXiv 2403.04873)：以金錢與傷害為技能量測，建模玩家對己方/友方/敵方統計的影響；與 Plus-Minus 及業界個人統計比較，宣稱更能區分職業選手；以 discrimination、independence、stability 評估；**摘要未處理少局玩家** — [arXiv 2403.04873](https://arxiv.org/abs/2403.04873)
- fastelo：統計誤差 (多局會縮) 與系統誤差 (同一玩家所有局共享、多局不縮) 分開報告；5 局合併誤差 59–138 Elo vs 單局 20 手 ±300 — [fastelo](https://huggingface.co/irsotarriva/fastelo)
- Chess.com：個人穩定估計約需 50 局 — [hissha blog](https://chess.com/blog/hissha/accuracy-and-ratings-on-chess-com)
- SkillCraft 原作者：無個人內變異估計 — [Thompson 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075129)

### Inferences
- 我方「以整題訓練的序列模型 + 12 模型 ensemble」已是隱式的部分池化：模型在訓練中學到「5 局的證據該信多少」。顯式階層模型 (每題潛在段位 z ~ 類別先驗，各局 posterior 相乘再乘先驗) 的可期望增益主要在**校正**而非排序；在 exact/adjacent 計分下，校正改善會透過決策規則 (見 §6) 體現。
- 文獻的收縮都是針對「抽樣誤差」；我方 2/3 的持續性誤差 (標籤≠表現) 是「每位玩家的隨機效應」而非「噪聲」，任何只用該玩家對局的方法都無法消除；唯一可能縫合的是**先驗結構**：若「標籤段位 − 表現段位」的偏差在母體中有系統性 (例如高段者標籤偏高、某些段位長期不更新)，可從訓練集估「表現→標籤」的轉移矩陣 P(label | performance) 並在解碼時乘上去 (這是一種 label-noise-aware decoding，見 §6 推論)。
- TrueSkill 數字 (2 人對戰 12 局才收斂、追蹤變化需 50–100 局) 說明：5–20 局本就不足以消抽樣誤差，故**題內局數應明確進入模型／校正**；局數少的題目應更向先驗 (訓練集段位邊際) 收縮。

### Gaps
- 沒有找到「階層收縮 vs 直接分類」在少觀測等級預測任務上的對照實驗 (esports 或棋類)；所有來源皆為機制說明或教學。
- 未找到以「P(label | true performance) 轉移矩陣」做標籤雜訊校正的 rating 文獻；此為我方自行推論。

---

## 6. 有序、帶噪標籤的競賽實務：序數回歸 vs 分類；門檻最佳化、期望值解碼、鄰近標籤平滑；哪些一貫有效

### Takeaway
QWK 競賽 (Prudential、PetFinder、APTOS、CrowdFlower) 的一致做法是「先得到一個一維連續分數，再在 OOF 預測上最佳化切點」；分數來源可以是回歸 (APTOS 1st: SmoothL1)、多類機率的期望值 (PetFinder 10th: Σ k·p_k)、或**累積二元分類器 P(y>k) 之和** (Prudential 1st，宣稱比回歸器與目標更相關)；切點初始化於訓練標籤分布的分位點，再做座標式貪婪搜尋。附加技巧：以訓練分布強制最低類比例 (PetFinder 10th)、對公開測試集做軟 pseudo-label 再微調 (APTOS 1st: +0.001 private)。警告：門檻在 public/private 間不穩 (Prudential 1st、APTOS 1st 皆靠 LB 探測)，且 QWK 本身獎勵「預測變異數匹配」的量綱效應，未必代表模型更好。

### Cited Findings
- **Prudential 1st (Gábor S.)**：不用回歸器；訓 7 個 XGBClassifier，第 i 個預測 P(Response > i)，7 個機率相加為連續分數，「與目標的相關性比 XGB 回歸器好」；切點初始化於訓練集 Response 分布對應的分位點；對每個邊界試鄰近值、保留最能提升 kappa 者 (各邊界獨立)，反覆到不再變動，自評「相當天真」但「穩健高效」；校正在重複 CV 的 OOF 預測上做；曾嘗試依二元變數分兩群各自校正，結果不穩；外層 k-fold 內再 3-fold 模擬 public/private 切分；觀察到「CV 相近的兩種校正，public 較好的在 private 常較差」；最佳 private 0.68002 但未選到；唯一 ensemble 是取少數解的中位數；特徵工程為每個二元 keyword-value 的目標均值再取 mean/min — [Kaggle Prudential 討論串 19010 (全文)](https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19010)
- Prudential 2nd (Zhurakovskyi)：回歸 → 以 Michael Hartman 公開腳本的「offset」法把回歸預測對映到標籤 — [zhurak/kaggle-prudential](https://github.com/zhurak/kaggle-prudential)；教學重現：offset 最佳化後訓練集 kappa 0.649→0.720 — [bhishanpoudel offset notebook](https://bhishanpoudel123.github.io/Projects/Prudential_Insurance/b05_prudential_modelling_xgboost_linear_poisson_reg_offset.html)；Patrick Kennedy 投影片：fmin 對 offset 初始值敏感、施加順序也有影響、最終 4 層 stack — [SlideShare](https://de.slideshare.net/oldmanpat/gapresentation-59689312)
- **PetFinder (QWK, 5 類)**：原 1st 隊 Bestpetting 後被查出以爬取的真實領養資料作弊 — [The Register 2020-01-21](https://www.theregister.com/2020/01/21/ai_kaggle_contest_cheat/)；10th place：多類 LightGBM 輸出以 `sum([0,1,2,3,4]*prob_matrix)` 轉成期望值，再用自製 QWK-optR 設 1–4 類切點；因開源 optR「預測太少 0 類」，**強制把排序後最低 `int(0.95*len_0)-1` 位以下設為 0 類** (len_0 = 訓練集 0 類數量)；stack 含 LightGBM×6 (多類+回歸)、CatBoost×2、NN×3、線性回歸；最佳單模多類 LGBM LB 0.471、NN 約 0.44；最終 private 約 0.442 (檔名推測) — [chizhu 10th solution](https://github.com/chizhu/kaggle-petfinder-adoption-prediction-10th-solution)；參賽者筆記：回歸輸出需切 5 類，以 Nelder-Mead 最佳化邊界 (XGB+OptimizedRounder kappa 0.407)；「最佳化 RMSE 不等於最佳化 Regressor+OptimizedRounder 的 QWK」 — [Medium PetFinder write-up](https://medium.com/@meilingxt/petfinder-my-adoption-prediction-8c512c2abcb1)
- **APTOS 2019 1st (QWK, 5 級)**：8 個 CNN (inception_resnet_v2、inception_v4、seresnext50、seresnext101 各 2) 簡單平均；**只用 SmoothL1Loss (回歸)**；切點從 [0.5,1.5,2.5,3.5] 改為 [0.7,1.5,2.5,3.5]，由 public LB 探測而非最佳化程序；stage 2 加入 public test 的**軟 pseudo-label**、IDRiD (標籤與 stage-1 預測平均)、Messidor (依真值分組取 stage-1 軟預測平均、離群值裁至組均值、人工設 0.5 dispersion)，每模型再訓 10 epoch；無 CV (2015 資料的 CV 與 LB 不相關)；private：單模 0.923–0.931、stage-1 ensemble 0.934、stage-2 0.935、改門檻後 0.936 — [Kaggle APTOS 討論串 108065 (全文)](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- APTOS 48th：比較回歸與分類後，「回歸 + 門檻處理」較強 (日文投影片) — [speakerdeck icebee16](https://speakerdeck.com/icebee16/aptos2019-48th-solution)
- **CrowdFlower 1st (QWK, 4 類)**：最佳單模 XGBoost linear booster public 0.69322／private 0.70768；最終為「35 個最佳 public LB 提交的中位數 ensemble」public 0.70807／private 0.72189；hyperopt 調參；解碼細節在隨附 PDF — [ChenglongChen/kaggle-CrowdFlower](https://github.com/ChenglongChen/kaggle-CrowdFlower)
- 理論：QWK 曾是多個 Kaggle 高知名度競賽指標 (含 ASAP 作文評分)，可直接最大化 — [Vaughn & Justice, arXiv 1509.07107](https://arxiv.org/pdf/1509.07107)；批評：QWK 準則的解「基於回歸函數，但把預測變異數人為拉到與原變數相等」，扭曲預測而未改善決策函數的統計性質 — [Nedel'ko 2018, Bulletin of Irkutsk State University](https://mathnet.ru/eng/iigum329)；深度序數回歸把 QWK 當 loss 以懲罰遠距誤分 — [arXiv 2010.08376](https://arxiv.org/pdf/2010.08376)
- OSIC Pulmonary Fibrosis 的指標為 Laplace log-likelihood (連續量+信賴區間)，非序數 QWK，不在此比較 (研究者判斷，未另搜)。

### Inferences
- **我方指標與 QWK 的根本差異**：QWK 是全體預測的邊際分布函數 (非逐題可加)，所以「調門檻讓預測直方圖像訓練直方圖」有結構性增益；我方分數 S = 1·[ŷ=y] + e^-1·[|ŷ−y|=1] 是**逐題可加**的，Bayes 最佳決策為 ŷ* = argmax_k [p_k + e^-1 (p_{k−1} + p_{k+1})] (以校正後的 posterior p)。這比 argmax p_k 更傾向選「鄰居機率總和大」的中間類；若目前解碼仍是 argmax p_k 或 expected-rank 四捨五入，這是零成本可試的第一步 (需先做溫度校正)。
- Kaggle 的門檻法仍有用處作為**對照與穩健化**：(a) 把 expected rank E[k] (或 Prudential 式 Σ_k P(y>k)) 當一維分數，在 OOF 上以座標貪婪搜尋 9 個切點直接最大化我方指標；(b) 若 (a) 與 Bayes 決策的 CV 差距很小，表示 posterior 校正良好；若 (a) 明顯較好，表示 posterior 有系統性偏斜 (例如對高段過度自信)，該去修校正而非留在門檻。
- Prudential 1st 的「累積二元分類器 P(y>k) 相加」是一個未試的 head 設計：對帶噪序數標籤，相鄰類混淆只影響一個二元器，噪聲被局部化；可作為序列模型的替代輸出層 (CORAL/CORN 類)。
- PetFinder 10th 的「以訓練分布強制極端類比例」提示：檢查我方預測的 10 類邊際分布是否相對訓練集明顯向中間收縮 (序數模型常見)；若是，分布匹配式切點可把邊緣段位 (1 與 10) 拉回。
- APTOS 1st 的軟 pseudo-label (+0.001) 是小而真實的增益；對我方：以高信心測試題 (例如 max 期望分數 >0.8) 的軟標籤重訓序列模型；風險是放大持續性誤差 (pseudo-label 承襲同一玩家的系統偏差)，故只應在 ensemble 的一部分成員上做並以 CV 監控。
- 標籤平滑向鄰類：APTOS 1st 用的是「與模型預測平均」而非鄰類平滑；文獻中沒有「向相鄰等級平滑」一貫有效的證據——在我方指標下，用 soft target [k−1: e^-1/(1+2e^-1), k: 1/(1+2e^-1), k+1: e^-1/(1+2e^-1)] 訓練等於直接以指標形狀當 loss，理論上合理但無外部實證 (gap)。
- Public/private 不穩定 (Prudential、APTOS) 的教訓：門檻與校正只在 OOF 上定，不以 public LB 微調；我方 Public 0.7625 > CV 0.726 的差距若來自測試分布，門檻法反而可能過擬合 public。

### Gaps
- 未找到任何競賽系統性比較「期望值解碼 vs 門檻最佳化 vs argmax」並附數字；僅有單一隊伍的敘述 (APTOS 48th 說回歸+門檻 > 分類)。
- PetFinder 真正 1st (作弊) 與遞補名次的完整寫法未取得；Kaggle 頁面 JS 渲染，僅 10th 的 GitHub 可讀。
- 「向相鄰類標籤平滑」在序數競賽中的效果無任何來源。
- CrowdFlower 的解碼細節在未取得的 PDF 中。

---

## 總表：各方法 → 我方可移轉性 (研究者判斷)

| 方法 (來源) | 輸入 | 模型 | 多局聚合 | 決策規則 | 報告準確率 | 可移轉性判斷 |
|---|---|---|---|---|---|---|
| Finding Elo 1st ([13008](https://www.kaggle.com/c/finding-elo/discussion/13008)) | 單局著手 + Stockfish 評分差統計、位置特徵 | ExtraTrees/ElasticNet/AE+LR 平均；目標 = (雙方平均, 黑白差) | 無 (單局) | 回歸 (MAE) | LB 155.78 Elo | 高：目標重參數化 (本方+對手, 差) 以處理對手強度混淆；robust 統計 (較差一半中位數) |
| Finding Elo 2nd (同上) | + 開局資料庫 Elo 統計；Stockfish 16 s/手 | OLS→RF 堆疊 | 無 | 回歸 | 160.68 | 低 (無外部段位資料庫)；堆疊線性預測當特徵可試 |
| Maia 吻合率峰值 ([CSSLab](https://csslab.cs.toronto.edu/blog/2020/08/24/maia_chess_kdd/)) | 每手 top-1 吻合 | 9 個 rating-bin 模型 | — | 取峰值 band | 46%→52%+ 吻合 | 已做 (我方等價物) |
| fastelo ([HF](https://huggingface.co/irsotarriva/fastelo)) | 每手 p(move\|pos, Elo) 於 12 個 Elo | Elo 條件化 transformer | 各局依長度加權；系統誤差不縮 | 似然曲線峰值+寬度 → 單調偏差校正 (手數參數) | 20 手 ±300、40 手 ±200；5 局 59–138 Elo | 高：每手 log-lik 截斷、手數參數化校正、統計/系統誤差分離；對手強度為棄用整局網路之因 |
| RatingNet ([2409.11506](https://arxiv.org/html/2409.11506v2)) | 棋盤平面 + 時鐘 | CNN→BiLSTM→雙 Elo 回歸 | 無 | 回歸 | MAE 182 (含鐘) / 239 (無) / 346 基準 | 中：證明端到端序列回歸可行，但時鐘佔 24% 增益 (我方無) |
| Chessable ([blog](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)) | 64 層棋盤/著手編碼 | 每手輸出 rating 分布 | 逐手貝氏更新 | 後驗峰值 | 無數字 | 中：天真貝氏乘積可當 ensemble 成員/基線 |
| Chess.com accuracy ([hissha](https://chess.com/blog/hissha/accuracy-and-ratings-on-chess-com)) | 每局 accuracy | 線性 | 需 ~50 局 | 線性回歸 | 中位誤差 48.5 Elo (≥40 局) | 低 (說明 5–20 局不足；對手偏差 ±100) |
| SkillCraft ([PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075129)) | 單局動作/注意力統計 | cforest，只比相隔兩級 | 無 (每人一局) | 二分類 | 78–99% (兩級距) | 低：證實相鄰級不可分；無個人一致性估計 |
| Chen 2020 ([ANU](https://researchportalplus.anu.edu.au/en/publications/improving-starcraft-ii-player-league-prediction-with-macro-level-/)) | 單局 + 巨觀 Spending Quotient | 分類器 (未詳) | 無 | 多類 | 61.7% exact / 94.5% ±1 | 中：增益來自「新層次特徵」而非模型 |
| Dota 2 pipeline ([Springer](https://link.springer.com/article/10.1007/s44163-024-00139-y)) | 單局 OpenDota 特徵 | LR/Ridge 最佳 | 無 | 二分類 | AUC 0.79–0.81 | 低 |
| Dota 2 ANN ([IEEE](https://ieeexplore.ieee.org/document/10770753)) | 策略/物品/移動 + 分階段切割 | ANN | 無 | 回歸等級 | MAE 0.60、R² 0.85 (snippet) | 中：階段切割特徵；誤差 ≈ 真實等級波動 (標籤噪聲天花板) |
| CS:GO Twitch ([1907.01615](https://arxiv.org/abs/1907.01615)) | 影音/聊天模態機率 | 階層貝氏跨證據池化 | 跨模態潛在技能 | 後驗 | 無數字 | 中：結構 = 每局 posterior × 先驗 |
| FPS 鍵鼠 ([CIG 2013](https://research.manchester.ac.uk/en/publications/predicting-skill-from-gameplay-input-to-a-first-person-shooter/)) | 幾秒–1 分鐘輸入統計 | RF | 無 | 分類 | 76–77% | 低–中：「低階風格統計」(top-k 直方圖、熵) 可補 |
| Prudential 1st ([19010](https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19010)) | 表格 | 7 個 P(y>k) 二元器相加 | — | OOF 上座標貪婪切點 (初值 = 訓練分位) | private 0.680 (未選) | 高：累積二元 head；切點法當對照 |
| PetFinder 10th ([GitHub](https://github.com/chizhu/kaggle-petfinder-adoption-prediction-10th-solution)) | 表格/圖/文 | 多類 LGBM 等 stack | — | 期望值 Σk·p_k + optR 切點 + 強制 0 類比例 | LB 0.471 單模 | 高：期望值分數 + 分布匹配檢查邊緣段位 |
| APTOS 1st ([108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)) | 影像 | 8 CNN 平均，SmoothL1 回歸 | — | 固定切點 (LB 探測) | private 0.934→0.936 | 中：軟 pseudo-label 測試集 (+0.001)；LB 探測不可取 |

### 給我方的具體未試清單 (按預期成本效益)
1. **Bayes 最佳解碼**：ŷ = argmax_k [p_k + e^-1 (p_{k−1}+p_{k+1})]，先做溫度校正；零成本。
2. **一維分數 + OOF 切點最佳化** (expected rank 或 Σ P(y>k))，作為校正診斷與對照；若勝過 1，修校正。
3. **對手強度去混淆**：若題目含對手著手，加入「對手段位估計」作條件或差分特徵；或改目標為 (平均, 差) (Finding Elo 1st；fastelo 棄整局網路之因)。
4. **fastelo 式工程**：每手 log-lik 截斷；以「局數×手數」參數化的單調偏差校正；題內局數進入模型。
5. **累積二元 head (P(y>k))** 替代 softmax，局部化相鄰標籤噪聲 (Prudential 1st)。
6. **邊際分布檢查**：預測 10 類直方圖 vs 訓練集；若邊緣段位被壓縮，用分布匹配切點 (PetFinder 10th)。
7. **軟 pseudo-label 高信心測試題**，只用於部分 ensemble 成員並以 CV 監控 (APTOS 1st +0.001)。
8. **標籤噪聲先驗**：以訓練集估 P(label | performance-rank) 轉移矩陣並在解碼時乘入 (本研究推論，無外部實證)。
9. 階層收縮本身：對 1/3 抽樣誤差有理論上的校正作用，但文獻無「勝過直接分類」的對照證據；優先度低於 1–7。
