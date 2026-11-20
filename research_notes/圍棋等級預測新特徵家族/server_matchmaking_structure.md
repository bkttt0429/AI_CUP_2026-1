# 線上圍棋伺服器的配對、讓子／貼目與升降段機制：與棋力無關的結構性等級訊號

研究日期：2026-11-04。範圍：Fox（野狐）、Tygem（弈城/타이젬）、KGS、OGS、Pandanet IGS、幽玄の間／Igo Online 等伺服器的配對、讓子、貼目、升降段規則，以及已公開的「非著手品質」等級相關統計；目標是從 AI CUP 2026 資料（SGF + 目標方顏色 + HD[n]）中找出可計算的結構性特徵。

資料集既知事實（題目給定，非本研究引用）：100 萬局、8,007 位玩家、10 類（10–12k、7–9k、4–6k、1–3k、1D–6D）、HD0 = 分先貼 6.5 目（79%）、HD1 = 不貼目（17%）、HD2–9 = 讓子（3.6%），無時鐘、無勝負、無日期、無對手身分；測試題為同一玩家 5/10/15/20 局、同一目標顏色、僅 HD0/HD1。

---

## 關鍵問題 1：各伺服器的自動配對怎麼選對手？段差對應什麼讓子／貼目？哪一家的慣例符合「HD0 = 6.5 目分先、HD1 = 不貼目、HD2+ = 讓子且白方較強」？

### Takeaway
「分先 6.5 目／差一級不貼目（先）／差兩級以上讓等同級差的子數」是日韓（及中國傳統）業餘手合慣例，**明確不符合** KGS（差一級用 0.5 目）、OGS（差一級 0.5 目）與 Pandanet IGS（半級一階、1 級差為白方倒貼 5.5 目、永無「不貼目」局）。Fox 以中國規則運作（貼 7.5 目 / 3¾ 子）是本研究未能以一手來源確證的常識，若題目的「6.5 目」為字面事實，則 Fox 機率下降；最吻合的候選是 Tygem 家族（弈城／타이젬）或日式規則伺服器，但沒有任何一家的官方手合表能在本次研究中被直接抓到（見 Gaps）。

### Cited Findings
- **Pandanet IGS（官方）**：段差 0 → 無讓子、黑貼 6.5；段差 0.5 → 黑貼 0.5；段差 1 → **白方倒貼 5.5**；段差 1.5 → 2 子黑貼 0.5；段差 2 → 2 子白貼 5.5；…；段差 8.5 → 9 子黑貼 0.5；段差 9 → 9 子白貼 5.5。每級拆成兩個半級步。自報段位帶「?」，20 局後認證；非 19 路、與職業、與初學者班、段差超過 3 級、自由室對局皆不計分。 — [Pandanet "Rating / handicap" 頁](https://pandanet-igs.com/communities/pandanet/60)
- **Pandanet 賽事規則**：讓子上限 9 子（「difference in rating is more than 9, please set handicap to 9」）。 — [IWAG 2002 rules](https://pandanet.co.jp/event/iwag2002/e/rule-p.htm)
- **KGS（官方）**：以最近 180 天所有計分局連續計算段位；段位 30k–9d，rating 無上限；僅 19 路計分局且**讓子 ≤ 6 子**才計入評分；例子：2d 對 3k 應讓 4 子。 — [KGS rank help](https://www.gokgs.com/help/rank.html)
- **KGS 自動配對（官方）**：自動配對一律日本規則、僅 19 路；有「最大段差」偏好設定；兩方皆勾選 free/rated 時定為 rated；逃跑即標記 escaper。頁面**未**說明讓子／貼目公式與顏色分配。 — [KGS automatch help](https://www.gokgs.com/help/automatch.html)
- **KGS 讓子公式（Sensei's Library，社群維基，可能過時）**：差一級 = 「一子」但以**貼 0.5 目取代 6.5 目**來實現；公式 H = 段差/2 + 0.5，H 為整數時貼 0.5，否則進位且貼 6.5。 — [Sensei's Library ?topic=2464](https://senseis.xmp.net/?topic=2464)（本次 WebFetch 回 403，內容取自搜尋摘要）
- **OGS 自動讓子（論壇使用者整理，非官方文件）**：同級 6.5 目（日本規則）；差 1 級 **0.5 目無子**（「拿掉 6.5 目而不是給一子，0.5 讓和局歸白」）；差 2 級 0.5 目 + 1 子；差 3 級 0.5 目 + 2 子；段差用原始 rating 計算而非顯示整數段。 — [OGS forum "Difference in ranks"](https://forums.online-go.com/t/difference-in-ranks/21772/17)；另一討論串說 1 handicap = 不貼目、2 handicap = 兩子無貼目（中國規則則 1.5 目），段差向下取整。 — [How Automatic Handicap Works on OGS](https://forums.online-go.com/t/how-automatic-handicap-works-on-ogs/44721)；兩者對「差 2 級」的子數說法互相矛盾。
- **OGS 2023 公告**：19 路自動配對限制為 ±9；公告中僅給 9×9/13×13 的讓子貼目表（例：13×13 段差 1 → 1 子、日本規則 2.5 目），未說明分先局顏色分配。 — [Draws, handicap & komi changes](https://forums.online-go.com/t/draws-handicap-komi-changes-and-ranking-limits-lifted/50489)
- **OGS 2021 評分改版**：rank = ln(rating/525) × 23.15；目標是「一級差 = 19 路一子差，不論段位高低」；取消滑動視窗。 — [2021 Rating and rank adjustments](https://forums.online-go.com/t/2021-rating-and-rank-adjustments/33389)
- **日式伺服器 Igo Online（Sakura Entertainment，官方規則頁）**：分先貼 6.5；**段差 1 → 貼 0 目**；段差 2 以上每級加 1 子（或改用每級加 13 目的倒貼）；段差 29 上限。 — [Igo Online game rules](https://www.sakura-ent.net/igoonline-sp/game?lang=en)
- **日本棋院級位認定大會**：互先貼 6 目半、置碁「1 級差 1 子」（搜尋摘要，PDF 未逐字核對）。 — [日本棋院活動 PDF 列表](https://u-gen.nihonkiin.or.jp/download/u-guide_webbook_total.pdf)
- **Fox 自動配對**：客戶端有 blitz（5 分）與 regular（20 分）兩種 automatch；討論串完全沒提段差窗口、讓子、貼目、顏色分配。 — [L19 "Fox Go Server"](https://lifein19x19.com/viewtopic.php?p=259042)
- **Tygem 自動配對（匿名玩家回憶）**：有人把 automatch 設成「段差 ±2，所以最多只會遇到 2 子讓子」；另一人認為「KGS 一子 ≈ Tygem 兩級」。 — [L19 search summary of Tygem threads](https://lifein19x19.com/viewtopic.php?p=193934)
- **一般慣例（Wikipedia）**：差一級不讓子，弱方執黑、強方不收貼目；每多一級加一子；名目讓子高估實質優勢約半子（白方在讓子後先行）。 — [Handicapping in Go](https://en.wikipedia.org/wiki/Handicapping_in_Go)
- **中國傳統慣例（萌娘百科，二手）**：分先黑貼 3¾ 子；讓先不貼；讓子要貼還讓子數的一半。 — [萌娘百科 圍棋](https://zh.moegirl.org.cn/%E5%9B%B4%E6%A3%8B)
- **Tygem 背景**：韓國 TongYang Online 所有，前身 ICBL（曹薰鉉 2000 創）；AlphaGo 用了 Tygem 800 萬個人類局面訓練。頁面無手合／段位制度。 — [Wikipedia Tygem](https://en.wikipedia.org/wiki/Tygem)

### Inferences
- 排除法：HD1 若為字面上的「貼 0 目」，KGS（0.5）、OGS（0.5）、IGS（0.5 或白倒貼 5.5）都不符；HD0 = 6.5 又排除中國規則伺服器（7.5）。剩下 Tygem 家族（韓國規則 6.5、정선＝不貼目）、日式伺服器（幽玄の間、Igo Online 類）。**但若主辦單位把 0.5 目也歸為「不貼目」，KGS/OGS 又回到候選名單**，需以 SGF 的 KM 欄位（若保留）或實際終局數子驗證。
- HD2–HD9 共佔 3.6%，而 Fox/Tygem 式自動配對通常只在 ±1～±2 級內配對（上述玩家回憶），因此讓子 ≥2 的局大概多半來自手動邀約／指導棋，而非自動配對；這些局的對手段位分佈與 HD0/HD1 不同（假說）。測試僅 HD0/HD1，所以 HD2+ 的任何特徵只能作為訓練時的玩家層資訊，無法在測試使用。
- 若伺服器採「段差 = 子數」且 HD1 嚴格對應差一級，則 HD1 局的對手段位 = 本人 ±1 級（方向由顏色決定）；HD0 局對手 = 同級。這使「對手著手品質」在 HD0 局是本人等級的直接代理、在 HD1 局是偏移 ±1 級的代理——可用顏色 × HD 做偏移校正後再平均（現有「own-minus-opponent」部分涵蓋，但未必有做 ±1 偏移校正）。

### Gaps
- Fox、Tygem/弈城、弈客、101weiqi 的**官方**自動配對段差窗口、讓子表、貼目與分先局顏色分配規則：Sensei's Library（403）、百度百科（403）、中文/韓文官方幫助頁皆未取得；Fox 使用中國規則貼 3¾ 子屬本人常識，未取得一手引用。
- 幽玄の間的手合表只找到「依 rating 決定手合」的賽事公告，無官方表。
- 分先局顏色（隨機／猜先）在各伺服器的實作沒有任何官方文件可引。

---

## 關鍵問題 2：在這些慣例下，一位玩家「局的分佈」透露什麼等級資訊？顏色份額、讓子方向、6.5 目份額、典型對手段位；kyu 與 dan 的階梯是否結構不同？

### Takeaway
在「同級分先、差一級先、差 n 級讓 n 子」的伺服器上，玩家 HD1 局中執黑／執白的比例等於相鄰上下兩級人口密度之比，HD0 份額反映本級相對密度；這是與棋力無關、純由伺服器人口分佈決定的等級訊號，但它在測試中受「同一題同一顏色」的取樣方式折損，只剩「HD1 份額 × 顏色」這個已被現有 signed-handicap 特徵涵蓋的方向資訊。升降段機制（Fox/Tygem 式「最近 20 局 14 勝升、13 敗降」）說明大多數玩家在單一段位停留期間的勝率被夾在 35%–70% 之間，而低級別門檻更低、升級更快，造成 kyu 類別的標籤更「跨級」。

### Cited Findings
- **Fox 升降段（部落格作者抄錄客戶端「Ranked game rules」3–4D 列）**：只看目前段位最近 20 局；20 局中 ≥14 勝升一段、≥13 敗降一段；前 20 局 ≥18 勝升兩段、17 敗降兩段；若單升已達成但雙升仍可能，延後至雙升不可能時才升；統計窗只顯示「連勝 8 升／連敗 4 降」。作者稱 Fox 是「分級聯賽制而非 rating」。其他段位列未公開。 — [dfan: Fox Go Server rank calculator](https://dfan.org/blog/posts/2025/10/23/fox-rank-calculator/)
- **Tygem 升降段（玩家回憶，自述「if I remember correctly」）**：以最近 20 局為準，20 局中 14 勝或 14 敗即升降，升降後重新計 20 局；KGS 3k ≈ Tygem/WBaduk 1d（約 4 級差）；另一人 KGS 1D 升到 Tygem 3D 後覺得對局勢均力敵。 — [L19 Tygem/KGS thread](https://lifein19x19.com/viewtopic.php?p=193934)
- **Igo Online 升段點數門檻**：8D–5D 需 200 點、4D–1D 100 點、1K–9K 70 點、10K–18K 50 點、19K–25K 20 點、26K–30K 10 點；每局 ±1 到 ±19 點依段差與對局類型。 — [Igo Online rules](https://www.sakura-ent.net/igoonline-sp/game?lang=en)
- **IGS 計分權重**：段差 0 勝負各 ±100；段差 1 強者勝 84／弱者勝 116；段差 3 為 64/136；段差 ≥3.5 不計分；前 20 局特殊計算、21–50 局 ×150%、50–100 局 ×120%。初學者班 5 勝即升 17k。 — [Pandanet rating page](https://pandanet-igs.com/communities/pandanet/60)
- **線上段位對應（中國，2023）**：「升業 5 網路（弈城/野狐）5D 差不多是均值」、福建約需野狐 7D；線下升段賽每級通過率約 23%（7 局 5 勝）。 — [新浪體育 2023-07-24](https://sports.sina.cn/others/qipai/2023-07-24/detail-imzcucmx9090178.d.html)
- **Fox/Tygem 與 KGS 段位差（論壇軼事）**：「1–2d Fox ≈ 2–1k KGS」、「Fox 與 Tygem 整體比 KGS 弱約 2 子」；OGS 2024 跨伺服器問卷僅 69 筆 Fox 樣本，作者自稱樣本很少。 — [L19 p=175858](https://www.lifein19x19.com/viewtopic.php?p=175858)；[OGS Go Rankings Survey Feb 2024](https://forums.online-go.com/t/go-rankings-survey-feb-2024/50803)
- **OGS 段位人口分佈（2021 非官方直方圖）**：人數在最弱端最高並單調遞減，低段尾巴厚；本頁未附讓子局頻率。 — [Unofficial OGS rank histogram 2021](https://forums.online-go.com/t/unofficial-ogs-rank-histogram-2021/35705)
- **KGS 高段升段偏易（論壇批評）**：「Promotion of KGS high dan ranks is too easy（高段有通膨）」、系統「不試圖維持段距一致」。 — [L19 KGS Ranking adjustment](https://lifein19x19.com/viewtopic.php?p=253302)
- **分先局黑勝率（來源伺服器不明的資料庫，L19）**：6.5 目時黑勝 50.713%，7.5 目時 47.977%；2002–2007 年分先 6.5 目黑勝率 49.1% → 50.7%。 — [L19 "The size of komi"](https://www.lifein19x19.com/viewtopic.php?p=240143)

### Inferences
- **人口密度比假說（可量化）**：令 n(r) 為伺服器在段位 r 的活躍人口。若自動配對在 ±1 級內近乎均勻取對手，則玩家 r 的 HD0 份額 ∝ n(r)、HD1 執黑份額 ∝ n(r+1)、HD1 執白份額 ∝ n(r−1)。因此 P(HD1 執黑)/P(HD1 執白) ≈ n(r+1)/n(r−1)，是段位的平滑函數。OGS 直方圖為單調遞減（弱端多），若資料來源伺服器類似，則低級別玩家更常「執白讓先」、高段玩家更常「執黑被讓先」——方向與現有 signed-handicap 特徵一致，但可進一步把它**校準成每類別的經驗先驗**（訓練集中各類別的 HD1 執黑/執白比例），在測試時當 likelihood 用，而非只當原始特徵。
- **測試取樣的折損**：同一題全部同色，所以「執黑份額」不可用；HD1 份額在執黑題中讀作 n(r+1)/[n(r)+n(r+1)]，在執白題中讀作 n(r−1)/[n(r)+n(r−1)]——兩種題型要用**不同的**類別先驗（條件於顏色），否則訊號被平均掉。
- **升降段門檻的含意**：Fox/Tygem 式 14/20 升、13/20 降，表示停留在某段位的玩家其滾動勝率約在 35%–70% 之間；加上無勝負欄位，勝率本身無法用，但「終局方式」（見問題 3）可部分代理。
- **kyu/dan 階梯差異**：Igo Online 的點數門檻 kyu 明顯低於 dan（50–70 vs 100–200），IGS 初學者 5 勝即升級；dfan 說 Fox 各段位列不同（僅 3–4D 已知）。推論：kyu 玩家每級停留局數少、升級快，因此以「單一類別」標籤的 kyu 玩家其 50 局樣本更可能跨越 2–3 個實際段位 → kyu 類別的標籤雜訊比 dan 類別大（假說，需用資料中各類別的「局間著手品質變異」驗證）。
- 分先局黑勝率 ≈ 50% 顯示顏色在 HD0 局**不**帶勝負資訊；顏色的等級訊號全部來自 HD1/HD2+ 的「弱方執黑」約定。

### Gaps
- 任何一家目標候選伺服器（Fox、Tygem/弈城）的**各段位人口分佈**數字皆未找到（野狐排行榜有反爬蟲保護，見 [clawhub 說明](https://clawhub.ai/zhangbin2025/weiqi-player)）；只能從訓練集本身反推相對密度。
- 未找到任何伺服器「讓子局佔比按段位」的公開統計。
- Fox 3–4D 以外的升降段門檻、Tygem/弈城的官方升段規定皆未取得。

---

## 關鍵問題 3：各伺服器的對局長度、認輸習慣、虛手、超時——可從著手序列計算的部分，有沒有按段位的公開數據？

### Takeaway
公開資料給了基準值（職業平均 211 手；OGS 數子終局 19 路分先局平均 253.5 手；OGS 認輸平均在第 168 手、落後約 63 目；5.9% 的認輸者其實領先），但**沒有任何來源提供按段位分層的對局長度或認輸率**；這些在本資料集上是 trivial CPU 即可算出的特徵（手數、是否以雙虛手結束、最後落子方、尾端虛手數），且 Fox 資料論文的篩選條件（≥50 手、以雙 pass 或認輸結束）暗示 Fox 原始資料中確有大量「斷線／超時」終局可被識別。

### Cited Findings
- **職業對局長度**：平均 211 手；認輸局 186 手、數子局 257 手。 — [van Ebbenhorst Tengbergen (CWI) gostat](https://homepages.cwi.nl/~aeb/go/misc/gostat.html)
- **OGS 大樣本（27,085,332 筆紀錄）**：19 路 12,739,207 局；2,632,713 局「非讓子、有計分結果」的 19 路局平均 **253.5 手**（排除 >500 手；長度 = 著手串長度 − 2）；偶數長度尖峰被歸因於 AGA 規則白方最後虛手；規則欄 Japanese 19.5M、Chinese 7.1M、Korean 0.2M。無按段位分層。 — [OGS forum: Statistics from a 27M game sample](https://forums.online-go.com/t/statistics-from-a-27m-game-sample/39850)
- **OGS 認輸統計（236 位即時對局玩家各一局）**：排除誤判後認輸平均在第 **168.19 手**、平均落後 **63.17 目**（KataGo 估）；**5.93%** 在領先時認輸；樣本多為單位數 kyu，最高 4 dan；作者承認 9 個段位分箱「可能不足以顯示相關性」，未給數字。 — [OGS forum: When do people usually resign?](https://forums.online-go.com/t/when-do-people-usually-resign-some-statistics/52977)
- **KGS 1d 認輸率（匿名小樣本）**：10 位 1d 各 40 局，平均 53% 認輸，極值 77% 與 32%。 — [L19 via search summary](https://www.lifein19x19.com/viewtopic.php?p=210911)
- **Fox 資料篩選（學術論文）**：僅保留 ≥50 手、且「以雙方虛手或一方認輸結束，而非斷線或超時」的局；雙方須同一段位組；每組訓練 5,000 局；Fox 資料約 2017 年。 — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- **KGS 評分排除**：少於 10 手的局通常不計分。 — [KGS rank help](https://www.gokgs.com/help/rank.html)
- **L19 長度直方圖討論（2013）**：觀察到對局長度「雙峰」（約 140 手與 280 手附近），推測與早認輸有關；無數字。 — [L19 histogram thread](https://lifein19x19.com/viewtopic.php?p=142719)

### Inferences
- 可計算特徵：(a) 總手數；(b) 終局型態三分類：雙虛手（數子）、以單一方最後落子結束（認輸／超時／逃跑）、尾端含單虛手；(c) 最後落子者是否為目標方（認輸局中通常**對手**落子後本人認輸，故「最後落子 = 對手」偏向本人輸；但超時／逃跑者是最後未落子方，混淆）；(d) 數子局中尾端虛手數（日本規則兩次、AGA 白方多一次）。這些與認輸習慣（dan 玩家更早、更常認輸；kyu 玩家下到數子）相關的方向是圈內共識，但**沒有引用得到的數字**，應以訓練集自行量化。
- 若資料來源伺服器像 Fox 一樣有大量斷線／超時局，則「極短局（<50 手）佔比」可能帶玩家層的行為訊號（逃跑、網路品質），與等級的相關未知。

### Gaps
- 無任何伺服器的「按段位對局長度」「按段位認輸率」「按段位雙虛手終局率」公開表格。
- 無「誰最後落子」與勝負關係的公開統計。

---

## 關鍵問題 4：Fox/Tygem/KGS 資料集的已公開分析中，有無「非著手品質」的段位相關統計（長度、黑白勝率、讓子頻率、開局選點、三三流行度）？

### Takeaway
三個主要學術資料來源（featurecat Fox 資料集的 2025 強度估計論文、Moudřík 等人的 KGS 10 萬局研究、KataGo HumanSL）都**只以著手為特徵**，沒有報告按段位的長度／顏色／讓子統計；唯一與結構相關的量化結果是「同一玩家多局」的強度估計上限遠低於「跨玩家隨機抽局」（5 局 58.1% vs 64.8%；15 局 63.2% vs 87.5%），顯示玩家個人風格是精準度瓶頸，正是結構性特徵可能補足之處。開局三三等 AI 時代著法的按段位頻率沒有公開數字，但 KataGo 把人類資料分成「preaz（AlphaZero 前）」與一般段位 profile，證實開局分佈同時依賴年代與段位。

### Cited Findings
- **Fox 資料集（featurecat，約 2017 年）強度估計**：11 組（3–5k、1–2k、1d…9d）；每組訓練 5,000 局；過濾條件如上。11 類準確率（±1 類）：隨機抽局 1 局 37.5%(82.8)、5 局 64.8%(98.8)、10 局 79.6%(99.9)、15 局 87.5%(100)、20 局 91.7%(100)；**同一玩家抽局** 5 局 58.1%(95.3)、10 局 61.0%(96.2)、15 局 63.2%(96.6)。無讓子／貼目／長度／顏色統計。因 2017 年「AlphaZero 式開局尚未流行」而使用 KataGo HumanSL 的 preaz profile。 — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)
- **KGS 10 萬局（Moudřík, Baudiš, Neruda 2015）**：20k–6d 共 26 級、每級 120 位玩家、每人 10–50 局；排除讓子局、僅 19 路同等級對局；強度回歸 RMSE：只用 Pattern 2.788 級、全部特徵 2.659 級、僅勝負統計 6.792 級（均值回歸 7.503）；未分析對局長度。 — [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)
- **KataGo HumanSL**：v1.15.0 釋出的人類著手模型「以大量不同段位與年代的人類對局訓練」，可依 rank 或 preaz profile 產生著手分佈；未列出伺服器與每段位局數。 — [KataGo v1.15.0 release notes](https://newreleases.io/project/github/lightvector/KataGo/release/v1.15.0)
- **職業資料集**：PAGE 98,525 局 2,007 位職業棋士 70 年、逐手 KataGo 標註（非業餘、無段位）。 — [arXiv 2211.01559](https://arxiv.org/abs/2211.01559)
- **三三入侵的歷史統計（GoGoD 職業庫，論壇分析）**：AlphaGo 前白方早三三 104 局、黑方僅 10 局；白三三時白勝 59%。無業餘按段位資料。 — [L19 Uberdude's journal](https://www.lifein19x19.com/viewtopic.php?p=221841)
- **分先黑勝率**：見問題 2（6.5 目約 50.7%，7.5 目約 48.0%，來源伺服器不明）。 — [L19 "The size of komi"](https://www.lifein19x19.com/viewtopic.php?p=240143)

### Inferences
- 本資料集的設定（同一玩家 5–20 局）對應論文的「player-specific」情境，其天花板（11 類 ~60%）遠低於隨機抽局，意味著「玩家層固定效應」才是誤差主因；結構特徵（HD 份額、終局型態、長度分佈）是**玩家層**而非局層訊號，理論上與著手品質互補。
- 開局統計（第一手角位、是否 3-3、早三三入侵次數、AI 定式使用）按類別分佈完全可從資料算出；其效力取決於資料年代（AI 後時代高段模仿 AI 開局的比例應更高），年代未知是風險。

### Gaps
- 沒有任何 Fox/Tygem/KGS 公開分析報告「按段位」的：平均手數、黑白勝率、讓子頻率、第一手選點、三三流行度。featurecat/go-dataset 的 README 無法透過搜尋取得。
- 資料集年代不明，無法判斷「AI 式開局」特徵的可用程度。

---

## 關鍵問題 5：匿名化與「每位玩家恰屬一類」的取樣如何洩漏等級？標籤是對局當時段位還是快照？

### Takeaway
在 Fox/Tygem 式「最近 20 局 14 勝升級」制度下，一位進步中的玩家每級只停留 20–40 局，而資料中玩家中位數 51 局、上四分位數以上達數百局，因此「恰屬一類」的標籤**不可能對所有人都是逐局段位**（否則跨級玩家會被切成多個類別）；最可能是「以某時點段位為標籤、再取該玩家全部／部分局」（快照）或「只選段位穩定的玩家」（過濾）。兩種方式都會讓類別內的著手品質分佈隨玩家局數而異，並使 kyu 類別更雜（假說，需以資料檢驗）。

### Cited Findings
- Fox：升降段只看「目前段位最近 20 局」，≥14 勝升、前 20 局 ≥18 勝雙升。 — [dfan Fox rank calculator](https://dfan.org/blog/posts/2025/10/23/fox-rank-calculator/)
- Tygem：以最近 20 局 14 勝/敗升降，升降後重算。 — [L19 Tygem thread](https://lifein19x19.com/viewtopic.php?p=193934)
- IGS：前 20 局特殊計分、21–50 局 150%，即新帳號段位變動快。 — [Pandanet rating page](https://pandanet-igs.com/communities/pandanet/60)
- KGS：段位由最近 180 天計分局連續重算（非離散升降），rating 連續、rank 取整。 — [KGS rank help](https://www.gokgs.com/help/rank.html)
- OGS：2021 後取消滑動視窗，勝則升、敗則降（連續 rating）。 — [2021 Rating and rank adjustments](https://forums.online-go.com/t/2021-rating-and-rank-adjustments/33389)
- Fox 資料論文的作法：以「對局當時雙方段位」篩選同組對局（每局標籤取自 SGF 當時段位），每位玩家取 20 局做 player-specific 測試。 — [arXiv 2505.00279](https://arxiv.org/abs/2505.00279)

### Inferences
- **快照假說**：若標籤為抽取時段位，則玩家較早的局多半是在更低段位下的（進步方向為主），於是「每局著手品質的時間趨勢」本身即含標籤資訊；但資料無日期，除非主辦單位保留了檔案順序（可檢驗：若順序含時間，則同一玩家的著手品質應與序號有單調趨勢）。
- **過濾假說**：若只選段位在抽取期間未變的玩家，則訓練集會系統性缺少「快速進步的玩家」，而測試集若同法產生，則模型對穩定玩家較準；對 kyu 類別（升級快）會過度篩掉進步者，使 kyu 類樣本偏向停滯型玩家（著手品質偏類別下緣）。
- **局數洩漏**：訓練集中「玩家總局數」與類別的關係（dan 玩家是否局數更多）可直接檢驗，但測試題固定 5/10/15/20 局，此訊號在測試不可用，只能用來做訓練時的分層加權。
- **HD1 方向與升降段的交互**：剛升段的玩家（勝率 70% 上下）面對同級對手表現偏強、剛降段者偏弱；這是類別內「跨級雜訊」的來源之一，無法從結構特徵消除，但解釋為何 player-specific 準確率上限較低。

### Gaps
- 主辦單位如何決定「唯一類別」（快照 / 過濾 / 多數決）未公開；本節全部為假說。
- 檔案順序是否保留時間資訊，需以資料檢驗。

---

## 可計算的結構性特徵總表（定義 / 機制 / 期望強度 / 是否已涵蓋 / 成本 / 風險）

| 特徵 | 定義（僅用 SGF + 顏色 + HD） | 連結到等級的機制 | 期望強度（引用） | 是否已被現有特徵涵蓋 | 成本 | 風險（測試分佈） |
|---|---|---|---|---|---|---|
| F1 HD1 份額 × 顏色（條件先驗） | 題內 HD1 局比例，分別對「執黑題」「執白題」建立各類別經驗分佈，當 likelihood 使用 | 自動配對 ±1 級；HD1 執黑 ⇔ 對手高一級、執白 ⇔ 低一級；比例 ∝ 相鄰段位人口密度 n(r±1)/n(r)（[Pandanet](https://pandanet-igs.com/communities/pandanet/60)、[Igo Online](https://www.sakura-ent.net/igoonline-sp/game?lang=en) 的「段差＝手合」規則；OGS 人口單調遞減 [直方圖](https://forums.online-go.com/t/unofficial-ogs-rank-histogram-2021/35705)） | 未有公開數字；資料本身可估。HD1 佔全資料 17% → 20 局題平均 3.4 局 HD1，變異大 | 部分（share of HD1、signed direction）；**「條件於顏色的類別先驗」**是新的用法 | trivial | 測試同色 → 執黑題與執白題要分開校準；HD2+ 不在測試 |
| F2 對手品質的 ±1 級偏移校正 | HD0 局：對手≈同級；HD1 執黑：對手＝+1 級；HD1 執白：對手＝−1 級；將對手著手品質依此平移後再聚合 | 同上手合規則 | 理論上把對手品質估計的偏差從 ±1 級壓到 0；數字未有來源 | 部分（opponent quality、own−opp），但偏移校正是否已做需確認 | trivial | 若伺服器實為 KGS/OGS（0.5 目制），HD1 仍對應 1 級差，校正仍成立 |
| F3 終局型態 | 雙虛手結束 / 無虛手結束（認輸、超時、逃跑）/ 單虛手；尾端虛手數 | 認輸習慣隨段位改變（圈內共識；OGS 平均認輸第 168 手、5.9% 誤認輸 [OGS](https://forums.online-go.com/t/when-do-people-usually-resign-some-statistics/52977)；KGS 1d 認輸率 ~53% [L19](https://www.lifein19x19.com/viewtopic.php?p=210911)） | 無按段位公開數字；需自算 | 否 | trivial | 測試題取樣若排除斷線局（如 Fox 論文作法），型態分佈會偏；測試局數少（5 局）時雜訊大 |
| F4 對局長度分佈 | 手數的均值/中位數/最小值/<50 手比例 | 數子局平均 ~253.5 手 vs 認輸局更短（[OGS 27M](https://forums.online-go.com/t/statistics-from-a-27m-game-sample/39850)、職業 186/257 [CWI](https://homepages.cwi.nl/~aeb/go/misc/gostat.html)）；段位差異無公開數字 | 弱 | 否 | trivial | 同上 |
| F5 最後落子方 | 最後一手是否為目標方；與 F3 交互 | 認輸局：對手落子後本人認輸 → 最後落子＝對手 偏向本人敗；但超時者＝最後未落子方（混淆） | 無來源；需自算 | 否 | trivial | 無勝負欄位，只能間接 |
| F6 開局統計 | 第一手位置（星/小目/三三）、前 20 手內三三入侵次數、前 10 手是否模仿 AI 開局 | AI 時代高段更常模仿 AI 開局；KataGo 以 preaz/rank profile 區分年代與段位 [KataGo](https://newreleases.io/project/github/lightvector/KataGo/release/v1.15.0) | 無按段位數字（GoGoD 僅職業 [L19](https://www.lifein19x19.com/viewtopic.php?p=221841)） | 部分（著手品質模型隱含） | trivial | 資料年代未知 |
| F7 玩家局數 | 訓練集中該玩家總局數 | 升降段制度與活躍度 | 未知 | 否 | trivial | **測試不可用**（固定 5/10/15/20 局），僅供訓練分層 |
| F8 HD2+ 份額與方向 | 訓練集中玩家讓子局比例、作為讓方/被讓方 | 自動配對多在 ±1–2 級，讓子局多為手動邀約（假說） | 未知 | 是（stones 特徵） | trivial | **測試不含 HD2+**，只能用於訓練時的玩家層描述 |

## 伺服器身分判定清單（供主辦資料驗證）
1. 檢查 SGF 是否保留 KM/RU 欄位：KM 6.5 + RU Japanese/Korean → 排除 Fox（中國規則 7.5，未取得一手引用）、支持 Tygem/日式；KM 0.5 → 指向 KGS/OGS。
2. HD1 局若能數子驗證「貼 0 目」而非 0.5：排除 KGS/OGS/IGS（[Sensei's KGS](https://senseis.xmp.net/?topic=2464)、[OGS](https://forums.online-go.com/t/difference-in-ranks/21772/17)、[Pandanet](https://pandanet-igs.com/communities/pandanet/60)）。
3. HD2 局讓 2 子 → 符合 Igo Online／日本棋院「段差＝子數」（[Igo Online](https://www.sakura-ent.net/igoonline-sp/game?lang=en)）與 Pandanet（1.5–2 級 → 2 子）；KGS 公式 H = 段差/2+0.5 下 2 級差 → 1.5 進位 2 子亦相容，無法單憑 HD2 區分。
4. HD 最大值 9 與 Pandanet 賽事上限 9、OGS 自動配對 ±9 一致（[IWAG](https://pandanet.co.jp/event/iwag2002/e/rule-p.htm)、[OGS](https://forums.online-go.com/t/draws-handicap-komi-changes-and-ranking-limits-lifted/50489)），不具鑑別力。
