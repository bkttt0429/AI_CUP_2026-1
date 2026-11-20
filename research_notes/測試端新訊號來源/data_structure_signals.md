# 測試集「資料結構」訊號：競賽前例、規則邊界與本題可做的檢查

（範圍：AI CUP 2026 競賽二，圍棋棋力預測。只談特徵以外的資料結構訊號：排序、指派、格式殘留、跨集合重疊、測試集的類別設計。排行榜探測另見 `research_notes/測試位移與排行榜探測方法/leaderboard_feedback.md`，這裡不重複。）

本機觀察（2026-11-08，只在登入節點用 `head`/`cut` 讀了檔頭幾行，沒有跑任何計算）：

- 測試檔欄位為 `question_id,num_games,sgf_1..sgf_20,color_1..color_20`；SGF 根節點只有 `(;SZ[19]HD[n]`，接著是 `;B[xx];W[xx]…`，結尾 `)`。看過的 5 局都沒有 `RE`、`CA`、`KM` 等屬性。來源：`/home/u2507115/BK_AICUP_2026-1/data/tests/rank_prediction_test_public.csv` 第 2 列。
- **每題的執色全部相同**。public 前 12 題、private 前 8 題都是整題 B 或整題 W：public 9 題 W、3 題 B；private 5 題 W、3 題 B。來源：兩個測試檔前 13／9 列的 `color_*` 欄。
- 訓練檔 `train_1D.csv` 前 5 列都是 `color=B`，同一位玩家的 `g_0000004`、`g_0000005` 相鄰。來源：`/home/u2507115/BK_AICUP_2026-1/data/training/train_1D.csv` 前 6 列。
- 後驗檔的欄位順序是 `p_D,…,p_6D,question_id`（`question_id` 在最後一欄）。來源：`/home/u2507115/BK_AICUP_2026-1/submission_rank_ct9_proba.csv` 檔頭。

---

## 問題一：競賽中有記錄的 ID 順序、列順序、檔案結構、重複紀錄與去匿名化洩漏，以及主辦方怎麼處理

### Takeaway
資料結構洩漏在 Kaggle 一再左右名次：ID 或列順序（Telstra、Bosch、TalkingData）、檔案中繼資料（Truly Native、RentHop 資料夾時間）、測試集抽樣方式（Quora 配對、Santander 合成列、Recursion 每盤一次的指派）、跨集合重複（HPA）。只要訊號**來自主辦方釋出的資料本身**、並公開分享，通常不會被取消資格。被取消資格的案例都是**從競賽外取得測試答案**，例如 PetFinder 爬官網、IJCNN 以外部爬取資料去匿名化，後者雖然得勝，作者自己也主張規則應該禁止。主辦方的修補方式是剔除受污染的測試批次（Recursion 的 HUVEC-18），或事後調整設計。我沒有找到任何一則主辦方明文判定「使用資料內結構屬違規」的裁決。

### Cited Findings

**ID 與列順序**
- Telstra Network Disruptions：資料沒有直接提供時間，但可以從某些檔案的列順序還原；排行榜前約 15 人在 0.42–0.44，其餘都高於 0.48，這個斷層被解讀為洩漏 — [gereleth Telstra writeup（搜尋摘要）](https://gereleth.github.io/Telstra-Network-Disruptions-Writeup/)；講義把 Telstra 列為「row order leak」的例子 — [Coursera 'How to Win a Data Science Competition' 講義逐字稿 basic data leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/01_basic-data-leaks.en.txt)
- Bosch Production Line Performance（2016）：Id／列順序的「magic features」由參賽者在公開 kernel 釋出。第 3 名隊伍說它「opened the competition」，而且一直是最強特徵；論壇上有人說加入第一個 ID／列順序特徵前 AUC 是 0.921 — [Kaggle Blog 第 3 名訪談](https://medium.com/kaggle-blog/bosch-production-line-performance-competition-winners-interview-3rd-place-team-data-property-8b24c3747321)；[Bosch 討論串](https://www.kaggle.com/c/bosch-production-line-performance/discussion/24291)
- TalkingData Mobile User Demographics：資料有重複列，相鄰列常同標籤 — [Coursera 講義 basic leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/01_basic-data-leaks.en.txt)。Kaggle 產品回饋區有人主張，把列順序與 ID 隨機化得更好，就能消除 TalkingData 的洩漏 — [Kaggle feature request「Handling Leaks in Competition」（搜尋摘要，頁面需 JS 無法全文抓取）](https://www.kaggle.com/discussions/product-feedback/23835)
- Caterpillar：把 ID 當特徵「略微改善」結果。講師的建議是不要假設 ID 沒有意義，而且在競賽中可以「exploit the leak in every way possible」，同時強調洩漏是主辦方無心造成的 — [Coursera 講義 basic leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/01_basic-data-leaks.en.txt)
- KDD Cup 2008（乳癌偵測）：優勝隊報告說 patient ID 帶有大量判斷惡性候選的資訊，原因是主辦方準備資料時意外造成人工相依 — [IBM 研究報告 RC24707（搜尋摘要）](https://dominoweb.draco.res.ibm.com/reports/rc24707.pdf)；[Rosset et al. RC24758](https://dominoweb.draco.res.ibm.com/reports/rc24758.pdf)。同一團隊後來以〈Medical data mining: insights from winning two competitions〉發表 — [論文頁](https://datalearner.com/academic/journal-papers/1384-5810/volumes-and-issues/20/paper-detail/9312)
- Kaufman, Rosset, Perlich, Stitelman（ACM TKDD 2012）把洩漏形式化，提出「learn-predict separation」作為避免方法，並在無法控制資料蒐集時提供偵測方法；文中提到競賽特有的第二種洩漏，以 KDD Cup 2008 為例 — [DOI 10.1145/2382577.2382579（搜尋摘要；PDF 本機無法解析）](https://www.doi.org/10.1145/2382577.2382579)

**檔案中繼資料**
- Truly Native?：zip 檔裡的日期讓分數「nearly perfect」。講師指出真正的問題出在資料蒐集：拿掉日期後，文字仍是蒐集時間的代理 — [Coursera 講義 basic leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/01_basic-data-leaks.en.txt)；[Coursera 講義 rare leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- Two Sigma Connect: Rental Listing（RentHop，2017）：圖片資料夾的建立時間和 interest level 相關，被稱為「magic leak」，很多人直接用它取代影像處理 — [lstmemery 簡報](https://lstmemery.github.io/renthop-presentation)；[Medium 參賽者文章](https://medium.com/@MARatsimbazafy/predicting-apartment-interest-from-listing-with-structured-data-free-text-geolocalization-time-52ad866f2fae)

**測試集抽樣與設計**
- Quora Question Pairs：主辦方沒有評估所有配對，「nonrandom subsampling」本身就洩漏標籤：常出現的題目較可能是重複題，鄰居集合相似的兩題也較可能重複。另外 public 與 private 的目標比例相同，可以用常數提交探出比例來重新平衡 — [Coursera 講義 rare leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- Expedia Hotel Recommendations：可以由距離特徵反推兩點座標，把訓練標籤對到測試 — [Coursera 講義 rare leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- Santander Customer Transaction Prediction（2019）：測試集混有合成列，判別規則是「一列只要有任一特徵值在該欄只出現一次就是真實列，否則是合成列」。此法出自 YaG320 的 kernel〈List of Fake Samples and Public/Private LB split〉，把合成列排除在建模之外可以提高分數 — [lovkush-a 部落格（二手）](https://lovkush-a.github.io/python/data%20science/2020/07/18/santander3.html)；[Pendulibrium 銀牌文](https://medium.com/pendulibrium/how-we-won-a-silver-medal-on-kaggle-b06cbe3dee8c)
- Recursion Cellular Image Classification（2019）：實驗設計讓每種 siRNA 在一個 experiment（更精確地說，一個 plate）裡只出現一次。參賽者用模型輸出機率解指派問題（匈牙利演算法），「多くの参加者がこのリークを利用していました」。另一個洩漏被發現後，主辦方把 HUVEC-18 實驗從測試集移除 — [guchiBLO 賽後整理](https://guchio3.hatenablog.com/entry/2019/10/07/212204)。siRNA 以 277 個為一組放在同一 plate；利用實驗結構後約 90% 孔位預測正確，不利用時約 65–75% — [Harvard QBRC 課程說明（搜尋摘要）](https://www.hsph.harvard.edu/qbrc/?p=309)
- Santander Value Prediction（2018）：洩漏經公開 kernel 散布，有隊伍說自己原本在前 14%，洩漏出現後名次大變；有簡報把當時的情況寫成「Leak! Competitors revolt!」 — [2econsulting 心得](https://2econsulting.github.io/kaggle/2018/08/30/kaggleSantander.html)；[Calgary R 簡報 PDF](https://imstatsbee.github.io/calgaryr/slides/c1819/20190320_alastair_muir.pdf)

**用匿名化資料內的欄位把同一實體的紀錄串起來**
- IEEE-CIS Fraud Detection（2019，第 1 名）：由 `card1`、`addr1`、`D1` 組出「credit card client unique ID」。UID 本身不進模型，只當聚合鍵；最後把同一 UID 的預測換成該客戶的平均預測。Public 0.9677、Private 0.9459。文章沒有把它當成洩漏或違規來討論 — [NVIDIA 技術部落格（第 1 名隊員 Chris Deotte）](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)

**跨集合重複紀錄**
- HPA Image Classification（2018）：主辦方事後文章指出兩種洩漏來源：訓練影像誤入測試集，以及同一批次影像分屬訓練與測試。參賽者用 perceptual hashing 找出相同或近似的影像；優勝模型的 metric learning 部分「mainly works by detecting images from the same sample」，屬於非預期的捷徑 — [Springer Nature Communities〈Behind the HPA Image Classification Competition〉](https://communities.springernature.com/posts/behind-the-hpa-image-classification-competition)。主辦方允許使用外部 HPAv18 影像（約 78,000 張） — [Nature Methods 主辦方論文，PMC6976526（搜尋摘要）](https://pmc.ncbi.nlm.nih.gov/articles/PMC6976526)

**去匿名化與從外部取得測試答案（取消資格的界線）**
- IJCNN 2011 Social Network Challenge（Kaggle）：資料是去除身分的 Flickr 圖。Narayanan、Shi、Rubinstein 用自己爬取的 Flickr 資料去匿名化大部分測試集，「effectively game the competition」，再和連結預測混合，以 AUC 0.981 獲勝。作者主張應以競賽規則抑制這類 gaming，並認為技術偵測不太可能萬無一失 — [arXiv 1102.4374](https://arxiv.org/abs/1102.4374)；[Microsoft Research 頁](https://www.microsoft.com/en-us/research/?p=160937)；[KDnuggets 轉載](https://www.kdnuggets.com/2011/03/link-prediction-by-de-anonymization-social-network-challenge.html)
- PetFinder.my Adoption Prediction（2019–2020）：第 1 名 BestPetting 被取消資格。他們從 PetFinder.my 網站爬取資料、事先得知隱藏測試集的領養結果，再把答案編碼成看似隨機的識別欄；而且只用了部分答案，讓分數看起來「realistic」。事情是第 6 名在協助部署優勝模型時發現的，隊長致歉並承諾退還獎金 — [Vice](https://www.vice.com/en/article/kaggle-data-science-community-rocked-by-pet-adoption-contest-cheating-scandal)；[The Register](https://www.theregister.com/2020/01/21/ai_kaggle_cheat/)
- Deepfake Detection Challenge（2020）：Facebook 與 Kaggle 取消了表面上的第 1 名，並因此受到批評（只取得標題，理由未抓取） — [Synced Review](https://syncedreview.com/2020/06/14/facebook-and-kaggle-face-backlash-after-disqualifying-apparent-deepfake-detection-challenge-winner/)

### Inferences
- 前例大致分成三層：
  1. **主辦方釋出資料內的結構**：ID、列順序、檔案時間、抽樣設計、跨集合重複、同實體串接。Kaggle 生態把這些視為可用，公開分享即可，沒有看到因此被取消資格的例子。
  2. **從競賽外取得測試答案，或用外部資料去匿名化**：PetFinder 被取消資格；IJCNN 雖然獲勝，作者自己主張規則應該禁止。
  3. **主辦方事後修補**：剔除受污染的測試批次（Recursion HUVEC-18）。修補的時點不一定，有時出現在賽後的 Private 評分。
- 本題的資料使用條款明文禁止「識別真實使用者」（見問題二）。所以第 2 層在本題不只是灰色地帶，而是明確禁止。第 1 層裡的「同實體串接」（IEEE-CIS 型）在本題對應到「把兩筆匿名紀錄串成同一玩家」，碰到這個條款的精神，應列為 **needs organiser approval**。
- 和本題最接近的是 Recursion（已知每格配額，用指派問題）和 Quora（測試配對的抽樣方式本身有訊號）。兩者都利用「主辦方如何組題」，不涉及任何人的身分。

### Gaps
- Kaggle 官方對「賽中發現洩漏」的通用政策我沒拿到原文：討論頁需要 JS，只取得標題與搜尋摘要。
- IJCNN 2011 時 Kaggle 是否事先知情、是否正式核可該勝利，抓到的來源都沒有寫；去匿名化的比例也沒有取得。
- Santander Value Prediction 洩漏後 Kaggle 的正式處置（例如是否延長或改測試集）沒有找到一手公告。
- 我記得 2016 年 Melbourne 癲癇預測賽有「測試檔與訓練重複、重新釋出」的事件，但可取得的主辦方論文（PMC6136083）沒有提到，**未證實，不列為事實**。
- Recursion 主辦方是否正式表態允許「匈牙利指派」，來源只說很多人用，沒有官方裁決。
- 2023 年 AI CUP 圍棋賽或其他 AI CUP 賽事是否有過「洩漏使用」的裁決，公開網路找不到。

---

## 問題二：AI CUP／AIdea 對使用測試集結構、排行榜探測、外部資料與「識別」的規則

### Takeaway
競賽二頁面的規則禁止多帳號、人為修正上傳結果、私下分享、抄襲作弊，外部資料須揭露。**對「利用測試集結構」與「排行榜探測」沒有任何明文**，對測試集如何抽樣也完全沒說明。唯一碰到「識別」的是資料集 README：「棋譜已匿名化，不得用於識別真實使用者。」官方 Q&A 在競賽一的語境禁止混用 Public／Private 候選池、禁止人工標註測試資料，可以看出主辦方對「跨集合使用」與「人工介入」敏感。凡屬「把匿名紀錄串連」的做法都應先經表單詢問。

### Cited Findings
- 競賽二規則原文：「測試資料集預測結果每日上傳至多 5 次，並以最後一次上傳之預測結果獲得之分數進行排名。」「參賽隊伍不得註冊多個帳號參加同一項競賽，違者喪失競賽資格，參賽隊伍個別成員自行以個人帳號上傳結果，也視同使用多重帳號參賽。」「參賽者所上傳的結果不得使用人為修正，以避免影響比賽公平性。」「切勿在競賽過程中將程式碼、報告草稿等資料，私下分享或傳遞給其他隊伍。」「已有具體事證，所屬隊伍有任何抄襲、作弊、或詐欺等行為」「參賽隊伍如違反本競賽辦法之相關規定，將取消其參賽資格。」外部自製或開源資料、生成式 AI 須遵守著作權法，並在報告中說明是否使用額外資源與來源 — [AIdea 競賽二頁面](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 同一頁面沒有任何關於識別玩家或匿名化的規則，也沒有描述測試題怎麼抽樣（類別平衡、num_games）。2026-11-04 之後沒有關於測試集的新公告，最新公告是 11/02 的 HD 更新 — [AIdea 競賽二頁面（2026-11-08 抓取）](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 資料集 README：「棋譜已匿名化，不得用於識別真實使用者。」「僅供 2026 AI CUP 競賽使用」 — [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)（轉引自本機 `research_notes/類似任務他人解法/aicup_context.md`）
- 官方 Q&A（2026-10-22 版）Q10，競賽一候選池：可用於校正、正規化，但 Public／Private 候選池不得混用，不得人工標註測試資料。Q&A 沒有任何題目談到等級標籤如何指定、測試題的同色或類別平衡、測試玩家是否和訓練玩家互斥 — [官方 Q&A Google Doc](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)（轉引自 `aicup_context.md`）。官方正式回覆須透過提問表單 — [提問表單](https://docs.google.com/forms/d/e/1FAIpQLSf0255uJMW5MsQNODY6EBoz3GHPVuaRbCL4PMS7EbCXh06YMQ/viewform)
- 教學 repo：public 與 private 各 400 題，每題 5–20 局；缺任何一題，該次提交無效 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)
- 對照（不是本賽規則）：國家海洋研究院主辦、工研院等協辦、繳交範例放在 AIdea 的「2026 海廢影像辨識國際 AI 競賽」在競賽辦法第 4 條明文寫「參賽隊伍不得以多重帳號、人工標註測試資料、攻擊排行榜系統，或其他不公平方式影響競賽結果；違者將取消參賽或得獎資格」 — [DIGITIMES 競賽頁（2026-11-08 直接讀取頁面確認）](https://www.digitimes.com.tw/seminar/mdivrc_2026/)

### Inferences
- **低風險**（演算法性、只用本賽釋出的檔案、在報告揭露）：題號／列順序檢查、num_games 與執色的設計推斷、格式殘留統計、依已知或推估的類別組成做先驗調整或配額決策。「不得人為修正」針對的是人工改答案；配額式指派是固定程式產生的輸出，應不屬之，但報告裡要寫清楚演算法與動機。
- **needs organiser approval**：
  - (a) 用訓練與測試之間的完全相同棋譜，把訓練標籤或 HD 推得的段位差轉移到測試玩家。
  - (b) 把兩個測試題串成同一位真實使用者，例如一題 B、一題 W；或串連對手關係。
  - (c) 把競賽二的測試資料和競賽一的查詢或候選池互相比對。

  (a) 的本質是「重複紀錄洩漏」，類似 HPA；但機制是「串連兩筆匿名紀錄」，和 IEEE-CIS 型的同實體串接同類。條款寫的是「識別真實使用者」，嚴格來說沒有識別出真人；但 Q10 對跨集合混用的態度，加上「不得人為修正」與「作弊」條款的裁量空間，會讓事後被判違規的風險不可忽略。
- **明確禁止**：反推 `player_id` 雜湊、比對外部棋站公開棋譜找出測試局（PetFinder／IJCNN 型）、人工標註、多帳號。
- 規則對「探測」沒有明文，但有每日 5 次與單帳號的限制。探測本身的效益與風險見另一份筆記。

### Gaps
- AIdea 平台的一般使用條款（不是單一競賽頁）是否有「不得利用系統漏洞」之類的總則，沒有找到。
- 主辦方是否會回答「測試集是否類別平衡」，沒有前例；Q&A 至今未涵蓋。

---

## 問題三：各類訊號在本題要做的具體檢查（S1–S5）

### Takeaway
值得先做的是 **S2（指派）與 S5（類別設計）**。400＝10×40、每個 num_games 100 題＝10×10，若「執色 × num_games」八格各恰 50 題＝10×5，就強烈暗示主辦方做了分層的因子設計，類別多半也是每格等量。這時可以做 Recursion 式的配額決策，前提是要在模擬題上驗證。
- **S1（排序）**：先驗低。題內局序已確認無趨勢；從檔頭幾行看，num_games 與執色在題號上是混的。
- **S3（格式）**：看起來很乾淨，根節點都是 `(;SZ[19]HD[n]`。
- **S4（重複棋譜）**：預期為零，Rinrin-lab 報告訓練集 100 萬局沒有任何兩局著手序列相同。這一項是資料完整性檢查，**只統計、不使用**。

所有計算都很輕（單核幾分鐘內），但仍依規定走 sbatch。

### Cited Findings
- 訓練集「每級 100,000 局」「執黑／執白各半」 — [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)（轉引自 `aicup_context.md`）
- 各級玩家數差異大（A 1,582 人、1D 433 人）；「100 萬局中沒有任何兩局的著手序列相同」 — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)（轉引自 `aicup_context.md`）
- v1.1 相對 v1.0 只加入 HD[n]，列順序、player_id、game_id、rank、color、棋步都不變 — [資料集 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases)（轉引自 `aicup_context.md`）
- 先驗調整：Saerens、Latinne、Decaestecker（2002）提出用 EM 在新資料上同時重估後驗與先驗，不必重訓模型，最大化新資料的概似；用估計先驗調整輸出能提高分類率，接近用真實先驗的結果 — [Neural Computation 14(1) PDF](https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf)；[mlanthology](https://mlanthology.org/neco/2002/saerens2002neco-adjusting)
- 已知配額時的指派決策：Recursion 參賽者以模型機率解指派問題（匈牙利演算法），利用「每 plate 每 siRNA 一次」 — [guchiBLO](https://guchio3.hatenablog.com/entry/2019/10/07/212204)
- 抽樣方式本身會帶訊號（Quora 的 nonrandom subsampling） — [Coursera 講義 rare leaks](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- 去除「列順序與 ID」結構的標準修補是把它們隨機化 — [Kaggle feature request（搜尋摘要）](https://www.kaggle.com/discussions/product-feedback/23835)

### Inferences

以下記號：檔案 f ∈ {pub, priv}；題 q 的後驗 p_k(q)，k = 0..9 依序為 D, C, B, A, 1D…6D（取自 `submission_rank_ct9_proba.csv`，注意 `question_id` 是最後一欄）；後驗平均等級 m_q = Σ_k k·p_k(q)；argmax 類 a_q；題號 i_q 取 `pub_q_0001` 的數字部分。所有 p 值用打亂題號（或打亂標記）的 10,000 次置換求得。

**S1 排序訊號（題號、列順序、欄順序）**
- C1.1：確認列順序等於題號順序（0001–0400 連續、遞增），兩檔表頭逐字相同。若相同，「列順序」與「題號順序」是同一個檢查。
- C1.2 題號 vs 等級：每檔計算以下統計量，各自求置換 p 值。
  - Spearman ρ(i_q, m_q)。
  - 把題號切成 10 個連續區塊（每塊 40 題），對 m_q 做 Kruskal–Wallis。
  - m_q 的 lag-1…lag-20 自相關，用來抓「先按類排序、再在小區塊內打亂」的生成方式。
  - 對 a_q 序列做 runs test。

  再在每個 num_games 子序列內各做一次。判讀：p 都 > 0.05 就關閉這條線。
- C1.3 題號 vs 設計變數：對 num_games 和執色序列做同樣的 runs／自相關，以及 10 區塊 × 4 n、10 區塊 × 2 色的卡方。設計變數若在題號上看得出結構，代表生成後沒有打亂，C1.2 的先驗就要提高；看起來隨機則反之。
- C1.4 pub／priv 同號對齊：比較 pub_q_i 與 priv_q_i 的 num_games 一致率（機會 1/4）、執色一致率（機會約 1/2）、m 的 Spearman。若一致率遠高於機會，表示兩檔由同一條交錯清單切出，同號題的類別可能也對齊。這會讓 public 上得到的資訊能按題號轉移到 private，屬高價值發現。從前 8 題看是機會水準：n 一致 2/8、執色一致 5/8。
- C1.5 題內局序 vs 局的屬性：等級趨勢已確認沒有，這裡只看「sgf_k 是否依某屬性排序」。每題計算 Spearman(k, 手數) 與 Spearman(k, HD)，對 800 題取平均，再和題內打亂的置換分布比較。若有排序，只能說明局序是時間序之類，不直接給等級。先驗低。

**S2 指派訊號（num_games、執色怎麼配給玩家）**
- C2.1 設計格：每檔做「執色（整題 B／W）× num_games」2×4 交叉表，並確認沒有混色題。
  - 若八格都恰為 50：在「每題執色獨立、機率 1/2」的虛無下，四個 n 層同時恰好 50:50 的機率約 0.08⁴ ≈ 4×10⁻⁵，可視為主辦方按執色分層，接著推論類別多半也是按 (類 × 色 × n) 每格 5 題的因子設計。
  - 若不是 50，就計算各層 W 比例與二項分布的偏離。
  - 混色題比例若為 0：在「從玩家的局隨機抽」的虛無下，20 局全同色的機率約 2×0.5²⁰，所以同色一定是設計出來的，不是巧合。
- C2.2 執色 vs 等級：比較 W 題與 B 題的後驗平均 m 分布（Mann–Whitney），兩組各做 EM 先驗估計（見 C5.1），再做 2×10 的「期望計數」卡方。前 20 題 W:B = 14:6，可能暗示 W 題較多；若真如此，要看是否和等級有關，例如某些等級的玩家在來源資料裡多執白、讓子棋的上手。
- C2.3 num_games vs 等級：對 m_q 依 n 做 Kruskal–Wallis，四個 n 層各做 EM 先驗。對照組用訓練檔：每個等級算「每位玩家的局數」分布，以及局數 ≥5／10／15／20 的玩家比例。若測試題是「從局數夠的玩家中均勻抽」，n 會和等級相關，類別先驗會 ∝ 合格玩家數（例如 A 1,582 人 vs 1D 433 人），非常不均勻；若是「每類每層等量」則 n 和等級無關。比較三種假說下的隱含先驗和 EM 估計：(i) 每類等量；(ii) ∝ 合格玩家數；(iii) ∝ 局數，也就是均勻。
- C2.4 訓練端的「玩家」是否本身就是（使用者 × 執色）：對每個訓練 player_id 計算 B 局比例，統計純 B、純 W、混色的玩家數。若訓練玩家幾乎都是單色，代表匿名代號可能以「使用者 × 執色」為單位，這時測試的一題 B、一題 W 可能是同一位真實使用者。**只能記錄這個事實，不可據以串連題目（needs organiser approval）。**

**S3 格式殘留（屬性順序、空白、著手字串、pass 編碼、屬性有無）**

原則：一個殘留要能用，必須在訓練集中**依等級**有明顯差異，**而且**出現在測試裡。否則關閉。
- C3.1 根節點字串：取每局從 `(` 到第一個 `;B[` 或 `;W[` 之前的子字串，在訓練（按等級）與測試（按檔）做完全字串的次數表。預期只有 `(;SZ[19]HD[0..9]`。任何額外屬性（CA、RE、KM、PL、AB、AW、HA）或順序不同，都要列出等級分布。
- C3.2 節點語法：以 `;` 切開，每節點比對正規式 `^[BW]\[([a-s]{2})?\]$`。不符者依「型態」歸類計數，例如 `B[tt]`、大寫座標、空白、換行、`C[]`、`BL/WL/OB/OW`、`AB/AW/AE`、`;;`、超出 a–s 的座標，再按等級與測試檔做表。
- C3.3 pass 編碼與位置：`B[]`／`W[]` 和 `B[tt]` 各自的出現率；pass 出現在最後兩手（終局）和中盤的比例。按等級統計。
- C3.4 輪替違規：連續兩手同色的局數比例（可能是讓子擺放或剪接），按等級統計。
- C3.5 長度一致性：設手數 N、根節點長度 L_root，檢查 `len(sgf) == L_root + 6·N + 1`。不等就代表字串裡還有別的東西，依等級列出。順帶檢查 CRLF／LF、BOM、CSV 引號、尾端空白，以檔為單位，只為確認沒有隱藏欄位。
- C3.6 截斷：各等級訓練局與測試局的最大手數、手數分布尾端是否在某固定值堆積（例如被截在 N 手），以及最後一手是否在某些等級常見為 pass。
- C3.7 HD 欄一致性：測試每局是否都有 `HD[n]`（v1.1 格式），HD 分布和訓練是否相同。

**S4 跨集合重疊（不識別任何人）**
- C4.1 精確重複：每局去掉根節點，取著手字串 `;B[..];W[..]…` 的 SHA-1。對以下集合求交集，各報計數：
  - 訓練 100 萬局 vs 測試約 1 萬局（public 400 題 + private 400 題，每題 5–20 局）。
  - 測試題內。
  - 測試題間：pub 內、priv 內、pub × priv。
- C4.2 方向：對每個命中，記錄測試該局的 `color_k` 和訓練列的 `color` 是否相同。同色表示可能是同一位玩家，這會和「測試玩家與訓練不重疊」的假設衝突；異色表示訓練玩家是測試玩家的對手。
- C4.3 近似重複：前 60 手相同、總手數相同、HD 相同，用來抓截斷或尾端差異。只計數。
- 判讀與處置：
  - 預期 C4.1 為 0：訓練內 100 萬局都沒有重複，主辦方多半做過去重，或每局只以一方視角收錄。
  - 若 > 0：**停止，不得用於預測**。把計數（不附任何識別資訊）透過官方表單回報，詢問能否使用。理由：用 HD 推段位差、或沿用訓練標籤，等於串連兩筆匿名紀錄來推測試標籤（needs organiser approval，見問題二）。
  - 測試題間的重疊（兩位測試玩家互為對手）同理，也是 needs organiser approval。
  - 不做以下事情：和競賽一的候選池或查詢比對（跨競賽，Q10 精神，needs organiser approval）、和任何外部棋站比對（明確禁止）。

**S5 測試集的類別設計與合法用法（配額式決策）**
- C5.1 EM 先驗估計（Saerens 2002）：
  - 先確認 ct9 後驗所隱含的訓練先驗 π^tr。若模型用類別平衡權重訓練，π^tr 為均勻。
  - 對每檔（以及每個 n 層、每個執色）迭代：π_k ← (1/Q) Σ_q w_k(q)，其中 w_k(q) = p_k(q)·π_k/π^tr_k ÷ Σ_j p_j(q)·π_j/π^tr_j。迭代到收斂，對題目做 bootstrap 取 95% CI。
  - 平衡檢定：LR = 2[ℓ(π̂) − ℓ(uniform)]，其中 ℓ(π) = Σ_q log Σ_k p_k(q)·π_k/π^tr_k，名目上近似 χ²₉。
  - 這裡後驗校準不完美，所以 null 分布要用模擬題產生，見 C5.2。
- C5.2 用平衡模擬題校準 null：在 `data/mock_tests/` 中已知是類別平衡的模擬題組（例如 `fold1_100*`、`fold1_n20*`，需先確認其組成），用同一模型的後驗跑 C5.1，得到「真的平衡時 π̂ 與 LR 的分布」。測試檔的 LR 落在這個分布內，才可說和平衡一致。
- C5.3 配額式決策（Recursion 型）：
  - 得分矩陣 S[q,c] = Σ_k p_k(q)·s(c,k)，其中 c = k 時 s = 1，|c−k| = 1 時 s = e⁻¹，其餘為 0。
  - 硬配額：最大化 Σ_q S[q, c_q]，限制每類恰 40 題（或在因子設計下每 (色 × n) 格每類 5 題）。可解為 400×400 指派（每類複製 40 個槽，用匈牙利演算法），或最小成本流。
  - 軟版本：把後驗用均勻先驗重加權後，再做逐題期望得分決策；或用 Lagrange 乘子給每類一個偏移量，調到計數接近配額。
  - 驗證：在平衡模擬題上比較「逐題期望得分」與「配額決策」的差，並在刻意不平衡的模擬題（例如抽掉某兩級各 20 題）上量下檔風險。
  - 本機記憶顯示 M50 LP／pseudo-test 已做過（ct12 ≈ ct9）。這裡的新增價值是 C2.1：若證實是因子設計，配額可以細到每格，限制更緊。
- C5.4 向主辦方提問（建議，低風險、可公開）。用表單問：
  1. 測試集各等級題數是否相同、是否依 num_games 或執色分層。
  2. 若訓練與測試之間有完全相同的棋譜，能否使用。
  3. 競賽一與競賽二的資料能否互相比對。

  前一項若獲肯定，C5.3 就從「賭組成」變成「使用公開設計」。後兩項是在使用任何串連類做法之前必須先得到的許可。

**優先順序（建議）**
1. C2.1、C2.4、C4.1：最便宜，結果是二元的，而且決定後面要不要做。
2. C5.1、C5.2：判斷平衡與否。
3. C2.2、C2.3：判斷設計假說。
4. C1.2–C1.4：快速關閉。
5. S3 全部：一個批次腳本就能跑完。

### Gaps
- 測試集實際的抽樣設計（類別、執色、num_games 如何分層）沒有任何官方文字；以上只能從檔案結構推斷。
- 只看了 20 題的執色，W:B = 14:6 是小樣本；八格是否恰為 50，要跑 C2.1 才知道。
- 訓練玩家是否單色（C2.4）沒有現成資料，`aicup_context.md` 與 README 只說「執黑／執白各半」，是以局為單位。
- 我沒有找到任何前例或規則，判定「用 HD 與重複棋譜推段位差」是否違反「不得識別真實使用者」；這必須由主辦方解釋。
