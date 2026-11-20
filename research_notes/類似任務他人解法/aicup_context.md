# AI CUP 2026 圍棋棋力預測競賽：公開脈絡（官方頁面、前屆賽事、參賽者公開討論、天花板估計）

研究日期：2026-11-05。所有網頁均為公開頁面，未登入任何系統。官方陳述與參賽者推測分開標示；凡引用本機檔案（非公開）者，以絕對路徑標示並註明「本機」。

---

## 關鍵問題 1：官方頁面、規則、baseline、Q&A（匿名化、HD[]、同色、等級標籤、伺服器）、姊妹賽（競賽一）

### Takeaway
官方資訊分散在四處：AIdea 競賽頁（規則、時程、獎金、公告）、GitHub 資料集 repo（欄位、v1.1 的 HD[n]）、GitHub 教學 repo（baseline ResNet，Public 0.4631／0.2845）、以及一份 Google Doc 官方 Q&A（2026-10-22 版，14 題）。官方明文允許 KataGo 等預訓練模型與外部棋譜，只須在最終報告揭露；Q13 說明約 80% 互先、16% 讓先、3% 讓 2–9 子，讓子棋的 B/W 標記與實際相反、讓子石未記錄，但「color 欄位是否也要對調」官方自己標為「二擇一、待確認」。官方**從未**說明棋譜來自哪個伺服器、等級標籤如何指定、測試題是否同色或類別平衡。

### Cited Findings

**官方競賽頁（AIdea，競賽二：從圍棋棋譜預測玩家棋力等級）**
- 題目名稱「圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級」；議題提供單位「國立東華大學 智慧科技中心」；指導單位教育部資訊及科技教育司；運籌單位「教育部人工智慧競賽與標註資料蒐集計畫辦公室」；平台為工研院 AIdea；標示「非開放議題」 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 時程（台灣時間）：報名 2026/09/08 11:00–2026/11/03 23:59:59；訓練資料下載與模型訓練 2026/9/18 11:00–2026/11/24；測試資料釋出與正式開賽 2026/11/04 11:00；上傳截止 2026/11/24 23:59:59；Private Leaderboard 成績公布 2026/11/25 16:00；報告上傳 2026/11/26–2026/12/06；最終排名 2027/01/04 11:00；頒獎典禮 2027 年初（暫訂 Q1） — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 任務原文：「從查詢棋譜集，預測下出這些棋譜的玩家之棋力。」等級為 6 個段位（1D–6D）加 4 個級位群（A–D），每級 100,000 局 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 評分：完全正確 1 分、相鄰一級 1/e ≈ 0.368、其餘 0；S_strength = (1/N) Σ score_i — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 上傳限制：每天最多 5 次；排名以**最後一次上傳**為準；競賽期間只顯示 Public 分數；Public 同分以上傳時間早者優先 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 規則：外部自製或開源資料、生成式 AI 工具皆可用，需符合著作權法，「參賽隊伍應於比賽結束繳交的書面報告文件中提供相關來源」並說明其對結果的貢獻；得獎報告須含模型演算法、前處理與訓練程式、參數（含訓練權重）、執行環境、外部資源來源；不得私下共享程式與特徵（官方討論區公開分享不在此限）；不得人工修改結果；「參賽隊伍不得註冊多個帳號參加同一項競賽」；主辦單位得於競賽中調整資料集 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 獎金（學生組）：總額 NT$125,000，第一名 40,000、第二名 25,000、第三名 15,000、優等 3 隊各 5,000、佳作 10 隊各 3,000；前 16 名學生隊得教育部獎狀；「排名前30名且為前25%之隊伍(含社會人士)」繳交報告並通過審查者得電子證書；證書資格須超過 baseline **0.4631** — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 參加人數／組數：2026-11-05 抓取時顯示 **420 人／277 組** — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)；本機 REPORT.md（撰寫於 v1.0 時期）記錄同頁當時為 213 人／138 組 — 本機 `/home/u2507115/BK_AICUP_2026-1/REPORT.md` 第 28 行（非公開）
- 公告三則：2026/10/21 教學簡報更新，「如簡報內容與競賽頁面公告之規則或說明有任何差異，均以競賽頁面公告內容為準。」；2026/10/29 彙整 Q&A 文件，「請參賽者於提出問題前，先行查閱 QA 文件」，官方正式回覆須透過表單；2026/11/02 訓練資料集更新：「尚未包含棋局的讓子資訊（HD[n]）」，已補上並釋出 v1.1 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 頁面所列連結：官方 Q&A 文件 https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit ；提問表單 https://docs.google.com/forms/d/e/1FAIpQLSf0255uJMW5MsQNODY6EBoz3GHPVuaRbCL4PMS7EbCXh06YMQ/viewform ；FB 粉絲專頁 https://www.facebook.com/AICUPrealtask ；報名系統 https://go.aicup.tw/ ；討論區只有頁內 `#topic-discuss` 分頁，抓取時無任何貼文內容 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
- 頁面**未說明**：棋譜來源伺服器、匿名化方法、等級標籤如何指定、測試題的執色與類別分布 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)

**官方 Q&A 文件（2026-10-22 版，公開 Google Doc，全文已下載）**
- 文件標題「AI CUP 2026 圍棋競賽 常見問題 Q&A」，「更新日期 2026年10月22日」，適用「棋力預測」與「玩家辨識」兩子題；「本文未涵蓋的事項，以競賽頁公告的競賽規則為準。」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q1：「能否使用開源的預訓練模型（例如 KataGo 公開權重），在推論時抽取特徵，或當作自行訓練模型的初始權重？」答「可以。……官方 baseline 沒有附預訓練 checkpoint，是為了讓大家有相同的起點，並不是禁止使用。教學簡報第 80 頁提到『使用 pretrained 權重是現成的改進方向』……KataGo 請依 KataGo network license 的規定。」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q2：可使用本競賽以外的公開對弈棋譜資料集訓練或預訓練。Q3：權利人僅表示「可用於研究」亦可用，智財爭議由隊伍自負。Q4：預訓練模型作者未公布訓練資料清單仍可用。Q5：不需事先報備，但最終報告須寫「來源網址、版本或檔名（例如 KataGo network 名稱）、授權條款、使用方式：特徵抽取、初始化或額外訓練資料、這些外部資源對分數的貢獻，最好附消融實驗」；「如果沒有揭露，經查證後視為違規。」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q6–Q8：Triplet Loss／centroid／Top-5 只是 baseline，不是指定方法；ArcFace、CosFace、非嵌入式分類或排序模型、手工特徵、集成皆可；「篩選 ≥2 局玩家」是 baseline 自己的設定。Q10：候選池棋譜可用於校正、正規化，但 Public／Private 候選池不得混用、不得人工標註測試資料 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q11：「原始對局是否統一採用日本規則？」「是。規則資訊（RU）沒有保留」。Q12：互先貼目統一 6.5 目，KM 未保留 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q13（文件自己標「此題待確認」）：「沒有排除」讓子棋；表格：HD[0] 互先、貼 6.5 目、約 80%、黑白標記與實際一致；HD[1] 讓先、不貼目和棋白勝、約 16%、一致；HD[2]～HD[9] 讓 2～9 子、不貼目和棋白勝、約 3%、「與實際相反」。三點注意：「1. 讓子石沒有記錄在棋譜中。……需要時請依下表的標準星位，自行補上黑子。2. 著手的 B/W 標記與實際黑白相反。 SGF 中第一手標為 B 的著手，實際上是白棋（上手）下的。……【二擇一：『color 欄位也依原始標記，請一併對調。』／『color 欄位已是實際顏色，不需對調。』】3. RE 結果欄位為實際勝負，不需對調。」範例：「(;CA[UTF-8]HD[2]RE[W+R];B[pp];W[dd];...」應視為「(;HA[2]AB[pd][dp]PL[W]RE[W+R];W[pp];B[dd];...」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q14：「RU、KM、HA、AB、AW、PL，以及玩家名稱、段位、日期等資訊都已移除。」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- Q&A **沒有**任何題目談到：棋譜伺服器、等級標籤如何指定、測試題同色／類別平衡、測試玩家是否與訓練玩家互斥 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- 實際 v1.1 檔案與 Q&A 範例不一致：Q&A 範例根節點有 `CA[UTF-8]HD[2]RE[W+R]`，但 v1.1 實際檔案只有 `SZ[19]HD[n]`，沒有 `RE[]`（Rinrin-lab 參賽者與本機核對均指出） — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)；本機 `/home/u2507115/BK_AICUP_2026-1/METHOD_NOTES.md` 第 482 行（非公開）

**官方資料集 repo 與 release**
- repo About：「2026 AI CUP 圍棋競賽訓練資料集：100 萬盤匿名棋譜，依 10 級棋力分檔，執黑／執白各半」；10 個 CSV（train_D … train_6D），每檔 100,000 列；欄位 `player_id`「玩家代號，同一玩家在所有列一致」、`game_id`「棋譜編號」、`rank`「該玩家的棋力等級，等於檔名的等級」、`color`「該玩家在這盤的執色，B 或 W」、`sgf_content`「單行 SGF，含棋盤大小、讓子資訊 HD[n] 與落子」；「棋譜已匿名化，不得用於識別真實使用者。」；「僅供 2026 AI CUP 競賽使用」；README 未說明等級如何指定、來源伺服器、測試集、授權 — [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)
- Release v1.0（9 月 3 日，jt851113）`aicup2026_go_training.tar.xz` 214 MB；v1.1（10 月 31 日）`aicup2026_go_training_v1.1.tar.xz` 210 MB、解壓約 1.3 GB，「其餘內容（列順序、player_id、game_id、rank、color、棋步）與 v1.0 相同」，範例 `(;SZ[19]HD[2];B[pd];…` — [Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases)

**官方教學 repo（baseline）**
- 作者「NDHU AI Lab 的 Serkan Kavak」；兩子題：棋力預測（10 級，「範圍從 12 級（12k）到 6 段（6d）」）與玩家辨識（metric learning）；等級對應原文「10 級為 D (10~12k), C (7~9k), B (4~6k), A (1~3k), 1D, 2D, 3D, 4D, 5D, 6D」（Rinrin-lab 轉錄版；官方 README 經轉換後顯示為「D (1012k), C (79k), B (46k), A (13k)」） — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)；[Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)
- Baseline 架構：`GoRankResNet`（棋力）與 `GoPlayerResNet`（玩家，Triplet Loss）；參賽者轉述細節為「ResNet 20 blocks、256 channels（約 26.5M 參數），500 個輸入通道＝最後 250 個局面的黑白盤面，cross-entropy」 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)；[Rinrin-lab](https://github.com/Rinrin-lab/AICUP-2026-AlphaGo)
- Baseline Public 分數：棋力預測 **0.4631**、玩家辨識 **0.2845**；「兩個 baseline 都是照 notebook 原樣跑出來的，沒有使用預訓練權重。」 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)
- 測試集結構：兩子題各有 public 與 private 各 **400 題**，每題 **5–20 局**；欄位 `question_id`（`pub_`／`priv_` 前綴）、`num_games`、`sgf_1..sgf_20`、`color_1..color_20`；提交一份 800 列的 `submission_rank.csv`（`question_id`,`pred_rank`）；「缺任何一份的題目即為缺題，該次提交無效、不予計分。」；「Public 分數在競賽期間即時公布於排行榜；Private 分數於競賽結束後公布，並作為最終排名依據。」 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)
- 玩家辨識候選池：每池 400 位玩家、每人 100 局（40,000 局）；「Public 與 private 兩池互斥，不可混用。」 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)
- 硬體：棋力 baseline 需 ≥9 GB VRAM，特徵約 1.7 GB WebDataset；玩家辨識特徵約 15 GB、約 100 萬小檔 — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)

**姊妹賽（競賽一：從圍棋棋譜集辨識玩家）**
- 「從查詢棋譜集，預測出是候選玩家集哪一位玩家下的棋。」同一主辦單位、同一時程（11/04 釋出測試、11/24 截止、11/25 公布 Private）；評分 e^-(r-1)（Top-1 1.000、Top-2 0.368、Top-3 0.135、Top-4 0.050、Top-5 0.018），400 題平均；baseline 0.2845；每日 5 次上傳；獎金結構與競賽二相同（NT$125,000）；2026-11-05 顯示 **353 人／241 組** — [AIdea 競賽一](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)
- aicup.tw 官方 2026 競賽頁列出兩個競賽的連結與「五支教學影片（EP1–EP5）、課程簡報、baseline repo」；「教學簡報僅供參考，請參賽者以競賽頁面公告為主。」 — [aicup.tw 2026 競賽頁](https://www.aicup.tw/ai-cup-2026-competition)
- 主題預告（7 月 9 日）：「透過深度學習分析真實人類棋譜，預測棋力與下棋風格」 — [aicup.tw 秋季賽主題](https://www.aicup.tw/post/ai-cup-2026-%E7%A7%8B%E5%AD%A3%E8%B3%BD%E7%AB%B6%E8%B3%BD%E4%B8%BB%E9%A1%8C%E6%90%B6%E5%85%88%E5%85%AC%E9%96%8B%EF%BC%81)
- 巡迴課程由顏士淨主講：9/24 高雄大學、10/2 暨南大學、10/13 文化大學、10/20 東華大學（機器學習／顏士淨） — [aicup.tw 巡迴課程](https://www.aicup.tw/post/%E3%80%90ai-cup-2026%E3%80%91%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E5%9C%8D%E6%A3%8B%E6%A3%8B%E5%8A%9B%E8%88%87%E9%A2%A8%E6%A0%BC%E9%A0%90%E6%B8%AC%E7%AB%B6%E8%B3%BD%E5%B7%A1%E8%BF%B4%E8%AA%B2%E7%A8%8B)

### Inferences
- 官方對預訓練模型（含 KataGo 權重、KataGo Human SL）的立場明確：允許、不需報備、須於報告揭露並附授權與（最好）消融實驗。這解除了本機 METHOD_NOTES 第 191 行記下的「預訓練權重合規未問」風險（本機檔案，非公開）。
- 「等級標籤如何指定」與「來源伺服器」官方始終未答；Q14 說玩家名稱、段位、日期已移除，暗示原始 SGF 本來帶有段位，標籤最可能是伺服器當時的段位（推測，無官方佐證）。
- 讓子棋的 `color` 欄位語意官方自己尚未定案（Q13「二擇一」），而 v1.1 又沒有 Q&A 範例裡的 RE[]，顯示官方資料說明文件與實際檔案並未同步校對；任何依賴 HD/color 的特徵都該對兩種解釋做穩健性測試。
- 競賽二的隊伍數從 138 組增至 277 組（約一倍），競賽一 241 組；兩賽同為 NDHU 智慧科技中心出題，同一份資料與教學。

### Gaps
- 公開排行榜：AIdea 頁面 HTML 不含任何分數；猜測的 `/leaderboard` 路徑回 404；排行榜需登入，依指示未嘗試。無法給出 Public 最高分與其時間序列。
- 棋譜來源伺服器、等級標籤指定方式、測試題是否同色／類別平衡、測試玩家是否出現在訓練集：官方頁面、README、Q&A 全無說明。
- 官方教學簡報（第 80 頁提及 pretrained 權重）與 EP1–EP5 影片的實際內容未取得（aicup.tw 頁面無直接連結；YouTube 播放清單頁面只能確認有影片，無法擷取標題）。
- AIdea 討論區（#topic-discuss）內容未能抓取（SPA 分頁，HTML 中無貼文）。
- FB 粉絲專頁（AICUPrealtask）需登入，未嘗試。

---

## 關鍵問題 2：主辦單位前屆圍棋競賽（2023 秋季賽「圍棋棋力模仿與棋風辨識」；2024、2025 無圍棋題）：排行榜、冠軍解法、Public/Private 教訓

### Takeaway
主辦單位（教育部 AI CUP，出題同為 NDHU AI Lab）只辦過一次圍棋前作：**2023 秋季賽「圍棋棋力模仿與棋風辨識競賽」**（三個模型 Dan／Kyu／PlayStyle，TensorFlow 2.9 baseline，T-Brain 平台，508 隊，總獎金約 31–32 萬）；2024 與 2025 的 AI CUP 均無圍棋題。該屆的得獎名單 PDF、排行榜與任何得獎隊伍的公開技術寫作（GitHub／HackMD／Medium／iThome）皆未能在公開網路找到；僅有兩則學校新聞（陽明交大 優等、逢甲 佳作）。因此「前屆 Public/Private 落差教訓」沒有可引用的公開資料。

### Cited Findings
- 2023 秋季賽題目「圍棋棋力模仿與棋風辨識競賽」，「報名日期：2023/09/01~2023/11/20」，「總獎金：31萬」，競賽頁 https://tbrain.trendmicro.com.tw/Competitions/Details/29，並列得獎名單貼文連結 — [aicup.tw 2023 競賽頁](https://www.aicup.tw/copy-of-ai-cup-2023-1)
- 獎金獵人頁面記載「總獎金320,000 TWD，最高獎金60,000 TWD」，與 aicup.tw「31萬」略有出入 — [bhuntr](https://bhuntr.com/tw/competitions/edr25rlnxmv2keh9i1)；對照 [aicup.tw 2023 競賽頁](https://www.aicup.tw/copy-of-ai-cup-2023-1)
- 2023 baseline repo：「There are 3 AI models required to train in this contest, namely Dan, Kyu, and Play Style models.」「The baseline models are trained by using TensorFlow 2.9」；含 `Dan Training Tutorial.ipynb`、`Kyu Training Tutorial.ipynb`、`PlayStyle Training Tutorial.ipynb`、`Create Public Upload CSV.ipynb`、`model_dan_tutorial.h5` 等；README 未給資料來源、評分與分數 — [AILAB-NDHU/AIcupTutorial](https://github.com/AILAB-NDHU/AIcupTutorial)
- 2023 教材公告（2023-10-05）：baseline code 連結即上述 repo；巡迴課程影片 https://reurl.cc/1GGWoW ；課程簡報 https://drive.google.com/file/d/17C5NP03aaBEQoo8duL_6kVd8waXC54Wl/view — [aicup.tw 教材出爐](https://www.aicup.tw/post/2023-ai-cup-%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E3%80%90%E5%9C%8D%E6%A3%8B%E6%A3%8B%E5%8A%9B%E6%A8%A1%E4%BB%BF%E8%88%87%E6%A3%8B%E9%A2%A8%E8%BE%A8%E8%AD%98%E7%AB%B6%E8%B3%BD%E3%80%91%E6%95%99%E6%9D%90%E5%87%BA%E7%88%90%E5%95%A6)
- 2023 競賽 9/1 開跑，主題標籤「#讓AI成為棋靈王」，由教育部主辦 — [aicup.tw 2023 秋季賽](https://www.aicup.tw/post/2023-ai-cup-%E7%A7%8B%E5%AD%A3%E8%B3%BD)；[en.aicup.tw](https://en.aicup.tw/post/2023-ai-cup-autumn-contest)
- 參賽規模：逢甲大學新聞稿稱得獎隊伍「從全國508隊中脫穎而出」，黃愈楨、蘇桓均獲佳作與一萬元獎金；做法描述為「先將大量棋譜轉換為標準 SGF 格式，抽取有利於預測的特徵，再建立深度神經網路進行預測」 — [逢甲應數系](https://apmath.fcu.edu.tw/en/news/2024032801/)；[中央社稿](https://www.cna.com.tw/postwrite/chi/366681)
- 競賽說明提到「CNN、ResNet、Transformer 等深度學習方法可以達到初步成果，若要更好的效果，則可考慮特徵設計或 meta learning、few-shot learning 等進階方法」（搜尋摘要轉述自競賽說明） — [逢甲應數系](https://apmath.fcu.edu.tw/en/news/2024032801/)
- 陽明交大資工系：袁賢銘教授指導高振群同學獲「競賽優等」（公告 2024-04-02），未描述方法與分數 — [NYCU CS 公告](https://www.cs.nycu.edu.tw/announcements/detail/11400)
- 得獎名單貼文（2024-01-08 發布、01-19 更新）只附 PDF「AI CUP 2023 公告版得獎名單-圍棋0119.pdf」(1.68 MB)，頁面 HTML 中抓不到 PDF 直接連結 — [aicup.tw 得獎名單貼文](https://www.aicup.tw/post/%E3%80%90%E5%BE%97%E7%8D%8E%E5%90%8D%E5%96%AE%E3%80%91ai-cup-2023-%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E3%80%8C%E5%9C%8D%E6%A3%8B%E6%A3%8B%E5%8A%9B%E6%A8%A1%E4%BB%BF%E8%88%87%E6%A3%8B%E9%A2%A8%E8%BE%A8%E8%AD%98%E3%80%8D%E7%AB%B6%E8%B3%BD)
- 2024 AI CUP 競賽列表為太陽能發電預測與金融問答等，無圍棋 — [en.aicup.tw 2024](https://en.aicup.tw/ai-cup-2024-competition)
- 2025 AI CUP 四題：玉山警示帳戶、電腦斷層心臟肌肉分割、桌球智慧球拍、醫病語音敏感資料；「None of the 2025 topics concerns Go or any board game」 — [aicup.tw 2025 競賽頁](https://www.aicup.tw/ai-cup-2025-competition)
- 2023 T-Brain 競賽頁已失效：tbrain.trendmicro.com.tw 301 轉到 www.trendaisecurity.com/Competitions/Details/29，該頁回 404；Wayback Machine 在此環境無法存取 — [T-Brain 原址](https://tbrain.trendmicro.com.tw/Competitions/Details/29)

### Inferences
- 2023 題目的核心是「模仿」（預測某棋力的下一手）與「棋風辨識」，與 2026 的「從 5–20 局預測等級」雖同出一個實驗室、同一類資料，但任務形式不同，前屆解法不能直接套用；真正可承接的是 2026 競賽一（玩家辨識）與 2023 PlayStyle 子題。
- 2023 的 Private/Public 落差、過擬合教訓沒有公開文獻；主辦單位在 2026 以「每日 5 次、最後一次為準」加「Private 400 題獨立」的規則，與一般 AIdea 競賽相同。

### Gaps
- 2023 得獎名單 PDF 無法取得（Wix 以 JS 載入檔案連結）；得獎隊伍（第一至三名）名稱、分數、學校皆無。
- 2023 排行榜（Public/Private 分數、隊伍數）無法取得（T-Brain 頁面已下線、Wayback 被封鎖）。
- 搜遍 GitHub／HackMD／Medium／iThome，找不到任何 2023 圍棋競賽得獎隊伍的公開程式或心得文。
- 2023 課程簡報（Google Drive）與影片內容未能讀取。

---

## 關鍵問題 3：本屆公開排行榜動態與參賽者公開討論（GitHub、FB、Discord、PTT、Dcard、知乎）

### Takeaway
公開可見的參賽者討論**只在 GitHub**：一個認真的實驗 repo（Rinrin-lab/AICUP-2026-AlphaGo）在玩家層級 held-out 上從 baseline 0.4631 做到約 0.658（含「對手視角」項），並記錄多項資料觀察（讓子比例、等級人數不均、1D–3D 最弱、對手等級相關 0.954）；另一個 repo（youyiwangwww）做競賽一，發現「開局指紋」是最穩定的訊號；其餘 repo（Kung56/cup2026、Danny3900/KyuDan、Ayandadlamini12/student_test）只是官方教學的複製。PTT、Dcard、Facebook、Discord、知乎均搜不到本屆討論。排行榜本身不可見。

### Cited Findings

**Rinrin-lab/AICUP-2026-AlphaGo（棋力預測實驗紀錄，分支 rank-prediction-experiments）**
- 分數表（held-out，玩家層級切分 801 人／97,126 局／2,388 模擬題；單 seed 標準誤約 ±0.008）：官方 baseline Public 0.4631；plus（局層級切分）0.623 但作者自認洩漏「高估約 0.07」；plus v1（玩家層級切分）0.553「第一個誠實分數」；plus v2（cosine LR＋ordinal label smoothing ε=0.1）0.621；move v1 0.638（10 seeds 0.634）；move v1 以 64 局面推論 10 seeds 0.640；v1+v3 集成 0.643；v1+v3+對手視角（權重 1.0）**0.6581**（採用） — [Rinrin-lab](https://github.com/Rinrin-lab/AICUP-2026-AlphaGo)
- 模型：「plus」系列 570 通道（加氣、提子、合法手）、8 重對稱增強、bf16；「move」系列（最佳單模）每手 18 通道、每局取樣 16 個局面、小 ResNet（6 blocks、96 ch）＋attention pooling，約 1.77M 參數；move v3 加局面層 Transformer 與 gate；「question」系列跨局 attention 的 v4 0.6231 未採用；最終 `predict_ensemble.py` 平均 move v1 與 v3 的 log-prob、64 局面／局、加對手視角項 — [Rinrin-lab](https://github.com/Rinrin-lab/AICUP-2026-AlphaGo)
- 未用預訓練權重；KataGo 列為「下一步」：「KataGo 走子品質特徵（實際落子的 policy 機率、勝率／目數損失）」，理由「唯一能補上『判斷好壞』、也能分辨同一盤棋中誰下得比較好的方向」；「競賽規則允許使用開源資源，需在最終報告揭露來源與貢獻。」 — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)
- 資料觀察：「玩家數差異大（A 1582 人、1D 433 人）」；「棋力越高棋局越短（D 235 手、6D 213 手）」；「100 萬局中沒有任何兩局的著手序列相同」（故對手視角無洩漏）；「兩人預測等級的相關 0.954」（支持伺服器依等級配對）；「1D～3D 一直是最弱的等級（約 0.44～0.49）」；差兩級以上的錯題從 21.6% 降到 11.9% — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)
- 作者指出 Q&A 與實際檔案不符（Q&A 範例根節點 CA/HD/RE，實際只有 SZ[19]），並建議向主辦確認 `color_k` 在讓子棋是否要對調 — [Rinrin-lab](https://github.com/Rinrin-lab/AICUP-2026-AlphaGo)

**youyiwangwww/aicup2026-chess-player-id（競賽一：玩家辨識）**
- 兩條基線：Baseline 0「Opening Fingerprint」（前 40 手目標玩家落子 19×19 熱圖，cosine 相似度，無訓練）；Baseline 1 Triplet（17 通道：8 自己＋8 對手＋1 執色，ResNet，TripletMarginLoss margin 0.2）；「Current strongest：Color-aware Opening」、「Opening是目前最穩定主訊號」；Strength Estimator（rlglab）、MiniZero、policy fingerprint 列為未整合；只用 train_A.csv 一檔；嚴格玩家互斥與 SGF 字串重疊檢查 — [youyiwangwww](https://github.com/youyiwangwww/aicup2026-chess-player-id)
- 無任何公開排行榜分數；mock 結果作者自註「僅驗證流程，不是真實模型效果」 — [youyiwangwww](https://github.com/youyiwangwww/aicup2026-chess-player-id)

**官方教學複製品（無新資訊）**
- Kung56/cup2026、Danny3900/KyuDan 文字與官方教學相同（含「NDHU AI Lab 的 Serkan Kavak」致謝、Public 0.4631／0.2845）；Ayandadlamini12/student_test 自稱教學副本 — [Kung56/cup2026](https://github.com/Kung56/cup2026)；[Danny3900/KyuDan](https://github.com/Danny3900/KyuDan)；[student_test](https://github.com/Ayandadlamini12/student_test)

**社群論壇**
- 搜尋「AI CUP 圍棋 棋力預測 Dcard／PTT／Facebook 2026」與精確詞「圍棋棋力與風格預測競賽」僅回傳 GitHub 與 aicup.tw 官方頁，無任何 PTT、Dcard、FB 貼文 — [搜尋結果示例：GitHub 資料集](https://github.com/AILAB-NDHU/aicup2026-go-dataset)
- 競賽頁指向 FB 粉絲專頁 https://www.facebook.com/AICUPrealtask（需登入，未抓取） — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)

### Inferences
- 公開可見的參賽者「上限」是 Rinrin-lab 的 held-out 0.658（非 Public 分數）；該 repo 的 1D–3D 最弱、對手等級高度相關兩點，與本機 METHOD_NOTES 的發現（1D/2D/3D 命中 0.34/0.32/0.37；對手通道是唯一標籤側訊息）一致（本機 `/home/u2507115/BK_AICUP_2026-1/METHOD_NOTES.md` 第 215、405 行，非公開）。
- 沒有任何公開參賽者提到利用測試集結構（同色、類別平衡）；外部資料亦只停在「下一步用 KataGo」。
- 兩個認真 repo 都獨立採用玩家層級切分並警告局層級切分會洩漏約 0.07，可視為本題的共識做法。

### Gaps
- 排行榜 Public 最高分、隊伍數隨時間變化：不可見（需登入）。
- Discord、知乎、Threads 等未搜到任何相關內容；是否有私密社群不可知。

---

## 關鍵問題 4：任務天花板的公開估計；是否有人利用測試集結構或外部資料；規則是否允許外部預訓練網路

### Takeaway
規則**明確允許**外部預訓練網路（含 KataGo 權重）與外部棋譜，條件是報告揭露（官方 Q&A Q1–Q5、Q9）。學術天花板來自兩篇密切相關的論文：Chen–Shih–Wu（ICLR 2025，中研院）的 Strength Estimator 在圍棋「15 局即達 80% 以上」等級準確率；Kuboki–Ogawa–Hsueh–**顏士淨（NDHU，本賽出題者）**–Ikeda（arXiv 2505.00279）用 KataGo HumanSL 多等級 policy 的幾何平均機率＋SE 分數＋KataGo 目數損失統計，餵 LightGBM，在野狐 11 級組上 n=5／10／20 局分別 64.8％／79.6％／91.7％（±1 級 98.8／99.9／100％）。但這些是在「同一等級組隨機抽局」的協定下，不是本賽的「同一玩家 5–20 局、10 級含密集 1D–6D」；公開參賽者的 held-out 僅約 0.66。沒有任何公開參賽者提到利用同色或類別平衡。

### Cited Findings

**規則面**
- 官方 Q&A Q1：可用 KataGo 公開權重抽特徵或當初始權重；Q2：可用外部公開棋譜；Q5：不需報備，報告須揭露來源、版本、授權、用法、貢獻（最好附消融）；Q9：「可以，而且需要在報告中揭露。」 — [官方 Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)
- 競賽頁規則第 4、5 點：外部自製或開源資料、生成式 AI 工具可用，需合法授權並於書面報告列來源與貢獻 — [AIdea 競賽二](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)

**天花板：Chen, Shih, Wu — Strength Estimation and Human-Like Strength Adjustment in Games（ICLR 2025）**
- arXiv:2502.17109（2025-02-24 投稿，03-21 修訂），ICLR 2025；「Our strength estimator significantly achieves over 80% accuracy in predicting ranks by observing 15 games only, whereas the previous method reached 49% accuracy for 100 games.」 — [arXiv 2502.17109](https://arxiv.org/abs/2502.17109)；[ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/17a1a1439421f1837e10cd612bf92861-Paper-Conference.pdf)
- 方法：「We use a neural network to estimate a strength score (β) of an action at a given state based on human game records」，「based on Bradley-Terry model, we sequentially minimize the loss to ensure that the strength scores are strictly in desired order」；另設 r∞ 為最弱等級處理未見狀態；「SE achieves over 80% accuracy within just 15 and 26 games in Go and chess」；程式與 Go／chess 模型 .tar.gz 公開 — [中研院專案頁](https://rlg.iis.sinica.edu.tw/papers/strength-estimator/)；程式 repo [rlglab/strength-estimator](https://github.com/rlglab/strength-estimator)
- 專案頁與摘要未說明圍棋資料伺服器、局數、等級範圍、等級類別數 — [中研院專案頁](https://rlg.iis.sinica.edu.tw/papers/strength-estimator/)

**天花板：Kuboki, Ogawa, Hsueh, Yen, Ikeda — Policies of Multiple Skill Levels for Better Strength Estimation in Games（arXiv 2505.00279，2025-05-01）**
- 作者單位：Kuboki、Ogawa、Hsueh、Ikeda 為 JAIST；Shi-Jim Yen 為「National Dong Hwa University, Taiwan」（即本賽出題單位所屬學校） — [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- 資料：「Fox Go dataset」（featurecat/go-dataset，約 2017 年對局）；11 個等級組 3–5k、1–2k、1d、2d…8d、9d；「we collected 5,000 matches for each of the 11 rank groups, using 55,000 matches in total」，每局兩個玩家資料點共 110,000；測試每組 900 局 — [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- 評估協定：每組隨機抽 n 個資料點（n∈{1,5,10,15,20}）重複 500 次，特徵取平均；準確率＝預測等級組命中比例，另報 accuracy±1；第 6 節另做「每人 20 局、每組 100 人」的玩家層級評估 — [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- 圍棋結果（提出法／Chen et al.；括號為 ±1 級）：n=1 37.5 (82.8)／33.7 (74.0)；n=5 64.8 (98.8)／56.3 (96.6)；n=10 79.6 (99.9)／71.4 (99.7)；n=15 87.5 (100)／78.2 (99.9)；n=20 91.7 (100)／83.8 (100) — [arXiv HTML](https://arxiv.org/html/2505.00279v1)；摘要：「accuracy of 80% in strength estimation when given 10 matches, which increased to 92% when given 20 matches」 — [arXiv abs](https://arxiv.org/abs/2505.00279)
- 特徵與模型：多等級 policy 直接用現成 **KataGo HumanSL**（10 個等級 10k, 8k, …, 2k, 1d, 3d, …, 9d）與 Maia（棋）；policy 特徵＝各等級下實際著手 prior 的幾何平均（等價於每手平均 log-likelihood）；加 Chen 的 SE 強度分數平均；加強網損失統計 MeanLoss／MedLoss／StdLoss（圍棋用 KataGo scoreLead）；「employed LightGBM with default settings for regression model training」後四捨五入到最近等級組 — [arXiv HTML](https://arxiv.org/html/2505.00279v1)

**較早文獻（背景）**
- Moudřík 等，Evaluating Go Game Records for Prediction of Player Attributes（2015）：以棋譜模式特徵預測棋力與棋風的早期工作 — [arXiv 1512.08969](https://arxiv.org/pdf/1512.08969)

**參賽者對天花板與結構利用的公開發言**
- Rinrin-lab 的 held-out 最佳 0.6581；1D–3D 約 0.44–0.49；差兩級以上錯題 11.9% — [Rinrin-lab](https://github.com/Rinrin-lab/AICUP-2026-AlphaGo)
- 官方巡迴課程講師即顏士淨 — [aicup.tw 巡迴課程](https://www.aicup.tw/post/%E3%80%90ai-cup-2026%E3%80%91%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E5%9C%8D%E6%A3%8B%E6%A3%8B%E5%8A%9B%E8%88%87%E9%A2%A8%E6%A0%BC%E9%A0%90%E6%B8%AC%E7%AB%B6%E8%B3%BD%E5%B7%A1%E8%BF%B4%E8%AA%B2%E7%A8%8B)
- 未在任何公開 repo、頁面找到參賽者提及「測試題同色」「類別平衡」或使用外部棋譜資料集；外部網路僅見 Rinrin-lab 的「下一步 KataGo」與 youyiwangwww 列為「未整合」的 Strength Estimator — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)；[youyiwangwww](https://github.com/youyiwangwww/aicup2026-chess-player-id)

### Inferences
- 出題者（顏士淨）本人合著的論文把「KataGo HumanSL 多等級 policy 幾何平均機率＋SE 分數＋強網目數損失→LightGBM」當作 state of the art，且在 10 局達 ~80%、20 局 ~92%（野狐 11 組）。這是對本賽「可以怎麼做」最強的間接官方訊號，也與 Q&A Q1 特別點名 KataGo 一致。
- 論文數字不能直接換算成本賽分數：(a) 協定是同等級組隨機抽局，不是同一玩家的 5–20 局——玩家固定偏差（同一人的所有題一起錯）在隨機抽局下被平均掉；(b) 野狐 3k–9d 的 11 組跨度大（±1 級 98.8% 即 n=5），本賽 10 級裡 1D–6D 密集、A–D 各含 3 級，公開參賽者實測 1D–3D 僅 0.44–0.49；(c) 本賽分數對相鄰級只給 1/e。合理推斷：本賽的實際天花板遠低於 92%，公開最佳 held-out 約 0.66，本機 fold 0 偷看標籤扣除每人固定偏差的診斷上界 0.844（本機 `/home/u2507115/BK_AICUP_2026-1/METHOD_NOTES.md` 第 215 行，非公開）是目前唯一的「結構性上限」估計。
- 兩篇論文都以 Fox（野狐）資料為實驗平台，而本賽日本規則、貼 6.5 目、讓先不貼目和棋白勝的設定不像野狐（中國規則、貼 7.5 目）；伺服器仍不可考（推測）。

### Gaps
- 沒有任何公開來源（官方或參賽者）給出本賽題型（10 級、5–20 局、同一玩家）的天花板估計。
- 無法確認 Chen et al. ICLR 2025 的圍棋資料伺服器與等級範圍（摘要與專案頁未載；全文 PDF 未讀）。
- 無人公開討論測試集同色、類別平衡或 Public/Private 題目分布；也無人公開使用外部棋譜資料集的結果。
- 本賽棋譜來源伺服器未知，故無法判斷 Fox 資料上訓練的 HumanSL／SE 權重與本賽分布的差距。
