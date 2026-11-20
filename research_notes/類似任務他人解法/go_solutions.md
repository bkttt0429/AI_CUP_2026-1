# 類似任務他人解法：圍棋棋力／段位預測的完整 pipeline 調查（2026-11-05）

對照基準（我方）：KataGo humanSL 18 個 rank-conditioned policy 對每手的 likelihood ＋ 微調的 10 級 rank heads（己方與對手著手）＋ 每局面 KataGo trunk 向量 → 小型 TCN/MLP，以整題（whole-question）loss 訓練；12 模型 seed／架構 ensemble；expected-score 決策。4-fold CV 0.726，Public 0.7625。題目：每題 5–20 局、單一顏色、有 HD 標記、無時鐘／結果／日期；10 類（D、C、B、A、1D–6D）；評分 exact 1 / 相鄰 e^-1 / 其他 0。

---

## Q1. AI CUP 歷屆圍棋題（2023／2024／2026）：誰贏、用什麼架構、如何跨局彙整

### Takeaway
2024 年 AI CUP **沒有**圍棋題；2023 秋季賽「圍棋棋力模仿與棋風辨識競賽」是三個子題（Dan 著手預測、Kyu 著手預測、PlayStyle），與本屆「從棋譜預測棋力等級」不同，且金銀銅得主與其方法在公開網路上找不到任何寫作或程式碼（只有逢甲佳作、陽明交大優等的獲獎新聞）。對我方最有價值的是**本屆（2026）其他參賽者的公開實驗紀錄**（Rinrin-lab）：純棋盤平面 ResNet＋attention pooling，不用 KataGo，自建 player-level 切分模擬賽分 0.658；其中 label smoothing、8 對稱增強、log-prob 平均、對手視角四項為可驗證的具體增益。

### Cited Findings

**AI CUP 2024（無圍棋題）**
- 官方 2024 競賽頁列出四題：太陽能發電預測、玉山 RAG/LLM 金融問答、生成式 AI 無人機導航（兩軌）、跨相機多目標車輛追蹤；無任何圍棋題 — [AI CUP 2024 競賽頁](https://en.aicup.tw/ai-cup-2024-competition)
- 「競賽二：從圍棋棋譜預測玩家棋力等級」「競賽一：從圍棋棋譜集辨識玩家」實際掛在 2026 頁面（頁首標題仍寫 AI CUP 2024，為網站沿用舊標題）— [en.aicup.tw/ai-cup-2026-competition](https://en.aicup.tw/ai-cup-2026-competition)

**AI CUP 2023 秋季賽「圍棋棋力模仿與棋風辨識競賽」**
- 報名 2023/09/01–11/20，總獎金 31 萬，官方頁面無得獎名單（連結標示「尚未公布」）— [AI CUP 2023 頁](https://en.aicup.tw/copy-of-ai-cup-2023-1)
- 獎項結構：金牌 1（6 萬）、銀牌 1（4 萬）、銅牌 1（3 萬）、優等 2、佳作 9；主辦建議方法 CNN、ResNet、GAN／Transformer — [獎金獵人](https://bhuntr.com/tw/competitions/edr25rlnxmv2keh9i1)
- 官方教材 repo 為 NDHU AI Lab 的 AIcupTutorial：競賽要求「3 個模型」：Dan、Kyu、PlayStyle；資料在 CSVs 資料夾；baseline 以 TensorFlow 2.9 訓練並附 .h5；README 未給 metric 與 baseline 分數 — [AILAB-NDHU/AIcupTutorial](https://github.com/AILAB-NDHU/AIcupTutorial)；教材公告（簡報 Google Drive、影片 reurl.cc/1GGWoW）— [aicup.tw 教材公告](https://www.aicup.tw/post/2023-ai-cup-%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E3%80%90%E5%9C%8D%E6%A3%8B%E6%A3%8B%E5%8A%9B%E6%A8%A1%E4%BB%BF%E8%88%87%E6%A3%8B%E9%A2%A8%E8%BE%A8%E8%AD%98%E7%AB%B6%E8%B3%BD%E3%80%91%E6%95%99%E6%9D%90%E5%87%BA%E7%88%90%E5%95%A6)；官方 YouTube 系列「手把手帶你成為下一個棋靈王 Part4：機器學習步驟」— [YouTube](https://www.youtube.com/watch?v=40QoKwAA9vo)
- 逢甲數據科學碩士學程（林英志指導，黃愈楨、蘇桓均）在 508 隊中獲佳作；做法：「先將大量棋譜轉換為標準 SGF 格式，再抽取有利於預測目標的特徵，建立深度神經網路」；使用校方 GPU 平台嘗試多種特徵組合與架構 — [逢甲統計系新聞](https://stat.fcu.edu.tw/news-detail/?id=41484)；[逢甲應數英文版](https://apmath.fcu.edu.tw/en/news/2024032801/)
- 陽明交大資工高振群（袁賢銘指導）獲「優等」；公告無方法細節 — [NYCU CS 公告](https://www.cs.nycu.edu.tw/announcements/detail/11400)

**AI CUP 2026（本屆）公開資料與他隊紀錄**
- 官方資料集 repo：100 萬局匿名棋譜，10 級各 10 萬，執黑／執白各半；欄位 player_id, game_id, rank, color, sgf_content；v1.1 於每局 SGF 加入 HD[n]；僅供 2026 AI CUP 使用 — [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)
- 官方教材（Serkan Kavak, NDHU AI Lab）：棋力題 GoRankResNet（ResNet 分類），特徵以 WebDataset gzip shards（約 1.7 GB）儲存；每子題 Public/Private 各 400 題、每題 5–20 局；**Baseline Public 0.4631**（棋力）、0.2845（玩家辨識）；需 ≥9 GB VRAM — [Kung56/cup2026（教材鏡像）](https://github.com/Kung56/cup2026)；同內容 — [Danny3900/KyuDan](https://github.com/Danny3900/KyuDan)
- **Rinrin-lab/AICUP-2026-AlphaGo（branch rank-prediction-experiments）實驗紀錄**，全部為自建 held-out 模擬分數、未報公開榜分數：
  - 驗證協定：以**玩家**切分（801 位 held-out 玩家、97,126 局、2,388 道模擬題、10 seeds、SE 約 ±0.008）；官方以「局」切分的驗證會**高估約 0.07** — [Rinrin-lab README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)
  - 主模型 move v1：每局從目標玩家非 pass 著手抽 16 個局面，每局面 18 通道（己／敵／空、最近 8 手、氣數、顏色），6-block×96ch 小 ResNet → 256-d，attention pooling 彙整 16 局面 → 10 類，約 1.77M 參數 — 同上
  - 訓練：batch 64 局（1024 局面）、AdamW lr 1e-3 wd 0.01、cosine（500-step warmup）30 epochs、**label smoothing 0.1、8 種對稱增強**、bf16；RTX 3060 每 epoch 約 55 分 — 同上
  - 跨局彙整：對每局的 log-prob 取平均再 argmax；**log 平均比機率平均高 +0.05～0.06** — 同上
  - 分數：官方 baseline Public 0.4631；plus 系列（570 通道）最佳 0.621；move v1 約 0.638；v1+v3 合併、每局 64 局面 0.643；**＋對手視角（同局以對手顏色再預測一次，w=1.0）0.6581**（10 seeds，+0.015）；對手與目標預測等級相關 0.954 — 同上
  - 無效／負向：整題（question-level）模型最佳 0.6231、10-seed 0.6245（低於 move v1）；**expected-score 決策 0.6177（低於 argmax）**；逐級 bias 校正無效；player-level Transformer 無效；真實著手平面無效；BatchNorm＋高 lr 造成 epoch 間大幅震盪 — 同上
  - 作者結論：label smoothing 貢獻了「smoothing＋cosine 約 +0.07」的大部分；1D–3D 一直是最弱等級（約 0.44～0.49）；模型判斷的是整局棋的強度而非誰下得較好；瓶頸在「著手品質判斷」；KataGo（實際著手 policy 機率、勝率／目數損失）列為下一步但尚無結果 — 同上
  - 讓子：官方 Q&A 稱 SGF root 有 HD/RE/CA 但釋出資料（v1.0）沒有；Q&A 稱約 16% 為讓 1 子、約 3% 為讓 2–9 子；作者加了 --fix-handicap、--swap-color-for-handicap 旗標預設關閉 — 同上
- GitHub API 搜尋（aicup 2026 go／圍棋／weiqi）只找到三個相關 repo：Rinrin-lab/AICUP-2026-AlphaGo、AILAB-NDHU/aicup2026-go-dataset、youyiwangwww/aicup2026-go-player-id（玩家辨識：17 通道盤面歷史＋ResNet encoder＋triplet loss）— [GitHub API](https://api.github.com/search/repositories?q=aicup+2026+go+OR+%E5%9C%8D%E6%A3%8B+OR+weiqi&sort=updated&per_page=30)

### Inferences
- 2023 題是「模仿特定棋力的著手預測＋棋風分類」，與本屆的「集合→等級」不同；即使找到得主方法，可直接移植的部分有限（最多是棋盤平面 CNN 的設計與訓練技巧）。
- Rinrin-lab 的 0.658（player-level 模擬）對照我方 CV 0.726，差距約 0.07，與我方使用 KataGo humanSL／trunk 特徵的方向一致（他們把 KataGo 列為下一步）；但其**四項具體技巧**仍值得在我方 pipeline 驗證：(1) label smoothing 0.1；(2) 8 對稱 TTA／增強（我方若在抽 trunk 向量時只用原盤面，可對 8 個對稱各抽一次取平均）；(3) 跨局 log-prob 平均而非機率平均；(4) 對手視角（我方已用對手著手，但可檢查是否以「對手預測等級」做為第二路訊號）。
- 他們的 expected-score 決策**輸給** argmax（0.6177 vs 0.638），與我方採 expected-score 的經驗相反；差異可能來自校準程度（label smoothing 後的機率較平坦）；建議我方在相同校準狀態下重測兩種決策。
- 「以局切分高估 0.07」是對我方 CV 協定的提醒：若我方 4-fold 以 player_id 切分則無此問題；若有任何 fold 混入同玩家局，Public 0.7625 > CV 0.726 的差距方向反而說明切分是乾淨的。

### Gaps
- 2023 金／銀／銅牌隊伍與方法：aicup.tw 得獎名單頁、TBrain 排行榜均未被搜尋索引到；Medium／HackMD／鐵人賽零篇心得。
- 2026 本屆：除 Rinrin-lab 外無其他隊伍公開方法；無人報公開榜分數可與我方 0.7625 對照。
- 官方 Q&A 原文（讓子比例 16%/3%）未直接取得，係轉引自 Rinrin-lab README。

---

## Q2. Chen, Shih & Wu（ICLR 2025）「Strength Estimator」完整 pipeline

### Takeaway
以 AlphaZero 式 20-block/256ch ResNet（18 輸入平面）端到端訓練一個「著手強度分數 β」，損失為 Bradley–Terry／listwise 排序 likelihood（每個 rank 抽 m=7 個 state-action 取平均 β，再按 rank 由強到弱做 softmax 排序損失），外加「隨機合法著手」的 r∞ 最弱級；推論時將玩家所有著手的 β 取算術平均（＝λ 的幾何平均），對照候選集算出的各級平均 β，取最近者。FoxWeiqi 11 級（3–5k … 9d）、495k 局訓練；**隨機從同一級抽 N 局**時 15 局 >80%、100 局 97.5%；容忍 ±1 級時 1 局約 80%、6 局 >99%。注意：此評估不是「同一玩家的 N 局」，Kuboki 等人（Q3）顯示改為同一玩家取樣時掉到 51.6/59.4/61.1%（5/10/15 局）。

### Cited Findings
- 論文與程式：ICLR 2025 正式版 PDF — [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/file/17a1a1439421f1837e10cd612bf92861-Paper-Conference.pdf)；海報頁 — [iclr.cc/virtual/2025/poster/30488](https://iclr.cc/virtual/2025/poster/30488)；專案頁僅提供 Go 與 chess 的預訓練模型 .tar.gz，**無 GitHub、無 README、無資料、無指令**，另有 SLsum/SLvote 對照圖與 SE∞-MCTS 結果 — [rlg.iis.sinica.edu.tw/papers/strength-estimator](https://rlg.iis.sinica.edu.tw/papers/strength-estimator)
- 模型定義：SE(p)=λ 對 state-action pair p=(s,a) 給強度；兩 rank 比較 P(r1≻r2)=λ1/(λ1+λ2)；n 個 rank 泛化為 P(r1≻{r2..rn})=λ1/Σλ；**composite strength 用幾何平均** Λi=(Πλi,j)^(1/m)「以確保估計穩定且不受 m 影響」，與原 BT 的乘積彙整不同、並「減輕離群值影響」— ICLR PDF §3.1
- 訓練：網路輸出 β，λ=e^β，故 Λi=e^{mean β}；likelihood L=Π_{i=1}^{n-1} e^{βi}/Σ_{j≥i} e^{βj}，loss 為其負對數（逐級 softmax）；每個 rank 抽 m 個 state-action、平均 β 後計算；**r∞**：把任一 rank 的 (s,a) 的 a 換成隨機合法著手，定義為最弱級、ri≻r∞ 恆成立，用來處理人類棋譜中罕見的 out-of-distribution 著手 — ICLR PDF §3.2
- 資料：FoxWeiqi 公開下載，業餘 5k 到 9d；分級 9d,8d,…,1d,1–2k,3–5k 共 **n=11**；排除 5k 以下（「kyu 棋力變動快、棋譜與等級不一致」）；訓練 **495,000 局（每級 45,000）**；測試分 candidate（1,100 局，每級 100，用來算各級平均 β）與 query（9,900 局，每級 900）— ICLR PDF §4.1
- 架構與超參：MiniZero 框架；20 residual blocks、256 channels；輸出 policy p、value v、強度 β 三頭；policy/value 損失同 AlphaZero；訓練時每級隨機取 **m=7** 個 state-action；18 輸入通道（過去 8 手黑白各 8＋2 通道落子方）；「也做了 data augmentation 以增加多樣性」（未明說是對稱）；SGD、lr 0.01 於 100k 步減半、共 130k 步；Go 約 242 GPU-hours（RTX A5000）；SE∞ 訓練時 r∞ 樣本不計 policy/value 損失 — ICLR PDF 附錄 A、Table 3
- 推論／決策：先用 candidate 集算每級 βi；對未知等級的 query 局集合算 βu；**判為 |βu−βi| 最小的 ri**（rank-mean 對齊，無逐手數對齊）— ICLR PDF §4.2
- 評估協定：對 query 集每個 ri「隨機選 N 局、評估其中所有 state-action」預測一次，每個 N 重複 500 次 — ICLR PDF §4.2（即**同級隨機抽局，非同一玩家**）
- 結果（Go）：SE 與 SE∞ 幾乎相同，**15 局 >80%、100 局 97.5%**；監督式分類 baseline SLsum／SLvote（11 類逐局面分類，sum＝機率加總、vote＝逐局面投票）**100 局仍只有 49%**；允許 ±1 級時 SE 1 局近 80%、6 局 >99%，SL 系列 100 局仍不行 — ICLR PDF §4.2、Fig. 2a
- 逐級觀察：9d 最準（Fox 最高級含職業與 AI，與 8d 落差大）、8d 最差；3d 低於平均（新帳號初始最高只能選 3d，許多 3d 實際更強）；3–5k 低於平均（kyu 不穩定）— ICLR PDF §4.2、Fig. 2b
- 部分著手：只用每局 1 個著手、20 局時約 60%（全著手 80%）；**只用前 50 手（布局）整體掉、kyu 級掉最多**（「各級布局著手相似」）；只用後 50 手（官子）整體也掉，但 9d 幾乎不掉、8d 掉很多 — ICLR PDF 附錄 B.2–B.4
- 少量資料實驗：只用 2 級或 3 級的資料集，SE 仍可在評估若干局後達 >80%（圖 5）— ICLR PDF §4.4
- Chess（Lichess，Elo 1000–2600 分 8 級；candidate 960 局、query 9,600 局）：26 局 >80%、100 局 93.38% — ICLR PDF §4.5
- 摘要數字：「observing 15 games only >80%，previous method 49% for 100 games」；SE-MCTS 對人類著手吻合 51.33% vs 42.56% — [iclr.cc poster](https://iclr.cc/virtual/2025/poster/30488)
- 相關先前工作（論文引用）：Moudřík & Neruda 2016 三級分類＋sum/vote 彙整；Liu et al. 2020 以 MCTS 強度指標 z 對弈約 20 局收斂 — ICLR PDF §2.4

### Inferences
- **與我方差異 1（訓練目標）**：他們把「等級」當作有序排序問題，用 listwise BT 損失學一個標量 β，而非 10 類 CE；優點是天然 ordinal、可跨級比較、對 label 的「同級內實際強弱不一」容錯（β 只需排序正確）。可直接在我方 10-rank heads 旁加一個 BT 排序頭（每 batch 抽各級 m 個局面取平均 β 再做逐級 softmax），再把 β 的 bag 平均作為 TCN/MLP 的一維特徵或輔助損失。
- **與我方差異 2（r∞ 負類）**：隨機合法著手當最弱級是零成本的資料增強，可穩定 OOD 著手的 β；我方 humanSL likelihood 本身已對爛手給低機率，但微調 heads 可仿效。
- **與我方差異 3（彙整與決策）**：β 全著手平均→最近級均值（rank-mean 對齊）本質上是一維 nearest-centroid；比我方的 expected-score 更粗，但其「用 candidate 集重估各級均值」等同於在測試分佈上做校準 — 若 Public 與訓練分佈有偏移，可用 Public 題（無標籤，transductive）做 β 均值的 shift 校正。
- **可比性警告**：80%@15 局是「同級隨機抽局」；同一玩家 15 局只有約 61%（Kuboki Table 4 的 Chen 欄）。我方題目正是同一玩家 5–20 局，故真正可比的是 Q3 的 player-specific 數字。
- **階段資訊**：布局最不具鑑別力（尤其 kyu）、官子對高段最具鑑別力 → 支持我方 TCN 以手數為軸、並可考慮依手數分段 pooling（KataRank 亦如此，見 Q6）。
- 「data augmentation」存在但未明說對稱；我方若尚未對 trunk 抽取做 8 對稱 TTA，可試。

### Gaps
- 專案頁只有權重 tar.gz，無訓練程式；無法確認 augmentation 是否為 8 對稱、或 β 頭的細節（如是否 per-move mask）。
- 論文無「同一玩家 N 局」的結果（由 Kuboki 補上）。
- 無逐手數的 rank-mean 對齊（我方假設的「per move number alignment」在論文中不存在；對齐只在整體 β 層級）。

---

## Q3. Kuboki, Ogawa, Hsueh, Yen, Ikeda（arXiv 2505.00279 / AIIDE 2025）：多棋力 policy 特徵＋LightGBM

### Takeaway
在 Chen 等人的 β 之上加入 (a) **KataGo HumanSL 10 個棋力 profile 對實際著手機率的幾何平均**、(b) **KataGo scoreLead 的每手損失統計（mean/median/std，前 N_cut 手）**，將所有特徵在 N 局上取平均後送 **LightGBM 迴歸（預設參數）→ 四捨五入成級**。Fox 11 級、每級 5,000 局訓練，每級每個 n 合成 10,000 個袋。隨機抽局：5/10/15/20 局 64.8/79.6/87.5/91.7%（Chen 56.3/71.4/78.2/83.8%）；**同一玩家抽局：5/10/15 局 58.1/61.0/63.2%（Chen 51.6/59.4/61.1%；±1 級 95.3/96.2/96.6%）**。消融顯示 HumanSL prior 是主要增益，loss 特徵在同一玩家情境才有幫助。

### Cited Findings
- 書目：Kyota Kuboki, Tatsuyoshi Ogawa, Chu-Hsuan Hsueh, Shi-Jim Yen, Kokolo Ikeda，2025-05-01 投稿，25 頁 15 圖，cs.LG — [arXiv abs](https://arxiv.org/abs/2505.00279)；AIIDE 2025 會議版（頁面受 bot 防護無法讀取）— [ojs.aaai.org AIIDE 36812](https://ojs.aaai.org/index.php/AIIDE/article/view/36812)
- 多棋力 policy：Go 用 **KataGo HumanSL**，「10 個棋力等級」，列出 10k, 8k, …, 2k, 1d, 3d, …, 9d，profile 為 **preaz_xx**（因 2017 Fox 資料早於 AlphaZero 流行開局）；chess 用 Maia-1100…1900（9 級）；β 用 Chen 等人釋出的模型（含 r∞）；強 AI 評估：KataGo scoreLead、Leela Chess Zero — [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- 每手→每人特徵：Strength：k 個 (s,a) 的 β 算術平均；Priors：每個 profile j 的 **GM_j=(Π p_{j,i})^{1/k}**（實際著手機率的幾何平均），每 profile 一個值；Losses：l_i=v'_i−v_i（Go 用 scoreLead），取前 N_cut∈{50,100,∞} 手的 MeanLoss／MedLoss／StdLoss；Go 最終選 MeanLoss(N_cut=100) 與 MedLoss(∞)；chess 選 MeanLoss 與 StdLoss(50)；**N 局特徵取平均，不用直方圖** — 同上
- Meta 模型：**LightGBM 迴歸、預設設定**，預測值四捨五入為級；訓練集建構：每級重複 10,000 次（Go）／26,000 次（chess）：抽 n 局、平均特徵、配對級索引；n∈{1,5,10,15,20} — 同上
- 資料：Fox Go 11 級（3–5k, 1–2k, 1d–9d）；訓練每級 5,000 局（共 55,000）→ 110,000 個訓練點；隨機抽樣測試每級 900 局（9,900 點）；**玩家特定測試：每級 100 位玩家 × 20 局（22,000）**；chess Lichess 8 級 — 同上
- Table 1（隨機抽局，exact%（±1%））Go：n=1 37.5(82.8) vs Chen 33.7(74.0)；n=5 64.8(98.8) vs 56.3(96.6)；n=10 79.6(99.9) vs 71.4(99.7)；n=15 87.5(100) vs 78.2(99.9)；n=20 91.7(100) vs 83.8(100) — 同上
- **Table 4（Go，玩家特定抽局）**：n=5 Proposed 58.1(95.3) vs Chen 51.6(93.1)；n=10 61.0(96.2) vs 59.4(96.0)；n=15 63.2(96.6) vs 61.1(96.9)；同表隨機抽局欄 70.4/82.8/89.4（Proposed）— 同上
- Table 2 消融（Go 隨機抽局，Use All / w/o Strength / w/o Prior / w/o Loss）：n=5 64.8/60.0/58.1/64.8；n=10 79.6/73.5/73.0/79.8；n=20 91.7/87.7/86.4/91.9 — 同上
- Table 7 消融（Go 玩家特定）：n=5 58.1/53.1/52.5/**56.7**；n=10 61.0/56.5/59.0/61.7；n=15 63.2/57.8/61.1/62.8（loss 在玩家特定情境下對 n=5 有 +1.4）— 同上
- 單一 prior 很弱：chess n=20 僅 Maia-1500 13.8%、僅 Maia-1900 33.0%、兩者 65.1%（附錄 C）；「priors 與 losses 單獨皆弱」（附錄 F）— 同上
- 誤差分析：Chen 法在極端級較準、本法在中間級較準（附錄 B）；玩家特定抽局時錯誤更分散到相鄰級（附錄 D）；chess 各級 β 分佈重疊較多，故 prior/loss 在 chess 增益更大（§6.3）；玩家特定增益 Go 2.1、chess 9.6 個百分點（結論）— 同上
- 無程式碼連結 — 同上

### Inferences
- **我方已是其超集**（18 個 humanSL profile vs 10；微調 rank heads；trunk 向量；學習式 TCN 而非平均＋GBM），所以他們的主要訊號我方已有；真正可補的是 **KataGo 評估損失特徵**（scoreLead 每手損失 mean/median/std，前 100 手）：在同一玩家 5 局情境給 +1.4 點，雖小但便宜（我方若已跑 KataGo trunk，可一併輸出 scoreLead）。
- **玩家特定上限的參考值**：11 級 Fox、15 局、精確率約 63%、±1 級 96.6% → 以 AI CUP 評分（相鄰 e^-1≈0.368）換算約 0.63＋0.336×0.368≈0.75。我方 CV 0.726／Public 0.7625（10 級、5–20 局）與此同一量級，暗示剩餘誤差主要是**玩家本身跨局不一致＋等級標籤噪音**，不是特徵缺失。
- 幾何平均（log 平均）彙整 policy 機率、特徵在局間平均，與我方 whole-question 訓練精神一致；他們的「每級合成 10,000 個袋」是對 bag-level 訓練資料的大量重抽樣，我方 whole-question loss 若樣本數有限，可類似地對同玩家局做 bootstrap 組題擴增。
- 預設 LightGBM 迴歸＋四捨五入即達此成績，顯示 bag-level 的非線性需求不高；我方 12 模型 ensemble 的邊際效益可對照此檢查。

### Gaps
- 無程式碼；AIIDE 版本頁面被 bot 防護擋住，是否有修正數字未知。
- preaz 的確切 10 個 rank 清單（原文列舉與「10」不吻合）。
- 未報告 n=2–4、6–9；未報告 10 級以外（kyu 低段）的表現。

---

## Q4. Neubauer（TU Wien 2024 Diplomarbeit）：KataGo trunk 向量＋Set Transformer，對照 Glicko-2

### Takeaway
以 KataGo kata1-b18c384nbt 的 **trunk 輸出（19×19×384）在「落子點」的 384 維向量**（pre-move 盤面）作為每手特徵，不用任何 head 輸出；把一位玩家最近 ≤500 手（約 4 局）當**無序集合**，用 Set Transformer（ISAB、61 個 inducing points、d=254、深度 5、2.47M 參數）回歸出 rating；訓練目標＝未來 10 局後的 Glicko-2 rating（MSE）＋ 兩人 rating 經 Bradley–Terry 預測勝負的 NLL ＋ L2。OGS 2021 資料（10k 訓練、5k 驗證、5k 測試、5k「稀少歷史」展示集），排除讓子、未排名、≤5 秒/手。結果：測試集 log-likelihood **−0.58（＝Glicko-2 −0.58）**；雙方 ≤4 局歷史時 **−0.55 vs Glicko-2 −0.64**。用 head 輸出（勝率、目差、policy 值、損失）的 basic model 明顯較差（−0.61／−0.57），點數損失單一特徵的 stochastic model 更差 → **trunk ≫ head 輸出**。

### Cited Findings
- 書目：Peter Neubauer，指導 Günther Raidl，2024-10-21，86 頁，關鍵字 skill assessment, rating systems, Set Transformer, transfer learning — [reposiTUm](https://repositum.tuwien.at/handle/20.500.12708/212251?mode=full)；全文 — [neubauer_24.pdf](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- 資料與清理：OGS 2021 collection；排除非 19 路、**讓子局**、≤5 秒/手、<20 手、未排名；只取 counting／resignation／timeout 結果，後兩者勝者改以 KataGo 判定並丟棄 KataGo 信心 <60% 的局；**訓練 10,000 局、驗證／測試各 5,000、展示集 5,000（雙方最近 ≤4 局）**；訓練局額外要求標籤與勝負一致且雙方有 ≥10 局未來歷史；**每位玩家最近手數窗 N=500** — 論文 §4.1
- 特徵：KataGo 網路 kata1-b18c384nbt-s9131461376-d4087399203；輸入 18 空間＋10 全局特徵（KataGo 原規格）；basic model 用 head 輸出（pre/post 勝率、pre/post 目差、著手 policy 值、最大 policy 值 → 歸一化後 6 維）；**full model 只用 pre-move trunk 輸出 b×b×c（c=384）在落子位置的向量** — 論文 §4.2、§4.5–4.6；README 稱之為「pick features」，另有 head／full trunk 兩種 — [Animiral/go-strength-model](https://github.com/Animiral/go-strength-model)
- 架構：Set Transformer，編碼器為 ISAB（m 個 inducing points，O(mn)），解碼器為單一 attention query（seed vector）pooling → FC → 標量 rating；LayerNorm＋residual — 論文 §4.4
- 訓練目標：rating 標籤 r(p|t)＝該玩家 **A=10 局之後**的 Glicko-2 rating（標準化：均值 1623.19、SD 315.81）；損失 L_t = L_s（BT 勝負 NLL，γ=s_r·ln10/400）＋ τ_r(L_b+L_w)（雙方 rating MSE）＋ τ_θ L_θ；驗證只用 L_s；Adam — 論文 §4.7
- 最佳超參（隨機搜尋五輪）：lr 2.14e-4、decay 0.995、τ_r 2.05、τ_θ 73.9、深度 5、d=254、d_q=154、**m=61**，2,469,388 參數，驗證 λ≈−0.578；初步搜尋顯示**窗口 N 越大越好**，故固定 500；平均 30 epochs 早停 — 論文 §5.2、Table 5.1
- 主結果 Table 5.2（accuracy α / log-lik λ；Train／Val／Test／Exhibition）：50:50：—/−0.69 全部；**Glicko-2：89.7%/−0.40、68.6%/−0.58、69.1%/−0.58、64.4%/−0.64**；Future Glicko-2（上界）：75.4%/−0.51 測試、83.4%/−0.38 展示；Stochastic（只用點數損失）：58.5%/−1.18 測試、64.4%/−2.03 展示；**Basic（head 特徵）：65.2%/−0.61 測試、70.9%/−0.57 展示；Full（trunk）：68.8%/−0.58 測試、73.7%/−0.55 展示** — 論文 §5.3
- 結論與偏差：500 手≈4 局即與 Glicko-2 相當；預測向均值回歸、低分段更分散、**完全不給最高分段**；可能原因：KataGo trunk「未針對區分爛手最佳化」、訓練分佈集中於 SDK（建議改為跨 rating 均勻抽樣）；模型對 KataGo 棋風有偏好（其他強 bot 被判較低）— 論文 §6.1–6.2
- 資料偏差：OGS 以 Glicko-2 配對，兩系統在同資料上比較時對 Glicko-2 有利有弊；OGS 曾改制並回溯 — 論文 §6.3
- 程式與重現：訓練碼、KataGo fork（僅 CUDA）、goratings fork 皆公開；README 無報告數字；網站 howdeepisyourgo.org 將原始輸出以 a≈334.03、b≈1595.1 線性映射到 OGS 尺度；查詢限 100 KB／1000 手 — 論文 §7.1；[GitHub](https://github.com/Animiral/go-strength-model)；[KataGo fork](https://github.com/Animiral/KataGo)
- OGS 論壇回饋：作者稱 4–5 局時「約與 Glicko-2 一樣準」；使用者測試 5 局 correspondence 得 2.0d、5 局長時限 1.2d、5 局快棋（Tygem）0.1k，認為與快棋品質較差一致；有人要求 MSE/RMSE 但未給；討論舊局與時間切分問題 — [OGS 論壇 How Deep is your Go?](https://forums.online-go.com/t/how-deep-is-your-go/53060)

### Inferences
- **與我方差異 1（集合 vs 序列）**：他們把 500 手當無序集合用 inducing-point attention 挑「少數關鍵手」；我方用 TCN（有序、按手數）。兩者可互補：在我方 bag-level 加一層 ISAB／attention pooling 跨局跨手（KataRank 亦如此），可能比平均更會抓「決定性失誤」。
- **與我方差異 2（trunk 取點）**：他們只取**落子位置**的 384 維向量（pre-move），不是全盤 pooled trunk。若我方用的是全盤平均／policy-head 前的 pooled 向量，可試改為（或加上）「落子點向量」＋「KataGo 最佳點向量」差；他們的 basic vs full 對照（−0.61→−0.58、展示 −0.57→−0.55）是 trunk 有用的直接證據。
- **排除讓子**：他們直接丟棄讓子局；我方有 HD 標記且 v1.1 才補上，與 Rinrin-lab 的疑慮一致 — 讓子局的對手視角特徵需另行處理（HD≥2 的局無部分子）。
- 他們的標籤是「未來 rating」（去噪），我方標籤固定；但「label 來自未來、模型向均值回歸」提示可對我方極端級（D、6D）做 class-prior 重加權。
- 不可比的部分：評分是勝負 log-lik 而非等級精確率；無法換算到我方指標。

### Gaps
- 無逐級準確率、無 RMSE；展示集之外未報告「局數 vs 準確率」曲線。
- 釋出權重是否存在不明確（README 提到 nets/search2 目錄但未說明）。

---

## Q5. Moudřík & Baudiš（2012–2016）：Pachi 樣式特徵→bagged NN；後續 CNN

### Takeaway
每局用 Pachi 匯出每手屬性，彙整成每位玩家的評估向量（最常見 1000 個空間樣式的出現頻率／局數、ω-local sente/gote 序列數、邊距×手數 16 格直方圖、提子 9 格直方圖、勝負與點差統計），以 20 個單隱層（20 單元）NN 的 bagging 迴歸 rank。KGS 26 級（20k–6d）× 每級 120 個樣本、每樣本 10–50 局：**RMSE 2.66 級（均值基準 7.50）；只用樣式特徵 2.79**，其餘特徵加總只再降 0.13。2016 年改用 CNN（盤面＋前 4 手，3 類），188,700 局 KGS，單局面 71.5%、整局彙整 77.9%。程式開源於 GoStyle。

### Cited Findings
- 2015 論文全文 — [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)；2012 前作「On Move Pattern Trends in a Large Go Games Corpus」— [arXiv 1209.5251](https://arxiv.org/abs/1209.5251)
- 原始處理：Pachi 對每手輸出 atari 旗標、atari 逃脫、提子數、與前手的 gridcular 距離、邊距、空間樣式（大小 2–6）— 1512.08969 §III.A
- 特徵：樣式＝整庫最常見 N=1000 個樣式（黑先正規化、旋轉鏡射不變）在樣本中的出現次數 ÷ 局數；ω-local（ω=10）sente/gote 序列每局平均數；邊距 {1–2,3,4,≥5} × 手數 {1–10,11–64,65–200,≥201} 16 格直方圖；提子 3 階段 × {己、敵、差} 9 格；勝負方式 4 種＋點差 2 個 — §III.B–F
- 模型：bagged feedforward NN（1 隱層 20 單元、sigmoid、RPROP ≤100 迭代）× 20；目標線性縮放到 [−1,1]；前作中 stacked ensembles 最佳但此文選 bagged NN；基準為均值迴歸 — §IV.A–B
- 資料：KGS 存檔；每位玩家在某 rank 期間的局 ≥10 局才用，>50 局則隨機抽 10–50 局（避免固定截斷造成偏差）；**26 級各 120 個樣本**；y=20（20k）…1（1k）、0（1d）…−5（6d）— §V.A
- 結果 Table I（RMSE，10-fold×5）：均值 7.503；**Patterns 2.788**；ω-local 5.765；Border 5.818；Captured 5.904；Win/Loss 6.792；Win/Loss Points 5.116；**全部 2.659** — §V.A
- 棋風資料：GoGoD 25 位職業各 192 局分成 12 組×16 局，專家問卷 4 個量表；全特徵 RMSE 1.615（均值 2.168）— §V.B
- 程式：GoStyle 專案、Python、Orange、FANN、Pachi — §VIII；網站 http://gostyle.j2m.cz
- 2016 CNN：Moudřík & Neruda, "Determining Player Skill in the Game of Go with Deep Neural Networks", TPNC 2016, LNCS 10071；188,700 局；三級；**單局面 71.5%、整局 77.9%** — [researchr](https://researchr.org/publication/MoudrikN16)；[Springer DOI](https://link.springer.com/10.1007/978-3-319-49001-4_15)（需登入）
- OGS 論壇 2016 貼文（作者 bronislav/Josef）：輸入「整盤＋前 4 手」、三類（弱／中／強）、單次預測約 71%（開局更低）、整局約 78%、96% 情況真實類別在前兩名；OGS API 資料；Keras；i7＋GTX 580 每局 <1 秒；OGS 方面提到可用於臨時段位與 sandbagger 偵測 — [OGS 論壇](https://forums.online-go.com/t/strength-prediction-using-cnn-collaboration-with-ogs/8485)
- Chen 等人把 Moudřík 2016 的 sum／vote 彙整做成 SLsum／SLvote baseline，11 級下 100 局僅 49% — ICLR PDF §2.4、§4.2

### Inferences
- 樣式頻率特徵「除以局數」的正規化與我方 bag-level 平均同理；其中**樣式頻率**（而非著手品質）單獨就解釋了絕大部分 RMSE 下降，這是與我方「品質為主」訊號正交的「習慣／棋風」訊號；可用便宜的 3×3／5×5 樣式直方圖（黑先正規化、對稱不變）作 bag 特徵試驗。
- 他們的樣本規模（10–50 局）與我方（5–20）相近，RMSE 2.66 級（26 級尺度）顯示純統計特徵在 10 級尺度約相當於 ±1 級，低於我方現況，驗證了 KataGo 特徵的必要性。
- 監督式逐局面分類再 sum/vote 的天花板（49%@100 局，11 級）是 Chen 的核心反例：我方 rank heads 是逐手分類的變體，但因接 TCN/whole-question loss 而非單純 sum，已避開此限制。

### Gaps
- Moudřík 2016 blog（jmoudrik.github.io，含 CNN 細節與 71,119 局數字）已 404；deep-go-wrap 的 CNN 架構細節未取得。
- 2015 論文無「局數 vs RMSE」曲線。

---

## Q6. 其他圍棋棋力推估工具／論文（GoKibitz、OGS、KataRank、野狐／弈客、日文文獻、西洋棋類比）

### Takeaway
最接近我方架構的公開專案是 **KataRank**（2026，GitHub）：KataGo trunk＋10 個標量/手、黑白雙流 Set Transformer＋因果交叉注意力、分段（開／中／終）attention pooling、**ordinal logistic 頭（29 級 20k–9d）**、損失＝rating MSE＋BT pairwise＋以 HumanSL rank 為錨的 ordinal NLL；Full（含 trunk）±1 準確率 70.7%、MAE 1.26 級，只用 10 個標量 57.2% → trunk 約 +13 點。弈客「鷹眼」是 KataGo 復盤報告（含「吻合度」）但演算法未公開；野狐 AI 棋力評估查無資料；日文文獻有 2016 IPSJ 單局 CNN 棋力迴歸（囲碁クエスト 13 路）與 2018 FIT policy-network 評估著手推估棋力（無法取得全文）；西洋棋 CNN-LSTM（Omori & Tadepalli 2024）MAE 182、拿掉時鐘變 239。

### Cited Findings
- **KataRank**（ahillzhao-msn）：29 類 20k–9d；輸入 KataGo「trunk embeddings、policy priors、value estimates」；Full 模型每手 1034 維（含 trunk）、Lite／Mobile 只用 10 個標量；主訓練目標 meanLogPrior（KataGo policy 對實際著手的 log prior 平均），並以 HumanSL profile 的 rank 錨（humanRankIdx/Exp/Sd）；架構 DualViewSetTransformer：黑白兩流各自 ISAB 編碼、**因果交叉注意力（只看已下的對手著手）**、StreamPooling 對開局／中盤／終盤分段 attention pooling、OrdinalLogisticHead；損失＝rating MSE＋Bradley–Terry pairwise＋rank-anchor ordinal NLL；參數 Full 2.26M、Lite 540K；**結果：Full ±1 準確率 70.7%、rating 相關 0.9936、rank MAE 1.26；Lite 57.2%／1.73；Mobile 42.4%／2.48**（Mobile 可能偏樂觀，b6c96 在同一批局微調過）；作者稱 Full 領先 Lite 13 點「可能主要來自 trunk」，trunk A/B 待做；資料來自 GoPredict 資料庫（局數未給）；需客製 KataGo fork 的 batch_analysis — [GitHub KataRank](https://github.com/ahillzhao-msn/KataRank)
- go-rank-estimator（maksimKorzh）：以 KataGo 處理 SGF 給出段位估計的 Python 腳本；README 未說明統計量與映射 — [GitHub](https://github.com/maksimKorzh/go-rank-estimator)
- 弈客「鷹眼」：App Store 說明「基於高算力圍棋 AI KataGo 生成復盤報告，多維度數據包括勝率圖、選點與變化圖、評分、**吻合度**、複雜度與穩定性」；吻合度計算方式未公開 — [App Store 弈客圍棋](https://apps.apple.com/app/id988682022)
- 日文：荒木伸夫・保木邦仁・村松正和「畳み込みニューラルネットワークを用いた囲碁における1局の棋譜からの棋力推定」情報処理学会論文誌 Vol.57 No.11（2016-11）：以 CNN 從**一局**棋譜估計棋力；用囲碁クエスト 13 路棋譜訓練；估 rating 的均方誤差小於既有方法；另做上級／中級／初級三分類，比較「先估 rating 再分類」與「直接分類」兩法 — [IPSJ 記錄](https://ipsj.ixsq.nii.ac.jp/records/175954)（頁面抓取時 503，內容取自搜尋摘要）；PDF — [IPSJ-JNL5711005.pdf](https://ipsj.ixsq.nii.ac.jp/record/175954/files/IPSJ-JNL5711005.pdf)
- 日文 FIT2018「用圍棋 AI 的 policy network 評估學習者棋譜中的著手以推估棋力」— [F-022.pdf](https://www.ieice.org/publications/conference-FIT-DVDs/FIT2018/data/pdf/F-022.pdf)（HTTP 405，無法讀取）；2019 小坂悠登碩士論文「囲碁AIを用いたプレイヤーの棋力推定」— 搜尋摘要提及，未取得全文
- 高知工科大 2025 學士論文摘要：把 Chen 等人的 Strength Estimator（MiniZero）用於**將棋**（floodgate 棋譜，約 160 萬局面），在未知棋譜上多數 rank 達 90% 以上估計精度（PDF 文字擷取為亂碼，數字僅由可辨識片段判讀）— [a1260313.pdf](https://www.kochi-tech.ac.jp/library/ron/pdf/2025/03/13/a1260313.pdf)
- 台灣：陽明交大吳毅成團隊「具 40 個穩定棋力等級的圍棋程式，具棋力檢測功能，可從棋局自動判斷玩家棋力」（新聞，無技術細節）— [公視新聞](https://news.pts.org.tw/article/416322)
- 西洋棋 Omori & Tadepalli 2024「Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM」：12 個棋子平面＋每手剩餘時鐘；4 層 CNN→每手向量＋時鐘→BiLSTM→FC；Lichess 約 120 萬局（2021-04～2024-07，每月 3 萬）；80/20 切分；標籤為雙方 rating（均值 1514、SD 366）；**MAE 182；無時鐘 239；均值基準 346**；Classical 151；每手輸出一條 rating 軌跡、取最後值；無跨局彙整；限制：高分局稀少（約 2922/3163 的局被估成 ~1460）、快棋隨機性 — [arXiv 2409.11506](https://arxiv.org/html/2409.11506v1)
- 西洋棋「Rating Your Moves: Guess the Elo, Move by Move」（Chess.com 部落格，非論文）— [chess.com](https://www.chess.com/blog/Chessable/rating-your-moves-guess-the-elo-move-by-move)

### Inferences
- KataRank 是「我方＋Neubauer」的合體：trunk＋humanSL 錨＋ordinal 頭＋BT 損失＋分段 attention pooling。其中**兩個我方可能沒有的設計**：(1) 黑白雙流＋因果交叉注意力（對手剛下的手作為 context，而非只平均對手特徵）；(2) ordinal logistic 頭（cumulative-link），與我方 expected-score 決策同屬 ordinal 處理，但在訓練端即內建相鄰級結構，和 e^-1 相鄰評分相容。其 ±1 準確率 70.7%（29 級）無法直接換算到我方 10 級。
- 弈客／野狐的商用「AI 評級」可推定是 KataGo 吻合度／損失統計的規則式映射（同 Kuboki 的 loss 特徵），無公開證據顯示優於學習式方法。
- 西洋棋的結論「時鐘貢獻 57 個 rating 點」對我方不適用（無時鐘），但其「高分段稀少→估值壓向中間」與 Neubauer「不給最高分」一致：我方對 6D 的召回值得單獨檢查。

### Gaps
- GoKibitz：未找到任何「從棋譜估棋力」的功能或文獻（本次搜尋無結果）。
- 野狐 AI 棋力評估：中文搜尋三次均無官方說明；弈客吻合度定義未公開。
- IPSJ 2016 與 FIT2018 全文未取得（503／405），無確切數字。
- KataRank 的訓練資料量、exact 準確率未揭露。

---

## Q7. 各方法「有而我方沒有」的做法、證據強度與可移植性（彙整）

### Takeaway
在「同一玩家 5–20 局、無時鐘／結果」的限制下，文獻給出的可移植項目依證據強度排序：(1) **KataGo scoreLead 損失統計**（Kuboki，玩家特定 n=5 +1.4 點）；(2) **label smoothing 0.1 ＋ 8 對稱增強 ＋ log-prob 跨局平均**（Rinrin-lab，本屆同資料，各 +0.05～0.07 級別，但其基線遠低於我方，增益會縮水）；(3) **Bradley–Terry／listwise 排序頭＋r∞ 隨機著手負類**（Chen，對比監督分類 49%→80%，但對比對象弱）；(4) **ordinal 頭／cumulative link**（KataRank，無消融）；(5) **跨局跨手 attention（ISAB）pooling、落子點 trunk 向量、黑白雙流因果交叉注意力**（Neubauer trunk vs head 有對照、其餘無消融）；(6) **測試集 rank-mean 重校準（transductive）**（Chen 的 candidate 集機制，可用 Public 無標籤題做 β/logit 均值平移）；(7) **樣式頻率特徵**（Moudřík，2015 年資料、尺度不同）。沒有任何文獻報告 pseudo-labelling／transductive 訓練或顯式 label-noise 處理；Chen 與 Neubauer 都以「允許 ±1 級」或「未來 rating 標籤」繞過標籤噪音。

### Cited Findings
- 玩家特定情境中 loss 特徵的增益（Kuboki Table 7：n=5 58.1 vs w/o Loss 56.7）— [arXiv HTML](https://arxiv.org/html/2505.00279v1)
- label smoothing／cosine 約 +0.07、log 平均 +0.05～0.06、對手視角 +0.015、expected-score 決策 −0.02（Rinrin-lab）— [README](https://raw.githubusercontent.com/Rinrin-lab/AICUP-2026-AlphaGo/rank-prediction-experiments/README.md)
- SE vs SLsum/SLvote：80%@15 局 vs 49%@100 局；r∞ 用於 OOD 著手；「data augmentation」存在（ICLR PDF 附錄 A）— [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/17a1a1439421f1837e10cd612bf92861-Paper-Conference.pdf)
- trunk vs head：Full −0.58／−0.55 vs Basic −0.61／−0.57（Neubauer Table 5.2）— [neubauer_24.pdf](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)
- KataRank Full（含 trunk）70.7% ±1 vs Lite 57.2% — [KataRank](https://github.com/ahillzhao-msn/KataRank)
- 樣式特徵單獨 RMSE 2.79 vs 全部 2.66（Moudřík Table I）— [arXiv 1512.08969](https://arxiv.org/abs/1512.08969)
- 布局 50 手對 kyu 鑑別力最低、官子 50 手對 9d 幾乎不掉（Chen 附錄 B.3–B.4）— ICLR PDF

### Inferences（對我方的具體建議，依預期 CP 值）
1. **加 KataGo 評估損失特徵**（每手 scoreLead／winrate 損失的 mean、median、std，前 100 手與全局）進 bag 特徵；我方若已有 trunk 抽取流程，邊際成本最低。證據：Kuboki 玩家特定 +1.4 點。
2. **訓練技巧 A/B**：label smoothing（0.05–0.1）、8 對稱 TTA（對 trunk／humanSL 抽取各做 8 次取平均 — 注意 humanSL 的 policy 本身不完全對稱不變，TTA 可減少此噪音）、跨局 log-prob 平均（我方 whole-question 模型若在 logit 層平均，等價）。證據：本屆同資料但低基線。
3. **ordinal／BT 輔助頭**：在 TCN/MLP 輸出旁加 (a) cumulative-link ordinal 頭，或 (b) Chen 式 listwise BT 頭（每 batch 各級抽 m 題平均 β 再逐級 softmax），推論時以 β 的 nearest-centroid 或與 CE 機率融合。證據：Chen（對比弱基線）、KataRank（無消融）— 屬中等證據，需自行 A/B。
4. **transductive 校準**：用 Public/Private 題的 logit 或 β 均值對訓練分佈的平移做校正（Chen 的 candidate 集思路），尤其針對 Rinrin-lab 與 Chen 都指出最弱的中段（1D–3D）。
5. **集合注意力 pooling**：以 ISAB（m≈32–64）取代或並聯 bag-level 平均，讓模型聚焦少數決定性著手；並考慮開／中／終分段 pooling（Chen 的階段分析支持階段差異）。
6. **讓子局處理**：Neubauer 直接排除；本屆 HD≥2 局缺子、HD=1 約 16%；至少確認對手視角特徵在 HD≥2 局是否被正確遮罩。
7. **低優先**：3×3／5×5 樣式頻率直方圖（Moudřík）作為棋風訊號。

### Gaps
- 沒有任何來源在「同一玩家、5–20 局、10 級」協定下報告可直接對照的分數；最接近者 Kuboki 玩家特定 15 局 63.2%（11 級 Fox）。
- 所有「增益」數字來自不同資料集與基線，無法線性外推到我方 0.726；每項皆需在我方 4-fold 上實測。
- 文獻中未見 pseudo-labelling、semi-supervised 或顯式 label-noise 建模用於圍棋棋力預測。
