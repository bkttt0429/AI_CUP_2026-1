# 非線性動力學／混沌／複雜度量測作為技能指標：文獻盤點與對圍棋逐手軌跡的可轉移性評估

研究日期：2026-11-04。範圍：permutation entropy、sample/approximate entropy、multiscale entropy、Lyapunov 代理量、RQA/cross-RQA、Hurst/DFA、Lempel–Ziv/壓縮性、棋形分形維度、criticality、entropy production、Kolmogorov 代理量。評估對象：每局約 100–150 手目標玩家著手、每位玩家 5–20 局、可得 28 個 rank-conditioned KataGo policy 的每手 log-prob 以及任意 KataGo 輸出（value、score lead、ownership、policy 分佈）。已知死亡結果：TSFEL 274 個線性時序特徵、lag-1/2 自相關、趨勢、勝率曲線形狀（波動度、drawdown、decisive time、blunder clustering）、response geometry；63 手感受野的 TCN 不優於逐手 MLP。

---

## 關鍵問題 1：這些量測在哪些領域被證明能分辨專家與生手？（運動控制、生理、眼動、音樂、棋類分析）

### Takeaway
運動控制／體育科學有大量「專家 vs 生手」的 entropy / DFA 證據，但方向不一致（有的專家 entropy 較高、有的較低），且 2024 年 scoping review 顯示 81% 的研究結論「專家變異度較低」而且**與所用指標無關**（線性 SD 與非線性 entropy 給同樣結論，即互相冗餘）。棋類方面：沒有任何已發表研究把 RQA/SampEn/PE 用在棋局評估曲線上；最接近的是西洋棋的 DFA（Ribeiro 2013；Schaigorodsky 2013）與「位置銳度 Δ」（Emergent complexity 2024；Barthelemy 2025），圍棋只有 Georgeot & Giraud 的複雜網路分析，且 1d 與 9d 的差異「relatively small」。

### Cited Findings

**運動控制／體育科學**
- 2024 年 scoping review（Scand J Med Sci Sports, DOI 10.1111/sms.14706）納入 59 篇研究，48 篇（約 81%）發現高技能運動員的動作變異度**較低**；此結果「independent of the criteria used to define expertise」、儀器與變異度指標（舉例指標含 standard deviation 與 approximate entropy）— [UQTR eprint 11611](https://depot-e.uqtr.ca/id/eprint/11611/)
- Stergiou 等人的 optimal movement variability 框架（J Neurol Phys Ther 2006, 30(3):120-129）：健康、成熟的技能對應「最適量」的變異度；低於最適 = 僵硬可預測，高於最適 = 噪音不穩定；失去複雜度與老化、疾病相關 — [Stergiou 等人（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC2652347)
- 2020 年飛鏢研究：在高度練習的運動員中，「moderate movement variability results in optimal throwing performance」（非單調、U 形關係） — [sportpedagogy.org.ua](https://sportpedagogy.org.ua/index.php/ppcs/article/view/1526)
- 攀岩（Cordier 等，Human Movement Science 1994，「Entropy as a global variable of the learning process」）：專家攀爬軌跡的 entropy 一般低於非專家 — [bohrium 條目](https://www.bohrium.com/paper-details/entropy-as-a-global-variable-of-the-learning-process/812451307502698497-7686)
- 桌球（J Motor Behavior 2025）：專家動作 entropy **高於**生手，解釋為「strategically sacrificing trajectory predictability to enhance haptic perception accuracy」 — [IIASA pure](https://pure.iiasa.ac.at/20858)
- 冬季兩項射擊姿勢控制（ISBS 2016）：8 位菁英 vs 15 位中等 vs 15 位生手；菁英在靜立時 body sway 較小且 SampEn **較高**，但在瞄準階段方向**反轉**；作者承認 entropy 解讀模糊（「vigilant system or noise」） — [ISBS 2016](https://ojs.ub.uni-konstanz.de/cpa/article/view/6845)
- 另一項冬季兩項研究（8 位國家級 vs 15 位三個月訓練 vs 15 位無經驗）：射擊與瞄準時菁英的 SampEn **顯著較低**（p<0.05），作者指出「difficult to clearly determine whether higher or lower entropy reflects postural control efficiency」 — [PMC9064080](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064080)
- 菁英步槍射手（J Motor Behavior 2020, DOI 10.1080/00222895.2020.1723478）：13 位射手 vs 11 位非射手，射手 SampEn 較低、entropic half-life 較長 — [NIH Brage](https://nih.brage.unit.no/nih-xmlui/handle/11250/2684010)
- 競走（Preatoni & Rodano, ISBS 2008）：用 SampEn 比較不同技能等級，結論「Sample entropy appeared a valid means for individual monitoring in sports」 — [ISBS 2008 PDF](https://ojs.ub.uni-konstanz.de/cpa/article/view/1860/1729)
- 新技能學習（Taylor 等, ISBS 2018）：用 SampEn 量關節變異度，所有受試者協調變異度在學習中顯著下降；較具適應力的組在關鍵點協調變異度較高 — [ISBS 2018](https://commons.nmu.edu/isbs/vol36/iss1/178)

**生理（心率複雜度）**
- 2026 CINC 摘要（12 人跑步機）：運動員在加速與穩態跑步時 DFA α1 較低（較具適應性），非運動員較高（較僵硬）；傳統 SDRR、RMSSD 「showed limited ability to distinguish training status under load」（樣本極小、初步） — [CINC 2026 #269](https://cinc.org/2026/Program/accepted/269.pdf)
- 16 位訓練良好的自行車手：運動負荷增加時 RR 間期整體變異度與複雜度皆下降；DFA α1 下降幅度與有氧能力呈反比 — [PMC6407005](https://pmc.ncbi.nlm.nih.gov/articles/PMC6407005)
- Multiscale entropy 教學（Costa 等）：單尺度 SampEn 下白噪音高於 1/f 噪音，但 1/f 在各尺度幾乎不變、白噪音單調下降、尺度 >4 後低於 1/f；「In contrast with the conclusions drawn from single-scale entropy-based analyses」，MSE 顯示 1/f 更複雜 — [PhysioNet MSE tutorial](https://archive.physionet.org/physiotools/mse/tutorial/node2.html)
- Time irreversibility／entropy production：生理訊號的時間不可逆性在年輕健康者最高、隨老化與病理下降（Costa 的 multiscale asymmetry、visibility-graph 方法）；**搜尋到的文獻無任何「專家 vs 生手」應用** — [Frontiers Netw Physiol 2022](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2022.877474/pdf)；[arXiv 1108.1691](https://arxiv.org/pdf/1108.1691)

**眼動／注視轉移 entropy**
- 教師課堂監控（Learning and Instruction, 2026）：專家注視轉移機率較高、entropy **較低**（較穩定）；生手掃視更具探索性 — [IPN Kiel](https://www.leibniz-ipn.de/en/research/publications/classroom-disruptions)
- 遙操作（6 位生手、4 位專家）：引入 gaze transition entropy 與 stationary gaze entropy，專家「more focused, goal-oriented」、生手「more erratic scanning」 — [Alparslan Univ. repository](https://akademikarsiv.alparslan.edu.tr/items/1bff51f4-9967-4f4e-ab6c-67c25994b3ee/full)
- 相反結果（Frontiers in Education 2020，教師）：專家 scanpath **更複雜**、更頻繁重訪所有學生、轉移機率近乎均等 — [Frontiers 2020](https://www.frontiersin.org/articles/10.3389/feduc.2020.612175/pdf)
- 西洋棋眼動：Sheridan & Reingold 2014 發現專家與生手都較多注視相關區域，但專家更早鎖定；**未找到任何用 entropy 量西洋棋專家眼動的研究** — [Frontiers Psychol 2014](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.00941)

**隨機序列生成（「隨機性即技能」的遊戲）**
- Delogu 等 2020（Morra 手指遊戲）：9 位錦標賽專家 vs 9 位學生，序列只取前 57 個數字，用 5 個隨機性指標（Redundancy、FPR、NSQ、Coupon、Repetition gap）。比賽中專家 vs 非專家：Redundancy 3.31 (1.74) vs 15.53 (4.43)、NSQ 15.82 (5.19) vs 37.90 (6.53)、Coupon 13.14 vs 35.01，貝氏 evidence ratio >1000、分佈重疊 0.08–0.23；但在標準 RNGT 作業中兩組幾乎無差異（ER=0.02）；結論：隨機生成能力可程序化、不需注意力控制 — [PMC7541701](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541701)
- Wong, Merholz & Maoz 2021（Sci Rep, DOI 10.1038/s41598-021-99967-6）：以 normalized Lempel–Ziv complexity（除以 1000 條等長偽隨機序列的平均 LZC，長度 192）量人類在競爭環境的隨機性；本身未分技能，引述 Delogu 指出「expert-level players … have been shown to be more random than novice players during the game」 — [PMC8526708](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526708)

**電競／即時戰略**
- StarCraft 2（AIIDE 2013）：1,297 局 replay，用 SMO 分類器預測聯賽等級，加權準確率 47.3%（基線 25.5%）；約 2.5 分鐘即可估計 — [AIIDE 2013](https://ojs.aaai.org/index.php/AIIDE/article/view/12682)
- StarCraft 2 motor chunking（Topics in Cognitive Science 2017）：3,317 位玩家、七個技能層級、996,163 個動作序列；chunk 起始動作的延遲較長；其他預測僅弱支持。**未找到以 action-sequence entropy 區分技能的研究** — [Zendy / TopiCS](https://zendy.io/title/10.1111/tops.12254)

**棋類分析（西洋棋）**
- Ribeiro 等 2013（PLoS ONE 8:e54165）：73,444 局高階棋譜，用 Crafty（Elo 2950）逐手評估白方優勢 A(m)。優勢的擴散：開局前 7–10 手幾乎不擴散，和棋 10<m<100 超擴散 α=1.49±0.01，勝負局 10≲m≲40 超擴散 α>2；增量 ΔA 的 DFA Hurst 指數集中於 h≈0.35（SD≈0.1，反持續），洗牌後 ≈0.54；分佈非高斯且 150 年來形狀穩定；**未對個別玩家 Elo 作檢定**；超擴散被「surmise」為玩家 fitness 差異造成，toy model A(m+½)=A(m)+Φ_i+η 給 α=2 — [ar5iv 1212.2787](https://ar5iv.arxiv.org/html/1212.2787)
- Schaigorodsky, Perotti & Billoni 2013（Physica A）：ChessDB 約 1.4×10⁶ 局**跨局按時間排序**，把開局相似度／流行度編成整數序列，R/S 與 DFA 給 H≈0.75（PAR）、0.67（SAR），洗牌 ≈0.5；依 Elo 分三段（<2200、2200–2399、≥2400）：短時間尺度 H≈0.5 無差，中尺度低 Elo 相關較強，長尺度反轉為高 Elo 最強；解讀為「high level players vary the lines of play more often at short time scales」。**注意：這是跨局（population-level）序列，不是單局內的時間序列** — [ar5iv 1307.0729](https://ar5iv.labs.arxiv.org/html/1307.0729)
- 「Emergent Complexity in the Decision-Making Process of Chess Players」（arXiv 2406.15463, 2024）：以 Stockfish 16 深度 20 計算 Δ=|E1−E2|（最佳與次佳著手評估差，即「決定性／銳度」）。FIDE 3000 局、Lichess 高 Elo（>2500）1000 局、低 Elo（<1500）3000 局、隨機合法著手 1000 局：**Δ 的平均與標準差隨等級上升而下降**，隨機對局的平均 Δ 最高可達 FIDE 的 4 倍；度分佈的變異數隨等級下降；節點流行度分佈「do not appear to depend on the competitive level」 — [arXiv 2406.15463v2](https://arxiv.org/html/2406.15463v2)
- Barthelemy 2025「Chess variation entropy」：對 principal variation 的每一半著以 p₁=1/(1+e^(−Δ/Δ₀)) 定義找到最佳著的機率、S=−log₂∏p₁；「beginners consistently experiencing the highest entropy and experts the lowest」（beginners 6<S<7，experts 約 4 以下）。**但 beginners/intermediates/experts 是用 Δ₀=100/50/10 centipawn 的假設性偵測門檻套在同一批 100 局 2023 世界快棋賽棋譜上，並非真實玩家分組**；不報告錯誤率 — [arXiv 2505.03251](https://arxiv.org/html/2505.03251)
- Islas-García & del Castillo-Mussot 2024（IJMPC 35(4)）：每種棋子佔據密度的 entropy 模式「independent of player skill level」（Elo）— 一個負結果 — [IDEAS/RePEc](https://ideas.repec.org/a/wsi/ijmpcx/v35y2024i04ns0129183124500438.html)
- Guid & Bratko「Computer Analysis of World Chess Champions」：以引擎最佳著與實際著手評估差作錯誤量，定義位置複雜度；「players made little errors in simple positions, and the error rate increased with increasing complexity」；Capablanca 複雜度低、Tal/Steinitz 高 — [ChessBase](https://en.chessbase.com/post/newsdetail.asp?newsid=3455)；Biswas & Regan 2015 延伸為 depth／complexity 度量 — [SciTePress 2015](https://www.scitepress.org/papers/2015/52883/52883.pdf)
- Regan & Haworth 2011「Intrinsic Chess Ratings」（AAAI）：以兩個技能參數 s（sensitivity）、c（consistency）描述著手品質分佈，建立與 Elo 的對應；c 是分佈形狀參數而非平均 — [AAAI 2011](https://ojs.aaai.org/index.php/AAAI/article/view/7951)
- ChessEngineLab「sharpness vs accuracy」：385 局 GM 棋局（約 25,000 位置）以 Lc0 WDL 定義 sharpness，GM 的 Lichess accuracy 隨 sharpness 上升而明顯下降；作者自己 42 局的 accuracy 在最低 sharpness 最高、約 0.8 後持平；作者推測「lower-level games are more random, so mistakes occur in any kind of position」；未按等級分層 — [ChessEngineLab](https://chessenginelab.substack.com/p/looking-at-sharpness-and-accuracy)
- Lichess 社群「volatility of chess games」討論：作者自承測試顯示該量「depended a lot on the opponents of the players」；評論者指出引擎量測多為「errors upon errors」 — [Lichess discuss](https://lichess.org/ublog/H6MWvX98/discuss)
- Lichess accuracy 本身即以評估波動加權：把棋局切成約 10 ply 的區段，波動大的區段著手權重較高 — [ChessEngineLab: different accuracies](https://chessenginelab.substack.com/p/different-accuracies)
- Maia（KDD 2020）：Maia-1900 預測 1100 級著手準確率 46%，最佳 >52%，各模型在其訓練等級達峰 — [maiachess.com](https://maiachess.com/blog/introducing_maia)；Otter（2026）：top-1 準確率由 <1100 的 49.48% 升至 1900–1999 的 57.38% — [arXiv 2608.05206](https://arxiv.org/pdf/2608.05206)
- Erdős Institute「Guess the Elo」：約百萬局 TWIC 棋譜 + Stockfish centipawn loss，**單局**指標與 Elo 只有弱相關，跨多局聚合後才有希望 — [Erdős Institute](https://www.erdosinstitute.org/project-database/fall-2024/data-science-boot-camp/guess-the-elo-(chess))

**圍棋（統計物理／複雜網路）**
- Georgeot & Giraud 2012（EPL 97, 68002）：日本職業賽約 865 局 + u-go.net 4000 局業餘棋譜；3×3 局部棋形（1,107 種）頻率服從 Zipf 律，指數 ≈1.06（Honinbo −1.058、Meijin −1.056、Judan −1.065、Kisei −1.067、amateur −1.081）；度分佈冪律 γ=1.0、入出對稱；clustering >0.7；譜有巨大 gap。職業 vs 業餘：「the amateur database departs significantly from all the professional ones」（著手距離分佈）、特徵向量「different for different tournaments and from professional to amateur」、Zipf 斜率業餘略陡 — [ar5iv 1105.2470](https://ar5iv.arxiv.org/html/1105.2470)
- Kandiah, Georgeot & Giraud 2014（EPJ B 87, 246）：135,663 局線上對局（1d–9d）；三種網路（3×3：1,107 節點；3×3+atari：2,051；菱形 plaquette：193,995）。1d/1d vs 9d/9d（各 2,731 局）菱形網路有差異（mean r_j≈1.33），但 6d/6d 同級樣本間也有差異，1d–9d 差「has some statistical significance」且「the difference is relatively small」；開局／中盤／終盤（各 50 手）網路譜差異明顯（終盤特徵值雲較大、開局 gap 較小） — [ar5iv 1405.6077](https://ar5iv.arxiv.org/html/1405.6077)
- 圍棋棋子分佈的 multifractal 分析（APS March Meeting 2010 海報）：估計 generalized dimensions，分形維度隨格子尺寸增加而增大；**無技能關聯** — [APS 2010](https://archive.aps.org/mar/2010/c1/147)；GoBase 稱棋譜統計分析顯示棋形「of a fractal nature」但無數值 — [GoBase](https://gobase.org/go-7.html)
- 圍棋提子「雪崩」與 SOC：僅有 2003 年新聞群組的非正式模擬（叢集尺寸在 19×19 飽和於 L≈20） — [sci.physics.research](https://groups.google.com/g/sci.physics.research/c/Rri1idGmwyU/m/2BJA3Wwil6gJ)
- 「Intelligence at the Edge of Chaos」（arXiv 2410.02536）：訓練資料的 Lempel–Ziv 複雜度與下游西洋棋著手預測準確率 Pearson r=0.4371（p=9.73×10⁻³）；這是關於**模型訓練資料**的複雜度，不是玩家技能 — [ar5iv 2410.02536](https://ar5iv.labs.arxiv.org/html/2410.02536)

**圍棋等級估計的現行最佳做法（對照）**
- 「Policies of Multiple Skill Levels for Better Strength Estimation in Games」（arXiv 2505.00279, 2025）：Fox 伺服器 11 個等級組各 5,000 局；特徵 = 各技能等級 KataGo HumanSL policy（10k…9d 共 10 個）對實際著手機率的**每局幾何平均**（LL geometric means）、Chen 等 strength score 的算術平均、以及 KataGo scoreLead 損失的 Mean/Med/Std（前 N_cut 手）；LightGBM 回歸；**完全不建模著手時間順序**。隨機抽樣 n=10 局 79.6%（±1 級 99.9%）、n=20 局 91.7%；player-specific 抽樣 n=10 61.0%、n=15 63.2% — [arXiv 2505.00279](https://arxiv.org/html/2505.00279)
- 「Derived metrics for the game of Go」（arXiv 2009.01606）：以 score estimate 差定義著手效果、以搜尋相對原始 policy 的資訊增量定義網路內在強度，用於作弊偵測 — [arXiv 2009.01606](https://arxiv.org/abs/2009.01606)

### Inferences
- 運動控制文獻的「專家 entropy 較低／較高」方向取決於任務（姿勢控制 vs 操作任務）與訊號（COP vs 軌跡），不存在可直接搬到圍棋的通則；且 scoping review 顯示 entropy 類指標與 SD 類指標結論相同，暗示在這些資料中非線性量主要重述了變異量。
- 西洋棋唯一帶 Elo 分層的「非線性」正結果（Schaigorodsky 2013）是跨局、跨年的開局流行度序列（長度 10⁶），描述的是棋手群體的開局記憶，與單局內 120 點的著手品質序列完全不同層次，不可外推。
- 棋類中真正隨等級單調變化且有獨立證據的「entropy 類」量是**位置層級**的銳度（Δ 的平均與 SD 隨等級下降；GM 準確率隨 sharpness 下降），而非著手品質序列的時間結構。它量的是「玩家把局面導向多銳」，是玩家可控的行為量，不是混沌量。
- Kandiah 2014 顯示圍棋棋形網路的**對局階段效應大於等級效應**（開局／中盤／終盤譜差異明顯，1d vs 9d 卻只「relatively small」），任何整局計算的 entropy/RQA 量都會被階段混合主導，需先分階段。

### Gaps
- 未找到任何把 RQA、SampEn、PE、MSE 或 Lyapunov 用在西洋棋或圍棋**單局評估曲線／著手品質序列**上的已發表研究。
- 未找到西洋棋或圍棋專家眼動 entropy 的研究。
- 未找到音樂表演以 entropy/DFA 區分專業程度的直接證據（本次搜尋未涵蓋到）。
- EEG multiscale entropy 區分踢球專家與生手的研究在搜尋摘要中出現但未取得可引用的 URL，故不列入。
- Ribeiro 2013 的 Hurst 指數是否隨個別玩家 Elo 變化：原文未檢定，無資料。

---

## 關鍵問題 2：每個量測各自捕捉了線性／頻譜統計與 TCN 捕捉不到的什麼？需要哪種輸入（評估序列／著手類別／棋盤位置）？

### Takeaway
理論上這些量各有獨特對象（ordinal 結構、模板遞迴、尺度律、字串重複、幾何維度），但要「存活」的前提是序列內存在**非線性時間依賴**。專案的 TCN 結果（63 手非線性感受野無增益）是對「任何 63 手內可局部計算的非線性時間統計」的強負證據；只有超出該窗口的整局全域量（DFA、整局 RQA、整局 LZ）與**跨玩家耦合量**（cross-RQA）在原理上未被排除，而這些正是在 N≈120 最不可靠的量。另有一類「entropy」（位置 policy entropy／top-2 gap）其實是**位置屬性**而非時序量，有獨立文獻支持，應與混沌量分開看待。

### Cited Findings

**定義與可計算性（供後續表格）**
- Permutation entropy（Bandt & Pompe 2002）：以 m 個連續值的排序模式（ordinal pattern）之 Shannon entropy 定義；對非線性單調變換不變、對觀測噪音穩健、計算極快；需弱平穩假設；混沌系統有「forbidden patterns」 — [Rosso 等 arXiv 1105.4550](https://arxiv.org/pdf/1105.4550)；Zunino & Kulp 2017 主張 PE 比範圍型符號化更適合有非平穩、非線性與噪音的實際序列 — [Lumped PE 摘要（POSTECH）](https://remotecenter.postech.ac.kr/handle/2014.oak/105703)
- Sample entropy（Richman & Moorman 2000, DOI 10.1152/ajpheart.2000.278.6.H2039）：不計自我匹配以消除 ApEn 對規律性的偏差，為短而雜訊多的生物序列設計；相對一致性（relative consistency）被後續研究質疑，在 N=50 的正弦訊號不成立 — [Entropy 2020, 22(6):694](https://www.mdpi.com/1099-4300/22/6/694)
- Multiscale entropy（Costa 等）：對原序列做非重疊窗平均的粗粒化（長度 N/τ），每個尺度算 SampEn；r 通常取 SD 的 10–20% — [PhysioNet MSE background](https://archive.physionet.org/physiotools/mse/tutorial/node1.html)
- RQA（Marwan 等 2007, Physics Reports 438:237）：RR = 遞迴點密度；DET = 構成長度 ≥ l_min 對角線的遞迴點比例（可預測性）；LAM = 垂直線比例（層流／間歇）；ENTR = 對角線長度分佈的 Shannon entropy（**對 bin 數敏感**）；TT = 平均垂直線長；DIV = 1/最長對角線（曾被宣稱估計最大 Lyapunov 指數，但關係「more complex」，白噪音亦偶有長對角線）；「can provide useful information even for short and non-stationary data」 — [Wikipedia RQA](https://en.wikipedia.org/wiki/Recurrence_quantification_analysis)；[Marwan 2007 PDF](https://publications.pik-potsdam.de/pubman/item/item_20541_1/component/file_20542/6996oa.pdf)
- 類別序列的 RQA／CRQA（crqa R 套件）：名目資料的遞迴以「相同即遞迴」定義，設 delay=1、embed=1、radius<1 即可，不需相空間重建；用於對話、注視等行為編碼 — [R Journal 2021](https://journal.r-project.org/articles/RJ-2021-062)
- Hurst／DFA：Ribeiro 2013 對西洋棋評估增量做線性去趨勢 DFA，得 h≈0.35（反持續） — [ar5iv 1212.2787](https://ar5iv.arxiv.org/html/1212.2787)
- Lempel–Ziv：Wong 等 2021 以等長偽隨機序列平均正規化 LZC，長度約 192 — [PMC8526708](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526708)
- 位置銳度 Δ=|E1−E2|：最佳與次佳著評估差；mean/SD 隨等級下降 — [arXiv 2406.15463v2](https://arxiv.org/html/2406.15463v2)；KataGo 原始 policy 分佈即可給出等價量（例如某局面單一著手 policy 質量 43%、次佳 16% 的實例） — [L19 論壇](https://www.lifein19x19.com/viewtopic.php?p=258885)

**對照：現行最佳圍棋等級估計只用聚合量**
- 2505.00279 的特徵全部是每局的平均／幾何平均／中位數／標準差，無時序建模，仍達 10 局 79.6%、20 局 91.7% — [arXiv 2505.00279](https://arxiv.org/html/2505.00279)

### Inferences（逐量測評估表）

下表為本研究者綜合文獻與專案負結果的推論；「文獻證據」欄只列真有出處的項目。

| 量測 | 需要的輸入 | 理論上捕捉什麼（線性/TCN 捕不到的） | 技能分辨的已發表證據 | N≈120 可靠性（見 KQ3） | 相對於專案死亡結果的預期增量 | 成本 | 風險 |
|---|---|---|---|---|---|---|---|
| Permutation entropy（m=3–4） | 每手品質序列（log-prob 或 Δscore） | ordinal 模式分佈的非均勻性＝非線性序列相依；若各手品質近似條件獨立，PE≈log(m!) 對所有人相同 | 無棋類證據；生理/EEG 分類可行 | m=3 可穩定（~50–100 點）；m=4 邊緣 | **極低**：TCN 已涵蓋任何 63 手內 ordinal 統計；專案 lag-1/2 自相關已死 | 極低 | 幾乎確定冗餘 |
| Sample/Approximate entropy | 同上（連續值） | 模板遞迴率的條件機率，對振幅尺度敏感（r∝SD，而 SD 已死） | 運動控制大量、方向不一致 | 建議 N>200；N≤200「極度參數敏感」 | 極低 | 低 | 參數敏感、與 SD/錯誤率冗餘 |
| Multiscale entropy | 同上 | 跨尺度的複雜度輪廓 | 生理（1/f vs 白噪音） | 不可行：尺度 τ 使 N→120/τ | 無 | 低 | 不可計算 |
| DFA／Hurst | 同上或勝率曲線 | 長程相關／尺度律（超出 63 手窗口的唯一一類） | 西洋棋跨局序列有 Elo 差；單局增量為反持續 h≈0.35 且未分 Elo | N<~500 不可靠；反持續序列 DFA 與 HK 皆失效 | 低；且評估增量的反持續性主要來自引擎評估的均值回歸，不是技能 | 低 | 在最需要的區域（反持續）方法失效 |
| Lyapunov 代理量 | 同上 | 軌跡發散率 | 無 | Eckmann–Ruelle：D≤2log₁₀N≈4.2；需數百點 | 無 | 低 | 無意義（非自治系統、噪音膨脹） |
| RQA（RR/DET/LAM）—連續 | 同上（嵌入） | 相空間中的遞迴結構、層流段（例如「連續平庸手」段落） | 無棋類證據 | DET 最穩健、ENTR/Lmax 不穩 | 低：LAM/TT 與 blunder clustering（已死）高度相關 | 低 | 與既有死亡特徵冗餘 |
| RQA—類別（著手類型） | 著手類別序列（例：policy 排名桶、區域、手段類型） | 類別序列的重複結構 | 行為科學（對話／注視）為例示，無效果量 | embed=1 無嵌入損失；120 點 RR/DET 可算 | 低–中：與 n-gram 頻率高度重疊 | 低 | 類別定義任意 |
| Cross-RQA（兩位玩家） | 兩方品質序列或類別序列 | 雙方品質的**耦合**（領先/跟隨、同步犯錯）—本專案「對手影響」主題 | 舞者、足球 dyads（小樣本） | 單局 60×60 點噪音大；可按玩家跨局池化但對手每局不同 | 中（唯一未被 TCN 排除的新資訊類型，但 TCN 若輸入雙方序列亦應能學到局部耦合） | 低 | 對手混淆、功效低 |
| LZ / gzip 壓縮性（著手序列） | 著手類別序列或 SGF | 字串重複度＝可預測性 | Morra/RSG：專家更隨機（效果巨大，但那是「隨機性即技能」） | 需等長代理序列正規化；120 點偏差大 | **幾乎零**：rank-conditioned policy 的 cross-entropy 是遠優於 LZ 的「壓縮器」，Maia/Otter 準確率隨等級單調上升證明可預測性已被 log-prob 捕捉 | 極低 | 冗餘 |
| 棋形分形維度／multifractal | 棋盤位置（重播） | 棋子空間分佈的幾何 | 僅海報，無技能關聯 | 19×19=361 格只有 2–3 個 box 尺度 | 未知、可能很低 | 中 | 無先驗、尺度太少 |
| Criticality／edge-of-chaos 指標 | 位置或評估序列 | 「臨界」的多尺度訊號 | 無（圍棋 SOC 僅非正式討論） | 不適用 | 無 | 高 | 概念不成立 |
| Entropy production／時間不可逆性 | 評估序列 | 增量分佈的時間不對稱 | 生理（年齡/病理），無技能 | 多尺度不對稱需長序列 | 低；單一尺度的不對稱（例如「大跌後小回」）已被 drawdown/recovery 特徵涵蓋 | 低 | 冗餘 |
| Kolmogorov 代理（整局棋譜） | SGF | 同 LZ | 無 | 同 LZ | 零 | 極低 | 冗餘 |
| **位置 policy entropy／top-2 gap（非混沌量）** | KataGo policy（評分時） | 玩家把局面導向的「銳度」分佈；玩家可控、與對手共同決定 | 西洋棋 Δ 的 mean/SD 隨等級下降（獨立資料集）；GM 準確率隨銳度下降 | 每手一個值，120 點即可做分佈統計 | **中**：若專案尚未把「玩家著手後局面的 policy entropy／top-2 gap 分佈」當特徵（response geometry 可能部分涵蓋），值得低成本測試 | 低（已有輸出） | 與對手強度混淆；可能已被 rank-policy log-prob 隱含 |

補充推論：
- TCN 的負結果比「線性特徵死亡」更強：一個帶非線性激活的 63 手卷積網路原則上能逼近任何 63 手窗口內的局部統計（含 ordinal 模式計數、模板匹配等 PE/SampEn 的構成成分），因此「非線性 vs 線性」不是瓶頸，瓶頸是序列內**缺乏可利用的時間相依**；在此條件下 PE、SampEn、局部 RQA 的期望值對所有等級趨於同一常數，只剩有限樣本噪音。
- 唯一在原理上超出 TCN 感受野的是整局尺度律（DFA）與整局遞迴（Lmax、ENTR），而這兩者在 N≈120 恰好最不可靠（KQ3）。
- 著手類別的 LZ/Kolmogorov 代理量本質是「無模型的可預測性」，而專案已持有 28 個 rank-conditioned 「有模型的可預測性」（log-prob），後者在資訊論上是嚴格更強的壓縮器；Maia/Otter 的「預測準確率隨等級單調上升」就是這個量在西洋棋的證據。
- 位置銳度／policy entropy 是文獻中唯一有獨立等級分層證據的「entropy」，但它是**位置屬性**：它反映玩家選擇進入哪類局面，與「混沌／複雜度」無關，應歸入「局面選擇風格」特徵家族測試，而非非線性動力學家族。

### Gaps
- 無任何研究比較「PE/SampEn/RQA 於棋局評估曲線」與「平均品質」的增量預測力；上表增量欄為推論。
- 本專案「response geometry」是否已包含位置 policy entropy／top-2 gap 的分佈統計，無從查證；若已含，則該列的「中」應降為「低」。
- 2505.00279 的 LL geometric mean 是否已隱含位置銳度（銳度高的位置所有 rank-policy 的 log-prob 都較低），未見分析。

---

## 關鍵問題 3：樣本量需求—這些量在 N≈120 的可靠性如何？逐局不可用是否可改為每位玩家跨局池化？

### Takeaway
文獻給出的穩定門檻：SampEn 建議 N>200（N≤200 極度參數敏感）；PE m=3 約 50–100 點可穩、m=4 約 150–450、m=5 約 250–950，但「分類可在穩定前成立」；DFA 一般需 ≥500 點，且對反持續序列失效；RQA 中 DET 最耐短序列、ENTR 中等、Lmax 需近全長；Lyapunov／維度受 Eckmann–Ruelle 上界 2log₁₀N≈4.2 限制。逐局 N≈120 僅 PE(m=3)、RR/DET、正規化 LZ 勉強可算；跨局池化（600–2,400 點）可解決長度問題，但會把不同對手、不同階段混在同一序列，破壞平穩性假設，且多尺度/粗粒化跨越局界無意義。

### Cited Findings
- Yentes 等 2013（Ann Biomed Eng 41(2):349-365, DOI 10.1007/s10439-012-0668-3）：ApEn 與 SampEn「extremely sensitive to parameter choices」，尤其 N≤200；SampEn 對資料長度較不敏感、相對一致性問題較少；建議「N larger than 200, an m of 2」並先測多個 r — [Europe PMC PMC6549512](https://europepmc.org/article/PMC/PMC6549512)
- Pincus & Goldberger 1994 的 ApEn/SampEn 長度準則：N（或粗粒化後 N/τ）應在 10^m 到 20^m 之間（m=2 → 100–400） — 轉引自 [arXiv 2206.13529](https://arxiv.org/pdf/2206.13529)
- SampEn 相對一致性在 N=50 的正弦訊號不成立 — [Entropy 2020, 22(6):694](https://www.mdpi.com/1099-4300/22/6/694)
- Cuesta-Frau 等 2019（Entropy 21(4):385）：PE 的經典建議 N≫m!（Bandt & Pompe 以降）及 5m!≤N；實測隨機序列 m=3…7 在約 50、200、500、20000、55000 點穩定（經典估計 60、240、1200、7200、50400）；真實資料 m=3 約 50–100、m=4 約 150–450、m=5 約 250–950 點；分類方面 EEG（Bonn）「as few as 100 samples」即顯著可分（N=3000 與 N=100 的 AUC 差約 0.1）、PAF 45–50 點可分；結論「shorter lengths than those suggested by N>>m! are sufficient」且「even very short time series can be robustly classified … before the stability point is reached」，理由是各類別的 PE–N 曲線近乎平行 — [Europe PMC PMC7514869](https://europepmc.org/article/PMC/PMC7514869)
- Riedl, Müller & Wessel 2013「Practical considerations of permutation entropy」（EPJ ST 222:249-262）為 PE 參數選擇的教學回顧 — [SFI 鏡像 PDF](https://wiki.santafe.edu/images/5/52/2013_RiedlM_EPJST.pdf)
- Multiscale entropy：粗粒化後長度 N/τ，大尺度時「dispersion … progressively increases due to the shortening」 — [PhysioNet MSE](https://archive.physionet.org/physiotools/mse/tutorial/node1.html)；Faes 等 2017 指 MSE/RMSE「cannot be reliably computed over short time series」並提出線性狀態空間替代 — [Faes 2017 PDF](https://air.unimi.it/retrieve/dfa8b99a-8e08-748b-e053-3a05fe0a3a96/Faes_C_2017.pdf)；Refined composite multiscale PE 被提出以減少對長度的依賴 — [NTNU scholar](https://scholar.lib.ntnu.edu.tw/en/publications/refined-composite-multiscale-permutation-entropy-to-overcome-mult-2/)
- DFA 短序列：Delignières 等 2006（J Math Psychol 50:525-544, DOI 10.1016/j.jmp.2006.07.004）以精確分形序列評估各法在短序列的偏差與變異，提出要分辨平均指數差約 0.08 所需的 SD — [bohrium 條目](https://www.bohrium.com/paper-details/fractal-analyses-for-short-time-series-a-re-assessment-of-classical-methods/812058172305965057-8711)；Yuan 等 2018（Physica A 505:179-189）提出偏差校正，fBm 長度 500 時偏差可忽略 — [IDEAS](https://ideas.repec.org/a/eee/phsmap/v505y2018icp179-189.html)；Carpena 等 2021：α1 在短尺度「systematically fails to detect the correct scaling」 — [PMC8775092](https://pmc.ncbi.nlm.nih.gov/articles/PMC8775092)；Mangalam & Likens 2025（Entropy 27(5):500）：Bayesian HK 法對**持續**序列短至 N=64 仍合理，但「both processes break down for anti-persistent time series」（DFA 與 HK 皆然）— [MDPI Entropy 27(5):500](https://www.mdpi.com/1099-4300/27/5/500)
- Eckmann & Ruelle 1992（Physica D 56:185-187）：由 N 點資料以 Grassberger–Procaccia 法估出的關聯維度上界為 2log₁₀N，達到上界時不能推論低維動力學 — [UNIGE archive](https://archive-ouverte.unige.ch/unige:92420)
- Rosenstein 等 1993：為「small data sets」設計的最大 Lyapunov 指數法，宣稱對嵌入維度、資料量、延遲、噪音穩健（作者自測） — [PhysioNet](https://www.physionet.org/content/lyapunov/1.0.0/RosensteinM93.html)；實務指南稱「a few hundred points」為實際下限且噪音會製造假正指數（非同儕審查） — [MetricGate](https://metricgate.com/docs/lyapunov-exponent-estimation/)
- RQA 短序列（Sviridova 等 2022, Sensors 22(14):5154）：PPG（409.6 Hz，嵌入 m=4、ε=10% attractor）達 <1% 誤差所需長度：DET 1,000 點（2.44 s≈3 個週期）、ENTR 7,000 點（≈21 週期）、L 43,000 點、Lmax 49,120 點（幾乎全長）；Rössler 模擬需 >25,000 點達 1%、10,000 點達 5%；DET「remains the feature requiring the shortest time series length」，Lmax「requires long time series」 — [Europe PMC PMC9324273](https://europepmc.org/article/PMC/PMC9324273)
- 類別 RQA 不需嵌入（delay=1、embed=1），故 120 點全部可用；但 R Journal 文章未給最小長度建議 — [R Journal 2021](https://journal.r-project.org/articles/RJ-2021-062)
- LZ 正規化：Wong 等 2021 以 1000 條等長偽隨機序列的平均 LZC 正規化（長度 192） — [PMC8526708](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526708)；Delogu 等 2020 僅用前 57 個數字即得到巨大專家效果 — [PMC7541701](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541701)
- Surrogate 檢定：以 100 條 surrogate 的 2.5–97.5 百分位判定非線性是常見做法；短而雜訊的實驗序列會讓簡單自相關誤導檢定（Theiler）；AAFT 在短序列有缺陷（Kugiumtzis） — [Schreiber & Schmitz arXiv chao-dyn/9909041](https://arxiv.org/pdf/chao-dyn/9909041)；[Kugiumtzis arXiv physics/9905021](https://arxiv.org/pdf/physics/9905021)

### Inferences
- 逐局（N≈120、目標玩家約 60–75 手；若雙方合併約 120–150 手）：
  - PE m=3（6 種模式）可穩定；m=4（24 種）需 ≥150 點，邊緣；m≥5 不可行。
  - SampEn m=2 低於 Yentes 建議門檻 200，估計值將被 r 的選擇主導。
  - MSE 尺度 2 即剩 60 點、尺度 3 剩 40 點，均低於 10^m=100。不可行。
  - DFA 在 120 點的 α 估計 SD 遠大於 0.08 的區辨需求；且若評估增量如西洋棋般反持續（h≈0.35），DFA/HK 均失效。
  - RQA：RR、DET 可算但噪音大；ENTR（依 bin 數）與 Lmax 不可用。
  - Lyapunov：D≤4.2 的上界加上非自治、噪音大的序列，數值無動力學意義。
- 跨局池化（5–20 局 → 600–2,400 點）在長度上足夠 SampEn、PE m=4–5、DFA，但：(a) 每局對手不同，池化序列的「對手」成分不平穩；(b) 開局／中盤／終盤的階段差異大於等級差異（Kandiah 2014），拼接會產生週期性的階段混合；(c) 粗粒化或嵌入窗跨越局界無意義，必須在局界截斷，等於回到逐局計算再平均。因此池化只能用在「每局算一個噪音大的統計量再跨局平均」的模式，這與 TSFEL 已做的方式相同。
- Cuesta-Frau 的「分類可在穩定前成立」要求各類別的 PE–N 曲線平行，即短序列偏差對所有等級相同；在本專案中每局手數與等級可能相關（高段局較長或較短），偏差將與等級混淆，應以固定長度截斷（例如前 100 手）控制。

### Gaps
- 沒有文獻給出 N=120 時 SampEn、DFA 或 RQA-ENTR 的具體偏差／方差數值（Delignières 2006 的分長度表格無法取得全文）。
- 短序列（~100 點）surrogate 非線性檢定的統計功效數字未找到。
- 沒有圍棋評估增量的 Hurst 指數報告，無法確認是否如西洋棋般反持續。

---

## 關鍵問題 4：對「混沌」的誠實評估—強評估器下的評估軌跡由雙方錯誤驅動（非平穩、非自治），動力系統解讀是否有意義？或只是錯誤率統計的非線性重述？

### Takeaway
現有最接近的實證（Ribeiro 2013）把西洋棋評估軌跡建模為「漂移（技能差）＋噪音（錯誤）」的隨機過程，超擴散歸因於玩家 fitness 差距，反持續性來自增量結構；作者明言「the diffusive scenario is not determined purely by the fitness of players」，且從未宣稱確定性混沌。位置銳度隨等級變化（Emergent complexity 2024）進一步說明軌跡統計是「誰在下」的函數，即系統是非自治的。因此對圍棋評估曲線作 Lyapunov／吸引子解讀沒有理論基礎；entropy/RQA 類量在此情境主要是錯誤率、錯誤幅度與位置銳度的非線性再描述。

### Cited Findings
- Ribeiro 等 2013：優勢增量 DFA h≈0.35（反持續），洗牌 ≈0.54；超擴散 α=1.49（和棋）、α>2（勝負局早期）；toy model A(m+½)=A(m)+Φ_i+η 以玩家 fitness 漂移 Φ_i 解釋 α=2；「We surmise」擴散指數上升源於 fitness 差距擴大；「In the case of chess, the diffusive scenario is not determined purely by the fitness of players」 — [ar5iv 1212.2787](https://ar5iv.arxiv.org/html/1212.2787)
- Emergent complexity 2024：Δ 的平均與 SD 隨等級上升下降；隨機合法著手的 Δ 平均達 FIDE 的 4 倍；勝方準確率在各深度均高於敗方 — [arXiv 2406.15463v2](https://arxiv.org/html/2406.15463v2)
- Barthelemy 2025 的 variation entropy 中「技能」是 Δ₀ 偵測門檻，等級分離是模型假設的直接結果 — [arXiv 2505.03251](https://arxiv.org/html/2505.03251)
- Lichess volatility 作者：該量「depended a lot on the opponents of the players」；評論者：引擎量測主要捕捉「errors upon errors」 — [Lichess discuss](https://lichess.org/ublog/H6MWvX98/discuss)
- Guid & Bratko：錯誤率隨位置複雜度上升；positional players「have less chance to commit gross tactical errors than aggressive tactical players」 — [ChessBase](https://en.chessbase.com/post/newsdetail.asp?newsid=3455)
- Kandiah 2014：圍棋棋形網路的開局／中盤／終盤譜差異明顯，1d vs 9d 差異「relatively small」 — [ar5iv 1405.6077](https://ar5iv.arxiv.org/html/1405.6077)
- 慢性心衰竭 HRV 研究：四個非線性變數與線性變數相關 >0.90 而被剔除；只有 EMD 與 symbolic dynamics 兩類增加預後資訊 — [Polimi repository](https://re.public.polimi.it/handle/11311/519670)；糖尿病社論：線性 HRV 指標「performed even better than most complexity measures」 — [Arq Bras Cardiol](https://abccardiol.org/wp-content/uploads/articles_xml/0066-782X-abc-111-01-0102/0066-782X-abc-111-01-0102.x66747.pdf)
- RQA 的 DIV 與 Lyapunov 指數關係「more complex」，白噪音亦可出現長對角線 — [Wikipedia RQA](https://en.wikipedia.org/wiki/Recurrence_quantification_analysis)
- Eckmann–Ruelle：N=120 時維度上界 2log₁₀120≈4.2 — [UNIGE archive](https://archive-ouverte.unige.ch/unige:92420)
- 2505.00279 顯示僅用每局聚合量（幾何平均 LL、平均 strength score、Mean/Med/Std loss）即達 20 局 91.7% — [arXiv 2505.00279](https://arxiv.org/html/2505.00279)

### Inferences
- 動力系統觀點要求自治、確定性、平穩的生成機制；棋局評估序列的生成器是「兩個時變的隨機策略 + 一個固定評估函數 + 單調推進的階段」，三個前提皆不成立。文獻中的「超擴散」「反持續」都是在隨機過程（漂移＋噪音＋評估器均值回歸）框架下解釋，無人宣稱確定性混沌。
- 在 TCN 無增益的條件下，每手品質近似「給定位置的條件獨立抽樣」，則 SampEn/PE/RQA 這些量的期望值由邊際分佈（錯誤率、錯誤幅度分佈、位置銳度）決定，與平均品質、SD、分位數等已測特徵在資訊上重疊；它們的差異只剩有限樣本噪音，這解釋了為何 TSFEL 274 個特徵（含許多非線性項）會全部死亡。
- 「entropy」在棋局文獻中有效的形態（Δ、policy entropy）是**橫斷面**位置量，不是時間序列量；把它誤歸為「非線性動力學」會導向錯誤的工具箱（嵌入、遞迴、尺度律），正確工具是分佈統計與對手調整。
- 評估增量的反持續性（若圍棋亦然）主要反映 KataGo 評估對短期波動的均值回歸與雙方交替修正，是評估器與交替結構的性質，不是玩家技能的性質。

### Gaps
- 圍棋 KataGo 評估增量的 Hurst／分佈形狀無已發表資料；本專案可低成本自行計算以確認（非本研究範圍）。
- 無文獻對「評估軌跡的 surrogate 非線性檢定」做過報告，無法引用證據支持或反駁「軌跡存在非線性時間相依」。

---

## 關鍵問題 5：哪些領域複雜度量測勝過平均表現量（效果量）？哪些領域失敗？

### Takeaway
文獻中複雜度量測明確勝出的情境有三種：(1) 「隨機性本身就是技能」的遊戲（Morra：專家 vs 非專家的隨機性指標分佈重疊僅 0.08–0.23，但這不是對平均表現的增量，而是表現本身）；(2) 生理訊號中單尺度平均／方差無法分辨但多尺度複雜度能分辨的病理（白噪音 vs 1/f；HRV 的 DFA α1 vs SDRR/RMSSD，n=12）；(3) 非單調的「最適變異度」（飛鏢）。失敗案例：HRV 非線性指標與線性指標 r>0.90、糖尿病分辨線性更好、西洋棋棋子 entropy 與 Elo 無關、圍棋棋形網路 1d–9d 差異小、單局 centipawn 統計與 Elo 弱相關、volatility 受對手主導。圍棋等級預測的情境屬於「平均品質已是強訊號、時間結構弱」，對應失敗組而非勝出組。

### Cited Findings

**勝出（含效果量）**
- Morra（Delogu 2020，n=9+9，序列 57 個數字）：Redundancy 3.31 vs 15.53、NSQ 15.82 vs 37.90、FPR 0.47 vs 0.57，evidence ratio >1000，分佈重疊 0.08、0.12、0.11；RNGT 中無差異（ER=0.02） — [PMC7541701](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541701)
- 跑步 HRV（CINC 2026，n=12）：DFA α1 區分運動員與非運動員，SDRR、RMSSD「limited ability」 — [CINC 2026 #269](https://cinc.org/2026/Program/accepted/269.pdf)
- MSE 白噪音 vs 1/f：單尺度結論相反，多尺度才正確 — [PhysioNet MSE](https://archive.physionet.org/physiotools/mse/tutorial/node2.html)
- 飛鏢：中等變異度最佳（非單調） — [sportpedagogy](https://sportpedagogy.org.ua/index.php/ppcs/article/view/1526)
- 心臟病篩檢（CINC 2025 預印本）：「especially nonlinear parameters contribute to discriminating heart diseases in men」 — [CINC 2025 #134](https://cinc.org/2025/Program/accepted/134_Preprint.pdf)；線性＋非線性組合優於單獨 — [Inderscience](https://www.inderscience.com/filter.php?aid=34173)
- 西洋棋跨局開局序列：高 Elo 在長時間尺度相關較強（Schaigorodsky 2013；僅圖示、無數值） — [ar5iv 1307.0729](https://ar5iv.labs.arxiv.org/html/1307.0729)
- 位置銳度 Δ 的平均與 SD 隨等級下降（Emergent complexity 2024；有三個資料集對照） — [arXiv 2406.15463v2](https://arxiv.org/html/2406.15463v2)

**冗餘或失敗**
- 2024 scoping review：48/59 研究結論「專家變異度較低」且與指標無關 → entropy 類與 SD 類結論相同 — [UQTR eprint 11611](https://depot-e.uqtr.ca/id/eprint/11611/)
- 冬季兩項 SampEn 方向在兩項研究中相反（靜立較高 vs 瞄準較低） — [ISBS 2016](https://ojs.ub.uni-konstanz.de/cpa/article/view/6845)；[PMC9064080](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064080)
- HRV CHF：四個非線性變數與線性 r>0.90 被剔除 — [Polimi](https://re.public.polimi.it/handle/11311/519670)；糖尿病：線性優於多數複雜度量 — [Arq Bras Cardiol](https://abccardiol.org/wp-content/uploads/articles_xml/0066-782X-abc-111-01-0102/0066-782X-abc-111-01-0102.x66747.pdf)
- 西洋棋棋子佔據 entropy 模式「independent of player skill level」 — [IDEAS/RePEc](https://ideas.repec.org/a/wsi/ijmpcx/v35y2024i04ns0129183124500438.html)
- 圍棋棋形網路 1d vs 9d「relatively small」差異，同級樣本間亦有差異 — [ar5iv 1405.6077](https://ar5iv.arxiv.org/html/1405.6077)
- 單局 centipawn 統計與 Elo 弱相關 — [Erdős Institute](https://www.erdosinstitute.org/project-database/fall-2024/data-science-boot-camp/guess-the-elo-(chess))
- Volatility「depended a lot on the opponents」 — [Lichess discuss](https://lichess.org/ublog/H6MWvX98/discuss)
- Mangalam & Likens 2025：DFA 與 HK 在反持續序列皆失效 — [MDPI Entropy 27(5):500](https://www.mdpi.com/1099-4300/27/5/500)
- Erdős、2505.00279 與 Maia/Otter 的一致訊息：聚合的著手可預測性／品質（不含時序）才是等級的主訊號 — [arXiv 2505.00279](https://arxiv.org/html/2505.00279)；[arXiv 2608.05206](https://arxiv.org/pdf/2608.05206)

### Inferences
- 複雜度量測勝出的共同結構是：(a) 任務的目標函數本身是不可預測性（Morra），或 (b) 訊號的生成器是自治振盪系統（心跳）而平均值被穩態調節固定，使資訊只剩在時間結構裡。圍棋等級預測兩者皆非：目標是品質（平均已強），且生成器非自治。
- 「最適變異度」啟示的是**非單調**關係（U 形），這在本專案可用既有分佈特徵的二次項或分位數差捕捉，不需要 entropy。
- 唯一值得以低成本測試的新項目依優先序：(1) 玩家著手後局面的 KataGo policy entropy／top-2 policy 質量差之分佈（均值、SD、分位數），分開「自己創造的銳度」與「面對的銳度」，並以對手 rank-policy 分數校正；(2) 雙方品質序列的 cross-RQA 或簡單的交叉相關／條件錯誤率（「對手大錯後我方下一手的品質」），作為「對手影響」主題的量化；(3) 僅作為確認性負控制的 PE(m=3) 與 DET，預期與平均品質冗餘。其餘（MSE、DFA、Lyapunov、分形、criticality、entropy production、Kolmogorov）建議不投入。

### Gaps
- 未找到任何「複雜度量測 vs 平均表現」在**決策／棋類**任務上的 head-to-head 增量比較；所有勝出案例來自運動／生理／隨機生成。
- 效果量：體育科學文獻多數未報告 Cohen's d（冬季兩項、步槍射手均僅有方向與 p 值）；僅 Morra 給出可換算的均值與 SD。
- Schaigorodsky 2013 的各 Elo 段 Hurst 數值只以圖呈現，無法引用具體數字。
