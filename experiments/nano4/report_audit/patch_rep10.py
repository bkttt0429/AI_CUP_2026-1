from pathlib import Path
P = Path("reports/go_report/build.py"); s = P.read_text()
CH10 = r'''
 dict(n=10, title="下一步：局面表徵、集成與量化方法", hook="把每手的局面直接交給模型、<br>多個模型平均，能把分數推到哪？",
  lead="第 8 章的 v2 只用逐手 log p。本章測試把每手的局面表徵交給模型、用整題標籤訓練；記錄集成的大幅增益與它還沒確認的部分；最後把量化交易處理重疊標籤與回測過擬合的方法對應到這裡，排出下一步。",
  meta="results/trunk_seq · folds 1–4 · 各臂 3 個 seed · 未用 fold 0", route=["表徵", "集成", "量化", "順序"],
  conclusion=f"新表徵加整題訓練通過門檻，時序層沒有；6 個模型等權平均是目前最強的離線候選（{num(VM['CT_ens']['score'])}），但要用新 seed 重現、用 fold 0 檢查一次，才進提交路徑。",
  next_q="第三輪的消融與重現結果出來後，決定是否進入重疊處理與 fold 0 檢查。", refs=[REF_TCN, REF_LDP, REF_GK, REF_DM, REF_PBO, REF_REPO],
  cards=[
   dict(id="card-trunk", q="逐手的局面表徵加整題損失，有沒有用？", h=f"有用：C、T 都比只用 R 高約 0.01；時序層沒有額外增益（T − C {signed(TSC['T vs C']['diff'])}）",
    formula="R0 只用 R ｜ C 逐手 MLP ｜ T 五層 TCN，感受野 63 ply；三臂同一訓練器",
    bullets=["每手 = humanv0 全局與落子點向量（各投影成 64 維）+ 逐手 log p + 角色；整題 CE 一路回傳到時序層", f"folds 2–4（事先指定）：C − R0 {signed(TSC['C vs R0']['diff'])} {ci(TSC['C vs R0'])}，T − R0 {signed(TSC['T vs R0']['diff'])} {ci(TSC['T vs R0'])}", f"T − C {signed(TSC['T vs C']['diff'])} {ci(TSC['T vs C'])}：看不出手順有額外幫助"],
    fig=fig(table(["模型", "fold 1", "fold 2", "fold 3", "fold 4", "平均"], [[n, *[num(x) for x in pf], num(sum(pf) / 4)] for n, pf in (("R0（只有 R）", TS4["arms"]["R0"]["per_fold"]), ("C（逐手 MLP）", TS4["arms"]["C"]["per_fold"]), ("T（TCN）", TS4["arms"]["T"]["per_fold"]), ("現行 MLP h32", VM["MLP"]["per_fold"]))]), "各臂 3 個 seed 的平均；停止點只在訓練玩家中選。fold 1 已在第一輪看過，判定只用 folds 2–4", "results/trunk_seq/es/compare_*.json", "folds 1–4"),
    cause=("舊的六個序列實驗只看 log p、用每局監督", "同輸入的 C 與 T 對照，整題 CE", "單模型 seed 雜訊大，平均之後呢？"),
    sec="10.1 trunk 向量與時序層",
    body=f"""<p>每手存兩個 trunk 向量：整盤平均與實際落子點，取自落子前的局面，用未改動的 humanv0（固定的「等級未知」metadata，不含任何標籤資訊），各以固定隨機投影壓到 64 維。抽取範圍是 folds 1–4 的評分樣本，共 2,624 萬個 ply，每局都與既有逐手分數逐手對齊。L1b 的 trunk 其實以 lr 1e-5 微調過，這裡刻意不用，因為 A、B 兩份 trunk 的向量對不齊。</p>
<p>第一輪固定 8 個 epoch：訓練 loss 從 1.0 掉到 0.4，驗證 NLL 從 1.1 升到 1.6，嚴重過擬合，那一輪不作結論。第二輪在訓練玩家中留 10% 選停止點，驗證折只評一次；C、T 多停在 1–3 epoch，R0 在 3–6 epoch。R0 少了那 10% 的玩家，所以比現行 MLP 低 {signed(-VM['R0']['diff']).replace('+', '')}。</p>
<p>結論只限於這個設定：未微調的 trunk、64 維投影、kernel 3 的 TCN。它說明逐手的局面表徵加整題訓練有價值，不排除更長程或更大的時序模型有用。</p>""",
    take="增益來自逐手表徵與整題訓練；手順在這個設定下沒有額外價值。"),
   dict(id="card-ens", q="多個模型平均能拿多少？", h=f"C、T 共 6 個模型平均到 {num(VM['CT_ens']['score'])}，比現行 MLP 高 {signed(VM['CT_ens']['diff'])}；這是事後發現的，要用新 seed 重現",
    formula="p = 6 個模型機率的等權平均；權重不在驗證折上調",
    bullets=[f"單模型對現行 MLP：C {signed(VM['C']['diff'])}、T {signed(VM['T']['diff'])}；各 3 個 seed 平均：C {signed(VM['C_ens']['diff'])}、T {signed(VM['T_ens']['diff'])}", f"6 個模型平均四折全正（{signed(min(VM['CT_ens']['per_fold_diff']))} 到 {signed(max(VM['CT_ens']['per_fold_diff']))}），直接平均 {num(VM['CT_ens']['direct'])}（現行 {num(VM['MLP']['direct'])}）", "沒有事先指定、沒用 fold 0、還沒進提交路徑"],
    fig=fig(table(["模型（同題）", "分數", "對現行 MLP", "95% CI"], [[n, num(VM[k]["score"]), signed(VM[k]["diff"]), ci(VM[k])] for n, k in (("R0 單模型", "R0"), ("R0，3 seed 平均", "R0_ens"), ("C 單模型", "C"), ("C，3 seed 平均", "C_ens"), ("T 單模型", "T"), ("T，3 seed 平均", "T_ens"), ("C + T，6 個模型平均", "CT_ens"))], bold_last=True), f"現行 MLP h32 在同一批題 {num(VM['MLP']['score'])}；單模型列是 3 個 seed 逐題分數的平均，不是機率平均", "results/trunk_seq/es/vs_mlp.json", "folds 1–4，只報告"),
    cause=("單模型 seed 間最多差約 0.01", "等權平均多個 seed 與兩種架構", "增益從哪來、能不能重現？"),
    sec="10.2 集成",
    body=f"""<p>R0 平均 3 個 seed 只多 {signed(VM['R0_ens']['diff'] - VM['R0']['diff'])}，C、T 卻多約 0.012：表徵模型的單模型雜訊大，而且停在不同地方，平均把雜訊抵銷掉。C 與 T 的錯誤也不完全相同，所以合在一起又再高約 0.005。</p>
<p>這組數字是看到單模型結果後才算的。第三輪已事先寫定：全新 seed 3–5 的 C、T 共 6 個模型平均，對同題現行 MLP，條件 (1) CI 下界 &gt; 0、(3) 四折至少三折 &gt; 0、(4) 差 ≥ 0.005；過了才談 fold 0。</p>""",
    take="目前最強的離線候選，但在重現與 fold 0 檢查之前不算數。"),
   dict(id="card-quant", q="量化交易的方法能用在哪裡？", h="標籤重疊、廣度定律、等權、多重檢定、分布漂移五件事直接對應；動能、均值回歸、組合最佳化不適用",
    formula="IR ≈ IC × √breadth（Grinold）",
    bullets=["標籤重疊：每人最多 20 局卻抽 32 題，同一局約出現在 20 題裡，對應樣本獨特性問題", "廣度定律與等權：多個低相關的模型等權平均；權重不在驗證折上最佳化", "回測紀律：事先指定、fold 0 一次、新 seed 重現；11/04 做對抗驗證"],
    fig=fig(table(["量化方法", "這裡的對應", "動作"], [["樣本獨特性、序列 bootstrap", "訓練題彼此重疊，模型 1–3 epoch 就記住棋局", "每 epoch 重抽題目、依重疊加權、每個模型只用 80% 玩家"], ["廣度定律 IR ≈ IC √N", "集成的增益來自模型間低相關", "量模型間相關、畫 seed 數曲線"], ["等權（1/N）", "集成權重", "一律等權"], ["回測過擬合（PBO、Deflated Sharpe）", "二十多次比較、事後發現的集成", "試驗紀錄、新 seed 重現、fold 0 只用一次"], ["對抗驗證", "測試分布可能和驗證題不同", "11/04 分類驗證題與測試題"], ["中性化、rank-gauss、meta-labeling", "局數效應、A／B 尺度、決策確定度", "之後再排；按局數標準化（#53）已在列"]], num_from=9), "動能、均值回歸、交易成本、Markowitz 權重最佳化不適用；最後一項在這裡等於在驗證折上調權重", "López de Prado 2018；Grinold 與 Kahn 1999；DeMiguel 等 2009；Bailey 等 2017"),
    cause=("單模型很快過擬合、集成增益很大", "借量化處理重疊標籤與回測過擬合的方法", "要怎麼排順序？"),
    sec="10.3 量化方法的對應",
    body="""<p><b>標籤重疊。</b>金融資料的標籤時間窗會重疊，樣本彼此不獨立，模型反覆學同一段資料而過擬合；López de Prado 用樣本獨特性加權與序列 bootstrap 處理。這裡的訓練題是從每人最多 20 局裡抽 32 題，同一局平均出現在約 20 題，正是同一個問題，也和 C、T 在 1–3 epoch 就過擬合一致。對應的做法是每個 epoch 重抽題目、按局的重疊次數給題目權重，以及真正的 bagging：每個模型只用隨機 80% 的玩家。</p>
<p><b>廣度定律與等權。</b>主動管理的基本定律說資訊比率約等於訊號品質乘以獨立下注數的平方根：多個不完全相關的模型平均，雜訊抵銷、訊號留下。DeMiguel 等人的結果是等權常勝過估計出來的最佳權重，因為權重本身有估計誤差；這裡的集成一律等權。</p>
<p><b>回測過擬合。</b>試很多組合再挑最好的，報告的分數就有選擇偏誤（Bailey 等人的 PBO、Deflated Sharpe）。事先指定規則、fold 0 只用一次、事後發現的結果用新 seed 重現，都是同一個紀律；之後再補一份完整的試驗紀錄。</p>
<p><b>分布漂移。</b>實盤前要確認市場和回測期一樣；這裡的對應是對抗驗證：拿到測試集後，訓練分類器分辨驗證題與測試題，分得出來就代表局數、讓子比例或配對方式不同。</p>""",
    take="量化方法在這裡的價值是處理重疊樣本與選模偏誤，不是新的訊號來源。"),
   dict(id="card-plan", q="接下來依序做什麼、怎麼判定？", h="四步：消融與新 seed 重現 → 處理標籤重疊 → 凍結配方用 fold 0 一次 → 接進提交路徑",
    formula="每一步的判定規則都在看結果前寫進架構文件 §4A",
    bullets=["第三輪：C 去 trunk 向量、C 去逐手分數各對 C；新 seed 3–5 的 6 模型平均對現行 MLP", "第四輪：每 epoch 重抽題目、重疊加權、80% 玩家 bagging，各自對同 seed 數的 C + T 集成", "11/04 照原計畫上傳現有文件；新候選要等 fold 0 與 Public 榜確認"],
    fig=fig(table(["步", "內容", "判定", "成本（估）"], [["1", "消融、新 seed 3–5 重現集成", "消融對 C；集成對現行 MLP，條件 (1)(3)(4)", "48 個 GPU 工作，約 2–3 小時"], ["2", "重抽題目、重疊加權、玩家 bagging", "對同 seed 數的 C + T 集成", "同量級"], ["3", "凍結配方，fold 0 檢查一次", "CI 下界 > 0；第五次使用，要使用者同意", "數十分鐘"], ["4", "接進提交路徑", "mock-40 對帳、Public 榜對照", "測試多一次 humanv0 前向，800 題約 220 萬 ply，8 卡數分鐘"]], num_from=9), "第 1 步消融若顯示 trunk 向量沒有貢獻，提交時就不必多跑 trunk", "架構設計 v2 §4A"),
    cause=("集成結果是事後發現的", "每一步先寫規則、再跑", "fold 0 與 Public 榜才是最後的確認"),
    sec="10.4 實驗順序",
    body="""<p>消融回答增益來源：若去掉 trunk 向量後 C 不變，增益來自逐手 log p 加整題訓練，提交時不必多一次 trunk 前向；若去掉逐手分數後 C 不變，代表 trunk 向量已涵蓋那些資訊。</p>
<p>提交路徑有兩件事要設計：測試時每局多跑一次 humanv0 前向取 trunk 向量；CV 裡每折只用 A 或 B 其中一個 head 的分數，正式提交要讓 A、B 各出一份 token 後把機率平均，和現行 MLP 的做法相同，這一步要用 mock-40 對帳。</p>
<p>前面三步都不碰 fold 0；第 3 步是 fold 0 的第五次使用，要使用者本人同意。</p>""",
    take="每一步都可能讓候選退出；fold 0 與 Public 榜確認之前，提交仍用現有文件。"),
  ]),
'''
R = [
 ('    take="目前的最佳估計只是離線分數；測試集公布後，第一件事是驗證這些假設。"),\n  ]),\n]', '    take="目前的最佳估計只是離線分數；測試集公布後，第一件事是驗證這些假設。"),\n  ]),' + CH10 + ']', 1),
 ('next_q="11/04 測試集公布後，先看 num_games 與 HD 計數、跑合法性檢查，再評分與提交。"', 'next_q="現有文件之外，下一個候選是什麼？第 10 章。"', 1),
 ('HD_STAB = {"10": 0.845, "5": 0.733}\n', '''HD_STAB = {"10": 0.845, "5": 0.733}
HDH = js("results/seq/hd_check_hd.json")
TSC = js("results/trunk_seq/es/compare_2_3_4.json")["pairs"]
TS4 = js("results/trunk_seq/es/compare_1_2_3_4.json")
VM = {**js("results/trunk_seq/es/vs_mlp.json")["folds 1 2 3 4"]}
VM["MLP"]["per_fold"] = VM["MLP"]["per_fold"]
assert round(score(HDH, "fused_opp_diff"), 4) == 0.6929 and round(score(HDH, "fused_opp_diff_hd"), 4) == 0.7026
assert round(TSC["T vs C"]["diff"], 4) == 0.0007 and round(TSC["C vs R0"]["diff"], 4) == 0.0098 and round(TSC["T vs R0"]["diff"], 4) == 0.0105
assert round(VM["CT_ens"]["score"], 4) == 0.7241 and round(VM["CT_ens"]["diff"], 4) == 0.0214 and round(VM["MLP"]["score"], 4) == 0.7026
REF_TCN = "Bai, S., Kolter, J. Z., &amp; Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. <em>arXiv:1803.01271</em>."
REF_LDP = "López de Prado, M. (2018). <em>Advances in Financial Machine Learning</em>. Wiley.（樣本獨特性、序列 bootstrap、meta-labeling）"
REF_GK = "Grinold, R. C., &amp; Kahn, R. N. (1999). <em>Active Portfolio Management</em> (2nd ed.). McGraw-Hill.（主動管理的基本定律）"
REF_DM = "DeMiguel, V., Garlappi, L., &amp; Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? <em>Review of Financial Studies</em>, 22(5), 1915–1953."
REF_PBO = "Bailey, D. H., Borwein, J. M., López de Prado, M., &amp; Zhu, Q. J. (2017). The probability of backtest overfitting. <em>Journal of Computational Finance</em>, 20(4), 39–69."
''', 1),
 # chapter 2: the L1b trunk is fine-tuned
 ('conclusion="KataGo 只當凍結的特徵抽取器：', 'conclusion="Human SL 只當凍結的特徵抽取器，L1b 則微調了 trunk：', 1),
 ('不做搜尋，網路權重凍結"', '不做搜尋；Human SL 權重不動，L1b 的 trunk 用本資料微調過"', 1),
 ('["trunk", "凍結；L1b 在它上面加 10 個自訓的等級', '["trunk", "Human SL 不動；L1b 以 lr 1e-5 微調後加 10 個自訓的等級', 1),
 ('take="KataGo 在這裡是凍結的特徵抽取器；所有訓練都發生在它之上或之後。"', 'take="Human SL 在這裡是凍結的特徵抽取器；L1b 是唯一動到 trunk 權重的地方（lr 1e-5）。"', 1),
 ('L1b 評分約 900 局面／秒／卡；全量兩側約 2.2 億局面，8 卡一節點約 8–9 小時', 'L1b 評分約 900 局面／秒／卡；評分樣本（每人 ≤ 20 局）兩側共約 4,000 萬局面，8 卡約 1.5 小時', 1),
 ('L1b 10 個 head 對目標手與對手手的 log p；trunk 凍結"', 'L1b 10 個 head 對目標手與對手手的 log p；Human SL 權重不動，L1b trunk 微調過"', 1),
 ('"10 個等級 head 接在凍結的 humanv0 trunk 上，一次前向出 10 個 log p"', '"10 個等級 head 接在 humanv0 trunk 上（第二階段起 trunk 以 lr 1e-5 微調），一次前向出 10 個 log p"', 1),
 # chapter 8: R-hdpatch is done
 ('conclusion="架構凍結；下一批最有價值的結果是修正後基準的 E10／E11，與 r3_pma 在同一份特徵上的增量。"', 'conclusion=f"R-hdpatch 已完成（fused_opp_diff {num(score(HDH, \'fused_opp_diff\'))}、加 HD {num(score(HDH, \'fused_opp_diff_hd\'))}）；下一個正式驗證是第 10 章的局面表徵與整題訓練。"', 1),
 ('next_q="模型定案之後，提交流程與未確認事項是什麼？第 9 章說明。"', 'next_q="現有文件的提交流程與未確認事項是什麼？第 9 章。"', 1),
 ('h="Stage 1 凍結 trunk 出每手 log p，', 'h="Stage 1 由 Human SL 與 L1b 出每手 log p，', 1),
 ('"Stage 1：不從頭訓 CNN、不微調 trunk；候選新訊號是 policy 摘要、強網棋損 Q、後續著手預測增益 G"', '"Stage 1：不從頭訓 CNN；Human SL 不動、L1b trunk 已微調；候選新訊號是 trunk 向量（第 10 章）、policy 摘要、強網棋損 Q、後續著手預測增益 G"', 1),
 ('h="R-hdpatch 只重評 HD≥2 的 36,421 局，是過渡版；R-full 要重訓 head 並全量重評"', 'h=f"R-hdpatch 已完成：fused_opp_diff {num(score(HD0, \'fused_opp_diff\'))} → {num(score(HDH, \'fused_opp_diff\'))}，加 HD {num(score(HD, \'fused_opp_diff_hd\'))} → {num(score(HDH, \'fused_opp_diff_hd\'))}；R-full 要重訓 head 並全量重評"', 1),
 ('formula="R-hdpatch ≈ 0.65 節點小時（L1b 兩側）；R-full (a) ≈ 68 GPU 小時 + Human SL"', 'formula="R-hdpatch 實測：L1b 四支各約 9 分、Human SL 1 小時 9 分；R-full (a) 重評約 1.5 小時（8 卡）＋ 重訓 head"', 1),
 ('"R-hdpatch 重評後不劣（下降 ≤ 0.002）才重擬合文件"', f'f"修正後 HD 5 維的增益 {{signed(comparison(HDH, \'fused_opp_diff_hd vs fused_opp_diff\')[\'diff\'])}} {{ci(comparison(HDH, \'fused_opp_diff_hd vs fused_opp_diff\'))}}；59 維與 64 維文件已用 --no-fold0 重擬合"', 1),
 ('take="主線一條（R-hdpatch → E10／E11）、探索一條（r3_pma）；其他候選等前兩項的結果表。"', 'take="R-hdpatch 已出結果；探索的優先順序改為第 10 章的局面表徵與集成，r3_pma 在後。"', 1),
 # chapter 9
 ('"R-hdpatch 的 E10／E11 出來後，決定 HD 版文件是否進場"', '"測試檔帶 HD 且 fold 0 檢查通過，HD 版文件才進場"', 1),
 ('R-hdpatch 是過渡版（head 不變），重擬合文件要再用一次 fold 0；', 'R-hdpatch 是過渡版（head 不變）；59 維與 64 維文件已用 --no-fold0 重擬合，沒有 fold 0 對照；', 1),
 ('take="提交路徑與離線路徑已逐項對上；R-hdpatch 完成後同一套對帳再做一次。"', 'take="提交路徑與離線路徑已逐項對上；HD 版文件的 mock-40 也已跑過：40／40 列、沒有退路題，和 59 維文件有 7 題決策不同。"', 1),
 # progress chart and cover
 ('("加讓子 HD 5 維（候選）", score(HD, "fused_opp_diff_hd"), None)]', '("加讓子 HD 5 維（修正前）", score(HD, "fused_opp_diff_hd"), None), ("讓子修正後 + HD（候選）", score(HDH, "fused_opp_diff_hd"), None)]', 1),
 ('    W, H, L, R, top = 640, 302, 178, 60, 40\n', '    W, H, L, R, top = 640, 342, 178, 60, 40\n', 1),
 ('"六個里程碑的離線分數：官方 baseline 0.440，Human SL 0.6192，fused 0.6545，fused_opp 0.6735，fused_opp_diff 0.6916，加 HD 0.6987"', '"七個里程碑的離線分數：官方 baseline 0.440，Human SL 0.6192，fused 0.6545，fused_opp 0.6735，fused_opp_diff 0.6916，加 HD 0.6987，讓子修正後 0.7026"', 1),
 ("；再加讓子 5 維的候選版 CV {num(score(HD,'fused_opp_diff_hd'))}，還沒做 fold 0 檢查。", "；讓子修正後再加 5 維的候選版 CV {num(score(HDH,'fused_opp_diff_hd'))}，還沒做 fold 0 檢查。第 10 章的 6 模型集成離線到 {num(VM['CT_ens']['score'])}，仍待重現與 fold 0 檢查。", 1),
]
for a, b, n in R:
    c = s.count(a)
    assert c == n, (a[:60], c)
    s = s.replace(a, b)
P.write_text(s)
print("patched", len(R))
