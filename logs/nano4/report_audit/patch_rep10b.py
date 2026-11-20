from pathlib import Path
P = Path("reports/go_report/build.py"); s = P.read_text()
R = [
('assert round(VM["CT_ens"]["score"], 4) == 0.7241 and round(VM["CT_ens"]["diff"], 4) == 0.0214 and round(VM["MLP"]["score"], 4) == 0.7026\n',
 '''assert round(VM["CT_ens"]["score"], 4) == 0.7241 and round(VM["CT_ens"]["diff"], 4) == 0.0214 and round(VM["MLP"]["score"], 4) == 0.7026
ABL4 = js("results/trunk_seq/es/compare_abl_1_2_3_4.json")
ABL = ABL4["pairs"]
VS = js("results/trunk_seq/es/vs_mlp_s345.json")["folds 1 2 3 4"]
V12 = js("results/trunk_seq/es/vs_mlp_s0_5.json")["folds 1 2 3 4"]
GBC = {**js("results/gbdt_hd/gbdt_hd_check.json")["comparisons"]["pre_specified"], **js("results/gbdt_hd/gbdt_hd_check.json")["comparisons"]["report_only"]}
assert round(ABL["C-notrunk vs C"]["diff"], 4) == -0.0053 and round(ABL["C-noscores vs C"]["diff"], 4) == -0.0003
assert round(VS["CT_ens"]["diff"], 4) == 0.0209 and round(V12["CT_ens"]["score"], 4) == 0.7258 and round(GBC["GBDT vs MLP0"]["diff"], 4) == -0.0056
'''),
('conclusion=f"新表徵加整題訓練通過門檻，時序層沒有；6 個模型等權平均是目前最強的離線候選（{num(VM[\'CT_ens\'][\'score\'])}），但要用新 seed 重現、用 fold 0 檢查一次，才進提交路徑。"',
 'conclusion=f"新表徵加整題訓練通過門檻，時序層沒有；增益主要來自 trunk 向量（去掉後 {signed(ABL[\'C-notrunk vs C\'][\'diff\'])}）。C + T 集成已用全新 seed 重現（{num(VS[\'CT_ens\'][\'score\'])}，{signed(VS[\'CT_ens\'][\'diff\'])}），還要用 fold 0 檢查一次、接進提交路徑。"'),
('next_q="第三輪的消融與重現結果出來後，決定是否進入重疊處理與 fold 0 檢查。"', 'next_q="下一步：處理標籤重疊、在新表徵上重比 GBDT，之後凍結配方用 fold 0 一次。"'),
('("C（逐手 MLP）", TS4["arms"]["C"]["per_fold"]), ', '("C（逐手 MLP）", TS4["arms"]["C"]["per_fold"]), ("C 去 trunk 向量", ABL4["arms"]["C-notrunk"]["per_fold"]), ("C 去逐手分數", ABL4["arms"]["C-noscores"]["per_fold"]), '),
('"各臂 3 個 seed 的平均；停止點只在訓練玩家中選。fold 1 已在第一輪看過，判定只用 folds 2–4"', '"各臂 3 個 seed 的平均；停止點只在訓練玩家中選。fold 1 已在第一輪看過，三臂的判定只用 folds 2–4；兩個消融列是第三輪"'),
('<p>結論只限於這個設定：', '<p>消融（folds 1–4、seed 0–2，事先指定）：去掉 trunk 向量 {signed(ABL[\'C-notrunk vs C\'][\'diff\'])} {ci(ABL[\'C-notrunk vs C\'])}，去掉逐手分數 {signed(ABL[\'C-noscores vs C\'][\'diff\'])} {ci(ABL[\'C-noscores vs C\'])}。增益主要來自 trunk 向量；有了 trunk 向量與 R 之後，逐手分數沒有額外貢獻，所以提交時多一次 trunk 前向是必要的。</p>\n<p>結論只限於這個設定：'),
('h=f"C、T 共 6 個模型平均到 {num(VM[\'CT_ens\'][\'score\'])}，比現行 MLP 高 {signed(VM[\'CT_ens\'][\'diff\'])}；這是事後發現的，要用新 seed 重現"', 'h=f"C、T 共 6 個模型平均比現行 MLP 高 {signed(VM[\'CT_ens\'][\'diff\'])}；用全新 seed 重現得到 {signed(VS[\'CT_ens\'][\'diff\'])}，四折全正"'),
('"沒有事先指定、沒用 fold 0、還沒進提交路徑"]', 'f"事先指定的重現（seed 3–5）：{num(VS[\'CT_ens\'][\'score\'])}，{signed(VS[\'CT_ens\'][\'diff\'])} {ci(VS[\'CT_ens\'])}，三個條件都過；沒用 fold 0、還沒進提交路徑"]'),
('[[n, num(VM[k]["score"]), signed(VM[k]["diff"]), ci(VM[k])] for n, k in (("R0 單模型", "R0"), ("R0，3 seed 平均", "R0_ens"), ("C 單模型", "C"), ("C，3 seed 平均", "C_ens"), ("T 單模型", "T"), ("T，3 seed 平均", "T_ens"), ("C + T，6 個模型平均", "CT_ens"))]',
 '[[n, num(d["score"]), signed(d["diff"]), ci(d)] for n, d in (("R0 單模型", VM["R0"]), ("C 單模型", VM["C"]), ("T 單模型", VM["T"]), ("C + T，seed 0–2（事後）", VM["CT_ens"]), ("C + T，seed 3–5（事先指定）", VS["CT_ens"]), ("C + T，seed 0–5，12 個模型", V12["CT_ens"]))]'),
('過了才談 fold 0。</p>', '三個條件都過：{signed(VS[\'CT_ens\'][\'diff\'])} {ci(VS[\'CT_ens\'])}，四折 {" / ".join(signed(x) for x in VS[\'CT_ens\'][\'per_fold_diff\'])}。從 6 個到 12 個模型只再多 {signed(V12[\'CT_ens\'][\'diff\'] - VS[\'CT_ens\'][\'diff\'])}，邊際效益已經小了。</p>'),
('take="目前最強的離線候選，但在重現與 fold 0 檢查之前不算數。"', 'take="已用新 seed 重現；fold 0 檢查與提交路徑對帳之前，仍只是離線候選。"'),
('h="四步：消融與新 seed 重現 → 處理標籤重疊 → 凍結配方用 fold 0 一次 → 接進提交路徑"', 'h="第 1 步已完成（trunk 向量有貢獻、集成重現）；接著處理標籤重疊、新表徵上的 GBDT，再用 fold 0 一次、接進提交路徑"'),
('"第三輪：C 去 trunk 向量、C 去逐手分數各對 C；新 seed 3–5 的 6 模型平均對現行 MLP"', 'f"GBDT 在現有 64 維特徵上重新公平比較：對 MLP {signed(GBC[\'GBDT vs MLP0\'][\'diff\'])}，和 MLP 平均也只多 {signed(GBC[\'MLP0+GBDT vs MLP0+MLP1\'][\'diff\'])}，都不通過"'),
('["1", "消融、新 seed 3–5 重現集成", "消融對 C；集成對現行 MLP，條件 (1)(3)(4)", "48 個 GPU 工作，約 2–3 小時"]', '["1（完成）", "消融、新 seed 3–5 重現集成", f"去 trunk {signed(ABL[\'C-notrunk vs C\'][\'diff\'])}；重現 {signed(VS[\'CT_ens\'][\'diff\'])}，通過", "48 個 GPU 工作"]'),
('["2", "重抽題目、重疊加權、玩家 bagging", "對同 seed 數的 C + T 集成", "同量級"]', '["2", "重抽題目、重疊加權、玩家 bagging；C、T 題目表徵上重比 GBDT", "對同 seed 數的 C + T 集成", "同量級；GBDT 每折約 40 秒"]'),
('"第 1 步消融若顯示 trunk 向量沒有貢獻，提交時就不必多跑 trunk"', '"消融顯示 trunk 向量有貢獻，所以提交時要多跑一次 humanv0 前向"'),
('<p>消融回答增益來源：若去掉 trunk 向量後 C 不變，增益來自逐手 log p 加整題訓練，提交時不必多一次 trunk 前向；若去掉逐手分數後 C 不變，代表 trunk 向量已涵蓋那些資訊。</p>', '<p>消融的答案是 trunk 向量：去掉它掉 0.005，去掉逐手分數幾乎不變。所以提交時要多一次 humanv0 前向；逐手分數那一欄可以考慮拿掉，簡化 token。</p>'),
("第 10 章的 6 模型集成離線到 {num(VM['CT_ens']['score'])}，仍待重現與 fold 0 檢查。", "第 10 章的 C + T 集成已用新 seed 重現（{num(VS['CT_ens']['score'])}，{signed(VS['CT_ens']['diff'])}），仍待 fold 0 檢查。"),
]
for a, b in R:
    c = s.count(a)
    assert c == 1, (a[:70], c)
    s = s.replace(a, b)
P.write_text(s); print("patched", len(R))
