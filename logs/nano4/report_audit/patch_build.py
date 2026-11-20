"""One-shot text fixes to build.py; each fragment must match exactly the expected number of times."""
from pathlib import Path
import re
P = Path("reports/go_report/build.py"); s = P.read_text()
R = [
 ("要求預測他在 10 個有序等級中的哪一級", "要求預測這位玩家在 10 個有序等級中的哪一級", 1),
 ("每題給一位玩家的 5–20 局 SGF 與他的執色", "每題給一位玩家的 5–20 局 SGF 與各局執色", 1),
 ("q<sub>k−1</sub> + q<sub>k+1</sub>）再取最大", "q<sub>k−1</sub> + q<sub>k+1</sub>) 再取最大", 1),
 ("它比官方的直接平均低約 0.007（提交候選：平衡版 0.6916、直接平均 0.6845），差來自各級題數不均。", "與官方同口徑的直接平均（每題等權）比它低約 0.007（提交候選：平衡版 0.6916、直接平均 0.6845），因為平衡版讓三種題長與各等級等權，直接平均則照內部驗證題的組成。", 1),
 ("但計分讓「差一級」有一半以上的價值；", "但計分讓「差一級」仍值 0.368 分；", 1),
 ("因為同一人的多題高度相關（第 5 章有實測）", "因為同一人的多題高度相關（第 7 章有實測）", 1),
 ('"下面的局面：12k 與 6d 兩個 profile 的分布總變異距離 0.4 以上"', 'f"下面的局面：12k 與 6d 兩個 profile 的分布總變異距離 {VIZ[\'position\'][\'tv_12k_6d\']:.2f}"', 1),
 ("每位玩家的特徵都來自沒看過他的模型", "每位玩家的特徵都來自沒看過這位玩家的模型", 2),
 ("Human SL 映射 +0.18，鑑別式 L1b +0.035，對手通道兩步 +0.036，讓子 5 維再 +0.007（候選）", "Human SL 映射 +0.17，鑑別式 L1b +0.035，對手通道兩步 +0.037，讓子 5 維再 +0.007（候選）", 1),
 ("fold 0 每次都略低於 CV（−0.001 到 −0.003），與折間變異相容；fold 0 已用四次，再用要使用者同意", "+0.17 是 fold 0 同題比較（0.6113 − 0.440），其餘是 CV；L1b 之後的三版 fold 0 略低於 CV（−0.001 到 −0.003），hsl 低 −0.008，都與折間變異相容；fold 0 已用四次，再用要使用者同意", 1),
 ("藍點是 folds 1–4 CV，空心點是 fold 0", "實心點是 folds 1–4 CV，空心點是 fold 0", 1),
 ("對手通道兩步共 +0.036，HD 5 維再 +0.0071", "對手通道兩步共 +0.037，HD 5 維再 +0.0071", 1),
 ("聚合器的增益上限約 0.01；", "單一聚合器的增益上限約 0.01，三者平均也只到約 +0.014；", 1),
 ("所有結構都落在 0.60–0.63：整題監督的集合模型比 MLP 多 +0.005 到 +0.009，上限約 0.01", "所有結構都落在 0.60–0.63：整題監督的集合模型比 MLP 多 +0.005 到 +0.009，單一結構上限約 0.01", 1),
 ("這三步是目前通過門檻的全部改進", "L1b 之後通過門檻的只有這三步", 1),
 ("對手通道三步（對手、差值、讓子）是目前全部的有效改進；它們都依賴「測試局也是配對產生」這個前提。", "對手、差值、讓子三步是 L1b 之後僅有的有效改進；它們都依賴「測試局也是配對產生」這個前提，讓子還要測試檔帶 HD。", 1),
 ("十六項比較只有對手通道通過；最差的是 CORAL（−0.026）與 TSFEL 全塊（−0.061）", "圖上 15 列（含一條純雜訊對照）只有對手、差值、讓子三步過門檻；最差的是 TSFEL 全塊（−0.061）與 CORAL（−0.026）", 1),
 ("藍色是已採用的三步。TSFEL 64／全塊超出軸範圍，未畫", "綠色是過門檻的三步（HD 還沒進提交）。TSFEL 64／全塊超出軸範圍，L2、softmax、提早停止只列在左欄，未畫", 1),
 ("十五項比較中只有對手通道的三步通過門檻", "十五項比較中只有對手、差值、讓子三步通過門檻", 1),
 ("通過門檻的只有對手通道三步；", "L1b 之後通過門檻的只有對手、差值、讓子三步；", 1),
 ("並發現訓練題局數混合是傷害的一大半來源。", "並看到訓練題固定 5 局能把 TSFEL 對新玩家的傷害縮小約七成；不過那個改動同時換了標準化與優化，機制沒有分離。", 1),
 ("它的大半成分是抽樣雜訊，且混合局數訓練讓 5 局題的權重過大。", "它的大半成分是抽樣雜訊；訓練題固定 5 局能縮小傷害，但機制尚未分離。", 1),
 ("同一人多題全對或全錯的比例都比獨立假設高一個量級：", "同一人多題全對或全錯的比例都是獨立假設的 3–4 倍（18.4% 對 5.1%、14.1% 對 3.8%）：", 1),
 ("fused_opp 的 OOF 題，每人 7 題", "fused_opp 的 OOF 題，每人 4 或 7 題", 1),
 ("固定 5 局訓練把傷害縮小 0.0108：訓練題局數混合時，TSFEL 在多局題上雜訊較小，MLP 學到的權重對 5 局題過大。同一個改動也換了標準化與優化，機制沒有分離，", "固定 5 局訓練把傷害縮小 0.0108。一個可能的解釋是：訓練題局數混合時，TSFEL 在多局題上雜訊較小，MLP 學到的權重對 5 局題過大。但同一個改動也換了標準化與優化，機制沒有分離，", 1),
 ("HD 五欄與 l1b／opp 同級（gap 欄最穩，讓先與否最不穩）", "HD 五欄與 l1b／opp 同級（hd_gap_down 最穩，讓先比例最不穩）", 1),
 ("混合局數訓練仍讓雜訊塊的權重偏向多局題。", "混合局數訓練仍可能讓雜訊塊的權重偏向多局題。", 1),
 ("純 CPU 便宜項是否核定（第 6.2 節）", "純 CPU 便宜項是否核定（第 8.2 節）", 1),
 ("步 3 重評 HD≥2（五支 GPU 工作）— 已啟動", "步 3 重評 HD≥2（五支 GPU 工作）— 進行中（11/02：四支 L1b 已完成，Human SL 在跑）", 1),
 ("只重評 HD≥2 的 36,421 局，11/04 用它；R-full", "只重評 HD≥2 的 36,421 局，是 11/04 前做得完的過渡版；R-full", 1),
 ("R-hdpatch 是過渡版（head 不變），11/04 用它並標明；", "R-hdpatch 是過渡版（head 不變）；重評後不劣（下降 ≤ 0.002）才重擬合文件，這要再用一次 fold 0。11/04 用哪個文件由使用者決定，採用過渡版時要標明；", 1),
 ("現行提交模型 fused_opp 在 folds 1–4 CV 是 <strong>{num(score(OPP,'fused_opp'))}</strong>，在未參與任何選擇的 fold 0 是 <strong>{num(score(FOLD0,'fused_opp_avg'))}</strong>，同一組 fold 0 題上官方 baseline 是 0.440。", "提交候選 fused_opp_diff 在 folds 1–4 CV 是 <strong>{num(score(ERR,'mlp_diff'))}</strong>，在 fold 0 採用檢查是 <strong>{num(score(DIFF0,F0D_KEY))}</strong>，同一組 fold 0 題上官方 baseline 是 0.440；再加讓子 5 維的候選版 CV {num(score(HD,'fused_opp_diff_hd'))}，還沒做 fold 0 檢查。", 1),
 ("OUT.mkdir(parents=True, exist_ok=True)\n(OUT / \"assets\").mkdir(exist_ok=True)\n", "OUT.mkdir(parents=True, exist_ok=True)\n(OUT / \"assets\").mkdir(exist_ok=True)\nfor a in HERE.joinpath(\"assets\").iterdir():\n    (OUT / \"assets\" / a.name).write_bytes(a.read_bytes())\n", 1),
 ("ROOT = Path(__file__).resolve().parents[2]\nOUT = Path(__file__).resolve().parent / \"site\"\n", "HERE = Path(__file__).resolve().parent\nROOT = HERE.parents[1]\nOUT = HERE / \"site\"\n", 1),
]
for a, b, n in R:
    c = s.count(a)
    assert c == n, (a[:40], c, n)
    s = s.replace(a, b)
for name, nxt in (("chart_auc", "def chart_perclass"), ("chart_timeline", "def rank_strip")):
    i = s.index(f"def {name}("); j = s.index(nxt); s = s[:i] + s[j:]
P.write_text(s)
print("patched", len(R))
