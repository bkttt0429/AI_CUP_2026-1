from pathlib import Path
P = Path("reports/go_report/build.py"); s = P.read_text()
R = [
# ch1 data card: shorter heading and bullets, old-pipeline history gone
("100 萬局、8,007 位玩家、每級 10 萬局；每局只有一方是標籤玩家，沒有勝負、用時、對手等級；v1.1 補了讓子數 HD[n]", "100 萬局、8,007 位玩家；每局只有一方有標籤，沒有勝負與對手等級", 1),
("每位玩家只屬一個等級，平均 125 局；對手不在資料集中，等級未知；game_id 在同人同色內是隨機排列，沒有時間序", "每位玩家只屬一個等級，平均 125 局；對手等級未知，對局沒有時間序", 1),
("SGF 只有雙方著手與棋盤大小，沒有 RE、用時、目數；v1.1 加了 HD[n]：HD0 互先 79.3%、HD1 讓先 17.1%、HD2–9 讓子 3.6%（36,421 局、3,474 人）", "v1.1 加了讓子數 HD[n]：互先 79.3%、讓先 17.1%、讓子 3.6%（36,421 局）", 1),
("讓子局的讓子石未記錄、著手 B/W 標記相反，color 欄是實際顏色（上手執白）；舊流程因此把 3.6% 的局在缺子棋盤上拿對手的手評分", "讓子局沒有記錄讓子石、B/W 標記相反，color 欄是實際顏色", 1),
('[["局數、玩家", "1,000,000 局、8,007 人；每級 100,000 局"], ["每局"', '[["每局"', 1),
("資料約定 HD0 6.5、HD≥1 0；舊流程一律輸入 7.5（第 2 章）", "資料約定 HD0 6.5、HD≥1 0；特徵一律輸入 7.5（第 2 章）", 1),
("讓子局的三個問題（缺子、B/W 相反、實際顏色）在 v1.1 之前無從得知；修正寫在 <code>aicup.sgf.setup_and_moves</code>，只重評那 3.6% 的局正在進行（第 8 章）。", "讓子局的修正寫在 <code>aicup.sgf.setup_and_moves</code>，只需重評那 3.6% 的局（第 8 章）。", 1),
("已用四次（r3_pma、fused、fused_opp、fused_opp_diff），再用要使用者本人同意", "已用四次，再用要使用者本人同意", 1),
# ch2 replay
("貼目：資料約定 HD0 6.5、HD≥1 0，但舊流程一律輸入 7.5；這是特徵抽取設定，訓練與推論一致，只改推論端會錯配", "貼目一律輸入 7.5：這是特徵抽取設定，訓練與推論一致，只改推論端會錯配", 1),
("全量兩側約 2.2 億局面，8 卡一節點約 8–9 小時；Human SL 18 profile 是另一遍、更貴", "全量兩側約 2.2 億局面，8 卡一節點約 8–9 小時", 1),
("HD≥2：補標準星位、B/W 對調、目標取 color；重評分進行中", "HD≥2：補標準星位、B/W 對調、目標取 color", 1),
("<p>讓子局在 v1.1 之前是在缺子棋盤上拿對手的手當目標評分；修正後只重評那 36,421 局（R-hdpatch），head 不變；head 一旦重訓，全量分數都要重算。</p>", "<p>讓子修正只重評那 36,421 局（R-hdpatch），head 不變；head 一旦重訓，全量分數都要重算。</p>", 1),
# ch3: ens_top3 is not on the current path
('("聚合器集成 ens_top3", score(STACK3, "ens_top3"), None), ', '', 1),
("W, H, L, R, top = 640, 342, 178, 60, 40\n", "W, H, L, R, top = 640, 302, 178, 60, 40\n", 1),
("七個里程碑的離線分數：官方 baseline 0.440，Human SL 0.6192，ens_top3 0.6327，fused", "六個里程碑的離線分數：官方 baseline 0.440，Human SL 0.6192，fused", 1),
('STACK3 = js("results/l1b/stack_check3.json")\n', '', 1),
('assert round(score(STACK3, "ens_top3"), 4) == 0.6327 and round(score(OPP, "fused"), 4) == 0.6545', 'assert round(score(OPP, "fused"), 4) == 0.6545', 1),
("退路依序為 fused_opp、fused、ens_top3、rank_model，共用同一組分片檔，不必重新評分", "退路 fused_opp 共用同一組分片檔，不必重新評分", 1),
("退路依序為 fused_opp、fused、ens_top3、rank_model，共用同一組分片檔", "退路 fused_opp 共用同一組分片檔", 1),
("；HD 5 維是待採用的第 60–64 維", "", 1),
("MLP h32、300 epoch、l2 1e-4、類別平衡；A、B 機率平均；argmax 期望得分", "MLP h32、類別平衡；A、B 機率平均；argmax 期望得分", 1),
("+0.17 是 fold 0 同題比較（0.6113 − 0.440），其餘是 CV；L1b 之後的三版 fold 0 略低於 CV（−0.001 到 −0.003），hsl 低 0.008，都與折間變異相容；fold 0 已用四次，再用要使用者同意", "+0.17 是 fold 0 同題比較，其餘是 CV；L1b 之後的三版 fold 0 只比 CV 低 0.001–0.003", 1),
# ch4
("<p>Pilot 從 fold 0 與 fold 1 各取 200 人（每級 20 人）× 20 局，每手算 18 個 profile。Python 直讀權重與 KataGo 引擎對照過：20 個局面的目標落子最大差 3.8×10⁻⁴，argmax 全部一致。用 fold 1 擬合單調門檻再測 fold 0，分數升到 n=5 0.586、n=10 0.601、n=20 0.647。</p>", "<p>Pilot 從 fold 0 與 fold 1 各取 200 人（每級 20 人）× 20 局，每手算 18 個 profile。</p>", 1),
("\n<p>每手只存實際落子那一格的 log p：目標手 18 個 profile 與 10 個 head、對手手 10 個 head。重評分約 900 局面／秒／卡；全量兩側約 2.2 億局面，8 卡一節點約 8–9 小時。</p>", "", 1),
# ch5
("單一聚合器的增益上限約 0.01，三者平均最多約 +0.014；", "單一聚合器的增益上限約 0.01，三者平均 +0.011；", 1),
("所有結構都落在 0.60–0.63：整題監督的集合模型比 MLP 多 +0.005 到 +0.009，單一結構上限約 0.01", "所有結構都落在 0.60–0.63：集合模型只比 MLP 多 +0.005 到 +0.009", 1),
("三者機率平均 ens_top3 0.6306（逐 seed）；三個 seed 全部再平均是 0.6327", "三者機率平均 ens_top3 0.6306", 1),
('"fold 0：r3_pma 0.6233 [0.6093, 0.6371]，對當時提交 +0.0102", ', '', 1),
("集合模型都是 3 個 seed 分數的平均，ens_top3 也是", "集合模型都是 3 個 seed 分數的平均", 1),
(', "v1.1 的讓子局：讓子石未記錄、B/W 相反，36,421 局（3.6%）目前是在缺子棋盤上拿對手的手評分，重評中"]', ']', 1),
# ch6
("圖上 15 列（含一條純雜訊對照）只有對手、差值、讓子三步過門檻；最差的是 TSFEL 全塊（−0.061）與 CORAL（−0.026）", "圖上 15 列只有對手、差值、讓子三步過門檻；最差的是 TSFEL 全塊與 CORAL", 1),
("綠色是過門檻的三步（HD 還沒進提交）。TSFEL 64／全塊超出軸範圍，L2、softmax、提早停止只列在左欄，未畫", "綠色是過門檻的三步；其中一列是純雜訊對照。TSFEL 64／全塊超出軸範圍，未畫", 1),
("多數是少局題上的抽樣雜訊，不是模型脆弱：16 欄純雜訊只掉 −0.0014，TSFEL 16 欄掉 −0.0128", "多半是特徵本身的雜訊：16 欄純雜訊只掉 −0.0014，TSFEL 16 欄掉 −0.0128", 1),
# ch7
("部分是：同 n=5 下 TSFEL 對已見玩家比新玩家少掉 {signed(D5['base_tsfel16']['D_plain'])}，CI 不含 0；但對已見玩家也仍是負增益", "部分是：TSFEL 對已見玩家少掉 {signed(D5['base_tsfel16']['D_plain'])}，但仍是負增益", 1),
# ch8: review history and status notes
("與 GPT 三輪、Gemini 一輪審查後收斂的單一版本：表示 → 聚合 → 分類三層", "架構分表示 → 聚合 → 分類三層", 1),
("\n<p>刪掉的指示：全域玩家 ID 對抗、自監督為主、等級 CE 權重 0.1、「時序成功才測新訊號」、樹為唯一主比較、每層指紋否決、「相關 < 0.3 就刪」的硬門檻、文件內的時程。</p>", "", 1),
("R-hdpatch：維持 7.5、只重評 HD≥2 的 36,421 局，是 11/04 前做得完的過渡版；R-full 要重訓 head 並全量重評，時點由使用者決定", "R-hdpatch 只重評 HD≥2 的 36,421 局，是過渡版；R-full 要重訓 head 並全量重評", 1),
('[["1", "步 3 重評 HD≥2（五支 GPU 工作）— 進行中（11/02：四支 L1b 已完成，Human SL 在跑）"], ["2", "fold 0 第五次使用：擬合 HD 版提交文件（GPT 建議暫緩）"], ["3", "強網權重是否下載；Q 支線沒有它無法開始"], ["4", "貼目：維持 7.5 一致版到 11/04，或列入全部重建"], ["5", "純 CPU', '[["1", "fold 0 第五次使用：擬合 HD 版提交文件"], ["2", "強網權重是否下載；Q 支線沒有它無法開始"], ["3", "貼目：維持 7.5 一致版到 11/04，或列入全部重建"], ["4", "純 CPU', 1),
('"磁碟：五個評分目錄共 2.6 GB，補丁併回再加一份；home 剩 15 GB"', '"R-hdpatch 重評後不劣（下降 ≤ 0.002）才重擬合文件"', 1),
("貼目：舊流程對所有局輸入 7.5（資料約定 HD0 6.5、HD≥1 0），訓練與推論一致；敏感度抽樣要一路量到十類機率與決策，不用「每手 ±0.001」當判準", "貼目維持 7.5（訓練與推論一致）；要改先抽樣量到十類機率與決策", 1),
# ch9
('meta="main ef10885 · 提交文件 rank_model_fused_opp_diff.json"', 'meta="提交文件 rank_model_fused_opp_diff.json"', 1),
('"單卡與 4 卡分片也 40／40 相同；單卡 800 題約 2–3 小時", "權重 tar 不進 git，sha256 記錄在 release/；必要時上傳成 GitHub Release 資產"', '"單卡與 4 卡分片也 40／40 相同"', 1),
('[["pytest（PR #40 head）", "344 passed、3 skipped"], ["fold 0 重現"', '[["fold 0 重現"', 1),
("R-hdpatch 是過渡版（head 不變）；重評後不劣（下降 ≤ 0.002）才重擬合文件，這要再用一次 fold 0。11/04 用哪個文件由使用者決定，採用過渡版時要標明；", "R-hdpatch 是過渡版（head 不變），重擬合文件要再用一次 fold 0；11/04 用哪個文件由使用者決定；", 1),
# index
("<h3>已完成的章節</h3>", "<h3>章節</h3>", 1),
# fit guard
('''  paint(6);
})();
"""''', '''  paint(6);
})();
/* Present mode: shrink a card's text column until it clears the row below, whatever font the viewer has. */
(function () {
  function fit() {
    document.querySelectorAll('.report>div:first-child').forEach(e => { e.style.fontSize = ''; });
    const left = document.querySelector('.page.card.cur .report>div:first-child');
    if (!left || !document.body.classList.contains('m-present')) return;
    for (let s = 97; s >= 76 && left.scrollHeight > left.clientHeight + 1; s -= 3) left.style.fontSize = s + '%';
  }
  const watch = new MutationObserver(fit);
  watch.observe(document.querySelector('main'), { subtree: true, attributes: true, attributeFilter: ['class'] });
  watch.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  window.addEventListener('resize', fit);
  document.fonts?.ready.then(fit);
  fit();
})();
"""''', 1),
]
for a, b, n in R:
    c = s.count(a); assert c == n, (a[:50], c, n); s = s.replace(a, b)
P.write_text(s)
Q = Path("reports/go_report/shot.py"); q = Q.read_text()
R2 = [
('''  if (left && cause && hit(R(left), R(cause))) out.push('text overlaps cause');\n''', '''  if (left && cause && hit(R(left), R(cause))) out.push('text overlaps cause');\n  if (left && left.scrollHeight > left.clientHeight + 1) out.push('text overflows its column');\n''', 1),
("  return { id: page.id, issues: [...new Set(out)] };", "  return { id: page.id, issues: [...new Set(out)], fit: left ? left.style.fontSize : '' };", 1),
('''with sync_playwright() as p:
    b = p.chromium.launch()
    res = {}
    ctx = b.new_page(viewport={"width": 1366, "height": 768})
    ctx.goto((SITE / "talk.html").as_uri())
    ctx.wait_for_timeout(600)
''', '''# The stack asks for "Noto Sans TC"; give the page that face so wrapping matches a viewer who has it.
FONT = """@font-face{font-family:"Noto Sans TC";font-weight:400;src:local("Noto Sans CJK TC")}
@font-face{font-family:"Noto Sans TC";font-weight:700;src:local("Noto Sans CJK TC Bold")}"""
def walk(pg, tag, res, shots=None):
    pg.goto((SITE / "talk.html").as_uri())
    pg.add_style_tag(content=FONT)
    pg.evaluate("""() => Promise.all(['400 20px "Noto Sans TC"', '700 20px "Noto Sans TC"'].map(f => document.fonts.load(f)))""")
    pg.evaluate("window.dispatchEvent(new Event('resize'))")
    pg.wait_for_timeout(600)
    n = pg.evaluate("document.querySelectorAll('.page').length")
    for i in range(n):
        r = pg.evaluate(PROBE)
        res[f"{tag} {i+1:02d} {r['id']}"] = r["issues"]
        if r["fit"]:
            print(tag, i + 1, r["id"], "fit", r["fit"])
        if shots is not None:
            f = OUT / f"talk{i+1:02d}.png"
            pg.screenshot(path=str(f)); shots.append(f)
        pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(120)
with sync_playwright() as p:
    b = p.chromium.launch()
    res = {}
    shots = []
    walk(b.new_page(viewport={"width": 1366, "height": 768}), "1366", res, shots)
    walk(b.new_page(viewport={"width": 1757, "height": 855}), "1757", res)
''', 1),
('''    n = ctx.evaluate("document.querySelectorAll('.page').length")
    shots = []
    for i in range(n):
        r = ctx.evaluate(PROBE)
        res[f"{i+1:02d} {r['id']}"] = r["issues"]
        f = OUT / f"talk{i+1:02d}.png"
        ctx.screenshot(path=str(f)); shots.append(f)
        ctx.keyboard.press("ArrowRight"); ctx.wait_for_timeout(120)
''', '', 1),
]
for a, b, n in R2:
    c = q.count(a); assert c == n, (a[:50], c, n); q = q.replace(a, b)
Q.write_text(q)
print("ok")
