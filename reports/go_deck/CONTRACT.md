# 收尾簡報寫作規約（給寫作 agent）

這份簡報是 AI CUP 2026 圍棋棋力預測的收尾報告，**只有簡報模式**，一次一頁。全份 15 張，分給 5 個寫作 agent 並行寫，每人只寫自己擁有的檔案。
完整計畫（每張卡的內容要點）在 `/home/u2507115/.claude/plans/https-claude-ai-artifact-jejxazsvtftgqj-transient-naur.md` 的「敘事主線」表。

## 1. 聽眾與寫作規則（最重要）

- 聽眾：**沒看過這個專案的人**（評審、同學）。懂一點機器學習，但不懂圍棋 AI，也不知道我們的內部代號。
- 全部用**繁體中文**。句子短，一句一個意思。
- **工具不能只寫名字。** KataGo、Human SL、L1b、trunk、錨點、帳本、護欄……第一次出現時要說「它是什麼、為什麼用它、它給了我們什麼、量到多少」。例：不要寫「用 Human SL 特徵」，要寫「KataGo 另訓了一個模仿人類的模型：給局面和棋力設定，預測那個程度的人會下哪裡；我們拿它來問『這一手像幾級的人下的』」。
- **每個設計都要回答「為什麼」**，基於什麼觀察或量測。用 `dl.why` 框寫成「問題 → 作法 → 結果」。
- 內部代號（ct116、M183、fold 0、oh、S6at、SQ4…）盡量不出現；非出現不可時（例如最終檔名 ct116）要說明它是什麼。
- **數字一律要有出處**：只用來源檔裡查得到的數字（見第 6 節），查不到的句子刪掉，並在回報裡列出。不要四捨五入到改變意思；Public 分數寫 7 位或 4 位小數都可，同一張卡內一致。
- 不說「保證」。下限、預估都說「證據」「預估區間」。
- 故事線：比賽 → 兩個難點（① 相鄰等級分不開、② 標籤和棋譜實力不一致）→ 每手棋變證據 → 模型 → 認出對手破局 → 守住 Private → 收穫。**第 4–13 張每張左欄最上方放 `.solves` 標籤，標明這張解決哪個難點**（見第 3 節）。

## 2. 檔案擁有權

| agent | 只能寫這些檔（都在 `reports/go_deck/parts/`） |
|---|---|
| W1 | `c01_intro.html`、`a01_intro.js`、`講稿_01.md` |
| W2 | `c02_evidence.html`、`a02_evidence.js`、`講稿_02.md` |
| W3 | `c03_model.html`、`a03_model.js`、`講稿_03.md` |
| W4 | `c04_anchor.html`、`a04_anchor.js`、`講稿_04.md` |
| W5 | `c05_private.html`、`a05_private.js`、`講稿_05.md` |

**不要改**：`00_head.html`、`01_engine_head.js`、`98_engine_tail.js`、`99_foot.html`、`10_num.js`、`a90_terms.js`、`assemble.py`、`deck_check.py`、`check.sh`，以及 `reports/go_story/` 底下任何檔案（那是舊版，另一個 session 也在用）。
需要額外 CSS：在自己的卡片檔裡放一個 `<style>`，所有選擇器都以自己的卡片 id 開頭（`#card-hsl .xxx{…}`）。需要改共用檔或新增名詞提示：寫在回報裡，由主持人改。

## 3. 卡片 HTML 骨架

一般卡（左文右圖）：
```html
<section class="page card" id="card-hsl" data-ch="3" data-chtitle="每手棋變證據" data-title="KataGo 的人類棋風模型">
  <div class="report">
    <div class="txt">
      <div class="num">第 3 章 · 每手棋變證據</div>
      <div class="solves"><span class="d1">解決 ① 相鄰等級分不開</span></div>
      <h3>標題寫成一句主張</h3>
      <p class="lede">一句話結論（聽眾只記得這句也夠）。</p>
      <ul class="key"><li>要點 1</li><li>要點 2</li><li>要點 3</li></ul>
      <dl class="why"><dt>問題</dt><dd>…</dd><dt>作法</dt><dd>…</dd><dt>結果</dt><dd>…</dd></dl>
    </div>
    <div class="fig">
      <div class="anim" data-anim="ch3_hsl"></div>
      <div class="bars" id="ch3-bars"></div>   <!-- 選用 -->
    </div>
  </div>
</section>
```
- `data-chtitle` 只放在**該章第一張**卡（用來產生章節選單）；`data-ch` 每張都要。章號：1 比賽在做什麼、2 難在哪裡、3 每手棋變證據、4 模型與訓練、5 破局：認出對手、6 從 Public 到 Private、7 收穫。
- `.solves`：`<span class="d1">解決 ① 相鄰等級分不開</span>`、`<span class="d2">解決 ② 標籤和實力不一致</span>`，兩個都適用就放兩個；不直接解決難點的卡（例如 gate、帳本）用 `<span class="d0">守住成果</span>` 之類。
- 版面變體：`.report.even`（左右 1:1）、`.report.stack`（上下排；動畫用 `<div class="anim span" data-anim="…">` 放在 `.report` 的直接子層橫跨整張；要點可用 `ul.key.cols3` 三欄）。
- 其他可用元件：`.formula`（含 `.eq` 與 `dl`）、`.cause`（`<span class="k">問題</span>…<span class="arrow">→</span><span class="k go">作法</span>…`）、`table`（包在 `.table-wrap` 裡）、`.chips>.chip`、`.tag.t-m`（實測）`.t-i`（推論）`.t-p`（用到 Public 資訊）。
- 封面：`<section class="page cover" id="cover" data-title="封面">`，內含 `.cover-kicker`、`h2`、`.cover-sub`、`.kpis>.kpi>(.v,.l)`。結語：`<section class="page after" id="after" data-ch="7" data-chtitle="收穫" data-title="…">`，內含 `h2`、`.cols`、`ul`。
- **不要**用 `.deriv-open`、`section.lesson`、`data-sec`（推導層已拿掉）。
- 每頁文字量：左欄的導語＋要點＋why 框合計約 **220 個中文字以內**。超過時引擎會自動縮小字級，檢查會在字級低於 13px 時報錯；那就刪字或拆到講稿。
- id 一律用自己章的前綴：卡片 `card-…`（不和別章重複），圖表 `chN-…`，動畫名 `chN_…`（N 是你檔名的數字 01–05）。

## 4. 動畫 API（引擎沿用 go_story）

在 `a0N_*.js` 裡（所有 a*.js 在同一個函式裡執行，看得到 `NUM`、`ANIM`、`CHARTS` 與下列輔助函式）：
```js
  // 本章數字（每筆註明出處）
  NUM.ch3 = { profiles: [...], /* 出處：素材06 §… */ };
  ANIM.ch3_hsl = () => {           // 輔助函式要在函式裡呼叫（引擎呼叫前才設定 AD/HOLD）
    let s = '';
    s += tx(20, 30, 'an-t', '標題字');                       // 固定文字
    s += txA(20, 60, 'an-b', '第 2 秒淡入', 2);              // 淡入文字：txA(x, y, class, 文字, t0, t2?, extra?)
    s += `<rect x="20" y="80" width="0" height="16" fill="#0b6e69">${aTo('width', 0, 300, 3)}</rect>`;  // 屬性動畫
    s += `<line x1="20" y1="120" x2="400" y2="120" stroke="#d1343a" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(4, 0.6)}</line>`; // 畫線
    return {steps: ['① 先…', '② 再…', '③ 最後…'],      // 動畫上方的步驟說明（選用）
            svg: `<svg viewBox="0 0 640 300" role="img" aria-label="一句話描述">${s}</svg>`,
            note: '動畫下方一行小字（出處或口徑）'};
  };
  ANIM.ch3_hsl.dur = 10;           // 一輪秒數（預設 9）
  CHARTS['ch3-bars'] = () => barRows([[標籤, 值, 顯示文字, 顏色], ...], lo, hi);
```
- 時間函式：`aShow(t0, t2)` 淡入淡出、`aDraw(t0, d, t2)` 畫線、`aTo(attr, v0, v1, t0, d, t2)` 屬性變化、`aSeq(attr, [[t, v], …], discrete)` 自訂時間表（第一個 t 是 0、最後一個是 `AD`）。`kt()` 會把 HOLD 之前的時間壓縮成 0.75 倍並提前 0.3 秒，所以設計時間可以寫得寬鬆。
- 其他輔助：`tx`、`txA`、`barRows`、`f3`、`f4`。
- SVG 文字 class（字級固定，SVG 會隨寬度縮放）：`an-k`（13 灰）、`an-n`（13 粗黑）、`an-t`（14 粗黑）、`an-s`（11.5 灰）、`an-b`（13 粗綠）、`an-h`（13 粗橘）、`an-v`（15 粗）、`an-ok`（綠）、`an-no`（紅）、`an-w`（12 粗白）、`an-l`（12 白字描黑邊）。
- 右欄動畫 `viewBox="0 0 640 H"`，H 建議 240–340；橫跨整張（`.anim.span`）用 `960×H`、H ≤ 300。文字要留在 viewBox 內，中文字寬約 1 個字級、英數約 0.58 個字級。
- 配色（SVG 裡直接寫色碼）：墨 `#1d1f24`、灰字 `#5d6370`、線 `#d9dbe0`、格線 `#e3e6ea`、主色 `#0b6e69`、主色淡 `#e3f1ef`、橘 `#a4451c`、橘淡 `#fbeee6`、紅 `#d1343a`、藍 `#0072b2`、金 `#b98900`、難點①藍底 `#e1eefa`、難點②金底 `#fbf0d2`。每個形狀都要寫 `fill`。
- 簡報是淺色單一主題；不要用 `prefers-color-scheme`。
- **沿用舊動畫**：舊版在 `reports/go_story/parts/` —— `a01_task.js`（task、chart-players）、`a02_hard.js`（accum、bias、rp、chart-byn）、`a03_journey.js`（climb、uploads）、`a04_arch.js`（pipeline、profiles、opp、decide）、`a05_rules.js`（gate、forest）、`a06_wall.js`（wall、ident）、`a07_final.js`（probes、anchor、ladder、last、chart-final）。複製函式到自己的檔、改名成 `chN_…`、改用新數字。舊動畫讀的 `NUM.xxx`（例如 `NUM.probes`、`NUM.forest`）在新 `10_num.js` 裡**不存在**，要搬到自己的 `NUM.chN` 並重新核對出處。舊卡片文字在 `reports/go_story/parts/c0*.html` 可參考，但數字停在 11-11（ct44 時代），不要照抄。
- 共用數字在 `parts/10_num.js`：`NUM.kpi`、`NUM.levels`、`NUM.span`、`NUM.players`、`NUM.partial`、`NUM.test`、`NUM.baseline`、`NUM.climb`（CV 里程碑）、`NUM.uploads`（Public 路徑）、`NUM.final`、`NUM.live`、`NUM.fallbacks`、`NUM.rank`（名次，目前 null）。

## 5. 講稿 `講稿_0N.md`

每張卡一段：
```md
## 第 4 頁 · KataGo 的人類棋風模型
- 2–4 句口語講法，含右欄動畫怎麼講。
- 卡片上放不下的補充數字可以放這裡（一樣要有出處）。
```
頁碼依全份順序：封面 1、任務 2、相鄰等級 3、標籤 4、Human SL 5、刻度與內部向量 6、對手 7、架構 8、訓練 9、認出對手 10、錨點合併 11、gate 12、帳本 13、Private 14、收穫 15。

## 6. 數字來源（核對用；只讀）

repo：`/home/u2507115/BK_AICUP_2026-1`
- `reports/報告素材_2026-11-06.md`（素材06）：任務、資料、ct7 系統、§8 誤差結構與資訊牆、§6 里程碑、§7 實驗表。
- `reports/報告素材_2026-11-15_補充.md`（素材15）：A 狀態、B 里程碑表、C ct106 資料流、D 按活躍度認人、E 揭露、F 失敗表、G 瓶頸與護欄、H 組成探測與 ct116、I/J 11-17～18。
- `reports/architecture_v2_2026-11-02.md`（設計文件，以 M 號找條目，約 2,000 多行）：M37 採用規則、M54 帳本方程、M161 抽樣、M183 瓶頸、M201 兩半相似、M203 組成、M209 人工決定、M214 帶映射、M224 認人上限。
- `reports/問題陳述_認人瓶頸_2026-11-16.md`（認人瓶頸）、`reports/測試位移與排行榜探測方法.md`、`reports/上傳排程_2026-11-08.md`。
- `METHOD_NOTES.md`（11-02 以前的特徵與實驗）、`REPORT.md`（資料與規則）。
- `results/trunk_seq/es/public_composition.json`（Public 各級題數）。
用 grep 找數字，例如 `grep -n "0.948" reports/報告素材_2026-11-06.md`。

## 7. 檢查（必做，在計算節點跑；登入節點只能編輯、送工作、讀檔）

```bash
cd /home/u2507115/BK_AICUP_2026-1
sbatch --wait -J deckchk0N -t 00:15:00 -c 4 --exclude=25a-hgpn[001,003,004,007,062,009,039,006,008,010,155,083,146,171,172,174,145,147,159,161,166,167,177,069,082] \
  scripts/nano4/run.sbatch bash reports/go_deck/check.sh --only 0N
tail -40 $(ls -t logs/nano4/deckchk0N-*.out | head -1)
```
（Bash 的 timeout 設 900000。`--wait` 會等工作結束。）
- 結果：`reports/go_deck/shots/only_0N/report.json`，每頁截圖 `1366_*.png`、`1757_*.png`（動畫停在最後一格）、`sheet*.png`。**用 Read 打開 PNG 親眼看**：字有沒有擠、動畫是否被截、版面是否平衡。
- 要修到：沒有 console 錯誤、沒有 missing_anims、每頁 0 issues（字級 ≥ 13px、不超出螢幕、SVG 文字不重疊不出界）。
- 不要在登入節點跑 python / node。

## 8. 回報格式（完成時的最後訊息）

1. 寫了哪些卡（id、標題、用了哪個動畫，沿用或新寫）。
2. 數字清單：每個出現在卡片上的數字 → 出處（檔名 + 段落或 M 號 / 行號）。
3. 因為查不到出處而刪掉或改寫的句子。
4. 最後一次檢查的工作編號與結果（issues 數），以及你看了哪幾張截圖。
5. 需要主持人處理的事（共用 CSS、名詞提示新增、跨章的不一致）。
