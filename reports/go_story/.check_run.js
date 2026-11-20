
const window = {matchMedia: () => ({matches: false})};
const document = {querySelectorAll: () => [], getElementById: () => null, body: {classList: {add(){}, remove(){}}}};
/* ===== 數字只改這裡：圖表與動畫都從這裡讀；每一筆後面註明出處（設計文件 = reports/architecture_v2_2026-11-02.md） ===== */
window.NUM = {
  date: '2026-11-11',
  // 訓練集各級玩家數（每級 10 萬局）：REPORT.md §3.4（掃描 10 個訓練 CSV 的結果，第 140–151 行）
  players: [['D', 890], ['C', 817], ['B', 1048], ['A', 1582], ['1D', 433], ['2D', 610], ['3D', 618], ['4D', 640], ['5D', 658], ['6D', 711]],
  levels: ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D'],
  // production 留出題（794 人、4,000 題）上 ct3 集成按 n 的分數：results/r_variants/prod_heldout.json ct3.by_n
  gateByN: [[5, 0.6579], [10, 0.7253], [20, 0.7952]],
  // 離線里程碑：[名稱, 離線分數, Public 或 null, 備註, 口徑]；口徑 f0 ＝ fold 0（官方 baseline 在本協議下重訓、期望分決策）、
  //   mlp ＝ 單一 MLP 的 CV、ens ＝ 6 模型集成（3C＋3T）的 CV；CV 都是 folds 1–4、類別平衡。口徑不同的相鄰兩點，圖上只用灰色虛線相接
  // 出處：METHOD_NOTES.md 第 29 行（baseline 0.440）、第 195 行（hsl 0.6192、fused 0.6545）、第 494 行（_r 0.6929、_hd 0.7026）；
  //   results/baseline_official_eval.json；設計文件 §2、§3.1、第 180 行（同題 MLP 0.7026）、§4H 第 351 行（3C＋3T 0.7241、12 模型 0.7258）、
  //   §4J′ 第 421 行、§4M M8（第 591 行）、M14／M22（第 710、808 行）
  climb: [
    ['baseline', 0.440, 0.4631, '官方 baseline；fold 0', 'f0'],
    ['Human SL', 0.6192, null, '18 個 profile 的 log p＋MLP', 'mlp'],
    ['＋L1b', 0.6545, null, '鑑別式等級頭', 'mlp'],
    ['＋對手', 0.6735, null, '對手著手的 L1b 16 維', 'mlp'],
    ['＋差值', 0.6929, 0.7148, '差值 3 維、讓子修正（主檔）', 'mlp'],
    ['＋HD', 0.7026, null, '讓子 HD 5 維（64 維 R，MLP）', 'mlp'],
    ['C＋T', 0.7241, 0.7494, 'trunk token、6 模型集成（Public 檔是 12 模型，CV 0.7258）', 'ens'],
    ['＋40 局', 0.7288, 0.7625, '每人 40 局（CV 不含 fold 0；Public 為 ct3）', 'ens'],
    ['對手 HSL', 0.7398, 0.7687835, '24 個 CV 驗證成員（ct7）', 'ens'],
    // 11-11 加：設計文件 M33 結果（6 對 6 集成 0.7398 → 0.7458；ct9 的 Public 0.7622032）；M71 第 2 步（底 ＝ 標準 6 成員 seed 0–2，0.7458）
    //   與 M82 結果（五成分＋比對的錨點對底 +0.0193）。0.7651 ＝ 0.7458 ＋ 0.0193，是兩個實測相加，口徑另記為 anc；Public 是 ct44（另含兩個推移）
    ['＋對比', 0.7458, 0.7622032, '玩家對比損失（M33；Public 為 ct9）', 'ens'],
    ['＋錨點', 0.7651, 0.8067107, '對手錨點（M82；Public 為 ct44，另含兩個推移）', 'anc'],
  ],
  // Public 上傳紀錄（使用者回報的分數；設計文件 §4J′ 第 421 行、§4M 第 581、595、738、812 行）：[檔, 日期, Public, 內容]
  //   ct4 的 Public 與上傳順序（ct3 → ct4 → ct6）：使用者 11-05 18:52 貼的上傳清單與當時的檔案比對（ct4 與 ct7 在 Public 差 47 題）；設計文件未記錄
  uploads: [
    ['主檔', '11-04', 0.7148, 'fused_opp_diff_r：59 維題目特徵＋MLP'],
    ['C＋T', '11-04', 0.7494, 'trunk token、C／T 各 6 個'],
    ['ct2', '11-04', 0.7415, '按局數路由（未採用）'],
    ['ct3', '11-04', 0.7625, '＋fold 0、每人 40 局'],
    ['ct4', '11-05', 0.7687835, '＋第二組 L1b head 的成員'],
    ['ct6', '11-05', 0.7529805, '對手 HSL，12 個全資料模型'],
    ['ct7', '11-05', 0.7687835, '對手 HSL，24 個 CV 驗證過的成員'],
    // 11-06 之後（設計文件 M33 的 Public 回報、M52 的 ct13、M58 第二天、M71 的 ct40 讀數、「11/11 Public 讀數與規則判定」；上傳排程修訂 7–9）：
    //   只列模型檔與被保留的檔；另有 ct10、ct12 與 14 個被丟棄的探測檔沒有列在這裡（探測見 probes）
    ['ct9', '11-06', 0.7622032, '同配方＋玩家對比損失'],
    ['ct13', '11-07', 0.7748456, 'ct9 ＋ 1D→A 推移'],
    ['ct28', '11-09', 0.7827471, 'ct13 ＋ 4D→5D 推移'],
    ['ct40', '11-10', 0.7761668, '錨點在推移之前（組合有瑕疵）'],
    ['ct42', '11-11', 0.7880062, 'ct28 鎖定＋五成分錨點'],
    ['ct44', '11-11', 0.8067107, 'ct28 鎖定＋五成分與比對的錨點（最終檔）'],
  ],
  final: 'ct44',
  // 決策推移的 Public 探測（設計文件 M52 的 ct13、M53 階梯、M58 第 1–3 天；上傳排程表的翻動題數）：
  //   [方向, 檔, Public 翻動題數, X（一級淨對錯）, Y（兩級淨對錯）, 解碼, 處置]；X、Y 由 400 × ΔPublic ＝ 0.63212 X ＋ 0.36788 Y 唯一解出
  //   （基準：ct13 對 ct9，其餘對 ct13 的 0.7748456）；處置 keep 保留、keep2 未達事先門檻但由使用者決定保留、rule 依事先登記的 ct25 規則刪除、drop 丟棄
  probes: [
    ['1D→A', 'ct13', 8, 8, 0, '8 對 0 錯', 'keep'],
    ['B→A', 'ct14', 13, -1, 0, '6 對 7 錯', 'drop'],
    ['5D→6D', 'ct15', 7, -4, -1, '1 對 5 錯、1 題差兩級', 'drop'],
    ['2D→1D', 'ct16', 12, -1, 1, '5 對 6 錯、1 題真值 A', 'drop'],
    ['3D→2D', 'ct17', 8, -5, 0, '1 對 6 錯、其他 1', 'drop'],
    ['1D→2D', 'ct26', 15, 1, -3, '一級淨 +1、3 題真值 A', 'drop'],
    ['1D→A 第二帶', 'ct29', 4, -2, 0, '1 對 3 錯', 'drop'],
    ['A→1D', 'ct32', 9, -4, 0, '改錯比改對多 4 題', 'drop'],
    ['3D→4D', 'ct27', 12, -2, 0, '5 對 7 錯', 'drop'],
    ['4D→5D', 'ct28', 9, 5, 0, '7 對 2 錯', 'keep2'],
    ['5D→4D', 'ct19', 8, -5, -1, '1 對 6 錯、其他 1', 'drop'],
    ['B→C', 'ct22', 9, 3, 0, '6 對 3 錯', 'rule'],
    ['C→D', 'ct23', 9, 0, -1, '4 對 4 錯、其他 1', 'drop'],
    ['C→B', 'ct25', 8, -1, 1, '3 對 4 錯、其他 1', 'drop'],
    ['A→B', 'ct21', 6, 0, 0, '3 對 3 錯', 'drop'],
    ['4D→3D', 'ct18', 6, -3, 1, '1 對 4 錯、其他 1', 'drop'],
  ],
  // 對手錨點（設計文件 M71 第 2 步的校正表：互先局、開局模型；M71 測試端、M73、M82 的 CV；ct40／ct42／ct44 的 Public 讀數）
  anchor: {
    // 九個帶 ＝ 冒充者比對分數的 q50／80／90／95／98／99／99.5／99.9 切出來的區間；a ＝ 該帶的比對為真的機率
    bands: ['≤50', '–80', '–90', '–95', '–98', '–99', '–99.5', '–99.9', '>99.9'],
    a: [0, 0, 0.07, 0.19, 0.36, 0.53, 0.58, 0.72, 0.84],
    kp0: [1.6, 3.1, 1.6],   // K／P0 在 d ＝ −1、0、+1
    // 錨點加在 6 成員集成上的 CV 配對增益：[認對手的方法, 出處, seed 0–2, seed 6–8]
    cv: [['只用開局', 'M71', 0.0078, 0.0084], ['五成分', 'M73', 0.0125, 0.0133], ['五成分＋比對', 'M82', 0.0193, 0.0204]],
    pub: [['ct28', 0.7827471, '兩個推移'], ['ct40', 0.7761668, '錨點在前'], ['ct42', 0.7880062, '五成分'], ['ct44', 0.8067107, '＋比對']],
  },
  // production gate：集成增益曲線（k 個成員），results/r_variants/gate_diag.json 的 ct3_diversity／ct6_diversity.gain_curve；設計文件 §4M gate 診斷（第 801 行 k ＝ 4 的抄寫有誤，以 JSON 為準）
  gate: {k: [1, 2, 4, 6, 12], ct3: [0.710, 0.718, 0.722, 0.726, 0.726], ct6: [0.723, 0.731, 0.732, 0.731, 0.731], weakT: [0.709, 0.715, 0.720]},
  // 比較（多數事先登記；ct4＋ct7 疊加是使用者臨時提問）：[名稱, 差, CI 下, CI 上, 結果]
  //   結果：pass 通過並採用、adopt 次門檻但依事先的納入規則採用、sub 次門檻、fail 未過、ref 對照；設計文件 §3.1、§4M（M4 第 568 行、M8 第 591 行）
  forest: [
    ['對手 L1b 16 維（fold 0）', 0.0181, 0.0094, 0.0266, 'pass'],
    ['對手 − 本人差值 3 維', 0.0181, 0.0140, 0.0220, 'pass'],
    ['讓子 HD 5 維', 0.0071, 0.0034, 0.0107, 'pass'],
    ['對手 HSL 通道（M14）', 0.0096, 0.0062, 0.0130, 'pass'],
    ['ct7 型成員（M22）', 0.0110, 0.0073, 0.0148, 'pass'],
    // 11-11 加（11-06 之後通過的三項）：設計文件 M33 結果、M73 結果（seed 0–2）、M82 主檢定（seed 0–2）
    ['玩家對比損失（M33，ct9）', 0.0060, 0.0036, 0.0085, 'pass'],
    ['對手錨點：五成分對底（M73）', 0.0125, 0.0092, 0.0159, 'pass'],
    ['錨點：比對 對 五成分（M82）', 0.0058, 0.0027, 0.0087, 'pass'],
    ['每人 40 局（M8，依納入規則採用）', 0.0047, 0.0016, 0.0077, 'adopt'],
    ['按局數路由（M7）', 0.0039, 0.0021, 0.0057, 'sub'],
    ['日本規則 Human SL（M27，集成）', 0.0014, -0.0007, 0.0036, 'fail'],
    ['讓子方向（M15）', 0.0011, -0.0013, 0.0034, 'fail'],
    ['勝負代理（winlm）', 0.0005, -0.0020, 0.0030, 'fail'],
    ['純雜訊 16 維', -0.0014, -0.0042, 0.0012, 'ref'],
    ['ct4＋ct7 疊加（事後提問）', -0.0012, -0.0041, 0.0017, 'fail'],
    ['微調 trunk 當 token（M3）', -0.0016, -0.0050, 0.0018, 'fail'],
    ['全局注意力 S 臂（M21b）', -0.0024, -0.0092, 0.0046, 'fail'],
    ['區帶專家兩階段', -0.0076, -0.0111, -0.0040, 'fail'],
    ['注意力池化 PMA（M4）', -0.0088, -0.0127, -0.0050, 'fail'],
    ['TSFEL PCA-16', -0.0128, -0.0167, -0.0090, 'fail'],
    ['CORAL 有序輸出', -0.0263, -0.0309, -0.0217, 'fail'],
  ],
  // ct7 對 pub_q_0177 的十級機率（submission_rank_ct7_proba.csv）：最可能 2D，期望分決策選 3D
  decide: {qid: 'pub_q_0177', p: [0.001, 0.003, 0.018, 0.025, 0.053, 0.240, 0.227, 0.164, 0.143, 0.126]},
  // Public 排行榜（使用者 11-06 傳的 AIdea 截圖，前 6 名逐列抄錄；第 7 名被裁切）：[名次, 隊伍, 成績, 上傳時間, 次數]；我們 ＝ TEAM_11819
  //   11-11 起頁面不再畫這張榜（當時的最終檔是 ct7；之後的名次沒有記錄）；數字只留作 11-06 的紀錄，7.4 節的表是同一份
  board: {us: 'TEAM_11819', rows: [[1, 'TEAM_11819', 0.7687835, '2026-11-05 18:47:08', 7], [2, 'TEAM_11724', 0.7574623, '2026-11-04 17:34:42', 1],
    [3, 'TEAM_11677', 0.7479805, '2026-11-05 14:23:29', 5], [4, 'TEAM_12289', 0.7470608, '2026-11-05 09:27:19', 2],
    [5, 'TEAM_11245', 0.7464002, '2026-11-04 12:20:13', 1], [6, 'TEAM_12386', 0.7423199, '2026-11-05 13:27:05', 9]]},
  // M24 身分比對（results/r_variants/ident_pool_local.json、ident_ct8_local.json）
  ident: {
    top1ByN: [[5, 0.265], [10, 0.460], [15, 0.521], [20, 0.602]], top1: 0.454,
    medCorrect: 4.44, medWrong: 0.40, Bq: {q50: 0.465, q90: 1.764, q99: 6.147},
    testAboveQ99: [[5, 0.010], [10, 0.005], [15, 0.005], [20, 0.025]], inPool: [[5, 0.0], [10, 0.032], [15, 0.0], [20, 0.0]],
    gain: 0.0005, gainCI: [-0.0003, 0.0014], changed: 1,
  },
  // 競賽一 玩家辨識（AI_CUP_2026-2：METHOD_NOTES §2.11–§2.20、results/probe/public_analysis.json、results/pid_fusion3o_ref.json）
  pid: {
    // 架構圖的成分列：[名稱, 說明, 最終檔的融合權重]；最後一列是沒進最終檔的成分
    parts: [['開局', '黑方前 12 手，8 種對稱正規化', 4.69], ['風格', '32 維整局統計，LDA', 1.55], ['局部棋形', '每手的位移類別、線、階段', 0.70],
      ['原始方向', '開在哪個角、第一手原始座標', 1.38], ['參考玩家相似度', '對 2,917 位訓練玩家的似然', 3.06], ['未進最終檔', '原始座標開局、前 4 手方向、讓子', null]],
    // 單一成分的 Public Top-1（答案已知後離線計算）：[名稱, Top-1, 在新最終候選 fusion3o_ref＋trunk 裡]
    comp: [['trunk 風格', 0.968, true], ['原始座標開局', 0.953, false], ['開局（正規化）', 0.920, true], ['方向（前 4 手）', 0.890, false], ['風格', 0.665, true],
      ['參考玩家相似度', 0.635, true], ['方向（第 1 手）', 0.255, true], ['讓子旗標', 0.140, false], ['局部棋形', 0.083, true]],
    // Public：[版本, 不配對, 配對]（排行榜實測；答案已知後離線重算一致到第 7 位）
    pub: [['fusion（開局＋風格）', 0.9487291, 0.9704912], ['fusion3（＋局部棋形）', 0.9573466, 0.9833134], ['fusion3o（＋原始方向）', 0.9658747, 0.9873479],
      ['fusion3o_ref（＋參考玩家，最終檔）', 0.9759976, 0.9937580]],
    // 難負例候選庫（results/hard/hard_pools.json，fusion3、固定權重、不配對）
    hard: [['隨機 400 人', 0.9648], ['同級帶 D–B', 0.9550], ['同級帶 A–2D', 0.9582], ['同級帶 3D–4D', 0.9623], ['同級帶 5D–6D', 0.9317], ['開局最相近 400 人', 0.9311]],
    // Public 最後 3 題（3-環）：[題目, n, 原 Top-1, 正解, 正解原排名]；順序依環
    cycle: [['pub_q_0357', 10, 'pub_p_0114', 'pub_p_0052', '第 2 名'], ['pub_q_0018', 5, 'pub_p_0277', 'pub_p_0114', '第 3 名'], ['pub_q_0189', 5, 'pub_p_0052', 'pub_p_0277', '前 5 外']],
    // 8 個成分中幾個同意 fusion3o_ref 不配對的 Top-1：[同意數, 題數, 錯題數]
    agree: [[0, 8, 6], [1, 5, 3], [2, 21, 3], [3, 68, 0], [4, 85, 0], [5, 137, 0], [6, 67, 0], [7, 9, 0]],
  },
};


/* ===== 引擎（沿用模板）：狀態、長條圖、動畫輔助函數；各章的 NUM／CHARTS／ANIM 在下面的章節區塊 ===== */
(function(){
  let mode = 'read', cur = 0;
  const pages = [...document.querySelectorAll('main > .page')];
  const NUM = window.NUM || {};
  const pstr = p => p.map(q => q[0] + ',' + q[1]).join(' ');
  const f4 = x => (+x).toFixed(4), f3 = x => (+x).toFixed(3);
  // 長條圖：rows = [[標籤, 值, 顯示文字, 顏色]]；lo／hi 是刻度範圍
  const barRows = (rows, lo, hi) => rows.map(([label, v, text, col]) => `<div class="row"><span>${label}</span><div class="track"><div class="fill" style="width:${Math.max(1, Math.min(100, (v - lo) / (hi - lo) * 100)).toFixed(1)}%;background:${col || 'var(--accent)'}"></div></div><span class="val">${text}</span></div>`).join('');
  const CHARTS = {};
  // 簡報模式：整張卡要塞進一個畫面。內容比可用高度高時，依比例縮小這一頁的字（最小 11px）；動畫 SVG 另由 CSS 的 max-height 限制
  function fitFigs(page) {
    if (!page) return;
    const box = page.querySelector('.report') || page;
    box.style.fontSize = '';
    if (mode !== 'present' || window.innerWidth <= 820 || !page.classList.contains('cur')) return;
    const avail = window.innerHeight - 124;
    for (let k = 0; k < 5; k++) {
      const h = page.scrollHeight; if (h <= avail) break;
      const fs = parseFloat(getComputedStyle(box).fontSize);
      if (fs <= 11) break;
      box.style.fontSize = Math.max(11, fs * Math.max(0.82, avail / h)).toFixed(2) + 'px';
    }
  }
  // ===== 每張卡的動畫（.anim[data-anim]）：預設一輪 9 秒，全部用 SVG <animate>，所以能用同一條進度條拖曳（setCurrentTime）。
  //       圖上的框都來自 overlays.js（我們提交檔的框）；只是示意的在動畫下方註明。 =====
  // AD：這段一輪幾秒；HOLD 之後淡出、重來；RAW：時間照寫的走、不壓縮前搖。放進頁面前依 ANIM.x.dur／raw 設定（第 8 頁 20 秒）。
  let AD = 9, HOLD = 8.3, RAW = false;
  // 前搖縮短：各段動畫寫的是「設計時間」，kt() 換成實際時間——HOLD 以前先提前 0.3 秒、再壓成 0.75 倍，HOLD 以後不變
  const warp = t => (RAW || t >= HOLD ? t : Math.max(0, (t - 0.3) * 0.75));
  const kt = a => a.map(t => Math.min(1, Math.max(0, warp(t) / AD)).toFixed(4)).join(';');
  // 出現：t0 淡入、停到 t2 再淡出
  const aShow = (t0, t2 = HOLD) => `<animate attributeName="opacity" dur="${AD}s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="${kt([0, t0, t0 + 0.3, t2, t2 + 0.3, AD])}"/>`;
  // 畫線（元素要有 pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"）
  const aDraw = (t0, d = 0.5, t2 = HOLD) => `<animate attributeName="stroke-dashoffset" dur="${AD}s" repeatCount="indefinite" values="1;1;0;0" keyTimes="${kt([0, t0, t0 + d, AD])}"/>` + aShow(t0, t2);
  // 屬性從 v0 變到 v1，停到 t2 再回來
  const aTo = (attr, v0, v1, t0, d = 0.6, t2 = HOLD) => `<animate attributeName="${attr}" dur="${AD}s" repeatCount="indefinite" values="${v0};${v0};${v1};${v1};${v0}" keyTimes="${kt([0, t0, t0 + d, t2, AD])}"/>`;
  // 自訂時間表 [[時間, 值], …]：第一個時間是 0、最後一個是 AD
  const aSeq = (attr, pairs, discrete) => `<animate attributeName="${attr}" dur="${AD}s" repeatCount="indefinite"${discrete ? ' calcMode="discrete"' : ''} values="${pairs.map(p => p[1]).join(';')}" keyTimes="${kt(pairs.map(p => p[0]))}"/>`;
  const tx = (x, y, cls, t, extra = '') => `<text class="${cls}" x="${x}" y="${y}" ${extra}>${t}</text>`;
  const txA = (x, y, cls, t, t0, t2, extra = '') => `<text class="${cls}" x="${x}" y="${y}" opacity="0" ${extra}>${t}${aShow(t0, t2)}</text>`;
  const polyS = (p, attrs) => `<polygon points="${pstr(p)}" vector-effect="non-scaling-stroke" ${attrs}/>`;
  const polyA = (p, attrs, kids) => `<polygon points="${pstr(p)}" vector-effect="non-scaling-stroke" ${attrs}>${kids}</polygon>`;
  const nest = (R, crop, x, y, w, h, inner) => `<svg x="${x}" y="${y}" width="${w}" height="${h}" viewBox="${crop.map(v => (+v).toFixed(1)).join(' ')}" preserveAspectRatio="xMidYMid meet">`
    + `<rect x="${crop[0]}" y="${crop[1]}" width="${crop[2]}" height="${crop[3]}" fill="#0b0d10"/><image href="${R.o.src}" width="${R.o.w}" height="${R.o.h}"/>${inner}</svg>`;
  const inPoly = (pt, p) => { let c = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) { const [xi, yi] = p[i], [xj, yj] = p[j];
      if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c; }
    return c; };
  const ANIM = {};

  // ---- 第 1 章 ----
  CHARTS['chart-players'] = () => '<h4>訓練集各級玩家數（每級 10 萬局）</h4>' + barRows(NUM.players.map(([n, c]) => [n, c, c.toLocaleString(), 'var(--accent)']), 0, 1600);

  // 小棋盤：s 是邊長，k 是線數，stones = [[x, y, 'b'|'w'], …]（線的索引）
  const miniBoard = (x, y, s, k, stones, extra = '') => {
    const g = s / (k + 1); let o = `<g ${extra}><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="#dcb36a" stroke="#9c7a3c"/>`;
    for (let i = 1; i <= k; i++) o += `<line x1="${x + g}" y1="${y + i * g}" x2="${x + k * g}" y2="${y + i * g}" stroke="#8a6a2f" stroke-width=".7"/><line x1="${x + i * g}" y1="${y + g}" x2="${x + i * g}" y2="${y + k * g}" stroke="#8a6a2f" stroke-width=".7"/>`;
    stones.forEach(([i, j, c]) => { o += `<circle cx="${(x + i * g).toFixed(1)}" cy="${(y + j * g).toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="${c === 'b' ? '#1d1f24' : '#fff'}" stroke="#1d1f24" stroke-width=".8"/>`; });
    return o + '</g>';
  };

  // 第 1 頁右欄：一題 = 同一位玩家的幾局 → 模型 → 10 級之一 → 依真實等級計分
  ANIM.task = () => {
    const L = NUM.levels, RX = 452, RW = 80, RH = 19, top = 34, pick = 5;
    const ry = i => top + (9 - i) * (RH + 4);
    const games = [[[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 3, 'b']], [[2, 6, 'b'], [6, 2, 'w'], [6, 6, 'b'], [3, 3, 'w']],
      [[6, 2, 'b'], [2, 2, 'w'], [2, 6, 'b'], [5, 5, 'w'], [6, 4, 'b']], [[2, 2, 'b'], [6, 6, 'w'], [2, 6, 'b'], [6, 3, 'w']], [[6, 6, 'b'], [2, 2, 'w'], [6, 2, 'b'], [3, 6, 'w'], [4, 4, 'b']]];
    let s = tx(20, 22, 'an-k', '一題：同一位玩家');
    games.forEach((st, i) => {
      const x = 22 + (i % 3) * 64, y = 40 + Math.floor(i / 3) * 76, t0 = 0.4 + i * 0.32;
      s += `<g opacity="0">${aShow(t0)}${miniBoard(x, y, 56, 7, st)}` + tx(x + 28, y + 70, 'an-s', `第 ${i + 1} 局`, 'text-anchor="middle"') + '</g>';
    });
    s += txA(150, 140, 'an-k', '…', 2.0) + txA(22, 214, 'an-n', '共 5–20 局，目標執色固定', 2.2)
      + `<line x1="214" y1="128" x2="292" y2="128" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.6, 0.5)}</line>`
      + `<g opacity="0">${aShow(3.0)}<rect x="296" y="104" width="96" height="48" rx="8" fill="#e3f1ef" stroke="#0b6e69"/>` + tx(344, 133, 'an-b', '模型', 'text-anchor="middle"') + '</g>'
      + `<line x1="392" y1="128" x2="${RX - 8}" y2="${ry(pick) + RH / 2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(3.5, 0.4)}</line>`;
    L.forEach((n, i) => {
      const y = ry(i), t0 = 3.3 + i * 0.04;
      s += `<rect x="${RX}" y="${y}" width="${RW}" height="${RH}" rx="4" fill="#eef0f3" stroke="#c9ced6" opacity="0">${aShow(t0)}</rect>`
        + txA(RX + RW / 2, y + 14, 'an-k', n, t0, undefined, 'text-anchor="middle"');
    });
    s += `<rect x="${RX}" y="${ry(pick)}" width="${RW}" height="${RH}" rx="4" fill="#0b6e69" opacity="0">${aShow(4.0)}</rect>`
      + txA(RX + RW / 2, ry(pick) + 14, 'an-w', '預測 2D', 4.0, undefined, 'text-anchor="middle"')
      + txA(RX + RW + 8, ry(pick) + 14, 'an-v an-ok', '1', 4.9) + txA(RX + RW + 22, ry(pick) + 14, 'an-s', '若真的是 2D', 4.9)
      + txA(RX + RW + 8, ry(pick + 1) + 14, 'an-n', '0.368', 5.5) + txA(RX + RW + 8, ry(pick - 1) + 14, 'an-n', '0.368', 5.5)
      + txA(RX + RW + 8, ry(pick + 2) + 14, 'an-s', '其餘 0', 6.1) + txA(RX + RW + 8, ry(pick - 2) + 14, 'an-s', '其餘 0', 6.1)
      + txA(296, 196, 'an-k', '差一級', 5.5) + txA(296, 214, 'an-k', '也有 e⁻¹ 分', 5.5);
    return {steps: ['① 一題：同一位玩家的 5–20 局', '② 模型輸出 10 級之一', '③ 依真實等級計分'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="一題的輸入、輸出與計分方式的動畫">${s}</svg>`, note: '示意：棋盤是示意圖；計分規則是官方規則。'};
  };

  // ---- 第 2 章 ----
  CHARTS['chart-byn'] = () => '<h4>production 留出題的分數，依局數（ct3 集成，實測；長條從 0.50 起）</h4>'
    + barRows(NUM.gateByN.map(([n, v]) => [`n ＝ ${n} 局`, v, v.toFixed(3), n === 20 ? 'var(--accent)' : '#8a93a3']), 0.5, 0.85);

  // 第 2 頁右欄：局數越多，十級的機率越集中；但中心是「看起來像幾段」，不一定是標籤
  ANIM.accum = () => {
    const L = NUM.levels, base = 206, H = 150, X = i => 64 + i * 54, W = 36, lab = 5;
    const P = [[0.04, 0.06, 0.08, 0.11, 0.13, 0.14, 0.14, 0.12, 0.10, 0.08],
      [0.00, 0.01, 0.02, 0.05, 0.12, 0.25, 0.30, 0.17, 0.06, 0.02],
      [0.00, 0.00, 0.00, 0.01, 0.04, 0.22, 0.52, 0.18, 0.03, 0.00]];
    const h = p => (p / 0.55 * H).toFixed(1), T = [0.4, 2.9, 5.3];
    let s = `<line x1="40" y1="${base}" x2="${X(9) + W + 12}" y2="${base}" stroke="#c9ced6"/>`
      + `<rect x="${X(lab) - 6}" y="${base - H - 14}" width="${W + 12}" height="${H + 14}" rx="4" fill="none" stroke="#a4451c" stroke-width="1.6" stroke-dasharray="5 4"/>`
      + tx(X(lab) + W / 2, base - H - 20, 'an-h', '標籤 2D', 'text-anchor="middle"');
    L.forEach((n, i) => {
      const hs = P.map(r => h(r[i])), ys = hs.map(v => (base - v).toFixed(1));
      const seq = vals => [[0, 0], [T[0], 0], [T[0] + 0.5, vals[0]], [T[1], vals[0]], [T[1] + 0.5, vals[1]], [T[2], vals[1]], [T[2] + 0.5, vals[2]], [HOLD, vals[2]], [HOLD + 0.3, 0], [AD, 0]];
      const ysq = seq(ys).map(([t, v], k) => [t, (k < 2 || k > 7) ? base : v]);
      s += `<rect x="${X(i)}" y="${base}" width="${W}" height="0" rx="3" fill="${i === 6 ? '#0b6e69' : '#7fb8b3'}">${aSeq('y', ysq)}${aSeq('height', seq(hs))}</rect>`
        + tx(X(i) + W / 2, base + 18, 'an-k', n, 'text-anchor="middle"');
    });
    s += txA(40, 24, 'an-t', '看了 1 局：10 級都有可能', T[0], T[1]) + txA(40, 24, 'an-t', '看了 5 局：集中到兩三級', T[1] + 0.3, T[2])
      + txA(40, 24, 'an-t', '看了 20 局：更集中，但中心在 3D', T[2] + 0.3)
      + txA(40, 242, 'an-h', '證據越多越集中；中心是「看起來像幾段」，不一定是標籤', 6.6);
    return {steps: ['① 一局：10 級都有可能', '② 5 局：集中到兩三級', '③ 20 局：更集中，但不一定在標籤上'],
      svg: `<svg viewBox="0 0 640 248" role="img" aria-label="局數越多機率越集中，但中心不一定是標籤的動畫">${s}</svg>`,
      note: '示意：機率是示意數字；左欄的 n ＝ 5／10／20 分數是實測。'};
  };

  // 第 3 頁右欄：每位玩家兩組不重疊的 40 局各估一次，兩次落在標籤的同一側
  ANIM.bias = () => {
    const L = NUM.levels, X = v => 76 + v * 55, Y = r => 56 + r * 32;
    const P = [[3, 0.8, -0.18, 0.14], [5, -0.7, 0.16, -0.12], [7, 0.35, -0.2, 0.22], [1, -0.55, 0.12, -0.2], [8, -1.0, -0.15, 0.17], [4, 0.6, 0.2, -0.1]];
    let s = `<line x1="64" y1="238" x2="${X(9) + 12}" y2="238" stroke="#c9ced6"/>`;
    L.forEach((n, i) => { s += `<line x1="${X(i)}" y1="234" x2="${X(i)}" y2="242" stroke="#c9ced6"/>` + tx(X(i), 256, 'an-k', n, 'text-anchor="middle"'); });
    P.forEach(([lab, b, e1, e2], r) => {
      const y = Y(r), t0 = 0.4 + r * 0.12, x0 = X(lab), xb = X(lab + b);
      s += tx(16, y + 4, 'an-s', `玩家 ${r + 1}`)
        + `<line x1="70" y1="${y}" x2="${X(9) + 8}" y2="${y}" stroke="#eef0f3"/>`
        + `<line x1="${x0}" y1="${y - 9}" x2="${x0}" y2="${y + 9}" stroke="#1d1f24" stroke-width="3" opacity="0">${aShow(t0)}</line>`
        + `<line x1="${x0}" y1="${y}" x2="${xb}" y2="${y}" stroke="#a4451c" stroke-width="2" stroke-dasharray="4 3" opacity="0">${aShow(4.3)}</line>`
        + `<circle cx="${(X(lab + b + e1)).toFixed(1)}" cy="${y - 4}" r="6" fill="#0072b2" opacity="0">${aShow(2.0)}${aTo('cx', X(lab + b + e1).toFixed(1), X(lab + b + e1 * 0.25).toFixed(1), 5.6, 0.7)}</circle>`
        + `<circle cx="${(X(lab + b + e2)).toFixed(1)}" cy="${y + 4}" r="6" fill="#e69f00" opacity="0">${aShow(2.7)}${aTo('cx', X(lab + b + e2).toFixed(1), X(lab + b + e2 * 0.25).toFixed(1), 5.6, 0.7)}</circle>`;
    });
    s += tx(16, 22, 'an-k', '黑線：標籤　藍：前 40 局的估計　橙：另 40 局的估計')
      + txA(16, 40, 'an-h', '兩次估計落在同一側：偏差跟著人走', 4.3, 5.5)
      + txA(16, 40, 'an-h', '局數加倍：兩點靠攏，但不回到標籤', 5.8)
      + txA(640 - 12, 22, 'an-n', '同一人兩組局的偏差相關 0.964', 6.4, undefined, 'text-anchor="end"');
    return {steps: ['① 每位玩家的標籤', '② 兩組不同的局（各 40 局）各估一次', '③ 兩次落在同一側，局數加倍也不回到標籤'],
      svg: `<svg viewBox="0 0 640 266" role="img" aria-label="玩家層偏差的動畫">${s}</svg>`,
      note: '示意：點的位置是示意；0.964 是實測（anchor_kalman：2,524 位有 80 局的玩家，兩組不重疊的 40 局各估一次的本人證據相關）。'};
  };

  // 第 4 頁右欄：RP 在三年間隨機漫步、多次跨過段位邊界；8 局是期間內的無序樣本，標籤只有一個值
  ANIM.rp = () => {
    const X0 = 64, X1 = 612, N = 64, band = [[44, 104, '3D'], [104, 164, '2D'], [164, 224, '1D']];
    let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    // 固定的轉折點（y）加 ±5 的小抖動：2D → 3D → 2D → 1D → 2D → 3D → 2D → 1D → 2D；2D 帶約占一半時間，與「標籤 2D」一致
    const K = [140, 134, 130, 70, 90, 140, 150, 205, 170, 120, 96, 70, 140, 156, 200, 190, 150];
    const pts = [];
    for (let i = 0; i < N; i++) { const u = i / (N - 1) * (K.length - 1), j = Math.min(K.length - 2, Math.floor(u)), f = u - j; pts.push([X0 + i * (X1 - X0) / (N - 1), K[j] + (K[j + 1] - K[j]) * f + (rnd() - 0.5) * 10]); }
    const pick = [4, 11, 19, 26, 34, 41, 49, 57], inBand = yy => band.find(([a, b]) => yy >= a && yy < b)[2];
    const cnt = {}; pick.forEach(i => { const b = inBand(pts[i][1]); cnt[b] = (cnt[b] || 0) + 1; });
    let s = '';
    band.forEach(([a, b, n], i) => {
      s += `<rect x="${X0}" y="${a}" width="${X1 - X0}" height="${b - a}" fill="${i % 2 ? '#f3f8f7' : '#ffffff'}"/>` + tx(X0 - 10, (a + b) / 2 + 4, 'an-k', n, 'text-anchor="end"');
      if (i) s += `<line x1="${X0}" y1="${a}" x2="${X1}" y2="${a}" stroke="#c9ced6" stroke-dasharray="4 4"/>`;
    });
    s += tx(X0, 244, 'an-s', '2021') + tx(X1, 244, 'an-s', '2023', 'text-anchor="end"') + tx((X0 + X1) / 2, 244, 'an-s', '時間（資料裡沒有）', 'text-anchor="middle"')
      + `<polyline points="${pstr(pts.map(p => [p[0].toFixed(1), p[1].toFixed(1)]))}" fill="none" stroke="#1d1f24" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(0.4, 2.6)}</polyline>`
      + txA(X0, 24, 'an-t', 'RP 每局約 ±100 點，三年間多次跨過段位邊界', 0.4, 3.6);
    pick.forEach((i, k) => {
      const [px, py] = pts[i], t0 = 3.8 + k * 0.2, c = {'1D': '#0072b2', '2D': '#0b6e69', '3D': '#d1343a'}[inBand(py)];
      s += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="6.5" fill="${c}" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(t0)}</circle>`;
    });
    s += txA(X0, 24, 'an-t', '資料只給這 8 局，而且不按時間排', 3.9, 5.5)
      + `<line x1="${X0}" y1="134" x2="${X1}" y2="134" stroke="#a4451c" stroke-width="4" opacity="0">${aShow(5.7)}</line>`
      + txA(626, 128, 'an-h', '標籤：2D（一個代表值）', 5.7, undefined, 'text-anchor="end"')
      + txA(X0, 24, 'an-t', `8 局中 1D 時期 ${cnt['1D'] || 0} 局、2D ${cnt['2D'] || 0} 局、3D ${cnt['3D'] || 0} 局`, 6.4);
    return {steps: ['① RP 在期間內隨機漫步', '② 資料是期間內的幾局，沒有時間', '③ 標籤只有一個值'],
      svg: `<svg viewBox="0 0 640 252" role="img" aria-label="RP 漫步與期間代表值標籤的動畫">${s}</svg>`,
      note: '示意：RP 軌跡與 8 局是示意；1,000 點區間來自幽玄の間手冊；每局約 ±100 RP 是依手冊（1 級約 116、8 段約 35）推估。'};
  };

  // ---- 第 3 章 ----
  // 第 5 頁右欄：離線分數與 Public 一路上升（NUM.climb）；第 5 欄是口徑，口徑改變處只畫灰色虛線、不連實線
  //   11-11：點數從 9 加到 11（對比損失、對手錨點），橫向間距與縱軸上限跟著改
  ANIM.climb = () => {
    const C = NUM.climb, n = C.length, X = i => 74 + i * 54, lo = 0.40, hi = 0.82, Y = v => 222 - (v - lo) / (hi - lo) * 186;
    const t = i => 0.5 + i * 0.7;
    let s = '';
    [0.4, 0.5, 0.6, 0.7, 0.8].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="60" y1="${y}" x2="${X(n - 1) + 20}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(54, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(1), 'text-anchor="end"'); });
    s += `<circle cx="66" cy="14" r="5" fill="#0b6e69"/>` + tx(76, 18, 'an-s', '離線分數') + `<circle cx="146" cy="14" r="5" fill="#fff" stroke="#a4451c" stroke-width="2"/>` + tx(156, 18, 'an-s', 'Public');
    C.slice(1).forEach((c, j) => {
      const x1 = X(j), y1 = Y(C[j][1]).toFixed(1), x2 = X(j + 1), y2 = Y(c[1]).toFixed(1);
      if (C[j][4] === c[4]) s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#0b6e69" stroke-width="2.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t(j), t(j + 1) - t(j))}</line>`;
      else {
        // 「換口徑」三個字放在虛線下方的空白處：f0 → mlp、mlp → ens、ens → anc 各一個位置
        const [lx, ly, la] = ({mlp: [x1 + 40, 166, 'start'], ens: [(x1 + x2) / 2, 116, 'middle'], anc: [(x1 + x2) / 2, 106, 'middle']})[c[4]] || [(x1 + x2) / 2, 116, 'middle'];
        s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#8a93a3" stroke-width="1.6" stroke-dasharray="3 4" opacity="0">${aShow(t(j + 1))}</line>`
          + txA(lx, ly, 'an-s', '換口徑', t(j + 1), undefined, `text-anchor="${la}" style="fill:#8a93a3"`);
      }
    });
    C.forEach(([name, cv, pub], i) => {
      const t0 = t(i), x = X(i);
      const fmt = v => v.toFixed(Math.round(v * 1e4) % 10 === 0 ? 3 : 4);
      s += `<circle cx="${x}" cy="${Y(cv).toFixed(1)}" r="5" fill="#0b6e69" opacity="0">${aShow(t0)}</circle>`
        + txA(x, (Y(cv) + (pub != null ? 18 : -10)).toFixed(1), 'an-n', fmt(cv), t0 + 0.1, undefined, 'text-anchor="middle" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:#0b6e69"')
        + txA(x, 240, 'an-k', name, t0, undefined, 'text-anchor="middle" style="font-size:11px"');
      if (pub != null) s += `<circle cx="${x}" cy="${Y(pub).toFixed(1)}" r="5.5" fill="#fff" stroke="#a4451c" stroke-width="2" opacity="0">${aShow(t0 + 0.25)}</circle>`
        + txA(i ? x : x - 9, (i ? Y(pub) - 9 : Y(pub) + 4).toFixed(1), 'an-n', fmt(pub), t0 + 0.35, undefined, `text-anchor="${i ? 'middle' : 'end'}" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:#a4451c"`);
    });
    s += txA(220, 18, 'an-h', '大跳升只有 Human SL（+0.17）；之後每步 +0.005–0.035', t(n - 1) + 0.4);
    return {steps: ['① 每手似然（Human SL）', '② 等級頭（L1b）與對手', '③ 讓子、局面向量＋集成', '④ 更多局、對手 Human SL', '⑤ 對比損失、對手錨點'],
      svg: `<svg viewBox="0 0 640 252" role="img" aria-label="離線分數與 Public 分數一路上升的動畫">${s}</svg>`,
      note: '最左點是官方 baseline 在 fold 0 的分數；第 2–6 點是單一 MLP 的 CV（folds 1–4、類別平衡）；第 7–10 點是 6 模型集成的 CV；最後一點是 0.7458 加上對手錨點在同一個底上的配對增益 +0.0193（驗證端的錨點密度；測試端較低）。口徑改變處只用灰色虛線相接。Public 是使用者回報的上傳分數（C＋T 檔是 12 模型；最後一點是 ct44，另含兩個由 Public 選出的推移，見第 7 章）。'};
  };
  ANIM.climb.dur = 10;

  // 第 6 頁右欄：Public 上傳分數（NUM.uploads：模型檔與被保留的檔，共 13 個）；紅點是 ct6 與 ct40，NUM.final 是最終檔
  ANIM.uploads = () => {
    const U = NUM.uploads, n = U.length, X = i => 72 + i * 44, lo = 0.70, hi = 0.81, Y = v => 214 - (v - lo) / (hi - lo) * 160;
    const t = i => 0.5 + i * 0.5, fin = NUM.final, bad = {ct6: 1, ct40: 1};
    let s = '';
    [0.70, 0.72, 0.74, 0.76, 0.78, 0.80].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="50" y1="${y}" x2="${X(n - 1) + 26}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(44, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    U.slice(1).forEach((u, j) => {
      s += `<line x1="${X(j)}" y1="${Y(U[j][2]).toFixed(1)}" x2="${X(j + 1)}" y2="${Y(u[2]).toFixed(1)}" stroke="#8a93a3" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t(j), t(j + 1) - t(j))}</line>`;
    });
    U.forEach(([name, day, v], i) => {
      const col = bad[name] ? '#d1343a' : name === fin ? '#0b6e69' : '#5d6370', r = name === fin ? 7 : 5, t0 = t(i);
      // 分數標籤的位置依前後兩點決定，避開連線：谷底放下方、峰頂放上方、一路上升的放左上
      const up = i === 0 || U[i - 1][2] < v, down = i < n - 1 && U[i + 1][2] > v;
      const [lx, ly, la] = !up || i === 0 ? (down ? [X(i), Y(v) + 20, 'middle'] : [X(i) + 7, Y(v) - 7, 'start'])
        : down ? [X(i) - 7, Y(v) - 7, 'end'] : [X(i), Y(v) - (name === fin ? 17 : 11), 'middle'];
      s += `<circle cx="${X(i)}" cy="${Y(v).toFixed(1)}" r="${r}" fill="${col}" opacity="0">${aShow(t0)}</circle>`
        + txA(lx, ly.toFixed(1), 'an-n', v.toFixed(4), t0 + 0.1, undefined, `text-anchor="${la}" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`)
        + txA(X(i), 234, 'an-k', name, t0, undefined, 'text-anchor="middle" style="font-size:12px"')
        + txA(X(i), 249, 'an-s', day, t0, undefined, 'text-anchor="middle" style="font-size:10.5px"');
    });
    const i6 = U.findIndex(u => u[0] === 'ct6'), i40 = U.findIndex(u => u[0] === 'ct40'), iF = U.findIndex(u => u[0] === fin);
    if (i6 >= 0) s += txA(X(i6) + 4, (Y(U[i6][2]) + 36).toFixed(1), 'an-h', 'CV 預期 +0.010 → −0.0095', t(i6) + 0.4);
    if (i40 >= 0) s += txA(X(i40), (Y(U[i40][2]) + 38).toFixed(1), 'an-h', '先錨點後推移 → 掉分', t(i40) + 0.4, undefined, 'text-anchor="middle"');
    if (iF >= 0) s += `<circle cx="${X(iF)}" cy="${Y(U[iF][2]).toFixed(1)}" r="13" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(t(n - 1) + 0.5)}</circle>`
      + txA(X(iF), (Y(U[iF][2]) - 32).toFixed(1), 'an-b', '最終檔', t(n - 1) + 0.5, undefined, 'text-anchor="middle"');
    return {steps: ['① 11-04 至 11-06：以 CV 決策，Public 看方向', '② 11-07 至 11-09：解碼推移（ct13、ct28）', '③ 11-10 至 11-11：對手錨點（ct42 → ct44）'],
      svg: `<svg viewBox="0 0 640 258" role="img" aria-label="Public 上傳分數的動畫">${s}</svg>`,
      note: '分數是使用者回報的 Public。只畫模型檔與被保留的檔；11-07 到 11-10 另有 ct10、ct12 與 14 個被丟棄的探測檔（見 3.2 與 7.1 的表）。紅點：ct6（12 個全資料模型）與 ct40（錨點加在推移之前）。ct4 與 ct7 同分（0.7687835），兩檔在 Public 有 47 題決策不同。'};
  };

  // ---- 第 4 章 ----
  // 第 7 頁（橫跨整張卡）：從棋譜到等級的三個階段
  ANIM.pipeline = () => {
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, fill, stroke, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}"/>${inner}</g>`;
    const arrow = (x1, y1, x2, y2, t0) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.4)}</line>`
      + `<polygon points="${x2},${y2} ${x2 - 7},${y2 - 4} ${x2 - 7},${y2 + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.4)}</polygon>`;
    let s = '';
    // 段落標題
    [['階段一　每手證據', 214, 0.3], ['階段二　整局與整題', 560, 3.5], ['階段三　集成與決策', 830, 6.2]].forEach(([t, x, t0]) => { s += txA(x, 22, 'an-t', t, t0, undefined, 'text-anchor="middle"'); });
    // 棋譜
    s += box(14, 96, 104, 112, '#fff', '#c9ced6', 0.4, miniBoard(30, 104, 72, 7, [[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 4, 'b']]) + `<text x="66" y="198" text-anchor="middle" style="${Sm}">SGF 棋譜</text>`)
      + arrow(120, 152, 146, 152, 1.0)
      + box(150, 112, 104, 80, '#e3f1ef', '#0b6e69', 1.3, `<text x="202" y="144" text-anchor="middle" style="${Hd}">KataGo 重播</text><text x="202" y="164" text-anchor="middle" style="${Sm}">中國規則、不搜尋</text>`)
      + arrow(256, 152, 282, 152, 1.9);
    // token 162 維
    const seg = [['盤面 trunk', 64, '#9ecae1'], ['落點 trunk', 64, '#6baed6'], ['L1b', 11, '#0b6e69'], ['Human SL', 19, '#e69f00'], ['旗標', 4, '#8a93a3']];
    let y0 = 54; const SC = 0.95;
    s += txA(298, 46, 'an-k', '每手 token 162 維', 2.2, undefined, 'style="font-size:12px"');
    seg.forEach(([n, d, c], k) => {
      const h = Math.max(9, d * SC), t0 = 2.3 + k * 0.25;
      s += `<rect x="298" y="${y0.toFixed(1)}" width="26" height="${h.toFixed(1)}" fill="${c}" opacity="0">${aShow(t0)}</rect>`
        + txA(330, (y0 + h / 2 + 4).toFixed(1), 'an-s', `${n} ${d}`, t0);
      y0 += h + 2;
    });
    // 一局的 token 序列：本人手與對手手交替
    for (let k = 0; k < 14; k++) {
      const x = 420 + k * 9, own = k % 2 === 0, t0 = 3.6 + k * 0.05;
      s += `<rect x="${x}" y="110" width="7" height="84" rx="1.5" fill="${own ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`;
    }
    s += txA(420, 92, 'an-s', '一局的每一手', 3.6) + txA(420, 104, 'an-s', '深：本人　淺：對手', 3.6) + arrow(548, 152, 574, 152, 4.4);
    // C / T
    s += box(578, 92, 110, 50, '#fff', '#0b6e69', 4.7, `<text x="633" y="114" text-anchor="middle" style="${Hd}">C　逐手 MLP</text><text x="633" y="132" text-anchor="middle" style="${Sm}">隱藏 110</text>`)
      + box(578, 152, 110, 50, '#fff', '#0b6e69', 5.0, `<text x="633" y="174" text-anchor="middle" style="${Hd}">T　擴張卷積</text><text x="633" y="192" text-anchor="middle" style="${Sm}">5 層、感受野 63 手</text>`)
      + box(578, 214, 110, 44, '#f2f3f6', '#c9ced6', 5.5, `<text x="633" y="232" text-anchor="middle" style="${Hd}">＋ R 64 維</text><text x="633" y="249" text-anchor="middle" style="${Sm}">題目特徵</text>`)
      + txA(633, 84, 'an-s', '平均／標準差池化：每局 → 每題', 5.2, undefined, 'text-anchor="middle"')
      + arrow(690, 152, 716, 152, 6.0);
    // 24 個成員
    for (let k = 0; k < 24; k++) {
      const x = 724 + (k % 6) * 17, y = 104 + Math.floor(k / 6) * 17, t0 = 6.3 + k * 0.03;
      s += `<rect x="${x}" y="${y}" width="13" height="13" rx="2" fill="${k % 2 ? '#e69f00' : '#0b6e69'}" opacity="0">${aShow(t0)}</rect>`;
    }
    s += txA(724, 96, 'an-s', '24 個成員', 6.3) + txA(724, 190, 'an-s', '4 折×{C,T}×3 seed', 6.6)
      + txA(724, 206, 'an-s', '十級機率平均', 6.8) + arrow(830, 152, 852, 152, 7.4)
      + box(856, 106, 96, 92, '#0b6e69', '#0b6e69', 7.8, `<text x="904" y="134" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">期望分最高</text><text x="904" y="172" text-anchor="middle" style="font-size:30px;font-weight:700;fill:#fff">3D</text>`)
      + txA(952, 222, 'an-s', '鄰級機率 × 0.368 也算分', 8.2, undefined, 'text-anchor="end"');
    return {steps: ['階段一（表示）：重播、每手 162 維 token', '階段二（聚合）：C／T 讀整局、池化到每題', '階段三：24 個成員平均、期望分決策'],
      tech: 'KataGo b18c384nbt-humanv0（凍結）＋ L1b 微調 trunk 與 10 個等級頭；PyTorch；集成成員都經 CV 驗證',
      svg: `<svg viewBox="0 0 960 264" role="img" aria-label="從棋譜到等級的三階段架構動畫">${s}</svg>`,
      note: 'token 各段的長度依維度比例縮放；「3D」只是示意輸出。'};
  };
  Object.assign(ANIM.pipeline, {dur: 12});

  // 第 8 頁右欄：同一手在 18 個 profile 下的 log p（示意），以及 L1b 10 個等級頭
  ANIM.profiles = () => {
    const X = i => 260 + i * 20.5, Y = v => 30 + (-v - 0.8) / 2.8 * 104;   // log p −0.8 … −3.6
    const A = Array.from({length: 18}, (_, i) => -1.15 - 1.9 * Math.pow(i / 17, 1.1));
    const B = Array.from({length: 18}, (_, i) => -3.3 + 2.2 * Math.pow(i / 17, 0.9));
    const pts = arr => arr.map((v, i) => [X(i).toFixed(1), Y(v).toFixed(1)]);
    let s = miniBoard(18, 34, 200, 9, [[3, 3, 'b'], [7, 7, 'w'], [7, 3, 'b'], [3, 7, 'w'], [5, 2, 'b'], [6, 7, 'w'], [4, 5, 'b'], [8, 6, 'w']])
      + `<circle cx="${18 + 6 * 20}" cy="${34 + 5 * 20}" r="9" fill="none" stroke="#e69f00" stroke-width="3" opacity="0">${aShow(0.4)}</circle>`
      + `<circle cx="${18 + 4 * 20}" cy="${34 + 7 * 20}" r="9" fill="none" stroke="#0072b2" stroke-width="3" opacity="0">${aShow(0.4)}</circle>`
      + tx(18, 254, 'an-s', '同一局面的兩個候選手');
    s += `<line x1="${X(0)}" y1="138" x2="${X(17)}" y2="138" stroke="#c9ced6"/>` + tx(X(0), 152, 'an-s', '12k', 'text-anchor="middle"') + tx(X(17), 152, 'an-s', '6d', 'text-anchor="middle"')
      + tx((X(0) + X(17)) / 2, 152, 'an-s', '18 個 profile', 'text-anchor="middle"') + tx(250, 24, 'an-k', '每手的 log p')
      + `<polyline points="${pstr(pts(A))}" fill="none" stroke="#e69f00" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(0.8, 1.2)}</polyline>`
      + `<polyline points="${pstr(pts(B))}" fill="none" stroke="#0072b2" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.6, 1.2)}</polyline>`
      + txA(X(17) + 4, (Y(A[17]) + 4).toFixed(1), 'an-s', '橙', 1.9) + txA(X(17) + 4, (Y(B[17]) + 4).toFixed(1), 'an-s', '藍', 2.7)
      + txA(250, 170, 'an-k', '橙：弱 profile 較常下　藍：強 profile 較常下', 2.8);
    const L = NUM.levels, Lv = [-2.9, -2.6, -2.3, -2.05, -1.85, -1.68, -1.56, -1.5, -1.52, -1.6];
    L.forEach((n, i) => {
      const x = 262 + i * 36, h = (3.2 + Lv[i]) * 26, t0 = 4.2 + i * 0.08;
      s += `<rect x="${x}" y="${(240 - h).toFixed(1)}" width="26" height="${h.toFixed(1)}" rx="2" fill="#0b6e69" opacity="0">${aShow(t0)}</rect>`
        + txA(x + 13, 254, 'an-s', n, t0, undefined, 'text-anchor="middle"');
    });
    s += txA(250, 188, 'an-k', 'L1b：10 個等級頭，同一手在各級的 log p', 4.0);
    return {steps: ['① 同一局面、兩個候選手', '② 18 個 profile 的 log p', '③ L1b 10 個等級頭'],
      tech: '每手只存實際落子那一格：Human SL 18 個值、L1b 10 個值；對手的手用同樣方式評分',
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="每手證據的動畫">${s}</svg>`,
      note: '示意：曲線與長條的形狀是示意；實際每手只取實際落子那一格。'};
  };

  // 第 9 頁右欄：token 的 Human SL 欄原本只在本人手上有值；M14 把對手手上也填上
  ANIM.opp = () => {
    const n = 16, X = k => 40 + k * 36, W = 26;
    let s = tx(30, 22, 'an-k', '一局的 16 手（左起）') + tx(30, 40, 'an-s', '上段：L1b　下段：Human SL');
    for (let k = 0; k < n; k++) {
      const own = k % 2 === 0, x = X(k);
      s += `<rect x="${x}" y="54" width="${W}" height="34" rx="2" fill="${own ? '#0b6e69' : '#7fb8b3'}"/>`
        + `<rect x="${x}" y="92" width="${W}" height="44" rx="2" fill="${own ? '#e69f00' : '#ffffff'}" stroke="${own ? '#e69f00' : '#c9ced6'}" stroke-dasharray="${own ? '0' : '3 3'}"/>`
        + tx(x + W / 2, 152, 'an-s', own ? '本人' : '對手', 'text-anchor="middle"');
      if (!own) s += `<rect x="${x}" y="92" width="${W}" height="44" rx="2" fill="#f0b45a" opacity="0">${aShow(2.4 + k * 0.08)}</rect>`;
    }
    s += txA(30, 178, 'an-h', '之前：對手手上的 Human SL 欄是 0', 0.6, 2.2)
      + txA(30, 178, 'an-b', 'M14：對手手上也填對手的 Human SL 似然（不加維度）', 2.5);
    const bar = (y, label, v, col, t0) => `<rect x="190" y="${y}" width="0" height="16" rx="3" fill="${col}">${aTo('width', 0, ((v - 0.70) / 0.05 * 380).toFixed(1), t0)}</rect>`
      + tx(182, y + 13, 'an-k', label, 'text-anchor="end"') + txA(190 + (v - 0.70) / 0.05 * 380 + 6, y + 13, 'an-n', v.toFixed(4), t0 + 0.6);
    s += bar(198, '6 模型集成：之前', 0.7288, '#8a93a3', 4.2) + bar(222, '加對手 HSL', 0.7383, '#0b6e69', 4.8)
      + `<line x1="190" y1="194" x2="190" y2="242" stroke="#1d1f24" stroke-width="1"/>`
      + txA(30, 258, 'an-k', '集成 +0.0096；局數越少增益越大：n 5 +0.0129、10 +0.0090、20 +0.0067', 5.8);
    return {steps: ['① 本人手有 L1b 與 Human SL', '② 對手手只有 L1b', '③ 對手手補上 Human SL'],
      svg: `<svg viewBox="0 0 640 268" role="img" aria-label="對手 Human SL 通道的動畫">${s}</svg>`,
      note: '長條是 M14 第 2 步的 CV 實測（folds 1–4，6 對 6），長度從 0.70 起算；方塊是示意。'};
  };

  // 第 10 頁右欄：pub_q_0177 的十級機率 → 期望分；機率最高 2D，期望分最高 3D
  ANIM.decide = () => {
    const L = NUM.levels, p = NUM.decide.p, e = Math.exp(-1);
    const es = p.map((v, c) => v + e * ((p[c - 1] || 0) + (p[c + 1] || 0)));
    const am = p.indexOf(Math.max(...p)), ed = es.indexOf(Math.max(...es));
    const base = 214, X = i => 56 + i * 56, W = 30, Hs = v => v / 0.40 * 170;
    let s = `<line x1="40" y1="${base}" x2="${X(9) + W + 14}" y2="${base}" stroke="#c9ced6"/>` + tx(40, 22, 'an-k', `ct7 對 ${NUM.decide.qid} 的十級機率`);
    L.forEach((n, i) => {
      const h = Hs(p[i]), h2 = Hs(es[i]), t0 = 0.4 + i * 0.06;
      s += `<rect x="${X(i)}" y="${base}" width="${W}" height="0" rx="3" fill="${i === am ? '#e69f00' : '#9aa3b2'}">${aTo('y', base, (base - h).toFixed(1), t0)}${aTo('height', 0, h.toFixed(1), t0)}</rect>`
        + `<rect x="${X(i) - 4}" y="${(base - h2).toFixed(1)}" width="${W + 8}" height="${h2.toFixed(1)}" rx="3" fill="none" stroke="${i === ed ? '#0b6e69' : '#5d6370'}" stroke-width="${i === ed ? 3 : 1.4}" stroke-dasharray="${i === ed ? '0' : '4 3'}" opacity="0">${aShow(3.4 + i * 0.05)}</rect>`
        + tx(X(i) + W / 2, base + 18, 'an-k', n, 'text-anchor="middle"');
    });
    s += txA(X(am) + W / 2, (base - Hs(p[am]) - 8).toFixed(1), 'an-n', p[am].toFixed(3), 1.4, 3.2, 'text-anchor="middle" style="fill:#b06f00"')
      + txA(632, 22, 'an-h', `機率最高：${L[am]}（${p[am].toFixed(3)}）`, 1.6, undefined, 'text-anchor="end"')
      + txA(632, 42, 'an-b', `期望分最高：${L[ed]}（${es[ed].toFixed(3)}）`, 4.4, undefined, 'text-anchor="end"')
      + txA(X(ed) + W / 2, (base - Hs(es[ed]) - 8).toFixed(1), 'an-n', es[ed].toFixed(3), 4.4, undefined, 'text-anchor="middle" style="fill:#0b6e69"')
      + txA(40, 252, 'an-k', '虛線框 ＝ 期望分 ＝ p(c) ＋ 0.368［p(c−1) ＋ p(c+1)］：鄰居的機率也算分', 3.4);
    return {steps: ['① 24 個成員平均後的十級機率', '② 每級的期望分', '③ 選期望分最高的等級'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="期望分決策的動畫">${s}</svg>`,
      note: `機率是 ct7 在 ${NUM.decide.qid} 的實際輸出（submission_rank_ct7_proba.csv）；ct7 的 800 題中有 23 題兩種選法不同。`};
  };

  // ---- 第 5 章 ----
  // 第 11 頁右欄：production gate 上的集成增益曲線（NUM.gate，實測）
  ANIM.gate = () => {
    const G = NUM.gate, K = G.k, X = i => 90 + i * 118, lo = 0.705, hi = 0.737, Y = v => 214 - (v - lo) / (hi - lo) * 180;
    let s = '';
    [0.71, 0.72, 0.73].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="72" y1="${y}" x2="${X(K.length - 1) + 20}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(66, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    K.forEach((k, i) => { s += tx(X(i), 236, 'an-k', `${k} 個`, 'text-anchor="middle"'); });
    s += tx(X(2), 254, 'an-s', '集成的成員數（類別刻度，不等距）', 'text-anchor="middle"');
    const line = (vals, col, t0) => vals.slice(1).map((v, j) => `<line x1="${X(j)}" y1="${Y(vals[j]).toFixed(1)}" x2="${X(j + 1)}" y2="${Y(v).toFixed(1)}" stroke="${col}" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0 + j * 0.45, 0.45)}</line>`).join('')
      + vals.map((v, j) => `<circle cx="${X(j)}" cy="${Y(v).toFixed(1)}" r="4.5" fill="${col}" opacity="0">${aShow(t0 + j * 0.45)}</circle>`
        + txA(X(j), (Y(v) + (col === '#d1343a' ? -10 : 18)).toFixed(1), 'an-n', v.toFixed(3), t0 + j * 0.45 + 0.1, undefined, `text-anchor="middle" style="font-size:11px;fill:${col}"`)).join('');
    s += line(G.ct3, '#5d6370', 0.5) + line(G.ct6, '#d1343a', 3.0)
      + `<rect x="80" y="10" width="14" height="4" rx="2" fill="#5d6370"/>` + tx(100, 16, 'an-s', 'ct3：單調上升')
      + `<rect x="210" y="10" width="14" height="4" rx="2" fill="#d1343a"/>` + tx(230, 16, 'an-s', 'ct6：4 個以後下降')
      + txA(600, 160, 'an-h', '3 個弱 T 成員：' + G.weakT.map(v => v.toFixed(3)).join('／'), 5.4, undefined, 'text-anchor="end"')
      + txA(600, 180, 'an-b', '修正 → ct7：每個成員都有驗證分', 6.6, undefined, 'text-anchor="end"');
    return {steps: ['① ct3 的集成增益：一路上升', '② ct6：單模型較強，集成到 4 個以後下降', '③ 改用 CV 驗證過的成員'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="production gate 的集成增益曲線動畫">${s}</svg>`,
      note: '794 位留出玩家（788 人有題）、4,000 題（prod_heldout.json、gate_diag.json）；gate 的信賴區間約 ±0.016，只看方向。'};
  };

  // 第 12 頁右欄：事先登記比較的配對差與 95% CI（NUM.forest）
  ANIM.forest = () => {
    const F = NUM.forest, RH = 19, top = 40, lo = -0.032, hi = 0.028, XL = 250, XR = 624, X = v => XL + (v - lo) / (hi - lo) * (XR - XL);
    const col = {pass: '#0b6e69', adopt: '#0b6e69', sub: '#b98900', fail: '#d1343a', ref: '#8a93a3'};
    const H = top + F.length * RH + 34, dt = 0.28, tz = 0.3 + F.length * dt + 0.2;   // tz：所有列出現之後才畫通過區（11-11 列數從 18 加到 21）
    let s = `<line x1="${X(0)}" y1="${top - 8}" x2="${X(0)}" y2="${H - 30}" stroke="#1d1f24" stroke-width="1.2"/>`
      + `<line x1="${X(0.005)}" y1="${top - 8}" x2="${X(0.005)}" y2="${H - 30}" stroke="#0b6e69" stroke-width="1.4" stroke-dasharray="5 4"/>`
      + tx(X(0), top - 14, 'an-s', '0', 'text-anchor="middle"') + tx(X(0.005) + 4, top - 14, 'an-s', '門檻 +0.005')
      + [-0.03, -0.02, -0.01, 0.01, 0.02].map(v => tx(X(v), H - 16, 'an-s', (v > 0 ? '+' : '') + v.toFixed(2), 'text-anchor="middle"')).join('');
    F.forEach(([name, d, a, b, st], i) => {
      const y = top + i * RH + 8, t0 = 0.3 + i * dt, c = col[st];
      s += txA(XL - 10, y + 4, 'an-s', name, t0, undefined, 'text-anchor="end" style="font-size:11.5px;fill:#1d1f24"')
        + `<line x1="${X(a).toFixed(1)}" y1="${y}" x2="${X(b).toFixed(1)}" y2="${y}" stroke="${c}" stroke-width="2.4" opacity="0">${aShow(t0)}</line>`
        + `<circle cx="${X(d).toFixed(1)}" cy="${y}" r="4.5" fill="${st === 'adopt' ? '#fff' : c}"${st === 'adopt' ? ` stroke="${c}" stroke-width="2"` : ''} opacity="0">${aShow(t0)}</circle>`;
    });
    s += `<rect x="${X(0.005)}" y="${top - 8}" width="${(XR - X(0.005)).toFixed(1)}" height="${H - 22 - top}" fill="#0b6e69" fill-opacity=".07" opacity="0">${aShow(tz)}</rect>`
      + txA(10, 18, 'an-s', '綠：採用　空心綠：次門檻、依規則採用　黃：次門檻　紅：未通過　灰：雜訊對照', 0.3, undefined, 'style="fill:#1d1f24"')
      + txA(XR, top - 26, 'an-b', '通過區', tz, undefined, 'text-anchor="end"');
    return {steps: ['① 每一列是一個比較（多數事先登記）', '② 點是配對差、線是 95% CI', '③ 過門檻的：對手相關的幾步、讓子 5 維、對比損失、對手錨點'],
      svg: `<svg viewBox="0 0 640 ${H}" role="img" aria-label="預先登記比較的森林圖動畫">${s}</svg>`,
      note: '數字是設計文件 §3.1 與 §4M 的實測（玩家層 bootstrap 的配對差；多數為 folds 1–4，M21b 只用 fold 1、對手 L1b 一列為 fold 0；層級不同：R 層 MLP、單臂或集成，見 5.2 表）。ct4＋ct7 疊加是使用者臨時提問，非事先登記。M33、M73、M82 三列是 11-06 之後通過的（那時的門檻已改成 +0.0025 加上換一組 seed 重做，三列也都超過 +0.005；M73 一列是錨點對 6 成員集成的增益）。'};
  };
  ANIM.forest.dur = 10;

  // ---- 第 6 章 ----
  // 第 13 頁右欄：「普通池的 2D」與「強池的 1D」實力相同，對手也相同 → 棋譜分不出來
  ANIM.wall = () => {
    const Yl = v => 214 - v * 40;   // 實力刻度：0 → 214，每級 40
    const ladder = (x, shift, title, t0, names) => { let o = txA(x, 22, 'an-t', title, t0, undefined, 'text-anchor="middle"');
      names.forEach((n, i) => { const y = Yl(i + shift + 0.5);
        o += `<rect x="${x - 34}" y="${(y - 15).toFixed(1)}" width="68" height="30" rx="4" fill="#eef0f3" stroke="#c9ced6" opacity="0">${aShow(t0 + i * 0.08)}</rect>` + txA(x, (y + 5).toFixed(1), 'an-k', n, t0 + i * 0.08, undefined, 'text-anchor="middle"'); });
      return o; };
    let s = `<line x1="320" y1="40" x2="320" y2="214" stroke="#5d6370" stroke-width="2"/>` + tx(320, 232, 'an-s', '實際棋力', 'text-anchor="middle"')
      + ladder(120, 0, '普通池的掛牌', 0.4, ['1D', '2D', '3D', '4D']) + ladder(520, 1, '強池的掛牌', 0.9, ['1D', '2D', '3D']);   // 強池的 1D 與普通池的 2D 同高
    const yP = Yl(1.5);
    s += `<circle cx="142" cy="${yP}" r="9" fill="#0072b2" opacity="0">${aShow(1.8)}</circle>` + txA(80, yP + 5, 'an-n', '乙：掛 2D', 1.8, undefined, 'text-anchor="end" style="fill:#0072b2"')
      + `<circle cx="498" cy="${yP}" r="9" fill="#d1343a" opacity="0">${aShow(2.2)}</circle>` + txA(560, yP + 5, 'an-n', '甲：掛 1D', 2.2, undefined, 'style="fill:#d1343a"')
      + `<line x1="151" y1="${yP}" x2="310" y2="${yP}" stroke="#0072b2" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.8, 0.5)}</line>`
      + `<line x1="489" y1="${yP}" x2="330" y2="${yP}" stroke="#d1343a" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.8, 0.5)}</line>`
      + `<circle cx="320" cy="${yP}" r="7" fill="#1d1f24" opacity="0">${aShow(3.3)}</circle>` + txA(330, yP - 10, 'an-k', '同一個實力', 3.3);
    [-0.6, -0.25, 0.1, 0.45, 0.7].forEach((d, k) => {
      const y = (yP - d * 40).toFixed(1), t0 = 4.0 + k * 0.12;
      s += `<circle cx="${178 + k * 9}" cy="${y}" r="4" fill="#7fb8b3" opacity="0">${aShow(t0)}</circle><circle cx="${462 - k * 9}" cy="${y}" r="4" fill="#7fb8b3" opacity="0">${aShow(t0)}</circle>`;
    });
    s += txA(320, 252, 'an-k', '對手（綠點）也一樣強 → 兩人的棋譜分布相同', 4.2, 5.6, 'text-anchor="middle"')
      + txA(320, 252, 'an-h', '從棋譜分不出「1D」還是「2D」', 5.9, undefined, 'text-anchor="middle"')
      + txA(640 - 10, 40, 'an-n', '實測：本人與對手誤差相關 0.62', 6.4, undefined, 'text-anchor="end"');
    return {steps: ['① 條件命題：若兩個池的掛牌差一級', '② 甲、乙實力相同，對手也相同', '③ 棋譜相同、掛牌不同'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="共享偏差不可識別的動畫">${s}</svg>`,
      note: '示意：這是條件命題的圖解（若本人與對手的掛牌一起平移）；不需要兩個池，標籤是期間代表值、對手跟著當局掛牌配，也給出同樣的共享偏差（設計文件第 879、885 行）。0.62 是實測（anchor_kalman）。'};
  };

  // 第 14 頁右欄：比對分數（margin）的分布：A 池內、B 池外、測試題（形狀示意，標出的分位數實測）
  ANIM.ident = () => {
    const I = NUM.ident, X = m => 56 + m * 52, base = 210, MAXM = 10.5;
    const lpdf = (m, mu, sg) => m > 0 ? Math.exp(-Math.pow(Math.log(m) - mu, 2) / (2 * sg * sg)) / m : 0;
    const grid = Array.from({length: 85}, (_, i) => 0.02 + i * MAXM / 84);
    const norm = f => { const v = grid.map(f), mx = Math.max(...v); return v.map(x => x / mx); };
    const B = norm(m => lpdf(m, Math.log(I.Bq.q50), 1.04)), Aw = norm(m => lpdf(m, Math.log(I.medWrong), 1.04)), Ac = norm(m => lpdf(m, Math.log(I.medCorrect), 0.55));   // B：對數常態，q50 0.465、q90 ≈ 1.76、q99 ≈ 5.2，接近實測
    const A = Aw.map((v, i) => 0.55 * v + 0.45 * Ac[i]), T = B.map((v, i) => v * 0.985 + 0.015 * Ac[i]);
    const path = (v, h) => grid.map((m, i) => `${i ? 'L' : 'M'}${X(m).toFixed(1)},${(base - v[i] * h).toFixed(1)}`).join(' ');
    let s = `<line x1="50" y1="${base}" x2="${X(MAXM)}" y2="${base}" stroke="#c9ced6"/>`;
    [0, 2, 4, 6, 8, 10].forEach(m => { s += tx(X(m), base + 16, 'an-s', String(m), 'text-anchor="middle"'); });
    s += tx(X(5.2), base + 34, 'an-s', '比對分數（第一名減第二名，margin）', 'text-anchor="middle"')
      + `<path d="${path(B, 150)} L${X(MAXM)},${base} L${X(0.02)},${base} Z" fill="#8a93a3" fill-opacity=".25" stroke="#8a93a3" opacity="0">${aShow(0.4)}</path>`
      + txA(X(1.4), 66, 'an-k', 'B：玩家不在池裡', 0.4)
      + `<path d="${path(A, 120)}" fill="none" stroke="#0b6e69" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.6, 1.0)}</path>`
      + txA(X(I.Bq.q99) - 6, base - 70, 'an-b', `A：在池裡，認對的中位 ${I.medCorrect}`, 2.4, undefined, 'text-anchor="end"')
      + `<line x1="${X(I.Bq.q99).toFixed(1)}" y1="34" x2="${X(I.Bq.q99).toFixed(1)}" y2="${base}" stroke="#1d1f24" stroke-width="1.4" stroke-dasharray="5 4" opacity="0">${aShow(3.4)}</line>`
      + txA(X(I.Bq.q99) + 6, 44, 'an-k', `B 的 q99 ＝ ${I.Bq.q99.toFixed(2)}`, 3.4)
      + `<path d="${path(T, 150)}" fill="none" stroke="#e69f00" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(4.4, 1.0)}</path>`
      + txA(X(1.4), 86, 'an-h', '測試題：和 B 重疊', 5.0)
      + txA(X(I.Bq.q99) + 6, 64, 'an-n', '測試題超過 q99：', 5.8)
      + txA(X(I.Bq.q99) + 6, 82, 'an-n', I.testAboveQ99.map(([n, v]) => `${(v * 100).toFixed(1)}%`).join('／'), 5.8)
      + txA(X(I.Bq.q99) + 6, 100, 'an-s', '（n 5／10／15／20；虛無 1%）', 5.8);
    return {steps: ['① B：不在池裡的玩家', '② A：在池裡的玩家', '③ 測試題的分布與 B 相同'],
      tech: '開局語言模型、31 維風格、局部棋形、方位、讓子五個成分；每題對 8,007 位訓練玩家評分後融合',
      svg: `<svg viewBox="0 0 640 256" role="img" aria-label="身分比對分數分布的動畫">${s}</svg>`,
      note: '示意：曲線形狀是示意（B 用與實測分位數相近的對數常態）；中位數、q99 與超過 q99 的比例是實測（ident_pool_local.json）。'};
  };

  // ---- 第 7 章（11-11 改寫：前三頁是 11-06 之後的三步，第四頁是最終檔與揭露）----
  // 7·1 右欄：16 個決策推移的 Public 解碼（NUM.probes）。長條 ＝ X（差一級的淨對錯題數），由分數差唯一解出
  ANIM.probes = () => {
    const P = NUM.probes, RH = 14, top = 62, X0 = 322, U = 19, H = top + P.length * RH + 24, X = v => X0 + v * U;
    const col = {keep: '#0b6e69', keep2: '#0b6e69', rule: '#b98900', drop: '#d1343a'}, tail = {keep: '　保留', keep2: '　保留', rule: '　刪除', drop: ''};
    const tEnd = 2.0 + P.length * 0.3 + 0.4;
    let s = tx(12, 18, 'an-t', '一次上傳只多一個推移：分數差就是那幾題的對錯')
      + txA(12, 38, 'an-n', 'ct13 − ct9 ＝ +0.0126424 ＝ 8 × 0.63212 ÷ 400', 0.8, undefined, 'style="fill:#0b6e69"')
      + txA(628, 38, 'an-k', '→ 8 題全對', 1.5, undefined, 'text-anchor="end"')
      + tx(160, 54, 'an-s', 'X ＝ 改對 − 改錯', 'text-anchor="end"')
      + [-5, 0, 5].map(v => tx(X(v), 54, 'an-s', (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v), 'text-anchor="middle"')).join('')
      + `<line x1="${X0}" y1="${top - 2}" x2="${X0}" y2="${top + P.length * RH}" stroke="#1d1f24" stroke-width="1.2"/>`;
    P.forEach(([dir, file, nq, x, y, dec, st], i) => {
      const yy = top + i * RH + 7, t0 = 2.0 + i * 0.3, c = col[st], kept = st === 'keep' || st === 'keep2';
      if (kept) s += `<rect x="8" y="${yy - 7}" width="624" height="${RH}" fill="#0b6e69" fill-opacity=".09" opacity="0">${aShow(tEnd)}</rect>`;
      s += txA(160, yy + 4, 'an-s', `${dir}（${nq} 題）`, t0, undefined, `text-anchor="end" style="fill:#1d1f24${kept ? ';font-weight:700' : ''}"`)
        + (x ? `<rect x="${Math.min(X0, X(x))}" y="${yy - 4.5}" width="${Math.abs(x) * U}" height="9" rx="2" fill="${c}" opacity="0">${aShow(t0)}</rect>`
          : `<circle cx="${X0}" cy="${yy}" r="3.5" fill="${c}" opacity="0">${aShow(t0)}</circle>`)
        + txA(x >= 0 ? X(x) + 6 : X(x) - 6, yy + 4, 'an-s', (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x), t0, undefined, `text-anchor="${x >= 0 ? 'start' : 'end'}" style="fill:${c};font-weight:700"`)
        + txA(628, yy + 4, 'an-s', dec + tail[st], t0 + 0.1, undefined, `text-anchor="end"${st === 'drop' ? '' : ` style="fill:${c};font-weight:700"`}`);
    });
    s += txA(12, H - 7, 'an-b', '只留兩個：1D→A（8 題全對）與 4D→5D（使用者決定）', tEnd);
    return {steps: ['① 每個探測檔只比前一檔多一個推移', '② 分數差唯一分解成對錯題數', '③ 16 個方向留下 2 個'],
      svg: `<svg viewBox="0 0 640 ${H}" role="img" aria-label="十六個決策推移的 Public 解碼結果動畫">${s}</svg>`,
      note: 'X 由 400 × ΔPublic ＝ 0.63212 X ＋ 0.36788 Y 唯一解出（設計文件 M52–M58；帳本 m58_decode.py）。括號內是 Public 翻動的題數，右欄是解碼出的對錯。第一列對 ct9，其餘對 ct13。4D→5D 沒達到事先寫的門檻（9 題要 8 對），由使用者決定保留；B→C 依事先登記的規則刪除。'};
  };
  ANIM.probes.dur = 11;

  // 7·2 右欄：對手錨點的機制（上排，示意）與校正表（下排，NUM.anchor，實測）
  ANIM.anchor = () => {
    const A = NUM.anchor, Hd = 'font-size:12.5px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#fff" stroke="#c9ced6"/>${inner}</g>`;
    const arrow = (x1, y, x2, t0) => `<line x1="${x1}" y1="${y}" x2="${x2 - 6}" y2="${y}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.3)}</line>`
      + `<polygon points="${x2},${y} ${x2 - 7},${y - 4} ${x2 - 7},${y + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.3)}</polygon>`;
    // 上排：一局 → 在訓練玩家中比對 → 讀到一個標籤與一個比對分數
    let s = box(10, 12, 170, 88, 0.4, miniBoard(18, 22, 68, 7, [[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 5, 'w'], [5, 3, 'b']])
        + `<text x="96" y="46" style="${Hd}">測試題的一局</text><text x="96" y="68" style="${Sm}">只看對手的落子</text>`)
      + arrow(182, 56, 204, 1.2);
    let dots = '';
    for (let r = 0; r < 3; r++) for (let c = 0; c < 14; c++) dots += `<circle cx="${226 + c * 12}" cy="${54 + r * 13}" r="3.2" fill="#c9ced6"/>`;
    s += box(206, 12, 194, 88, 1.5, `<text x="303" y="36" text-anchor="middle" style="${Hd}">在 8,007 位訓練玩家中比對</text>${dots}`)
      + `<circle cx="${226 + 9 * 12}" cy="67" r="5.5" fill="#0b6e69" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(2.4)}</circle>`
      + arrow(402, 56, 424, 2.9)
      + box(426, 12, 204, 88, 3.2, `<text x="438" y="38" style="${Hd}">最像的那位：標籤 3D</text><text x="438" y="60" style="${Sm}">比對分數 ＝ 第一名 − 第二名</text><text x="438" y="80" style="${Sm}">和冒充者的分布比 → 落在哪一帶</text>`);
    // 下排左：九個帶的 a（比對為真的機率）
    const base = 226, BX = i => 20 + i * 35;
    s += txA(12, 124, 'an-k', 'a：這個比對是真的機率（依冒充者分位帶）', 4.2)
      + `<line x1="12" y1="${base}" x2="338" y2="${base}" stroke="#c9ced6"/>`
      + tx(BX(4) + 12, 255, 'an-s', '冒充者比對分數的分位（%）', 'text-anchor="middle"');
    A.a.forEach((v, i) => {
      const h = Math.max(1.5, v * 84), t0 = 4.4 + i * 0.12;
      s += `<rect x="${BX(i)}" y="${(base - h).toFixed(1)}" width="24" height="${h.toFixed(1)}" rx="2" fill="${v >= 0.5 ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`
        + txA(BX(i) + 12, (base - h - 5).toFixed(1), 'an-n', v ? v.toFixed(2) : '0', t0, undefined, 'text-anchor="middle" style="font-size:10.5px"')
        + tx(BX(i) + 12, 240, 'an-s', A.bands[i], 'text-anchor="middle" style="font-size:10px"');
    });
    // 下排右：真配對時標籤差 d 的倍率 K／P0
    const KX = i => 388 + i * 72;
    s += txA(372, 124, 'an-k', '真配對時，對手標籤落在各級的倍率 K／P0', 6.0)
      + `<line x1="384" y1="${base}" x2="628" y2="${base}" stroke="#c9ced6"/>`
      + `<line x1="384" y1="${base - 26}" x2="628" y2="${base - 26}" stroke="#5d6370" stroke-dasharray="4 3" opacity="0">${aShow(6.0)}</line>`
      + txA(634, base - 30, 'an-s', '背景 ＝ 1', 6.0, undefined, 'text-anchor="end" style="font-size:10.5px"')
      + tx(KX(1) + 24, 255, 'an-s', '對手的標籤比本人', 'text-anchor="middle"');
    ['低一級', '同級', '高一級'].forEach((n, i) => {
      const v = A.kp0[i], h = v * 26, t0 = 6.2 + i * 0.2;
      s += `<rect x="${KX(i)}" y="${(base - h).toFixed(1)}" width="48" height="${h.toFixed(1)}" rx="2" fill="${i === 1 ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`
        + txA(KX(i) + 24, (base - h - 5).toFixed(1), 'an-n', '×' + v.toFixed(1), t0, undefined, 'text-anchor="middle" style="font-size:11px"')
        + tx(KX(i) + 24, 240, 'an-s', n, 'text-anchor="middle"');
    });
    s += txA(12, 274, 'an-b', '題目後驗 × 各局證據 → 期望分決策；只在邊界區動，ct28 的 38 題鎖定', 7.4);
    return {steps: ['① 每一局：用對手的落子認人', '② 比對分數決定信多少（a）', '③ 標籤變成鄰近幾級的倍率，乘進後驗'],
      tech: '認對手：開局、風格、局部棋形、方位、讓子五個成分，加上學出來的條件化比對（下一頁）；校正：冒充者分位帶 × 混合模型，依驗證折交叉擬合',
      svg: `<svg viewBox="0 0 640 282" role="img" aria-label="對手錨點的機制與校正表動畫">${s}</svg>`,
      note: '上排是示意（「3D」只是例子）。下排是實測的校正表：互先局、只用開局模型時的值（設計文件 M71 第 2 步）；換成五成分後，同級的 K 從 0.50 升到 0.54。'};
  };
  ANIM.anchor.dur = 11;

  // 7·3 右欄：左 ＝ 錨點對底的 CV 增益（三種認對手的方法 × 兩組 seed）；右 ＝ Public 讀數（ct28 → ct40／ct42 → ct44）
  ANIM.ladder = () => {
    const A = NUM.anchor, BX = 112, SC = 9300;
    let s = tx(12, 20, 'an-t', '錨點加在集成上的 CV 增益') + tx(12, 37, 'an-s', '深：seed 0–2　淺：seed 6–8');
    A.cv.forEach(([name, src, v1, v2], i) => {
      const y = 52 + i * 46, t0 = 0.5 + i * 0.9;
      s += tx(BX - 8, y + 12, 'an-k', name, 'text-anchor="end" style="font-size:12px;fill:#1d1f24"') + tx(BX - 8, y + 27, 'an-s', src, 'text-anchor="end"')
        + [[v1, '#0b6e69', 0], [v2, '#7fb8b3', 17]].map(([v, c, dy]) => `<rect x="${BX}" y="${y + dy}" width="${(v * SC).toFixed(1)}" height="14" rx="3" fill="${c}" opacity="0">${aShow(t0 + dy / 60)}</rect>`
          + txA((BX + v * SC + 6).toFixed(1), y + dy + 11, 'an-n', '+' + v.toFixed(4), t0 + 0.2 + dy / 60, undefined, 'style="font-size:11px"')).join('');
    });
    s += txA(12, 206, 'an-b', '比對 對 五成分：+0.0058／+0.0071', 3.4) + txA(12, 224, 'an-k', '兩組 seed、八個折全正', 3.4);
    const P = A.pub, PX = i => 424 + i * 60, lo = 0.77, hi = 0.81, PY = v => 200 - (v - lo) / (hi - lo) * 150, fin = NUM.final;
    s += tx(372, 20, 'an-t', 'Public（同一批 400 題）');
    [0.77, 0.78, 0.79, 0.80, 0.81].forEach(g => { const y = PY(g).toFixed(1);
      s += `<line x1="394" y1="${y}" x2="630" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(388, (PY(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    const seg = (i, j, c, dash, t0) => `<line x1="${PX(i)}" y1="${PY(P[i][1]).toFixed(1)}" x2="${PX(j)}" y2="${PY(P[j][1]).toFixed(1)}" stroke="${c}" stroke-width="2.2"${dash ? ' stroke-dasharray="4 4"' : ''} opacity="0">${aShow(t0)}</line>`;
    const T = [4.0, 4.7, 5.5, 6.3];
    s += seg(0, 1, '#d1343a', true, T[1]) + seg(0, 2, '#8a93a3', false, T[2]) + seg(2, 3, '#0b6e69', false, T[3]);
    P.forEach(([name, v, sub], i) => {
      const col = name === 'ct40' ? '#d1343a' : name === fin ? '#0b6e69' : '#5d6370', r = name === fin ? 7 : 5, below = name === 'ct40';
      s += `<circle cx="${PX(i)}" cy="${PY(v).toFixed(1)}" r="${r}" fill="${col}" opacity="0">${aShow(T[i])}</circle>`
        + (name === 'ct42' ? txA(PX(i) - 8, (PY(v) - 6).toFixed(1), 'an-n', v.toFixed(4), T[i] + 0.1, undefined, `text-anchor="end" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`)   // 左上：避開往 ct44 的連線
          : txA(PX(i), (PY(v) + (below ? 18 : name === fin ? -20 : -10)).toFixed(1), 'an-n', v.toFixed(4), T[i] + 0.1, undefined, `text-anchor="middle" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`))
        + tx(PX(i), 218, 'an-k', name, 'text-anchor="middle" style="font-size:12px"') + tx(PX(i), 233, 'an-s', sub, 'text-anchor="middle"');
    });
    const iF = P.findIndex(p => p[0] === fin);
    if (iF >= 0) s += `<circle cx="${PX(iF)}" cy="${PY(P[iF][1]).toFixed(1)}" r="13" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(6.9)}</circle>`;
    s += txA(630, 253, 'an-h', 'ct44 對 ct28 +0.0240；CV 期望 +0.012–0.014', 7.3, undefined, 'text-anchor="end"');
    return {steps: ['① 認對手的方法變強，CV 增益跟著升', '② ct40：先錨點後推移，Public 掉分', '③ 鎖定推移後：ct42、ct44'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="錨點的 CV 增益與 Public 讀數動畫">${s}</svg>`,
      note: 'CV：錨點加在 6 成員集成上的配對增益（folds 1–4、類別平衡；驗證端的錨點密度，測試端約 0.74–0.85 倍）。M82 那次執行裡重算的五成分是 +0.0135／+0.0133。Public 是使用者回報的分數；兩檔差的雜訊約 ±0.008。'};
  };
  ANIM.ladder.dur = 10;

  // 7·4 右欄：Public 分數（橘色那一步是由 Public 回饋選出的兩個推移）
  CHARTS['chart-final'] = () => '<h4>Public 分數（使用者回報；長條從 0.40 起；橘色那一步用了 Public 回饋）</h4>'
    + barRows([['官方 baseline', 0.4631, '0.4631', '#9aa3b2'], ['主檔（MLP）', 0.7148, '0.7148', '#8a93a3'], ['C＋T', 0.7494, '0.7494', '#8a93a3'],
      ['ct7（11-06）', 0.7687835, '0.7687835', '#8a93a3'], ['ct9（對比損失）', 0.7622032, '0.7622032', '#8a93a3'], ['ct28（兩個推移）', 0.7827471, '0.7827471', 'var(--warn)'],
      ['ct42（五成分錨點）', 0.7880062, '0.7880062', '#7fb8b3'], ['ct44（最終檔）', 0.8067107, '0.8067107', 'var(--accent)']], 0.4, 0.82);

  // 7·4 左欄：最後一筆上傳決定 Private（NUM.uploads 的 13 個檔 ＋ 截止前的最後一筆）
  ANIM.last = () => {
    const U = NUM.uploads, n = U.length, X = i => 26 + i * 42, Y = 60, xl = X(n), fin = NUM.final;
    let s = `<line x1="12" y1="${Y}" x2="${xl + 24}" y2="${Y}" stroke="#d9dbe0" stroke-width="3"/>`
      + `<line x1="${xl + 28}" y1="${Y - 34}" x2="${xl + 28}" y2="${Y + 34}" stroke="#a4451c" stroke-width="2" stroke-dasharray="4 3"/>` + tx(xl + 34, Y - 22, 'an-k', '截止');
    U.forEach(([name], i) => { s += `<circle cx="${X(i)}" cy="${Y}" r="10" fill="${name === fin ? '#7fb8b3' : '#c9ced6'}" opacity="0">${aShow(0.3 + i * 0.12)}</circle>`
      + txA(X(i), Y + 28, 'an-s', name, 0.3 + i * 0.12, undefined, 'text-anchor="middle" style="font-size:10.5px"'); });
    // 先演錯的一支、最後停在正確狀態（減少動態效果時停在完成畫面，看到的是最終檔 ✓）
    s += `<circle cx="${xl}" cy="${Y}" r="15" fill="#d1343a" opacity="0">${aShow(2.6, 5.0)}</circle>` + txA(xl, Y + 4, 'an-l', '其他', 2.6, 5.0, 'text-anchor="middle"')
      + `<circle cx="${xl}" cy="${Y}" r="15" fill="#0b6e69" opacity="0">${aShow(5.6)}</circle>` + txA(xl, Y + 4, 'an-l', fin, 5.6, undefined, 'text-anchor="middle"')
      + txA(12, Y + 66, 'an-v an-no', `之後若再傳別的檔、沒有補傳 ${fin} → Private 用錯檔 ✗`, 3.0, 5.0)
      + txA(12, Y + 66, 'an-v an-ok', `Private 只看最後一筆：必須是 ${fin} ✓`, 6.0);
    return {steps: ['① 之前的上傳只影響 Public', '② 截止前最後一筆決定 Private'],
      svg: `<svg viewBox="0 0 640 ${Y + 82}" role="img" aria-label="最後一筆上傳的動畫">${s}</svg>`, note: '示意：只畫模型檔與被保留的檔；截止 11-24 23:59:59；每日 5 次上傳。'};
  };

  //（競賽一：玩家辨識，11-06 加；數字在 NUM.pid）
  // ---- 第 8 章 ----
  CHARTS['chart-pid-score'] = () => '<h4>正解排第幾名得幾分（官方規則）</h4>'
    + barRows([1, 2, 3, 4, 5].map(r => [`第 ${r} 名`, Math.exp(-(r - 1)), Math.exp(-(r - 1)).toFixed(3), r === 1 ? 'var(--accent)' : '#8a93a3']).concat([['不在前 5 名', 0, '0', '#8a93a3']]), 0, 1);

  // ---- 第 9 章 ----
  // 第 17 頁（橫跨整張卡）：五種成分 → 融合 → 一對一配對 → 前 5 名（靜態圖）
  CHARTS['chart-pid-arch'] = () => {
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, fill, stroke, inner, dash) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}"${dash ? ' stroke-dasharray="5 4"' : ''}/>${inner}`;
    const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2 - 7}" y2="${y2}" stroke="#5d6370" stroke-width="2"/><polygon points="${x2},${y2} ${x2 - 8},${y2 - 4.5} ${x2 - 8},${y2 + 4.5}" fill="#5d6370"/>`;
    const C = NUM.pid.parts, top = 18, step = 38, H = top + C.length * step + 22, mid = top + (C.length * step - 8) / 2;
    let s = box(12, mid - 82, 158, 64, '#fff', '#c9ced6', `<text x="91" y="${mid - 56}" text-anchor="middle" style="${Hd}">題目</text><text x="91" y="${mid - 36}" text-anchor="middle" style="${Sm}">同一人 5–20 局（黑）</text>`)
      + box(12, mid + 18, 158, 64, '#fff', '#c9ced6', `<text x="91" y="${mid + 44}" text-anchor="middle" style="${Hd}">候選人 × 400</text><text x="91" y="${mid + 64}" text-anchor="middle" style="${Sm}">每人 100 局（黑）</text>`);
    C.forEach(([name, sub, w], i) => {
      const y = top + i * step, last = w == null;
      s += box(214, y, 300, 30, last ? '#f2f3f6' : '#e3f1ef', last ? '#c9ced6' : '#0b6e69',
        `<text x="226" y="${y + 20}" style="${Hd}">${name}</text><text x="502" y="${y + 20}" text-anchor="end" style="${Sm}">${sub}</text>`, last)
        + (last ? '' : `<text x="522" y="${y + 20}" style="${Sm}">× ${w.toFixed(2)}</text>`);
    });
    s += arrow(172, mid - 50, 210, mid - 50) + arrow(172, mid + 50, 210, mid + 50)
      + `<text x="364" y="${H - 8}" text-anchor="middle" style="${Sm}">每個成分：一張 400 題 × 400 人的 log 分數表；「× 數字」是最終檔的融合權重</text>`
      + arrow(570, mid, 600, mid)
      + box(604, mid - 36, 120, 72, '#fff', '#0b6e69', `<text x="664" y="${mid - 8}" text-anchor="middle" style="${Hd}">逐列標準化</text><text x="664" y="${mid + 12}" text-anchor="middle" style="${Hd}">加權相加</text><text x="664" y="${mid + 29}" text-anchor="middle" style="${Sm}">5 個權重</text>`)
      + arrow(726, mid, 752, mid)
      + box(756, mid - 36, 112, 72, '#fff', '#0b6e69', `<text x="812" y="${mid - 8}" text-anchor="middle" style="${Hd}">一對一配對</text><text x="812" y="${mid + 12}" text-anchor="middle" style="${Sm}">匈牙利法</text><text x="812" y="${mid + 29}" text-anchor="middle" style="${Sm}">決定 Top-1</text>`)
      + arrow(870, mid, 892, mid)
      + box(896, mid - 36, 56, 72, '#0b6e69', '#0b6e69', `<text x="924" y="${mid - 4}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">每題</text><text x="924" y="${mid + 16}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">前 5 名</text>`);
    return `<svg viewBox="0 0 960 ${H}" role="img" aria-label="玩家辨識的架構：五種成分、融合、一對一配對、前 5 名">${s}</svg>`;
  };

  CHARTS['chart-pid-comp'] = () => '<h4>每個成分單獨用時的 Public Top-1（答案已知後離線計算；綠色 ＝ 在新最終候選裡）</h4>'
    + barRows(NUM.pid.comp.map(([n, v, used]) => [n, v, v.toFixed(3), used ? 'var(--accent)' : '#a7afbb']), 0, 1);

  CHARTS['chart-pid-assign'] = () => '<h4>Public：同一份融合分數，每題各選 vs 一對一配對（長條從 0.94 起）</h4>'
    + NUM.pid.pub.map(([n, a, b]) => `<h4>${n}</h4>` + barRows([['每題各選', a, a.toFixed(4), '#8a93a3'], ['一對一配對', b, b.toFixed(4), 'var(--accent)']], 0.94, 1.0)).join('');

  // ---- 第 10 章 ----
  CHARTS['chart-pid-hard'] = () => '<h4>把模擬弄難後的 fusion3（不配對；長條從 0.80 起）</h4>'
    + barRows(NUM.pid.hard.map(([n, v]) => [n, v, v.toFixed(4), n === '隨機 400 人' ? 'var(--accent)' : '#8a93a3']), 0.8, 1.0);

  // ---- 第 11 章 ----
  // 第 21 頁右欄：Public 最後 3 題的 3-環。箭頭 ＝ 候選人從「現在佔著他的題」移到「他真正的題」
  CHARTS['chart-pid-cycle'] = () => {
    const N = NUM.pid.cycle, P = [[320, 62], [526, 226], [114, 226]], W = 212, H = 78;
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:12px;fill:#5d6370', Ok = 'font-size:12px;font-weight:700;fill:#0b6e69';
    const edge = (a, b) => {   // 從 a 框邊到 b 框邊
      const [x1, y1] = P[a], [x2, y2] = P[b], dx = x2 - x1, dy = y2 - y1;
      const cut = Math.min(W / 2 / Math.abs(dx || 1e-9), H / 2 / Math.abs(dy || 1e-9)) + 0.06;
      return [x1 + dx * cut, y1 + dy * cut, x2 - dx * cut, y2 - dy * cut];
    };
    let s = '<defs><marker id="pid-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#a4451c"/></marker></defs>';
    N.forEach(([q, n, top, truth, rk], i) => {
      const [cx, cy] = P[i];
      s += `<rect x="${cx - W / 2}" y="${cy - H / 2}" width="${W}" height="${H}" rx="8" fill="#fff" stroke="#c9ced6"/>`
        + `<text x="${cx}" y="${cy - 16}" text-anchor="middle" style="${Hd}">${q}（n＝${n}）</text>`
        + `<text x="${cx}" y="${cy + 5}" text-anchor="middle" style="${Sm}">原 Top-1：${top}</text>`
        + `<text x="${cx}" y="${cy + 26}" text-anchor="middle" style="${Ok}">正解：${truth}（${rk}）</text>`;
    });
    // 候選人 c 在題 i 是 Top-1、卻是題 j 的正解 → 箭頭 i → j
    N.forEach(([, , top], i) => {
      const j = N.findIndex(r => r[3] === top); if (j < 0) return;
      const [x1, y1, x2, y2] = edge(i, j), mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      s += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#a4451c" stroke-width="2.2" marker-end="url(#pid-ah)"/>`
        + `<text x="${mx.toFixed(1)}" y="${(my + (i === 0 ? -8 : 16)).toFixed(1)}" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#a4451c;paint-order:stroke;stroke:#fff;stroke-width:4px">${top}</text>`;
    });
    s += `<text x="320" y="300" text-anchor="middle" style="${Sm}">箭頭：候選人從目前佔著他的題，移回他真正的題；三個 Top-1 輪轉一次就全對</text>`;
    return `<svg viewBox="0 0 640 312" role="img" aria-label="Public 最後三題錯答形成的 3-環">${s}</svg>`;
  };

  // ---- 第 12 章 ----
  CHARTS['chart-pid-err'] = () => '<h4>Public：不配對時的錯誤率，依 8 個成分中幾個同意 Top-1（實測）</h4>'
    + barRows(NUM.pid.agree.map(([k, nq, w]) => [`${k} 個同意（${nq} 題）`, w / nq, `${w}／${nq}`, w ? '#d1343a' : '#8a93a3']), 0, 1);

  // ---- 名詞註記（11-06 加；使用者：「第一次看到報告的人不知道這啥，希望簡單註記、內容不要太多」）----
  // 每一頁（含推導節）第一次出現的術語加虛線底；滑鼠指向、鍵盤聚焦或手指點一下，顯示一行說明。頁面文字不變。
  // 名詞與說明只改這裡（[術語, 說明]）；英數術語只在前後不是英數、底線、點、斜線時才標，避免標到檔名裡。
  {
    const TERMS = [
      ['KataGo', '目前最強的開源圍棋 AI。這裡只借用它的神經網路：每個局面算一次（不搜尋），取每一手的機率與內部向量。'],
      ['Human SL', 'KataGo 的人類棋風模型（b18c384nbt-humanv0）：給定局面與棋力設定（12k–6d），預測那個棋力的人會下哪裡。'],
      ['humanv0', 'Human SL 的網路權重（b18c384nbt-humanv0）。'],
      ['幽玄の間', '日本棋院的線上對弈平台；本比賽的棋譜都來自這裡（2021–2023）。'],
      ['讓子', '棋力差較多時，弱方執黑、先在星位擺 2–9 顆子；記錄裡寫成 HD[2]–HD[9]。'],
      ['讓先', '棋力只差一點時，弱方執黑先下、白方不貼目；記錄裡寫成 HD[1]。'],
      ['互先', '棋力相近的一般對局：黑方先下、要貼目；記錄裡寫成 HD[0]。'],
      ['HD', '棋譜（SGF）裡的讓子標記：HD[0] 互先、HD[1] 讓先、HD[2–9] 讓 2–9 子。'],
      ['執色', '這位玩家在那一局執黑還是執白。'],
      ['貼目', '黑方先下占便宜，終局時補給白方的目數（例如 6.5 或 7.5 目）。'],
      ['中國規則', '數子計分的規則；這裡只是重播棋局時餵給網路的設定，不是那盤棋實際用的規則。'],
      ['段級位', '級（k）的數字越小越強，1k 之上是 1 段（d），段數越大越強。'],
      ['RP', '幽玄の間的積分（レーティング點數）；每 1,000 點換一個段級位。'],
      ['profile', 'Human SL 的棋力設定，從 12k 到 6d 共 18 個。'],
      ['L1b', '在 KataGo 主幹網路（小幅微調）上，為 10 個等級各接一個落子預測頭，用本比賽的標籤訓練；每手輸出「這一級的人會下這手」的 10 個 log 機率。'],
      ['trunk', '神經網路的主幹：每個局面算出一組內部特徵；我們取整盤與落點兩組，各固定投影成 64 維，當作「這手棋的樣子」。'],
      ['token', '把一手棋整理成的一個特徵向量。'],
      ['C＋T', '兩種讀整局的模型：C 對每一手各用一個小 MLP，再對整局取平均與標準差；T 用擴張卷積，每次看連續約 63 手（前後各約 31 手）。'],
      ['MLP', '多層感知器：最基本的全連接神經網路。'],
      ['集成', '把多個模型的機率平均，降低單一模型的隨機誤差。'],
      ['期望分', '把差一級的部分分也算進去的預期得分；選期望分最高的等級，不一定是機率最高的那個。'],
      ['baseline', '主辦方提供的參考做法與它的分數。'],
      ['Public', '比賽期間排行榜即時顯示的 400 題分數。'],
      ['Private', '截止後才公布的另外 400 題分數，決定最終名次。'],
      ['CV', '交叉驗證：玩家分成 5 份（fold），輪流留一份當考題；分數都來自沒看過那些玩家的模型。'],
      ['fold', '交叉驗證裡的一份玩家；共 5 份，編號 0–4。'],
      ['OOF', 'out-of-fold：每位玩家的預測都來自沒看過他的模型。'],
      ['bootstrap', '把玩家重抽幾千次、每次重算差值，得到差值的 95% 信賴區間。'],
      ['CI', '95% 信賴區間；下界大於 0 才算確定有增益。'],
      // 11-11 加（第 7 章的名詞）
      ['對手錨點', '認出這一局的對手是訓練集裡的哪位玩家，把他的標籤當成本人等級的一次觀測。'],
      ['條件化比對', '學出來的比對：把一局棋和每位候選人自己的棋各變成一個向量，看哪位候選人最接近。'],
      ['邊界區', '模型最看好的兩個等級相鄰、合計機率至少 0.5、兩者相差不到 3 倍的題；對手錨點只在這些題上改決策。'],
      ['冒充者', '對照組：本人明明不在比對庫裡，卻被配到的那位最像的人；用它的比對分數分布，判斷一個比對有多可能是真的。'],
      ['決策推移', '在兩個相鄰等級之間，把其中一級的勝算乘上一個常數（這裡是 2），只改因此翻動的題。'],
      ['對比損失', '訓練時多加的一項：同一位玩家的兩組棋局要得到相近的表示，不同玩家的要分開。'],
      ['Top-1', '排第一的候選人；Top-5 指前五名。'],
      ['一對一配對', '每位候選人只能配給一題；在這個限制下找總分最大的配法。'],
      ['匈牙利法', '求最佳一對一指派的經典演算法。'],
      ['LDA', '線性判別分析：每位候選人一個平均特徵、所有人共用一個變異結構，看題目最像誰。'],
      ['back-off', '先用完整的前文預測下一手；資料不夠時退回較短的前文，再不夠就用全庫平均。'],
      ['n-gram', '用前面幾手預測下一手的計數模型。'],
      ['Dirichlet', '一種平滑：資料少的候選人，機率往全庫的平均靠攏。'],
      ['cosine', '兩個向量夾角的餘弦，越接近 1 表示方向越像。'],
      ['SGF', '圍棋棋譜的標準文字格式。'],
      ['fusion', '玩家辨識最早的融合：開局＋風格兩個成分。'],
      ['fusion3', '開局＋風格＋局部棋形三個成分的融合。'],
      ['fusion3o', 'fusion3 再加原始方向（開在哪個角）。'],
      ['fusion5_raw4', 'fusion3 加前 4 手方向與讓子旗標的版本（沒有進最終檔）。'],
      ['fusion3o_ref', '玩家辨識的最終檔：fusion3o 再加參考玩家相似度。'],
    ];
    const LATIN = /^[A-Za-z0-9]/, BEFORE = /[A-Za-z0-9_./\-]/, AFTER = /[A-Za-z0-9_/]/;
    const edgeOK = (txt, i, t) => !LATIN.test(t) || (!BEFORE.test(txt[i - 1] || ' ') && !AFTER.test(txt[i + t.length] || ' '));
    const SKIP = 'script,style,svg,code,button,textarea,input,.src,.chips,.num,.an-steps,.an-note,.an-tech,.lb-head,h1,h2,.kpi .v,.term';
    const mark = root => {   // root：一頁或一個推導節；每個術語只標第一次出現
      const used = new Set(), nodes = [];
      const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {acceptNode: n => (n.nodeValue.trim() && n.parentElement && !n.parentElement.closest(SKIP)) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT});
      for (let n; (n = tw.nextNode());) nodes.push(n);
      nodes.forEach(node => {
        let cur = node;
        while (cur) {
          const txt = cur.nodeValue;
          let best = null;
          TERMS.forEach(([t, d]) => {
            if (used.has(t)) return;
            let i = txt.indexOf(t);
            while (i >= 0 && !edgeOK(txt, i, t)) i = txt.indexOf(t, i + 1);
            if (i >= 0 && (!best || i < best[0] || (i === best[0] && t.length > best[1].length))) best = [i, t, d];
          });
          if (!best) break;
          const [i, t, d] = best;
          used.add(t);
          const mid = cur.splitText(i), rest = mid.splitText(t.length);
          const span = document.createElement('span');
          span.className = 'term'; span.tabIndex = 0; span.dataset.def = d; span.textContent = t;
          mid.parentNode.replaceChild(span, mid);
          cur = rest;
        }
      });
    };
    // check.py 的 node 假 DOM 沒有這些 API：在那裡整段略過
    if (typeof document.createTreeWalker === 'function' && typeof document.addEventListener === 'function') {
      try {
        document.querySelectorAll('main > .page, main > .lesson').forEach(mark);
        const tip = document.createElement('div');
        tip.className = 'term-tip'; tip.id = 'term-tip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true;
        let on = null;
        const show = el => {
          on = el;
          const host = el.closest('dialog') || document.body;   // 推導彈窗是 modal：說明框要放進彈窗裡才看得到
          if (tip.parentNode !== host) host.appendChild(tip);
          tip.textContent = el.dataset.def; tip.hidden = false; el.setAttribute('aria-describedby', 'term-tip');
          const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, vw = document.documentElement.clientWidth;
          let y = r.bottom + 6;
          if (y + h > window.innerHeight - 8) y = r.top - h - 6;
          tip.style.left = Math.max(8, Math.min(r.left, vw - w - 8)) + 'px';
          tip.style.top = Math.max(8, y) + 'px';
        };
        const hide = () => { if (on) on.removeAttribute('aria-describedby'); on = null; tip.hidden = true; };
        const termOf = e => e.target && e.target.closest ? e.target.closest('.term') : null;
        document.addEventListener('pointerover', e => { const t = termOf(e); if (t && e.pointerType !== 'touch') show(t); });
        document.addEventListener('pointerout', e => { const t = termOf(e); if (t && e.pointerType !== 'touch' && !t.contains(e.relatedTarget)) hide(); });
        document.addEventListener('focusin', e => { const t = termOf(e); if (t) show(t); });
        document.addEventListener('focusout', e => { if (termOf(e)) hide(); });
        document.addEventListener('click', e => { const t = termOf(e); if (t) show(t); else if (!tip.hidden) hide(); });   // 點術語＝顯示（點一下聚焦也會顯示，不做開關以免手機上一閃即逝）；點別處收起
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && !tip.hidden) hide(); });
        window.addEventListener('scroll', hide, {passive: true});
      } catch (err) { console.warn('名詞註記載入失敗：' + err.message); }
    }
  }


  const OUT = {anims: {}, charts: {}, errors: []};
  Object.keys(ANIM).forEach(k => { const f = ANIM[k]; try { AD = (f && f.dur) || 9; HOLD = AD - 0.7; RAW = !!(f && f.raw); const an = f(); OUT.anims[k] = an ? {svg: an.svg, steps: an.steps || [], tech: an.tech || '', note: an.note || '', dur: AD} : null; } catch (e) { OUT.errors.push('ANIM.' + k + ': ' + e.message); } });
  Object.keys(CHARTS).forEach(k => { try { OUT.charts[k] = CHARTS[k](); } catch (e) { OUT.errors.push('CHARTS.' + k + ': ' + e.message); } });
  console.log(JSON.stringify(OUT));
})();
