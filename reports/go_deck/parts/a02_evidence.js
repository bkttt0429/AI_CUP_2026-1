  // ===== 第 3 章「每手棋變證據」（全份第 5–7 頁；W2：c02_evidence.html、a02_evidence.js）=====
  // 出處縮寫：素材06 = reports/報告素材_2026-11-06.md；MN = METHOD_NOTES.md；設計 = reports/architecture_v2_2026-11-02.md
  // 白話版（PLAIN.md）：分數一律換成「每 100 題拿幾分」＝分數 × 100；Human SL 在畫面上叫「AI 老師」（PLAIN §7.1）；它的刻度叫「老師尺」，10 個小網路叫「新尺」（§7.2）。
  // 整段包在區塊裡：輔助函式用 const，不和別章的同名函式衝突（所有 a*.js 在同一個函式裡執行）。
  {
  NUM.ch2 = {
    // 第 5 頁：Human SL 的判斷＋小型神經網路，在從沒訓練過的玩家（fold 0）上 0.6113；主辦方範例做法同一批題 0.440 用共用的 NUM.baseline.f0；封面 0.4631 是它的 Public 榜分數（NUM.baseline.pub，考題不同）
    //   （素材06 §6.1 第 0、1 列；MN §2.1 第 29 行、§2.10）
    hslF0: 0.6113,
    // 第 6 頁 (a)：本賽各級本人落子在老師 18 個等級上的平均位置（0 = 12 級 … 11 = 1 級、12 = 1 段 … 17 = 6 段；素材06 §8.5「尺度壓縮」列）
    pos: [5.93, 7.15, 8.07, 9.02, 9.60, 10.03, 10.47, 11.00, 11.72, 12.71],
    // 照名稱該在的位置（D＝10–12 級取中間 11 級 … A＝1–3 級取 2 級；1D–6D＝1–6 段；NUM.span）
    nominal: [1, 4, 7, 10, 12, 13, 14, 15, 16, 17],
    // 1D–4D 只跨 1.4 格（9.60 → 11.00）、只看一局的估計誤差約 ±1.5 格（素材06 §8.5）
    span14: 1.4, noise: 1.5,
    // 換新尺前後的模擬考分數：只用老師的判斷（hsl）0.6192383 → 再加新尺（10 個小網路＝鑑別式等級頭，fused）0.6544787，配對差 +0.0352（畫面 61.9 → 65.4、多 3.5）
    //   （素材06 §6.1 第 1、3 列、§7.1 #1；MN §2.10 第 195 行「對 hsl 0.6192 高 0.035」）
    hslCV: 0.6192383, l1bCV: 0.6544787, l1bGain: 0.0352,
    // (b) 拿掉內部印象 −0.0053、拿掉逐手分數 −0.0003（設計 §4A 第三輪；素材06 第 513 行「C-notrunk／C-noscores」）
    noTrunk: 0.0053, noScores: 0.0003,
    // 每一手的描述：各段長度照原本的大小比例（整盤 64、落點 64、新尺的答案 11、老師的判斷 19、旗標 4；素材06 §4.2）；畫面上不寫維度
    token: [['整盤的印象', 64, '#0072b2'], ['落點的印象', 64, '#e1eefa'], ['新尺的答案', 11, '#0b6e69'], ['老師的判斷', 19, '#b98900'], ['誰下的、第幾手', 4, '#5d6370']],
    // 第 7 頁：標籤和「只看對手的棋算出的平均等級」的排序吻合度 0.948、只看本人的棋 0.934（Spearman；MN §2.18 第 332 行；素材06 第 979 行）
    rhoOpp: 0.948, rhoOwn: 0.934,
    // 讀整局的模型也看對手每一手的老師判斷：6 個模型平均後 +0.0096；5／10／20 局的題 +0.0129／+0.0090／+0.0067（設計 M14 第 2 步，第 710 行；素材06 §7.1 #12）
    ensGain: 0.0096, byN: [['5 局的題', 0.0129], ['10 局的題', 0.0090], ['20 局的題', 0.0067]],
    // 四個補強的模擬考增益：對手等級頭 +0.0191（素材06 §7.1 #2）、差值 +0.0181（#3；MN §2.18）、讓子（修正重播後）+0.0097（#4；MN §2.24）、讀整局也看對手 +0.0096（#12）
    gains: [['對手的棋也用新尺量', 0.0191], ['對手比本人強多少', 0.0181], ['讓了幾子', 0.0097], ['讀整局時也看對手', 0.0096]],
  };
  // 分數 → 「每 100 題拿幾分」（PLAIN.md 第 4 節：× 100、取 1 位小數）
  const p100 = (v, d = 1) => (v * 100).toFixed(d);

  // 小棋盤（沿用 go_story a01_task.js 的 miniBoard）：s 是邊長，k 是線數，stones = [[x, y, 'b'|'w'], …]（線的索引）
  const board = (x, y, s, k, stones) => {
    const g = s / (k + 1); let o = `<g><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="#dcb36a" stroke="#9c7a3c"/>`;
    for (let i = 1; i <= k; i++) o += `<line x1="${x + g}" y1="${(y + i * g).toFixed(1)}" x2="${x + k * g}" y2="${(y + i * g).toFixed(1)}" stroke="#8a6a2f" stroke-width=".7"/><line x1="${(x + i * g).toFixed(1)}" y1="${y + g}" x2="${(x + i * g).toFixed(1)}" y2="${y + k * g}" stroke="#8a6a2f" stroke-width=".7"/>`;
    stones.forEach(([i, j, c]) => { o += `<circle cx="${(x + i * g).toFixed(1)}" cy="${(y + j * g).toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="${c === 'b' ? '#1d1f24' : '#fff'}" stroke="#1d1f24" stroke-width=".8"/>`; });
    return o + '</g>';
  };
  // 畫一條帶箭頭的線（t0 開始畫）
  const arrow = (x1, y1, x2, y2, t0) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.4)}</line>`
    + `<polygon points="${x2},${y2} ${x2 - 7},${y2 - 4} ${x2 - 7},${y2 + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.4)}</polygon>`;

  // 第 5 頁右欄：同一手問老師 18 次 → 連成曲線找最像的等級 → 整題平均成一份摘要（沿用 go_story ANIM.profiles，只換成白話）
  ANIM.ch2_hsl = () => {
    const BX = 18, BY = 30, BS = 168, g = BS / 10;
    const X = i => 232 + i * 21.6, W = 14, base = 168;
    const H = v => Math.max(0, (v + 3.2) / 2.0 * 108);
    const curve = (c, w, top) => Array.from({length: 18}, (_, i) => top - w * (i - c) ** 2);
    const lp = curve(9, 0.02, -1.2);                                        // 示意：真正那一手在 18 個等級下的機會，峰在 3 級
    const others = [curve(6, 0.025, -1.5), curve(12, 0.018, -1.4), curve(8, 0.03, -1.6)];   // 示意：同一題的其他手
    const avg = lp.map((v, i) => (v + others[0][i] + others[1][i] + others[2][i]) / 4);
    const pts = arr => arr.map((v, i) => `${(X(i) + W / 2).toFixed(1)},${(base - H(v)).toFixed(1)}`).join(' ');
    let s = tx(BX, 20, 'an-k', '下這手之前的棋盤')
      + board(BX, BY, BS, 9, [[3, 3, 'b'], [7, 7, 'w'], [7, 3, 'b'], [3, 7, 'w'], [5, 2, 'b'], [6, 7, 'w'], [4, 5, 'b'], [8, 6, 'w']]);
    const mx = BX + 6 * g, my = BY + 4 * g;   // 真正下的這一手（黑）
    s += `<g opacity="0">${aShow(0.4)}<circle cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="#1d1f24"/>`
      + `<circle cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="${(g * 0.72).toFixed(1)}" fill="none" stroke="#a4451c" stroke-width="3"/></g>`
      + txA(BX, 215, 'an-h', '他真正下的這一手', 0.4)
      + arrow(192, 112, 224, 112, 0.7)
      + tx(226, 20, 'an-k', '問 AI 老師 18 次：「如果是這一級的人，會下這裡嗎？」');
    // ① 18 根柱子：每一級各給這一手一個「會下這裡的機會」；② 連成曲線、最高點＝最像的等級（第 ③ 段變淡）
    let dim = '';
    lp.forEach((v, i) => { const h = H(v);
      dim += `<rect x="${X(i).toFixed(1)}" y="${(base - h).toFixed(1)}" width="${W}" height="${h.toFixed(1)}" rx="2" fill="#b98900" opacity="0">${aShow(0.9 + i * 0.07)}</rect>`; });
    dim += `<polyline points="${pts(lp)}" fill="none" stroke="#a4451c" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.6, 0.9)}</polyline>`
      + `<circle cx="${(X(9) + W / 2).toFixed(1)}" cy="${(base - H(lp[9])).toFixed(1)}" r="5" fill="#a4451c" opacity="0">${aShow(3.3)}</circle>`;
    s += `<g>${dim}${aTo('opacity', 1, 0.3, 4.6)}</g>`
      + txA((X(9) + W / 2).toFixed(1), (base - H(lp[9]) - 11).toFixed(1), 'an-h', '最像：老師尺上的 3 級', 3.3, 4.5, 'text-anchor="middle"');
    s += `<line x1="228" y1="${base}" x2="${X(17) + W + 4}" y2="${base}" stroke="#5d6370"/>`;
    [[0, '12 級'], [3, '9 級'], [6, '6 級'], [9, '3 級'], [12, '1 段'], [17, '6 段']].forEach(([i, t]) => { s += tx((X(i) + W / 2).toFixed(1), base + 16, 'an-s', t, 'text-anchor="middle"'); });
    s += txA(232, 202, 'an-s', '橫軸＝老師尺上的級段；柱子越高＝那一級的人越常下這裡', 0.9)
      + txA(232, 220, 'an-s', '老師的級段來自另一個網站，和本賽 10 級不是同一把尺（下一頁講）', 1.2);
    // ③ 同一題的其他手各有一條曲線 → 平均 → 一份「像幾級」的摘要
    others.forEach((c, k) => { s += `<polyline points="${pts(c)}" fill="none" stroke="#5d6370" stroke-width="1.4" stroke-opacity=".5" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(4.8 + k * 0.25, 0.6)}</polyline>`; });
    s += `<polyline points="${pts(avg)}" fill="none" stroke="#0b6e69" stroke-width="3.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(5.8, 0.8)}</polyline>`
      + txA(X(17) + W, 42, 'an-b', '整題所有手的平均', 5.9, undefined, 'text-anchor="end"');
    s += txA(BX, 241, 'an-t', '整題平均起來，得到一份「像幾級」的摘要', 6.4);
    const shape = avg.map(v => H(v) / 108);
    shape.forEach((v, i) => { s += `<rect x="${(BX + i * 13.5).toFixed(1)}" y="248" width="11.5" height="20" rx="2" fill="#b98900" fill-opacity="${(0.25 + 0.75 * v).toFixed(2)}" opacity="0">${aShow(6.6 + i * 0.03)}</rect>`; });
    s += txA(BX + 121, 284, 'an-s', '整條曲線的形狀', 6.6, undefined, 'text-anchor="middle"');
    ['有多像', '最像幾級', '平均像幾級', '幾手棋', '幾局棋', '各局差多少'].forEach((t, j) => {
      const x = 272 + j * 60, t0 = 7.2 + j * 0.12;
      s += `<rect x="${x}" y="248" width="55" height="20" rx="3" fill="#0b6e69" opacity="0">${aShow(t0)}</rect>` + txA(x + 27.5, 284, 'an-s', t, t0, undefined, 'text-anchor="middle"');
    });
    return {steps: ['① 換 18 種等級，各問一次', '② 連成曲線：最高點＝最像的等級', '③ 整題的手平均成一份摘要'],
      svg: `<svg viewBox="0 0 640 294" role="img" aria-label="同一手棋問 AI 老師 18 次：如果是老師尺上 12 級到 6 段的人，會不會下這裡；連成曲線找出最像的等級，整題再平均成一份摘要">${s}</svg>`,
      note: '示意圖：柱子和曲線的高低是示意。實際只記「真正下的那一手」在 18 種等級下有多常見。'};
  };
  ANIM.ch2_hsl.dur = 10;
  CHARTS['ch2-hslbars'] = () => '<h4>從沒訓練過的那批玩家當考題：每 100 題拿幾分（和封面 Public 榜的考題不同）</h4>'
    + barRows([['主辦方範例做法', NUM.baseline.f0 * 100, p100(NUM.baseline.f0) + ' 分', '#5d6370'], ['AI 老師的判斷', NUM.ch2.hslF0 * 100, p100(NUM.ch2.hslF0) + ' 分', 'var(--accent)']], 0, 70);

  // 第 6 頁右欄：① 老師尺把本賽 10 級擠在一起 → ② 加上新尺（10 個小網路）→ ③ 也記下 AI 老師看棋的印象（新寫，白話版）
  ANIM.ch2_scale = () => {
    const N = NUM.ch2, L = NUM.levels, X = i => 34 + i * 33;
    // ① 上段：老師的 18 個等級當刻度
    let s = tx(20, 20, 'an-t', '① 老師尺上，本賽 10 級擠在一起');
    s += `<line x1="${X(0) - 10}" y1="98" x2="${X(17) + 10}" y2="98" stroke="#5d6370"/>`;
    for (let i = 0; i < 18; i++) s += `<line x1="${X(i)}" y1="94" x2="${X(i)}" y2="102" stroke="#5d6370"/>`;
    [[0, '12 級'], [3, '9 級'], [6, '6 級'], [9, '3 級'], [12, '1 段'], [15, '4 段'], [17, '6 段']].forEach(([i, t]) => { s += tx(X(i), 118, 'an-k', t, 'text-anchor="middle"'); });
    // 10 個等級：從照名稱該在的位置滑到老師實際看出來的位置（1D–4D 用橘色）
    L.forEach((n, i) => {
      const x0 = X(N.nominal[i]).toFixed(1), x1 = X(N.pos[i]).toFixed(1), col = i >= 4 && i <= 7 ? '#a4451c' : '#0b6e69';
      s += `<circle cx="${x0}" cy="67" r="6" fill="${col}" opacity="0">${aShow(0.4)}${aTo('cx', x0, x1, 2.0, 1.2)}</circle>`
        + `<text class="an-t" x="${x0}" y="${i % 2 === 0 ? 52 : 88}" text-anchor="middle" opacity="0" style="fill:${col}">${n}${aShow(0.4)}${aTo('x', x0, x1, 2.0, 1.2)}</text>`;
    });
    s += txA(20, 141, 'an-k', '起點：照名稱該在的位置（例如 6D 對到老師尺上的 6 段）', 0.4, 1.9)
      + txA(20, 141, 'an-k', '終點：AI 老師實際看出來的平均位置（實測）', 3.3);
    const a = X(N.pos[4]), b = X(N.pos[7]), c = X(N.pos[5]), w = N.noise * 33;
    s += `<path d="M${a.toFixed(1)} 152 V158 H${b.toFixed(1)} V152" fill="none" stroke="#a4451c" stroke-width="2" opacity="0">${aShow(3.8)}</path>`
      + txA((b + 10).toFixed(1), 163, 'an-h', `1D 到 4D 只差 ${N.span14} 格`, 3.8)
      + `<g opacity="0">${aShow(4.4)}<line x1="${(c - w).toFixed(1)}" y1="181" x2="${(c + w).toFixed(1)}" y2="181" stroke="#d1343a" stroke-width="2"/>`
      + `<line x1="${(c - w).toFixed(1)}" y1="176" x2="${(c - w).toFixed(1)}" y2="186" stroke="#d1343a" stroke-width="2"/><line x1="${(c + w).toFixed(1)}" y1="176" x2="${(c + w).toFixed(1)}" y2="186" stroke="#d1343a" stroke-width="2"/>`
      + `<circle cx="${c.toFixed(1)}" cy="181" r="3" fill="#d1343a"/></g>`
      + txA((c + w + 10).toFixed(1), 185, 'an-k', `只看一局，會晃 ±${N.noise} 格`, 4.4)
      + txA(20, 185, 'an-k', '1 格＝老師尺上相鄰兩級', 4.4);
    // ② 中段：只用老師的判斷 vs 再加新尺（長條從 0 分起算，模擬考每 100 題）
    s += txA(20, 208, 'an-t', '② 換新尺：接上 10 個小網路，用本賽的棋重新學', 5.2);
    const BX0 = 140, sc = v => v / 70 * 400;
    [['老師的判斷', N.hslCV, '#5d6370'], ['老師的判斷＋新尺', N.l1bCV, '#0b6e69']].forEach(([lab, v, col], k) => {
      const y = 222 + k * 22, t0 = 5.6 + k * 0.5, len = sc(v * 100);
      s += txA(20, y + 12, 'an-k', lab, t0)
        + `<rect x="${BX0}" y="${y}" width="0" height="15" rx="3" fill="${col}">${aTo('width', 0, len.toFixed(1), t0)}</rect>`
        + txA((BX0 + len + 6).toFixed(1), y + 12, 'an-n', p100(v) + ' 分', t0 + 0.5);
    });
    s += txA(20, 286, 'an-b', `每 100 題多 ${p100(N.l1bGain)} 分（模擬考）`, 6.8);
    // ③ 下段：AI 老師看棋時的印象 → 濃縮成兩份 → 和兩種判斷一起存成「每一手的描述」
    s += `<line x1="10" y1="300" x2="630" y2="300" stroke="#e3e6ea" stroke-width="1.5"/>`
      + txA(20, 320, 'an-t', '③ 記印象：不只記老師的判斷，也記它看棋時的印象', 7.0);
    let st = '';
    for (let k = 0; k < 16; k++) st += `<rect x="24" y="${(332 + k * 5.5).toFixed(1)}" width="26" height="4.8" fill="#0072b2" fill-opacity="${(0.2 + ((k * 37) % 10) / 13).toFixed(2)}"/>`;
    s += `<g opacity="0">${aShow(7.3)}${st}</g>` + txA(12, 438, 'an-k', 'AI 老師看棋時的印象', 7.3)
      + arrow(58, 376, 108, 376, 7.7) + txA(83, 398, 'an-s', '濃縮', 7.7, undefined, 'text-anchor="middle"')
      + `<g opacity="0">${aShow(8.1)}<rect x="114" y="334" width="90" height="36" rx="4" fill="#0072b2"/></g>` + txA(159, 357, 'an-w', '整盤的印象', 8.1, undefined, 'text-anchor="middle"')
      + `<g opacity="0">${aShow(8.3)}<rect x="114" y="382" width="90" height="36" rx="4" fill="#e1eefa" stroke="#0072b2"/></g>` + txA(159, 405, 'an-n', '落點的印象', 8.3, undefined, 'text-anchor="middle"')
      + arrow(210, 376, 236, 376, 8.7);
    let y0 = 328;
    const labY = [350, 382, 401, 420, 439];
    N.token.forEach(([lab, d, col], k) => {
      const h = Math.max(7, d * 0.55), t0 = 9.0 + k * 0.2, cy = y0 + h / 2;
      s += `<rect x="242" y="${y0.toFixed(1)}" width="26" height="${h.toFixed(1)}" fill="${col}" stroke="${k === 1 ? '#0072b2' : col}" opacity="0">${aShow(t0)}</rect>`
        + `<line x1="270" y1="${cy.toFixed(1)}" x2="276" y2="${(labY[k] - 4).toFixed(1)}" stroke="#5d6370" opacity="0">${aShow(t0)}</line>`
        + txA(280, labY[k], 'an-s', lab, t0);
      y0 += h + 2;
    });
    s += txA(236, 458, 'an-k', '＝每一手的描述', 9.9);
    s += txA(400, 350, 'an-n', `拿掉印象：每 100 題少 ${p100(N.noTrunk)} 分`, 10.1)
      + txA(400, 374, 'an-k', `只拿掉老師和新尺的判斷：只少 ${p100(N.noScores, 2)} 分`, 10.4)
      + txA(400, 404, 'an-b', '印象已經包含了判斷，', 10.8) + txA(400, 424, 'an-b', '還多了棋形的細節', 10.8);
    return {steps: ['① 老師尺把本賽 10 級擠在一起', '② 換一把刻度對的新尺', '③ 也記下 AI 老師看棋的印象'],
      svg: `<svg viewBox="0 0 640 466" role="img" aria-label="上：老師尺上本賽 10 級擠在一起；中：加上新尺後每 100 題多 3.5 分；下：AI 老師看棋時的印象和判斷一起存成每一手的描述">${s}</svg>`,
      note: '圓點位置和分數都是實測。分數是模擬考（把一部分訓練玩家藏起來當考題）每 100 題拿幾分；考題和第 5 頁不同，所以起點是 61.9 不是 61.1。'};
  };
  ANIM.ch2_scale.dur = 13;

  // 第 7 頁右欄：① 對手的段位也透露本人 → ② 讀整局時對手的每一手也請 AI 老師看 → ③ 棋譜越少幫助越大（沿用 go_story ANIM.opp，白話版）
  ANIM.ch2_opp = () => {
    const N = NUM.ch2, n = 14, X = k => 110 + k * 35, W = 25;
    // ① 伺服器配實力相近的對手：只看對手的棋也排得出本人的強弱
    let s = `<g opacity="0">${aShow(0.3)}<circle cx="62" cy="48" r="20" fill="#0b6e69"/><circle cx="198" cy="48" r="20" fill="#e3f1ef" stroke="#0b6e69" stroke-width="2"/></g>`
      + txA(62, 52, 'an-w', '本人', 0.3, undefined, 'text-anchor="middle"') + txA(198, 53, 'an-n', '對手', 0.3, undefined, 'text-anchor="middle"')
      + `<g opacity="0">${aShow(0.6)}<line x1="92" y1="48" x2="168" y2="48" stroke="#5d6370" stroke-width="2"/>`
      + `<polygon points="168,48 161,44 161,52" fill="#5d6370"/><polygon points="92,48 99,44 99,52" fill="#5d6370"/></g>`
      + txA(130, 16, 'an-s', '伺服器會配實力相近的對手', 0.6, undefined, 'text-anchor="middle"')
      + txA(250, 30, 'an-n', `只看對手的棋排本人強弱：和標籤的吻合度 ${N.rhoOpp}`, 1.0)
      + txA(250, 52, 'an-k', `只看本人的棋：和標籤的吻合度 ${N.rhoOwn}`, 1.3)
      + txA(250, 74, 'an-s', '（吻合度 1＝和標籤的高低順序完全一樣）', 1.7);
    // ② 一局棋，本人和對手輪流下：上排＝新尺的答案，下排＝老師的判斷；對手那一格的下排原本是空的
    s += txA(30, 100, 'an-k', '一局棋：本人和對手輪流下，每一手都請 AI 老師和新尺打分數', 2.2)
      + txA(100, 128, 'an-s', '新尺的答案', 2.3, undefined, 'text-anchor="end"')
      + txA(100, 164, 'an-s', '老師的判斷', 2.3, undefined, 'text-anchor="end"');
    for (let k = 0; k < n; k++) {
      const own = k % 2 === 0, x = X(k), t0 = 2.3 + k * 0.04;
      s += `<g opacity="0">${aShow(t0)}<rect x="${x}" y="110" width="${W}" height="28" rx="2" fill="#0b6e69" fill-opacity="${own ? 1 : 0.5}"/>`
        + `<rect x="${x}" y="142" width="${W}" height="36" rx="2" fill="${own ? '#b98900' : '#ffffff'}" stroke="${own ? '#b98900' : '#d9dbe0'}" stroke-dasharray="${own ? '0' : '3 3'}"/></g>`
        + txA(x + W / 2, 194, 'an-s', own ? '本人' : '對手', t0, undefined, 'text-anchor="middle"');
      if (!own) s += `<rect x="${x}" y="142" width="${W}" height="36" rx="2" fill="#b98900" fill-opacity="0.5" opacity="0">${aShow(4.9 + k * 0.06)}</rect>`;
    }
    s += txA(30, 214, 'an-h', '之前：對手的手少了老師的判斷（虛線格）', 2.8, 4.5)
      + txA(30, 214, 'an-b', '補上：對手的每一手也有老師的判斷', 4.8);
    // ③ 補上後的模擬考增益（6 個模型一起平均後量；長條從 0 起算）：棋譜越少，幫助越大
    s += txA(30, 240, 'an-k', `補上後，每 100 題多 ${p100(N.ensGain)} 分；棋譜越少，幫助越大：`, 6.0);
    N.byN.forEach(([lab, v], k) => {
      const y = 250 + k * 18, t0 = 6.4 + k * 0.4, len = v * 100 / 1.5 * 300;
      s += txA(182, y + 12, 'an-k', lab, t0, undefined, 'text-anchor="end"')
        + `<rect x="190" y="${y}" width="0" height="14" rx="3" fill="#b98900">${aTo('width', 0, len.toFixed(1), t0)}</rect>`
        + txA((190 + len + 6).toFixed(1), y + 12, 'an-n', `+${p100(v)} 分`, t0 + 0.5);
    });
    return {steps: ['① 對手的段位也透露本人', '② 對手的每一手也請 AI 老師看', '③ 棋譜越少，幫助越大'],
      svg: `<svg viewBox="0 0 640 306" role="img" aria-label="伺服器配實力相近的對手，所以只看對手的棋也排得出本人強弱；讀整局時對手的每一手也請 AI 老師看，棋譜越少幫助越大">${s}</svg>`,
      note: '分數是模擬考（把一部分訓練玩家藏起來當考題），6 個模型一起平均後量；方塊是示意。'};
  };
  ANIM.ch2_opp.dur = 11;
  CHARTS['ch2-oppbars'] = () => '<h4>四個補強各多幾分（模擬考，每 100 題）</h4>'
    + barRows(NUM.ch2.gains.map(([l, v], k) => [l, v * 100, '+' + p100(v) + ' 分', k === 3 ? '#b98900' : 'var(--accent)']), 0, 2);
  }
