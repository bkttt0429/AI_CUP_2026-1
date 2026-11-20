  // ===== 第 1–2 章（W1）：任務、難點 ① 相鄰等級分不開、難點 ② 標籤和實力不一致 =====
  // 整段包在區塊裡：輔助函式（ch1Board）不會和別章的同名常數衝突
  // 白話版（2026-11-18 晚，PLAIN.md）：分數一律說成「每 100 題拿幾分」（分數 × 100，取 1 位小數）
  {
  // 本章數字（每筆註明出處）
  //   素材06 = reports/報告素材_2026-11-06.md、素材15 = reports/報告素材_2026-11-15_補充.md、設計 = reports/architecture_v2_2026-11-02.md
  NUM.ch1 = {
    // 最終系統（定案檔用的規則，交叉驗證整體 0.7900）按局數的分數：類別平衡、39,327 題、6,354 位玩家；無限局＝對 1/n 外推
    //   出處：素材15 G.5 表；設計 M183 結果 (A)（0.7401／0.7954／0.8344；局數無限多時 0.862，基礎單獨 0.818）
    //   注意：舊版 NUM.gateByN（0.658／0.725／0.795）是 ct3 集成在 production gate 的分數，口徑不同，不用
    byN: [['5 局', 0.740], ['10 局', 0.795], ['20 局', 0.834]], byNInf: 0.862,
    // 失分預算：0.210＝局數不夠 0.072（34%）＋局數再多也在 0.138（66%）——素材15 G.5；設計 M183 (A)（0.2100＝0.0721＋0.1379）
    loss: [['局數不夠', 0.072, '#0072b2'], ['再多局也補不回', 0.138, '#b98900']], lossAll: 0.210,
    // 同一人兩組不重疊的 40 局各估一次，本人證據誤差的相關（2,524 位有 80 局的玩家，早期模型）——素材06 §8.2；設計第 835 行
    halfCorr: 0.964,
    // 卡片文字另用到：每手證據的相鄰重疊係數 0.86–0.94（設計 §3.2 第 68 行；各邊界 METHOD_NOTES §2.20 第 389 行）、
    //   不經訓練的相鄰 AUC 1D|2D 0.62（METHOD_NOTES §2.4.1(d) 第 56 行）、分身帳號同級 0.67–0.82（素材15 第 309 行）
  };

  // 第 2 頁右欄下方：800 題分成兩個榜（NUM.test）；封面分數換算成「每 100 題」（NUM.live.pub × 100）
  CHARTS['ch1-boards'] = () => `<h4>${NUM.test.pub + NUM.test.priv} 道考題分成兩個排行榜</h4><div class="ch1-boards">`
    + `<div class="c1b c1pub"><b>Public 榜 · ${NUM.test.pub} 題</b><span>比賽中公開：每次上傳，馬上看到分數</span></div>`
    + `<div class="c1b c1priv"><b>Private 榜 · ${NUM.test.priv} 題</b><span>比賽結束才公布：決定名次，以最後一次上傳為準</span></div></div>`
    + `<p class="ch1-conv">分數是平均每題得分（滿分 1），這份簡報說成「每 100 題拿幾分」：封面的 ${NUM.kpi.public}，就是每 100 題拿 ${(NUM.live.pub * 100).toFixed(1)} 分</p>`;

  // 小棋盤：s 是邊長，k 是線數，stones = [[x, y, 'b'|'w'], …]（線的索引）
  const ch1Board = (x, y, s, k, stones, extra = '') => {
    const g = s / (k + 1); let o = `<g ${extra}><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="#dcb36a" stroke="#9c7a3c"/>`;
    for (let i = 1; i <= k; i++) o += `<line x1="${x + g}" y1="${y + i * g}" x2="${x + k * g}" y2="${y + i * g}" stroke="#8a6a2f" stroke-width=".7"/><line x1="${x + i * g}" y1="${y + g}" x2="${x + i * g}" y2="${y + k * g}" stroke="#8a6a2f" stroke-width=".7"/>`;
    stones.forEach(([i, j, c]) => { o += `<circle cx="${(x + i * g).toFixed(1)}" cy="${(y + j * g).toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="${c === 'b' ? '#1d1f24' : '#fff'}" stroke="#1d1f24" stroke-width=".8"/>`; });
    return o + '</g>';
  };

  // 第 2 頁右欄（沿用 go_story 的 task）：一題＝同一個人的幾局 → 模型 → 10 級之一 → 對照標籤（伺服器登記的等級）給分
  ANIM.ch1_task = () => {
    const L = NUM.levels.map(n => /^[A-D]$/.test(n) ? n + " 組" : n), RX = 452, RW = 80, RH = 17, top = 30, pick = 5;
    const ry = i => top + (9 - i) * (RH + 3);
    const games = [[[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 3, 'b']], [[2, 6, 'b'], [6, 2, 'w'], [6, 6, 'b'], [3, 3, 'w']],
      [[6, 2, 'b'], [2, 2, 'w'], [2, 6, 'b'], [5, 5, 'w'], [6, 4, 'b']], [[2, 2, 'b'], [6, 6, 'w'], [2, 6, 'b'], [6, 3, 'w']], [[6, 6, 'b'], [2, 2, 'w'], [6, 2, 'b'], [3, 6, 'w'], [4, 4, 'b']]];
    let s = tx(20, 20, 'an-k', '一題：同一個人下的棋');
    games.forEach((st, i) => {
      const x = 22 + (i % 3) * 64, y = 34 + Math.floor(i / 3) * 76, t0 = 0.4 + i * 0.32;
      s += `<g opacity="0">${aShow(t0)}${ch1Board(x, y, 56, 7, st)}` + tx(x + 28, y + 70, 'an-s', `第 ${i + 1} 局`, 'text-anchor="middle"') + '</g>';
    });
    s += txA(172, 146, 'an-k', '…', 2.0) + txA(22, 204, 'an-n', '5、10、15 或 20 局', 2.2)
      + `<line x1="214" y1="128" x2="292" y2="128" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.6, 0.5)}</line>`
      + `<g opacity="0">${aShow(3.0)}<rect x="296" y="104" width="96" height="48" rx="8" fill="#e3f1ef" stroke="#0b6e69"/>` + tx(344, 133, 'an-b', '模型', 'text-anchor="middle"') + '</g>'
      + `<line x1="392" y1="128" x2="${RX - 8}" y2="${ry(pick) + RH / 2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(3.5, 0.4)}</line>`;
    L.forEach((n, i) => {
      const y = ry(i), t0 = 3.3 + i * 0.04;
      // 被選中的那一格：原本的字在高亮出現時淡出，換成「答 2D」（不重疊）
      s += `<rect x="${RX}" y="${y}" width="${RW}" height="${RH}" rx="4" fill="#eef0f3" stroke="#c9ced6" opacity="0">${aShow(t0)}</rect>`
        + txA(RX + RW / 2, y + 13, 'an-k', n, t0, i === pick ? 3.9 : undefined, 'text-anchor="middle"');
    });
    s += `<rect x="${RX}" y="${ry(pick)}" width="${RW}" height="${RH}" rx="4" fill="#0b6e69" opacity="0">${aShow(4.0)}</rect>`
      + txA(RX + RW / 2, ry(pick) + 13, 'an-w', '答 2D', 4.0, undefined, 'text-anchor="middle"')
      + txA(RX + RW + 8, ry(pick) + 13, 'an-n an-ok', '1 分', 4.9) + txA(RX + RW + 36, ry(pick) + 13, 'an-s', '標籤是 2D', 4.9)
      + txA(RX + RW + 8, ry(pick + 1) + 13, 'an-n', `約 ${NUM.partial} 分`, 5.5) + txA(RX + RW + 8, ry(pick - 1) + 13, 'an-n', `約 ${NUM.partial} 分`, 5.5)
      + txA(RX + RW + 8, ry(pick + 2) + 13, 'an-s', '差更多 0 分', 6.1) + txA(RX + RW + 8, ry(pick - 2) + 13, 'an-s', '差更多 0 分', 6.1)
      + txA(300, 190, 'an-k', '差一級也有分：', 5.5) + txA(300, 208, 'an-k', '約三分之一分', 5.5);
    return {steps: ['① 一題＝同一個人的 5–20 局棋', '② 模型從本賽 10 級裡答一個', '③ 對照標籤（伺服器登記的等級）給分'],
      svg: `<svg viewBox="0 0 640 232" role="img" aria-label="一題的輸入、答案與給分方式的動畫">${s}</svg>`, note: '示意：棋盤是示意圖；給分規則是官方規則。'};
  };

  // 第 3 頁右欄下方：最終系統按局數的分數（NUM.ch1.byN，換成每 100 題；長條從 50 分起）；無限局是推估，用虛線條紋和實測區分
  CHARTS['ch1-byn'] = () => {
    const inf = NUM.ch1.byNInf, w = ((inf - 0.5) / 0.4 * 100).toFixed(1);
    return '<h4>做到最後的系統：看幾局棋，每 100 題拿幾分（模擬考）</h4>'
      + barRows(NUM.ch1.byN.map(([n, v]) => [n, v, (v * 100).toFixed(1) + ' 分', n === '20 局' ? 'var(--accent)' : '#8a93a3']), 0.5, 0.9)
      + `<div class="row"><span>局數無限多（推估）</span><div class="track"><div class="fill" style="width:${w}%;background:repeating-linear-gradient(135deg,#d9dbe0 0 4px,#ffffff 4px 8px);border:1.5px dashed #8a93a3;box-sizing:border-box"></div></div><span class="val">約 ${(inf * 100).toFixed(1)} 分</span></div>`
      + '<p class="ch1-cv">模擬考（交叉驗證）：把一部分訓練玩家藏起來當考題，自己出題自己改。長條從 50 分起畫。</p>';
  };

  // 第 3 頁右欄（沿用 go_story 的 accum）：局數越多，十級的機率越集中；但中心是「棋看起來像幾級」，不一定是標籤
  ANIM.ch1_accum = () => {
    const L = NUM.levels.map(n => /^[A-D]$/.test(n) ? n + " 組" : n), base = 252, H = 192, X = i => 64 + i * 54, W = 36, lab = 5;
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
      + txA(40, 290, 'an-h', '看越多局越集中；但集中的是「棋看起來像幾級」，不一定是標籤', 6.6);
    return {steps: ['① 看 1 局：10 級都有可能', '② 看 5 局：集中到兩三級', '③ 看 20 局：更集中，但不一定落在標籤上'],
      svg: `<svg viewBox="0 0 640 298" role="img" aria-label="看的局數越多，猜測越集中，但集中的地方不一定是標籤的動畫">${s}</svg>`,
      note: '示意：長條高低是示意；下方 5、10、20 局的分數是實測。'};
  };

  // 第 4 頁右欄下方：最終系統每 100 題還丟的 21.0 分從哪來（NUM.ch1.loss，換成每 100 題）
  CHARTS['ch1-loss'] = () => `<h4>做到最後，每 100 題還丟 ${(NUM.ch1.lossAll * 100).toFixed(1)} 分；三分之二看再多局也補不回（模擬考推估）</h4>`
    + barRows(NUM.ch1.loss.map(([n, v, c]) => [n, v, (v * 100).toFixed(1) + ' 分', c]), 0, 0.15);

  // 第 4 頁右欄（合併 go_story 的 rp 與 bias）：上半＝標籤是三年積分起伏中的一個值；下半＝同一人換一批棋重估，還是偏同一邊
  ANIM.ch1_label = () => {
    let s = '';
    // ---- 上半：積分漫步（設計時間 0.4–5.0）----
    const X0 = 64, X1 = 520, N = 64, band = [[26, 56, '3D'], [56, 86, '2D'], [86, 116, '1D']];
    let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    // 固定的轉折點（y）加小抖動：2D → 3D → 2D → 1D → 2D → 3D → 2D → 1D → 2D；2D 帶約占一半時間，與「標籤 2D」一致
    const K = [140, 134, 130, 70, 90, 140, 150, 205, 170, 120, 96, 70, 140, 156, 200, 190, 150].map(y => 26 + (y - 44) / 2);
    const pts = [];
    for (let i = 0; i < N; i++) { const u = i / (N - 1) * (K.length - 1), j = Math.min(K.length - 2, Math.floor(u)), f = u - j; pts.push([X0 + i * (X1 - X0) / (N - 1), K[j] + (K[j + 1] - K[j]) * f + (rnd() - 0.5) * 5]); }
    const pick = [4, 11, 19, 26, 34, 41, 49, 57], inBand = yy => band.find(([a, b]) => yy >= a && yy < b)[2];
    const cnt = {}; pick.forEach(i => { const b = inBand(pts[i][1]); cnt[b] = (cnt[b] || 0) + 1; });
    s += tx(16, 16, 'an-t', '① 積分三年來上上下下，標籤只記一個值')
      + txA(624, 16, 'an-s', '題目只給其中幾局，也不知道是哪天下的', 3.2, undefined, 'text-anchor="end"');
    band.forEach(([a, b, n], i) => {
      s += `<rect x="${X0}" y="${a}" width="${X1 - X0}" height="${b - a}" fill="${i % 2 ? '#f3f8f7' : '#ffffff'}"/>` + tx(X0 - 10, (a + b) / 2 + 4, 'an-k', n, 'text-anchor="end"');
      if (i) s += `<line x1="${X0}" y1="${a}" x2="${X1}" y2="${a}" stroke="#c9ced6" stroke-dasharray="4 4"/>`;
    });
    s += `<rect x="${X0}" y="26" width="${X1 - X0}" height="90" fill="none" stroke="#e3e6ea"/>`
      + tx(X0, 130, 'an-s', '2021') + tx(X1, 130, 'an-s', '2023', 'text-anchor="end"') + tx((X0 + X1) / 2, 130, 'an-s', '時間（題目裡看不到）', 'text-anchor="middle"')
      + `<polyline points="${pstr(pts.map(p => [p[0].toFixed(1), p[1].toFixed(1)]))}" fill="none" stroke="#1d1f24" stroke-width="1.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(0.4, 2.6)}</polyline>`;
    pick.forEach((i, k) => {
      const [px, py] = pts[i], t0 = 3.2 + k * 0.15, c = {'1D': '#0072b2', '2D': '#0b6e69', '3D': '#d1343a'}[inBand(py)];
      s += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" fill="${c}" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(t0)}</circle>`;
    });
    s += `<line x1="${X0}" y1="71" x2="${X1}" y2="71" stroke="#a4451c" stroke-width="3" opacity="0">${aShow(4.5)}</line>`
      + txA(530, 68, 'an-h', '標籤 2D', 4.5) + txA(530, 88, 'an-s', '只有一個值', 4.5)
      + txA(X0, 148, 'an-s', `題目抽到的 8 局：1D 時期 ${cnt['1D'] || 0} 局、2D ${cnt['2D'] || 0} 局、3D ${cnt['3D'] || 0} 局`, 5.0);
    // ---- 下半：誤差跟著人走（設計時間 5.4–9.3）----
    const L = NUM.levels.map(n => /^[A-D]$/.test(n) ? n + " 組" : n), X = v => 76 + v * 55, Y = r => 214 + r * 24, AX = 308;
    const P = [[3, 0.8, -0.18, 0.14], [6, -0.7, 0.16, -0.12], [1, 0.6, -0.2, 0.2], [8, -1.0, -0.15, 0.17]];
    s += tx(16, 174, 'an-t', '② 換一批棋重估，還是偏同一邊') + tx(16, 194, 'an-s', '黑線：標籤　藍點：用前 40 局估　橘點：用另外 40 局估')
      + `<line x1="64" y1="${AX}" x2="${X(9) + 12}" y2="${AX}" stroke="#c9ced6"/>`;
    L.forEach((n, i) => { s += `<line x1="${X(i)}" y1="${AX - 4}" x2="${X(i)}" y2="${AX + 4}" stroke="#c9ced6"/>` + tx(X(i), AX + 18, 'an-k', n, 'text-anchor="middle"'); });
    P.forEach(([lab, b, e1, e2], r) => {
      const y = Y(r), t0 = 5.4 + r * 0.12, x0 = X(lab), xb = X(lab + b);
      const x1 = X(lab + b + e1).toFixed(1), x2 = X(lab + b + e2).toFixed(1), x1b = X(lab + b + e1 * 0.25).toFixed(1), x2b = X(lab + b + e2 * 0.25).toFixed(1);
      s += tx(16, y + 4, 'an-s', `玩家 ${r + 1}`)
        + `<line x1="70" y1="${y}" x2="${X(9) + 8}" y2="${y}" stroke="#eef0f3"/>`
        + `<line x1="${x0}" y1="${y - 9}" x2="${x0}" y2="${y + 9}" stroke="#1d1f24" stroke-width="3" opacity="0">${aShow(t0)}</line>`
        + `<line x1="${x0}" y1="${y}" x2="${xb}" y2="${y}" stroke="#a4451c" stroke-width="2" stroke-dasharray="4 3" opacity="0">${aShow(7.0)}</line>`
        + `<circle cx="${x1}" cy="${y - 4}" r="6" fill="#0072b2" opacity="0">${aShow(6.0)}${aTo('cx', x1, x1b, 8.8, 0.7)}</circle>`
        + `<circle cx="${x2}" cy="${y + 4}" r="6" fill="#e69f00" opacity="0">${aShow(6.5)}${aTo('cx', x2, x2b, 8.8, 0.7)}</circle>`;
    });
    s += txA(624, 174, 'an-h', '兩次估的都偏同一邊', 7.0, 8.4, 'text-anchor="end"')
      + txA(624, 174, 'an-h', '多看幾局：兩點靠近，但不回到標籤', 8.7, undefined, 'text-anchor="end"')
      + txA(624, 194, 'an-n', `兩次偏得多像：${NUM.ch1.halfCorr}（1＝完全一樣）`, 7.4, undefined, 'text-anchor="end"');
    return {steps: ['① 積分會起伏，標籤只記一個值', '② 同一個人換一批棋重估', '③ 還是偏同一邊，多看局也拉不回'],
      svg: `<svg viewBox="0 0 640 334" role="img" aria-label="積分會起伏而標籤只記一個值、同一個人的誤差總是偏同一邊的動畫">${s}</svg>`,
      note: '示意：軌跡和點的位置是示意。0.964 是實測：早期模型、2,524 位各有 80 局的玩家。'};
  };
  ANIM.ch1_label.dur = 12;
  }
