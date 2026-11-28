  // ---- 兩項競賽檢討：第 4 頁（預估和實際）、第 5 頁（兩道門＋Public 走勢）、第 6 頁（落差拆三段）的動畫 ----
  // 數字全部取自 10_num.js 的 NUM.pid／NUM.go；這裡只放版面座標。輔助函式要在 ANIM 函式裡呼叫（引擎呼叫前才設定 AD／HOLD）。
  const RV = {ok: '#0b6e69', okSoft: '#bfe0db', okPale: '#e3f1ef', hot: '#a4451c', hotSoft: '#fbeee6', ink: '#1d1f24', mute: '#5d6370',
    faint: '#9aa0aa', line: '#d9dbe0', grey: '#c9ced6', priv: '#0072b2'};
  const RVBIG = 'font-size:17px;font-weight:700';
  const rvBox = (x, y, w, h, t0, stroke = RV.line, fill = '#ffffff') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${fill}" stroke="${stroke}" stroke-width="1.3" opacity="0">${aShow(t0)}</rect>`;
  const rvArrow = (x1, y, x2, t0, c = RV.mute) => `<g opacity="0">${aShow(t0)}<line x1="${x1}" y1="${y}" x2="${x2 - 6}" y2="${y}" stroke="${c}" stroke-width="2.2" stroke-linecap="round"/>`
    + `<polygon points="${x2},${y} ${x2 - 8},${y - 5} ${x2 - 8},${y + 5}" fill="${c}"/></g>`;
  const rvDown = (x, y1, y2, t0, c = RV.hot) => `<g opacity="0">${aShow(t0)}<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2 - 7}" stroke="${c}" stroke-width="2.4" stroke-linecap="round"/>`
    + `<polygon points="${x},${y2} ${x - 5},${y2 - 9} ${x + 5},${y2 - 9}" fill="${c}"/></g>`;

  // 4 右欄：兩把刻度相同的尺（0.01 ＝ 50 px），各自以賽前的中心對齊。淡色＝九成把握、深色＝八成、橘線＝實際成績；右邊徽章＝實際和中心差多少
  ANIM.rv_est = () => {
    const K = RV, PX = 5000;
    const ruler = (o, name, sub, y0, t0, dec, foot, good) => {
      const X = v => 330 + (v - o.est) * PX, ty = y0 + 84, th = 22, xa = X(o.priv).toFixed(1), F = v => v.toFixed(dec);
      const diff = Math.abs(o.priv - o.est).toFixed(dec === 3 ? 3 : 2);
      let s = rvBox(12, y0, 616, 172, t0) + txA(26, y0 + 28, 'an-t', name, t0) + txA(26 + name.length * 14 + 10, y0 + 28, 'an-s', sub, t0);
      // 九成把握（淡色）＋每 0.01 一格的刻度＋兩端數字
      s += `<rect x="${X(o.ci90[0]).toFixed(1)}" y="${ty}" width="${(X(o.ci90[1]) - X(o.ci90[0])).toFixed(1)}" height="${th}" rx="4" fill="${K.okSoft}" opacity="0">${aShow(t0 + 0.4)}</rect>`;
      for (let k = Math.ceil(o.ci90[0] * 100 - 1e-9); k <= Math.floor(o.ci90[1] * 100 + 1e-9); k++)
        s += `<line x1="${X(k / 100).toFixed(1)}" y1="${ty + th}" x2="${X(k / 100).toFixed(1)}" y2="${ty + th + 5}" stroke="${K.faint}" stroke-width="1.2" opacity="0">${aShow(t0 + 0.4)}</line>`;
      s += txA((X(o.ci90[0]) - 8).toFixed(1), ty + 16, '', F(o.ci90[0]), t0 + 0.6, undefined, `text-anchor="end" style="font-size:15px;font-weight:700;fill:${K.ink}"`)
        + txA((X(o.ci90[1]) + 8).toFixed(1), ty + 16, '', F(o.ci90[1]), t0 + 0.6, undefined, `style="font-size:15px;font-weight:700;fill:${K.ink}"`);
      // 八成把握（深色）＋中心
      s += `<rect x="${X(o.ci80[0]).toFixed(1)}" y="${ty}" width="${(X(o.ci80[1]) - X(o.ci80[0])).toFixed(1)}" height="${th}" rx="3" fill="${K.ok}" opacity="0">${aShow(t0 + 1.0)}</rect>`
        + txA(330, ty - 12, '', '預估中心 ' + F(o.est), t0 + 1.3, undefined, `text-anchor="middle" style="${RVBIG};fill:${K.ok}"`);
      // 實際成績：橘色粗線（白邊，在深色條上也看得清楚）＋大字＋一行小字
      s += `<line x1="${xa}" y1="${ty + 1}" x2="${xa}" y2="${ty + th + 14}" stroke="#ffffff" stroke-width="9" stroke-linecap="round" opacity="0">${aShow(t0 + 2.3)}</line>`
        + `<line x1="${xa}" y1="${ty + 1}" x2="${xa}" y2="${ty + th + 14}" stroke="${K.hot}" stroke-width="5" stroke-linecap="round" opacity="0">${aShow(t0 + 2.3)}</line>`
        + txA(xa, ty + th + 34, '', '實際 ' + o.priv.toFixed(4), t0 + 2.6, undefined, `text-anchor="middle" style="${RVBIG};fill:${K.hot}"`)
        + txA(xa, ty + th + 52, 'an-s', foot, t0 + 2.9, undefined, 'text-anchor="middle"');
      // 徽章：差多少
      s += `<rect x="538" y="${ty - 4}" width="82" height="30" rx="15" fill="${good ? K.okPale : K.hotSoft}" stroke="${good ? K.ok : K.hot}" stroke-width="1.3" opacity="0">${aShow(t0 + 3.2)}</rect>`
        + txA(579, ty + 16, '', '差 ' + diff, t0 + 3.2, undefined, `text-anchor="middle" style="font-size:15px;font-weight:700;fill:${good ? K.ok : K.hot}"`);
      return {s, diff};
    };
    const P = NUM.pid, G = NUM.go;
    const a = ruler(P, '玩家識別', '競賽一', 14, 0.3, 3, '依離線驗證與題目結構：' + P.est.toFixed(3), true);
    const b = ruler(G, '棋力預測', '競賽二', 198, 4.6, 2, '僅依離線驗證：' + G.cv.toFixed(3), false);
    return {steps: ['① 玩家識別：預估 ' + P.est.toFixed(3), '② 實際 ' + P.priv.toFixed(4) + '，誤差 ' + a.diff, '③ 棋力預測：預估 ' + G.est.toFixed(2), '④ 實際 ' + G.priv.toFixed(4) + '，誤差 ' + b.diff],
      svg: `<svg viewBox="0 0 640 384" role="img" aria-label="兩圖刻度相同。玩家識別：賽前 90% 區間 ${P.ci90[0]} 至 ${P.ci90[1]}、預估中心 ${P.est}，實際 ${P.priv.toFixed(4)}，誤差 ${a.diff}。棋力預測：賽前 90% 區間 ${G.ci90[0]} 至 ${G.ci90[1]}、預估中心 ${G.est}，實際 ${G.priv.toFixed(4)}，誤差 ${b.diff}">${a.s}${b.s}</svg>`,
      note: '淺色為賽前 90% 區間，深色為 80% 區間，橘線為 2026-11-25 公布的 Private 成績。兩圖刻度相同，每格 0.01。'};
  };
  ANIM.rv_est.dur = 10.5;

  // 5 右欄：上框＝新做法 → 第一道門（模擬考）→ 第二道門（換一批題重考）→ 採用；下面一條「Public 只當否決線」
  //   下框＝當過定案的 13 個檔在 Public 的分數（依時間排）；空心圓＝比前一版略低、照樣採用的兩次；虛線＝Private 成績
  ANIM.rv_gate = () => {
    const K = RV, P = NUM.pid, L = P.ladder;
    let s = rvBox(12, 10, 616, 176, 0.3) + txA(26, 34, 'an-t', '新方法須通過兩階段檢核，方予上傳', 0.3);
    [0, 1, 2].forEach(i => { s += `<rect x="${28 + i * 5}" y="${66 + i * 6}" width="78" height="46" rx="7" fill="#eef1f4" stroke="${K.grey}" stroke-width="1.2" opacity="0">${aShow(0.5 + i * 0.12)}</rect>`; });
    s += txA(77, 106, 'an-n', '新方法', 0.8, undefined, 'text-anchor="middle"');
    const gate = (x, t0, l1, l2, l3) => `<rect x="${x}" y="58" width="150" height="76" rx="9" fill="${K.okPale}" stroke="${K.ok}" stroke-width="1.5" opacity="0">${aShow(t0)}</rect>`
      + txA(x + 75, 82, 'an-n', l1, t0 + 0.1, undefined, `text-anchor="middle" style="fill:${K.ok}"`)
      + txA(x + 75, 102, 'an-k', l2, t0 + 0.2, undefined, `text-anchor="middle" style="fill:${K.ink}"`)
      + txA(x + 75, 120, 'an-k', l3, t0 + 0.2, undefined, `text-anchor="middle" style="fill:${K.ink}"`);
    s += rvArrow(122, 96, 142, 1.1) + gate(146, 1.3, '第一階段：離線驗證', '使用未曾用過的題目', '須優於現行採用檔')
      + rvArrow(300, 96, 322, 2.2) + gate(326, 2.4, '第二階段：重複驗證', '更換題組再驗證一次', '兩次均優於現行檔')
      + rvArrow(480, 96, 502, 3.3)
      + `<rect x="506" y="66" width="110" height="60" rx="9" fill="${K.ok}" opacity="0">${aShow(3.5)}</rect>`
      + txA(561, 93, '', '採用', 3.6, undefined, 'text-anchor="middle" style="font-size:17px;font-weight:700;fill:#ffffff"')
      + txA(561, 113, '', '隨後上傳', 3.6, undefined, 'text-anchor="middle" style="font-size:12px;fill:#ffffff"');
    // 否決線
    s += `<line x1="561" y1="126" x2="561" y2="146" stroke="${K.hot}" stroke-width="1.6" stroke-dasharray="3 3" opacity="0">${aShow(4.3)}</line>`
      + `<rect x="146" y="146" width="470" height="28" rx="14" fill="${K.hotSoft}" stroke="${K.hot}" stroke-width="1.3" opacity="0">${aShow(4.3)}</rect>`
      + txA(381, 165, 'an-n', 'Public 分數僅作否決依據：低於現行採用檔 ' + P.veto + ' 以上方予退回', 4.4, undefined, `text-anchor="middle" style="fill:${K.hot}"`);
    // 下框：Public 走勢
    const X = i => 70 + i * (520 / (L.length - 1)), Y = v => 372 - (v - 0.87) / 0.08 * 132, t1 = 5.2, t2 = 5.8, td = 2.2;
    s += rvBox(12, 196, 616, 200, t1) + txA(26, 220, 'an-t', '歷次採用的 ' + L.length + ' 份提交檔之 Public 分數', t1);
    [0.88, 0.90, 0.92].forEach(v => {
      s += `<line x1="64" y1="${Y(v).toFixed(1)}" x2="600" y2="${Y(v).toFixed(1)}" stroke="#eceef1" stroke-width="1" opacity="0">${aShow(t1 + 0.2)}</line>`
        + txA(58, (Y(v) + 4).toFixed(1), 'an-s', v.toFixed(2), t1 + 0.2, undefined, 'text-anchor="end"');
    });
    s += `<polyline points="${L.map((r, i) => X(i).toFixed(1) + ',' + Y(r[1]).toFixed(1)).join(' ')}" fill="none" stroke="${K.ok}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t2, td)}</polyline>`;
    L.forEach((r, i) => {
      const t0 = t2 + td * i / (L.length - 1);
      s += r[2] ? `<circle cx="${X(i).toFixed(1)}" cy="${Y(r[1]).toFixed(1)}" r="5" fill="#ffffff" stroke="${K.hot}" stroke-width="2.4" opacity="0">${aShow(t0)}</circle>`
        : `<circle cx="${X(i).toFixed(1)}" cy="${Y(r[1]).toFixed(1)}" r="4" fill="${K.ok}" opacity="0">${aShow(t0)}</circle>`;
    });
    // 日期（同一天的檔合成一個標籤）
    const days = [...new Set(L.map(r => r[0]))];
    days.forEach(d => { const xs = L.map((r, i) => r[0] === d ? X(i) : null).filter(v => v !== null);
      s += txA((xs.reduce((p, q) => p + q, 0) / xs.length).toFixed(1), 390, 'an-s', d, t1 + 0.3, undefined, 'text-anchor="middle"'); });
    const last = L.length - 1;
    s += txA(80, (Y(L[0][1]) + 15).toFixed(1), 'an-n', L[0][1].toFixed(4), t2, undefined, `style="fill:${K.ok}"`)
      + txA(600, (Y(L[last][1]) + 18).toFixed(1), 'an-n', L[last][1].toFixed(4), t2 + td, undefined, `text-anchor="end" style="fill:${K.ok}"`);
    // 空心圓的說明＋Private 虛線
    s += `<circle cx="306" cy="340" r="5" fill="#ffffff" stroke="${K.hot}" stroke-width="2.4" opacity="0">${aShow(t2 + td + 0.4)}</circle>`
      + txA(318, 344, 'an-s', '低於前版，因通過離線驗證而照常採用', t2 + td + 0.4, undefined, `style="fill:${K.hot}"`);
    const yp = Y(P.priv).toFixed(1), t3 = t2 + td + 1.2;
    s += `<line x1="64" y1="${yp}" x2="600" y2="${yp}" stroke="${K.priv}" stroke-width="1.8" stroke-dasharray="6 4" opacity="0">${aShow(t3)}</line>`
      + txA(80, (+yp - 7).toFixed(1), 'an-n', 'Private ' + P.priv.toFixed(4) + '（11-25 公布）', t3 + 0.2, undefined, `style="fill:${K.priv}"`);
    return {steps: ['① 第一階段：離線驗證', '② 第二階段：重複驗證', '③ Public 僅作否決依據', '④ Public ' + L[0][1].toFixed(4) + ' → ' + L[last][1].toFixed(4)],
      svg: `<svg viewBox="0 0 640 404" role="img" aria-label="新方法先通過離線驗證，再更換題組重複驗證，兩次均優於現行採用檔方予採用並上傳；Public 分數僅作否決依據，低於現行採用檔 ${P.veto} 以上方予退回。歷次採用的 ${L.length} 份提交檔之 Public 分數由 ${L[0][1].toFixed(4)} 升至 ${L[last][1].toFixed(4)}，其中兩次略低於前版仍照常採用；Private ${P.priv.toFixed(4)}">${s}</svg>`,
      note: '下圖每一點為一份曾採用的提交檔，依時間由左至右排列；空心圓為 Public 略低於前版的兩次（低 0.0009、0.0001）。藍色虛線為 2026-11-25 公布的 Private 成績。'};
  };
  ANIM.rv_gate.dur = 11.5;

  // 6 右欄：四根柱子，由左到右一段一段拿掉只存在 Public 的分數；最後一條虛線＝模擬考 0.790，Private 就停在這條線上
  ANIM.rv_bridge = () => {
    const K = RV, G = NUM.go, B = G.bridge, D = G.drops;
    const Y = v => 348 - (v - 0.76) / 0.11 * 290, bx = [110, 250, 390, 530], bw = 86, fill = [K.ok, '#2f837e', '#4d9993', K.priv];
    const cap = [['排行榜顯示的', 'Public'], ['扣除回推標籤', '後的 Public'], ['Public 題組的', '離線驗證估計'], ['Private', '實際成績']];
    const tb = [0.8, 2.4, 4.4, 6.4];
    let s = txA(22, 30, 'an-t', '同一份答案於兩榜相差 ' + (G.pub - G.priv).toFixed(3), 0.3)
      + `<line x1="100" y1="348" x2="626" y2="348" stroke="${K.grey}" stroke-width="1.4" opacity="0">${aShow(0.4)}</line>`;
    B.forEach((r, i) => {
      const y = Y(r[1]), cx = bx[i] + bw / 2;
      s += `<rect x="${bx[i]}" y="${y.toFixed(1)}" width="${bw}" height="${(348 - y).toFixed(1)}" rx="4" fill="${fill[i]}" opacity="0">${aShow(tb[i])}</rect>`
        + txA(cx, (y + 22).toFixed(1), '', r[1].toFixed(4), tb[i] + 0.2, undefined, 'text-anchor="middle" style="font-size:15px;font-weight:700;fill:#ffffff"')
        + txA(cx, 366, 'an-s', cap[i][0], tb[i] + 0.2, undefined, `text-anchor="middle" style="fill:${K.ink}"`)
        + txA(cx, 381, 'an-s', cap[i][1], tb[i] + 0.2, undefined, `text-anchor="middle" style="fill:${K.ink}"`);
    });
    const hotBig = `style="font-size:15px;font-weight:700;fill:${K.hot}"`, ink = `style="fill:${K.ink}"`;
    // 第一段：回推標籤
    s += rvDown(262, Y(B[0][1]) + 4, Y(B[1][1]) - 4, tb[1] + 0.3)
      + txA(272, 112, '', '−' + D[0].toFixed(4), tb[1] + 0.5, undefined, hotBig)
      + txA(272, 128, 'an-s', '回推標籤', tb[1] + 0.5, undefined, ink) + txA(272, 143, 'an-s', '僅對 Public 有效', tb[1] + 0.5, undefined, ink);
    // 第二段：Public 那批題多出來的
    s += rvDown(402, Y(B[1][1]) + 4, Y(B[2][1]) - 4, tb[2] + 0.3)
      + txA(412, 196, '', '−' + D[1].toFixed(3), tb[2] + 0.5, undefined, hotBig)
      + txA(412, 212, 'an-s', 'Public 題組', tb[2] + 0.5, undefined, ink) + txA(412, 227, 'an-s', '的超額分數', tb[2] + 0.5, undefined, ink);
    // 第三段：兩榜難度（太小，不畫箭頭）
    s += txA(573, 236, '', '−' + D[2].toFixed(3), tb[3] + 0.3, undefined, `text-anchor="middle" ${hotBig}`)
      + txA(573, 252, 'an-s', '兩榜難度差', tb[3] + 0.3, undefined, `text-anchor="middle" ${ink}`);
    // 模擬考的水準線
    const yc = Y(G.cv).toFixed(1);
    s += `<line x1="20" y1="${yc}" x2="630" y2="${yc}" stroke="${K.ink}" stroke-width="1.6" stroke-dasharray="6 4" opacity="0">${aShow(7.8)}</line>`
      + txA(8, (+yc - 7).toFixed(1), 'an-n', '離線驗證 ' + G.cv.toFixed(3), 8.0);
    s += txA(618, 30, 'an-n', D[0].toFixed(4) + '＋' + D[1].toFixed(3) + '＋' + D[2].toFixed(3) + '＝' + (D[0] + D[1] + D[2]).toFixed(4), 9.0, undefined, `text-anchor="end" style="fill:${K.hot}"`);
    return {steps: ['① 排行榜上的 Public ' + B[0][1].toFixed(4), '② 扣除回推標籤', '③ 扣除 Public 超額分數', '④ Private ' + B[3][1].toFixed(4) + '＝離線驗證'],
      svg: `<svg viewBox="0 0 640 392" role="img" aria-label="棋力預測：排行榜顯示的 Public ${B[0][1].toFixed(4)}；扣除回推標籤 ${D[0].toFixed(4)} 後為 ${B[1][1].toFixed(4)}；再扣除 Public 題組的超額分數 ${D[1].toFixed(3)} 後為 ${B[2][1].toFixed(4)}；兩榜難度差 ${D[2].toFixed(3)}，Private ${B[3][1].toFixed(4)}，與離線驗證 ${G.cv.toFixed(3)} 一致">${s}</svg>`,
      note: '三段合計 ' + (D[0] + D[1] + D[2]).toFixed(4) + '。其中 ' + D[1].toFixed(3) + ' 相當於 400 題中的 ' + (G.excess.picked + G.excess.luck + G.excess.scattered).toFixed(1) + ' 分：依 Public 挑選的規則 ' + G.excess.picked.toFixed(1) + ' 分、最後一次模型更新的偶然增益 ' + G.excess.luck.toFixed(1) + ' 分、其餘零散差異 ' + G.excess.scattered.toFixed(1) + ' 分。'};
  };
  ANIM.rv_bridge.dur = 11.5;

  // 7 右欄：上框＝一把尺，左端是我們的 Private、右端是榜上最高分；每一列是一項測過／查過的原因：
  //   淡色長條＝要補的差距，綠色＝這一項補回的部分，空心圓＝沒有補回。下框＝沒做過的兩件事。數字在 NUM.go.gapTests。
  ANIM.rv_gap = () => {
    const K = RV, G = NUM.go, T = G.gapTests, gap = G.top - G.priv;
    const X0 = 300, X1 = 610, PX = (X1 - X0) / gap, pts = gap * 400, nq = Math.round(pts / (1 - Math.exp(-1)));
    let s = rvBox(12, 10, 616, 236, 0.3) + txA(26, 34, 'an-t', '已測試或查證的項目：均無法彌補差距', 0.3);
    s += txA(X0, 60, 'an-n', '本隊 ' + G.priv.toFixed(4), 0.6, undefined, `text-anchor="middle" style="fill:${K.priv}"`)
      + txA(X1, 60, 'an-n', '最高分 ' + G.top.toFixed(4), 0.9, undefined, `text-anchor="end" style="fill:${K.ok}"`)
      + `<line x1="${X0}" y1="72" x2="${X1}" y2="72" stroke="${K.ink}" stroke-width="1.6" opacity="0">${aShow(0.9)}</line>`
      + `<line x1="${X0}" y1="66" x2="${X0}" y2="236" stroke="${K.priv}" stroke-width="1.4" stroke-dasharray="3 3" opacity="0">${aShow(0.6)}</line>`
      + `<line x1="${X1}" y1="66" x2="${X1}" y2="236" stroke="${K.ok}" stroke-width="1.4" stroke-dasharray="3 3" opacity="0">${aShow(0.9)}</line>`
      + txA((X0 + X1) / 2 + 8, 60, '', '差距 ' + gap.toFixed(3), 1.4, undefined, `text-anchor="middle" style="font-size:15px;font-weight:700;fill:${K.hot}"`)
      + txA(26, 60, 'an-s', '相當於 400 題中 ' + pts.toFixed(1) + ' 分，約 ' + nq + ' 題', 1.8);
    const ys = [100, 132, 164, 196, 228], tr = [2.7, 3.5, 5.2, 6.0, 7.0];
    T.forEach((r, i) => {
      const y = ys[i], t0 = tr[i];
      s += txA(26, y, 'an-k', r[0], t0, undefined, `style="fill:${K.ink}"`)
        + txA(288, y, 'an-n', r[1], t0 + 0.3, undefined, `text-anchor="end" style="fill:${r[2] ? K.ok : K.hot}"`);
      if (r[2] === null) return;
      s += `<rect x="${X0}" y="${y - 11}" width="${X1 - X0}" height="12" rx="6" fill="#eef1f4" opacity="0">${aShow(t0)}</rect>`;
      if (r[2] > 0) {
        const w = r[2] * PX;
        s += `<rect x="${X0}" y="${y - 11}" width="${w.toFixed(1)}" height="12" rx="6" fill="${K.ok}" opacity="0">${aShow(t0 + 0.3)}</rect>`
          + txA((X0 + w + 10).toFixed(1), y, 'an-n', '尚差 ' + (gap - r[2]).toFixed(3), t0 + 0.6, undefined, `style="fill:${K.hot}"`);
      } else s += `<circle cx="${X0}" cy="${y - 5}" r="5.5" fill="#ffffff" stroke="${K.hot}" stroke-width="2.2" opacity="0">${aShow(t0 + 0.3)}</circle>`;
    });
    const tb = 8.2;
    s += rvBox(12, 256, 616, 138, tb, K.hot, K.hotSoft) + txA(26, 280, 'an-t', '尚未進行的兩項工作', tb, undefined, `style="fill:${K.hot}"`);
    const todo = (x, t0, l1, l2) => `<rect x="${x}" y="292" width="290" height="58" rx="9" fill="#ffffff" stroke="${K.hot}" stroke-width="1.4" stroke-dasharray="6 4" opacity="0">${aShow(t0)}</rect>`
      + txA(x + 14, 316, 'an-n', l1, t0 + 0.1) + txA(x + 14, 336, 'an-s', l2, t0 + 0.2, undefined, `style="fill:${K.ink}"`);
    s += todo(24, tb + 0.4, '第二種特徵來源', '現有模型的特徵均出自同一個 Human SL 網路')
      + todo(326, tb + 1.0, '針對棋力預測蒐集外部或自製資料', '僅使用野狐一份資料，且係為玩家識別而選')
      + txA(26, 376, 'an-s', '上述兩項的效益尚無法估計；最高分隊伍的作法須待其報告公開。', tb + 1.8);
    return {steps: ['① 差距 ' + gap.toFixed(3) + '，約 ' + nq + ' 題', '② 加大模型、增加資料：無增益', '③ 外部棋譜、撤回規則：仍不足', '④ 尚未進行的兩項工作'],
      svg: `<svg viewBox="0 0 640 404" role="img" aria-label="本隊 Private ${G.priv.toFixed(4)}，排行榜最高分 ${G.top.toFixed(4)}，差距 ${gap.toFixed(3)}，相當於 400 題中多答對約 ${nq} 題。已測試或查證的五項：${T.map(r => r[0] + ' ' + r[1]).join('；')}。尚未進行的兩項工作：第二種特徵來源、針對棋力預測蒐集外部或自製資料">${s}</svg>`,
      note: '淺色長條為待彌補的差距 ' + gap.toFixed(3) + '，綠色為該項可補回的部分，空心圓表示未補回。「撤回規則」的 ' + T[3][1] + ' 為模型自估值（約 3 分），無法由 Private 總分驗證。'};
  };
  ANIM.rv_gap.dur = 11.5;

  // 8 右欄：上框＝棋譜 → Human SL 網路（同一副眼鏡）→ 三種用它的模型 → 模擬考；下面一行是模擬考的走勢。
  //   中框＝四次檢查（三次仍透過同一個網路，一次換了引擎但量的是對錯）。下框＝漏掉的：另一副眼鏡。數字在 NUM.go.climb／lensChecks。
  ANIM.rv_lens = () => {
    const K = RV, G = NUM.go, C = G.climb, L = G.lensChecks;
    let s = rvBox(12, 10, 616, 146, 0.3) + txA(26, 34, 'an-t', '所有模型的特徵均出自同一個網路', 0.3);
    [0, 1, 2].forEach(i => { s += `<rect x="${26 + i * 5}" y="${62 + i * 6}" width="74" height="42" rx="6" fill="#eef1f4" stroke="${K.grey}" stroke-width="1.2" opacity="0">${aShow(0.5 + i * 0.1)}</rect>`; });
    s += txA(73, 100, 'an-n', '棋譜', 0.7, undefined, 'text-anchor="middle"')
      + rvArrow(114, 91, 138, 0.9)
      + `<rect x="140" y="56" width="168" height="70" rx="10" fill="${K.okPale}" stroke="${K.ok}" stroke-width="1.8" opacity="0">${aShow(1.1)}</rect>`
      + txA(224, 78, 'an-n', 'Human SL 網路', 1.2, undefined, `text-anchor="middle" style="fill:${K.ok}"`)
      + txA(224, 97, 'an-k', '公開的圍棋 AI', 1.3, undefined, `text-anchor="middle" style="fill:${K.ink}"`)
      + txA(224, 115, 'an-k', '唯一的特徵來源', 1.3, undefined, `text-anchor="middle" style="fill:${K.ok}"`)
      + txA(26, 146, 'an-s', '離線驗證 ' + C.map(r => r[1].toFixed(3) + '（' + r[0] + '）').join('→ '), 1.9);
    s += rvArrow(312, 91, 330, 3.0);
    ['等級模型（兩類架構）', '各等級的著手預測頭', '對手辨識網路'].forEach((t, i) => {
      s += `<rect x="334" y="${50 + i * 29}" width="160" height="24" rx="12" fill="#ffffff" stroke="${K.grey}" stroke-width="1.3" opacity="0">${aShow(3.2 + i * 0.4)}</rect>`
        + txA(414, 67 + i * 29, 'an-k', t, 3.2 + i * 0.4, undefined, `text-anchor="middle" style="fill:${K.ink}"`);
    });
    s += rvArrow(498, 91, 516, 4.4)
      + `<rect x="520" y="60" width="98" height="62" rx="9" fill="${K.ok}" opacity="0">${aShow(4.6)}</rect>`
      + txA(569, 84, '', '離線驗證', 4.7, undefined, 'text-anchor="middle" style="font-size:12px;fill:#ffffff"')
      + txA(569, 108, '', G.cv.toFixed(3), 4.7, undefined, 'text-anchor="middle" style="font-size:17px;font-weight:700;fill:#ffffff"');
    // 中框：四次檢查
    s += rvBox(12, 164, 616, 136, 5.5) + txA(26, 188, 'an-t', '四項檢驗均顯示其特徵無遺漏', 5.5);
    L.forEach((r, i) => {
      const x = [24, 326][i % 2], y = [198, 246][Math.floor(i / 2)], t0 = 5.9 + i * 0.6, c = r[2] ? K.ok : K.priv;
      s += `<rect x="${x}" y="${y}" width="290" height="42" rx="8" fill="#f6f7f9" stroke="${K.line}" stroke-width="1.2" opacity="0">${aShow(t0)}</rect>`
        + txA(x + 12, y + 18, 'an-n', r[0], t0 + 0.1) + txA(x + 12, y + 34, 'an-s', r[1], t0 + 0.2, undefined, `style="fill:${K.ink}"`)
        + `<rect x="${x + 206}" y="${y + 10}" width="76" height="22" rx="11" fill="${r[2] ? K.okPale : '#dbe9f4'}" stroke="${c}" stroke-width="1.1" opacity="0">${aShow(t0 + 0.3)}</rect>`
        + txA(x + 244, y + 25, '', r[2] ? '同一網路' : '評估對象不同', t0 + 0.3, undefined, `text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:${c}"`);
    });
    // 下框：漏掉的
    s += rvBox(12, 308, 616, 88, 8.4, K.hot, K.hotSoft) + txA(26, 332, 'an-t', '未涵蓋：獨立的特徵來源', 8.4, undefined, `style="fill:${K.hot}"`)
      + txA(26, 354, 'an-k', '未曾測試不經 Human SL、同樣估計「著手近似何種段位」的特徵來源。', 8.8, undefined, `style="fill:${K.ink}"`)
      + txA(26, 374, 'an-s', '例如他人公開釋出的棋力估計模型，或以 100 萬局競賽棋譜從頭訓練的模型。其效益尚無法估計。', 9.4);
    return {steps: ['① 帶來最大幅度的進步', '② 各層模型均以其為基礎', '③ 四項檢驗均無異狀', '④ 未涵蓋：獨立的特徵來源'],
      svg: `<svg viewBox="0 0 640 404" role="img" aria-label="棋譜先經 Human SL 網路產生特徵，再交由等級模型、各等級的著手預測頭與對手辨識網路使用，離線驗證 ${G.cv.toFixed(3)}；離線驗證依序為 ${C.map(r => r[1].toFixed(3) + '（' + r[0] + '）').join('、')}。四項檢驗：${L.map(r => r[0] + '：' + r[1]).join('；')}；前三項仍以同一網路為基礎。未涵蓋：不經 Human SL、同樣估計著手近似何種段位的特徵來源，未曾測試">${s}</svg>`,
      note: 'Human SL 網路即 KataGo 作者公開的 humanv0。四個離線驗證分數的計算方式不盡相同（依序為官方範例、單一模型、多模型平均、完整規則），僅供觀察趨勢。「同一網路」指該項檢驗仍以 Human SL 網路的特徵為基礎；「評估對象不同」指改用另一引擎，但其評估的是著手優劣，而非著手近似何種段位。'};
  };
  ANIM.rv_lens.dur = 11.5;
