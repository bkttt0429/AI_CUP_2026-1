  // ---- 第 6 章 從 Public 到 Private（第 13–15 頁，W5）；第 7 章收穫、揭露（第 16–17 頁）只用表格與方框、沒有動畫 ----
  // 白話版（PLAIN.md）：動畫字不用 X／Y、p 值、後驗、推移、護欄這些詞；「分」在本章兩種用法：
  //   「400 題總分」＝ Public 分數 × 400（1 分＝多猜中一題；Public 0.0025）；「每 100 題」＝ Public 分數 × 100。
  // 出處縮寫（只寫在這裡與講稿；投影片上不印出處碼）：設計 = reports/architecture_v2_2026-11-02.md（以 M 號找）；
  //   素材15 = reports/報告素材_2026-11-15_補充.md；素材06 = reports/報告素材_2026-11-06.md。
  NUM.ch5 = {
    // 從分數差回推的實例：[改了哪些題, 400 題總分變化（＝400 × Public 分數差）, 猜中淨增 X, 「猜中或差一級」淨增 Y, 回推出來, 學到什麼]
    //   400 × 差 ＝ 0.632 × X ＋ 0.368 × Y（猜中 1＝0.632＋0.368）；改動少時只有一組整數解（設計 M54；排行榜無條件捨去：M202）
    //   1D→A：ct13 − ct9 ＝ 0.7748456 − 0.7622032 ＝ +0.0126424 ＝ 8 × 0.63212 ÷ 400（設計「ct13 Public」，M52 之後；素材15 B）
    //   5D→6D：ct15 − ct13 ＝ −0.0072409 ＝（−4 × 0.63212 − 1 × 0.36788）÷ 400；7 題裡 1 題 6D、5 題 5D、1 題 4D（設計「ct15 Public」）
    //   11 題：ct104 − ct104b ＝ −2.264 分 ＝ (−3, −1)；ct104 多做的「A 改別級 9 題＋5D→6D 2 題」（設計 M130 之後的 ct104b 讀數）
    //   定案 ct116 − ct112 ＝ +2.5285 分，(+4, 0)；pub_q_0157 原本被算成 2D、權重 0.99，讀數證明不是（設計 M211；素材15 H.5、H.6）
    //   ct118d − ct116 ＝ +6.0570 分，(+9, +1)；只改 Public 18 題，Private 與 ct116 相同（設計 M220；素材15 J.7）
    ledger: [
      ['8 題：1D 改成 A', '+5.057', 8, 0, '8 題都由差一級變猜中', '全對，定案不再改'],
      ['7 題：5D 改成 6D', '−2.896', -4, -1, '只對 1 題', '寫成禁止規則'],
      ['11 題：A 改別級、5D 改 6D', '−2.264', -3, -1, '淨少猜中 3 題', '兩條禁止規則得證'],
      ['定案檔 ct116：Public 改 7 題', '+2.528', 4, 0, '多猜中 4 題', '總數準，單題不一定'],
      ['ct118d：只改 Public 18 題', '+6.057', 9, 1, '多猜中 9 題', 'Private 一題沒動'],
    ],
    // Public 各級精確題數：5 個「整份答同一級」的上傳解出（results/trunk_seq/es/public_composition.json；設計 M202、M206；素材15 H.3）
    comp: [['D', 51], ['C', 44], ['B', 49], ['A', 56], ['1D', 24], ['2D', 34], ['3D', 35], ['4D', 36], ['5D', 36], ['6D', 35]],
    // 11-18 四批（不看答案的規則圈出的題）：[方向, Public 題數, Private 對應題數, X, Y, 結果]
    //   規則與門檻（淨分 ≥ 0.25 × 題數＝每改 4 題淨多 1 分；> 0 但不到 → 只改 Public；≤ 0 → 放棄）：設計 M214；素材15 I.2
    //   讀數：設計 M220；素材15 J.7。淨分＝0.632X＋0.368Y（對定案檔 ct116，單位是 400 題總分）
    bands: [['2D 改 3D', 7, 10, -3, 0, '放棄'], ['3D 改 4D', 6, 10, -1, -1, '放棄'], ['C 改 B', 7, 7, 2, -1, '只改 Public'], ['5D 改 4D', 4, 7, -2, 0, '放棄']],
    // 對照：較早真的改到 Private 的三次 [方向, Public 試改的結果, Private 怎麼改, 依據, 顏色]
    //   1D→A：Public 8 題全對，Private 照改 10 題（設計「ct13 Public」、M54、M201 (4)）
    //   4D→5D：Public 9 題對 7（X +5），Private 照改 11 題；沒達事前登錄門檻（9 題 ≥ 8：設計 M58 第二天「五筆皆未達事先登錄門檻（9 題 ≥ 8…）」、上傳排程 修訂 3；規則判 drop，P(signal) 0.43），使用者 11-09 02:30「可以賭」（上傳排程 ct28 列「9／11」、M58、M63）；
    //          11-17 試探 probe_4D5D_locked：3 題裡 2 題是 4D → 參賽者把 Private 6 題改回 4D（設計 M206、M209；素材15 H.5）
    //   A→B：被「原本答 A 的不改」擋住的 Public 7 題改 B 淨多對 2 題（事前門檻 +3，未達）→ 參賽者只放寬 Private 把握最高的 2 題（設計 M206、M209；素材15 H.5）
    moved: [['1D 改 A', '8 題全對', '照改 10 題', '全對，保留', '#0b6e69'],
      ['4D 改 5D', '9 題對 7', '照改 11 題，後來改回 6 題', '沒達門檻，參賽者決定', '#a4451c'],
      ['A 改 B（被擋下的題）', '7 題淨多對 2 題', '先改最有把握的 2 題，11-23 再改其餘 4 題', '沒達門檻，參賽者決定的例外', '#a4451c']],
    //   11-23 使用者「好 改那4題」：同一帶其餘 4 題 priv_q_0062／0125／0204／0325 也由 A 改 B（合併後驗本來就答 B，B 的機率 0.50–0.64），
    //          定案檔改成 ct118e_a2b4（Public 400 列與 ct118e 相同）；自估約 +0.4 到 +0.6 分（400 題總分），實際在 ±2.5 分之間
    // 兩榜像不像：六項不看答案的比較（題目層置換檢定；設計 M201 (1)；素材15 H.2）。[白話名稱, 看不出差別?]
    //   p 值依序 0.67（讓子局比例）、0.76（每局認出的對手數）、0.41（合併後驗最大機率）、0.32（後驗平均等級）、0.78（ct112 各級答案數）、
    //   0.064（基礎模型最大機率 0.647 對 0.626：Private 題略難，自估約 −0.01＝每 100 題約少 1 分）；只抓得到每級約 5 個百分點以上的差（設計 M203 (3)）
    sim: [['讓子棋的比例', true], ['認得出對手的比例', true], ['最終模型的把握', true], ['模型猜的平均等級', true], ['答案的等級分布', true], ['不認對手時的把握', false]],
    // 預估（每 100 題）：
    //   78.2 = M201 (2) ct112 在合併後驗下的 Private 自估（隱含各級等權，最保守）
    //   80.4–81.6 = M203 (2) 0.804（全部題、量到的組成、規則本身不含推移）到 M204 0.8159（R0）；中間的 0.8106 是 R1b（含兩條推移＝ct112 做法）
    //   中心 81–83：素材15 H.2（11-17，以量到的組成重估；對象是 ct112）；九成把握 77–84、單半標準差 0.014：M201 (5)（較早的計算，當時中心 0.80–0.82；與 81–83 同一次的區間來源沒有給）
    //   ct116 比 ct112 多改的 Private 8 題：M209「依合併後驗 6 題改回 4D 值 +1.43 分；兩題 A→B 依 Public 的比例估約 +0.8 分」（400 題總分；無法驗證，不加進預估）
    //   78.2 最保守的原因：M203「M201 用模型自己的後驗（隱含各級等權的先驗）…偏保守」；M201 (4)「若其實只有後驗認為的四成多：就是自估的 0.782」
    //   11-21 結案時的重算（設計「結案開始：定案檔與 Private 的期待」，M264；results/r_variants/m264_private_estimate.json）取代上面的數字：
    //     定案答案（ct116）在合併後驗下的自估 0.7872（抽樣標準差 0.0142；已含 Private 基礎層略難約 0.016）；下緣 0.793（Public 多出來的全是運氣）、
    //     上緣 0.829（Private 也多出和 Public 一樣的 +0.042）；中心約 0.81（0.80–0.82），八成 0.785–0.835、九成 0.78–0.84（標準差約 0.019）。
    //     11-23 多改的 4 題 A→B 自估約 +0.001–0.0015，不改變這些區間。舊的 est [81, 83]、ci90 [77, 84] 只留在上面的註解裡
    //   11-25 公布的 Private 成績（使用者回報；NUM.final.priv）：0.7902471 ＝（276 ＋ 109 × e^-1）÷ 400。在九成範圍內、比中心 0.81 低 0.02、
    //     比自估 0.787 高 0.003。Public 0.8587 − Private 0.7902 ＝ 0.0685 ＝ 0.0255（回推標籤，只在 Public）＋ 0.042（Public 多出的：
    //     用 Public 挑的兩批改動 5.0 分、最後一步 ct109→ct112 的運氣 5.3 分、分散的 6.5 分，400 題總分；設計 M201）＋ 0.001（兩榜難度與 12 題改動）
    est: [80, 82], ci90: [78, 84], self: 78.7, comparable: 83.3, actual: 79.02471,
    ladder: [['模型自己估（最保守）', [78.2], '#5d6370', '78.2'], ['用 Public 量到的各級題數重算', [80.4, 81.6], '#0b6e69', '80.4–81.6']],
  };

  // 本章小工具（名稱都以 ch5 開頭，避免和別章衝突）：確定性亂數（點陣用）、打勾、鎖、框
  const ch5rng = seed => () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const ch5pick = (rnd, n, k) => { const out = new Set(); while (out.size < k) out.add(Math.floor(rnd() * n)); return out; };
  const ch5check = (x, y, c, t0) => `<polyline points="${x},${y} ${x + 4},${y + 4} ${x + 11},${y - 5}" fill="none" stroke="${c}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity="0">${aShow(t0)}</polyline>`;
  const ch5lock = (x, y, s, c, t0) => `<g opacity="0">${aShow(t0)}<path d="M${x + s * 0.22},${y} v${-s * 0.26} a${s * 0.28},${s * 0.28} 0 0 1 ${s * 0.56},0 v${s * 0.26}" fill="none" stroke="${c}" stroke-width="${(s * 0.12).toFixed(1)}"/>`
    + `<rect x="${x}" y="${y}" width="${s}" height="${s * 0.8}" rx="${s * 0.12}" fill="${c}"/><circle cx="${x + s / 2}" cy="${y + s * 0.36}" r="${s * 0.09}" fill="#ffffff"/></g>`;
  const ch5box = (x, y, w, h, t0, stroke = '#d9dbe0') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#ffffff" stroke="${stroke}" stroke-width="1.3" opacity="0">${aShow(t0)}</rect>`;
  const CH5 = {ok: '#0b6e69', near: '#b98900', bad: '#d1343a', grey: '#c9ced6', ink: '#1d1f24', mute: '#5d6370', hot: '#a4451c'};
  const CH5BIG = 'font-size:20px;font-weight:700';

  // 13 右欄：收銀台反推。① 三個單價標籤 ② 例一框：8 格由「差一級」逐一翻成「猜中」，計數器加到 +5.057＝排行榜讀數
  //   ③ 例二框：7 格只有 1 格變猜中 → 禁止規則 ④ 方塊堆：每格 1 題，堆高＝Public 各級題數。另外三次試探（11 題、ct116、ct118d）在講稿。
  ANIM.ch5_ledger = () => {
    const C = NUM.ch5.comp, L = NUM.ch5.ledger, K = CH5, SQ = 28, GAP = 34;
    const tags = [['猜中 1 分', K.ok, '#e3f1ef'], ['差一級 約 0.368 分', K.near, '#fbf0d2'], ['猜錯 0 分', K.bad, '#fbe9e9']];
    let s = '';
    // ① 單價標籤
    tags.forEach(([lab, c, bg], i) => {
      const x = 12 + i * 144, t0 = 0.3 + i * 0.2;
      s += `<rect x="${x}" y="8" width="136" height="28" rx="14" fill="${bg}" stroke="${c}" stroke-width="1.5" opacity="0">${aShow(t0)}</rect>`
        + txA(x + 68, 27, 'an-n', lab, t0, undefined, `text-anchor="middle" style="fill:${c}"`);
    });
    s += txA(448, 27, 'an-k', '單價固定，像蘋果、橘子', 0.9, undefined, `style="fill:${K.ink}"`);
    // ② 例一框：1D 改 A，8 題全由差一級變猜中（每格 +0.632）
    const t1 = 1.4, tf = i => 1.9 + i * 0.3;
    s += ch5box(12, 48, 302, 152, t1) + txA(26, 72, 'an-t', '例一：8 題 1D 改成 A', t1);
    let g = '';
    for (let i = 0; i < 8; i++) g += `<rect x="${26 + i * GAP}" y="84" width="${SQ}" height="${SQ}" rx="5" fill="${K.near}">${aSeq('fill', [[0, K.near], [tf(i), K.ok], [AD, K.ok]], true)}</rect>`;
    s += `<g opacity="0">${aShow(t1)}${g}</g>` + txA(26, 138, 'an-k', '400 題總分多了', t1);
    // 計數器：每翻一格換一個數字（一次只看得到一個）
    const vals = ['+0'].concat([1, 2, 3, 4, 5, 6, 7, 8].map(k => '+' + (0.63212 * k).toFixed(3)));
    vals.forEach((v, k) => {
      const a = k === 0 ? t1 : tf(k - 1), b = k === 8 ? HOLD + 0.3 : tf(k);
      s += `<text x="138" y="140" opacity="0" style="${CH5BIG};fill:${k === 8 ? K.ok : K.ink}">${v}${aSeq('opacity', [[0, 0], [a, 1], [b, 0], [AD, 0]], true)}</text>`;
    });
    s += ch5check(28, 171, K.ok, tf(7) + 0.4) + txA(46, 176, 'an-b', '＝排行榜讀數：8 題全對', tf(7) + 0.4);
    // ③ 例二框：5D 改 6D。改之前 5 格猜中、2 格差一級；改之後只有 1 格變猜中，5 格變差一級、1 格變猜錯
    const t2 = 5.0, before = [K.ok, K.ok, K.ok, K.ok, K.ok, K.near, K.near], after = [K.near, K.near, K.near, K.near, K.near, K.ok, K.bad];
    s += ch5box(326, 48, 302, 152, t2) + txA(340, 72, 'an-t', '例二：7 題 5D 改成 6D', t2);
    g = '';
    before.forEach((c0, i) => { g += `<rect x="${340 + i * GAP}" y="84" width="${SQ}" height="${SQ}" rx="5" fill="${c0}">${aSeq('fill', [[0, c0], [t2 + 0.6 + i * 0.12, after[i]], [AD, after[i]]], true)}</rect>`; });
    s += `<g opacity="0">${aShow(t2)}${g}</g>`
      + txA(340, 138, 'an-k', '400 題總分少了', t2 + 1.2)
      + txA(452, 140, '', L[1][1], t2 + 1.2, undefined, `style="${CH5BIG};fill:${K.bad}"`)
      + txA(340, 176, 'an-h', '只對 1 題：寫成禁止規則', t2 + 1.6, undefined, `style="fill:${K.bad}"`);
    // ④ 方塊堆：每格 1 題、每排 4 格，堆高就是 Public 那一級的題數
    const t3 = 7.2, base = 340, P = 5, B = 4;
    s += ch5box(12, 212, 616, 158, t3) + txA(26, 236, 'an-t', '5 次整份答同一級：解出每級幾題（1 格＝1 題）', t3);
    C.forEach(([lv, n], i) => {
      const x0 = 60 + i * 56, rows = Math.ceil(n / B), top = base - rows * P, t0 = t3 + 0.3 + i * 0.07;
      let blk = '';
      for (let k = 0; k < n; k++) blk += `<rect x="${x0 + (k % B) * P}" y="${base - (Math.floor(k / B) + 1) * P}" width="${P - 1}" height="${P - 1}" fill="${K.ok}"/>`;
      s += `<g opacity="0">${aShow(t0)}${blk}</g>`
        + `<rect x="${x0 - 1}" y="${top - 1}" width="${B * P + 1}" height="${rows * P + 1}" fill="#ffffff" opacity="0">${aShow(t0)}${aTo('height', rows * P + 1, 0, t0, 0.5)}</rect>`
        + `<text class="an-n" x="${x0 + 10}" y="${top - 6}" text-anchor="middle" opacity="0">${n}<tspan x="${x0 + 10}" y="${base + 16}" style="font-size:12px;font-weight:400;fill:${K.mute}">${lv}</tspan>${aShow(t0 + 0.4)}</text>`;
    });
    return {steps: ['① 三種得分像固定單價', '② 8 題全變對，分數差對上', '③ 7 題只對 1 題：禁止規則', '④ 解出 Public 每級幾題'],
      svg: `<svg viewBox="0 0 640 378" role="img" aria-label="收銀台式反推：8 個方塊由差一級翻成猜中、分數差對上排行榜；7 個方塊只有 1 個變對；Public 各級題數的方塊堆">${s}</svg>`,
      note: '400 題總分 ÷ 4 就是每 100 題。一題由差一級變猜中 +0.6321，由猜錯變猜中 +1；分數顯示到小數第 7 位，所以改的題不多時只湊得出一種組合。'};
  };
  ANIM.ch5_ledger.dur = 10;

  // 14 右欄：兩個城市的民調。① 兩個框（兩榜各 400 題）＋同一個篩子，各圈一批（Public 7 題、Private 10 題；點的位置是示意）
  //   ② Public 那批試改：第一根溫度計沒到門檻 ③ Private 那批鎖住不動 ④ 11-18 的四批並排，都沒到門檻。較早真的改到 Private 的三次在講稿。
  ANIM.ch5_band = () => {
    const B = NUM.ch5.bands, K = CH5, rnd = ch5rng(20261118), SP = 8;
    const selP = ch5pick(rnd, 400, B[0][1]), selQ = ch5pick(rnd, 400, B[0][2]);
    let s = '';
    // ① 兩個框＋點陣＋篩子
    [['Public 榜 400 題', selP, 12], ['Private 榜 400 題', selQ, 428]].forEach(([nm, sel, bx]) => {
      const gx = bx + 24, gy = 46, cx = bx + 100;
      let dots = '', hot = '';
      for (let i = 0; i < 400; i++) {
        const x = gx + (i % 20) * SP, y = gy + Math.floor(i / 20) * SP;
        dots += `<circle cx="${x}" cy="${y}" r="2.6" fill="${K.grey}"/>`;
        if (sel.has(i)) hot += `<circle cx="${x}" cy="${y}" r="4.2" fill="${K.hot}" stroke="#ffffff" stroke-width="1"/>`;
      }
      s += ch5box(bx, 8, 200, 222, 0.3) + txA(cx, 32, 'an-t', nm, 0.3, undefined, 'text-anchor="middle"')
        + `<g opacity="0">${aShow(0.3)}${dots}</g>` + `<g opacity="0">${aShow(1.7)}${hot}</g>`
        + txA(cx, 220, 'an-n', '圈到 ' + sel.size + ' 題', 2.0, undefined, `text-anchor="middle" style="fill:${K.hot}"`);
    });
    s += txA(320, 40, 'an-t', '同一個篩子', 0.8, undefined, 'text-anchor="middle"')
      + `<polygon points="266,52 374,52 342,98 298,98" fill="#e3f1ef" stroke="${K.ok}" stroke-width="2" opacity="0">${aShow(0.8)}</polygon>`
      + [66, 80].map(y => `<line x1="${266 + (y - 52) * 0.7}" y1="${y}" x2="${374 - (y - 52) * 0.7}" y2="${y}" stroke="${K.ok}" stroke-dasharray="3 3" opacity="0">${aShow(0.8)}</line>`).join('')
      + txA(320, 124, 'an-k', '答 2D，但 3D 也很可能', 1.0, undefined, `text-anchor="middle" style="fill:${K.ink}"`)
      + txA(320, 144, 'an-s', '（只看模型的把握，不看答案）', 1.0, undefined, 'text-anchor="middle"')
      + `<line x1="292" y1="90" x2="222" y2="90" stroke="${K.ok}" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.3, 0.4)}</line>`
      + `<polygon points="215,90 224,85 224,95" fill="${K.ok}" opacity="0">${aShow(1.6)}</polygon>`
      + `<line x1="348" y1="90" x2="418" y2="90" stroke="${K.ok}" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.3, 0.4)}</line>`
      + `<polygon points="425,90 416,85 416,95" fill="${K.ok}" opacity="0">${aShow(1.6)}</polygon>`;
    // ③ Private 那批鎖住
    s += `<rect x="482" y="80" width="92" height="78" rx="8" fill="#ffffff" fill-opacity="0.94" stroke="#d9dbe0" opacity="0">${aShow(4.4)}</rect>`
      + ch5lock(513, 104, 30, K.mute, 4.4)
      + txA(528, 150, 'an-h', '鎖住不動', 4.6, undefined, 'text-anchor="middle"');
    // ②④ 下框：四個並排的小溫度計＝四批的 400 題總分淨多幾分；粗橫線＝門檻（每改 4 題淨多 1 分）
    const y0 = 318, U = 17, CX = [118, 262, 406, 550], T = [2.8, 5.4, 5.9, 6.4];
    s += ch5box(12, 244, 616, 126, 2.8) + txA(26, 268, 'an-t', '11-18 試的四批', 2.8)
      + txA(614, 268, 'an-b', '都沒到門檻：Private 不動', 7.2, undefined, 'text-anchor="end"');
    B.forEach(([nm, np, , dE, dW], i) => {
      const cx = CX[i], t0 = T[i], net = 0.632 * dE + 0.368 * dW, thr = 0.25 * np, h = Math.abs(net) * U, c = net > 0 ? K.ok : K.bad, ty = y0 - thr * U;
      s += `<rect x="${cx - 8}" y="278" width="16" height="80" rx="8" fill="#f2f3f6" stroke="${i === 0 ? K.hot : '#d9dbe0'}" stroke-width="${i === 0 ? 2 : 1}" opacity="0">${aShow(t0)}</rect>`
        + `<line x1="${cx - 12}" y1="${y0}" x2="${cx + 12}" y2="${y0}" stroke="${K.mute}" stroke-width="1.2" opacity="0">${aShow(t0)}</line>`
        + (net < 0 ? `<rect x="${cx - 5}" y="${y0}" width="10" height="0" rx="2" fill="${c}">${aTo('height', 0, h.toFixed(1), t0 + 0.3, 0.5)}</rect>`
          : `<rect x="${cx - 5}" y="${y0}" width="10" height="0" rx="2" fill="${c}">${aTo('y', y0, (y0 - h).toFixed(1), t0 + 0.3, 0.5)}${aTo('height', 0, h.toFixed(1), t0 + 0.3, 0.5)}</rect>`)
        + `<line x1="${cx - 15}" y1="${ty}" x2="${cx + 15}" y2="${ty}" stroke="${K.ink}" stroke-width="3" opacity="0">${aShow(t0 + 0.2)}</line>`
        + txA(cx + 20, (net < 0 ? y0 + h + 4 : y0 - h + 4).toFixed(1), 'an-n', (net > 0 ? '+' : '−') + Math.abs(net).toFixed(2), t0 + 0.7, undefined, `style="fill:${c}"`)
        + txA(cx - 16, y0 + 5, 'an-n', nm, t0, undefined, `text-anchor="end" style="fill:${i === 0 ? K.hot : K.ink}"`);
      if (i === 0) s += txA(cx + 20, ty + 4, 'an-s', '門檻', t0 + 0.2);
    });
    return {steps: ['① 同一個篩子在兩榜各圈一批', '② Public 那批試改：沒到門檻', '③ Private 那批鎖住不動', '④ 11-18 四批都沒到門檻'],
      svg: `<svg viewBox="0 0 640 376" role="img" aria-label="兩個城市的民調：同一個篩子在 Public、Private 各圈一批題；Public 那批的淨分沒到門檻，Private 那批鎖住；11-18 四批都沒到門檻">${s}</svg>`,
      note: '溫度計＝和 ct116 比，Public 400 題總分淨多幾分（÷ 4 就是每 100 題）；粗橫線＝門檻：每改 4 題淨多 1 分。點的位置只是示意。'};
  };
  ANIM.ch5_band.dur = 10;


  // 15 右欄：① 左框 800 題看起來像隨機分成兩半（點的顏色是示意）＋右框六個徽章（五個看不出差別、一個 Private 略低）
  //   ② 下框：賽前九成把握的範圍條，兩端大字「下限約 0.78／上限約 0.84」③ 中間填色「最可能 0.80–0.82」（11-21 結案時的計算，M264）
  //   ④ 淡色參考記號 78.7（模型對最終答案的自估，只畫三角形）、83.3（沒用回推標籤的 Public＝ct112）；
  //     橘色粗豎線＋大字＝11-25 公布的實際成績 0.7902（11-26 加入）；鎖：最後上傳的答案檔與 276／109／15
  ANIM.ch5_floor = () => {
    const S = NUM.ch5.sim, E = NUM.ch5.est, CI = NUM.ch5.ci90, SELF = NUM.ch5.self, CMP = NUM.ch5.comparable, ACT = NUM.ch5.actual, fin = NUM.final, K = CH5, PUB = K.ok, PRV = '#0072b2', FAINT = '#9aa0aa';
    const rnd = ch5rng(116), idx = [...Array(800).keys()];
    for (let i = 799; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const isPub = new Set(idx.slice(0, 400));
    let a = '', b = '';
    for (let i = 0; i < 800; i++) {
      const c = `<circle cx="${(28 + (i % 40) * 6.8).toFixed(1)}" cy="${(46 + Math.floor(i / 40) * 6.8).toFixed(1)}" r="2.3"/>`;
      if (isPub.has(i)) a += c; else b += c;
    }
    // ① 左框：點陣先全灰，再依兩榜上色；右框：六個徽章
    let s = ch5box(12, 8, 300, 198, 0.3) + txA(26, 32, 'an-t', '800 題，看起來像隨機分成兩半', 0.3)
      + `<g opacity="0" fill="${K.grey}">${aShow(0.3)}${aSeq('fill', [[0, K.grey], [1.0, PUB], [AD, PUB]], true)}${a}</g>`
      + `<g opacity="0" fill="${K.grey}">${aShow(0.3)}${aSeq('fill', [[0, K.grey], [1.0, PRV], [AD, PRV]], true)}${b}</g>`
      + `<circle cx="32" cy="190" r="4.5" fill="${PUB}" opacity="0">${aShow(1.1)}</circle>` + txA(42, 195, 'an-n', 'Public 400 題', 1.1, undefined, `style="fill:${PUB}"`)
      + `<circle cx="170" cy="190" r="4.5" fill="${PRV}" opacity="0">${aShow(1.1)}</circle>` + txA(180, 195, 'an-n', 'Private 400 題', 1.1, undefined, `style="fill:${PRV}"`);
    s += ch5box(324, 8, 304, 198, 1.6) + txA(338, 32, 'an-t', '六項比較，五項看不出差別', 1.6);
    S.forEach(([nm, ok], i) => {
      const x = 334 + (i % 2) * 146, y = 44 + Math.floor(i / 2) * 52, t0 = 1.8 + i * 0.2;
      s += `<rect x="${x}" y="${y}" width="138" height="46" rx="10" fill="${ok ? '#e3f1ef' : '#fbeee6'}" opacity="0">${aShow(t0)}</rect>`;
      if (ok) s += ch5check(x + 9, y + 23, K.ok, t0) + txA(x + 26, y + 28, 'an-k', nm, t0, undefined, `style="fill:${K.ink}"`);
      else s += `<circle cx="${x + 14}" cy="${y + 13}" r="7" fill="${K.hot}" opacity="0">${aShow(t0)}</circle>`
        + `<rect x="${x + 12.8}" y="${y + 8}" width="2.4" height="6" fill="#ffffff" opacity="0">${aShow(t0)}</rect><rect x="${x + 12.8}" y="${y + 15.5}" width="2.4" height="2.4" fill="#ffffff" opacity="0">${aShow(t0)}</rect>`
        + txA(x + 26, y + 18, 'an-k', nm, t0, undefined, `style="fill:${K.ink}"`)
        + txA(x + 9, y + 40, 'an-h', 'Private 略低約 0.01', t0 + 0.2);
    });
    // ② 下框：賽前九成把握的範圍條（只畫 78–84），兩端粗豎線＋大字
    const D2 = v => (v / 100).toFixed(2);   // 顯示成排行榜的寫法（0.xx）；資料仍是每 100 題
    const X = v => 146 + (v - CI[0]) * (348 / (CI[1] - CI[0])), t2 = 3.3, t3 = 4.6, t4 = 5.8, t6 = 7.0, ty = 272, th = 22, BIG = 'font-size:18px;font-weight:700';
    const A4 = (ACT / 100).toFixed(4), xa = X(ACT).toFixed(1);
    s += ch5box(12, 214, 616, 144, t2) + txA(26, 236, 'an-t', 'Private 分數：賽前預估和實際（和排行榜同一種分數，滿分 1）', t2)
      + `<rect x="${X(CI[0])}" y="${ty}" width="${X(CI[1]) - X(CI[0])}" height="${th}" rx="4" fill="#bfe0db" opacity="0">${aShow(t2)}</rect>`
      + [CI[0], CI[1]].map(v => `<line x1="${X(v)}" y1="${ty - 10}" x2="${X(v)}" y2="${ty + th + 8}" stroke="${K.ok}" stroke-width="4" stroke-linecap="round" opacity="0">${aShow(t2 + 0.2)}</line>`).join('')
      + txA(X(CI[0]) - 10, ty + 17, '', '下限約 ' + D2(CI[0]), t2 + 0.3, undefined, `text-anchor="end" style="${BIG};fill:${K.ink}"`)
      + txA(X(CI[1]) + 10, ty + 17, '', '上限約 ' + D2(CI[1]), t2 + 0.3, undefined, `style="${BIG};fill:${K.ink}"`);
    // ③ 中間填色：賽前最可能 80–82（11-21 結案時的計算）
    s += `<rect x="${X(E[0])}" y="${ty}" width="${X(E[1]) - X(E[0])}" height="${th}" rx="3" fill="${K.ok}" opacity="0">${aShow(t3)}</rect>`
      + txA((X(E[0]) + X(E[1])) / 2, ty - 10, '', '最可能 ' + D2(E[0]) + '–' + D2(E[1]), t3 + 0.2, undefined, `text-anchor="middle" style="${BIG};fill:${K.ok}"`);
    // ④ 淡色參考記號：自估只畫三角形（實際成績的大字就在旁邊，說明寫在框底那行），ct112 的 Public 有一行小字
    [[SELF, ''], [CMP, (CMP / 100).toFixed(3) + ' 沒用回推標籤的 Public（含運氣）']].forEach(([v, lab], i) => {
      const x = X(v), t0 = t4 + i * 0.3;
      s += `<polygon points="${x.toFixed(1)},${ty + th + 2} ${(x - 5).toFixed(1)},${ty + th + 10} ${(x + 5).toFixed(1)},${ty + th + 10}" fill="${FAINT}" opacity="0">${aShow(t0)}</polygon>`
        + (lab ? txA(x.toFixed(1), ty + th + 26, 'an-s', lab, t0, undefined, `text-anchor="middle" style="fill:${FAINT}"`) : '');
    });
    // ⑤ 實際成績（11-25 公布）：橘色粗豎線穿過範圍條＋大字
    s += `<line x1="${xa}" y1="${ty - 12}" x2="${xa}" y2="${ty + th + 14}" stroke="${K.hot}" stroke-width="5" stroke-linecap="round" opacity="0">${aShow(t6)}</line>`
      + txA(xa, ty + th + 35, '', '實際 ' + A4, t6 + 0.3, undefined, `text-anchor="middle" style="${BIG};fill:${K.hot}"`);
    s += txA(26, 350, 'an-s', '賽前九成把握 ' + D2(CI[0]) + '–' + D2(CI[1]) + '、最可能 ' + D2(E[0]) + '–' + D2(E[1]) + '。灰色三角形＝模型自己估的 ' + (SELF / 100).toFixed(3), t2 + 0.5);
    // 鎖：最後上傳的答案檔與成績
    const t5 = 8.2;
    s += ch5lock(24, 376, 22, K.ok, t5)
      + txA(58, 381, 'an-t', '最後上傳的答案檔 ' + fin.file + '：Private ' + fin.priv.toFixed(7), t5 + 0.2)
      + txA(58, 397, 'an-s', '400 題：猜中 ' + fin.exact + ' 題、差一級 ' + fin.near + ' 題、其餘 ' + fin.miss + ' 題', t5 + 0.5);
    return {steps: ['① 兩榜很像：五項看不出差別', '② 賽前九成把握 ' + D2(CI[0]) + '–' + D2(CI[1]), '③ 賽前最可能 ' + D2(E[0]) + '–' + D2(E[1]), '④ 實際 ' + A4 + '：在範圍內，比最可能低'],
      svg: `<svg viewBox="0 0 640 402" role="img" aria-label="兩榜看起來像隨機分成兩半、六項比較五項看不出差別；Private 分數賽前九成把握下限約 ${D2(CI[0])}、上限約 ${D2(CI[1])}，最可能 ${D2(E[0])}–${D2(E[1])}；實際 ${fin.priv.toFixed(7)}，在範圍內、比最可能低；最後上傳的答案檔 ${fin.file}">${s}</svg>`,
      note: '橘色豎線：11-25 公布的 Private 成績 ' + fin.priv.toFixed(7) + '。灰色三角形 ' + (SELF / 100).toFixed(3) + '：模型對最終答案的自估（賽前最保守的估法）；' + (CMP / 100).toFixed(3) + '：沒用回推標籤的 Public 分數（ct112），含運氣。分數和排行榜同一種寫法。點的顏色只是示意。'};
  };
  ANIM.ch5_floor.dur = 11;
