  // ---- 第 4 章 模型與訓練（第 8–9 頁；W3；白話版 11-18）----
  // 本章數字。出處縮寫：素材06 = reports/報告素材_2026-11-06.md、設計 = reports/architecture_v2_2026-11-02.md
  NUM.ch3 = {
    // 連看前後 63 手：T 支的膨脹 1–16、kernel 3，感受野 63 手（素材06 §4.3）
    rf: 63,
    // 每一手的描述（素材06 §4.2 token 162 維）：棋形印象＝盤面 64＋落點 64、像幾級下的＝等級頭 11＋Human SL 19、誰下／第幾手＝旗標 4；畫面只用長度比例，不顯示維度
    token: [['AI 的棋形印象', 128, '#0072b2'], ['像幾級下的', 30, '#b98900'], ['誰下、第幾手', 4, '#5d6370']],
    // 整題線索（素材06 §4.2 R 表）：Human SL 24、等級頭 16、對手 16、差值 3、讓子 5；畫面只列名稱
    r: ['老師的判斷', '新尺（10 個小網路）', '對手的程度', '讓子幾顆'],
    // 「同一個人的兩份棋譜，判斷要接近」（設計 M33）：每批 16 位玩家 × 每人 2 題；6 對 6 集成模擬考 0.7398 → 0.7458（畫成每 100 題 74.0 → 74.6）
    contrast: {players: 16, perPlayer: 2, before: 0.7398, after: 0.7458},
    // 作答的例子：上傳檔 ct7 對 pub_q_0177 的十級把握（submission_rank_ct7_proba.csv 第 178 列，取 3 位小數）：最有把握 2D，平均拿分最多 3D
    //   800 題中兩種選法不同的題數：同一檔重算（job 803320：23 題，Public 9、Private 14）；模擬考上這樣選 +0.0023（設計 M75）
    decide: {qid: 'pub_q_0177', p: [0.001, 0.003, 0.018, 0.025, 0.053, 0.240, 0.227, 0.164, 0.143, 0.126], differ: 23, gain: 0.0023},
  };

  // 第 8 頁（橫跨整張卡）：一題從每一手、每一局到整題，最後說出每一級幾成把握
  ANIM.ch3_pipeline = () => {
    const f1 = v => (+v).toFixed(1);
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const Wh = 'font-size:12px;font-weight:700;fill:#fff', WhS = 'font-size:11.5px;fill:#e3f1ef';
    const box = (x, y, w, h, fill, stroke, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="${stroke}"/>${inner}</g>`;
    const ct = (x, y, st, t) => `<text x="${x}" y="${y}" text-anchor="middle" style="${st}">${t}</text>`;
    const arrow = (x1, y1, x2, y2, t0) => {   // 任意方向的箭頭：線畫出來後箭頭淡入
      const L = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / L, uy = (y2 - y1) / L, bx = x2 - ux * 7, by = y2 - uy * 7, px = -uy * 4, py = ux * 4;
      return `<line x1="${x1}" y1="${y1}" x2="${f1(bx)}" y2="${f1(by)}" stroke="#5d6370" stroke-width="1.8" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.35)}</line>`
        + `<polygon points="${x2},${y2} ${f1(bx + px)},${f1(by + py)} ${f1(bx - px)},${f1(by - py)}" fill="#5d6370" opacity="0">${aShow(t0 + 0.35)}</polygon>`;
    };
    let s = '';
    // 三個階段的標題（評審的比喻）與分隔線
    s += txA(204, 18, 'an-t', '① 每一手：像看每個動作', 0.3, undefined, 'text-anchor="middle"')
      + txA(485, 18, 'an-t', '② 每一局：像看整場', 5.0, undefined, 'text-anchor="middle"')
      + txA(762, 18, 'an-t', '③ 整題：像看多場平均', 6.8, undefined, 'text-anchor="middle"');
    [403, 568].forEach(x => { s += `<line x1="${x}" y1="30" x2="${x}" y2="220" stroke="#e3e6ea" stroke-dasharray="4 4"/>`; });

    // 每一手的描述（第 3 章做好的；各段長度只表示份量）
    s += txA(12, 42, 'an-s', '每一手的描述', 0.4);
    let y0 = 50;
    NUM.ch3.token.forEach(([n, d, c], k) => {
      const h = Math.max(18, d * 0.75), t0 = 0.5 + k * 0.2;
      s += `<rect x="14" y="${f1(y0)}" width="24" height="${f1(h)}" fill="${c}" stroke="${c}" opacity="0">${aShow(t0)}</rect>`
        + txA(44, f1(y0 + h / 2 + 4), 'an-s', n, t0);
      y0 += h + 2;
    });
    s += arrow(120, 128, 140, 76, 1.5) + arrow(120, 128, 140, 180, 1.5);

    // 讀法一：逐手看（深色＝本人的手，淺色＝對手的手）
    const TX = k => 150 + k * 20, own = k => k % 2 === 0;
    s += txA(146, 48, 'an-n', '逐手看（逐手 MLP）', 1.7);
    for (let k = 0; k < 12; k++) {
      const t0 = 1.8 + k * 0.04, x = TX(k);
      s += `<rect x="${x}" y="58" width="14" height="20" rx="2" fill="${own(k) ? '#0b6e69' : '#e3f1ef'}" stroke="#0b6e69" opacity="0">${aShow(t0)}</rect>`
        + `<line x1="${x + 7}" y1="79" x2="${x + 7}" y2="87" stroke="#5d6370" stroke-width="1.5" opacity="0">${aShow(2.4 + k * 0.04)}</line>`
        + `<circle cx="${x + 7}" cy="92" r="4" fill="#a4451c" opacity="0">${aShow(2.4 + k * 0.04)}</circle>`;
    }
    s += txA(146, 114, 'an-s', '每一手只看它自己', 2.8);

    // 讀法二：連看前後 63 手（示意：從中間一手往兩邊連到間隔 1、2、4 的手）
    s += txA(146, 136, 'an-n', `連看前後 ${NUM.ch3.rf} 手（擴張卷積）`, 3.1);
    for (let k = 0; k < 12; k++) {
      s += `<rect x="${TX(k)}" y="176" width="14" height="20" rx="2" fill="${own(k) ? '#0b6e69' : '#e3f1ef'}" stroke="#0b6e69" opacity="0">${aShow(3.2 + k * 0.04)}</rect>`;
    }
    const fx = TX(6) + 7;
    s += `<rect x="${TX(6) - 2}" y="174" width="18" height="24" rx="3" fill="none" stroke="#a4451c" stroke-width="2.2" opacity="0">${aShow(3.7)}</rect>`;
    [[1, 9, 3.8], [2, 16, 4.1], [4, 26, 4.4]].forEach(([d, h, t0]) => {
      [-1, 1].forEach(sg => {
        const x2 = fx + sg * d * 20;
        s += `<path d="M${fx},175 Q${(fx + x2) / 2},${175 - 2 * h} ${x2},175" fill="none" stroke="#a4451c" stroke-width="1.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.4)}</path>`;
      });
    });
    s += txA(146, 214, 'an-s', '橘線：中間那一手連前後一起看', 4.6);

    // ② 每一局：本人與對手的手分開算整局平均，再合成這一局的總評
    s += `<g opacity="0">${aShow(5.0)}<polyline points="388,66 398,66 398,186 388,186" fill="none" stroke="#5d6370" stroke-width="1.5"/></g>`
      + arrow(398, 66, 410, 66, 5.0) + arrow(398, 118, 410, 118, 5.0)
      + box(412, 44, 146, 44, '#0b6e69', '#0b6e69', 5.2, ct(485, 62, Wh, '本人的手') + ct(485, 80, WhS, '算平均和起伏'))
      + box(412, 96, 146, 44, '#e3f1ef', '#0b6e69', 5.5, ct(485, 114, Hd, '對手的手') + ct(485, 132, Sm, '算平均和起伏'))
      + arrow(485, 142, 485, 160, 5.9)
      + box(412, 162, 146, 44, '#a4451c', '#a4451c', 6.1, ct(485, 180, Wh, '這一局的總評') + ct(485, 198, 'font-size:11.5px;fill:#fbeee6', '兩邊合起來'));

    // ③ 整題：5–20 局的總評再平均，加上第 3 章的整題線索，由小型神經網路合起來判斷
    s += arrow(560, 184, 576, 184, 6.6);
    for (let k = 0; k < 6; k++) {
      s += `<rect x="580" y="${112 + k * 15}" width="54" height="10" rx="2" fill="#a4451c" fill-opacity="${(0.45 + k * 0.1).toFixed(2)}" opacity="0">${aShow(6.8 + k * 0.08)}</rect>`;
    }
    s += txA(640, 160, 'an-s', '5–20 局', 7.2) + arrow(607, 108, 607, 94, 7.6)
      + box(576, 48, 124, 44, '#e3f1ef', '#0b6e69', 7.8, ct(638, 66, Hd, '各局取平均') + ct(638, 84, Sm, '也記下起伏'));
    const R = NUM.ch3.r;
    s += box(716, 106, 124, 100, '#fff', '#b98900', 8.3, ct(778, 124, Hd, '第 3 章的線索')
        + R.map((n, i) => tx(726, 142 + i * 16, 'an-s', n)).join(''))
      + arrow(700, 70, 714, 70, 8.6) + arrow(778, 104, 778, 94, 8.6)
      + box(716, 48, 124, 44, '#0b6e69', '#0b6e69', 8.9, ct(778, 66, Wh, '合起來判斷') + ct(778, 84, WhS, '兩層 MLP'))
      + arrow(840, 70, 854, 70, 9.2);
    const p = NUM.ch3.decide.p;
    s += txA(856, 40, 'an-k', '每一級幾成把握', 9.3);
    NUM.levels.forEach((n, i) => {
      const y = 52 + i * 16.5, t0 = 9.4 + i * 0.05;
      s += txA(880, f1(y + 9.5), 'an-s', n, t0, undefined, 'text-anchor="end"')
        + `<rect x="884" y="${f1(y)}" width="0" height="10" rx="2" fill="#0b6e69">${aTo('width', 0, f1(Math.max(1.5, p[i] / 0.25 * 64)), t0)}</rect>`;
    });
    return {svg: `<svg viewBox="0 0 960 226" role="img" aria-label="模型怎麼讀一題：每一手用兩種讀法看，本人和對手的手分開算平均和起伏，各局取平均和起伏並加上第 3 章的線索，最後說出每一級各有幾成把握">${s}</svg>`,
      note: '深色格＝本人的手，淺色格＝對手的手。右邊 10 條長條是我們上傳檔對一道 Public 題的真實輸出；下一頁講怎麼從它選出答案。'};
  };
  Object.assign(ANIM.ch3_pipeline, {dur: 12});

  // 第 9 頁右欄：上半＝同一個人的兩份棋譜判斷要接近，下半＝選平均拿分最多的一級
  ANIM.ch3_train = () => {
    const C3 = NUM.ch3, L = NUM.levels, e = Math.exp(-1);
    let s = '';
    // ① 練習：4 位玩家（示意；實際每次 16 位）各 2 份棋譜，從混在一起到同一人靠攏
    s += tx(16, 20, 'an-t', '① 練習：同一個人的兩份棋譜，判斷要接近')
      + `<rect x="16" y="32" width="300" height="132" rx="8" fill="#fff" stroke="#d9dbe0"/>`
      + tx(26, 50, 'an-s', '模型對每份棋譜的印象（示意）');
    const cols = ['#0b6e69', '#a4451c', '#0072b2', '#b98900'];
    const P = [   // [第 1 份起點, 第 2 份起點, 第 1 份終點, 第 2 份終點]
      [[60, 140], [250, 90], [60, 82], [82, 92]],
      [[112, 88], [284, 146], [262, 80], [284, 90]],
      [[172, 150], [40, 100], [62, 138], [84, 148]],
      [[220, 112], [134, 128], [250, 136], [272, 146]],
    ];
    const T0 = 1.4, D = 1.6;
    P.forEach(([a, b, c, d], k) => {
      s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${cols[k]}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0">${aShow(0.6)}`
        + `${aTo('x1', a[0], c[0], T0, D)}${aTo('y1', a[1], c[1], T0, D)}${aTo('x2', b[0], d[0], T0, D)}${aTo('y2', b[1], d[1], T0, D)}</line>`;
      [[a, c], [b, d]].forEach(([u, v]) => {
        s += `<circle cx="${u[0]}" cy="${u[1]}" r="8" fill="${cols[k]}" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(0.5)}${aTo('cx', u[0], v[0], T0, D)}${aTo('cy', u[1], v[1], T0, D)}</circle>`;
      });
    });
    s += txA(166, 116, 'an-s', '同一人靠在一起', 3.2, undefined, 'text-anchor="middle"');
    const bar = (y, label, v, col, t0) => {   // 長條從 70 分起畫，5 分＝150
      const w = (v - 0.70) / 0.05 * 150;
      return txA(412, y + 11, 'an-k', label, t0, undefined, 'text-anchor="end"')
        + `<rect x="420" y="${y}" width="0" height="14" rx="3" fill="${col}">${aTo('width', 0, w.toFixed(1), t0)}</rect>`
        + txA((426 + w).toFixed(1), y + 11, 'an-n', `${(v * 100).toFixed(1)} 分`, t0 + 0.6);
    };
    const K = C3.contrast;
    s += txA(332, 52, 'an-k', `每次練習放 ${K.players} 位玩家，每人 ${K.perPlayer} 份棋譜`, 0.6)
      + txA(332, 74, 'an-k', '同一人的兩份拉近，不同人的推開', 1.6)
      + txA(332, 96, 'an-k', '用意：學這個人，不學這幾局的運氣', 2.6)
      + txA(332, 120, 'an-s', '整體模擬考（多個模型平均），每 100 題', 3.2)
      + bar(132, '加要求之前', K.before, '#5d6370', 3.6) + bar(152, '加要求之後', K.after, '#0b6e69', 4.2)
      + `<line x1="420" y1="128" x2="420" y2="170" stroke="#1d1f24" stroke-width="1"/>`;

    // ② 作答：最有把握 2D，平均拿分最多 3D
    const p = C3.decide.p;
    const es = p.map((v, c) => v + e * ((p[c - 1] || 0) + (p[c + 1] || 0)));
    const am = p.indexOf(Math.max(...p)), ed = es.indexOf(Math.max(...es));
    const base = 334, X = i => 64 + i * 56, W = 30, Hs = v => v / 0.40 * 112;
    s += `<line x1="16" y1="176" x2="624" y2="176" stroke="#e3e6ea"/>`
      + txA(16, 202, 'an-t', '② 作答：選平均拿分最多的一級', 5.6)
      + txA(16, 222, 'an-s', '一道 Public 題：24 個模型平均後，每一級的把握', 5.7)
      + `<line x1="40" y1="${base}" x2="${X(9) + W + 14}" y2="${base}" stroke="#d9dbe0"/>`;
    L.forEach((n, i) => {
      const h = Hs(p[i]), h2 = Hs(es[i]), t0 = 5.8 + i * 0.06;
      s += `<rect x="${X(i)}" y="${base}" width="${W}" height="0" rx="3" fill="${i === am ? '#a4451c' : '#5d6370'}" fill-opacity="${i === am ? 1 : 0.45}">${aTo('y', base, (base - h).toFixed(1), t0)}${aTo('height', 0, h.toFixed(1), t0)}</rect>`
        + `<rect x="${X(i) - 4}" y="${(base - h2).toFixed(1)}" width="${W + 8}" height="${h2.toFixed(1)}" rx="3" fill="none" stroke="${i === ed ? '#0b6e69' : '#5d6370'}" stroke-width="${i === ed ? 3 : 1.4}" stroke-dasharray="${i === ed ? '0' : '4 3'}" opacity="0">${aShow(8.6 + i * 0.05)}</rect>`
        + tx(X(i) + W / 2, base + 18, 'an-k', n, 'text-anchor="middle"');
    });
    const pct = v => `${Math.round(v * 100)}%`;
    s += txA(X(am) + W / 2, (base - Hs(p[am]) - 8).toFixed(1), 'an-n', pct(p[am]), 6.8, 8.4, 'text-anchor="middle" style="fill:#a4451c"')
      + txA(624, 202, 'an-h', `最有把握：${L[am]}（${pct(p[am])}）`, 7.0, undefined, 'text-anchor="end"')
      + txA(624, 222, 'an-b', `平均拿分最多：${L[ed]}`, 9.4, undefined, 'text-anchor="end"')
      + txA(X(ed) + W / 2, (base - Hs(es[ed]) - 8).toFixed(1), 'an-n', `${es[ed].toFixed(3)} 分`, 9.4, undefined, 'text-anchor="middle" style="fill:#0b6e69"')
      + txA(16, 372, 'an-k', '實心＝把握；虛線框＝選這級平均拿幾分（猜中 1 分，差一級約 0.368 分）', 8.6);
    return {steps: ['① 同一個人的兩份棋譜拉近', '② 24 個模型平均出每一級的把握', '③ 選平均拿分最多的一級'],
      svg: `<svg viewBox="0 0 640 380" role="img" aria-label="上半：同一位玩家的兩份棋譜，模型的印象慢慢靠近；下半：一道題每一級的把握與平均能拿的分數，選平均拿分最多的那一級">${s}</svg>`,
      note: `上半是示意（實際每次 ${C3.contrast.players} 位玩家）；長條從 70 分起畫。下半是上傳檔對一道 Public 題的真實輸出。這樣選答案，模擬考每 100 題多 ${(C3.decide.gain * 100).toFixed(2)} 分。`};
  };
  Object.assign(ANIM.ch3_train, {dur: 12});
