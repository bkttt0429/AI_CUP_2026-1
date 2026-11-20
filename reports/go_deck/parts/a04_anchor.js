  // ===== 第 5 章 破局：認出對手（W4；第 10–12 頁：card-anchor-idea、card-anchor-fuse、card-gate）=====
  // 整段包在一個區塊裡：本章的輔助函式（ch4Board、ch4Arrow）不和別章的同名 const 衝突。
  {
    // ---- 本章數字。出處簡稱：素材06 = reports/報告素材_2026-11-06.md、素材15 = reports/報告素材_2026-11-15_補充.md、
    //      設計 = reports/architecture_v2_2026-11-02.md（以 M 號為準）、問題陳述 = reports/問題陳述_認人瓶頸_2026-11-16.md、
    //      MN = METHOD_NOTES.md ----
    NUM.ch4 = {
      // 測試對手約 75% 是有標籤的訓練玩家：問題陳述 §1、§2；素材15 G.6（M187：混合分布估計 0.74–0.78）
      inPool: 0.75,
      // 池內對手被清楚認出的只有約四成（每局 0.29 個）：問題陳述 §2；素材15 G.6（M187）
      clearPerGame: 0.29,
      // 名下 ≥ 300 局的玩家占 11.6%，擁有 48% 的訓練局：素材15 D.1（M160）
      active: {players: 0.116, games: 0.48},
      // 列數加權的留出本人局 Top-1（全體網路、1.2 萬步的先驅；三個種子）：素材15 D.2（M161）
      //   顯示名稱（白話）：等機率抽人＝每人一樣常見；按局數抽人＝常客優先猜；＋每人 16 局、網路加寬＝再改進練法
      top1: [['每人一樣常見', [0.331, 0.378, 0.335]], ['常客優先猜', [0.416, 0.413, 0.414]], ['再改進練法', [0.471]]],
      // 第 11 頁 Public 線：從 CV 0.7458 那版（ct9）的上傳起算，到 ct112。[檔, x 軸第一行, 第二行, Public, 種類]
      //   種類：base 基礎集成、probe 用 Public 探測決定的推移、rej 被否決的上傳（不在採用路徑上）、'' 一般採用
      //   出處：素材15 B 里程碑表（ct9 0.7622032 11-06；ct13 0.7748456 11-07；ct28 0.7827471「11-09 上傳、11-10 定案」；
      //   ct40 0.7761668 11-10「開局模型認對手…否決」＝第一個含錨點的上傳；ct44 0.8067107；ct75u 0.8101304 11-13；ct106 0.8126304 11-14）；
      //   素材15 A（ct109 0.8164516 約 17:00、ct112 0.8331743 約 22:40，都是 11-15）
      pub: [['ct9', '11-06', '', 0.7622032, 'base'], ['ct13', '11-07', '', 0.7748456, 'probe'], ['ct28', '11-09', '', 0.7827471, 'probe'],
        ['ct40', '11-10', '', 0.7761668, 'rej'], ['ct44', '11-11', '', 0.8067107, ''], ['ct75u', '11-13', '', 0.8101304, ''],
        ['ct106', '11-14', '', 0.8126304, ''], ['ct109', '11-15', '17 時', 0.8164516, ''], ['ct112', '11-15', '22 時', 0.8331743, '']],
      // 第 11 頁模擬考折線的 x 軸名稱：只看棋＝NUM.climb 的「＋對比損失」（基礎集成 0.7458）；＋借等級（及後續改良）＝ct104b 規則 0.7781（累積 M71–M129b，含兩條用 Public 資訊定的禁止規則：素材15 B、E）；＋按可靠度合併＋禁止規則＝ct106 規則 0.7836（學出來的合併 L0＋四條防護：素材15 C 第 6、7 列）；＋常客優先猜＝ct112 規則 0.7900（素材15 D.6）
      cvLabels: [['只看棋'], ['＋借等級', '（及後續', '改良）'], ['＋按可靠度', '合併＋禁止', '規則'], ['＋常客', '優先猜']],
      // 第 12 頁閘門動畫的數字來源（圖上不畫分數，講稿引用）：[名稱, 區, 差（第一組 seed）, CI 下, CI 上, 差（第二組）, CI 下, CI 上, 結果, 出處]；第二組沒有做的填 null
      //   區 A ＝ 對當時基礎模型的配對差（M37 門檻 +0.0025；M37 以前的列當時門檻是 +0.005，兩種門檻都沒過）；
      //   區 B ＝ 認人網路的池測試：整條規則（含錨點）對當時現行規則的配對差（門檻兩組都 ≥ +0.0010）
      forest: [
        ['第一版借等級（認出對手、借他的等級）', 'A', 0.0074, 0.0043, 0.0107, 0.0070, 0.0039, 0.0101, 'pass', '錨點第一版（只用開局模型）：設計 M71 第 2 步結果（底 0.7458／0.7437）'],
        ['改用梯度提升樹（GBDT）取代 MLP', 'A', -0.0056, -0.0095, -0.0014, null, null, null, 'fail', '樹模型 GBDT 取代 MLP：素材06 §7.2-C「§4B 樹公平比較」(A)'],
        ['讀整局改用循環神經網路 GRU', 'A', -0.0004, -0.0029, 0.0022, null, null, null, 'fail', '序列編碼器 GRU：素材06 §7.2-B「六種序列 encoder」'],
        ['讀整局改用反覆讀取的 CTM', 'A', -0.0030, -0.0057, -0.0004, null, null, null, 'fail', '序列編碼器 CTM：素材06 §7.2-B「六種序列 encoder」'],
        ['把每手分數當曲線，自動算統計量', 'A', -0.0128, -0.0167, -0.0090, null, null, null, 'fail', 'TSFEL 時間序列特徵：素材06 §7.2-A（PCA-16）；MN §2.22'],
        ['請頂尖 AI 評每一手虧了多少', 'A', -0.0024, -0.0047, -0.0001, -0.0019, -0.0041, 0.0006, 'fail', '強 AI 每手勝率／目數損失：設計 M74 A 部分結果'],
        ['一級一級問「比這級強嗎」（CORAL）', 'A', -0.0263, -0.0309, -0.0217, null, null, null, 'fail', '順序型輸出頭 CORAL：素材06 §7.2-C「CORAL／BT 有序輸出」'],
        ['練習題照考題：同一人都執黑或都執白', 'A', 0.00248, -0.0002, 0.0052, null, null, null, 'fail', '標籤端單色訓練：素材06 §7.2-H「單色訓練 sc50」（label_side_diag D3）；差值用 JSON 原值 0.0024842（素材06 第 91 行），表上四捨五入是 +0.0025'],
        ['最後再參考名下局數等背景資訊', 'A', 0.0007, -0.0015, 0.0029, null, null, null, 'fail', '標籤端共變量疊加（log n）：素材06 §7.2-H「標籤端 stacker」、§7 表第 926 行'],
        ['常客優先猜（採用）：練習和猜都偏向常客', 'B', 0.0064, 0.0042, 0.0085, 0.0077, 0.0055, 0.0098, 'pass', '按活躍度認人：素材15 D.6（ct112 對 ct106 規則）'],
        ['認人 Transformer 加寬到 768', 'B', 0.0010, -0.0006, 0.0026, 0.0006, -0.0010, 0.0022, 'fail', '認人網路加寬到 768：素材15 H.1；設計 M181'],
        ['多看「這手和一般下法差多少」', 'B', 0.0002, -0.0014, 0.0019, 0.0005, -0.0011, 0.0021, 'fail', '選擇殘差輸入：素材15 I.1；設計 M200 步驟二'],
        ['加 17.6 萬個外部帳號當陌生人（後續實驗，沒採用）', 'B', 0.0008, -0.0009, 0.0025, 0.0001, -0.0017, 0.0019, 'fail', '外部帳號當陌生人（Fox Go Dataset 176,420 個帳號，素材15 I.4；與進了定案檔的 S3（9,685 個帳號，素材15 E）是不同的實驗）：設計 M218b 結果；素材15 J.1'],
        ['多看落點周圍 9×9 的棋形', 'B', -0.0011, -0.0025, 0.0003, -0.0021, -0.0035, -0.0006, 'fail', '落點 9×9 棋形視窗：設計 M219 步驟二；素材15 J.3'],
        ['三種認人模型一起用', 'B', 0.0003, -0.0016, 0.0021, 0.0004, -0.0016, 0.0022, 'fail', '三個家族合併：設計 M225 結果；素材15 J.4'],
        ['棋形視窗＋外部帳號合在一個模型', 'B', -0.0002, -0.0019, 0.0015, -0.0007, -0.0025, 0.0011, 'fail', '視窗＋陌生人同一網路：設計 M221 步驟二結果；素材15 J.6'],
      ],
      // 第 12 頁「採用」框：留在最終系統、量過有效的訊號，依故事分 4 組。[卡上名稱, 卡上數字（每 100 題）]；數字沿用第 5–11 頁
      kept: [
        [ // 每一手的證據（第 5–6 頁）
          ['AI 老師 18 級判斷', '+17.1（另批考題）'],   // fold 0（從沒訓練過的玩家）0.4399 → 0.6113：素材06 §6.1 第 0、1 列；第 5 頁 44.0 → 61.1
          ['換一把新尺', '+3.5'],                     // L1b 對 hsl +0.0352 [+0.0303, +0.0404]：素材06 §7.1 #1
          ['內部印象', '+0.5'],                       // 拿掉 trunk 向量 −0.0053 [−0.0089, −0.0014]：素材06 §4.1.3「消融」（屬於讀整局 +2.1 的一部分）
        ],
        [ // 看對手（第 7 頁）
          ['對手也用新尺量', '+1.9'],                 // +0.0191 [0.0148, 0.0235]：素材06 §7.1 #2
          ['對手比本人強多少', '+1.8'],               // +0.0181 [+0.0140, +0.0220]：素材06 §7.1 #3
          ['讓了幾子', '+1.0'],                       // 讓子重評後 +0.0097 [0.0060, 0.0133]：素材06 §7.1 #4、§6.1 第 7 列
          ['讀整局也看對手', '+1.0'],                 // token oh 集成 +0.0096 [+0.0062, +0.0130]：素材06 §7.1 #12
        ],
        [ // 模型與訓練（第 8–9 頁）
          ['讀整局：兩種讀法', '+2.1'],               // C＋T 集成對同題 MLP +0.0214（seed 0–2）／+0.0209 [+0.0164, +0.0255]（seed 3–5）：素材06 §6.1 第 9 列、§7.1 #7
          ['每人 40 局練習題', '+0.47'],              // M8 集成 +0.0047 [+0.0016, +0.0077]，四折全正（當時依放寬規則納入）：素材06 §7.1 #9
          ['同一人判斷要接近', '+0.6'],               // ct9 對比損失 +0.0060 [+0.0036, +0.0085]，四折全正：素材15 B ct9 列
          ['選平均拿分最多', '+0.23（沒走門檻）'],     // 期望分決策對 argmax +0.0023 [+0.0011, +0.0034]／+0.0018 [+0.0007, +0.0029]，一開始就用、事後量：設計 M75
        ],
        [ // 認出對手（第 10–11 頁）
          ['借等級及後續改良', '+3.2（累積）'],        // 基礎 0.7458 → ct104b 規則 0.7781（M71–M129b 累積；第一版 +0.0074）：素材15 A、B
          ['按可靠度合併', '+0.55'],                  // ct106：學出來的合併＋四條禁止規則對 ct104b 規則 +0.0055 [+0.0036, +0.0074]／+0.0054：素材15 B ct106 列
          ['常客優先猜', '+0.64'],                    // ct112 規則對 ct106 規則 +0.0064 [+0.0042, +0.0085]／+0.0077：素材15 D.6
        ],
      ],
    };

    // 小棋盤：s 是邊長，k 是線數，stones = [[x, y, 'b'|'w'], …]（線的索引）
    const ch4Board = (x, y, s, k, stones) => {
      const g = s / (k + 1);
      let o = `<rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="#dcb36a" stroke="#9c7a3c"/>`;
      for (let i = 1; i <= k; i++) o += `<line x1="${(x + g).toFixed(1)}" y1="${(y + i * g).toFixed(1)}" x2="${(x + k * g).toFixed(1)}" y2="${(y + i * g).toFixed(1)}" stroke="#8a6a2f" stroke-width=".7"/>`
        + `<line x1="${(x + i * g).toFixed(1)}" y1="${(y + g).toFixed(1)}" x2="${(x + i * g).toFixed(1)}" y2="${(y + k * g).toFixed(1)}" stroke="#8a6a2f" stroke-width=".7"/>`;
      stones.forEach(([i, j, c]) => { o += `<circle cx="${(x + i * g).toFixed(1)}" cy="${(y + j * g).toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="${c === 'b' ? '#1d1f24' : '#fff'}" stroke="#1d1f24" stroke-width=".8"/>`; });
      return o;
    };
    const ch4Arrow = (x1, y, x2, t0) => `<line x1="${x1}" y1="${y}" x2="${x2 - 6}" y2="${y}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.3)}</line>`
      + `<polygon points="${x2},${y} ${x2 - 7},${y - 4} ${x2 - 7},${y + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.3)}</polygon>`;
    const ch4Box = (x, y, w, h, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#fff" stroke="#c9ced6"/>${inner}</g>`;
    const ch4Halo = 'paint-order:stroke;stroke:#fff;stroke-width:3px';   // 文字壓在線上時的白邊

    // ---- 第 10 頁右欄：對手的一局 → 在 8,007 位老玩家中找最像的人 → 借他的等級推回本人；下排：常客優先猜（實測）----
    ANIM.ch4_ident = () => {
      const N = NUM.ch4, A = N.active;
      // 上排（示意）
      const B0 = {x: 18, y: 46, s: 104, k: 7}, g = B0.s / (B0.k + 1);
      const stones = [[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 5, 'w'], [5, 3, 'b'], [3, 4, 'b'], [5, 6, 'w']];
      const rings = stones.filter(st => st[2] === 'w').map(([i, j]) => `<circle cx="${(B0.x + i * g).toFixed(1)}" cy="${(B0.y + j * g).toFixed(1)}" r="${(g * 0.62).toFixed(1)}" fill="none" stroke="#a4451c" stroke-width="2" opacity="0">${aShow(1.0)}</circle>`).join('');
      let s = ch4Box(8, 8, 172, 162, 0.4, tx(94, 32, 'an-t', '測試題的一局', 'text-anchor="middle"') + ch4Board(B0.x, B0.y, B0.s, B0.k, stones)
          + tx(130, 76, 'an-s', '黑：本人') + tx(130, 100, 'an-s', '白：對手') + tx(130, 132, 'an-s', '要認的') + tx(130, 150, 'an-s', '是白方'))
        + rings + ch4Arrow(182, 89, 204, 1.4);
      let dots = '';
      for (let r = 0; r < 5; r++) for (let c = 0; c < 16; c++) dots += `<circle cx="${222 + c * 12}" cy="${54 + r * 15}" r="3.4" fill="#c9ced6"/>`;
      s += ch4Box(206, 8, 214, 162, 1.6, tx(313, 32, 'an-t', '8,007 位老玩家', 'text-anchor="middle"') + dots
          + tx(313, 140, 'an-s', '每人記下一份棋風', 'text-anchor="middle"') + tx(313, 158, 'an-s', '找出棋風最像的人', 'text-anchor="middle"'))
        + `<rect x="215" y="44" width="14" height="80" rx="3" fill="#0b6e69" fill-opacity=".18" opacity="0">${aSeq('x', [[0, 215], [2.1, 215], [3.3, 395], [AD, 395]])}${aShow(2.1, 3.4)}</rect>`
        + `<circle cx="342" cy="84" r="5.8" fill="#0b6e69" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(3.4)}</circle>`
        + `<circle cx="342" cy="84" r="10.5" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(3.4)}</circle>`
        + ch4Arrow(422, 89, 444, 3.6);
      s += ch4Box(446, 8, 186, 162, 3.9, tx(539, 32, 'an-t', '最像的那位', 'text-anchor="middle"')
          + tx(539, 76, 'an-v', '他是 3D', 'text-anchor="middle" style="fill:#0b6e69"')
          + tx(539, 108, 'an-k', '認得很有把握', 'text-anchor="middle"') + tx(539, 130, 'an-s', '→ 借他的等級', 'text-anchor="middle"'));
      // 借來的等級回流到本人：從右框底部畫回左框底部
      s += `<path d="M539,172 C539,214 94,214 94,180" fill="none" stroke="#0b6e69" stroke-width="2.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(4.7, 0.9)}</path>`
        + `<polygon points="94,173 88,183 100,183" fill="#0b6e69" opacity="0">${aShow(5.6)}</polygon>`
        + txA(316, 228, 'an-b', '兩人下互先，多半同級：本人大概也是 3D', 5.6, undefined, 'text-anchor="middle"')
        + txA(316, 248, 'an-s', '實測：認得最有把握的那批，約一半同級、九成差一級以內', 6.0, undefined, 'text-anchor="middle"');
      // 下排（實測）：常客下了將近一半的棋；優先猜常客，第一個猜的就猜中的比例上升
      const D = 20;   // 下排整體往下移
      s += `<line x1="8" y1="${242 + D}" x2="632" y2="${242 + D}" stroke="#e3e6ea"/>` + txA(8, 264 + D, 'an-t', '常下棋的人優先猜', 6.4);
      [['玩家', A.players, 290 + D], ['棋局', A.games, 320 + D]].forEach(([n, v, y], i) => {
        const t0 = 6.7 + i * 0.4, W = 190, w = W * v;
        s += txA(8, y + 1, 'an-k', n, t0)
          + `<rect x="44" y="${y - 11}" width="${W}" height="15" rx="3" fill="#e3e6ea" opacity="0">${aShow(t0)}</rect>`
          + `<rect x="44" y="${y - 11}" width="0" height="15" rx="3" fill="#a4451c" opacity="0">${aTo('width', 0, w.toFixed(1), t0, 0.5)}${aShow(t0)}</rect>`
          + txA((44 + w + 6).toFixed(1), y + 1, 'an-n', (v * 100).toFixed(v * 100 % 1 ? 1 : 0) + '%', t0 + 0.4, undefined, 'style="fill:#a4451c"');
      });
      s += txA(8, 348 + D, 'an-s', '橘色＝常客（名下 300 局以上）', 7.2);
      s += txA(290, 264 + D, 'an-k', '第一個猜的就猜中', 7.2);
      const X0 = 410, SC = 300, pc = v => Math.round(v * 100);
      N.top1.forEach(([n, vs], i) => {
        const y = 288 + D + i * 26, t0 = 7.4 + i * 0.45, a = Math.min(...vs), b = Math.max(...vs), col = i ? '#0b6e69' : '#8a93a3';
        s += txA(402, y + 4, 'an-s', n, t0, undefined, `text-anchor="end" style="fill:#1d1f24${i ? ';font-weight:700' : ''}"`)
          + `<rect x="${X0}" y="${y - 7}" width="0" height="14" rx="2" fill="${col}" opacity="0">${aTo('width', 0, (a * SC).toFixed(1), t0, 0.5)}${aShow(t0)}</rect>`
          + (b > a ? `<rect x="${(X0 + a * SC).toFixed(1)}" y="${y - 7}" width="${((b - a) * SC).toFixed(1)}" height="14" fill="${col}" fill-opacity=".35" opacity="0">${aShow(t0 + 0.5)}</rect>` : '')
          + txA((X0 + b * SC + 6).toFixed(1), y + 4, 'an-n', (b > a ? `${pc(a)}–${pc(b)}` : `${pc(a)}`) + '%', t0 + 0.5, undefined, `style="font-size:11.5px;fill:${col}"`);
      });
      s += txA(290, 366 + D, 'an-s', '淺色段＝重訓三次的範圍；47% 只跑過一次', 8.6);
      return {steps: ['① 拿出對手下的棋', '② 在 8,007 位老玩家裡找最像的人', '③ 借他的等級，推回本人'],
        svg: `<svg viewBox="0 0 640 ${376 + D}" role="img" aria-label="認出對手、借他的等級推回本人，以及優先猜常客讓猜中比例上升的動畫">${s}</svg>`,
        note: '上排是示意（「3D」只是例子）。限制：老玩家當對手的局，只有約四成認得清楚。下排是實測：拿模型練習時沒看過的棋局去認，第一個猜的就是棋的主人的比例。'};
    };
    ANIM.ch4_ident.dur = 10;

    // ---- 第 11 頁右欄：模擬考與 Public 榜兩條進度線，都換算成每 100 題拿幾分
    //      （模擬考讀共用的 NUM.climb；Public 用 NUM.ch4.pub，從模擬考 74.6 分那版的上傳 ct9 起算）----
    ANIM.ch4_climb = () => {
      const C = NUM.climb, L = NUM.ch4.cvLabels, P = NUM.ch4.pub;
      const k0 = Math.max(0, C.findIndex(c => c[4] === 'rule') - 1), cv = C.slice(k0, k0 + 4);   // 只看棋的基礎模型＋三個含借等級的規則
      const lo = 0.74, hi = 0.84, Y = v => 334 - (v - lo) / (hi - lo) * 280, XC = i => 70 + i * 60, XP = i => 298 + i * 40, XD = 276;
      const p100 = v => (v * 100).toFixed(1);
      const lab = (x, y, v, col, t0, anc = 'middle', fs = 11.5) => txA(x, y.toFixed(1), 'an-n', p100(v), t0, undefined, `text-anchor="${anc}" style="font-size:${fs}px;${ch4Halo};fill:${col}"`);
      let s = '';
      [0.74, 0.76, 0.78, 0.80, 0.82, 0.84].forEach(g => { const y = Y(g).toFixed(1);
        s += `<line x1="50" y1="${y}" x2="634" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(44, (Y(g) + 4).toFixed(1), 'an-s', (g * 100).toFixed(0), 'text-anchor="end"'); });
      s += `<line x1="${XD}" y1="30" x2="${XD}" y2="336" stroke="#d9dbe0" stroke-width="1.5"/>`
        + tx(52, 18, 'an-t', '模擬考') + tx(XD + 12, 18, 'an-t', 'Public 榜') + tx(4, 38, 'an-s', '每 100 題拿幾分');
      const tc = i => 0.4 + i * 0.6;
      cv.forEach(([, v], i) => {
        if (i) s += `<line x1="${XC(i - 1)}" y1="${Y(cv[i - 1][1]).toFixed(1)}" x2="${XC(i)}" y2="${Y(v).toFixed(1)}" stroke="#0b6e69" stroke-width="2.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(tc(i - 1) + 0.15, 0.45)}</line>`;
        s += `<circle cx="${XC(i)}" cy="${Y(v).toFixed(1)}" r="5" fill="#0b6e69" opacity="0">${aShow(tc(i))}</circle>`
          + (i ? lab(XC(i), Y(v) - 10, v, '#0b6e69', tc(i) + 0.1) : lab(XC(i) - 8, Y(v) - 5, v, '#0b6e69', tc(i) + 0.1, 'end'))   // 第一點的標籤放左邊，避開往上的連線
          + (L[i] || [cv[i][0]]).map((t, j) => txA(XC(i), 354 + j * 16, 'an-s', t, tc(i), undefined, 'text-anchor="middle"')).join('');
      });
      const tp = i => 3.0 + i * 0.4;
      let prev = -1;   // 採用路徑上的前一點：沒採用的上傳不在路徑上，只用灰色虛線接出去
      P.forEach(([name, d1, d2, v, kind], i) => {
        const x = XP(i), y = Y(v), rej = kind === 'rej', probe = kind === 'probe', col = rej ? '#5d6370' : '#a4451c';
        if (prev >= 0) s += `<line x1="${XP(prev)}" y1="${Y(P[prev][3]).toFixed(1)}" x2="${x}" y2="${y.toFixed(1)}" stroke="${rej ? '#8a93a3' : '#a4451c'}" stroke-width="2" stroke-opacity=".75"${rej ? ' stroke-dasharray="4 3"' : ' pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"'} opacity="0">${rej ? aShow(tp(i)) : aDraw(tp(i) - 0.3, 0.35)}</line>`;
        if (!rej) prev = i;
        const below = name === 'ct106' || name === 'ct9';   // 與鄰點的標籤太近：放到點的下方
        s += `<circle cx="${x}" cy="${y.toFixed(1)}" r="5" fill="${probe || rej ? '#fff' : '#a4451c'}" stroke="${rej ? '#8a93a3' : '#a4451c'}" stroke-width="2" opacity="0">${aShow(tp(i))}</circle>`
          + lab(x, below ? y + 19 : y - 10, v, col, tp(i) + 0.1)
          + txA(x, 354, 'an-s', d1, tp(i), undefined, 'text-anchor="middle"')
          + (d2 ? txA(x, 370, 'an-s', d2, tp(i), undefined, 'text-anchor="middle"') : '')
          + (rej ? txA(x, y + 19, 'an-s', '沒採用', tp(i) + 0.1, undefined, `text-anchor="middle" style="${ch4Halo};fill:#d1343a;font-weight:700"`) : '');
      });
      const i44 = P.findIndex(u => u[0] === 'ct44'), i112 = P.length - 1, tEnd = tp(P.length - 1) + 0.5;
      // 圖例（放在右下角的空白處）：空心＝試探調整、灰＝第一次借等級
      s += `<g opacity="0">${aShow(tp(3) + 0.2)}<circle cx="452" cy="289" r="5" fill="#fff" stroke="#a4451c" stroke-width="2"/>${tx(462, 293, 'an-s', '空心＝試探性上傳後整批調整')}`
        + `<circle cx="452" cy="311" r="5" fill="#fff" stroke="#8a93a3" stroke-width="2"/>${tx(462, 315, 'an-s', '灰＝第一次借等級，Public 掉分')}</g>`;
      if (i44 >= 0) s += txA(XP(i44) - 12, (Y(P[i44][3]) - 31).toFixed(1), 'an-b', '改良後採用', tEnd, undefined, 'text-anchor="end"');
      s += txA(634, (Y(P[i112][3]) - 28).toFixed(1), 'an-b', '常客優先猜', tEnd + 0.3, undefined, 'text-anchor="end"')
        + `<circle cx="${XP(i112)}" cy="${Y(P[i112][3]).toFixed(1)}" r="11" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(tEnd + 0.3)}</circle>`;
      return {steps: ['① 模擬考：一步步加上借等級', '② 同期 Public 榜：從同一個起點算起'],
        svg: `<svg viewBox="0 0 640 394" role="img" aria-label="加入借來的等級之後，模擬考和同期 Public 榜的分數一起上升的動畫">${s}</svg>`,
        note: '都換算成每 100 題拿幾分。綠＝模擬考：第一點是第 4 章只看棋的模型；「借等級」那一點累積了好幾步改良，也含兩條看過試探結果才定的禁止規則；下一點換成學出來的合併，禁止規則共四條。橘＝Public 榜；空心＝試探性上傳之後整批調整的版本（第 13 頁）；灰＝第一次加入借等級的上傳，Public 掉分，沒採用。'};
    };
    ANIM.ch4_climb.dur = 9;

    // ---- 第 12 頁右欄：閘門＋「採用」框（場景式；採用框是主角）----
    //   ① 左邊一疊試過的想法、一道門「模擬考門檻（重訓也要過）」 ② 被擋下的掉進灰色「沒過」堆（寫 4 個代表名字），
    //      過門的綠卡依故事分 4 組進「採用」框 ③ 另一小組橘卡：靠 Public 試探確認、沒走模擬考門檻（第 13 頁） ④ 大字結論
    //   綠卡上的數字是各自加入當時在模擬考量到的「每 100 題多幾分」，口徑不同、不能相加；出處見 NUM.ch4.kept 與講稿。
    ANIM.ch4_forest = () => {
      const K = NUM.ch4.kept, CW = 110, CH = 38, GX = 124;
      const GRP = [['每一手的證據（第 5–6 頁）', '#0072b2', '#e1eefa'], ['看對手（第 7 頁）', '#8a6a00', '#fbf0d2'],
        ['模型與訓練（第 8–9 頁）', '#5d6370', '#eef0f3'], ['認出對手（第 10–11 頁）', '#0b6e69', '#e3f1ef']];
      const PX = g => [148, 392][g % 2], PY = g => [92, 204][g >> 1];
      const gray = [['梯度提升樹 GBDT', [5, 248]], ['', [9, 252]], ['循環網路 GRU', [5, 270]], ['一級級問 CORAL', [5, 292]], ['', [8, 300]], ['Transformer 加寬', [5, 314]]];
      const T0 = 0.3, T3 = 6.3, T4 = 7.4;   // 設計時間
      const tm = (vals, times) => `<animateTransform attributeName="transform" type="translate" dur="${AD}s" repeatCount="indefinite" values="${vals.map(v => v.join(',')).join(';')}" keyTimes="${kt(times)}"/>`;
      // 出場時間：灰卡 1.1 起每 0.7 秒一張；綠卡第 g 組 1.5 + g 秒起、每張隔 0.22 秒
      const ev = [];
      gray.forEach((c, i) => ev.push(['gray', i, 1.1 + i * 0.7]));
      K.forEach((grp, g) => grp.forEach((c, j) => ev.push(['green', [g, j], 1.5 + g * 1.0 + j * 0.22])));
      ev.sort((a, b) => a[2] - b[2]);
      const NL = ev.length, deckY = L => 92 - L * 2;
      let s = `<rect x="4" y="40" width="106" height="74" rx="8" fill="#f4f5f7" stroke="#c3c8d0" stroke-dasharray="4 3" opacity="0">${aShow(T0)}</rect>`;
      ev.forEach((e, k) => { const L = NL - 1 - k; s += `<rect x="8" y="${deckY(L)}" width="96" height="18" rx="3" fill="#fff" stroke="#9aa1ac" opacity="0">${aShow(T0, e[2])}</rect>`; });
      s += txA(56, 130, 'an-k', '試過的想法', T0, undefined, 'text-anchor="middle" style="font-weight:700;fill:#1d1f24"');
      // 門：牌子三行＋門柱
      s += `<g opacity="0">${aShow(T0)}<rect x="60" y="2" width="128" height="56" rx="6" fill="#e3f1ef" stroke="#0b6e69"/>`
        + `<rect x="${GX - 4}" y="58" width="8" height="170" rx="2" fill="#1d1f24"/><rect x="${GX - 14}" y="226" width="28" height="6" rx="2" fill="#1d1f24"/></g>`
        + txA(GX, 18, 'an-t', '模擬考門檻', T0, undefined, 'text-anchor="middle"')
        + txA(GX, 35, 'an-s', '+0.25／認人 +0.10', T0, undefined, 'text-anchor="middle" style="fill:#0b6e69;font-weight:700"')
        + txA(GX, 52, 'an-s', '重訓也要過', T0, undefined, 'text-anchor="middle" style="fill:#0b6e69;font-weight:700"');
      // 沒過堆
      s += `<rect x="2" y="240" width="116" height="98" rx="8" fill="#eef0f3" stroke="#c3c8d0" opacity="0">${aShow(1.0)}</rect>`
        + txA(61, 354, 'an-k', '沒過', 1.0, undefined, 'text-anchor="middle" style="font-weight:700"');
      // 採用框與四組
      s += `<rect x="140" y="62" width="496" height="254" rx="10" fill="#fff" stroke="#0b6e69" stroke-width="2.5" opacity="0">${aShow(1.3)}</rect>`
        + txA(152, 83, 'an-t', '採用：留在最終系統的訊號', 1.3, undefined, 'style="fill:#0b6e69"');
      GRP.forEach(([title, col, tint], g) => {
        const t = 1.4 + g * 1.0;
        s += `<rect x="${PX(g)}" y="${PY(g)}" width="236" height="106" rx="7" fill="${tint}" stroke="${col}" stroke-opacity=".6" opacity="0">${aShow(t)}</rect>`
          + txA(PX(g) + 8, PY(g) + 15, 'an-k', title, t, undefined, `style="font-weight:700;fill:${col}"`);
      });
      let back = '', front = '';
      ev.forEach(([kind, id, t]) => {
        if (kind === 'gray') {
          const [name, to] = gray[id], top = [8, 54], gate = [GX - 4 - 106, 70];
          const g = `<g transform="translate(${top.join(',')})" opacity="0">${tm([top, top, gate, gate, to, to], [0, t, t + 0.35, t + 0.45, t + 0.85, AD])}${aShow(t)}`
            + `<rect x="0" y="0" width="106" height="20" rx="3" fill="#fff" stroke="#9aa1ac">${aSeq('fill', [[0, '#fff'], [t + 0.45, '#fff'], [t + 0.65, '#d3d7de'], [AD, '#d3d7de']])}</rect>`
            + (name ? `<text x="53" y="14.5" class="an-s" text-anchor="middle" style="fill:#1d1f24;font-weight:700">${name}</text>` : '') + `</g>`;
          if (name) front += g; else back += g;
        } else {
          const [g, j] = id, [name, val] = K[g][j], from = [GX + 8, 62], to = [PX(g) + 6 + (j % 2) * 114, PY(g) + 22 + (j >> 1) * 42];
          front += `<g transform="translate(${from.join(',')})" opacity="0">${tm([from, from, to, to], [0, t, t + 0.55, AD])}${aShow(t)}`
            + `<rect x="0" y="0" width="${CW}" height="${CH}" rx="5" fill="#bfe3dd" stroke="#0b6e69" stroke-width="1.4"/>`
            + `<text x="${CW / 2}" y="15" class="an-s" text-anchor="middle" style="fill:#1d1f24;font-weight:700">${name}</text>`
            + `<text x="${CW / 2}" y="32" class="an-s" text-anchor="middle" style="fill:#0b6e69;font-weight:700">${val}</text></g>`;
        }
      });
      s += back + front;
      // 橘色小組：靠 Public 試探確認，沒走模擬考門檻
      s += `<rect x="140" y="324" width="496" height="58" rx="8" fill="#fbeee6" stroke="#a4451c" opacity="0">${aShow(T3)}</rect>`
        + txA(152, 342, 'an-k', 'Public 試探確認（第 13 頁）：沒走模擬考門檻', T3, undefined, 'style="font-weight:700;fill:#a4451c"');
      ['整批改 1D 為 A', '整批改 4D 為 5D', '四條禁止規則'].forEach((n, i) => {
        const x = 148 + i * 156, t = T3 + 0.2 + i * 0.2;
        s += `<rect x="${x}" y="350" width="150" height="24" rx="4" fill="#f6d9c7" stroke="#a4451c" opacity="0">${aShow(t)}</rect>`
          + txA(x + 75, 366.5, 'an-s', n, t, undefined, 'text-anchor="middle" style="fill:#1d1f24;font-weight:700"');
      });
      const nKept = K.reduce((a, g) => a + g.length, 0);
      s += txA(388, 412, 'an-v', `留下 ${nKept} 個訊號；其餘方向都被擋下`, T4, undefined, 'text-anchor="middle" style="font-size:20px;fill:#0b6e69"')
        + txA(388, 434, 'an-s', '11-06 之後的 16 個代表方向只過了 2 個', T4 + 0.3, undefined, 'text-anchor="middle"');
      return {steps: ['① 試過的想法都要過門檻', '② 大多被擋下；過門的進「採用」', '③ 另有幾條靠 Public 試探確認', `④ 留下 ${nKept} 個訊號`],
        svg: `<svg viewBox="0 0 640 446" role="img" aria-label="閘門動畫：試過的想法過模擬考門檻，留在最終系統的 ${nKept} 個訊號分四組放進採用框，另有三條靠 Public 試探確認">${s}</svg>`,
        note: '綠卡上的數字是模擬考每 100 題多幾分，各自在加入當時量的，不能直接相加。AI 老師那張考的是從沒訓練過的那批玩家；選平均拿分最多是一開始就用的作答規則，事後才量。橘卡靠 Public 試探確認，沒走模擬考門檻（第 13 頁）。'};
    };
    ANIM.ch4_forest.dur = 12;
  }
