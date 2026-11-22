  // ---- 名詞註記（沿用 go_story；詞條改成收尾簡報的新手版）----
  // 每一頁第一次出現的術語加虛線底；滑鼠指向、鍵盤聚焦或手指點一下，顯示一行說明。頁面文字不變。
  // 名詞與說明只改這裡（[術語, 說明]）；英數術語只在前後不是英數、底線、點、斜線時才標，避免標到檔名裡。
  {
    const TERMS = [
      // 白話版（2026-11-18 第二輪）：先講人話，術語本身也給一句外行懂的解釋
      ['KataGo', '世界頂尖的免費開源圍棋 AI，棋力遠超職業棋手。我們只借它「看棋盤」的能力，不用它下棋。'],
      ['Human SL', 'KataGo 團隊另外訓練的「模仿人類」版本：告訴它棋手是幾級，它就猜那個程度的人會下哪裡。我們反過來問：這一手像幾級的人下的？'],
      ['humanv0', 'Human SL 公開下載的模型檔名稱；比賽允許使用公開的預訓練模型。'],
      ['幽玄の間', '日本棋院的線上對弈平台；本比賽的棋譜都來自這裡（2021–2023）。'],
      ['讓子', '兩人棋力差較多時，弱的一方先在棋盤上擺 2–9 顆子；差越多擺越多。'],
      ['段級位', '級的數字越小越強（10 級比 12 級強）；1 級再往上是 1 段，段的數字越大越強。'],
      ['等級頭', '我們自己加的 10 個小網路，各自專門判斷「這一手像不像某一級的人下的」，用本比賽的等級標籤訓練。'],
      ['內部印象', 'AI 看完棋盤後留下的一大串數字，記著棋形和風格，比最後給的一個分數資訊更多。'],
      ['模擬考', '把一部分訓練玩家藏起來當考題，用沒看過他們的模型作答、自己改分；一個改動要不要採用，先看模擬考。'],
      ['標籤', '伺服器上登記的玩家等級，也就是資料給的「標準答案」；它不一定等於這幾局棋表現出來的實力。'],
      ['交叉驗證', '本簡報叫「模擬考」：把一部分訓練玩家藏起來當考題，用沒看過他們的模型作答、自己改分。'],
      ['CV', '交叉驗證的縮寫，本簡報叫「模擬考」。'],
      ['fold', '模擬考把訓練玩家分成 5 份，每份叫一個 fold，輪流當考題。'],
      ['集成', '多個模型各自作答再平均，像請多位評審打分取平均，比單一模型穩。'],
      ['期望分', '把「差一級也有 0.368 分」算進去的平均得分；沒把握時，選中間的等級常比選最可能的等級划算。'],
      ['baseline', '主辦方提供的參考做法，和它在 Public 榜的分數。'],
      ['Public', 'Public 榜：比賽期間公開的 400 題，每次上傳立刻看到分數。'],
      ['Private', 'Private 榜：隱藏的另外 400 題，比賽結束才公布，決定最終名次。'],
      ['每 100 題', '比賽分數是平均每題得分（滿分 1）；乘以 100，就是每 100 題拿幾分。'],
      ['seed', '隨機起點。同一個設定換一個隨機起點重新訓練，結果還成立，才不是運氣。'],
      ['信賴區間', '把考題重新抽樣幾千次、每次重算差距得到的範圍；整個範圍都大於 0，才算確定有效。'],
      ['參考等級', '認出對手是哪位老玩家後，借用他已知的等級，再依配對規則（互先多半同級、讓先多半差一級）推回本人等級。'],
      ['老玩家', '訓練資料裡的 8,007 位玩家；他們的等級是已知的。'],
      ['常客', '名下棋局特別多的玩家：只占 11.6%，卻下了 48% 的訓練棋局；認人時優先猜他們。'],
      ['認人模型', '看一個人的棋風，猜他是訓練資料裡 8,007 位玩家中的哪一位。'],
      ['後驗', '模型看完所有證據後，對 10 個等級各有幾成把握。'],
      ['帳本', '把每次上傳的檔案和 Public 分數記成一本帳，用分數差回推哪些題改對。'],
      ['試探性上傳', '故意只改一小批題再上傳，從分數變化回推：這批題裡幾題變對、幾題差一級。'],
      ['探測', '故意只改一小批題再上傳，從分數變化回推：這批題裡幾題變對、幾題差一級。'],
      ['禁止規則', '看過試探性上傳的結果後寫成的固定規則，共四條（例：原本答 A 的題不改）；這些方向已證實容易改錯，之後每次改答案都先過這幾條。'],
      ['護欄', '看過試探性上傳的結果後寫成的固定規則，共四條（例：原本答 A 的題不改）；這些方向已證實容易改錯，之後每次改答案都先過這幾條。'],
      ['推移', '把猶豫在相鄰兩級之間的題，整批往其中一級調。'],
      ['後驗帶', '按模型把握度圈出的一批題（不看答案），所以 Public 和 Private 兩邊都圈得出同類的題。'],
      ['鎖定', '試探性上傳已確認方向的題；之後的版本不再改它們的答案。'],
      ['對比損失', '訓練時多一個要求：同一個人的兩份棋譜，模型給的判斷要接近；讓模型學「這個人」，不學「這幾局的運氣」。'],
      ['交叉熵', '最常見的訓練目標：猜對等級就給獎勵，把正確答案猜得越沒把握，懲罰越重。'],
      ['Top-1', '第一個猜的就猜中的比例。'],
      ['AUC', '隨機抽一位較強、一位較弱的玩家，模型把較強的排在前面的機率；50% 等於亂猜，100% 全對。'],
      ['MLP', '多層感知器：最基本的神經網路，幾層「加權相加再過一個開關」疊起來。'],
      ['ResNet', '殘差卷積網路：看棋盤用的深層卷積網路，每一層把新算的結果加回原輸入，疊很深也好訓練。'],
      ['殘差塊', '每一層把新算出的結果加回原本的輸入，疊深了也好訓練。'],
      ['擴張卷積', '跳著看的卷積：間隔 1、2、4、8、16 手各看一次，疊 5 層就能看到前後 63 手。'],
      ['CNN', '卷積神經網路：用同一組小窗口沿著序列（或棋盤）滑過去找局部樣式。'],
      ['policy head', 'AI 老師最後預測「下一手下在哪」的那一層。'],
      ['邏輯斯迴歸', '最簡單的分類模型：把幾路證據各乘一個權重相加，換算成每一級的機率；權重從模擬考資料學出來。'],
      ['GBDT', '梯度提升樹：很多棵決策樹一棵接一棵修正前一棵的錯；表格資料常用，這裡輸給 MLP。'],
      ['GRU', '循環神經網路的一種：一手一手往下讀，邊讀邊更新記憶。'],
      ['CTM', '一種反覆讀取的網路：對一局的描述固定讀 8 次、每次修正判斷。'],
      ['CORAL', '有序輸出：不直接猜 10 級之一，而是逐級回答「比這一級強嗎」。'],
      ['注意力池化', '讓模型自己學每一手、每一局的權重再加總，取代直接平均。'],
      ['Transformer', '一種能同時看一整段棋步、找出彼此關聯的神經網路（近年的語言模型也用它）；我們的認人模型用它把一局前 256 手壓成一個棋風指紋。'],
      ['執色', '同一題裡，這位玩家都執黑或都執白。'],
      ['互先', '棋力相近的一般對局：猜子決定誰先下，先下的一方終局要補對方幾目。'],
      ['讓先', '棋力差一點時，弱的一方固定先下、不補目。'],
      ['積分', '伺服器依勝負加減的分數；積分到了就升段（級），掉了就降。'],
      ['分身帳號', '同一個人開的另一個帳號，棋風幾乎一樣。'],
      ['定案檔', '我們選定、拿去算 Private 名次的那一份答案檔（ct118e_a2b4：ct116 的 Private 答案再改 4 題）。'],
      ['ct118e_a2b4', '定案檔（11-23）：Public 和 ct118e 一樣，Private 是 ct116 再把 4 題 A 改 B。'],
      ['ct116', '我們的上傳檔編號；定案檔的 Private 答案以它為底（含參賽者決定的 8 題）。'],
      ['ct118e', 'Public 最高的檔（0.8587182）；只改了 Public 榜的題，Private 答案和 ct116 完全一樣，是定案檔的備援。'],
      ['ct118d', '11-18 的最後一次上傳；只改了 Public 榜的題，Private 答案和 ct116 完全一樣。'],
      ['ct112', '我們的上傳檔編號；ct112 是還沒做逐題人工修改的最後一版，是第一個備援檔。'],
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
        document.querySelectorAll('main > .page').forEach(mark);
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
