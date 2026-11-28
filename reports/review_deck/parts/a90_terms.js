  // ---- 名詞註記（沿用 go_deck；詞條改成兩項競賽檢討用的幾個）----
  // 每一頁第一次出現的術語加虛線底；滑鼠指向、鍵盤聚焦或手指點一下，顯示一行說明。頁面文字不變。
  // 名詞與說明只改這裡（[術語, 說明]）；英數術語只在前後不是英數、底線、點、斜線時才標，避免標到檔名裡。
  {
    const TERMS = [
      ['Public', 'Public 榜：競賽期間公開的半數題目，每次上傳即公布分數。'],
      ['Private', 'Private 榜：未公開的另一半題目，競賽結束後公布分數，為最終排名依據。'],
      ['離線驗證', '保留部分有標籤的訓練玩家作為題目，由未見過這些玩家的模型作答並自行評分；棋力預測採交叉驗證。'],
      ['採用檔', '經選定、作為 Private 排名依據的提交檔。'],
      ['否決依據', '事先訂定的下限：新檔的 Public 分數未低於現行採用檔 0.0025 以上時，不因 Public 而退回。'],
      ['回推標籤', '由 Public 分數反推答錯的題目後直接修正的答案，僅對已公布分數的 Public 題目有效。'],
      ['80% 區間', '賽前預估的範圍：依本隊估計，實際成績有 80% 的機率落於其中。'],
      ['90% 區間', '賽前預估的範圍：依本隊估計，實際成績有 90% 的機率落於其中。'],
      ['Human SL', 'KataGo 作者公開的圍棋 AI（b18c384nbt-humanv0），以大量附段位的人類棋譜訓練，可估計特定段位的棋手在某一局面的著手。本隊的逐手特徵均出自此網路。'],
      ['特徵來源', '將棋譜轉換為數值特徵的方式。同一份棋譜改由另一個網路處理，可得到不同的資訊。'],
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
