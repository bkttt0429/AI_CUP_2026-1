
  // ---- 長條圖與 data-n 數字 ----
  Object.keys(CHARTS).forEach(id => { const el = document.getElementById(id); if (!el) return; try { el.innerHTML = CHARTS[id](); } catch (err) { el.textContent = '圖表載入失敗：' + err.message; } });

  // 放進頁面＋進度條。平常只露出底部細線（.an-mini），滑到底部才展開控制列；拖曳時暫停、放開後照原本狀態繼續；
  // 每次重新看到動畫（切頁、捲回來）就從頭播；減少動態效果時停在完成畫面（ANIM.x.final，預設倒數 1.5 秒），仍可拖曳。
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ctlHTML = d => '<div class="an-zone"><div class="an-ctl"><button type="button" class="an-pp" aria-label="暫停">❚❚</button>'
    + `<input type="range" class="an-seek" min="0" max="${d - 0.05}" step="0.05" value="0" aria-label="拖曳動畫進度"><span class="an-time">0.0 秒</span></div></div><div class="an-mini"></div>`;
  const anIO = 'IntersectionObserver' in window
    ? new IntersectionObserver(es => es.forEach(e => { if (e.target._an) e.target._an(e.isIntersecting); }), {threshold: 0.3}) : null;
  document.querySelectorAll('.anim[data-anim]').forEach(el => {
    try {
      const f = ANIM[el.dataset.anim];
      AD = (f && f.dur) || 9; HOLD = AD - 0.7; RAW = !!(f && f.raw);   // 這段的長度、要不要壓縮前搖（產生動畫字串時會用到）
      const an = f && f(), dur = AD, fin = (f && f.final) || AD - 1.5;
      if (!an) { el.remove(); return; }
      el.innerHTML = (an.steps && an.steps.length ? '<div class="an-steps">' + an.steps.map(t => `<span>${t}</span>`).join('') + '</div>' : '')
        + (an.tech ? `<div class="an-tech"><b>技術</b><span>${an.tech}</span></div>` : '')
        + `<div class="an-stage">${an.svg}${ctlHTML(dur)}</div>` + (an.note ? `<div class="an-note">${an.note}</div>` : '');
      const sv = el.querySelector('.an-stage > svg'), ctl = el.querySelector('.an-ctl'), mini = el.querySelector('.an-mini');
      if (!sv || !sv.setCurrentTime) { el.querySelector('.an-zone').remove(); mini.remove(); return; }
      const seek = ctl.querySelector('.an-seek'), pp = ctl.querySelector('.an-pp'), tm = ctl.querySelector('.an-time');
      let playing = true, resume = false, raf = 0, away = true;
      const show = t => { seek.value = t.toFixed(2); tm.textContent = t.toFixed(1) + ' 秒'; mini.style.width = (t / dur * 100).toFixed(2) + '%'; };
      const play = p => { playing = p; if (p) sv.unpauseAnimations(); else sv.pauseAnimations();
        pp.textContent = p ? '❚❚' : '▶'; pp.setAttribute('aria-label', p ? '暫停' : '播放'); };
      const loop = () => { if (!el.classList.contains('scrub')) show(sv.getCurrentTime() % dur); raf = requestAnimationFrame(loop); };
      const endScrub = () => { if (!el.classList.contains('scrub')) return; el.classList.remove('scrub'); if (resume) play(true); resume = false; };
      pp.addEventListener('click', () => play(!playing));
      seek.addEventListener('pointerdown', () => { resume = playing; play(false); el.classList.add('scrub'); });
      seek.addEventListener('input', () => { sv.setCurrentTime(+seek.value); show(+seek.value); });
      ['change', 'pointerup', 'pointercancel'].forEach(ev => seek.addEventListener(ev, endScrub));
      sv.querySelectorAll('[data-seek]').forEach(g => {   // 動畫裡的分段按鈕：點了跳到那一段
        const go = () => { const t = +g.dataset.seek; sv.setCurrentTime(t); show(t); };
        g.addEventListener('click', go);
        g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      });
      el._an = vis => {   // 看得到才更新進度條；從看不到變成看得到時從頭播
        if (vis && !raf) { if (away && playing) sv.setCurrentTime(0); away = false; loop(); }
        else if (!vis) { cancelAnimationFrame(raf); raf = 0; away = true; }
      };
      if (reduceMotion) { sv.setCurrentTime(fin); play(false); }
      show(reduceMotion ? fin : 0);
      if (anIO) anIO.observe(el); else loop();
    } catch (err) { el.textContent = '動畫載入失敗：' + err.message; }
  });
  AD = 9; HOLD = 8.3; RAW = false;

  // ---- 章節選單 ----
  const menu = document.getElementById('ch-menu'), route = document.getElementById('route');
  let mh = '<a href="#cover"><span>封面</span><span class="meta">第 1 頁</span></a><hr>';
  pages.forEach((p, i) => {
    // 分隔頁（data-divider，例如「1106 報告」）在選單裡當成分段標題
    if (p.dataset.divider) mh += `<hr><a href="#${p.id}"><span>${p.dataset.divider}</span><span class="meta">第 ${i + 1} 頁</span></a>`;
    if (p.dataset.chtitle) {
      const n = pages.filter(q => q.dataset.ch === p.dataset.ch).length;
      mh += `<a href="#${p.id}"><span>${p.dataset.ch} · ${p.dataset.chtitle}</span><span class="meta">第 ${i + 1}${n > 1 ? '–' + (i + n) : ''} 頁</span></a>`;
      if (route) route.insertAdjacentHTML('beforeend', `<li><b>第 ${p.dataset.ch} 章</b><a href="#${p.id}">${p.dataset.chtitle}</a><span class="c">${n} 頁</span></li>`);
    }
  });
  menu.innerHTML = mh;
  document.getElementById('menu-btn').addEventListener('click', () => menu.showModal());
  menu.addEventListener('click', e => {
    if (e.target === menu) { menu.close(); return; }
    const a = e.target.closest('a'); if (!a) return;
    e.preventDefault(); menu.close(); go(pages.findIndex(p => '#' + p.id === a.getAttribute('href')));
  });
  if (route) route.addEventListener('click', e => {
    const a = e.target.closest('a'); if (!a) return;
    e.preventDefault(); go(pages.findIndex(p => '#' + p.id === a.getAttribute('href')));
  });

  // ---- 換頁（只有簡報模式：一次一頁）----
  const pg = document.getElementById('pg'), bPrev = document.querySelector('.pg-prev'), bNext = document.querySelector('.pg-next');
  function updatePager() { pg.textContent = `${cur + 1}／${pages.length}`; bPrev.disabled = cur === 0; bNext.disabled = cur === pages.length - 1; }
  function go(i) {
    if (i < 0 || i >= pages.length) return;
    cur = i;
    pages.forEach((p, j) => p.classList.toggle('cur', j === cur)); window.scrollTo(0, 0); fitFigs(pages[cur]);
    history.replaceState(null, '', location.pathname + location.search + '#' + pages[cur].id);
    updatePager(); updateChrome();
  }
  // 頁首的目前章節與進度線；講稿抽屜的內容
  const chTitle = {}; pages.forEach(p => { if (p.dataset.chtitle) chTitle[p.dataset.ch] = p.dataset.chtitle; });
  const curCh = document.getElementById('cur-ch'), prog = document.getElementById('deck-prog');
  const NOTES = window.NOTES || {}, notes = document.getElementById('notes'), notesB = document.getElementById('notes-b'), notesT = document.getElementById('notes-t');
  function updateChrome() {
    const ch = pages[cur].dataset.ch;
    if (curCh) curCh.textContent = ch && chTitle[ch] ? `第 ${ch} 章 · ${chTitle[ch]}` : '';
    if (prog) prog.style.width = ((cur + 1) / pages.length * 100).toFixed(2) + '%';
    if (notes && !notes.hidden) fillNotes();
  }
  function fillNotes() {
    const n = NOTES[cur + 1];
    notesT.textContent = `第 ${cur + 1} 頁${n ? ' · ' + n.t : ''}`;
    notesB.innerHTML = n ? n.h : '<p>這一頁沒有講稿。</p>';
  }
  bPrev.addEventListener('click', () => go(cur - 1));
  bNext.addEventListener('click', () => go(cur + 1));
  window.addEventListener('resize', () => fitFigs(pages[cur]));
  window.addEventListener('hashchange', () => { const k = pages.findIndex(p => '#' + p.id === location.hash); if (k >= 0 && k !== cur) go(k); });

  // ---- 鍵盤：→／PageDown／空白 下一頁，←／PageUp 上一頁，Home／End 首尾 ----
  document.addEventListener('keydown', e => {
    if (e.target.closest('input,textarea') || menu.open) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(cur + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(cur - 1); }
    else if (e.key === 'Home') { e.preventDefault(); go(0); }
    else if (e.key === 'End') { e.preventDefault(); go(pages.length - 1); }
    else if ((e.key === 'n' || e.key === 'N') && notes) { notes.hidden = !notes.hidden; if (!notes.hidden) fillNotes(); }
  });

  // ---- 手機：左右滑動換頁 ----
  let tx0 = null, ty0 = null;
  document.addEventListener('touchstart', e => { if (e.touches.length === 1) { tx0 = e.touches[0].clientX; ty0 = e.touches[0].clientY; } }, {passive: true});
  document.addEventListener('touchend', e => {
    if (tx0 === null || e.target.closest('.an-ctl,input')) { tx0 = null; return; }
    const dx = e.changedTouches[0].clientX - tx0, dy = e.changedTouches[0].clientY - ty0; tx0 = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) go(cur + (dx < 0 ? 1 : -1));
  }, {passive: true});

  // ---- 起始頁：#頁 id ----
  const h0 = pages.findIndex(p => '#' + p.id === location.hash);
  go(h0 >= 0 ? h0 : 0);
})();

</script>
