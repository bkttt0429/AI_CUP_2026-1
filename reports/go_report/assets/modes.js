/* Page behaviour for the CTM lessons: three modes, present-mode paging, the shared lesson
   dialog (sections are moved in and back out, so notes mode and the dialog show the same DOM),
   and URL / localStorage state. Classic script, loaded last. */
(function () {
  const body = document.body;
  const talk = !!body.dataset.talk;
  if (talk) body.className = 'm-present';
  const pages = [...document.querySelectorAll('.page')];
  const secs = [...document.querySelectorAll('.lesson')];
  const dlg = document.getElementById('dlg');
  const pg = document.getElementById('pg');
  const MODES = ['present', 'read', 'notes'];
  let i = 0, openSec = -1, lastBtn = null;

  const store = {
    get() { try { return localStorage.getItem('ctm-mode'); } catch (e) { return null; } },
    set(v) { try { localStorage.setItem('ctm-mode', v); } catch (e) { /* storage blocked: just don't remember */ } },
  };
  const indexMode = !pages.length ? new URLSearchParams(location.search).get('mode') ?? store.get() : null;
  const mode = () => pages.length ? body.className.replace(/^m-/, '') : MODES.includes(indexMode) ? indexMode : 'read';
  const chapterUrl = (file, m, hash) => `${file}?mode=${m}${hash ? '#' + hash : ''}`;

  /* ---------- header chapter menu (also used by the index) ---------- */
  const titleBtn = document.querySelector('.ch-menu-btn');
  const chapterMenu = document.querySelector('.ch-menu-dlg');
  if (titleBtn) {
    const header = titleBtn.closest('.top');
    const syncHeaderHeight = () => body.style.setProperty('--top-bar-h', header.getBoundingClientRect().height + 'px');
    syncHeaderHeight();
    new ResizeObserver(syncHeaderHeight).observe(header);
  }
  if (chapterMenu && titleBtn) {
    const items = [...chapterMenu.querySelectorAll('a[href]')];
    function positionMenu() {
      if (!chapterMenu.open) return;
      const anchor = titleBtn.getBoundingClientRect();
      const mobile = matchMedia('(max-width:600px)').matches;
      const left = mobile ? 8 : Math.max(8, Math.min(anchor.left, innerWidth - chapterMenu.offsetWidth - 8));
      const top = anchor.bottom + 6;
      chapterMenu.style.left = left + 'px';
      chapterMenu.style.top = top + 'px';
      chapterMenu.style.maxHeight = Math.max(0, innerHeight - top - 8) + 'px';
    }
    function openMenu() {
      if (dlg?.open || chapterMenu.open) return;
      syncLinks();
      chapterMenu.showModal();
      titleBtn.setAttribute('aria-expanded', 'true');
      positionMenu();
      const current = chapterMenu.querySelector('[aria-current="page"]') || items[0];
      current.focus({ preventScroll: true });
      current.scrollIntoView({ block: 'nearest' });
    }
    titleBtn.addEventListener('click', openMenu);
    chapterMenu.addEventListener('close', () => {
      titleBtn.setAttribute('aria-expanded', 'false');
      titleBtn.focus({ preventScroll: true });
    });
    chapterMenu.addEventListener('click', e => {
      if (e.target.closest('a[aria-current="page"]')) { e.preventDefault(); chapterMenu.close(); return; }
      if (e.target !== chapterMenu) return;
      const r = chapterMenu.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) chapterMenu.close();
    });
    window.addEventListener('resize', positionMenu);
    window.addEventListener('scroll', positionMenu, true);
    // Capture owns all keys while modal, before the page / derivation router sees them.
    document.addEventListener('keydown', e => {
      if (chapterMenu.open) {
        e.stopImmediatePropagation();
        const at = items.indexOf(document.activeElement);
        let next;
        if (e.key === 'ArrowDown') next = (at + 1) % items.length;
        else if (e.key === 'ArrowUp') next = (at - 1 + items.length) % items.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = items.length - 1;
        if (next !== undefined) { e.preventDefault(); items[next].focus(); }
        else if (e.key === 'Escape') { e.preventDefault(); chapterMenu.close(); }
        else if (['ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', ' ', 'd', 'D'].includes(e.key)) e.preventDefault();
        return;
      }
      if (dlg?.open || e.defaultPrevented || e.isComposing || e.ctrlKey || e.altKey || e.metaKey) return;
      if (e.key !== 'g' && e.key !== 'G') return;
      const textControl = 'input, select, textarea, [contenteditable], [role="textbox"], [role="combobox"]';
      if ([e.target, document.activeElement].some(el => el instanceof Element && el.closest(textControl))) return;
      e.preventDefault(); openMenu();
    }, true);
  }
  // The index needs navigation only; it has no animation core, pager or lesson dialog.
  if (!pages.length) { syncLinks(); return; }
  NB.mountAll();
  body.dataset.mounted = String(Math.round(performance.now()));
  const dbody = dlg.querySelector('.dlg-body'), dtitle = document.getElementById('dlg-t');

  function syncURL() {
    try {
      const u = new URL(location.href);
      u.searchParams.set('mode', mode());
      if (openSec >= 0) u.searchParams.set('deriv', '1'); else u.searchParams.delete('deriv');
      // Chapter dialogs link their section; talk keeps the current report page's hash.
      const card = !talk && openSec >= 0 ? pages.find(p => p.dataset.sec === String(openSec)) : null;
      if (card) u.hash = card.id;
      else if (mode() === 'present' && pages[i]) u.hash = pages[i].id;
      history.replaceState(null, '', u);
    } catch (e) { /* some file:// contexts refuse replaceState; the page still works */ }
  }

  /* ---------- lesson dialog ---------- */
  secs.forEach(s => { s._home = document.createComment('lesson-home'); s.before(s._home); });

  function returnSec() {
    if (openSec < 0) return;
    const s = secs[openSec];
    (s.anims || []).forEach(a => a.pause());
    if (talk) s.hidden = true;
    s._home.after(s);
    openSec = -1;
  }
  function openPopup(k) {
    if (k < 0 || k >= secs.length) return;
    returnSec();
    const s = secs[k];
    if (talk) s.hidden = false;
    dbody.appendChild(s);
    openSec = k;
    dtitle.textContent = s.dataset.title;
    if (talk) document.getElementById('dlg-k').textContent = `第 ${s.dataset.chapter} 章 · 推導`;
    dlg.querySelector('.prev').disabled = k === 0;
    dlg.querySelector('.nextb').disabled = k === secs.length - 1;
    if (!dlg.open) { lastBtn = document.querySelector(`.deriv-open[data-sec="${k}"]`); dlg.showModal(); }
    dbody.scrollTop = 0;
    NB.mountIn(s);
    (s.anims || []).forEach(a => a.auto());
    syncURL();
  }
  dlg.addEventListener('close', () => {
    returnSec();
    if (lastBtn) lastBtn.focus({ preventScroll: true });
    syncURL();
  });
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
  dlg.querySelector('.x').onclick = () => dlg.close();
  dlg.querySelector('.prev').onclick = () => openPopup(openSec - 1);
  dlg.querySelector('.nextb').onclick = () => openPopup(openSec + 1);
  dlg.querySelector('.whole').onclick = () => {
    const s = secs[openSec];
    if (talk) {
      location.href = chapterUrl(`ch${s.dataset.chapter.padStart(2, '0')}.html`, 'notes', `sec-${s.dataset.local}`);
      return;
    }
    returnSec();                       // move the section home now: 'close' fires asynchronously
    lastBtn = null;
    dlg.close();
    setMode('notes');
    s.scrollIntoView({ block: 'start' });
    window.scrollBy(0, -60);
  };
  document.querySelectorAll('.deriv-open').forEach(b => { b.onclick = () => openPopup(+b.dataset.sec); });

  /* ---------- chapters: neighbours, mode-keeping links ---------- */
  const prevCh = body.dataset.prev || '', nextCh = body.dataset.next || '';
  const goChapter = (file, hash) => { if (file) location.href = chapterUrl(file, mode(), hash); };
  // Header menu, chapter footer and bridge links carry the current mode.
  function syncLinks() {
    document.querySelectorAll('.ch-menu-dlg a[href], .chapter-nav a[href], .bridge a[href]').forEach(a => {
      a.dataset.file ??= a.getAttribute('href').split(/[?#]/)[0];
      a.dataset.hash ??= a.getAttribute('href').split('#')[1] || '';
      if (a.closest('.ch-menu-dlg') || /^ch\d\d\.html$/.test(a.dataset.file)) a.setAttribute('href', chapterUrl(a.dataset.file, mode(), a.dataset.hash));
    });
  }
  const hint = document.querySelector('.pager .pg-hint');
  let hintTimer = 0;
  function say(text) {
    if (!hint) return;
    hint.textContent = text;
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => { hint.textContent = ''; }, 1800);
  }
  const pgPrev = document.querySelector('.pager .pg-prev'), pgNext = document.querySelector('.pager .pg-next');
  if (pgPrev) pgPrev.onclick = () => step(-1);
  if (pgNext) pgNext.onclick = () => step(1);

  // Question, trace and route anchors point inside pages; present mode selects their containing page.
  const hashTarget = hash => hash ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
  const pageIndex = target => target ? pages.indexOf(target.closest('.page')) : -1;
  function showHash(hash) {
    const at = pageIndex(hashTarget(hash));
    if (at < 0) return;
    show(at);
    // Keep the more precise question anchor after show() synchronizes the containing page.
    try { history.replaceState(null, '', hash); } catch (e) { /* file:// can refuse history writes */ }
  }
  document.addEventListener('click', e => {
    if (mode() !== 'present' || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const link = e.target.closest('a[href^="#"]');
    const at = link ? pageIndex(hashTarget(link.getAttribute('href'))) : -1;
    if (at >= 0) { e.preventDefault(); showHash(link.getAttribute('href')); }
  });
  window.addEventListener('hashchange', () => {
    if (mode() === 'present') showHash(location.hash);
  });

  /* ---------- pages and modes ---------- */
  function show(n, direction = n - i) {
    i = Math.max(0, Math.min(pages.length - 1, n));
    pages.forEach((p, k) => {
      p.classList.toggle('cur', k === i);
      if (k === i && mode() === 'present' && matchMedia('(max-width:600px)').matches) p.scrollTop = 0;
      (p.anims || []).forEach(a => {
        if (k !== i || mode() !== 'present') a.pause();
        else if (a.steps?.length) a.seek(direction < 0 || matchMedia('(prefers-reduced-motion: reduce)').matches ? a.steps.length - 1 : 0);
        else a.auto();
      });
    });
    const chapter = talk ? pages[i].dataset.chapter || pages[i].dataset.label || '封面' : body.dataset.chapter;
    pg.textContent = `${chapter} · ${i + 1}／${pages.length}`;
    if (pgPrev) pgPrev.disabled = i === 0 && (talk || !prevCh);
    if (pgNext) pgNext.disabled = i === pages.length - 1 && (talk || !nextCh);
    syncURL();
  }
  // → past the last page opens the next chapter; ← before the first returns to the previous chapter's last page
  function step(delta) {
    const n = i + delta;
    if (talk && (n < 0 || n >= pages.length)) return say(n < 0 ? '已是第一頁' : '已是最後一頁');
    if (n >= pages.length) return nextCh ? goChapter(nextCh, 'opening') : say('已是最後一頁');
    if (n < 0) return prevCh ? goChapter(prevCh, 'last') : say('已是第一頁');
    show(n, delta);
  }
  function setMode(m, direction = 1) {
    if (dlg.open) { returnSec(); dlg.close(); }
    body.className = 'm-' + m;
    document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === m));
    if (!talk) store.set(m);
    syncLinks();
    if (m === 'notes') NB.observeLessons();
    if (m === 'present') show(i, direction);
    else {
      pages.forEach(p => (p.anims || []).forEach(a => { if (a.steps?.length) a.seek(a.steps.length - 1); else a.pause(); }));
      secs.forEach(s => (s.anims || []).forEach(a => { if (a.steps?.length) a.seek(a.steps.length - 1); }));
      syncURL();
    }
  }
  document.querySelectorAll('.modes button').forEach(b => { b.onclick = () => { setMode(b.dataset.mode); if (b.dataset.mode !== 'present') window.scrollTo(0, 0); }; });

  function currentCardSec() {
    if (mode() === 'present') return pages[i].dataset.sec;
    const c = pages.find(p => p.dataset.sec !== undefined && p.getBoundingClientRect().bottom > 80);
    return c && c.dataset.sec;
  }
  // A focused control keeps the keys it uses itself: sliders, selects and text fields take the arrows
  // and PageUp/PageDown; buttons take Space. Arrows on a button (e.g. 推導 ▸ after Esc) still turn the page.
  const ARROW_OWNERS = 'input, select, textarea, [contenteditable], [role="slider"]';
  const SPACE_OWNERS = 'input, select, textarea, button, [contenteditable], [role="button"], [role="slider"]';
  function keyForControl(e) {
    const owners = e.key === ' ' ? SPACE_OWNERS : ARROW_OWNERS;
    return e.defaultPrevented || [e.target, document.activeElement].some(el => el instanceof Element && el.closest(owners));
  }
  document.addEventListener('keydown', e => {
    if (dlg.open) return;                         // Esc closes the dialog natively; arrows do nothing
    if ((e.key === 'd' || e.key === 'D') && mode() !== 'notes') {
      const s = currentCardSec();
      if (s !== undefined) openPopup(+s);
      return;
    }
    if (mode() !== 'present') return;
    if (['ArrowRight', 'ArrowLeft', 'PageDown', 'PageUp', ' '].includes(e.key) && keyForControl(e)) return;
    const a = pages[i].anims?.find(a => a.steps?.length);
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); if (!a?.next()) step(1); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); if (!a?.prev()) step(-1); }
    else if (e.key === 'Escape' && !talk) setMode('read');
  });

  /* ---------- talk-only page list ---------- */
  if (talk && document.querySelector('.page-list-dlg')) {   // pages without the list (e.g. test fixtures) skip it
    const list = document.querySelector('.page-list-dlg');
    const listBtn = document.querySelector('.pg-list');
    let listReturn = listBtn;
    const items = [...list.querySelectorAll('[data-page]')];
    function openList(trigger = listBtn) {
      if (dlg.open || list.open) return;
      listReturn = trigger;
      items.forEach(b => {
        if (+b.dataset.page === i) b.setAttribute('aria-current', 'page');
        else b.removeAttribute('aria-current');
      });
      listBtn.setAttribute('aria-expanded', 'true');
      titleBtn?.setAttribute('aria-expanded', 'true');
      list.showModal();
      items[i].focus({ preventScroll: true });
      items[i].scrollIntoView({ block: 'center' });
    }
    listBtn.addEventListener('click', () => openList(listBtn));
    titleBtn?.addEventListener('click', () => openList(titleBtn));
    list.querySelector('.page-list-close').addEventListener('click', () => list.close());
    list.addEventListener('close', () => {
      listBtn.setAttribute('aria-expanded', 'false');
      titleBtn?.setAttribute('aria-expanded', 'false');
      listReturn.focus({ preventScroll: true });
    });
    list.addEventListener('click', e => {
      const item = e.target.closest('[data-page]');
      if (item) { e.preventDefault(); show(+item.dataset.page); list.close(); return; }
      if (e.target !== list) return;
      const r = list.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) list.close();
    });
    // Capture keeps the existing page / derivation key router inactive inside this modal.
    document.addEventListener('keydown', e => {
      if (list.open) { e.stopImmediatePropagation(); return; }
      if (dlg.open || e.defaultPrevented || e.isComposing || e.ctrlKey || e.altKey || e.metaKey) return;
      if (e.key !== 'g' && e.key !== 'G') return;
      const textControl = 'input, select, textarea, [contenteditable], [role="textbox"], [role="combobox"]';
      if ([e.target, document.activeElement].some(el => el instanceof Element && el.closest(textControl))) return;
      e.preventDefault();
      openList();
    }, true);
    document.querySelectorAll('.deriv-open').forEach(b => b.addEventListener('click', e => {
      if (list.open) { e.preventDefault(); e.stopImmediatePropagation(); }
    }, true));
  }

  /* ---------- initial state from URL, then localStorage ---------- */
  const params = new URLSearchParams(location.search);
  const asked = params.get('mode');
  const start = talk ? 'present' : asked !== null ? (MODES.includes(asked) ? asked : 'read') : (MODES.includes(store.get()) ? store.get() : 'read');
  // #last (used when paging back from the next chapter) means the last page
  const target = location.hash === '#last' ? pages.at(-1)
    : hashTarget(location.hash);
  const at = pageIndex(target);
  if (at >= 0) i = at;
  setMode(start, location.hash === '#last' ? -1 : 1);
  if (start !== 'present' && target) target.scrollIntoView({ block: 'start' });
  if (params.get('deriv') === '1' && start !== 'notes' && at >= 0 && pages[at].dataset.sec !== undefined) openPopup(+pages[at].dataset.sec);
})();
