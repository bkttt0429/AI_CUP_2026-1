/* Animation core for the CTM lessons (classic script; works from file://).
   Chapter files call NB.register(name, builder); NB.mountAll() builds report figures first.
   builder(fig) -> {dur, loop?, still?, draw(p), bind?(anim), steps?} and fills '.stage'.
   Optional steps: [{label, at}], with strictly increasing at in (0, 1] and last at = 1.
   A stepped Anim has a zero-based step; seek(k) pauses and draws steps[k].at (invalid k
   returns false). next()/prev() return true when consumed, false at the boundary;
   atStart()/atEnd() report those boundaries. next() animates forward with its target
   step current immediately; prev()/seek() are instant. Continuous playback follows
   the last reached step.
   Its still is always the last step, including read/notes, reduced motion and print. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let reduce = motion.matches;
  const builders = {}, all = [], suspended = new Set();
  const mounted = new WeakMap();
  let lessonObserver, printing = false;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      all.forEach(a => { if (a.run || a.segment) {
        const segment = a.segment;
        a.pause(); a.segment = segment; suspended.add(a);
      } });
    } else {
      const resume = [...suspended];
      suspended.clear();
      resume.forEach(a => a.segment ? a.schedule() : a.play());
    }
  });
  motion.addEventListener('change', e => {
    reduce = e.matches;
    if (reduce) all.forEach(a => { a.pause(); a.draw(a.still); });
  });
  // Printing (notes mode → PDF) mounts deferred lessons and shows every figure's still frame.
  window.addEventListener('beforeprint', () => {
    printing = true;
    mountIn(document);
    all.forEach(a => { a.pause(); a.draw(a.still); });
  });
  window.addEventListener('afterprint', () => { printing = false; });

  function mk(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  class Anim {
    constructor(fig, o) {
      const steps = o.steps ?? [];
      if (o.steps !== undefined && (!Array.isArray(steps) || !steps.length ||
        steps.some((s, k) => !s || typeof s.label !== 'string' || !Number.isFinite(s.at) || s.at <= (k ? steps[k - 1].at : 0) || s.at > 1) || steps.at(-1).at !== 1)) {
        throw new RangeError('NB: steps require labels and increasing at in (0, 1], ending at 1');
      }
      Object.assign(this, { fig, dur: o.dur, loop: !!o.loop, draw: o.draw, still: steps.length ? 1 : o.still ?? 1, steps, step: 0, t: 0, run: false, last: 0, raf: null, generation: 0, segment: null });
      this.btn = fig.querySelector('.play');
      this.btn.onclick = () => (this.run ? this.pause() : this.play());
      if (steps.length) {
        const row = document.createElement('div'); row.className = 'step-row';
        const dots = document.createElement('div');
        dots.className = 'step-dots'; dots.setAttribute('role', 'group'); dots.setAttribute('aria-label', '圖解步驟');
        this.dots = steps.map((s, k) => {
          const b = document.createElement('button');
          b.type = 'button'; b.className = 'step-dot';
          b.setAttribute('aria-label', `${k < 20 ? String.fromCodePoint(0x2460 + k) : k + 1} ${s.label}`);
          b.onclick = () => this.seek(k); dots.appendChild(b);
          return b;
        });
        this.name = document.createElement('span'); this.name.className = 'step-name';
        this.name.setAttribute('aria-label', '目前步驟');
        row.append(this.btn, dots, this.name);
        fig.querySelector('.ctl').prepend(row);
        this.draw = p => {
          o.draw.call(this, p);
          this.t = p * this.dur;
          this.step = this.segment?.step ?? 0;
          if (!this.segment) steps.forEach((s, k) => { if (p >= s.at) this.step = k; });
          this.dots.forEach((b, k) => {
            if (k === this.step) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
            b.toggleAttribute('data-past', k <= this.step);
          });
          this.name.textContent = `${steps[this.step].label}　${this.step + 1}／${steps.length}`;
        };
      }
      this.pause();
      this.draw(reduce || (steps.length && !document.body.classList.contains('m-present')) ? this.still : steps[0]?.at ?? 0);
    }
    seek(k) {
      if (!Number.isInteger(k) || !this.steps[k]) return false;
      this.pause(); this.draw(this.steps[k].at); return true;
    }
    atStart() { return this.steps.length ? this.step === 0 : this.t <= 0; }
    atEnd() { return this.steps.length ? this.step === this.steps.length - 1 : this.t >= this.dur; }
    next() {
      if (this.segment) {
        const end = this.segment.at;
        this.pause(); this.draw(end);
        // finishing an animating step consumes the key, so the last step is seen before the page turns
        if (this.atEnd()) return true;
      }
      if (!this.steps.length || this.atEnd()) return false;
      const step = this.step + 1;
      if (reduce) return this.seek(step);
      const from = this.t / this.dur, at = this.steps[step].at;
      this.pause();
      this.segment = { step, from, at, duration: clamp(this.dur * (at - from), 0.35, 1.6), elapsed: 0 };
      this.draw(from); this.schedule();
      return true;
    }
    prev() { if (!this.steps.length || this.atStart()) return false; return this.seek(this.step - 1); }
    tick(now, generation) {
      if ((!this.run && !this.segment) || generation !== this.generation) return;
      this.raf = null;
      const dt = this.last ? Math.min(now - this.last, 64) / 1000 : 0;
      this.last = now;
      if (this.segment) {
        const s = this.segment;
        s.elapsed = Math.min(s.duration, s.elapsed + dt);
        if (s.elapsed >= s.duration) { this.pause(); this.draw(s.at); return; }
        this.draw(s.from + (s.at - s.from) * s.elapsed / s.duration);
      } else {
        this.t += dt;
        let p = this.t / this.dur, wrapped = null;
        if (p >= 1) {
          if (this.loop) {
            this.t %= this.dur; p = this.t / this.dur;
            // A stepped loop exposes its final checkpoint for this frame before wrapping.
            if (this.steps.length) { wrapped = this.t; p = 1; }
          }
          else { this.t = this.dur; this.draw(1); this.pause(); return; }
        }
        this.draw(p);
        if (wrapped !== null) this.t = wrapped;
      }
      if ((this.run || this.segment) && generation === this.generation) this.raf = requestAnimationFrame(now => this.tick(now, generation));
    }
    schedule() {
      this.last = 0;
      if (document.hidden) { suspended.add(this); return; }
      const generation = ++this.generation;
      this.raf = requestAnimationFrame(now => this.tick(now, generation));
    }
    play() {
      if (this.run) return;
      const segment = this.segment;
      this.pause();
      if (segment) this.draw(this.t / this.dur);
      if (document.hidden) { suspended.add(this); return; }
      if (!this.loop && this.t >= this.dur) {
        this.t = 0;
        if (this.steps.length) this.draw(this.steps[0].at);
      }
      this.run = true; this.btn.textContent = '❚❚';
      this.btn.setAttribute('aria-pressed', 'true'); this.btn.setAttribute('aria-label', '暫停');
      this.schedule();
    }
    pause() {
      this.run = false; this.segment = null; this.generation++;
      if (this.raf !== null) cancelAnimationFrame(this.raf);
      this.raf = null;
      suspended.delete(this);
      this.btn.textContent = '▶';
      this.btn.setAttribute('aria-pressed', 'false'); this.btn.setAttribute('aria-label', '播放');
    }
    restart() { this.pause(); this.t = 0; this.draw(this.steps[0]?.at ?? 0); this.play(); }
    auto() { if (reduce || (this.steps.length && (printing || !document.body.classList.contains('m-present')))) { this.pause(); this.draw(this.still); } else this.restart(); }
  }

  function ensureMounted(fig) {
    if (mounted.has(fig)) return mounted.get(fig);
    const make = builders[fig.dataset.anim];
    if (!make) { console.warn('NB: no animation named', fig.dataset.anim); return; }
    const spec = make(fig), a = new Anim(fig, spec);
    fitLabels(fig);
    mounted.set(fig, a);
    if (spec.bind) spec.bind(a);
    all.push(a);
    const owner = fig.closest('.lesson') || fig.closest('.page');
    (owner.anims = owner.anims || []).push(a);
    new IntersectionObserver(es => es.forEach(e => {
      const b = document.body.classList;
      if (printing || b.contains('m-present')) return;             // pages drive their own animations
      if (fig.closest('.lesson') && !b.contains('m-notes')) return; // lesson figs follow the dialog
      if (e.isIntersecting) a.auto(); else a.pause();
    }), { threshold: 0.5 }).observe(fig);
    return a;
  }

  // Phones scroll a figure sideways rather than shrink its labels below 11 px: --fig-min-w is the
  // rendered width at which the smallest label reaches 11 px (CSS applies it only at ≤ 600 px).
  const swipe = new ResizeObserver(es => es.forEach(e => {
    const stage = e.target;
    stage.closest('.fig').classList.toggle('fig-scrolls', stage.scrollWidth > stage.clientWidth + 1);
  }));
  function fitLabels(fig) {
    const stage = fig.querySelector('.stage'), svg = stage.querySelector('svg'), vb = svg?.viewBox.baseVal;
    if (!vb?.width) return;
    let min = Infinity;
    svg.querySelectorAll('text, tspan').forEach(t => { const f = parseFloat(getComputedStyle(t).fontSize); if (f > 0 && f < min) min = f; });
    if (min === Infinity) return;
    fig.style.setProperty('--fig-min-w', Math.ceil(vb.width * 11 / min) + 'px');
    const hint = document.createElement('p');
    hint.className = 'fig-swipe'; hint.setAttribute('aria-hidden', 'true'); hint.textContent = '← 左右滑動看完整圖 →';
    stage.after(hint);
    swipe.observe(stage);
  }

  function mountIn(root) {
    if (root.matches?.('.fig[data-anim]')) ensureMounted(root);
    root.querySelectorAll('.fig[data-anim]').forEach(ensureMounted);
  }

  function observeLessons(root = document) {
    // Reduced-motion notes expose all still readouts immediately, including off-screen figures.
    if (document.body.classList.contains('m-notes') && reduce) {
      root.querySelectorAll('.lesson').forEach(mountIn);
      return;
    }
    lessonObserver ??= new IntersectionObserver(es => es.forEach(e => {
      if (!printing && e.isIntersecting && document.body.classList.contains('m-notes')) {
        ensureMounted(e.target);
        lessonObserver.unobserve(e.target);
      }
    }), { rootMargin: innerHeight + 'px 0px' });
    root.querySelectorAll('.lesson .fig[data-anim]').forEach(fig => {
      if (!mounted.has(fig)) { lessonObserver.unobserve(fig); lessonObserver.observe(fig); }
    });
  }

  function mountAll(root = document) {
    root.querySelectorAll('.fig[data-anim]').forEach(fig => {
      if (!fig.closest('.lesson')) ensureMounted(fig);
    });
    if (document.body.classList.contains('m-notes')) observeLessons(root);
  }

  // Shared helpers for the chapter animation files.
  const stage = (fig, svg) => { fig.querySelector('.stage').innerHTML = svg; return fig.querySelector('svg'); };
  const slider = (fig, html) => { const l = document.createElement('label'); l.className = 'slider'; l.innerHTML = html; fig.querySelector('.ctl').appendChild(l); return l; };
  function choice(fig, { label, values, value, onChange }) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'fig-choice';
    const next = () => values[(values.indexOf(value) + 1) % values.length];
    function update() {
      button.textContent = label + ' = ' + value;
      button.setAttribute('aria-label', label + ' = ' + value + '，按一下換成 ' + next());
    }
    function set(v) {
      if (!values.includes(v)) throw new RangeError('NB: invalid choice value');
      value = v; update(); onChange(value);
    }
    button.onclick = () => set(next());
    update(); fig.querySelector('.stage').appendChild(button);
    return { get value() { return value; }, set };
  }
  // Scientific notation with Unicode superscripts: sci(16777216) -> '1.68×10⁷', sci(0.0408) -> '4.08×10⁻²'.
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  function sci(x, digits = 3) {
    if (x === 0) return '0';
    const [m, e] = x.toExponential(digits - 1).split('e');
    return m + '×10' + String(+e).split('').map(c => SUP[c]).join('');
  }
  // Fixed decimals with a real minus sign (U+2212); never prints "−0.00".
  const sgn = (x, d) => { const s = Math.abs(x).toFixed(d); return (x < 0 && /[1-9]/.test(s) ? '−' : '') + s; };
  // Unicode subscript digits: sub(12) -> '₁₂'.
  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(c => SUB[c]).join('');
  // SVG path through [x, y] points.
  const poly = pts => pts.map(([x, y], j) => (j ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1)).join('');
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  // Smoothing helpers: animations step through discrete states (ticks, tokens, path steps); these
  // turn the jump between two states into an eased glide so nothing moves in a single frame.
  const ease = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };            // smoothstep
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpArr = (a, b, t) => a.map((v, i) => lerp(v, b[i], t));
  // Position in a sequence of n states over p ∈ [0, 1]. Each state's slot spends its first `move`
  // fraction gliding in from the previous state, then holds: draw lerp(state[prev], state[k], e).
  function beat(p, n, move = 0.5) {
    const f = clamp(p, 0, 1) * n, k = Math.min(n - 1, Math.floor(f));
    return { k, prev: Math.max(0, k - 1), e: k === 0 ? 1 : ease((f - k) / move), f };
  }
  // Opacity for looping animations: fades out over the last `w` of the cycle and back in over the
  // first `w`, so the wrap from the final state to the first is not a jump.
  const loopFade = (p, w = 0.06) => Math.min(ease(p / w), ease((1 - p) / w));
  // A polyline that reaches a fractional index f along pts (grows smoothly instead of point by point).
  function partial(pts, f) {
    const k = Math.floor(clamp(f, 0, pts.length - 1)), t = f - k;
    const out = pts.slice(0, k + 1);
    if (t > 0 && k + 1 < pts.length) out.push(lerpArr(pts[k], pts[k + 1], t));
    return out;
  }

  window.NB = { register(name, builder) { builders[name] = builder; }, mountAll, mountIn, ensureMounted, observeLessons, mk, Anim, stage, slider, choice, sci, sgn, sub, poly, clamp,
    ease, lerp, lerpArr, beat, loopFade, partial };
})();
