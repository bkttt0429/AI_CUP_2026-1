/* The rank strip of chapter 1: pick the true level, read what each prediction scores. */
(function () {
  const root = document.getElementById('rankStrip');
  if (!root) return;
  const btns = [...root.querySelectorAll('button')];
  function paint(truth) {
    btns.forEach((b, i) => {
      const d = Math.abs(i - truth), s = d === 0 ? 1 : d === 1 ? Math.exp(-1) : 0;
      b.style.setProperty('--score', s);
      b.classList.toggle('hi', s === 1);
      b.setAttribute('aria-pressed', String(i === truth));
      b.querySelector('.sc').textContent = s === 1 ? '1' : s ? '.368' : '0';
    });
  }
  btns.forEach((b, i) => b.addEventListener('click', () => paint(i)));
  paint(6);
})();
