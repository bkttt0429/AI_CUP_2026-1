#!/usr/bin/env python3
"""Layout check of the built report: one screenshot and overlap / overflow measurements per slide of talk.html,
plus chapter pages in read mode at desktop and phone width. Writes reports/go_report/shots/."""
import json, os, sys
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

SITE = Path(__file__).resolve().parent / "site"
OUT = Path(__file__).resolve().parent / "shots"
OUT.mkdir(exist_ok=True)
PROBE = """() => {
  const page = document.querySelector('.page.cur'); const out = [];
  const R = e => e.getBoundingClientRect();
  const hit = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 2 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 2;
  const pr = R(page);
  page.querySelectorAll('.fig,.cause,.report>div:first-child,.deriv-open,h3,.opening ol').forEach(e => { const r = R(e);
    if (r.bottom > pr.bottom + 1 || r.right > pr.right + 1 || r.top < pr.top - 1) out.push('outside page: ' + e.tagName + '.' + e.className); });
  const fig = page.querySelector('.report>.fig'), cause = page.querySelector('.cause'), left = page.querySelector('.report>div:first-child'), btn = page.querySelector('.deriv-open');
  if (fig && cause && hit(R(fig), R(cause))) out.push('fig overlaps cause');
  if (fig && left && hit(R(fig), R(left))) out.push('fig overlaps text');
  if (left && cause && hit(R(left), R(cause))) out.push('text overlaps cause');
  if (btn && cause) cause.querySelectorAll('.pair,.bridge-text').forEach(p => { if (hit(R(p), R(btn))) out.push('button overlaps cause'); });
  if (fig) { const st = fig.querySelector('.stage'), sr = R(st);
    [...st.children].forEach(c => { const r = R(c); if (r.bottom > sr.bottom + 2 || r.top < sr.top - 2 || r.right > sr.right + 2) out.push('stage child overflows: ' + c.tagName + ' ' + Math.round(r.height) + ' in ' + Math.round(sr.height)); });
    st.querySelectorAll('.table-wrap').forEach(t => { if (t.scrollWidth > t.clientWidth + 1) out.push('table scrolls sideways'); });
    if (R(st).height < 120) out.push('stage very short: ' + Math.round(R(st).height));
    st.querySelectorAll('svg').forEach(s => { const sv = R(s), ts = [...s.querySelectorAll('text')], rs = ts.map(R);
      rs.forEach((r, i) => { if (r.left < sv.left - 3 || r.right > sv.right + 3 || r.top < sv.top - 3 || r.bottom > sv.bottom + 3) out.push('svg text clipped: ' + ts[i].textContent);
        for (let j = i + 1; j < rs.length; j++) if (hit(r, rs[j])) out.push('svg text collides: ' + ts[i].textContent + ' / ' + ts[j].textContent); }); }); }
  const art = page.querySelector('.cv-art'), head = page.querySelector('.cv-head');
  if (art && head) { if (hit(R(art), R(head))) out.push('cover art overlaps title'); page.querySelectorAll('.cv-head>*').forEach(e => { if (R(e).bottom > pr.bottom + 1) out.push('cover text cut: ' + e.className); }); }
  return { id: page.id, issues: [...new Set(out)] };
}"""
with sync_playwright() as p:
    b = p.chromium.launch()
    res = {}
    ctx = b.new_page(viewport={"width": 1366, "height": 768})
    ctx.goto((SITE / "talk.html").as_uri())
    ctx.wait_for_timeout(600)
    n = ctx.evaluate("document.querySelectorAll('.page').length")
    shots = []
    for i in range(n):
        r = ctx.evaluate(PROBE)
        res[f"{i+1:02d} {r['id']}"] = r["issues"]
        f = OUT / f"talk{i+1:02d}.png"
        ctx.screenshot(path=str(f)); shots.append(f)
        ctx.keyboard.press("ArrowRight"); ctx.wait_for_timeout(120)
    for w, tag in ((1366, "desk"), (400, "phone")):
        v = b.new_page(viewport={"width": w, "height": 800})
        for name in ["index"] + [f"ch{k:02d}" for k in range(1, 9)]:
            v.goto((SITE / f"{name}.html").as_uri() + "?mode=read"); v.wait_for_timeout(250)
            over = v.evaluate("document.documentElement.scrollWidth - innerWidth")
            res[f"{tag} {name}"] = [f"body overflow {over}px"] if over > 1 else []
            if name in ("index", "ch06"):
                v.screenshot(path=str(OUT / f"{tag}_{name}.png"), full_page=(tag == "desk" and name == "index"))
    b.close()
(OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
for j in range(0, len(shots), 6):
    ims = [Image.open(f).convert("RGB").resize((683, 384)) for f in shots[j:j + 6]]
    sheet = Image.new("RGB", (683 * 2 + 6, 384 * 3 + 12), "#888")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % 2) * 689, (k // 2) * 390))
    sheet.save(OUT / f"sheet{j // 6 + 1}.png")
bad = {k: v for k, v in res.items() if v}
print(len(res), "views;", len(bad), "with issues")
print(json.dumps(bad, ensure_ascii=False, indent=1))
