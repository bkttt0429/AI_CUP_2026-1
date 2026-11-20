#!/usr/bin/env python3
"""Browser check of the closing deck (run on a compute node through sbatch; needs Playwright in ~/.cache/go_deck_tools):
  sbatch -J deckchk -t 00:20:00 -c 4 scripts/nano4/run.sbatch bash reports/go_deck/check.sh            # whole deck
  sbatch -J deckchk -t 00:20:00 -c 4 scripts/nano4/run.sbatch bash reports/go_deck/check.sh --only 02  # one chapter
For every slide, at 1366x768 and 1757x855 in present mode, with each animation paused at 35 %, 60 %, 85 % and its final frame:
  - the slide fits the screen (no page scroll), the text was not shrunk below 13 px by fitFigs
  - visible SVG texts do not collide and stay inside their SVG; HTML boxes do not scroll sideways
  - every <animate> has as many values as keyTimes and keyTimes rise 0 -> 1; no '載入失敗'; every data-anim rendered
At 400x800: no sideways body scroll. Whole deck only: ids unique, every NUM.kpi value appears in the slide text.
Writes reports/go_deck/shots/<tag>/: one PNG per slide (final animation frame), sheet*.png contact sheets, report.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import assemble  # noqa: E402

from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

SET_T = """(frac) => { const page = document.querySelector('main>.page.cur'); if (!page) return;
  page.querySelectorAll('.an-stage>svg').forEach(s => { const sk = s.parentElement.querySelector('.an-seek');
    const dur = sk ? +sk.max + 0.05 : 9; s.pauseAnimations(); s.setCurrentTime(frac < 0 ? Math.max(0, dur - 1.5) : frac * dur); }); }"""

PROBE = """() => {
  const page = document.querySelector('main>.page.cur'); const out = [];
  if (!page) return {id: null, issues: ['no current page']};
  const R = e => e.getBoundingClientRect();
  const hit = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1.5 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1.5;
  const vis = e => { for (let x = e; x && x !== page; x = x.parentElement) { const cs = getComputedStyle(x);
      if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity < 0.05) return false; } return true; };
  const over = document.documentElement.scrollHeight - innerHeight;
  if (over > 2) out.push('slide taller than screen by ' + Math.round(over) + 'px');
  const box = page.querySelector('.report') || page, fs = parseFloat(getComputedStyle(box).fontSize);
  if (fs < 13) out.push('text shrunk to ' + fs.toFixed(1) + 'px (too much on this slide)');
  page.querySelectorAll('.table-wrap,.bars,.formula,.why,.txt,.fig').forEach(e => { if (e.scrollWidth > e.clientWidth + 2) out.push('scrolls sideways: ' + e.className); });
  page.querySelectorAll('.an-stage>svg').forEach(s => { const sv = R(s), name = s.closest('.anim')?.dataset.anim;
    const ts = [...s.querySelectorAll('text')].filter(t => t.textContent.trim() && vis(t)), rs = ts.map(R);
    rs.forEach((r, i) => { if (r.width === 0) return;
      if (r.left < sv.left - 2 || r.right > sv.right + 2 || r.top < sv.top - 2 || r.bottom > sv.bottom + 2) out.push(name + ': text outside svg: ' + ts[i].textContent.slice(0, 40));
      for (let j = i + 1; j < rs.length; j++) if (rs[j].width && hit(r, rs[j])) out.push(name + ': texts collide: ' + ts[i].textContent.slice(0, 30) + ' / ' + ts[j].textContent.slice(0, 30)); }); });
  return {id: page.id, fs: +fs.toFixed(1), ch: (document.getElementById('cur-ch') || {}).textContent || '', issues: [...new Set(out)]};
}"""


METRICS = """() => {
  const page = document.querySelector('main>.page.cur'); if (!page) return null;
  const R = e => e.getBoundingClientRect();
  const vis = e => { for (let x = e; x && x !== page; x = x.parentElement) { const cs = getComputedStyle(x);
      if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity < 0.05) return false; } return true; };
  const count = root => { let n = 0; if (!root) return 0;
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {acceptNode: t => t.parentElement.closest('svg,style,script,.an-ctl') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT});
    for (let t; (t = tw.nextNode());) n += t.nodeValue.replace(/\\s+/g, '').length; return n; };
  const left = page.querySelector('.report>.txt') || page.querySelector('.af-top') || page;
  const px = [];
  page.querySelectorAll('.an-stage>svg').forEach(s => { const vb = s.viewBox && s.viewBox.baseVal, sc = vb && vb.width ? R(s).width / vb.width : 1;
    s.querySelectorAll('text').forEach(t => { if (t.textContent.trim() && vis(t)) px.push(parseFloat(getComputedStyle(t).fontSize) * sc); }); });
  px.sort((a, b) => a - b);
  const content = page.querySelector('.report') || page.firstElementChild;
  let bottom = 0; page.querySelectorAll('.report>*,.af-top,.af-disc,.kpis,.metaline,h2').forEach(e => { bottom = Math.max(bottom, R(e).bottom); });
  return {leftChars: count(left), allChars: count(page), svgTexts: px.length,
          svgMin: px.length ? +px[0].toFixed(1) : null, svgP10: px.length ? +px[Math.floor(px.length * 0.1)].toFixed(1) : null,
          bottomGap: Math.round(R(page).bottom - bottom)};
}"""

STATIC = """() => { const out = [];
  document.querySelectorAll('animate').forEach(a => { const v = (a.getAttribute('values') || '').split(';'), k = (a.getAttribute('keyTimes') || '').split(';').map(Number);
    const name = a.closest('.anim')?.dataset.anim;
    if (v.length !== k.length) out.push(name + ': animate values ' + v.length + ' != keyTimes ' + k.length);
    else if (Math.abs(k[0]) > 1e-9 || Math.abs(k[k.length - 1] - 1) > 1e-9 || k.some((x, i) => i && x < k[i - 1] - 1e-9)) out.push(name + ': keyTimes not rising 0..1'); });
  document.querySelectorAll('.anim,.bars,.fig').forEach(e => { if (/載入失敗/.test(e.textContent)) out.push('load failure: ' + e.textContent.slice(0, 120)); });
  const ids = [...document.querySelectorAll('[id]')].map(e => e.id), dup = [...new Set(ids.filter((x, i) => ids.indexOf(x) !== i))];
  if (dup.length) out.push('duplicate ids: ' + dup.join(', '));
  const rendered = [...document.querySelectorAll('.anim[data-anim]')].map(e => e.dataset.anim);
  const txt = document.querySelector('main').innerText.replace(/\\s+/g, '');
  const kpi = (window.NUM && window.NUM.kpi) || {};
  const NT = window.NOTES || {}, np = document.querySelectorAll('main>.page').length;
  const notesMissing = [...Array(np).keys()].map(i => i + 1).filter(k => !NT[k]);
  if (notesMissing.length) out.push('pages without speaker notes: ' + notesMissing.join(', '));
  const extra = Object.keys(NT).map(Number).filter(k => k > np);
  if (extra.length) out.push('speaker notes for pages that do not exist: ' + extra.join(', '));
  return {issues: out, rendered, kpiMissing: Object.entries(kpi).filter(([k, v]) => !txt.includes(String(v).replace(/\\s+/g, ''))).map(([k, v]) => k + '=' + v),
          pages: document.querySelectorAll('main>.page').length};
}"""


DUMP = """() => { const pg = document.querySelector('main>.page.cur'); if (!pg) return null;
  const all = [...document.querySelectorAll('main>.page')], nt = (window.NOTES || {})[all.indexOf(pg) + 1];
  const svg = [...new Set([...pg.querySelectorAll('svg text')].map(t => t.textContent.trim()).filter(Boolean))];
  const steps = [...pg.querySelectorAll('.an-steps, .an-note, .cap')].map(e => e.textContent.trim()).filter(Boolean);
  const notes = nt ? nt.h.replace(/<\\/(p|li)>/g, '\\n').replace(/<[^>]+>/g, '').trim() : '';
  return {id: pg.id, title: pg.dataset.title || '', text: pg.innerText.trim(), svg, steps, notes}; }"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    a = ap.parse_args()
    cards, scripts = assemble.pick(a.only)
    html = assemble.build(cards, scripts)
    tag = ("only_" + a.only.replace(",", "_")) if a.only else "full"
    src = assemble.JOBS / f"check_{tag}.html"
    assemble.JOBS.mkdir(parents=True, exist_ok=True)
    # publish wraps the page in <!doctype html><head>charset+viewport</head><body>; do the same locally so body exists for the scripts
    src.write_text('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                   '<style>:root{color-scheme:light}body{margin:0}</style></head><body>\n' + html + '\n</body></html>')
    if not a.only:
        (D / "index.html").write_text(html)
    want = sorted(set(re.findall(r'data-anim="([^"]+)"', "".join(c.read_text() for c in cards))))
    out = D / "shots" / tag
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    res = {"tag": tag, "cards": [c.name for c in cards], "scripts": [s.name for s in scripts], "console": [], "slides": {}}
    with sync_playwright() as p:
        b = p.chromium.launch()
        shots, texts = [], []
        for (w, h) in ((1366, 768), (1757, 855)):
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.on("console", lambda m: res["console"].append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: res["console"].append(f"pageerror: {e}"))
            pg.goto(src.as_uri())
            pg.wait_for_timeout(500)
            try:
                pg.evaluate("document.fonts.ready.then(() => 1)")
            except Exception:
                pass
            if w == 1366:
                st = pg.evaluate(STATIC)
                res["static"] = [x for x in st["issues"] if not (a.only and "speaker notes" in x)]   # 單章模式頁碼對不上全份講稿
                res["missing_anims"] = [x for x in want if x not in st["rendered"]]
                res["kpi_missing"] = st["kpiMissing"] if not a.only else []
                n = st["pages"]
            pg.keyboard.press("Home")
            pg.wait_for_timeout(150)
            for i in range(n):
                issues, rid, fs = [], None, None
                for frac in (0.35, 0.6, 0.85, -1):
                    pg.evaluate(SET_T, frac)
                    pg.wait_for_timeout(40)
                    r = pg.evaluate(PROBE)
                    rid, fs = r["id"], r.get("fs")
                    chl = r.get("ch", "")
                    issues += [x for x in r["issues"] if x not in issues]
                    if w == 1366 and frac == -1:
                        texts.append(pg.evaluate(DUMP))
                key = f"{i + 1:02d} {rid}"
                res["slides"].setdefault(key, {})[f"{w}x{h}"] = {"fs": fs, "ch": chl, "issues": issues, "m": pg.evaluate(METRICS)}
                if w == 1366 and i == min(4, n - 1):   # 講稿抽屜：按 N 打開、內容是這一頁、再按 N 收起
                    pg.keyboard.press("n"); pg.wait_for_timeout(80)
                    nd = pg.evaluate("(() => { const d = document.getElementById('notes'); return d ? {open: !d.hidden, t: document.getElementById('notes-t').textContent, len: document.getElementById('notes-b').textContent.length} : null; })()")
                    pg.keyboard.press("n"); pg.wait_for_timeout(80)
                    closed = pg.evaluate("(() => { const d = document.getElementById('notes'); return d ? d.hidden : null; })()")
                    res["notes_drawer"] = {"page": i + 1, **(nd or {}), "closed_again": closed}
                f = out / f"{w}_{i + 1:02d}_{rid}.png"
                pg.screenshot(path=str(f))
                if w == 1366:
                    shots.append(f)
                pg.keyboard.press("ArrowRight")
                pg.wait_for_timeout(150)
            pg.close()
        ph = b.new_page(viewport={"width": 400, "height": 800})
        ph.goto(src.as_uri())
        ph.wait_for_timeout(400)
        ph.keyboard.press("Home")
        for i in range(n):
            o = ph.evaluate("document.documentElement.scrollWidth - innerWidth")
            rid = ph.evaluate("document.querySelector('main>.page.cur')?.id")
            res["slides"].setdefault(f"{i + 1:02d} {rid}", {})["400x800"] = {"issues": [f"body scrolls sideways {o}px"] if o > 1 else []}
            if i < 2 or o > 1:
                ph.screenshot(path=str(out / f"phone_{i + 1:02d}_{rid}.png"))
            ph.keyboard.press("ArrowRight")
            ph.wait_for_timeout(100)
        b.close()
    for j in range(0, len(shots), 6):
        ims = [Image.open(f).convert("RGB").resize((683, 384)) for f in shots[j:j + 6]]
        sheet = Image.new("RGB", (683 * 2, 384 * 3), "white")
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % 2) * 683, (k // 2) * 384))
        sheet.save(out / f"sheet{j // 6 + 1}.png")
    md = []   # 每頁文字（左欄＋動畫字＋講稿），給審稿用
    for i, t in enumerate(texts):
        if not t:
            continue
        md.append(f"## 第 {i + 1} 頁 · {t['title']}（#{t['id']}）\n\n### 投影片文字\n{t['text']}\n\n### 動畫與圖表文字\n" + " ／ ".join(t["svg"])
                  + ("\n\n" + "\n".join(t["steps"]) if t["steps"] else "") + f"\n\n### 講稿\n{t['notes']}\n")
    (out / "pages.md").write_text("\n".join(md))
    (out / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    bad = sum(len(v["issues"]) for s in res["slides"].values() for v in s.values()) + len(res["static"]) + len(res["missing_anims"]) + len(res["kpi_missing"])
    print(json.dumps({k: v for k, v in res.items() if k != "slides"}, ensure_ascii=False, indent=1))
    for k, s in res["slides"].items():
        for vp, v in s.items():
            if v["issues"]:
                print(k, vp, "fs", v.get("fs"), "::", " | ".join(v["issues"]))
    print("notes drawer:", res.get("notes_drawer"))
    print("chapter labels:", {k: v.get("1366x768", {}).get("ch") for k, v in res["slides"].items()})
    print("metrics (slide | viewport | body px | left chars | all chars | svg texts | svg min px | svg p10 px | bottom gap px):")
    for k, s_ in res["slides"].items():
        for vp, v in s_.items():
            m = v.get("m")
            if m:
                print(f"  {k:24s} {vp:9s} {v.get('fs')!s:>5} {m['leftChars']:>5} {m['allChars']:>5} {m['svgTexts']:>4} {m['svgMin']!s:>6} {m['svgP10']!s:>6} {m['bottomGap']:>5}")
    print(f"{tag}: {len(res['slides'])} slides, {bad} issues; shots in {out}")


if __name__ == "__main__":
    main()
