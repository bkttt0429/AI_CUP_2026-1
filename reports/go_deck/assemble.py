"""Assemble the closing deck from parts/ (head, chapter cards c*.html, footer, NUM, engine, chapter scripts a*.js, engine tail).
  python3 reports/go_deck/assemble.py                 -> reports/go_deck/index.html (+ 講稿.md from parts/講稿_*.md)
  python3 reports/go_deck/assemble.py --only 02       -> ~/.cache/go_deck_jobs/only_02.html: cover-less page with c02*.html + a02*.js only
Card files c0N_*.html and script files a0N_*.js are joined in file-name order; a9*.js (terms) always come last.
"""
import argparse
import base64
import html as H
import json
import os
import re
from pathlib import Path

D = Path(__file__).resolve().parent
P = D / "parts"
JOBS = Path(os.path.expanduser("~/.cache/go_deck_jobs"))


def md_inline(x):
    x = H.escape(x)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", x)


def notes_json():
    """講稿_*.md → {頁碼: {t: 標題, h: html}}：以「## 第 N 頁 · 標題」分段，- 開頭的行變成條列。"""
    out = {}
    for f in sorted(P.glob("講稿_*.md")):
        cur = None
        for line in f.read_text().splitlines():
            m = re.match(r"^##\s*第\s*(\d+)\s*頁\s*[·・]?\s*(.*)$", line)
            if m:
                cur = int(m.group(1)); out[cur] = {"t": m.group(2).strip(), "lines": []}; continue
            if cur is not None and line.strip():
                out[cur]["lines"].append(line)
    res = {}
    for k, v in out.items():
        h, inlist = [], False
        for ln in v["lines"]:
            st = ln.strip()
            if st.startswith("#"):
                continue
            if re.match(r"^[-*]\s+", st):
                if not inlist:
                    h.append("<ul>"); inlist = True
                h.append("<li>" + md_inline(re.sub(r"^[-*]\s+", "", st)) + "</li>")
            else:
                if inlist:
                    h.append("</ul>"); inlist = False
                h.append("<p>" + md_inline(st) + "</p>")
        if inlist:
            h.append("</ul>")
        res[k] = {"t": v["t"], "h": "".join(h)}
    return res


MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "svg": "image/svg+xml"}


def inline_assets(html):
    """src="asset:NAME" → data URI of parts/assets/NAME (the deck is published as one self-contained page)."""
    def rep(m):
        f = P / "assets" / m.group(1)
        return 'src="data:%s;base64,%s"' % (MIME[f.suffix[1:].lower()], base64.b64encode(f.read_bytes()).decode())
    return re.sub(r'src="asset:([^"]+)"', rep, html)


def build(cards, scripts):
    out = [(P / "00_head.html").read_text()]
    out += [c.read_text() for c in cards]
    out.append((P / "99_foot.html").read_text())
    out.append("<script>\n" + (P / "10_num.js").read_text() + "\n</script>\n")
    out.append("<script>\nwindow.NOTES = " + json.dumps(notes_json(), ensure_ascii=False).replace("</", "<\\/") + ";\n</script>\n")
    out.append((P / "01_engine_head.js").read_text())
    out += ["\n  // ===== " + s.name + " =====\n" + s.read_text() for s in scripts]
    out.append((P / "98_engine_tail.js").read_text())
    return inline_assets("\n".join(out))


def pick(only=None):
    cards = sorted(P.glob("c*.html"))
    scripts = sorted(P.glob("a*.js"))
    if only:
        keys = [k.strip().zfill(2) for k in only.split(",") if k.strip()]
        cards = [c for c in cards if any(c.name.startswith("c" + k) for k in keys)]
        scripts = [s for s in scripts if any(s.name.startswith("a" + k) for k in keys) or s.name.startswith("a9")]
    return cards, scripts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma list of chapter numbers, e.g. 02 or 01,02")
    a = ap.parse_args()
    cards, scripts = pick(a.only)
    html = build(cards, scripts)
    if a.only:
        JOBS.mkdir(parents=True, exist_ok=True)
        out = JOBS / ("only_" + a.only.replace(",", "_") + ".html")
    else:
        out = D / "index.html"
        notes = sorted(P.glob("講稿_*.md"))
        if notes:
            (D / "講稿.md").write_text("# 講稿（收尾簡報）\n\n" + "\n\n".join(n.read_text().strip() for n in notes) + "\n")
    out.write_text(html)
    print(f"{out}: {len(html):,} chars; cards {[c.name for c in cards]}; scripts {[s.name for s in scripts]}")


if __name__ == "__main__":
    main()
