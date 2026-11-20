"""Text-only review helper: reads .check_out.json, estimates each text/shape bbox and its visibility window
(real seconds, after kt() warp), and lists text pairs that overlap while both are visible.
  python3 reports/go_story/.review_overlap.py [anim ...]
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

D = Path(__file__).resolve().parent
NS = ""
FS = {"an-k": 13, "an-n": 13, "an-t": 14, "an-l": 12, "an-v": 15, "an-s": 11.5, "an-w": 12, "an-b": 13, "an-h": 13}


def tw(s, fs):
    w = 0.0
    for ch in s:
        o = ord(ch)
        w += fs * (1.0 if (0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F) else 0.58)
    return w


def vis_window(el, parents, dur):
    """Return (fade_in_start, full_on, full_off, fade_out_end) in seconds, or None if static."""
    chain = [el] + parents
    win = None
    for e in chain:
        for a in e.findall(NS + "animate"):
            if a.get("attributeName") == "opacity":
                k = [float(x) * dur for x in a.get("keyTimes").split(";")]
                v = a.get("values").split(";")
                if v == ["0", "0", "1", "1", "0", "0"]:
                    w = (k[1], k[2], k[3], k[4])
                    win = w if win is None else (max(win[0], w[0]), max(win[1], w[1]), min(win[2], w[2]), min(win[3], w[3]))
    return win


def walk(el, parents, out, dur):
    tag = el.tag.replace(NS, "")
    if tag == "text":
        s = "".join(t for t in el.itertext()).strip()
        cls = (el.get("class") or "").split()
        fs = max([FS.get(c, 0) for c in cls] + [0]) or 13
        st = el.get("style") or ""
        m = re.search(r"font-size:\s*([\d.]+)px", st)
        if m:
            fs = float(m.group(1))
        x, y = float(el.get("x")), float(el.get("y"))
        w = tw(s, fs)
        anc = el.get("text-anchor", "start")
        x0 = x if anc == "start" else (x - w / 2 if anc == "middle" else x - w)
        out.append(dict(kind="text", s=s, fs=fs, bb=(x0, y - 0.82 * fs, x0 + w, y + 0.22 * fs), win=vis_window(el, parents, dur)))
    elif tag in ("rect", "circle", "line", "polyline", "path", "polygon"):
        bb = None
        try:
            if tag == "rect":
                x, y, w, h = (float(el.get(a, 0)) for a in ("x", "y", "width", "height"))
                # animated final size
                for a in el.findall(NS + "animate"):
                    if a.get("attributeName") in ("height", "width", "y"):
                        vals = [float(v) for v in a.get("values").split(";")]
                        if a.get("attributeName") == "height":
                            h = max(vals)
                        if a.get("attributeName") == "width":
                            w = max(vals)
                        if a.get("attributeName") == "y":
                            y = min(vals)
                bb = (x, y, x + w, y + h)
            elif tag == "circle":
                cx, cy, r = float(el.get("cx")), float(el.get("cy")), float(el.get("r"))
                bb = (cx - r, cy - r, cx + r, cy + r)
            elif tag == "line":
                xs = [float(el.get("x1")), float(el.get("x2"))]; ys = [float(el.get("y1")), float(el.get("y2"))]
                bb = (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, ValueError):
            bb = None
        if bb:
            out.append(dict(kind=tag, s=el.get("fill") or el.get("stroke") or "", bb=bb, win=vis_window(el, parents, dur)))
    for c in el:
        if c.tag.replace(NS, "") != "animate":
            walk(c, [el] + parents, out, dur)


def overlap_t(a, b, dur):
    wa = a["win"] or (0, 0, dur, dur); wb = b["win"] or (0, 0, dur, dur)
    lo, hi = max(wa[1], wb[1]), min(wa[2], wb[2])   # both fully visible
    return (lo, hi) if hi > lo + 0.05 else None


def inter(a, b, pad=0.0):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0); iy = min(ay1, by1) - max(ay0, by0)
    return (ix, iy) if ix > pad and iy > pad else None


def main():
    d = json.loads((D / ".check_out.json").read_text())
    names = sys.argv[1:] or list(d["anims"])
    for k in names:
        an = d["anims"][k]; dur = an["dur"]
        root = ET.fromstring(an["svg"])
        vb = [float(v) for v in root.get("viewBox").split()]
        els = []
        for c in root:
            walk(c, [], els, dur)
        texts = [e for e in els if e["kind"] == "text"]
        print(f"=== {k}  dur {dur}s  viewBox {vb[2]:.0f}x{vb[3]:.0f}  texts {len(texts)}")
        for t in texts:
            x0, y0, x1, y1 = t["bb"]
            if x0 < vb[0] - 1 or x1 > vb[2] + 1 or y0 < vb[1] - 1 or y1 > vb[3] + 1:
                print(f"  OUT  [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] {t['s'][:40]}")
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ii = inter(a["bb"], b["bb"], 0.5)
                if not ii:
                    continue
                ot = overlap_t(a, b, dur)
                if ot:
                    print(f"  TXT  {a['s'][:28]!r} x {b['s'][:28]!r}  overlap {ii[0]:.0f}x{ii[1]:.0f}px  t {ot[0]:.2f}-{ot[1]:.2f}s"
                          f"  A[{a['bb'][0]:.0f},{a['bb'][1]:.0f},{a['bb'][2]:.0f},{a['bb'][3]:.0f}] B[{b['bb'][0]:.0f},{b['bb'][1]:.0f},{b['bb'][2]:.0f},{b['bb'][3]:.0f}]")
        if "-s" in sys.argv[0:0] or True:
            shapes = [e for e in els if e["kind"] != "text"]
            for t in texts:
                for sh in shapes:
                    ii = inter(t["bb"], sh["bb"], 1.0)
                    if not ii:
                        continue
                    ot = overlap_t(t, sh, dur)
                    if ot:
                        print(f"  SHP  {t['s'][:30]!r} x {sh['kind']}({sh['s']}) [{sh['bb'][0]:.0f},{sh['bb'][1]:.0f},{sh['bb'][2]:.0f},{sh['bb'][3]:.0f}] overlap {ii[0]:.0f}x{ii[1]:.0f} t {ot[0]:.2f}-{ot[1]:.2f}")
        # final frame (reduced motion): what is visible at AD-1.5
        fin = dur - 1.5
        hidden = [t["s"][:24] for t in texts if t["win"] and not (t["win"][1] <= fin <= t["win"][2])]
        print(f"  FINAL@{fin}s hidden texts: {hidden}")


if __name__ == "__main__":
    main()
