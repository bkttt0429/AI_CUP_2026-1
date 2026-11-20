"""Text-only helper: for every <line> in each animation SVG, list text boxes it actually crosses while both are visible."""
import json, sys, importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ov", D / ".review_overlap.py"); ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
d = json.loads((D / ".check_out.json").read_text())

def seg_hits_box(x1, y1, x2, y2, b, n=200):
    for k in range(n + 1):
        t = k / n; x = x1 + (x2 - x1) * t; y = y1 + (y2 - y1) * t
        if b[0] + 0.5 < x < b[2] - 0.5 and b[1] + 0.15 * (b[3] - b[1]) < y < b[3] - 0.15 * (b[3] - b[1]):
            return (round(x), round(y))
    return None

for k in (sys.argv[1:] or d["anims"]):
    an = d["anims"][k]; dur = an["dur"]; root = ET.fromstring(an["svg"])
    els = []
    for c in root: ov.walk(c, [], els, dur)
    texts = [e for e in els if e["kind"] == "text"]
    lines = []
    def lw(el, parents):
        if el.tag == "line":
            lines.append((el, ov.vis_window(el, parents, dur)))
        for c in el:
            if c.tag != "animate": lw(c, [el] + parents)
    for c in root: lw(c, [])
    for el, win in lines:
        x1, y1, x2, y2 = (float(el.get(a)) for a in ("x1", "y1", "x2", "y2"))
        for t in texts:
            hit = seg_hits_box(x1, y1, x2, y2, t["bb"])
            if hit and ov.overlap_t(t, {"win": win}, dur):
                print(f"{k}: line ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) crosses {t['s'][:30]!r} at {hit}  t {ov.overlap_t(t, {'win': win}, dur)}")
