#!/usr/bin/env python3
"""Builds the Go rank-prediction report (the chapter / report-card layout of the user's research notes).
Every chart number is read from results/*.json at build time. Output: reports/go_report/site/.
Run from anywhere: python3 -I reports/go_report/build.py
"""
import html
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "site"
BRAND = "AICUP 棋力預測"
DATE = "2026 年 11 月 2 日"
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def js(p):
    return json.loads((ROOT / p).read_text())


OPP = js("results/l1b/opp_check.json")
FOLD0 = js("results/l1b/opp_fold0_check.json")
XFIT = js("results/l1b/xfit_center_check.json")
ORD = js("results/l1b/ordinal_check.json")
SE = js("results/l1b/se_check.json")
QUAL = js("results/quality/quality_check.json")
LAD = js("results/ladder/cv_summary.json")
VIZ = js("results/viz_examples.json")
HD = js("results/seq/hd_check.json")
ERR = js("results/l1b/error_analysis.json")
FP2 = js("results/seq/fingerprint_check_v2.json")
FS = js("results/seq/feature_stability.json")
HIST = js("results/seq/hist_check.json")
SHAPE = js("results/seq/shape_check.json")
GB = js("results/gbdt/gbdt_check.json")
DIFF0 = js("results/models/rank_model_fused_opp_diff.json")["fold0_check"]
F0D_KEY = next(k for k in DIFF0["scores"] if "diff" in k and "avg" in k)


def score(d, k):
    return d["scores"][k]["balanced_mean_over_n"]


def comparison(d, name):
    c = d["comparisons"]
    if name in c:
        return c[name]
    for group in c.values():
        if isinstance(group, dict) and name in group:
            return group[name]
    raise KeyError(name)


# Guards: the prose below quotes these numbers, so a changed result file must stop the build.
assert round(score(OPP, "fused_opp"), 4) == 0.6735 and round(score(FOLD0, "fused_opp_avg"), 4) == 0.6716
assert round(score(XFIT, "fused_opp_x5c"), 4) == 0.6704 and round(score(XFIT, "fused_opp_c"), 4) == 0.6683
assert round(score(OPP, "fused"), 4) == 0.6545
assert round(score(ERR, "mlp_diff"), 4) == 0.6916 and round(score(HD, "fused_opp_diff_hd"), 4) == 0.6987 and round(score(DIFF0, F0D_KEY), 4) == 0.6888
assert round(score(FOLD0, "hsl") - 0.440, 2) == 0.17 and round(score(OPP, "fused") - score(OPP, "hsl"), 3) == 0.035 and round(score(ERR, "mlp_diff") - score(OPP, "fused"), 3) == 0.037


def esc(s):
    return html.escape(str(s), quote=False)


def num(x, d=4):
    return f"{x:.{d}f}".replace("-", "−")


def signed(x, d=4):
    return ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"


def ci(c):
    return f"[{signed(c['ci95'][0])}, {signed(c['ci95'][1])}]"


# ---------------------------------------------------------------- charts (static SVG, native hover titles)
INK, MUTED, LINE, ACCENT, WARN, S1 = "#1d1f24", "#5d6370", "#d9dbe0", "#0b6e69", "#a4451c", "#0072b2"


def t(x, y, s, size=12, fill=INK, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}>{esc(s)}</text>'


def line(x1, y1, x2, y2, c=LINE, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{d} fill="none"/>'


def dot(x, y, r, fill, title=None, stroke=None):
    s = f' stroke="{stroke}" stroke-width="2"' if stroke else ' stroke="#ffffff" stroke-width="1.5"'
    body = f"<title>{esc(title)}</title>" if title else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{s}>{body}</circle>'


def poly(pts, c, w=2):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>'


def svg(w, h, body, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">{"".join(body)}</svg>'


def legend(items, x, y):
    out, cx = [], x
    for name, color, kind in items:
        if kind == "open":
            out.append(f'<circle cx="{cx + 6}" cy="{y - 4}" r="5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        elif kind == "line":
            out.append(line(cx, y - 4, cx + 14, y - 4, color, 2))
        else:
            out.append(f'<circle cx="{cx + 6}" cy="{y - 4}" r="5" fill="{color}"/>')
        out.append(t(cx + 18, y, name, 12, MUTED))
        cx += 26 + 13 * sum(2 if ord(ch) > 255 else 1 for ch in name) / 2 + 8
    return "".join(out)


def chart_progress():
    rows = [("官方 baseline", None, 0.440), ("Human SL + MLP", score(OPP, "hsl"), score(FOLD0, "hsl")),
            ("加鑑別式 L1b（fused）", score(OPP, "fused"), score(FOLD0, "fused_avg")),
            ("加對手落子（fused_opp）", score(OPP, "fused_opp"), score(FOLD0, "fused_opp_avg")), ("加差值（fused_opp_diff）", score(ERR, "mlp_diff"), score(DIFF0, F0D_KEY)),
            ("加讓子 HD 5 維（候選）", score(HD, "fused_opp_diff_hd"), None)]
    W, H, L, R, top = 640, 302, 178, 60, 40
    xs = lambda v: L + (v - 0.40) / (0.70 - 0.40) * (W - L - R)
    b = [legend([("folds 1–4 CV", ACCENT, "dot"), ("fold 0 留出", WARN, "open")], L, 18)]
    for v in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        b += [line(xs(v), top - 6, xs(v), H - 26), t(xs(v), H - 8, f"{v:.2f}", 11, MUTED, "middle")]
    for i, (name, cv, f0) in enumerate(rows):
        y = top + 12 + i * 40
        last = i == len(rows) - 1
        b.append(t(L - 10, y + 4, name, 12.5, INK, "end", 700 if last else None))
        if cv is not None and f0 is not None:
            b.append(line(xs(cv), y, xs(f0), y, MUTED, 1))
        if f0 is not None:
            b.append(dot(xs(f0), y, 6, "#ffffff", f"{name} fold 0：{f0:.4f}", WARN))
        if cv is not None:
            b.append(dot(xs(cv), y, 6, ACCENT, f"{name} CV：{cv:.4f}"))
        lab = max(v for v in (cv, f0) if v is not None)
        txt = f"{cv:.4f}" if cv is not None else f"{f0:.3f}"
        b.append(t(xs(lab) + 11, y + 4, txt, 12, INK, "start", 700 if last else None))
    return svg(W, H, b, "六個里程碑的離線分數：官方 baseline 0.440，Human SL 0.6192，fused 0.6545，fused_opp 0.6735，fused_opp_diff 0.6916，加 HD 0.6987")


def board_svg(prof, size=236):
    p = VIZ["position"]
    pol = p["policy"][prof]
    m, step = 12, (size - 24) / 18
    pos = lambda i: m + i * step
    b = [f'<rect width="{size}" height="{size}" rx="4" fill="#e4bd6f"/>']
    for i in range(19):
        b += [line(pos(0), pos(i), pos(18), pos(i), "#7a5b24", .6), line(pos(i), pos(0), pos(i), pos(18), "#7a5b24", .6)]
    for sx in (3, 9, 15):
        for sy in (3, 9, 15):
            b.append(f'<circle cx="{pos(sx):.1f}" cy="{pos(sy):.1f}" r="1.6" fill="#7a5b24"/>')
    for x, y, c in p["stones"]:
        f, st = ("#16191b", "#16191b") if c == "B" else ("#f7f7f3", "#8f8e86")
        b.append(f'<circle cx="{pos(x):.1f}" cy="{pos(y):.1f}" r="{step * .46:.1f}" fill="{f}" stroke="{st}" stroke-width=".7"/>')
    for i, q in enumerate(pol[:361]):
        if q < 0.01:
            continue
        x, y = i % 19, i // 19
        b.append(f'<circle cx="{pos(x):.1f}" cy="{pos(y):.1f}" r="{max(1.5, step * 1.15 * math.sqrt(q)):.1f}" fill="{S1}" fill-opacity=".72"><title>{"ABCDEFGHJKLMNOPQRST"[x]}{19 - y}：{q:.1%}</title></circle>')
    px, py = p["played_xy"]
    b.append(f'<circle cx="{pos(px):.1f}" cy="{pos(py):.1f}" r="{step * .62:.1f}" fill="none" stroke="#b84a00" stroke-width="2"/>')
    return svg(size, size, b, f"{prof} profile 的落子機率，橘圈是玩家實際下的 {p['played']}")


def chart_boards():
    p = VIZ["position"]
    cells = []
    for prof in ("12k", "6d"):
        q = math.exp(p["played_logp"][prof])
        cells.append(f'<div class="board-cell">{board_svg(prof)}<p>rank_{prof}：實際落子 {esc(p["played"])} 的機率 {q:.1%}</p></div>')
    return '<div class="boards">' + "".join(cells) + "</div>"


def cum_panel(a, b_, W=560, H=150):
    idx = VIZ["profiles"]
    series = []
    for g in VIZ["games"]:
        run, pts = 0.0, [(0, 0.0)]
        for k, row in enumerate(g["logp"], 1):
            run += row[idx.index(a)] - row[idx.index(b_)]
            pts.append((k, run))
        series.append((g["rank"], pts))
    allv = [v for _, pts in series for _, v in pts]
    lo, hi = min(allv + [0]), max(allv + [0])
    pad = (hi - lo) * .08
    lo, hi = lo - pad, hi + pad
    L, R, top, bot = 44, 72, 22, 22
    nx = max(len(p) for _, p in series)
    xs = lambda k: L + k / nx * (W - L - R)
    ys = lambda v: top + (hi - v) / (hi - lo) * (H - top - bot)
    out = [t(L, 13, f"log p({a}) − log p({b_})，累積（nats）", 12, MUTED)]
    step = 10 ** math.floor(math.log10(max(hi - lo, 1e-6)))
    if (hi - lo) / step < 3:
        step /= 2
    v = math.ceil(lo / step) * step
    while v <= hi:
        out += [line(L, ys(v), W - R, ys(v), MUTED if abs(v) < 1e-9 else LINE, 1), t(L - 6, ys(v) + 4, f"{v:g}".replace("-", "−"), 11, MUTED, "end")]
        v += step
    for name, pts in series:
        c = S1 if name == "D" else WARN
        pp = [(xs(k), ys(val)) for k, val in pts]
        out.append(poly(pp, c))
        out.append(t(pp[-1][0] + 6, pp[-1][1] + 4, f"{name} 玩家 {signed(pts[-1][1], 1)}", 12, INK))
    out.append(t(W - R, H - 6, "目標玩家的第幾手", 11, MUTED, "end"))
    return svg(W, H, out, f"兩位玩家沿落子累積的 {a} 對 {b_} 對數似然比")


def chart_compression():
    prof = VIZ["profiles"]
    i = prof.index
    nominal = ["11k", "8k", "5k", "2k", "1d", "2d", "3d", "4d", "5d", "6d"]
    peak = [i("6k"), i("5k"), i("4k"), i("3k"), i("2k"), i("2k"), (i("2k") + i("1k")) / 2, i("1k"), i("1d"), i("2d")]
    W, H, L, R, top, bot = 560, 300, 44, 56, 34, 28
    xs = lambda k: L + (k + .5) / 10 * (W - L - R)
    ys = lambda v: top + (17 - v) / 17 * (H - top - bot)
    b = [legend([("標稱位置", MUTED, "open"), ("實測似然峰值", ACCENT, "dot")], L, 16)]
    for name in ("12k", "9k", "6k", "3k", "1d", "3d", "6d"):
        b += [line(L, ys(i(name)), W - R, ys(i(name))), t(L - 6, ys(i(name)) + 4, name, 11, MUTED, "end")]
    for k, lv in enumerate(LEVELS):
        b.append(t(xs(k), H - 8, lv, 12, INK, "middle"))
    b.append(poly([(xs(k), ys(i(n))) for k, n in enumerate(nominal)], LINE, 1.5))
    b.append(poly([(xs(k), ys(v)) for k, v in enumerate(peak)], ACCENT, 2))
    for k, n in enumerate(nominal):
        b.append(dot(xs(k), ys(i(n)), 4.5, "#ffffff", f"{LEVELS[k]} 標稱 {n}", MUTED))
    labels = ["6k", "5k", "4k", "3k", "2k", "2k", "2k/1k", "1k", "1d", "2d"]
    for k, v in enumerate(peak):
        b.append(dot(xs(k), ys(v), 5, ACCENT, f"{LEVELS[k]} 實測峰值 {labels[k]}"))
    b.append(t(xs(0), ys(peak[0]) - 10, "6k", 12, INK, "middle"))
    b.append(t(xs(9) + 12, ys(peak[9]) + 4, "2d", 12, INK))
    return svg(W, H, b, "10 個等級的似然峰值只落在 KGS 6k 到 2d 之間，遠比標稱範圍窄")


def chart_ladder():
    r = LAD["rungs"]
    pick = [("r0_sum", "似然相加"), ("r1_clip", "每局截斷"), ("r2_staged", "分段統計 + MLP"), ("r2_whole", "整局向量 + MLP"), ("r3_null", "集合模型空對照"),
            ("r2_mlp24", "24 維 + MLP（基準）"), ("r2_pooled_delta", "合併差分 + MLP"), ("r3_settf", "Set Transformer"), ("r3_deepsets", "DeepSets"),
            ("r4_xattn", "cross-attention 8 ticks"), ("r3_pma", "PMA"), ("r4_xattn_t1", "cross-attention 1 tick"), ("r5_ctm", "CTM 8 ticks")]
    vals = sorted(((r[k]["balanced_mean_over_n"], n, k) for k, n in pick))
    vals.append((r["ens_top3"]["balanced_mean_over_n"], "ens_top3（三者平均）", "ens"))
    W, L, R, top = 560, 188, 56, 12
    H = top + len(vals) * 21 + 26
    base = r["r2_mlp24"]["balanced_mean_over_n"]
    xs = lambda v: L + (v - 0.595) / (0.640 - 0.595) * (W - L - R)
    b = []
    for v in (0.60, 0.61, 0.62, 0.63, 0.64):
        b += [line(xs(v), top, xs(v), H - 22), t(xs(v), H - 6, f"{v:.2f}", 11, MUTED, "middle")]
    b.append(line(xs(base), top, xs(base), H - 22, MUTED, 1, "4 3"))
    for j, (v, n, k) in enumerate(vals):
        y = top + 10 + j * 21
        hi = k in ("ens", "r2_mlp24")
        b.append(t(L - 8, y + 4, n, 12, INK, "end", 700 if hi else None))
        b.append(dot(xs(v), y, 5, ACCENT if k == "ens" else (MUTED if k != "r2_mlp24" else INK), f"{n}：{v:.4f}"))
        b.append(t(xs(v) + 9, y + 4, f"{v:.4f}", 11, INK if hi else MUTED))
    return svg(W, H, b, "聚合器梯子：所有結構都落在 0.60 到 0.63 之間")


def chart_perclass():
    a, c = OPP["scores"]["hsl"]["per_class"], OPP["scores"]["fused_opp"]["per_class"]
    W, H, L, R, top, bot = 560, 268, 40, 16, 34, 26
    xs = lambda k: L + (k + .5) / 10 * (W - L - R)
    ys = lambda v: top + (0.90 - v) / 0.5 * (H - top - bot)
    b = [legend([("hsl", MUTED, "dot"), ("fused_opp", ACCENT, "dot")], L, 16)]
    for v in (.4, .5, .6, .7, .8, .9):
        b += [line(L, ys(v), W - R, ys(v)), t(L - 6, ys(v) + 4, f"{v:.1f}", 11, MUTED, "end")]
    for k, lv in enumerate(LEVELS):
        b.append(line(xs(k), ys(a[lv]), xs(k), ys(c[lv]), MUTED, 1.5))
        b.append(dot(xs(k), ys(a[lv]), 4.5, MUTED, f"{lv} hsl {a[lv]:.3f}"))
        b.append(dot(xs(k), ys(c[lv]), 5, ACCENT, f"{lv} fused_opp {c[lv]:.3f}"))
        b.append(t(xs(k), H - 8, lv, 12, INK, "middle"))
        b.append(t(xs(k), ys(c[lv]) - 10, f"{c[lv]:.2f}", 11, INK, "middle"))
    return svg(W, H, b, "每個等級的分數都上升，1D 到 3D 仍最低")


def chart_forest():
    rows = [("加對手 16 維（已採用）", comparison(OPP, "fused_opp vs fused"), True),
            ("加差值 3 維（已採用）", diff_c, True),
            ("加讓子 HD 5 維（候選）", hd_c, True),
            ("棋力標量：目標 9 維", comparison(SE, "fused_opp_se vs fused_opp"), False),
            ("落子品質特徵", comparison(QUAL, "fused_opp_q vs fused_opp"), False),
            ("最後一層：CORAL", comparison(ORD, "coral vs mlp"), False),
            ("最後一層：成對 BT", comparison(ORD, "bt vs mlp"), False),
            ("五折交叉擬合 x5", comparison(XFIT, "fused_opp_x5 vs fused_opp"), False),
            ("每局形狀 8 維", shape_c, False),
            ("每手 pmi 直方圖 10 維", hist_c, False),
            ("TSFEL PCA-16", TSFEL16, False),
            ("純雜訊 16 維（對照）", NOISE16, False),
            ("xgboost 代 MLP", gbm_c, False),
            ("MLP h128", h128_c, False),
            ("MLP L1 1e-3", L1, False)]
    W, L, R, top = 600, 186, 58, 14
    H = top + len(rows) * 26 + 48
    xs = lambda v: L + (v + 0.035) / 0.065 * (W - L - R)
    b = []
    for v in (-.03, -.02, -.01, 0, .01, .02, .03):
        c = MUTED if v == 0 else LINE
        b += [line(xs(v), top, xs(v), H - 44, c), t(xs(v), H - 28, f"{v:+.2f}".replace("-", "−") if v else "0", 11, MUTED, "middle")]
    b.append(line(xs(.005), top, xs(.005), H - 44, WARN, 1, "4 3"))
    b.append(t(xs(.005) + 4, H - 4, "門檻 +0.005", 11, WARN))
    for j, (n, c, ok) in enumerate(rows):
        y = top + 13 + j * 26
        col = ACCENT if ok else MUTED
        b.append(t(L - 8, y + 4, n, 12, INK, "end", 700 if ok else None))
        b.append(line(xs(c["ci95"][0]), y, xs(c["ci95"][1]), y, col, 2))
        b.append(dot(xs(c["diff"]), y, 5, col, f"{n}：{signed(c['diff'])} {ci(c)}"))
        b.append(t(W - R + 6, y + 4, signed(c["diff"]), 11.5, INK, "start", 700 if ok else None))
    return svg(W, H, b, "十五項比較中只有對手、差值、讓子三步通過門檻")


def chart_pipeline():
    W, H = 560, 408
    def box(x, y, w, h, ttl, sub=None, main=False):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{"#e3f1ef" if main else "#ffffff"}" stroke="{ACCENT if main else "#b9c2c9"}" stroke-width="{1.5 if main else 1}"/>'
        s += t(x + w / 2, y + (20 if sub else h / 2 + 5), ttl, 13, INK, "middle", 700)
        if sub:
            s += t(x + w / 2, y + 37, sub, 11.5, MUTED, "middle")
        return s
    def arrow(x1, y1, x2, y2):
        ang = math.atan2(y2 - y1, x2 - x1)
        a = [(x2 - 7 * math.cos(ang - .45), y2 - 7 * math.sin(ang - .45)), (x2 - 7 * math.cos(ang + .45), y2 - 7 * math.sin(ang + .45))]
        return line(x1, y1, x2, y2, MUTED, 1.2) + f'<polygon points="{x2:.1f},{y2:.1f} {a[0][0]:.1f},{a[0][1]:.1f} {a[1][0]:.1f},{a[1][1]:.1f}" fill="{MUTED}"/>'
    b = [box(150, 4, 260, 38, "一題：同一玩家 5–20 局")]
    b += [arrow(280, 42, 95, 66), arrow(280, 42, 280, 66), arrow(280, 42, 465, 66)]
    b += [box(5, 68, 180, 54, "Human SL 18 profile", "目標落子 → 24 維"), box(195, 68, 170, 54, "L1b A／B", "目標落子 → 16 維", True), box(375, 68, 180, 54, "L1b A／B", "對手落子 → 16 維", True)]
    b += [arrow(95, 122, 245, 150), arrow(280, 122, 280, 150), arrow(465, 122, 315, 150)]
    b += [box(150, 152, 260, 42, "每個模型一份 59 維特徵（含差值 3）")]
    b += [arrow(235, 194, 175, 218), arrow(325, 194, 385, 218)]
    b += [box(85, 220, 180, 42, "MLP（A 的特徵）"), box(295, 220, 180, 42, "MLP（B 的特徵）")]
    b += [arrow(175, 262, 245, 286), arrow(385, 262, 315, 286)]
    b += [box(150, 288, 260, 42, "10 類機率平均", None, True)]
    b += [arrow(280, 330, 280, 352)]
    b += [box(100, 354, 360, 50, "期望得分決策", "argmax [ q(k) + e⁻¹ ( q(k−1) + q(k+1) ) ]")]
    return svg(W, H, b, "提交管線：三路特徵加差值接成 59 維，經 A、B 兩個 MLP 平均機率後做期望得分決策")


def rank_strip():
    cells = "".join(f'<button type="button" data-i="{k}" aria-pressed="false"><span class="lv">{lv}</span><span class="rg">{rg}</span><span class="sc"></span></button>'
                    for k, (lv, rg) in enumerate(zip(LEVELS, ["10–12k", "7–9k", "4–6k", "1–3k", "1d", "2d", "3d", "4d", "5d", "6d"])))
    return f'<div class="strip-wrap"><div class="strip" id="rankStrip" role="group" aria-label="點選真實等級，看每個預測的得分">{cells}</div><p class="strip-note">點一格設定真實等級，格內數字是猜這一級能拿的分數。D–A 各涵蓋 3 個級位，1D–6D 各 1 段。</p></div>'


def table(head, rows, bold_last=False, num_from=1):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = []
    for r_i, r in enumerate(rows):
        tds = "".join(f'<td class="n">{c}</td>' if k >= num_from else f"<td>{c}</td>" for k, c in enumerate(r))
        cls = ' class="hl"' if bold_last and r_i == len(rows) - 1 else ""
        body.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def fig(content, cap, data, scope=None):
    chips = f'<span class="source-chip" data-field="資料"><b>資料</b> {data}</span>'
    if scope:
        chips += f'<span class="source-chip source-scope" data-field="範圍"><b>範圍</b> {scope}</span>'
    return f'<div class="fig"><div class="stage">{content}</div><div class="cap">{cap}</div><div class="sources"><div class="source-chips" role="group" aria-label="出處">{chips}</div></div></div>'


C = lambda k: comparison(XFIT, k)
se_c, se2_c = comparison(SE, "fused_opp_se vs fused_opp"), comparison(SE, "fused_opp_se2 vs fused_opp")
coral_c, bt_c = comparison(ORD, "coral vs mlp"), comparison(ORD, "bt vs mlp")
q_c, opp_c = comparison(QUAL, "fused_opp_q vs fused_opp"), comparison(OPP, "fused_opp vs fused")
f0opp = FOLD0["comparisons"]["fused_opp_avg vs fused_avg"]
diff_c, hd_c = comparison(ERR, "mlp_diff vs mlp"), comparison(HD, "fused_opp_diff_hd vs fused_opp_diff")
hist_c, shape_c = comparison(HIST, "fused_opp_diff_hist vs fused_opp_diff"), comparison(SHAPE, "fused_opp_shape vs fused_opp")
gbm_c, h128_c = comparison(ERR, "gbm vs mlp"), comparison(ERR, "mlp_h128 vs mlp")
ens_tree_c, ens_tree_mlp_c = comparison(GB, "ens_top3+xgb_fused vs ens_top3"), comparison(GB, "ens_top3+xgb_fused vs ens_top3+mlp_fused")
f0diff = DIFF0["fused_opp_diff_avg_vs_fused_opp_avg"]
D5, FSS = FP2["same_n5"], FS["summary"]
# Results that live only on PR branches (tsfel_check.json PR #39, qmodel_check.json PR #43, seq_check_mamba3.json PR #44, feature_stability_hd.json PR #48): quoted as constants.
TSFEL16 = dict(diff=-0.0128, ci95=[-0.0167, -0.0090]); NOISE16 = dict(diff=-0.0014, ci95=[-0.0042, 0.0012]); L1 = dict(diff=0.0006, ci95=[-0.0026, 0.0038])
HD_STAB = {"10": 0.845, "5": 0.733}
REF_CTM = "Darlow, L., Regan, C., Risi, S., Seely, J., &amp; Jones, L. (2025). Continuous thought machines. <em>NeurIPS 2025</em>. arXiv:2505.05522."
REF_KG = "Wu, D. J. (2019). Accelerating self-play learning in Go. <em>arXiv:1902.10565</em>.（本研究使用 KataGo 的 b18c384nbt-humanv0 權重）"
REF_SE = "Chen, Shih, &amp; Wu. <em>Strength Estimation and Human-Like Strength Adjustment in Games</em>. ICLR 2025."
REF_MS = "Kuboki, Ogawa, Hsueh, Yen, &amp; Ikeda. <em>Policies of Multiple Skill Levels for Better Strength Estimation in Games</em>. AIIDE 2025. arXiv:2505.00279."
REF_CORAL = "Cao, W., Mirjalili, V., &amp; Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. <em>Pattern Recognition Letters</em>, 140."
REF_REPO = "本專案紀錄：<code>METHOD_NOTES.md</code>、<code>docs/NANO4.md</code>、<code>results/</code>、<code>reports/architecture_v2_2026-11-02.md</code>；GitHub issue #6、#23、#36、#42。"

CHAPTERS = [
 dict(n=1, title="任務與資料", hook="只憑 5–20 局棋譜，<br>要判斷一位陌生玩家屬於哪個等級。",
  lead="AI CUP 2026 秋季賽「棋力預測」給同一位未知玩家的 5–20 局 SGF 棋譜，要求預測這位玩家在 10 個有序等級中的哪一級。本章先把題目、計分、資料與驗證方式說清楚，後面每一章的數字都建立在這些約定上。",
  meta="訓練 100 萬局 · 8,007 人 · 測試 Public 400 + Private 400 題", route=["任務", "資料", "驗證"],
  conclusion="輸出是十類機率加期望得分決策；資料裡只有落子、執色與讓子數，沒有勝負與對手等級；離線比較一律依玩家留出，並事先指定通過條件。",
  next_q="我們用什麼工具把一手棋變成數字？第 2 章介紹 KataGo 與 Human SL。", refs=[REF_REPO],
  cards=[
   dict(id="card-task", q="題目到底給什麼、要我們交什麼？", h="每題給一位玩家的 5–20 局 SGF 與各局執色，交一個等級；猜中 1 分、差一級 0.368 分、其餘 0 分",
    formula="k̂ = argmax<sub>k</sub> [ q<sub>k</sub> + e<sup>−1</sup> ( q<sub>k−1</sub> + q<sub>k+1</sub> ) ]",
    bullets=["輸入：一題一列，question_id、num_games、sgf_1…sgf_20、color_1…color_20；每局是完整 SGF（雙方著手），color 是目標玩家的執色", "輸出：submission_rank.csv，每題一個標籤 D、C、B、A、1D–6D；Public 400 題先、Private 400 題後，名次依 Private", "2026-11-04 11:00 釋出測試集，11-24 23:59:59 截止；每天 5 次上傳，以最後一次計分"],
    fig=fig(rank_strip(), "互動示意：點一格設定真實等級，看猜每一級各拿幾分", "AIdea 競賽頁面、Tutorial README"),
    cause=("10 類有序，差一級仍有分", "輸出機率，選期望得分最高的等級", "資料裡有什麼、沒有什麼？"),
    sec="1.1 任務與計分",
    body="""<p>十個等級由弱到強是 D、C、B、A、1D、2D、3D、4D、5D、6D：D–A 各涵蓋三個級位（10–12k、7–9k、4–6k、1–3k），1D–6D 各一段。計分對相鄰等級給 e⁻¹ 的部分分數、差兩級以上給 0，所以它是半徑 1 的帶狀獎勵，不是有序距離；最佳決策是對每個 k 算「猜 k 的期望得分」q<sub>k</sub> + e⁻¹(q<sub>k−1</sub> + q<sub>k+1</sub>) 再取最大，不一定是機率最高的那一類。</p>
<p>官方只說每題 5–20 局、以 num_games 給定；實際分布要等測試集釋出才知道。官方 baseline（Tutorial notebook：整局畫成平面進 ResNet，逐局 softmax 平均）在 Public 是 0.4631。</p>
<p>本報告的主要指標是「各級平均官方分」：先按 5、10、20 局分開算，每種題長在 10 個等級間等權平均，再取三種題長的平均；與官方同口徑的直接平均（每題等權）比它低約 0.007（提交候選：平衡版 0.6916、直接平均 0.6845），因為平衡版讓三種題長與各等級等權，直接平均則照內部驗證題的組成。兩種口徑都會報，差值與信賴區間一律用同一份配對預測表。</p>""",
    take="任務是十類有序分類，但計分讓「差一級」仍值 0.368 分；決策規則要照計分來，不是照準確率。"),
   dict(id="card-data", q="訓練資料裡有什麼、沒有什麼？", h="100 萬局、8,007 位玩家；每局只有一方有標籤，沒有勝負與對手等級",
    formula="train_{rank}.csv：player_id、game_id、rank、color、sgf_content",
    bullets=["每位玩家只屬一個等級，平均 125 局；對手等級未知，對局沒有時間序", "v1.1 加了讓子數 HD[n]：互先 79.3%、讓先 17.1%、讓子 3.6%（36,421 局）", "讓子局沒有記錄讓子石、B/W 標記相反，color 欄是實際顏色"],
    fig=fig(table(["欄位或事實", "內容"], [["每局", "一方是標籤玩家（player_id、rank、color），另一方未知"], ["沒有的", "勝負、目數、用時、玩家名、對手等級、時間序"], ["讓子（v1.1）", "HD0 79.3%、HD1 17.1%、HD2–9 3.6%；讓子石未記錄，B/W 相反"], ["貼目", "資料約定 HD0 6.5、HD≥1 0；特徵一律輸入 7.5（第 2 章）"], ["評分時取樣", "每位玩家只取 20 局（全量評分 7,948 人、147,780 局）"]], num_from=9), "資料集 v1.1（2026-11-01），issue #42 的普查", "data/training、results/data/hd_table.csv"),
    cause=("要知道哪些訊號根本不在資料裡", "逐欄盤點，讓子資訊另外普查", "離線分數怎樣才可信？"),
    sec="1.2 資料與棋譜",
    body="""<p>這些事實決定了後面的路：沒有勝負與目數，所以「這盤誰贏」不能當特徵；對手等級未知，但線上配對依實力，對手的落子仍帶有目標玩家的資訊（第 5 章）；讓子數由配對系統依兩人等級差決定，是資料裡唯一直接寫著「標籤側」的欄位（第 5 章的 HD 特徵）。</p>
<p>讓子局的修正寫在 <code>aicup.sgf.setup_and_moves</code>，只需重評那 3.6% 的局（第 8 章）。</p>""",
    take="資料只給落子、執色、讓子數；凡是需要勝負或對手等級的方法，一開始就排除。"),
   dict(id="card-protocol", q="離線分數怎樣才可信？", h="所有實驗依玩家整人留出，通過條件事先指定，fold 0 只作採用前的單次檢查",
    formula="(1) CI 下界 &gt; 0　(3) ≥ 3／4 折為正　(4) 差 ≥ 0.005",
    bullets=["同一人不會同時出現在訓練與驗證；folds 1–4 留一折 CV，39,327 題", "fold 0：9,872 題、1,594 人；已用四次，再用要使用者本人同意", "驗證題每人 1×20 + 2×10 + 4×5 局是內部設計；官方只說 5–20 局，實際分布 11/04 才知道"],
    fig=fig(table(["項目", "設定"], [["切分", "依 player_id 分 5 折（seed 42）"], ["驗證題", "每人 1 題 20 局、2 題 10 局、4 題 5 局"], ["訓練題", "每人 32 題，5–20 局隨機；約 152,000 題"], ["不確定性", "玩家層級 bootstrap，5,000 次，兩模型共用抽樣"], ["學習式特徵", "PCA、標準化、校準、選特徵都只在訓練內玩家擬合"]], num_from=9), "條件編號沿用聚合器梯子；條件 (2) 是多 seed 一致，單次確定性擬合的模型不適用", "METHOD_NOTES §2.1"),
    cause=("反覆比較容易選到運氣", "條件事先指定，fold 0 只當檢查", "我們用什麼工具把棋譜變成數字？"),
    sec="1.3 驗證協議",
    body="""<p>5 折切分依 <code>player_id</code> 整人留出。比較時以玩家為重抽樣單位，因為同一人的多題高度相關（第 7 章有實測）。新實驗只比 folds 1–4，並事先指定要比哪一條、條件是什麼；接近門檻才補預先固定的少量種子。單折雜訊底線：同一特徵的 float32 捨入可讓單折動 ±0.001。</p>
<p>離線分數不能和排行榜直接比：題目、玩家與題長分布都可能不同。官方 baseline 在 Public 是 0.4631，同一模型在本協議下是 0.399（argmax）與 0.440（期望得分決策）。</p>""",
    take="以玩家為單位切分與重抽樣；比較事先指定；fold 0 不參與選擇，所以 0.69 左右的分數可以互相比較，但不等於排行榜分數。"),
  ]),
 dict(n=2, title="工具：KataGo 與 Human SL", hook="一手棋怎麼變成數字？<br>我們用 KataGo 的人類落子模型，不做搜尋。",
  lead="每一章的特徵都從同一個來源算出：KataGo 的 Human SL 網路對「這個局面、這一手」給出的機率。本章介紹 KataGo 是什麼、Human SL 的等級 profile 怎麼用、重播與評分的設定與成本，以及為何單手證據必須沿整題累積。",
  meta="b18c384nbt-humanv0 · 18 個 profile（12k–6d）· 一次前向，無搜尋", route=["KataGo", "profile", "累積"],
  conclusion="KataGo 只當凍結的特徵抽取器：重播每局、對每手算實際落子在各等級 profile 下的 log p；單手證據弱，相鄰等級更弱，所以一切都在題目層級累積。",
  next_q="用這些每手數字能直接判斷等級嗎？現行架構怎麼接？第 3 章。", refs=[REF_KG, REF_REPO],
  cards=[
   dict(id="card-katago", q="KataGo 是什麼？我們用它的哪一部分？", h="KataGo 是開源圍棋引擎；我們只用它的類神經網路做一次前向，不做搜尋，網路權重凍結",
    formula="trunk（18 blocks × 384 channels）→ policy head：p(a | 局面, metadata)，a ∈ 361 點 + pass",
    bullets=["網路輸入是局面的二值平面（棋子、氣、歷史、劫）與全域特徵（規則、貼目、metadata）；輸出 policy（下一手分布）、value、目數、領地", "humanv0 是 KataGo 的 Human SL 網路：用線上人類棋譜訓練、以 metadata 指定雙方等級，可模仿指定等級的人類下法", "我們直接以 PyTorch 跑權重，與 KataGo 引擎對照過：20 個局面的目標落子最大差 3.8×10⁻⁴，argmax 全部一致"],
    fig=fig(table(["元件", "我們怎麼用"], [["重播", "KataGo 的 Python Board／GameState 重播每局，產生每手前的輸入平面"], ["Human SL policy", "18 個等級 profile 各一次前向，取實際落子那一格的 log p（第 4 章）"], ["trunk", "凍結；L1b 在它上面加 10 個自訓的等級 head，一次前向出 10 個 log p（第 4 章）"], ["value／score head", "試過當落子品質特徵，未採用（第 6 章）"], ["搜尋（MCTS）", "不用；強網棋損 Q 是待核定的支線（第 8 章）"]], num_from=9), "KataGo v1.15 的 b18c384nbt-humanv0 權重；程式在 repo/KataGo 與 src/aicup/humansl.py、l1b.py", "docs/ARCHITECTURE.md"),
    cause=("要把落子變成可比較的數字", "借用已學會人類下法的網路", "等級 profile 給的是什麼？"),
    sec="2.1 KataGo 與 humanv0",
    body="""<p>KataGo（Wu, 2019）是自我對弈訓練的圍棋引擎；它的網路是一個殘差式 trunk 接多個 head。本專案用的 humanv0 不是最強的自弈網路，而是「Human SL」：同一個結構改用人類棋譜監督訓練，並把雙方等級當 metadata 輸入，所以同一個局面換一個 profile，policy 就變成那個等級的人會怎麼下。</p>
<p>我們不改權重、不做搜尋：每個局面對每個 profile 做一次前向，只記錄實際落子那一格的 log p。18 個 profile 對應 12k 到 6d 共 18 級，比競賽的 10 個等級細，映射要用本資料學（第 4 章）。</p>""",
    take="KataGo 在這裡是凍結的特徵抽取器；所有訓練都發生在它之上或之後。"),
   dict(id="card-evidence", q="等級 profile 給的證據長什麼樣？", h="同一局面，弱棋與強棋 profile 的落子分布差很多；但單手的差會被其他手蓋過，相鄰等級更是小一個量級",
    formula="L<sub>k</sub>(Q) = Σ<sub>a ∈ Q</sub> log p(a | H, k)",
    bullets=[f"下面的局面：12k 與 6d 兩個 profile 的分布總變異距離 {VIZ['position']['tv_12k_6d']:.2f}", "沿一局累積對數似然比，遠距等級（12k vs 6d）幾十手就分開，相鄰（1d vs 2d）起伏小得多", "每手 pmi 在相鄰等級間的分布重疊 0.86–0.94：單手幾乎分不出相鄰等級"],
    fig=fig(chart_boards(), f"D 級玩家第 {VIZ['position']['t']} 手，橘圈是實際落子，圓面積正比於落子機率；兩分布總變異距離 {VIZ['position']['tv_12k_6d']:.2f}", "results/viz_examples.json", "單一局面的例子"),
    cause=("落子裡有等級訊號嗎？", "看同一局面在不同 profile 下的分布，再沿手數累積", "重播與評分的設定是什麼？"),
    sec="2.2 沿落子累積證據",
    body=f"""<p>下面兩張圖沿目標玩家的落子累積對數似然比。上圖比遠距的 12k 與 6d，訊號隨手數明顯分開；下圖比相鄰的 1d 與 2d，曲線起伏小得多。兩張圖的縱軸刻度不同。</p>
<div class="fig">{cum_panel('12k', '6d')}</div><div class="fig">{cum_panel('1d', '2d')}</div>
<p class="note">資料：<code>results/viz_examples.json</code>，兩局棋，用來說明問題的形狀，不是統計結論。</p>""",
    take="遠距等級靠累積就能分開；相鄰等級的訊號小一個量級，這是後面每一章的主要困難。"),
   dict(id="card-replay", q="重播與評分用什麼設定？成本多少？", h="所有局用 RULES_CHINESE、whiteKomi 7.5 重播；每手只存實際落子那一格；Human SL 18 次前向、L1b 一次前向出 10 維",
    formula="每局 → 每手 (18 + 10 + 10) 個 log p → 每局加總 → 每題統計",
    bullets=["貼目一律輸入 7.5：這是特徵抽取設定，訓練與推論一致，只改推論端會錯配", "L1b 評分約 900 局面／秒／卡；全量兩側約 2.2 億局面，8 卡一節點約 8–9 小時", "每手只存實際落子的 log p（不存 362 維分布、value、trunk 向量）；每手 10 個 head 的 log p 存在 moves_*.npz，可直接做逐手統計"],
    fig=fig(table(["設定", "值"], [["重播", "KataGo Board／GameState，RULES_CHINESE，whiteKomi 7.5（所有局）"], ["讓子修正（v1.1）", "HD≥2：補標準星位、B/W 對調、目標取 color"], ["Human SL", "18 個 profile，各一次前向，目標手的 log p"], ["L1b", "凍結 trunk + 10 個 head，一次前向；目標手與對手手各 10 維"], ["精度", "bf16 autocast；分片與單卡決策一致，每手差 ≤ 4×10⁻³ nats"], ["儲存", "每手實際落子的 log p；每局加總；題目層 59（+5）維"]], num_from=9), "docs/NANO4.md §4.4、§4.19；results/l1b/scores3/fold3 的 meta", "nano4 實測"),
    cause=("特徵要可重現、可對帳", "固定規則與貼目、只存需要的量", "這些數字怎麼接成提交模型？"),
    sec="2.3 重播與評分",
    body="""<p>貼目是網路的全域輸入之一，所以 7.5 一律的設定會影響每手 log p；但它對訓練與推論一致。是否改成資料約定的貼目，要先抽樣量到十類機率與決策（第 8 章），不用「每手差 ±0.001」當判準。</p>
<p>讓子修正只重評那 36,421 局（R-hdpatch），head 不變；head 一旦重訓，全量分數都要重算。</p>""",
    take="評分設定固定且可對帳，之後所有比較才有同一個基準。"),
  ]),
 dict(n=3, title="現行架構：提交候選", hook="從棋譜到十類機率，<br>現在的模型是怎麼接的？",
  lead="本章先把現在要提交的模型整個講一遍：重播 → 每手 log p（Human SL 18、L1b 目標與對手各 10）→ 每局加總 → 59 維題目特徵 → MLP → 期望得分決策。之後的章節再回頭說明每一塊是怎麼被證據選進來的。",
  meta="fused_opp_diff：CV 0.6916 · fold 0 0.6888 · 退路 fused_opp 0.6735 / 0.6716", route=["管線", "分數"],
  conclusion="提交候選 fused_opp_diff 由三路每手證據加差值組成，A、B 兩套 out-of-fold 特徵各走一個 MLP 後平均；從官方 baseline 0.44 到 0.69，增益來自鑑別式 L1b 與對手通道。",
  next_q="每手證據是怎麼從 Human SL 走到 L1b 的？第 4 章。", refs=[REF_REPO],
  cards=[
   dict(id="card-pipeline", q="從棋譜到十類機率，現在的管線是什麼？", h="三路每手證據加差值接成 59 維，A、B 兩個 MLP 平均，再做期望得分決策；每一層都按玩家 cross-fit",
    formula="p = ½ [ MLP(x<sub>A</sub>) + MLP(x<sub>B</sub>) ]，x = hsl<sub>24</sub> ⊕ l1b<sub>16</sub> ⊕ opp<sub>16</sub> ⊕ diff<sub>3</sub>",
    bullets=["Stage 1 表示：每手 Human SL 18 個 profile 的 log p、L1b 10 個 head 對目標手與對手手的 log p；trunk 凍結", "Stage 2 聚合：每局各維度加總，題內做統計（相對曲線、峰值、posterior-mean index、log 局數、log 總手數…）→ 59 維", "Stage 3 分類：MLP h32、類別平衡；A、B 機率平均；argmax 期望得分"],
    fig=fig(chart_pipeline(), "退路 fused_opp 共用同一組分片檔，不必重新評分", "docs/NANO4.md §4.13；架構圖頁 claude.ai/artifact/YAJNakFnSxj99rbJMkwWC5"),
    cause=("前面兩章的工具", "把每手證據聚合成題目層特徵，再分類", "這一版的分數與成本？"),
    sec="3.1 管線",
    body="""<p>Cross-fit：L1b 的 head A 用 folds 1、2 的玩家訓練，評 folds 0、3、4；head B 用 3、4 訓練，評 0、1、2。每位玩家的特徵都來自沒看過這位玩家的模型；fold 0 兩個模型都沒看過。題目層的 MLP 也只用訓練折的題，標準化與任何學習式特徵都只在訓練內玩家擬合。</p>
<p>對手通道：同一組 head 評同一局棋裡另一色的落子，得到第二份證據；差值 3 維是每局「對手 pmi − 自己 pmi」的題內統計。讓子 5 維（互先／讓先／讓子比例、有號讓子數 up／down）已在 CV 通過，是否進提交看測試檔有沒有帶 HD。</p>""",
    take="三層各自可換，但每一層的學習式輸出都必須是 out-of-fold 的，否則題目層會學到洩漏。"),
   dict(id="card-progress", q="這一版的分數怎麼來的？代價多少？", h="從官方 baseline 0.44 到 0.69：Human SL 映射 +0.17，鑑別式 L1b +0.035，對手通道兩步 +0.037，讓子 5 維再 +0.007（候選）",
    formula="CV（folds 1–4 留一折）與 fold 0（只在採用前查一次）兩個數",
    bullets=["+0.17 是 fold 0 同題比較，其餘是 CV；L1b 之後的三版 fold 0 只比 CV 低 0.001–0.003", "fused_opp_diff：命中 0.548、±1 內 0.919；到 0.80 若 ±1 內不變，命中要 73.1%", "推論成本：mock 40 題單卡 298 秒；正式 800 題單卡約 2–3 小時、8 卡約 20 分"],
    fig=fig(chart_progress(), "實心點是 folds 1–4 CV，空心點是 fold 0；HD 5 維只有 CV", "results/l1b/opp_check.json、opp_fold0_check.json、error_analysis.json、rank_model_fused_opp_diff.json、hd_check.json"),
    cause=("要知道每一塊各值多少", "同一套協定逐步加", "每手證據是怎麼演進的？"),
    sec="3.2 分數與成本",
    body=f"""<p>各里程碑：Human SL + MLP {num(score(OPP,'hsl'))}／fold 0 {num(score(FOLD0,'hsl'))}；加鑑別式 L1b 的 fused {num(score(OPP,'fused'))}／{num(score(FOLD0,'fused_avg'))}；加對手的 fused_opp {num(score(OPP,'fused_opp'))}／{num(score(FOLD0,'fused_opp_avg'))}；加差值的 fused_opp_diff {num(score(ERR,'mlp_diff'))}／{num(score(DIFF0,F0D_KEY))}；再加 HD 5 維 {num(score(HD,'fused_opp_diff_hd'))}（CV）。</p>
<p>六種序列架構、三組題目層特徵塊、換模型（樹、h128、L2、softmax、提早停止、L1）都在 ±0.005 內或更差（第 6 章）。</p>""",
    take="分數的每一步都有配對比較與 fold 0 對照；下一步的增益要在修正後的基準上重新量（第 8 章）。"),
  ]),
 dict(n=4, title="每手證據：Human SL 到 L1b", hook="既有的人類落子模型能直接判斷等級嗎？<br>用本資料的標籤訓練會更好嗎？",
  lead="KataGo Human SL（humanv0）可依等級 profile 產生人類落子機率，是最便宜的起點；L1b 在同一個 trunk 上加 10 個等級 head，用本資料標籤訓練。本章說明兩者的訊號，以及為何只有鑑別式目標才把題目分數拉開。",
  meta="humanv0 18 個 profile · L1b 模型 A（folds 1、2）與 B（folds 3、4）", route=["尺度", "映射", "L1b"],
  conclusion="Human SL 的序數訊號可用但尺度要學（0.6192）；L1b 的增益來自鑑別式目標（+0.0352），fold 0 的檢查確認沒有洩漏。",
  next_q="目標玩家的落子之外還有證據嗎？第 5 章看聚合方式與對手的落子。", refs=[REF_KG, REF_REPO],
  cards=[
   dict(id="card-scale", q="Human SL 的等級和本資料的等級對得上嗎？", h="訊號單調但尺度被壓縮：10 個等級只對到 KGS 6k–2d，硬分箱僅 0.25",
    formula="pmi = Σ<sub>r</sub> r · softmax( L<sub>r</sub> )",
    bullets=["似然峰值隨等級單調右移；posterior-mean index 與等級的 Spearman 0.944", "零樣本硬分箱官方分 0.25（段位幾乎全錯）", "1D–4D 擠在 2k–1k，尺度必須學"],
    fig=fig(chart_compression(), "每級 20 人、每人 20 局；3D 的峰值落在 2k、1k 之間，畫在中點", "results/pilot_humansl.json", "fold 0 pilot，200 人"),
    cause=("預訓練模型能零樣本判斷嗎？", "每級 20 人掃 18 個 profile", "尺度要學，怎麼學？"),
    sec="4.1 Pilot：零樣本的結果",
    body="""<p>Pilot 從 fold 0 與 fold 1 各取 200 人（每級 20 人）× 20 局，每手算 18 個 profile。</p>""",
    take="Human SL 的序數訊號可用，等級尺度不能照搬，必須用本資料學映射。"),
   dict(id="card-mlp", q="用本資料學映射能到多少？", h="24 維題目特徵加 MLP 到 CV 0.6192，比官方 baseline 高約 0.17",
    formula="x(Q) ∈ ℝ<sup>24</sup> → MLP(h = 32) → q(k | Q)",
    bullets=["18 維相對曲線、峰值、log 局數、log 總手數等 24 維", "fold 0：0.6113，與 CV 的差在雜訊內", "1D、2D、3D 仍只有 0.47、0.42、0.50"],
    fig=fig(table(["模型（同一組 fold 0 題）", "官方分"], [["官方 baseline，argmax", "0.399"], ["官方 baseline，期望得分", "0.440"], ["單調門檻", "0.597"], ["邏輯迴歸", "0.600"], ["MLP h32（每人 32 題）", num(score(FOLD0, "hsl"))]], bold_last=True), "官方 baseline 為 Tutorial notebook 原封重跑，只改切分", "results/baseline_official_eval.json、opp_fold0_check.json"),
    cause=("尺度要學", "題目特徵 + 小 MLP", "每手證據能不能用本資料的標籤變好？"),
    sec="4.2 題目特徵與第一個基準",
    body=f"""<p>全量評分涵蓋 7,948 人、147,780 局、1,643 萬個目標落子，每手 18 個 profile。題目特徵 24 維：18 維相對每手 log p 曲線、最佳 profile 的每手 log p、posterior-mean index、拋物線內插峰值、log 總手數、log 局數、各局最佳 profile 的標準差。folds 1–4：門檻 0.596、邏輯迴歸 0.609、MLP h32 {num(score(OPP,'hsl'))}。換線性或非線性分類器不是往上的路。</p>""",
    take="學出來的映射讓分數從 0.25 升到約 0.62；再往上要改每手證據本身。"),
   dict(id="card-disc", q="每手證據能不能用本資料的標籤變好？", h="能，但只有鑑別式目標有效：NLL 變好不帶動題目分數，加等級交叉熵後 fused 升到 0.6545",
    formula="L = NLL + λ · CE( log p<sub>1…10</sub>(a), k )，λ = 1",
    bullets=["10 個等級 head 接在凍結的 humanv0 trunk 上，一次前向出 10 個 log p", "A 訓 folds 1、2 評 0、3、4；B 訓 3、4 評 0、1、2：每位玩家的特徵都來自沒看過這位玩家的模型", f"fused {num(score(OPP,'fused'))}，對 hsl +0.0352 [+0.0303, +0.0404]；fold 0 {num(score(FOLD0,'fused_avg'))}，+0.0422 [+0.0323, +0.0524]"],
    fig=fig(table(["階段", "inner NLL", "rank CE", "fused CV"], [["1 凍結 trunk", "1.6164 → 1.4885", "—", "0.6257"], ["2 微調 trunk", "→ 1.4339", "—", "0.6250"], ["3 鑑別式", "→ 1.5156", "2.286 → 2.137", "0.6545"]], bold_last=True), "NLL 與 rank CE 為模型 A 的內層分數", "results/l1b/heads*_A.json、cv_summary{,2,3}.json"),
    cause=("Human SL 不是為本資料的等級訓練", "在 trunk 上訓 head，保持 out-of-fold；直接訓練 head 之間的等級鑑別力", "目標玩家的落子用完了，還有別的證據嗎？"),
    sec="4.3 L1b：結構、鑑別式目標與洩漏檢查",
    body=f"""<p>metadata 用 humanv0 訓練時見過的「等級未知」列，等級資訊只經由 10 個 head 注入，所以一次 trunk 前向就得到 10 個等級的 log p。生成式 NLL 要每個 head 把自己等級的落子預測好，但判斷等級需要的是 head 之間的差異；鑑別式目標把 10 個 head 對實際落子的 log p 當 logits 預測玩家等級，選模看 NLL + CE。</p>
<p>增幅遠大於所有聚合器實驗，所以先查洩漏：A、B 的訓練與出特徵 folds 沒有交叉；再用兩個模型都沒看過的 fold 0 檢查（經使用者核准、不作選擇）：fused_A 0.6479、fused_B 0.6510、平均 0.6535，與 CV 一致。測試時 A、B 各出一套特徵、各走一次 MLP，機率平均。</p>""",
    take="對這個計分，目標函數比模型容量重要；每手層級的 NLL 進步不保證傳到題目層級（第 6 章還會看到兩次）。"),
  ]),
 dict(n=5, title="聚合與第二份證據", hook="同樣的逐局證據，<br>換聚合器或加對手的落子，能再往上嗎？",
  lead="本章先比一整條聚合器梯子，說明增益上限；再加入對手的落子、對手與自己的差值、讓子資訊，L1b 之後通過門檻的只有這三步。",
  meta="fused_opp_diff：59 維 = 24 + 16 + 16 + 3 · HD 再加 5 維", route=["聚合器", "對手", "差值與讓子"],
  conclusion="單一聚合器的增益上限約 0.01，三者平均 +0.011；對手通道兩步共 +0.037，HD 5 維再 +0.0071。提交候選 fused_opp_diff：CV 0.6916、fold 0 0.6888。",
  next_q="其他方向為什麼都沒過？第 6 章列出全部比較。", refs=[REF_CTM, REF_REPO],
  cards=[
   dict(id="card-ladder", q="換更強的聚合器能多拿分嗎？", h="所有結構都落在 0.60–0.63：集合模型只比 MLP 多 +0.005 到 +0.009",
    formula="CTM − 基準 = +0.0087　　ens_top3 − 基準 = +0.0113",
    bullets=["PMA、cross-attention、CTM 兩兩差 ≤ 0.0005；三者機率平均 ens_top3 0.6306", "相鄰等級 AUC 從 5 局到 20 局只升 ≤ 0.05：瓶頸是每手證據的解析度，不是證據量"],
    fig=fig(chart_ladder(), "虛線是 24 維 + MLP 基準。集合模型都是 3 個 seed 分數的平均", "results/ladder/cv_summary.json、stack_check3.json", "folds 1–4"),
    cause=("更強的聚合器能多拿分嗎？", "同輸入同題目比一整條梯子", "增益這麼小，別的證據在哪？"),
    sec="5.1 聚合器梯子",
    body=f"""<p>四個事先指定的門檻都不成立；只看長度的 <code>r2_length_only</code> 是 {num(LAD['rungs']['r2_length_only']['balanced_mean_over_n'])}。這條線之後被 L1b 取代，集合模型沒有在新的 59 維特徵上重測；第 8 章把「<code>r3_pma</code> 接現行每局訊號」列為待排的對照。</p>
<p>六種逐手序列 encoder（conv、GRU、CTM、Transformer、Mamba、Mamba-3）用每局交叉熵加 base + Δ 的接法，全部落在 0.6705–0.6737，對 fused_opp 0.6735 都在雜訊內；逐手層級的順序訊號沒有傳到題目層。</p>""",
    take="聚合方式有小幅增益，但處理不了證據本身的解析度。"),
   dict(id="card-opp", q="對手的落子帶有目標玩家的等級資訊嗎？", h=f"帶有：加入對手 16 維後 CV {num(score(OPP,'fused_opp'))}，fold 0 {num(score(FOLD0,'fused_opp_avg'))}",
    formula="fused_opp = hsl<sub>24</sub> ⊕ l1b<sub>16</sub> ⊕ opp<sub>16</sub>",
    bullets=[f"對 fused {signed(opp_c['diff'])} {ci(opp_c)}，三條件全過", f"fold 0 對 fused {signed(f0opp['diff'])} {ci(f0opp)}", "標籤對對手 pmi 的 Spearman 0.948，高於對自己的 0.934"],
    fig=fig(chart_perclass(), "各級官方分，fused_opp 標數值。1D–3D 上升，但仍是最低的三級", "results/l1b/opp_check.json", "folds 1–4"),
    cause=("目標玩家的證據已用盡", "同一組 OOF 模型評對手的落子", "對手減自己的差值還有訊號嗎？"),
    sec="5.2 對手特徵",
    body="""<p><code>l1b_score.py --opponent</code> 用同一組 out-of-fold 模型評同一批對局裡另一色的落子，同樣做 16 維題目特徵。線上配對依實力，對手是目標玩家等級的第二次量測。</p>
<p>機制假說（未驗證）：目標玩家的棋風偏差在同一題的 n 局裡是常數，不隨 n 平均掉；對手是配對系統抽出的 n 個不同人，其偏差以 1/√n 抵消，所以對手 pmi 與標籤的相關反而更高。可檢驗的推論：對手欄的 split-half 相關應低於自己欄而增益不低——第 7 章的數字方向相符。</p>
<p class="note">前提：測試集的局也是相近實力的線上配對。訓練資料的折只按目標玩家切，head 可能看過對手的其他棋；這條路徑 fold 0 抓不到，只能在 Public 榜上檢查。</p>""",
    take="對手特徵的增益和鑑別式 L1b 同一量級。"),
   dict(id="card-diff", q="對手減自己的差值、讓子資訊，還能加分嗎？", h=f"能：差值 3 維 {signed(diff_c['diff'])} 到 {num(score(ERR,'mlp_diff'))}；HD 5 維再 {signed(hd_c['diff'])} 到 {num(score(HD,'fused_opp_diff_hd'))}",
    formula="diff = pmi<sub>opp</sub> − pmi<sub>own</sub>（每局，取題內統計）",
    bullets=[f"差值 3 維：{signed(diff_c['diff'])} {ci(diff_c)}，四折全正；fold 0 {num(score(DIFF0,F0D_KEY))}，對 fused_opp {signed(f0diff['diff'])} {ci(f0diff)}", f"HD 5 維（互先／讓先／讓子比例、有號讓子數 up／down）：{signed(hd_c['diff'])} {ci(hd_c)}，增益集中在有讓子局的題"],
    fig=fig(table(["題目分組", "題數", "fused_opp", "fused_opp_diff", "+ HD 5"], [[n, f"{HD['buckets'][k]['questions']:,}", num(HD['buckets'][k]['fused_opp']), num(HD['buckets'][k]['fused_opp_diff']), num(HD['buckets'][k]['fused_opp_diff_hd'])] for n, k in (("只有互先局", "no_hd"), ("含讓先、無讓子", "sente_only"), ("含讓子局", "stones"))]), "按題內是否含讓先／讓子局分組的各級平均官方分", "results/seq/hd_check.json、error_analysis.json", "folds 1–4"),
    cause=("殘差對（對手 − 自己）仍有 −0.213 的相關", "把差值直接當特徵；讓子是配對系統給的標籤側資訊", "其他方向為什麼都沒過？"),
    sec="5.3 差值特徵與讓子資訊",
    body=f"""<p>加差值後殘差對（對手 − 自己）的相關從 −0.213 降到 −0.130，模型學到了那一欄的主要線性部分。HD 的增益在「含讓子局」的題最大（{num(HD['buckets']['stones']['fused_opp_diff'])} → {num(HD['buckets']['stones']['fused_opp_diff_hd'])}），符合它是標籤側資訊的預期；能不能用在正式提交，要看 11/04 的測試檔有沒有帶 HD[]。</p>
<p>提交候選 <code>rank_model_fused_opp_diff.json</code>（fold 0 {num(score(DIFF0,F0D_KEY))}）與退路 <code>rank_model_fused_opp.json</code> 都在 main；HD 版的文件要再用一次 fold 0，待使用者決定。</p>""",
    take="對手、差值、讓子三步是 L1b 之後僅有的有效改進；它們都依賴「測試局也是配對產生」這個前提，讓子還要測試檔帶 HD。"),
  ]),
 dict(n=6, title="沒有通過的方向", hook="論文方法、新特徵塊、換模型，<br>為什麼都停在門檻之下？",
  lead="本章把所有事先指定的比較放在同一張圖上：棋力標量、落子品質、有序最後一層、五折交叉擬合、每局形狀、直方圖、TSFEL、序列 encoder、樹模型與 MLP 變體。全程沒有碰 fold 0。",
  meta="本章實驗全部未過門檻", route=["全部比較", "為什麼"],
  conclusion="L1b 之後通過門檻的只有對手、差值、讓子三步；每手層級的進步常傳不到題目層級，題目層多加一塊欄位常變差。",
  next_q="錯誤到底在哪裡？第 7 章切開每題機率，並用三個診斷找原因。", refs=[REF_SE, REF_MS, REF_CORAL, REF_REPO],
  cards=[
   dict(id="card-forest", q="哪些方向試過、結果如何？", h="圖上 15 列只有對手、差值、讓子三步過門檻；最差的是 TSFEL 全塊與 CORAL",
    formula="所有比較對當時的現行模型，玩家 bootstrap 95% CI",
    bullets=[f"棋力標量 {signed(se_c['diff'])}、落子品質 {signed(q_c['diff'])}、CORAL {signed(coral_c['diff'])}、成對 BT {signed(bt_c['diff'])}、x5 {signed(C('fused_opp_x5 vs fused_opp')['diff'])}", f"每局形狀 8 維 {signed(shape_c['diff'])}、直方圖 10 維 {signed(hist_c['diff'])}、TSFEL PCA-16 {signed(TSFEL16['diff'])}（4／64／全：−0.0014／−0.0375／−0.0610）", f"xgboost 代 MLP {signed(gbm_c['diff'])}；MLP h128 {signed(h128_c['diff'])}、L2×100 −0.0189、softmax −0.0245、提早停止 −0.0043、L1 {signed(L1['diff'])}"],
    fig=fig(chart_forest(), "點是差值，線段是玩家 bootstrap 95% CI；綠色是過門檻的三步；其中一列是純雜訊對照。TSFEL 64／全塊超出軸範圍，未畫", "results/l1b、results/seq、results/gbdt 的 *_check.json", "folds 1–4"),
    cause=("每個方向各有道理", "同一套協定逐項比", "為什麼每手變好、題目不變？"),
    sec="6.1 全部比較",
    body=f"""<p>論文方法：Chen 等的 Bradley–Terry 棋力標量 head 訓練成功（成對準確率 0.52 → 0.74），但接上 fused_opp 顯著變差；L1b 已有一個有序標量 <code>pmi_sum</code>，兩者只差訓練目標。落子品質（humanv0 9d 的勝率與目數損失）接上 Human SL 有幫助，接上 fused_opp 沒有。CORAL 共用一個標量與 9 個門檻，20 局題差最多（−0.041）：官方分是半徑 1 的帶狀獎勵，差 2 級和差 9 級一樣是 0，全局有序距離不是對的目標。</p>
<p>資料量：五折交叉擬合讓每個 head 多看一倍玩家，每手 rank CE 確實變好（2.100–2.110 對 2.137／2.111），題目層卻 {signed(C('fused_opp_x5 vs fused_opp')['diff'])}；刻度校正後相同，刻度假設不成立。</p>
<p>樹模型：同一份特徵上 xgboost 比 MLP 差；ens_top3 + xgb 對 ens_top3 {signed(ens_tree_c['diff'])} {ci(ens_tree_c)}，但對 ens_top3 + 多一個 MLP 是 {signed(ens_tree_mlp_c['diff'])} {ci(ens_tree_mlp_c)}——多一個成員的功勞不能算在樹上。</p>""",
    take="在已有多等級似然的前提下，標量、品質、有序輸出、更多資料、更大模型都沒有額外價值。"),
   dict(id="card-noise", q="題目層多加一塊欄位為什麼常變差？", h="多半是特徵本身的雜訊：16 欄純雜訊只掉 −0.0014，TSFEL 16 欄掉 −0.0128",
    formula="退步隨維度單調放大：+4 → −0.001，+16 → −0.013，+64 → −0.038，+267 → −0.061",
    bullets=[f"純雜訊對照 {signed(NOISE16['diff'])} {ci(NOISE16)}：MLP 對多餘欄位有抵抗力", "TSFEL 塊只有 pc1–pc4 可靠，pc5–pc16 的 split-half 相關 0.0–0.34（第 7 章）", "h16、L2、L1、提早停止都救不回 TSFEL；問題在特徵，不在正則化"],
    fig=fig(table(["加在 fused_opp_diff 上的塊", "維度", "差值", "95% CI"], [["純雜訊", "16", signed(NOISE16['diff']), ci(NOISE16)], ["每局形狀（對 fused_opp）", "8", signed(shape_c['diff']), ci(shape_c)], ["每手 pmi 直方圖", "10", signed(hist_c['diff']), ci(hist_c)], ["TSFEL PCA", "16", signed(TSFEL16['diff']), ci(TSFEL16)], ["TSFEL PCA", "64", "−0.0375", "[−0.0423, −0.0326]"], ["TSFEL 全塊", "267", "−0.0610", "[−0.0667, −0.0551]"]]), "同一套 CV；TSFEL 與雜訊對照來自 PR #39／#43 的結果檔", "results/seq/{shape,hist}_check.json、tsfel_check.json、qmodel_check.json", "folds 1–4"),
    cause=("三條線都是「加欄位就退步」", "用純雜訊對照分開模型與特徵", "錯誤到底在哪裡？"),
    sec="6.2 雜訊對照",
    body="""<p>純雜訊幾乎不掉，所以「MLP 對多餘欄位沒抵抗力」不成立；TSFEL、直方圖、形狀帶的是在 5–20 局的題上估不準的量。第 7 章用同玩家重抽與同 n 比較把這件事量化，並看到訓練題固定 5 局能把 TSFEL 對新玩家的傷害縮小約七成；不過那個改動同時換了標準化與優化，機制沒有分離。</p>""",
    take="新特徵塊要先問「同一位玩家換一組棋，它穩不穩」，再問它有沒有訊號。"),
  ]),
 dict(n=7, title="錯誤結構與診斷", hook="這麼多方向都停住，<br>錯誤到底在哪裡？",
  lead="本章切開 39,327 題的 out-of-fold 機率看錯誤落在哪裡，再用三個診斷回答「新特徵為什麼傷人」：同玩家留出局、同 n 的已見玩家相對優勢、特徵重抽穩定性。",
  meta="fused_opp OOF · 39,327 題、6,354 人 · 三個診斷 job 670875／671226／671227", route=["錯誤結構", "已見玩家", "穩定性"],
  conclusion="六成的錯誤變異在玩家之間；TSFEL 對已見玩家有 +0.007–0.008 的相對優勢但仍是負增益；它的大半成分是抽樣雜訊；訓練題固定 5 局能縮小傷害，但機制尚未分離。",
  next_q="這些證據怎麼變成規則與下一步？第 8 章是架構 v2。", refs=[REF_REPO],
  cards=[
   dict(id="card-errors", q="錯誤主要來自哪裡？", h=f"錯誤是玩家層級的：帶號誤差的變異數 {ERR['residuals']['between_player_share']*100:.1f}% 在玩家之間",
    formula="到 0.80：若 ±1 內維持 0.919，命中要 73.1%（現在 0.548）",
    bullets=[f"同一人全對 {ERR['residuals']['all_right']['observed']*100:.1f}%（獨立期望 {ERR['residuals']['all_right']['independent']*100:.1f}%）、全錯 {ERR['residuals']['all_wrong']['observed']*100:.1f}%（{ERR['residuals']['all_wrong']['independent']*100:.1f}%）", "多給棋局只壓玩家內：n=5／10／20 命中 0.496／0.544／0.570", "校準貼對角線；偷看標籤扣每人偏差的 0.844 不是可達目標"],
    fig=fig(table(["同一人的多題", "實測", "若獨立"], [["全對", f"{ERR['residuals']['all_right']['observed']*100:.1f}%", f"{ERR['residuals']['all_right']['independent']*100:.1f}%"], ["全錯", f"{ERR['residuals']['all_wrong']['observed']*100:.1f}%", f"{ERR['residuals']['all_wrong']['independent']*100:.1f}%"]], num_from=1), "fused_opp 的 OOF 題，每人 4 或 7 題", "results/l1b/error_analysis.json", "folds 1–4"),
    cause=("實驗持續停在 0.69 附近", "切開每題機率看錯誤結構", "新特徵是在記玩家嗎？"),
    sec="7.1 錯誤結構",
    body="""<p>同一人多題全對或全錯的比例都是獨立假設的 3–4 倍（18.4% 對 5.1%、14.1% 對 3.8%）：伺服器等級與這些局下出來的棋力之間，每人有一個固定落差。對手通道能抵一部分（第 5 章），但加差值後殘差對（對手 − 自己）仍有 −0.130 的相關。</p>""",
    take="短期的分數空間在把差一級變成命中；要往上，需要能估每人固定落差的新證據。"),
   dict(id="card-fingerprint", q="新特徵塊是在記玩家嗎？", h=f"部分是：TSFEL 對已見玩家少掉 {signed(D5['base_tsfel16']['D_plain'])}，但仍是負增益",
    formula="D(5) = gain(已見玩家留出局) − gain(新玩家 n=5)",
    bullets=[f"TSFEL：D(5) {signed(D5['base_tsfel16']['D_plain'])} {ci(dict(ci95=D5['base_tsfel16']['D_plain_ci95']))}（直接平均）；類別平衡 {signed(D5['base_tsfel16']['D_balanced'])} {ci(dict(ci95=D5['base_tsfel16']['D_balanced_ci95']))}", f"形狀 8 {signed(D5['base_shape8']['D_plain'])}、純雜訊 {signed(D5['base_noise16']['D_plain'])}：未偵測到已見玩家相對優勢", f"訓練題固定 5 局後，TSFEL 對新玩家從 {signed(D5['base_tsfel16']['gain_new_plain'])} 縮到 {signed(D5['fixed5_base_tsfel16']['gain_new_plain'])}，D(5) 縮到 {signed(D5['fixed5_base_tsfel16']['D_plain'])}（CI 跨 0）"],
    fig=fig(table(["特徵塊", "新玩家增益", "已見玩家增益", "D(5)", "95% CI"], [[n, signed(D5[k]['gain_new_plain']), signed(D5[k]['gain_seen_plain']), signed(D5[k]['D_plain']), ci(dict(ci95=D5[k]['D_plain_ci95']))] for n, k in (("形狀 8", "base_shape8"), ("TSFEL PCA-16", "base_tsfel16"), ("純雜訊 16", "base_noise16"), ("TSFEL，訓練題固定 5 局", "fixed5_base_tsfel16"))]), "訓練玩家每人 20 局留 5 局組成「已見玩家」題；全域 player_id 整包 bootstrap 5,000 次，兩組共用抽樣", "results/seq/fingerprint_check_v2.json", "folds 1–4，直接平均"),
    cause=("TSFEL 在訓練題上好、驗證折上差", "同一位訓練玩家留出的新局 vs 新玩家", "特徵本身穩不穩？"),
    sec="7.2 已見玩家相對優勢",
    body="""<p>在這套切分與訓練流程下，TSFEL 對已見玩家留出局的相對影響優於新玩家的 5 局題，幅度 0.007–0.008 超過工程門檻，所以「指紋」成分存在；但它量到的是相對優勢，還沒唯一識別出「記住玩家身分」的機制，而且 TSFEL 對已見玩家也仍是 −0.0066 到 −0.0075，不構成採用理由。</p>
<p>固定 5 局訓練把傷害縮小 0.0108。一個可能的解釋是：訓練題局數混合時，TSFEL 在多局題上雜訊較小，MLP 學到的權重對 5 局題過大。但同一個改動也換了標準化與優化，機制沒有分離，不能寫成「69% 是局數錯配」。固定 5 局的模型在 10／20 局題上只有 0.13–0.15，因為 log(局數) 在訓練時是常數，模型沿這個方向外推失敗（零變異欄的尺度設為 1，不是數值問題）。</p>""",
    take="已見玩家相對優勢存在但不改結論；採用判準仍是新玩家的增益，D 只當診斷。"),
   dict(id="card-stability", q="特徵本身穩不穩？", h=f"同一位玩家換一組棋：hsl 的 split-half 相關 {FSS['10']['hsl24']['median_split_half_corr']:.2f}，TSFEL PCA-16 只有 {FSS['10']['tsfel']['median_split_half_corr']:.2f}，接近純雜訊",
    formula="W = ½·mean<sub>p</sub>(z<sub>pA</sub> − z<sub>pB</sub>)²，between = var<sub>p</sub>(½(z<sub>pA</sub> + z<sub>pB</sub>))；純雜訊時比值 2、相關 0",
    bullets=["20 局切成不相交兩半（10+10、5+5），特徵以其他三折的 pilot 題定尺度", f"TSFEL 玩家內／玩家間 {FSS['10']['tsfel']['mean_ratio_W_over_between']:.2f}／{FSS['5']['tsfel']['mean_ratio_W_over_between']:.2f}；只有 pc1–pc4 可靠", f"diff 3 只有 {FSS['10']['diff3']['median_split_half_corr']:.2f}／{FSS['5']['diff3']['median_split_half_corr']:.2f} 卻帶來 +0.018：穩定性是診斷，不是採用標準"],
    fig=fig(table(["特徵塊", "相關 n=10", "n=5", "玩家內／間 n=10", "n=5"], [[n, f"{FSS['10'][k]['median_split_half_corr']:.3f}", f"{FSS['5'][k]['median_split_half_corr']:.3f}", f"{FSS['10'][k]['mean_ratio_W_over_between']:.2f}", f"{FSS['5'][k]['mean_ratio_W_over_between']:.2f}"] for n, k in (("hsl 24", "hsl24"), ("l1b 16", "l1b16"), ("opp 16", "opp16"), ("diff 3", "diff3"), ("形狀 8", "shape"), ("TSFEL PCA-16", "tsfel"))] + [["HD 5", f"{HD_STAB['10']:.3f}", f"{HD_STAB['5']:.3f}", "0.18", "0.34"]]), "split-half 相關中位數與玩家內／玩家間變異比；HD 5 來自 PR #48 的 feature_stability_hd.json", "results/seq/feature_stability.json", "folds 1–4"),
    cause=("加欄位就退步", "量同一玩家換棋後特徵動多少", "證據怎麼變成規則？"),
    sec="7.3 特徵重抽穩定性",
    body="""<p>HD 五欄與 l1b／opp 同級（hd_gap_down 最穩，讓先比例最不穩）。對手欄與差值欄因兩半抽到不同對手，天生偏低，所以穩定性不能當硬門檻，也不套用在對手校準與跨通道差值欄。</p>
<p>穩定性從 n=10 到 n=5 全面下降，而題目層雖有 log 局數作輸入，混合局數訓練仍可能讓雜訊塊的權重偏向多局題。可測的便宜項（待核定）：按 n 分別標準化或分別訓練、重加權訓練題的局數分布（官方測試是 5–20 局，不把其他 n 歸零）、以玩家內／玩家間變異比做乘法收縮。</p>""",
    take="新特徵塊先做 split-half 診斷：低可靠度成分限維、縮小試驗或整塊消融；採用仍看新玩家增益、CI 與分折穩定性。"),
  ]),
 dict(n=8, title="架構 v2 與規則", hook="證據怎麼變成規則？<br>接下來要做哪幾件事？",
  lead="架構分表示 → 聚合 → 分類三層，主線保留現行 MLP，先完成重播與快取修正，新訊號各自小規模驗證，樹只作對照與融合候選。五張架構圖另有頁面。",
  meta="reports/architecture_v2_2026-11-02.md · 架構圖 claude.ai/artifact/YAJNakFnSxj99rbJMkwWC5", route=["三層", "規則", "基準與支線"],
  conclusion="架構凍結；下一批最有價值的結果是修正後基準的 E10／E11，與 r3_pma 在同一份特徵上的增量。",
  next_q="模型定案之後，提交流程與未確認事項是什麼？第 9 章說明。", refs=[REF_REPO],
  cards=[
   dict(id="card-stages", q="三層各做什麼？", h="Stage 1 凍結 trunk 出每手 log p，Stage 2 每局加總再題內統計，Stage 3 MLP 加期望得分決策；每層按玩家 cross-fit",
    formula="棋譜 → 重播（讓子修正）→ 表示 → 聚合 → 分類與決策",
    bullets=["Stage 1：不從頭訓 CNN、不微調 trunk；候選新訊號是 policy 摘要、強網棋損 Q、後續著手預測增益 G", "Stage 2：固定統計 59 維 + HD 5；學習式輸出一律固定語義與尺度；整題監督的 r3_pma 是替代聚合", "Stage 3：MLP h32 基準；樹做同特徵四格對照與融合，融合的對照是多加一個不同種子的 MLP"],
    fig=fig(table(["層", "現行", "候選（都待核定）"], [["重播", "KataGo RULES_CHINESE，whiteKomi 7.5 一律", "讓子局補星位、B/W 對調、目標取 color"], ["Stage 1 表示", "Human SL 18 + L1b 10（目標、對手）", "policy 摘要、跨 head 敏感度加權與截斷、Q、G"], ["Stage 2 聚合", "每局加總 → 59 維 + HD 5", "逐局讓子校準、按 n 處理、r3_pma"], ["Stage 3 分類", "MLP h32，A／B 機率平均，期望得分決策", "樹對照／融合、帶狀效用損失、以官方分選 temperature"]], num_from=9), "每一層第 k 折的學習式輸出只來自沒看過第 k 折玩家的模型", "架構設計 v2 §4"),
    cause=("前面七章的證據", "把有效的、無效的、未測的各放到一層", "採用與否怎麼判？"),
    sec="8.1 三層",
    body="""<p>時間對齊規格只有一版：每 ply 一列，28 維分數（Human SL 18 只在目標手有觀測，L1b 10 是當前落子者），加角色、t/T、有效遮罩；缺值不以 0 進標準化。每局的學習式輸出用 10 類機率或 log-softmax，各折獨立訓的 encoder 隱藏向量不拼成同一張表。</p>""",
    take="三層只是放東西的位置；真正固定的是下一張卡的規則。"),
   dict(id="card-rules", q="一個新候選怎樣才算通過？", h="同題同抽樣同種子的四格比較，CI 下界 > 0、四折至少三折正、差 ≥ 0.005；穩定性與 D(5) 只作診斷",
    formula="{MLP, 樹} × {R, R + 新塊}；E／S／D 三組四格",
    bullets=["主線以修正後基準 R 為共同基準；每筆比較註明用哪個 R", "新特徵塊首輪 ≤ 16 欄；先過 split-half 診斷，低可靠度成分限維或整塊消融，不設硬門檻", "嚴格小型試驗（head A → folds 1、2 訓 → fold 3 下游 → fold 4 評）只篩成本與捷徑，不作採用證據"],
    fig=fig(table(["規則", "內容"], [["四格命名", "重播 × HD：E00/E01/E10/E11；新訊號 × 聚合：S00/S01/S10/S11；Q × G：D00/D10/D01/D11"], ["正式確認", "四折 OOF、玩家整包 bootstrap、兩模型共用抽樣；報每折、每種 n、各等級與 HD 分組"], ["特徵選擇", "整塊加入或移除重訓比較；單欄 permutation importance 只用來解釋"], ["校準與決策", "先比不校準與單一 temperature，只在訓練內玩家選；期望得分決策不變"], ["不預設", "HD 增益與重播增益可以相加；舊集合模型增益保證新版本"]], num_from=9), "規則與實驗清單都待使用者核定", "架構設計 v2 §5"),
    cause=("反覆比較會選到運氣", "同一套協定、同一門檻", "基準 R 怎麼來？"),
    sec="8.2 規則",
    body="""<p>純 CPU 的待核定項（都在 R 上、一次只改一項）：按 n 條件化標準化；以玩家內／玩家間變異比做乘法收縮 x／(1 + 10·r／n)；逐局對手讓子校準再聚合；跨 10 個 head 的每手標準差加權與 [−2.5, 2.5] 截斷加總（每手 10 head 的 log p 已存在，不必重評分）；帶狀效用損失與直接以官方分選 temperature。「訓練題只抽 5／10／20」因官方是 5–20 局而不採。</p>""",
    take="規則固定之後，候選可以依結果退出，不必每次失敗再加補救。"),
   dict(id="card-R", q="修正後的基準 R 與支線是什麼？", h="R-hdpatch 只重評 HD≥2 的 36,421 局，是過渡版；R-full 要重訓 head 並全量重評",
    formula="R-hdpatch ≈ 0.65 節點小時（L1b 兩側）；R-full (a) ≈ 68 GPU 小時 + Human SL",
    bullets=["貼目維持 7.5（訓練與推論一致）；要改先抽樣量到十類機率與決策", "R-full (a) 維持 7.5、head 在修正資料上重訓；(b) 改資料約定貼目則 cache_A 1,015 萬局面全部受影響", "支線 Q（強網棋損，先 100 局工程試驗）與 G（BERT 式後續著手預測增益，五個視圖控制）各自小規模；雙頭只在三條件同時成立時啟動"],
    fig=fig(table(["待使用者決定", "內容"], [["1", "fold 0 第五次使用：擬合 HD 版提交文件"], ["2", "強網權重是否下載；Q 支線沒有它無法開始"], ["3", "貼目：維持 7.5 一致版到 11/04，或列入全部重建"], ["4", "純 CPU 便宜項是否核定（第 8.2 節）"]], num_from=9), "R-hdpatch 重評後不劣（下降 ≤ 0.002）才重擬合文件", "架構設計 v2 §6–§8"),
    cause=("3.6% 的局在錯的棋盤上評分", "先出過渡版，再決定完整修復的範圍", "提交流程與風險呢？"),
    sec="8.3 基準版本、支線與待決事項",
    body="""<p>E 四格：E00 舊重播、E01 舊重播 + HD 5、E10 新重播、E11 新重播 + HD 5；必報 E10−E00、E11−E10、E11−E01 與交互。G 支線的固定規格：小型 Transformer 4 層、錨點前 ≤ 32 ply 後 ≤ 8 ply、實際手永遠換成 MASK、不在被遮手座標抽向量；Past／Own／Opp／Shuffle／Both 五個視圖都要在訓練中出現，Own ≈ Shuffle 就只是局部座標資訊。</p>""",
    take="主線一條（R-hdpatch → E10／E11）、探索一條（r3_pma）；其他候選等前兩項的結果表。"),
  ]),
 dict(n=9, title="提交流程與未確認事項", hook="模型定案之後，<br>提交流程與風險是什麼？",
  lead="提交候選是 A、B 兩個 L1b 加 Human SL、再加差值 3 維的 fused_opp_diff。本章說明管線、已完成的對帳，以及 11/04 之前無法確認的事項。",
  meta="提交文件 rank_model_fused_opp_diff.json", route=["對帳", "未確認"],
  conclusion="管線與離線路徑逐項對上；排行榜分數、測試集的局數分布、HD 欄位與配對來源仍待 11/04 確認。",
  next_q="11/04 測試集公布後，先看 num_games 與 HD 計數、跑合法性檢查，再評分與提交。", refs=[REF_REPO],
  cards=[
   dict(id="card-checks", q="提交路徑和離線路徑一致嗎？", h="一致：mock test 與離線路徑 40／40 決策相同，每局 L1b 總和對帳通過；單卡與分片也相同",
    formula="p = ½ [ MLP(x<sub>A</sub>) + MLP(x<sub>B</sub>) ]",
    bullets=["375／375 局的每手 L1b 總和對帳通過；fused_opp_diff 的 mock-40 最大機率差 5.1×10⁻⁴，0 題決策不同", "單卡與 4 卡分片也 40／40 相同"],
    fig=fig(table(["檢查", "結果"], [["fold 0 重現", "對 opp_fold0_check.json 差 0"], ["每局對帳：目標與對手", "375／375 局，每手 ≤ 7×10⁻³ nats"], ["管線 vs 離線（fused_opp_diff）", "40／40，max |Δp| 5.1×10⁻⁴"], ["單卡 vs 4 卡", "40／40，max |Δp| 2.7×10⁻⁴"], ["帶 HD[] 的 mock-40", "data/mock_tests/fold0_40_v11，報告列每局 HD 計數"]], num_from=9), "退路 fused_opp 共用同一組分片檔", "docs/NANO4.md §4.13"),
    cause=("離線分數要能在提交路徑重現", "用 fold 0 做 40 題官方格式的假測試", "還有什麼沒辦法事先確認？"),
    sec="9.1 對帳",
    body="""<p>每局棋跑 18 個 Human SL profile 與 2 個 L1b trunk（目標與對手），分片評分後合併，再做特徵與決策。帶 HD[] 的 mock-40（data/mock_tests/fold0_40_v11）已準備，HD 版文件的 predict_submission 報告會列每局 HD 值的計數，測試檔有沒有帶 HD 一跑就知道。</p>""",
    take="提交路徑與離線路徑已逐項對上；R-hdpatch 完成後同一套對帳再做一次。"),
   dict(id="card-open", q="還有什麼不能確定？", h="排行榜分數、每題局數的實際分布、測試檔是否帶 HD、對手配對的前提，都要等 11/04",
    formula="本報告的分數皆為離線題，尚無 Public／Private 成績",
    bullets=["測試集尚未公布，0.69 不能當排行榜預測", "官方只說每題 5–20 局；內部驗證用 5／10／20，第一件事是看 Public／Private 的 num_games 分布", "對手通道假設測試局是相近實力的配對；head 可能看過對手其他棋的路徑只能在 Public 榜上看"],
    fig=fig(table(["11/04 的檢查順序", "看什麼"], [["1", "num_games 分布、每題是否帶 HD[]、SGF 合法性（check_legal）"], ["2", "玩家是否與訓練集重疊、各級題數是否相近"], ["3", "fused_opp_diff 與 fused_opp 各上傳一次，做 Public 榜對照"], ["4", "R-hdpatch 的 E10／E11 出來後，決定 HD 版文件是否進場"]], num_from=9), "上傳每天 5 次，以最後一次計分", "AIdea 競賽頁面"),
    cause=("離線驗證到此為止", "把未知列清楚，並準備退路", "11/04 後依序驗證"),
    sec="9.2 未確認事項與退路",
    body="""<p>A、B 各只看過 40% 的玩家，全量模型是否更好只有間接證據（第 6 章的 x5 沒有比較好）。R-hdpatch 是過渡版（head 不變），重擬合文件要再用一次 fold 0；11/04 用哪個文件由使用者決定；完整修復的版本名要不同，每筆比較註明用哪個 R。</p>""",
    take="目前的最佳估計只是離線分數；測試集公布後，第一件事是驗證這些假設。"),
  ]),
]

for ch in CHAPTERS:
    ch["total"] = len(ch["cards"]) + 1
N_PAGES = sum(c["total"] for c in CHAPTERS)
TALK_PAGES = N_PAGES + 2


def page_head(title, extra=""):
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="assets/report.css">
</head>
{extra}"""


def menu(current=None):
    items = [f'<a href="index.html"{" aria-current=page" if current == 0 else ""}><span>章節架構</span> <span class="ch-menu-meta">{"目前" if current == 0 else ""}</span></a><hr>']
    for c in CHAPTERS:
        cur = c["n"] == current
        items.append(f'<a href="ch{c["n"]:02d}.html"{" aria-current=page" if cur else ""}><span>{c["n"]} · {c["title"]}</span> <span class="ch-menu-meta">{c["total"]} 頁{" · 目前" if cur else ""}</span></a>')
    items.append(f'<hr><a href="talk.html"><span>整份報告（{TALK_PAGES} 頁）</span> <span class="ch-menu-meta"></span></a>')
    return f'<dialog class="ch-menu-dlg" id="ch-menu" aria-label="章節">{"".join(items)}</dialog>'


def build_chapter_body(ch, offset=0, prefix="", talk=False):
    """The chapter's pages and lessons; prefix namespaces ids for talk.html, offset numbers its dialog sections."""
    n, cards = ch["n"], ch["cards"]
    pid = lambda s: prefix + s
    ch_next = next((c for c in CHAPTERS if c["n"] == n + 1), None)
    lis = "".join(f'<li id="{pid(f"opening-q{i+1}")}"><b>{i+1}</b><a href="#{pid(c["id"])}">{c["q"]}</a></li>' for i, c in enumerate(cards))
    route = '<span aria-hidden="true">→</span>'.join(f'<a href="#{pid(c["id"])}">{r}</a>' for c, r in zip(cards, ch["route"]))
    attrs = f' data-chapter="{n}"' if talk else ""
    out = [f"""<section class="page opening" id="{pid('opening')}"{attrs}>
  <div class="kicker">{BRAND} · 第 {n} 章 · {ch['title']}</div>
  <p class="hook">{ch['hook']}</p>
  <p class="lead">{ch['lead']}</p>
  <nav class="opening-route" aria-label="本章路線">{route}</nav>
  <ol>{lis}</ol>
  <div class="meta"><span>{ch['meta']}</span><span>{len(cards)} 張報告卡 · 推導見「推導 ▸」</span></div>
</section>"""]
    for i, c in enumerate(cards):
        sec = offset + i
        if i + 1 < len(cards):
            nxt, href = cards[i + 1]["h"], f'#{pid(cards[i + 1]["id"])}'
        elif ch_next:
            nxt = f'第 {ch_next["n"]} 章　{ch_next["title"]}'
            href = f'#c{ch_next["n"]:02d}-opening' if talk else f'ch{ch_next["n"]:02d}.html#opening'
        else:
            nxt, href = None, None
        bridge = f'<span class="bridge"><a href="{href}">接續：{nxt}</a><span class="bridge-text">接續：{nxt}</span></span>' if nxt else ""
        lim, go, stuck = c["cause"]
        cardattrs = (f' data-chapter="{n}"' if talk else "") + f' data-sec="{sec}"'
        out.append(f"""<section class="page card" id="{pid(c['id'])}"{cardattrs}>
  <div class="report">
    <div>
      <div class="num"><a class="trace" href="#{pid(f'opening-q{i+1}')}">第 {n} 章 · 問題 {i+1}／{len(cards)}</a></div>
      <h3>{c['h']}</h3>
      <div class="formula">{c['formula']}</div>
      <ul>{''.join(f'<li>{b}</li>' for b in c['bullets'])}</ul>
    </div>
    {c['fig']}
    <div class="cause"><span class="pair"><span class="k">問題</span><b>{lim}</b></span><span class="arrow">→</span><span class="pair"><span class="k go">作法</span><b>{go}</b></span><span class="arrow">→</span><span class="pair"><span class="k stuck">限制</span><b>{stuck}</b></span>{bridge}</div>
  </div>
  <button class="deriv-open" data-sec="{sec}" aria-haspopup="dialog">推導 ▸</button>
</section>
<section class="lesson" id="{pid(f'sec-{i}')}" data-title="{c['sec']}"{f' data-chapter="{n}" data-local="{i}" hidden' if talk else ''}>
  <h3>{c['sec']}</h3>
  <p class="proves">對應報告卡 {i+1}：{c['h']}</p>
  {c['body']}
  <p class="takeaway">{c['take']}</p>
</section>""")
    return "\n".join(out)


def recap(ch):
    refs = "".join(f"<li>{r}</li>" for r in ch["refs"])
    return f"""<section class="after" id="recap">
  <h3>本章結論</h3><p>{ch['conclusion']}</p>
  <div class="next"><b>待解問題　</b>{ch['next_q']}</div>
  <section class="refs"><h3>參考文獻</h3><ol>{refs}</ol></section>
</section>"""


DLG = """<dialog class="lesson-dlg" id="dlg" aria-labelledby="dlg-t">
  <div class="dlg-head"><span class="k" id="dlg-k">{k}</span><span class="t" id="dlg-t"></span><button class="x" aria-label="關閉">✕</button></div>
  <div class="dlg-body"></div>
  <div class="dlg-foot"><button class="prev">‹ 上一節</button><button class="nextb">下一節 ›</button><button class="whole">閱讀全章</button></div>
</dialog>"""


def chapter_page(ch):
    n = ch["n"]
    prev_c = next((c for c in CHAPTERS if c["n"] == n - 1), None)
    next_c = next((c for c in CHAPTERS if c["n"] == n + 1), None)
    nav = ""
    if prev_c:
        nav += f'<a class="prev" href="ch{prev_c["n"]:02d}.html">‹ 第 {prev_c["n"]} 章　{prev_c["title"]}</a>'
    if next_c:
        nav += f'<a class="next" href="ch{next_c["n"]:02d}.html">第 {next_c["n"]} 章　{next_c["title"]} ›</a>'
    data = f'data-chapter="{n}"' + (f' data-prev="ch{prev_c["n"]:02d}.html"' if prev_c else "") + (f' data-next="ch{next_c["n"]:02d}.html"' if next_c else "")
    return page_head(f"{BRAND} 第 {n} 章：{ch['title']}") + f"""<body class="m-read" {data}>
<header class="top"><h1><button type="button" class="ch-menu-btn" aria-haspopup="dialog" aria-controls="ch-menu" aria-expanded="false">{BRAND} · 第 {n} 章 {ch['title']} <span class="ch-menu-caret" aria-hidden="true">▾</span></button></h1><div class="modes" role="group" aria-label="模式">
    <button data-mode="present" aria-pressed="false">簡報</button>
    <button data-mode="read" aria-pressed="true">閱讀</button>
    <button data-mode="notes" aria-pressed="false">筆記</button>
  </div></header>
{menu(n)}
<main>
{build_chapter_body(ch)}
{recap(ch)}
<nav class="chapter-nav" aria-label="上一章／下一章">{nav}</nav>
</main>
{DLG.format(k=f'第 {n} 章 · 推導')}
<nav class="pager" aria-label="頁面導覽">
  <button type="button" class="pg-prev" aria-label="上一頁">‹</button>
  <span id="pg" title="← → 換頁 · D 推導 · Esc 離開">{n} · 1／{ch['total']}</span>
  <button type="button" class="pg-next" aria-label="下一頁">›</button>
  <span class="pg-hint" aria-live="polite"></span>
</nav>
<script src="assets/core.js"></script>
<script src="assets/report.js"></script>
<script src="assets/modes.js"></script>
</body>
</html>
"""


LEAD = "只憑 5–20 局落子，預測一位未知玩家屬於 10 個有序等級中的哪一級。本報告沿因果鏈說明每一步解決了什麼、代價是什麼，也記錄沒有通過驗證的方向。"
TITLE_MAIN = "從落子推論棋手的棋力"
INDEX_SUMMARY = table(["章", "研究問題", "報告卡結論", "資料"], [[f'{c["n"]} {c["title"]}', re.sub("<br>", "", c["hook"]), "；".join(x["h"] for x in c["cards"]), c["meta"]] for c in CHAPTERS], num_from=9)


def index_page():
    lst = "".join(f'<li><span class="n">第 {c["n"]} 章</span><a href="ch{c["n"]:02d}.html">{c["title"]}</a><span class="c">{len(c["cards"])} 張報告卡</span></li>' for c in CHAPTERS)
    return page_head("AICUP 棋力預測架構") + f"""<body class="m-read">
<header class="top"><h1><button type="button" class="ch-menu-btn" aria-haspopup="dialog" aria-controls="ch-menu" aria-expanded="false">{BRAND} · 章節架構 <span class="ch-menu-caret" aria-hidden="true">▾</span></button></h1></header>
{menu(0)}
<main>
<section class="index-hero rep-hero" aria-label="封面">
  <p class="kicker">AI CUP 2026 秋季賽 · 圍棋棋力預測</p>
  <h2>{TITLE_MAIN}</h2>
  <p class="paper">{LEAD}</p>
  <p class="cv-date">{DATE}　·　離線分數，測試集 11/04 公布</p>
  <p class="cv-cta"><a class="cv-go" href="talk.html">播放整份報告（{TALK_PAGES} 頁）</a><a href="ch01.html">從第 1 章讀起</a></p>
</section>
<div class="prose rep-result">
<h2>目前結果</h2>
<p>提交候選 fused_opp_diff 在 folds 1–4 CV 是 <strong>{num(score(ERR,'mlp_diff'))}</strong>，在 fold 0 採用檢查是 <strong>{num(score(DIFF0,F0D_KEY))}</strong>，同一組 fold 0 題上官方 baseline 是 0.440；再加讓子 5 維的候選版 CV {num(score(HD,'fused_opp_diff_hd'))}，還沒做 fold 0 檢查。兩次真正的跳升都來自訓練每手證據模型與增加證據來源，換聚合器與最後一層都沒有明顯增益。</p>
<div class="fig">{chart_progress()}</div>
<p class="note">這些都是離線題上的分數，不能直接和排行榜比；官方 baseline 在 Public 是 0.4631。</p>
</div>
<h3>章節</h3>
<ul class="index-list">{lst}</ul>
<div class="prose"><h2>章節架構</h2>
<p>各章以一個研究問題開場，以報告卡回答；每張卡的「推導 ▸」開啟對應一節的完整推導，每張卡的「限制」就是下一張卡要回答的問題。</p>
{INDEX_SUMMARY}
<h2>閱讀方式</h2>
<ul>
<li><strong>閱讀</strong>（預設）：開場 → 報告卡 → 本章結論 → 參考文獻。</li>
<li><strong>簡報</strong>：一次一頁，←／→ 或空白鍵換頁，<code>D</code> 開啟推導，<code>Esc</code> 離開。</li>
<li><strong>筆記</strong>：整章依序展開，每張卡之後緊接其推導。</li>
<li><strong>整份報告</strong>：<code>talk.html</code> 把封面、導讀與 {len(CHAPTERS)} 章串成 {TALK_PAGES} 頁的簡報。</li>
</ul>
</div>
</main>
<script src="assets/modes.js"></script>
</body>
</html>
"""


def talk_page():
    body, off = [], 0
    route = "".join(f'<li><span class="cv-no"><span class="vh">第 </span>{c["n"]}<span class="vh"> 章</span></span><a class="cv-ch" href="#c{c["n"]:02d}-opening">{c["title"]}</a><span class="cv-q">{re.sub("<br>", "", c["hook"])}</span></li>' for c in CHAPTERS)
    body.append(f"""<section class="page cover cur" id="cover">
  <figure class="cv-art rep-art">{board_svg('6d', 236)}</figure>
  <div class="cv-head">
    <h2 class="cv-title" style="--title-len:10"><span class="cv-brand">{BRAND}</span><span class="vh">：</span><span class="cv-main">{TITLE_MAIN}</span></h2>
    <p class="cv-sub">AI CUP 2026 秋季賽 · 圍棋棋力預測 · 研究報告</p>
    <p class="lead">{LEAD}</p>
    <p class="cv-date">{DATE}</p>
  </div>
</section>
<section class="page route" id="route" data-label="導讀">
  <div class="cv-head"><h2 class="rt-title">導讀：{len(CHAPTERS)} 章的因果鏈</h2><p class="lead">提交候選 fused_opp_diff CV {num(score(ERR,'mlp_diff'))}、fold 0 {num(score(DIFF0,F0D_KEY))}。以下依問題的先後說明它怎麼來。</p></div>
  <ol class="cv-route" aria-label="章節路線">{route}</ol>
</section>""")
    plist = ['<section class="page-list-group" aria-label="封面"><h3>封面</h3><ol><li><button type="button" class="page-list-item" data-page="0"><span class="page-list-number">1</span><span>' + TITLE_MAIN + '</span></button></li><li><button type="button" class="page-list-item" data-page="1"><span class="page-list-number">2</span><span>導讀</span></button></li></ol></section>']
    page = 2
    for c in CHAPTERS:
        body.append(build_chapter_body(c, off, f'c{c["n"]:02d}-', True))
        off += len(c["cards"])
        items = [f'<li><button type="button" class="page-list-item" data-page="{page}"><span class="page-list-number">{page+1}</span><span>開場 · {c["title"]}</span></button></li>']
        page += 1
        for x in c["cards"]:
            items.append(f'<li><button type="button" class="page-list-item" data-page="{page}"><span class="page-list-number">{page+1}</span><span>{x["h"]}</span></button></li>')
            page += 1
        plist.append(f'<section class="page-list-group" aria-label="第 {c["n"]} 章"><h3>第 {c["n"]} 章</h3><ol>{"".join(items)}</ol></section>')
    assert page == TALK_PAGES
    return page_head(f"{BRAND}：{TITLE_MAIN}") + f"""<body class="m-present" data-talk="1">
<header class="top"><h1><button type="button" class="ch-menu-btn" aria-haspopup="dialog" aria-controls="page-list" aria-expanded="false">{BRAND}：{TITLE_MAIN} <span class="ch-menu-caret" aria-hidden="true">▾</span></button></h1></header>
<main>
{chr(10).join(body)}
</main>
{DLG.format(k='推導')}
<nav class="pager" aria-label="頁面導覽">
  <button type="button" class="pg-prev" disabled aria-label="上一頁">‹</button>
  <button type="button" id="pg" class="pg-list" aria-haspopup="dialog" aria-expanded="false" title="← → 換頁 · D 推導 · Esc 離開 · G 目錄">封面 · 1／{TALK_PAGES}</button>
  <button type="button" class="pg-next" aria-label="下一頁">›</button>
  <span class="pg-hint" aria-live="polite"></span>
</nav>
<dialog class="page-list-dlg" id="page-list" aria-label="目錄">
  <div class="page-list-head"><h2>目錄 <span>{TALK_PAGES} 頁</span></h2><button type="button" class="page-list-close" aria-label="關閉目錄">✕</button></div>
  <div class="page-list-body">{''.join(plist)}</div>
</dialog>
<script src="assets/core.js"></script>
<script src="assets/report.js"></script>
<script src="assets/modes.js"></script>
</body>
</html>
"""


CSS = """/* Additions for the rank-prediction report; everything else is the research notes' shared style.css. */
.strip{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:2px}
.strip button{border:0;padding:8px 1px 6px;border-radius:4px;cursor:pointer;font:inherit;color:var(--ink);display:grid;justify-items:center;gap:0;
  background:color-mix(in srgb,var(--accent) calc(var(--score,0) * 85%),var(--deriv))}
.strip button[aria-pressed="true"]{box-shadow:inset 0 0 0 2px var(--ink)}
.strip button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.strip .lv{font-weight:700;font-size:14px}.strip .rg{font-size:10px;color:var(--muted);white-space:nowrap}.strip .sc{font-size:12px;font-variant-numeric:tabular-nums}
.strip button.hi,.strip button.hi .rg{color:#fff}
.strip-note{font-size:13px;color:var(--muted);margin:8px 0 0}
.boards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.boards,.strip-wrap,.fig>.stage>.table-wrap{width:100%;min-width:0}
.board-cell{min-width:0;display:grid;gap:4px}.board-cell p{margin:0;font-size:12px;color:var(--muted);text-align:center}
.fig .table-wrap table{font-size:13.5px;width:100%;border-collapse:collapse}
.fig .table-wrap th,.fig .table-wrap td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
.fig .table-wrap th{font-size:12.5px;background:var(--accent-soft)}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.hl td{font-weight:700}
.lesson .fig{margin:10px 0}
.rep-hero h2{text-wrap:balance}
.rep-result{margin-top:18px}
.rep-art svg{display:block;width:100%;height:auto;max-height:100%}
/* talk cover: title block beside the board, no bleed banner */
body.m-present .cover.cur{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);grid-template-rows:minmax(0,1fr);grid-template-areas:"head art";column-gap:3em}
.cover .rep-art{margin:0;display:flex;align-items:center;justify-content:center;min-height:0}
body.m-present .cover .rep-art svg{width:auto;height:calc(var(--stage-w) * .38);max-width:100%}
@media (min-width:761px){
  /* present: the figure gets the wider column; a figure and its caption stay together instead of the caption sinking to the foot */
  body.m-present .report{grid-template-columns:minmax(0,.92fr) minmax(0,1.08fr)}
  body.m-present .report .fig:has(.sources)>.stage{flex:0 1 auto;display:block}
  body.m-present .report .fig:has(.sources)>.stage>svg{height:auto;max-height:calc(var(--stage-w) * .36);width:100%}
  body.m-present .report .fig:has(.sources) .boards svg{height:auto;max-height:calc(var(--stage-w) * .27)}
}
.after>p{margin:8px 0}
"""
JS = """/* The rank strip of chapter 1: pick the true level, read what each prediction scores. */
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
/* Present mode: shrink a card's text column until it clears the row below, whatever font the viewer has. */
(function () {
  function fit() {
    document.querySelectorAll('.report>div:first-child').forEach(e => { e.style.fontSize = ''; });
    const left = document.querySelector('.page.card.cur .report>div:first-child');
    if (!left || !document.body.classList.contains('m-present')) return;
    for (let s = 97; s >= 76 && left.scrollHeight > left.clientHeight + 1; s -= 3) left.style.fontSize = s + '%';
  }
  const watch = new MutationObserver(fit);
  watch.observe(document.querySelector('main'), { subtree: true, attributes: true, attributeFilter: ['class'] });
  watch.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  window.addEventListener('resize', fit);
  document.fonts?.ready.then(fit);
  fit();
})();
"""

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "assets").mkdir(exist_ok=True)
for a in HERE.joinpath("assets").iterdir():
    (OUT / "assets" / a.name).write_bytes(a.read_bytes())
(OUT / "assets/report.css").write_text(CSS)
(OUT / "assets/report.js").write_text(JS)
(OUT / "index.html").write_text(index_page())
(OUT / "talk.html").write_text(talk_page())
for ch in CHAPTERS:
    (OUT / f"ch{ch['n']:02d}.html").write_text(chapter_page(ch))
print("built", len(CHAPTERS), "chapters,", sum(len(c["cards"]) for c in CHAPTERS), "cards,", TALK_PAGES, "talk pages")
