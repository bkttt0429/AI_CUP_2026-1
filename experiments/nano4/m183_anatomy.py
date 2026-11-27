#!/usr/bin/env python
"""Section 4M M183 (2026-11-16, report only): anatomy of what is still lost under the ct112 rule, from m183_dump.npz
(cross-validation, seeds 0-2; per question the base posterior, the anchor evidence, the pooled posterior and the decisions).

All rates are class-balanced inside each size (weight 1 / number of questions of that size and true level), sizes weighted
equally - the convention of the cross-validation score. Score of an answer: 1 exact, exp(-1) one level off, 0 otherwise.

A  what is lost: exact / one level off / two or more, by size; the 1/n extrapolation of the score; by true level.
B  the two channels separately - own play (base decision) and opponents (argmax of the anchor evidence, questions with at
   least K games carrying anchor evidence of a factor >= 2): how often each is right, how often they agree, and how often the
   label is the level they agree on; the loss of the final answer split by what the channels said.
C  is the pooled posterior honest: the score it expects of its own answer against the score obtained, by confidence quarter;
   the score if the truth were always the better of its two first choices.
D  dose-response: score against the number of games with anchor evidence, by size (base alone and final).
E  the continuous readings: expected level of the base and of the anchors minus the label - spread, correlation between the
   channels, and the part of the spread that does not shrink with the number of games.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; E1 = float(np.exp(-1.0))


def main():
    z = np.load(ROOT / "results/r_variants" / (sys.argv[1] if len(sys.argv) > 1 else "m183_dump.npz"), allow_pickle=True); name = sys.argv[2] if len(sys.argv) > 2 else "SQ4"
    y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); fin, plain, base = z[name].astype(int), z["anat_plain"].astype(int), z["anat_base"].astype(int); post = z["anat_post"]; logp = z["anat_logp"]; L = z["anat_L"]; Lall = z["anat_Lall"]; AN = z["anat_AN"].astype(int)
    sizes = sorted(set(n)); w = np.zeros(len(y))
    for s in sizes:
        for c in range(10):
            m = (n == s) & (y == c); w[m] = 1.0 / max(int(m.sum()), 1) / 10.0 / len(sizes)
    sc = lambda d: np.where(d == y, 1.0, np.where(np.abs(d - y) == 1, E1, 0.0))  # noqa: E731
    wm = lambda x, m=None: float((w * x)[m].sum() / w[m].sum()) if m is not None else float((w * x).sum() / w.sum())  # noqa: E731
    print(f"questions {len(y)}, players {len(set(pid))}; score of the final answers {wm(sc(fin)):.4f} (without the guards {wm(sc(plain)):.4f}, base alone {wm(sc(base)):.4f})")
    # ---------------------------------------------------------------- A
    print("\nA. what is lost (final answers)")
    Sn = {}
    for s in sizes:
        m = n == s; ex, o1 = wm(fin == y, m), wm(np.abs(fin - y) == 1, m); Sn[s] = wm(sc(fin), m); print(f"  {s:2d} games: score {Sn[s]:.4f} | exact {ex:.3f}, one level off {o1:.3f}, two or more {1 - ex - o1:.3f} | of the loss {1 - Sn[s]:.3f}: one level off {o1 * (1 - E1) / (1 - Sn[s]):.2f}, two or more {(1 - ex - o1) / (1 - Sn[s]):.2f}")
    xs = np.array([1.0 / s for s in sizes]); ys_ = np.array([Sn[s] for s in sizes]); slope, icpt = np.polyfit(xs, ys_, 1); tot = float(np.mean(ys_))
    print(f"  score against 1/n: {icpt:.4f} {slope:+.3f}/n -> with unlimited games {icpt:.3f}; the same for the base alone: " + "{:.3f}".format(np.polyfit(xs, [wm(sc(base), n == s) for s in sizes], 1)[1]))
    print(f"  budget of the loss 1 - {tot:.4f} = {1 - tot:.4f}: too few games {icpt - tot:.4f} ({(icpt - tot) / (1 - tot):.0%}), left with unlimited games {1 - icpt:.4f} ({(1 - icpt) / (1 - tot):.0%})")
    print("  by true level (score / exact / mean signed error): " + "  ".join(f"{LV[c]} {wm(sc(fin), y == c):.3f}/{wm(fin == y, y == c):.2f}/{wm(fin - y, y == c):+.2f}" for c in range(10)))
    # ---------------------------------------------------------------- B
    print("\nB. the two channels: own play (base decision b) and opponents (a = level with the most anchor evidence)")
    a1 = L.argmax(1); a5 = Lall.argmax(1); k2 = AN[:, 1]
    for lab, a in (("newest identifier only", a1), ("all six anchor inputs summed", a5)):
        print(f"  anchor reading = {lab}")
        for s in sizes:
            for K in (1, 3, 5, 8):
                m = (n == s) & (k2 >= K)
                if m.sum() < 300:
                    continue
                ag = m & (a == base); print(f"    {s:2d} games, >= {K} games with anchor evidence ({float(m.sum() / (n == s).sum()):.2f} of the questions): right - base {wm(base == y, m):.3f}, anchors {wm(a == y, m):.3f}, final {wm(fin == y, m):.3f}; base and anchors agree {wm(a == base, m):.3f}; "
                                                f"when they agree the label is that level {wm(a == y, ag):.3f} (one level off {wm(np.abs(a - y) == 1, ag):.3f}); when they disagree the label is the base's {wm(base == y, m & ~(a == base)):.3f} / the anchors' {wm(a == y, m & ~(a == base)):.3f} / neither {wm((base != y) & (a != y), m & ~(a == base)):.3f}")
    a = a5; loss = 1.0 - sc(fin); tl = float((w * loss).sum()); cats = [("no game with anchor evidence", k2 == 0), ("1-2 such games", (k2 >= 1) & (k2 <= 2)), ("3 or more: base and anchors agree, label elsewhere", (k2 >= 3) & (a == base) & (base != y)),
                                                                     ("3 or more: they disagree, one of them is right", (k2 >= 3) & (a != base) & ((a == y) | (base == y))), ("3 or more: they disagree, neither is right", (k2 >= 3) & (a != base) & (a != y) & (base != y)),
                                                                     ("3 or more: both right (final answer wrong)", (k2 >= 3) & (a == base) & (base == y))]
    print("  where the loss of the final answers sits (share of the loss | share of the questions | score of the final answers there):")
    for lab, m in cats:
        print(f"    {lab:55s} {float((w * loss)[m].sum() / tl):.3f} | {float(w[m].sum() / w.sum()):.3f} | {wm(sc(fin), m) if m.any() else float('nan'):.3f}")
    # ---------------------------------------------------------------- C
    print("\nC. is the pooled posterior honest (decision without the guards)")
    SMx = np.eye(10) + E1 * (np.eye(10, k=1) + np.eye(10, k=-1)); es = post @ SMx; own = es[np.arange(len(y)), plain]; got = sc(plain); qs = np.quantile(own, [0.25, 0.5, 0.75]); bq = np.digitize(own, qs)
    print(f"  score it expects of its own answers {wm(own):.4f}, obtained {wm(got):.4f}; by confidence quarter (expected / obtained / share of the loss): " + "  ".join(f"{wm(own, bq == k):.3f}/{wm(got, bq == k):.3f}/{float((w * (1 - got))[bq == k].sum() / (w * (1 - got)).sum()):.2f}" for k in range(4)))
    o2 = np.argsort(-post, axis=1)[:, :2]; in2 = (o2[:, 0] == y) | (o2[:, 1] == y); o3 = np.argsort(-post, axis=1)[:, :3]; in3 = (o3 == y[:, None]).any(1)
    print(f"  truth is the first choice {wm(o2[:, 0] == y):.3f}, among the first two {wm(in2):.3f}, the first three {wm(in3):.3f}; score with a perfect pick between the first two {wm(np.where(in2, 1.0, got)):.3f}")
    ent = -(post * np.log(np.maximum(post, 1e-12))).sum(1); print("  mean entropy of the posterior (nats) by size: " + "  ".join(f"{s}: {wm(ent, n == s):.3f}" for s in sizes) + f"; of the base alone: " + "  ".join(f"{s}: {wm(-(np.exp(logp) * logp).sum(1), n == s):.3f}" for s in sizes))
    # ---------------------------------------------------------------- D
    print("\nD. score against the number of games with anchor evidence (factor >= 2): share of the questions | base alone | final")
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 12), (13, 20)]
    for s in sizes:
        row = []
        for lo, hi in bins:
            m = (n == s) & (k2 >= lo) & (k2 <= hi)
            if m.sum() >= 200:
                row.append(f"{lo}-{hi}: {float(m.sum() / (n == s).sum()):.2f} | {wm(sc(base), m):.3f} | {wm(sc(fin), m):.3f}")
        print(f"  {s:2d} games: " + "   ".join(row) + f"   mean number {float(k2[n == s].mean()):.2f} ({float(k2[n == s].mean()) / s:.2f} per game)")
    # ---------------------------------------------------------------- E
    print("\nE. continuous readings minus the label (levels)")
    lev = np.arange(10.0); eb = np.exp(logp) @ lev - y; pa = np.exp(Lall - Lall.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True); ea = pa @ lev - y; ef = post @ lev - y; inner = (y >= 1) & (y <= 8)
    for s in sizes:
        m = (n == s) & inner; m3_ = m & (k2 >= 3)
        print(f"  {s:2d} games (true levels C-5D): sd of base {np.sqrt(wm(eb ** 2, m) - wm(eb, m) ** 2):.3f}, pooled {np.sqrt(wm(ef ** 2, m) - wm(ef, m) ** 2):.3f}; with >= 3 anchor games: base {np.sqrt(wm(eb ** 2, m3_) - wm(eb, m3_) ** 2):.3f}, anchors {np.sqrt(wm(ea ** 2, m3_) - wm(ea, m3_) ** 2):.3f}, correlation of the two {float(np.corrcoef(eb[m3_], ea[m3_])[0, 1]):.3f}")
    vb = [wm(eb ** 2, (n == s) & inner) - wm(eb, (n == s) & inner) ** 2 for s in sizes]; vf = [wm(ef ** 2, (n == s) & inner) - wm(ef, (n == s) & inner) ** 2 for s in sizes]
    for lab, v in (("base", vb), ("pooled", vf)):
        sl, ic = np.polyfit(xs, v, 1); print(f"  variance of ({lab} expected level - label) = {ic:.3f} {sl:+.2f}/n -> the part that more games do not remove: sd {np.sqrt(max(ic, 0)):.3f} levels; per-game noise sd {np.sqrt(max(sl, 0)):.2f}")


if __name__ == "__main__":
    main()
