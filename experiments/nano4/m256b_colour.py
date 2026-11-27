#!/usr/bin/env python
"""M256b (2026-11-20), report only, added after seeing M256 (D): does the anchor reading of a SINGLE-COLOUR set of games carry
a colour-dependent offset? (design doc M256b)

Validation games (identifier SQ2, statistic m15; the production evidence of m103_stability.evidence, cross-fitted by fold):
 1. per game, by group (even / HD1 target black / HD1 target white) and the target's colour: mean (identified opponent's
    label - own label) over the clearly identified games, and the single game's anchor reading (expected level under
    pi x evidence) minus the own label;
 2. per validation player (20 games): the anchor-only expected level of the black subset minus that of the white subset
    (both with >= 2 clear opponents), overall, by the player's level, and for even games only.

  python -I logs/nano4/m256b_colour.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

m3, b, s2, FOLDS = mb7.m3, mb7.b, mb7.s2, mb7.FOLDS; RV = ROOT / "results/r_variants"; STAT, X = "m15", "SQ2"; LV = np.arange(10.0)


def soft(L):
    L = L - L.max(-1, keepdims=True); P = np.exp(L); return P / P.sum(-1, keepdims=True)


def main():
    val = json.load(open(RV / "m255_val.json")); xscale = val["density_ratio"]; lpi = np.log(np.array(val["seeds_0"]["pi"])); name = f"{X}~{STAT}"; V, T, key_pid, used = b.sources([name]); z = V[name]
    key = z["key"].astype(str); pid = z["pid"].astype(str); fold = z["fold"].astype(int); y = z["y"].astype(int); hd = z["hd"].astype(int); tb = z["target_black"].astype(bool); lab = z["opp_lab"].astype(int); m = z["opp_m"].astype(float); thr = float(np.quantile(z["imp_m"], 0.95)); clear = m > thr
    E = np.zeros((len(key), 10)); row = {k: i for i, k in enumerate(key)}
    for v in FOLDS:
        for k, e in m3.evidence(z, v, 1, 1, 1, xscale).items():
            E[row[str(k)]] = e
    g = np.asarray(s2.group_of(hd, tb)); read = soft(lpi[None] + E) @ LV; rep = {}
    print(f"validation games {len(key)}; clearly identified (m15 above the impostor q95 = {thr:.2f}) {clear.mean():.3f}; black share {tb.mean():.3f}; HD1 share {float((hd > 0).mean()):.3f}")
    print("\n1. per game: group x target colour | games | clear | mean (opponent label - own label) over the clear games (se) | single-game anchor reading - own label, clear games")
    for k, gn in ((0, "even"), (1, "HD1, target black"), (2, "HD1, target white")):
        for c, cn in ((True, "black"), (False, "white")):
            s = (g == k) & (tb == c)
            if s.sum() < 50:
                continue
            sc = s & clear; d = (lab - y)[sc]; r_ = (read - y)[sc]; rep[f"{gn}|{cn}"] = {"n": int(s.sum()), "clear": int(sc.sum()), "opp_minus_own": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))), "reading_minus_own": float(r_.mean())}
            print(f"   {gn:18s} | target {cn}: {int(s.sum()):6d} | {int(sc.sum()):6d} | {d.mean():+.3f} ({d.std() / np.sqrt(len(d)):.3f}) | {r_.mean():+.3f}")
    ev = (g == 0) & clear; db = (lab - y)[ev & tb].mean() - (lab - y)[ev & ~tb].mean(); print(f"   even games, black minus white: opponent label - own label {db:+.3f}; by the own level: " + " ".join(f"{c}: {(lab - y)[ev & tb & (y == c)].mean() - (lab - y)[ev & ~tb & (y == c)].mean():+.2f}" for c in range(10)))
    rep["even_black_minus_white"] = float(db)
    # 2. per player: black subset minus white subset
    order = np.argsort(pid, kind="stable"); up, start = np.unique(pid[order], return_index=True); out = []
    for a_, e_ in zip(start, list(start[1:]) + [len(order)]):
        ix = order[a_:e_]
        for tag, sel in (("all", np.ones(len(ix), bool)), ("even", g[ix] == 0)):
            bi, wi = ix[sel & tb[ix]], ix[sel & ~tb[ix]]
            if clear[bi].sum() >= 2 and clear[wi].sum() >= 2:
                out.append((tag, int(y[ix[0]]), float(soft(lpi + E[bi].sum(0)) @ LV), float(soft(lpi + E[wi].sum(0)) @ LV), len(bi), len(wi)))
    print("\n2. per player: anchor-only expected level of the BLACK games minus that of the WHITE games (both subsets with >= 2 clear opponents)")
    for tag in ("all", "even"):
        R = np.array([(o[1], o[2], o[3]) for o in out if o[0] == tag]); d = R[:, 1] - R[:, 2]; rep[f"player_{tag}"] = {"n": int(len(d)), "mean": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))), "median": float(np.median(d)), "black_minus_truth": float((R[:, 1] - R[:, 0]).mean()), "white_minus_truth": float((R[:, 2] - R[:, 0]).mean())}
        print(f"   {tag:4s} games: {len(d)} players; black - white mean {d.mean():+.3f} (se {d.std() / np.sqrt(len(d)):.3f}), median {np.median(d):+.3f}; black subset - truth {float((R[:, 1] - R[:, 0]).mean()):+.3f}, white subset - truth {float((R[:, 2] - R[:, 0]).mean()):+.3f}; by level: " + " ".join(f"{c}: {d[R[:, 0] == c].mean():+.2f}" for c in range(10) if (R[:, 0] == c).sum() >= 20))
    (RV / "m256b_colour.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m256b_colour.json")


if __name__ == "__main__":
    main()
