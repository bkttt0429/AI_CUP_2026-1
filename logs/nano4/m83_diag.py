"""Section 4M M83 diagnostics (2026-11-11, report only; after ct42 0.7880062 / ct44 0.8067107 on Public).
 1 exact duplicates: the same move sequence (>= 30 moves) in two training rows, in a validation question's game and another player's row,
   in a test game and a training row, in two test questions
 2 what the anchors move in validation at the test's density (0.851, weights x0.851), decision inside the boundary region as in ct44:
   per direction (from -> to) how many, how often right (truth = to) / broke an exact answer (truth = from), mean gain; per true class the gain
  python -I logs/nano4/m83_diag.py
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced  # noqa: E402
import m71_cv_density as cd  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m81_tactic_id as m81  # noqa: E402

MOVE = re.compile(r";[BW]\[[a-s]{0,2}\]")


def gh(s):
    m = MOVE.findall(s)
    return hashlib.md5("".join(m).encode()).hexdigest()[:16] if len(m) >= 30 else None


rows = m81.load_rows(); rows["h"] = [gh(s) for s in rows.sgf_content]; ok = rows[rows.h.notna()]
cnt = ok.groupby("h").size(); dup = ok[ok.h.map(cnt) > 1]; g = dup.groupby("h").agg(players=("player_id", "nunique"), colours=("color", "nunique"), n=("player_id", "size"))
print(f"training rows: {len(rows)} ({len(ok)} with >= 30 moves); rows whose move sequence appears in another row: {len(dup)} in {len(g)} groups; groups with two different players {int((g.players > 1).sum())} (opposite colours {int(((g.players > 1) & (g.colours > 1)).sum())}), same player twice {int((g.players == 1).sum())}")
used = set()
for v in (1, 2, 3, 4):
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
    for ng, gk in zip(m["n_games"], m["game_keys"]):
        if ng == 20:
            used.update(gk)
q = ok[ok.game_key.isin(used)]; other = dup[dup.h.isin(set(q.h))].groupby("h").player_id.nunique()
print(f"validation question games: {len(q)}; with the same game in ANOTHER player's training row: {int(q.h.map(other).fillna(1).gt(1).sum())}")
tg = m81.load_tests(); th = [gh(g_[0]) for _, g_ in tg]; by_h = ok.groupby("h").indices; hit = [(j, by_h[h_]) for j, h_ in enumerate(th) if h_ is not None and h_ in by_h]
print(f"test games: {len(tg)}; with the same move sequence in a training row: {len(hit)}")
if hit:
    T = np.load(ROOT / "results/r_variants/m82_test.npz"); col = ok.color.to_numpy(); pid = ok.player_id.to_numpy(); rk = ok["rank"].to_numpy(); opp = same = agree = 0; labs = Counter()
    for j, idx in hit:
        for i in idx:
            if col[i] != tg[j][1][1]:
                opp += 1; labs[rk[i]] += 1; agree += int(T["f5e_opp_best"][j] == pid[i])
            else:
                same += 1
    print(f"   training row is the OPPONENT's record (opposite colour): {opp} (the identifier's top candidate is that player in {agree}); same colour: {same}; opponent labels: {dict(labs)}")
tq = Counter(); first = {}
for (qid, _), h_ in zip(tg, th):
    if h_ is not None:
        if h_ in first and first[h_] != qid:
            tq[h_] += 1
        first.setdefault(h_, qid)
print(f"test games whose move sequence appears in a different test question: {sum(tq.values())}")

d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); base = {k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; R = 0.851
for name, pre in (("five + matcher (ct44's identifier)", "f5e_"), ("five (ct42's identifier)", "f5_")):
    v = {**base, "opp_lab": d[pre + "opp_lab"], "opp_m": d[pre + "opp_m"], "imp_lab": d[pre + "imp_lab"], "imp_m": d[pre + "imp_m"]}
    ys, ns, a0, a1 = [], [], [], []
    for draw in (1, 2, 3):
        for yy, nn, pp, P, L in mb.fold_data(v, cd.thin(v, R, draw), R, (0, 1, 2)):
            d0, _, dAr, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); ys.append(yy); ns.append(nn); a0.append(d0); a1.append(dAr)
    y, n, d0, d1 = map(np.concatenate, (ys, ns, a0, a1)); gain = mb.S[d1, y] - mb.S[d0, y]; ch = d0 != d1
    print(f"{name}: changed {ch.mean():.3f} of the questions; right {float((y[ch] == d1[ch]).mean()):.3f}, broke an exact answer {float((y[ch] == d0[ch]).mean()):.3f}, mean gain per change {gain[ch].mean():+.3f}; up moves {int((d1 > d0).sum())}, down moves {int((d1 < d0).sum())}")
    tab = Counter(zip(d0[ch], d1[ch])); print("   direction: count, right, broke, mean gain")
    for (a, b), c in sorted(tab.items(), key=lambda kv: -kv[1])[:18]:
        s = ch & (d0 == a) & (d1 == b); print(f"   {LEVELS[a]:>2s}->{LEVELS[b]:<2s} {c:5d}  {float((y[s] == b).mean()):.3f}  {float((y[s] == a).mean()):.3f}  {gain[s].mean():+.3f}")
    print("   gain by true class (class mean over the three sizes): " + " ".join(f"{LEVELS[c]} {np.mean([gain[(y == c) & (n == m)].mean() for m in SIZES]):+.4f}" for c in range(10)) + f" | balanced {np.mean([balanced(gain, y, n == m) for m in SIZES]):+.4f}")
