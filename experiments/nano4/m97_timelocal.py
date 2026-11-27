"""Section 4M M97 (2026-11-13, report only): is a player's paired level locally stable in time, and are the test questions local in
time like the validation questions? M96 showed the label is the centre of the whole period and that a single anchor scatters with sd
1.07 around a player's own consensus. If the paired level moves slowly (a rating walk), anchors from games close in time agree more
than anchors far apart. Validation questions are consecutive games; how the test questions were drawn is not known (their game
order is shuffled).
(A) training pairs of M96 (seen-level identification, top tiers, even games): semi-variance of the opponents' labels of one owner
    as a function of the distance in his game order.
(B) the forward anchors actually used (SP2e identifier, confident bands, even games): semi-variance between two anchors of the same
    question, validation (consecutive games) against test, by question size; and the same for the identified opponents' handicap-
    free base statistic "share of pairs with the same label".
  python -I logs/nano4/m97_timelocal.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
import m86_bundle as b  # noqa: E402
from ens_compare import FOLDS, load_fold  # noqa: E402

RV = ROOT / "results/r_variants"; rng = np.random.default_rng(97)
rows = m81.load_rows(); lab_of = dict(zip(rows.player_id, rows["rank"].map(b.RANK))); players = np.array(sorted(rows.player_id.unique())); ylab = np.array([lab_of[p] for p in players]); del rows
P = np.load(RV / "m96_pairs.npz"); ow, op, tier, pos = P["owner"], P["opp"], P["tier"], P["pos"]; top = tier >= 5; x = ylab[op].astype(float)
print("(A) training pairs, top tiers (>= q95), even games: semi-variance 0.5 * E[(x_i - x_j)^2] of two opponents' labels of the same owner, by distance in the owner's game order (fraction of his games)")
bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]; acc = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1); same = np.zeros(len(bins) - 1); o_sort = np.argsort(ow[top], kind="stable"); o_, x_, p_ = ow[top][o_sort], x[top][o_sort], pos[top][o_sort]; st = np.flatnonzero(np.r_[True, o_[1:] != o_[:-1], True])
for a_, e_ in zip(st[:-1], st[1:]):
    if e_ - a_ >= 2:
        xi, pi = x_[a_:e_], p_[a_:e_]; i, j = np.triu_indices(e_ - a_, 1); h = np.abs(pi[i] - pi[j]); d2 = (xi[i] - xi[j]) ** 2; k = np.digitize(h, bins) - 1; np.add.at(acc, k, d2); np.add.at(cnt, k, 1); np.add.at(same, k, d2 == 0)
for k in range(len(bins) - 1):
    print(f"    distance {bins[k]:.2f}-{min(bins[k + 1], 1.0):.2f}: {int(cnt[k])} pairs, semi-variance {0.5 * acc[k] / max(cnt[k], 1):.3f}, same label {same[k] / max(cnt[k], 1):.3f}")
V, T, key_pid, used = b.sources(["5e", "SP2e"]); v, t = V["SP2e"], T["SP2e"]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); nb = int(vband.max()) + 1
tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"])
qof = {}; qn = {}
for f in FOLDS:
    y, n, pid, q = load_fold(f)
    for i, (g_, n_) in enumerate(zip(q["game_keys"], n)):
        for k_ in g_:
            qof.setdefault((k_, int(n_)), f"{f}:{i}")
tn = dict(zip(*np.unique(t["qid"], return_counts=True)))


def within(groups, xs, label):
    d = defaultdict(list)
    for g_, x1 in zip(groups, xs):
        d[g_].append(x1)
    sv, sm, npair, nq = 0.0, 0.0, 0, 0
    for g_, xv in d.items():
        if len(xv) >= 2:
            xv = np.array(xv, float); i, j = np.triu_indices(len(xv), 1); d2 = (xv[i] - xv[j]) ** 2; sv += d2.sum(); sm += (d2 == 0).sum(); npair += len(d2); nq += 1
    print(f"    {label}: {nq} questions with >= 2 such anchors, {npair} pairs; semi-variance {0.5 * sv / max(npair, 1):.3f}, same label {sm / max(npair, 1):.3f}")
    return 0.5 * sv / max(npair, 1), npair


print(f"(B) forward anchors of the SP2e identifier ({nb} bands), even games: two anchors inside the same question")
for lo, bname in ((nb - 2, "top two bands"), (nb - 4, "top four bands")):
    for size in (10, 20):
        sv = (vg == 0) & (vband >= lo); gv = [qof.get((k_, size)) for k_ in v["key"][sv]]; keep = [g_ is not None for g_ in gv]; a1 = within([g_ for g_, k_ in zip(gv, keep) if k_], v["opp_lab"][sv][keep], f"validation, {size}-game questions (consecutive games), {bname}")
        stt = (tg == 0) & (tb >= lo) & np.array([tn[q_] == size for q_ in t["qid"]]); a2 = within(t["qid"][stt], t["opp_lab"][stt], f"test, {size}-game questions, {bname}")
        # reference: validation anchors of the same players re-paired across DIFFERENT players of the same level would be the "unrelated" limit; here the null of interest is time, so also report the test / validation ratio
        print(f"       test / validation semi-variance ratio: {a2[0] / max(a1[0], 1e-9):.3f}")
    stt = (tg == 0) & (tb >= lo) & np.array([tn[q_] == 15 for q_ in t["qid"]]); within(t["qid"][stt], t["opp_lab"][stt], f"test, 15-game questions, {bname}")
# bootstrap the ratio for the 20-game questions, top four bands
lo = nb - 4; sv = (vg == 0) & (vband >= lo); gv = np.array([qof.get((k_, 20), "") for k_ in v["key"][sv]]); xv_ = v["opp_lab"][sv]; stt = (tg == 0) & (tb >= lo) & np.array([tn[q_] in (10, 15, 20) for q_ in t["qid"]]); gt = t["qid"][stt]; xt_ = t["opp_lab"][stt]


def per_q(groups, xs):
    d = defaultdict(list)
    for g_, x1 in zip(groups, xs):
        if g_ != "":
            d[g_].append(x1)
    out = []
    for xv in d.values():
        if len(xv) >= 2:
            xv = np.array(xv, float); i, j = np.triu_indices(len(xv), 1); out.append(((xv[i] - xv[j]) ** 2).sum() / 2 / len(i))
    return np.array(out)


qv, qt = per_q(gv, xv_), per_q(gt, xt_); bs = [qt[rng.integers(len(qt), size=len(qt))].mean() / qv[rng.integers(len(qv), size=len(qv))].mean() for _ in range(3000)]
print(f"    per-question semi-variance (top four bands; validation 20-game, test 10/15/20-game): validation mean {qv.mean():.3f} ({len(qv)} questions), test mean {qt.mean():.3f} ({len(qt)} questions); ratio {qt.mean() / qv.mean():.3f}, bootstrap 95% [{np.quantile(bs, 0.025):.3f}, {np.quantile(bs, 0.975):.3f}]")
