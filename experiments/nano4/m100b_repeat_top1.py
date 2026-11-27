"""Section 4M M100 stage 0b (2026-11-13, report only): rematches inside a question, measured directly on the identifier's output.
M100's cosine test could not separate "the same opponent twice" from "two opponents of the same level" (opponents inside a question
share the level). Here: how often do two games of the same question get the SAME first candidate from the SP2e identifier, against
the same count after the games are shuffled between questions of the same level (validation: true level; test: ct75's answer).
A repeated first candidate that is far above the shuffled rate is a rematch (or the same look-alike twice, which is just as
dependent). Also: how the margins of such pairs compare, and what share of the confident anchors are in such pairs (the current
evidence counts them as independent anchors).
  python -I logs/nano4/m100b_repeat_top1.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m71_step2 as s2  # noqa: E402
import m86_bundle as b  # noqa: E402
from ens_compare import FOLDS, load_fold  # noqa: E402

RV = ROOT / "results/r_variants"; rng = np.random.default_rng(1002)
V, T, key_pid, used = b.sources(["5e", "SP2e"]); v, t = V["SP2e"], T["SP2e"]; dv = np.load(RV / "m84_SP2e_val.npz"); dt = np.load(RV / "m84_SP2e_test.npz"); bv, bt = dv["f_opp_best"].astype(str), dt["f_opp_best"].astype(str)
ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"]))


def stats(groups, best, band, level, label):
    """groups: question of every game; level: level of the question's player (for the shuffle)."""
    d = defaultdict(list)
    for i, g_ in enumerate(groups):
        d[g_].append(i)

    def count(best_):
        pairs = games = conf = conf_rep = 0; n_games = 0
        for ii in d.values():
            c = Counter(best_[ii]); rep = {k for k, x in c.items() if x >= 2}; pairs += sum(x * (x - 1) // 2 for x in c.values()); n_games += len(ii)
            for i in ii:
                r_ = best_[i] in rep; games += r_
                if band[i] >= 4:
                    conf += 1; conf_rep += r_
        return pairs, games / n_games, conf, conf_rep
    p, g_share, conf, conf_rep = count(best); idx = np.arange(len(best)); sh = []
    for _ in range(20):
        perm = idx.copy()
        for lv in np.unique(level):
            s = np.flatnonzero(level == lv); perm[s] = s[rng.permutation(len(s))]
        sh.append(count(best[perm])[1])  # the candidates travel, the question structure stays
    n_pairs = sum(len(ii) * (len(ii) - 1) // 2 for ii in d.values())
    print(f"{label}: {len(d)} questions, {len(best)} games; games whose first candidate is also the first candidate of another game of the question: {g_share:.4f} (same-level shuffle {np.mean(sh):.4f} +- {np.std(sh):.4f}); repeated pairs {p} of {n_pairs} ({p / n_pairs:.5f}); among the confident anchors (margin >= q95, {conf}): {conf_rep / max(conf, 1):.4f} are in such a pair", flush=True)


qof = {}; qlev = {}
for f in FOLDS:
    y, n, pid, q = load_fold(f)
    for i, (g_, n_, y_) in enumerate(zip(q["game_keys"], n, y)):
        for k_ in g_:
            qof[(k_, int(n_))] = f"{f}:{i}"; qlev[f"{f}:{i}"] = int(y_)
for size in (5, 10, 20):
    gv = np.array([qof.get((k_, size), "") for k_ in v["key"]]); s = gv != ""; stats(gv[s], bv[s], vband[s], np.array([qlev[x] for x in gv[s]]), f"validation, {size}-game questions (consecutive games)")
c75 = pd.read_csv(ROOT / "submission_rank_ct75.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nq = dict(zip(*np.unique(t["qid"], return_counts=True))); lev_t = c75.reindex(t["qid"]).to_numpy()
for size in (5, 10, 15, 20):
    s = np.array([nq[x] == size for x in t["qid"]]); stats(t["qid"][s], bt[s], tb[s], lev_t[s], f"test, {size}-game questions")
