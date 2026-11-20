"""Section 4M M100 stage 0 (2026-11-13, report only, no level labels): does the same opponent appear more than once inside a
question? Identification of a single game is right 40% of the time; with two games of the same person it would be far better
(diarisation: cluster first, identify the cluster). Measured with the M90 vectors of the opponent side:
  reference "same person": two different games of the same training player (own side), players the network did not see;
  reference "different persons": own-side games of two different unseen players of the same level;
  inside a question: all pairs of opponent-side vectors of the question's games, validation (fold v under network v, the
  20-game questions) and test (network -1), against the pairs of opponent-side vectors from DIFFERENT questions.
Cosine of the three-seed mean of unit vectors per network. Reported: quantiles of each distribution, the share of within-question
pairs above the 99.9th percentile of the between-question pairs, and the implied share of games that have a rematch partner.
  python -I logs/nano4/m100_rematch.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m91_mirror as mm  # noqa: E402
from ens_compare import FOLDS, load_fold  # noqa: E402

TEST = ROOT / "results/ct_submission/vec"; rng = np.random.default_rng(100)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def load(net, name):
    return unit(np.mean([unit(np.load(mm.EMB / f"emb_{net}{f'_s{s}' if s else ''}.npz")[name].astype(np.float32)) for s in (0, 1, 2)], axis=0))


def within(vecs, groups):
    d = defaultdict(list)
    for i, g_ in enumerate(groups):
        d[g_].append(i)
    out = []; sizes = []
    for g_, ii in d.items():
        if len(ii) >= 2:
            v = vecs[ii]; s = v @ v.T; iu = np.triu_indices(len(ii), 1); out.append(s[iu]); sizes.append(len(ii))
    return np.concatenate(out), d


def between(vecs, groups, n=400_000):
    i = rng.integers(len(vecs), size=n); j = rng.integers(len(vecs), size=n); k = np.array(groups)[i] != np.array(groups)[j]
    return np.einsum("ij,ij->i", vecs[i[k]], vecs[j[k]])


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    D = mm.load_meta(log); kidx = {k: i for i, k in enumerate(D["key"])}; qs = lambda x: " ".join(f"{v:.3f}" for v in np.quantile(x, [0.5, 0.9, 0.99, 0.999]))  # noqa: E731
    for v in (1, 2):
        own, opp = load(v, "own"), load(v, "opp"); fv = np.flatnonzero(D["fold"] == v); pl = D["owner"][fv]; same, _ = within(own[fv[:60000]], pl[:60000]); same = rng.choice(same, min(len(same), 400_000), replace=False); diff = between(own[fv], list(pl))
        y, n, pid, q = load_fold(v); rows, grp = [], []
        for i, (g_, n_) in enumerate(zip(q["game_keys"], n)):
            if n_ == 20:
                r = [kidx[k] for k in g_ if k in kidx]; rows += r; grp += [i] * len(r)
        rows = np.array(rows); w, d = within(opp[rows], grp); bt = between(opp[rows], grp); thr = np.quantile(bt, 0.999); thr_s = np.quantile(diff, 0.999); det = float((same > thr_s).mean())
        log(f"network {v}: cosine quantiles 0.5 / 0.9 / 0.99 / 0.999 | same unseen player, two own games: {qs(same)} | two different unseen players: {qs(diff)} | share of same-player pairs above the different-player 99.9th percentile ({thr_s:.3f}): {det:.3f}")
        ex = float((w > thr).mean()); n_pairs = len(w); log(f"   validation 20-game questions ({len(d)}): opponent sides of two games of the SAME question: {qs(w)} | of DIFFERENT questions: {qs(bt)} | within-question pairs above the between-question 99.9th percentile ({thr:.3f}): {ex:.4f} of {n_pairs} pairs (nominal 0.0010) -> rematch pairs about {(ex - 0.001) / max(det, 1e-9):.4f} of the pairs, i.e. about {(ex - 0.001) / max(det, 1e-9) * 19:.3f} rematch partners per game of a 20-game question")
        if v == 1:
            det1, thr_keep = det, thr_s
    topp = load(-1, "t_opp"); tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); tq = tab["question_id"].to_numpy(); nq = dict(zip(*np.unique(tq, return_counts=True)))
    for size in (5, 10, 15, 20):
        s = np.array([nq[x] == size for x in tq]); w, d = within(topp[s], list(tq[s])); bt = between(topp[s], list(tq[s])); thr = np.quantile(bt, 0.999); ex = float((w > thr).mean())
        log(f"test, {size}-game questions ({len(d)}), all-players network: same question: {qs(w)} | different questions: {qs(bt)} | within-question pairs above the between-question 99.9th percentile ({thr:.3f}): {ex:.4f} of {len(w)} pairs -> about {(ex - 0.001) / max(det1, 1e-9) * (size - 1):.3f} rematch partners per game (detection rate taken from network 1's unseen players: {det1:.3f})")
    # the test opponents are mostly SEEN players for the all-players network; show its own reference too
    own = load(-1, "own"); pl = D["owner"]; i = rng.choice(len(pl), 60000, replace=False); same, _ = within(own[i], pl[i]); diff = between(own[i], list(pl[i])); log(f"all-players network, seen players: same player two own games: {qs(same)} | different players: {qs(diff)} | detection at the different-player 99.9th percentile: {float((same > np.quantile(diff, 0.999)).mean()):.3f}")


if __name__ == "__main__":
    main()
