#!/usr/bin/env python
"""Section 4M M183 part G, second step (2026-11-16, report only): permutation tests for "repeated opponents cluster".

(1) training side: the validation players' 20 games; statistic = number of same-opponent pairs whose game numbers are closer
    than 1,000 / 3,000 and the mean log distance of the same-opponent pairs; null = game numbers shuffled inside each question.
(2) test side, label-free: inside each test question the games in file order (the running game key); pairs of games whose
    opponent side is recognised with a clear margin; statistic = number of same-opponent pairs at position distance 1 / <= 2
    and the mean position distance of the same-opponent pairs; null = positions shuffled inside each question.
    A clear excess at distance 1 means the games of a test question are listed in time order.

  python -I logs/nano4/m183_time2.py [tag, default SQ4]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def perm_test(groups, stats, n_perm=2000, seed=0):
    """groups: list of (positions, same-pair index arrays i, j). stats: dict name -> function of the distances of the same pairs."""
    rng = np.random.default_rng(seed); obs = {k: f(np.concatenate([np.abs(p[i] - p[j]) for p, i, j in groups])) for k, f in stats.items()}; null = {k: [] for k in stats}
    for _ in range(n_perm):
        d = np.concatenate([np.abs((pp := rng.permutation(p))[i] - pp[j]) for p, i, j in groups])
        for k, f in stats.items():
            null[k].append(f(d))
    return {k: (obs[k], float(np.mean(null[k])), float(np.std(null[k])), float((np.sum(np.array(null[k]) >= obs[k]) + 1) / (n_perm + 1)), float((np.sum(np.array(null[k]) <= obs[k]) + 1) / (n_perm + 1))) for k in stats}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "SQ4"; m3 = mb7.m3; q = np.load(mb7.CACHE / "queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}
    z = np.load(ROOT / f"results/r_variants/m127_{tag}.npz"); TV = z["TV"]; st = TV[:, 0] - TV[:, 1:5].mean(1); best = z["best"]
    show = lambda r: "; ".join(f"{k}: observed {o:.3f}, shuffled {mu:.3f} (sd {sd:.3f}), p(more) {ph:.4f}, p(less) {pl:.4f}" for k, (o, mu, sd, ph, pl) in r.items())  # noqa: E731
    for qq in (0.95, 0.99):
        thr = float(np.quantile(st[Q:2 * Q], qq)); groups = []
        for v in mb7.FOLDS:
            y, n, pid, qd = m3.load_fold(v); gk = list(qd["game_keys"])
            for j in np.flatnonzero(np.asarray(n) == 20):
                ks = [k for k in gk[j] if k in row]; r = np.array([row[k] for k in ks]); ok = st[r] >= thr
                if ok.sum() < 2:
                    continue
                ids = np.array([int(k.split("g_")[1]) for k in ks])[ok].astype(float); b_ = best[r][ok]; i, jj = np.triu_indices(len(ids), 1); s_ = b_[i] == b_[jj]
                if s_.any():
                    groups.append((ids, i[s_], jj[s_]))
        res = perm_test(groups, {"pairs closer than 1,000": lambda d: float((d < 1000).sum()), "closer than 3,000": lambda d: float((d < 3000).sum()), "mean log10 distance": lambda d: float(np.log10(np.maximum(d, 1)).mean())})
        print(f"(1) training side, margin above q{int(qq * 100)}: {sum(len(g[1]) for g in groups)} same-opponent pairs in {len(groups)} questions | " + show(res), flush=True)
    # ---------------------------------------------------------------- test side
    tg = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv"))], ignore_index=True); tg["task"] = tg["task"].astype(int); tg = tg.sort_values("task").reset_index(drop=True); assert (tg.task.to_numpy() == np.arange(len(tg))).all(); tq = np.arange(2 * Q + H, 2 * Q + H + T); assert (q["side"][tq] == 1).all()
    rows = q["qtest"][tq].astype(int); assert rows.min() >= 0 and rows.max() < len(tg); qid = tg.question_id.to_numpy()[rows]; print(f"   [fix 2026-11-19] games table sorted by task; question ids equal to m82_test.npz in {float((qid == np.load(ROOT / 'results/r_variants/m82_test.npz')['qid'].astype(str)).mean()):.4f} of the rows", flush=True); num = tg.game_key.str.lstrip("g").astype(int).to_numpy()[rows]; sto = st[2 * Q:2 * Q + T]; bo = best[2 * Q:2 * Q + T]; null_t = st[2 * Q + T:2 * Q + 2 * T]
    df = pd.DataFrame({"qid": qid, "num": num, "st": sto, "best": bo}); df["pos"] = df.groupby("qid").num.rank(method="first").astype(int) - 1; sizes = df.groupby("qid").size()
    print(f"\n(2) test side: {len(df)} games in {df.qid.nunique()} questions; the game numbers inside a question are consecutive in {float((df.groupby('qid').num.agg(lambda x: x.max() - x.min() + 1) == sizes).mean()):.3f} of the questions")
    for qq in (0.95, 0.99):
        thr = float(np.quantile(null_t, qq)); groups = []; byd = {}; tot = {}
        for _, g in df[df.st >= thr].groupby("qid"):
            if len(g) < 2:
                continue
            p = g.pos.to_numpy().astype(float); b_ = g.best.to_numpy(); i, jj = np.triu_indices(len(p), 1); s_ = b_[i] == b_[jj]; d = np.abs(p[i] - p[jj]).astype(int)
            for dd, ss in zip(d, s_):
                k = 1 if dd == 1 else 2 if dd == 2 else 4 if dd <= 4 else 9 if dd <= 9 else 19; tot[k] = tot.get(k, 0) + 1; byd[k] = byd.get(k, 0) + int(ss)
            if s_.any():
                groups.append((p, i[s_], jj[s_]))
        print(f"  margin above the target side's q{int(qq * 100)} ({thr:.2f}): {int((df.st >= thr).sum())} games; same-opponent share by position distance " + "  ".join(f"{'1' if k == 1 else '2' if k == 2 else '3-4' if k == 4 else '5-9' if k == 9 else '10+'}: {byd.get(k, 0) / max(tot.get(k, 0), 1):.4f} ({byd.get(k, 0)}/{tot.get(k, 0)})" for k in (1, 2, 4, 9, 19)))
        if groups:
            res = perm_test(groups, {"pairs at distance 1": lambda d: float((d == 1).sum()), "at distance <= 2": lambda d: float((d <= 2).sum()), "mean distance": lambda d: float(d.mean())})
            print(f"    {sum(len(g[1]) for g in groups)} same-opponent pairs in {len(groups)} questions | " + show(res), flush=True)


if __name__ == "__main__":
    main()
