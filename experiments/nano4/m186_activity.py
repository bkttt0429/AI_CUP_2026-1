#!/usr/bin/env python
"""Section 4M M186 (2026-11-16, report only): are the test players like the LIGHT validation players?

Test questions hold almost no repeated opponent (M183 G) and no game against another test player (M185), while the
validation players (the owners of the training games) do both. If that is a matter of activity, the lightest validation
players should look like the test players in these label-free respects - and then what the pipeline does on them is the
better guide to the test set.
By the number of games under the validation player's name (bands), 20-game questions and all sizes:
  label-free: games with a clearly recognised opponent per game (m15 above the target side's q95), share of such pairs that
  are the same opponent, median games of the recognised opponents; the same three numbers for the test questions;
  with labels: base / final score (class-balanced inside the band), exact, the score the pooled posterior expects of its own
  answers against the score obtained (is the pool honest for light players too), anchor games per game.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); E1 = float(np.exp(-1.0))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def main():
    q = np.load(mb7.CACHE / "queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}; players = q["players"].astype(str)
    zz = np.load(ROOT / "results/r_variants/m127_SQ4.npz"); st = zz["TV"][:, 0] - zz["TV"][:, 1:5].mean(1); best = zz["best"]; thr_v = float(np.quantile(st[Q:2 * Q], 0.95)); thr_t = float(np.quantile(st[2 * Q + T:2 * Q + 2 * T], 0.95))
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ngp = pf.n_games.astype(int); ngc = ngp.reindex(players).to_numpy()
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); post = z["anat_post"]; fin = z["SQ4"].astype(int); plain = z["anat_plain"].astype(int); base = z["anat_base"].astype(int); AN = z["anat_AN"].astype(int)
    conf, rep, pairs, medg = [], [], [], []
    for v in mb7.FOLDS:
        _, _, _, qd = mb7.m3.load_fold(v)
        for g in qd["game_keys"]:
            r = np.array([row[k] for k in g if k in row], dtype=np.int64); ok = st[r] >= thr_v; b_ = best[r][ok]; c = len(b_); conf.append(c); pairs.append(c * (c - 1) // 2); rep.append(sum(k * (k - 1) // 2 for k in np.unique(b_, return_counts=True)[1])); medg.append(float(np.median(ngc[b_])) if c else np.nan)
    conf, rep, pairs, medg = map(np.array, (conf, rep, pairs, medg)); assert len(conf) == len(y); ng = ngp.reindex(pid).to_numpy()
    sc = lambda d: np.where(d == y, 1.0, np.where(np.abs(d - y) == 1, E1, 0.0))  # noqa: E731
    SMx = np.eye(10) + E1 * (np.eye(10, k=1) + np.eye(10, k=-1)); own = (post @ SMx)[np.arange(len(y)), plain]
    def bal(x, m):
        tot, k = 0.0, 0
        for s in sorted(set(n)):
            for c in range(10):
                mm = m & (n == s) & (y == c)
                if mm.any():
                    tot += float(x[mm].mean()); k += 1
        return tot / max(k, 1)
    edges = [20, 30, 45, 70, 110, 200, 400, 10 ** 6]
    print("validation players by the number of games under their name (all question sizes; class-balanced inside the band):")
    print("  band | players | recognised opponents per game | same-opponent share of such pairs | median games of the recognised opponents | anchor games per game | base | final | exact | pooled expects / obtains (no guards)")
    for i in range(len(edges) - 1):
        m = (ng >= edges[i]) & (ng < edges[i + 1])
        print(f"  [{edges[i]}, {edges[i + 1] if edges[i + 1] < 10 ** 6 else 'inf'}) | {len(set(pid[m]))} | {float(conf[m].sum() / n[m].sum()):.3f} | {float(rep[m].sum() / max(pairs[m].sum(), 1)):.4f} | {float(np.nanmedian(medg[m])):.0f} | {float((AN[m, 1] / n[m]).mean()):.3f} | {bal(sc(base), m):.4f} | {bal(sc(fin), m):.4f} | {bal((fin == y).astype(float), m):.3f} | {bal(own, m):.4f} / {bal(sc(plain), m):.4f}")
    # the test questions, label-free
    tg = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv"))], ignore_index=True); qo = np.arange(2 * Q + H, 2 * Q + H + T); rows = q["qtest"][qo].astype(int); qid = tg.question_id.to_numpy()[rows]; sto = st[2 * Q:2 * Q + T]; bo = best[2 * Q:2 * Q + T]
    df = pd.DataFrame({"qid": qid, "ok": sto >= thr_t, "best": bo}); c_t = df.groupby("qid").ok.sum(); n_t = df.groupby("qid").size(); rp = tp = 0; med = []
    for _, g in df[df.ok].groupby("qid"):
        c = len(g); tp += c * (c - 1) // 2; rp += sum(k * (k - 1) // 2 for k in g.best.value_counts().to_numpy()); med.append(float(np.median(ngc[g.best.to_numpy()])))
    print(f"test questions (label-free): recognised opponents per game {float(c_t.sum() / n_t.sum()):.3f}; same-opponent share of such pairs {rp / max(tp, 1):.4f}; median games of the recognised opponents {float(np.median(med)):.0f}")
    for s in (5, 10, 20):
        mv = n == s; mt = n_t == s; print(f"  {s:2d}-game questions: recognised opponents per game - validation {float(conf[mv].sum() / n[mv].sum()):.3f} (lightest band {float(conf[mv & (ng < 30)].sum() / max(n[mv & (ng < 30)].sum(), 1)):.3f}), test {float(c_t[mt].sum() / n_t[mt].sum()):.3f}; questions without any: validation {float((conf[mv] == 0).mean()):.3f} (lightest {float((conf[mv & (ng < 30)] == 0).mean()):.3f}), test {float((c_t[mt] == 0).mean()):.3f}")


if __name__ == "__main__":
    main()
