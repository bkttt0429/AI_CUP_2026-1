#!/usr/bin/env python
"""Section 4M M187 (2026-11-16, report only): how much is left in "recognise more opponents", and a last look at time.

(1) share of the opponents that ARE training players. The opponent side's margin statistic is a mixture of "in the pool"
    (shape taken from the held-out own games of the pool players, weighted by the owner's number of games - opponents are met
    in proportion to activity) and "not in the pool" (the target side: never a candidate). pi(t) = (P_opp(> t) - P_out(> t))
    / (P_in(> t) - P_out(> t)) over a range of thresholds; test and validation. With it: of the opponents in the pool, the
    share recognised with a clear margin (above the null's q95), and what perfect recognition would add in anchor games.
(2) is the file order of a test question a time order? If ranks drift and the games are listed in time order, the labels of
    the recognised even-game opponents trend along the position inside the question. Pooled within-question correlation of
    position and opponent label, permutation test (positions shuffled inside the question); the same for the validation
    players along their game numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def trend(groups, n_perm=2000, seed=0):
    """groups: list of (positions, labels). Pooled within-group correlation, permutation of the positions inside each group."""
    rng = np.random.default_rng(seed)
    def stat(perm):
        sxy = sxx = syy = 0.0
        for p, l in groups:
            pp = rng.permutation(p) if perm else p; a = pp - pp.mean(); b = l - l.mean(); sxy += float(a @ b); sxx += float(a @ a); syy += float(b @ b)
        return sxy / np.sqrt(max(sxx * syy, 1e-12))
    obs = stat(False); null = np.array([stat(True) for _ in range(n_perm)]); return obs, float(null.std()), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    tag = "SQ4"; q = np.load(mb7.CACHE / "queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); players = q["players"].astype(str); plab = q["plab"].astype(int)
    z = np.load(ROOT / f"results/r_variants/m127_{tag}.npz"); st = z["TV"][:, 0] - z["TV"][:, 1:5].mean(1); best = z["best"]; hs = z["HV"][:, 0] - z["HV"][:, 1:5].mean(1); truth = z["held_truth"].astype(int)
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players).to_numpy(); w = ng[truth] / ng[truth].sum()
    print("(1) share of the opponents that are training players (mixture of the held-out own games' shape and the target side's)")
    for lab, opp, out in (("validation", st[:Q], st[Q:2 * Q]), ("test", st[2 * Q:2 * Q + T], st[2 * Q + T:2 * Q + 2 * T])):
        row = []
        for qq in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            t = float(np.quantile(out, qq)); pin = float((w * (hs > t)).sum()); pi = (float((opp > t).mean()) - (1 - qq)) / max(pin - (1 - qq), 1e-9); row.append(f"q{int(qq * 100)}: {pi:.3f} (in-pool above the line {pin:.3f})")
        t95 = float(np.quantile(out, 0.95)); rec = float((opp > t95).mean()) - 0.05; print(f"  {lab}: pi by threshold - " + "  ".join(row) + f" | recognised with a clear margin (above q95, net of the null) {rec:.3f} of the games")
    # ---------------------------------------------------------------- (2)
    print("\n(2) trend of the recognised even-game opponents' labels along the order of the games")
    tg = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv"))], ignore_index=True); tg["task"] = tg["task"].astype(int); tg = tg.sort_values("task").reset_index(drop=True); assert (tg.task.to_numpy() == np.arange(len(tg))).all(); qo = np.arange(2 * Q + H, 2 * Q + H + T); rows = q["qtest"][qo].astype(int); qid = tg.question_id.to_numpy()[rows]; print(f"   [fix 2026-11-19] games table sorted by task; question ids equal to m82_test.npz in {float((qid == np.load(ROOT / 'results/r_variants/m82_test.npz')['qid'].astype(str)).mean()):.4f} of the rows", flush=True); num = tg.game_key.str.lstrip("g").astype(int).to_numpy()[rows]
    meta = pd.read_csv(ROOT / "results/data/test_games.csv", dtype=str); meta["num"] = np.arange(len(meta)); hd_t = meta.set_index("num").hd.astype(float).reindex(num).to_numpy() if len(meta) == T else np.zeros(T)
    sto = st[2 * Q:2 * Q + T]; bo = best[2 * Q:2 * Q + T]
    for qq in (0.90, 0.95, 0.99):
        thr = float(np.quantile(st[2 * Q + T:2 * Q + 2 * T], qq)); df = pd.DataFrame({"qid": qid, "num": num, "ok": (sto >= thr) & (hd_t == 0), "lab": plab[bo].astype(float)}); df["pos"] = df.groupby("qid").num.rank(method="first"); groups = [(g.pos.to_numpy(float), g.lab.to_numpy(float)) for _, g in df[df.ok].groupby("qid") if len(g) >= 3 and g.lab.std() > 0]
        obs, sd, p = trend(groups); print(f"  test, margin above q{int(qq * 100)}: {sum(len(g[0]) for g in groups)} anchors in {len(groups)} questions; within-question correlation of position and opponent label {obs:+.4f} (shuffled sd {sd:.4f}, two-sided p {p:.3f})")
    row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}; hdv = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "hd"], dtype={"game_key": str}).set_index("game_key").hd.astype(float).reindex(q["qkey"][:Q].astype(str)).to_numpy(); thr = float(np.quantile(st[Q:2 * Q], 0.95)); groups = []
    for v in mb7.FOLDS:
        _, n, _, qd = mb7.m3.load_fold(v)
        for j in np.flatnonzero(np.asarray(n) == 20):
            ks = [k for k in qd["game_keys"].iloc[j] if k in row]; r = np.array([row[k] for k in ks], dtype=np.int64); ok = (st[r] >= thr) & (hdv[r] == 0)
            if ok.sum() >= 3:
                ids = np.array([int(k.split("g_")[1]) for k in ks], float)[ok]; l = plab[best[r][ok]].astype(float)
                if l.std() > 0:
                    groups.append((ids.argsort().argsort().astype(float), l))
    obs, sd, p = trend(groups); print(f"  validation (20 games per player, ordered by game number), margin above q95: {sum(len(g[0]) for g in groups)} anchors in {len(groups)} players; correlation {obs:+.4f} (shuffled sd {sd:.4f}, p {p:.3f})")


if __name__ == "__main__":
    main()
