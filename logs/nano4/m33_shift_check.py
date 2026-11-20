"""M33 after the Public score (2026-11-06, report only): is the CV gain of the pc members carried by question types the test does not have?
The test has no HD >= 2 game (and 22.8 % HD 1 games); the validation questions do. On folds 1-4, ct7's 6 members vs the 6 pc members:
the class-balanced official-score difference (paired player bootstrap) and the plain question average (the official averaging) on
all questions, on the test-like questions (no HD >= 2 game) and on the others; and how many decisions differ.
  python -I logs/nano4/m33_shift_check.py --out results/trunk_seq/es/m33_shift_check.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402


def probs(v, tags):
    return np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in tags for s in (0, 1, 2)], axis=0)


def boot_plain(d, pid, B=5000, seed=0):
    players = np.unique(pid); idx = {p: np.flatnonzero(pid == p) for p in players}; rng = np.random.default_rng(seed)
    vals = [np.concatenate([d[idx[p]] for p in rng.choice(players, len(players))]).mean() for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"]
    S_new, S_old, Y, N, PID, MAXHD, DEC = [], [], [], [], [], [], []
    for v in (1, 2, 3, 4):
        y, n, pid, q = load_fold(v)
        Pn, Po = probs(v, ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh")), probs(v, ("C-f0-x40-oh", "T-f0-x40-oh"))
        rn, sn = evaluate(Pn, y, q, sizes=SIZES); ro, so = evaluate(Po, y, q, sizes=SIZES)
        S_new.append(sn); S_old.append(so); Y.append(y); N.append(n); PID.append(pid)
        MAXHD.append(np.array([max(hd.get(k, 0.0) for k in ks) for ks in q["game_keys"]]))
        DEC.append(np.abs(sn - so) > 1e-9)
    sn, so, y, n, pid, mh, dec = (np.concatenate(x) for x in (S_new, S_old, Y, N, PID, MAXHD, DEC))
    rep = {"questions": int(len(y)), "share_with_hd2plus": float((mh >= 2).mean()), "subsets": {}}
    print(f"{len(y)} validation questions; with an HD >= 2 game: {(mh >= 2).mean():.3f}; score changes on {dec.mean():.3f} of the questions")
    for name, m in (("all", np.ones(len(y), bool)), ("test-like (no HD >= 2)", mh < 2), ("with HD >= 2", mh >= 2)):
        d = paired_diff(sn[m], so[m], y[m], pid[m], n[m], sizes=SIZES, n=5000)
        plain = float((sn[m] - so[m]).mean()); ci = boot_plain(sn[m] - so[m], pid[m])
        rep["subsets"][name] = {"n": int(m.sum()), "balanced": d, "plain_mean_diff": plain, "plain_ci95": ci, "changed_share": float(dec[m].mean())}
        print(f"{name:24s} n {m.sum():6d}  class-balanced {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  plain average {plain:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"changed {dec[m].mean():.3f}  by n " + " ".join(f"{k}: {x:+.4f}" for k, x in d["by_n"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
