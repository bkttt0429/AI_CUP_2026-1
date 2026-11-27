"""Section 4M M51b check 1 (2026-11-07; user: 「1 2 都做」): does M33's adoption (ct9 over ct7, +0.0060 class-balanced on all validation
questions) hold under the test-like protocol -- questions holding an HD >= 2 game dropped (the test has none), n = 5 / 10 / 20 weighted
equally? Report only; from the stored out-of-fold probabilities (results/trunk_seq/es/val{v}/<tag>_s<seed>.npz).
  ct9 = C/T-pc-f0-x40-oh seeds 0-2, ct7 = C/T-f0-x40-oh seeds 0-2, each fold's six members averaged, expected-score decision.
  For all questions and for the HD-free subset: class-balanced paired difference with the player bootstrap (aicup.qeval.paired_diff),
  per-fold differences, the test-like plain difference (plain mean within n, mean over n; player bootstrap), by n, and the n-balanced
  per-class difference a_c = E[s9 - s7 | Y = c].
  python -I logs/nano4/m51b_ct9_testlike.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m47_prior import boot_players, nplain  # noqa: E402

TAGS = {"ct9": ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"), "ct7": ("C-f0-x40-oh", "T-f0-x40-oh")}


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = {k: np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in tags for s in (0, 1, 2)], axis=0) for k, tags in TAGS.items()}
        mh = np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]])
        D[v] = {"y": y, "n": n, "pid": pid, "hd2": mh >= 2, "s": {k: qscore(decide(P[k]), y) for k in TAGS}}
    rep = {}
    for subset in ("all", "hd_free"):
        keep = {v: (np.ones(len(D[v]["y"]), bool) if subset == "all" else ~D[v]["hd2"]) for v in FOLDS}
        y = np.concatenate([D[v]["y"][keep[v]] for v in FOLDS]); n = np.concatenate([D[v]["n"][keep[v]] for v in FOLDS]); pid = np.concatenate([D[v]["pid"][keep[v]] for v in FOLDS])
        s9 = np.concatenate([D[v]["s"]["ct9"][keep[v]] for v in FOLDS]); s7 = np.concatenate([D[v]["s"]["ct7"][keep[v]] for v in FOLDS])
        dd = paired_diff(s9, s7, y, pid, n, n=5000)
        pf = [balanced_mean_over_n(D[v]["s"]["ct9"][keep[v]], D[v]["y"][keep[v]], D[v]["n"][keep[v]]) - balanced_mean_over_n(D[v]["s"]["ct7"][keep[v]], D[v]["y"][keep[v]], D[v]["n"][keep[v]]) for v in FOLDS]
        plain = nplain(s9, n) - nplain(s7, n); ci_plain = boot_players(s9, s7, n, pid, B=3000)
        a = [float(np.mean([(s9 - s7)[(y == c) & (n == m)].mean() for m in (5, 10, 20)])) for c in range(10)]
        rule = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
        rep[subset] = {"questions": int(len(y)), "class_balanced": dd, "per_fold": pf, "plain": plain, "plain_ci95": ci_plain, "a_by_class": dict(zip(LEVELS, a)), "pass_rule": bool(rule),
                       "score_ct9": float(balanced_mean_over_n(s9, y, n)), "score_ct7": float(balanced_mean_over_n(s7, y, n))}
        print(f"{subset:8s} ({len(y)} questions): ct9 {rep[subset]['score_ct9']:.4f} vs ct7 {rep[subset]['score_ct7']:.4f}; class-balanced {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] "
              f"folds " + " ".join(f"{e:+.4f}" for e in pf) + f"; by n " + " ".join(f"{k} {x:+.4f}" for k, x in dd["by_n"].items())
              + f"; test-like plain {plain:+.4f} [{ci_plain[0]:+.4f}, {ci_plain[1]:+.4f}]" + ("  -> rule PASS" if rule else "  -> rule no"), flush=True)
        print(f"         a_c (ct9 - ct7 | Y = c, n-balanced): " + " ".join(f"{lv} {x:+.3f}" for lv, x in zip(LEVELS, a)), flush=True)
    Path(ROOT / "results/trunk_seq/es/m51b_ct9_testlike.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
