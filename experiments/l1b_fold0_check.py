#!/usr/bin/env python
"""Leak check of the L1b fused features on fold 0 (issue #6, 2026-10-26; the user's decision, no selection on it).

Neither L1b model saw fold 0 (A trained on folds 1-2, B on 3-4), and both scored it. The MLPs (MLPModel(hidden=32))
are fitted once on folds 1-4 -- training questions qfeatures.sample_questions, --per-player per player,
RandomState(42) over the folds in ascending order, features as in the CV (folds 1-2 from B's scores, 3-4 from
A's) -- and scored on fold 0's pilot questions (1 x 20, 2 x 10, 4 x 5 games per player): hsl alone, and fused
with fold 0's L1b features from A, from B, and the mean of the two probabilities (the test-time rule). Expected
if the CV gain is real: fused clearly above hsl on fold 0 too; hsl itself should reproduce the ladder's
r2_mlp24 retrained on folds 1-4 (0.6113). Each --run is one L1b generation: NAME A_DIR B_DIR.

  python -I scripts/l1b_fold0_check.py --run stage1 results/l1b/scores results/l1b/scores_B \\
      --run disc results/l1b/scores3 results/l1b/scores3_B --out results/l1b/fold0_check.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
import l1b_cv  # noqa: E402

TRAIN_FOLDS = (1, 2, 3, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "A_DIR", "B_DIR"))
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    out = {"train_folds": list(TRAIN_FOLDS), "per_player": a.per_player, "runs": {}}
    hsl_done = None
    for name, a_dir, b_dir in a.run:
        src = {1: b_dir, 2: b_dir, 3: a_dir, 4: a_dir}  # the CV's out-of-fold features
        data = {}
        for f in TRAIN_FOLDS:
            hsl, l1b = l1b_cv.load_fold(a.hsl, src[f], f)
            data[f] = {"hsl": hsl, "l1b": l1b, "players": l1b_cv.players_from_scores(hsl, folds_df)}
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in TRAIN_FOLDS], ignore_index=True)
        X_tr, y_tr, _ = l1b_cv.feature_sets(pd.concat([data[f]["hsl"] for f in TRAIN_FOLDS], ignore_index=True),
                                            pd.concat([data[f]["l1b"] for f in TRAIN_FOLDS], ignore_index=True), train_q)
        hsl0, l1b0_a = l1b_cv.load_fold(a.hsl, a_dir, 0)
        _, l1b0_b = l1b_cv.load_fold(a.hsl, b_dir, 0)
        val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))
        X_a, y, n = l1b_cv.feature_sets(hsl0, l1b0_a, val)
        X_b, y_b, _ = l1b_cv.feature_sets(hsl0, l1b0_b, val)
        assert np.array_equal(y, y_b)
        pid = val["player_id"].to_numpy(dtype=str)
        mlp_h = MLPModel(hidden=32).fit(X_tr["hsl"], y_tr)
        mlp_f = MLPModel(hidden=32).fit(X_tr["fused"], y_tr)
        P = {"hsl": mlp_h.predict_proba(X_a["hsl"]), "fused_A": mlp_f.predict_proba(X_a["fused"]), "fused_B": mlp_f.predict_proba(X_b["fused"])}
        P["fused_avg"] = (P["fused_A"] + P["fused_B"]) / 2
        res = {k: evaluate(p, y, val) for k, p in P.items()}
        s = {k: r[1] for k, r in res.items()}
        run = {"n_train_questions": int(len(train_q)), "n_questions": int(len(y)), "n_players": int(len(set(pid))), "scores": {}, "vs_hsl": {}}
        for k in P:
            run["scores"][k] = {"balanced_mean_over_n": balanced_mean_over_n(s[k], y, n), "nll_balanced": res[k][0]["nll_balanced"],
                                "balanced_by_n": {str(m): balanced_mean_over_n(s[k], y, n, sizes=(m,)) for m in SIZES},
                                "per_class": {LEVELS[c]: float(np.mean(s[k][y == c])) for c in range(len(LEVELS)) if np.any(y == c)}}
            if k != "hsl":
                run["vs_hsl"][k] = paired_diff(s[k], s["hsl"], y, pid, n, n=a.n_boot)
        out["runs"][name] = run
        hsl_done = run["scores"]["hsl"]["balanced_mean_over_n"]
        print(f"[{name}] fold 0: {run['n_players']} players, {run['n_questions']} questions; trained on {run['n_train_questions']} questions")
        for k, v in run["scores"].items():
            d = run["vs_hsl"].get(k)
            extra = f"  vs hsl {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""
            print(f"  {k:10s} {v['balanced_mean_over_n']:.4f}  NLL {v['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {v['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
                  + extra + "  1D-3D " + " ".join(f"{v['per_class'][c]:.3f}" for c in ("1D", "2D", "3D")), flush=True)
    out["hsl_reference_ladder_r2_mlp24_fold0"] = 0.6113
    out.update(environment("cpu"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} (hsl {hsl_done:.4f}; the ladder's r2_mlp24 retrained on folds 1-4 had 0.6113)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
