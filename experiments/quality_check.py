#!/usr/bin/env python
"""Does move quality add to the best features? Folds 1-4 CV of fused_opp plus humanv0's point / win-rate losses (idea #2).

Question features from scripts/quality_score.py's per-game table: the move-weighted means over the question's games
of the target's and the opponent's statistics (quality_score.COLS but n), the spread of the target's per-game mean
point loss, and the share of games the target ends ahead (end lead clipped to +-30 points). Protocol as
scripts/l1b_opp_check.py (folds 1-4, pilot questions, 32 training questions per player, MLPModel(hidden=32)).

Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_q (fused_opp
+ quality) vs fused_opp. Reported only: hsl_q vs hsl, q alone, fused_opp vs hsl.

  python -I scripts/quality_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp \\
      results/l1b/scores3_B_opp --quality results/quality --out results/quality/quality_check.json
"""
import argparse
import glob
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
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402
import quality_score as qs  # noqa: E402

STATS = [c for c in qs.COLS if c != "n"]
SETS = ("hsl", "q", "hsl_q", "fused_opp", "fused_opp_q")
PRE = [("fused_opp_q", "fused_opp")]
REPORT = [("hsl_q", "hsl"), ("fused_opp", "hsl"), ("fused_opp_q", "hsl")]


def read_quality(fold_dir):
    files = sorted(glob.glob(str(Path(fold_dir) / "quality_s*_p*.csv")))
    if not files:
        raise FileNotFoundError(f"no quality tables in {fold_dir}")
    return pd.concat([pd.read_csv(p, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}) for p in files],
                     ignore_index=True).set_index("game_key")


def quality_matrix(table, questions):
    """[Q, 2 x len(STATS) + 2] question features from the per-game table (NaN stages filled with the column mean)."""
    rows = []
    for keys in questions["game_keys"]:
        g = table.loc[list(keys)]
        f = []
        for side in ("t", "o"):
            w = g[f"{side}_n"].to_numpy(float)
            for c in STATS:
                v = g[f"{side}_{c}"].to_numpy(float)
                m = np.isfinite(v) & (w > 0)
                f.append(float(np.average(v[m], weights=w[m])) if m.any() else np.nan)
        lm = g["t_loss_mean"].to_numpy(float)
        f.append(float(np.nanstd(lm)) if np.isfinite(lm).sum() > 1 else 0.0)
        f.append(float((np.clip(g["t_end_lead"].to_numpy(float), -30, 30) > 0).mean()))
        rows.append(f)
    X = np.asarray(rows, dtype=np.float64)
    col_mean = np.nanmean(X, axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(col_mean, np.nonzero(bad)[1])
    return X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="+", required=True)
    ap.add_argument("--quality", required=True, help="quality_score.py output directory (fold{F} subdirectories)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
        qual = read_quality(Path(a.quality) / f"fold{f}")
        missing = set(hsl["game_key"]) - set(qual.index)
        if missing:
            raise ValueError(f"fold {f}: {len(missing)} games have no quality row")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "qual": qual, "players": players, "val": l1b_cv.pilot_questions(players)}
        print(f"fold {f}: {len(players)} players, {len(hsl)} games", flush=True)

    def feats(hsl, l1b, opp, qual, questions):
        X, y, n = oc.features(hsl, l1b, opp, questions)
        Xq = quality_matrix(qual, questions)
        return {"hsl": X["hsl"], "q": Xq, "hsl_q": np.hstack([X["hsl"], Xq]), "fused_opp": X["fused_opp"],
                "fused_opp_q": np.hstack([X["fused_opp"], Xq])}, y, n

    rows, per_fold, nll = [], {s: [] for s in SETS}, {s: [] for s in SETS}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=key == "qual")  # noqa: E731
        qual_tr = pd.concat([data[f]["qual"] for f in folds if f != v])
        X_tr, y_tr, _ = feats(cat("hsl"), cat("l1b"), cat("opp"), qual_tr, train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = feats(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], data[v]["qual"], val)
        sc = {}
        for s in SETS:
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, val)
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v}: " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS))}, "per_fold": per_fold[s]}
            for s in SETS}

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= len(folds) - 1), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}
    for s, h in head.items():
        print(f"{s:12s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]) + "  1D-3D " + " ".join(f"{h['per_class'][c]:.3f}" for c in ("1D", "2D", "3D")))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "quality_dir": a.quality, "stats": STATS, "scores": head, "comparisons": comps,
           "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
