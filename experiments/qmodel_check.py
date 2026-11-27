#!/usr/bin/env python
"""Folds 1-4 CV of the QUESTION-level model: does regularisation help fused_opp_diff, and does it make the model robust to
an extra block of columns? (issue #41)

Three lines (2.16 shape statistics, 2.20 histograms, 2.22 TSFEL) lost score when a block of extra columns was appended to
fused_opp_diff, and 2.22 showed a dose-response (more columns, bigger loss). The competition's own post-test lists the
remedies: more data, L1 regularisation, lower model complexity. Two explanations compete: the question-level
MLPModel(hidden=32) (fixed 300 epochs, l2 1e-4, no early stopping) is fragile to extra columns, or the blocks carry
player-specific shifts across folds. This script separates them. Models on the same training questions (protocol as
scripts/l1b_opp_check.py; fused_opp_diff with mlp must reproduce 0.6916):

  mlp       MLPModel(hidden=32)                          the current question-level model (baseline)
  mlp_es    + inner early stopping                       10% of the training players held out per rank (aicup.settrain.inner_players,
                                                         INNER_SEED), class-balanced CE on them every 10 epochs, patience 5
  mlp_l1    + L1 1e-3 on the input layer + early stopping  sparse input selection      <- pre-specified vs mlp on base
  mlp_l2    l2 1e-2 (100x), fixed epochs
  mlp_h16   hidden 16, fixed epochs
  softmax   SoftmaxModel(l2=1e-3)                        the linear reference
  mlp_pp64  mlp trained on 64 sampled questions per player instead of 32 ("more data"; base only)

Feature sets: base = fused_opp_diff (59); base_tsfel16 = + the PCA-16 TSFEL block of scripts/tsfel_check.py (from its cached
per-fold tables, --tsfel-cache; standardised and PCA-fitted on the training questions); base_noise16 = + 16 standard-normal
columns (RandomState(--seed + fold)), the pure-noise control. Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in
>= 3 of 4 folds, (4) >= 0.005): mlp_l1 on base vs mlp on base. Reported only: every other model on base vs mlp on base;
and per model the robustness deltas (model on base_tsfel16 minus model on base, model on base_noise16 minus model on base).
Reading rule, written before the run: if mlp loses >= 0.005 on base_noise16 while mlp_es / mlp_l1 lose < 0.002, the
question-level model is fragile to extra columns and the three lines' "fail" verdicts rest partly on the model (re-test
them under the regularised model, a new issue); if mlp does not lose on base_noise16, the TSFEL loss was the block's own
shift and the verdicts stand. Fold 0 is refused.

  python -I scripts/qmodel_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --tsfel-cache results/seq/tsfel --out results/l1b/qmodel_check.json
"""
import argparse
import sys
import time
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
from aicup.rankmodel import MLPModel, SoftmaxModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.seqtsfel import feature_columns, load_tsfel_table  # noqa: E402
from aicup.settrain import INNER_SEED, inner_players  # noqa: E402
import error_analysis as ea  # noqa: E402
import l1b_cv  # noqa: E402
from tsfel_check import tsfel_blocks  # noqa: E402

BASE = "fused_opp_diff"
SETS = ("base", "base_tsfel16", "base_noise16")
MODELS = ("mlp", "mlp_es", "mlp_l1", "mlp_l2", "mlp_h16", "softmax", "mlp_pp64")
PRE = ("mlp_l1", "mlp")
INNER_FRAC, EVAL_EVERY, PATIENCE = 0.1, 10, 5
K_TSFEL, NOISE_DIM = 16, 16


def key(model, fset):
    return f"{model}@{fset}"


def fit_predict(model, X_tr, y_tr, X_va, inner, pid_tr=None):
    """Probabilities of one model on X_va. `inner` [N_tr] bool marks the training questions whose players are held out for the
    early-stopping models (they train on the rest); the fixed-epoch models use every training question."""
    if model == "softmax":
        return SoftmaxModel(l2=1e-3).fit(X_tr, y_tr).predict_proba(X_va), None
    if model in ("mlp", "mlp_pp64"):
        return MLPModel(hidden=32).fit(X_tr, y_tr).predict_proba(X_va), None
    if model == "mlp_l2":
        return MLPModel(hidden=32, l2=1e-2).fit(X_tr, y_tr).predict_proba(X_va), None
    if model == "mlp_h16":
        return MLPModel(hidden=16).fit(X_tr, y_tr).predict_proba(X_va), None
    m = MLPModel(hidden=32, l1=1e-3 if model == "mlp_l1" else 0.0)
    m.fit(X_tr[~inner], y_tr[~inner], val=(X_tr[inner], y_tr[inner]), eval_every=EVAL_EVERY, patience=PATIENCE)
    return m.predict_proba(X_va), m.best_epoch


def load_tsfel_cache(cache_dir, folds):
    """{fold: table} from --tsfel-cache/fold<F>.npz, or None when any is missing."""
    paths = {f: Path(cache_dir) / f"fold{f}.npz" for f in folds}
    if not all(p.exists() for p in paths.values()):
        return None
    return {f: load_tsfel_table(p) for f, p in paths.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--per-player-more", type=int, default=64, help="training questions per player of mlp_pp64")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--tsfel-cache", default=str(ROOT / "results" / "seq" / "tsfel"), help="scripts/tsfel_check.py's per-fold tables (fold<F>.npz)")
    ap.add_argument("--no-tsfel", action="store_true", help="skip the base_tsfel16 set (no TSFEL cache)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the noise block (+ fold)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    t0 = time.time()
    try:
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
        tables = None if a.no_tsfel else load_tsfel_cache(a.tsfel_cache, folds)
        if tables is None and not a.no_tsfel:
            raise FileNotFoundError(f"{a.tsfel_cache}: fold<F>.npz missing for some of {folds}; run scripts/tsfel_check.py first or pass --no-tsfel")
        if tables is not None:
            for f in folds:
                if set(tables[f]["game_key"]) != set(data[f]["hsl"]["game_key"]):
                    raise ValueError(f"fold {f}: the TSFEL table covers other games than {a.hsl}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    sets = [s for s in SETS if not (s == "base_tsfel16" and tables is None)]
    cols = sorted(set().union(*[feature_columns(tables[f]) for f in folds])) if tables is not None else []
    combos = [(m, s) for m in MODELS for s in sets if not (m == "mlp_pp64" and s != "base")]
    rows, per_fold, nll, dims, epochs, inner_sizes = [], {key(*c): [] for c in combos}, {key(*c): [] for c in combos}, {}, {key(*c): [] for c in combos}, []
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        rng_more = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_more = pd.concat([sample_questions(data[f]["players"], a.per_player_more, rng_more) for f in folds if f != v], ignore_index=True)
        cat = lambda k: pd.concat([data[f][k] for f in folds if f != v], ignore_index=True)  # noqa: E731
        hsl_tr, l1b_tr, opp_tr = cat("hsl"), cat("l1b"), cat("opp")
        X_tr, y_tr, _ = ea.feature_sets(hsl_tr, l1b_tr, opp_tr, train_q)
        X_more, y_more, _ = ea.feature_sets(hsl_tr, l1b_tr, opp_tr, train_more)
        d = data[v]
        X_va, y_va, n_va = ea.feature_sets(d["hsl"], d["l1b"], d["opp"], d["val"])
        inner = inner_players(train_q["player_id"].to_numpy(dtype=str), y_tr, INNER_FRAC, seed=INNER_SEED + v)
        inner_sizes.append(int(inner.sum()))
        feats = {"base": (X_tr[BASE], X_va[BASE])}
        if tables is not None:
            blocks, _ = tsfel_blocks({"train": pd.concat([tables[f] for f in folds if f != v], ignore_index=True), "val": tables[v]}, cols, train_q, d["val"], [K_TSFEL])
            feats["base_tsfel16"] = (np.hstack([X_tr[BASE], blocks[K_TSFEL][0]]), np.hstack([X_va[BASE], blocks[K_TSFEL][1]]))
        nrng = np.random.RandomState(a.seed + v)
        feats["base_noise16"] = (np.hstack([X_tr[BASE], nrng.randn(len(X_tr[BASE]), NOISE_DIM)]), np.hstack([X_va[BASE], nrng.randn(len(X_va[BASE]), NOISE_DIM)]))
        sc = {}
        for m, s in combos:
            k = key(m, s)
            if m == "mlp_pp64":
                P, be = fit_predict(m, X_more[BASE], y_more, feats[s][1], None)
                dims[k] = int(X_more[BASE].shape[1])
            else:
                P, be = fit_predict(m, feats[s][0], y_tr, feats[s][1], inner)
                dims[k] = int(feats[s][0].shape[1])
            res, sc[k] = evaluate(P, y_va, d["val"])
            per_fold[k].append(res["balanced_mean_over_n"])
            nll[k].append(res["nll_balanced"])
            epochs[k].append(be)
        print(f"fold {v} ({len(train_q)} training questions, {int(inner.sum())} inner): " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in sc) + f"  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **sc}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {k: {"balanced_mean_over_n": balanced_mean_over_n(t[k].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[k])), "n_features": dims[k],
                "balanced_by_n": {str(m): balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[k].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[k], "best_epoch": epochs[k]} for k in per_fold}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    ref = key("mlp", "base")
    comps = {"pre_specified": {f"{key(PRE[0], 'base')} vs {ref}": compare(key(PRE[0], "base"), ref)},
             "report_only": {f"{key(m, 'base')} vs {ref}": compare(key(m, "base"), ref) for m in MODELS if m not in ("mlp", PRE[0])},
             "robustness": {f"{key(m, s)} vs {key(m, 'base')}": compare(key(m, s), key(m, "base")) for m in MODELS if m != "mlp_pp64" for s in sets if s != "base"}}
    for k, h in head.items():
        be = [e for e in h["best_epoch"] if e is not None]
        print(f"{k:22s} ({h['n_features']:3d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"])
              + (f"  best epochs {be}" if be else ""))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + ("  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}" if g != "robustness" else ""))
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "per_player_more": a.per_player_more, "inner_frac": INNER_FRAC, "inner_questions": inner_sizes,
           "eval_every": EVAL_EVERY, "patience": PATIENCE, "k_tsfel": K_TSFEL, "noise_dim": NOISE_DIM, "noise_seed": a.seed, "tsfel_cache": None if tables is None else str(a.tsfel_cache),
           "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp, "models": list(MODELS), "sets": sets, "scores": head, "comparisons": comps,
           "min_diff": l1b_cv.MIN_DIFF, "seconds": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
