#!/usr/bin/env python
"""S00 / S01 (issue #52, architecture v2 Stage 2): the set model r3_pma on R-hdpatch per-game tokens against the 64-dim MLP.

Per-game token: the ladder's staged Human SL vector (gamevec: 4 move-number stages x 75 fields = 300, from
scripts/ladder_build_vectors.py on results/scores_hd) + log1p(target moves) [the ladder's token, 301] + the L1b sums of the
target's moves (10) + of the opponent's moves (10) + the game's pmi difference (opponent - own, 1) + the game's handicap
features (aicup.handicap.game_hd_features, 5): 327 columns; qglob as in the ladder. S01: r3_pma (SetTransformer without SABs:
game encoder -> PMA pooling -> MLP) trained with the 10/19 ladder settings (settrain.TrainCfg via ladder.train_cfg: inner
early stopping on 10% of each rank's players, 4 evaluations per epoch, patience 8; 3 seeds) as a delta on cross-fitted base
logits of S00, the loss on whole questions, 10-class probabilities out. S00 (the control): the same per-game signals through
fixed statistics into the current MLP h32, i.e. the 64 fused_opp_diff_hd features (scripts/hd_check.py) -- the set model sees
no signal the control lacks. Same questions, sampling and seeds as hd_check (folds 1-4 leave-one-out, 32 training
questions per player, pilot validation questions); player bootstrap 5,000. Pre-specified: S01 vs S00 (conditions (1) CI lower
bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005), S01's per-question score being the mean over seeds; reported per fold,
per n, per rank and per HD bucket, and per seed. Fold 0 is not used. Out of scope: SABs, cross-attention, CTM, TCN, wiring
r3_pma into the submission.

  python -I scripts/ladder_build_vectors.py --folds 1 2 3 4 --scores results/scores_hd --out results/ladder/vectors_hd
  python -I scripts/s01_check.py --vectors results/ladder/vectors_hd --l1b results/l1b/scores3_hd results/l1b/scores3_B_hd \\
      --opp results/l1b/scores3_opp_hd results/l1b/scores3_B_opp_hd --hsl results/scores_hd --hd-table results/data/hd_table.csv \\
      --models-dir results/models/s01 --out results/seq/s01_check.json
"""
import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.aggregators import SetTransformer, n_params  # noqa: E402
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES  # noqa: E402
from aicup.handicap import HD_FEATURE_NAMES, game_hd_features, read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, load_vectors, to_json, train_cfg  # noqa: E402
from aicup.ladder_data import pack_questions  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.setmodel_io import save_set_model  # noqa: E402
from aicup.settrain import crossfit_base_logits, fit_set_model, predict_set_model  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402

RUNG = "r3_pma"
EXTRA_NAMES = [f"l1b_sum_{lv}" for lv in LEVELS] + [f"opp_sum_{lv}" for lv in LEVELS] + ["opp_minus_own_pmi"] + list(HD_FEATURE_NAMES)
PRE = [("S01", "S00")]
BUCKETS = ("no_hd", "sente_only", "stones")


def game_extra(vec, l1b, opp, hd):
    """[G, 26] aligned with vec["game_key"]: the L1b sums of the target's and the opponent's moves, the game's opp - own pmi and
    its five handicap features (KeyError when a game lacks a row)."""
    cols = sum_cols(LEVELS)
    li, oi, hi = l1b.set_index("game_key"), opp.set_index("game_key"), hd.drop_duplicates("game_key").set_index("game_key")
    keys = vec["game_key"].tolist()
    L, O = li.loc[keys, cols].to_numpy(dtype=float), oi.loc[keys, cols].to_numpy(dtype=float)
    diff = posterior_mean_index(np.ascontiguousarray(O)) - posterior_mean_index(np.ascontiguousarray(L))
    H = game_hd_features(hi.loc[keys, "hd"].to_numpy(dtype=float), hi.loc[keys, "color"].to_numpy(dtype=str))
    return np.hstack([L, O, diff[:, None], H]).astype(np.float32)


def pack_tokens(questions, vec, extra):
    """ladder_data.pack_questions on the staged vectors with `extra` [G, 26] appended to every game's token."""
    packed = pack_questions(questions, vec, "staged", True)
    Q, M, D = packed["tokens"].shape
    ext = np.zeros((Q, M, extra.shape[1]), dtype=np.float32)
    for q, ix in enumerate(packed["idx"]):
        ext[q, :len(ix)] = extra[ix]
    packed["tokens"] = np.concatenate([packed["tokens"], ext], axis=2)
    return packed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vectors", required=True, help="scripts/ladder_build_vectors.py's directory built from the _hd Human SL scores")
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores_hd"))
    ap.add_argument("--hd-table", required=True, help="scripts/hd_table.py's CSV")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="override the rung's TrainCfg epochs (tests)")
    ap.add_argument("--batch", type=int, default=None, help="override the rung's TrainCfg batch (tests)")
    ap.add_argument("--models-dir", default=None, help="save every split's and seed's set model here (setmodel_io)")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    t0 = time.time()
    try:
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        hd_all = read_hd_table(a.hd_table)
        vec, fold_of = load_vectors(a.vectors, folds)
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            if set(vec["game_key"][fold_of == f].tolist()) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the vectors in {a.vectors} cover other games than {a.hsl}")
            hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
            if len(set(hsl["game_key"]) - set(hd["game_key"])):
                raise ValueError(f"fold {f}: scored games missing from {a.hd_table}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
        extra = game_extra(vec, pd.concat([data[f]["l1b"] for f in folds]), pd.concat([data[f]["opp"] for f in folds]), hd_all)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    cfg0 = train_cfg(RUNG, 0, a.device)
    if a.epochs is not None or a.batch is not None:
        cfg0 = replace(cfg0, **{k: v for k, v in (("epochs", a.epochs), ("batch", a.batch)) if v is not None})
    rows, per_split, per_fold = [], {}, {"S00": [], "S01": [], **{f"S01_s{s}": [] for s in a.seeds}}
    for v in folds:
        t1 = time.time()
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng).assign(fold=f) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, H_va = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        X_tr, X_va = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        # S00: the control; its cross-fitted logits are the base of S01 (the validation base is S00 itself)
        P00 = MLPModel(hidden=32).fit(X_tr, y_tr).predict_proba(X_va)
        fold_col = train_q["fold"].to_numpy()
        tr_folds = sorted(set(fold_col.tolist()))
        by_fold, base_val = crossfit_base_logits(lambda: MLPModel(hidden=32), {f: X_tr[fold_col == f] for f in tr_folds},
                                                 {f: y_tr[fold_col == f] for f in tr_folds}, X_va)
        base_tr = np.concatenate([by_fold[f] for f in tr_folds])
        tok_tr, tok_va = pack_tokens(train_q, vec, extra), pack_tokens(d["val"], vec, extra)
        res00, s00 = evaluate(P00, y_va, d["val"])
        per_fold["S00"].append(res00["balanced_mean_over_n"])
        seed_scores, info = [], {"S00": res00["balanced_mean_over_n"], "n_train": int(len(train_q)), "token_dim": int(tok_tr["tokens"].shape[2]), "seeds": {}}
        for seed in a.seeds:
            cfg = replace(cfg0, seed=int(seed))
            fitted = fit_set_model(lambda dim: SetTransformer(dim, layers=0), tok_tr, base_tr, cfg)
            if not np.isfinite(fitted.losses).all():
                raise FloatingPointError(f"{RUNG} seed {seed} split {v}: non-finite training loss")
            P01 = predict_set_model(fitted, tok_va, base_val)
            if not np.isfinite(P01).all():
                raise FloatingPointError(f"{RUNG} seed {seed} split {v}: non-finite probabilities")
            res01, s01 = evaluate(P01, y_va, d["val"])
            seed_scores.append(s01)
            per_fold[f"S01_s{seed}"].append(res01["balanced_mean_over_n"])
            inner = fitted.inner_ce
            info["seeds"][str(seed)] = {"balanced_mean_over_n": res01["balanced_mean_over_n"], "nll_balanced": res01["nll_balanced"], "best_epoch": fitted.best_epoch,
                                       "inner_ce_init": inner[0] if inner else None, "inner_ce_best": inner[fitted.best_eval] if inner else None,
                                       "stopped_early": fitted.stopped_early, "n_params": n_params(fitted.model), "train_loss": fitted.losses}
            if a.models_dir:
                save_set_model(fitted, RUNG, int(seed), Path(a.models_dir) / f"{RUNG}_v{v}_s{seed}.pt",
                               extra={"experiment": "s01_check", "split": int(v), "token_extra": EXTRA_NAMES, "vectors": str(a.vectors)})
            print(f"  split {v} seed {seed}: S01 {res01['balanced_mean_over_n']:.4f} (best epoch {fitted.best_epoch}, inner CE {info['seeds'][str(seed)]['inner_ce_init']} -> "
                  f"{info['seeds'][str(seed)]['inner_ce_best']})", flush=True)
        s01_mean = np.mean(seed_scores, axis=0)
        per_fold["S01"].append(balanced_mean_over_n(s01_mean, y_va, n_va))
        info["S01"] = per_fold["S01"][-1]
        info["wall_s"] = time.time() - t1
        per_split[str(v)] = info
        print(f"fold {v} ({len(train_q)} training questions, tokens {info['token_dim']}): S00 {per_fold['S00'][-1]:.4f}  S01 {per_fold['S01'][-1]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, "bucket": hd_check.bucket_of(H_va),
                                  "S00": s00, "S01": s01_mean, **{f"S01_s{s}": sc for s, sc in zip(a.seeds, seed_scores)}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    labels = ["S00", "S01", *[f"S01_s{s}" for s in a.seeds]]
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in labels}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}
        by_n = {str(m): paired_diff(t[x].to_numpy()[n == m], t[z].to_numpy()[n == m], y[n == m], pid[n == m], n[n == m], n=a.n_boot) for m in SIZES if (n == m).any()}
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values()), "by_n": by_n}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE},
             "per_seed": {f"S01_s{s} vs S00": compare(f"S01_s{s}", "S00") for s in a.seeds}}
    buckets = {b: {"questions": int((t["bucket"] == b).sum()), **{s: float(t.loc[t["bucket"] == b, s].mean()) if (t["bucket"] == b).any() else None for s in ("S00", "S01")}}
               for b in BUCKETS}
    for s in labels:
        h = head[s]
        print(f"{s:8s} {h['balanced_mean_over_n']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    for b, r in buckets.items():
        print(f"bucket {b:11s} {r['questions']:5d} questions  S00 {r['S00'] if r['S00'] is None else round(r['S00'], 4)}  S01 {r['S01'] if r['S01'] is None else round(r['S01'], 4)}")
    out = {"rung": RUNG, "folds": folds, "seeds": a.seeds, "per_player": a.per_player, "n_boot": a.n_boot, "min_diff": l1b_cv.MIN_DIFF, "train_cfg": asdict(cfg0),
           "token_names": {"ladder": "staged 300 + log1p(n_moves)", "extra": EXTRA_NAMES}, "control_features": list(FUSED_OPP_DIFF_HD_FEATURE_NAMES),
           "vectors": a.vectors, "hd_table": a.hd_table, "models_dir": a.models_dir, "per_split": per_split, "scores": head, "comparisons": comps, "buckets": buckets,
           "wall_s": time.time() - t0, **environment(a.device)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
