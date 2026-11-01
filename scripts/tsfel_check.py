#!/usr/bin/env python
"""Folds 1-4 CV: do TSFEL's descriptors of the per-move score series add to fused_opp_diff? (issue #36)

The user's ask: let TSFEL (Time Series Feature Extraction Library) generate the per-game features instead of hand-picked
shape statistics (2.16) or a histogram (2.20). aicup.seqtsfel runs TSFEL's statistical, temporal, spectral and fractal
descriptors (about 120 per series) on a game's pmi_t and s_t series; the question-level feature is the nanmean over the
question's games. Because the MLP degraded whenever a block of extra columns was appended (2.16, 2.20), the block is
standardised with the training questions' statistics, NaN imputed by the training mean, and reduced with a PCA fitted on
the training questions only: fused_opp_diff_tsfel<k> for each --pca k (pre-specified k = 16), plus fused_opp_diff_tsfel_all
(the whole standardised block). Protocol as scripts/l1b_opp_check.py (pilot validation questions, sample_questions per
player with RandomState(42 + v), MLPModel(hidden=32), player bootstrap); fused_opp_diff must reproduce 0.6916.
Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_tsfel16 vs
fused_opp_diff. Reported only: the other k and tsfel_all vs fused_opp_diff; and, as in 2.15 step 0, a per-game
class-balanced softmax regression on folds 1-4 (train three, test one) with the mean alone (pmi_Mean, log n_moves)
against mean + the PCA-16 TSFEL block: its balanced-accuracy / NLL gain is the game-level information TSFEL adds beyond
the mean. The per-fold TSFEL tables are cached as --cache-dir/fold<F>.npz (about 11 ms per series to build; npz files
under results/seq are not committed). Fold 0 is refused.

  python -I scripts/tsfel_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --workers 12 --out results/seq/tsfel_check.json
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
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel, SoftmaxModel, class_weights  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.seqtsfel import feature_columns, fold_tsfel_table, load_tsfel_table, pca_apply, pca_fit, question_tsfel, save_tsfel_table, standardise_impute  # noqa: E402
import error_analysis as ea  # noqa: E402
import l1b_cv  # noqa: E402
import seq_diag  # noqa: E402

K_PRE = 16
BASE = "fused_opp_diff"


def set_name(k):
    return f"{BASE}_tsfel{k}"


def tsfel_tables(where, folds, cache_dir, workers, rebuild, log=print):
    """{fold: table} from the cache (fold<F>.npz) or built from the per-move arrays and cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in folds:
        path = cache_dir / f"fold{f}.npz"
        t0 = time.time()
        if path.exists() and not rebuild:
            out[f] = load_tsfel_table(path)
            log(f"fold {f}: TSFEL table cached ({len(out[f])} games, {len(feature_columns(out[f]))} features) from {path}", flush=True)
        else:
            out[f] = fold_tsfel_table(Path(where[f]) / f"fold{f}", workers=workers)
            save_tsfel_table(path, out[f])
            log(f"fold {f}: TSFEL table built ({len(out[f])} games, {len(feature_columns(out[f]))} features) in {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


def tsfel_blocks(tables, cols, train_q, val_q, pcas):
    """Standardised TSFEL question blocks: {"all": (Z_tr, Z_va), k: (P_tr, P_va)} plus the PCA diagnostics."""
    T_tr, T_va = question_tsfel(tables["train"], train_q, cols), question_tsfel(tables["val"], val_q, cols)
    Z_tr, Z_va, keep = standardise_impute(T_tr, T_va)
    blocks, diag = {"all": (Z_tr, Z_va)}, {"kept_columns": int(len(keep)), "pca": {}}
    for k in pcas:
        pca = pca_fit(Z_tr, k)
        blocks[k] = (pca_apply(pca, Z_tr), pca_apply(pca, Z_va))
        diag["pca"][str(k)] = {"components": int(pca["components"].shape[1]), "explained": float(sum(pca["explained"]))}
    return blocks, diag


def game_level(tables, folds, k=K_PRE):
    """Per-game softmax regression (as scripts/seq_diag.py): mean only (pmi_Mean, log n_moves) vs mean + the PCA-k TSFEL block
    fitted on the training games; balanced accuracy and NLL per validation fold and their gains."""
    results = {"mean_only": [], "mean_plus_tsfel": []}
    for v in folds:
        tr = pd.concat([tables[f] for f in folds if f != v], ignore_index=True)
        te = tables[v]
        cols = [c for c in feature_columns(tr) if c in te.columns]
        y_tr, y_te = tr["rank"].map(RANK_TO_IDX).to_numpy(), te["rank"].map(RANK_TO_IDX).to_numpy()
        base_tr = np.column_stack([tr["pmi_Mean"].to_numpy(dtype=float) if "pmi_Mean" in tr else np.zeros(len(tr)), np.log(np.maximum(tr["n_moves"].to_numpy(dtype=float), 1))])
        base_te = np.column_stack([te["pmi_Mean"].to_numpy(dtype=float) if "pmi_Mean" in te else np.zeros(len(te)), np.log(np.maximum(te["n_moves"].to_numpy(dtype=float), 1))])
        base_tr, base_te = np.nan_to_num(base_tr), np.nan_to_num(base_te)
        Z_tr, Z_te, _ = standardise_impute(tr[cols].to_numpy(dtype=float), te.reindex(columns=cols).to_numpy(dtype=float))
        pca = pca_fit(Z_tr, k)
        X = {"mean_only": (base_tr, base_te), "mean_plus_tsfel": (np.hstack([base_tr, pca_apply(pca, Z_tr)]), np.hstack([base_te, pca_apply(pca, Z_te)]))}
        for name, (a, b) in X.items():
            m = SoftmaxModel(l2=1e-3).fit(a, y_tr, w=class_weights(y_tr))
            acc, nll = seq_diag._balanced(m.predict_proba(b), y_te)
            results[name].append({"fold": v, "balanced_acc": acc, "balanced_nll": nll, "n_games": int(len(te))})
    gain_acc = [b["balanced_acc"] - a["balanced_acc"] for a, b in zip(results["mean_only"], results["mean_plus_tsfel"])]
    gain_nll = [a["balanced_nll"] - b["balanced_nll"] for a, b in zip(results["mean_only"], results["mean_plus_tsfel"])]
    return {"k": k, "per_game_logistic": results, "gain": {"balanced_acc": gain_acc, "balanced_nll": gain_nll,
                                                          "mean_balanced_acc": float(np.mean(gain_acc)), "mean_balanced_nll": float(np.mean(gain_nll))}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--pca", type=int, nargs="+", default=[4, 16, 64], help=f"PCA sizes of the TSFEL block (must include {K_PRE}, the pre-specified one)")
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "seq" / "tsfel"), help="per-fold TSFEL tables (fold<F>.npz)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the TSFEL extraction")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached TSFEL tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    pcas = sorted(set(a.pca))
    if K_PRE not in pcas:
        ap.error(f"--pca must include {K_PRE} (the pre-specified comparison)")
    t0 = time.time()
    try:
        import importlib
        importlib.import_module("tsfel")  # fail early, before the folds are read
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
        tables = tsfel_tables(where, folds, a.cache_dir, a.workers, a.rebuild)
        for f in folds:
            if set(tables[f]["game_key"]) != set(data[f]["hsl"]["game_key"]):
                raise ValueError(f"fold {f}: the TSFEL table covers other games than {a.hsl}; use --rebuild")
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        print(e, file=sys.stderr)
        return 2
    cols = sorted(set().union(*[feature_columns(tables[f]) for f in folds]))
    sets = [BASE] + [set_name(k) for k in pcas] + [f"{BASE}_tsfel_all"]
    rows, per_fold, nll, dims, diags = [], {s: [] for s in sets}, {s: [] for s in sets}, {}, {}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = ea.feature_sets(cat("hsl"), cat("l1b"), cat("opp"), train_q)
        d = data[v]
        X_va, y_va, n_va = ea.feature_sets(d["hsl"], d["l1b"], d["opp"], d["val"])
        blocks, diags[str(v)] = tsfel_blocks({"train": pd.concat([tables[f] for f in folds if f != v], ignore_index=True), "val": tables[v]}, cols, train_q, d["val"], pcas)
        feats = {BASE: (X_tr[BASE], X_va[BASE]), f"{BASE}_tsfel_all": (np.hstack([X_tr[BASE], blocks["all"][0]]), np.hstack([X_va[BASE], blocks["all"][1]]))}
        for k in pcas:
            feats[set_name(k)] = (np.hstack([X_tr[BASE], blocks[k][0]]), np.hstack([X_va[BASE], blocks[k][1]]))
        sc = {}
        for s in sets:
            tr, va = feats[s]
            dims[s] = int(tr.shape[1])
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(tr, y_tr).predict_proba(va), y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in sets) + f"  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in sets}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])), "n_features": dims[s],
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in sets}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{set_name(K_PRE)} vs {BASE}": compare(set_name(K_PRE), BASE)},
             "report_only": {f"{s} vs {BASE}": compare(s, BASE) for s in sets if s not in (BASE, set_name(K_PRE))}}
    gl = game_level(tables, folds, K_PRE)
    for s, h in head.items():
        print(f"{s:24s} ({h['n_features']:3d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    print(f"TSFEL: {len(cols)} feature names over the folds; kept after standardisation per fold " + " ".join(str(diags[str(v)]["kept_columns"]) for v in folds)
          + "; PCA explained " + " ".join(f"k={k}: " + "/".join(f"{diags[str(v)]['pca'][str(k)]['explained']:.2f}" for v in folds) for k in pcas))
    print(f"game-level softmax (k={K_PRE}): mean-only acc " + " ".join(f"{r['balanced_acc']:.4f}" for r in gl["per_game_logistic"]["mean_only"])
          + " | mean+tsfel acc " + " ".join(f"{r['balanced_acc']:.4f}" for r in gl["per_game_logistic"]["mean_plus_tsfel"])
          + f" | gain acc {gl['gain']['mean_balanced_acc']:+.4f} NLL {gl['gain']['mean_balanced_nll']:+.4f}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp,
           "cache_dir": str(a.cache_dir), "pca": pcas, "k_pre": K_PRE, "tsfel_feature_names": cols, "blocks": diags, "scores": head, "comparisons": comps,
           "game_level": gl, "min_diff": l1b_cv.MIN_DIFF, "seconds": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
