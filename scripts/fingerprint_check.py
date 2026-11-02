#!/usr/bin/env python
"""Report only (issue #42 / design-doc discussion): do the feature blocks that failed on NEW players help on SEEN players?

Per validation fold v (folds 1-4): the training folds' players keep 15 of their 20 scored games for the training questions
(sample_questions, 32 per player) and hold out 5; (A) = fold v's pilot questions (new players), (B) = pilot questions
built from the held-out games of the TRAINING players (seen players, unseen games). Feature sets: base = fused_opp_diff
(59), + shape 8 (issue #21), + TSFEL PCA-16 (issue #36, standardisation and PCA fitted on the training questions),
+ 16 pure-noise columns. MLPModel(hidden=32) as everywhere. Reported per fold and pooled: every set's score on (A) and (B)
by question size, and the paired difference to base on (A) and on (B) with a player bootstrap. A block whose gain is much
larger on (B) than on (A) fits the training players themselves. Fold 0 refused.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.fused import question_diff_features  # noqa: E402
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.qeval import SIZES, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.seqshape import SHAPE, fold_shape_table, question_shape  # noqa: E402
from aicup.seqtsfel import feature_columns, load_tsfel_table, pca_apply, pca_fit, question_tsfel, standardise_impute  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("base", "base_shape8", "base_tsfel16", "base_noise16")


def split_players(players, frac, rng):
    tr, ho = [], []
    for p in players.itertuples():
        keys = list(p.game_keys)
        idx = rng.permutation(len(keys))
        k = max(1, int(round(frac * len(keys))))
        ho.append({"player_id": p.player_id, "rank": p.rank, "game_keys": [keys[i] for i in idx[:k]]})
        tr.append({"player_id": p.player_id, "rank": p.rank, "game_keys": [keys[i] for i in idx[k:]]})
    return pd.DataFrame(tr), pd.DataFrame(ho)


def raw_blocks(hsl, l1b, opp, tsfel, shape, cols, q):
    X, y, n = oc.features(hsl, l1b, opp, q)
    base = np.hstack([X["fused_opp"], question_diff_features(l1b, opp, q)])
    return base, question_tsfel(tsfel, q, cols), question_shape(shape, q, cols=SHAPE), y, n


def sizes_present(n):
    return tuple(m for m in SIZES if np.any(n == m)) or SIZES


def by_size(scores, n):
    return {str(m): float(np.mean(scores[n == m])) for m in SIZES if np.any(n == m)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", default=["results/l1b/scores3", "results/l1b/scores3_B"])
    ap.add_argument("--opp", nargs="+", default=["results/l1b/scores3_opp", "results/l1b/scores3_B_opp"])
    ap.add_argument("--hsl", default="results/scores")
    ap.add_argument("--tsfel-cache", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--heldout-frac", type=float, default=0.25, help="scored players carry 20 games each, so 0.25 holds out exactly 5: one 5-game question per seen player")
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-out", default=None, help="per-question scores of every arm (CSV; the diagnostics are recomputable from it)")
    ap.add_argument("--n-boot-d", type=int, default=2000, help="joint player bootstrap draws for D(5) = gain(B) - gain(A)")
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    t0 = time.time()
    where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players_from_scores(hsl, folds_df),
                   "tsfel": load_tsfel_table(Path(a.tsfel_cache) / f"fold{f}.npz"), "shape": fold_shape_table(Path(where[f]) / f"fold{f}")}
        print(f"fold {f}: {len(data[f]['players'])} players, {len(hsl)} games ({time.time() - t0:.0f} s)", flush=True)
    cols = sorted(set().union(*[set(feature_columns(d["tsfel"])) for d in data.values()]))
    rows, per_fold = [], {}
    for v in folds:
        rng = np.random.RandomState(a.seed + v)
        train_f = [f for f in folds if f != v]
        tr_players, ho_players = zip(*[split_players(data[f]["players"], a.heldout_frac, rng) for f in train_f])
        train_q = pd.concat([sample_questions(p, a.per_player, rng) for p in tr_players], ignore_index=True)
        seen_q = pd.concat([pilot_questions(p) for p in ho_players], ignore_index=True)
        if len(seen_q) == 0:
            raise ValueError("no seen-player questions: hold out at least 5 games per player")
        val_q = pilot_questions(data[v]["players"])
        cat = lambda key: pd.concat([data[f][key] for f in train_f], ignore_index=True)  # noqa: E731
        H, L, O, T, S = cat("hsl"), cat("l1b"), cat("opp"), cat("tsfel"), cat("shape")
        b_tr, t_tr, s_tr, y_tr, _ = raw_blocks(H, L, O, T, S, cols, train_q)
        b_se, t_se, s_se, y_se, n_se = raw_blocks(H, L, O, T, S, cols, seen_q)
        d = data[v]
        b_va, t_va, s_va, y_va, n_va = raw_blocks(d["hsl"], d["l1b"], d["opp"], d["tsfel"], d["shape"], cols, val_q)
        z_tr, z_va, _ = standardise_impute(t_tr, t_va)
        _, z_se, _ = standardise_impute(t_tr, t_se)
        pca = pca_fit(z_tr, 16)
        p_tr, p_va, p_se = pca_apply(pca, z_tr), pca_apply(pca, z_va), pca_apply(pca, z_se)
        noise = lambda k: rng.normal(size=(k, 16))  # noqa: E731
        X = {"base": (b_tr, b_va, b_se), "base_shape8": tuple(np.hstack([b, s]) for b, s in ((b_tr, s_tr), (b_va, s_va), (b_se, s_se))),
             "base_tsfel16": tuple(np.hstack([b, p]) for b, p in ((b_tr, p_tr), (b_va, p_va), (b_se, p_se))),
             "base_noise16": tuple(np.hstack([b, noise(len(b))]) for b in (b_tr, b_va, b_se))}
        sc = {}
        for s in SETS:
            m = MLPModel(hidden=32).fit(X[s][0], y_tr)
            _, sc[(s, "new")] = evaluate(m.predict_proba(X[s][1]), y_va, val_q)
            _, sc[(s, "seen")] = evaluate(m.predict_proba(X[s][2]), y_se, seen_q)
        # arm 2: every training question has exactly 5 games (the same players and game pools); base and the TSFEL block only
        train_q5 = pd.concat([sample_questions(p, a.per_player, rng, sizes=[5]) for p in tr_players], ignore_index=True)
        b5_tr, t5_tr, _, y5_tr, _ = raw_blocks(H, L, O, T, S, cols, train_q5)
        z5_tr, z5_va, _ = standardise_impute(t5_tr, t_va)
        _, z5_se, _ = standardise_impute(t5_tr, t_se)
        pca5 = pca_fit(z5_tr, 16)
        X5 = {"base": (b5_tr, b_va, b_se), "base_tsfel16": (np.hstack([b5_tr, pca_apply(pca5, z5_tr)]), np.hstack([b_va, pca_apply(pca5, z5_va)]), np.hstack([b_se, pca_apply(pca5, z5_se)]))}
        sc5 = {}
        for s in X5:
            m = MLPModel(hidden=32).fit(X5[s][0], y5_tr)
            _, sc5[(s, "new")] = evaluate(m.predict_proba(X5[s][1]), y_va, val_q)
            _, sc5[(s, "seen")] = evaluate(m.predict_proba(X5[s][2]), y_se, seen_q)
        train_n = train_q["n_games"].value_counts().sort_index()
        fold_res = {}
        for s in SETS:
            fold_res[s] = {"new_by_n": by_size(sc[(s, "new")], n_va), "seen_by_n": by_size(sc[(s, "seen")], n_se)}
        per_fold[str(v)] = {"train_questions": int(len(train_q)), "new_questions": int(len(val_q)), "seen_questions": int(len(seen_q)),
                            "seen_by_n_count": {str(m): int((n_se == m).sum()) for m in SIZES}, "sets": fold_res,
                            "train_n_games_distribution": {str(k): int(c) for k, c in train_n.items()}, "train5_questions": int(len(train_q5)),
                            "fixed5_arm": {s: {"new_by_n": by_size(sc5[(s, "new")], n_va), "seen_by_n": by_size(sc5[(s, "seen")], n_se)} for s in X5}}
        print(f"fold {v}: training-question n distribution {dict(train_n)}; fixed-5 arm: " + "  ".join(f"{s}: new {np.mean(sc5[(s, 'new')]):.4f} seen {np.mean(sc5[(s, 'seen')]):.4f}" for s in X5), flush=True)
        print(f"fold {v}: train {len(train_q)} q, new {len(val_q)} q, seen {len(seen_q)} q (n5/10/20 {per_fold[str(v)]['seen_by_n_count']}); "
              + "  ".join(f"{s}: new {np.mean(sc[(s, 'new')]):.4f} seen {np.mean(sc[(s, 'seen')]):.4f}" for s in SETS) + f"  ({time.time() - t0:.0f} s)", flush=True)
        for kind, q, y, n in (("new", val_q, y_va, n_va), ("seen", seen_q, y_se, n_se)):
            rows.append(pd.DataFrame({"kind": kind, "fold": v, "player_id": q["player_id"].to_numpy(dtype=str), "y": y, "n": n, **{s: sc[(s, kind)] for s in SETS},
                                      **{f"fixed5_{s}": sc5[(s, kind)] for s in X5}}))
    t = pd.concat(rows, ignore_index=True)
    out = {"folds": folds, "heldout_frac": a.heldout_frac, "seed": a.seed, "per_fold": per_fold, "pooled": {}, "gain": {}}
    for kind in ("new", "seen"):
        tk = t[t["kind"] == kind]
        y, pid, n = tk["y"].to_numpy(), tk["player_id"].to_numpy(dtype=str), tk["n"].to_numpy()
        sizes = sizes_present(n)
        out["pooled"][kind] = {"questions": int(len(tk)), "sizes": list(sizes), **{s: {"by_n": by_size(tk[s].to_numpy(), n)} for s in SETS}}
        for s in SETS[1:]:
            dd = paired_diff(tk[s].to_numpy(), tk["base"].to_numpy(), y, pid, n, sizes=sizes, n=a.n_boot)
            out["gain"].setdefault(s, {})[kind] = dd
            print(f"[{kind}] {s} vs base: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  by n " + " ".join(f"{k}: {val:+.4f}" for k, val in dd["by_n"].items()), flush=True)
    for s in SETS:
        print(f"{s:13s} new by n " + " ".join(f"{k}: {val:.4f}" for k, val in out["pooled"]["new"][s]["by_n"].items())
              + "   seen by n " + " ".join(f"{k}: {val:.4f}" for k, val in out["pooled"]["seen"][s]["by_n"].items()), flush=True)
    # same-n comparison: D(5) = gain on seen players' held-out 5-game questions minus gain on new players' 5-game questions,
    # plain and class-balanced, with a joint bootstrap over global player ids (a player is a new player in its own fold and a
    # seen player in the other three, so the two populations are the same players)
    t5 = t[t["n"] == 5].reset_index(drop=True)
    pid5 = t5["player_id"].to_numpy(dtype=str)
    uniq, inv = np.unique(pid5, return_inverse=True)
    is_b = (t5["kind"] == "seen").to_numpy()
    y5 = t5["y"].to_numpy()
    rng_b = np.random.RandomState(123)
    draws = rng_b.randint(0, len(uniq), size=(a.n_boot_d, len(uniq)))

    def gains(delta, w):
        wa, wb = w * (~is_b), w * is_b
        plain = (np.sum(wb * delta) / max(wb.sum(), 1e-12), np.sum(wa * delta) / max(wa.sum(), 1e-12))
        bal_b = np.mean([np.sum(wb * (y5 == c) * delta) / max(np.sum(wb * (y5 == c)), 1e-12) for c in range(10) if np.sum(wb * (y5 == c)) > 0])
        bal_a = np.mean([np.sum(wa * (y5 == c) * delta) / max(np.sum(wa * (y5 == c)), 1e-12) for c in range(10) if np.sum(wa * (y5 == c)) > 0])
        return plain, (bal_b, bal_a)

    out["same_n5"] = {"questions_new": int((~is_b).sum()), "questions_seen": int(is_b.sum()), "players": int(len(uniq)),
                      "players_in_both": int(len(set(pid5[is_b]) & set(pid5[~is_b])))}
    w1 = np.ones(len(t5))
    for s in [x for x in SETS[1:]] + ["fixed5_base_tsfel16"]:
        base_col = "fixed5_base" if s.startswith("fixed5_") else "base"
        delta = t5[s].to_numpy() - t5[base_col].to_numpy()
        (pb, pa), (bb, ba) = gains(delta, w1)
        boot_plain, boot_bal = [], []
        for d in draws:
            w = np.bincount(d, minlength=len(uniq))[inv].astype(float)
            (qb, qa), (cb, ca) = gains(delta, w)
            boot_plain.append(qb - qa)
            boot_bal.append(cb - ca)
        out["same_n5"][s] = {"gain_new_plain": pa, "gain_seen_plain": pb, "D_plain": pb - pa, "D_plain_ci95": [float(np.percentile(boot_plain, 2.5)), float(np.percentile(boot_plain, 97.5))],
                             "gain_new_balanced": ba, "gain_seen_balanced": bb, "D_balanced": bb - ba, "D_balanced_ci95": [float(np.percentile(boot_bal, 2.5)), float(np.percentile(boot_bal, 97.5))]}
        r = out["same_n5"][s]
        print(f"[same n=5] {s}: gain new {pa:+.4f} seen {pb:+.4f} -> D {pb - pa:+.4f} [{r['D_plain_ci95'][0]:+.4f}, {r['D_plain_ci95'][1]:+.4f}] (plain); "
              f"balanced new {ba:+.4f} seen {bb:+.4f} D {bb - ba:+.4f} [{r['D_balanced_ci95'][0]:+.4f}, {r['D_balanced_ci95'][1]:+.4f}]", flush=True)
    for s in ("fixed5_base", "fixed5_base_tsfel16"):
        out["pooled"]["fixed5_" + s] = {k: by_size(t[t["kind"] == k][s].to_numpy(), t[t["kind"] == k]["n"].to_numpy()) for k in ("new", "seen")}
        print(f"{s:21s} new by n " + " ".join(f"{k}: {val:.4f}" for k, val in out["pooled"]["fixed5_" + s]["new"].items()) + "   seen by n " + " ".join(f"{k}: {val:.4f}" for k, val in out["pooled"]["fixed5_" + s]["seen"].items()), flush=True)
    if a.rows_out:
        t.to_csv(a.rows_out, index=False)
        print(f"wrote {a.rows_out} ({len(t)} rows)", flush=True)
    out.update({"seconds": time.time() - t0, **environment("cpu")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
