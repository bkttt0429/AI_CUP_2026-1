#!/usr/bin/env python
"""Architecture v2 section 4I, Stage 1: the end-position proxy ("終局優勢代理") as eight label-free question columns (O),
screened with the MLP h32 of 4B/4D. Folds 1-4; fold 0 is refused.

  build [--partial]   per-game table: the target-view value / score at T, T-10, T-30 (scripts/endgame_proxy.py extract) joined
                      with the target's and the opponent's per-game L1b pmi (posterior_mean_index of the ten sums, the same
                      statistic as the R diff columns) -> results/endgame/game_table.csv; the eight O columns of every training
                      and validation question of folds 1-4 -> results/endgame/o_val{v}.npz; coverage, input versions and
                      distribution diagnostics -> results/endgame/o_quality.json. --partial reports what the finished folds allow.
  mlp --val-fold v    MLP h32 seed 0 on R + HD + O (72) and on R + HD (64), same questions / fold / seed, plus the report-only
                      O + HD (13) -> results/endgame/mlp_val{v}.npz (with 4D's stored MLP0 for a consistency check)
  eval                pre-specified pair MLP(R+HD+O) vs MLP(R+HD): conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0,
                      (4) difference >= 0.005; by n and per level; report-only pairs; strict OOF R² (three folds fit, one evaluated;
                      player-equal and class-balanced weights) -> results/endgame/stage1.json
  smoke               exercises every code path on fold 1's validation questions (tiny MLPs)

O columns (question means of per-game statistics; v = P(target ahead) at the last position T, score = Human SL whiteScore in the
target's view): o_win = share of v_T > 0.5; o_v = mean v_T; o_v3 = mean v over {T, T-10, T-30}; o_human_score = mean
clip(score_T, -30, 30) / 30; o_decisive = mean |v_T - 0.5|; o_decisive_share = share of |v_T - 0.5| > 0.4; o_upset = share of
v_T > 0.5 among the games whose opponent pmi exceeds the target's by more than 0.5 (0.5, neutral, when there is no such game);
o_upset_share = share of such games (0 when none).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.metric import LEVELS, score  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import completed_parts, read_scored_games  # noqa: E402

CACHE, TUNE, EG = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd", ROOT / "results/endgame"
OWN_DIRS = [ROOT / "results/l1b" / d for d in ("scores3_B_hd", "scores3_hd")]
OPP_DIRS = [ROOT / "results/l1b" / d for d in ("scores3_B_opp_hd", "scores3_opp_hd")]
FOLDS = (1, 2, 3, 4)
O_COLS = ["o_win", "o_v", "o_v3", "o_human_score", "o_decisive", "o_decisive_share", "o_upset", "o_upset_share"]
COL_DIFF, COL_OPP, COL_GAP_UP, COL_GAP_DOWN, HD = 56, 57, 62, 63, slice(59, 64)  # R columns: opp - own pmi, opp pmi, hd gaps, HD block
UPSET_MARGIN, DECISIVE, SCORE_CLIP = 0.5, 0.4, 30.0
STR = {"game_key": str, "player_id": str, "rank": str, "color": str}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- build
def locate(dirs, fold):
    hits = [d for d in dirs if completed_parts(d / f"fold{fold}")]
    if len(hits) != 1:
        raise FileNotFoundError(f"fold {fold}: scored in {len(hits)} of {[str(d) for d in dirs]}")
    return hits[0] / f"fold{fold}"


def pmi_of(fold_dir):
    """game_key -> posterior_mean_index of the ten L1b sums (0 = D ... 9 = 6D), as the R diff columns compute it."""
    t = read_scored_games(fold_dir)
    return pd.Series(posterior_mean_index(t[sum_cols(LEVELS)].to_numpy(dtype=float)), index=t["game_key"].to_numpy())


def file_version(path):
    h = hashlib.md5(path.read_bytes()).hexdigest()
    return {"path": str(path.relative_to(ROOT)), "md5": h, "bytes": path.stat().st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}


def game_table(folds):
    parts = []
    for f in folds:
        g = pd.read_csv(EG / f"games_fold{f}.csv", dtype=STR)
        g["fold"] = f
        own, opp = pmi_of(locate(OWN_DIRS, f)), pmi_of(locate(OPP_DIRS, f))
        g["own_pmi"], g["opp_pmi"] = g["game_key"].map(own).to_numpy(), g["game_key"].map(opp).to_numpy()
        if not g["game_key"].is_unique:
            raise ValueError(f"fold {f}: duplicated game keys in the extraction")
        missing = int(g["own_pmi"].isna().sum() + g["opp_pmi"].isna().sum())
        if missing:
            raise ValueError(f"fold {f}: {missing} extracted games without L1b scores")
        bad = {c: int(g[c].isna().sum()) for c in ("v_0", "v_10", "v_30", "score_0") if g[c].isna().any()}
        if bad:
            raise ValueError(f"fold {f}: missing proxy values {bad}")
        if not (((g["v_0"] >= 0) & (g["v_0"] <= 1)).all() and np.isfinite(g["score_0"]).all()):
            raise ValueError(f"fold {f}: proxy values out of range")
        parts.append(g)
        log(f"fold {f}: {len(g)} games, {g.player_id.nunique()} players, last mover ahead {np.mean((g.v_0 > 0.5) == (g.last_mover_is_target == 1)):.3f}")
    return pd.concat(parts, ignore_index=True)


def question_o(table, game_keys):
    """[Q, 8] O columns of questions given as lists of game keys (every key must be in `table`)."""
    row = pd.Series(np.arange(len(table)), index=table["game_key"].to_numpy())
    lens = np.array([len(ks) for ks in game_keys], dtype=int)
    if (lens < 1).any():
        raise ValueError("empty question")
    flat = np.concatenate([np.asarray(ks, dtype=object) for ks in game_keys])
    idx = row.reindex(flat).to_numpy(dtype=float)
    if np.isnan(idx).any():
        raise KeyError(f"{int(np.isnan(idx).sum())} question games are not in the proxy table")
    idx = idx.astype(int)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    v0, v10, v30 = (table[c].to_numpy(dtype=float)[idx] for c in ("v_0", "v_10", "v_30"))
    sc = np.clip(table["score_0"].to_numpy(dtype=float)[idx], -SCORE_CLIP, SCORE_CLIP) / SCORE_CLIP
    own, opp = table["own_pmi"].to_numpy(dtype=float)[idx], table["opp_pmi"].to_numpy(dtype=float)[idx]
    win, conf = (v0 > 0.5).astype(float), np.abs(v0 - 0.5)
    v3 = np.nanmean(np.column_stack([v0, v10, v30]), axis=1)
    weaker = (opp > own + UPSET_MARGIN).astype(float)  # the target is the estimated weaker side

    def seg_mean(a):
        return np.add.reduceat(a, starts) / lens

    n_weak, upset_wins = np.add.reduceat(weaker, starts), np.add.reduceat(weaker * win, starts)
    o_upset = np.where(n_weak > 0, upset_wins / np.maximum(n_weak, 1.0), 0.5)
    O = np.column_stack([seg_mean(win), seg_mean(v0), seg_mean(v3), seg_mean(sc), seg_mean(conf), seg_mean((conf > DECISIVE).astype(float)),
                         o_upset, n_weak / lens])
    assert O.shape == (len(lens), len(O_COLS)) and np.isfinite(O).all()
    return O


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def residual(a, on):
    A = np.column_stack([np.ones(len(on)), on])
    beta, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ beta


def quality(O, y, X):
    """Distribution and association diagnostics of the O columns of one question set (labels used only here, for reporting)."""
    own = X[:, COL_OPP] - X[:, COL_DIFF]
    ry = residual(y.astype(float), own)
    out = {"questions": int(len(y)), "nan": int(np.isnan(O).sum()), "neutral_upset_share": float((O[:, O_COLS.index("o_upset_share")] == 0).mean()),
           "corr_own_pmi_y": corr(own, y), "columns": {}}
    for j, c in enumerate(O_COLS):
        o = O[:, j]
        out["columns"][c] = {"mean": float(o.mean()), "std": float(o.std()), "quantiles_5_25_50_75_95": [float(x) for x in np.quantile(o, [0.05, 0.25, 0.5, 0.75, 0.95])],
                             "corr_y": corr(o, y), "corr_own_pmi": corr(o, own), "partial_corr_y_given_own": corr(residual(o, own), ry),
                             "mean_by_level": {LEVELS[k]: float(o[y == k].mean()) for k in range(10) if (y == k).any()}}
    return out


def cmd_build(a):
    have = [f for f in FOLDS if (EG / f"games_fold{f}.csv").exists()]
    if not have:
        raise SystemExit("no extraction CSV in results/endgame")
    table = game_table(have)
    table.to_csv(EG / "game_table.csv", index=False)
    log(f"game table: {len(table)} games of folds {have}; upset-eligible games (opp pmi > own + {UPSET_MARGIN}) {np.mean(table.opp_pmi > table.own_pmi + UPSET_MARGIN):.3f}")
    report = {"folds": have, "inputs": [file_version(EG / f"games_fold{f}.csv") for f in have],
              "scores": {f: {"own": str(locate(OWN_DIRS, f).relative_to(ROOT)), "opp": str(locate(OPP_DIRS, f).relative_to(ROOT))} for f in have},
              "games": int(len(table)), "columns": O_COLS, "upset_margin": UPSET_MARGIN, "decisive": DECISIVE, "score_clip": SCORE_CLIP, "per_fold": {}}
    keys_in = set(table["game_key"])
    complete = True
    for v in FOLDS:
        J = json.loads((CACHE / f"r_val{v}.json").read_text())
        z = np.load(CACHE / f"r_val{v}.npz")
        rep, O = {}, {}
        for split in ("train", "val"):
            ks = J[split]["game_keys"]
            covered = np.array([all(k in keys_in for k in q) for q in ks])
            rep[split] = {"questions": int(len(ks)), "covered": float(covered.mean())}
            if covered.all():
                O[split] = question_o(table, ks)
            elif a.partial and covered.any():
                sub = [q for q, c in zip(ks, covered) if c]
                Xs, ys = z["X_tr" if split == "train" else "X_va"][covered], z["y_tr" if split == "train" else "y_va"][covered].astype(int)
                rep[split]["quality_partial"] = quality(question_o(table, sub), ys, Xs.astype(float))
        if "train" in O and "val" in O:
            np.savez(EG / f"o_val{v}.npz", O_tr=O["train"].astype(np.float32), O_va=O["val"].astype(np.float32))
            rep["val"]["quality"] = quality(O["val"], z["y_va"].astype(int), z["X_va"].astype(float))
            rep["written"] = f"results/endgame/o_val{v}.npz"
        else:
            complete = False
        report["per_fold"][str(v)] = rep
        q = rep["val"].get("quality") or rep["val"].get("quality_partial")
        line = f"val fold {v}: train covered {rep['train']['covered']:.3f}, val covered {rep['val']['covered']:.3f}"
        if q:
            line += "; val O corr with y / own pmi / partial: " + " ".join(
                f"{c} {q['columns'][c]['corr_y']:+.3f}/{q['columns'][c]['corr_own_pmi']:+.3f}/{q['columns'][c]['partial_corr_y_given_own']:+.3f}" for c in O_COLS)
            line += f"; neutral o_upset share {q['neutral_upset_share']:.3f}"
        log(line)
    report["complete"] = complete
    (EG / "o_quality.json").write_text(json.dumps(report, indent=1, default=float))
    log("build done" if complete else "build partial (not every fold is extracted yet)")


# ------------------------------------------------------------------------------------------------------------------- mlp
def feature_sets(X_tr, X_va, O_tr, O_va):
    return {"MLP": (X_tr, X_va), "MLP+O": (np.hstack([X_tr, O_tr]), np.hstack([X_va, O_va])), "O+HD": (np.hstack([O_tr, X_tr[:, HD]]), np.hstack([O_va, X_va[:, HD]]))}


def cmd_mlp(a):
    v = a.val_fold
    if v not in FOLDS:
        raise SystemExit("folds 1-4 only")
    z, o = np.load(CACHE / f"r_val{v}.npz"), np.load(EG / f"o_val{v}.npz")
    y_tr, y_va = z["y_tr"].astype(int), z["y_va"].astype(int)
    q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
    sets = feature_sets(z["X_tr"].astype(float), z["X_va"].astype(float), o["O_tr"].astype(float), o["O_va"].astype(float))
    P = {"MLP0": np.load(TUNE / f"tune_val{v}.npz")["mlp0"]}
    for name, (A, B) in sets.items():
        P[name] = MLPModel(hidden=32, seed=0).fit(A, y_tr).predict_proba(B)
        log(f"fold {v}: {name} ({A.shape[1]} features) {evaluate(P[name], y_va, q)[0]['balanced_mean_over_n']:.4f}")
    np.savez(EG / f"mlp_val{v}.npz", **{k: p.astype(np.float32) for k, p in P.items()})
    log(f"fold {v}: MLP0 (4D stored) {evaluate(P['MLP0'], y_va, q)[0]['balanced_mean_over_n']:.4f}; mlp done")


# ------------------------------------------------------------------------------------------------------------------ eval
R2_SPECS = {"own": ("own",), "opp+gap": ("opp", "gap"), "own+opp+gap": ("own", "opp", "gap"), "own+opp+gap+O": ("own", "opp", "gap", "O")}


def design(X, O, cols):
    own, opp, gap = X[:, COL_OPP] - X[:, COL_DIFF], X[:, COL_OPP], X[:, COL_GAP_UP] - X[:, COL_GAP_DOWN]
    parts = {"own": own[:, None], "opp": opp[:, None], "gap": gap[:, None], "O": O}
    return np.column_stack([np.ones(len(X))] + [parts[c] for c in cols])


def weights(kind, y, pid):
    if kind == "player_equal":
        w = 1.0 / pd.Series(pid).map(pd.Series(pid).value_counts()).to_numpy(dtype=float)
    else:
        w = 1.0 / np.bincount(y, minlength=10)[y].astype(float)
    return w / w.mean()


def weighted_r2(A_tr, y_tr, w_tr, A_va, y_va, w_va):
    sw = np.sqrt(w_tr)
    beta, *_ = np.linalg.lstsq(A_tr * sw[:, None], y_tr * sw, rcond=None)
    pred = A_va @ beta
    ybar = np.average(y_va, weights=w_va)
    return float(1 - np.sum(w_va * (y_va - pred) ** 2) / np.sum(w_va * (y_va - ybar) ** 2)), beta


def oof_r2(folds=FOLDS):
    out = {}
    for v in folds:
        z, o = np.load(CACHE / f"r_val{v}.npz"), np.load(EG / f"o_val{v}.npz")
        J = json.loads((CACHE / f"r_val{v}.json").read_text())
        y_tr, y_va = z["y_tr"].astype(int), z["y_va"].astype(int)
        X_tr, X_va, O_tr, O_va = z["X_tr"].astype(float), z["X_va"].astype(float), o["O_tr"].astype(float), o["O_va"].astype(float)
        pid_tr, pid_va = np.array(J["train"]["player_id"], dtype=str), np.array(J["val"]["player_id"], dtype=str)
        for kind in ("player_equal", "class_balanced"):
            w_tr, w_va = weights(kind, y_tr, pid_tr), weights(kind, y_va, pid_va)
            for name, cols in R2_SPECS.items():
                r2, beta = weighted_r2(design(X_tr, O_tr, cols), y_tr, w_tr, design(X_va, O_va, cols), y_va, w_va)
                out.setdefault(kind, {}).setdefault(name, {"per_fold": {}, "beta_fold": {}})
                out[kind][name]["per_fold"][str(v)] = r2
                out[kind][name]["beta_fold"][str(v)] = [float(b) for b in beta]
    for kind in out:
        for name in out[kind]:
            out[kind][name]["mean"] = float(np.mean(list(out[kind][name]["per_fold"].values())))
    return out


def cmd_eval(a):
    rows, per_fold, per_class = [], {}, {}
    for v in FOLDS:
        z = np.load(CACHE / f"r_val{v}.npz")
        q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
        y = z["y_va"].astype(int)
        P = dict(np.load(EG / f"mlp_val{v}.npz"))
        S = {}
        for k, p in P.items():
            m, s = evaluate(np.asarray(p, dtype=float), y, q)
            S[k] = s
            per_fold.setdefault(k, []).append(balanced_mean_over_n(s, y, q["n_games"].to_numpy()))
            per_class.setdefault(k, []).append(m["per_class"])
        rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
        log(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in P))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
    models = [c for c in t.columns if c not in ("y", "pid", "n")]
    out = {"scores": {}, "comparisons": {}, "per_class": {}}
    for k in models:
        out["scores"][k] = {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": {s: balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)) for s in SIZES}}
        out["per_class"][k] = {lvl: float(np.mean([pc[lvl] for pc in per_class[k]])) for lvl in LEVELS}
        print(f"{k:8s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k]) + "  by n " + " ".join(f"{s}: {out['scores'][k]['by_n'][s]:.4f}" for s in SIZES), flush=True)
    for group, pairs in (("pre_specified", [("MLP+O", "MLP")]), ("report_only", [("O+HD", "MLP"), ("MLP", "MLP0"), ("MLP+O", "MLP0")])):
        out["comparisons"][group] = {}
        for x, w in pairs:
            dd = paired_diff(t[x].to_numpy(), t[w].to_numpy(), y, pid, n, n=5000)
            pf = [p - r for p, r in zip(per_fold[x], per_fold[w])]
            cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
            lv = {lvl: out["per_class"][x][lvl] - out["per_class"][w][lvl] for lvl in LEVELS}
            out["comparisons"][group][f"{x} vs {w}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values()), "per_level_diff": lv}
            print(f"[{group}] {x} vs {w}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
                  + "  by n " + " ".join(f"{s}: {dd['by_n'][s]:+.4f}" for s in SIZES) + f"  -> {'PASS' if all(cond.values()) else 'no'}", flush=True)
            print("    per level " + " ".join(f"{lvl} {d:+.3f}" for lvl, d in lv.items()), flush=True)
    out["oof_r2"] = oof_r2()
    for kind in out["oof_r2"]:
        print(f"OOF R2 ({kind}): " + "  ".join(f"{name} {r['mean']:.3f} [" + " ".join(f"{x:.3f}" for x in r["per_fold"].values()) + "]" for name, r in out["oof_r2"][kind].items()), flush=True)
    if (EG / "o_quality.json").exists():
        out["inputs"] = json.loads((EG / "o_quality.json").read_text()).get("inputs")
    (EG / "stage1.json").write_text(json.dumps(out, indent=1, default=float))
    log("eval done")


# ----------------------------------------------------------------------------------------------------------------- smoke
def cmd_smoke(a):
    table = game_table([1])
    J = json.loads((CACHE / "r_val1.json").read_text())
    z = np.load(CACHE / "r_val1.npz")
    ks, y, X = J["val"]["game_keys"], z["y_va"].astype(int), z["X_va"].astype(float)
    pid = np.array(J["val"]["player_id"], dtype=str)
    O = question_o(table, ks)
    # direct recomputation of two questions
    for i in (0, len(ks) // 2):
        g = table.set_index("game_key").loc[ks[i]]
        assert abs(O[i, 0] - (g.v_0 > 0.5).mean()) < 1e-9 and abs(O[i, 3] - (np.clip(g.score_0, -30, 30) / 30).mean()) < 1e-9
        m = g.opp_pmi > g.own_pmi + UPSET_MARGIN
        assert abs(O[i, 6] - ((g.v_0[m] > 0.5).mean() if m.any() else 0.5)) < 1e-9 and abs(O[i, 7] - m.mean()) < 1e-9
    qual = quality(O, y, X)
    assert qual["nan"] == 0
    log("quality: " + " ".join(f"{c} corr_y {qual['columns'][c]['corr_y']:+.3f}" for c in O_COLS) + f"; neutral upset {qual['neutral_upset_share']:.3f}")
    # tiny MLPs on a pseudo split of the validation questions (players disjoint), every eval path
    players = np.array(sorted(set(pid)))
    half = set(np.random.RandomState(0).permutation(players)[: len(players) // 2])
    tr = np.array([p in half for p in pid])
    sets = feature_sets(X[tr], X[~tr], O[tr], O[~tr])
    q = pd.DataFrame({"player_id": pid[~tr], "n_games": np.array(J["val"]["n_games"])[~tr]})
    P = {}
    for name, (A, B) in sets.items():
        P[name] = MLPModel(hidden=32, seed=0, epochs=20).fit(A, y[tr]).predict_proba(B)
        log(f"smoke {name}: {A.shape[1]} features, score {evaluate(P[name], y[~tr], q)[0]['balanced_mean_over_n']:.4f}")
    dd = paired_diff(evaluate(P["MLP+O"], y[~tr], q)[1], evaluate(P["MLP"], y[~tr], q)[1], y[~tr], pid[~tr], q["n_games"].to_numpy(), n=200)
    log(f"smoke paired diff {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]")
    for kind in ("player_equal", "class_balanced"):
        w_tr, w_va = weights(kind, y[tr], pid[tr]), weights(kind, y[~tr], pid[~tr])
        r2s = {name: weighted_r2(design(X[tr], O[tr], cols), y[tr], w_tr, design(X[~tr], O[~tr], cols), y[~tr], w_va)[0] for name, cols in R2_SPECS.items()}
        log(f"smoke R2 ({kind}): " + " ".join(f"{k} {r:.3f}" for k, r in r2s.items()))
    log("SMOKE OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--partial", action="store_true")
    m = sub.add_parser("mlp")
    m.add_argument("--val-fold", type=int, required=True)
    sub.add_parser("eval")
    sub.add_parser("smoke")
    a = ap.parse_args()
    EG.mkdir(parents=True, exist_ok=True)
    {"build": cmd_build, "mlp": cmd_mlp, "eval": cmd_eval, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
