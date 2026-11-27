"""Section 4M M50 (2026-11-07; GPT review of ct9 / ct10 / ct12): which file wins for every test class distribution the unlabelled test
predictions allow? Report only.
Validation questions made test-like: those with an HD >= 2 game dropped (the test has none); n = 5, 10, 20 weighted equally.
Cross-fitted decisions per question: ct9 (the ensemble), ct10 (M46's natural-frequency A|1D specialist), ct12 (M49's balanced specialists at
2D|3D and A|1D). a_c = E[s_x - s_9 | Y = c] (n-balanced), with a player bootstrap SE.
C[i, j] = P(ct9 decides i | Y = j) (n-balanced, same questions); m = ct9's decision shares on the test (all 800, and Private alone).
Feasible set Q = {q in the simplex : |C q - m|_k <= 2 sqrt(m_k (1 - m_k) / N)}; min / max of a^T q over Q by linear programming.
SVD of C: the weakest directions v and a^T v. Pseudo-tests: 400 validation questions from distinct players, n equal, classes uniform or
at the validation shares; 2,000 draws; P(file x beats file y).
  python -I logs/nano4/m50_lp.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m44_calib import logit_fit_auc  # noqa: E402
from m46_a1d import A, D1, xprobs  # noqa: E402
from m46_a1d import fold_in as fold_in_a1d  # noqa: E402
from m49_select import FIXED, fold_in, specialist  # noqa: E402


def decisions(seeds, hd):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in seeds], axis=0)
        mh = np.array([max(hd.get(k, 0.0) for k in ks) for ks in q["game_keys"]])
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "X": xprobs(P, n), "hd2": mh >= 2}
    out = {"y": [], "n": [], "pid": [], "hd2": [], "d9": [], "d10": [], "d12": []}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        sel = lambda u: (D[u]["y"] == A) | (D[u]["y"] == D1)  # noqa: E731
        X = np.concatenate([D[u]["X"][sel(u)] for u in tr]); lab = np.concatenate([(D[u]["y"][sel(u)] == D1).astype(float) for u in tr])
        s10 = logit_fit_auc(X, lab, D[v]["X"], None)
        out["d10"].append(decide(fold_in_a1d(D[v]["P"], s10)[0]))
        out["d12"].append(decide(fold_in(D[v]["P"], {k: specialist(D, tr, k, D[v]["X"]) for k in FIXED})))
        out["d9"].append(decide(D[v]["P"]))
        for k in ("y", "n", "pid", "hd2"):
            out[k].append(D[v][k])
    return {k: np.concatenate(x) for k, x in out.items()}


def nbal_class_mean(x, y, n, c):
    return float(np.mean([x[(y == c) & (n == m)].mean() for m in (5, 10, 20)]))


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); Pt = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float)
    dt = decide(Pt); pub = pr["question_id"].str.startswith("pub").to_numpy()
    rep = {}
    for seeds in ((0, 1, 2), (3, 4, 5)):
        R = decisions(seeds, hd); keep = ~R["hd2"]
        y, n, pid = R["y"][keep], R["n"][keep], R["pid"][keep]
        s = {f: qscore(R[f"d{f}"][keep], y) for f in (9, 10, 12)}
        C = np.zeros((10, 10))
        for j in range(10):
            C[:, j] = np.mean([np.bincount(R["d9"][keep][(y == j) & (n == m)], minlength=10) / max(((y == j) & (n == m)).sum(), 1) for m in (5, 10, 20)], axis=0)
        val_prior = np.bincount(y, minlength=10) / len(y)
        res = {"questions": int(keep.sum())}
        rng = np.random.default_rng(0); players = np.unique(pid); idx = {p: np.flatnonzero(pid == p) for p in players}
        A_vec = {}
        for a_, b_ in ((12, 9), (10, 9), (12, 10)):
            dlt = s[a_] - s[b_]; a = np.array([nbal_class_mean(dlt, y, n, c) for c in range(10)])
            bs = []
            for _ in range(300):
                ii = np.concatenate([idx[p] for p in rng.choice(players, len(players))])
                bs.append([np.mean([dlt[ii][(y[ii] == c) & (n[ii] == m)].mean() for m in (5, 10, 20)]) for c in range(10)])
            se = np.std(bs, axis=0); A_vec[(a_, b_)] = (a, np.array(bs))
            r = {"a": a.tolist(), "se": se.tolist()}
            for prior_name, q in (("uniform", np.full(10, 0.1)), ("validation", val_prior)):
                vals = np.array(bs) @ q
                r[f"at_{prior_name}"] = [float(a @ q), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            for set_name, m_idx in (("test800", np.ones(len(dt), bool)), ("private400", ~pub)):
                m = np.bincount(dt[m_idx], minlength=10) / m_idx.sum(); eps = 2 * np.sqrt(m * (1 - m) / m_idx.sum())
                A_ub = np.vstack([C, -C]); b_ub = np.r_[m + eps, eps - m]
                lo = linprog(a, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 10)), b_eq=[1], bounds=[(0, 1)] * 10)
                hi = linprog(-a, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 10)), b_eq=[1], bounds=[(0, 1)] * 10)
                if lo.success and hi.success:
                    r[f"lp_{set_name}"] = [float(lo.fun), float(-hi.fun)]
                    bs_lo = [float(linprog(bb, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 10)), b_eq=[1], bounds=[(0, 1)] * 10).fun) for bb in np.array(bs)[:100]]
                    bs_hi = [float(-linprog(-bb, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 10)), b_eq=[1], bounds=[(0, 1)] * 10).fun) for bb in np.array(bs)[:100]]
                    r[f"lp_{set_name}_with_a_uncertainty"] = [float(np.percentile(bs_lo, 5)), float(np.percentile(bs_hi, 95))]
                else:
                    r[f"lp_{set_name}"] = None
            res[f"ct{a_}-ct{b_}"] = r
            print(f"seeds {seeds} ct{a_} - ct{b_}: a by class " + " ".join(f"{lv} {x:+.3f}" for lv, x in zip(LEVELS, a)), flush=True)
            print(f"    uniform prior {r['at_uniform'][0]:+.4f} [{r['at_uniform'][1]:+.4f}, {r['at_uniform'][2]:+.4f}]  validation prior {r['at_validation'][0]:+.4f} "
                  f"[{r['at_validation'][1]:+.4f}, {r['at_validation'][2]:+.4f}]  LP over Q(test 800) {r['lp_test800']}  Q(private 400) {r['lp_private400']}  "
                  f"LP incl. a uncertainty (800) {r.get('lp_test800_with_a_uncertainty')}", flush=True)
        U, S_, Vt = np.linalg.svd(C)
        res["singular_values"] = S_.tolist()
        weak = [{"sigma": float(S_[i]), "v": Vt[i].tolist(), **{f"a_dot_v ct{x}-ct{z}": float(A_vec[(x, z)][0] @ Vt[i]) for (x, z) in A_vec}} for i in (9, 8, 7)]
        res["weak_directions"] = weak
        print(f"seeds {seeds}: singular values of C " + " ".join(f"{x:.3f}" for x in S_), flush=True)
        for w in weak:
            print(f"    sigma {w['sigma']:.3f}: v " + " ".join(f"{lv} {x:+.2f}" for lv, x in zip(LEVELS, w["v"])) + "  a.v: " + ", ".join(f"{k.split(' ')[1]} {x:+.3f}" for k, x in w.items() if k.startswith("a_dot")), flush=True)
        # pseudo-tests
        sims = {}
        by = {(c, m): np.flatnonzero((y == c) & (n == m)) for c in range(10) for m in (5, 10, 20)}
        for prior_name, q in (("uniform", np.full(10, 0.1)), ("validation", val_prior)):
            diffs = {k: [] for k in ("ct12-ct9", "ct10-ct9", "ct12-ct10")}
            for _ in range(2000):
                cls = rng.choice(10, 400, p=q); ns = rng.choice((5, 10, 20), 400)
                pick = []; used = set()
                for c, m in zip(cls, ns):
                    cand = by[(c, m)]
                    for _t in range(20):
                        i = cand[rng.integers(len(cand))]
                        if pid[i] not in used:
                            used.add(pid[i]); pick.append(i); break
                pick = np.array(pick)
                for k, (a_, b_) in (("ct12-ct9", (12, 9)), ("ct10-ct9", (10, 9)), ("ct12-ct10", (12, 10))):
                    diffs[k].append(float((s[a_][pick] - s[b_][pick]).mean()))
            sims[prior_name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "p_positive": float(np.mean(np.array(v) > 0))} for k, v in diffs.items()}
            print(f"seeds {seeds} pseudo-tests ({prior_name} prior, 400 questions): " + "  ".join(f"{k} mean {d['mean']:+.4f} sd {d['sd']:.4f} P(>0) {d['p_positive']:.2f}" for k, d in sims[prior_name].items()), flush=True)
        res["pseudo_tests"] = sims
        rep["".join(map(str, seeds))] = res
    Path(ROOT / "results/trunk_seq/es/m50_lp.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
