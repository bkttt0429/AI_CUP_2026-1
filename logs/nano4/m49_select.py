"""Section 4M M49 (2026-11-07; user: 可以只在少數邊界上使用): M48's 'bal' specialists folded into ct9 at chosen boundaries only.
A: fixed boundaries 2D|3D and A|1D (picked from M48 seeds 0-2), primary on seeds 3-5. B: cross-fit choice -- for each outer fold, an inner
cross-fit over the other three folds; a boundary is used when the specialist's balanced accuracy beats ct9's log-odds by > 0.005.
  python -I logs/nano4/m49_select.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m46_a1d import xprobs  # noqa: E402
from m47_prior import nplain  # noqa: E402
from m48_chain import wlogit  # noqa: E402

FIXED = (LEVELS.index("2D"), LEVELS.index("A"))  # lower level k of the boundaries 2D|3D and A|1D


def specialist(D, train_folds, k, target):
    sel = lambda u: (D[u]["y"] == k) | (D[u]["y"] == k + 1)  # noqa: E731
    X = np.concatenate([D[u]["X"][sel(u)] for u in train_folds]); lab = np.concatenate([(D[u]["y"][sel(u)] == k + 1).astype(float) for u in train_folds])
    w = np.where(lab == 1, 0.5 / lab.mean(), 0.5 / (1 - lab.mean()))
    return 1 / (1 + np.exp(-wlogit(X, lab, w, target)))


def bal_acc(score, lab):
    pr = score > 0
    return 0.5 * ((pr[lab == 1]).mean() + (~pr[lab == 0]).mean())


def fold_in(P, s_by_k):
    Q = P.copy()
    for k, s in s_by_k.items():
        d = decide(Q); route = (d == k) | (d == k + 1); m = Q[:, k] + Q[:, k + 1]
        Q[route, k + 1] = m[route] * s[route]; Q[route, k] = m[route] * (1 - s[route])
    return Q


def evaluate(D, Sn, So, label, seeds):
    y, n, pid = (np.concatenate([D[v][kk] for v in FOLDS]) for kk in ("y", "n", "pid"))
    sn, so = np.concatenate([Sn[v] for v in FOLDS]), np.concatenate([So[v] for v in FOLDS])
    dd = paired_diff(sn, so, y, pid, n, n=5000)
    pf = [balanced_mean_over_n(Sn[v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(So[v], D[v]["y"], D[v]["n"]) for v in FOLDS]
    ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
    r = {**dd, "per_fold": pf, "testlike_plain": nplain(sn, n) - nplain(so, n), "changed": int((sn != so).sum()), "pass_new_rule": bool(ok)}
    print(f"seeds {seeds} {label:14s}: class-balanced {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf)
          + f"  test-like plain {r['testlike_plain']:+.4f}  score changed on {r['changed']} questions" + ("  -> PASS" if ok else "  -> no"), flush=True)
    return r


def run(seeds):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in seeds], axis=0)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "X": xprobs(P, n)}
    out = {}
    SnA, So, SnB, chosen = {}, {}, {}, {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        So[v] = qscore(decide(D[v]["P"]), D[v]["y"])
        SnA[v] = qscore(decide(fold_in(D[v]["P"], {k: specialist(D, tr, k, D[v]["X"]) for k in FIXED})), D[v]["y"])
        use = []
        for k in range(9):  # inner cross-fit on the three training folds
            gains = []
            for w_ in tr:
                inner = [u for u in tr if u != w_]; m = (D[w_]["y"] == k) | (D[w_]["y"] == k + 1); lab = (D[w_]["y"][m] == k + 1).astype(int)
                s = specialist(D, inner, k, D[w_]["X"][m]); lp = np.log(np.clip(D[w_]["P"][m], 1e-9, 1))
                gains.append(bal_acc(np.log(s / (1 - s)), lab) - bal_acc(lp[:, k + 1] - lp[:, k], lab))
            if np.mean(gains) > 0.005:
                use.append(k)
        chosen[str(v)] = [f"{LEVELS[k]}|{LEVELS[k + 1]}" for k in use]
        SnB[v] = qscore(decide(fold_in(D[v]["P"], {k: specialist(D, tr, k, D[v]["X"]) for k in use})), D[v]["y"])
    print(f"seeds {seeds}: cross-fit choice per outer fold {chosen}", flush=True)
    out["A_fixed"] = evaluate(D, SnA, So, "A fixed", seeds); out["B_crossfit"] = evaluate(D, SnB, So, "B cross-fit", seeds); out["B_choice"] = chosen
    return out


def main():
    rep = {"s012": run((0, 1, 2)), "s345": run((3, 4, 5))}
    Path(ROOT / "results/trunk_seq/es/m49_select.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
