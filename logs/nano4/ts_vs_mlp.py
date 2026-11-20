#!/usr/bin/env python
"""Report only: C, T and R0 of results/trunk_seq/es against the existing MLP h32 on the same 64 R features and questions, paired by question."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff
from aicup.rankmodel import MLPModel
C = ROOT / "results/trunk_seq"; ES = C / "es"
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--tag", default=""); ap.add_argument("--arms", nargs="+", default=["R0", "C", "T"]); A = ap.parse_args()
folds = (1, 2, 3, 4)
rows = {}
out = {}
for v in folds:
    z = np.load(C / f"r_val{v}.npz"); q = pd.DataFrame(json.loads((C / f"r_val{v}.json").read_text())["val"])
    y = z["y_va"]
    P = MLPModel(hidden=32).fit(z["X_tr"], z["y_tr"]).predict_proba(z["X_va"])
    res, s_mlp = evaluate(P, y, q)
    d = {"MLP": s_mlp}
    probs = {}
    for arm in A.arms:
        fs = [f for f in sorted((ES / f"val{v}").glob(f"{arm}_s*.npz")) if int(f.stem.rsplit("_s", 1)[1]) in A.seeds]
        assert len(fs) == len(A.seeds), (arm, v, fs)
        L = [np.load(f) for f in fs]
        d[arm] = np.mean([l["scores"] for l in L], axis=0)
        probs[arm] = np.mean([l["proba"] for l in L], axis=0)
        d[arm + "_ens"] = evaluate(probs[arm], y, q)[1]
    d["CT_ens"] = evaluate((probs["C"] + probs["T"]) / 2, y, q)[1]
    d["TMLP_ens"] = evaluate((probs["T"] + P) / 2, y, q)[1]
    rows[v] = (y, q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy(), d)
    print(f"fold {v}: MLP {res['balanced_mean_over_n']:.4f}", flush=True)
for name, use in (("folds 2 3 4", (2, 3, 4)), ("folds 1 2 3 4", folds)):
    y = np.concatenate([rows[v][0] for v in use]); pid = np.concatenate([rows[v][1] for v in use]); n = np.concatenate([rows[v][2] for v in use])
    S = {k: np.concatenate([rows[v][3][k] for v in use]) for k in rows[1][3]}
    print("==", name)
    pf = {k: [balanced_mean_over_n(rows[v][3][k], rows[v][0], rows[v][2]) for v in use] for k in S}
    block = {}
    for k in S:
        line = f"{k:9s} {np.mean(pf[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in pf[k]) + f"  direct {S[k].mean():.4f}"
        if k != "MLP":
            dd = paired_diff(S[k], S["MLP"], y, pid, n, n=5000)
            e = [a - b for a, b in zip(pf[k], pf["MLP"])]
            block[k] = {"score": float(np.mean(pf[k])), "diff": dd["diff"], "ci95": dd["ci95"], "by_n": dd["by_n"], "per_fold_diff": e, "direct": float(S[k].mean())}
            line += f"  vs MLP {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in e) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items())
        print(line, flush=True)
    out[name] = {"MLP": {"score": float(np.mean(pf["MLP"])), "per_fold": pf["MLP"], "direct": float(S["MLP"].mean())}, **block}
(ES / f"vs_mlp{A.tag}.json").write_text(json.dumps(out, indent=1, default=float))
