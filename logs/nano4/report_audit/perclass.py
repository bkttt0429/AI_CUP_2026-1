import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src")]
from aicup.qeval import evaluate, SIZES
from aicup.metric import LEVELS
C, O = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd"
rows = []
for v in (1, 2, 3, 4):
    z = np.load(C / f"r_val{v}.npz"); q = pd.DataFrame(json.loads((C / f"r_val{v}.json").read_text())["val"]); y = z["y_va"].astype(int)
    for name, P in (("MLP", np.load(O / f"tune_val{v}.npz")["mlp0"]), ("TREE", np.load(O / f"xgb_val{v}.npz")["proba"])):
        P = np.asarray(P, dtype=float); s = evaluate(P, y, q)[1]
        rows.append(pd.DataFrame({"m": name, "v": v, "y": y, "n": q["n_games"].to_numpy(), "s": s, "pred": (P * np.arange(10)).sum(1)}))
t = pd.concat(rows)
print("questions per level:", t[t.m == "MLP"].groupby("y").size().to_dict())
for n in SIZES + (0,):
    u = t if n == 0 else t[t.n == n]
    a = u.groupby(["m", "y"])["s"].mean().unstack(0)
    print(f"n={'all' if n == 0 else n}: per level TREE - MLP: " + " ".join(f"{LEVELS[k]} {a.loc[k, 'TREE'] - a.loc[k, 'MLP']:+.3f}" for k in a.index))
u = t.groupby(["m", "y"])["pred"].mean().unstack(0)
print("mean expected level per true level: " + " ".join(f"{LEVELS[k]} M{u.loc[k, 'MLP']:.2f}/T{u.loc[k, 'TREE']:.2f}" for k in u.index))
