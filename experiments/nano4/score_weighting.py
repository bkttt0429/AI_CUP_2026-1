import numpy as np, pandas as pd
t = pd.read_csv("results/l1b/oof_fused_opp_proba.csv", dtype={"player_id": str, "question_id": str})
d = (t["pred"] - t["y"]).abs().to_numpy(); s = np.where(d == 0, 1.0, np.where(d == 1, np.exp(-1), 0.0)); n = t["n_games"].to_numpy(); y = t["y"].to_numpy()
print(f"fused_opp OOF, {len(t)} questions: plain mean score {s.mean():.4f}; exact {(d==0).mean():.4f} within1 {(d<=1).mean():.4f}; exact+e^-1*(within1-exact) = {(d==0).mean()+np.exp(-1)*((d<=1).mean()-(d==0).mean()):.4f}")
for m in (5, 10, 20):
    k = n == m; print(f"  n={m}: {k.sum()} q, plain mean {s[k].mean():.4f}, class-balanced {np.mean([s[k & (y==c)].mean() for c in range(10)]):.4f}")
print(f"mean over n of plain {np.mean([s[n==m].mean() for m in (5,10,20)]):.4f}; mean over n of class-balanced {np.mean([np.mean([s[(n==m)&(y==c)].mean() for c in range(10)]) for m in (5,10,20)]):.4f}  (error_analysis mlp 0.6735)")
print("questions per class:", np.bincount(y, minlength=10).tolist())
