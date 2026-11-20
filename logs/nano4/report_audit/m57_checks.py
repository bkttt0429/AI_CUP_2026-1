# (1) is ct13's private 1D->A set a subset of ct10's? (2) colour split public vs private: question-clustered permutation test.
import numpy as np, pandas as pd
d9, d10, d13 = (pd.read_csv(f"submission_rank_{k}.csv", dtype=str).set_index("question_id")["pred_rank"] for k in ("ct9", "ct10", "ct13"))
d10, d13 = d10.reindex(d9.index), d13.reindex(d9.index)
priv = ~d9.index.str.startswith("pub")
s13 = set(d9.index[priv & (d13 != d9)]); s10 = set(d9.index[priv & (d10 == "A") & (d9 == "1D")])
print(f"private: ct13 flips {len(s13)}, ct10 1D->A flips {len(s10)}, ct13 subset of ct10: {s13 <= s10}, ct10 extra {len(s10 - s13)}")
g = pd.read_csv("results/trunk_seq/es/m51_test_games.csv")
q = g.groupby("question_id").agg(black=("color", lambda c: (c == "B").sum()), n=("color", "size"), half=("half", "first"))
obs = q[q.half == "public"].black.sum() / q[q.half == "public"].n.sum() - q[q.half == "private"].black.sum() / q[q.half == "private"].n.sum()
rng = np.random.default_rng(0); halves = q.half.to_numpy(); B = q.black.to_numpy(); N = q.n.to_numpy(); null = []
for _ in range(20000):
    h = rng.permutation(halves); null.append(B[h == "public"].sum() / N[h == "public"].sum() - B[h == "private"].sum() / N[h == "private"].sum())
null = np.array(null)
print(f"colour: public-private black share diff {obs:+.4f}; question-level permutation sd {null.std():.4f}, z {obs / null.std():+.2f}, two-sided p {(np.abs(null) >= abs(obs)).mean():.4f}")
