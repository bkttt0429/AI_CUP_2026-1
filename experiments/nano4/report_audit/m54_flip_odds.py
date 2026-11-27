# ct13's 18 flipped questions (1D -> A): ct9's odds P(A)/P(1D), n, half -- are the 10 Private flips as confident as the 8 Public ones?
import numpy as np, pandas as pd
pr = pd.read_csv("submission_rank_ct9_proba.csv").set_index("question_id"); d9 = pd.read_csv("submission_rank_ct9.csv", dtype=str).set_index("question_id")["pred_rank"]
d13 = pd.read_csv("submission_rank_ct13.csv", dtype=str).set_index("question_id")["pred_rank"]
ng = pd.concat([pd.read_csv(f"data/tests/rank_prediction_test_{s}.csv", usecols=["question_id", "num_games"]) for s in ("public", "private")]).set_index("question_id")["num_games"]
fl = [q for q in d9.index if d9[q] != d13[q]]
rows = [(q, "public" if q.startswith("pub") else "private", int(ng[q]), pr.loc[q, "p_A"] / pr.loc[q, "p_1D"], pr.loc[q, "p_A"] + pr.loc[q, "p_1D"], pr.loc[q, "p_2D"], pr.loc[q, "p_B"]) for q in fl]
df = pd.DataFrame(rows, columns=["q", "half", "n", "odds_A_1D", "mass_A+1D", "p_2D", "p_B"]).sort_values(["half", "odds_A_1D"], ascending=[False, False])
pd.set_option("display.width", 200); print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for h in ("public", "private"):
    s = df[df.half == h]; print(f"{h}: n counts {s['n'].value_counts().sort_index().to_dict()}, odds median {s['odds_A_1D'].median():.3f}, min {s['odds_A_1D'].min():.3f}, mass A+1D median {s['mass_A+1D'].median():.3f}")
# the next band: ct9 says 1D with odds in [1/2.6, 1/2)
nxt = [(q, "public" if q.startswith("pub") else "private", pr.loc[q, "p_A"] / pr.loc[q, "p_1D"]) for q in d9.index if d9[q] == "1D" and 1 / 2.6 <= pr.loc[q, "p_A"] / pr.loc[q, "p_1D"] < 0.5]
print("next band (odds 1/2.6 .. 1/2):", [(q, h, round(o, 3)) for q, h, o in nxt])
