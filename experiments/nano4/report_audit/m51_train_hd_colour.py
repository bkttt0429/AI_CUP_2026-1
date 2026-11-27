# M51 side table: in the training games, how often is the labelled player the lower hand (black) of an HD1 game, overall and by rank;
# the test's figure is 0.471 (public 0.417, private 0.525).
import pandas as pd
t = pd.read_csv("results/data/hd_table.csv", dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False, na_values={"hd": [""]})
t["hd"] = pd.to_numeric(t["hd"], errors="coerce")
h1 = t[t["hd"] == 1]
print(f"train: HD1 share {float((t['hd'] == 1).mean()):.3f}; target black among HD1 games {float((h1['color'] == 'B').mean()):.3f}; target black overall {float((t['color'] == 'B').mean()):.3f}")
levels = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
print("by rank, HD1 share / target black among HD1 / hd_gap_up mean / hd_gap_down mean (per game):")
for lv in levels:
    s = t[t["rank"] == lv]; s1 = s[s["hd"] == 1]
    up = ((s["hd"].fillna(0) > 0) & (s["color"] == "W")).mean(); down = ((s["hd"].fillna(0) > 0) & (s["color"] == "B")).mean()
    print(f"  {lv:3s} HD1 {float((s['hd'] == 1).mean()):.3f}  black|HD1 {float((s1['color'] == 'B').mean()):.3f}  HD>=1 & upper {up:.3f}  HD>=1 & lower {down:.3f}")
