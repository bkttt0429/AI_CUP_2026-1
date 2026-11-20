"""GPT round 4 (2026-11-05): does the OPPONENT population a player meets carry a stable signature (a matchmaking-pool fingerprint)?
Games: HD[0], the target plays White, so Black (the opponent) makes the first move before the target has played anything. Mark the
opponent's first move: corner 4-4 (star), 3-4, 3-3, other. Per player: random split of the qualifying games into halves; across players
WITHIN groups of the same label and similar strength (oh-ensemble E quintile within label, folds 1-4 CV players), the correlation of the
first-half and second-half star rates; the null: the second halves permuted among the players of the same group. A real > null gap means
different players meet opponents with different habits even at the same label and strength (era / table / repeated opponents). Then: does
the opponents' star rate predict the sign of the player's persistent bias (E - y) within groups? Also reported for all 8,007 training
players grouped by label only (no E).
  python -I logs/nano4/pool_fingerprint.py --out results/r_variants/pool_fingerprint.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import load_games, player_games  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
STAR = {"dd", "pd", "dp", "pp"}
KOMOKU = {"cd", "dc", "qd", "pc", "cp", "dq", "qp", "pq"}
SANSAN = {"cc", "qc", "cq", "qq"}


def first_move_type(sgf):
    m = re.search(r";B\[([a-s]{2})\]", sgf)
    if not m:
        return None
    mv = m.group(1)
    return "star" if mv in STAR else "komoku" if mv in KOMOKU else "sansan" if mv in SANSAN else "other"


def halves_corr(df, group_col, rng, n_perm=200, min_games=6):
    """df: player_id, group, rate_h1, rate_h2 (players with >= min_games qualifying games). Real corr vs permuted-within-group null."""
    d = df.dropna()
    real = float(np.corrcoef(d.rate_h1, d.rate_h2)[0, 1])
    null = []
    for _ in range(n_perm):
        p2 = d.groupby(group_col)["rate_h2"].transform(lambda x: x.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_numpy())
        null.append(np.corrcoef(d.rate_h1, p2)[0, 1])
    null = np.array(null)
    return {"players": int(len(d)), "real_corr": real, "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)), "z": float((real - null.mean()) / (null.std() + 1e-12))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--min-games", type=int, default=6); a = ap.parse_args()
    g = player_games()
    gw = g[g["rank"].notna()].copy()
    # only games where the target is White: the colour column is in the per-level train CSVs; player_games keeps player_id, rank, game_key -> reload colour
    frames = []
    for p in sorted((ROOT / "data/train").glob("train_*.csv")) if (ROOT / "data/train").exists() else []:
        frames.append(pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]))
    if not frames:
        from aicup.scoring import TRAIN_DIR
        for p in sorted(Path(TRAIN_DIR).glob("train_*.csv")):
            frames.append(pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]))
    c = pd.concat(frames, ignore_index=True); c["game_key"] = c["rank"] + ":" + c["game_id"]
    c = c[c.color == "W"]
    print(f"{len(g)} training games, {len(c)} with the target as White", flush=True)
    data = load_games(c.game_key.tolist())
    rows = []
    for r in c.itertuples():
        sgf = data[r.game_key][0]
        hd = re.search(r"HD\[(\d+)\]", sgf)
        if hd and int(hd.group(1)) != 0:
            continue
        t = first_move_type(sgf)
        if t is not None:
            rows.append((r.player_id, r.rank, t))
    F = pd.DataFrame(rows, columns=["player_id", "rank", "first"])
    print(f"{len(F)} qualifying games (HD 0, target White); first-move mix: {F['first'].value_counts(normalize=True).round(3).to_dict()}", flush=True)
    rng = np.random.default_rng(0)
    per = []
    for pid, s in F.groupby("player_id"):
        if len(s) < a.min_games:
            continue
        idx = rng.permutation(len(s)); h1, h2 = s.iloc[idx[: len(s) // 2]], s.iloc[idx[len(s) // 2:]]
        per.append({"player_id": pid, "rank": s["rank"].iat[0], "n": len(s), "rate_h1": (h1["first"] == "star").mean(), "rate_h2": (h2["first"] == "star").mean(),
                    "komoku_h1": (h1["first"] == "komoku").mean(), "komoku_h2": (h2["first"] == "komoku").mean(), "star_all": (s["first"] == "star").mean()})
    P = pd.DataFrame(per); P["group"] = P["rank"]
    rep = {"qualifying_games": int(len(F)), "players_ge_min": int(len(P)), "min_games": a.min_games, "first_move_mix": F["first"].value_counts(normalize=True).to_dict()}
    rep["all_players_by_label"] = {"star": halves_corr(P, "group", rng), "komoku": halves_corr(P.rename(columns={"komoku_h1": "rate_h1", "komoku_h2": "rate_h2", "rate_h1": "x1", "rate_h2": "x2"}), "group", rng)}
    print(f"all players (groups = label): star rate split-half corr {rep['all_players_by_label']['star']['real_corr']:.3f} vs permuted-within-label {rep['all_players_by_label']['star']['null_mean']:.3f} (97.5% {rep['all_players_by_label']['star']['null_q975']:.3f}), z {rep['all_players_by_label']['star']['z']:.1f}; "
          f"komoku {rep['all_players_by_label']['komoku']['real_corr']:.3f} vs {rep['all_players_by_label']['komoku']['null_mean']:.3f}", flush=True)
    # CV players with E: groups = label x E quintile within label; and the bias test
    Es = []
    for v in (1, 2, 3, 4):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        Pm = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        meta["E"] = Pm @ np.arange(10); Es.append(meta[meta.n_games == 20].drop_duplicates("player_id")[["player_id", "E"]])
    E = pd.concat(Es).set_index("player_id")["E"]
    Q = P[P.player_id.isin(E.index)].copy(); Q["E"] = E.reindex(Q.player_id).to_numpy(); Q["y"] = Q["rank"].map(RANK); Q["bias"] = Q.E - Q.y
    Q["group"] = Q["rank"] + "_" + Q.groupby("rank")["E"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")).astype(str)
    rep["cv_players_by_label_and_E"] = {"star": halves_corr(Q, "group", rng), "komoku": halves_corr(Q.rename(columns={"komoku_h1": "rate_h1", "komoku_h2": "rate_h2", "rate_h1": "x1", "rate_h2": "x2"}), "group", rng)}
    r = rep["cv_players_by_label_and_E"]["star"]
    print(f"CV players (groups = label x E quintile, {r['players']} players): star split-half corr {r['real_corr']:.3f} vs null {r['null_mean']:.3f} (97.5% {r['null_q975']:.3f}), z {r['z']:.1f}", flush=True)
    # does the opponents' first-move habit predict the bias (within label)?
    Q["bias_dm"] = Q.bias - Q.groupby("rank")["bias"].transform("mean"); Q["star_dm"] = Q.star_all - Q.groupby("rank")["star_all"].transform("mean")
    rep["bias_vs_opp_star_within_label"] = {"spearman": spearman(Q.star_dm.to_numpy(), Q.bias_dm.to_numpy()), "players": int(len(Q))}
    print(f"within-label Spearman(opponents' star rate, bias E - y) = {rep['bias_vs_opp_star_within_label']['spearman']:+.3f} over {len(Q)} players")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
