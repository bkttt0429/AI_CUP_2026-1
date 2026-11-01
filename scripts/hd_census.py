#!/usr/bin/env python
"""Census of the handicap (HD) and result (RE) properties of the training SGFs (dataset v1.1; issue #42).

v1.1 (2026-10-31) adds HD[n] to every sgf_content. The official Q&A (Q13) says: HD[0] even games about 80%, HD[1] "讓先" about
16% (colours as recorded), HD[2]-HD[9] handicap games about 3% whose handicap stones are not recorded and whose B/W move
marks are reversed; whether the `color` column of those rows is also reversed is still "to be confirmed" there. This script
reads data/training/train_*.csv (or --train-dir) and reports, per file and overall: the SGF properties present; the HD
distribution (None = property absent = v1.0 data); HD by rank and by colour; RE presence and its distribution by HD; for
HD >= 2 the labelled player's rank by `color` (if `color` is the ACTUAL colour, the labelled player is higher-ranked when
`color` is W, the stronger side; if `color` follows the reversed marks, the opposite) and whether the first recorded move
sits on one of the standard handicap points (it should not, when the stones are really missing); the share of players
with at least one HD >= 1 / HD >= 2 game. Nothing is replayed; CPU, a few minutes over a million rows.

  python -I scripts/hd_census.py --out results/data/hd_census.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.scoring import TRAIN_DIR  # noqa: E402
from aicup.sgf import HANDICAP_STONES, handicap, parse_moves, properties, result  # noqa: E402

STAR_POINTS = sorted(set(HANDICAP_STONES[9]))


def first_move(sgf):
    mv = parse_moves(sgf)
    if not mv or mv[0][1] is None:
        return None
    c, x, y = mv[0]
    return c, chr(97 + x) + chr(97 + y)


def census(train_dir):
    frames = []
    for p in sorted(Path(train_dir).glob("train_*.csv")):
        t0 = time.time()
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False)
        df["hd"] = df["sgf_content"].map(handicap)
        df["re"] = df["sgf_content"].map(result)
        df["props"] = df["sgf_content"].map(lambda s: ",".join(properties(s)))
        fm = df["sgf_content"].map(first_move)
        df["first_color"] = fm.map(lambda v: v[0] if v else None)
        df["first_star"] = fm.map(lambda v: (v[1] in STAR_POINTS) if v else None)
        df["n_moves"] = df["sgf_content"].map(lambda s: len(parse_moves(s)))
        frames.append(df.drop(columns=["sgf_content"]))
        print(f"{p.name}: {len(df)} rows, HD present {int(df['hd'].notna().sum())}, RE present {int(df['re'].notna().sum())} ({time.time() - t0:.0f} s)", flush=True)
    if not frames:
        raise FileNotFoundError(f"no train_*.csv in {train_dir}")
    return pd.concat(frames, ignore_index=True)


def summarise(t):
    t = t.copy()
    t["hd_key"] = t["hd"].map(lambda v: "absent" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v)))
    t["rank_idx"] = t["rank"].map(RANK_TO_IDX)
    out = {"rows": int(len(t)), "players": int(t["player_id"].nunique()), "properties": t["props"].value_counts().head(20).to_dict(),
           "hd": t["hd_key"].value_counts().sort_index().to_dict(),
           "hd_by_rank": {lv: t.loc[t["rank"] == lv, "hd_key"].value_counts().to_dict() for lv in LEVELS if (t["rank"] == lv).any()},
           "hd_by_color": {c: t.loc[t["color"] == c, "hd_key"].value_counts().to_dict() for c in ("B", "W")},
           "re_present": int(t["re"].notna().sum()),
           "re_by_hd": {k: g["re"].fillna("absent").map(lambda r: r if r in ("absent",) else (r.split("+")[0] + "+" + ("R" if r.endswith("R") else "pts") if "+" in r else r)).value_counts().head(8).to_dict()
                        for k, g in t.groupby("hd_key")},
           "n_moves_by_hd": {k: {"mean": float(g["n_moves"].mean()), "median": float(g["n_moves"].median())} for k, g in t.groupby("hd_key")}}
    hc = t[t["hd"].map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and int(v) >= 2)]
    if len(hc):
        by_c = {c: {"rows": int((hc["color"] == c).sum()), "mean_rank_idx": float(hc.loc[hc["color"] == c, "rank_idx"].mean())} for c in ("B", "W")}
        out["handicap_games"] = {"rows": int(len(hc)), "labelled_rank_by_color": by_c,
                                 "reading": "color is the ACTUAL colour if mean_rank_idx(W) > mean_rank_idx(B) (White is the stronger side in a handicap game); the reverse if color follows the reversed marks",
                                 "first_move_mark": hc["first_color"].value_counts().to_dict(),
                                 "first_move_on_star_point_share": float(hc["first_star"].astype(float).mean()),
                                 "by_hd_color": {f"{int(h)}": hc.loc[hc["hd"] == h, "color"].value_counts().to_dict() for h in sorted(hc["hd"].dropna().unique())}}
        even = t[t["hd_key"] == "0"]
        if len(even):
            out["even_games_first_move_on_star_point_share"] = float(even["first_star"].astype(float).mean())
    g = t.groupby("player_id")["hd"].apply(lambda s: (s.fillna(-1).astype(float) >= 1).mean())
    g2 = t.groupby("player_id")["hd"].apply(lambda s: (s.fillna(-1).astype(float) >= 2).mean())
    out["players_with_any_hd1_share"] = float((g > 0).mean())
    out["players_with_any_hd2_share"] = float((g2 > 0).mean())
    out["player_hd1_share_quantiles"] = {str(q): float(g.quantile(q)) for q in (0.5, 0.9, 0.99)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", default=str(TRAIN_DIR))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        t = census(a.train_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    s = summarise(t)
    print(f"{s['rows']} rows, {s['players']} players; HD: {s['hd']}; RE present {s['re_present']}")
    print("properties:", s["properties"])
    if "handicap_games" in s:
        h = s["handicap_games"]
        print(f"handicap games (HD >= 2): {h['rows']}; labelled rank by color: {h['labelled_rank_by_color']}; first move mark {h['first_move_mark']}; "
              f"first move on a star point {h['first_move_on_star_point_share']:.3f} (even games {s.get('even_games_first_move_on_star_point_share', float('nan')):.3f})")
    print(f"players with any HD>=1 game {s['players_with_any_hd1_share']:.3f}, HD>=2 {s['players_with_any_hd2_share']:.3f}")
    out = {**s, "train_dir": str(a.train_dir), "star_points": STAR_POINTS, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
