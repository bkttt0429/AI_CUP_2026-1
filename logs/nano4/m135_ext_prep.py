#!/usr/bin/env python
"""Section 4M M135 (2026-11-14): the external (Fox) ranked units as a game table in the competition's layout, so the project's
scoring scripts read them unchanged (AICUP_TRAIN_DIR / AICUP_FOLDS_CSV / --folds-csv, pseudo-fold 9).

Sources (read only): /work/u2507115/ext_fox/<selection>/select.csv of the player-identification session (game_key "x:<rank>:<id>",
hashed account, the subject's colour, sgf). Unit = (hashed account, rank folder). Per rank up to --per-rank units, the units with
more games first (capped at --max-games), random among equals (seed 0); a unit needs --min-games games; a game id is used once.
Nothing here reads account names or compares external accounts with the competition's players.

Written under --out (default /work/u2507115/ext_rank):
  training/train_F<rank>.csv   player_id, game_id, rank, color, sgf_content   (player_id = <account>f<rank>, game_id = x<id>, rank = F<rank>)
  player_folds.csv             rank, player_id, n_games, fold (= 9)
  hd_table.csv                 game_key, player_id, rank, color, hd (= 0: every external game is an even game)
  units.json                   counts per rank
  smoke_keys.csv               the games of two units per rank (for the smoke run's --game-keys)

  python -I logs/nano4/m135_ext_prep.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/work/u2507115/ext_fox")
RANKS = ["5k", "4k", "3k", "2k", "1k", "1d", "2d", "3d", "4d", "5d", "6d"]
SELECTIONS = ["pilot", "midrank", "scale1", "scale2_5d", "scale3"]
HEAD = re.compile(r"^\(;SZ\[19\]HD\[0\];")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/work/u2507115/ext_rank")
    ap.add_argument("--per-rank", type=int, default=1000)
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "training").exists() and any((out / "training").glob("train_*.csv")):
        raise SystemExit(f"{out}/training already holds game tables; this script does not overwrite a selection")
    frames = []
    for s in SELECTIONS:
        d = pd.read_csv(SRC / s / "select.csv", dtype=str, keep_default_na=False)
        parts = d["game_key"].str.split(":", expand=True)
        assert (parts[0] == "x").all() and parts.shape[1] == 3, f"{s}: unexpected game keys"
        d["rank"], d["gid"], d["sel"] = parts[1], parts[2], s
        d = d[d["rank"].isin(RANKS)]
        frames.append(d)
        print(f"{s}: {len(d)} games of the wanted ranks", flush=True)
    d = pd.concat(frames, ignore_index=True)
    ok = d["sgf_content"].map(lambda x: bool(HEAD.match(x))) & d["color"].isin(["B", "W"]) & d["gid"].str.fullmatch(r"\d+") & d["account"].str.fullmatch(r"[0-9a-f]{8,}")
    print(f"dropped {int((~ok).sum())} rows with an unexpected header / colour / id", flush=True)
    d = d[ok].drop_duplicates(["rank", "account", "gid"]).reset_index(drop=True)
    rng = np.random.RandomState(a.seed)
    used, rows, report = set(), [], {}
    for r in RANKS:
        dr = d[d["rank"] == r]
        idx = dr.groupby("account", sort=True).indices  # positions within dr
        accounts = sorted(idx)
        n = np.array([len(idx[x]) for x in accounts])
        capped = np.minimum(n, a.max_games)
        tie = rng.rand(len(accounts))
        order = np.lexsort((tie, -capped))
        chosen, sizes = 0, []
        gid, col, sgf = dr["gid"].to_numpy(), dr["color"].to_numpy(), dr["sgf_content"].to_numpy()
        for j in order:
            if chosen >= a.per_rank or capped[j] < a.min_games:
                break
            pos = idx[accounts[j]]
            pos = pos[rng.permutation(len(pos))]
            keep = [p for p in pos if gid[p] not in used][:a.max_games]
            if len(keep) < a.min_games:
                continue
            pid = f"{accounts[j]}f{r}"
            for p in keep:
                used.add(gid[p])
                rows.append((pid, f"x{gid[p]}", f"F{r}", col[p], sgf[p]))
            chosen += 1
            sizes.append(len(keep))
        report[r] = {"units": chosen, "games": int(np.sum(sizes)), "units_with_20": int(np.sum(np.array(sizes) >= 20)), "units_with_10_or_more": int(np.sum(np.array(sizes) >= 10)),
                     "min_games": int(np.min(sizes)) if sizes else 0, "accounts_available": int(len(accounts))}
        print(f"{r}: {report[r]}", flush=True)
    t = pd.DataFrame(rows, columns=["player_id", "game_id", "rank", "color", "sgf_content"])
    assert t["game_id"].is_unique
    (out / "training").mkdir(parents=True, exist_ok=True)
    for r, g in t.groupby("rank", sort=True):
        g.to_csv(out / "training" / f"train_{r}.csv", index=False)
    per = t.groupby(["rank", "player_id"], sort=True).size().reset_index(name="n_games")
    per["fold"] = 9
    per[["rank", "player_id", "n_games", "fold"]].to_csv(out / "player_folds.csv", index=False)
    hd = pd.DataFrame({"game_key": t["rank"] + ":" + t["game_id"], "player_id": t["player_id"], "rank": t["rank"], "color": t["color"], "hd": 0.0})
    hd.to_csv(out / "hd_table.csv", index=False)
    smoke = per.groupby("rank").head(2)["player_id"]
    hd[hd["player_id"].isin(set(smoke))][["game_key"]].to_csv(out / "smoke_keys.csv", index=False)
    (out / "units.json").write_text(json.dumps({"args": vars(a), "ranks": report, "units": int(len(per)), "games": int(len(t))}, indent=1) + "\n")
    print(f"written {out}: {len(per)} units, {len(t)} games", flush=True)


if __name__ == "__main__":
    main()
