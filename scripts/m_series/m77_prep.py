"""Section 4M M77 data (2026-11-10): every training game and every test game as move arrays for the whole-game identity encoder.
POS [N, 256] uint16 (0-360 board point y * 19 + x, 361 pass, 362 padding), COL [N, 256] uint8 (0 B, 1 W, 2 padding), the row's
player index, fold, actual colour, HD; the same for the test games (question id, target colour). Output results/m77/data.npz.
  python -I scripts/m_series/m77_prep.py
"""
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402

T = 256
RANK = {lv: i for i, lv in enumerate(LEVELS)}


def arr(sgf):
    m = parse_moves(sgf)[:T]
    pos = np.full(T, 362, np.uint16); col = np.full(T, 2, np.uint8)
    for t, (c, x, y) in enumerate(m):
        pos[t] = 361 if x is None else y * 19 + x; col[t] = 0 if c == "B" else 1
    h = re.search(r"HD\[(\d+)\]", sgf)
    return pos, col, len(m), int(h.group(1)) if h else 0


def pack(sgfs):
    with mp.get_context("fork").Pool(12) as pool:
        res = pool.map(arr, sgfs, chunksize=1024)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), np.array([r[2] for r in res], np.int16), np.array([r[3] for r in res], np.int8)


def main():
    t0 = time.time()
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
    POS, COL, LEN, HD = pack(rows.sgf_content.tolist()); print(f"training games packed: {POS.shape} ({time.time() - t0:.0f} s)", flush=True)
    recs = []
    for h in ("public", "private"):
        df = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{h}.csv", dtype=str, keep_default_na=False)
        for d in df.to_dict("records"):
            for k in range(1, int(d["num_games"]) + 1):
                s, c = d.get(f"sgf_{k}", ""), d.get(f"color_{k}", "")
                if s and c in ("B", "W"):
                    recs.append((d["question_id"], s, c))
    TPOS, TCOL, TLEN, THD = pack([r[1] for r in recs])
    np.savez(ROOT / "results/m77/data.npz", POS=POS, COL=COL, LEN=LEN, hd=HD, subj=(rows.color.to_numpy() == "W").astype(np.uint8), player=rows.player_id.map(pidx).to_numpy(np.int32),
             fold=rows.player_id.map(folds).to_numpy(np.int8), key=(rows["rank"] + ":" + rows["game_id"]).to_numpy(str), players=players, plab=plab,
             TPOS=TPOS, TCOL=TCOL, TLEN=TLEN, thd=THD, tsubj=np.array([r[2] == "W" for r in recs], np.uint8), tqid=np.array([r[0] for r in recs]))
    print(f"written: {len(rows)} training games ({int((HD >= 2).sum())} with HD >= 2, median length {int(np.median(LEN))}), {len(recs)} test games ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
