"""Duplicate games (2026-11-05): the same move sequence listed under two players.  Training: both players are labelled, so the pairs
give the server's matchmaking table P(other player's label | target label, HD) exactly.  Test: two questions sharing a game are two
players who met, which ties their ranks (rank_A ~ rank_B +- HD).
  python -I logs/nano4/dup_scan.py --out results/r_variants/dup_scan.json
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.scoring import load_games, player_games  # noqa: E402
import predict_submission as ps  # noqa: E402


def move_hash(sgf):
    moves = re.findall(r";([BW]\[[a-z]{0,2}\])", sgf)
    hd = re.search(r"HD\[(\d+)\]", sgf)
    return hashlib.sha1("".join(moves).encode()).hexdigest(), (int(hd.group(1)) if hd else -1), len(moves)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    g = player_games()
    print(f"train: {len(g)} rows, {g.player_id.nunique()} players, {g.game_key.nunique()} game keys", flush=True)
    data = load_games(g.game_key.tolist())
    rows = []
    for r in g.itertuples():
        sgf, colour = data[r.game_key]
        h, hd, n = move_hash(sgf)
        rows.append((h, r.player_id, r.rank, colour, hd, n))
    T = pd.DataFrame(rows, columns=["h", "player_id", "rank", "color", "hd", "n_moves"])
    T = T[T.n_moves >= 30]
    cnt = T.groupby("h").player_id.nunique()
    dup_h = cnt[cnt >= 2].index
    D = T[T.h.isin(dup_h)].sort_values("h")
    rep = {"train_games": int(len(T)), "train_distinct_sequences": int(T.h.nunique()), "train_duplicate_sequences": int(len(dup_h)),
           "train_games_in_duplicates": int(len(D))}
    print(f"train: {len(dup_h)} move sequences appear under >= 2 players ({len(D)} rows)", flush=True)
    pairs = []
    for h, grp in D.groupby("h"):
        ps_ = grp.drop_duplicates("player_id")
        if len(ps_) == 2 and set(ps_.color) == {"B", "W"}:
            b, w = ps_[ps_.color == "B"].iloc[0], ps_[ps_.color == "W"].iloc[0]
            pairs.append({"rank_B": b["rank"], "rank_W": w["rank"], "hd": int(b["hd"])})
    P = pd.DataFrame(pairs)
    rep["train_pairs_B_W"] = int(len(P))
    if len(P):
        L = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; idx = {l: i for i, l in enumerate(L)}
        P["gap_W_minus_B"] = P.rank_W.map(idx) - P.rank_B.map(idx)
        tab = P.groupby("hd").gap_W_minus_B.value_counts().unstack(fill_value=0)
        rep["gap_by_hd"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in tab.iterrows()}
        print("train pairs: rank(W) - rank(B) by HD:"); print(tab.to_string())
        rep["share_equal_given_hd0"] = float((P[P.hd == 0].gap_W_minus_B == 0).mean()) if (P.hd == 0).any() else None
        rep["share_gap1_given_hd1"] = float((P[P.hd == 1].gap_W_minus_B == 1).mean()) if (P.hd == 1).any() else None
    # test
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS)
    trows = []
    for qi, q in enumerate(questions):
        for sgf, colour in q["games"]:
            h, hd, n = move_hash(sgf)
            trows.append((h, q["question_id"], colour, hd, n))
    Q = pd.DataFrame(trows, columns=["h", "question_id", "color", "hd", "n_moves"])
    cq = Q.groupby("h").question_id.nunique()
    dups = cq[cq >= 2].index
    links = Q[Q.h.isin(dups)].sort_values("h")
    rep["test_games"] = int(len(Q)); rep["test_duplicate_sequences"] = int(len(dups)); rep["test_questions_linked"] = int(links.question_id.nunique())
    rep["test_links"] = [{"h": h[:10], "questions": grp.question_id.tolist(), "colors": grp.color.tolist(), "hd": int(grp.hd.iloc[0])} for h, grp in links.groupby("h")][:50]
    # test games that also occur in training
    train_h = set(T.h)
    rep["test_games_also_in_train"] = int(Q.h.isin(train_h).sum())
    print(f"test: {len(Q)} games, {len(dups)} sequences shared by >= 2 questions ({links.question_id.nunique()} questions linked); {rep['test_games_also_in_train']} test games also in the training set")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str)); print("written", a.out)


if __name__ == "__main__":
    main()
