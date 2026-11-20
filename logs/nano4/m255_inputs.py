#!/usr/bin/env python
"""M255 (2026-11-20): inputs of step 2 of the cross-competition linkage (design doc M255, registered before any result).

For each of the 20 pairs of M253 (threshold 0.5; results/r_variants/m253_xvotes.json) the matched competition-1 candidate's
10 games become one pseudo-question "B" in the official rank-test format (question_id pub_bNN, num_games 10, sgf_1..10,
color_1..10; game_id order). Six of the matched test questions are copied unchanged under the ids pub_cNN as controls of the
recomputation (base posterior against submission_rank_ct9_proba.csv, block scores and S7t vectors against the cache).
Read-only on the other session's raw files. Writes results/m255/inputs/{bq.csv, ctl.csv, map.csv, ctl_map.csv}.

  python -I logs/nano4/m255_inputs.py
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path.cwd(); SRC = Path.home() / "BK_AICUP_2026-2/data/tests_1111"; OUT = ROOT / "results/m255/inputs"
CTL = ("pub_q_0020", "priv_q_0071", "pub_q_0013", "priv_q_0073", "pub_q_0298", "priv_q_0381")  # sizes 5, 5, 10, 10, 15, 20
HD = re.compile(r"HD\[(\d+)\]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(ROOT / "results/r_variants/m253_xvotes.json"))["candidates of pub2 / priv2"]["lo_0.5"]["pairs"]; assert len(pairs) == 20
    test = pd.concat([pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{h}.csv", dtype=str, keep_default_na=False) for h in ("public", "private")], ignore_index=True)
    cols = list(test.columns); assert cols[:2] == ["question_id", "num_games"] and len(cols) == 42 and len(test) == 800; tq = test.set_index("question_id")
    cand = pd.concat([pd.read_csv(SRC / f"player_identification_candidates2_{h}.csv", dtype=str, keep_default_na=False) for h in ("public", "private")], ignore_index=True)
    rows, mp = [], []
    for i, (q, c, n, s1, s2) in enumerate(pairs):
        pid = c.split(":")[-1]; g = cand[cand.player_id == pid].sort_values("game_id"); assert len(g) == 10 and int(tq.loc[q, "num_games"]) == n, (q, pid, len(g))
        own = {tq.loc[q, f"sgf_{k}"] for k in range(1, n + 1)}; assert not (own & set(g.sgf_content)), (q, pid); assert all(HD.search(s) for s in g.sgf_content) and set(g.color) <= {"B", "W"}
        qid = f"pub_b{i:02d}"; row = {k: "" for k in cols}; row.update(question_id=qid, num_games="10")
        for k, (sg, col) in enumerate(zip(g.sgf_content, g.color), 1):
            row[f"sgf_{k}"] = sg; row[f"color_{k}"] = col
        rows.append(row); mp.append({"b_question": qid, "test_question": q, "candidate": pid, "n_test": n, "test_color": tq.loc[q, "color_1"], "share_q": s1, "share_c": s2, "game_ids": "|".join(g.game_id), "colors": "".join(g.color), "hd1": sum(HD.search(s).group(1) != "0" for s in g.sgf_content)})
    pd.DataFrame(rows, columns=cols).to_csv(OUT / "bq.csv", index=False); pd.DataFrame(mp).to_csv(OUT / "map.csv", index=False)
    ctl = tq.loc[list(CTL)].reset_index(); cm = pd.DataFrame({"ctl_question": [f"pub_c{i:02d}" for i in range(len(CTL))], "test_question": list(CTL), "n_test": ctl.num_games.astype(int)}); ctl["question_id"] = cm.ctl_question
    ctl[cols].to_csv(OUT / "ctl.csv", index=False); cm.to_csv(OUT / "ctl_map.csv", index=False)
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()[:12]  # noqa: E731
    print(f"B questions: {len(rows)} (10 games each; colours W {sum(m['colors'].count('W') for m in mp)}, B {sum(m['colors'].count('B') for m in mp)}; games with HD >= 1: {sum(m['hd1'] for m in mp)}); md5 {md5(OUT / 'bq.csv')}")
    print(f"controls: {len(ctl)} original test questions, sizes {ctl.num_games.tolist()}; md5 {md5(OUT / 'ctl.csv')}")
    print(pd.DataFrame(mp)[["b_question", "test_question", "candidate", "n_test", "test_color", "colors", "hd1"]].to_string(index=False)); print("written", OUT)


if __name__ == "__main__":
    main()
