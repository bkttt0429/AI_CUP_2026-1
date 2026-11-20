"""Section 4M M95 step 1c (2026-11-13, CPU): exact move-sequence comparison between the 10,000 rank-test games and every
competition-1 game of the v2 release (old pools: candidates and questions, public and private; added pools pub2 / priv2:
candidates2 and test2). The 11-06 scan (cross_task_overlap) covered only the 10-30 pools. A game = the string of its moves
(;B[..] / ;W[..] in order); also the first 40 moves alone (shared openings between two different games are not a match, so that
count is only a hint). Reports the number of identical games per file and the (rank question -> competition-1 player) pairs.
  python -I logs/nano4/m95_dupscan.py
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m81_tactic_id as m81  # noqa: E402
from aicup_pid import pools  # noqa: E402

T1 = Path.home() / "BK_AICUP_2026-2/data/tests_1111"; MV = re.compile(r";\s*([BW])\[([a-s]{0,2})\]")
seq = lambda sgf: "".join(c + p + "," for c, p in MV.findall(sgf))  # noqa: E731

rank = defaultdict(list); n_rank = 0
for qid, g in m81.load_tests():
    sgf = g[0] if isinstance(g, (tuple, list)) else g; s = seq(sgf); rank[s].append(qid); n_rank += 1
print(f"rank test: {n_rank} games, {len(rank)} distinct move sequences; plies: median {sorted(len(k.split(',')) for k in rank)[len(rank) // 2]}", flush=True)
hits = []
for name in ("candidates_public", "candidates_private", "candidates2_public", "candidates2_private"):
    c = pd.read_csv(T1 / f"player_identification_{name}.csv", dtype=str, keep_default_na=False); n = 0
    for r in c.itertuples():
        s = seq(r.sgf_content)
        if s in rank and len(s) > 40:
            n += 1; hits += [(q, name, r.player_id, r.game_id) for q in rank[s]]
    print(f"{name}: {len(c)} games of {c.player_id.nunique()} players; identical to a rank-test game: {n}", flush=True)
for name in ("test_public", "test_private", "test2_public", "test2_private"):
    n = g_ = 0
    for q in pools.read_questions(T1 / f"player_identification_{name}.csv"):
        for sgf, colour in q["games"]:
            s = seq(sgf); g_ += 1
            if s in rank and len(s) > 40:
                n += 1; hits += [(x, name, q["question_id"], "") for x in rank[s]]
    print(f"{name}: {g_} games; identical to a rank-test game: {n}", flush=True)
by = Counter((q, f, p) for q, f, p, _ in hits); print(f"pairs (rank question, file, competition-1 player or question) with identical games: {len(by)}")
for (q, f, p), k in sorted(by.items(), key=lambda kv: -kv[1])[:60]:
    print(f"  {q} <-> {f} {p}: {k} identical game(s)")
