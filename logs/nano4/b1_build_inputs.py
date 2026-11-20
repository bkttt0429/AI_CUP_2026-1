"""Section 4M M62 (2026-11-08): inputs of the B1 support run for competition 1's player-identification session.
Competition 1's old Public candidates (400 players x 100 black games) become 2,000 pseudo-questions of 20 games in the official rank-test
format (question_id pub_c<player>_k<chunk>, num_games 20, sgf_1..20, color_1..20), chunked in game_id order; its 400 Public questions
(already in that format, 5/10/15/20 black games) are copied unchanged. Both go through ct9's test-side path (predict_submission.py shards ->
trunk_predict.py extract -> predict with the 24 ct_cv_m33 members); the resulting 10-dim probabilities are evaluation inputs for the
identification side only. Nothing here trains anything or touches competition 2's test. Also writes cands_map.csv (question -> player, games).
  python -I logs/nano4/b1_build_inputs.py
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
SRC = Path.home() / "BK_AICUP_2026-2" / "data" / "tests"
OUT = ROOT / "results" / "b1_public" / "inputs"
N = 20


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(SRC / "player_identification_test_public.csv", dtype=str, keep_default_na=False)
    cols = list(q.columns)
    assert cols[:2] == ["question_id", "num_games"] and len(cols) == 42, cols
    assert q["question_id"].str.startswith("pub_").all() and len(q) == 400
    c = pd.read_csv(SRC / "player_identification_candidates_public.csv", dtype=str, keep_default_na=False)
    rows, mp = [], []
    for pid, g in c.groupby("player_id", sort=True):
        g = g.sort_values("game_id")
        sg, gid, col = g["sgf_content"].tolist(), g["game_id"].tolist(), g["color"].tolist()
        assert len(sg) % N == 0 and set(col) == {"B"}, (pid, len(sg), set(col))
        core = pid.replace("pub_p_", "")
        for j in range(len(sg) // N):
            qid = f"pub_c{core}_k{j}"
            row = {"question_id": qid, "num_games": str(N)}
            for i in range(N):
                row[f"sgf_{i + 1}"] = sg[j * N + i]
                row[f"color_{i + 1}"] = col[j * N + i]
            rows.append(row)
            mp.append({"question_id": qid, "player_id": pid, "chunk": j, "game_ids": "|".join(gid[j * N:(j + 1) * N])})
    cands = pd.DataFrame(rows, columns=cols)
    cands.to_csv(OUT / "cands.csv", index=False)
    q.to_csv(OUT / "quest.csv", index=False)
    pd.DataFrame(mp).to_csv(OUT / "cands_map.csv", index=False)
    print(f"candidates: {c['player_id'].nunique()} players, {len(c)} games -> {len(cands)} pseudo-questions of {N}; md5 {md5(OUT / 'cands.csv')}")
    print(f"questions: {len(q)} rows, num_games {q['num_games'].value_counts().sort_index().to_dict()}; md5 {md5(OUT / 'quest.csv')}")
    print("written", OUT)


if __name__ == "__main__":
    main()
