"""PR #4 checks on the 40-question mock test: single GPU vs 4 shards + merge vs the recorded fixture."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/mock_fold0_40"
ONE = ROOT / "results/mock_fold0_40"
X4 = ROOT / "results/mock_fold0_40_x4"


def sub(d):
    return pd.read_csv(d / "submission_rank.csv", dtype=str)


def proba(d):
    p = pd.read_csv(d / "submission_rank_proba.csv")
    return p.set_index("question_id")[[c for c in p.columns if c.startswith("p_")]]


def agree(a, b):
    m = a.merge(b, on="question_id", suffixes=("_a", "_b"), how="outer")
    return int((m["pred_rank_a"] == m["pred_rank_b"]).sum()), len(m), list(a["question_id"]) == list(b["question_id"])


one, x4 = sub(ONE), sub(X4)
offline = pd.read_csv(FIX / "offline_predictions.csv", dtype=str)
lab01 = pd.read_csv(FIX / "submission_rank.csv", dtype=str)
answers = pd.read_csv(FIX / "answers.csv", dtype=str)
for name, a, b in (("single vs fixture offline_predictions", one, offline), ("single vs fixture lab01 submission", one, lab01),
                   ("4 shards + merge vs single", x4, one)):
    n, tot, same_order = agree(a, b)
    print(f"{name}: pred_rank {n}/{tot} equal, same row order {same_order}")
pa, p4, pf = proba(ONE), proba(X4), proba(FIX)
print(f"proba max |4 shards - single| = {np.abs(p4.values - pa.loc[p4.index].values).max():.3e}")
print(f"proba max |single (H200) - fixture (lab01)| = {np.abs(pa.values - pf.loc[pa.index].values).max():.3e}")

# per-game sums and per-move log p: the union of the 4 shard files vs the single run's shard 0 of 1
with np.load(ONE / "submission_rank_shard0of1.npz", allow_pickle=False) as z:
    t1, n1, s1 = z["task"], z["n_moves"], z["sums"]
    m1 = (z["move_logp"], z["move_task"], z["move_t"])
    meta1 = json.loads(str(z["__meta__"]))
parts = []
for i in range(4):
    with np.load(X4 / f"submission_rank_shard{i}of4.npz", allow_pickle=False) as z:
        parts.append({k: z[k] for k in ("task", "n_moves", "sums", "move_logp", "move_task", "move_t")})
        meta = json.loads(str(z["__meta__"]))
        print(f"shard {i}: {meta['n_tasks']} games, {meta['positions']} positions, {meta['scoring_wall_s']:.0f} s, "
              f"questions {meta['questions_sha256'][:12]}")
print(f"single: {meta1['n_tasks']} games, {meta1['positions']} positions, {meta1['scoring_wall_s']:.0f} s, "
      f"questions {meta1['questions_sha256'][:12]}")
t4 = np.concatenate([p["task"] for p in parts])
order = np.argsort(t4)
assert np.array_equal(t4[order], np.sort(t1)), "the shards do not cover the same tasks as the single run"
o1 = np.argsort(t1)
n4 = np.concatenate([p["n_moves"] for p in parts])[order]
s4 = np.concatenate([p["sums"] for p in parts])[order]
print(f"n_moves equal: {np.array_equal(n4, n1[o1])}; sums max |diff| {np.abs(s4 - s1[o1]).max():.3e} "
      f"(max |sum| {np.abs(s1).max():.1f})")
key1 = np.lexsort((m1[2], m1[1]))
mt = np.concatenate([p["move_task"] for p in parts]); mm = np.concatenate([p["move_t"] for p in parts])
ml = np.concatenate([p["move_logp"] for p in parts])
key4 = np.lexsort((mm, mt))
same_idx = np.array_equal(mt[key4], m1[1][key1]) and np.array_equal(mm[key4], m1[2][key1])
print(f"per-move rows: single {len(m1[0])}, shards {len(ml)}, same (task, t) index {same_idx}; "
      f"move_logp max |diff| {np.abs(ml[key4] - m1[0][key1]).max():.3e}")

acc = one.merge(answers, on="question_id")
print("reports:", {d.name: {k: json.loads((d / 'submission_rank_report.json').read_text())[k]
                            for k in ('mode', 'nshards', 'exit_code', 'problems', 'fallback_questions', 'games_dropped')}
                   for d in (ONE, X4)})
