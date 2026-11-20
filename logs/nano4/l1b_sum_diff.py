"""Distribution of the per-game sum differences, pipeline shard vs offline tables, for Human SL and the L1b models
(PR #10 step 3 follow-up; not in git)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
for d in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(d))
import predict_submission as ps
import check_game_vectors as cgv
from aicup.metric import LEVELS
from aicup.scoring import read_scored_games
inputs = ["tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv", "tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv"]
out = Path("results/mock_fold0_40_fused/submission_rank.csv")
questions, _ = ps.load_questions(inputs)
tasks, owner, dropped, passes = ps.build_tasks(questions)
per_task, _, moves = ps.load_shards(out, 1, len(tasks), ps.questions_sha(inputs, questions), with_moves=True)
vec = ps.game_vectors(per_task, moves)
with np.load("results/ladder/vectors/fold0.npz", allow_pickle=False) as z:
    ref = {k: z[k] for k in ("game_key", "staged", "whole", "sums", "n_moves")}
match, un, amb = cgv.match_tasks(vec, ref, 0.5)
print("matched", int((match >= 0).sum()), "unmatched", len(un), "ambiguous", len(amb))
n = np.asarray(vec["n_moves"], dtype=float)
hs = np.abs(np.asarray(vec["sums"], float) - np.asarray(ref["sums"], float)[match])
def show(name, d):
    g = d.max(axis=1)
    pm = g / n
    print(f"{name:10s} per game max|Δ|: median {np.median(g):.4f} p95 {np.quantile(g,.95):.4f} max {g.max():.4f} | per move: median {np.median(pm):.5f} p95 {np.quantile(pm,.95):.5f} max {pm.max():.5f} | mean |Δ| over all cells {d.mean():.4f}")
show("HSL", hs)
models, l1b_per_task, _ = ps.load_shards_l1b(out, 1)
for j, (m, d) in enumerate(zip(models, ("results/l1b/scores3/fold0", "results/l1b/scores3_B/fold0"))):
    t = read_scored_games(d).set_index("game_key")
    cols = [f"sum_{lv}" for lv in LEVELS]
    diffs = np.stack([np.abs(t.loc[str(ref["game_key"][match[k]]), cols].to_numpy(float) - l1b_per_task[k][j]) for k in range(len(match))])
    show(m["name"], diffs)
    # signed mean difference per level: a systematic offset would point at a real mismatch, not noise
    sg = np.stack([l1b_per_task[k][j] - t.loc[str(ref["game_key"][match[k]]), cols].to_numpy(float) for k in range(len(match))])
    print("   signed mean Δ per move by head:", " ".join(f"{lv}:{v:+.5f}" for lv, v in zip(LEVELS, (sg / n[:, None]).mean(axis=0))))
