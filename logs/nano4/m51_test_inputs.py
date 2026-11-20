"""Section 4M M51 (2026-11-07): the test questions' model inputs, saved for the input-shift diagnosis (m51_shift.py). Report only.

Rebuilds what scripts/trunk_predict.py predict computed for ct9's run (shards results/ct6/submission_rank.csv x 8, vec
results/ct_submission/vec, the saved score arrays scores_oh_<L1b model>.npy):
  R [M, 800, 64]  the fused_opp_diff_hd features per L1b model m (question_r), questions in trunk_predict's order;
  qid, n_games, half per question;
  m51_test_games.csv  per usable game: task, question_id, half, hd (HD[] of the record, NaN when absent), the target's colour, n_moves;
  tok_<model>_{target,opp}_{mean,sd}  per raw token column (0-157: trunk 128, L1b 11, Human SL 19) over the target plies and the
                                      opponent plies of the test's tokens, as the "-oh" members read them.
  python -I logs/nano4/m51_test_inputs.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import predict_submission as ps  # noqa: E402
import trunk_predict as tp  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from aicup.sgf import handicap  # noqa: E402

SHARDS, NSHARDS, VEC = ROOT / "results/ct6/submission_rank.csv", 8, ROOT / "results/ct_submission/vec"
OUT = ROOT / "results/trunk_seq/es"
N_TOKEN_COLS = 158  # trunk 64 + 64, then the 30 scores: columns 158-161 are the flags


def column_stats(X, rows):
    """mean, sd of X[rows, :N_TOKEN_COLS] in float64, in chunks (X is float16 and large)."""
    s, s2, cnt = np.zeros(N_TOKEN_COLS), np.zeros(N_TOKEN_COLS), 0
    for i in range(0, len(rows), 500_000):
        blk = X[rows[i:i + 500_000], :N_TOKEN_COLS].astype(np.float64)
        s += blk.sum(0); s2 += (blk * blk).sum(0); cnt += len(blk)
    mean = s / max(cnt, 1)
    return mean, np.sqrt(np.maximum(s2 / max(cnt, 1) - mean * mean, 0.0)), cnt


def main():
    t0 = time.time()
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS, False)
    tasks, owner, dropped, _ = ps.build_tasks(questions)
    sha = ps.questions_sha(ps.DEFAULT_INPUTS, questions)
    per_task, _, _ = ps.load_shards(str(SHARDS), NSHARDS, len(tasks), sha, with_moves=False)
    l1b_models, l1b_per_task, _, opp_per_task = ps.load_shards_l1b(str(SHARDS), NSHARDS, with_moves=False)
    names = [m["name"] for m in l1b_models]
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    qis = sorted(by_q)
    task_games = {k: (sgf, c) for k, sgf, c in tasks}
    R = np.stack([tp.question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, task_games, m) for m in range(len(names))])
    qid = np.array([questions[qi]["question_id"] for qi in qis])
    n_games = np.array([len(by_q[qi]) for qi in qis])
    half = np.where(pd.Series(qid).str.startswith("pub").to_numpy(), "public", "private")
    print(f"{len(qis)} questions, {len(tasks)} usable games ({len(dropped)} dropped), R {R.shape} from L1b models {names} ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for qi in qis:
        q_id = questions[qi]["question_id"]
        for k in by_q[qi]:
            sgf, c = task_games[k]
            h = handicap(sgf)
            rows.append((k, q_id, "public" if q_id.startswith("pub") else "private", np.nan if h is None else float(h), c, int(per_task[k][0])))
    games = pd.DataFrame(rows, columns=["task", "question_id", "half", "hd", "color", "n_moves"])
    games.to_csv(OUT / "m51_test_games.csv", index=False)
    hd1 = games[games["hd"] == 1]
    print(f"games: HD1 share {float((games['hd'] == 1).mean()):.3f}; among HD1 games the target is black (lower hand) in {float((hd1['color'] == 'B').mean()):.3f} "
          f"(public {float((hd1[hd1['half'] == 'public']['color'] == 'B').mean()):.3f}, private {float((hd1[hd1['half'] == 'private']['color'] == 'B').mean()):.3f}); "
          f"target black overall {float((games['color'] == 'B').mean()):.3f}", flush=True)
    arrays = {"R": R, "qid": qid, "n_games": n_games, "half": half, "l1b_names": np.array(names)}
    for nm in names:
        sc_path = VEC / f"scores_oh_{nm}.npy"
        if not sc_path.exists():
            print(f"no {sc_path.name}: token statistics skipped for {nm}", flush=True)
            continue
        X, _ = ts.raw_tokens([(VEC, sc_path)])
        tgt = np.flatnonzero(X[:, 158].astype(np.float32) > 0.5)
        opp = np.flatnonzero(X[:, 158].astype(np.float32) <= 0.5)
        for role, rws in (("target", tgt), ("opp", opp)):
            mean, sd, cnt = column_stats(X, rws)
            arrays[f"tok_{nm}_{role}_mean"], arrays[f"tok_{nm}_{role}_sd"], arrays[f"tok_{nm}_{role}_plies"] = mean, sd, np.int64(cnt)
        print(f"tokens {nm}: {len(X)} plies, {len(tgt)} target plies ({time.time() - t0:.0f} s)", flush=True)
        del X
    np.savez(OUT / "m51_test_inputs.npz", **arrays)
    print(f"written {OUT / 'm51_test_inputs.npz'} and m51_test_games.csv ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
