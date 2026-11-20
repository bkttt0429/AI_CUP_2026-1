#!/usr/bin/env python
"""M262 helper (2026-11-20): scripts/ai_state_extract.py for the TEST vec directory (results/ct_submission/vec), whose part tables
carry task ids instead of SGFs; the SGFs come from the two released test files through predict_submission's task builder. The
replay of every part is compared with the stored plies (pos / mover), as in ai_state_extract.py.

  python -I logs/nano4/m262_test_state.py --out /work/u2507115/m262_ai/test
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.humansl import load_humansl  # noqa: E402
from aicup.scoring import completed_parts, part_paths  # noqa: E402
import ai_state_extract as ax  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_extract as te  # noqa: E402
import trunk_predict as tp  # noqa: E402

VEC = ROOT / "results/ct_submission/vec"; INPUTS = [str(ROOT / "data/tests/rank_prediction_test_public.csv"), str(ROOT / "data/tests/rank_prediction_test_private.csv")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--batch", type=int, default=512); a = ap.parse_args()
    questions, _ = ps.load_questions(INPUTS, False); tasks, owner, dropped, _ = ps.build_tasks(questions); sgf_of = {tp.task_key(k): (sgf, c) for k, sgf, c in tasks}; qid_of = {tp.task_key(k): questions[owner[k]]["question_id"] for k, _, _ in tasks}
    print(f"{len(questions)} questions, {len(tasks)} games ({len(dropped)} dropped)", flush=True)
    config = torch.load(str(ax.CKPT), map_location="cpu", weights_only=False)["config"]; pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,)); out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, _ = load_humansl(str(ax.CKPT), "cuda"); model.eval(); parts = list(completed_parts(VEC)); t0, plies = time.time(), 0
        for n_part, (s, k) in enumerate(parts):
            dst = out_dir / f"ai_s{s}_p{k:03d}.npy"
            if dst.exists():
                continue
            csv_path, npz_path = part_paths(VEC, s, k); tab = pd.read_csv(csv_path, dtype={"game_key": str, "question_id": str}, keep_default_na=False); keys = tab["game_key"].tolist()
            assert all(qid_of[q] == x and sgf_of[q][1] == c for q, x, c in zip(keys, tab["question_id"], tab["color"])), f"{csv_path}: the task table does not match the released test files"
            with np.load(npz_path) as z:
                pos0, mover0 = z["pos"], z["mover"]
            done = list(pool.imap(te.replay, [(q, *sgf_of[q]) for q in keys], chunksize=1)); bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done]); pos = np.concatenate([d[3] for d in done]); mover = np.concatenate([d[4] for d in done])
            if not (np.array_equal(pos, pos0) and np.array_equal(mover, mover0)):
                raise ValueError(f"{npz_path}: the replay does not reproduce the stored plies")
            rows = ax.state_rows(model, bins, globs, pos, a.batch); assert np.isfinite(rows.astype(np.float32)).all()
            tmp = dst.with_name(dst.name + ".tmp.npy"); np.save(tmp, rows); os.replace(tmp, dst); plies += len(rows)
            if n_part % 10 == 0 or n_part == len(parts) - 1:
                print(f"part {n_part + 1}/{len(parts)}: {plies} plies, {plies / max(time.time() - t0, 1e-9):.0f} plies/s", flush=True)
        print(f"done: {plies} plies, {(time.time() - t0) / 60:.1f} min; written {out_dir}", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
