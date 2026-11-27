"""Section 4M M74 (2026-11-10): per-ply game STATE from a strong non-human KataGo net (kata1-b18c384nbt-s9996604416, no search; external
model, to be disclosed), aligned ply by ply with an existing trunk-vector directory (results/trunk_vec, results/trunk_vec_x40, a test
vec directory), so the sequence arms can read who is ahead, by how much, and what each move cost.
For every part <vec>/fold<f>/moves_s<s>_p<k>.npz the games are replayed exactly as scripts/trunk_extract.py does (same replay function;
the stored pos / mover are compared), and <out>/fold<f>/ai_s<s>_p<k>.npy holds [plies, 6] float16 for the side to move:
  0 log p of the played move under the strong policy   1 played move == the policy's argmax   2 policy entropy
  3 log p played - log p best                           4 win probability                    5 score lead (points)
  bash scripts/nano4/shards.sh 8 python -I scripts/ai_state_extract.py --vec results/trunk_vec --out results/ai_state/base --folds 0 1 2 3 4
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.humansl import load_humansl  # noqa: E402
from aicup.scoring import completed_parts, load_games, part_paths  # noqa: E402
import trunk_extract as te  # noqa: E402

CKPT = ROOT / "models/katago/kata1_b18c384nbt/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt"


def state_rows(model, bins, globs, pos, batch):
    out = np.empty((len(pos), 6), np.float16); has_meta = model.get_has_metadata_encoder()
    with torch.no_grad():
        for i in range(0, len(pos), batch):
            j = min(len(pos), i + batch)
            b = torch.as_tensor(bins[i:j], device="cuda").float(); g = torch.as_tensor(globs[i:j], device="cuda").float(); tgt = torch.as_tensor(pos[i:j].astype(np.int64), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = model(b, g, input_meta=torch.zeros(j - i, 192, device="cuda")) if has_meta else model(b, g)
            head = model.postprocess_output(outs)[0]
            lp = torch.log_softmax(head[0][:, 0, :].float(), dim=1); p = lp.exp()
            lp_t = lp.gather(1, tgt[:, None])[:, 0]; best = lp.max(dim=1).values
            ent = -(p * lp.clamp(min=-30)).sum(1)
            win = torch.softmax(head[1].float(), dim=1)[:, 0]; lead = head[10].float().reshape(-1)
            out[i:j] = torch.stack([lp_t.clamp(min=-30), (lp.argmax(1) == tgt).float(), ent, (lp_t - best).clamp(min=-30), win, lead.clamp(-300, 300)], dim=1).cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vec", required=True); ap.add_argument("--out", required=True); ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--flat", action="store_true", help="the vec directory holds the parts itself (a test directory), no fold<f> level")
    a = ap.parse_args()
    config = torch.load(str(CKPT), map_location="cpu", weights_only=False)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda"); model.eval()
        for f in ([None] if a.flat else a.folds):
            fold_dir = Path(a.vec) if f is None else Path(a.vec) / f"fold{f}"; out_dir = Path(a.out) if f is None else Path(a.out) / f"fold{f}"; out_dir.mkdir(parents=True, exist_ok=True)
            parts = [(s, k) for s, k in completed_parts(fold_dir) if s % a.nshards == a.shard]; t0, plies = time.time(), 0
            for n_part, (s, k) in enumerate(parts):
                dst = out_dir / f"ai_s{s}_p{k:03d}.npy"
                if dst.exists():
                    continue
                csv_path, npz_path = part_paths(fold_dir, s, k)
                tab = pd.read_csv(csv_path, dtype={"game_key": str}, keep_default_na=False); keys = tab["game_key"].tolist()
                with np.load(npz_path) as z:
                    pos0, mover0 = z["pos"], z["mover"]
                if "sgf" in tab.columns:  # a test directory keeps its SGFs in the part table
                    tasks = [(q, s_, c) for q, s_, c in zip(keys, tab["sgf"], tab["color"])]
                else:
                    sgf = load_games(keys); tasks = [(q, *sgf[q]) for q in keys]
                done = list(pool.imap(te.replay, tasks, chunksize=1))
                bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done]); pos = np.concatenate([d[3] for d in done]); mover = np.concatenate([d[4] for d in done])
                if not (np.array_equal(pos, pos0) and np.array_equal(mover, mover0)):
                    raise ValueError(f"{npz_path}: the replay does not reproduce the stored plies")
                rows = state_rows(model, bins, globs, pos, a.batch)
                if not np.isfinite(rows.astype(np.float32)).all():
                    raise ValueError(f"{npz_path}: non-finite state")
                tmp = dst.with_name(dst.name + ".tmp.npy"); np.save(tmp, rows); os.replace(tmp, dst); plies += len(rows)
                if n_part % 10 == 0 or n_part == len(parts) - 1:
                    print(f"{fold_dir.name} part {n_part + 1}/{len(parts)}: {plies} plies, {plies / max(time.time() - t0, 1e-9):.0f} plies/s", flush=True)
            print(f"{fold_dir} done: {plies} plies this run, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
