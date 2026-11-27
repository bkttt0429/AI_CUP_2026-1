#!/usr/bin/env python
"""Section 4M M197 (2026-11-16, report only): small-sample probe of the "choice residual" feature, full against reduced.

The identity network already receives the engine trunk feature AT THE PLAYED POINT (64-d random projection, u). Not given:
the same feature of the OTHER legal points. Proposed addition (outside discussion): r = feature at the played point minus the
policy-weighted mean of the feature over the legal points (what was chosen relative to what the baseline policy would have
chosen on average), plus three policy scalars (surprisal, entropy, rank of the played move). Storing it for all games costs
disk: full = both players' moves, 64-d (about 14.5 GB); reduced = the subject's moves only, 32-d (about 4.2 GB).
This probe extracts the FULL version for 400 players x 50 games and asks, with a deliberately simple reader (mean over the
moves of the first 80 plies and of the rest; whitened nearest prototype; 40 games per player to enrol, 10 to query; players
are each other's candidates): does r add identity information on top of u at all, how much is lost at 32 dimensions, and
does the opponent's r (context) matter. Also the extraction speed (plies per second on one GPU).

  sbatch -p 8gpus --gres=gpu:1 -c 12 ... run.sbatch python -I logs/nano4/m197_choice_probe.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m81_tactic_id as m81  # noqa: E402
import trunk_extract as te  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import _GraphExtraOutputs, neutral_rows  # noqa: E402

N_PL, N_G, N_ENROL, SPLIT = 300, 40, 30, 80  # second run: smaller sample (the replay is CPU-bound, about 800 plies/s per 10 workers)
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def safe_replay(task):
    try:
        return te.replay(task)
    except Exception as e:  # noqa: BLE001
        return (task[0], None, repr(e)[:80])


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rows = m81.load_rows(); rng = np.random.default_rng(0); cnt = rows.groupby("player_id").size(); pl = rng.choice(cnt.index[cnt >= N_G + 10].to_numpy(), N_PL, replace=False)
    sel = pd.concat([g.sample(N_G, random_state=0) for _, g in rows[rows.player_id.isin(set(pl))].groupby("player_id")], ignore_index=True); log(f"sample: {N_PL} players x {N_G} games = {len(sel)} games")
    tasks = list(zip(sel.game_key, sel.sgf_content, sel.color)); owner = dict(zip(sel.game_key, sel.player_id)); del rows
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]; pool = mp.get_context("fork").Pool(WORKERS, initializer=te.init_worker, initargs=(config,))
    model, _ = load_humansl(str(te.CKPT), "cuda"); model.eval(); nrows = torch.as_tensor(neutral_rows(), dtype=torch.float32, device="cuda"); proj = None; feats = {}; plies = 0; t1 = time.time()
    for start in range(0, len(tasks), 200):
        res = [d for d in pool.map(safe_replay, tasks[start:start + 200], chunksize=2) if d[1] is not None and len(d[3]) > 0]
        lens = np.array([len(d[3]) for d in res]); bins = np.concatenate([d[1] for d in res]); globs = np.concatenate([d[2] for d in res]); pos = np.concatenate([d[3] for d in res]).astype(np.int64); mover = np.concatenate([d[4] for d in res]).astype(np.int64); tgt = np.concatenate([d[5] for d in res]); n = len(pos)
        U = np.zeros((n, 64), np.float32); Rr = np.zeros((n, 64), np.float32); SC = np.zeros((n, 3), np.float32)
        with torch.no_grad():
            for i in range(0, n, 384):
                j = min(n, i + 384); b = torch.as_tensor(bins[i:j], device="cuda").float(); g = torch.as_tensor(globs[i:j], device="cuda").float(); c = torch.as_tensor(mover[i:j], device="cuda"); extra = _GraphExtraOutputs(["trunkfinal"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(b, g, input_meta=nrows[c], extra_outputs=extra)
                trunk = extra.returned["trunkfinal"].float().flatten(2); logits = model.postprocess_output(out)[0][0][:, 0, :].float()  # [B, 384, 361], [B, 362]
                if proj is None:
                    proj = te.projections(trunk.shape[1], trunk.device)[1]
                occ = (b[:, 1] + b[:, 2]).flatten(1) > 0; legal = torch.cat([~occ, torch.ones(j - i, 1, dtype=torch.bool, device="cuda")], 1); lp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1); p = lp.exp()
                a = torch.as_tensor(pos[i:j], device="cuda"); on = a < te.PASS_POS; mu = torch.einsum("bcp,bp->bc", trunk, p[:, :361]); u = torch.zeros_like(mu); u[on] = trunk[on.nonzero()[:, 0], :, a[on]]
                lpa = lp.gather(1, a[:, None])[:, 0]; ent = -(p * lp.masked_fill(~legal, 0.0)).sum(1); rank = (lp > lpa[:, None]).sum(1).float()
                U[i:j] = (u @ proj).cpu().numpy(); Rr[i:j] = ((u - mu) @ proj).cpu().numpy(); SC[i:j] = torch.stack([-lpa.clamp(min=-20.0), ent, torch.log1p(rank)], 1).cpu().numpy()
        plies += n; off = np.concatenate([[0], np.cumsum(lens)])
        for k, d in enumerate(res):
            s = slice(off[k], off[k + 1]); t = np.arange(lens[k]); me = tgt[s]; row = []
            for seg in (t < SPLIT, t >= SPLIT):
                for arr, who in ((U[s], me), (Rr[s], me), (SC[s], me), (Rr[s], ~me)):
                    m = seg & who; row.append(arr[m].mean(0) if m.any() else np.zeros(arr.shape[1], np.float32))
            feats[d[0]] = np.concatenate(row)
        if (start // 200) % 20 == 0:
            log(f"  {start + len(res)} games, {plies} plies, {plies / (time.time() - t1):.0f} plies/s")
    pool.close(); pool.join(); rate = plies / (time.time() - t1); log(f"extracted {len(feats)} games, {plies} plies: {rate:.0f} plies/s on one GPU with {WORKERS} replay workers -> all 1,000,000 games (about 215 million plies): {215e6 / rate / 3600:.1f} GPU-hours")
    keys = np.array(list(feats)); X = np.stack([feats[k] for k in keys]); own = np.array([owner[k] for k in keys]); up, y = np.unique(own, return_inverse=True); np.savez(ROOT / "results/r_variants/m197_probe.npz", X=X, y=y)
    # layout of one segment: u 64 | r 64 | scalars 3 | opponent's r 64  (two segments)
    W = 64 + 64 + 3 + 64; col = lambda a, b_: np.concatenate([np.arange(a, b_), np.arange(W + a, W + b_)])  # noqa: E731
    sets = {"u (already in the input)": col(0, 64), "r, 64-d": col(64, 128), "r, 32-d": col(64, 96), "policy scalars only": col(128, 131), "u + r 64-d": np.concatenate([col(0, 64), col(64, 128)]), "u + r 32-d": np.concatenate([col(0, 64), col(64, 96)]),
            "u + scalars": np.concatenate([col(0, 64), col(128, 131)]), "u + r 64-d + scalars": col(0, 131), "u + r 32-d + scalars": np.concatenate([col(0, 64), col(64, 96), col(128, 131)]), "u + r 64-d + scalars + opponent's r 64-d": col(0, 195), "u + r 64-d + scalars + opponent's r 32-d": np.concatenate([col(0, 131), col(131, 163)])}
    order = np.argsort(y, kind="stable"); rank_in = np.zeros(len(y), int)
    for c in range(len(up)):
        idx = order[np.searchsorted(y[order], c):np.searchsorted(y[order], c + 1)]; rank_in[rng.permutation(idx)] = np.arange(len(idx))
    enrol = rank_in < N_ENROL; print(f"\nprobe: {len(up)} players, {int(enrol.sum())} enrolment games, {int((~enrol).sum())} query games; whitened nearest prototype (owner first | in the first five)")
    for name, cols in sets.items():
        A = X[:, cols].astype(np.float64); mu_, sd_ = A[enrol].mean(0), A[enrol].std(0) + 1e-9; A = (A - mu_) / sd_; M = np.zeros((len(up), A.shape[1])); np.add.at(M, y[enrol], A[enrol]); M /= np.bincount(y[enrol], minlength=len(up))[:, None]
        Sw = np.cov((A[enrol] - M[y[enrol]]).T) + 0.05 * np.eye(A.shape[1]); Wm = np.linalg.cholesky(np.linalg.inv(Sw)); Q = A[~enrol] @ Wm; Pm = M @ Wm; D = ((Q ** 2).sum(1)[:, None] - 2 * Q @ Pm.T + (Pm ** 2).sum(1)[None]); r = (D < D[np.arange(len(Q)), y[~enrol]][:, None]).sum(1)
        print(f"  {name:44s} {len(cols):4d} dims: {float((r == 0).mean()):.3f} | {float((r < 5).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
