"""Permutation importance of ct9's inputs (2026-11-07, report only; user: 目前用的特徵影響最高的前 10 特徵). On validation fold v, the 6 pc
members (C/T-pc-f0-x40-oh seeds 0-2, ct9's members for that fold) and their probability average with the expected-score decision; the
drop of the class-balanced official score when one input is shuffled (one fixed permutation, the same for every member):
  token groups (shuffled across the fold's plies, within the role where a role is named): board trunk (0-63), played-point trunk (64-127),
    own L1b (128-138, target plies), opponent L1b (opponent plies), own Human SL (139-157, target plies), opponent Human SL (opponent
    plies), move number and pass flags (160-161);
  R: each of the 64 columns and each block (Human SL 24, L1b 16, opponent 16, diff 3, HD 5), shuffled across questions. The question
    embedding from the tokens is computed once per member, so an R shuffle only reruns the last MLP.
  python -I logs/nano4/feature_importance.py --val-fold 1 --out results/r_variants/importance_val1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_seq_check as ts  # noqa: E402
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES as R_NAMES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
TOKEN_GROUPS = {"board trunk": (slice(0, 64), None), "played-point trunk": (slice(64, 128), None), "own L1b": (slice(128, 139), 1),
                "opponent L1b": (slice(128, 139), 0), "own Human SL": (slice(139, 158), 1), "opponent Human SL": (slice(139, 158), 0),
                "move number + pass": (slice(160, 162), None)}
R_BLOCKS = {"R Human SL (24)": range(0, 24), "R L1b (16)": range(24, 40), "R opponent (16)": range(40, 56), "R diff (3)": range(56, 59), "R HD (5)": range(59, 64)}


def z_and_logits_fn(model, Xs, offsets, keys, device, batch=64):
    Z = []
    with torch.no_grad():
        for i in range(0, len(keys), batch):
            tokens, mask, game_of = ts.batch_tokens(Xs, offsets, keys[i:i + batch], device)
            v = model.encode_games(tokens, mask)
            Z.append(torch.relu(model.question(model.pool_games(v, game_of, len(keys[i:i + batch])))).cpu())
    return torch.cat(Z)


def probs_from(model, R_std, Z, device):
    with torch.no_grad():
        return torch.softmax(model.mlp(torch.cat([torch.as_tensor(R_std, dtype=torch.float32), Z], 1).to(device)), 1).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    t0, v, device = time.time(), a.val_fold, torch.device("cuda")
    rng = np.random.RandomState(1000 + v)
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.json"))["val"]
    keys = m["game_keys"]; y = np.array([RANK[r] for r in m["rank"]]); n = np.array(m["n_games"])
    import pandas as pd
    q = pd.DataFrame(m)
    R_raw = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
    X_raw, offsets = ts.raw_tokens([(ts.VEC / f"fold{v}", ts.scores_path(ROOT / "results/trunk_seq", v, ts.VEC, 3, True))])
    tgt = X_raw[:, 158].astype(np.float32) > 0.5
    members = [ts.load_checkpoint(ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.pt", device) for arm in ("C", "T") for s in (0, 1, 2)]
    print(f"fold {v}: {len(keys)} questions, {len(X_raw)} plies, {len(members)} members ({time.time() - t0:.0f} s)", flush=True)

    def score(P):
        return balanced_mean_over_n(evaluate(P, y, q, sizes=SIZES)[1], y, n)

    def token_pass(X_mod):
        Zs, Rs = [], []
        for model, tstats, rstats, _ in members:
            Xs = torch.as_tensor(ts.standardise_tokens(X_mod.copy(), tstats), device=device)
            Zs.append(z_and_logits_fn(model, Xs, offsets, keys, device)); Rs.append(((R_raw - rstats[0]) / rstats[1]).astype(np.float32))
            del Xs; torch.cuda.empty_cache()
        return Zs, Rs

    Zs, Rs = token_pass(X_raw)
    P0 = np.mean([probs_from(mod, R, Z, device) for (mod, _, _, _), R, Z in zip(members, Rs, Zs)], axis=0)
    base = score(P0)
    rep = {"val_fold": v, "base": base, "token_groups": {}, "r_columns": {}, "r_blocks": {}}
    print(f"base {base:.4f} ({time.time() - t0:.0f} s)", flush=True)
    perm_q = rng.permutation(len(keys))
    for name, cols in list(R_BLOCKS.items()) + [(c, [i]) for i, c in enumerate(R_NAMES)]:
        cols = list(cols)
        P = []
        for (mod, _, _, _), R, Z in zip(members, Rs, Zs):
            Rp = R.copy(); Rp[:, cols] = R[perm_q][:, cols]; P.append(probs_from(mod, Rp, Z, device))
        drop = base - score(np.mean(P, axis=0))
        (rep["r_blocks"] if name in R_BLOCKS else rep["r_columns"])[name] = drop
    print("R blocks: " + ", ".join(f"{k} {x:+.4f}" for k, x in rep["r_blocks"].items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    for name, (cols, role) in TOKEN_GROUPS.items():
        rows = np.arange(len(X_raw)) if role is None else np.flatnonzero(tgt == bool(role))
        X_mod = X_raw.copy(); X_mod[rows, cols] = X_raw[rows[rng.permutation(len(rows))], cols]
        Zp, _ = token_pass(X_mod)
        Pp = np.mean([probs_from(mod, R, Z, device) for (mod, _, _, _), R, Z in zip(members, Rs, Zp)], axis=0)
        rep["token_groups"][name] = base - score(Pp)
        print(f"token group {name:20s} drop {rep['token_groups'][name]:+.4f} ({time.time() - t0:.0f} s)", flush=True)
        del X_mod
    Path(a.out).write_text(json.dumps(rep, indent=1)); print("written", a.out)


if __name__ == "__main__":
    main()
