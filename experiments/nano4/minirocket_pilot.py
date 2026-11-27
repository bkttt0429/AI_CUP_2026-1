#!/usr/bin/env python
"""Section 4M M41 (2026-11-07; user: 「MINIROCKET 這生成特徵如何」→「好」): MINIROCKET random-convolution features of each game's per-ply
rank-evidence series, pooled per question, ridge to the rank index, and the partial correlation with the label given E (ct9's val1 members).

Series (from the token score arrays, oh layout: columns 128-138 L1b level + 10 centred, 139-157 Human SL level + 18 centred, 158 is_target):
for the target's plies and the opponent's, the posterior expected level of the 10 L1b heads and the posterior expected position of the
18 Human SL profiles -- 4 series a game, each cut to the first 128 plies of that side (zero-padded, valid length kept).
MINIROCKET (Dempster et al. 2021), univariate per series: the 84 length-9 kernels with three weights 2 and six -1, dilations 1 2 3 5 8 12,
'same' zero padding, two biases per (series, kernel, dilation) at the 25 % and 75 % quantiles of the outputs on 512 training games, PPV over
the valid positions only: 4 x 84 x 6 x 2 = 4,032 features a game; a question is the mean over its games.
Training: players of folds 0, 2, 3, 4, seven questions each (aicup.pilot.pilot_questions: 1 x 20, 2 x 10, 4 x 5); ridge on standardised
features, lambda picked on 10 % held-out training players from {0.01, 0.1, 1, 10, 100, 1e3, 1e4} x the row count. Evaluation: fold 1's 20-game
validation questions (r_val1.json), s = the ridge prediction; partial(s, y | E) with a player bootstrap; control: the same pipeline on the 4
series' per-game mean and std (8 features). Continue (pre-registered): partial CI lower > 0 and >= 0.08.
  python -I logs/nano4/minirocket_pilot.py --out results/r_variants/m41_minirocket.json [--smoke]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_seq_check as ts  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
L_MAX, DILATIONS, QUANTILES = 128, (1, 2, 3, 5, 8, 12), (0.25, 0.75)
TRAIN_FOLDS = (0, 2, 3, 4)


def kernels():
    W = -np.ones((84, 9), np.float32)
    for i, c in enumerate(itertools.combinations(range(9), 3)):
        W[i, list(c)] = 2.0
    return torch.as_tensor(W)[:, None, :]


def game_series(X, offsets, keys):
    """[G, 4, L_MAX] float32 series and [G, 4] valid lengths (target L1b, target HSL, opponent L1b, opponent HSL)."""
    S = np.zeros((len(keys), 4, L_MAX), np.float32); lens = np.zeros((len(keys), 4), np.int64)
    for g, k in enumerate(keys):
        s, n = offsets[k]
        x = X[s:s + n].astype(np.float32)
        tgt = x[:, 158] > 0.5
        l1b, hsl = x[:, 129:139], x[:, 140:158]
        el = (np.exp(l1b - l1b.max(1, keepdims=True)) / np.exp(l1b - l1b.max(1, keepdims=True)).sum(1, keepdims=True)) @ np.arange(10)
        eh = (np.exp(hsl - hsl.max(1, keepdims=True)) / np.exp(hsl - hsl.max(1, keepdims=True)).sum(1, keepdims=True)) @ np.arange(18)
        for c, (v, m) in enumerate(((el, tgt), (eh, tgt), (el, ~tgt), (eh, ~tgt))):
            seq = v[m][:L_MAX]
            S[g, c, :len(seq)] = seq; lens[g, c] = len(seq)
    return S, lens


def conv_all(S, W, device):
    """{(c, d): [G, 84, L]} convolution outputs of every series channel c and dilation d."""
    out = {}
    x = torch.as_tensor(S, device=device)
    for c in range(4):
        for d in DILATIONS:
            out[(c, d)] = torch.nn.functional.conv1d(x[:, c:c + 1], W.to(device), padding=4 * d, dilation=d)
    return out


def fit_biases(S, lens, W, device):
    B = {}
    conv = conv_all(S, W, device)
    for (c, d), o in conv.items():
        valid = torch.arange(L_MAX, device=device)[None, :] < torch.as_tensor(lens[:, c], device=device)[:, None]
        vals = o.permute(1, 0, 2)[:, valid]  # [84, n_valid]
        B[(c, d)] = torch.quantile(vals.float(), torch.tensor(QUANTILES, device=device), dim=1).T  # [84, 2]
    return B


def transform(S, lens, W, B, device, batch=2048):
    F = []
    for lo in range(0, len(S), batch):
        conv = conv_all(S[lo:lo + batch], W, device)
        ln = torch.as_tensor(lens[lo:lo + batch], device=device)
        parts = []
        for (c, d), o in conv.items():
            valid = (torch.arange(L_MAX, device=device)[None, :] < ln[:, c:c + 1]).float()  # [g, L]
            ppv = ((o[:, :, None, :] > B[(c, d)][None, :, :, None]).float() * valid[:, None, None, :]).sum(-1) / ln[:, c].clamp(min=1)[:, None, None]
            parts.append(ppv.flatten(1))
        F.append(torch.cat(parts, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(F)


def summary(S, lens):
    out = np.zeros((len(S), 8), np.float32)
    for c in range(4):
        for g in range(len(S)):
            v = S[g, c, :lens[g, c]]
            out[g, 2 * c], out[g, 2 * c + 1] = (v.mean(), v.std()) if len(v) else (0.0, 0.0)
    return out


def question_matrix(Fg, index, questions):
    return np.stack([Fg[[index[k] for k in ks]].mean(axis=0) for ks in questions["game_keys"]])


def ridge_fit_predict(Xtr, ytr, ptr, Xev, rng):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z, Ze = (Xtr - mu) / sd, (Xev - mu) / sd
    players = np.unique(ptr); held = set(rng.permutation(players)[:len(players) // 10]); h = np.array([p in held for p in ptr])

    def solve(A, y, lam):
        A1 = np.hstack([A, np.ones((len(A), 1), A.dtype)])
        R = A1.T @ A1 + lam * np.eye(A1.shape[1]); R[-1, -1] -= lam
        return np.linalg.solve(R, A1.T @ y)
    best = None
    for f in (0.01, 0.1, 1, 10, 100, 1e3, 1e4):
        lam = f * len(Z)
        w = solve(Z[~h], ytr[~h], lam)
        mse = float(np.mean((np.hstack([Z[h], np.ones((h.sum(), 1))]) @ w - ytr[h]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, f)
    w = solve(Z, ytr, best[1] * len(Z))
    return np.hstack([Ze, np.ones((len(Ze), 1))]) @ w, best


def main():
    from oppview_eval import boot, partial
    from pilot_stats import spearman
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda"); rng = np.random.RandomState(0)
    all_games, folds_df = player_games(), pd.read_csv(FOLDS_CSV)
    tr_q = []
    for f in TRAIN_FOLDS:
        pl = fold_players(all_games, folds_df, f, max_games=20, min_games=20)
        if a.smoke:
            pl = pl.iloc[::40]
        tr_q.append(pilot_questions(pl))
    tr_q = pd.concat(tr_q, ignore_index=True)
    ev = pd.DataFrame(json.load(open(ROOT / "results/trunk_seq/r_val1.json"))["val"]); ev["row"] = np.arange(len(ev))
    ev20 = ev[ev.n_games == 20].drop_duplicates("player_id")
    if a.smoke:
        ev20 = ev20.iloc[::20]
    keys_needed = sorted({k for ks in tr_q["game_keys"] for k in ks} | {k for ks in ev20["game_keys"] for k in ks})
    print(f"training questions {len(tr_q)} ({tr_q.player_id.nunique()} players), evaluation 20-game questions {len(ev20)}; games {len(keys_needed)}", flush=True)
    S_all, L_all, index = [], [], {}
    for f in sorted(set(TRAIN_FOLDS) | {1}):
        X, offsets = ts.raw_tokens([(ts.VEC / f"fold{f}", ts.scores_path(ROOT / "results/trunk_seq", f, ts.VEC, 3, True))])
        ks = [k for k in keys_needed if k in offsets]
        S, ln = game_series(X, offsets, ks)
        for k in ks:
            index[k] = len(index)
        S_all.append(S); L_all.append(ln)
        print(f"fold {f}: {len(ks)} games' series ({time.time() - t0:.0f} s)", flush=True)
        del X
    S, ln = np.concatenate(S_all), np.concatenate(L_all)
    missing = [k for k in keys_needed if k not in index]
    assert not missing, f"{len(missing)} games without tokens, e.g. {missing[:3]}"
    W = kernels()
    tr_keys = sorted({k for ks in tr_q["game_keys"] for k in ks})
    samp = np.array([index[k] for k in rng.permutation(tr_keys)[:512]])
    B = fit_biases(S[samp], ln[samp], W, device)
    Fg = transform(S, ln, W, B, device)
    Sg = summary(S, ln)
    print(f"MINIROCKET features {Fg.shape} ({time.time() - t0:.0f} s)", flush=True)
    ytr = tr_q["rank"].map(RANK).to_numpy(float); ptr = tr_q["player_id"].to_numpy(str)
    meta = ev
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val1/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    E = (P @ np.arange(len(LEVELS)))[ev20["row"].to_numpy()]; yev = ev20["rank"].map(RANK).to_numpy(float)
    rep = {"train_questions": int(len(tr_q)), "eval_players": int(len(ev20)), "features": int(Fg.shape[1]), "arms": {}}
    for name, G in (("minirocket", Fg), ("summary_control", Sg)):
        s, (mse, lam) = ridge_fit_predict(question_matrix(G, index, tr_q), ytr, ptr, question_matrix(G, index, ev20), rng)
        if len(yev) >= 30:
            pe = partial(s, yev, [E]); ci = boot(lambda idx: partial(s[idx], yev[idx], [E[idx]]), len(s))
        else:
            pe, ci = float("nan"), [float("nan"), float("nan")]
        r = {"partial_given_E": pe, "ci95": ci, "spearman_y": spearman(s, yev), "lambda_factor": lam, "held_mse": mse, "continue": bool(ci[0] > 0 and pe >= 0.08)}
        rep["arms"][name] = r
        print(f"{name:16s} partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  Spearman y {r['spearman_y']:.3f}  lambda {lam:g} x n  held MSE {mse:.3f}"
              + ("  -> CONTINUE" if r["continue"] and name == "minirocket" else ""), flush=True)
    rep["E_spearman_y"] = spearman(E, yev)
    print(f"reference: E Spearman y {rep['E_spearman_y']:.3f} on {len(yev)} players ({time.time() - t0:.0f} s)")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
