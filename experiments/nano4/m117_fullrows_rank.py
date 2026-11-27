#!/usr/bin/env python
"""M117: does the fixed random projection of the trunk vectors lose RANK information? (report only)

Every stored trunk feature of this project is the humanv0 trunk's 384-channel vector pushed through a fixed random projection to
64 dimensions (board-pooled and played-point). The player-identification session stored, for 77,372 training games of folds 1-4,
the usual 256-dim row (mean / std of the two projected vectors over the target's plies) next to the same statistics of the
UNPROJECTED vectors (1,536 dims), both from one trunk pass (/work/u2507115/pid_v2/fullrows/rows, read only here). It measured
that its identity embedding goes from 0.277 to 0.485 on the unprojected rows. This script asks the same question for the level.

The same per-game learner (10 levels, cross-entropy) is trained on each representation
  rp256   the usual row                         full    mean / std of the two unprojected vectors (1,536)
  rp128m  the two projected means only          fullm   the two unprojected means only (768)
with players split by their fold (train on three of folds 1-4, test on the fourth; a tenth of the training players is held out
to pick the weight decay and the stopping epoch; three seeds averaged). Learners: linear, and one hidden layer of 1,024 units.
Reported per representation and learner: per-game exact / score, and the competition score (1, 1/e, 0) of the pooled decision
(arg max of the summed per-game log-probabilities) over 5 games, 10 games and all of a player's sampled games (8-16), with a
bootstrap over players of the difference from rp256; and the split of the expected-level error into a per-player part and a
per-game part. Fold 0 is never read.

  python -I logs/nano4/m117_fullrows_rank.py --gpus 2 3 4
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import player_games  # noqa: E402

C = 384
E1 = float(np.exp(-1.0))
WDS = (1e-4, 1e-2, 1e-1)
SEEDS = (0, 1, 2)
LEARNERS = {"lin": 0, "mlp": 1024}
FOLDS = (1, 2, 3, 4)
OUT = ROOT / "results" / "r_variants"


def load_rows(d):
    keys, rp, full = [], [], []
    for f in sorted(glob.glob(f"{d}/*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        keys.append(z["keys"].astype(str))
        rp.append(z["feats"].astype(np.float32))
        full.append(z["full"].astype(np.float32))
    return np.concatenate(keys), np.concatenate(rp), np.concatenate(full)


def representations(RP, FULL):
    return {"rp256": RP, "full": FULL, "rp128m": RP[:, np.r_[0:64, 128:192]], "fullm": FULL[:, np.r_[0:C, 2 * C:3 * C]]}


def fit(Xtr, ytr, Xva, yva, hidden, wd, seed, max_epochs=80, patience=8, batch=1024, lr=1e-3):
    """One learner with early stopping on the held-out players' per-game cross-entropy; returns (net, best loss, epochs used)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if hidden:
        net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, len(LEVELS)))
    else:
        net = nn.Linear(d, len(LEVELS))
    net = net.to(Xtr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad, used = float("inf"), None, 0, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(Xtr.device)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.cross_entropy(net(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = nn.functional.cross_entropy(net(Xva), yva).item()
        if va < best - 1e-4:
            best, bad, used = va, 0, ep + 1
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, best, used


def run_fold(v, gpu, reps, y, fold, player, out_path):
    """Train on the players of the other folds, write the per-game log-probabilities of fold v's games for every (rep, learner)."""
    import torch
    t0 = time.time()
    dev = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    te = np.flatnonzero(fold == v)
    pool = np.flatnonzero(fold != v)
    players = np.unique(player[pool])
    held = set(np.random.RandomState(v).permutation(players)[: max(1, len(players) // 10)].tolist())
    is_va = np.array([p in held for p in player[pool]])
    tr, va = pool[~is_va], pool[is_va]
    yt = torch.as_tensor(y).to(dev)
    res, info = {}, {}
    for name, X in reps.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = torch.as_tensor(((X - mu) / sd).astype(np.float32)).to(dev)
        for learner, hidden in LEARNERS.items():
            first = {wd: fit(Z[tr], yt[tr], Z[va], yt[va], hidden, wd, SEEDS[0]) for wd in WDS}
            wd = min(first, key=lambda w: first[w][1])
            nets = [first[wd]] + [fit(Z[tr], yt[tr], Z[va], yt[va], hidden, wd, s) for s in SEEDS[1:]]
            with torch.no_grad():
                p = torch.stack([torch.softmax(n(Z[te]), 1) for n, _, _ in nets]).mean(0)
            res[f"{name}|{learner}"] = torch.log(p.clamp_min(1e-12)).cpu().numpy().astype(np.float32)
            info[f"{name}|{learner}"] = {"wd": wd, "val_ce": [float(n[1]) for n in nets], "epochs": [int(n[2]) for n in nets],
                                         "val_ce_by_wd": {str(w): float(first[w][1]) for w in WDS}}
            print(f"  fold {v} {name:7s} {learner}: wd {wd:g}, held-out CE {np.mean([n[1] for n in nets]):.4f}, epochs {[n[2] for n in nets]} ({time.time() - t0:.0f} s)", flush=True)
        del Z
    np.savez(out_path, idx=te, info=json.dumps(info), **res)


def score(pred, truth):
    d = np.abs(pred - truth)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def questions(player, keys, test_games, min_all=8):
    """Per player: its games in a fixed random order -> lists of game indices for 5-game, 10-game and all-game questions."""
    by = {}
    for i in test_games:
        by.setdefault(player[i], []).append(i)
    q = {"5": [], "10": [], "all": []}
    for p, idx in by.items():
        idx = np.array(sorted(idx, key=lambda i: keys[i]))
        idx = idx[np.random.RandomState(int(p[:8], 16) % 2 ** 32).permutation(len(idx))]
        for j in range(0, len(idx) - 4, 5):
            q["5"].append(idx[j:j + 5])
        if len(idx) >= 10:
            q["10"].append(idx[:10])
        if len(idx) >= min_all:
            q["all"].append(idx)
    return q


def boot(d, cluster, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num, den = np.bincount(inv, weights=d, minlength=len(ids)), np.bincount(inv, minlength=len(ids)).astype(float)
    pick = rng.integers(0, len(ids), (B, len(ids)))
    bs = num[pick].sum(1) / den[pick].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="/work/u2507115/pid_v2/fullrows/rows")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="m117_fullrows_rank")
    a = ap.parse_args(argv)
    t0 = time.time()
    keys, RP, FULL = load_rows(a.rows)
    assert len(set(keys)) == len(keys)
    games = player_games()
    folds = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold", "n_games"]]
    meta = pd.DataFrame({"game_key": keys}).merge(games, on="game_key", how="left").merge(folds, on="player_id", how="left")
    assert len(meta) == len(keys) and meta.player_id.notna().all() and meta.fold.notna().all()
    assert (meta["rank"].to_numpy() == np.array([k.split(":")[0] for k in keys])).all()
    fold = meta.fold.to_numpy().astype(int)
    assert not (fold == 0).any(), "fold 0 is never read here"
    y = meta["rank"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy().astype(np.int64)
    player = meta.player_id.to_numpy().astype(str)
    n_games = meta.n_games.to_numpy().astype(int)
    print(f"{len(keys)} games of {len(set(player))} players; by fold: " + ", ".join(f"{v}: {int((fold == v).sum())} games / {len(set(player[fold == v]))} players" for v in FOLDS) + f" ({time.time() - t0:.0f} s)", flush=True)
    reps = representations(RP, FULL)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {v: OUT / f"{a.tag}_fold{v}.npz" for v in FOLDS}
    kids = []
    for i, v in enumerate(FOLDS):
        pid = os.fork()  # the parent never touches CUDA
        if pid == 0:
            try:
                run_fold(v, a.gpus[i % len(a.gpus)], reps, y, fold, player, paths[v])
                os._exit(0)
            except BaseException:
                import traceback
                traceback.print_exc()
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(1)
        kids.append(pid)
    bad = [k for k in kids if os.waitpid(k, 0)[1] != 0]
    if bad:
        raise SystemExit(f"{len(bad)} fold workers failed")
    print(f"all folds trained ({time.time() - t0:.0f} s)", flush=True)

    names = [f"{r}|{l}" for r in reps for l in LEARNERS]
    L = {n: np.zeros((len(keys), len(LEVELS)), np.float32) for n in names}
    info = {}
    for v in FOLDS:
        z = np.load(paths[v])
        info[str(v)] = json.loads(str(z["info"]))
        for n in names:
            L[n][z["idx"]] = z[n]
    q = questions(player, keys, np.arange(len(keys)))
    rng = np.random.default_rng(0)
    out = {"games": int(len(keys)), "players": int(len(set(player))), "fits": info, "scores": {}, "diff_vs_rp256": {}, "error_split": {}}
    S = {}
    lv = np.arange(len(LEVELS))
    for n in names:
        row = {"game_exact": float((L[n].argmax(1) == y).mean()), "game_score": float(score(L[n].argmax(1), y).mean())}
        for size, qs in q.items():
            pred = np.array([L[n][g].sum(0).argmax() for g in qs])
            truth = np.array([y[g[0]] for g in qs])
            s = score(pred, truth)
            S[(n, size)] = s
            row[f"q{size}"] = float(s.mean())
            row[f"q{size}_exact"] = float((pred == truth).mean())
            if size == "all":
                row["qall_balanced"] = float(np.mean([s[truth == c].mean() for c in lv if (truth == c).any()]))
                row["qall_by_level"] = {LEVELS[c]: float(s[truth == c].mean()) for c in lv if (truth == c).any()}
                act = np.array([n_games[g[0]] for g in qs])
                row["qall_le40"], row["qall_gt40"] = float(s[act <= 40].mean()), float(s[act > 40].mean())
                row["qall_by_fold"] = {str(v): float(s[np.array([fold[g[0]] for g in qs]) == v].mean()) for v in FOLDS}
        out["scores"][n] = row
        # expected-level error: per-player and per-game parts (players of the all-game questions; per-level mean removed)
        e = (np.exp(L[n]) * lv).sum(1) - y
        for c in lv:
            e[y == c] -= e[y == c].mean()
        m = np.array([e[g].mean() for g in q["all"]])
        w = np.concatenate([e[g] - e[g].mean() for g in q["all"]])
        k = np.array([len(g) for g in q["all"]])
        s2w = float((w ** 2).sum() / (k - 1).sum())
        s2p = float(m.var() - s2w * (1.0 / k).mean())
        out["error_split"][n] = {"per_game_sd": s2w ** 0.5, "per_player_sd": max(s2p, 0.0) ** 0.5}
    sizes = {size: (np.array([player[g[0]] for g in qs]), np.array([fold[g[0]] for g in qs]), np.array([n_games[g[0]] for g in qs])) for size, qs in q.items()}
    for l in LEARNERS:
        for r in reps:
            if r == "rp256":
                continue
            row = {}
            for size in q:
                d = S[(f"{r}|{l}", size)] - S[(f"rp256|{l}", size)]
                cl, fo, act = sizes[size]
                row[f"q{size}"] = {"diff": float(d.mean()), "ci95": boot(d, cl, a.boot, rng), "by_fold": [float(d[fo == v].mean()) for v in FOLDS]}
                if size == "all":
                    row["qall_le40"] = {"diff": float(d[act <= 40].mean()), "ci95": boot(d[act <= 40], cl[act <= 40], a.boot, rng)}
            out["diff_vs_rp256"][f"{r}|{l}"] = row
    d = S[("fullm|mlp", "all")] - S[("rp128m|mlp", "all")]
    out["diff_fullm_vs_rp128m_mlp_qall"] = {"diff": float(d.mean()), "ci95": boot(d, sizes["all"][0], a.boot, rng)}

    print(f"\nquestions: 5 games {len(q['5'])}, 10 games {len(q['10'])}, all games {len(q['all'])} (mean {np.mean([len(g) for g in q['all']]):.1f} games)")
    print(f"{'representation|learner':24s} game exact  game score   q5      q10     qall   (exact)  balanced  <=40 g  per-player sd / per-game sd")
    for n in names:
        r, e = out["scores"][n], out["error_split"][n]
        print(f"{n:24s} {r['game_exact']:.4f}      {r['game_score']:.4f}      {r['q5']:.4f}  {r['q10']:.4f}  {r['qall']:.4f}  {r['qall_exact']:.4f}   {r['qall_balanced']:.4f}    {r['qall_le40']:.4f}  {e['per_player_sd']:.3f} / {e['per_game_sd']:.3f}")
    print("\ndifference from rp256 (same learner), bootstrap over players:")
    for n, row in out["diff_vs_rp256"].items():
        print(f"  {n:12s} " + "  ".join(f"q{s} {row[f'q{s}']['diff']:+.4f} [{row[f'q{s}']['ci95'][0]:+.4f}, {row[f'q{s}']['ci95'][1]:+.4f}]" for s in q)
              + f"  folds(qall) {' '.join(f'{x:+.3f}' for x in row['qall']['by_fold'])}  <=40 games {row['qall_le40']['diff']:+.4f} [{row['qall_le40']['ci95'][0]:+.4f}, {row['qall_le40']['ci95'][1]:+.4f}]")
    b = out["diff_fullm_vs_rp128m_mlp_qall"]
    print(f"  means only, unprojected against projected (mlp, qall): {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
    (OUT / f"{a.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    for p in paths.values():
        p.unlink()
    print(f"wrote {OUT / (a.tag + '.json')} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
