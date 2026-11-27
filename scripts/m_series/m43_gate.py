"""Section 4M M43 (2026-11-07; user: 「把錯的 30% 再 train 一個模型去修正，然後用類似貝氏的方式選相信誰」): an error gate and an
error-only corrector on top of the pc ensemble's out-of-fold probabilities, cross-fitted over the four validation folds.

For seed set S (0 1 2 = ct9, or 3 4 5): P = mean of C/T-pc-f0-x40-oh seeds S on each validation fold; y-hat = the expected-score decision.
Features X: log P (10), entropy, top-2 margin, n_games, and the 64 R columns of r_val{v}_f0_x40 (X_va, the validation questions' order).
Trained on three folds' validation questions, scored on the fourth:
  gate      g = P(y-hat != y | X), MLP h32, class-balanced;
  corrector q(y | X, y-hat wrong), MLP h32 on the wrong questions only; at prediction q(y-hat) = 0, renormalised;
  bayes     P'(y) = (1 - g) 1[y = y-hat] + g q(y), then the expected-score decision;
  stack     (control) a 10-class MLP h32 on all questions, expected-score decision.
Against ct9's decision: paired player bootstrap (class-balanced official score) and the plain average; the new rule (CI lower > 0,
>= 3 of 4 folds > 0, diff >= 0.0025) must hold for both seed sets.
  python -I scripts/m_series/m43_gate.py --seeds 0 1 2 --out results/trunk_seq/es/m43_s012.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402

E1 = float(np.exp(-1.0))
FOLDS = (1, 2, 3, 4)


def decide(P):
    es = P.copy(); es[:, 1:] += E1 * P[:, :-1]; es[:, :-1] += E1 * P[:, 1:]
    return es.argmax(1)


def qscore(d, y):
    k = np.abs(d - y)
    return np.where(k == 0, 1.0, np.where(k == 1, E1, 0.0))


def features(P, n, R):
    lp = np.log(np.clip(P, 1e-9, 1)); ent = -(P * lp).sum(1, keepdims=True); s = np.sort(P, 1)
    return np.hstack([lp, ent, (s[:, -1] - s[:, -2])[:, None], n[:, None].astype(float), R]).astype(np.float32)


def train_mlp(X, y, n_out, groups, seed, hidden=32, epochs=80, lr=1e-3, wd=1e-4, batch=512, patience=6):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    players = np.unique(groups); held = set(rng.permutation(players)[:len(players) // 10]); h = np.array([g in held for g in groups])
    mu, sd = X[~h].mean(0), X[~h].std(0) + 1e-6
    Xt, Xh = torch.as_tensor((X[~h] - mu) / sd), torch.as_tensor((X[h] - mu) / sd)
    yt, yh = torch.as_tensor(y[~h]).long(), torch.as_tensor(y[h]).long()
    cnt = np.bincount(y[~h], minlength=n_out).astype(float); cw = torch.as_tensor(np.where(cnt > 0, cnt.sum() / (n_out * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, n_out))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, since = float("inf"), None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            j = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(net(Xt[j]), yt[j], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy(net(Xh), yh, weight=cw))
        if vl < best - 1e-5:
            best, best_state, since = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    return lambda Z: torch.softmax(net(torch.as_tensor((Z - mu) / sd)), 1).detach().numpy()


def auc(score, label):
    import pandas as pd
    r = pd.Series(score).rank().to_numpy(); n1 = label.sum(); n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in a.seeds], axis=0)
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
        assert len(R) == len(y), f"fold {v}: R rows {len(R)} != {len(y)} questions"
        d = decide(P)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "X": features(P, n, R), "d": d}
    yall = np.concatenate([D[v]["y"] for v in FOLDS]); dall = np.concatenate([D[v]["d"] for v in FOLDS]); k = np.abs(dall - yall)
    rep = {"seeds": a.seeds, "main": {"exact": float((k == 0).mean()), "adjacent": float((k == 1).mean()), "two_plus": float((k >= 2).mean())}, "folds": {}}
    print(f"seeds {a.seeds}: main decision exact {rep['main']['exact']:.3f}, adjacent {rep['main']['adjacent']:.3f}, two or more off {rep['main']['two_plus']:.3f}", flush=True)
    S = {"main": {}, "bayes": {}, "stack": {}}
    gates, wrongs = [], []
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        X = np.concatenate([D[u]["X"] for u in tr]); y = np.concatenate([D[u]["y"] for u in tr]); d = np.concatenate([D[u]["d"] for u in tr])
        groups = np.concatenate([D[u]["pid"] for u in tr]); wrong = (d != y).astype(int)
        gate = train_mlp(X, wrong, 2, groups, seed=v)
        corr = train_mlp(X[wrong == 1], y[wrong == 1], 10, groups[wrong == 1], seed=10 + v)
        stack = train_mlp(X, y, 10, groups, seed=20 + v)
        Xv, yv, dv = D[v]["X"], D[v]["y"], D[v]["d"]
        g = gate(Xv)[:, 1]; qv = corr(Xv); qv[np.arange(len(dv)), dv] = 0.0; qv /= qv.sum(1, keepdims=True)
        Pb = g[:, None] * qv; Pb[np.arange(len(dv)), dv] += 1.0 - g
        S["main"][v] = qscore(dv, yv); S["bayes"][v] = qscore(decide(Pb), yv); S["stack"][v] = qscore(decide(stack(Xv)), yv)
        gates.append(g); wrongs.append((dv != yv).astype(int))
        print(f"fold {v}: gate mean {g.mean():.3f} vs wrong rate {(dv != yv).mean():.3f}; bayes changes {(decide(Pb) != dv).mean():.3f} of the decisions", flush=True)
    g_all, w_all = np.concatenate(gates), np.concatenate(wrongs)
    pmax = np.concatenate([D[v]["P"].max(1) for v in FOLDS])
    rep["gate"] = {"auc": auc(g_all, w_all), "auc_of_1_minus_pmax": auc(1 - pmax, w_all), "mean_g": float(g_all.mean()), "wrong_rate": float(w_all.mean())}
    print(f"gate AUC {rep['gate']['auc']:.4f} (the main model's own 1 - max p: {rep['gate']['auc_of_1_minus_pmax']:.4f})", flush=True)
    y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in ("y", "n", "pid"))
    for arm in ("bayes", "stack"):
        s_new, s_old = np.concatenate([S[arm][v] for v in FOLDS]), np.concatenate([S["main"][v] for v in FOLDS])
        dd = paired_diff(s_new, s_old, y, pid, n, n=5000)
        pf = [balanced_mean_over_n(S[arm][v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(S["main"][v], D[v]["y"], D[v]["n"]) for v in FOLDS]
        ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
        rep["folds"][arm] = {**dd, "per_fold_diff": pf, "plain_diff": float((s_new - s_old).mean()), "pass_new_rule": bool(ok)}
        print(f"{arm} vs ct9 decision: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + f"  plain {rep['folds'][arm]['plain_diff']:+.4f}" + ("  -> PASS (new rule)" if ok else "  -> no"), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
