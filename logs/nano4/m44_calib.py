"""Section 4M M44 (2026-11-07; user: 「試試看你的方法」 after M43): a calibrated gate with an expected-score switch rule (G1), and a
feature-conditional recalibration of the ensemble posterior (G2; G2-nodis without the member-disagreement features), on the same out-of-
fold data and cross-fit as M43. Each outer fold's training folds are split by player 90 % / 10 % (fixed seed): G1's gate and corrector
(natural-frequency cross-entropy) train on the 90 %, the gate is Platt-calibrated and tau picked on the 10 %; G2 trains on the 90 % and
stops on the 10 %. Inputs: M43's 77 columns + 4 member-disagreement columns (sd of the 6 members' expected rank, share of members whose
decision differs from the ensemble's, KL(C mean || T mean), |E_C - E_T|). Diagnostic: cross-fitted logistic AUC for y-hat != y from
1 - max p alone vs + the 4 disagreement columns. New rule for adoption, in both seed sets.
  python -I logs/nano4/m44_calib.py --seeds 0 1 2 --out results/trunk_seq/es/m44_s012.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import E1, FOLDS, auc, decide, features, qscore  # noqa: E402

TAUS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def disagreement(Pm_C, Pm_T, d_ens):
    Pm = Pm_C + Pm_T
    Em = np.stack([p @ np.arange(10) for p in Pm]); dm = np.stack([decide(p) for p in Pm])
    PC, PT = np.mean(Pm_C, 0), np.mean(Pm_T, 0)
    kl = (PC * (np.log(np.clip(PC, 1e-9, 1)) - np.log(np.clip(PT, 1e-9, 1)))).sum(1)
    return np.stack([Em.std(0), (dm != d_ens[None]).mean(0), kl, np.abs(PC @ np.arange(10) - PT @ np.arange(10))], 1).astype(np.float32)


def split90(groups, seed):
    rng = np.random.RandomState(seed); players = np.unique(groups)
    held = set(rng.permutation(players)[:len(players) // 10]); return np.array([g in held for g in groups])


def fit_net(net, loss_fn, X, y, groups, seed, epochs=80, lr=1e-3, wd=1e-4, batch=512, patience=6, extra=None):
    """Natural-frequency training with early stopping on 10 % of the given rows' players; returns the standardisation."""
    torch.manual_seed(seed); h = split90(groups, seed + 1000)
    mu, sd = X[~h].mean(0), X[~h].std(0) + 1e-6
    T = lambda Z: torch.as_tensor((Z - mu) / sd)  # noqa: E731
    Xt, Xh, yt, yh = T(X[~h]), T(X[h]), torch.as_tensor(y[~h]).long(), torch.as_tensor(y[h]).long()
    et, eh = (None, None) if extra is None else (torch.as_tensor(extra[~h]), torch.as_tensor(extra[h]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, state, since = float("inf"), None, 0
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            j = perm[i:i + batch]
            loss = loss_fn(net, Xt[j], yt[j], None if et is None else et[j]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net, Xh, yh, eh))
        if vl < best - 1e-5:
            best, state, since = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(state); net.eval()
    return mu, sd


def mlp(d_in, d_out):
    return torch.nn.Sequential(torch.nn.Linear(d_in, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, d_out))


class Recal(torch.nn.Module):
    """t(x) = softplus(.) starting at 1, b(x) starting at 0: logits = log P / t + b."""

    def __init__(self, d):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1)); self.head = torch.nn.Linear(32, 11)
        torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)

    def forward(self, x, logp):
        o = self.head(self.body(x)); t = torch.nn.functional.softplus(o[:, 0] + 0.5413) + 1e-3
        return logp / t[:, None] + o[:, 1:]


def ce_plain(net, x, y, _):
    return torch.nn.functional.cross_entropy(net(x), y)


def ce_recal(net, x, y, logp):
    return torch.nn.functional.cross_entropy(net(x, logp), y)


def platt(z, w):
    """(a, b) of sigmoid(a z + b) fitted to the 0/1 labels w by Newton steps."""
    a, b = 1.0, 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a * z + b))); g = np.array([((p - w) * z).sum(), (p - w).sum()])
        H = np.array([[(p * (1 - p) * z * z).sum(), (p * (1 - p) * z).sum()], [(p * (1 - p) * z).sum(), (p * (1 - p)).sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g); a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return a, b


def logit_fit_auc(Ftr, wtr, Fte, wte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9; A = np.hstack([np.ones((len(Ftr), 1)), (Ftr - mu) / sd]); beta = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ beta)); H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-6 * np.eye(A.shape[1]); step = np.linalg.solve(H, A.T @ (p - wtr)); beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return 1 / (1 + np.exp(-np.hstack([np.ones((len(Fte), 1)), (Fte - mu) / sd]) @ beta))


def es_matrix(P):
    es = P.copy(); es[:, 1:] += E1 * P[:, :-1]; es[:, :-1] += E1 * P[:, 1:]; return es


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        PmC = [np.load(ES / f"val{v}/C-pc-f0-x40-oh_s{s}.npz")["proba"].astype(np.float64) for s in a.seeds]
        PmT = [np.load(ES / f"val{v}/T-pc-f0-x40-oh_s{s}.npz")["proba"].astype(np.float64) for s in a.seeds]
        P = np.mean(PmC + PmT, axis=0); d = decide(P)
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32); assert len(R) == len(y)
        X0 = features(P, n, R); Dis = disagreement(PmC, PmT, d)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "d": d, "X": np.hstack([X0, Dis]), "X0": X0, "dis": Dis, "logp": np.log(np.clip(P, 1e-9, 1)).astype(np.float32)}
    rep = {"seeds": a.seeds, "arms": {}, "diagnostic": {}}
    # diagnostic: does member disagreement predict errors beyond 1 - max p?
    s1, s5, ww = [], [], []
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        f1 = lambda u: (1 - D[u]["P"].max(1))[:, None]  # noqa: E731
        f5 = lambda u: np.hstack([f1(u), D[u]["dis"]])  # noqa: E731
        wtr = np.concatenate([(D[u]["d"] != D[u]["y"]).astype(float) for u in tr])
        s1.append(logit_fit_auc(np.concatenate([f1(u) for u in tr]), wtr, f1(v), None)); s5.append(logit_fit_auc(np.concatenate([f5(u) for u in tr]), wtr, f5(v), None))
        ww.append((D[v]["d"] != D[v]["y"]).astype(int))
    w = np.concatenate(ww); a1, a5 = auc(np.concatenate(s1), w), auc(np.concatenate(s5), w)
    rep["diagnostic"] = {"auc_1_minus_pmax": a1, "auc_plus_disagreement": a5, "delta": a5 - a1}
    print(f"seeds {a.seeds}: error AUC 1 - max p {a1:.4f}; + 4 disagreement columns {a5:.4f} (delta {a5 - a1:+.4f})", flush=True)
    S = {k: {} for k in ("main", "G1", "G2", "G2-nodis")}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        cat = lambda k: np.concatenate([D[u][k] for u in tr])  # noqa: E731
        X, X0, y, d, groups, logp = cat("X"), cat("X0"), cat("y"), cat("d"), cat("pid"), cat("logp")
        h = split90(groups, 77 + v); wrong = (d != y).astype(int)
        # G1: natural-frequency gate and corrector on 90 %, Platt + tau on 10 %
        gate = mlp(X.shape[1], 2); mg, sg = fit_net(gate, ce_plain, X[~h], wrong[~h], groups[~h], seed=v)
        corr = mlp(X.shape[1], 10); m_wr = (~h) & (wrong == 1); mc, sc = fit_net(corr, ce_plain, X[m_wr], y[m_wr], groups[m_wr], seed=10 + v)

        def g1_parts(Z, dz, ab=None):
            with torch.no_grad():
                lg = torch.log_softmax(gate(torch.as_tensor((Z - mg) / sg)), 1).numpy(); qq = torch.softmax(corr(torch.as_tensor((Z - mc) / sc)), 1).numpy()
            z = lg[:, 1] - lg[:, 0]
            g = 1 / (1 + np.exp(-(ab[0] * z + ab[1]))) if ab is not None else 1 / (1 + np.exp(-z))
            qq[np.arange(len(dz)), dz] = 0.0; qq /= qq.sum(1, keepdims=True)
            Pm = g[:, None] * qq; Pm[np.arange(len(dz)), dz] += 1 - g
            es = es_matrix(Pm); gain = es.max(1) - es[np.arange(len(dz)), dz]
            return z, es.argmax(1), gain
        zh, _, _ = g1_parts(X[h], d[h]); ab = platt(zh, wrong[h].astype(float))
        _, alt_h, gain_h = g1_parts(X[h], d[h], ab)
        best_tau = max(TAUS, key=lambda t: balanced_mean_over_n(qscore(np.where(gain_h > t, alt_h, d[h]), y[h]), y[h], cat("n")[h]))
        _, alt_v, gain_v = g1_parts(D[v]["X"], D[v]["d"], ab)
        dec_g1 = np.where(gain_v > best_tau, alt_v, D[v]["d"])
        # G2: feature-conditional recalibration, with and without the disagreement columns
        dec = {}
        for name, XX, XV in (("G2", X, D[v]["X"]), ("G2-nodis", X0, D[v]["X0"])):
            net = Recal(XX.shape[1]); mu, sd = fit_net(net, ce_recal, XX, y, groups, seed=20 + v, extra=logp)
            with torch.no_grad():
                Pv = torch.softmax(net(torch.as_tensor((XV - mu) / sd), torch.as_tensor(D[v]["logp"])), 1).numpy()
            dec[name] = decide(Pv)
        yv = D[v]["y"]
        S["main"][v] = qscore(D[v]["d"], yv); S["G1"][v] = qscore(dec_g1, yv); S["G2"][v] = qscore(dec["G2"], yv); S["G2-nodis"][v] = qscore(dec["G2-nodis"], yv)
        print(f"fold {v}: G1 Platt a {ab[0]:.2f} b {ab[1]:.2f}, tau {best_tau}, switches {(dec_g1 != D[v]['d']).mean():.3f}; G2 changes {(dec['G2'] != D[v]['d']).mean():.3f}; "
              f"G2-nodis changes {(dec['G2-nodis'] != D[v]['d']).mean():.3f}", flush=True)
    y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in ("y", "n", "pid"))
    for arm in ("G1", "G2", "G2-nodis"):
        s_new, s_old = np.concatenate([S[arm][v] for v in FOLDS]), np.concatenate([S["main"][v] for v in FOLDS])
        dd = paired_diff(s_new, s_old, y, pid, n, n=5000)
        pf = [balanced_mean_over_n(S[arm][v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(S["main"][v], D[v]["y"], D[v]["n"]) for v in FOLDS]
        ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
        rep["arms"][arm] = {**dd, "per_fold_diff": pf, "plain_diff": float((s_new - s_old).mean()), "pass_new_rule": bool(ok)}
        print(f"{arm:9s} vs ct9 decision: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + f"  plain {rep['arms'][arm]['plain_diff']:+.4f}" + ("  -> PASS (new rule)" if ok else "  -> no"), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
