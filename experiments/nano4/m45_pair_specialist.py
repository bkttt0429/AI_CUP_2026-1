"""Section 4M M45 (2026-11-07, diagnostic): can a two-class specialist separate an adjacent pair better than ct9 itself? For each boundary
(B|A, A|1D, 1D|2D, 2D|3D), the validation questions whose true level is one of the two; cross-fitted over the folds; specialists: an MLP h32
(natural frequencies) and a logistic regression on M43's 77 columns; baseline: ct9's log P(hi) - log P(lo). Pooled AUC, the specialist's
AUC minus ct9's with a player bootstrap (1,000). Continue: delta >= 0.01 with CI lower > 0.
  python -I logs/nano4/m45_pair_specialist.py --out results/trunk_seq/es/m45_pairs.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, features  # noqa: E402
from m44_calib import ce_plain, fit_net, logit_fit_auc, mlp  # noqa: E402

PAIRS = (("B", "A"), ("A", "1D"), ("1D", "2D"), ("2D", "3D"))


def auc(s, lab):
    r = pd.Series(s).rank().to_numpy(); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
        D[v] = {"y": y, "pid": pid, "P": P, "X": features(P, n, R)}
    rep = {}
    rng = np.random.RandomState(0)
    for lo_name, hi_name in PAIRS:
        lo, hi = LEVELS.index(lo_name), LEVELS.index(hi_name)
        S_base, S_mlp, S_lin, LAB, PID = [], [], [], [], []
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]
            sel = lambda u: (D[u]["y"] == lo) | (D[u]["y"] == hi)  # noqa: E731
            X = np.concatenate([D[u]["X"][sel(u)] for u in tr]); lab = np.concatenate([(D[u]["y"][sel(u)] == hi).astype(int) for u in tr])
            g = np.concatenate([D[u]["pid"][sel(u)] for u in tr])
            m = sel(v); Xv, labv, Pv = D[v]["X"][m], (D[v]["y"][m] == hi).astype(int), D[v]["P"][m]
            net = mlp(X.shape[1], 2); mu, sd = fit_net(net, ce_plain, X, lab, g, seed=v)
            with torch.no_grad():
                lg = torch.log_softmax(net(torch.as_tensor((Xv - mu) / sd)), 1).numpy()
            S_mlp.append(lg[:, 1] - lg[:, 0]); S_lin.append(logit_fit_auc(X, lab.astype(float), Xv, None))
            S_base.append(np.log(np.clip(Pv[:, hi], 1e-9, 1)) - np.log(np.clip(Pv[:, lo], 1e-9, 1)))
            LAB.append(labv); PID.append(D[v]["pid"][m])
        sb, sm, sl, lab, pid = (np.concatenate(x) for x in (S_base, S_mlp, S_lin, LAB, PID))
        players = np.unique(pid); idx = {p: np.flatnonzero(pid == p) for p in players}
        res = {"questions": int(len(lab)), "auc_ct9": auc(sb, lab), "acc_ct9": float(((sb > 0) == lab).mean()), "acc_mlp": float(((sm > 0) == lab).mean()),
               "acc_logistic": float(((sl > 0.5) == lab).mean()), "share_hi": float(lab.mean())}
        print(f"{lo_name:>3s}|{hi_name:<3s} pairwise accuracy (choose between the two): ct9 {res['acc_ct9']:.4f}  mlp {res['acc_mlp']:.4f}  logistic {res['acc_logistic']:.4f}  (share of {hi_name} {res['share_hi']:.3f})", flush=True)
        for name, s in (("mlp", sm), ("logistic", sl)):
            d0 = auc(s, lab) - res["auc_ct9"]; bs = []
            for _ in range(1000):
                ii = np.concatenate([idx[p] for p in rng.choice(players, len(players))]); bs.append(auc(s[ii], lab[ii]) - auc(sb[ii], lab[ii]))
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res[name] = {"auc": auc(s, lab), "delta": d0, "ci95": ci, "continue": bool(d0 >= 0.01 and ci[0] > 0)}
        rep[f"{lo_name}|{hi_name}"] = res
        print(f"{lo_name:>3s}|{hi_name:<3s} {res['questions']:5d} questions  ct9 AUC {res['auc_ct9']:.4f}  "
              + "  ".join(f"{k} {res[k]['auc']:.4f} (delta {res[k]['delta']:+.4f} [{res[k]['ci95'][0]:+.4f}, {res[k]['ci95'][1]:+.4f}])" + (" CONTINUE" if res[k]["continue"] else "") for k in ("mlp", "logistic")), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1)); print("written", a.out)


if __name__ == "__main__":
    main()
