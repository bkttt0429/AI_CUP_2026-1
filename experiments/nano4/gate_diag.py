"""Production-gate diagnostics on the saved per-model probabilities (prod_heldout.P.npz; 794 held-out inner players, 4000 questions):
H2 diversity   pairwise correlation of the members' residuals (E_m - y) and decision agreement, ensemble-gain curve (k = 1, 2, 4, 6, 12)
compositions   ct6 C-only / T-only, ct3 + ct6 (24), the two ensembles' probability average, size routing (ct6 for n <= 12, ct3 above)
H1 calibration bias-only multinomial recalibration (P' ~ P exp(b_c)) and temperature, fitted on one half of the players (NLL), applied to
               the other half, swapped and concatenated; for ct3 and ct6 alike
All differences against the ct3 ensemble with the paired player bootstrap; the gate's CI is about +-0.016, so this is direction only.
  python -I logs/nano4/gate_diag.py --inp results/r_variants/prod_heldout.P.npz --out results/r_variants/gate_diag.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402

K = len(LEVELS)


def scores_of(P, y):
    pred = np.array([expected_score_decision(p) for p in P])
    return np.array([score(int(a), int(b)) for a, b in zip(pred, y)]), pred


def recal(P_fit, y_fit, P_apply, mode):
    """Bias-only (b_c) or temperature (t) recalibration fitted by NLL on the fit half, applied to the other half."""
    lp = torch.log(torch.tensor(P_fit, dtype=torch.float64).clamp_min(1e-9)); yt = torch.tensor(y_fit)
    if mode == "bias":
        b = torch.zeros(K, dtype=torch.float64, requires_grad=True); params = [b]
        f = lambda L: L + b  # noqa: E731
    else:
        t = torch.ones(1, dtype=torch.float64, requires_grad=True); params = [t]
        f = lambda L: L / t  # noqa: E731
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(f(lp), yt); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        La = torch.log(torch.tensor(P_apply, dtype=torch.float64).clamp_min(1e-9))
        return torch.softmax(f(La), dim=1).numpy(), [float(x) for x in params[0].detach().numpy().ravel()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str)
    sets = {name: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(name + "__")} for name in ("ct3", "ct6")}
    rep = {"questions": int(len(y)), "players": int(len(set(pid)))}
    ens = {name: np.mean(list(M.values()), axis=0) for name, M in sets.items()}
    S = {name: scores_of(P, y)[0] for name, P in ens.items()}
    base = S["ct3"]
    def report(label, P):
        s, pred = scores_of(P, y); d = paired_diff(s, base, y, pid, n)
        rep[label] = {"score": balanced_mean_over_n(s, y, n), "vs_ct3": d["diff"], "ci95": d["ci95"], "by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in (5, 10, 20)},
                      "mean_E_minus_y": float((P @ np.arange(K) - y).mean())}
        print(f"{label:34s} {rep[label]['score']:.4f}  vs ct3 {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  by n 5/10/20 "
              + "/".join(f"{rep[label]['by_n'][m]:.3f}" for m in ('5', '10', '20')) + f"  E-y {rep[label]['mean_E_minus_y']:+.3f}", flush=True)
    for name in ("ct3", "ct6"):
        report(f"{name} ensemble (12)", ens[name])
    # H2 diversity
    for name, M in sets.items():
        names = sorted(M); E = {m: M[m] @ np.arange(K) for m in names}; dec = {m: scores_of(M[m], y)[1] for m in names}
        rc = [np.corrcoef(E[a_] - y, E[b_] - y)[0, 1] for a_, b_ in itertools.combinations(names, 2)]
        ag = [float((dec[a_] == dec[b_]).mean()) for a_, b_ in itertools.combinations(names, 2)]
        singles = {m: balanced_mean_over_n(scores_of(M[m], y)[0], y, n) for m in names}
        rng = np.random.default_rng(0); curve = {}
        for k in (1, 2, 4, 6, 12):
            vals = [balanced_mean_over_n(scores_of(np.mean([M[m] for m in rng.choice(names, k, replace=False)], axis=0), y)[0], y, n) for _ in range(30 if k < 12 else 1)]
            curve[str(k)] = float(np.mean(vals))
        rep[f"{name}_diversity"] = {"mean_resid_corr": float(np.mean(rc)), "min_resid_corr": float(np.min(rc)), "mean_decision_agreement": float(np.mean(ag)),
                                    "singles_mean": float(np.mean(list(singles.values()))), "singles": singles, "gain_curve": curve,
                                    "C_only": balanced_mean_over_n(scores_of(np.mean([M[m] for m in names if "_C-" in m], axis=0), y)[0], y, n),
                                    "T_only": balanced_mean_over_n(scores_of(np.mean([M[m] for m in names if "_T-" in m], axis=0), y)[0], y, n)}
        print(f"{name}: member residual corr mean {np.mean(rc):.3f} (min {np.min(rc):.3f}), decision agreement {np.mean(ag):.3f}, singles mean {np.mean(list(singles.values())):.4f}, "
              f"gain curve k=1/2/4/6/12 " + "/".join(f"{curve[k]:.4f}" for k in ('1', '2', '4', '6', '12')) + f", C-only {rep[name + '_diversity']['C_only']:.4f} T-only {rep[name + '_diversity']['T_only']:.4f}", flush=True)
    # compositions
    report("ct6 C-only (6)", np.mean([P for m, P in sets["ct6"].items() if "_C-" in m], axis=0))
    report("ct6 T-only (6)", np.mean([P for m, P in sets["ct6"].items() if "_T-" in m], axis=0))
    report("ct3 + ct6 (24)", np.mean(list(sets["ct3"].values()) + list(sets["ct6"].values()), axis=0))
    report("0.5 ct3 + 0.5 ct6 ensembles", 0.5 * ens["ct3"] + 0.5 * ens["ct6"])
    route = np.where((n <= 12)[:, None], ens["ct6"], ens["ct3"]); report("route: ct6 n<=12, ct3 n>12 (post hoc)", route)
    # H1 calibration, player 2-fold
    players = np.array(sorted(set(pid))); half = set(np.random.default_rng(1).permutation(players)[: len(players) // 2]); inA = np.array([p in half for p in pid])
    for name in ("ct3", "ct6"):
        for mode in ("bias", "temp"):
            Pc = np.zeros_like(ens[name]); params = []
            for fit, app in ((inA, ~inA), (~inA, inA)):
                Pc[app], prm = recal(ens[name][fit], y[fit], ens[name][app], mode); params.append(prm)
            report(f"{name} recalibrated ({mode}, 2-fold players)", Pc); rep[f"{name} recalibrated ({mode}, 2-fold players)"]["params"] = params
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
