"""Section 4M M87 (2026-11-11): the mixture weight a_b = 1 - nominal_b / share_b treats every query as null when it computes the
excess; with a share pi of real matches the null mass in band b is (1 - pi) nominal_b. Correction: 1 - pi = min(1, share_0 / nominal_0)
(the lowest band taken as pure null), a_b = clip(1 - (1 - pi) nominal_b / share_b, 0, 0.98), a_top and K recomputed the same way.
Identifier 5e (m82_val.npz). Primary: corrected vs current evidence, M37, seeds 0-2 and 6-8.
  python -I scripts/m_series/m87_pi0.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)
NOM = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))


def corrected(z, tr):
    """s2.fit with the null mass scaled by (1 - pi) per group; returns edges, cal (a, ratio replaced), and the (1 - pi) used."""
    edges, cal, band, g = s2.fit(z, tr); d = z["opp_lab"] - z["y"]; om = {}
    for k in (0, 1, 2):
        c = cal[k]; share = c["share"]; q = float(min(1.0, share[0] / NOM[0])); om[k] = q; s = tr & (g == k)
        a = np.clip(1 - q * NOM / np.maximum(share, 1e-9), 0, 0.98); top = s & (band >= 5); a_top = float(np.clip(1 - q * NOM[5:].sum() / max(top.sum() / max(s.sum(), 1), 1e-9), 0, 0.98))
        hist = (np.array([(d[top] == x).sum() for x in s2.D]) + 1.0) / (top.sum() + len(s2.D)); K = np.clip(hist - (1 - a_top) * c["P0"], 0, None); K = K / K.sum()
        cal[k] = {**c, "a_old": c["a"], "a": a, "ratio": np.minimum(K / c["P0"], 30.0), "K": K, "a_top": a_top}
    return edges, cal, om


def evidence(z, v, fix):
    tr = z["fold"] != v; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    if fix:
        edges, cal, _ = corrected(z, tr)
    else:
        edges, cal, _, _ = s2.fit(z, tr)
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s = g == k; a = cal[k]["a"][band[s]][:, None]; out[s] = np.log((1 - a) + a * cal[k]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9])
    return dict(zip(z["key"][te], out))


def main():
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}
    _, cal, om = corrected(z, np.ones(len(z["y"]), bool))
    for k, nm in ((0, "HD0"), (1, "HD1 target Black"), (2, "HD1 target White")):
        print(f"{nm}: 1 - pi = {om[k]:.3f}; a before " + " ".join(f"{x:.2f}" for x in cal[k]["a_old"]) + " | after " + " ".join(f"{x:.2f}" for x in cal[k]["a"]) + f" | a_top {cal[k]['a_top']:.2f} | K(-2..2) " + " ".join(f"{cal[k]['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)))
    EV = {fix: {v: evidence(z, v, fix) for v in FOLDS} for fix in (False, True)}; rep = {}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {"base": [], False: [], True: []}; ys, ns, pids = [], [], []
        for v in FOLDS:
            y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); y = np.asarray(y)
            S["base"].append(SM[(P @ SM).argmax(1), y]); ys.append(y); ns.append(np.asarray(n)); pids.append(np.asarray(pid))
            for fix in (False, True):
                L = np.zeros_like(P)
                for i, gk in enumerate(q["game_keys"]):
                    for k in gk:
                        if k in EV[fix][v]:
                            L[i] += EV[fix][v][k]
                S[fix].append(SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), y])
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); res = {}
        for label, a, b in (("current vs base", False, "base"), ("corrected vs base", True, "base"), ("CORRECTED vs CURRENT (primary)", True, False)):
            dd = paired_diff(np.concatenate(S[a]), np.concatenate(S[b]), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w, yy, nn, sizes=SIZES) for x, w, yy, nn in zip(S[a], S[b], ys, ns)]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}
            print(f"  seeds {seeds} {label:32s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((np.concatenate(S[a]) != np.concatenate(S[b])).mean()):.3f} -> {'PASS' if good else 'no'}")
        gain = np.concatenate(S[True]) - np.concatenate(S[False]); print("  corrected - current by true class: " + " ".join(f"{LEVELS[c]} {np.mean([gain[(y == c) & (n == m)].mean() for m in SIZES]):+.4f}" for c in range(10)))
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["CORRECTED vs CURRENT (primary)"]["pass"])
    rep["adopt"] = bool(all(ok)); print(f"PRE-REGISTERED: corrected mixture weights replace the current ones: {'YES' if rep['adopt'] else 'NO'}")
    Path(ROOT / "results/r_variants/m87_pi0.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
