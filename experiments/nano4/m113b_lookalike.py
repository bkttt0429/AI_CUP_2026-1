"""Section 4M M113b (2026-11-14, report only, CPU): the target's look-alikes in the identity space, variants of the reading.
A question = the mean target-side identity vector of its games (three seeds of the all-games network; five members with S2y),
compared with the prototypes of the training players outside the validation fold. Reading: level distribution of the neighbours,
weights exp(beta x similarity); "balanced" divides each level's weight by its number of candidates (the candidates are 3.7 times
more often A than 1D). Per-game variant: every game's own reading, logs added with the question-size tempering.
Each reading pooled into the base (P x reading^w) and on top of the current evidence; w and beta listed, nothing selected here.
  python -I logs/nano4/m113b_lookalike.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m111_label_readings as m1  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM; CACHE = ROOT / "results/m84_cache"; BETA = (5.0, 10.0, 20.0, 30.0); WK = (0.2, 0.3, 0.5, 0.75, 1.0)


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool); edges, cal, om = m87.corrected(z, ones); _, _, band, g = s2.fit(z, ones)
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    VAR = {}
    for nm in ("S2e", "S2y"):
        v_ = np.load(CACHE / f"var_{nm}.npz"); VAR[nm] = (v_["qvec"], v_["protos"])
    F = {}
    for v in FOLDS:
        tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); y, n, pid, qd = m3.load_fold(v); y, n = np.asarray(y), np.asarray(n); Qn = len(y)
        qi_all = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(qd["game_keys"])]); gi_all = np.concatenate([[row[k] for k in g_ if k in row] for g_ in qd["game_keys"]]).astype(int); ok = g[gi_all] >= 0; qi, gi = qi_all[ok], gi_all[ok]; kk, bb, lab = g[gi], band[gi], z["opp_lab"][gi]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]
        L_std = np.zeros((Qn, 10)); np.add.at(L_std, qi, np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab])); L_std *= Tn
        cand = np.flatnonzero(pfold != v); oh = np.eye(10)[plab[cand]]; nc = oh.sum(0); RD = {}
        for nm, (QV, PR) in VAR.items():
            ns_ = QV.shape[0]; Sq = np.zeros((Qn, len(cand))); Sg = np.zeros((len(gi_all), len(cand)), np.float32) if nm == "S2e" else None
            for s_ in range(ns_):
                tv = QV[s_, Q + gi_all].astype(np.float64); pr = PR[s_, v - 1][cand].astype(np.float64); tq = np.zeros((Qn, 128)); np.add.at(tq, qi_all, tv); tq /= np.maximum(np.linalg.norm(tq, axis=1, keepdims=True), 1e-9); Sq += tq @ pr.T / ns_
                if Sg is not None:
                    Sg += (tv @ pr.T / ns_).astype(np.float32)
            for be in BETA:
                w_ = np.exp(be * (Sq - Sq.max(1, keepdims=True))) @ oh; RD[(nm, "question", "natural", be)] = np.log((w_ + 1e-6) / (w_ + 1e-6).sum(1, keepdims=True)); wb = w_ / nc; RD[(nm, "question", "balanced", be)] = np.log((wb + 1e-9) / (wb + 1e-9).sum(1, keepdims=True))
                if Sg is not None and be in (10.0, 20.0):
                    wg = np.exp(be * (Sg - Sg.max(1, keepdims=True))).astype(np.float64) @ oh / nc; lg = np.log((wg + 1e-9) / (wg + 1e-9).sum(1, keepdims=True)); Lg = np.zeros((Qn, 10)); np.add.at(Lg, qi_all, lg); RD[(nm, "per game", "balanced", be)] = Lg * Tn / np.maximum(np.bincount(qi_all, minlength=Qn), 1)[:, None] ** 0.5
        F[v] = dict(y=y, n=n, L_std=L_std, RD=RD)
    keys = list(F[FOLDS[0]]["RD"]); out = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {}
        for v in FOLDS:
            f = F[v]; y, n = f["y"], f["n"]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = lambda L_: float(balanced_mean_over_n(SM[mb.decisions(P, L_, 0.5, 3.0, 0.6)[1], y], y, n, sizes=SIZES))  # noqa: E731
            S.setdefault("base", []).append(dec(np.zeros_like(P))); S.setdefault("evidence", []).append(dec(f["L_std"]))
            for k_ in keys:
                for w in WK:
                    S.setdefault((k_, w, "base"), []).append(dec(w * f["RD"][k_])); S.setdefault((k_, w, "evidence"), []).append(dec(f["L_std"] + w * f["RD"][k_]))
        m = {k_: (float(np.mean(x)), x) for k_, x in S.items()}; base, evd = m["base"][0], m["evidence"][0]; print(f"\n=== seeds {seeds}: base {base:.4f}; current evidence {evd:.4f} ===\n   reading (members, unit, level weights, sharpness): gain over the base at weights {WK} | gain on top of the current evidence at the same weights (folds of the best)")
        for k_ in keys:
            gb = [m[(k_, w, "base")][0] - base for w in WK]; ge = [m[(k_, w, "evidence")][0] - evd for w in WK]; wbest = WK[int(np.argmax(ge))]; pf = [a_ - c_ for a_, c_ in zip(m[(k_, wbest, "evidence")][1], m["evidence"][1])]
            print(f"   {k_[0]:3s} {k_[1]:8s} {k_[2]:8s} {k_[3]:4.0f}: " + " ".join(f"{x:+.4f}" for x in gb) + " | " + " ".join(f"{x:+.4f}" for x in ge) + "  (" + " ".join(f"{e:+.4f}" for e in pf) + ")", flush=True)
        out[seeds[0]] = {str(k_): v_[0] for k_, v_ in m.items()}
    (ROOT / "results/r_variants/m113b_lookalike.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
