#!/usr/bin/env python
"""Section 4M M136 (2026-11-14, report only): the look-alike reading with identity vectors of different networks.

The reading of M123 (balanced level weights, sharpness 20, centred per level) with the query / prototype vectors of several
results/m84_cache/var_<name>.npz files, pooled into the current base probabilities with fixed weights; the class-balanced score
of the base decision and of the ct75u rule against no pooling (seeds 0-2 members, four folds, paired by question).

  python -I logs/nano4/m136_lookalike_sources.py --sources S2 S3c S3 S4 S4e S2y S4ft S4e+S3 S4e+S2y
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; WS = (0.2, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="+", default=["S2", "S3c", "S3", "S4", "S4e", "S2y", "S4ft", "S4e+S3", "S4e+S2y"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a_ = ap.parse_args()
    V, T, key_pid, used = b.sources(["5e", *[REC[k][0] for k in REC]])
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    names = sorted({x for s in a_.sources for x in s.split("+")}); VARS = {}
    for nm in names:
        v_ = np.load(CACHE / f"var_{nm}.npz"); VARS[nm] = (v_["qvec"], v_["protos"]); print(f"var_{nm}: {VARS[nm][0].shape[0]} member(s)", flush=True)
    base, rule, RD, meta = {}, {}, {}, {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, spec in REC.items():
            idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)
        TQ = {}
        for nm in names:
            QV, PR = VARS[nm]; tq = np.zeros((QV.shape[0], len(y), 128))
            for m in range(QV.shape[0]):
                np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
            TQ[nm] = (tq / np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9), [PR[m, v - 1][cand].astype(np.float64) for m in range(QV.shape[0])])
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in a_.seeds], axis=0)
        meta[v] = (y, n, pid); base[(v, None)] = SM[np.asarray(b.decide(P), int), y]; rule[(v, None)] = SM[mf.rule75u(P, Ls)[1], y]
        for src in a_.sources:
            tq = np.concatenate([TQ[x][0] for x in src.split("+")]); pr = [p_ for x in src.split("+") for p_ in TQ[x][1]]
            lr_, top, _ = la.reading(tq, pr, plab[cand], la.BE); lr_ = lr_ - lr_.mean(0, keepdims=True); RD[(v, src)] = float(np.mean(top))
            for w in WS:
                Pw = la.pooled(P, lr_, w); base[(v, src, w)] = SM[np.asarray(b.decide(Pw), int), y]; rule[(v, src, w)] = SM[mf.rule75u(Pw, Ls)[1], y]
        print(f"fold {v} done", flush=True)
    y, n, pid = (np.concatenate([meta[v][k] for v in FOLDS]) for k in (0, 1, 2)); out = {}
    lvl = lambda tab, key: np.concatenate([tab[(v, *key)] if key else tab[(v, None)] for v in FOLDS])  # noqa: E731
    print(f"no pooling: base decision {balanced_mean_over_n(lvl(base, ()), y, n, sizes=SIZES):.4f}, ct75u rule {balanced_mean_over_n(lvl(rule, ()), y, n, sizes=SIZES):.4f}")
    print("gain of pooling the reading (class-balanced, four folds pooled; [95% CI by player bootstrap] at w = 0.5 on the ct75u rule):")
    for src in a_.sources:
        r = {}
        for w in WS:
            r[f"base_w{w}"] = float(balanced_mean_over_n(lvl(base, (src, w)), y, n, sizes=SIZES) - balanced_mean_over_n(lvl(base, ()), y, n, sizes=SIZES))
            r[f"rule_w{w}"] = float(balanced_mean_over_n(lvl(rule, (src, w)), y, n, sizes=SIZES) - balanced_mean_over_n(lvl(rule, ()), y, n, sizes=SIZES))
        dd = paired_diff(lvl(rule, (src, 0.5)), lvl(rule, ()), y, pid, n, sizes=SIZES, n=3000); r["rule_w0.5_ci"] = dd["ci95"]; r["rule_w0.5_by_n"] = dd["by_n"]
        pf = [float(balanced_mean_over_n(rule[(v, src, 0.5)], meta[v][0], meta[v][1], sizes=SIZES) - balanced_mean_over_n(rule[(v, None)], meta[v][0], meta[v][1], sizes=SIZES)) for v in FOLDS]
        r["mean_top_similarity"] = float(np.mean([RD[(v, src)] for v in FOLDS])); out[src] = {**r, "rule_w0.5_per_fold": pf}
        print(f"  {src:9s} base: " + " ".join(f"w{w}: {r[f'base_w{w}']:+.4f}" for w in WS) + " | ct75u rule: " + " ".join(f"w{w}: {r[f'rule_w{w}']:+.4f}" for w in WS)
              + f"  [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()), flush=True)
    for x, c in (("S3", "S3c"), ("S4", "S2"), ("S4e+S3", "S4e"), ("S4e+S2y", "S4e"), ("S4ft", "S4e")):
        if x in a_.sources and c in a_.sources:
            dd = paired_diff(lvl(rule, (x, 0.5)), lvl(rule, (c, 0.5)), y, pid, n, sizes=SIZES, n=3000); out[f"{x} - {c}"] = dd
            print(f"  ct75u rule at w = 0.5: {x} minus {c}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]")
    (ROOT / "results/r_variants/m136_lookalike_sources.json").write_text(json.dumps(out, indent=1, default=float)); print("written results/r_variants/m136_lookalike_sources.json")


if __name__ == "__main__":
    main()
