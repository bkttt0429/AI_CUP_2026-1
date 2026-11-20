#!/usr/bin/env python
"""Section 4M M132 (2026-11-14, report only): splitting the ct104b rule by the KIND of change and by the SEGMENT of the question.

The ct104b rule = M129's bundle (m15 for the five recipes, look-alike weights 0.5 / 0.5 / 0.1 by question size) with two kinds of
changes not applied (an answer is never moved out of A; no 5D->6D). Here, in CV (folds 1-4, seeds 0-2 and 6-8):
 (1) by kind (old answer -> new answer): number and mean gain per change on each half of the folds and each seed set; a
     cross-fitted kind filter (kinds with a positive mean on one half are kept on the other half) against the unfiltered rule;
 (2) by segment (question size x old answer): the gain of every look-alike weight; cross-fitted weights per segment against the
     weights per size;
 (3) next to every kind the direction the Public probes recorded for that boundary (M63), and the Private changes of ct104b by kind.
  python -I logs/nano4/m132_split_table.py
"""
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
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0); W0 = {5: 0.5, 10: 0.5, 20: 0.1}
A_, D5, D6 = LEVELS.index("A"), LEVELS.index("5D"), LEVELS.index("6D")
# Public direction statistics recorded before today (design doc M63) and today's decode: the side of a boundary the test favours
FAVOUR = {("1D", "A"): "with (1D->A 14 of 16)", ("A", "1D"): "AGAINST (A side favoured)", ("4D", "5D"): "with (5D 13 / 4D 3)", ("5D", "4D"): "AGAINST (5D favoured)", ("5D", "6D"): "AGAINST (5D 6 / 6D 1)", ("6D", "5D"): "with",
          ("3D", "2D"): "AGAINST (3D 5 / 2D 0)", ("2D", "3D"): "with", ("B", "C"): "with, weak (C 6 / B 3)", ("C", "B"): "against, weak", ("A", "B"): "AGAINST (today: leaving A lost)"}


def guard(d_new, d_old):
    d = d_new.copy(); m = ((d_old == A_) & (d_new != d_old)) | ((d_old == D5) & (d_new == D6)); d[m] = d_old[m]; return d


def main():
    rep = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = rep["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~m15" for x in ids]]); rec = dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec[k] = (f"{idn}~m15", enc, cc, temp, pi0, bmin, scale[idn])
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); players = q["players"].astype(str); plab = q["plab"]
    pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {0: {}, 2: {}}
        for A, R in ((0, REC), (2, rec)):
            for name, (idn, enc, cc, temp, pi0, bmin, sc) in R.items():
                ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[A][name] = L * Tn if temp else L
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); Fd[v] = (y, n, pid, Ls, lr_ - lr_.mean(0, keepdims=True))
    half_of = {1: 0, 2: 0, 3: 1, 4: 1}; rows = []; D = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        for v in FOLDS:
            y, n, pid, Ls, RD = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); _, d_old = mf.rule75u(P, Ls[0])
            dw = {w: guard(mf.rule75u(P if w == 0 else la.pooled(P, RD, w), Ls[2])[1], d_old) for w in WS}; D[(seeds[0], v)] = (y, n, pid, d_old, dw)
            d_new = np.select([n == m for m in SIZES], [dw[W0[m]] for m in SIZES])
            for i in np.flatnonzero(d_new != d_old):
                rows.append((seeds[0], v, half_of[v], int(n[i]), int(d_old[i]), int(d_new[i]), float(SM[d_new[i], y[i]] - SM[d_old[i], y[i]])))
    df = pd.DataFrame(rows, columns=["seeds", "fold", "half", "n", "old", "new", "gain"]); df["kind"] = [f"{LEVELS[o]}->{LEVELS[w_]}" for o, w_ in zip(df.old, df.new)]; nq = sum(len(Fd[v][0]) for v in FOLDS)
    print(f"ct104b rule in CV: {len(df)} changed decisions over two seed sets of {nq} questions; mean gain per change {df.gain.mean():+.3f}")
    print("\n(1) by kind: changes (both seed sets) | mean gain: all, half {1,2}, half {3,4}, seeds 0-2, seeds 6-8 | Public direction on record")
    kinds = df.groupby("kind").gain.agg(["size", "mean", "sum"]).sort_values("size", ascending=False)
    for k, r in kinds.iterrows():
        g = df[df.kind == k]; o, w_ = k.split("->"); parts = [g.gain.mean(), g[g.half == 0].gain.mean(), g[g.half == 1].gain.mean(), g[g.seeds == 0].gain.mean(), g[g.seeds == 6].gain.mean()]
        if r["size"] >= 30:
            print(f"   {k:9s} {int(r['size']):5d} | " + "  ".join(f"{x:+.3f}" for x in parts) + f" | total {r['sum']:+6.1f} | {FAVOUR.get((o, w_), '')}")
    small = kinds[kinds["size"] < 30]; print(f"   other kinds (< 30 changes each): {int(small['size'].sum())} changes, total {small['sum'].sum():+.1f}")
    # cross-fitted kind filter: kinds with a positive mean on the other half (seeds 0-2) are kept
    out = {}
    for thr in (0.0, 0.03, 0.06):
        keep = {h: set(k for k, g in df[(df.half == 1 - h) & (df.seeds == 0)].groupby("kind") if g.gain.mean() > thr and len(g) >= 10) for h in (0, 1)}; res = []
        for seeds in (0, 6):
            new_s, old_s, ys, ns, ps = [], [], [], [], []
            for v in FOLDS:
                y, n, pid, d_old, dw = D[(seeds, v)]; d_new = np.select([n == m for m in SIZES], [dw[W0[m]] for m in SIZES]); kind = np.array([f"{LEVELS[o]}->{LEVELS[w_]}" for o, w_ in zip(d_old, d_new)]); ok = np.array([k in keep[half_of[v]] for k in kind]) | (d_new == d_old)
                d_f = np.where(ok, d_new, d_old); new_s.append(SM[d_f, y]); old_s.append(SM[d_new, y]); ys.append(y); ns.append(n); ps.append(pid); res.append((int((d_new != d_old).sum()), int((d_f != d_old).sum())))
            dd = paired_diff(np.concatenate(new_s), np.concatenate(old_s), np.concatenate(ys), np.concatenate(ps), np.concatenate(ns), sizes=SIZES, n=3000)
            pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new_s, old_s, ys, ns)]
            out[f"filter>{thr}|{seeds}"] = {**dd, "per_fold": pf}
            print(f"   kind filter (mean gain on the other half > {thr:g}), seeds {seeds}: against the unfiltered ct104b rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  changes kept {sum(r[1] for r in res[-4:])} of {sum(r[0] for r in res[-4:])}")
        print(f"      kinds kept from half {{1,2}}: {sorted(keep[1])}\n      kinds kept from half {{3,4}}: {sorted(keep[0])}")
    # (2) by segment: size x old answer
    print("\n(2) by segment (question size x old answer): gain of each look-alike weight against weight 0 (seeds 0-2 | seeds 6-8), per question of the segment; questions")
    seg = {}
    for (sd_, v), (y, n, pid, d_old, dw) in D.items():
        for m in SIZES:
            for lv in range(10):
                s = (n == m) & (d_old == lv)
                if s.any():
                    for w in WS:
                        a = seg.setdefault((m, lv, sd_, half_of[v]), {}).setdefault(w, [0.0, 0]); a[0] += float((SM[dw[w][s], y[s]] - SM[dw[0.0][s], y[s]]).sum()); a[1] += int(s.sum())
    for m in SIZES:
        for lv in range(10):
            line = f"   n={m:2d} {LEVELS[lv]:3s}"
            for sd_ in (0, 6):
                tot = {w: sum(seg[(m, lv, sd_, h)][w][0] for h in (0, 1) if (m, lv, sd_, h) in seg) for w in WS}; cnt = sum(seg[(m, lv, sd_, h)][0.0][1] for h in (0, 1) if (m, lv, sd_, h) in seg)
                line += "  | " + " ".join(f"{w:g}:{tot[w] / max(cnt, 1):+.4f}" for w in WS[1:]) + f" ({cnt})"
            print(line)
    for sd_sel in (0,):  # weights per segment chosen on the other half (seeds 0-2), scored on both seed sets
        pick = {h: {(m, lv): max(WS, key=lambda w: seg.get((m, lv, sd_sel, 1 - h), {}).get(w, [0.0, 0])[0]) for m in SIZES for lv in range(10)} for h in (0, 1)}
        for seeds in (0, 6):
            new_s, old_s, ys, ns, ps = [], [], [], [], []; nch = [0, 0]
            for v in FOLDS:
                y, n, pid, d_old, dw = D[(seeds, v)]; d_cur = np.select([n == m for m in SIZES], [dw[W0[m]] for m in SIZES]); d_seg = d_old.copy()
                for m in SIZES:
                    for lv in range(10):
                        s = (n == m) & (d_old == lv); d_seg[s] = dw[pick[half_of[v]][(m, lv)]][s]
                new_s.append(SM[d_seg, y]); old_s.append(SM[d_cur, y]); ys.append(y); ns.append(n); ps.append(pid); nch[0] += int((d_cur != d_old).sum()); nch[1] += int((d_seg != d_old).sum())
            dd = paired_diff(np.concatenate(new_s), np.concatenate(old_s), np.concatenate(ys), np.concatenate(ps), np.concatenate(ns), sizes=SIZES, n=3000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new_s, old_s, ys, ns)]
            out[f"segment_weights|{seeds}"] = {**dd, "per_fold": pf}
            print(f"   weights per segment (chosen on the other half), seeds {seeds}: against the ct104b rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  changes {nch[1]} against {nch[0]}")
        print("      weights chosen from half {1,2} (rows = size, columns = D..6D): " + " | ".join(f"n={m}: " + " ".join(f"{pick[1][(m, lv)]:g}" for lv in range(10)) for m in SIZES))
        print("      weights chosen from half {3,4}:                                " + " | ".join(f"n={m}: " + " ".join(f"{pick[0][(m, lv)]:g}" for lv in range(10)) for m in SIZES))
    # (3) the test file by kind
    rd = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank  # noqa: E731
    c75, c104b = rd("ct75u"), rd("ct104b"); ch = c104b != c75; pub = c75.index.str.startswith("pub_"); cvm = df.groupby("kind").gain.mean()
    print("\n(3) ct104b's changes on the test by kind: Public / Private | CV mean gain per change | Public direction on record")
    tk = pd.Series([f"{a}->{c}" for a, c in zip(c75[ch], c104b[ch])], index=c75.index[ch])
    for k, cnt in tk.value_counts().items():
        o, w_ = k.split("->"); pb = int((tk[tk.index.str.startswith("pub_")] == k).sum()); print(f"   {k:9s} {pb} / {cnt - pb} | {cvm.get(k, float('nan')):+.3f} | {FAVOUR.get((o, w_), '')}")
    (ROOT / "results/r_variants/m132_split_table.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
