#!/usr/bin/env python
"""Section 4M M120 (2026-11-14): the formal test of the symmetry-augmented identity networks in the anchor pipeline, and the candidate.

v7 = bundle v5's recipe (encoder evidence + class-conditional tables + tempering + corrected mixture weights, every band, test-side
rules) with identifier --id in place of SP2e (default SP2y4: the phase matcher, S2y's five sequence members and the three
symmetry-augmented seeds, each source with its own fusion weight; m84_variant.py --stage eval --tag SP2y4 writes its files).
Registered (design doc M120 and its amendment of 07:06), both needed for a file:
  (1) recipe level   v7's region decisions against v5's: CI lower bound > 0, >= 3 of 4 folds positive, difference >= 0.0025,
                     on seeds 0-2 and 6-8;
  (2) file level     the ct75u composition with v7 in place of v5 (new changes need v3 = v7; reversals of ct44's changes and
                     changes outside the boundary region need all of v3, v7, s2, s3c, s3) against the ct75u composition:
                     difference >= 0 on both seed sets.
Report only: by question size, the players with <= 40 games, the identifiers of --also as v7.
--build (only if both hold): ct100 (ct76-ct99 are taken) by m104_build.py's code path with v7 in place of v5; the same path with v5 must reproduce
ct75u exactly; the feasible Public readings relative to ct75u.

  python -I logs/nano4/m121_sym_anchor_test.py --id SP2y4 --also SP4e SP2e4 [--build]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m105_voters as m5  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
FIVE = ("v3", "v5", "s2", "s3c", "s3")
RV = ROOT / "results/r_variants"
E1 = math.exp(-1.0)


def compose(d0, d44, dR, dA, region, five, locked=None):
    """The ct75u composition for the voters `five` (the first two are the pair whose shared new changes are taken)."""
    keep = d44 != d0
    free = ~region & ~keep if locked is None else ~region & ~keep & ~locked
    r = np.where(keep, d44, np.where(dR[five[0]] == dR[five[1]], dR[five[1]], d44))
    a5, c5 = m5.modal(np.stack([dR[k] for k in five]), d44)
    o5, k5 = m5.modal(np.stack([dA[k] for k in five]), d0)
    s = keep & (c5 >= len(five)); r[s] = a5[s]
    s = free & (k5 >= len(five)); r[s] = o5[s]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", default="SP2y4")
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--name", default="ct100")
    a_ = ap.parse_args()
    new_ids = [a_.id, *a_.also]
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3", *new_ids])
    missing = [x for x in new_ids if x not in used]
    if a_.id in missing:
        sys.exit(f"identifier {a_.id}: evaluation files missing; run m84_variant.py --stage eval --tag {a_.id} first")
    new_ids = [x for x in new_ids if x in used]
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)
        dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    rec = dict(REC); rep = {"id": a_.id, "density_ratio": {}, "heldout": {}}
    for x in new_ids:
        r = m5.density_ratio(V[x], T[x], key_pid, dec, ct9s, nominal); rec[f"v7:{x}"] = (x, 1, 1, 1, 1, 1, round(r, 3)); rep["density_ratio"][x] = r
        ev = RV / f"m84_{x}_eval.json"
        if ev.exists():
            rep["heldout"][x] = json.loads(ev.read_text()).get("heldout", {}).get("fused")
        print(f"identifier {x}: test-side density ratio {r:.3f}; held-out fused top-1 / top-5 / top-20 {rep['heldout'].get(x)}", flush=True)
    for x in ("SP2e",):
        print(f"identifier {x} (v5): density ratio recorded {REC['v5'][6]}, computed {m5.density_ratio(V[x], T[x], key_pid, dec, ct9s, nominal):.3f}")
    ngames = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id").n_games
    Fd = {}
    for v in FOLDS:
        y, n, pid, q = LF[v]; gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in rec.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)
    yy, nn, pp = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); low = ngames.reindex(pp).to_numpy() <= 40
    verdict = {}
    for x in new_ids:
        v7 = f"v7:{x}"; five7 = ("v3", v7, "s2", "s3c", "s3"); ok1, ok2 = [], []
        for seeds in ((0, 1, 2), (6, 7, 8)):
            S = {k: [] for k in ("v5", "v7", "u5", "u7", "base", "v44")}; changed = 0
            for v in FOLDS:
                y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR, dA = {}, {}
                d0, _, d44, _, region = mb.decisions(P, Ls["v44"], 0.5, 3.0, 0.6)
                for name in (*FIVE, v7):
                    _, dA[name], dR[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
                u5, u7 = compose(d0, d44, dR, dA, region, FIVE), compose(d0, d44, dR, dA, region, five7); changed += int((u5 != u7).sum())
                for k, d in (("v5", dR["v5"]), ("v7", dR[v7]), ("u5", u5), ("u7", u7), ("base", d0), ("v44", d44)):
                    S[k].append(SM[d, y])
            fs = lambda k: np.array([balanced_mean_over_n(s_, Fd[v][0], Fd[v][1], sizes=SIZES) for s_, v in zip(S[k], FOLDS)])  # noqa: E731
            print(f"\n=== {x} as v7, seeds {seeds}: base {fs('base').mean():.4f}, ct44 rule {fs('v44').mean():.4f}, v5 recipe {fs('v5').mean():.4f}, v7 recipe {fs('v7').mean():.4f}, ct75u rule {fs('u5').mean():.4f}, with v7 {fs('u7').mean():.4f} ===")
            out = {}
            for tag, new, old in (("recipe", "v7", "v5"), ("file", "u7", "u5")):
                A_, B_ = np.concatenate(S[new]), np.concatenate(S[old]); d = paired_diff(A_, B_, yy, pp, nn, sizes=SIZES, n=5000); pf = (fs(new) - fs(old)).tolist()
                dl = paired_diff(A_[low], B_[low], yy[low], pp[low], nn[low], sizes=SIZES, n=3000)
                print(f"  [{tag}] {new} vs {old}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {v_:+.4f}" for m_, v_ in d["by_n"].items())
                      + f"  changed {float((A_ != B_).mean()):.4f}  | players with <= 40 games {dl['diff']:+.4f} [{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]", flush=True)
                out[tag] = {**d, "per_fold": pf, "low_activity": dl, "share_changed": float((A_ != B_).mean())}
            ok1.append((out["recipe"]["ci95"][0] > 0) and (out["recipe"]["diff"] >= 0.0025) and (sum(e > 0 for e in out["recipe"]["per_fold"]) >= 3)); ok2.append(out["file"]["diff"] >= 0)
            rep[f"{x}|seeds_{seeds[0]}"] = {**out, "decisions_changed_by_the_composition": changed}
        verdict[x] = {"recipe_M37": bool(all(ok1)), "file_not_worse": bool(all(ok2))}
        print(f"REGISTERED ({'primary' if x == a_.id else 'report only'}) {x}: (1) recipe level {'PASS' if all(ok1) else 'NO'}; (2) file level {'PASS' if all(ok2) else 'NO'}")
    rep["verdict"] = verdict; rep["adopt"] = bool(verdict[a_.id]["recipe_M37"] and verdict[a_.id]["file_not_worse"])
    (RV / "m121_sym_anchor_test.json").write_text(json.dumps(rep, indent=1, default=float))
    if not a_.build:
        return
    if not rep["adopt"]:
        print("no file: the registered conditions are not both met"); return
    # ---- test side: m104_build.py's path, v7 in place of v5
    v7 = f"v7:{a_.id}"; five7 = ("v3", v7, "s2", "s3c", "s3")
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c75s, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75s", "ct75u")); locked = c28 != c9; keep = c44 != c28
    nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); dR, dA = {}, {}; region = None
    for name in (*FIVE, v7):
        idn, enc, cc, temp, pi0, bmin, scale = rec[name]; Lq = (pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, scale, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq
        d0, da, dr, _, region = mb.decisions(P9, Lq, 0.5, 3.0, 0.6); dR[name] = np.where(locked, c28, lev[dr]); dA[name] = np.where(locked, c28, lev[da])

    def build(five):
        r = np.where(keep, c44, np.where(dR[five[0]] == dR[five[1]], dR[five[1]], c44)); vR = np.stack([dR[k] for k in five]); vA = np.stack([dA[k] for k in five])
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != c44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != c9); fin = r.copy(); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]
        return fin, r, rev, out
    chk, chk_s, _, _ = build(FIVE); assert (chk_s == c75s).all() and (chk == c75u).all(), (int((chk_s != c75s).sum()), int((chk != c75u).sum())); print("\nchecks: the same path with v5 reproduces ct75s and ct75u exactly")
    fin, r7, rev, out = build(five7); assert (fin[locked] == c28[locked]).all()
    for nm, o in (("ct75u", c75u), ("ct75s", c75s), ("ct44", c44), ("ct28", c28)):
        ch = fin != o; j = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((j == 1).sum())}, two or more {int((j >= 2).sum())}")
    for i in np.flatnonzero(fin != c75u):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct44 {c44[i]}, ct75u {c75u[i]} -> {fin[i]}  (v3 {dR['v3'][i]}, v5 {dR['v5'][i]}, v7 {dR[v7][i]})")
    f = ROOT / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten (choose another --name)")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["question_id", "pred_rank"]); w.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw)
    print(f"{f.name} written, md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct75u": qids[fin != c75u].tolist(), "public": int(((fin != c75u) & pub).sum()), "private": int(((fin != c75u) & ~pub).sum())}
    (RV / "m121_sym_anchor_test.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
