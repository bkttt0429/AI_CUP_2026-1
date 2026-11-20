"""Section 4M M91 (2026-11-13): the mirror channel with the M90 sequence identity networks. A question's player also appears as the
OPPONENT inside other players' training rows; the owner's level of such a row is known. Candidate = a question (prototype = sum of the
own-side vectors of its games), query = the opponent-side vector of a training row, score = cosine standardised per candidate over
all queries, averaged over the three seeds; top-1 and margin per query.
Validation (fold v, networks trained without fold v's players): every fold-v player gets ONE of his questions (5 / 10 / 20 games with
probability .25 / .25 / .5), two half-pools; D1 queries = rows whose owner is not in fold v; null = own-side rows of the other
half-pool's players; D2 (report) = opponent side of those same rows. Band edges per group from the null pooled over all folds and
repetitions. Matches de-duplicated per (candidate, owner). Calibration cross-fitted over folds: P0 from the lowest band, K from the
top bands, a_b by maximum likelihood on the label-difference histogram of each band; tempering by the number of matches (three
buckets, cross-fitted). Primary: base + bundle-v5 forward evidence + mirror vs base + bundle-v5 forward evidence, M37, seeds 0-2 and
6-8. Writes results/r_variants/m91_mirror.json and m91_cal.npz (edges, calibration, temperatures) for the test side.
  python -I logs/nano4/m91_mirror.py [--folds 1,2,3,4] [--reps 4] [--tag mirror]
"""
import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
import m86_bundle as b  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
RV, EMB = ROOT / "results/r_variants", ROOT / "results/m91"
QS = np.array([0.9, 0.99, 0.999, 0.9999, 0.99999]); NOM = np.diff(np.concatenate([[0.0], QS, [1.0]])); NB = len(QS) + 1
NSEED, TGRID, MIN_NULL = (0, 1, 2), (0.25, 0.5, 0.75, 1.0), 30
SM = b.SM


def bucket(m):
    return np.where(m <= 5, 0, np.where(m <= 20, 1, 2))


def load_meta(log):
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}
    meta = pd.DataFrame({"owner": rows.player_id.map(pidx).to_numpy(), "lab": rows["rank"].map(RANK).to_numpy(), "black": (rows.color == "B").to_numpy(),
                         "hd": rows.sgf_content.str.extract(r"HD\[(\d+)\]")[0].fillna("0").astype(int).to_numpy()}, index=rows.game_key.to_numpy()); del rows
    R = np.load(EMB / "rows_1.npz"); key = R["key"].astype(str)
    for v in (2, 3, 4):
        assert np.array_equal(np.load(EMB / f"rows_{v}.npz")["key"].astype(str), key), f"row order of net {v} differs"
    m = meta.loc[key]; fo = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; pfold = np.array([int(fo.get(p, -9)) for p in players])
    D = dict(key=key, owner=m.owner.to_numpy(), lab=m.lab.to_numpy(), black=m.black.to_numpy(), hd=m.hd.to_numpy(), players=players, pidx=pidx); D["fold"] = pfold[D["owner"]]
    D["g_opp"] = s2.group_of(D["hd"], ~D["black"]); D["g_own"] = s2.group_of(D["hd"], D["black"])  # the "target" of a query is the embedded side of the row
    agree = float(((R["subj"] == 0) == D["black"]).mean()); log(f"rows: {len(key)} games of {len(np.unique(D['owner']))} players; owner colour agrees with the stored subject colour in {agree:.4f}; HD0 {float((D['hd'] == 0).mean()):.3f}, HD1 {float((D['hd'] == 1).mean()):.3f}")
    assert agree > 0.99
    return D


def calibrate(counts, hist0, hist):
    """Per group: P0 (lowest band), K (top bands minus the null part), a_b by maximum likelihood on each band's label-difference
    histogram; counts [3, NB] of all queries, hist0 [3, 19] of band-0 queries, hist [3, NB, 19] of de-duplicated matches."""
    cal = {}; grid = np.linspace(0.0, 0.98, 99)
    for g in range(3):
        share = counts[g] / max(counts[g].sum(), 1); P0 = (hist0[g] + 1.0) / (hist0[g].sum() + 19); nz = [k for k in range(1, NB) if hist[g, k].sum() >= 50]; top = nz[-2:] if len(nz) >= 2 else nz
        a = np.zeros(NB); K = P0.copy(); a_top = 0.0
        if top:
            ht = hist[g, top].sum(0); a_top = float(np.clip(1 - NOM[top].sum() / max(share[top].sum(), 1e-12), 0, 0.98)); K = np.clip((ht + 0.5) / (ht.sum() + 9.5) - (1 - a_top) * P0, 1e-6, None); K = K / K.sum()
            if a_top >= 0.05:
                for k in range(1, NB):
                    if hist[g, k].sum() >= 20:
                        ll = [(hist[g, k] * np.log(x * K + (1 - x) * P0)).sum() for x in grid]; a[k] = grid[int(np.argmax(ll))]
            else:
                K = P0.copy()
        cal[g] = {"a": a, "ratio": np.minimum(K / P0, 30.0), "share": share, "a_top": a_top, "K": K, "P0": P0, "n": hist[g].sum(1)}
    return cal


def evidence(cal, lab, grp, band, scale=1.0):
    out = np.zeros((len(lab), 10))
    for g in range(3):
        s = grp == g
        if s.any():
            a = (scale * cal[g]["a"][band[s]])[:, None]; out[s] = np.log((1 - a) + a * cal[g]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9])
    return out


def distractors(Eown, Eopp, D, v, qi, dev, chunk=100_000):
    """The seen training players (owner fold != v, >= 8 rows) as competing candidates. A query whose best match is one of them is not
    a mirror match: the opponent of that row is most likely that training player (added after the fold-1 trial, see the design doc).
    Returns best(E_list, rows, mask_owner) -> the best standardised distractor score of every row (mean over seeds)."""
    import torch
    C = len(D["players"]); own_t = torch.as_tensor(D["owner"], device=dev); use = torch.as_tensor(np.flatnonzero(D["fold"] != v), device=dev); cnt = np.bincount(D["owner"][D["fold"] != v], minlength=C); seen = np.flatnonzero(cnt >= 8)
    col = np.full(C, -1, np.int64); col[seen] = np.arange(len(seen)); col_t = torch.as_tensor(col, device=dev); PS, MU, SD = [], [], []; qi_t = torch.as_tensor(qi, device=dev)
    for s in range(len(Eown)):
        P = torch.zeros((C, Eown[s].shape[1]), device=dev).index_add_(0, own_t[use], Eown[s][use]); P = torch.nn.functional.normalize(P[torch.as_tensor(seen, device=dev)], dim=1); s1 = torch.zeros(len(seen), device=dev, dtype=torch.float64); s2_ = torch.zeros_like(s1)
        for i in range(0, len(qi), chunk):
            S = (Eopp[s][qi_t[i:i + chunk]] @ P.T).double(); s1 += S.sum(0); s2_ += (S * S).sum(0)
        mu = s1 / len(qi); PS.append(P); MU.append(mu.float()); SD.append((s2_ / len(qi) - mu * mu).clamp(min=1e-12).sqrt().float() + 1e-6)

    def best(E_list, rows, mask_owner):
        rows_t = torch.as_tensor(rows, device=dev); out = torch.zeros(len(rows), device=dev)
        for i in range(0, len(rows), chunk):
            r = rows_t[i:i + chunk]; Z = 0.0
            for s in range(len(E_list)):
                Z = Z + (E_list[s][r] @ PS[s].T - MU[s]) / SD[s]
            Z = Z / len(E_list)
            if mask_owner:
                c = col_t[own_t[r]]; ok = c >= 0; Z[torch.arange(len(r), device=dev)[ok], c[ok]] = -float("inf")
            out[i:i + chunk] = Z.max(dim=1).values
        return out
    return best, len(seen)


def dedup(q, owner, band, extra):
    """Keep, for each (question slot, owner), the row with the highest band."""
    o = np.lexsort((-band, owner, q)); q, owner, band = q[o], owner[o], band[o]; first = np.ones(len(q), bool); first[1:] = (q[1:] != q[:-1]) | (owner[1:] != owner[:-1])
    return q[first], owner[first], band[first], [x[o][first] for x in extra]


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", default="1,2,3,4"); ap.add_argument("--reps", type=int, default=4); ap.add_argument("--tag", default="mirror"); ap.add_argument("--no-distractors", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; D = load_meta(log); kidx = {k: i for i, k in enumerate(D["key"])}; folds = [int(x) for x in a_.folds.split(",")]; REPS = a_.reps; full = folds == list(FOLDS) and REPS == 4
    REC = {}; NULL = defaultdict(list)
    for v in folds:
        E = [np.load(EMB / f"emb_{v}{f'_s{s}' if s else ''}.npz") for s in NSEED]; Eown = [torch.as_tensor(z["own"].astype(np.float32), device=dev) for z in E]; Eopp = [torch.as_tensor(z["opp"].astype(np.float32), device=dev) for z in E]; del E
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); qrows = [torch.as_tensor([kidx[k] for k in g_ if k in kidx], dtype=torch.long, device=dev) for g_ in gk]; by_p = defaultdict(lambda: defaultdict(list))
        for i, (p_, m_) in enumerate(zip(pid, n)):
            by_p[p_][int(m_)].append(i)
        plist = np.array(sorted(by_p)); pown = np.array([D["pidx"][p_] for p_ in plist]); qi = np.flatnonzero(D["fold"] != v); qi_t = torch.as_tensor(qi, device=dev); runs = []
        fv = np.flatnonzero(D["fold"] == v); fpos = np.full(len(D["key"]), -1, np.int64); fpos[fv] = np.arange(len(fv)); NEG = -float("inf")
        if a_.no_distractors:
            dz_q = torch.full((len(qi),), NEG, device=dev); dz_own = torch.full((len(fv),), NEG, device=dev); dz_opp = dz_own.clone(); n_dis = 0
        else:
            best, n_dis = distractors(Eown, Eopp, D, v, qi, dev); dz_q = best(Eopp, qi, True); dz_own = best(Eown, fv, False); dz_opp = best(Eopp, fv, False)
        for rep in range(REPS):
            rng = np.random.default_rng(100 * v + rep); perm = rng.permutation(len(plist)); halves = (perm[:len(perm) // 2], perm[len(perm) // 2:]); asg = np.zeros(len(plist), np.int64)
            for i_, p_ in enumerate(plist):
                size = int(rng.choice([5, 10, 20], p=[0.25, 0.25, 0.5])); size = size if size in by_p[p_] else sorted(by_p[p_])[-1]; asg[i_] = int(rng.choice(by_p[p_][size]))
            for h in (0, 1):
                cq = asg[halves[h]]; nrow = np.flatnonzero(np.isin(D["owner"], pown[halves[1 - h]])); nrow_t = torch.as_tensor(nrow, device=dev); Z = Zn = Z2 = 0.0; npos = torch.as_tensor(fpos[nrow], device=dev)
                for s in range(len(NSEED)):
                    P = torch.stack([torch.nn.functional.normalize(Eown[s][qrows[i]].sum(0), dim=0) for i in cq]); S = Eopp[s][qi_t] @ P.T; mu, sd = S.mean(0), S.std(0) + 1e-6
                    Z = Z + (S - mu) / sd; del S; Zn = Zn + (Eown[s][nrow_t] @ P.T - mu) / sd; Z2 = Z2 + (Eopp[s][nrow_t] @ P.T - mu) / sd

                def stat(Zs, dz):
                    """Top pool candidate and the margin over the runner-up among the pool and the distractors; -inf when a distractor wins."""
                    t = (Zs / len(NSEED)).topk(2, dim=1); p1, p2 = t.values[:, 0], t.values[:, 1]; m = torch.where(p1 > dz, p1 - torch.maximum(p2, dz), torch.full_like(p1, NEG))
                    return t.indices[:, 0].cpu().numpy().astype(np.int32), m.cpu().numpy()
                (j, mg), (_, mn), (j2, m2) = stat(Z, dz_q), stat(Zn, dz_own[npos]), stat(Z2, dz_opp[npos]); del Z, Zn, Z2
                for g in range(3):
                    NULL[g].append(mn[D["g_own"][nrow] == g])
                runs.append(dict(rep=rep, cq=cq, j=j, mg=mg, nrow=nrow, mn=mn, j2=j2, m2=m2))
        REC[v] = dict(y=y, n=n, pid=pid, gk=gk, qi=qi, runs=runs, n_cand=len(plist) // 2); log(f"fold {v}: {len(plist)} players in two half-pools, {len(qi)} D1 queries, {REPS} assignments, {n_dis} seen training players as competing candidates; a pool candidate wins {float(np.mean([np.isfinite(r_['mg']).mean() for r_ in runs])):.4f} of the queries and {float(np.mean([np.isfinite(r_['mn']).mean() for r_ in runs])):.4f} of the null rows"); del Eown, Eopp, dz_q, dz_own, dz_opp
    # band edges per group from the pooled null (no labels involved)
    edges = {}
    for g in range(3):
        x = np.concatenate(NULL[g]); e = np.quantile(x, QS, method="higher"); e[len(x) * (1 - QS) < MIN_NULL] = np.inf; edges[g] = e
        log(f"group {g}: pooled null {len(x)} rows, {float(np.isfinite(x).mean()):.4f} of them won by a pool candidate; edges " + " ".join(f"{x_:.3f}" for x_ in e))
    band_of = lambda m, g, ed: np.where(g == 0, np.searchsorted(ed[0], m), np.where(g == 1, np.searchsorted(ed[1], m), np.where(g == 2, np.searchsorted(ed[2], m), 0)))  # noqa: E731
    # per fold: counts, histograms, matches
    ST = {}
    for v in folds:
        r = REC[v]; y = r["y"]; qi = r["qi"]; gq, lq, oq = D["g_opp"][qi], D["lab"][qi], D["owner"][qi]; counts = np.zeros((3, NB)); counts2 = np.zeros((3, NB)); countsn = np.zeros((3, NB)); hist0 = np.zeros((3, 19)); hist = np.zeros((3, NB, 19)); hist2 = np.zeros((3, NB, 19)); M = defaultdict(list)
        for run in r["runs"]:
            band = band_of(run["mg"], gq, edges); qc = run["cq"][run["j"]]; d = np.clip(lq - y[qc] + 9, 0, 18); ok = gq >= 0
            np.add.at(counts, (gq[ok], band[ok]), 1); z0 = ok & (band == 0); np.add.at(hist0, (gq[z0], d[z0]), 1)
            k = ok & (band >= 1); qd, od, bd, (ld, gd) = dedup(qc[k], oq[k], band[k], [lq[k], gq[k]]); np.add.at(hist, (gd, bd, np.clip(ld - y[qd] + 9, 0, 18)), 1)
            for name, arr in (("rep", np.full(len(qd), run["rep"])), ("q", qd), ("lab", ld), ("grp", gd), ("band", bd)):
                M[name].append(arr)
            nr = run["nrow"]; g2 = D["g_opp"][nr]; b2 = band_of(run["m2"], g2, edges); ok2 = g2 >= 0; np.add.at(counts2, (g2[ok2], b2[ok2]), 1); q2 = run["cq"][run["j2"]]; k2 = ok2 & (b2 >= 1); np.add.at(hist2, (g2[k2], b2[k2], np.clip(D["lab"][nr][k2] - y[q2[k2]] + 9, 0, 18)), 1)
            gn = D["g_own"][nr]; bn = band_of(run["mn"], gn, edges); np.add.at(countsn, (gn[gn >= 0], bn[gn >= 0]), 1)
        ST[v] = dict(counts=counts, counts2=counts2, countsn=countsn, hist0=hist0, hist=hist, hist2=hist2, **{k_: np.concatenate(x) for k_, x in M.items()})
        st_ = ST[v]; cntq = np.zeros((REPS, len(y)), np.int64); k2_ = st_["band"] >= 2; np.add.at(cntq, (st_["rep"][k2_], st_["q"][k2_]), 1); mq = cntq[st_["rep"], st_["q"]]; dd_ = st_["lab"] - y[st_["q"]]; h_ = (st_["grp"] == 0) & k2_
        with np.errstate(all="ignore"):
            log(f"fold {v} HD0 matches at >= q99, by the candidate's number of such matches (owner at the candidate's level / within one / matches): " + " | ".join(f"{lo}-{hi if hi < 10 ** 9 else 'inf'}: {(dd_[h_ & (mq >= lo) & (mq <= hi)] == 0).mean():.2f} / {(np.abs(dd_[h_ & (mq >= lo) & (mq <= hi)]) <= 1).mean():.2f} / {int((h_ & (mq >= lo) & (mq <= hi)).sum())}" for lo, hi in ((1, 5), (6, 20), (21, 10 ** 9))))
        sh = lambda c: " ".join(f"{x:.5f}" for x in c[0] / max(c[0].sum(), 1))  # noqa: E731
        log(f"fold {v} HD0 share by band (nominal {' '.join(f'{x:.5f}' for x in NOM)}): D1 {sh(counts)} | D2 (unseen rows) {sh(counts2)} | null {sh(countsn)}; de-duplicated matches {len(ST[v]['q'])}")
    tot = lambda name, fs: sum(ST[u][name] for u in fs)  # noqa: E731
    cal_all = calibrate(tot("counts", folds), tot("hist0", folds), tot("hist", folds)); rep_out = {"folds": folds, "reps": REPS, "edges": {g: edges[g].tolist() for g in range(3)}, "calibration": {}}
    for g in range(3):
        c = cal_all[g]; c2 = tot("counts2", folds)[g]; h2 = tot("hist2", folds)[g]
        log(f"group {g} (all folds): share by band " + " ".join(f"{x:.5f}" for x in c["share"]) + " | a by band (label fit) " + " ".join(f"{x:.2f}" for x in c["a"]) + f" | a_top {c['a_top']:.2f} | matches by band " + " ".join(str(int(x)) for x in c["n"]) + " | K(-2..2) " + " ".join(f"{c['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)) + " | P0(-2..2) " + " ".join(f"{c['P0'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)))
        log(f"   D2 share by band " + " ".join(f"{x:.5f}" for x in c2 / max(c2.sum(), 1)) + " | D2 same-level rate by band " + " ".join(f"{(h2[k][9] / max(h2[k].sum(), 1)):.2f}" for k in range(NB)) + " | D1 same-level rate by band " + " ".join(f"{(tot('hist', folds)[g][k][9] / max(tot('hist', folds)[g][k].sum(), 1)):.2f}" for k in range(NB)))
        rep_out["calibration"][g] = {"share": c["share"].tolist(), "a": c["a"].tolist(), "a_top": c["a_top"], "matches": c["n"].tolist(), "share_d2": (c2 / max(c2.sum(), 1)).tolist(), "share_null": (tot("countsn", folds)[g] / max(tot("countsn", folds)[g].sum(), 1)).tolist()}
    # forward evidence of bundle v5 (ct75's recipe) per question
    V, _, _, used = b.sources(["5e", "SP2e"]); assert "SP2e" in used; LF = {}
    for v in folds:
        r = REC[v]; nq = len(r["y"]); Tn = np.array([b.TN[int(m)] for m in r["n"]])[:, None]; em, ee = b.evidence(V["SP2e"], v, 1, 1), b.evidence(V["enc"], v, 0, 1); L = np.zeros((nq, 10))
        for i, g_ in enumerate(r["gk"]):
            for k in g_:
                if k in em:
                    L[i] += em[k]
                if k in ee:
                    L[i] += b.U_ENC * ee[k]
        LF[v] = L * Tn

    def mirror(v, excl):
        """Summed mirror log-evidence [REPS, nq, 10] and match counts [REPS, nq] of fold v, calibrated without the folds in excl."""
        fs = [u for u in folds if u not in excl] or folds; cal = calibrate(tot("counts", fs), tot("hist0", fs), tot("hist", fs)); st = ST[v]; nq = len(REC[v]["y"]); L = np.zeros((REPS, nq, 10)); m = np.zeros((REPS, nq), np.int64)
        np.add.at(L, (st["rep"], st["q"]), evidence(cal, st["lab"], st["grp"], st["band"])); k = st["band"] >= 2; np.add.at(m, (st["rep"][k], st["q"][k]), 1)
        return L, m
    P012 = {v: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0) for v in folds}
    sc_of = lambda P, L, yv: SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), yv]  # noqa: E731
    LM, MC, TP = {}, {}, {}
    for v in folds:
        LM[v], MC[v] = mirror(v, {v}); oth = [u for u in folds if u != v]; best, best_s = (1.0, 1.0, 1.0), -1.0
        inner = {u: mirror(u, {u, v}) for u in oth}
        for T3 in itertools.product(TGRID, repeat=3):
            tot_s = []
            for u in oth:
                Lu, mu_ = inner[u]; r = REC[u]
                for rp in range(REPS):
                    sel = np.unique(np.concatenate([run["cq"] for run in r["runs"] if run["rep"] == rp])); T = np.array(T3)[bucket(mu_[rp][sel])][:, None]
                    tot_s.append(balanced_mean_over_n(sc_of(P012[u][sel], LF[u][sel] + T * Lu[rp][sel], r["y"][sel]), r["y"][sel], r["n"][sel], sizes=SIZES))
            if np.mean(tot_s) > best_s + 1e-9:
                best_s, best = float(np.mean(tot_s)), T3
        TP[v] = best; mm = np.concatenate([MC[v][rp][np.unique(np.concatenate([run["cq"] for run in REC[v]["runs"] if run["rep"] == rp]))] for rp in range(REPS)])
        log(f"fold {v}: temperatures by match count (<= 5, 6-20, > 20) picked on the other folds: {best}; matches at >= q99 per assigned question: mean {mm.mean():.1f}, median {np.median(mm):.0f}, share with none {float((mm == 0).mean()):.3f}, share with > 20 {float((mm > 20).mean()):.3f}")
    ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        cols = defaultdict(list)
        for v in folds:
            r = REC[v]; P = P012[v] if seeds == (0, 1, 2) else np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            for rp in range(REPS):
                sel = np.unique(np.concatenate([run["cq"] for run in r["runs"] if run["rep"] == rp])); T = np.array(TP[v])[bucket(MC[v][rp][sel])][:, None]; yv = r["y"][sel]; Lm = LM[v][rp][sel]
                for name, L in (("base", np.zeros((len(sel), 10))), ("fwd", LF[v][sel]), ("mir", T * Lm), ("both", LF[v][sel] + T * Lm), ("both_T1", LF[v][sel] + Lm)):
                    cols[name].append(sc_of(P[sel], L, yv))
                cols["y"].append(yv); cols["n"].append(r["n"][sel]); cols["pid"].append(r["pid"][sel]); cols["fold"].append(np.full(len(sel), v)); cols["m"].append(MC[v][rp][sel])
        A = {k: np.concatenate(x) for k, x in cols.items()}; res = {}
        for label, x, w in (("forward_vs_base", "fwd", "base"), ("mirror_vs_base", "mir", "base"), ("both_vs_forward", "both", "fwd"), ("both_untempered_vs_forward", "both_T1", "fwd"), ("both_vs_base", "both", "base")):
            dd = paired_diff(A[x], A[w], A["y"], A["pid"], A["n"], sizes=SIZES, n=5000); pf = [balanced_mean_over_n(A[x][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) - balanced_mean_over_n(A[w][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) for v in folds]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}
            log(f"  seeds {seeds} {label:28s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x_:+.4f}" for m_, x_ in dd["by_n"].items()) + f"  changed {float((A[x] != A[w]).mean()):.3f} -> {'PASS' if good else 'no'}")
        for lo, hi in ((0, 0), (1, 5), (6, 20), (21, 10 ** 9)):
            s = (A["m"] >= lo) & (A["m"] <= hi)
            if s.sum() > 50:
                log(f"     questions with {lo}-{hi if hi < 10 ** 9 else 'inf'} matches: share {float(s.mean()):.3f}; mean score base {A['base'][s].mean():.4f}, forward {A['fwd'][s].mean():.4f}, both {A['both'][s].mean():.4f}")
        rep_out[f"seeds_{seeds[0]}"] = res; ok.append(res["both_vs_forward"]["pass"])
    passed = bool(all(ok) and full); log(f"PRE-REGISTERED (four folds, four assignments): mirror evidence on top of the bundle-v5 forward evidence: {'PASS' if passed else ('NO' if full else 'not the registered run')}")
    rep_out.update(passed_cv=passed, temperatures={int(v): list(TP[v]) for v in folds}); (RV / f"m91_{a_.tag}.json").write_text(json.dumps(rep_out, indent=1, default=float))
    Tall = tuple(float(np.median([TP[v][k] for v in folds])) for k in range(3))
    np.savez(RV / f"m91_{a_.tag}_cal.npz", edges=np.stack([edges[g] for g in range(3)]), a=np.stack([cal_all[g]["a"] for g in range(3)]), ratio=np.stack([cal_all[g]["ratio"] for g in range(3)]), share=np.stack([cal_all[g]["share"] for g in range(3)]), a_top=np.array([cal_all[g]["a_top"] for g in range(3)]), T=np.array(Tall), n_cand=np.array([REC[v]["n_cand"] for v in folds]))
    log("written")


if __name__ == "__main__":
    main()
