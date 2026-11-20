#!/usr/bin/env python
"""M262 step 3 (2026-11-20), exploratory on the Public readings (stated as such in the design doc): which FORM of the outcome
input fits best on the fair-likelihood ruler? Copy of m262_like.py with other covariates: the registered primary (win share of the
even games - 0.5), the binomial z, wins minus losses, a shrunk share, versions by question size, and combinations with the
opponents-minus-own strength gap. The primary is kept unless another form is better by >= +2.

  python -I logs/nano4/m262_form.py --perms 300
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m229_like as ml  # noqa: E402  (pair_dist, LATER)

LEV = led.LEVELS; K = np.arange(10.0); SIZES = (5, 10, 15, 20)
MODEL = {"ct40", "ct42", "ct44", "ct64", "ct71", "ct72", "ct73", "ct75n", "ct75s", "ct75u", "ct75s_out", "ct75s_out_up", "ct104", "ct104b", "ct105", "ct106", "ct109", "ct112"}
BAND = {"probe_4D5D_locked", "probe_A2B", "probe_2D3D", "probe_3D4D", "probe_C2B", "probe_5D4D", "probe_D2C", "probe_5D6D", "probe_1D2A", "probe_B2", "probe_noLA9"}
NEW = [("upload_probe/probe_D2C.csv", 0.8379152), ("upload_probe/probe_5D6D.csv", 0.8331743), ("upload_probe/probe_1D2A.csv", 0.8410758), ("upload_probe/probe_B2.csv", 0.8284334), ("upload_probe/probe_noLA9.csv", 0.8523970)]
GAM = (-1.0, -0.7, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.7, 1.0); KINDS = ("shift",); MOVE = re.compile(r";([BW])\[([a-z]*)\]")
G = {}


def soft(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def tilt(LP, x, kind, g):
    return soft(LP + g * x[:, None] * K[None, :]) if kind == "shift" else soft(LP * np.clip(1 + g * x, 0.3, 3.0)[:, None])


def parts(Pp):
    """Log-likelihood of the Public readings under truths ~ Pp [400, 10]: model-driven steps, band probes, exact composition."""
    A, pairs, cnt = G["A"], G["pairs"], G["cnt"]; out = []
    for js in (G["jm"], G["jb"]):
        t = 0.0
        for j in js:
            k, p, idx, sols = pairs[j]; D, n = ml.pair_dist(Pp, A[p], A[k], idx); t += math.log(max(sum(D[n + x, n + y] for x, y in sols), 1e-300))
        out.append(t)
    mu = Pp.sum(0); var = (Pp * (1 - Pp)).sum(0); out.append(float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum()))
    return np.array(out)


def scan(X):
    """Gain against the current posterior [covariate, kind, gamma, part]."""
    LP, ref = G["LP"], G["ref"]; out = np.zeros((X.shape[1], len(KINDS), len(GAM), 3))
    for j in range(X.shape[1]):
        for a, kind in enumerate(KINDS):
            for c, g in enumerate(GAM):
                out[j, a, c] = parts(tilt(LP, X[:, j], kind, g)) - ref
    return out


def work(task):
    fam, r = task; X = G["X"][fam]; rng = np.random.default_rng(262000 + r); perm = np.arange(len(X))
    if fam == "A":
        for s in SIZES:
            m = np.flatnonzero(G["n"] == s); perm[m] = rng.permutation(m)
    else:
        perm = rng.permutation(len(X))
    return fam, r, scan(X[perm]).sum(-1)


def question_stats():
    """Per question from the released files: mean number of moves, share of games whose last move is the player's (games ending in a pass left out), share ending in a pass."""
    csv.field_size_limit(10 ** 9); out = {}
    for half in ("public", "private"):
        with open(ROOT / f"data/tests/rank_prediction_test_{half}.csv") as fh:
            for row in csv.DictReader(fh):
                n = int(row["num_games"]); ln, own, ps = [], [], []
                for i in range(1, n + 1):
                    mv = MOVE.findall(row[f"sgf_{i}"]); ln.append(len(mv)); last = mv[-1]; is_pass = last[1] in ("", "tt"); ps.append(float(is_pass))
                    if not is_pass:
                        own.append(float(last[0] == row[f"color_{i}"]))
                out[row["question_id"]] = (float(np.mean(ln)), float(np.mean(own)) if own else 0.5, float(np.mean(ps)), n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--perms", type=int, default=300); a_ = ap.parse_args(); t0 = time.time()
    z = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P0 = z["P"][names.index("current")].astype(np.float64); qid = z["qid"].astype(str); nq = z["nq"].astype(int)
    base = z["base"].astype(np.float64); R = z["R_cur"].astype(np.float64); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); pos = {q: i for i, q in enumerate(qid)}
    # ------------------------------------------------------------ covariates (all label-free)
    t82 = np.load(ROOT / "results/r_variants/m82_test.npz"); sq = np.load(ROOT / "results/r_variants/m84_SQ4~m15_test.npz"); assert np.array_equal(t82["qid"], sq["qid"])
    gq = np.array([pos[q] for q in t82["qid"].astype(str)]); hd = t82["hd"].astype(int); tb = t82["target_black"].astype(float); om = sq["f_opp_m"].astype(np.float64); ol = sq["f_opp_lab"].astype(np.float64)
    assert hd.max() <= 1 and np.array_equal(np.bincount(gq, minlength=800), nq), "games and questions are not aligned"
    per = lambda v: np.bincount(gq, weights=v, minlength=800)  # noqa: E731
    black = per(tb) / nq; assert set(np.unique(black)) <= {0.0, 1.0}, "a test question has both colours"
    hd1 = per((hd == 1).astype(float)) / nq; conf = om >= np.quantile(om, 0.7); an = per(conf.astype(float)); ce = conf & (hd == 0); ne = per(ce.astype(float)); s1 = per(ce * ol); s2_ = per(ce * ol * ol)
    eb = base @ K; el = P0 @ K; ela = soft(R) @ K; amean = np.where(ne > 0, s1 / np.maximum(ne, 1), np.nan); asd = np.where(ne >= 2, np.sqrt(np.maximum(s2_ / np.maximum(ne, 1) - (s1 / np.maximum(ne, 1)) ** 2, 0.0)), np.nan)
    ent = lambda P: -(P * np.log(np.maximum(P, 1e-300))).sum(1)  # noqa: E731
    qs = question_stats(); st = np.array([qs[q] for q in qid]); assert np.array_equal(st[:, 3].astype(int), nq)
    oc = np.load(ROOT / "results/r_variants/m262_outcomes.npz"); op = {str(x): i for i, x in enumerate(oc["qid"])}; oi = np.array([op[x] for x in qid]); ne = oc["n_even"][oi].astype(float); assert np.array_equal(oc["n"][oi].astype(int), nq)
    sh_even = np.where(ne > 0, oc["wins_even"][oi] / np.maximum(ne, 1), 0.5) - 0.5; sh_all = oc["share_all"][oi] - 0.5; sh_last_clean = oc["share_last"][oi] - 0.5
    print(f"clean win share: questions with an even game {int((ne > 0).sum())}; mean share even {float((sh_even[ne > 0] + 0.5).mean()):.3f}, all games {float((sh_all + 0.5).mean()):.3f}; correlation of the clean share (all games) with M246's 'moved last' share {float(np.corrcoef(sh_all, st[:, 1])[0, 1]):.3f}", flush=True)
    # the opponents' strength minus the player's own: posterior-mean level index of the ten L1b heads per game (the shard files of 11-04 hold no opponent Human SL sums; ct6_shift_diag.py used those)
    sys.path.insert(0, str(ROOT / "scripts")); import predict_submission as ps; from aicup.evidence import posterior_mean_index  # noqa: E401,E402
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS); tasks, owner, _, _ = ps.build_tasks(questions); own_s, opp_s = {}, {}
    for i in range(8):
        with np.load(ROOT / f"submission_rank_shard{i}of8.npz", allow_pickle=False) as zz:
            for k_, s_, o_ in zip(zz["task"].tolist(), zz["l1b_sums"], zz["l1b_opp_sums"]):
                own_s[k_], opp_s[k_] = s_, o_
    assert len(own_s) == len(tasks) == len(hd) and all(questions[owner[k_]]["question_id"] == str(t82["qid"][k_]) for k_ in range(0, len(tasks), 97)), "shard files and tasks do not match"
    pm = lambda S: np.asarray(posterior_mean_index(S.reshape(-1, S.shape[-1]))).reshape(len(tasks), -1).mean(1)  # noqa: E731  (mean over the head sets when the sums are [games, sets, 10])
    Sopp, Sown = np.stack([np.asarray(opp_s[k_], float) for k_ in range(len(tasks))]), np.stack([np.asarray(own_s[k_], float) for k_ in range(len(tasks))]); print(f"L1b sums per game: own {Sown.shape}, opponent {Sopp.shape}", flush=True)
    gapg = pm(Sopp) - pm(Sown); ev = (hd == 0).astype(float)
    gap_all = per(gapg) / nq; gap_even = np.where(per(ev) > 0, per(gapg * ev) / np.maximum(per(ev), 1), 0.0)
    print(f"opponents' pmi - own pmi per question (all games): mean {gap_all.mean():+.3f}, sd {gap_all.std():.3f}; by size " + " ".join(f"{s_}: {gap_all[nq == s_].mean():+.3f}" for s_ in SIZES) + f"; correlation with the clean win share (all games) {float(np.corrcoef(gap_all, sh_all)[0, 1]):+.3f}", flush=True)
    zs = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    wv = oc["wins_even"][oi].astype(float); bz = np.where(ne > 0, (wv - ne / 2) / np.sqrt(np.maximum(ne, 1) / 4), 0.0); diff = 2 * wv - ne; shr = (wv + 1) / (ne + 2) - 0.5
    cov = {"clean win share, even games (registered primary)": sh_even, "binomial z of the wins": bz, "wins minus losses": diff, "shrunk share (wins + 1) / (even games + 2)": shr,
           "win share x short questions (5, 10)": sh_even * (nq <= 10), "win share x long questions (15, 20)": sh_even * (nq >= 15), "binomial z x short": bz * (nq <= 10), "binomial z x long": bz * (nq >= 15),
           "combined: z(win share) - z(opponents' pmi - own pmi)": zs(sh_even) - zs(gap_even), "combined: z(binomial z) - z(gap)": zs(bz) - zs(gap_even)}
    cn = list(cov); XA = np.zeros((800, len(cn)))
    for j, k in enumerate(cn):
        v = cov[k]; ok = np.isfinite(v); XA[ok, j] = (v[ok] - v[ok].mean()) / max(v[ok].std(), 1e-9)  # questions without the quantity sit at the mean (0)
    bn = [f"question size {s}" for s in SIZES]; XB = np.stack([(nq == s).astype(float) for s in SIZES], 1); XB = (XB - XB.mean(0)) / XB.std(0)
    print(f"covariates: {len(cn)} + {len(bn)} size dummies; defined on (questions): " + ", ".join(f"{k.split(' ')[0]}.. {int(np.isfinite(cov[k]).sum())}" for k in cn))
    print("   Public mean / Private mean (raw units): " + " | ".join(f"{k}: {np.nanmean(cov[k][pub]):.3f} / {np.nanmean(cov[k][~pub]):.3f}" for k in cn), flush=True)
    # ------------------------------------------------------------ the Public readings
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *ml.LATER, *NEW]; ups = [(f, s) for f, s in ups if (ROOT / f).exists() or print(f"   (missing, left out: {f})")]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k])
        if len(idx) == 0:
            continue
        sols = led.decompose(400 * (S[k] - S[p]), len(idx))
        if sols:
            pairs.append((k, p, idx, sols))
        else:
            print(f"   (no decomposition, left out: {ups[k][0]} against {ups[p][0]}, {len(idx)} questions)")
    name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    jm = [j for j, x in enumerate(pairs) if name(x[0]) in MODEL]; jb = [j for j, x in enumerate(pairs) if name(x[0]) in BAND]
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[lv] for lv in LEV], float)
    print(f"{len(ups)} uploads, {len(pairs)} pairs: model-driven {len(jm)} ({sum(len(pairs[j][2]) for j in jm)} question changes), band probes {len(jb)} ({sum(len(pairs[j][2]) for j in jb)}), not used (chosen with Public information) {len(pairs) - len(jm) - len(jb)}")
    print("   band probes found: " + ", ".join(f"{name(pairs[j][0])} vs {name(pairs[j][1])} ({len(pairs[j][2])})" for j in jb), flush=True)
    LP = np.log(np.maximum(P0[pi], 1e-300)); G.update(A=A, pairs=pairs, jm=jm, jb=jb, cnt=cnt, LP=LP, n=nq[pi], X={"A": XA[pi], "B": XB[pi]}); G["ref"] = parts(P0[pi])
    print(f"current posterior: log-likelihood model-driven {G['ref'][0]:.2f}, band probes {G['ref'][1]:.2f}, composition {G['ref'][2]:.2f}; one evaluation takes {time.time() - t0:.1f} s incl. loading", flush=True)
    one = np.ones((len(pi), 1)); g1 = scan(one)[0]
    print("\n== reference (closed in M229 / M235): the same tilts for every question (x = 1); fair gain by gamma " + " ".join(f"{g:+.1f}" for g in GAM))
    for a, kind in enumerate(KINDS):
        print(f"   {kind}: " + " ".join(f"{v:+.2f}" for v in g1[a].sum(-1)))
    # ------------------------------------------------------------ observed scan and permutations
    t1 = time.time(); obs = {"A": scan(G["X"]["A"]), "B": scan(G["X"]["B"])}; print(f"\nobserved scan done in {time.time() - t1:.0f} s; {a_.perms} permutations on {len(os.sched_getaffinity(0))} processes", flush=True)
    tasks = [(fam, r) for r in range(1, a_.perms + 1) for fam in ("A", "B")]; null = {"A": [], "B": []}
    with Pool(len(os.sched_getaffinity(0))) as pool:
        for i, (fam, r, gain) in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            null[fam].append(gain)  # [covariate, kind, gamma]
            if (i + 1) % 80 == 0:
                print(f"   {i + 1} / {len(tasks)} permutation tasks, {time.time() - t1:.0f} s", flush=True)
    rep = {"gammas": GAM, "kinds": KINDS, "reference": g1.sum(-1).tolist(), "families": {}}
    lev = np.array(LEV); dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(P0); LPall = np.log(np.maximum(P0, 1e-300))
    for fam, nm, Xall in (("A", cn, XA), ("B", bn, XB)):
        o = obs[fam]; fair = o.sum(-1); best = fair.max(-1); NF = np.array(null[fam]); N = NF.max(-1); T = float(best.max()); Tn = N.reshape(len(N), -1).max(1); pg = (1 + int((Tn >= T).sum())) / (len(Tn) + 1)
        i4p, i4m = GAM.index(0.4), GAM.index(-0.4)
        if fam == "A":  # the registered primary reading: the clean win share, gamma < 0 only, one-sided permutation p
            neg = [c_ for c_, g_ in enumerate(GAM) if g_ < 0]; posi = [c_ for c_, g_ in enumerate(GAM) if g_ > 0]; rep["primary"] = {}
            for j_ in range(len(nm)):
                on, nn_ = float(fair[j_, 0, neg].max()), NF[:, j_, 0][:, neg].max(1); gb = GAM[neg[int(fair[j_, 0, neg].argmax())]]; pn = (1 + int((nn_ >= on).sum())) / (len(nn_) + 1); op_ = float(fair[j_, 0, posi].max())
                print(f"   PRIMARY-TYPE reading | {nm[j_]}: best gain with gamma < 0: {on:+.2f} at {gb:+.1f} (parts {o[j_, 0, GAM.index(gb), 0]:+.2f} {o[j_, 0, GAM.index(gb), 1]:+.2f} {o[j_, 0, GAM.index(gb), 2]:+.2f}), one-sided permutation p {pn:.4f}; best gain with gamma > 0: {op_:+.2f}; by gamma " + " ".join(f"{g_:+.1f}: {fair[j_, 0, c_]:+.2f}" for c_, g_ in enumerate(GAM)))
                rep["primary"][nm[j_]] = {"gain_neg": on, "gamma": gb, "p_neg": pn, "gain_pos": op_, "by_gamma": fair[j_, 0].tolist()}
        print(f"\n== family {fam} ({'stratified by question size' if fam == 'A' else 'unstratified'} permutations): largest fair gain {T:+.2f}; under permutation the largest gain has median {np.median(Tn):+.2f}, 90 % {np.quantile(Tn, 0.9):+.2f}, 95 % {np.quantile(Tn, 0.95):+.2f}, max {Tn.max():+.2f}  ->  FAMILY-WISE p = {pg:.3f}")
        print("   covariate | tilt: best gamma, fair gain (model-driven + band probes + composition), own permutation p (not corrected), mean loss of a +-0.4 tilt")
        rows = {}
        for j, k in enumerate(nm):
            for a, kind in enumerate(KINDS):
                c = int(fair[j, a].argmax()); pj = (1 + int((N[:, j, a] >= best[j, a]).sum())) / (len(N) + 1); loss = -(fair[j, a, i4p] + fair[j, a, i4m]) / 2
                print(f"   {k:48s} | {kind}: gamma {GAM[c]:+.1f}, {best[j, a]:+6.2f} ({o[j, a, c, 0]:+.2f} {o[j, a, c, 1]:+.2f} {o[j, a, c, 2]:+.2f}), p {pj:.3f}, loss at +-0.4 {loss:5.2f}")
                rows[f"{k} | {kind}"] = {"gamma": GAM[c], "gain": float(best[j, a]), "parts": o[j, a, c].tolist(), "p": pj, "loss_04": float(loss), "by_gamma": fair[j, a].tolist()}
        order = np.dstack(np.unravel_index(np.argsort(-best, axis=None), best.shape))[0][:3]; print("   the three largest settings and the answers they would change under the ct112 rule (Public / Private; kinds):")
        for j, a in order:
            c = int(fair[j, a].argmax()); dd = dec(tilt(LPall, Xall[:, j], KINDS[a], GAM[c])); ch = dd != d0; kinds = {}
            for u, w in zip(d0[ch], dd[ch]):
                kinds[f"{lev[u]}>{lev[w]}"] = kinds.get(f"{lev[u]}>{lev[w]}", 0) + 1
            print(f"      {nm[j]} | {KINDS[a]} {GAM[c]:+.1f}: {int((ch & pub).sum())} / {int((ch & ~pub).sum())}   " + ", ".join(f"{u} {w}" for u, w in sorted(kinds.items(), key=lambda kv: -kv[1])[:8]))
        rep["families"][fam] = {"largest_gain": T, "p_family": pg, "null_quantiles": {"median": float(np.median(Tn)), "q90": float(np.quantile(Tn, 0.9)), "q95": float(np.quantile(Tn, 0.95)), "max": float(Tn.max())}, "rows": rows}
        print(f"   REGISTERED: family-wise p {'<= 0.05 -> the largest setting is a candidate direction (next step registered separately)' if pg <= 0.05 else '> 0.05 -> no covariate direction is visible with the present readings'}", flush=True)
    (ROOT / "results/r_variants/m262_form.json").write_text(json.dumps(rep, indent=1)); print(f"\nwritten results/r_variants/m262_form.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
