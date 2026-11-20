#!/usr/bin/env python
"""M247 (2026-11-19), report only: is the colour of a test question (every test question is all-black or all-white) a prior for the
level - because of the way the colour was chosen for each test player? (design doc M247, registered before any reading)

(A) training players (games without two or more handicap stones, players with >= 20 such games): theta_k = P(all-black question | level k)
    under three ways of choosing the colour, and the share of HD1 games inside the question each way predicts;
(B) test side without labels: by colour the counts, HD1 shares, expected levels; theta_k by EM on the pooled posterior;
(C) the fair composite likelihood of the Public readings (as in m246_scan.py) when the pooled posterior is multiplied by
    theta_k (black question) or 1 - theta_k (white question) to the power lambda; permutation p values (colour permuted among the Public
    questions of the same size, EM re-estimated when it used Public colours). PRIMARY: logistic theta estimated on the Private half only;
(D) answers that would change under the ct112 rule; the 38 locked shift questions by colour.

  python -I logs/nano4/m247_colour.py --perms 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m229_like as ml  # noqa: E402
import m246_scan as m6  # noqa: E402  (parts, G, soft, MODEL, BAND, NEW)

LEV = led.LEVELS; K = np.arange(10.0); SIZES = (5, 10, 15, 20)


def em_free(P, c, it=300):
    th = np.full(10, 0.5)
    for _ in range(it):
        W = P * np.where(c[:, None] > 0.5, th[None, :], 1 - th[None, :]); W /= W.sum(1, keepdims=True); th = ((W * c[:, None]).sum(0) + 0.5) / (W.sum(0) + 1.0)  # pseudo-count: one question, half black
    return np.clip(th, 0.02, 0.98)


def em_logit(P, c):
    def nll(ab):
        th = 1 / (1 + np.exp(-(ab[0] + ab[1] * (K - 4.5))))
        return -np.log(np.maximum((P * np.where(c[:, None] > 0.5, th[None, :], 1 - th[None, :])).sum(1), 1e-300)).sum()
    r = minimize(nll, [0.0, 0.0], method="BFGS"); return r.x


def logit_theta(ab):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * (K - 4.5))))


def tilt(P, th, c, lam):
    Q = P * np.where(c[:, None] > 0.5, th[None, :], 1 - th[None, :]) ** lam
    return Q / Q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--perms", type=int, default=200); a_ = ap.parse_args(); t0 = time.time(); rep = {}
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[lv] for lv in LEV], float); pi_pub = cnt / cnt.sum()
    # ------------------------------------------------------------ (A) training players
    ht = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str, "player_id": str}); ht = ht[ht.hd <= 1].copy(); ht["b"] = (ht.color == "B").astype(float); ht["h1"] = (ht.hd == 1).astype(float); ht["h1b"] = ht.b * ht.h1; ht["h1w"] = (1 - ht.b) * ht.h1
    g = ht.groupby("player_id"); T = pd.DataFrame({"lev": g["rank"].first().map(led.RANK), "n": g.size(), "nb": g.b.sum(), "h1b": g.h1b.sum(), "h1w": g.h1w.sum()}); T["nw"] = T.n - T.nb; n_all = len(T); T = T[T.n >= 20].copy()
    lev = T.lev.to_numpy().astype(int); nb, nw, n = T.nb.to_numpy(), T.nw.to_numpy(), T.n.to_numpy(); h1b, h1w = T.h1b.to_numpy(), T.h1w.to_numpy()
    rule = {"majority colour": np.where(nb > nw, 1.0, np.where(nb < nw, 0.0, 0.5)), "in proportion to the games": nb / n,
            "a colour with >= 20 games": np.where((nb >= 20) & (nw >= 20), 0.5, np.where(nb >= 20, 1.0, np.where(nw >= 20, 0.0, np.nan)))}
    print(f"(A) training players with >= 20 games without two or more handicap stones: {len(T)} of {n_all}")
    print("    by level: players | share of black games | share of HD1 games | black among the HD1 games | " + " | ".join(rule))
    TH = {}
    for k_, pb in rule.items():
        TH[k_] = np.array([np.nanmean(pb[lev == c]) for c in range(10)])
    for c in range(10):
        m = lev == c
        print(f"      {LEV[c]:3s} {int(m.sum()):5d} | {float((nb[m] / n[m]).mean()):.3f} | {float(((h1b[m] + h1w[m]) / n[m]).mean()):.3f} | {float(h1b[m].sum() / max((h1b[m] + h1w[m]).sum(), 1)):.3f} | " + " | ".join(f"{TH[k_][c]:.3f}" for k_ in rule))
    w = pi_pub[lev] / np.bincount(lev, minlength=10)[lev]  # players weighted to the Public level mix
    sb = np.where(nb > 0, h1b / np.maximum(nb, 1), 0.0); sw = np.where(nw > 0, h1w / np.maximum(nw, 1), 0.0)
    print("    predicted share of HD1 games inside a question (players weighted to the Public level mix): all | black questions | white questions | share of black questions")
    for k_, pb in rule.items():
        ok = np.isfinite(pb); ww, p_ = w[ok], pb[ok]; bq = float((ww * p_ * sb[ok]).sum() / (ww * p_).sum()); wq = float((ww * (1 - p_) * sw[ok]).sum() / (ww * (1 - p_)).sum()); fb = float((ww * p_).sum() / ww.sum())
        print(f"      {k_:28s}: {fb * bq + (1 - fb) * wq:.3f} | {bq:.3f} | {wq:.3f} | {fb:.3f}"); rep.setdefault("train_rules", {})[k_] = {"theta": TH[k_].tolist(), "hd1_all": fb * bq + (1 - fb) * wq, "hd1_black": bq, "hd1_white": wq, "share_black": fb}
    ok = np.ones(len(T), bool); print(f"      (random colour, for reference)  : {float((w * 0.5 * (sb + sw)).sum() / w.sum()):.3f} | {float((w * sb).sum() / w.sum()):.3f} | {float((w * sw).sum() / w.sum()):.3f} | 0.500", flush=True)
    # ------------------------------------------------------------ (B) test side
    z = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P0 = z["P"][names.index("current")].astype(np.float64); qid = z["qid"].astype(str); nq = z["nq"].astype(int); base = z["base"].astype(np.float64)
    pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); pos = {q: i for i, q in enumerate(qid)}; t82 = np.load(ROOT / "results/r_variants/m82_test.npz"); gq = np.array([pos[q] for q in t82["qid"].astype(str)])
    per = lambda v: np.bincount(gq, weights=v, minlength=800)  # noqa: E731
    c = per(t82["target_black"].astype(float)) / nq; assert set(np.unique(c)) <= {0.0, 1.0}; hd1 = per((t82["hd"] == 1).astype(float)) / nq
    dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(P0); print("\n(B) test side, no labels. group: questions | share of HD1 games | questions without an HD1 game | base expected level | pooled expected level")
    for nm, m in (("Public black", pub & (c == 1)), ("Public white", pub & (c == 0)), ("Private black", ~pub & (c == 1)), ("Private white", ~pub & (c == 0))):
        print(f"      {nm:14s}: {int(m.sum()):4d} | {float(hd1[m].mean()):.3f} | {float((hd1[m] == 0).mean()):.3f} | {float((base[m] @ K).mean()):.2f} | {float((P0[m] @ K).mean()):.2f}")
    print("    share of black questions by the level the ct112 rule answers (Public | Private | questions): " + "  ".join(f"{LEV[k]} {float(c[pub & (d0 == k)].mean()):.2f}|{float(c[~pub & (d0 == k)].mean()):.2f}|{int((d0 == k).sum())}" for k in range(10)))
    EST = {}
    for nm, m in (("all 800", np.ones(800, bool)), ("Public only", pub), ("Private only", ~pub)):
        th = em_free(P0[m], c[m]); ab = em_logit(P0[m], c[m]); rng = np.random.default_rng(247); idx = np.flatnonzero(m); bs = np.array([em_logit(P0[j], c[j])[1] for j in (rng.choice(idx, len(idx)) for _ in range(300))])
        EST[nm] = (th, ab); print(f"    EM theta_k = P(black | level), {nm:12s}: " + " ".join(f"{LEV[k]} {th[k]:.2f}" for k in range(10)) + f"   | logistic slope per level {ab[1]:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], theta D -> 6D {logit_theta(ab)[0]:.2f} -> {logit_theta(ab)[9]:.2f}")
        rep.setdefault("em", {})[nm] = {"theta_free": th.tolist(), "logit": ab.tolist(), "slope_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print("    for comparison, theta_k of the training rules: " + " | ".join(f"{k_}: " + " ".join(f"{x:.2f}" for x in TH[k_]) for k_ in rule), flush=True)
    # ------------------------------------------------------------ (C) the Public readings
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *ml.LATER, *m6.NEW]; ups = [(f, s) for f, s in ups if (ROOT / f).exists()]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k])
        if len(idx) == 0:
            continue
        sols = led.decompose(400 * (S[k] - S[p]), len(idx))
        if sols:
            pairs.append((k, p, idx, sols))
    name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    jm = [j for j, x in enumerate(pairs) if name(x[0]) in m6.MODEL]; jb = [j for j, x in enumerate(pairs) if name(x[0]) in m6.BAND]
    m6.G.update(A=A, pairs=pairs, jm=jm, jb=jb, cnt=cnt); ref = m6.parts(P0[pi]); print(f"\n(C) {len(ups)} uploads: model-driven pairs {len(jm)}, band probes {len(jb)}; current posterior {ref.sum():.2f}")
    npub = nq[pi]; priv = ~pub

    def thetas(cc):
        """theta of every setting for the colour vector cc (800); the settings that use Public colours are re-estimated."""
        return {**{f"training rule: {k_}": TH[k_] for k_ in rule}, "EM free, all 800": em_free(P0, cc), "EM logistic, all 800": logit_theta(em_logit(P0, cc)),
                "EM free, Private only": em_free(P0[priv], cc[priv]), "PRIMARY EM logistic, Private only": logit_theta(em_logit(P0[priv], cc[priv]))}
    LAM = (1.0, 0.5); th0 = thetas(c); sn = list(th0); obs = np.array([[m6.parts(tilt(P0[pi], th0[s], c[pi], lam)) - ref for lam in LAM] for s in sn]); rng = np.random.default_rng(2470); null = []
    for r in range(a_.perms):
        cc = c.copy(); perm = np.arange(len(pi))
        for s_ in SIZES:
            m = np.flatnonzero(npub == s_); perm[m] = rng.permutation(m)
        cc[pi] = c[pi][perm]; thp = thetas(cc); null.append([[float((m6.parts(tilt(P0[pi], thp[s], cc[pi], lam)) - ref).sum()) for lam in LAM] for s in sn])
    null = np.array(null); lev_ = np.array(LEV); print(f"    setting | lambda: fair gain (model-driven + band probes + composition), permutation p ({a_.perms} permutations of the colour among Public questions of the same size) | answers changed under the ct112 rule, Public / Private, kinds")
    for i, s in enumerate(sn):
        for j, lam in enumerate(LAM):
            gain = float(obs[i, j].sum()); pv = (1 + int((null[:, i, j] >= gain).sum())) / (len(null) + 1); dd = dec(tilt(P0, th0[s], c, lam)); ch = dd != d0; kinds = {}
            for u, w_ in zip(d0[ch], dd[ch]):
                kinds[f"{lev_[u]}>{lev_[w_]}"] = kinds.get(f"{lev_[u]}>{lev_[w_]}", 0) + 1
            print(f"      {s:36s} | {lam:.1f}: {gain:+6.2f} ({obs[i, j, 0]:+.2f} {obs[i, j, 1]:+.2f} {obs[i, j, 2]:+.2f}), p {pv:.3f}, null median {np.median(null[:, i, j]):+.2f} / 95 % {np.quantile(null[:, i, j], 0.95):+.2f} | {int((ch & pub).sum())} / {int((ch & ~pub).sum())}  " + ", ".join(f"{u} {v}" for u, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:7]))
            rep.setdefault("public", {})[f"{s} | {lam}"] = {"gain": gain, "parts": obs[i, j].tolist(), "p": pv, "changed_public": int((ch & pub).sum()), "changed_private": int((ch & ~pub).sum()), "kinds": kinds}
    pr = rep["public"]["PRIMARY EM logistic, Private only | 1.0"]; print(f"    REGISTERED (primary: logistic theta from the Private half, lambda 1, on Public): gain {pr['gain']:+.2f}, p {pr['p']:.3f} -> {'PASS (gain >= +3.0 and p <= 0.05): go to step 2' if pr['gain'] >= 3.0 and pr['p'] <= 0.05 else 'not passed'}")
    sec = [(s, rep["public"][f"training rule: {s} | 1.0"]) for s in rule]; print("    REGISTERED (secondary: a training rule with gain >= +5.0 and p <= 0.01): " + "; ".join(f"{s} {v['gain']:+.2f}, p {v['p']:.3f} -> {'PASS' if v['gain'] >= 5.0 and v['p'] <= 0.01 else 'no'}" for s, v in sec), flush=True)
    # ------------------------------------------------------------ (D) the locked shift questions by colour
    c9 = led.decisions("submission_rank_ct9.csv").reindex(qid).to_numpy().astype(int); c116 = led.decisions("submission_rank_ct116.csv").reindex(qid).to_numpy().astype(int); c118d = led.decisions("submission_rank_ct118d.csv").reindex(qid).to_numpy().astype(int); lk = np.asarray(z["locked"], bool); l28 = np.asarray(z["l28"], int)
    print("\n(D) the locked shift questions (ct28 against ct9) by colour: kind | Public black, white | Private black, white | Public questions whose best-known answer (ct118d) still is the shifted level: black, white")
    kinds = sorted({(int(a), int(b_)) for a, b_ in zip(c9[lk], l28[lk])})
    for a, b_ in kinds:
        m = lk & (c9 == a) & (l28 == b_); kept = m & pub & (c118d == b_)
        print(f"      {LEV[a]}>{LEV[b_]}: {int((m & pub & (c == 1)).sum())}, {int((m & pub & (c == 0)).sum())} | {int((m & ~pub & (c == 1)).sum())}, {int((m & ~pub & (c == 0)).sum())} | {int((kept & (c == 1)).sum())}, {int((kept & (c == 0)).sum())}")
    print("    final file ct116, share of black questions by answered level (Public | Private): " + "  ".join(f"{LEV[k]} {float(c[pub & (c116 == k)].mean()):.2f}|{float(c[~pub & (c116 == k)].mean()):.2f}" for k in range(10)))
    print("    best-known Public answers (ct118d), share of black questions by answered level: " + "  ".join(f"{LEV[k]} {float(c[pub & (c118d == k)].mean()):.2f} ({int((pub & (c118d == k)).sum())})" for k in range(10)))
    (ROOT / "results/r_variants/m247_colour.json").write_text(json.dumps(rep, indent=1, default=float)); print(f"\nwritten results/r_variants/m247_colour.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
