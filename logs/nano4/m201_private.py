#!/usr/bin/env python
"""M201 (2026-11-16), REPORT ONLY, no labels: how the Private half of the test set compares with the Public half, and what the
final file (ct112) should score there by its own posterior.

(1) composition of the two halves: question sizes, games, handicap share, recognised opponents (statistic m15 of the
    opponent side above the target-side q90 / q98 of the whole test set), the levels answered and the posterior mass per level;
(2) self-expected score of a file = mean over questions of sum_c posterior[c] * score(answer, c), under the pooled posterior
    behind ct112 ("learned") and under the base posterior; per half, per size; its sampling sd from the same posterior;
    Public: against the readings actually obtained (ledger);
(3) what is Public-only: the 38 locked answers of ct28 (taken from Public probes) - where they sit, and how many differ
    from the answer the posterior alone would give; the Public readings of ct9 -> ct28;
(4) the chain of files: questions changed per half between consecutive finals and their self-expected gain per half.

  python -I logs/nano4/m201_private.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; RV = ROOT / "results/r_variants"
S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])  # S[answer, truth]


def main():
    z = np.load(RV / "m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = {"learned": z["learned"].astype(np.float64), "base": z["base"].astype(np.float64)}
    for k in P:
        P[k] = P[k] / P[k].sum(1, keepdims=True)
    pub = np.char.startswith(qid, "pub"); halves = (("Public", pub), ("Private", ~pub)); ledger = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}
    files = ["ct9", "ct28", "ct44", "ct75u", "ct104b", "ct106", "ct109", "ct112", "ct113"]; D = {}
    for f in files:
        p = ROOT / f"submission_rank_{f}.csv"
        if p.exists():
            D[f] = led.decisions(p).reindex(qid).to_numpy().astype(int)
    T = np.load(RV / "m84_SQ4~m15_test.npz", allow_pickle=True); tq = T["qid"].astype(str); pos = {q: i for i, q in enumerate(qid)}; gi = np.array([pos[q] for q in tq]); n = np.bincount(gi, minlength=len(qid))
    hd = T["hd"]; om, tm = T["f_opp_m"].astype(np.float64), T["f_tgt_m"].astype(np.float64); q90, q98 = np.quantile(tm, 0.90), np.quantile(tm, 0.98); a90 = np.bincount(gi, om > q90, len(qid)); a98 = np.bincount(gi, om > q98, len(qid)); t90 = np.bincount(gi, tm > q90, len(qid))
    hq = np.bincount(gi, hd != 0, len(qid)); rng = np.random.default_rng(201)

    def perm(x, m):  # two-sided permutation p of the difference of the Public and Private means of a per-question quantity (questions of the mask m)
        x = np.asarray(x, float)[m]; g = pub[m]; d0 = x[g].mean() - x[~g].mean(); k = int(g.sum()); d = np.array([(lambda o: x[o[:k]].mean() - x[o[k:]].mean())(rng.permutation(len(x))) for _ in range(4000)])
        return d0, float((np.abs(d) >= abs(d0) - 1e-15).mean())
    print("== (1) composition of the two halves (no labels)")
    for nm, m in halves:
        print(f"  {nm}: {int(m.sum())} questions, by size " + ", ".join(f"{s_}: {int((m & (n == s_)).sum())}" for s_ in (5, 10, 15, 20)) + f"; games {int(n[m].sum())}; handicap games {hq[m].sum() / n[m].sum():.3f}; opponent side above the target-side q90 {a90[m].sum() / n[m].sum():.4f} (target side {t90[m].sum() / n[m].sum():.4f}), above q98 {a98[m].sum() / n[m].sum():.4f}; "
              f"questions without a recognised opponent (q90) {float((a90[m] == 0).mean()):.3f}; recognised opponents per question {a90[m].mean():.2f}")
    allq = np.ones(len(qid), bool)
    for lab, x in (("handicap share of a question's games", hq / n), ("recognised opponents (q90) per game of a question", a90 / n), ("recognised (q98) per game", a98 / n), ("largest posterior probability (learned)", P["learned"].max(1)), ("largest posterior probability (base)", P["base"].max(1)),
                   ("posterior mean level (learned)", P["learned"] @ np.arange(10)), ("posterior mean level (base)", P["base"] @ np.arange(10))):
        d0, p_ = perm(x, allq); print(f"  Public minus Private, {lab}: {d0:+.4f} (permutation p {p_:.3f}); Public {np.asarray(x)[pub].mean():.4f}, Private {np.asarray(x)[~pub].mean():.4f}")
    print("  levels answered by ct112 (count)        " + " ".join(f"{l:>4}" for l in LEV))
    for nm, m in halves:
        print(f"    {nm:8s}                              " + " ".join(f"{int((D['ct112'][m] == c).sum()):4d}" for c in range(10)))
    for k in P:
        print(f"  posterior mass per level ({k}), sum over questions")
        for nm, m in halves:
            print(f"    {nm:8s}                              " + " ".join(f"{P[k][m][:, c].sum():4.0f}" for c in range(10)))
    cA, cB = np.stack([(D["ct112"][m][:, None] == np.arange(10)).sum(0) for _, m in halves]); e = (cA + cB) / 2; chi = float((((cA - e) ** 2 + (cB - e) ** 2) / np.maximum(e, 1e-9)).sum()); sim = []
    for _ in range(4000):
        o = rng.permutation(len(qid)); a_ = np.bincount(D["ct112"][o[:int(pub.sum())]], minlength=10); b_ = np.bincount(D["ct112"][o[int(pub.sum()):]], minlength=10); e_ = (a_ + b_) / 2; sim.append((((a_ - e_) ** 2 + (b_ - e_) ** 2) / np.maximum(e_, 1e-9)).sum())
    print(f"  answered levels, Public against Private: chi-square {chi:.1f}, permutation p {float((np.array(sim) >= chi).mean()):.3f}")

    def exp_score(f, k, m):
        e_ = (P[k][m] * S[D[f][m]]).sum(1); v = (P[k][m] * S[D[f][m]] ** 2).sum(1) - e_ ** 2
        return float(e_.mean()), float(np.sqrt(v.sum()) / m.sum())
    print("\n== (2) self-expected score (mean over questions; sd = sampling sd of the half's mean under the same posterior)")
    for f in ("ct9", "ct106", "ct109", "ct112"):
        for k in ("learned", "base"):
            row = []
            for nm, m in halves:
                mu, sd = exp_score(f, k, m); row.append(f"{nm} {mu:.4f} (sd {sd:.4f})" + (f", obtained {ledger.get(f'submission_rank_{f}.csv', float('nan')):.4f}" if nm == "Public" else ""))
            print(f"  {f:6s} under the {k:7s} posterior: " + " | ".join(row))
    print("  ct112 under the learned posterior, by size: " + " | ".join(f"{nm} " + " ".join(f"{s_}: {exp_score('ct112', 'learned', m & (n == s_))[0]:.3f}" for s_ in (5, 10, 15, 20)) for nm, m in halves))
    best = {k: (P[k] @ S.T).argmax(1) for k in P}  # the answer with the largest expected score under the posterior alone (no guards, no locks)
    locked = D["ct28"] != D["ct9"]; print(f"\n== (3) what is Public-only\n  locked answers (ct28 != ct9): {int(locked.sum())}; Public {int((locked & pub).sum())}, Private {int((locked & ~pub).sum())}; ct112 keeps ct28's answer on {int((D['ct112'][locked] == D['ct28'][locked]).sum())} of them")
    lk = locked & pub; diff = lk & (D["ct112"] != best["learned"]); e_lock = (P["learned"][lk] * S[D["ct112"][lk]]).sum(1); e_free = (P["learned"][lk] * S[best["learned"][lk]]).sum(1)
    print(f"  locked Public questions where the posterior alone would answer otherwise: {int(diff.sum())} of {int(lk.sum())}; by the posterior (which does not know the probes) the locked answers score {e_lock.mean():.3f} per question, its own answers {e_free.mean():.3f}")
    print(f"  Public readings: ct9 {ledger['submission_rank_ct9.csv']:.4f} -> ct28 {ledger['submission_rank_ct28.csv']:.4f} = {400 * (ledger['submission_rank_ct28.csv'] - ledger['submission_rank_ct9.csv']):+.2f} points from the {int(lk.sum())} locked answers (Public only; nothing of this exists on Private)")
    for nm, m in halves:  # the locked answers per kind of push, and how the posterior (which never saw a Public reading) looks at them in each half
        for a, b_ in sorted({(int(x), int(y_)) for x, y_ in zip(D["ct9"][locked], D["ct28"][locked])}):
            k = m & locked & (D["ct9"] == a) & (D["ct28"] == b_)
            if k.any():
                pl = P["learned"][k]; pb = P["base"][k]
                print(f"    {nm:8s} {LEV[a]}->{LEV[b_]}: {int(k.sum()):2d} questions; posterior of the pushed level against the original one, learned {pl[:, b_].mean():.3f} : {pl[:, a].mean():.3f}, base {pb[:, b_].mean():.3f} : {pb[:, a].mean():.3f}; the learned posterior alone would answer the pushed level on {int((best['learned'][k] == b_).sum())}, the original on {int((best['learned'][k] == a).sum())}, another on {int(((best['learned'][k] != a) & (best['learned'][k] != b_)).sum())}; sizes " + " ".join(f"{s_}:{int((n[k] == s_).sum())}" for s_ in (5, 10, 15, 20)))
    for nm, m in halves:  # answers that rest on a Public-derived direction: questions where ct112 differs from the posterior's own best answer although not locked (guards)
        g = m & ~locked & (D["ct112"] != best["learned"]); mv = sorted({(int(x), int(y_)) for x, y_ in zip(best["learned"][g], D["ct112"][g])}); e_g = float(((P["learned"][g] * S[D["ct112"][g]]).sum(1) - (P["learned"][g] * S[best["learned"][g]]).sum(1)).sum())
        print(f"    {nm:8s} not locked, ct112 differs from the posterior's own best answer on {int(g.sum())} questions (self-expected cost {e_g:+.2f} points): " + ", ".join(f"{LEV[a]}->{LEV[b_]} {int((g & (best['learned'] == a) & (D['ct112'] == b_)).sum())}" for a, b_ in mv))
    un = ~locked; agree =float((D["ct112"][un] == best["learned"][un]).mean()); print(f"  outside the locked questions ct112 equals the posterior's own best answer on {agree:.3f} of the questions (the rest: guards); Public {float((D['ct112'][un & pub] == best['learned'][un & pub]).mean()):.3f}, Private {float((D['ct112'][un & ~pub] == best['learned'][un & ~pub]).mean()):.3f}")
    mu_pub_un, _ = exp_score("ct112", "learned", pub & un); mu_pri, sd_pri = exp_score("ct112", "learned", ~pub); print(f"  self-expected score of ct112 on the Public questions that are NOT locked {mu_pub_un:.4f}; on Private {mu_pri:.4f} (sd {sd_pri:.4f})")
    print("\n== (4) the chain of finals: questions changed per half and their self-expected gain (points of 400; learned | base posterior)")
    chain = [f for f in ("ct9", "ct28", "ct44", "ct75u", "ct104b", "ct106", "ct109", "ct112") if f in D]
    for a, b_ in zip(chain[:-1], chain[1:]):
        row = []
        for nm, m in halves:
            ch = m & (D[a] != D[b_]); g = {k: float(((P[k][ch] * S[D[b_][ch]]).sum(1) - (P[k][ch] * S[D[a][ch]]).sum(1)).sum()) for k in P}
            obs = f", obtained {400 * (ledger[f'submission_rank_{b_}.csv'] - ledger[f'submission_rank_{a}.csv']):+.2f}" if nm == "Public" and all(f"submission_rank_{x}.csv" in ledger for x in (a, b_)) else ""
            row.append(f"{nm} {int(ch.sum()):3d} questions, expected {g['learned']:+.2f} | {g['base']:+.2f}{obs}")
        print(f"  {a:>6} -> {b_:6}: " + " ; ".join(row))
    for a in ("ct9", "ct44", "ct106"):
        ch = ~pub & (D[a] != D["ct112"]); mv = D["ct112"][ch] - D[a][ch]; print(f"  Private, ct112 against {a}: {int(ch.sum())} questions differ (one level {int((np.abs(mv) == 1).sum())}, more {int((np.abs(mv) > 1).sum())}; upward {int((mv > 0).sum())}, downward {int((mv < 0).sum())}); Public: {int((pub & (D[a] != D['ct112'])).sum())}")


if __name__ == "__main__":
    main()
