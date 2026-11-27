#!/usr/bin/env python
"""Section 4M M245 (2026-11-19), report only, no test labels: is the level mix of the test questions the same at every question
size? On the test set the base model answers A far more often than 1D in 5-game questions and the reverse in 20-game questions.
Either the model has a size-dependent bias (then the validation confusion matrices of each size, applied to ONE level mix,
reproduce the test answer counts of every size), or the level mix itself changes with the size (question size tied to the kind
of player). Validation: players with 20 or more games (the test-like population; all players as a second reading), confusion
P(answer | true level) per size for the base decision and for the ct112 rule (15 games = mean of 10 and 20).
Test: answer counts per size of the base decision and of ct112, Public and Private.
(1) Expected counts under the exact Public mix at every size, Pearson chi-square with a parametric bootstrap p-value.
(2) The level mix of each size estimated from the answer counts alone (maximum likelihood through the confusion matrix),
    with a bootstrap interval; the A and 1D shares by size are the reading.

  python -I logs/nano4/m245_bysize.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m58_decode as led  # noqa: E402

m3, b, s2, ES, FOLDS = mb7.m3, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS
LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def conf(d, y, m):
    C = np.zeros((10, 10))
    for c in range(10):
        k = m & (y == c); C[c] = (np.bincount(d[k], minlength=10) + 0.5) / (k.sum() + 5.0)
    return C


def mle_mix(counts, C, iters=400):
    """EM for the level mix pi given answer counts and the confusion matrix C[true, answer]."""
    pi = np.full(10, 0.1)
    for _ in range(iters):
        post = pi[:, None] * C; post = post / post.sum(0, keepdims=True); pi = (post * counts[None, :]).sum(1); pi = pi / pi.sum()
    return pi


def main():
    rng = np.random.default_rng(245); z = np.load(ROOT / "results/r_variants/m244b_decisions.npz", allow_pickle=True); y, n, pid = z["y"], z["n"], z["pid"].astype(str); rule112 = mf7.guard2(z["linear_plain"], z["rule"])
    LF = {v: m3.load_fold(v) for v in FOLDS}; base = np.concatenate([np.asarray(b.decide(np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)), int) for v in FOLDS])
    assert len(base) == len(y) and (np.concatenate([np.asarray(LF[v][0]) for v in FOLDS]) == y).all()
    act = pd.concat([pd.read_csv(p, usecols=["player_id"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))]).player_id.value_counts(); ng = act.reindex(pid).to_numpy()
    g = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); qid = g["qid"].astype(str); nq = g["nq"].astype(int); pub = np.char.startswith(qid, "pub"); P9 = g["base"].astype(float); tb = np.asarray(b.decide(P9 / P9.sum(1, keepdims=True)), int)
    t112 = led.decisions(ROOT / "submission_rank_ct112.csv").reindex(qid).to_numpy().astype(int)
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; pi_pub = np.array([comp[l] for l in LEV], float); pi_pub /= pi_pub.sum()
    for pop_name, pm in (("validation players with 20 or more games", ng >= 20), ("all validation players", np.ones(len(y), bool))):
        print(f"\n######## confusion matrices from: {pop_name}")
        for dec_name, dv, dt in (("base decision", base, tb), ("ct112 rule", rule112, t112)):
            C = {s: conf(dv, y, pm & (n == s)) for s in (5, 10, 20)}; C[15] = (C[10] + C[20]) / 2
            print(f"\n== {dec_name}: share of the answers A / 1D by size under the Public mix (expected from validation) against the test set (Public | Private)")
            tot_chi = {"pub": 0.0, "priv": 0.0, "both": 0.0}
            for s in (5, 10, 15, 20):
                e = pi_pub @ C[s]; out = [f"n={s:2d}: expected A {e[3]:.3f} 1D {e[4]:.3f}"]
                for hn, hm in (("pub", pub), ("priv", ~pub), ("both", np.ones(800, bool))):
                    cnt = np.bincount(dt[hm & (nq == s)], minlength=10).astype(float); N = cnt.sum(); chi = float(((cnt - N * e) ** 2 / (N * e)).sum()); tot_chi[hn] += chi
                    sim = []
                    for _ in range(2000):
                        tr = rng.choice(10, int(N), p=pi_pub); a = np.array([rng.choice(10, p=C[s][c]) for c in tr]); c2 = np.bincount(a, minlength=10); sim.append(float(((c2 - N * e) ** 2 / (N * e)).sum()))
                    out.append(f"{hn}: A {cnt[3] / N:.3f} 1D {cnt[4] / N:.3f} (chi2 {chi:.1f}, p {float((np.array(sim) >= chi).mean()):.3f})")
                print("   " + " | ".join(out), flush=True)
            print(f"   all answers of the size, counts on the test set (both halves) minus expected: " + " ; ".join(f"n={s}: " + " ".join(f"{LEV[a]} {np.bincount(dt[nq == s], minlength=10)[a] - 200 * (pi_pub @ C[s])[a]:+.1f}" for a in range(10)) for s in (5, 10, 15, 20)))
            print("   level mix estimated from the answer counts of each size (both halves, 200 questions; share with a 90 % bootstrap interval):")
            for s in (5, 10, 15, 20):
                cnt = np.bincount(dt[nq == s], minlength=10).astype(float); pi = mle_mix(cnt, C[s]); bs = np.array([mle_mix(np.bincount(rng.choice(10, 200, p=cnt / cnt.sum()), minlength=10).astype(float), C[s], 150) for _ in range(300)])
                print(f"     n={s:2d}: " + "  ".join(f"{LEV[c]} {pi[c]:.3f} [{np.percentile(bs[:, c], 5):.2f}, {np.percentile(bs[:, c], 95):.2f}]" for c in range(10)) + f"   | kyu share {pi[:4].sum():.3f} [{np.percentile(bs[:, :4].sum(1), 5):.2f}, {np.percentile(bs[:, :4].sum(1), 95):.2f}], A minus 1D {pi[3] - pi[4]:+.3f} [{np.percentile(bs[:, 3] - bs[:, 4], 5):+.2f}, {np.percentile(bs[:, 3] - bs[:, 4], 95):+.2f}]", flush=True)
    print("\nvalidation, players with 20 or more games: true share A / 1D by size (the same players at every size): " + ", ".join(f"n={s}: {float((y[(ng >= 20) & (n == s)] == 3).mean()):.3f} / {float((y[(ng >= 20) & (n == s)] == 4).mean()):.3f}; base answers {float((base[(ng >= 20) & (n == s)] == 3).mean()):.3f} / {float((base[(ng >= 20) & (n == s)] == 4).mean()):.3f}" for s in (5, 10, 20)))
    print("REPORT ONLY")


if __name__ == "__main__":
    sys.exit(main())
