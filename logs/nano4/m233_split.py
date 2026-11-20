#!/usr/bin/env python
"""Section 4M M233 (2026-11-18), report only, no training: does pooling the games of a question by mean / std lose
information that reading the games in separate pieces and ADDING the evidence would keep?
Stored validation posteriors of the base ensemble (3 C + 3 T members). For every validation player whose 20-game question,
two 10-game questions and four 5-game questions are nested (checked on the game keys), the same 20 games are read as
  (a) one question of 20 games (the production reading: mean / std over the games),
  (b) two halves: T * (log P_half1 + log P_half2), (c) four quarters: T * sum log P_quarter   (T cross-fitted over folds),
  (d) arithmetic means of the halves' / quarters' posteriors, (e) (a) + lambda * (b or c)  (lambda cross-fitted, 0 allowed).
The same on the 10-game questions (one question of 10 games against its two 5-game quarters).

  python -I logs/nano4/m233_split.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
from aicup.qeval import balanced, paired_diff  # noqa: E402

m3, b, s2, ES, FOLDS, SM = mb7.m3, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.SM
TG = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2); LG = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def soft(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    LF = {v: m3.load_fold(v) for v in FOLDS}; rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        D = {20: [], 10: []}
        for v in FOLDS:
            y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = [frozenset(str(k) for k in g) for g in qd["game_keys"]]
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); lp = np.log(np.maximum(P, 1e-12)); by = {}
            for i, p_ in enumerate(pid):
                by.setdefault(p_, []).append(i)
            for p_, ii in by.items():
                for big, small in ((20, 10), (10, 5)):
                    for f in (i for i in ii if n[i] == big):
                        parts = [i for i in ii if n[i] == small and gk[i] <= gk[f]]
                        if len(parts) != 2 or (gk[parts[0]] & gk[parts[1]]):
                            continue
                        rec = dict(v=v, pid=p_, y=int(y[f]), full=lp[f], two=lp[parts].sum(0), two_mean=np.log(np.maximum(P[parts].mean(0), 1e-12)))
                        if big == 20:
                            qq = [i for i in ii if n[i] == 5 and gk[i] <= gk[f]]
                            if len(qq) != 4 or len(frozenset().union(*[gk[i] for i in qq])) != 20:
                                continue
                            rec.update(four=lp[qq].sum(0), four_mean=np.log(np.maximum(P[qq].mean(0), 1e-12)))
                        D[big].append(rec)
            if seeds == (0, 1, 2):
                print(f"fold {v}: questions {len(y)} (20 / 10 / 5 games: {int((n == 20).sum())} / {int((n == 10).sum())} / {int((n == 5).sum())}); nested so far: 20-game {len(D[20])}, 10-game {len(D[10])}", flush=True)
        out = {}
        for big in (20, 10):
            R = D[big]
            if len(R) < 500:
                print(f"\nseeds {seeds}, {big}-game questions: only {len(R)} have nested pieces - the questions of a player are not nested; no comparison"); continue
            fold = np.array([r["v"] for r in R]); y = np.array([r["y"] for r in R]); pid = np.array([r["pid"] for r in R]); nn = np.full(len(R), big); A = {k: np.stack([r[k] for r in R]) for k in R[0] if k not in ("v", "pid", "y")}
            sc = lambda Z: SM[np.asarray(b.decide(soft(Z)), int), y]  # noqa: E731  per-question official score of the decision
            bal = lambda s_, m=None: float(balanced(s_, y, m))  # noqa: E731

            def crossfit(make, grid):
                """Per-question scores with the grid value chosen on the other folds; also the values chosen."""
                S = {g: sc(make(g)) for g in grid}; res = np.zeros(len(y)); ch = []
                for v in FOLDS:
                    te = fold == v; best = max(grid, key=lambda g: (bal(S[g], ~te), -abs(g - (1.0 if grid is TG else 0.0)))); res[te] = S[best][te]; ch.append(best)
                return res, ch, {g: bal(S[g]) for g in grid}
            base = sc(A["full"]); res = {"n_questions": int(len(y)), "players": int(len(set(pid.tolist()))), "one question": bal(base)}
            print(f"\nseeds {seeds}, {big}-game questions with nested pieces: {len(y)} ({res['players']} players); one question of {big} games (mean / std pooling): {bal(base):.4f}")
            arms = [("two pieces, log evidence added (T cross-fitted)", lambda T: T * A["two"], TG), ("two pieces, posteriors averaged", lambda _: A["two_mean"], (1.0,)),
                    ("one question + lambda x two pieces", lambda g: A["full"] + g * A["two"], LG)]
            if big == 20:
                arms += [("four pieces, log evidence added (T cross-fitted)", lambda T: T * A["four"], TG), ("four pieces, posteriors averaged", lambda _: A["four_mean"], (1.0,)),
                         ("one question + lambda x four pieces", lambda g: A["full"] + g * A["four"], LG), ("one question + lambda x (two + four pieces)", lambda g: A["full"] + g * (A["two"] + A["four"]), LG)]
            for name, make, grid in arms:
                s_, ch, tab = crossfit(make, grid); d_ = paired_diff(s_, base, y, pid, nn, sizes=(big,), n=5000); pf = [bal(s_, fold == v) - bal(base, fold == v) for v in FOLDS]
                print(f"  {name:48s}: {bal(s_):.4f}  minus one question {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + (f"  chosen {ch}; whole grid " + " ".join(f"{g}: {x:.4f}" for g, x in tab.items()) if len(grid) > 1 else ""), flush=True)
                res[name] = {**d_, "per_fold": pf, "chosen": ch, "grid": {str(g): x for g, x in tab.items()}}
            ag = lambda X: float((soft(A["full"]).argmax(1) == soft(X).argmax(1)).mean())  # noqa: E731
            print(f"  most likely level the same as the one-question reading: two pieces {ag(A['two']):.3f}" + (f", four pieces {ag(A['four']):.3f}" if big == 20 else "")
                  + f"; mean entropy (nats): one question {float(-(soft(A['full']) * A['full']).sum(1).mean()):.3f}, two pieces added at T = 1 {float(-(soft(A['two']) * np.log(np.maximum(soft(A['two']), 1e-12))).sum(1).mean()):.3f}")
            out[str(big)] = res
        rep[f"seeds_{seeds[0]}"] = out
    (ROOT / "results/r_variants/m233_split.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nREPORT ONLY; written m233_split.json")


if __name__ == "__main__":
    sys.exit(main())
