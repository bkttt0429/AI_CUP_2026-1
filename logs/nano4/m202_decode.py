#!/usr/bin/env python
"""M202 (2026-11-16): the level composition of the Public half from the Public scores of constant-answer files.

A file that answers level k on every question scores (n_k + exp(-1) * (n_{k-1} + n_{k+1})) / 400 on the 400 Public questions;
at 7 decimals the pair (n_k, n_{k-1} + n_{k+1}) is unique. The files all-A, all-1D, all-4D, all-5D give the exact counts of
A, 1D, 4D, 5D and through the neighbour sums those of B, 2D, 3D, 6D, and C + D; all-C completes the ten levels.
Readings registered in the design doc (M202) before any score: the ratios 5D : 4D and A : 1D.

  python -I logs/nano4/m202_decode.py A=0.2431234 1D=0.1512345 4D=0.1612345 5D=0.2012345 [C=0.2112345]
Writes results/trunk_seq/es/public_composition.json (not the upload ledger: these files change every answer, the ledger's
exact / within-one decomposition of a set of changed questions does not apply to them).
"""
import json
import math
import sys
from pathlib import Path

LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; E = math.exp(-1.0); ROOT = Path.cwd()


def solve(score):
    # the leaderboard TRUNCATES to 7 decimals (seen 2026-11-16: all-1D 0.1427728 for 0.14277287, all-A 0.2071379 for 0.20713800),
    # so the true value lies in [score, score + 1e-7): 400 * that = a window of 4e-5 points above the printed number
    t = 400.0 * score; out = [(n, m) for n in range(401) for m in range(401 - n) if -2e-6 < n + E * m - t < 4.2e-5]
    return out


def main():
    sc = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[1:] if "=" in a}; assert sc and all(k in LEV for k in sc), "arguments: LEVEL=score ..."
    n, nb = {}, {}
    for k, s in sc.items():
        sol = solve(s)
        if len(sol) != 1:
            sys.exit(f"all-{k}: score {s} has {len(sol)} decompositions {sol[:5]} - check the digits")
        n[k], nb[k] = sol[0]; i = LEV.index(k); print(f"all-{k}: score {s:.7f} -> {k} = {n[k]}, " + " + ".join(LEV[j] for j in (i - 1, i + 1) if 0 <= j < 10) + f" = {nb[k]}")
    changed = True
    while changed:  # neighbour sums -> further levels
        changed = False
        for k in list(nb):
            i = LEV.index(k); nbrs = [LEV[j] for j in (i - 1, i + 1) if 0 <= j < 10]; unknown = [x for x in nbrs if x not in n]
            if len(unknown) == 1:
                n[unknown[0]] = nb[k] - sum(n[x] for x in nbrs if x in n); changed = True
            elif not unknown and sum(n[x] for x in nbrs) != nb[k]:
                sys.exit(f"inconsistent: neighbours of {k} sum to {sum(n[x] for x in nbrs)}, the score says {nb[k]}")
    if any(v < 0 for v in n.values()):
        sys.exit(f"negative count {n} - check the digits")
    rest = [x for x in LEV if x not in n]; left = 400 - sum(n.values())
    if len(rest) == 1:
        n[rest[0]] = left; rest = []
    print("Public composition (questions of 400): " + ", ".join(f"{x} {n[x]}" for x in LEV if x in n) + (f"; {' + '.join(rest)} = {left}" if rest else ""))
    sd = lambda c: math.sqrt(c * (1 - c / 400.0))  # noqa: E731  (binomial sd of the same level's count in the other half)
    print("Private, if both halves are draws from one population: the same counts with a sampling sd of " + ", ".join(f"{x} {sd(n[x]):.1f}" for x in LEV if x in n))
    rep = {"scores": sc, "counts": n, "neighbour_sums": nb, "unresolved": {"levels": rest, "sum": left} if rest else None}
    for a, b, lo, hi, what in (("5D", "4D", 1.0, 1.3, "the 4D->5D push"), ("A", "1D", 1.3, 2.0, "the 1D->A push and 'never leave A'")):
        if a in n and b in n:
            r = n[a] / max(n[b], 1); verdict = "supported by the composition" if r >= hi else ("NOT supported by the composition" if r <= lo else "in between: kept, recorded")
            rep[f"{a}:{b}"] = r; print(f"registered reading, {a} : {b} = {n[a]} : {n[b]} = {r:.2f} (thresholds {lo} / {hi}) -> {what}: {verdict}")
    if "6D" in n and "5D" in n:
        print(f"recorded only: 6D : 5D = {n['6D']} : {n['5D']} = {n['6D'] / max(n['5D'], 1):.2f} ('no 5D -> 6D')" + (f"; kyu share {sum(n[x] for x in ('D', 'C', 'B', 'A')) / 400:.3f}" if all(x in n for x in ("D", "C", "B", "A")) else (f"; kyu share {(left + n.get('B', 0) + n.get('A', 0)) / 400:.3f}" if rest == ["D", "C"] and "B" in n and "A" in n else "")))
    out = ROOT / "results/trunk_seq/es/public_composition.json"; out.write_text(json.dumps(rep, indent=1)); print(f"written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
