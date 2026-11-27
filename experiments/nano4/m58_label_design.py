"""Section 4M M58 label-decoding design (2026-11-08; user approved): a NON-ADAPTIVE batch of K query files that, decoded jointly with the
existing uploads by m58_decode.py's MILP, determines the true class of the public questions routed to A / 1D (ct9 decision A or 1D, 82
public questions; the 8 ct13 flips are already known A). Query k = ct13 with a random half of those questions flipped A <-> 1D (private rows
untouched). Simulation: truths drawn from ct9's posterior with the A|1D shift (A odds x2 on 1D-decided questions) and the 8 known set to A;
for K in KS, the share of the 74 unknown questions whose class is determined (one feasible class). Then --write builds the K* query files.
  python -I logs/nano4/m58_label_design.py --sims 12                 # choose K
  python -I logs/nano4/m58_label_design.py --write 20                # submission_rank_lab01..lab20.csv
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.submission import validate_submission, write_submission  # noqa: E402
from m58_decode import LEVELS, RANK, decisions, determined, prior_from_ct9, solve, xy  # noqa: E402

KS = (16, 24, 32)
A, D1 = RANK["A"], RANK["1D"]


def design(n_items, kmax, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(kmax, n_items)).astype(bool)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sims", type=int, default=0); ap.add_argument("--write", type=int, default=0); a = ap.parse_args()
    d9, d13 = decisions("submission_rank_ct9.csv"), decisions("submission_rank_ct13.csv").reindex(decisions("submission_rank_ct9.csv").index)
    pub = [q for q in d9.index if q.startswith("pub")]
    U = [q for q in pub if d9[q] in (A, D1)]
    known = [q for q in pub if d13[q] != d9[q]]
    unknown = [q for q in U if q not in known]
    D = design(len(U), max(KS + (a.write,)))
    print(f"routed public questions {len(U)}, known (ct13 flips) {len(known)}, unknown {len(unknown)}", flush=True)
    prior = prior_from_ct9()
    if a.sims:
        rng = np.random.default_rng(0)
        res = {k: [] for k in KS}
        for sim in range(a.sims):
            t0 = time.time()
            truth = {}
            for q in U:
                p = prior[q].copy()
                if d9[q] == D1:
                    p[A] *= 2.0
                p /= p.sum(); truth[q] = int(rng.choice(10, p=p))
            for q in known:
                truth[q] = A
            # observations: ct13 vs ct9, and the K queries (each vs ct9)
            def obs_of(dec):
                ch = [q for q in U if dec[q] != d9[q]]
                X = sum(xy(int(d9[q]), int(dec[q]), truth[q])[0] for q in ch); Y = sum(xy(int(d9[q]), int(dec[q]), truth[q])[1] for q in ch)
                return {"name": "", "moves": {q: (int(d9[q]), int(dec[q])) for q in ch}, "X": X, "Y": Y}
            base = obs_of(d13)
            for k in KS:
                obs = [base]
                for j in range(k):
                    dec = d13.copy()
                    for i, q in enumerate(U):
                        if D[j, i]:
                            dec[q] = D1 if d13[q] == A else A if d13[q] == D1 else d13[q]
                    obs.append(obs_of(dec))
                qs = sorted({q for o in obs for q in o["moves"]})
                cand = {q: sorted({c for dd in ({int(d9[q])} | {o["moves"][q][1] for o in obs if q in o["moves"]}) for c in range(dd - 2, dd + 3) if 0 <= c <= 9}) for q in qs}
                t1 = time.time(); mapa = solve(qs, cand, obs, prior, time_limit=60)
                if mapa in (None, "timeout"):
                    res[k].append((0.0, 0.0)); print(f"   K {k}: MAP {mapa}", flush=True); continue
                dm = determined(qs, cand, obs, prior, mapa, only=[q for q in unknown if q in qs], time_limit=10)
                det = [q for q, v in dm.items() if v]
                print(f"   K {k}: MAP {time.time() - t1:.0f} s total", flush=True)
                res[k].append((len(det) / len(unknown), float(np.mean([mapa[q] == truth[q] for q in unknown if q in mapa]))))
            print(f"sim {sim}: " + "  ".join(f"K {k}: determined {res[k][-1][0]:.2f}, MAP correct {res[k][-1][1]:.2f}" for k in KS) + f" ({time.time() - t0:.0f} s)", flush=True)
        for k in KS:
            arr = np.array(res[k]); print(f"K {k:2d}: determined share mean {arr[:, 0].mean():.3f} (min {arr[:, 0].min():.2f}); MAP correct {arr[:, 1].mean():.3f}", flush=True)
    if a.write:
        order = pd.read_csv(ROOT / "submission_rank_ct13.csv", dtype=str)["question_id"]
        for j in range(a.write):
            dec = d13.copy()
            for i, q in enumerate(U):
                if D[j, i]:
                    dec[q] = D1 if d13[q] == A else A if d13[q] == D1 else d13[q]
            out = ROOT / f"submission_rank_lab{j + 1:02d}.csv"
            rows = [(q, LEVELS[int(dec[q])]) for q in order]
            write_submission(rows, out)
            problems = validate_submission(out, order.tolist(), 800, {"pub_": 400, "priv_": 400})
            ch = int((dec != d13).sum())
            print(f"{out.name}: {ch} public flips vs ct13 (private untouched: {int((dec[[q for q in d13.index if q.startswith('priv')]] != d13[[q for q in d13.index if q.startswith('priv')]]).sum()) == 0}); "
                  f"problems {problems or 'none'}; md5 {hashlib.md5(out.read_bytes()).hexdigest()[:12]}", flush=True)


if __name__ == "__main__":
    main()
