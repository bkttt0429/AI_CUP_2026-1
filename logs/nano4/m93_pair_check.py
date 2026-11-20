"""Section 4M M93 (2026-11-13, report only): are games that involve test players present in the training rows at all? M91 found
mirror matches for validation players (who are training players) but no excess for the 800 test questions. A targeted check with a
much higher prior: when a question's game has a confidently identified opponent B (a training player), B's own ~125 rows are the most
likely place for further games between the two. For each such pair (A, B): the largest standardised similarity between A's prototype
and the opponent side of B's rows, against the same statistic for a random other training player B' of the SAME level and a similar number of rows (control; level-matched
because B's opponents are of A's level anyway). Same network
(fold-1 network, three seeds), same code for two pools: (V) the fold-1 players with one of their validation questions each;
(T) the 800 test questions. If training rows held games of test players, (T) would separate from its control as (V) does.
  python -I logs/nano4/m93_pair_check.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m91_mirror as mm  # noqa: E402
from ens_compare import load_fold  # noqa: E402

V_NET, TEST = 1, ROOT / "results/ct_submission/vec"


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; D = mm.load_meta(log); N = len(D["key"]); C = len(D["players"]); kidx = {k: i for i, k in enumerate(D["key"])}; rng = np.random.default_rng(93); F = torch.nn.functional
    Z = [np.load(mm.EMB / f"emb_{V_NET}{f'_s{s}' if s else ''}.npz") for s in mm.NSEED]; Eown = [torch.as_tensor(z["own"].astype(np.float32), device=dev) for z in Z]; Eopp = [torch.as_tensor(z["opp"].astype(np.float32), device=dev) for z in Z]
    Town = [torch.as_tensor(z["t_own"].astype(np.float32), device=dev) for z in Z]; Topp = [torch.as_tensor(z["t_opp"].astype(np.float32), device=dev) for z in Z]; del Z
    seen_rows = np.flatnonzero(D["fold"] != V_NET); own_t = torch.as_tensor(D["owner"].copy(), device=dev); sr_t = torch.as_tensor(seen_rows, device=dev); cnt = np.bincount(D["owner"][seen_rows], minlength=C); seen = np.flatnonzero(cnt >= 8); seen_t = torch.as_tensor(seen, device=dev)
    PS = [F.normalize(torch.zeros((C, 128), device=dev).index_add_(0, own_t[sr_t], e[sr_t])[seen_t], dim=1) for e in Eown]; rows_of = defaultdict(list)
    for r in seen_rows:
        rows_of[D["owner"][r]].append(r)
    rows_of = {k: np.array(v) for k, v in rows_of.items()}; sizes = np.array([len(rows_of[p]) for p in seen]); ylab = np.full(C, -1, np.int64); ylab[D["owner"]] = D["lab"]; slab = ylab[seen]
    # standardisation of the forward scores over a sample of opponent-side rows
    samp = torch.as_tensor(rng.choice(seen_rows, 100_000, replace=False), device=dev); FMU, FSD = [], []
    for s in range(3):
        S = Eopp[s][samp] @ PS[s].T; FMU.append(S.mean(0)); FSD.append(S.std(0) + 1e-6); del S

    def pool(name, own_vecs, opp_vecs):
        """own_vecs / opp_vecs: per seed, a list over questions of [n_games, 128] tensors (own side, opponent side of the question's games)."""
        nq = len(own_vecs[0]); P = [torch.stack([F.normalize(x.sum(0), dim=0) for x in own_vecs[s]]) for s in range(3)]; MU, SD = [], []
        for s in range(3):
            s1 = torch.zeros(nq, device=dev, dtype=torch.float64); s2_ = torch.zeros_like(s1)
            for i in range(0, len(seen_rows), 100_000):
                S = (Eopp[s][sr_t[i:i + 100_000]] @ P[s].T).double(); s1 += S.sum(0); s2_ += (S * S).sum(0)
            mu = s1 / len(seen_rows); MU.append(mu.float()); SD.append((s2_ / len(seen_rows) - mu * mu).clamp(min=1e-12).sqrt().float() + 1e-6)
        pairs = []  # (question, identified player column, margin)
        for i in range(nq):
            Zf = sum((opp_vecs[s][i] @ PS[s].T - FMU[s]) / FSD[s] for s in range(3)) / 3; t = Zf.topk(2, dim=1)
            pairs += [(i, int(c), float(m)) for c, m in zip(t.indices[:, 0].tolist(), (t.values[:, 0] - t.values[:, 1]).tolist())]
        return P, MU, SD, pairs

    def stat(P, MU, SD, q, col):
        r = torch.as_tensor(rows_of[seen[col]], device=dev); z = sum((Eopp[s][r] @ P[s][q] - MU[s][q]) / SD[s][q] for s in range(3)) / 3
        return float(z.max()), int((z > 4).sum()), len(r), int((z > 5).sum())
    # pool V: fold-1 players, one validation question each (a 10-game question when there is one)
    y, n, pid, q = load_fold(V_NET); gk = list(q["game_keys"]); by_p = defaultdict(dict)
    for i, (p_, m_) in enumerate(zip(pid, n)):
        by_p[p_].setdefault(int(m_), i)
    qs = [d_.get(10, next(iter(d_.values()))) for d_ in by_p.values()]; rows_v = [torch.as_tensor([kidx[k] for k in gk[i] if k in kidx], dtype=torch.long, device=dev) for i in qs]
    PV = pool("validation", [[e[r] for r in rows_v] for e in Eown], [[e[r] for r in rows_v] for e in Eopp])
    tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); qid = tab["question_id"].to_numpy(); uq = np.unique(qid); rows_t = [torch.as_tensor(np.flatnonzero(qid == x), device=dev) for x in uq]
    PT = pool("test", [[e[r] for r in rows_t] for e in Town], [[e[r] for r in rows_t] for e in Topp])
    thr = np.quantile([m for _, _, m in PV[3]], [0.8, 0.9, 0.95]); log(f"forward margins: validation quantiles 0.8 / 0.9 / 0.95 = {' / '.join(f'{x:.2f}' for x in thr)}; share of test games above them: " + " / ".join(f"{float(np.mean([m >= x for _, _, m in PT[3]])):.3f}" for x in thr) + " (validation 0.200 / 0.100 / 0.050)")
    for name, (P, MU, SD, pairs) in (("validation (fold-1 players, one question each)", PV), ("test (800 questions)", PT)):
        for lab_, lo in (("top 20% forward margin", thr[0]), ("top 5% forward margin", thr[2]), ("all games", -1e9)):
            sel = [(i, c) for i, c, m in pairs if m >= lo]; A, B = [], []
            for i, c in sel:
                A.append(stat(P, MU, SD, i, c)); cands = np.flatnonzero((slab == slab[c]) & (np.abs(sizes - sizes[c]) <= max(10, int(0.2 * sizes[c])))); cands = cands[cands != c]; c2 = int(rng.choice(cands)) if len(cands) else int(rng.integers(len(seen))); B.append(stat(P, MU, SD, i, c2))
            a, b_ = np.array(A), np.array(B)
            log(f"{name}, {lab_}: {len(sel)} pairs; largest standardised similarity over the identified player's rows: mean {a[:, 0].mean():.2f} (control {b_[:, 0].mean():.2f}); share above 4 / 5 / 6: {float((a[:, 0] > 4).mean()):.3f} / {float((a[:, 0] > 5).mean()):.3f} / {float((a[:, 0] > 6).mean()):.3f} (control {float((b_[:, 0] > 4).mean()):.3f} / {float((b_[:, 0] > 5).mean()):.3f} / {float((b_[:, 0] > 6).mean()):.3f}); identified player's rows with similarity above 4, per pair: {a[:, 1].mean():.3f} (control {b_[:, 1].mean():.3f}), above 5: {a[:, 3].mean():.4f} (control {b_[:, 3].mean():.4f}); pairs with two or more rows above 4: {float((a[:, 1] >= 2).mean()):.4f} (control {float((b_[:, 1] >= 2).mean()):.4f}); rows per player {a[:, 2].mean():.0f} (control {b_[:, 2].mean():.0f})")
    log("done")


if __name__ == "__main__":
    main()
