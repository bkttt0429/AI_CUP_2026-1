"""Section 4M M95 step 1 (2026-11-13, label-free existence check; user: "我判斷規則允許"): are rank-test players among competition 1's
candidates? Per fold network v (three seeds; nobody involved was seen by it): candidates = the competition-1 candidates of one pool
(old Public: 400 x 100 games; pub2 / priv2: 600 x 10 games), prototype = sum of the own-side vectors of the candidate's games;
queries = question prototypes of (T) the 800 rank-test questions, (V) one validation question per fold-v training player with the
test-like size mix (the reference: same structure, same pipeline), (P) the old Public pool's own 400 questions (positive control:
the true candidate is known from the scorer-confirmed answers). Score = cosine, standardised per candidate over the V queries,
mean over seeds; per query the best candidate's score and the margin over the second.
Reading rule written before the run: overlap exists if, in at least 3 of the 4 fold networks, at least 2% of the test questions
score above the reference's 99.9th percentile in some pool; no overlap if every pool stays within 3x the nominal 0.1%.
  python -I logs/nano4/m95_xcheck.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m91_mirror as mm  # noqa: E402
from ens_compare import FOLDS, load_fold  # noqa: E402

X, TEST, TRUTH = ROOT / "results/m95", ROOT / "results/ct_submission/vec", Path.home() / "BK_AICUP_2026-2/results/probe/upload_fix.csv"


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; F = torch.nn.functional; D = mm.load_meta(log); kidx = {k: i for i, k in enumerate(D["key"])}; G = np.load(X / "games.npz"); grp = G["group"].astype(str)
    tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); tq = tab["question_id"].to_numpy(); tqs = np.unique(tq); t_rows = [np.flatnonzero(tq == x) for x in tqs]
    pools = {}
    for p in ("oldpub", "pub2", "priv2"):
        names = np.array(sorted(set(x for x in grp if x.startswith(p + ":c:")))); pools[p] = (names, [np.flatnonzero(grp == x) for x in names])
    qn = np.array(sorted(set(x for x in grp if x.startswith("oldpub:q:")))); q_rows = [np.flatnonzero(grp == x) for x in qn]; truth = pd.read_csv(TRUTH, dtype=str).set_index("question_id")["top_1"].to_dict()
    cpos = {x.split(":c:")[1]: i for i, x in enumerate(pools["oldpub"][0])}; q_true = np.array([cpos.get(truth.get(x.split(":q:")[1], ""), -1) for x in qn]); log(f"pools: " + ", ".join(f"{p} {len(v[0])} candidates" for p, v in pools.items()) + f"; old Public questions {len(qn)} ({int((q_true >= 0).sum())} with a known candidate); rank-test questions {len(tqs)}")
    proto = lambda E, rows: torch.stack([F.normalize(E[torch.as_tensor(r, device=dev)].sum(0), dim=0) for r in rows])  # noqa: E731
    rep = {}; hits = defaultdict(lambda: defaultdict(list))
    for v in FOLDS:
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); by_p = defaultdict(lambda: defaultdict(list)); rng = np.random.default_rng(950 + v)
        for i, (p_, m_) in enumerate(zip(pid, n)):
            by_p[p_][int(m_)].append(i)
        v_rows = []
        for p_ in sorted(by_p):
            size = int(rng.choice([5, 10, 20], p=[0.25, 0.25, 0.5])); size = size if size in by_p[p_] else sorted(by_p[p_])[-1]; v_rows.append(np.array([kidx[k] for k in gk[int(rng.choice(by_p[p_][size]))] if k in kidx]))
        Z = {p: {"T": 0.0, "V": 0.0, "P": 0.0} for p in pools}
        for s in mm.NSEED:
            sfx = f"{v}{f'_s{s}' if s else ''}"; e = np.load(mm.EMB / f"emb_{sfx}.npz"); own = torch.as_tensor(e["own"].astype(np.float32), device=dev); town = torch.as_tensor(e["t_own"].astype(np.float32), device=dev); xo = torch.as_tensor(np.load(X / f"xemb_{sfx}.npz")["own"].astype(np.float32), device=dev); del e
            QT, QV, QP = proto(town, t_rows), proto(own, v_rows), proto(xo, q_rows)
            for p, (names, rows) in pools.items():
                PC = proto(xo, rows); SV = QV @ PC.T; mu, sd = SV.mean(0), SV.std(0) + 1e-6
                for name, Q in (("T", QT), ("V", QV), ("P", QP)):
                    Z[p][name] = Z[p][name] + ((Q @ PC.T) - mu) / sd / len(mm.NSEED)
        for p, (names, rows) in pools.items():
            top = {k: Z[p][k].topk(2, dim=1) for k in ("T", "V", "P")}; b = {k: top[k].values[:, 0].cpu().numpy() for k in top}; mg = {k: (top[k].values[:, 0] - top[k].values[:, 1]).cpu().numpy() for k in top}; jt = top["T"].indices[:, 0].cpu().numpy()
            q99, q999 = np.quantile(b["V"], [0.99, 0.999]); m99, m999 = np.quantile(mg["V"], [0.99, 0.999]); sT, sT9 = float((b["T"] > q999).mean()), float((b["T"] > q99).mean()); msT, msT9 = float((mg["T"] > m999).mean()), float((mg["T"] > m99).mean())
            line = f"network {v}, pool {p}: best standardised score, reference (validation, {len(b['V'])}) median {np.median(b['V']):.2f} q99 {q99:.2f} q99.9 {q999:.2f} max {b['V'].max():.2f} | rank test ({len(b['T'])}) median {np.median(b['T']):.2f} q99 {np.quantile(b['T'], 0.99):.2f} max {b['T'].max():.2f}; share above the reference's q99 / q99.9: {sT9:.4f} / {sT:.4f} (nominal 0.0100 / 0.0010) | margin: share above the reference's q99 / q99.9: {msT9:.4f} / {msT:.4f}"
            if p == "oldpub":
                ok = q_true >= 0; zt = Z[p]["P"].cpu().numpy(); jp = top["P"].indices[:, 0].cpu().numpy(); line += f" | positive control (the pool's own questions): top-1 correct {float((jp[ok] == q_true[ok]).mean()):.3f}; true candidate's score median {np.median(zt[np.flatnonzero(ok), q_true[ok]]):.2f}; share above the reference's q99.9: score {float((b['P'][ok] > q999).mean()):.3f}, margin {float((mg['P'][ok] > m999).mean()):.3f}"
            log(line); rep[f"net{v}_{p}"] = {"ref_q99": float(q99), "ref_q999": float(q999), "test_share_q99": sT9, "test_share_q999": sT, "test_margin_share_q999": msT, "test_max": float(b["T"].max()), "ref_max": float(b["V"].max())}
            for i in np.flatnonzero((b["T"] > q999) | (mg["T"] > m999)):
                hits[p][(tqs[i], names[jt[i]])].append((v, float(b["T"][i]), float(mg["T"][i])))
    log("rank-test questions above the reference's q99.9 (score or margin), by pool: question -> candidate, networks, mean score, mean margin")
    for p in pools:
        rows = sorted(hits[p].items(), key=lambda kv: (-len(kv[1]), -np.mean([x[1] for x in kv[1]]))); log(f"  pool {p}: {len(rows)} pairs; seen under >= 3 of the 4 networks: {sum(len(x[1]) >= 3 for x in rows)}")
        for (qq, cc), xs in rows[:25]:
            log(f"    {qq} -> {cc}: networks {[x[0] for x in xs]}, score {np.mean([x[1] for x in xs]):.2f}, margin {np.mean([x[2] for x in xs]):.2f}")
        rep[f"pairs_{p}"] = [{"question": qq, "candidate": cc, "networks": [x[0] for x in xs], "score": float(np.mean([x[1] for x in xs])), "margin": float(np.mean([x[2] for x in xs]))} for (qq, cc), xs in rows]
    (mm.RV / "m95_xcheck.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
