"""Section 4M M95 step 1b (2026-11-13, label-free; written after step 1, whose threshold turned out to have little power: the pool's
own questions passed the reference's q99.9 in only 10-22% of the cases although their top-1 was right in 94%). Same question with
more power: all 15 M90 networks averaged (nobody involved was seen by any of them), and a reference made of true non-members: the
old Public pool's own 400 questions scored against (a) the old Public candidates WITHOUT their true candidate, (b) the pub2 / priv2
candidates. Positive control: the same questions with their true candidate. Scores are standardised per candidate within each
query set. Reported: share of the rank-test questions above the reference's q95 / q99 / q99.9 (best score and margin), the positive
control's detection rate at the same thresholds, the implied share of rank-test players in the pool, and the best pairs.
  python -I logs/nano4/m95_xcheck2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m88_seqid as m  # noqa: E402
import m91_mirror as mm  # noqa: E402

X, TEST, TRUTH = ROOT / "results/m95", ROOT / "results/ct_submission/vec", Path.home() / "BK_AICUP_2026-2/results/probe/upload_fix.csv"


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; F = torch.nn.functional; G = np.load(X / "games.npz"); grp = G["group"].astype(str)
    tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); tq = tab["question_id"].to_numpy(); tqs = np.unique(tq); t_rows = [np.flatnonzero(tq == x) for x in tqs]
    pools = {p: (np.array(sorted(set(x for x in grp if x.startswith(p + ":c:")))),) for p in ("oldpub", "pub2", "priv2")}; pools = {p: (v[0], [np.flatnonzero(grp == x) for x in v[0]]) for p, v in pools.items()}
    qn = np.array(sorted(set(x for x in grp if x.startswith("oldpub:q:")))); q_rows = [np.flatnonzero(grp == x) for x in qn]; truth = pd.read_csv(TRUTH, dtype=str).set_index("question_id")["top_1"].to_dict(); cpos = {x.split(":c:")[1]: i for i, x in enumerate(pools["oldpub"][0])}; q_true = torch.as_tensor(np.array([cpos[truth[x.split(":q:")[1]]] for x in qn]), device=dev)
    proto = lambda E, rows: torch.stack([F.normalize(E[torch.as_tensor(r, device=dev)].sum(0), dim=0) for r in rows])  # noqa: E731
    std = lambda S: (S - S.mean(0)) / (S.std(0) + 1e-6)  # noqa: E731
    ZT = {p: 0.0 for p in pools}; ZP = {p: 0.0 for p in pools}; tags = [f"{v}{f'_s{s}' if s else ''}" for s in (0, 1, 2) for v in m.NETS]
    for tag in tags:
        town = torch.as_tensor(np.load(mm.EMB / f"emb_{tag}.npz")["t_own"].astype(np.float32), device=dev); xo = torch.as_tensor(np.load(X / f"xemb_{tag}.npz")["own"].astype(np.float32), device=dev); QT, QP = proto(town, t_rows), proto(xo, q_rows)
        for p, (names, rows) in pools.items():
            PC = proto(xo, rows); ZT[p] = ZT[p] + std(QT @ PC.T) / len(tags); ZP[p] = ZP[p] + std(QP @ PC.T) / len(tags)
    rep = {}; ar = torch.arange(len(qn), device=dev)
    for p, (names, rows) in pools.items():
        zt, zp = ZT[p], ZP[p]; tT = zt.topk(2, dim=1); bT = tT.values[:, 0].cpu().numpy(); mT = (tT.values[:, 0] - tT.values[:, 1]).cpu().numpy(); jT = tT.indices[:, 0].cpu().numpy()
        if p == "oldpub":
            tP = zp.topk(2, dim=1); bP = tP.values[:, 0].cpu().numpy(); mP = (tP.values[:, 0] - tP.values[:, 1]).cpu().numpy(); acc = float((tP.indices[:, 0] == q_true).float().mean()); zn = zp.clone(); zn[ar, q_true] = -float("inf")
        else:
            zn = zp
        tN = zn.topk(2, dim=1); bN = tN.values[:, 0].cpu().numpy(); mN = (tN.values[:, 0] - tN.values[:, 1]).cpu().numpy(); line = f"pool {p} ({len(names)} candidates), 15 networks: best score, reference (non-members, {len(bN)}) median {np.median(bN):.2f} q95 {np.quantile(bN, 0.95):.2f} q99 {np.quantile(bN, 0.99):.2f} max {bN.max():.2f} | rank test median {np.median(bT):.2f} q95 {np.quantile(bT, 0.95):.2f} q99 {np.quantile(bT, 0.99):.2f} max {bT.max():.2f}"
        out = {}
        for name, q_ in (("q95", 0.95), ("q99", 0.99), ("q99.9", 0.999)):
            eb, em = np.quantile(bN, q_), np.quantile(mN, q_); sb, sm_ = float((bT > eb).mean()), float((mT > em).mean()); line += f" | above the reference's {name}: score {sb:.4f}, margin {sm_:.4f} (nominal {1 - q_:.4f})"; out[name] = {"score": sb, "margin": sm_}
            if p == "oldpub":
                db, dm = float((bP > eb).mean()), float((mP > em).mean()); f_b = (sb - (1 - q_)) / max(db - (1 - q_), 1e-9); line += f"; positive control detected {db:.3f} / {dm:.3f} -> implied share of rank-test players in this pool {f_b:+.4f}"; out[name].update(detect_score=db, detect_margin=dm, implied_share=f_b)
        if p == "oldpub":
            line += f" | positive control top-1 correct {acc:.3f}, its best score median {np.median(bP):.2f}, margin median {np.median(mP):.2f}; reference margin median {np.median(mN):.2f}; rank-test margin median {np.median(mT):.2f}"
        log(line); order = np.argsort(-bT)[:8]; log("   best rank-test pairs: " + "; ".join(f"{tqs[i]} -> {names[jT[i]].split(':c:')[1]} (score {bT[i]:.2f}, margin {mT[i]:.2f})" for i in order)); rep[p] = out
    (mm.RV / "m95_xcheck2.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
