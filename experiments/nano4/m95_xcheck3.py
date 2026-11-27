"""Section 4M M95 step 1d (2026-11-13, label-free): how many rank-test players are among the pub2 / priv2 candidates? Steps 1 / 1b
found no overlap with the old Public pool and a few strong pairs in the added pools. A cleaner count by split halves: a question's
games are split in two at random; each half is scored against the pool (15 M90 networks averaged, scores standardised per candidate
within the query set); "agree" = both halves pick the same candidate. A member's halves agree often, a non-member's rarely.
Controls with the same code: (P10) the old Public pool's own questions against that pool with every candidate cut to 10 random
games (members: the power with 10-game candidates); (N10) the same with the true candidate removed (non-members); (T-old10) the
rank-test questions against the cut old Public pool (non-members with the rank test's own structure, since steps 1 / 1b showed no
overlap there). Then the rank test against pub2 and priv2. Questions with at least 10 games only.
  python -I logs/nano4/m95_xcheck3.py
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
    dev = "cuda"; F = torch.nn.functional; rng = np.random.default_rng(953); G = np.load(X / "games.npz"); grp = G["group"].astype(str)
    tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); tq = tab["question_id"].to_numpy(); tqs = np.unique(tq); t_rows = [np.flatnonzero(tq == x) for x in tqs]
    names = {p: np.array(sorted(set(x for x in grp if x.startswith(p + ":c:")))) for p in ("oldpub", "pub2", "priv2")}; rows = {p: [np.flatnonzero(grp == x) for x in names[p]] for p in names}
    rows["old10"] = [rng.choice(r, 10, replace=False) for r in rows["oldpub"]]; names["old10"] = names["oldpub"]
    qn = np.array(sorted(set(x for x in grp if x.startswith("oldpub:q:")))); q_rows = [np.flatnonzero(grp == x) for x in qn]; truth = pd.read_csv(TRUTH, dtype=str).set_index("question_id")["top_1"].to_dict(); cpos = {x.split(":c:")[1]: i for i, x in enumerate(names["oldpub"])}; q_true = np.array([cpos[truth[x.split(":q:")[1]]] for x in qn])

    def halves(rs):
        out = []
        for r in rs:
            p = rng.permutation(r); out.append((p[:len(p) // 2], p[len(p) // 2:]))
        return [x[0] for x in out], [x[1] for x in out]
    tA, tB = halves(t_rows); qA, qB = halves(q_rows); big_t = np.array([len(r) >= 10 for r in t_rows]); big_q = np.array([len(r) >= 10 for r in q_rows])
    proto = lambda E, rs: torch.stack([F.normalize(E[torch.as_tensor(r, device=dev)].sum(0), dim=0) for r in rs])  # noqa: E731
    std = lambda S: (S - S.mean(0)) / (S.std(0) + 1e-6)  # noqa: E731
    tags = [f"{v}{f'_s{s}' if s else ''}" for s in (0, 1, 2) for v in m.NETS]; Z = {}
    for tag in tags:
        town = torch.as_tensor(np.load(mm.EMB / f"emb_{tag}.npz")["t_own"].astype(np.float32), device=dev); xo = torch.as_tensor(np.load(X / f"xemb_{tag}.npz")["own"].astype(np.float32), device=dev)
        Q = {"T": [proto(town, r) for r in (t_rows, tA, tB)], "P": [proto(xo, r) for r in (q_rows, qA, qB)]}
        for p in ("old10", "pub2", "priv2"):
            PC = proto(xo, rows[p])
            for k in ("T", "P"):
                for h in range(3):
                    Z[(k, p, h)] = Z.get((k, p, h), 0.0) + std(Q[k][h] @ PC.T) / len(tags)
    rep = {}

    def summary(label, k, p, big, mask_true=False, true=None):
        zs = [Z[(k, p, h)].clone() for h in range(3)]
        if mask_true:
            ar = torch.arange(len(true), device=dev); tt = torch.as_tensor(true, device=dev)
            for z in zs:
                z[ar, tt] = -float("inf")
        top = [z.topk(2, dim=1) for z in zs]; j = [t.indices[:, 0].cpu().numpy() for t in top]; mg = (top[0].values[:, 0] - top[0].values[:, 1]).cpu().numpy(); ag = (j[1] == j[2]) & big; ag3 = ag & (j[0] == j[1]); n = int(big.sum())
        extra = f"; top-1 correct (whole question) {float((j[0] == true).mean()):.3f}, both halves correct {float(((j[1] == true) & (j[2] == true))[big].mean()):.3f}" if (true is not None and not mask_true) else ""
        log(f"{label}: {n} questions with >= 10 games; both halves pick the same candidate {int(ag.sum())} ({ag.sum() / n:.4f}); and the whole question picks it too {int(ag3.sum())} ({ag3.sum() / n:.4f}); whole-question margin median {np.median(mg):.2f}; agreeing questions with margin above 0.8 / 1.0 / 1.5: {int((ag3 & (mg > 0.8)).sum())} / {int((ag3 & (mg > 1.0)).sum())} / {int((ag3 & (mg > 1.5)).sum())} ({(ag3 & (mg > 1.0)).sum() / n:.4f} above 1.0){extra}")
        rep[label] = {"n": n, "agree": int(ag.sum()), "agree3": int(ag3.sum())}
        return ag3, j[0], mg
    summary("P10  old Public questions vs their pool cut to 10 games (members)", "P", "old10", big_q, true=q_true)
    summary("N10  the same with the true candidate removed (non-members)", "P", "old10", big_q, mask_true=True, true=q_true)
    summary("T-old10  rank test vs the cut old Public pool (non-members, rank-test structure)", "T", "old10", big_t)
    for p in ("pub2", "priv2"):
        summary(f"P-{p}  old Public questions vs {p} (non-members unless the pools share players)", "P", p, big_q)
        ag3, j0, mg = summary(f"T-{p}  rank test vs {p}", "T", p, big_t); idx = np.flatnonzero(ag3); idx = idx[np.argsort(-mg[idx])]
        pubq = np.char.startswith(tqs[idx].astype(str), "pub_"); log(f"   agreeing rank-test questions: {len(idx)} ({int(pubq.sum())} Public, {int((~pubq).sum())} Private); by question size: " + ", ".join(f"{s}: {int(sum(len(t_rows[i]) == s for i in idx))}" for s in (10, 15, 20)))
        log("   " + "; ".join(f"{tqs[i]} ({len(t_rows[i])} games) -> {names[p][j0[i]].split(':c:')[1]} (margin {mg[i]:.2f})" for i in idx[:40])); rep[f"pairs_{p}"] = [{"question": str(tqs[i]), "candidate": names[p][j0[i]].split(":c:")[1], "margin": float(mg[i]), "games": int(len(t_rows[i]))} for i in idx]
    (mm.RV / "m95_xcheck3.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
