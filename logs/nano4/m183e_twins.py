#!/usr/bin/env python
"""Section 4M M183e (2026-11-16, report only): is the player part "the same person at another level"?

M154: about a tenth of the players have a near-identical second code (nearest prototype with similarity >= 0.9, far above
the second nearest); its level is the same in 70-80 % and adjacent otherwise, never two apart - the same person under
another code. If what is left of the loss is a person straddling two levels, then for a player whose two disjoint halves
agree on a level that is NOT the label, the twin's label should be that agreed level far more often than for an arbitrary
neighbour level.

Prototypes: all-players network of the second production set (results/m161q/out_-1_s30.npz and s31, mean of the cosines).
Twin of a player = the nearest other prototype if similarity >= 0.90 and the gap to the second nearest >= 0.05.
Rows: players whose halves agree and are right / agree and are wrong / disagree. Columns: twin's label equals the player's
label / equals the agreed answer / other. Control for "agree and wrong": the same table with the level on the OTHER side of
the label (label minus the direction of the answer) - by symmetry a stranger of similar strength would land there as often.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def main():
    q = np.load(mb7.CACHE / "queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); S = 0
    for s in (30, 31):
        P = np.load(ROOT / f"results/m161q/out_-1_s{s}.npz")["protos"].astype(np.float32); P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12); S = S + P @ P.T
    S /= 2; np.fill_diagonal(S, -1.0); o = np.argsort(-S, axis=1)[:, :2]; s1 = S[np.arange(len(S)), o[:, 0]]; s2 = S[np.arange(len(S)), o[:, 1]]
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(players).to_numpy()
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); dec = z["anat_plain"].astype(int)
    df = pd.DataFrame({"pid": pid, "n": n, "y": y, "dec": dec, "row": np.arange(len(y))}); sub = df[df.n == 10].sort_values(["pid", "row"]); g = sub.groupby("pid"); ok = g.size() == 2; sub = sub[sub.pid.isin(ok.index[ok])]
    D = sub.dec.to_numpy().reshape(-1, 2); Y = sub.y.to_numpy().reshape(-1, 2)[:, 0]; PP = sub.pid.to_numpy().reshape(-1, 2)[:, 0]; idx = {p: i for i, p in enumerate(players)}; pi = np.array([idx[p] for p in PP]); assert (plab[pi] == Y).all()
    for thr, gap in ((0.90, 0.05), (0.85, 0.03), (0.95, 0.05)):
        tw = (s1[pi] >= thr) & (s1[pi] - s2[pi] >= gap); tl = plab[o[pi, 0]]; print(f"twin = nearest prototype with similarity >= {thr} and gap >= {gap}: {int(tw.sum())} of {len(pi)} validation players ({float(tw.mean()):.3f}); twin's level equals the player's {float((tl[tw] == Y[tw]).mean()):.3f}, one apart {float((np.abs(tl[tw] - Y[tw]) == 1).mean()):.3f}, two or more {float((np.abs(tl[tw] - Y[tw]) >= 2).mean()):.3f}")
        ag = D[:, 0] == D[:, 1]; A = D[:, 0]
        for lab, sel in (("halves agree and are right", ag & (A == Y)), ("halves agree, label elsewhere", ag & (A != Y)), ("halves disagree", ~ag)):
            m = sel & tw
            if m.sum() < 20:
                print(f"    {lab}: {int(m.sum())} players with a twin (too few)"); continue
            line = f"    {lab}: {int(sel.sum())} players, {int(m.sum())} with a twin ({float(m.sum() / max(sel.sum(), 1)):.3f}); twin's label = the player's label {float((tl[m] == Y[m]).mean()):.3f}"
            if "elsewhere" in lab:
                mirror = 2 * Y[m] - A[m]; line += f", = the agreed answer {float((tl[m] == A[m]).mean()):.3f}, = the level on the other side of the label {float((tl[m] == mirror).mean()):.3f}; among twins whose label differs from the player's: on the answer's side {int(((tl[m] != Y[m]) & (np.sign(tl[m] - Y[m]) == np.sign(A[m] - Y[m]))).sum())}, on the other side {int(((tl[m] != Y[m]) & (np.sign(tl[m] - Y[m]) == -np.sign(A[m] - Y[m]))).sum())}"
            elif "disagree" in lab:
                line += f", = one of the two answers that is not the label {float(((tl[m] != Y[m]) & ((tl[m] == D[m, 0]) | (tl[m] == D[m, 1]))).mean()):.3f}"
            print(line)
        # the reverse view: players with a twin of a DIFFERENT level - where do our answers go?
        dm = tw & (tl != Y); a20 = df[df.n == 20].set_index("pid").dec.reindex(PP).to_numpy()
        if dm.sum() >= 30:
            print(f"    players whose twin carries a different level ({int(dm.sum())}): our 20-game answer is the label {float((a20[dm] == Y[dm]).mean()):.3f}, the twin's level {float((a20[dm] == tl[dm]).mean()):.3f}, something else {float(((a20[dm] != Y[dm]) & (a20[dm] != tl[dm])).mean()):.3f}; players whose twin carries the same level ({int((tw & (tl == Y)).sum())}): the label {float((a20[tw & (tl == Y)] == Y[tw & (tl == Y)]).mean()):.3f}; players without a twin ({int((~tw).sum())}): the label {float((a20[~tw] == Y[~tw]).mean()):.3f}")


if __name__ == "__main__":
    main()
