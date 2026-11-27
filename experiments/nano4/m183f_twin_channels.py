#!/usr/bin/env python
"""Section 4M M183f (2026-11-16, report only): for players whose twin carries a DIFFERENT level - which channel says what?

The base models and the look-alike reading were trained on / compare with the twin's games (another fold), so their answer
may simply copy the twin's label. The anchors do not know the twin: they read the labels of the opponents met in the
player's own games. If the anchors also point to the twin's level, the player's games really were played at that level and
the player's label is the odd one; if the anchors point to the label, the other two channels were misled by the twin.
Per channel (20-game question): answer = the label / the twin's level / something else; the mirror level (the label's other
neighbour) as the chance reference. Twin = nearest prototype, similarity >= 0.85, gap >= 0.03 (224 players in M183e).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def main():
    q = np.load(mb7.CACHE / "queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); S = 0
    for s in (30, 31):
        P = np.load(ROOT / f"results/m161q/out_-1_s{s}.npz")["protos"].astype(np.float32); P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12); S = S + P @ P.T
    S /= 2; np.fill_diagonal(S, -1.0); o = np.argsort(-S, axis=1)[:, :2]; s1 = S[np.arange(len(S)), o[:, 0]]; s2 = S[np.arange(len(S)), o[:, 1]]; idx = {p: i for i, p in enumerate(players)}
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); fold = pf.fold.astype(int).reindex(players).to_numpy(); ng = pf.n_games.astype(int).reindex(players).to_numpy()
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); m20 = n == 20; pi = np.array([idx[p] for p in pid[m20]]); Y = y[m20]
    ch = {"own play (base)": z["anat_base"].astype(int)[m20], "look-alike reading": z["anat_R"][m20].argmax(1), "anchors (newest identifier)": z["anat_L"][m20].argmax(1), "anchors (all six inputs)": z["anat_Lall"][m20].argmax(1), "pooled, no guards": z["anat_plain"].astype(int)[m20]}; k2 = z["anat_AN"][m20][:, 1]
    tw = (s1[pi] >= 0.85) & (s1[pi] - s2[pi] >= 0.03); tl = plab[o[pi, 0]]; tf = fold[o[pi, 0]]; print(f"20-game questions {len(Y)}; players with a twin {int(tw.sum())}; the twin sits in the player's own fold (not a candidate, not in the base's training folds) in {float((tf[tw] == fold[pi][tw]).mean()):.3f}; twin's games: median {int(np.median(ng[o[pi, 0]][tw]))}")
    for lab, sel in (("twin one level apart", tw & (np.abs(tl - Y) == 1)), ("twin of the same level", tw & (tl == Y)), ("no twin", ~tw)):
        print(f"{lab}: {int(sel.sum())} players (with >= 3 anchor games {int((sel & (k2 >= 3)).sum())})")
        for nm, a in ch.items():
            s_ = sel & (k2 >= 3) if nm.startswith("anchors") else sel
            if "apart" in lab:
                mir = 2 * Y - tl; print(f"    {nm:28s}: the label {float((a[s_] == Y[s_]).mean()):.3f} | the twin's level {float((a[s_] == tl[s_]).mean()):.3f} | the mirror level {float((a[s_] == mir[s_]).mean()):.3f} | else {float(((a[s_] != Y[s_]) & (a[s_] != tl[s_]) & (a[s_] != mir[s_])).mean()):.3f}")
            else:
                print(f"    {nm:28s}: the label {float((a[s_] == Y[s_]).mean()):.3f} | one level above {float((a[s_] == Y[s_] + 1).mean()):.3f} | one below {float((a[s_] == Y[s_] - 1).mean()):.3f}")
    d = tw & (np.abs(tl - Y) == 1); same_f = tf == fold[pi]
    for lab, sel in (("twin one level apart, twin in ANOTHER fold (seen by the base, a candidate of the look-alike reading)", d & ~same_f), ("twin one level apart, twin in the SAME fold (unseen)", d & same_f)):
        print(f"{lab}: {int(sel.sum())} players - " + "; ".join(f"{nm}: label {float((a[sel] == Y[sel]).mean()):.2f} / twin {float((a[sel] == tl[sel]).mean()):.2f}" for nm, a in ch.items()) if sel.sum() >= 8 else f"{lab}: {int(sel.sum())} players (too few)")


if __name__ == "__main__":
    main()
