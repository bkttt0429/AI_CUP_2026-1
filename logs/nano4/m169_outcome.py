#!/usr/bin/env python
"""Section 4M M169 step 1 (2026-11-15, report only): does the outcome of a game (proxy: who made the last move) change what a
recognised opponent's level says about the target's level? (design doc M169)

(1) every validation / test game: colour of the last move, whether the record ends with a pass -> results/r_variants/
    m169_last_val.npz (key, last_is_target, ends_pass, plies) and m169_last_test.npz (row-aligned with the identifier's test
    rows; the alignment is checked against the stored handicap and colour);
(2) the pilot games with a strong engine's final win rate (results/ai_search_moves/fold1): agreement of the proxy;
(3) validation, even games, identifier given: by margin band group and by the proxy, the distribution of
    d = candidate label - target label.

  python -I logs/nano4/m169_outcome.py SX4L~m15
"""
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; MV = re.compile(r";([BW])\[([a-z]{0,2})\]"); HD = re.compile(r"HD\[(\d+)\]")


def last(sgf, color):
    m = MV.findall(sgf)
    if not m:
        return -1, 0, 0
    c, xy = m[-1]; return int(c == color), int(xy in ("", "tt")), len(m)


def main():
    tag = sys.argv[1]; csv.field_size_limit(10 ** 9); z = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); keys = z["key"].astype(str); want = set(keys); info = {}
    for f in sorted((ROOT / "data/training").glob("train_*.csv")):
        with open(f, newline="") as fh:
            r = csv.reader(fh); next(r)
            for pid, gid, rank, color, sgf in r:
                k = f"{rank}:{gid}"
                if k in want:
                    info[k] = (*last(sgf, color), color == "B")
    A = np.array([info[k] for k in keys]); lt, ep, npl, tbk = A[:, 0], A[:, 1].astype(bool), A[:, 2], A[:, 3].astype(bool); assert np.array_equal(tbk, z["target_black"]), "colour mismatch on the validation side"
    np.savez(RV / "m169_last_val.npz", key=keys, last_is_target=lt, ends_pass=ep, plies=npl)
    print(f"validation games {len(keys)}: target made the last move {float((lt == 1).mean()):.3f}; record ends with a pass {float(ep.mean()):.3f}; by target colour (B / W) last-is-target {float((lt[tbk] == 1).mean()):.3f} / {float((lt[~tbk] == 1).mean()):.3f}")
    # ---------------------------------------------------------------- the test side, row-aligned with the identifier files
    t = dict(np.load(RV / f"m84_{tag}_test.npz", allow_pickle=True)); rows = {}
    for nm in ("public", "private"):
        d = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{nm}.csv", dtype=str)
        for _, r in d.iterrows():
            rows[r.question_id] = [(*last(r[f"sgf_{j}"], r[f"color_{j}"]), r[f"color_{j}"] == "B", int(HD.search(r[f"sgf_{j}"]).group(1))) for j in range(1, int(r.num_games) + 1)]
    seen = {}; T_ = []
    for qid in t["qid"].astype(str):
        j = seen.get(qid, 0); seen[qid] = j + 1; T_.append(rows[qid][j])
    T_ = np.array(T_); ok_c = float((T_[:, 3].astype(bool) == t["target_black"]).mean()); ok_h = float((T_[:, 4] == t["hd"]).mean()); print(f"test games {len(T_)}: alignment with the identifier rows - colour equal {ok_c:.4f}, handicap equal {ok_h:.4f}"); assert ok_c == 1.0 and ok_h == 1.0, "test rows are not aligned"
    np.savez(RV / "m169_last_test.npz", qid=t["qid"], last_is_target=T_[:, 0], ends_pass=T_[:, 1].astype(bool), plies=T_[:, 2])
    print(f"   target made the last move {float((T_[:, 0] == 1).mean()):.3f}; ends with a pass {float(T_[:, 1].mean()):.3f}; handicap > 1: {int((T_[:, 4] > 1).sum())} games")
    # ---------------------------------------------------------------- the proxy against a strong engine's final win rate
    rws, seen2 = [], set()
    for npz in sorted(glob.glob(str(ROOT / "results/ai_search_moves/fold1/shard*.npz"))):
        games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str}); zz = np.load(npz); df = pd.DataFrame({k: zz[k] for k in ("game", "t", "side", "win", "lead")}).sort_values(["game", "t"])
        for g, s in df.groupby("game"):
            key = games["game_key"].iloc[g]
            if key in seen2:
                continue
            seen2.add(key); tc = 0 if games["color"].iloc[g] == "B" else 1; mine = s["side"].to_numpy() == tc; w = s["win"].to_numpy(float); ld = s["lead"].to_numpy(float)
            rws.append((key, float(w[-1] if mine[-1] else 1 - w[-1]), float(ld[-1] if mine[-1] else -ld[-1]), float(mine[-1])))
    if rws:
        P = pd.DataFrame(rws, columns=["key", "wt", "lead", "last_eval_is_target"]); P["k2"] = P.key.astype(str); pk = {k: i for i, k in enumerate(keys)}; hit = P.k2.isin(pk)
        print(f"pilot games with a strong engine: {len(P)}; found among the validation games {int(hit.sum())} (key example {P.k2.iloc[0]})")
        if hit.any():
            Q_ = P[hit]; ii = np.array([pk[k] for k in Q_.k2]); l_ = lt[ii]; e_ = ep[ii]; w_ = Q_.wt.to_numpy() > 0.5
            print(f"   proxy 'target made the last move' vs final win rate > 0.5: agreement {float((l_[~e_] == w_[~e_]).mean()):.3f} on the {int((~e_).sum())} games not ending with a pass; on games ending with a pass {float((l_[e_] == w_[e_]).mean()) if e_.any() else float('nan'):.3f} ({int(e_.sum())}); share of clear final positions (|win rate - 0.5| > 0.4) {float((np.abs(Q_.wt.to_numpy() - 0.5) > 0.4).mean()):.3f}")
    # ---------------------------------------------------------------- the kernel by outcome
    ev = (z["hd"] == 0) & ~ep; tb = z["target_black"]; edges = {True: np.quantile(z["f_imp_m"][tb], QS), False: np.quantile(z["f_imp_m"][~tb], QS)}
    band = np.where(tb, np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"])); d = z["f_opp_lab"].astype(int) - z["y"].astype(int); y = z["y"].astype(int)
    print(f"\n{tag}: even games not ending with a pass {int(ev.sum())}; d = candidate label - target label, by band group and by who made the last move (W = the target, L = the opponent)")
    for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8)):
        line = []
        for o, on in ((1, "W"), (0, "L")):
            s = ev & (band >= lo) & (band <= hi) & (lt == o); dd = d[s]; line.append(f"{on}: n {int(s.sum())}, mean d {dd.mean():+.3f}, d<0 {float((dd < 0).mean()):.3f}, d=0 {float((dd == 0).mean()):.3f}, d>0 {float((dd > 0).mean()):.3f}")
        s1, s0 = (ev & (band >= lo) & (band <= hi) & (lt == o_) for o_ in (1, 0)); print(f"  {nm:22s} " + " | ".join(line) + f" | difference of the means {d[s1].mean() - d[s0].mean():+.3f}")
    top = ev & (band >= 4); print("  top bands (>= q95) by target level: difference of the mean d (W minus L): " + "  ".join(f"{c}: {d[top & (y == c) & (lt == 1)].mean() - d[top & (y == c) & (lt == 0)].mean():+.2f}" for c in range(10)))
    print("  top bands, share of games where the target made the last move, by d: " + "  ".join(f"d={x:+d}: {float((lt[top & (d == x)] == 1).mean()):.3f} (n {int((top & (d == x)).sum())})" for x in (-2, -1, 0, 1, 2)))


if __name__ == "__main__":
    main()
