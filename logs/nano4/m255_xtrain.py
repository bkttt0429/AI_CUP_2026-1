#!/usr/bin/env python
"""M255b (2026-11-20), report only: are competition 1's pub2 / priv2 candidates also among competition 2's LABELLED training
players? (design doc M255b, registered before any result)

T1  exact comparison: every candidate game (2 x 600 x 10) against the 1,000,000 training rows (the whole SGF text, and the text
    with the HD tag removed).
T2  two-way game-by-game votes (m253_xvotes.py's vote / mutual, unchanged) between the validation players of fold v (their 20
    scored games, own side, the three M90 networks of fold v = var_S2e members, which never saw them) and the 1,200 candidates
    (results/m95 vectors of the same three networks). Pairs = mutual favourites, min share >= 0.5 / 0.7.
Output: results/r_variants/m255_xtrain.json (pairs with the training player's label and fold).

  python -I logs/nano4/m255_xtrain.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m253_xvotes as mx  # noqa: E402
import m81_tactic_id as m81  # noqa: E402

X95 = ROOT / "results/m95"; RV = ROOT / "results/r_variants"; CACHE = ROOT / "results/m84_cache"; SRC = Path.home() / "BK_AICUP_2026-2/data/tests_1111"; HD = re.compile(r"HD\[\d+\]")


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rep = {}
    # ---------------------------------------------------------------- T1
    rows = m81.load_rows(); cand = pd.concat([pd.read_csv(SRC / f"player_identification_candidates2_{h}.csv", dtype=str, keep_default_na=False) for h in ("public", "private")], ignore_index=True)
    full = dict(zip(rows.sgf_content, rows.player_id)); bare = dict(zip(rows.sgf_content.map(lambda s: HD.sub("", s)), rows.player_id)); hit_f = cand.sgf_content.map(full.get); hit_b = cand.sgf_content.map(lambda s: bare.get(HD.sub("", s)))
    rep["exact"] = {"candidate_games": int(len(cand)), "identical_text": int(hit_f.notna().sum()), "identical_without_hd": int(hit_b.notna().sum())}; ex_pairs = set(zip(cand.player_id[hit_b.notna()], hit_b[hit_b.notna()]))
    log(f"T1: {len(cand)} candidate games against {len(rows)} training rows: identical text {rep['exact']['identical_text']}, identical without the HD tag {rep['exact']['identical_without_hd']}; (candidate, training player) pairs with a shared game {len(ex_pairs)}"); del full, bare
    # ---------------------------------------------------------------- T2
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); dv = np.load(RV / "m82_val.npz"); pidv = dv["pid"].astype(str); foldv = dv["fold"].astype(int); yv = dv["y"].astype(int); assert np.array_equal(q["qkey"][:Q].astype(str), dv["key"].astype(str))
    QV = np.load(CACHE / "var_S2e.npz")["qvec"]; g = np.load(X95 / "games.npz"); grp = g["group"].astype(str); cr = np.flatnonzero(np.char.startswith(grp, "pub2:c:") | np.char.startswith(grp, "priv2:c:")); cn, uc = np.unique(grp[cr], return_inverse=True); nc = len(cn); kc = np.bincount(uc, minlength=nc); log(f"T2: {nc} candidates, {len(cr)} games")
    pairs = []
    for v in (1, 2, 3, 4):
        idx = np.flatnonzero(foldv == v); assert (q["net"][Q + idx] == v - 1).all(); names, uq = np.unique(pidv[idx], return_inverse=True); nq = len(names); kq = np.bincount(uq, minlength=nq); lab = np.zeros(nq, int); lab[uq] = yv[idx]
        vq = [mx.unit_norm(QV[m_, Q + idx].astype(np.float64)) for m_ in range(3)]; vc = [mx.unit_norm(np.load(X95 / f"xemb_{v}{f'_s{s}' if s else ''}.npz")["own"][cr].astype(np.float64)) for s in (0, 1, 2)]
        W1 = mx.vote(vq, vc, uq, uc, nq, nc); W2 = mx.vote(vc, vq, uc, uq, nc, nq); mu = mx.mutual(W1, W2, kq, kc, 0.5)
        pairs += [{"fold": v, "player": str(names[a]), "label": int(lab[a]), "candidate": str(cn[b_]).split(":")[-1], "share_q": s1, "share_c": s2, "shared_game": bool((str(cn[b_]).split(":")[-1], str(names[a])) in ex_pairs)} for a, b_, s1, s2 in mu]
        f1 = W1.argmax(1); top = W1[np.arange(nq), f1] / np.maximum(kq, 1); log(f"   fold {v}: {nq} players ({int(kq.min())}-{int(kq.max())} games); mutual pairs >= 0.5: {len(mu)}, >= 0.7: {sum(min(a[2], a[3]) >= 0.7 for a in mu)}; share of a player's games for the most voted candidate 50 / 90 / 99 %: " + " / ".join(f"{float(np.quantile(top, x)):.2f}" for x in (0.5, 0.9, 0.99)))
    P = pd.DataFrame(pairs); rep["pairs"] = pairs; n5 = len(P); n7 = int((P[["share_q", "share_c"]].min(1) >= 0.7).sum()) if n5 else 0
    log(f"T2 total: {n5} pairs at >= 0.5 ({n7} at >= 0.7) among {int(len(set(pidv)))} validation players; with a shared game (T1) {int(P.shared_game.sum()) if n5 else 0}; candidates paired in two or more folds {int((P.candidate.value_counts() >= 2).sum()) if n5 else 0}")
    if n5:
        print("   pairs by the training player's label: " + ", ".join(f"{k}: {v_}" for k, v_ in P.label.value_counts().sort_index().items()))
        multi = P[P.candidate.isin(P.candidate.value_counts()[lambda s: s >= 2].index)].sort_values("candidate")
        if len(multi):
            print("   candidates paired with several training players (labels): " + "; ".join(f"{c}: {list(d.label)}" for c, d in multi.groupby("candidate")))
    (RV / "m255_xtrain.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m255_xtrain.json")


if __name__ == "__main__":
    main()
