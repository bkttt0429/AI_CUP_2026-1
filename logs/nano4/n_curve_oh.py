"""Section 4M M19 (2026-11-05): the learned ensembles' score at n = 20 / 40 / 80 games per question, fold 1 validation players with 80 scored
games (base 20 + x40 + x80 tables), the saved fold-1 checkpoints: the base family (val1_{C,T}_s0-2, trained on folds 2-4, 20 games) at
n = 20 / 40 / 80 and the oh family (val1_{C,T}-x40-oh_s0) at n = 20 / 40 (no x80 opponent HSL). The by-n fit of the CV ensembles
(0.677 / 0.735 / 0.774 at 5 / 10 / 20) extrapolates to 0.871; the fixed per-game decoder saturates at 0.63 (n_asymptote). Which is it?
Caveat: the pooling was trained on n <= 20, so the n = 40 / 80 numbers are a lower bound of a model trained at that n.
  python -I logs/nano4/n_curve_oh.py --out results/r_variants/n_curve_learned.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, evaluate  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def tables(f):
    hsl = [read_scored_games(d / f"fold{f}") for d in (ts.HSL, ts.EXTRAS[40]["hsl"], ts.EXTRAS[80]["hsl"])]
    l1b = [read_scored_games(ROOT / "results/l1b" / (ts.L1B[f] + sfx) / f"fold{f}") for sfx in ("", "_x40", "_x80")]
    opp = [read_scored_games(ROOT / "results/l1b" / (ts.OPP[f] + sfx) / f"fold{f}") for sfx in ("", "_x40", "_x80")]
    return hsl, l1b, opp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq")); ap.add_argument("--out", required=True)
    a = ap.parse_args(); f = a.fold; t0 = time.time(); device = "cuda"
    hsl, l1b, opp = tables(f)
    keys_by = [t.groupby("player_id")["game_key"].apply(list).to_dict() for t in hsl]
    players = [p for p in keys_by[0] if len(keys_by[0][p]) == 20 and len(keys_by[1].get(p, [])) == 20 and len(keys_by[2].get(p, [])) == 40]
    rank = hsl[0].drop_duplicates("player_id").set_index("player_id")["rank"]
    qs = []
    for p in players:
        k20 = sorted(keys_by[0][p]); k40 = k20 + sorted(keys_by[1][p]); k80 = k40 + sorted(keys_by[2][p])
        for n, k in ((20, k20), (40, k40), (80, k80)):
            qs.append({"player_id": p, "rank": rank[p], "n_games": n, "game_keys": k})
    Q = pd.DataFrame(qs)
    print(f"fold {f}: {len(players)} players with 80 games, {len(Q)} questions ({time.time() - t0:.0f} s)", flush=True)
    H, L, O = (pd.concat(x, ignore_index=True) for x in (hsl, l1b, opp))
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    X, y, n, _ = hd_check.features(H, L, O, hd_all[hd_all["game_key"].isin(set(H["game_key"]))], Q)
    R_raw = X["fused_opp_diff_hd"]; keys = Q["game_keys"].tolist()
    fams = {"base": ([str(ROOT / f"results/models/ct_check/val1_{arm}_s{s}.pt") for arm in ("C", "T") for s in range(3)], False, (20, 40, 80)),
            "oh": ([str(ROOT / f"results/models/ct_check/val1_{arm}-x40-oh_s0.pt") for arm in ("C", "T")], True, (20, 40))}
    rep = {"fold": f, "players": len(players)}
    for fam, (paths, oh, sizes) in fams.items():
        dirs = [(ts.VEC, ts.scores_path(a.cache, f, ts.VEC, 3, oh)), (ts.EXTRAS[40]["vec"], ts.scores_path(a.cache, f, ts.EXTRAS[40]["vec"], 3, oh))]
        if 80 in sizes:
            dirs.append((ts.EXTRAS[80]["vec"], ts.scores_path(a.cache, f, ts.EXTRAS[80]["vec"], 3, False)))
        X_raw, offsets = ts.raw_tokens([(d / f"fold{f}", sc) for d, sc in dirs])
        use = Q["n_games"].isin(sizes).to_numpy()
        kq = [k for k, u in zip(keys, use) if u]
        Ps = []
        for p in paths:
            model, stats, (r_mean, r_std), side = ts.load_checkpoint(p, device)
            Xd = torch.as_tensor(ts.standardise_tokens(X_raw.copy(), stats), device=device)
            Ps.append(ts.predict(model, ((R_raw[use] - r_mean) / r_std).astype(np.float32), kq, Xd, offsets, device))
            del Xd, model; torch.cuda.empty_cache()
        P = np.mean(Ps, axis=0)
        _, sc = evaluate(P, y[use], Q[use], sizes=sizes)
        E = P @ np.arange(len(LEVELS))
        out = {"models": len(paths), "by_n": {str(m): balanced_mean_over_n(sc, y[use], n[use], sizes=(m,)) for m in sizes},
               "single_by_n": {str(m): float(np.mean([balanced_mean_over_n(evaluate(Pi, y[use], Q[use], sizes=sizes)[1], y[use], n[use], sizes=(m,)) for Pi in Ps])) for m in sizes},
               "sd_E_minus_y_by_n": {str(m): float((E - y[use])[n[use] == m].std()) for m in sizes}}
        rep[fam] = out
        print(f"{fam} ({len(paths)} models): ensemble by n {out['by_n']}  single-model mean {out['single_by_n']}  sd(E-y) {out['sd_E_minus_y_by_n']}  ({time.time() - t0:.0f} s)", flush=True)
        del X_raw
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
