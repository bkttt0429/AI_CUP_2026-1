"""Section 4M M60 (2026-11-08): colour-pure questions in cross-validation. Every test question holds games of ONE colour only (400/400 per half),
whereas 98% of training players mix colours and the validation questions mix them too. Does ct9 behave differently on colour-pure questions,
and could that explain test-only effects (A|1D) or reveal new ones? Report only.
Fold v (validation players, their 20 scored games as in the standard validation questions): question sets
  mixed : disjoint chunks of n = 5 / 10 from the 20 games in their stored order (the standard construction; sanity check vs stored OOF)
  pureB : disjoint chunks of n = 5 / 10 from the player's black games only;   pureW : the same from white games only.
ct9's six members of fold v predict each set (R = fused_opp_diff_hd features of the set, tokens of the fold). Reported per set: class-balanced
official score, mean residual (expected level - truth) per true class, A|1D first-band precision (decided 1D with P(A)/P(1D) >= 1/2), and the
confusion matrix; pure vs mixed on the same players.
  python -I logs/nano4/m60_colour_pure_cv.py --val-fold 1
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n  # noqa: E402
from aicup.scoring import fold_players, player_games, read_scored_games  # noqa: E402
from m43_gate import decide, qscore  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
A, D1 = RANK["A"], RANK["1D"]


def chunks(pid, rank, keys, tag, sizes=(5, 10)):
    rows = []
    for n in sizes:
        for j in range(len(keys) // n):
            rows.append({"question_id": f"{tag}_{pid[:8]}_n{n}_{j}", "rank": rank, "player_id": pid, "n_games": n, "game_keys": keys[j * n:(j + 1) * n]})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); a = ap.parse_args()
    t0, v, device = time.time(), a.val_fold, torch.device("cuda")
    players = fold_players(player_games(), pd.read_csv(FOLDS_CSV), v)
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    colour = hd_all.drop_duplicates("game_key").set_index("game_key")["color"].to_dict()
    hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[v], v)
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[v] / f"fold{v}")
    scored = set(hsl["game_key"])
    sets = {"mixed": [], "pureB": [], "pureW": []}
    for p in players.itertuples():
        keys = [k for k in p.game_keys if k in scored]
        sets["mixed"] += chunks(p.player_id, p.rank, keys, "m")
        sets["pureB"] += chunks(p.player_id, p.rank, [k for k in keys if colour.get(k) == "B"], "b")
        sets["pureW"] += chunks(p.player_id, p.rank, [k for k in keys if colour.get(k) == "W"], "w")
    hd_f = hd_all[hd_all["game_key"].isin(scored)]
    X_raw, offsets = ts.raw_tokens([(ts.VEC / f"fold{v}", ts.scores_path(ROOT / "results/trunk_seq", v, ts.VEC, 3, True))])
    members = [ts.load_checkpoint(ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.pt", device) for arm in ("C", "T") for s in (0, 1, 2)]
    print(f"fold {v}: {len(players)} players; questions " + ", ".join(f"{k} {len(q)}" for k, q in sets.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    rep = {"val_fold": v, "sets": {}}
    preds = {}
    stats_cache = {}
    for name, rows in sets.items():
        q = pd.DataFrame(rows)
        Xf, y, n, _ = hd_check.features(hsl, l1b, opp, hd_f, q)
        R = Xf["fused_opp_diff_hd"].astype(np.float32)
        keys = q["game_keys"].tolist(); y = np.array([RANK[r] for r in q["rank"]]); n = q["n_games"].to_numpy()
        Ps = []
        for model, tstats, (r_mean, r_std), _ in members:
            h = id(tstats)
            if h not in stats_cache:
                stats_cache.clear(); torch.cuda.empty_cache()
                stats_cache[h] = torch.as_tensor(ts.standardise_tokens(X_raw.copy(), tstats), device=device)
            Ps.append(ts.predict(model, ((R - r_mean) / r_std).astype(np.float32), keys, stats_cache[h], offsets, device))
        P = np.mean(Ps, axis=0); d = decide(P); s = qscore(d, y)
        preds[name] = (q, P, y, n)
        np.savez(ROOT / f"results/trunk_seq/es/m60_{name}_val{v}.npz", proba=P.astype(np.float32), y=y, n=n, player_id=q["player_id"].to_numpy().astype(str),
                 question_id=q["question_id"].to_numpy().astype(str))
        E = (P * np.arange(10)).sum(1)
        band = (d == D1) & (P[:, A] / P[:, D1] >= 0.5)
        res = {LEVELS[c]: float(np.mean(E[y == c] - c)) if (y == c).any() else None for c in range(10)}
        conf = np.zeros((10, 10), int)
        for yy, dd in zip(y, d):
            conf[yy, dd] += 1
        rep["sets"][name] = {"questions": int(len(y)), "score": float(balanced_mean_over_n(s, y, n, sizes=(5, 10))), "residual_by_class": res,
                             "first_band": int(band.sum()), "first_band_precision_A": float((y[band] == A).mean()) if band.any() else None,
                             "mean_residual": float(np.mean(E - y)), "confusion": conf.tolist()}
        r = rep["sets"][name]
        print(f"{name:6s} {len(y):5d} q: score {r['score']:.4f}; mean residual {r['mean_residual']:+.3f}; first band {r['first_band']} q, precision A "
              f"{r['first_band_precision_A'] if r['first_band_precision_A'] is None else round(r['first_band_precision_A'], 3)}; residual by class "
              + " ".join(f"{k} {x:+.2f}" for k, x in res.items() if x is not None) + f" ({time.time() - t0:.0f} s)", flush=True)
    # sanity: the mixed set reproduces the stored OOF probabilities of the standard validation questions
    try:
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        stored = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        key_stored = {tuple(k): i for i, k in enumerate(m["game_keys"])}
        qm, Pm, _, _ = preds["mixed"]
        idx = [(i, key_stored.get(tuple(k))) for i, k in enumerate(qm["game_keys"])]
        idx = [(i, j) for i, j in idx if j is not None]
        diff = np.abs(Pm[[i for i, _ in idx]] - stored[[j for _, j in idx]]).max(1)
        rep["sanity"] = {"matched": len(idx), "max_abs_dp_median": float(np.median(diff)), "max_abs_dp_max": float(diff.max())}
        print(f"sanity: {len(idx)} mixed questions matched to stored OOF; max |dp| median {np.median(diff):.4f}, max {diff.max():.4f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"sanity check skipped: {e}", flush=True)
    Path(ROOT / f"results/trunk_seq/es/m60_colour_pure_val{v}.json").write_text(json.dumps(rep, indent=1)); print("written", flush=True)


if __name__ == "__main__":
    main()
