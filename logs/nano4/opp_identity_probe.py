"""Opponent identity probe (user 2026-11-06): are the opponents in the training games themselves training players, and can a single game
point at them? Per game, two 128-d style vectors from the trunk-vector parts (fixed random projections, both players' plies): the mean over
the TARGET's plies (own) and over the OPPONENT's plies (opp). Player centroids from own vectors (all folds, ~8k players).
  A  own leave-one-game-out retrieval among all players (single-game identification ceiling with these features);
  B  impostor null: for each own vector, the best cosine among the OTHER players (what a wrong match scores);
  C  opponents: best cosine among players other than the row's own player; the share above the null's 99 / 99.9 percentiles minus 1% / 0.1%
     = a lower bound on identifiable training-player opponents;
  D  label consistency of the confident opponent matches: |rank(matched player) - (y_target - signed_gap)| <= 1 vs the same for random players.
Zero duplicate games in the 1M rows means sampled-vs-sampled games were either deduplicated (opponents can be training players) or excluded
(they cannot); C and D tell which, if the features are strong enough to see it.
  python -I logs/nano4/opp_identity_probe.py --out results/r_variants/opp_identity_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.handicap import read_hd_table, signed_gap  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import completed_parts, part_paths  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def game_vectors(fold_dir):
    keys, pids, ranks, cols, own, opp = [], [], [], [], [], []
    for s, k in completed_parts(fold_dir):
        csv_path, npz_path = part_paths(fold_dir, s, k)
        tab = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
        with np.load(npz_path, allow_pickle=False) as z:
            X = np.hstack([z["g"], z["l"]]).astype(np.float32); row = z["game_row"]; tgt = z["target"]
        for g in range(len(tab)):
            m = row == g
            own.append(X[m & tgt].mean(0)); opp.append(X[m & ~tgt].mean(0))
        keys += tab.game_key.tolist(); pids += tab.player_id.tolist(); ranks += tab["rank"].tolist(); cols += tab.color.tolist()
    return pd.DataFrame({"game_key": keys, "player_id": pids, "rank": ranks, "color": cols}), np.array(own), np.array(opp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    T, O, P = [], [], []
    for f in (0, 1, 2, 3, 4):
        t, o, p = game_vectors(ts.VEC / f"fold{f}"); T.append(t); O.append(o); P.append(p); print(f"fold {f}: {len(t)} games", flush=True)
    T = pd.concat(T, ignore_index=True); own = np.vstack(O); opp = np.vstack(P)
    mu, sd = own.mean(0), own.std(0) + 1e-9; own = (own - mu) / sd; opp = (opp - mu) / sd
    players, inv = np.unique(T.player_id.to_numpy(), return_inverse=True); n_p = len(players)
    cnt = np.bincount(inv, minlength=n_p); S = np.zeros((n_p, own.shape[1])); np.add.at(S, inv, own)
    C = S / cnt[:, None]; Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    on = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-9); pn = opp / (np.linalg.norm(opp, axis=1, keepdims=True) + 1e-9)
    print(f"{len(T)} games, {n_p} players, {own.shape[1]} dims", flush=True)
    rep = {"games": int(len(T)), "players": int(n_p)}
    # A + B in chunks
    loo = (S[inv] - own) / np.maximum(cnt[inv] - 1, 1)[:, None]; loo_n = loo / (np.linalg.norm(loo, axis=1, keepdims=True) + 1e-9)
    own_true = (on * loo_n).sum(1); best_wrong = np.empty(len(T)); rank_true = np.empty(len(T), int)
    opp_best = np.empty(len(T)); opp_arg = np.empty(len(T), int)
    for i in range(0, len(T), 20000):
        j = min(len(T), i + 20000)
        sim = on[i:j] @ Cn.T; sim[np.arange(j - i), inv[i:j]] = -2  # exclude own (full) centroid
        best_wrong[i:j] = sim.max(1); rank_true[i:j] = (sim > own_true[i:j, None]).sum(1)
        simo = pn[i:j] @ Cn.T; simo[np.arange(j - i), inv[i:j]] = -2  # the opponent is never the row's own player
        opp_best[i:j] = simo.max(1); opp_arg[i:j] = simo.argmax(1)
    rep["A_own_loo"] = {"top1": float((rank_true == 0).mean()), "top5": float((rank_true < 5).mean()), "top20": float((rank_true < 20).mean()), "median_rank": float(np.median(rank_true) + 1)}
    q99, q999 = np.quantile(best_wrong, 0.99), np.quantile(best_wrong, 0.999)
    rep["B_impostor_null"] = {"q50": float(np.median(best_wrong)), "q99": float(q99), "q999": float(q999), "true_match_median": float(np.median(own_true)), "true_match_share_above_q99": float((own_true > q99).mean())}
    rep["C_opponents"] = {"median_best": float(np.median(opp_best)), "share_above_q99": float((opp_best > q99).mean()), "share_above_q999": float((opp_best > q999).mean()),
                          "excess_above_q99": float((opp_best > q99).mean() - 0.01), "excess_above_q999": float((opp_best > q999).mean() - 0.001)}
    print(f"A own LOO: top-1 {rep['A_own_loo']['top1']:.4f} top-5 {rep['A_own_loo']['top5']:.4f} top-20 {rep['A_own_loo']['top20']:.4f} median rank {rep['A_own_loo']['median_rank']:.0f}")
    print(f"B impostor null: median {rep['B_impostor_null']['q50']:.3f}, q99 {q99:.3f}, q99.9 {q999:.3f}; true-match median {rep['B_impostor_null']['true_match_median']:.3f}, share of true matches above q99 {rep['B_impostor_null']['true_match_share_above_q99']:.3f}")
    print(f"C opponents: median best {rep['C_opponents']['median_best']:.3f}; above q99 {rep['C_opponents']['share_above_q99']:.4f} (excess {rep['C_opponents']['excess_above_q99']:+.4f}); above q99.9 {rep['C_opponents']['share_above_q999']:.5f} (excess {rep['C_opponents']['excess_above_q999']:+.5f})")
    # D label consistency of confident opponent matches
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    y = T["rank"].map(RANK).to_numpy(); gap = signed_gap(T.game_key.map(hd).fillna(0).to_numpy(), T.color.to_numpy())
    prank = pd.Series(T["rank"].map(RANK).to_numpy()).groupby(inv).first().reindex(range(n_p)).to_numpy()
    implied_opp = y - gap  # the opponent's displayed class implied by the HD rule (dan steps = 1 class; kyu bands coarser)
    rng = np.random.default_rng(0)
    out = {}
    for name, mask in (("all", np.ones(len(T), bool)), ("above_q99", opp_best > q99), ("above_q999", opp_best > q999)):
        if mask.sum() == 0:
            continue
        agree = np.abs(prank[opp_arg[mask]] - implied_opp[mask]) <= 1
        rnd = np.abs(prank[rng.integers(0, n_p, mask.sum())] - implied_opp[mask]) <= 1
        out[name] = {"games": int(mask.sum()), "match_label_within_1": float(agree.mean()), "random_player_within_1": float(rnd.mean())}
        print(f"D {name:10s}: {mask.sum():7d} games; matched player's label within 1 of the HD-implied opponent class {agree.mean():.3f} vs random player {rnd.mean():.3f}")
    rep["D_label_consistency"] = out
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
