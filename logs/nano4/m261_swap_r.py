#!/usr/bin/env python
"""M261 (2026-11-20), report only, question-level pilot: ROLE SWAP - the opponents' side of the training games as new training
units. (design doc M261, registered before any reading; protocol of r_variants.py: 64 question features, MLP h32, seeds 0-2,
folds 1-4, the standard pilot validation questions)

In an even game the two players have the same listed rank at the time of the game. So the opponents' moves in owner P's even games
are games played at P's listed rank: with the two sides' features exchanged (the opponents' HSL / L1b scores as the "own" side, P's
as the "opponent" side) and P's label they form a pseudo player "the opponents of P". No rescoring: the opponent-side scores exist.
Arms:
  base     the present training questions (32 per player)
  swap     base + 32 questions per pseudo player; games whose opponent is identified (best candidate of results/m260/rows.npz, any
           margin) as a player of the validation fold are left out
  swap_nl  the same without that filter (how large is the leak)
  mix      base + the same number of mixed questions of REAL own-side games (n players of one level, one game each): more questions
           and several people per question, but no new data
  dbl      base with 64 questions per player

  python -I logs/nano4/m261_swap_r.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import RANK_TO_IDX  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

FOLDS, SEEDS, PER, SET = (1, 2, 3, 4), (0, 1, 2), 32, "fused_opp_diff_hd"; ARMS = ("base", "swap", "swap_nl", "mix", "dbl"); OUT = ROOT / "results/r_variants/m261_swap_r.json"


def mixed_questions(players, per_player, rng, sizes=range(5, 21)):
    """Per player `per_player` questions of its level: n different players of that level, one of their scored games each."""
    by_rank = {r: g for r, g in players.groupby("rank")}; rows = []
    for p in players.itertuples():
        pool = by_rank[p.rank]; keys = pool["game_keys"].to_numpy(); allowed = [s for s in sizes if s <= len(pool)]
        for j in range(per_player):
            size = int(rng.choice(allowed)); who = rng.choice(len(pool), size=size, replace=False)
            rows.append({"question_id": f"m_{p.player_id[:8]}_{j}", "rank": p.rank, "player_id": "M" + p.player_id, "n_games": size, "game_keys": [keys[i][int(rng.randint(len(keys[i])))] for i in who]})
    return pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])


def main():
    t0 = time.time(); folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str}); pfold = folds_df.set_index("player_id")["fold"].astype(int); hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    z = np.load(ROOT / "results/m260/rows.npz"); rkey = np.load(ROOT / "results/m260/rows_key.npy").astype(str); q = np.load(ROOT / "results/m84_cache/queries.npz"); players_all = q["players"].astype(str)
    cand_fold = pd.Series(pfold.reindex(players_all[z["top_i"][:, 0].astype(int)]).to_numpy(), index=rkey); assert cand_fold.index.is_unique and cand_fold.notna().all()
    print(f"row-level opponent identification: {len(rkey)} rows; best candidate by fold " + " ".join(f"{f}: {float((cand_fold == f).mean()):.3f}" for f in range(5)), flush=True)
    data = {}
    for f in FOLDS:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f); opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}"); oh = read_scored_games(ts.OPP_HSL / f"fold{f}")
        assert set(oh["game_key"]) == set(hsl["game_key"]) == set(opp["game_key"]); hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]; players = l1b_cv.players_from_scores(hsl, folds_df)
        even = set(hd.loc[hd["hd"].astype(float) == 0, "game_key"]); cf = cand_fold.reindex(hsl["game_key"]); assert cf.notna().all(), "scored games missing from the row table"
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "oh": oh, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players), "even": even, "cf": dict(zip(hsl["game_key"], cf.astype(int)))}
        print(f"fold {f}: {len(players)} players, {len(hsl)} games ({len(even)} even), {len(data[f]['val'])} validation questions ({time.time() - t0:.0f} s)", flush=True)
    scores, per_fold, ys, pids, ns, info = {k: [] for k in ARMS}, {k: [] for k in ARMS}, [], [], [], {}
    for v in FOLDS:
        train = [f for f in FOLDS if f != v]; cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        pl = pd.concat([data[f]["players"] for f in train], ignore_index=True); even = set().union(*[data[f]["even"] for f in train]); cf = {}
        for f in train:
            cf.update(data[f]["cf"])
        feat = lambda hsl_, l1b_, opp_, qs: hd_check.features(hsl_, l1b_, opp_, cat("hd"), qs)[0][SET]  # noqa: E731
        H, L, O, OH = cat("hsl"), cat("l1b"), cat("opp"), cat("oh")
        rng0 = np.random.RandomState(l1b_cv.TRAIN_SEED + v); q_base = pd.concat([sample_questions(data[f]["players"], PER, rng0) for f in train], ignore_index=True)  # exactly r_variants.py's training questions
        X_base = feat(H, L, O, q_base); y_of = lambda qs: qs["rank"].map(RANK_TO_IDX).astype(int).to_numpy(); y_base = y_of(q_base)  # noqa: E731

        def pseudo(filter_leak):
            rows = [{"rank": p.rank, "player_id": "S" + p.player_id, "game_keys": [k for k in p.game_keys if k in even and not (filter_leak and cf[k] == v)]} for p in pl.itertuples()]
            return pd.DataFrame([r for r in rows if len(r["game_keys"]) >= 5], columns=["rank", "player_id", "game_keys"])
        ps, psn = pseudo(True), pseudo(False); q_s = sample_questions(ps, PER, np.random.RandomState(261000 + v)); q_sn = sample_questions(psn, PER, np.random.RandomState(261100 + v))
        X_s, X_sn = feat(OH, O, L, q_s), feat(OH, O, L, q_sn)  # roles exchanged: the opponents' HSL and L1b scores as the own side, the owner's L1b scores as the opponent side
        q_m = mixed_questions(pl, PER, np.random.RandomState(261200 + v)); X_m = feat(H, L, O, q_m)
        q_d = pd.concat([sample_questions(data[f]["players"], PER, np.random.RandomState(261300 + v + 10 * f)) for f in train], ignore_index=True); X_d = feat(H, L, O, q_d)
        d = data[v]; X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"]); X_va = X_va[SET]
        sets = {"base": (X_base, y_base), "swap": (np.vstack([X_base, X_s]), np.concatenate([y_base, y_of(q_s)])), "swap_nl": (np.vstack([X_base, X_sn]), np.concatenate([y_base, y_of(q_sn)])),
                "mix": (np.vstack([X_base, X_m]), np.concatenate([y_base, y_of(q_m)])), "dbl": (np.vstack([X_base, X_d]), np.concatenate([y_base, y_of(q_d)]))}
        info[str(v)] = {"pseudo_players": int(len(ps)), "pseudo_players_nofilter": int(len(psn)), "games_per_pseudo": float(ps["game_keys"].map(len).mean()), "games_per_pseudo_nofilter": float(psn["game_keys"].map(len).mean()),
                        "feature_mean_abs_diff_swap_vs_base": float(np.abs(X_s.mean(0) - X_base.mean(0)).mean()), "feature_sd_ratio": float(np.median(X_s.std(0)[X_base.std(0) > 0] / X_base.std(0)[X_base.std(0) > 0]))}
        for name, (Xt, yt) in sets.items():
            P = np.mean([MLPModel(hidden=32, seed=s).fit(Xt, yt).predict_proba(X_va) for s in SEEDS], axis=0); res, sc = evaluate(P, y_va, d["val"]); scores[name].append(sc); per_fold[name].append(res["balanced_mean_over_n"])
        # the swapped questions scored by the base model (how well does the present model read the opponents' side?)
        Pb = np.mean([MLPModel(hidden=32, seed=s).fit(X_base, y_base).predict_proba(X_s) for s in SEEDS[:1]], axis=0); ys_ = y_of(q_s); lv = np.arange(10.0); info[str(v)]["base_on_swapped"] = {"exact": float((Pb.argmax(1) == ys_).mean()), "within1": float((np.abs(Pb.argmax(1) - ys_) <= 1).mean()), "mean_signed": float((Pb @ lv - ys_).mean())}
        ys.append(y_va); pids.append(d["val"]["player_id"].to_numpy(dtype=str)); ns.append(n_va)
        print(f"fold {v}: {len(q_base)} base questions, pseudo players {len(ps)} ({info[str(v)]['games_per_pseudo']:.1f} games each; without the filter {len(psn)}, {info[str(v)]['games_per_pseudo_nofilter']:.1f}); the base model on the swapped questions: exact {info[str(v)]['base_on_swapped']['exact']:.3f}, within one {info[str(v)]['base_on_swapped']['within1']:.3f}, mean signed error {info[str(v)]['base_on_swapped']['mean_signed']:+.3f} | "
              + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in ARMS) + f"  ({time.time() - t0:.0f} s)", flush=True)
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns); S = {k: np.concatenate(scores[k]) for k in ARMS}; rep = {"arms": {}, "pairs": {}, "info": info}
    for k in ARMS:
        rep["arms"][k] = {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": {str(m): balanced_mean_over_n(S[k], y, n, sizes=(m,)) for m in SIZES}}
        print(f"{k:8s} {rep['arms'][k]['score']:.4f}  folds {[round(x, 4) for x in per_fold[k]]}  by n " + " ".join(f"{m}: {rep['arms'][k]['by_n'][str(m)]:.4f}" for m in SIZES))
    for k, ref in (("swap", "base"), ("swap", "mix"), ("swap", "dbl"), ("swap_nl", "swap"), ("mix", "base"), ("dbl", "base")):
        dd = paired_diff(S[k], S[ref], y, pid, n, n=5000); pf = [a - b for a, b in zip(per_fold[k], per_fold[ref])]; rep["pairs"][f"{k} vs {ref}"] = {**dd, "per_fold_diff": pf}
        print(f"{k} vs {ref}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()))
    a, b_ = rep["pairs"]["swap vs base"], rep["pairs"]["swap vs mix"]; ok = a["diff"] >= 0.003 and a["ci95"][0] > 0 and sum(e > 0 for e in a["per_fold_diff"]) >= 3 and b_["ci95"][0] > 0
    verdict = "PASS: new data from the opponents' side (second step: the C / T member trainer, registered separately)" if ok else ("closed: the opponents' side is already used up by the opponent channel" if a["diff"] < 0.001 else "in between: reported, no second step")
    print(f"REGISTERED: swap vs base {a['diff']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}], swap vs mix {b_['diff']:+.4f} [{b_['ci95'][0]:+.4f}, {b_['ci95'][1]:+.4f}]  ->  {verdict}")
    rep["verdict"] = verdict; OUT.write_text(json.dumps(rep, indent=1, default=float)); print("written", OUT, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
