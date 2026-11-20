#!/usr/bin/env python
"""Komi sensitivity (architecture v2 section 6.1 item 1; user-approved scope 2026-11-03). Report only.

The same games are scored twice: with whiteKomi 7.5 (the pipeline's setting for every game) and with the dataset's komi
(HD0 6.5, HD >= 1 none), everything else equal (fixed replay with handicap stones, Human SL 18 profiles, L1b heads A and
B on the target's and the opponent's moves). The difference is followed down the pipeline: per move, per game sum, per
question feature, and at the submission document's probabilities and expected-score decisions. Questions are n=5 pilot
questions of folds 1-4 grouped by handicap content (A: all games HD0; B: some HD1, no HD>=2; C: some HD>=2). Fold 0 refused.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from aicup.fused import fused_opp_diff_features, fused_predict, load_fused_model  # noqa: E402
from aicup.handicap import question_hd_features_from_games, read_hd_table  # noqa: E402
from aicup.humansl import PROFILES, load_humansl, meta_rows, score_positions as hsl_score  # noqa: E402
from aicup.l1b import load_heads, score_positions as l1b_score  # noqa: E402
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402


def decide(p):
    d = expected_score_decision(p)
    return LEVELS.index(d) if isinstance(d, str) else int(d)

from aicup.pilot import start_state  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
import l1b_cv  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
_FEATURES = None


def init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def extract(task):
    """(key, setting, target positions, opponent positions) under the setting's komi; the pool's unit of work."""
    key, setting, sgf, color, komi = task
    rules = dict(GameState.RULES_CHINESE, whiteKomi=komi)
    other = "W" if color == "B" else "B"
    return key, setting, positions(sgf, color, _FEATURES, rules), positions(sgf, other, _FEATURES, rules)


def positions(sgf, target_color, features, rules):
    """Inputs before every move of target_color under `rules` (pilot.game_positions with a rules override)."""
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, rules)
    P = features.pass_pos
    bins, globs, legal, target, colors = [], [], [], [], []
    for t, (c, x, y) in enumerate(moves):
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if c == target_color:
            board = gs.board
            b, g = gs.get_input_features(features)
            mask = np.zeros(P + 1, dtype=bool)
            for i in range(P):
                mask[i] = board.would_be_legal(pla, features.tensor_pos_to_loc(i, board))
            mask[P] = True
            bins.append(b[0].astype(np.uint8))
            globs.append(g[0].astype(np.float32))
            legal.append(mask)
            target.append(P if x is None else features.loc_to_tensor_pos(loc, board))
            colors.append(0 if target_color == "B" else 1)
        gs.play(pla, loc)
    if not target:
        return None
    return {"bins": np.stack(bins), "globs": np.stack(globs), "legal": np.stack(legal), "target": np.asarray(target), "colors": np.asarray(colors)}


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max())} if x.size else {"n": 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", default=["results/l1b/scores3_hd", "results/l1b/scores3_B_hd"])
    ap.add_argument("--hsl", default="results/scores_hd")
    ap.add_argument("--hd-table", default="results/data/hd_table.csv")
    ap.add_argument("--heads", nargs=2, default=["results/l1b/heads3_A.pt", "results/l1b/heads3_B.pt"], metavar=("A", "B"))
    ap.add_argument("--docs", nargs="+", default=["results/models/rank_model_fused_opp_diff_r.json"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-group", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not used here")
    t0 = time.time()
    hd_all = read_hd_table(a.hd_table).drop_duplicates("game_key").set_index("game_key")["hd"]
    where = l1b_cv.locate_folds(a.l1b, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    qs = []
    for f in folds:
        hsl, _ = l1b_cv.load_fold(a.hsl, where[f], f)
        q = pilot_questions(players_from_scores(hsl, folds_df))
        q = q[q["n_games"] == 5].copy()
        q["fold"] = f
        qs.append(q)
    q = pd.concat(qs, ignore_index=True)
    hd_max = np.array([np.nanmax([hd_all.get(k, np.nan) for k in ks]) for ks in q["game_keys"]])
    assert np.isfinite(hd_max).all(), "every v1.1 game carries HD[]"
    q["group"] = np.where(hd_max >= 2, "C_stones", np.where(hd_max >= 1, "B_sente", "A_even"))
    rng = np.random.RandomState(a.seed)
    picked = pd.concat([q[q["group"] == g].sample(n=min(a.per_group, (q["group"] == g).sum()), random_state=rng) for g in ("A_even", "B_sente", "C_stones")], ignore_index=True)
    keys = sorted({k for ks in picked["game_keys"] for k in ks})
    games = load_games(keys)
    print(f"{len(picked)} questions, {len(keys)} games ({time.time() - t0:.0f} s); groups {picked['group'].value_counts().to_dict()}", flush=True)

    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))  # before CUDA, as score_folds.py
    model, features = load_humansl(str(CKPT), "cuda")
    heads = {m: load_heads(p, model, "cuda")[0] for m, p in zip("AB", a.heads)}  # (heads, sidecar meta, warnings)
    rows = meta_rows([(p, p) for p in PROFILES])
    docs = {Path(p).stem: load_fused_model(p) for p in a.docs}
    komi_of = {"k75": lambda hd: 7.5, "kdata": lambda hd: 6.5 if hd == 0 else 0.0}
    per_game = {k: {"hd": float(hd_all[k])} for k in keys}
    tasks = [(k, s, games[k][0], games[k][1], komi_of[s](per_game[k]["hd"])) for k in keys for s in ("k75", "kdata")]
    n_pos, done = 0, 0
    for key, s, tgt, opp in pool.imap_unordered(extract, tasks, chunksize=4):
        done += 1
        if tgt is None or opp is None:
            per_game[key][s] = None
            continue
        h = hsl_score(model, tgt["bins"], tgt["globs"], tgt["legal"], tgt["target"], tgt["colors"], rows)
        rec = {"mv_hsl": h, "hsl": h.sum(axis=0), "n": len(tgt["target"]), "l1b": {}, "opp": {}, "opp_n": len(opp["target"]), "mv_l1b": {}}
        for m, hm in heads.items():
            lt = l1b_score(model, hm, tgt["bins"], tgt["globs"], tgt["legal"], tgt["target"], tgt["colors"])
            lo = l1b_score(model, hm, opp["bins"], opp["globs"], opp["legal"], opp["target"], opp["colors"])
            rec["mv_l1b"][m] = lt
            rec["l1b"][m] = lt.sum(axis=0)
            rec["opp"][m] = lo.sum(axis=0)
        per_game[key][s] = rec
        n_pos += len(tgt["target"]) + len(opp["target"])
        if done % 100 == 0:
            print(f"  {done}/{len(tasks)} game-settings, {n_pos} positions ({time.time() - t0:.0f} s)", flush=True)
    pool.close()
    pool.join()
    settings = komi_of

    out = {"folds": folds, "per_group": a.per_group, "seed": a.seed, "settings": {"k75": "whiteKomi 7.5 for every game (pipeline)", "kdata": "HD0 6.5, HD>=1 0.0 (dataset convention)"},
           "docs": {k: {"feature_set": d[0]["feature_set"], "file_sha256": d[0]["file_sha256"]} for k, d in docs.items()}, "groups": {}}
    for g in ("A_even", "B_sente", "C_stones"):
        sub = picked[picked["group"] == g]
        d_move_hsl, d_move_rel_hsl, d_move_l1b, d_sum_hsl, d_sum_l1b, d_sum_opp = [], [], [], [], [], []
        feat_a, feat_b, prob = [], [], {k: {"maxdp": [], "changed": 0, "dec_k75": [], "dec_kdata": []} for k in docs}
        n_games = 0
        for row in sub.itertuples():
            recs = [per_game[k] for k in row.game_keys]
            if any(r["k75"] is None or r["kdata"] is None for r in recs):
                continue
            n_games += len(recs)
            for r in recs:
                a_, b_ = r["k75"], r["kdata"]
                dm = b_["mv_hsl"] - a_["mv_hsl"]
                d_move_hsl.append(np.abs(dm).ravel())
                d_move_rel_hsl.append(np.abs(dm - dm.mean(axis=1, keepdims=True)).ravel())
                d_move_l1b.append(np.abs(b_["mv_l1b"]["A"] - a_["mv_l1b"]["A"]).ravel())
                d_sum_hsl.append(b_["hsl"] - a_["hsl"])
                d_sum_l1b.append(b_["l1b"]["A"] - a_["l1b"]["A"])
                d_sum_opp.append(b_["opp"]["A"] - a_["opp"]["A"])
            feats, probs = {}, {}
            for s in ("k75", "kdata"):
                hs = np.stack([r[s]["hsl"] for r in recs]); nm = np.array([r[s]["n"] for r in recs], dtype=float)
                l1 = np.stack([[r[s]["l1b"]["A"], r[s]["l1b"]["B"]] for r in recs]); op = np.stack([[r[s]["opp"]["A"], r[s]["opp"]["B"]] for r in recs])
                on = np.array([r[s]["opp_n"] for r in recs], dtype=float)
                feats[s] = fused_opp_diff_features(hs, l1[:, 0, :], nm, op[:, 0, :], on)
                probs[s] = {}
                for k, (doc, mlp) in docs.items():
                    order = [0 if "A" in m["name"].split("_")[-1] else 1 for m in doc["l1b_models"]]
                    hdf = question_hd_features_from_games([games[kk] for kk in row.game_keys]) if doc["feature_set"] == "fused_opp_diff_hd" else None
                    probs[s][k] = fused_predict(mlp, hs, l1, nm, order, op, on, feature_set=doc["feature_set"], hd_feats=hdf)
            feat_a.append(feats["k75"]); feat_b.append(feats["kdata"])
            for k in docs:
                pa, pb = probs["k75"][k], probs["kdata"][k]
                da, db = decide(pa), decide(pb)
                prob[k]["maxdp"].append(float(np.abs(pb - pa).max())); prob[k]["changed"] += int(da != db)
                prob[k]["dec_k75"].append(da); prob[k]["dec_kdata"].append(db)
        FA, FB = np.stack(feat_a), np.stack(feat_b)
        sd = FA.std(axis=0) + 1e-12
        rel = np.abs(FB - FA) / sd
        out["groups"][g] = {
            "questions": int(len(feat_a)), "games": n_games,
            "per_move_abs_dlogp": {"hsl_all_profiles": stats(np.concatenate(d_move_hsl)), "hsl_profile_centred": stats(np.concatenate(d_move_rel_hsl)), "l1b_A_heads": stats(np.concatenate(d_move_l1b))},
            "per_game_dsum": {"hsl_signed_mean_by_profile": dict(zip(PROFILES, np.mean(d_sum_hsl, axis=0).round(4).tolist())), "hsl_abs": stats(np.abs(d_sum_hsl)), "l1b_A_abs": stats(np.abs(d_sum_l1b)), "opp_A_abs": stats(np.abs(d_sum_opp))},
            "question_feature_shift_in_sd": {"median_over_features": float(np.median(rel)), "p95_over_features": float(np.percentile(rel, 95)), "worst_feature_index": int(np.median(rel, axis=0).argmax()), "worst_feature_median": float(np.median(rel, axis=0).max())},
            "decisions": {k: {"max_abs_dp": stats(v["maxdp"]), "changed": v["changed"], "changed_share": v["changed"] / max(len(v["maxdp"]), 1)} for k, v in prob.items()},
        }
        print(f"{g}: {len(feat_a)} q; per-move |dlogp| hsl mean {out['groups'][g]['per_move_abs_dlogp']['hsl_all_profiles']['mean']:.4f} (centred {out['groups'][g]['per_move_abs_dlogp']['hsl_profile_centred']['mean']:.4f}) l1b {out['groups'][g]['per_move_abs_dlogp']['l1b_A_heads']['mean']:.4f}; "
              f"feature shift median {out['groups'][g]['question_feature_shift_in_sd']['median_over_features']:.3f} sd; decisions changed " + ", ".join(f"{k} {v['changed']}/{len(v['maxdp'])}" for k, v in prob.items()), flush=True)
    out["positions_scored"] = n_pos
    out["seconds"] = time.time() - t0
    out.update(environment("cuda"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
