"""Test-question linkage, structure check only (user 2026-11-05 「好」): do the official rank-test questions contain the same
player more than once? Analysis only -- nothing here feeds a submission. Using any linkage would first need the organisers'
answer (REPORT.md 2.7 "不得用於識別真實使用者", 2.9).

Similarity of two questions = competition 1's fusion4 (opening + style + local + HD-habit components of ~/BK_AICUP_2026-2's
aicup_pid, weights and cfg from its results/pid_fusion4.json): every question's games are one "candidate", the target's moves
play Black (a White question's colours are swapped), questions are compared within one colour, each component is standardised
per query over the other questions (diagonal excluded) and the weighted sum is the score.

Per pool: mutual top-1 rate (q's most similar question p has q as its most similar), the top-1 standardised score z1 and the
margin z1 - z2, the share of mutual pairs whose sizes differ. Calibration pools from folds 1-4 training players (HD <= 1 games
only, as the test; single-colour questions; sizes 5 / 10 / 15 / 20 a quarter each; classes equally represented):
  A_same   P/4 players x 4 questions (one per size), all four in the player's majority colour
  A_mixed  P/4 players x 4 questions, each question's colour drawn at random among those with enough games
  B        P players x 1 question
for P = 400 (one test file) and 800 (both files), seeds 0-2. Test pools: public (400), private (400), both (800).

PRE-SPECIFIED reading (written before the test pools are scored); primary statistic = mutual top-1 rate of the 800 pool:
  test <= max(B_800) + 0.02     -> no evidence of repeated players; the linkage line closes
  test >= min(A_mixed_800)      -> repeated players likely; report, and ask the organisers before any use
  otherwise                     -> inconclusive (report the 400 pools and the size structure)

  sbatch -J linkchk -t 02:00:00 scripts/nano4/run.sbatch python -I logs/nano4/test_linkage_check.py --out results/r_variants/test_linkage_check.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
PID = Path(os.environ.get("AICUP2_ROOT", Path.home() / "BK_AICUP_2026-2"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(PID / "src")]
import aicup_pid  # noqa: E402,F401  (also puts this repository's src / scripts on sys.path)
from aicup_pid.components import fit_from_parsed, fit_hd, fit_local, score_components  # noqa: E402
from aicup_pid.opening import parse  # noqa: E402
from aicup_pid.style import game_features  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.sgf import handicap  # noqa: E402
from aicup.submission import read_test  # noqa: E402

SIZES = (5, 10, 15, 20)
TEST_DIR = ROOT / "data/tests"
TRAIN_DIR = ROOT / "data/training"
SGF = {}  # game key -> sgf string, filled before the workers fork


def swap(moves):
    return [("W" if c == "B" else "B", x, y) for c, x, y in moves]


def similarity(qs, cfg, comps, weights):
    """(question ids in model order, [Q, Q] fused standardised similarity with -inf on the diagonal) of one colour group."""
    moves, feats = {}, {}
    for q in qs:
        for k in q["keys"]:
            m = parse(SGF[k])
            moves[k] = swap(m) if q["colour"] == "W" else m
            feats[k] = game_features(moves[k])
    cand = {q["qid"]: q["keys"] for q in qs}
    opening, style = fit_from_parsed(cand, moves, feats, cfg)
    local = fit_local(cand, moves, cfg) if "local" in comps else None
    hd = fit_hd(cand, SGF, cfg) if "hd" in comps else None
    order = opening.players
    S = score_components(list(comps), opening, style, [cand[p] for p in order], moves, feats, local=local, hd=hd, sgfs=SGF)
    Z = np.zeros((len(order), len(order)))
    for w, c in zip(weights, comps):
        M = np.asarray(S[c], dtype=float).copy()
        np.fill_diagonal(M, np.nan)
        mu, sd = np.nanmean(M, axis=1, keepdims=True), np.nanstd(M, axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        Z += w * np.nan_to_num((M - mu) / sd, nan=0.0)
    np.fill_diagonal(Z, -np.inf)
    return order, Z


def pool_stats(job):
    name, pool, cfg, comps, weights = job
    t0 = time.time()
    rows = []
    for col in ("B", "W"):
        qs = [q for q in pool if q["colour"] == col]
        if len(qs) < 3:
            continue
        order, Z = similarity(qs, cfg, comps, weights)
        info = {q["qid"]: q for q in qs}
        top = np.argsort(-Z, axis=1, kind="stable")[:, :2]
        for i, p in enumerate(order):
            t1, t2 = top[i]
            rows.append({"qid": p, "colour": col, "size": info[p]["size"], "player": info[p].get("player"), "z1": float(Z[i, t1]),
                         "z2": float(Z[i, t2]), "mutual": bool(top[t1, 0] == i), "top1_size": info[order[t1]]["size"],
                         "top1_player": info[order[t1]].get("player")})
    df = pd.DataFrame(rows)
    out = {"questions": int(len(df)), "colour_counts": df["colour"].value_counts().to_dict(), "mutual_top1": float(df["mutual"].mean()),
           "z1_median": float(df["z1"].median()), "z1_q90": float(df["z1"].quantile(0.9)), "margin_median": float((df["z1"] - df["z2"]).median()),
           "margin_q90": float((df["z1"] - df["z2"]).quantile(0.9)),
           "mutual_pairs_diff_size": float((df.loc[df["mutual"], "size"] != df.loc[df["mutual"], "top1_size"]).mean()) if df["mutual"].any() else float("nan"),
           "mutual_by_size": {str(s): float(df.loc[df["size"] == s, "mutual"].mean()) for s in SIZES if (df["size"] == s).any()},
           "wall_s": round(time.time() - t0, 1)}
    if df["player"].notna().all():
        same = df["player"] == df["top1_player"]
        sib = df.groupby(["colour", "player"])["qid"].transform("count") > 1
        out.update({"share_with_same_colour_sibling": float(sib.mean()), "top1_same_player": float(same.mean()),
                    "top1_same_player_given_sibling": float(same[sib].mean()) if sib.any() else float("nan"),
                    "mutual_precision": float(same[df["mutual"]].mean()) if df["mutual"].any() else float("nan")})
    return name, out


def load_training(folds):
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    keep = set(folds_df.loc[folds_df["fold"].isin(folds), "player_id"])
    by_player, rank_of = {}, {}
    for r in LEVELS:
        df = pd.read_csv(TRAIN_DIR / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str)
        df = df[df["player_id"].isin(keep)]
        for pid, gid, col, sgf in zip(df["player_id"], df["game_id"], df["color"], df["sgf_content"]):
            if (handicap(sgf) or 0) > 1:
                continue
            key = f"{r}:{gid}"
            SGF[key] = sgf
            by_player.setdefault(pid, {"B": [], "W": []})[col].append(key)
            rank_of[pid] = r
    return by_player, rank_of


def build_pool(design, P, seed, by_player, rank_of):
    rng = np.random.RandomState(seed)
    per_class = (P // 4 if design != "B" else P) // len(LEVELS)
    players = {r: sorted(p for p, rr in rank_of.items() if rr == r) for r in LEVELS}
    pool = []
    sizes_b = list(np.repeat(SIZES, P // 4))
    rng.shuffle(sizes_b)
    for r in LEVELS:
        cands = list(rng.permutation(players[r]))
        got = 0
        for pid in cands:
            if got == per_class:
                break
            g = by_player[pid]
            qs = []
            if design == "A_same":
                col = "B" if len(g["B"]) >= len(g["W"]) else "W"
                if len(g[col]) < sum(SIZES):
                    continue
                keys, off = list(rng.permutation(g[col])), 0
                for s in rng.permutation(SIZES):
                    qs.append((col, int(s), keys[off:off + s]))
                    off += s
            elif design == "A_mixed":
                avail = {c: list(rng.permutation(g[c])) for c in "BW"}
                for s in rng.permutation(SIZES):
                    cols = [c for c in "BW" if len(avail[c]) >= s]
                    if not cols:
                        qs = []
                        break
                    c = cols[rng.randint(len(cols))]
                    qs.append((c, int(s), avail[c][:s]))
                    avail[c] = avail[c][s:]
                if not qs:
                    continue
            else:
                s = int(sizes_b[len(pool)]) if len(pool) < len(sizes_b) else int(rng.choice(SIZES))
                cols = [c for c in "BW" if len(g[c]) >= s]
                if not cols:
                    continue
                c = cols[rng.randint(len(cols))]
                qs.append((c, s, list(rng.permutation(g[c]))[:s]))
            for j, (c, s, ks) in enumerate(qs):
                pool.append({"qid": f"{design}_{pid[:8]}_{j}", "colour": c, "size": s, "keys": ks, "player": pid})
            got += 1
    return pool


def test_pools():
    pools = {}
    for tag, fname in (("public", "rank_prediction_test_public.csv"), ("private", "rank_prediction_test_private.csv")):
        pool = []
        for q in read_test(TEST_DIR / fname):
            cols = {c for _, c in q["games"]}
            if len(cols) != 1:
                raise SystemExit(f"{q['question_id']}: colours {cols}")
            keys = []
            for i, (sgf, _) in enumerate(q["games"]):
                k = f"{q['question_id']}:{i}"
                SGF[k] = sgf
                keys.append(k)
            pool.append({"qid": q["question_id"], "colour": cols.pop(), "size": len(keys), "keys": keys})
        pools[f"test_{tag}"] = pool
    pools["test_both"] = pools["test_public"] + pools["test_private"]
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fusion", default=str(PID / "results/pid_fusion4.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    fus = json.loads(Path(a.fusion).read_text())
    comps, weights, cfg = tuple(fus["components"]), list(fus["weights"]), fus["cfg"]
    by_player, rank_of = load_training(a.folds)
    print(f"training: {len(by_player)} players of folds {a.folds}, {len(SGF)} HD <= 1 games ({time.time() - t0:.0f} s)", flush=True)
    jobs = []
    for P in (400, 800):
        for design in ("A_same", "A_mixed", "B"):
            for s in a.seeds:
                pool = build_pool(design, P, 1000 * P + s, by_player, rank_of)
                jobs.append((f"{design}_{P}_s{s}", pool, cfg, comps, weights))
    for name, pool in test_pools().items():
        jobs.append((name, pool, cfg, comps, weights))
    print("pools: " + ", ".join(f"{j[0]} {len(j[1])}" for j in jobs), flush=True)
    with mp.get_context("fork").Pool(min(a.workers, len(jobs))) as pool:
        res = dict(pool.map(pool_stats, jobs, chunksize=1))
    keys = ["mutual_top1", "z1_median", "margin_median", "mutual_pairs_diff_size", "top1_same_player_given_sibling", "mutual_precision"]
    for name in [j[0] for j in jobs]:
        r = res[name]
        print(f"{name:16s} q {r['questions']:4d}  " + "  ".join(f"{k} {r[k]:.3f}" for k in keys if k in r)
              + "  mutual by size " + " ".join(f"{k}:{v:.2f}" for k, v in r["mutual_by_size"].items()), flush=True)
    b800 = max(res[f"B_800_s{s}"]["mutual_top1"] for s in a.seeds)
    am800 = min(res[f"A_mixed_800_s{s}"]["mutual_top1"] for s in a.seeds)
    t = res["test_both"]["mutual_top1"]
    verdict = ("no evidence of repeated players" if t <= b800 + 0.02 else "repeated players likely" if t >= am800 else "inconclusive")
    rep = {"rule": {"test_both_mutual_top1": t, "max_B_800": b800, "min_A_mixed_800": am800, "verdict": verdict},
           "components": comps, "weights": weights, "cfg": cfg, "folds": a.folds, "seeds": a.seeds, "pools": res}
    print(f"PRE-SPECIFIED: test_both mutual top-1 {t:.3f} vs max B_800 {b800:.3f} (+0.02) / min A_mixed_800 {am800:.3f} -> {verdict}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
