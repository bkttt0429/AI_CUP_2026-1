#!/usr/bin/env python
"""Evaluate the official rank baseline on the project's fold-0 questions -> results/baseline_official_eval.json.

Question sets:
  fold0        the Human SL pipeline's evaluation questions: every fold-0 player with >= 5 games,
               up to 20 of them (scoring.fold_players), carved 1x20 / 2x10 / 4x5 (pilot.pilot_questions)
  pilot_fold0  the 200-player pilot set of METHOD_NOTES §2.4.1, for a direct comparison with its numbers
Inference is the notebook's: the official parser per game, GoRankResNet softmax in batches of
64, the arithmetic mean over a question's games; the mean is decided by argmax (the notebook)
and by metric.expected_score_decision. A game that does not parse is left out of its
questions, and a question with no game left gets LEVELS[0], as in the notebook. Per-game
probabilities go to --probs-out. Refuses to run if a fold-0 player is in the training shards.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.baseline import TUTORIAL_DIR, decide, game_features, load_tutorial, mean_probs, report  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.pilot import pilot_questions, select_players  # noqa: E402
from aicup.scoring import fold_players, load_games, player_games  # noqa: E402

GPU_BATCH = 64
SIZES = (5, 10, 20)
RULES = {"argmax_mean_softmax": "argmax", "expected_score_mean_softmax": "expected_score"}


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def tutorial_provenance(tutorial_dir=TUTORIAL_DIR):
    git = subprocess.run(["git", "-C", str(tutorial_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"dir": str(tutorial_dir), "commit": git.stdout.strip() or None,
            "sha256": {f: hashlib.sha256((Path(tutorial_dir) / f).read_bytes()).hexdigest()
                       for f in ("utils.py", "network.py", "rank-prediction.ipynb")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "baseline_official" / "best_model.pth"))
    ap.add_argument("--features", default=str(ROOT / "data" / "baseline_official"))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-players", type=int, default=0, help="first N players of each set (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_official_eval.json"))
    ap.add_argument("--probs-out", default=str(ROOT / "results" / "baseline_official" / "game_probs_fold0.csv"))
    a = ap.parse_args()
    feat, ckpt_dir = Path(a.features), Path(a.ckpt).parent
    t0 = time.time()

    folds = pd.read_csv(ROOT / "splits" / "player_folds.csv")
    fold0_ids = set(folds.loc[folds["fold"] == 0, "player_id"])
    leak = {f"fold0_players_in_{tag}": len(set(pd.read_csv(feat / f"{tag}_games.csv", usecols=["player_id"])["player_id"]) & fold0_ids)
            for tag in ("train", "val")}
    if any(leak.values()):
        raise SystemExit(f"fold-0 players in the baseline's training data: {leak}")

    sets = {"fold0": fold_players(player_games(), folds, 0),
            "pilot_fold0": select_players(pd.read_csv(ROOT / "splits" / "questions_fold0.csv"), 0)}
    if a.limit_players:
        sets = {name: players.head(a.limit_players) for name, players in sets.items()}
    qsets = {name: pilot_questions(players) for name, players in sets.items()}
    keys = sorted({k for qs in qsets.values() for ks in qs["game_keys"] for k in ks})
    data = load_games(keys)
    log(f"{len(keys)} games for " + ", ".join(f"{n}: {len(qs)} questions" for n, qs in qsets.items()))

    import torch

    pool = mp.get_context("fork").Pool(a.nproc)   # before CUDA initialises
    device = torch.device(a.device)
    net = load_tutorial()[1](num_classes=len(LEVELS))
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.eval().to(device)

    probs, failed, buf = {}, [], []

    def flush():
        x = torch.from_numpy(np.stack([f for _, f in buf])).float()
        with torch.no_grad():
            p = net(x.to(device)).softmax(1).cpu().numpy()
        for (k, _), row in zip(buf, p):
            probs[k] = row
        buf.clear()

    for i, (k, f) in enumerate(zip(keys, pool.imap(game_features, [data[k][0] for k in keys], chunksize=16))):
        if f is None:
            failed.append(k)
            continue
        buf.append((k, f))
        if len(buf) == GPU_BATCH:
            flush()
        if (i + 1) % 5000 == 0:
            log(f"  {i + 1}/{len(keys)} games")
    if buf:
        flush()
    pool.close()
    log(f"inference done: {len(probs)} games, {len(failed)} failed to parse")

    fold0_keys = [k for ks in qsets["fold0"]["game_keys"] for k in ks if k in probs]
    game_acc = float(np.mean([int(np.argmax(probs[k])) == RANK_TO_IDX[k.split(":")[0]] for k in set(fold0_keys)]))
    eval_sets = {}
    for name, qs in qsets.items():
        y = qs["rank"].map(RANK_TO_IDX).to_numpy()
        P = mean_probs(probs, qs["game_keys"])
        entry = {"players": int(qs["player_id"].nunique()), "games": len({k for ks in qs["game_keys"] for k in ks}),
                 "questions_by_n": {m: int((qs["n_games"] == m).sum()) for m in SIZES},
                 "questions_without_games": int(np.isnan(P).any(axis=1).sum())}
        for label, rule in RULES.items():
            entry[label] = report(decide(P, rule), y, qs, SIZES)
        eval_sets[name] = entry

    split = json.loads((feat / "split.json").read_text())
    run = json.loads((ckpt_dir / "run.json").read_text()) if (ckpt_dir / "run.json").exists() else None
    history = pd.read_csv(ckpt_dir / "history.csv").to_dict("records") if (ckpt_dir / "history.csv").exists() else None
    import webdataset as wds
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "what": "official AI CUP 2026 rank baseline (Tutorial rank-prediction.ipynb) retrained and evaluated "
                "under the project's player-grouped protocol (METHOD_NOTES §2.1)",
        "tutorial": tutorial_provenance(),
        "deviations_from_notebook": [
            "training pool: players of folds 1-4 only (fold 0 never seen) instead of all 1,000,000 games",
            "validation: whole players of folds 1-4, about 10,000 games per rank (~VAL_SIZE 100,000) instead of 100,000 random games",
            "evaluation: the project's fold-0 pseudo-questions (5/10/20 games), not the official public test",
            f"extraction processes: {split['nproc']} (notebook NPROC 8); only changes which games share a shard",
            "DataLoader pin_memory only when CUDA is available (on for every GPU run)",
            f"environment: torch {torch.__version__}, {run['device'] if run else a.device}, webdataset {wds.__version__} "
            "(notebook: torch 2.6.0+cu124, RTX 4090)",
        ],
        "decision_rules": {"argmax_mean_softmax": "the notebook: argmax of the mean of per-game softmax",
                           "expected_score_mean_softmax": "metric.expected_score_decision of the same mean"},
        "split": {k: split[k] for k in ("seed", "train_folds", "val_games_per_rank", "nproc", "per_shard", "limit",
                                        "train", "val", "val_games_by_rank")},
        "leakage_check": leak,
        "training": {"run": run, "history": history},
        "checkpoint": str(a.ckpt),
        "inference": {"device": a.device, "gpu_batch": GPU_BATCH, "games_scored": len(probs),
                      "minutes": round((time.time() - t0) / 60, 2)},
        "failed_games": len(failed),
        "failed_game_keys": failed[:50],
        "game_accuracy_fold0": game_acc,
        "eval_sets": eval_sets,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    scored = [k for k in keys if k in probs]
    table = pd.DataFrame(np.array([probs[k] for k in scored]).reshape(-1, len(LEVELS)), columns=[f"p_{lv}" for lv in LEVELS]).round(6)
    table.insert(0, "game_key", scored)
    Path(a.probs_out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.probs_out, index=False)

    log(f"fold-0 game-level accuracy {game_acc:.4f}")
    for name, entry in eval_sets.items():
        for label in RULES:
            r = entry[label]
            by_n = "  ".join(f"n={m} {r['by_n'][m]['balanced']:.4f} [{r['by_n'][m]['balanced_ci95'][0]:.3f}, "
                             f"{r['by_n'][m]['balanced_ci95'][1]:.3f}]" for m in SIZES)
            lo, hi = r["balanced_mean_over_n_ci95"]
            log(f"{name} {label}: balanced {by_n} | mean over n {r['balanced_mean_over_n']:.4f} [{lo:.3f}, {hi:.3f}] | "
                f"question mean {r['question_mean']:.4f}")
    log(f"wrote {a.out} and {a.probs_out}")


if __name__ == "__main__":
    main()
