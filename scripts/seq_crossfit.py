#!/usr/bin/env python
"""Cross-fit a move-sequence encoder over the CV folds and write its per-game outputs in l1b_score.py's layout (issue #19).

For every fold k of --folds the encoder (aicup.seqmodel; --arch sum / conv / gru / ctm_seq) is trained on the other folds'
games (per-game rank cross-entropy, inner early stopping by players) and applied to fold k's games, so every fold's outputs
come from a model that never saw it. Fold k's outputs go to <out>/fold{k}/games_s0_p000.csv with sum_<rank> = base + Δ
(the target's summed L1b log p corrected by the encoder; the "sum" encoder reproduces the l1b_score sums exactly), a
moves_s0_p000.npz with base and delta per game, and meta_s0.json (arch, cfg, train_folds, inner CE, state sha256); the
encoder itself is saved as <out>/encoder_f{k}.pt. scripts/seq_check.py then compares the fused_opp features built on these
tables with the plain sums under the usual CV. Inputs per step: the target's 18 Human SL + 10 L1b log p and the opponent's
10 L1b log p (aicup.seqdata). Fold 0 is refused.

  sbatch ... torchrun is not needed: one GPU, e.g.
  python -I scripts/seq_crossfit.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --arch gru --out results/seq/gru
"""
import argparse
import json
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import part_paths  # noqa: E402
from aicup.seqdata import concat_sequences, fold_sequences  # noqa: E402
from aicup.seqmodel import ARCHS, SeqCfg, fit_seq_encoder, predict_seq, save_encoder, state_sha256  # noqa: E402
import l1b_cv  # noqa: E402


def write_fold(out, fold, seq, pred, meta_extra):
    fold_dir = Path(out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = part_paths(fold_dir, 0, 0)
    table = pd.DataFrame({"game_key": seq["game_key"], "player_id": seq["player_id"], "rank": seq["rank"], "color": seq["color"], "n_moves": seq["n_moves"]})
    for j, lv in enumerate(LEVELS):
        table[f"sum_{lv}"] = pred[:, j]
    np.savez_compressed(npz_path, base=seq["base"], delta=pred - seq["base"], n_moves=seq["n_moves"])
    table.to_csv(csv_path, index=False)
    meta = {"fold": fold, "shard": 0, "nshards": 1, "games": int(len(table)), "positions": int(seq["n_moves"].sum()), "side": "target",
            "finished": datetime.now().isoformat(timespec="seconds"), **meta_extra, **environment("cpu")}
    (fold_dir / "meta_s0.json").write_text(to_json(meta, indent=1) + "\n")
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="*", default=[], help="l1b_score.py --opponent directories; omit for target steps only")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--out", required=True)
    for f in fields(SeqCfg):
        if f.name == "arch":
            continue
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--only-folds", type=int, nargs="+", default=None,
                    help="fit and write only these folds of --folds (the training sets still come from --folds), so one arch's folds can run as separate jobs")
    ap.add_argument("--max-games", type=int, default=0, help="smoke tests: only the first N games of every fold")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not a CV fold here")
    targets = sorted(set(a.only_folds)) if a.only_folds else folds
    if not set(targets) <= set(folds):
        ap.error(f"--only-folds {targets} must be among --folds {folds}")
    out = Path(a.out)
    if any(part_paths(out / f"fold{k}", 0, 0)[0].exists() for k in targets) and not a.overwrite:
        print(f"{out} already has outputs; pass --overwrite", file=sys.stderr)
        return 2
    cfg = SeqCfg(**{f.name: (a.arch if f.name == "arch" else getattr(a, f.name)) for f in fields(SeqCfg)})
    try:
        where = l1b_cv.locate_folds(a.l1b, folds)
        where_o = l1b_cv.locate_folds(a.opp, folds) if a.opp else None
        seqs = {}
        for f in folds:
            t0 = time.time()
            seqs[f] = fold_sequences(a.hsl, where[f], where_o[f] if where_o else None, f)
            if a.max_games:
                keep = np.arange(min(a.max_games, len(seqs[f]["rank_idx"])))
                s = seqs[f]
                X = np.concatenate([s["X"][s["offsets"][g]:s["offsets"][g + 1]] for g in keep])
                seqs[f] = {**{k: s[k][keep] for k in ("game_key", "player_id", "rank", "rank_idx", "color", "n_moves", "n_steps", "base")},
                           "offsets": np.concatenate([[0], np.cumsum(s["n_steps"][keep])]), "X": X, "games": s["games"].iloc[keep].reset_index(drop=True)}
            print(f"fold {f}: {len(seqs[f]['rank_idx'])} games, {len(seqs[f]['X'])} steps ({time.time() - t0:.0f} s)", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    for k in targets:
        t0 = time.time()
        train = concat_sequences([seqs[f] for f in folds if f != k])
        print(f"== fold {k}: training {cfg.arch} on folds {[f for f in folds if f != k]} ({len(train['rank_idx'])} games)", flush=True)
        model, meta = fit_seq_encoder(train, cfg)
        pred = predict_seq(model, seqs[k], cfg)
        sha = state_sha256(model)
        enc_path = out / f"encoder_f{k}.pt"
        out.mkdir(parents=True, exist_ok=True)
        save_encoder(model, cfg, {**meta, "train_folds": [f for f in folds if f != k], "state_sha256": sha}, enc_path)
        extra = {"arch": cfg.arch, "train_folds": [f for f in folds if f != k], "cfg": meta["cfg"], "inner_ce_init": meta["inner_ce_init"],
                 "inner_ce_best": meta["inner_ce_best"], "best_epoch": meta["best_epoch"], "stopped_early": meta["stopped_early"],
                 "n_params": meta["n_params"], "train_wall_seconds": meta["wall_seconds"], "n_train_games": meta["n_train_games"],
                 "heads": {"path": str(enc_path), "state_sha256": sha, "train_folds": [f for f in folds if f != k]},
                 "mean_abs_delta": float(np.abs(pred - seqs[k]["base"]).mean())}
        write_fold(out, k, seqs[k], pred, extra)
        print(f"fold {k}: inner CE {meta['inner_ce_init']:.4f} -> {meta['inner_ce_best']:.4f} at epoch {meta['best_epoch']:.2f}, "
              f"mean |Δ| {extra['mean_abs_delta']:.4f}, {time.time() - t0:.0f} s", flush=True)
    # rebuilt from every fold's meta on disk, so the job that finishes last leaves the summary of all of them
    summary = {}
    for k in folds:
        path = out / f"fold{k}" / "meta_s0.json"
        if path.exists():
            m = json.loads(path.read_text())
            summary[str(k)] = {kk: m[kk] for kk in ("inner_ce_init", "inner_ce_best", "best_epoch", "n_params", "train_wall_seconds", "mean_abs_delta")}
    (out / "summary.json").write_text(to_json({"arch": cfg.arch, "cfg": meta["cfg"], "folds": summary, **environment("cpu")}, indent=1) + "\n")
    print(f"wrote {out}/fold*/ and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
