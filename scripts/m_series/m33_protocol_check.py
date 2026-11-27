"""M33 protocol check (compute node; after training, before any comparison is read): every -pc member's sidecar against its ct7 (M22)
twin -- same training questions, held-out players, inner questions, steps per epoch, folds, seed, data and optimiser settings and score
directories; player_contrast 0.1; the same parameter count (the contrastive head is training-only); validation probabilities on disk.
  python -I scripts/m_series/m33_protocol_check.py
"""
import json
import sys
from pathlib import Path

KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",
        "opp_hsl", "opp_hsl_r", "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "drop", "mix", "swa", "pseudo",
        "hsl_dir", "vec_dir", "l1b_set", "l1b_dirs", "opp_dirs", "opp_hsl_dirs", "extra_dirs", "n_r", "d_token", "n_scores")

ROOT = Path.cwd()
bad = 0
for v in (1, 2, 3, 4):
    for arm in ("C", "T"):
        for s in (0, 1, 2):
            fn, fc = ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.json", ROOT / f"results/models/ct_cv/val{v}_{arm}-f0-x40-oh_s{s}.json"
            npz = ROOT / f"results/trunk_seq/es/val{v}/{arm}-pc-f0-x40-oh_s{s}.npz"
            if not fn.exists() or not npz.exists():
                print(f"MISSING {fn.name if not fn.exists() else npz.name}"); bad = 1; continue
            n, c = json.loads(fn.read_text()), json.loads(fc.read_text())
            diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}
            ok = not diff and n.get("player_contrast") == 0.1 and n.get("tag") == f"{arm}-pc-f0-x40-oh" and n["n_params"] == c["n_params"]
            print(f"pc val{v} {arm} s{s}: {'same' if not diff else diff}; params {c['n_params']} -> {n['n_params']}; best inner {n.get('best_inner'):.4f} vs ct7 {c.get('best_inner'):.4f}; "
                  f"stop epoch {n['best_step'] / n['steps_per_epoch']:.2f} vs {c['best_step'] / c['steps_per_epoch']:.2f}" + ("" if ok else "  <-- MISMATCH"))
            bad |= not ok
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
