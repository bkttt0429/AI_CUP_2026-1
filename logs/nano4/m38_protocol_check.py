"""M38 protocol check (compute node; after training, before any comparison is read): every pc seed 6-29 member's sidecar against the
pc seed-0 member of the same fold and arm (ct9) -- the same training questions, held-out players, inner questions, steps per epoch,
folds and optimiser settings, player_contrast 0.1, its own seed, and the validation probabilities on disk.
  python -I logs/nano4/m38_protocol_check.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "val_fold", "inner_frac", "extra_games", "with_fold0",
        "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "opp_hsl", "opp_hsl_r", "hsl_dir", "vec_dir", "l1b_set", "n_params", "player_contrast")
bad, n_ok = 0, 0
for v in (1, 2, 3, 4):
    for arm in ("C", "T"):
        c = json.loads((ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s0.json").read_text())
        for s in range(6, 30):
            fn = ROOT / f"results/models/ct_cv_m38/val{v}_{arm}-pc-f0-x40-oh_s{s}.json"
            npz = ROOT / f"results/trunk_seq/es/val{v}/{arm}-pc-f0-x40-oh_s{s}.npz"
            if not fn.exists() or not npz.exists():
                print(f"MISSING val{v} {arm} s{s}"); bad = 1; continue
            n = json.loads(fn.read_text())
            diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}
            ok = not diff and n.get("seed") == s and n.get("tag") == f"{arm}-pc-f0-x40-oh"
            n_ok += ok
            if not ok:
                print(f"val{v} {arm} s{s}: {diff} seed {n.get('seed')} tag {n.get('tag')}  <-- MISMATCH")
            bad |= not ok
print(f"{n_ok} of 192 members match their ct9 twin's settings")
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
