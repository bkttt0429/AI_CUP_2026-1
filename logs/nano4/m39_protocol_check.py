"""M39 / M40 protocol check (compute node; after training, before any comparison is read): each listed member's sidecar against its pc
twin (same fold and seed; results/models/ct_cv_m33 for seeds 0-2, ct_cv_m37 for 3-5): same training questions, held-out players, inner
questions, steps per epoch, folds and optimiser settings; the variant's own setting; R +20 columns and +640 parameters for -rh.
  python -I logs/nano4/m39_protocol_check.py --tags C-pc0.2-f0-x40-oh T-pc0.2-f0-x40-oh --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",
        "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "opp_hsl", "opp_hsl_r", "hsl_dir", "vec_dir", "l1b_set")
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
a = ap.parse_args()
bad = 0
for tag in a.tags:
    arm = tag[0]
    for v in (1, 2, 3, 4):
        for s in a.seeds:
            fn = ROOT / f"results/models/ct_cv_m39/val{v}_{tag}_s{s}.json"
            twin = ROOT / (f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.json" if s < 3 else f"results/models/ct_cv_m37/val{v}_{arm}-pc-f0-x40-oh_s{s}.json")
            npz = ROOT / f"results/trunk_seq/es/val{v}/{tag}_s{s}.npz"
            if not (fn.exists() and npz.exists() and twin.exists()):
                print(f"MISSING {tag} val{v} s{s}"); bad = 1; continue
            n, c = json.loads(fn.read_text()), json.loads(twin.read_text())
            diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}
            rh = tag.endswith("-rh")
            ok = not diff and n.get("tag") == tag and (n["n_params"] - c["n_params"] == (640 if rh else 0)) and (n.get("n_r") == c.get("n_r") + (20 if rh else 0))
            if not ok or (v == 1 and s == a.seeds[0]):
                print(f"{tag} val{v} s{s}: {'same' if not diff else diff}; pc {n.get('player_contrast')} temp {n.get('pc_temp')} views {n.get('pc_views')} n_r {n.get('n_r')}; "
                      f"best inner {n.get('best_inner'):.4f} vs pc {c.get('best_inner'):.4f}" + ("" if ok else "  <-- MISMATCH"))
            bad |= not ok
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
