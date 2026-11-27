"""M36 protocol check (compute node; after training, before any comparison is read): every -ra member's sidecar against its M33 (pc)
twin -- the M30 keys except n_r (64 -> 84), player_contrast 0.1, r_extra set, parameters + 20 x 32 (the R MLP's first layer), and the
validation probabilities on disk.
  python -I logs/nano4/m36_protocol_check.py
"""
import json
import sys
from pathlib import Path

KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",
        "opp_hsl", "opp_hsl_r", "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "drop", "mix", "swa", "pseudo",
        "hsl_dir", "vec_dir", "l1b_set", "l1b_dirs", "opp_dirs", "opp_hsl_dirs", "extra_dirs", "d_token", "n_scores", "player_contrast")
ROOT = Path.cwd()
bad = 0
for v in (1, 2, 3, 4):
    for arm in ("C", "T"):
        for s in (0, 1, 2):
            tag = f"{arm}-pc-f0-x40-oh-ra"
            fn, fc = ROOT / f"results/models/ct_cv_m36/val{v}_{tag}_s{s}.json", ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.json"
            npz = ROOT / f"results/trunk_seq/es/val{v}/{tag}_s{s}.npz"
            if not fn.exists() or not npz.exists():
                print(f"MISSING {fn.name if not fn.exists() else npz.name}"); bad = 1; continue
            n, c = json.loads(fn.read_text()), json.loads(fc.read_text())
            diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}
            ok = not diff and n.get("n_r") == c.get("n_r") + 20 and bool(n.get("r_extra")) and n.get("tag") == tag and n["n_params"] - c["n_params"] == 20 * 32
            print(f"ra val{v} {arm} s{s}: {'same' if not diff else diff}; n_r {c.get('n_r')} -> {n.get('n_r')}; params {c['n_params']} -> {n['n_params']}; "
                  f"best inner {n.get('best_inner'):.4f} vs pc {c.get('best_inner'):.4f}" + ("" if ok else "  <-- MISMATCH"))
            bad |= not ok
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
