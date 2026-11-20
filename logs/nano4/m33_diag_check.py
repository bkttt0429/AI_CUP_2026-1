"""M33 diagnostics protocol check (compute node; after training, before any comparison is read): every -pb and -rr member's sidecar
against its ct7 (M22) twin -- the same keys as the M30 check, the right pair_batches / player_contrast / tag, the same parameter count,
and the validation probabilities on disk.
  python -I logs/nano4/m33_diag_check.py
"""
import json
import sys
from pathlib import Path

KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",
        "opp_hsl", "opp_hsl_r", "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "drop", "mix", "swa", "pseudo",
        "hsl_dir", "vec_dir", "l1b_set", "l1b_dirs", "opp_dirs", "opp_hsl_dirs", "extra_dirs", "n_r", "d_token", "n_scores")
ROOT = Path.cwd()
ARMS = {"pb": ("{a}-pb-f0-x40-oh", True), "rr": ("{a}-f0-x40-oh-rr", False)}
bad = 0
for name, (pat, pair) in ARMS.items():
    for v in (1, 2, 3, 4):
        for arm in ("C", "T"):
            for s in (0, 1, 2):
                tag = pat.format(a=arm)
                fn, fc = ROOT / f"results/models/ct_cv_m33d/val{v}_{tag}_s{s}.json", ROOT / f"results/models/ct_cv/val{v}_{arm}-f0-x40-oh_s{s}.json"
                npz = ROOT / f"results/trunk_seq/es/val{v}/{tag}_s{s}.npz"
                if not fn.exists() or not npz.exists():
                    print(f"MISSING {fn.name if not fn.exists() else npz.name}"); bad = 1; continue
                n, c = json.loads(fn.read_text()), json.loads(fc.read_text())
                diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}
                ok = not diff and n.get("pair_batches") is pair and n.get("player_contrast") == 0.0 and n.get("tag") == tag and n["n_params"] == c["n_params"]
                print(f"{name} val{v} {arm} s{s}: {'same' if not diff else diff}; best inner {n.get('best_inner'):.4f} vs ct7 {c.get('best_inner'):.4f}; "
                      f"stop epoch {n['best_step'] / n['steps_per_epoch']:.2f} vs {c['best_step'] / c['steps_per_epoch']:.2f}" + ("" if ok else "  <-- MISMATCH"))
                bad |= not ok
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
