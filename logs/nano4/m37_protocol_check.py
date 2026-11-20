"""M37 protocol check (compute node; after training, before any comparison is read): every family member's sidecar against its pc twin
(same fold and seed: results/models/ct_cv_m33 for seeds 0-2, ct_cv_m37 for 3-5) -- the same training questions, held-out players,
inner questions, steps per epoch, folds, seed and optimiser settings, player_contrast 0.1, and the family's own flag.
  python -I logs/nano4/m37_protocol_check.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",
        "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "drop", "mix", "swa", "pseudo", "vec_dir", "player_contrast")
FAM = {"pc": ("{a}-pc-f0-x40-oh", "ct_cv_m37", {}), "ohr": ("{a}-pc-f0-x40-ohr", "ct_cv_m37", {"opp_hsl_r": True}),
       "uq": ("{a}-uq-pc-f0-x40-oh", "ct_cv_m37", {"uniqueness": True}), "jp": ("{a}-pc-f0-x40-oh-jp", "ct_cv_m37_jp", {"hsl_dir": None}),
       "h4": ("{a}-pc-f0-h4-x40-oh", "ct_cv_m37", {"l1b_set": 4})}
bad = 0
for fam, (pat, d, flags) in FAM.items():
    for v in (1, 2, 3, 4):
        for arm in ("C", "T"):
            for s in range(6):
                if fam == "pc" and s < 3:
                    continue
                tag = pat.format(a=arm)
                fn = ROOT / f"results/models/{d}/val{v}_{tag}_s{s}.json"
                twin = ROOT / (f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.json" if s < 3 else f"results/models/ct_cv_m37/val{v}_{arm}-pc-f0-x40-oh_s{s}.json")
                npz = ROOT / f"results/trunk_seq/es/val{v}/{tag}_s{s}.npz"
                if not fn.exists() or not npz.exists() or not twin.exists():
                    print(f"MISSING {fn.name if not fn.exists() else (npz.name if not npz.exists() else twin.name)}"); bad = 1; continue
                n, c = json.loads(fn.read_text()), json.loads(twin.read_text())
                diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k) and not (fam == "uq" and k == "train_questions")}
                ok = not diff and n.get("tag") == tag and n.get("player_contrast") == 0.1
                if fam == "ohr":
                    ok &= bool(n.get("opp_hsl_r"))
                if fam == "h4":
                    ok &= n.get("l1b_set") == 4
                if fam == "jp":
                    ok &= str(n.get("hsl_dir", "")).endswith("_jp")
                if not ok or s in (0, 3):
                    print(f"{fam} val{v} {arm} s{s}: {'same' if not diff else diff}; best inner {n.get('best_inner'):.4f} vs pc {c.get('best_inner'):.4f}" + ("" if ok else "  <-- MISMATCH"))
                bad |= not ok
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
