"""M27 protocol check (run on a compute node). --stage r: the jp R caches against M22's: identical training and validation questions,
and which of the 64 R columns differ (only the Human SL ones should). --stage members: every jp member's sidecar against its M22 twin:
same training questions, held-out players, inner questions, steps per epoch, parameter counts; hsl_dir must be results/scores_hd_jp.
Exit 1 on any mismatch.
  python -I logs/nano4/m27_protocol_check.py --stage r
  python -I logs/nano4/m27_protocol_check.py --stage members
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); T = ROOT / "results/trunk_seq"


def stage_r():
    bad = 0
    for v in (1, 2, 3, 4):
        jp, cn, base = (json.loads((T / f).read_text()) for f in (f"r_val{v}_f0_x40_jp.json", f"r_val{v}_f0_x40.json", f"r_val{v}.json"))
        same_tr, same_va = jp["train"] == cn["train"], jp["val"] == base["val"]
        zj, zc = np.load(T / f"r_val{v}_f0_x40_jp.npz"), np.load(T / f"r_val{v}_f0_x40.npz")
        diff_tr = np.flatnonzero(np.any(np.abs(zj["X_tr"] - zc["X_tr"]) > 1e-6, axis=0)); diff_va = np.flatnonzero(np.any(np.abs(zj["X_va"] - zc["X_va"]) > 1e-6, axis=0))
        y_ok = np.array_equal(zj["y_tr"], zc["y_tr"]) and np.array_equal(zj["y_va"], zc["y_va"])
        print(f"val{v}: training questions identical {same_tr}, validation questions identical {same_va}, labels identical {y_ok}; R shape {zj['X_tr'].shape}; "
              f"columns that differ: train {diff_tr.tolist()}, val {diff_va.tolist()}")
        bad |= not (same_tr and same_va and y_ok)
    return bad


def stage_members():
    bad = 0; keys = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "n_params", "block_params", "train_folds", "seed", "val_fold", "inner_frac", "extra_games", "opp_hsl")
    for v in (1, 2, 3, 4):
        for arm in ("C", "T"):
            for s in (0, 1, 2):
                fj, fc = ROOT / f"results/models/ct_cv_jp/val{v}_{arm}-f0-x40-oh-jp_s{s}.json", ROOT / f"results/models/ct_cv/val{v}_{arm}-f0-x40-oh_s{s}.json"
                if not fj.exists():
                    print(f"MISSING {fj.name}"); bad = 1; continue
                j, c = json.loads(fj.read_text()), json.loads(fc.read_text())
                diff = {k: (j.get(k), c.get(k)) for k in keys if j.get(k) != c.get(k)}
                ok = not diff and j.get("hsl_dir") == "results/scores_hd_jp"
                print(f"val{v} {arm} s{s}: {'same' if not diff else diff}; hsl_dir {j.get('hsl_dir')}; best inner jp {j.get('best_inner'):.4f} vs ct7 {c.get('best_inner'):.4f}")
                bad |= not ok
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("r", "members"), required=True); a = ap.parse_args()
    bad = stage_r() if a.stage == "r" else stage_members()
    print("PROTOCOL CHECK", "FAIL" if bad else "PASS"); sys.exit(1 if bad else 0)
