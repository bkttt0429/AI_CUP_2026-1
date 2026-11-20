"""Training-loss trace of r5_ctm_minmax_pc vs r5_ctm (seed 0) from the CV attrs (GitHub PR #3)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import load_cv  # noqa: E402
for name in ("r5_ctm_s0", "r5_ctm_minmax_pc_s0", "r5_ctm_minmax_s0"):
    _, m = load_cv(ROOT / "results/ladder/cv" / f"{name}.npz")
    print("==", name, "keys:", sorted(m)[:40])
    tl = m.get("train_loss")
    if isinstance(tl, dict):
        for v, losses in tl.items():
            a = np.asarray(losses, dtype=float)
            q = [a[min(len(a) - 1, i)] for i in (0, 10, 50, 100, 200, len(a) - 1)] if len(a) else []
            print(f"  split {v}: steps {len(a)}  loss at 0/10/50/100/200/last: " + " ".join(f"{x:.4f}" for x in q))
    print("  best_epoch", m.get("best_epoch"), "inner_ce_init", m.get("inner_ce_init"), "inner_ce_best", m.get("inner_ce_best"))
