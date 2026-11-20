"""Section 4M M40: the 20 soft-histogram columns (r_variants.opphist_features: HD-aligned opponent per-game tempered posteriors, own
per-game posteriors) of every training and validation question of the R cache r_val{v}_f0_x40, in its order, from the same
cross-fitted L1b tables as R (base 20 games plus games 21-40). Writes results/trunk_seq/r_extra_opphist_f0_x40_val{v}.npz (X_tr, X_va).
  python -I logs/nano4/opphist_r_extra.py --val-fold 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from r_variants import opphist_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); a = ap.parse_args()
    v = a.val_fold
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.json"))
    tr, va = pd.DataFrame(m["train"]), pd.DataFrame(m["val"])
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    dirs = ts.L1B_SETS[3]
    train_folds = ts.training_folds(v, 0, True)
    l1bs, opps = [], []
    for f in sorted(set(train_folds) | {v}):
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / dirs["l1b"][f], f)
        opp = read_scored_games(ROOT / "results/l1b" / dirs["opp"][f] / f"fold{f}")
        base = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))],
                "players": l1b_cv.players_from_scores(hsl, folds_df)}
        t = ts.extra_tables(f, base, folds_df, hd_all, dirs, 40) if f in train_folds else base
        l1bs.append(t["l1b"]); opps.append(t["opp"])
        print(f"fold {f}: {len(t['l1b'])} games", flush=True)
    l1b, opp = pd.concat(l1bs, ignore_index=True), pd.concat(opps, ignore_index=True)
    assert l1b["game_key"].is_unique and opp["game_key"].is_unique
    hd = hd_all[hd_all["game_key"].isin(set(l1b["game_key"]))]
    X_tr = opphist_features(l1b, opp, hd, tr).astype(np.float32)
    X_va = opphist_features(l1b, opp, hd, va).astype(np.float32)
    assert np.isfinite(X_tr).all() and np.isfinite(X_va).all()
    assert np.allclose(X_tr[:, :10].sum(1), 1, atol=1e-4) and np.allclose(X_tr[:, 10:].sum(1), 1, atol=1e-4)
    z = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")
    assert len(X_tr) == len(z["X_tr"]) and len(X_va) == len(z["X_va"]), "row counts differ from the R cache"
    out = ROOT / f"results/trunk_seq/r_extra_opphist_f0_x40_val{v}.npz"
    np.savez(out, X_tr=X_tr, X_va=X_va)
    print(f"val{v}: X_tr {X_tr.shape}, X_va {X_va.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
