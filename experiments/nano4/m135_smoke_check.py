#!/usr/bin/env python
"""M135 smoke checks on a scored external root: the int8 trunk vectors against the fp16 ones, the score array's shape, and the
scores by external rank (the best Human SL profile and the L1b expected level should rise with the rank)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import trunk_seq_check as ts  # noqa: E402
from aicup.scoring import completed_parts, part_paths, read_scored_games  # noqa: E402

root = Path(sys.argv[1]); e = ts.ext_dirs(root)
# int8 against fp16, game by game (the two runs may cut the parts differently)
def load(vec):
    out = {}
    for s, k in completed_parts(vec):
        c, n = part_paths(vec, s, k); tab = pd.read_csv(c, dtype=str, keep_default_na=False)
        with np.load(n) as z:
            g, l = ts.part_gl(z, vec); row = z["game_row"]
        for i, key in enumerate(tab["game_key"]):
            m = row == i; out[key] = np.hstack([g[m], l[m]]).astype(np.float32)
    return out
if (root / "trunk_vec_fp16" / "fold9").exists():
    q, f = load(e["vec"]), load(root / "trunk_vec_fp16" / "fold9")
    assert set(q) == set(f), (len(q), len(f))
    st = np.load(e["vec"] / "quant8.npz"); sd = st["sd"]
    err = np.concatenate([np.abs(q[k] - f[k]) / sd for k in q]); clip = np.concatenate([(np.abs((f[k] - st["mu"]) / sd) > 6.35).mean(axis=0, keepdims=True) for k in q]).mean()
    print(f"trunk vectors: {len(q)} games, {len(err)} plies; int8 against fp16 in sd units: mean |diff| {err.mean():.4f}, 99.9% {np.quantile(err, 0.999):.4f}, max {err.max():.3f}; share of values beyond the int8 range {clip:.5f}")
sc = np.load(e["scores"]); print(f"score array {sc.shape}, finite {bool(np.isfinite(sc).all())}; zero rows {(np.abs(sc).sum(1) == 0).mean():.4f}")
ix = np.load(e["index"]); print(f"index: {len(ix['key'])} games, {int(ix['n'].sum())} plies, target share {ix['target'].mean():.3f}, plies/game {ix['n'].mean():.1f} (min {int(ix['n'].min())}, max {int(ix['n'].max())})")
if completed_parts(e["vec"]):
    X, off = ts.ext_tokens(root, "cpu", (np.zeros(ts.D_TOKEN - 4, np.float32), np.ones(ts.D_TOKEN - 4, np.float32)))
    print(f"tokens: {tuple(X.shape)}, {len(off)} games (every trunk part lined up with the score array); games without trunk vectors {len(set(ix['key'].astype(str)) - set(off))}")
STR = {"game_key": str, "player_id": str, "rank": str, "color": str}
hsl, l1b, opp = (pd.read_csv(e["games"][k], dtype=STR, keep_default_na=False) for k in ("hsl", "l1b", "opp"))
ohsl = read_scored_games(e["ohsl"]) if completed_parts(e["ohsl"]) else hsl.copy()
print(f"games: hsl {len(hsl)}, l1b {len(l1b)}, opp {len(opp)}, opp-hsl {len(ohsl)}; same keys {set(hsl.game_key) == set(l1b.game_key) == set(opp.game_key) == set(ohsl.game_key)}")
hc = [c for c in hsl.columns if c.startswith("sum_")]; lc = [c for c in l1b.columns if c.startswith("sum_")]
def post_mean(t, cols):
    z = t[cols].to_numpy(float) / t["n_moves"].to_numpy(float)[:, None]; z = z * 60.0; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return (p * np.arange(len(cols))).sum(1)
hsl["e"], l1b["e"], opp["e"], ohsl["e"] = post_mean(hsl, hc), post_mean(l1b, lc), post_mean(opp, lc), post_mean(ohsl, hc)
order = sorted(set(hsl["rank"]), key=ts.ext_rank_order)
print("per rank (mean over games; index into the 18 profiles / the 10 levels):")
for r in order:
    print(f"  {r:5s} games {int((hsl['rank'] == r).sum()):4d}  moves/game {hsl.loc[hsl['rank'] == r, 'n_moves'].mean():6.1f}  HSL own {hsl.loc[hsl['rank'] == r, 'e'].mean():5.2f}  HSL opp {ohsl.loc[ohsl['rank'] == r, 'e'].mean():5.2f}  L1b own {l1b.loc[l1b['rank'] == r, 'e'].mean():5.2f}  L1b opp {opp.loc[opp['rank'] == r, 'e'].mean():5.2f}")
comp = read_scored_games(ROOT / "results/scores_hd/fold1"); comp["e"] = post_mean(comp, hc)
print("competition fold 1 for comparison (HSL own by level): " + "  ".join(f"{r} {comp.loc[comp['rank'] == r, 'e'].mean():.2f}" for r in ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")))
X, y, qq, ranks = ts.ext_questions(root)
print(f"questions: {len(qq)} of {qq['player_id'].nunique()} units, R {X.shape}, finite {bool(np.isfinite(X).all())}, ranks {ranks}, sizes {sorted(set(qq['n_games']))[:3]}..{max(qq['n_games'])}")
