#!/usr/bin/env python
"""Section 4M M161 (2026-11-15): identifiers whose fusion weights are fitted with the held-out games weighted by ACTIVITY.

m84_variant.stage_eval_multi fits one weight per source on the 3,244 held-out own games, every game counting the same (two games
of each of 1,622 players). Opponents appear in proportion to their activity, and the by-games networks are better for busy players
and no better for the others: here every held-out game counts in proportion to its player's number of training games
(ident_pool.fit_weights replaced by a weighted Newton fit of the same likelihood). Everything else as m143_ident.py.

  python -I logs/nano4/m161_ident_w.py --tag SP6aw:P,S6at --tag SPA6aw:P,S2y,S4ft,S4Lt,S6at
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m127_conf_family as m127  # noqa: E402
import m84_variant as mv  # noqa: E402

_q = np.load(mv.CACHE / "queries.npz"); NG = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(_q["players"].astype(str)).to_numpy().astype(np.float64)


def fit_weights_by_activity(Zs, truth, l2=1e-3, iters=60):
    """ident_pool.fit_weights with every query weighted by the number of training games of its true player."""
    Z = np.stack(Zs).astype(np.float32); J, Q, _ = Z.shape; w = np.ones(J); qi = np.arange(Q); sw = NG[np.asarray(truth)]; sw = sw / sw.sum()
    for _ in range(iters):
        logits = np.einsum("j,jqc->qc", w, Z); logits -= logits.max(1, keepdims=True); P = np.exp(logits); P /= P.sum(1, keepdims=True)
        EZ = np.einsum("jqc,qc->jq", Z, P); Zt = Z[:, qi, truth]; grad = ((EZ - Zt) * sw).sum(1) + 2 * l2 * w
        A = (Z * (P * sw[:, None])[None]).reshape(J, -1); H = A @ Z.reshape(J, -1).T - (EZ * sw) @ EZ.T + 2 * l2 * np.eye(J)
        step = np.linalg.solve(H, grad); w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--tag", action="append", required=True); ap.add_argument("--stat", default="m15"); a_ = ap.parse_args()
    mv.finalize = lambda: print("(finalize skipped: upload_next/README.md is not touched)", flush=True); ip.fit_weights = fit_weights_by_activity
    for spec in a_.tag:
        name, vars_ = spec.split(":"); src = []
        assert name not in mv.TAGS and name not in mv.MULTI, f"{name} is an existing tag"
        for v in vars_.split(","):
            n = int(np.load(mv.CACHE / f"var_{v}.npz", mmap_mode="r")["protos"].shape[0]); src.append((v, list(range(n))))
        mv.TAGS[name] = (src[-1][0], src[-1][1], None); mv.MULTI[name] = src
        print(f"== identifier {name} (fusion weights fitted by activity): block + " + " + ".join(f"{v} ({len(m)} members)" for v, m in src), flush=True)
        mv.stage_eval_multi(name); m127.fuse(name); m127.quick_write([name], a_.stat)


if __name__ == "__main__":
    main()
