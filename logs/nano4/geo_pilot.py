"""Architecture v2 section 4L.1 part A: response geometry (hypothesis H1) from the stored trunk parts' move positions, no network.

Per ply (both players): Chebyshev distance to the opponent's last move (d_opp) and to the player's own last move (d_own); local =
d_opp <= 2, tenuki = d_opp >= 6, low = distance to the edge <= 1; phases ply < 50 / 50-149 / >= 150. Per game the target's and
the opponent's rates and means; then the step-0 signal report (pilot_stats) with own / opponent pmi from the OOF L1b tables.

  sbatch -c 12 ... scripts/nano4/run.sbatch python -I logs/nano4/geo_pilot.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs" / "nano4")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import completed_parts, part_paths, read_scored_games  # noqa: E402
import pilot_stats as st  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

OUT = ROOT / "results" / "pilot"
PASS = 361
LOCAL, TENUKI, LOW = 2, 6, 1
PHASES = ((0, 50), (50, 150), (150, 10**6))
SIDES = ("tgt", "opp")


def game_features(pos, t, target):
    """The per-side statistics of one game from its ply positions (pos 361 = pass), ply indices t and target flags."""
    n = len(pos)
    x, y = pos // 19, pos % 19
    is_pass = pos == PASS
    d_opp = np.full(n, np.nan)
    d_own = np.full(n, np.nan)
    if n > 1:
        ok = ~is_pass[1:] & ~is_pass[:-1]
        d_opp[1:][ok] = np.maximum(np.abs(x[1:] - x[:-1]), np.abs(y[1:] - y[:-1]))[ok]
    if n > 2:
        ok = ~is_pass[2:] & ~is_pass[:-2]
        d_own[2:][ok] = np.maximum(np.abs(x[2:] - x[:-2]), np.abs(y[2:] - y[:-2]))[ok]
    edge = np.minimum(np.minimum(x, 18 - x), np.minimum(y, 18 - y)).astype(float)
    edge[is_pass] = np.nan
    out = {}
    for side, mask in (("tgt", target), ("opp", ~target)):
        d = d_opp[mask]
        dv = d[np.isfinite(d)]
        e = edge[mask]
        out[f"{side}_n"] = int(mask.sum())
        out[f"{side}_local"] = float((dv <= LOCAL).mean()) if len(dv) else np.nan
        out[f"{side}_tenuki"] = float((dv >= TENUKI).mean()) if len(dv) else np.nan
        out[f"{side}_dopp_mean"] = float(dv.mean()) if len(dv) else np.nan
        out[f"{side}_dopp_med"] = float(np.median(dv)) if len(dv) else np.nan
        ev = e[np.isfinite(e)]
        out[f"{side}_low"] = float((ev <= LOW).mean()) if len(ev) else np.nan
        out[f"{side}_pass"] = float(is_pass[mask].mean()) if mask.any() else np.nan
        do = d_own[mask]
        dov = do[np.isfinite(do)]
        out[f"{side}_down_mean"] = float(dov.mean()) if len(dov) else np.nan
        for p, (lo, hi) in enumerate(PHASES, start=1):
            m = mask & (t >= lo) & (t < hi)
            dp = d_opp[m]
            dp = dp[np.isfinite(dp)]
            out[f"{side}_local_p{p}"] = float((dp <= LOCAL).mean()) if len(dp) >= 5 else np.nan
    return out


def pmi_table(fold):
    own = read_scored_games(ROOT / "results/l1b" / ts.L1B[fold] / f"fold{fold}").set_index("game_key")
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[fold] / f"fold{fold}").set_index("game_key")
    cols = sum_cols(LEVELS)
    return pd.DataFrame({"own_pmi": posterior_mean_index(own[cols].to_numpy(dtype=float)), "opp_pmi": posterior_mean_index(opp.loc[own.index, cols].to_numpy(dtype=float))}, index=own.index)


def main():
    t0 = time.time()
    rows = []
    for f in ts.FOLDS:
        vec = ts.VEC / f"fold{f}"
        pm = pmi_table(f)
        n_games = 0
        for s, k in completed_parts(vec):
            csv_path, npz_path = part_paths(vec, s, k)
            tab = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
            with np.load(npz_path) as z:
                pos, t, target, row = z["pos"].astype(np.int64), z["t"].astype(np.int64), z["target"].astype(bool), z["game_row"].astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(tab["n_plies"].to_numpy())])
            for g, rec in enumerate(tab.to_dict("records")):
                sl = slice(starts[g], starts[g + 1])
                assert (row[sl] == g).all()
                feats = game_features(pos[sl], t[sl], target[sl])
                rows.append({"game_key": rec["game_key"], "player_id": rec["player_id"], "rank": rec["rank"], "fold": f, "n_plies": int(rec["n_plies"]),
                             "own_pmi": pm["own_pmi"].get(rec["game_key"], np.nan), "opp_pmi": pm["opp_pmi"].get(rec["game_key"], np.nan), **feats})
                n_games += 1
        print(f"fold {f}: {n_games} games ({time.time() - t0:.0f} s)", flush=True)
    games = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_csv(OUT / "geo_games.csv", index=False)
    cols = [c for c in games.columns if c.startswith(SIDES) and not c.endswith("_n")]
    out, ref, table, H = st.report(games, cols)
    st.print_report(out, ref, table, "geometry (H1), all scored games of folds 1-4")
    by_rank = games.groupby("rank")[["tgt_local", "tgt_tenuki", "tgt_low", "tgt_dopp_mean", "opp_local"]].mean().loc[list(LEVELS)]
    print(by_rank.round(3).to_string())
    st.dump(OUT / "geo_report.json", out, ref, {"games": len(games), "by_rank": by_rank.to_dict(), "thresholds": {"local": LOCAL, "tenuki": TENUKI, "low": LOW},
                                                "wall_s": time.time() - t0})
    print(f"written {OUT / 'geo_report.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
