"""Section 4M M42 (2026-11-07; user: 用更強的模型判斷勝負、每一手都加上強模型的勝率): player-level features from M25's strong-net
searched per-position values (kata1-b18c384nbt, 100 visits; results/ai_search_moves/fold1: 600 fold-1 players x 20 games).
win / lead are the side-to-move's; side 0 = Black to move. Per game, target view: wt = win if the target is to move else 1 - win.
  A (inferred results): w_strong = mean final wt (the last evaluated position), w_hard = share of games with final wt > 0.5,
    w_last = share of games whose last move is the target's (the winlm proxy); agreement of w_hard and w_last is reported.
  B (per-move win-rate trajectory): mean / sd of wt before the target's moves, share of them with wt > 0.5, mean change of wt caused by
    the target's moves (wt after - before), mean final target-view lead.
Statistics as psych_eval.py: Spearman with the label, partial | E (ct9's val1 members C/T-pc seeds 0-2) with a player bootstrap, OOF ridge
composites of A, B and A+B. Continue (pre-registered): a composite's partial CI lower > 0 and >= 0.08.
  python -I logs/nano4/winrate_pilot.py --out results/r_variants/m42_winrate.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ai_quality_eval import ridge_oof  # noqa: E402
from oppview_eval import boot, partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
A_COLS = ("w_strong", "w_hard", "w_last")
B_COLS = ("wt_mean", "wt_sd", "ahead_share", "move_delta", "final_lead")


def game_rows(moves_dir):
    rows, seen = [], set()
    for npz in sorted(glob.glob(f"{moves_dir}/shard*.npz")) + sorted(glob.glob(f"{moves_dir}/bench*.npz")):
        games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str})
        z = np.load(npz); df = pd.DataFrame({k: z[k] for k in ("game", "t", "side", "win", "lead")}).sort_values(["game", "t"])
        for g, s in df.groupby("game"):
            key = games["game_key"].iloc[g]
            if key in seen:
                continue
            seen.add(key)
            tc = 0 if games["color"].iloc[g] == "B" else 1
            side, win, lead, t = s["side"].to_numpy(), s["win"].to_numpy(float), s["lead"].to_numpy(float), s["t"].to_numpy()
            mine = side == tc
            wt = np.where(mine, win, 1.0 - win); lt = np.where(mine, lead, -lead)
            nxt = np.r_[t[1:] == t[:-1] + 1, False]
            delta = (wt[1:] - wt[:-1])[(mine & nxt)[:-1]] if len(t) > 1 else np.zeros(0)
            rows.append({"game_key": key, "player_id": games["player_id"].iloc[g], "rank": games["rank"].iloc[g], "final_wt": wt[-1], "last_is_target": float(mine[-1]),
                         "wt_mine": wt[mine], "delta": delta, "final_lead": float(np.clip(lt[-1], -40, 40))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--moves", default=str(ROOT / "results/ai_search_moves/fold1")); ap.add_argument("--out", required=True); a = ap.parse_args()
    G = game_rows(a.moves)
    agree = float(((G.final_wt > 0.5).astype(float) == G.last_is_target).mean())
    print(f"{len(G)} games, {G.player_id.nunique()} players; inferred-result agreement (final wt > 0.5 vs last move by the target) {agree:.3f}; "
          f"target inferred win share {(G.final_wt > 0.5).mean():.3f}", flush=True)
    feats = []
    for pid, s in G.groupby("player_id"):
        wm = np.concatenate(list(s.wt_mine)); dl = np.concatenate(list(s.delta))
        feats.append({"player_id": pid, "y": RANK[s["rank"].iloc[0]], "w_strong": s.final_wt.mean(), "w_hard": (s.final_wt > 0.5).mean(), "w_last": s.last_is_target.mean(),
                      "wt_mean": wm.mean(), "wt_sd": wm.std(), "ahead_share": (wm > 0.5).mean(), "move_delta": dl.mean() if len(dl) else 0.0, "final_lead": s.final_lead.mean()})
    g = pd.DataFrame(feats).set_index("player_id")
    meta = pd.DataFrame(json.load(open(ROOT / "results/trunk_seq/r_val1.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val1/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS))
    E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    g["E"] = E.reindex(g.index); g = g.dropna(subset=["E"])
    y, Ev = g.y.to_numpy(float), g.E.to_numpy(float)
    rep = {"players": int(len(g)), "games": int(len(G)), "agreement_hard_vs_last": agree, "features": {}, "composites": {}}
    for c in A_COLS + B_COLS:
        x = g[c].to_numpy(float); pe = partial(x, y, [Ev]); ci = boot(lambda idx: partial(x[idx], y[idx], [Ev[idx]]), len(x))
        rep["features"][c] = {"spearman_y": spearman(x, y), "partial_given_E": pe, "ci95": ci}
        print(f"{c:12s} Spearman y {rep['features'][c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    for name, cols in (("A_results", A_COLS), ("B_trajectory", B_COLS), ("A+B", A_COLS + B_COLS)):
        comp = ridge_oof(g[list(cols)].to_numpy(float), y); pe = partial(comp, y, [Ev]); ci = boot(lambda idx: partial(comp[idx], y[idx], [Ev[idx]]), len(y))
        rep["composites"][name] = {"spearman_y": spearman(comp, y), "partial_given_E": pe, "ci95": ci, "continue": bool(ci[0] > 0 and pe >= 0.08)}
        print(f"composite {name:12s} Spearman y {rep['composites'][name]['spearman_y']:.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
              + ("  -> CONTINUE" if rep["composites"][name]["continue"] else ""), flush=True)
    rep["E_spearman_y"] = spearman(Ev, y)
    print(f"reference: E Spearman y {rep['E_spearman_y']:.3f} on {len(y)} players")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
