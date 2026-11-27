"""Section 4M M29 (2026-11-06; user: 賽局理論、心理特徵、時序特徵 → "跑跑看"): state-dependent behaviour from the strong net's searched
per-position values (logs/nano4/ai_search_pilot.py --save-moves -> results/ai_search_moves/fold1/shard*.npz + shard*_games.csv).
Per move t (mover's view): point loss pl = lead_t + lead_{t+1}, win-rate loss wl = win_t - (1 - win_{t+1}); blunder = wl > 0.10.
Player-level features, pooled over the player's 20 games, for the target (t_) and the opponents (o_):
  tilt          P(a blunder within the mover's next 5 moves | blunder now) - P(a blunder within the next 5 moves)
  clutch        mean pl in undecided positions (0.2 < win < 0.8) - mean pl in decided ones (win < 0.1 or > 0.9)
  ab_mean       mean pl when ahead (win > 0.7) - mean pl when behind (win < 0.3)
  ab_sd         sd of pl when behind - sd when ahead (risk taking when behind)
  punish        after an opponent blunder, the mover's summed wl over its next 5 moves (how much of the gift is given back)
  persist_lost  own moves per game after the mover's win rate first drops below 0.05 (playing on in lost positions)
  persist_won   own moves per game after it first exceeds 0.95 (finishing won games)
Evaluation as logs/nano4/ai_quality_eval.py: Spearman with the label, partial | E (the oh ensemble), OOF ridge composite of the 14 columns
(continue condition: partial | E CI lower > 0 and >= 0.08); plus each feature against the bias E - y and the OOF ridge R^2 of the bias.
  python -I logs/nano4/psych_eval.py --moves results/ai_search_moves/fold1 --out results/r_variants/psych_pilot.json
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
NAMES = ("tilt", "clutch", "ab_mean", "ab_sd", "punish", "persist_lost", "persist_won")
BL, WIN5 = 0.10, 5


def per_move(npz, games):
    """One row per move with position t and t+1 both answered: game, t, side, win, pl, wl, is_target."""
    z = np.load(npz); df = pd.DataFrame({k: z[k] for k in ("game", "t", "side", "win", "lead")}).sort_values(["game", "t"])
    nxt = df.groupby("game").shift(-1)
    ok = (nxt["t"] == df["t"] + 1)
    m = df[ok].copy(); n = nxt[ok]
    m["pl"] = (m["lead"] + n["lead"]).clip(-10, 40); m["wl"] = (m["win"] - (1 - n["win"])).clip(-1, 1)
    tcol = games["color"].map({"B": 0, "W": 1}).to_numpy()
    m["is_target"] = m["side"].to_numpy() == tcol[m["game"].to_numpy()]
    m["player_id"] = games["player_id"].to_numpy()[m["game"].to_numpy()]; m["rank"] = games["rank"].to_numpy()[m["game"].to_numpy()]
    return m


def side_features(m):
    """Features of one player's moves of one role (pooled over games); m has the columns of per_move, sorted by game, t."""
    out = {}
    b = (m["wl"] > BL).to_numpy(); g = m["game"].to_numpy()
    nxt_any = np.zeros(len(m), bool)  # a blunder among the mover's next 5 moves in the same game
    for k in range(1, WIN5 + 1):
        same = np.r_[g[k:] == g[:-k], np.zeros(k, bool)]
        shifted = np.r_[b[k:], np.zeros(k, bool)]
        nxt_any |= same & shifted
    out["tilt"] = float(nxt_any[b].mean() - nxt_any.mean()) if b.sum() >= 3 else np.nan
    w = m["win"].to_numpy(); pl = m["pl"].to_numpy()
    und, dec = (w > 0.2) & (w < 0.8), (w < 0.1) | (w > 0.9)
    out["clutch"] = float(pl[und].mean() - pl[dec].mean()) if und.sum() >= 10 and dec.sum() >= 10 else np.nan
    ah, be = w > 0.7, w < 0.3
    out["ab_mean"] = float(pl[ah].mean() - pl[be].mean()) if ah.sum() >= 10 and be.sum() >= 10 else np.nan
    out["ab_sd"] = float(pl[be].std() - pl[ah].std()) if ah.sum() >= 10 and be.sum() >= 10 else np.nan
    lost, won = [], []
    for _, s in m.groupby("game"):
        ww = s["win"].to_numpy()
        i = np.flatnonzero(ww < 0.05); lost.append(len(ww) - i[0] - 1 if len(i) else 0)
        j = np.flatnonzero(ww > 0.95); won.append(len(ww) - j[0] - 1 if len(j) else 0)
    out["persist_lost"], out["persist_won"] = float(np.mean(lost)), float(np.mean(won))
    return out


def punish_feature(m_all, pid, is_target):
    """After a blunder by the OTHER side, the role's summed wl over its next 5 moves; mean over events."""
    ev = []
    for _, s in m_all[m_all.player_id == pid].groupby("game"):
        s = s.sort_values("t"); other = s[s.is_target != is_target]; mine = s[s.is_target == is_target]
        for t0 in other.loc[other.wl > BL, "t"].to_numpy():
            nx = mine[mine.t > t0].head(WIN5)
            if len(nx) == WIN5:
                ev.append(nx.wl.sum())
    return float(np.mean(ev)) if len(ev) >= 3 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--moves", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()
    parts = []
    for npz in sorted(glob.glob(f"{a.moves}/shard*.npz")) + sorted(glob.glob(f"{a.moves}/bench*.npz")):
        games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str})
        m = per_move(npz, games); m["game"] = m["game"].astype(str) + "@" + Path(npz).stem; parts.append(m)
    M = pd.concat(parts, ignore_index=True)
    print(f"{M.game.nunique()} games, {M.player_id.nunique()} players, {len(M)} moves; blunder rate (wl > {BL}) target {(M[M.is_target].wl > BL).mean():.3f}, opponent {(M[~M.is_target].wl > BL).mean():.3f}", flush=True)
    rows = []
    for pid, s in M.groupby("player_id"):
        r = {"player_id": pid, "y": RANK[s["rank"].iloc[0]]}
        for role, pre in ((True, "t_"), (False, "o_")):
            sub = s[s.is_target == role].sort_values(["game", "t"])
            f = side_features(sub); f["punish"] = punish_feature(M, pid, role)
            r.update({pre + k: f[k] for k in NAMES})
        rows.append(r)
    g = pd.DataFrame(rows).set_index("player_id")
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    g["E"] = E.reindex(g.index); g = g.dropna(subset=["E"]); g["bias"] = g.E - g.y
    y, Ev, bias = g.y.to_numpy(float), g.E.to_numpy(float), g.bias.to_numpy(float)
    feats = [p + k for p in ("t_", "o_") for k in NAMES]
    rep = {"players": int(len(g)), "features": {}}
    if len(g) < 30:
        print(f"only {len(g)} players: statistics skipped (smoke)"); print(g[feats].describe().T.round(3).to_string())
        Path(a.out).write_text(json.dumps({"players": int(len(g)), "smoke": True}, indent=1)); return
    for c in feats:
        x = g[c].to_numpy(float); ok = np.isfinite(x)
        pe = partial(x[ok], y[ok], [Ev[ok]]); ci = boot(lambda idx: partial(x[ok][idx], y[ok][idx], [Ev[ok][idx]]), int(ok.sum()))
        rep["features"][c] = {"spearman_y": spearman(x[ok], y[ok]), "partial_given_E": pe, "ci95": ci, "spearman_bias": spearman(x[ok], bias[ok]), "n": int(ok.sum())}
        print(f"{c:15s} n {ok.sum():3d}  Spearman y {rep['features'][c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  Spearman bias {rep['features'][c]['spearman_bias']:+.3f}", flush=True)
    X = g[feats].to_numpy(float); X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))
    comp = ridge_oof(X, y); est = partial(comp, y, [Ev]); ci = boot(lambda idx: partial(comp[idx], y[idx], [Ev[idx]]), len(y))
    pb = ridge_oof(X, bias); r2 = 1 - np.sum((bias - pb) ** 2) / np.sum((bias - bias.mean()) ** 2)
    rep["composite"] = {"spearman_y": spearman(comp, y), "partial_given_E": est, "ci95": ci}
    rep["bias_oof_r2"] = float(r2); rep["continue"] = bool(ci[0] > 0 and est >= 0.08)
    print(f"composite (14): Spearman y {rep['composite']['spearman_y']:.3f}  partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]; OOF ridge R^2 of the bias E - y {r2:+.4f}; "
          f"continue (CI lower > 0 and >= 0.08): {'YES' if rep['continue'] else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
