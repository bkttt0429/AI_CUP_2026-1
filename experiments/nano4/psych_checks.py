"""Section 4M M29 interpretation checks (pre-registered 2026-11-06 06:28, before the M29 result; quant design audit wf_12b418f1).
They only decide how M29 is read; the main judgment (composite partial | E) stays the one in psych_pilot.json.
  (1) incremental OOF ridge R^2 of the residual y - E over [E]: [E, 14 features] vs [E] (the raw bias R^2 is inflated by shrinkage)
  (2) state-adjusted features: an expected-behaviour model fitted on the moves of all players (5-fold cross-fit by player) over cells
      (mover's win-rate decile x |lead| bucket x move-number phase x colour); tilt / punish / clutch / ab_mean / ab_sd recomputed as
      observed - expected, re-evaluated as in psych_eval (partial | E, composite)
  (3) judge offset: mean wl / pl and blunder rate by colour, win-rate decile and phase (both roles)
  (4) half split: the player's two disjoint 10-game validation questions A and B; features from A's games, E from question B
      (and B -> A), so the features and E do not share sampling error
  python -I logs/nano4/psych_checks.py --moves results/ai_search_moves/fold1 --out results/r_variants/psych_checks.json
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
from psych_eval import BL, NAMES, WIN5, per_move, punish_feature, side_features  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
ADJ = ("tilt", "clutch", "ab_mean", "ab_sd", "punish")
WIN_EDGES = np.linspace(0, 1, 11)[1:-1]
LEAD_EDGES = np.array([2.0, 5.0, 10.0, 20.0])
T_EDGES = np.array([50, 100, 150, 200])


def load_moves(moves_dir):
    parts = []
    for npz in sorted(glob.glob(f"{moves_dir}/shard*.npz")):
        games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str})
        m = per_move(npz, games); m["game_key"] = games["game_key"].to_numpy()[m["game"].to_numpy()]
        m["game"] = m["game"].astype(str) + "@" + Path(npz).stem; parts.append(m)
    M = pd.concat(parts, ignore_index=True)
    M["cell"] = (np.digitize(M.win, WIN_EDGES) * 1000 + np.digitize(M.lead.abs(), LEAD_EDGES) * 100
                 + np.digitize(M.t, T_EDGES) * 10 + M.side.astype(int))
    M["blunder"] = (M.wl > BL).astype(float)
    return M.sort_values(["player_id", "game", "t"]).reset_index(drop=True)


def expected(M, k=5, seed=0):
    """Cross-fitted cell means of blunder, wl and pl: the cell means of the players in the other k-1 player folds (both roles)."""
    pids = np.array(sorted(M.player_id.unique())); f = dict(zip(pids, np.random.RandomState(seed).randint(0, k, len(pids))))
    fold = M.player_id.map(f).to_numpy()
    out = np.full((len(M), 3), np.nan)
    for i in range(k):
        tr = M[fold != i]; tab = tr.groupby("cell")[["blunder", "wl", "pl"]].mean(); glob_ = tr[["blunder", "wl", "pl"]].mean()
        te = fold == i
        out[te] = tab.reindex(M.cell[te]).fillna(glob_).to_numpy()
    M["e_blunder"], M["e_wl"], M["e_pl"] = out[:, 0], out[:, 1], out[:, 2]
    return M


def adj_side(m):
    """State-adjusted features of one player's moves of one role (sorted by game, t)."""
    out = {}
    b = m.blunder.to_numpy() > 0; p = m.e_blunder.to_numpy(); g = m.game.to_numpy()
    obs = np.zeros(len(m), bool); log_none = np.zeros(len(m))
    for k in range(1, WIN5 + 1):
        same = np.r_[g[k:] == g[:-k], np.zeros(k, bool)]
        obs |= same & np.r_[b[k:], np.zeros(k, bool)]
        log_none += np.where(same, np.log1p(-np.clip(np.r_[p[k:], np.zeros(k)], 0, 0.999)), 0.0)
    r = obs - (1 - np.exp(log_none))
    out["tilt"] = float(r[b].mean() - r.mean()) if b.sum() >= 3 else np.nan
    w = m.win.to_numpy(); rp = (m.pl - m.e_pl).to_numpy()
    und, dec = (w > 0.2) & (w < 0.8), (w < 0.1) | (w > 0.9)
    out["clutch"] = float(rp[und].mean() - rp[dec].mean()) if und.sum() >= 10 and dec.sum() >= 10 else np.nan
    ah, be = w > 0.7, w < 0.3
    ok = ah.sum() >= 10 and be.sum() >= 10
    out["ab_mean"] = float(rp[ah].mean() - rp[be].mean()) if ok else np.nan
    out["ab_sd"] = float(rp[be].std() - rp[ah].std()) if ok else np.nan
    out["frac_undecided"] = float(und.mean())
    return out


def adj_punish(s, is_target):
    """After a blunder by the other role: the role's summed (wl - expected wl) over its next 5 moves, minus 5 x its mean residual."""
    mine_all = s[s.is_target == is_target]; base = float((mine_all.wl - mine_all.e_wl).mean())
    ev = []
    for _, gs in s.groupby("game"):
        other = gs[gs.is_target != is_target]; mine = gs[gs.is_target == is_target]
        for t0 in other.loc[other.wl > BL, "t"].to_numpy():
            nx = mine[mine.t > t0].head(WIN5)
            if len(nx) == WIN5:
                ev.append(float((nx.wl - nx.e_wl).sum()))
    return float(np.mean(ev) - WIN5 * base) if len(ev) >= 3 else np.nan


def features(M, keys=None):
    """Raw (psych_eval) and adjusted features per player; keys: optional {player_id: set of game_keys} restricting the games."""
    rows = []
    for pid, s in M.groupby("player_id", sort=True):
        if keys is not None:
            s = s[s.game_key.isin(keys.get(pid, ()))]
            if s.empty:
                continue
        r = {"player_id": pid, "y": RANK[s["rank"].iloc[0]]}
        for role, pre in ((True, "t_"), (False, "o_")):
            sub = s[s.is_target == role].sort_values(["game", "t"])
            f = side_features(sub); f["punish"] = punish_feature(s, pid, role)
            r.update({pre + k: f[k] for k in NAMES})
            fa = adj_side(sub); fa["punish"] = adj_punish(s, role)
            r.update({pre + "adj_" + k: fa[k] for k in ADJ}); r[pre + "frac_undecided"] = fa["frac_undecided"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("player_id")


def fill(X):
    X = np.asarray(X, float); return np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))


def r2(t, p):
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def incr_r2(F, y, E):
    """OOF ridge R^2 of y - E from [E, F] minus from [E]; player bootstrap CI on the fixed OOF predictions."""
    res = y - E; p0 = ridge_oof(E[:, None], res); p1 = ridge_oof(np.column_stack([E, fill(F)]), res)
    d = r2(res, p1) - r2(res, p0)
    ci = boot(lambda i: r2(res[i], p1[i]) - r2(res[i], p0[i]), len(y))
    return {"r2_E": r2(res, p0), "r2_E_feats": r2(res, p1), "delta": d, "ci95": ci}


def evaluate(g, cols, E, y, label):
    rep = {}
    for c in cols:
        x = g[c].to_numpy(float); ok = np.isfinite(x)
        pe = partial(x[ok], y[ok], [E[ok]]); ci = boot(lambda i: partial(x[ok][i], y[ok][i], [E[ok][i]]), int(ok.sum()))
        rep[c] = {"spearman_y": spearman(x[ok], y[ok]), "partial_given_E": pe, "ci95": ci, "spearman_bias": spearman(x[ok], (E - y)[ok]), "n": int(ok.sum())}
        print(f"  [{label}] {c:20s} n {ok.sum():3d}  Spearman y {rep[c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    X = fill(g[cols].to_numpy(float)); comp = ridge_oof(X, y)
    est = partial(comp, y, [E]); ci = boot(lambda i: partial(comp[i], y[i], [E[i]]), len(y))
    comp_r = ridge_oof(np.column_stack([E, X]), y - E)   # composite fitted to the residual, as the audit asked
    est_r = partial(comp_r, y, [E]); ci_r = boot(lambda i: partial(comp_r[i], y[i], [E[i]]), len(y))
    inc = incr_r2(g[cols].to_numpy(float), y, E)
    print(f"  [{label}] composite fit to y: partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]; fit to y - E: {est_r:+.3f} [{ci_r[0]:+.3f}, {ci_r[1]:+.3f}]; "
          f"incremental R^2 of y - E {inc['delta']:+.4f} [{inc['ci95'][0]:+.4f}, {inc['ci95'][1]:+.4f}]", flush=True)
    return {"features": rep, "composite_y": {"partial_given_E": est, "ci95": ci}, "composite_resid": {"partial_given_E": est_r, "ci95": ci_r}, "incremental_r2": inc}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--moves", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()
    M = expected(load_moves(a.moves))
    print(f"{M.game.nunique()} games, {M.player_id.nunique()} players, {len(M)} moves, {M.cell.nunique()} state cells", flush=True)
    rep = {}
    # (3) judge offset
    M["wbin"] = np.digitize(M.win, WIN_EDGES); M["phase"] = np.digitize(M.t, T_EDGES)
    def by(col):
        t = M.groupby(col)[["wl", "pl", "blunder"]].mean()
        return {str(k): {c: float(t.loc[k, c]) for c in t.columns} for k in t.index}
    off = {"overall": {c: float(M[c].mean()) for c in ("wl", "pl", "blunder")}, "by_side": by("side"), "by_win_decile": by("wbin"), "by_phase": by("phase"),
           "share_by_win_decile": {str(k): float(v) for k, v in M.wbin.value_counts(normalize=True).sort_index().items()}}
    rep["judge_offset"] = off
    print("(3) judge offset: overall", {k: round(v, 4) for k, v in off["overall"].items()})
    for name in ("by_side", "by_win_decile", "by_phase"):
        print(f"  {name}: " + "; ".join(f"{k}: wl {v['wl']:+.3f} pl {v['pl']:+.2f} bl {v['blunder']:.3f}" for k, v in off[name].items()), flush=True)
    # E per question
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS))
    q20 = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")
    # (1) + (2) on all 20 games
    g = features(M)
    searched = M.groupby("player_id").game_key.apply(set)
    same20 = np.mean([set(q20.game_keys[p]) == searched[p] for p in g.index if p in q20.index])
    print(f"searched games = the 20-game question's games for {same20:.3f} of players", flush=True)
    g["E"] = q20.E.reindex(g.index); g = g.dropna(subset=["E"]); y, E = g.y.to_numpy(float), g.E.to_numpy(float)
    raw = [p + k for p in ("t_", "o_") for k in NAMES]
    adj = [p + "adj_" + k for p in ("t_", "o_") for k in ADJ] + [p + k for p in ("t_", "o_") for k in ("persist_lost", "persist_won")]
    rep["state_occupancy"] = {f"{p}{k}": {"spearman_frac_undecided": spearman(g[f"{p}{k}"], g[f"{p}frac_undecided"]),
                                          "adj_spearman_frac_undecided": spearman(g[f"{p}adj_{k}"], g[f"{p}frac_undecided"])}
                              for p in ("t_", "o_") for k in ("tilt", "punish", "clutch")}
    rep["state_occupancy"]["frac_undecided_spearman_y"] = {p: spearman(g[f"{p}frac_undecided"], y) for p in ("t_", "o_")}
    print("state occupancy:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rep["state_occupancy"].items()}), flush=True)
    print("(1)/(2) all 20 games", flush=True)
    rep["raw_20"] = evaluate(g, raw, E, y, "raw")
    rep["adj_20"] = evaluate(g, adj, E, y, "adj")
    # (4) half split
    q10 = meta[meta.n_games == 10]
    halves = {}
    for pid, grp in q10.groupby("player_id"):
        if pid in g.index and len(grp) == 2:
            ka, kb = set(grp.game_keys.iloc[0]), set(grp.game_keys.iloc[1])
            if not (ka & kb) and (ka | kb) == searched.get(pid, set()):
                halves[pid] = ((ka, grp.E.iloc[0]), (kb, grp.E.iloc[1]))
    print(f"(4) half split: {len(halves)} players with two disjoint 10-game questions covering the 20 searched games", flush=True)
    rep["half_split"] = {"players": len(halves)}
    for d, (src, tgt) in enumerate(((0, 1), (1, 0))):
        keys = {p: h[src][0] for p, h in halves.items()}
        gh = features(M, keys); gh["E"] = pd.Series({p: h[tgt][1] for p, h in halves.items()}).reindex(gh.index)
        gh["E_same"] = pd.Series({p: h[src][1] for p, h in halves.items()}).reindex(gh.index)
        gh = gh.dropna(subset=["E"]); yh = gh.y.to_numpy(float)
        tag = f"A{src}->E{tgt}"
        rep["half_split"][tag] = {"raw_cross": evaluate(gh, raw, gh.E.to_numpy(float), yh, f"{tag} raw, E other half"),
                                  "adj_cross": evaluate(gh, adj, gh.E.to_numpy(float), yh, f"{tag} adj, E other half"),
                                  "adj_same": evaluate(gh, adj, gh.E_same.to_numpy(float), yh, f"{tag} adj, E same half")}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
