"""Section 4M M29 interpretation checks v2 (post hoc: written 2026-11-06 07:22 after v1 = psych_checks.py and its review wf_ec1b8c8a-275;
decision mapping in the design doc before this ran). Fixes of the review's five confirmed flaws:
  (c) player-scaled expected model: p_i(cell) = k_i * p_pop(cell), k_i = the player's observed total / population-expected total for that
      role (no labels; clipped to [0.2, 5]); pl residuals also divided by the cross-fitted cell sd
  (d) punish window skips the role's first reply (it shares win_(t0+1) with the gift's wl, so search noise reads as give-back)
  (a) incremental R^2 of y - E: mean over 10 fold seeds, against a 200-permutation null (feature rows shuffled across players)
  (b) half split: features from half A, partial | [E_A, E_B]; both directions pooled with one joint player bootstrap; benchmark
      partial(E_A, y | E_B) = what a second 10-game reading of skill adds
  (e) judge table on best == 1 moves only (the played move is the search's top move): mean / sd / quantiles of wl
Primary reading: 20-game v2 composite partial | E (CI lower > 0 and >= 0.08) AND permutation-calibrated incremental R^2 p < 0.05.
  python -I logs/nano4/psych_checks_v2.py --moves results/ai_search_moves/fold1 --out results/r_variants/psych_checks_v2.json
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
from oppview_eval import partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402
from psych_checks import LEAD_EDGES, T_EDGES, WIN_EDGES, fill, r2  # noqa: E402
from psych_eval import BL, WIN5, per_move, side_features  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
V2 = ("tilt", "clutch", "ab_mean", "ab_sd", "punish")
KCLIP = (0.2, 5.0)


def load_moves(moves_dir):
    parts = []
    for npz in sorted(glob.glob(f"{moves_dir}/shard*.npz")):
        games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str})
        m = per_move(npz, games); z = np.load(npz)
        b = pd.DataFrame({"game": z["game"], "t": z["t"], "best": z["best"]})
        m = m.merge(b, on=["game", "t"], how="left", validate="1:1")
        m["game_key"] = games["game_key"].to_numpy()[m["game"].to_numpy()]
        m["game"] = m["game"].astype(str) + "@" + Path(npz).stem; parts.append(m)
    M = pd.concat(parts, ignore_index=True)
    M["cell"] = (np.digitize(M.win, WIN_EDGES) * 1000 + np.digitize(M.lead.abs(), LEAD_EDGES) * 100
                 + np.digitize(M.t, T_EDGES) * 10 + M.side.astype(int))
    M["blunder"] = (M.wl > BL).astype(float)
    return M.sort_values(["player_id", "game", "t"]).reset_index(drop=True)


def expected(M, k=5, seed=0):
    """Cross-fitted (by player folds) cell means of blunder, wl, pl and the cell sd of pl, pooled over both roles."""
    pids = np.array(sorted(M.player_id.unique())); f = dict(zip(pids, np.random.RandomState(seed).randint(0, k, len(pids))))
    fold = M.player_id.map(f).to_numpy(); out = np.full((len(M), 4), np.nan)
    for i in range(k):
        tr = M[fold != i]; g = tr.groupby("cell")
        tab = pd.concat([g[["blunder", "wl", "pl"]].mean(), g["pl"].std(ddof=0).rename("pl_sd")], axis=1)
        glob_ = pd.Series({"blunder": tr.blunder.mean(), "wl": tr.wl.mean(), "pl": tr.pl.mean(), "pl_sd": tr.pl.std(ddof=0)})
        te = fold == i; out[te] = tab.reindex(M.cell[te]).fillna(glob_).to_numpy()
    M["e_blunder"], M["e_wl"], M["e_pl"], M["e_plsd"] = out[:, 0], out[:, 1], out[:, 2], np.maximum(out[:, 3], 0.5)
    return M


def kscale(obs, exp):
    e = float(np.sum(exp)); return float(np.clip(np.sum(obs) / e, *KCLIP)) if e > 0 else 1.0


def v2_side(m):
    """Player-scaled features of one player's moves of one role (sorted by game, t)."""
    out = {}
    b = m.blunder.to_numpy() > 0; g = m.game.to_numpy()
    p = np.clip(kscale(m.blunder, m.e_blunder) * m.e_blunder.to_numpy(), 0, 0.999)
    obs = np.zeros(len(m), bool); log_none = np.zeros(len(m))
    for k in range(1, WIN5 + 1):
        same = np.r_[g[k:] == g[:-k], np.zeros(k, bool)]
        obs |= same & np.r_[b[k:], np.zeros(k, bool)]
        log_none += np.where(same, np.log1p(-np.r_[p[k:], np.zeros(k)]), 0.0)
    r = obs - (1 - np.exp(log_none))
    out["tilt"] = float(r[b].mean() - r.mean()) if b.sum() >= 3 else np.nan
    w = m.win.to_numpy(); pl = m.pl.to_numpy()
    rp = pl - kscale(pl, m.e_pl) * m.e_pl.to_numpy(); z = rp / m.e_plsd.to_numpy()
    und, dec = (w > 0.2) & (w < 0.8), (w < 0.1) | (w > 0.9)
    out["clutch"] = float(rp[und].mean() - rp[dec].mean()) if und.sum() >= 10 and dec.sum() >= 10 else np.nan
    ah, be = w > 0.7, w < 0.3; ok = ah.sum() >= 10 and be.sum() >= 10
    out["ab_mean"] = float(rp[ah].mean() - rp[be].mean()) if ok else np.nan
    out["ab_sd"] = float(z[be].std() - z[ah].std()) if ok else np.nan
    out["frac_undecided"] = float(und.mean())
    return out


def v2_punish(s, is_target):
    """After a blunder by the other role: the role's summed player-scaled wl residual over its 2nd-6th replies, minus 5 x its mean residual."""
    mine_all = s[s.is_target == is_target]; kw = kscale(mine_all.wl, mine_all.e_wl)
    base = float((mine_all.wl - kw * mine_all.e_wl).mean()); ev = []
    for _, gs in s.groupby("game"):
        other = gs[gs.is_target != is_target]; mine = gs[gs.is_target == is_target]
        for t0 in other.loc[other.wl > BL, "t"].to_numpy():
            nx = mine[mine.t > t0].iloc[1:1 + WIN5]
            if len(nx) == WIN5:
                ev.append(float((nx.wl - kw * nx.e_wl).sum()))
    return float(np.mean(ev) - WIN5 * base) if len(ev) >= 3 else np.nan


def features(M, keys=None):
    rows = []
    for pid, s in M.groupby("player_id", sort=True):
        if keys is not None:
            s = s[s.game_key.isin(keys.get(pid, ()))]
            if s.empty:
                continue
        r = {"player_id": pid, "y": RANK[s["rank"].iloc[0]]}
        for role, pre in ((True, "t_"), (False, "o_")):
            sub = s[s.is_target == role].sort_values(["game", "t"])
            raw = side_features(sub); f = v2_side(sub); f["punish"] = v2_punish(s, role)
            r.update({pre + "v2_" + k: f[k] for k in V2}); r[pre + "frac_undecided"] = f["frac_undecided"]
            r[pre + "persist_lost"], r[pre + "persist_won"] = raw["persist_lost"], raw["persist_won"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("player_id")


def boot_multi(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def incr_r2_perm(F, y, E, seeds=range(10), B=200, seed=0):
    res = y - E; F = fill(F)
    p0 = {s: ridge_oof(E[:, None], res, seed=s) for s in seeds}
    def delta(Fm):
        return float(np.mean([r2(res, ridge_oof(np.column_stack([E, Fm]), res, seed=s)) - r2(res, p0[s]) for s in seeds]))
    obs = delta(F); rng = np.random.default_rng(seed)
    null = np.array([delta(F[rng.permutation(len(y))]) for _ in range(B)])
    return {"delta": obs, "null_mean": float(null.mean()), "null_q95": float(np.percentile(null, 95)), "excess": float(obs - null.mean()),
            "p": float((1 + np.sum(null >= obs)) / (B + 1))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--moves", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args(); rep = {}
    M = expected(load_moves(a.moves))
    print(f"{M.game.nunique()} games, {M.player_id.nunique()} players, {len(M)} moves; best == 1 share {M.best.mean():.3f}", flush=True)
    # (e) judge table on the moves where the player chose the search's top move
    B1 = M[M.best == 1].copy(); B1["wbin"] = np.digitize(B1.win, WIN_EDGES); B1["phase"] = np.digitize(B1.t, T_EDGES)
    def tab(col):
        out = {}
        for k, s in B1.groupby(col):
            q = np.percentile(s.wl, [5, 25, 50, 75, 95])
            out[str(k)] = {"n": int(len(s)), "mean": float(s.wl.mean()), "sd": float(s.wl.std()), "q05_25_50_75_95": [float(x) for x in q],
                           "blunder_rate": float((s.wl > BL).mean())}
        return out
    rep["judge_best1"] = {"overall": tab(np.zeros(len(B1), int)), "by_side": tab("side"), "by_win_decile": tab("wbin"), "by_phase": tab("phase")}
    for name in ("overall", "by_side", "by_win_decile", "by_phase"):
        print(f"(e) best==1 wl {name}: " + "; ".join(f"{k}: mean {v['mean']:+.3f} sd {v['sd']:.3f} med {v['q05_25_50_75_95'][2]:+.3f} bl {v['blunder_rate']:.3f}"
                                                     for k, v in rep["judge_best1"][name].items()), flush=True)
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS))
    q20 = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")
    searched = M.groupby("player_id").game_key.apply(set)
    g = features(M); same20 = [p for p in g.index if p in q20.index and set(q20.game_keys[p]) == searched[p]]
    rep["same20_share"] = len(same20) / len(g); assert rep["same20_share"] == 1.0, rep["same20_share"]
    g["E"] = q20.E.reindex(g.index); y, E = g.y.to_numpy(float), g.E.to_numpy(float)
    cols = [p + "v2_" + k for p in ("t_", "o_") for k in V2] + [p + k for p in ("t_", "o_") for k in ("persist_lost", "persist_won")]
    occ = {}
    for p in ("t_", "o_"):
        for k in ("tilt", "punish", "clutch"):
            x = g[f"{p}v2_{k}"].to_numpy(float); fu = g[f"{p}frac_undecided"].to_numpy(float); ok = np.isfinite(x)
            occ[f"{p}{k}"] = {"spearman_frac_undecided": spearman(x, fu), "partial_frac_undecided_given_E": partial(x[ok], fu[ok], [E[ok]])}
    rep["state_occupancy_v2"] = occ; print("(c) state occupancy v2:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in occ.items()}), flush=True)
    # 20 games: single features (descriptive), composite fit to y (primary), incremental R^2 with permutation null (primary)
    feats = {}
    for c in cols:
        x = g[c].to_numpy(float); ok = np.isfinite(x)
        pe = partial(x[ok], y[ok], [E[ok]]); ci = boot_multi(lambda i: partial(x[ok][i], y[ok][i], [E[ok][i]]), int(ok.sum()))
        feats[c] = {"spearman_y": spearman(x, y), "partial_given_E": pe, "ci95": ci, "n": int(ok.sum())}
        print(f"  {c:18s} n {ok.sum():3d}  Spearman y {feats[c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  (descriptive)", flush=True)
    X = fill(g[cols].to_numpy(float)); comp = ridge_oof(X, y)
    est = partial(comp, y, [E]); ci = boot_multi(lambda i: partial(comp[i], y[i], [E[i]]), len(y))
    inc = incr_r2_perm(g[cols].to_numpy(float), y, E)
    primary = bool(ci[0] > 0 and est >= 0.08 and inc["p"] < 0.05)
    rep["v2_20"] = {"features": feats, "composite": {"spearman_y": spearman(comp, y), "partial_given_E": est, "ci95": ci}, "incremental_r2": inc,
                    "posthoc_signal": primary}
    print(f"v2 composite (14, 20 games): Spearman y {spearman(comp, y):.3f}  partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]; incremental R^2 of y - E "
          f"{inc['delta']:+.4f} (permutation null mean {inc['null_mean']:+.4f}, q95 {inc['null_q95']:+.4f}; excess {inc['excess']:+.4f}, p {inc['p']:.3f}); "
          f"post-hoc signal: {'YES' if primary else 'no'}", flush=True)
    # half split: features from half A, conditioned on both halves' E; pooled over the two directions
    q10 = meta[meta.n_games == 10]; halves = {}
    for pid, grp in q10.groupby("player_id"):
        if pid in g.index and len(grp) == 2:
            ka, kb = set(grp.game_keys.iloc[0]), set(grp.game_keys.iloc[1])
            if not (ka & kb) and (ka | kb) == searched.get(pid, set()):
                halves[pid] = ((ka, float(grp.E.iloc[0])), (kb, float(grp.E.iloc[1])))
    pids = sorted(halves); n = len(pids); print(f"(b) half split: {n} players", flush=True)
    yh = g.y.reindex(pids).to_numpy(float); dirs = []
    for src, tgt in ((0, 1), (1, 0)):
        gh = features(M, {p: halves[p][src][0] for p in pids}).reindex(pids)
        EA = np.array([halves[p][src][1] for p in pids]); EB = np.array([halves[p][tgt][1] for p in pids])
        comp_h = ridge_oof(fill(gh[cols].to_numpy(float)), yh); dirs.append((comp_h, EA, EB))
    def pooled(i, which):
        v = []
        for comp_h, EA, EB in dirs:
            if which == "feat":
                v.append(partial(comp_h[i], yh[i], [EA[i], EB[i]]))
            else:
                v.append(partial(EA[i], yh[i], [EB[i]]))
        return float(np.mean(v))
    idx = np.arange(n)
    hs = {"players": n,
          "composite_partial_given_EA_EB": {"estimate": pooled(idx, "feat"), "ci95": boot_multi(lambda i: pooled(i, "feat"), n),
                                            "by_direction": [partial(c, yh, [a_, b_]) for c, a_, b_ in dirs]},
          "benchmark_partial_EA_given_EB": {"estimate": pooled(idx, "bench"), "ci95": boot_multi(lambda i: pooled(i, "bench"), n)}}
    rep["half_split_v2"] = hs
    print(f"(b) pooled composite partial | [E_A, E_B] {hs['composite_partial_given_EA_EB']['estimate']:+.3f} {[round(x, 3) for x in hs['composite_partial_given_EA_EB']['ci95']]} "
          f"(directions {[round(x, 3) for x in hs['composite_partial_given_EA_EB']['by_direction']]}); benchmark partial(E_A, y | E_B) "
          f"{hs['benchmark_partial_EA_given_EB']['estimate']:+.3f} {[round(x, 3) for x in hs['benchmark_partial_EA_given_EB']['ci95']]}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
