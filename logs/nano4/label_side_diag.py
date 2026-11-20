"""D1-D3: what the label-side channels still hold (user 2026-11-05, 「好」 to the analysis of the signal Human SL / L1b cannot
capture). Report only, except D3's one pre-specified comparison. Same data, questions and protocol as logs/nano4/r_variants.py
(folds 1-4, 32 training questions per player drawn with RandomState(TRAIN_SEED + v), MLP h32, seeds 0-2 averaged, class-balanced
official score, player bootstrap); its `base` arm must reproduce M10's base per fold. fold 0 is not touched.

D1  Does the opponent channel have a per-player floor? Arms are column subsets of the 64-dim fused_opp_diff_hd
    (HSL 24 | L1b 16 | opp 16 | diff 3 | HD 5): own = HSL + L1b (40), opp = opp 16, opp_hd = opp 16 + HD 5, base = 64.
    Per arm and question size: official score; signed error e = E[rank] - label with the (class, size) mean removed (the
    shrinkage every player of a class shares); persistent variance = covariance of e between a player's disjoint questions of
    one size (n=10: two halves, n=5: four quarters), sampling = var(e) - persistent. Measurement model: own = r + u + noise/sqrt(n)
    keeps the floor var(u); opp averages n different opponents, so its persistent variance should be small and its score should
    rise faster with n. Player bootstrap CIs for the slope [score(20) - score(5)] of opp minus own and for persistent var(n=10)
    of opp minus own; the correlation of the two arms' persistent errors.
D2  The same question without a model: players with 20 scored games, two disjoint 10-game halves (selection order), mean per-game
    L1b pmi of the target (own) and of the opponent (opp). Within class (class means removed): persistent variance =
    cov(half A, half B), in class units via the slope of class-mean pmi on the class index (a rank uniform inside its class has
    SD sqrt(1/12) = 0.29 class units); persistent own-opp correlation = cross-half cov / sqrt(persistent vars). Kyu classes (D-A,
    three kyu each) and dan classes (1D-6D, one dan each) separately. HD1 calibration: per game d = opp pmi - own pmi by signed gap
    (-1 target is the lower hand, 0 even, +1 upper hand), within class, in class units: about +-1 for dan and +-1/3 for kyu if HD
    counts server ranks and opponents of even games share the target's rank.
D3  Single-colour questions (the test's structure: every test question has one colour; standard validation questions are single-
    colour only 2.4-12.5 % of the time). Validation sc questions = per player and colour, disjoint 10- and 5-game chunks of that
    colour's scored games (the scored games are <= 20 per player, so n = 15 / 20 single-colour questions cannot be built).
    Arms: std (= D1's base) and sc50 (every second of a player's 32 training questions drawn from one colour's games, its own
    RandomState(TRAIN_SEED + 1000 + v)). Pre-specified: sc50 vs std on the sc questions (sizes 5, 10), conditions (1) CI lower
    bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005. Report only: sc50 vs std on the standard questions; std on sc vs mixed
    questions of the same players and sizes; the sc questions holding an HD1 game separately.

  sbatch -J lsdiag -t 02:00:00 scripts/nano4/run.sbatch python -I logs/nano4/label_side_diag.py --out results/r_variants/label_side_diag.json
  python -I logs/nano4/label_side_diag.py --out ... --from-npz results/r_variants/label_side_diag.npz   # report only, no fitting
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table, signed_gap  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

K = len(LEVELS)
ARMS = {"own": np.r_[0:40], "opp": np.r_[40:56], "opp_hd": np.r_[40:56, 59:64], "base": np.r_[0:64]}
SIZES, SC_SIZES = (5, 10, 20), (5, 10)
SUBSETS = {"kyu": list(range(0, 4)), "dan": list(range(4, 10)), "all": list(range(K))}
QCOLS = ["question_id", "rank", "player_id", "n_games", "game_keys"]


# ---------------------------------------------------------------------------------------------------------------- questions
def single_colour_questions(players, colour_of, sizes=(10, 5)):
    """Per player and colour: disjoint chunks of `sizes` games of that colour (scored games in selection order)."""
    rows = []
    for p in players.itertuples():
        for c in ("B", "W"):
            kc = [k for k in p.game_keys if colour_of[k] == c]
            for n in sizes:
                for j in range(len(kc) // n):
                    rows.append({"question_id": f"{p.rank}_{p.player_id[:8]}_{c}{n}_{j}", "rank": p.rank, "player_id": p.player_id,
                                 "n_games": n, "game_keys": kc[j * n:(j + 1) * n]})
    return pd.DataFrame(rows, columns=QCOLS)


def sample_questions_sc50(players, per_player, rng, colour_of, sizes=range(5, 21)):
    """sample_questions with every second question drawn from one colour's games (uniformly among the colours with >= 5 scored
    games); a player without such a colour keeps the mixed draw."""
    sizes = list(sizes)
    rows = []
    for p in players.itertuples():
        keys = list(p.game_keys)
        if not [s for s in sizes if s <= len(keys)]:
            continue
        pools = [pool for pool in ([k for k in keys if colour_of[k] == c] for c in ("B", "W")) if len(pool) >= sizes[0]]
        for j in range(per_player):
            pool = pools[rng.randint(len(pools))] if (j % 2 == 1 and pools) else keys
            size = int(rng.choice([s for s in sizes if s <= len(pool)]))
            pick = rng.choice(len(pool), size=size, replace=False)
            rows.append({"question_id": f"s_{p.player_id[:8]}_{j}", "rank": p.rank, "player_id": p.player_id, "n_games": size,
                         "game_keys": [pool[i] for i in pick]})
    return pd.DataFrame(rows, columns=QCOLS)


def fit_predict(Xt, y, Xvs, seeds):
    """Seed-averaged class probabilities of MLP h32 on each matrix of Xvs (np.mean over seeds, as r_variants.py)."""
    preds = [[] for _ in Xvs]
    for s in seeds:
        m = MLPModel(hidden=32, seed=s).fit(Xt, y)
        for i, Xv in enumerate(Xvs):
            preds[i].append(m.predict_proba(Xv))
    return [np.mean(p, axis=0) for p in preds]


def game_table(f, hsl, l1b, opp, hd, players):
    """Per scored game of fold f: player, class, colour, own / opponent L1b pmi, signed HD gap, half (0 / 1 for players with 20
    scored games, by selection order; -1 otherwise)."""
    cols = sum_cols(LEVELS)
    li, oi, hi = l1b.set_index("game_key"), opp.set_index("game_key"), hd.drop_duplicates("game_key").set_index("game_key")
    keys = li.index.to_numpy()
    own = posterior_mean_index(li[cols].to_numpy(dtype=float))
    oppm = posterior_mean_index(oi.loc[keys, cols].to_numpy(dtype=float))
    colour = hsl.set_index("game_key").loc[keys, "color"].to_numpy(dtype=str)
    gap = signed_gap(hi.loc[keys, "hd"].to_numpy(dtype=float), colour)
    half = {}
    for p in players.itertuples():
        if len(p.game_keys) == 20:
            for i, k in enumerate(p.game_keys):
                half[k] = i // 10
    rank_idx = {lv: i for i, lv in enumerate(LEVELS)}
    return pd.DataFrame({"fold": f, "game_key": keys, "player_id": li["player_id"].to_numpy(dtype=str),
                         "cls": li["rank"].map(rank_idx).to_numpy(dtype=int), "colour": colour, "own": own, "opp": oppm,
                         "gap": gap, "half": [half.get(k, -1) for k in keys]})


# ---------------------------------------------------------------------------------------------------------------- statistics
class PlayerBoot:
    """Player bootstrap stratified by class (players resampled within each class, all their questions follow); every ci() call
    draws the same replicates."""

    def __init__(self, pid, y, n_boot):
        uniq, inv = np.unique(np.asarray(pid), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        self.q_of = [order[s:s + c] for s, c in zip(starts, counts)]
        cls = np.array([np.asarray(y)[q[0]] for q in self.q_of])
        self.by_c = [np.where(cls == k)[0] for k in range(K) if (cls == k).any()]
        self.n_boot = n_boot

    def ci(self, stat):
        rng = np.random.RandomState(42)
        vals = []
        for _ in range(self.n_boot):
            pick = np.concatenate([rng.choice(ix, len(ix)) for ix in self.by_c])
            vals.append(stat(np.concatenate([self.q_of[p] for p in pick])))
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        return [float(lo), float(hi)]


def strat_ci(cls, stat, n_boot):
    """Bootstrap CI of stat(rows) over rows (players) resampled within each class."""
    rng = np.random.RandomState(42)
    by_c = [np.where(cls == k)[0] for k in np.unique(cls)]
    vals = [stat(np.concatenate([rng.choice(ix, len(ix)) for ix in by_c])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def centred_errors(P, y, n):
    """e = E[rank] - label with the mean of e removed inside every (class, size) cell."""
    e = P @ np.arange(K) - y
    for k in range(K):
        for m in np.unique(n):
            sel = (y == k) & (n == m)
            if sel.any():
                e[sel] -= e[sel].mean()
    return e


def player_moments(e_a, e_b, y, pid, n, size):
    """Per player with >= 2 questions of `size`: (class, mean pairwise cross product a_i b_j over i != j, mean a_i b_i).
    With a = b: (class, persistent variance estimate, total variance)."""
    m = np.where(n == size)[0]
    df = pd.DataFrame({"pid": pid[m], "y": y[m], "a": e_a[m], "b": e_b[m]})
    rows = []
    for _, g in df.groupby("pid", sort=True):
        a, b = g["a"].to_numpy(), g["b"].to_numpy()
        k = len(a)
        if k < 2:
            continue
        rows.append((g["y"].iat[0], (a.sum() * b.sum() - (a * b).sum()) / (k * (k - 1)), (a * b).mean()))
    return np.array(rows)


def arm_scores(P, y, n, fold, sizes):
    s = evaluate(P, y, pd.DataFrame({"n_games": n}), sizes=sizes)[1]
    per_fold = [balanced_mean_over_n(s[fold == v], y[fold == v], n[fold == v], sizes=sizes) for v in np.unique(fold)]
    by_n = {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in sizes}
    return s, {"score": float(np.mean(per_fold)), "per_fold": per_fold, "by_n": by_n}


def d1_report(Z, n_boot):
    y, n, pid, fold = Z["y_val"], Z["n_val"], Z["pid_val"], Z["fold_val"]
    out, S, E = {"arms": {}}, {}, {}
    for arm in list(ARMS) + ["sc50"]:
        S[arm], out["arms"][arm] = arm_scores(Z[f"P_val_{arm}"], y, n, fold, SIZES)
        a = out["arms"][arm]
        a["slope_20_minus_5"] = a["by_n"]["20"] - a["by_n"]["5"]
        if arm in ARMS:
            E[arm] = centred_errors(Z[f"P_val_{arm}"], y.astype(float), n)
            comp = {}
            for size in SIZES:
                mm = n == size
                total_all = float((E[arm][mm] ** 2).mean())
                if size == 20:
                    comp[str(size)] = {"total": total_all}
                    continue
                M = player_moments(E[arm], E[arm], y, pid, n, size)
                comp[str(size)] = {"players": int(len(M)), "total": float(M[:, 2].mean()), "persistent": float(M[:, 1].mean()),
                                   "sampling": float(M[:, 2].mean() - M[:, 1].mean()), "persistent_share": float(M[:, 1].mean() / M[:, 2].mean())}
            a["error_variance"] = comp
        print(f"D1 {arm:6s} {a['score']:.4f}  by n " + " ".join(f"{m}: {a['by_n'][str(m)]:.4f}" for m in SIZES)
              + f"  slope20-5 {a['slope_20_minus_5']:+.4f}"
              + ("" if arm not in ARMS else "  persistent share n5/n10 "
                 + f"{a['error_variance']['5']['persistent_share']:.3f}/{a['error_variance']['10']['persistent_share']:.3f}"
                 + f"  persistent var n10 {a['error_variance']['10']['persistent']:.3f} of {a['error_variance']['10']['total']:.3f}"), flush=True)
    boot = PlayerBoot(pid, y, n_boot)

    def slope(s, idx):
        return (balanced_mean_over_n(s[idx], y[idx], n[idx], sizes=(20,)) - balanced_mean_over_n(s[idx], y[idx], n[idx], sizes=(5,)))

    out["pairs"] = {}
    for arm in ("opp", "opp_hd"):
        point = out["arms"][arm]["slope_20_minus_5"] - out["arms"]["own"]["slope_20_minus_5"]
        ci = boot.ci(lambda idx, arm=arm: slope(S[arm], idx) - slope(S["own"], idx))
        out["pairs"][f"slope {arm} - own"] = {"diff": point, "ci95": ci}
        print(f"D1 slope(20-5) {arm} - own: {point:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    for size in (5, 10):
        Mo, Mp = player_moments(E["own"], E["own"], y, pid, n, size), player_moments(E["opp"], E["opp"], y, pid, n, size)
        X = player_moments(E["own"], E["opp"], y, pid, n, size)
        cls = Mo[:, 0].astype(int)
        diff = float(Mp[:, 1].mean() - Mo[:, 1].mean())
        ci = strat_ci(cls, lambda r: Mp[r, 1].mean() - Mo[r, 1].mean(), n_boot)
        rho = float(X[:, 1].mean() / np.sqrt(max(Mo[:, 1].mean(), 1e-12) * max(Mp[:, 1].mean(), 1e-12)))
        rho_ci = strat_ci(cls, lambda r: X[r, 1].mean() / np.sqrt(max(Mo[r, 1].mean(), 1e-12) * max(Mp[r, 1].mean(), 1e-12)), n_boot)
        out["pairs"][f"persistent var opp - own n{size}"] = {"diff": diff, "ci95": ci}
        out["pairs"][f"persistent error correlation own~opp n{size}"] = {"rho": rho, "ci95": rho_ci}
        print(f"D1 n={size}: persistent var opp - own {diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]; "
              f"persistent error correlation own~opp {rho:.3f} [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")
    return out


def d2_report(G, n_boot):
    out = {}
    T = G[G["half"] >= 0]
    W = T.groupby(["player_id", "cls", "half"])[["own", "opp"]].mean().unstack("half")
    W.columns = [f"{c}{h}" for c, h in W.columns]
    W = W.reset_index().dropna()
    for name, classes in SUBSETS.items():
        P = W[W["cls"].isin(classes)].reset_index(drop=True)
        cls = P["cls"].to_numpy()
        avg = {x: (P[f"{x}0"] + P[f"{x}1"]).to_numpy() / 2 for x in ("own", "opp")}
        cm = {x: pd.Series(avg[x]).groupby(cls).mean() for x in avg}
        b = {x: float(np.polyfit(cm[x].index.to_numpy(dtype=float), cm[x].to_numpy(), 1)[0]) for x in avg}
        R = {}
        for x in ("own", "opp"):
            for h in (0, 1):
                v = P[f"{x}{h}"].to_numpy()
                R[f"{x}{h}"] = v - pd.Series(v).groupby(cls).transform("mean").to_numpy()

        def stats(r):
            po, pp = np.mean(R["own0"][r] * R["own1"][r]), np.mean(R["opp0"][r] * R["opp1"][r])
            cross = np.mean(R["own0"][r] * R["opp1"][r] + R["opp0"][r] * R["own1"][r]) / 2
            vo = np.mean((R["own0"][r] ** 2 + R["own1"][r] ** 2) / 2)
            vp = np.mean((R["opp0"][r] ** 2 + R["opp1"][r] ** 2) / 2)
            return {"persistent_own": po, "persistent_opp": pp, "sampling_own_half": vo - po, "sampling_opp_half": vp - pp,
                    "sd_own_class_units": np.sqrt(max(po, 0)) / b["own"], "sd_opp_class_units": np.sqrt(max(pp, 0)) / b["opp"],
                    "reliability_own": po / vo, "reliability_opp": pp / vp,
                    "persistent_corr_own_opp": cross / np.sqrt(max(po, 1e-12) * max(pp, 1e-12))}

        st = {k: float(v) for k, v in stats(np.arange(len(P))).items()}
        ci = {k: strat_ci(cls, lambda r, k=k: stats(r)[k], n_boot) for k in ("sd_own_class_units", "sd_opp_class_units", "persistent_corr_own_opp")}
        sp = lambda a, c: float(pd.Series(a).rank().corr(pd.Series(c).rank()))  # noqa: E731
        block = {"players": int(len(P)), "slope_pmi_per_class": b, **st, "ci95": ci,
                 "spearman_20games": {"own_label": sp(avg["own"], cls), "opp_label": sp(avg["opp"], cls), "own_opp": sp(avg["own"], avg["opp"])}}
        # HD1 calibration over every scored game of the subset's players (not only the 20-game players)
        g = G[G["cls"].isin(classes) & (np.abs(G["gap"]) <= 1)]
        d = (g["opp"] - g["own"]).to_numpy()
        cal = {}
        for s in (-1, 0, 1):
            cal[str(s)] = {"games": int((g["gap"] == s).sum())}
        shifts = {"-1": [], "1": []}
        for k in classes:
            gk = g["cls"].to_numpy() == k
            base0 = d[gk & (g["gap"].to_numpy() == 0)]
            for s in (-1, 1):
                ds = d[gk & (g["gap"].to_numpy() == s)]
                if len(ds) >= 30 and len(base0) >= 30:
                    shifts[str(s)].append(ds.mean() - base0.mean())
        for s in ("-1", "1"):
            raw = float(np.mean(shifts[s])) if shifts[s] else float("nan")
            cal[s].update({"d_shift_vs_even_pmi": raw, "d_shift_vs_even_class_units": raw / b["opp"], "classes": len(shifts[s])})
        cal["0"]["d_mean_even_class_units"] = float(np.mean(d[g["gap"].to_numpy() == 0]) / b["opp"])
        block["hd1_calibration"] = cal
        out[name] = block
        print(f"D2 {name:3s} players {len(P)}: slope own {b['own']:.3f} opp {b['opp']:.3f} pmi/class; persistent SD (class units) own "
              f"{st['sd_own_class_units']:.3f} {ci['sd_own_class_units']}  opp {st['sd_opp_class_units']:.3f} {ci['sd_opp_class_units']}; "
              f"reliability own {st['reliability_own']:.3f} opp {st['reliability_opp']:.3f}; persistent corr own~opp "
              f"{st['persistent_corr_own_opp']:.3f} {ci['persistent_corr_own_opp']}; Spearman own/opp/own~opp "
              f"{block['spearman_20games']['own_label']:.3f}/{block['spearman_20games']['opp_label']:.3f}/{block['spearman_20games']['own_opp']:.3f}")
        print(f"D2 {name:3s} HD1: d shift (class units) target lower {cal['-1']['d_shift_vs_even_class_units']:+.3f}, upper "
              f"{cal['1']['d_shift_vs_even_class_units']:+.3f}; even-game mean d {cal['0']['d_mean_even_class_units']:+.3f}; games "
              f"{cal['-1']['games']}/{cal['0']['games']}/{cal['1']['games']}", flush=True)
    return out


def d3_report(Z, n_boot):
    out = {}
    y, n, pid, fold, H = Z["y_sc"], Z["n_sc"], Z["pid_sc"], Z["fold_sc"], Z["H_sc"]
    S = {}
    for arm in ("std", "sc50"):
        S[arm], out[arm] = arm_scores(Z[f"P_sc_{arm}"], y, n, fold, SC_SIZES)
    dd = paired_diff(S["sc50"], S["std"], y, pid, n, sizes=SC_SIZES, n=n_boot)
    pf = [a - b for a, b in zip(out["sc50"]["per_fold"], out["std"]["per_fold"])]
    cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
    out["prespecified sc50 vs std on sc"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
    print(f"D3 sc questions {len(y)} (players {len(np.unique(pid))}): std {out['std']['score']:.4f}  sc50 {out['sc50']['score']:.4f}")
    print(f"D3 PRE-SPECIFIED sc50 vs std on sc: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds "
          + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items())
          + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    # sc50 vs std on the standard questions
    yv, nv, pv = Z["y_val"], Z["n_val"], Z["pid_val"]
    sv = {arm: evaluate(Z[f"P_val_{key}"], yv, pd.DataFrame({"n_games": nv}))[1] for arm, key in (("std", "base"), ("sc50", "sc50"))}
    dv = paired_diff(sv["sc50"], sv["std"], yv, pv, nv, n=n_boot)
    out["sc50 vs std on standard"] = dv
    print(f"D3 sc50 vs std on standard questions: {dv['diff']:+.4f} [{dv['ci95'][0]:+.4f}, {dv['ci95'][1]:+.4f}]  by n "
          + " ".join(f"{m}: {v:+.4f}" for m, v in dv["by_n"].items()))
    # std on sc vs mixed, same players and sizes
    keep = np.isin(pv, np.unique(pid)) & np.isin(nv, SC_SIZES)
    cs = np.r_[S["std"], sv["std"][keep]]
    cy, cn, cp = np.r_[y, yv[keep]], np.r_[n, nv[keep]], np.r_[pid, pv[keep]]
    flag = np.r_[np.ones(len(y), bool), np.zeros(keep.sum(), bool)]

    def gap(idx):
        a, b = idx[flag[idx]], idx[~flag[idx]]
        return balanced_mean_over_n(cs[a], cy[a], cn[a], sizes=SC_SIZES) - balanced_mean_over_n(cs[b], cy[b], cn[b], sizes=SC_SIZES)

    point = gap(np.arange(len(cs)))
    ci = PlayerBoot(cp, cy, n_boot).ci(gap)
    out["std: sc minus mixed, same players"] = {"diff": point, "ci95": ci, "mixed_questions": int(keep.sum())}
    print(f"D3 std on sc minus mixed (same players, n 5/10): {point:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    # questions holding an HD1 game
    hd1 = H[:, hd_check.COL_SENTE] > 0
    blk = {}
    for name, mask in (("with_hd1", hd1), ("without_hd1", ~hd1)):
        blk[name] = {"questions": int(mask.sum())}
        for arm in ("std", "sc50"):
            blk[name][arm] = balanced_mean_over_n(S[arm][mask], y[mask], n[mask], sizes=SC_SIZES)
        blk[name]["sc50_minus_std"] = blk[name]["sc50"] - blk[name]["std"]
    out["by_hd1"] = blk
    print("D3 by HD1: " + "; ".join(f"{k} {v['questions']} q: std {v['std']:.4f} sc50 {v['sc50']:.4f} ({v['sc50_minus_std']:+.4f})" for k, v in blk.items()))
    return out


# ---------------------------------------------------------------------------------------------------------------- main
def run_fits(a, t0):
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    data, games = {}, []
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
        players = l1b_cv.players_from_scores(hsl, folds_df)
        colour = dict(zip(hsl["game_key"], hsl["color"].astype(str)))
        d = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "colour": colour,
             "val": l1b_cv.pilot_questions(players), "sc": single_colour_questions(players, colour)}
        data[f] = d
        games.append(game_table(f, hsl, l1b, opp, hd, players))
        print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(d['val'])} standard / {len(d['sc'])} single-colour "
              f"validation questions ({time.time() - t0:.0f} s)", flush=True)
    m10 = ROOT / "results/r_variants/m10.json"
    anchor = json.loads(m10.read_text())["arms"]["base"]["per_fold"] if m10.exists() else None
    Z = {k: [] for k in ["y_val", "n_val", "pid_val", "fold_val", "y_sc", "n_sc", "pid_sc", "fold_sc", "H_sc"]
         + [f"P_val_{arm}" for arm in list(ARMS) + ["sc50"]] + ["P_sc_std", "P_sc_sc50"]}
    for i, v in enumerate(a.folds):
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        rng_sc = np.random.RandomState(l1b_cv.TRAIN_SEED + 1000 + v)
        train_sc = pd.concat([sample_questions_sc50(data[f]["players"], 32, rng_sc, data[f]["colour"]) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        Xs, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        Xs_sc, y_tr_sc, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_sc)
        d = data[v]
        Xv, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        Xc, y_sc, n_sc, H_sc = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["sc"])
        B_tr, B_tr_sc, B_va, B_sc = Xs["fused_opp_diff_hd"], Xs_sc["fused_opp_diff_hd"], Xv["fused_opp_diff_hd"], Xc["fused_opp_diff_hd"]
        msg = []
        for arm, cols in ARMS.items():
            Ps = fit_predict(B_tr[:, cols], y_tr, [B_va[:, cols]] + ([B_sc] if arm == "base" else []), a.seeds)
            Z[f"P_val_{arm}"].append(Ps[0])
            if arm == "base":
                Z["P_sc_std"].append(Ps[1])
            msg.append(f"{arm} {evaluate(Ps[0], y_va, d['val'])[0]['balanced_mean_over_n']:.4f}")
        Ps = fit_predict(B_tr_sc, y_tr_sc, [B_va, B_sc], a.seeds)
        Z["P_val_sc50"].append(Ps[0])
        Z["P_sc_sc50"].append(Ps[1])
        for key, val in (("y_val", y_va), ("n_val", n_va), ("pid_val", d["val"]["player_id"].to_numpy(dtype=str)), ("fold_val", np.full(len(y_va), v)),
                         ("y_sc", y_sc), ("n_sc", n_sc), ("pid_sc", d["sc"]["player_id"].to_numpy(dtype=str)), ("fold_sc", np.full(len(y_sc), v)), ("H_sc", H_sc)):
            Z[key].append(val)
        base_v = evaluate(Z["P_val_base"][-1], y_va, d["val"])[0]["balanced_mean_over_n"]
        check = "" if anchor is None else f"  (M10 base {anchor[i]:.4f}, diff {base_v - anchor[i]:+.1e})"
        print(f"fold {v} ({len(train_q)} / {len(train_sc)} training questions): " + "  ".join(msg) + check + f"  ({time.time() - t0:.0f} s)", flush=True)
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    G = pd.concat(games, ignore_index=True)
    for c in G.columns:
        Z[f"G_{c}"] = G[c].to_numpy(dtype=str if G[c].dtype == object else None)
    Z["anchor_m10"] = np.array(anchor if anchor is not None else [], dtype=float)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-npz", help="skip the fits and report from this file")
    a = ap.parse_args()
    if 0 in a.folds:
        ap.error("fold 0 is not evaluated here")
    t0 = time.time()
    npz = Path(a.from_npz) if a.from_npz else Path(a.out).with_suffix(".npz")
    if a.from_npz:
        Z = dict(np.load(npz))
    else:
        Z = run_fits(a, t0)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **Z)
        print(f"written {npz} ({time.time() - t0:.0f} s)", flush=True)
    G = pd.DataFrame({k[2:]: v for k, v in Z.items() if k.startswith("G_")})
    for c in ("fold", "cls", "half"):
        G[c] = G[c].astype(int)
    for c in ("own", "opp", "gap"):
        G[c] = G[c].astype(float)
    rep = {"folds": a.folds, "seeds": a.seeds, "n_boot": a.n_boot, "anchor_m10_base_per_fold": Z["anchor_m10"].tolist()}
    rep["D1"] = d1_report(Z, a.n_boot)
    rep["D2"] = d2_report(G, a.n_boot)
    rep["D3"] = d3_report(Z, a.n_boot)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
