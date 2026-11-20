"""Section 4M M10: three cheap R-level variants against the 64-dim fused_opp_diff_hd baseline, same questions and protocol as
scripts/hd_check.py (folds 1-4, 32 training questions per player, MLP h32, seeds averaged, class-balanced official score, paired
player bootstrap; conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005).

  pern   per-n standardisation of R (issue #53): mean/std per training question size; columns constant within a size (log_games)
         are left untouched; validation questions use their size's statistics
  headc  the L1b 16 features recomputed from per-move 10-head log p centred by the head mean and clamped to [-2.5, 2.5]
  headw  ... weighted per move by the cross-head std I_t / mean I_t (per game)
  fp     + 5 "fingerprint" / style-consistency columns: std over games of the per-game L1b pmi, mean pairwise L2 distance of the
         per-game HSL curves (mean - max), pairwise agreement of the target's canonical (8-symmetry) first move and first three
         moves, coefficient of variation of the target's move counts

  opphsl   section 4M M14: + 39 columns from the OPPONENT's moves scored under the 18 HSL profiles (results/scores_opp_hd):
           question mean and std over games of the per-game per-move mean log p curve (18 + 18), opponent HSL pmi mean and
           std (2), own HSL pmi - opponent HSL pmi (1);  opphsl_s = the last 3 columns only
  opphsl_hd  section 4M M15: opphsl + 4 columns of the target's IMPLIED displayed rank per game = opponent HSL pmi + the signed HD gap
           (handicap.signed_gap: +HD when the target is the upper hand / W, -HD when the lower / B, 0 even): question mean and std,
           mean over the HD >= 1 games (0 without any), mean over the even games (0 without any). --pairs a:b picks the comparisons.
  oppshift  Gemini 2026-11-05 (module A): + 18 columns from the HD-rule-aligned opponent L1b posteriors: per game the opponent's 10-level
           log-posterior (from its summed log p) shifted by the rule gap so it sits on the target's rank scale, averaged over games (10);
           anchor mean / se, even-game and handicap-game anchor means, Kalman-fused estimate with Gemini's constants, w_opp(n), drift gap
           (own - anchor), anchor span (8). oppshift_s = base + the 18 only; oppshift = opphsl + the 18.
  opphist   GPT round 8 (2026-11-05, mode-label hypothesis): if the label is the MODE of the player's per-game displayed ranks, the target is the
           most frequent matched rank, not the mean. + 20 columns: the ARITHMETIC mean over games of per-game tempered posteriors (sums / 10, so
           a game is a soft vote, not a near-one-hot) of (a) the opponent shifted by the HD rule (10) and (b) the target's own L1b (10) -- soft
           histograms of the per-game rank votes. opphist = opphsl + 20.
  winlm    2026-11-05 result proxy: + 2 columns from "the last mover is the target" (the loser resigns on their turn; results/endgame/
           game_table.csv): the question's win rate, and the mean residual of wins against the logistic of (own - opponent pmi) fitted on
           the training folds' games (the rating disequilibrium). Player-level partial vs y given E and the gap was +0.06 [+0.03, +0.09].

  kitchen   section 4M M31 (2026-11-06): every R family with a positive point estimate against its reference stacked on opphsl:
           opphsl 39 + implied rank 4 (opphsl_hd) + opphist 20 + oppshift 18 + winlm 2 (147 columns); kitchen_nowin drops winlm;
           opphsl_win = opphsl + winlm only. --save-scores writes the per-question scores of every arm (for the cross-fit selection).
  attn20    section 4M M36 (2026-11-06): opphsl + 20 columns, the question mean and std of the cross-fitted spatial-attention heads'
           per-game log-softmax (aicup.attnfeat; --attn-scores, --attn-games).
  autofeat  section 4M M32 (2026-11-06): OpenFE / AutoFeat-style generated features on opphsl's 103 columns (standardised on the training
           questions): x^2, sign log(1+|x|), a*b, a-b, a/(1+|b|) (about 21,000 candidates). Screened inside each outer fold's training
           folds only: the training players split in two halves by a fixed hash, an MLP h32 fit on half A, candidates ranked by |corr|
           with the expected-rank residual y - E[y] on half B; the top 16 kept greedily, skipping any with |corr| >= 0.9 to one kept.

  python -I logs/nano4/r_variants.py --out results/r_variants/m10.json
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
from aicup.opphsl import opp_hsl_features as _opp_hsl_features  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import build_matrix, sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import _part_moves, load_games, read_scored_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

KITCHEN = ("kitchen", "kitchen_nowin", "opphsl_win")  # section 4M M31
AF_K, AF_REDUND = 16, 0.9  # section 4M M32: generated features kept, redundancy threshold


def _af_column(Z, op, i, j):
    x = Z[:, i]
    if op == "sq":
        return x * x
    if op == "slog":
        return np.sign(x) * np.log1p(np.abs(x))
    y = Z[:, j]
    return x * y if op == "mul" else (x - y if op == "sub" else x / (1.0 + np.abs(y)))


def autofeat(Xt, y_tr, pid_tr, Xv, k=AF_K, redund=AF_REDUND, chunk=512):
    """Section 4M M32: (generated training columns [Q_tr, k], validation columns [Q_va, k], the kept specs with their half-B |corr|).
    Only the training questions are looked at: halves by player hash, MLP h32 on half A, |corr| with the residual on half B."""
    import zlib
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    sd[sd == 0] = 1.0
    Zt, Zv = ((Xt - mu) / sd).astype(np.float32), ((Xv - mu) / sd).astype(np.float32)
    half = np.array([zlib.crc32(str(x).encode()) % 2 for x in pid_tr])
    A, B = half == 0, half == 1
    P = MLPModel(hidden=32, seed=0).fit(Xt[A], y_tr[A]).predict_proba(Xt[B])
    r = y_tr[B] - P @ np.arange(P.shape[1])
    r = ((r - r.mean()) / r.std()).astype(np.float32)
    ZB, d = Zt[B], Zt.shape[1]
    specs = [("sq", i, -1) for i in range(d)] + [("slog", i, -1) for i in range(d)] + [(op, i, j) for op in ("mul", "sub") for i in range(d) for j in range(i + 1, d)] \
        + [("div", i, j) for i in range(d) for j in range(d) if i != j]
    score = np.zeros(len(specs), np.float32)
    for lo in range(0, len(specs), chunk):
        C = np.stack([_af_column(ZB, *s) for s in specs[lo:lo + chunk]], axis=1)
        cs = C.std(axis=0)
        ok = cs > 1e-6
        score[lo:lo + chunk][ok] = np.abs(((C[:, ok] - C[:, ok].mean(axis=0)) / cs[ok]).T @ r / len(r))
    kept, cols = [], []
    for s_i in np.argsort(-score):
        c = _af_column(ZB, *specs[s_i])
        c = (c - c.mean()) / (c.std() + 1e-12)
        if all(abs(float(c @ q) / len(c)) < redund for q in cols):
            kept.append((specs[s_i], float(score[s_i]))); cols.append(c)
            if len(kept) == k:
                break
    Ft = np.stack([_af_column(Zt, *s) for s, _ in kept], axis=1)
    Fv = np.stack([_af_column(Zv, *s) for s, _ in kept], axis=1)
    return Ft, Fv, [{"op": s[0], "i": int(s[1]), "j": int(s[2]), "abs_corr_half_b": c} for s, c in kept]
L1B_COLS = slice(24, 40)  # fused_opp_diff_hd layout: HSL 24 | L1b 16 | opp 16 | diff 3 | HD 5
CLAMP = 2.5


def per_n_standardise(X_tr, n_tr, X_va, n_va):
    """Per-size mean/std from the training questions; a column constant within a size keeps its values."""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    stats = {}
    for size in np.unique(n_tr):
        rows = n_tr == size
        mu, sd = X_tr[rows].mean(axis=0), X_tr[rows].std(axis=0)
        keep = sd < 1e-9
        mu[keep], sd[keep] = 0.0, 1.0
        stats[int(size)] = (mu, sd)
        X_tr[rows] = (X_tr[rows] - mu) / sd
    for size in np.unique(n_va):
        rows = n_va == size
        mu, sd = stats.get(int(size), (np.zeros(X_va.shape[1]), np.ones(X_va.shape[1])))
        X_va[rows] = (X_va[rows] - mu) / sd
    return X_tr, X_va


def l1b_variants(l1b, fold_dir):
    """Two copies of the L1b games table whose sum columns are the clamped (headc) and the std-weighted (headw) per-move sums."""
    moves = _part_moves(fold_dir)
    cols = sum_cols(LEVELS)
    c_tab, w_tab = l1b.copy(), l1b.copy()
    c_sums, w_sums = np.zeros((len(l1b), len(cols))), np.zeros((len(l1b), len(cols)))
    for i, key in enumerate(l1b["game_key"]):
        lp = np.asarray(moves[key][1], dtype=np.float64)
        centred = lp - lp.mean(axis=1, keepdims=True)
        c_sums[i] = np.clip(centred, -CLAMP, CLAMP).sum(axis=0)
        it = centred.std(axis=1)
        w = it / max(it.mean(), 1e-9)
        w_sums[i] = (lp * w[:, None]).sum(axis=0)
    c_tab[cols], w_tab[cols] = c_sums, w_sums
    return c_tab, w_tab


SYMS = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y),
        lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]


def canonical(seq):
    """The lexicographically smallest image of a move sequence under the 8 board symmetries (a pass is (-1, -1))."""
    best = None
    for s in SYMS:
        img = tuple(s(x, y) if x is not None else (-1, -1) for x, y in seq)
        if best is None or img < best:
            best = img
    return best


def opening_keys(games):
    """{game_key: (canonical first target move, canonical first three target moves, target move count)} from the SGFs."""
    sgf = load_games(list(games))
    out = {}
    for k, (text, color) in sgf.items():
        _, moves = setup_and_moves(text)
        own = [(x, y) for c, x, y in moves if c == color]
        out[k] = (canonical(own[:1]), canonical(own[:3]), len(own))
    return out


def fingerprint(hsl, l1b, questions, opening):
    """[Q, 5]: pmi_std, curve_pairdist, first1_agree, first3_agree, nmoves_cv."""
    hcols, lcols = sum_cols(l1b_cv.PROFILES) if hasattr(l1b_cv, "PROFILES") else [c for c in hsl.columns if c.startswith("sum_")], sum_cols(LEVELS)
    h = hsl.set_index("game_key")
    curves = h[hcols].to_numpy(dtype=float) / np.maximum(h["n_moves"].to_numpy(dtype=float), 1.0)[:, None]
    curves = curves - curves.max(axis=1, keepdims=True)
    hrow = {k: i for i, k in enumerate(h.index)}
    li = l1b.set_index("game_key")
    pmi = posterior_mean_index(li[lcols].to_numpy(dtype=float))
    lrow = {k: i for i, k in enumerate(li.index)}
    out = np.zeros((len(questions), 5))

    def agree(labels):  # share of game pairs with the same label
        _, counts = np.unique(np.array([repr(x) for x in labels]), return_counts=True)  # one string per game (a tuple would be flattened)
        k = len(labels)
        return float((counts * (counts - 1) / 2).sum() / (k * (k - 1) / 2)) if k > 1 else 0.0

    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys)
        out[qi, 0] = pmi[[lrow[k] for k in keys]].std()
        C = curves[[hrow[k] for k in keys]]
        D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(keys), 1)
        out[qi, 1] = D[iu].mean() if len(iu[0]) else 0.0
        o = [opening[k] for k in keys]
        out[qi, 2] = agree([x[0] for x in o])
        out[qi, 3] = agree([x[1] for x in o])
        nm = np.array([x[2] for x in o], dtype=float)
        out[qi, 4] = nm.std() / max(nm.mean(), 1.0)
    return out

def opp_hsl_features(hsl, opp_hsl, questions, summary_only=False):
    """[Q, 39] (or [Q, 3]: the last three columns) opponent-HSL question features (aicup.opphsl)."""
    out = _opp_hsl_features(hsl, opp_hsl, questions)
    return out[:, -3:] if summary_only else out


def implied_rank_features(opp_hsl, hd, questions):
    """[Q, 4] (section 4M M15): per game the opponent's HSL pmi + the signed HD gap = the target's implied displayed rank (profile index
    units, one profile = one rank); per question its mean and std over games, its mean over the HD >= 1 games (0 when there are none)
    and over the even games (0 when there are none)."""
    hcols = [c for c in opp_hsl.columns if c.startswith("sum_")]
    o = opp_hsl.set_index("game_key")
    pmi = pd.Series(posterior_mean_index(o[hcols].to_numpy(dtype=float)), index=o.index)
    h = hd.drop_duplicates("game_key").set_index("game_key")
    gap = pd.Series(signed_gap(h["hd"].to_numpy(), h["color"].to_numpy()), index=h.index)
    out = np.zeros((len(questions), 4))
    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys)
        imp = pmi.loc[keys].to_numpy() + gap.reindex(keys).fillna(0.0).to_numpy()
        ishd = gap.reindex(keys).fillna(0.0).to_numpy() != 0
        out[qi] = [imp.mean(), imp.std(), imp[ishd].mean() if ishd.any() else 0.0, imp[~ishd].mean() if (~ishd).any() else 0.0]
    return out


def oppshift_features(l1b, opp, hd, questions, sp2=0.45, sgo2=1.2, sga2=1.2, shift=True):
    """[Q, 18] (Gemini module A, 2026-11-05): HD-rule-aligned opponent L1b posteriors and anchor statistics."""
    lc = [f"sum_{lv}" for lv in LEVELS]
    o = opp.set_index("game_key"); own = l1b.set_index("game_key")
    h = hd.drop_duplicates("game_key").set_index("game_key")
    gap = pd.Series(signed_gap(h["hd"].to_numpy(), h["color"].to_numpy()), index=h.index)  # + when the target is the upper hand
    So = o[lc].to_numpy(float); Lo = So - np.logaddexp.reduce(So, axis=1, keepdims=True)  # per-game opponent log-posterior over the 10 levels
    e_opp = pd.Series(posterior_mean_index(So), index=o.index); e_own = pd.Series(posterior_mean_index(own[lc].to_numpy(float)), index=own.index)
    Lo = pd.DataFrame(Lo, index=o.index)
    out = np.zeros((len(questions), 18))
    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys); n = len(keys)
        gp = gap.reindex(keys).fillna(0.0).to_numpy(); eo = e_own.loc[keys].to_numpy(); ep = e_opp.loc[keys].to_numpy()
        if not shift:  # GPT's control ladder: the opponent evidence aggregated without the HD rule (direction and magnitude dropped)
            gp = np.zeros_like(gp)
        anchor = ep + gp
        L = Lo.loc[keys].to_numpy(); sh = np.zeros_like(L)
        for i, k in enumerate(np.rint(gp).astype(int)):  # shift the opponent's posterior up by the gap: index j of the opponent -> j + gap of the target
            for j in range(10):
                sh[i, min(max(j + k, 0), 9)] = np.logaddexp(sh[i, min(max(j + k, 0), 9)], L[i, j]) if sh[i, min(max(j + k, 0), 9)] != 0 else L[i, j]
        sh[sh == 0] = -30.0
        hd_mask = gp != 0
        var_own = sp2 + sgo2 / n; var_anc = (sp2 + sga2) / n; w = var_own / (var_own + var_anc)
        out[qi, :10] = sh.mean(axis=0)
        out[qi, 10:] = [anchor.mean(), anchor.std() / np.sqrt(n), anchor[~hd_mask].mean() if (~hd_mask).any() else anchor.mean(), anchor[hd_mask].mean() if hd_mask.any() else anchor.mean(),
                        (1 - w) * eo.mean() + w * anchor.mean(), w, eo.mean() - anchor.mean(), anchor.max() - anchor.min()]
    return out


def opphist_features(l1b, opp, hd, questions, temp=10.0):
    """[Q, 20]: soft histograms (arithmetic means of tempered per-game posteriors) of the HD-aligned opponent votes and of the own votes."""
    lc = [f"sum_{lv}" for lv in LEVELS]
    o = opp.set_index("game_key"); own = l1b.set_index("game_key"); h = hd.drop_duplicates("game_key").set_index("game_key")
    gap = pd.Series(signed_gap(h["hd"].to_numpy(), h["color"].to_numpy()), index=h.index)
    def tempered(S, n):
        L = S / np.maximum(n, 1.0)[:, None] * temp; L = L - L.max(axis=1, keepdims=True); P = np.exp(L); return P / P.sum(axis=1, keepdims=True)
    Po = pd.DataFrame(tempered(o[lc].to_numpy(float), o["n_moves"].to_numpy(float)), index=o.index)
    Pw = pd.DataFrame(tempered(own[lc].to_numpy(float), own["n_moves"].to_numpy(float)), index=own.index)
    out = np.zeros((len(questions), 20))
    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys); gp = np.rint(gap.reindex(keys).fillna(0.0).to_numpy()).astype(int)
        A = Po.loc[keys].to_numpy(); sh = np.zeros_like(A)
        for i, k in enumerate(gp):
            for j in range(10):
                sh[i, min(max(j + k, 0), 9)] += A[i, j]
        out[qi, :10] = sh.mean(axis=0); out[qi, 10:] = Pw.loc[keys].to_numpy().mean(axis=0)
    return out


def winlm_features(questions, lm, gap, slope):
    """[Q, 2]: win rate by the last-mover proxy, and the mean residual of wins against the logistic of the per-game strength gap."""
    out = np.zeros((len(questions), 2))
    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys); w = lm.reindex(keys).to_numpy(dtype=float); g = gap.reindex(keys).to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(g)
        if ok.any():
            p = 1 / (1 + np.exp(-(slope[0] + slope[1] * g[ok])))
            out[qi] = [w[ok].mean(), (w[ok] - p).mean()]
    return out


def fit_logit(x, w):
    b = np.zeros(2)
    for _ in range(50):
        z = b[0] + b[1] * x; p = 1 / (1 + np.exp(-z)); W = p * (1 - p) + 1e-9
        X = np.column_stack([np.ones_like(x), x]); H = X.T @ (X * W[:, None]); grad = X.T @ (w - p)
        b = b + np.linalg.solve(H + 1e-6 * np.eye(2), grad)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["pern", "headc", "headw", "fp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--pairs", nargs="*", default=None, help="comparisons a:b over the arms (default: every variant vs base)")
    ap.add_argument("--attn-scores", default=str(ROOT / "results/attn_scores"), help="section 4M M36: the score parts of e2e_pilot.py score")
    ap.add_argument("--attn-games", default=str(ROOT / "results/r_variants/attn_games.csv"))
    ap.add_argument("--save-scores", default=None, help="npz of the per-question scores of every arm, with y, player_id, n_games and fold")
    a = ap.parse_args()
    t0 = time.time()
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    data = {}
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
        players = l1b_cv.players_from_scores(hsl, folds_df)
        d = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}
        if "headc" in a.variants or "headw" in a.variants:
            d["l1b_c"], d["l1b_w"] = l1b_variants(l1b, ROOT / "results/l1b" / ts.L1B[f] / f"fold{f}")
        if "fp" in a.variants:
            d["opening"] = opening_keys(hsl["game_key"])
        if any(v in a.variants for v in ("opphsl", "opphsl_s", "opphsl_hd", "oppshift", "opphist", "autofeat", "attn20") + KITCHEN):
            d["opp_hsl"] = read_scored_games(ROOT / "results/scores_opp_hd" / f"fold{f}")
            if set(d["opp_hsl"]["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: opponent HSL scores cover different games than the HSL scores")
        if any(v in a.variants for v in ("winlm", "kitchen", "opphsl_win")):
            eg = pd.read_csv(ROOT / "results/endgame/game_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")
            eg = eg.reindex(hsl["game_key"])
            d["lm"] = eg["last_mover_is_target"].astype(float); d["gap"] = (eg["own_pmi"] - eg["opp_pmi"]).astype(float)
        data[f] = d
        print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(d['val'])} validation questions ({time.time() - t0:.0f} s)", flush=True)
    names = ["base"] + a.variants
    scores, per_fold, ys, pids, ns = {k: [] for k in names}, {k: [] for k in names}, [], [], []
    af_log = {}  # section 4M M32: the generated features kept per outer fold
    attn_scores = None
    if "attn20" in a.variants:  # section 4M M36
        from aicup.attnfeat import load_scores
        attn_scores = load_scores(a.attn_scores, a.attn_games)
        print(f"attn20: {len(attn_scores)} scored games", flush=True)
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, n_tr, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        B_tr, B_va = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        n_tr = train_q["game_keys"].map(len).to_numpy()
        variants = {"base": (B_tr, B_va)}
        if "pern" in a.variants:
            variants["pern"] = per_n_standardise(B_tr, n_tr, B_va, n_va)
        for name, key in (("headc", "l1b_c"), ("headw", "l1b_w")):
            if name in a.variants:
                Xt, Xv = B_tr.copy(), B_va.copy()
                Xt[:, L1B_COLS] = build_matrix(cat(key), train_q, LEVELS)[0]
                Xv[:, L1B_COLS] = build_matrix(d[key], d["val"], LEVELS)[0]
                variants[name] = (Xt, Xv)
        if "fp" in a.variants:
            opening = {}
            for f in train:
                opening.update(data[f]["opening"])
            Ft = fingerprint(cat("hsl"), cat("l1b"), train_q, opening)
            Fv = fingerprint(d["hsl"], d["l1b"], d["val"], d["opening"])
            variants["fp"] = (np.hstack([B_tr, Ft]), np.hstack([B_va, Fv]))
        for name, summ in (("opphsl", False), ("opphsl_s", True)):
            if name in a.variants:
                Ft = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q, summ)
                Fv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"], summ)
                variants[name] = (np.hstack([B_tr, Ft]), np.hstack([B_va, Fv]))
        if "opphsl_hd" in a.variants:
            Ft = np.hstack([opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q), implied_rank_features(cat("opp_hsl"), cat("hd"), train_q)])
            Fv = np.hstack([opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"]), implied_rank_features(d["opp_hsl"], d["hd"], d["val"])])
            variants["opphsl_hd"] = (np.hstack([B_tr, Ft]), np.hstack([B_va, Fv]))
        if "oppshift_nod" in a.variants:  # GPT 2026-11-05 control: same 18 columns with the HD shift dropped
            Fn_t = oppshift_features(cat("l1b"), cat("opp"), cat("hd"), train_q, shift=False); Fn_v = oppshift_features(d["l1b"], d["opp"], d["hd"], d["val"], shift=False)
            variants["oppshift_nod"] = (np.hstack([B_tr, Fn_t]), np.hstack([B_va, Fn_v]))
        if "oppshift" in a.variants or "oppshift_s" in a.variants:
            Ft = oppshift_features(cat("l1b"), cat("opp"), cat("hd"), train_q); Fv = oppshift_features(d["l1b"], d["opp"], d["hd"], d["val"])
            if "oppshift_s" in a.variants:
                variants["oppshift_s"] = (np.hstack([B_tr, Ft]), np.hstack([B_va, Fv]))
            if "oppshift" in a.variants:
                Ht = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q); Hv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])
                variants["oppshift"] = (np.hstack([B_tr, Ht, Ft]), np.hstack([B_va, Hv, Fv]))
        if "opphist" in a.variants:
            Ht = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q); Hv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])
            Gt = opphist_features(cat("l1b"), cat("opp"), cat("hd"), train_q); Gv = opphist_features(d["l1b"], d["opp"], d["hd"], d["val"])
            variants["opphist"] = (np.hstack([B_tr, Ht, Gt]), np.hstack([B_va, Hv, Gv]))
        if "winlm" in a.variants:
            lm_tr = pd.concat([data[f]["lm"] for f in train]); gap_tr = pd.concat([data[f]["gap"] for f in train])
            ok = lm_tr.notna() & gap_tr.notna(); slope = fit_logit(gap_tr[ok].to_numpy(float), lm_tr[ok].to_numpy(float))
            Ft = winlm_features(train_q, lm_tr, gap_tr, slope); Fv = winlm_features(d["val"], d["lm"], d["gap"], slope)
            variants["winlm"] = (np.hstack([B_tr, Ft]), np.hstack([B_va, Fv]))
        if any(k in a.variants for k in KITCHEN):  # section 4M M31: the positive families stacked on opphsl
            Ht = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q); Hv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])
            blocks_t, blocks_v = [B_tr, Ht], [B_va, Hv]
            if "kitchen" in a.variants or "kitchen_nowin" in a.variants:
                blocks_t += [implied_rank_features(cat("opp_hsl"), cat("hd"), train_q), opphist_features(cat("l1b"), cat("opp"), cat("hd"), train_q),
                             oppshift_features(cat("l1b"), cat("opp"), cat("hd"), train_q)]
                blocks_v += [implied_rank_features(d["opp_hsl"], d["hd"], d["val"]), opphist_features(d["l1b"], d["opp"], d["hd"], d["val"]),
                             oppshift_features(d["l1b"], d["opp"], d["hd"], d["val"])]
            if "kitchen" in a.variants or "opphsl_win" in a.variants:
                lm_tr = pd.concat([data[f]["lm"] for f in train]); gap_tr = pd.concat([data[f]["gap"] for f in train])
                ok = lm_tr.notna() & gap_tr.notna(); slope = fit_logit(gap_tr[ok].to_numpy(float), lm_tr[ok].to_numpy(float))
                Wt, Wv = winlm_features(train_q, lm_tr, gap_tr, slope), winlm_features(d["val"], d["lm"], d["gap"], slope)
            if "kitchen" in a.variants:
                variants["kitchen"] = (np.hstack(blocks_t + [Wt]), np.hstack(blocks_v + [Wv]))
            if "kitchen_nowin" in a.variants:
                variants["kitchen_nowin"] = (np.hstack(blocks_t), np.hstack(blocks_v))
            if "opphsl_win" in a.variants:
                variants["opphsl_win"] = (np.hstack([B_tr, Ht, Wt]), np.hstack([B_va, Hv, Wv]))
            print(f"fold {v}: kitchen columns " + ", ".join(f"{k} {variants[k][0].shape[1]}" for k in KITCHEN if k in variants), flush=True)
        if "attn20" in a.variants:  # section 4M M36: cross-fitted spatial-attention per-game log-softmax, question mean and std
            from aicup.attnfeat import question_features
            Ht = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q); Hv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])
            variants["attn20"] = (np.hstack([B_tr, Ht, question_features(train_q["game_keys"], attn_scores)]), np.hstack([B_va, Hv, question_features(d["val"]["game_keys"], attn_scores)]))
        if "autofeat" in a.variants:  # section 4M M32: generated features screened on this fold's training folds only
            Ht = opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q); Hv = opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])
            Xo_t, Xo_v = np.hstack([B_tr, Ht]), np.hstack([B_va, Hv])
            Ft, Fv, kept = autofeat(Xo_t, y_tr, train_q["player_id"].to_numpy(dtype=str), Xo_v)
            af_log[str(v)] = kept
            variants["autofeat"] = (np.hstack([Xo_t, Ft]), np.hstack([Xo_v, Fv]))
            print(f"fold {v}: autofeat kept " + ", ".join(f"{q['op']}({q['i']},{q['j']}) {q['abs_corr_half_b']:.3f}" for q in kept), flush=True)
        for name, (Xt, Xv) in variants.items():
            P = np.mean([MLPModel(hidden=32, seed=s).fit(Xt, y_tr).predict_proba(Xv) for s in a.seeds], axis=0)
            res, sc = evaluate(P, y_va, d["val"])
            scores[name].append(sc)
            per_fold[name].append(res["balanced_mean_over_n"])
        ys.append(y_va); pids.append(d["val"]["player_id"].to_numpy(dtype=str)); ns.append(n_va)
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in names) + f"  ({time.time() - t0:.0f} s)", flush=True)
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
    S = {k: np.concatenate(scores[k]) for k in names}
    rep = {"folds": a.folds, "seeds": a.seeds, "arms": {}, "pairs": {}, **({"autofeat": af_log} if af_log else {})}
    for k in names:
        rep["arms"][k] = {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": {str(m): balanced_mean_over_n(S[k], y, n, sizes=(m,)) for m in SIZES}}
        print(f"{k:6s} {rep['arms'][k]['score']:.4f}  folds {[round(x, 4) for x in per_fold[k]]}  by n " + " ".join(f"{m}: {rep['arms'][k]['by_n'][str(m)]:.4f}" for m in SIZES))
    need = max(1, len(a.folds) - 1)
    pairs = [tuple(x.split(":")) for x in a.pairs] if a.pairs else [(k, "base") for k in a.variants]
    for k, ref in pairs:
        dd = paired_diff(S[k], S[ref], y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[k], per_fold[ref])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= 0.005)}
        rep["pairs"][f"{k} vs {ref}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{k} vs {ref}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    if a.save_scores:
        fold = np.concatenate([np.full(len(x), v) for v, x in zip(a.folds, ys)])
        np.savez(a.save_scores, y=y, player_id=pid, n_games=n, fold=fold, **{f"S_{k}": S[k] for k in names})
        print("written", a.save_scores)
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
