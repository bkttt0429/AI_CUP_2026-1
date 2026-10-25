"""The L1b early signal (plan 2026-10-25-l1b-rank-heads, Task 6): L1b against Human SL on the same moves.

Both scorings are in scripts/score_folds.py's layout (games_s*_p*.csv + moves_s*_p*.npz per fold).
align_moves reads a fold from both directories and refuses anything but the same games with the same
moves (game_key set, n_moves and move indices t per game), so every comparison is paired move by move.
per_rank_nll gives each rank's mean per-move NLL under its own L1b head and under Human SL's best (and
pilot-peak) profile; question_sums turns per-move log p into per-question sums (one 20-game question
per player with exactly 20 scored games, as the pilot's n = 20 questions) and adjacent_auc is the
pilot's §2.4.1(d) statistic: posterior-mean index of the summed curve, Mann-Whitney AUC of each pair of
adjacent ranks (P(the higher rank's index is higher)).
"""
import numpy as np
import pandas as pd

from aicup.evidence import auc, posterior_mean_index
from aicup.humansl import PROFILES
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.scoring import completed_parts, part_paths

# §2.4.1(a): the Human SL profile where each rank's likelihood curve peaks (KGS scale compressed)
PILOT_PEAK = {"D": "6k", "C": "5k", "B": "4k", "A": "3k", "1D": "2k", "2D": "2k", "3D": "2k", "4D": "1k", "5D": "1d", "6D": "2d"}
# §2.4.1(d), n = 20 questions, the adjacent-rank AUCs of the frozen Human SL pilot (fold 0, 20 players per rank)
PILOT_AUC_REFERENCE = {"D|C": 0.87, "C|B": 0.77, "B|A": 0.79, "A|1D": 0.88, "1D|2D": 0.62, "2D|3D": 0.73, "3D|4D": 0.65,
                       "4D|5D": 0.86, "5D|6D": 0.89}
ADJACENT_PAIRS = [f"{a}|{b}" for a, b in zip(LEVELS, LEVELS[1:])]


def read_moves(fold_dir):
    """(games, logp [M, S] float32, game_row [M] into games, t [M]) of every finished part of one fold, sorted by
    (game_row, t). FileNotFoundError when the fold has no finished part."""
    parts = completed_parts(fold_dir)
    if not parts:
        raise FileNotFoundError(f"no finished parts in {fold_dir}")
    frames, logps, rows, ts, offset = [], [], [], [], 0
    for s, k in parts:
        csv_path, npz_path = part_paths(fold_dir, s, k)
        games = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
        with np.load(npz_path, allow_pickle=False) as z:
            logps.append(z["logp"].astype(np.float32))
            rows.append(z["game_row"].astype(np.int64) + offset)
            ts.append(z["t"].astype(np.int64))
        offset += len(games)
        frames.append(games)
    games = pd.concat(frames, ignore_index=True)
    if not games["game_key"].is_unique:
        raise ValueError(f"{fold_dir}: {int(games['game_key'].duplicated().sum())} games scored more than once")
    logp, game_row, t = np.concatenate(logps), np.concatenate(rows), np.concatenate(ts)
    order = np.lexsort((t, game_row))
    return games, logp[order], game_row[order], t[order]


def align_moves(l1b_dir, hsl_dir, fold):
    """(games, logp_l1b [M, 10], logp_hsl [M, 18], game_row [M], t [M]) of fold `fold` scored in both directories,
    in the L1b games' order. ValueError unless both have exactly the same games with the same moves."""
    games, lp_a, row_a, t_a = read_moves(f"{l1b_dir}/fold{fold}")
    games_b, lp_b, row_b, t_b = read_moves(f"{hsl_dir}/fold{fold}")
    keys_a, keys_b = set(games["game_key"]), set(games_b["game_key"])
    if keys_a != keys_b:
        raise ValueError(f"fold {fold}: the two scorings cover different games ({len(keys_a - keys_b)} only in {l1b_dir}, "
                         f"{len(keys_b - keys_a)} only in {hsl_dir}; e.g. {sorted(keys_a ^ keys_b)[:3]})")
    if lp_a.shape[1] != len(LEVELS) or lp_b.shape[1] != len(PROFILES):
        raise ValueError(f"fold {fold}: expected {len(LEVELS)} L1b heads and {len(PROFILES)} Human SL profiles, got {lp_a.shape[1]} and {lp_b.shape[1]}")
    # map the Human SL rows onto the L1b game order
    pos_b = {k: i for i, k in enumerate(games_b["game_key"])}
    remap = np.array([pos_b[k] for k in games["game_key"]], dtype=np.int64)  # L1b row -> HSL row
    n_a = np.bincount(row_a, minlength=len(games))
    n_b = np.bincount(row_b, minlength=len(games_b))[remap]
    if not np.array_equal(n_a, n_b):
        bad = games["game_key"][np.flatnonzero(n_a != n_b)[:3]].tolist()
        raise ValueError(f"fold {fold}: {int((n_a != n_b).sum())} games have different numbers of scored moves, e.g. {bad}")
    inv = np.empty(len(games_b), dtype=np.int64)
    inv[remap] = np.arange(len(games))  # HSL row -> L1b row
    row_b_in_a = inv[row_b]
    order = np.lexsort((t_b, row_b_in_a))
    lp_b, row_b_in_a, t_b = lp_b[order], row_b_in_a[order], t_b[order]
    if not np.array_equal(row_b_in_a, row_a) or not np.array_equal(t_b, t_a):
        diff = np.flatnonzero((row_b_in_a != row_a) | (t_b != t_a))[:1]
        where = games["game_key"][row_a[diff[0]]] if len(diff) else "?"
        raise ValueError(f"fold {fold}: the move indices differ between the two scorings (first at game {where})")
    return games, lp_a, lp_b, row_a, t_a


def per_rank_nll(logp_l1b, logp_hsl, rank_idx):
    """Mean per-move NLL by true rank: {"l1b": [10] (own head), "hsl_best": [10] (lowest-NLL profile of the rank),
    "hsl_best_profile": [10 names], "hsl_pilot_peak": [10] (PILOT_PEAK profile), "n_moves": [10]}; None where a
    rank has no moves."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)
    out = {"l1b": [], "hsl_best": [], "hsl_best_profile": [], "hsl_pilot_peak": [], "n_moves": []}
    for k, lv in enumerate(LEVELS):
        sel = rank_idx == k
        n = int(sel.sum())
        out["n_moves"].append(n)
        if n == 0:
            for key in ("l1b", "hsl_best", "hsl_best_profile", "hsl_pilot_peak"):
                out[key].append(None)
            continue
        out["l1b"].append(float(-logp_l1b[sel, k].mean()))
        hsl = -np.asarray(logp_hsl[sel], dtype=np.float64).mean(axis=0)
        j = int(np.argmin(hsl))
        out["hsl_best"].append(float(hsl[j]))
        out["hsl_best_profile"].append(PROFILES[j])
        out["hsl_pilot_peak"].append(float(hsl[PROFILES.index(PILOT_PEAK[lv])]))
    return out


def game_sums(games, logp, game_row):
    """Per game the summed log p [G, S] and move count [G]."""
    sums = np.zeros((len(games), logp.shape[1]), dtype=np.float64)
    np.add.at(sums, game_row, logp.astype(np.float64))
    return sums, np.bincount(game_row, minlength=len(games))


def question_sums(games, logp, game_row, n_games=20):
    """(sums [Q, S], rank_idx [Q], n_moves [Q]) of one question per player with exactly n_games scored games (the
    pilot's n = 20 questions); players with another number of games are left out."""
    g_sums, g_n = game_sums(games, logp, game_row)
    sums, ranks, n_moves = [], [], []
    for (pid, rank), idx in games.groupby(["player_id", "rank"], sort=True).indices.items():
        if len(idx) != n_games:
            continue
        sums.append(g_sums[idx].sum(axis=0))
        ranks.append(RANK_TO_IDX[rank])
        n_moves.append(int(g_n[idx].sum()))
    if not sums:
        return np.zeros((0, logp.shape[1])), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.stack(sums), np.asarray(ranks, dtype=np.int64), np.asarray(n_moves, dtype=np.int64)


def adjacent_auc(sums, rank_idx):
    """{pair: AUC} for every adjacent pair of ranks: posterior-mean index of the summed curve (softmax over heads or
    profiles), P(index of the higher rank > index of the lower) with ties half; None when a rank has no rows."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)
    pmi = posterior_mean_index(np.asarray(sums, dtype=np.float64)) if len(rank_idx) else np.zeros(0)
    out = {}
    for k, pair in enumerate(ADJACENT_PAIRS):
        lo, hi = pmi[rank_idx == k], pmi[rank_idx == k + 1]
        out[pair] = float(auc(hi, lo)) if len(lo) and len(hi) else None
    return out


def early_signal(l1b_dir, hsl_dir, folds):
    """The whole report over several folds (moves and questions pooled): per-rank NLL, adjacent-rank AUC on 20-game
    questions and on single games, for L1b and Human SL, with the pilot reference."""
    lp_a, lp_b, ranks = [], [], []
    q_sums = {"l1b": [], "hsl": []}
    q_ranks = []
    g_sums = {"l1b": [], "hsl": []}
    g_ranks = []
    players_by_rank = {lv: 0 for lv in LEVELS}
    for fold in folds:
        games, a, b, game_row, _ = align_moves(l1b_dir, hsl_dir, fold)
        rank_idx = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64)
        lp_a.append(a)
        lp_b.append(b)
        ranks.append(rank_idx[game_row])
        sa, ra, _ = question_sums(games, a, game_row)
        sb, rb, _ = question_sums(games, b, game_row)
        assert np.array_equal(ra, rb)
        q_sums["l1b"].append(sa)
        q_sums["hsl"].append(sb)
        q_ranks.append(ra)
        for k in ra:
            players_by_rank[LEVELS[k]] += 1
        ga, _ = game_sums(games, a, game_row)
        gb, _ = game_sums(games, b, game_row)
        g_sums["l1b"].append(ga)
        g_sums["hsl"].append(gb)
        g_ranks.append(rank_idx)
    rank_moves = np.concatenate(ranks)
    nll = per_rank_nll(np.concatenate(lp_a), np.concatenate(lp_b), rank_moves)
    qr, gr = np.concatenate(q_ranks), np.concatenate(g_ranks)
    return {"folds": [int(f) for f in folds], "l1b_dir": str(l1b_dir), "hsl_dir": str(hsl_dir), "n_moves": int(len(rank_moves)),
            "nll": nll, "n_players_by_rank": players_by_rank,
            "auc_question20": {src: adjacent_auc(np.concatenate(q_sums[src]), qr) for src in ("l1b", "hsl")},
            "auc_game": {src: adjacent_auc(np.concatenate(g_sums[src]), gr) for src in ("l1b", "hsl")},
            "pilot_reference": dict(PILOT_AUC_REFERENCE)}


def format_report(res):
    """The two tables as text."""
    lines = ["per-move NLL by true rank (L1b: own head; Human SL: the rank's best of 18 profiles, and the pilot-peak profile)",
             f"{'rank':>4s} {'moves':>8s} {'l1b':>8s} {'hsl_best':>9s} {'profile':>7s} {'peak':>8s} {'l1b-best':>9s}"]
    nll = res["nll"]
    for k, lv in enumerate(LEVELS):
        if nll["l1b"][k] is None:
            lines.append(f"{lv:>4s} {0:>8d} {'-':>8s}")
            continue
        lines.append(f"{lv:>4s} {nll['n_moves'][k]:>8d} {nll['l1b'][k]:>8.4f} {nll['hsl_best'][k]:>9.4f} {nll['hsl_best_profile'][k]:>7s} "
                     f"{nll['hsl_pilot_peak'][k]:>8.4f} {nll['l1b'][k] - nll['hsl_best'][k]:>+9.4f}")
    lines += ["", "adjacent-rank AUC of the posterior-mean index (20-game questions; single games in brackets); pilot n=20 reference",
              f"{'pair':>6s} {'l1b':>7s} {'hsl':>7s} {'l1b-hsl':>8s} {'pilot':>6s} {'(l1b':>7s} {'hsl)':>7s}"]
    for pair in ADJACENT_PAIRS:
        a, b = res["auc_question20"]["l1b"][pair], res["auc_question20"]["hsl"][pair]
        ga, gb = res["auc_game"]["l1b"][pair], res["auc_game"]["hsl"][pair]
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        d = "-" if a is None or b is None else f"{a - b:+.3f}"
        lines.append(f"{pair:>6s} {f(a):>7s} {f(b):>7s} {d:>8s} {res['pilot_reference'].get(pair, float('nan')):>6.2f} {f(ga):>7s} {f(gb):>7s}")
    return "\n".join(lines)
