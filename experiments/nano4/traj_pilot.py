"""Architecture v2 section 4L.1 part B: win-rate trajectory statistics (hypothesis H2) from humanv0's value and score heads, on a
pilot of 50 players x 20 games per rank (folds 1-4), two metadata variants: E1n = rank-unknown (what the trunk pass sees, free at test
time) and E1s = 9d vs 9d (the strong-human judge of quality_score.py). Every position before a move and the terminal position are
evaluated; v_t and lead_t are turned to the target's view; Δ of a ply = v_{t+1} - v_t. Twelve pre-specified per-game statistics for the
target and four for the opponent (section 4L.1), then the step-0 signal report (pilot_stats).

  sbatch -c 12 -t 03:00:00 ... scripts/nano4/run.sbatch python -I logs/nano4/traj_pilot.py --workers 10
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs" / "nano4")]
from katago.game.board import Board  # noqa: E402
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import load_humansl, meta_rows  # noqa: E402
from aicup.l1b import neutral_rows  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.pilot import start_state  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import load_games, read_scored_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402
import pilot_stats as st  # noqa: E402
import trunk_extract as te  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

OUT = ROOT / "results" / "pilot"
PLAYERS_PER_RANK, GAMES_PER_PLAYER = 50, 20
BLUNDER, DIP, OPP_GIFT, KEEP, PHASES, DECISIVE = 0.10, 0.05, 0.10, 0.02, ((0, 50), (50, 150), (150, 10**6)), 0.40
VARIANTS = ("n", "s")
_FEATURES = None


def init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def replay_full(task):
    """Inputs before every ply and at the terminal position: (key, bins [T+1], globs [T+1], to_move [T+1] 0/1, target [T] bool)."""
    key, sgf, color = task
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, te.RULES)
    bins, globs, to_move, target = [], [], [], []
    for c, x, y in moves:
        b, g = gs.get_input_features(_FEATURES)
        bins.append(b[0].astype(np.uint8))
        globs.append(g[0].astype(np.float32))
        to_move.append(0 if gs.board.pla == Board.BLACK else 1)
        target.append(c == color)
        gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))
    b, g = gs.get_input_features(_FEATURES)
    bins.append(b[0].astype(np.uint8))
    globs.append(g[0].astype(np.float32))
    to_move.append(0 if gs.board.pla == Board.BLACK else 1)
    return key, np.stack(bins), np.stack(globs), np.asarray(to_move, np.int8), np.asarray(target, bool), 0 if color == "B" else 1


def evaluate(model, bins, globs, to_move, rows, batch=256):
    """(win probability of the side to move [N], lead for the side to move [N]) under metadata rows [2, 192] indexed by the side to move."""
    device = next(model.parameters()).device
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device=device)
    win, lead = np.empty(len(to_move), np.float32), np.empty(len(to_move), np.float32)
    with torch.no_grad():
        for i in range(0, len(to_move), batch):
            j = min(len(to_move), i + batch)
            b = torch.as_tensor(bins[i:j], device=device).float()
            g = torch.as_tensor(globs[i:j], device=device).float()
            m = rows_t[torch.as_tensor(to_move[i:j], device=device).long()]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b, g, input_meta=m)
            post = model.postprocess_output(out)[0]
            win[i:j] = torch.softmax(post[1].float(), dim=1)[:, 0].cpu().numpy()  # endgame_proxy.heads: value {win, loss, noresult}
            lead[i:j] = post[10].float().cpu().numpy()
    return win, lead


def target_view(win, lead, to_move, target_color):
    mine = to_move == target_color
    return np.where(mine, win, 1.0 - win), np.where(mine, lead, -lead)


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag + 4:
        return np.nan
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def traj_stats(v, lead, target, suffix):
    """The section-4L.1 statistics of one game: v, lead [T+1] in the target's view, target [T] flags of the plies."""
    T = len(target)
    d = np.diff(v)  # d[t] = change caused by ply t
    dl = np.diff(lead)
    own, oppd = d[target], d[~target]
    o = {}
    o["mean_down"] = float(own.mean()) if len(own) else np.nan
    o["sd_down"] = float(own.std()) if len(own) > 1 else np.nan
    o["q10_down"] = float(np.quantile(own, 0.1)) if len(own) else np.nan
    o["blunder"] = float((own < -BLUNDER).mean()) if len(own) else np.nan
    o["ac1_down"], o["ac2_down"] = autocorr(own, 1), autocorr(own, 2)
    dips = np.flatnonzero(own < -DIP)
    dips = dips[dips + 3 < len(own)]
    o["recover"] = float(np.mean([own[j + 1:j + 4].sum() >= -own[j] for j in dips])) if len(dips) else np.nan
    gifts = np.flatnonzero((~target) & (d > OPP_GIFT))
    gifts = gifts[(gifts + 1 < T)]
    gifts = gifts[target[gifts + 1]]
    o["keep"] = float(np.mean([d[t + 1] >= -KEEP for t in gifts])) if len(gifts) else np.nan
    o["giveback"] = float(np.mean([d[t + 1] <= -0.5 * d[t] for t in gifts])) if len(gifts) else np.nan
    tt = np.arange(T + 1)
    for p, (lo, hi) in enumerate(PHASES, start=1):
        seg = v[(tt >= lo) & (tt < hi)]
        o[f"sd_v_p{p}"] = float(seg.std()) if len(seg) >= 5 else np.nan
    dec = np.flatnonzero(np.abs(v - 0.5) > DECISIVE)
    o["decisive_t"] = float(dec[0] / (T + 1)) if len(dec) else 1.0
    o["v_final"] = float(v[-1])
    o["mean_dlead"] = float(dl[target].mean()) if target.any() else np.nan
    x = -oppd  # the opponent's own change
    o["opp_mean"] = float(x.mean()) if len(x) else np.nan
    o["opp_sd"] = float(x.std()) if len(x) > 1 else np.nan
    o["opp_q10"] = float(np.quantile(x, 0.1)) if len(x) else np.nan
    o["opp_blunder"] = float((x < -BLUNDER).mean()) if len(x) else np.nan
    return {f"{k}_{suffix}": val for k, val in o.items()}


def sample_players(rng):
    """50 players per rank with exactly 20 scored games, spread over folds 1-4: (fold, player_id, rank, [game_key], {game_key: colour})."""
    cand = []
    colours = {}
    for f in ts.FOLDS:
        hsl = read_scored_games(ts.HSL / f"fold{f}")
        colours.update(dict(zip(hsl["game_key"], hsl["color"])))
        for (pid, rank), g in hsl.groupby(["player_id", "rank"]):
            if len(g) == GAMES_PER_PLAYER:
                cand.append((f, pid, rank, sorted(g["game_key"])))
    picked = []
    for rank in LEVELS:
        pool = [c for c in cand if c[2] == rank]
        by_fold = {f: [c for c in pool if c[0] == f] for f in ts.FOLDS}
        quota = {f: PLAYERS_PER_RANK // len(ts.FOLDS) + (1 if i < PLAYERS_PER_RANK % len(ts.FOLDS) else 0) for i, f in enumerate(ts.FOLDS)}
        for f in ts.FOLDS:
            idx = rng.permutation(len(by_fold[f]))[:quota[f]]
            picked += [by_fold[f][i] for i in idx]
    return picked, colours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    picked, colours = sample_players(np.random.RandomState(a.seed))
    meta = {k: {"game_key": k, "player_id": pid, "rank": rank, "fold": f} for f, pid, rank, keys in picked for k in keys}
    keys = sorted(meta)
    print(f"{len(picked)} players, {len(keys)} games ({time.time() - t0:.0f} s)", flush=True)
    pmi = {}
    for f in ts.FOLDS:
        own = read_scored_games(ROOT / "results/l1b" / ts.L1B[f] / f"fold{f}").set_index("game_key")
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}").set_index("game_key")
        cols = sum_cols(LEVELS)
        want = [k for k in keys if meta[k]["fold"] == f]
        po, pp = posterior_mean_index(own.loc[want, cols].to_numpy(dtype=float)), posterior_mean_index(opp.loc[want, cols].to_numpy(dtype=float))
        pmi.update({k: (float(x), float(y)) for k, x, y in zip(want, po, pp)})
    sgf = load_games(keys)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    model, _ = load_humansl(str(te.CKPT), "cuda")
    model.eval()
    rows = {"n": neutral_rows(), "s": meta_rows([("9d", "9d")])[0]}
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    recs, positions = [], 0
    try:
        for p in range(0, len(keys), a.part_size):
            part = keys[p:p + a.part_size]
            done = pool.map(replay_full, [(k, sgf[k][0], colours[k]) for k in part], chunksize=1)
            bins = np.concatenate([d[1] for d in done])
            globs = np.concatenate([d[2] for d in done])
            to_move = np.concatenate([d[3] for d in done])
            lens = np.array([len(d[3]) for d in done])
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            ev = {s: evaluate(model, bins, globs, to_move, rows[s], a.batch) for s in VARIANTS}
            for d, s0, n in zip(done, starts, lens):
                key, _, _, tm, target, tcol = d
                rec = {**meta[key], "n_plies": int(len(target)), "color": "B" if tcol == 0 else "W", "own_pmi": pmi[key][0], "opp_pmi": pmi[key][1]}
                for s in VARIANTS:
                    v, lead = target_view(ev[s][0][s0:s0 + n], ev[s][1][s0:s0 + n], tm, tcol)
                    rec.update(traj_stats(v.astype(np.float64), lead.astype(np.float64), target, s))
                recs.append(rec)
            positions += int(lens.sum())
            print(f"{len(recs)}/{len(keys)} games, {positions} positions, {positions / (time.time() - t0):.0f} pos/s", flush=True)
    finally:
        pool.close()
        pool.join()
    games = pd.DataFrame(recs)
    games.to_csv(OUT / "traj_games.csv", index=False)
    cols = [c for c in games.columns if c.endswith(("_n", "_s")) and c not in ("own_pmi",)]
    out, ref, table, H = st.report(games, cols)
    st.print_report(out, ref, table, "win-rate trajectory (H2), 500 players x 20 games, E1n (rank unknown) vs E1s (9d vs 9d)")
    by_rank = games.groupby("rank")[[f"{k}_s" for k in ("mean_down", "blunder", "ac1_down", "recover", "keep", "decisive_t", "v_final")]].mean().loc[list(LEVELS)]
    print(by_rank.round(3).to_string())
    pair = {k: {"split_half_r": (out[f"{k}_n"]["split_half_r"], out[f"{k}_s"]["split_half_r"]), "adjacent_auc": (out[f"{k}_n"]["adjacent_auc"], out[f"{k}_s"]["adjacent_auc"])}
            for k in sorted({c[:-2] for c in cols})}
    st.dump(OUT / "traj_report.json", out, ref, {"games": len(games), "players": len(picked), "positions": positions, "by_rank_E1s": by_rank.to_dict(),
                                                 "E1n_vs_E1s": pair, "thresholds": {"blunder": BLUNDER, "dip": DIP, "opp_gift": OPP_GIFT, "keep": KEEP, "decisive": DECISIVE},
                                                 "wall_s": time.time() - t0})
    print(f"written {OUT / 'traj_report.json'} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
