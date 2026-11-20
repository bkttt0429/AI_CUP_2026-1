#!/usr/bin/env python
"""Architecture v2 section 4I, stage 0: the end-position advantage proxy from humanv0's value and score heads, probed on about
100 sampled games before any batch extraction. Fold 0 is refused.

The records have no RE and (as far as the data shows) no passes, so the last position is a resignation or a truncation, not a
counted endgame: what the heads give is "who the network believes is ahead at the last recorded move", a proxy. Semantics
(KataGo python/play.py lines 244-247 and postprocess_single_heads_output): value_logits are {win, loss, noresult} for the side
to move; scoremean and lead are from the side to move's view as well; Human SL's score head is whiteScore (the predicted final
score difference under human continuation), not a strong net's lead, so the derived feature is called o_human_score.

Hard stops (any one fails -> stage 0 fails): replay or colour errors; the value perspective test (swapping the side to move on
the same board must flip the win probability; a white-perspective head would not move); komi not reaching the input features;
NaN / non-finite outputs; T, T-10, T-30 misaligned with the move list. Everything else (last mover ahead, sign agreement over
time, handicap monotonicity, komi and metadata sensitivity) is reported as diagnostics.

  python -I scripts/endgame_proxy.py probe --fold 1 --games 100 --out results/endgame/probe_fold1.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from aicup.humansl import load_humansl, make_profile_meta, stack_inputs  # noqa: E402
from aicup.l1b import neutral_meta  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.pilot import start_state  # noqa: E402
from aicup.scoring import fold_players, load_games, player_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
HD_TABLE = ROOT / "results/data/hd_table.csv"
OFFSETS = (0, 10, 30)  # T, T-10, T-30: plies before the last recorded move


def rules_with_komi(komi):
    return dict(GameState.RULES_CHINESE, whiteKomi=float(komi))


def data_komi(hd):
    """The data convention: 6.5 for even games, 0 with any handicap (HD 1 is sente, no komi)."""
    return 6.5 if hd == 0 else 0.0


def replay_states(sgf, komi, offsets=OFFSETS):
    """GameStates at the last position and `offsets` plies before it, plus the initial position; also the move list."""
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, rules_with_komi(komi))
    n = len(moves)
    want = {n - o for o in offsets if n - o >= 0} | {0}
    snaps = {}
    if 0 in want:
        snaps[0] = copy.deepcopy(gs)
    for t, (c, x, y) in enumerate(moves):
        pla = Board.BLACK if c == "B" else Board.WHITE
        if gs.board.pla != pla:
            raise ValueError(f"ply {t}: {c} to play but the board expects {gs.board.pla}")
        gs.play(pla, Board.PASS_LOC if x is None else gs.board.loc(x, y))
        if t + 1 in want:
            snaps[t + 1] = copy.deepcopy(gs)
    return snaps, moves, stones


def heads(model, features, states, meta):
    """value probabilities [N, 3] (side to move: win, loss, noresult), scoremean [N], lead [N] (side to move's view), and the
    global input rows [N, G] (to check that komi reaches the network)."""
    device = next(model.parameters()).device
    bins, globs, metas = stack_inputs(features, states, meta)
    with torch.no_grad():
        out = model(bins.to(device), globs.to(device), input_meta=metas.to(device))
        post = model.postprocess_output(out)[0]
    value = torch.softmax(post[1].float(), dim=1).cpu().numpy()
    return value, post[8].float().cpu().numpy(), post[10].float().cpu().numpy(), globs.numpy()


def target_view(value, score, lead, states, target_pla):
    """P(target wins), noresult, target score and lead from the side-to-move outputs."""
    mover_is_target = np.array([gs.board.pla == target_pla for gs in states])
    v = np.where(mover_is_target, value[:, 0], value[:, 1])
    sign = np.where(mover_is_target, 1.0, -1.0)
    return v, value[:, 2], score * sign, lead * sign, mover_is_target


def swapped(states):
    """The same boards with the other side to move (the perspective test)."""
    out = []
    for gs in states:
        g2 = copy.deepcopy(gs)
        g2.board.set_pla(Board.WHITE if gs.board.pla == Board.BLACK else Board.BLACK)
        out.append(g2)
    return out


def pick_games(fold, n_games, seed=0):
    """About n_games of the fold's scored selection, stratified over HD 0 / 1 / >= 2 (half, 30%, the rest)."""
    players = fold_players(player_games(), pd.read_csv(FOLDS_CSV), fold, max_games=20, min_games=5)
    keys = sorted(k for ks in players["game_keys"] for k in ks)
    hd = pd.read_csv(HD_TABLE, usecols=["game_key", "hd", "color"]).drop_duplicates("game_key").set_index("game_key")
    hd = hd.loc[hd.index.intersection(keys)]
    rng = np.random.RandomState(seed)
    groups = [(hd.index[hd.hd == 0], n_games // 2), (hd.index[hd.hd == 1], int(0.3 * n_games))]
    groups.append((hd.index[hd.hd >= 2], n_games - sum(g[1] for g in groups)))
    chosen = []
    for idx, k in groups:
        idx = np.array(sorted(idx))
        chosen += list(rng.permutation(idx)[:min(k, len(idx))])
    return chosen, hd


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def probe(a):
    if a.fold == 0:
        raise SystemExit("fold 0 is not used here")
    t0 = time.time()
    keys, hd = pick_games(a.fold, a.games)
    sgfs = load_games(keys)
    model, features = load_humansl(str(CKPT), "cuda")
    meta_neutral, meta_1d = neutral_meta(), make_profile_meta("1d")
    rows, hard = [], {"replay_errors": 0, "perspective_fail": 0, "komi_not_in_features": 0, "non_finite": 0, "misaligned": 0}
    t_replay, t_forward = 0.0, 0.0
    for k in keys:
        sgf, color = sgfs[k]
        target_pla = Board.BLACK if color == "B" else Board.WHITE
        game_hd = int(hd.loc[k, "hd"])
        try:
            t1 = time.time()
            snaps, moves, stones = replay_states(sgf, data_komi(game_hd))
            snaps_alt, _, _ = replay_states(sgf, 7.5)
            t_replay += time.time() - t1
        except Exception as e:  # noqa: BLE001 -- counted as a hard stop, the game is reported
            hard["replay_errors"] += 1
            rows.append({"game_key": k, "error": repr(e)})
            continue
        n = len(moves)
        plies = [n - o for o in OFFSETS if n - o >= 0]
        if any(p not in snaps for p in plies) or plies[0] != n:
            hard["misaligned"] += 1
        states = [snaps[p] for p in plies]
        t1 = time.time()
        value, score, lead, globs = heads(model, features, states, meta_neutral)
        value_alt, score_alt, _, globs_alt = heads(model, features, [snaps_alt[p] for p in plies], meta_neutral)
        value_1d, _, _, _ = heads(model, features, states, meta_1d)
        value_sw, _, _, _ = heads(model, features, swapped(states), meta_neutral)
        v0, _, _, _ = heads(model, features, [snaps[0]], meta_neutral)
        v0_sw, _, _, _ = heads(model, features, swapped([snaps[0]]), meta_neutral)
        t_forward += time.time() - t1
        if not (np.isfinite(value).all() and np.isfinite(score).all() and np.isfinite(lead).all()):
            hard["non_finite"] += 1
        if data_komi(game_hd) != 7.5 and np.array_equal(globs, globs_alt):
            hard["komi_not_in_features"] += 1
        # perspective (hard stop, HD >= 2 only): stones on the board, komi 0, so Black is far ahead at move 0 whoever moves. For a
        # side-to-move head value[0] is P(White wins) < 0.5 with White to move and P(Black wins) > 0.5 with Black to move; a
        # white-perspective head would stay below 0.5 in both. The last-position sum below is a diagnostic only: giving the other
        # side the move changes the evaluation itself (tempo), so it need not be exactly 1.
        if game_hd >= 2 and snaps[0].board.pla == Board.WHITE and not (v0[0, 0] < 0.5 < v0_sw[0, 0]):
            hard["perspective_fail"] += 1
        flip = value[:, 0] + value_sw[:, 0] + value[:, 2]
        v, nores, s_t, l_t, mover_is_target = target_view(value, score, lead, states, target_pla)
        v_alt, _, s_alt, _, _ = target_view(value_alt, score_alt, lead, states, target_pla)
        v_1d, _, _, _, _ = target_view(value_1d, score, lead, states, target_pla)
        last_mover_is_target = moves[-1][0] == color
        rows.append({"game_key": k, "hd": game_hd, "color": color, "n_moves": n, "n_pass": int(sum(x is None for _, x, _ in moves)),
                     "last_two_pass": int(all(x is None for _, x, _ in moves[-2:])), "last_pass": int(moves[-1][1] is None),
                     "n_stones": len(stones), "last_mover_is_target": bool(last_mover_is_target),
                     "v": v.tolist(), "noresult": nores.tolist(), "score": s_t.tolist(), "lead": l_t.tolist(), "mover_is_target": mover_is_target.tolist(),
                     "v_komi75": v_alt.tolist(), "score_komi75": s_alt.tolist(), "v_meta1d": v_1d.tolist(), "flip_sum": flip.tolist(),
                     "black_win_initial": float(v0[0, 0] if snaps[0].board.pla == Board.BLACK else v0[0, 1]),
                     "black_win_initial_swapped": float(v0_sw[0, 0] if snaps[0].board.pla == Board.WHITE else v0_sw[0, 1])})
    ok = [r for r in rows if "error" not in r]
    v_T = np.array([r["v"][0] for r in ok])
    v_10 = np.array([r["v"][1] for r in ok if len(r["v"]) > 1])
    v_30 = np.array([r["v"][2] for r in ok if len(r["v"]) > 2])
    conf = np.abs(v_T - 0.5)
    rep = {"fold": a.fold, "games": len(keys), "replayed": len(ok), "hard_stops": hard, "stage0_pass": all(x == 0 for x in hard.values()),
           "wall_s": time.time() - t0, "replay_s_per_game": t_replay / max(len(ok), 1), "forward_s_per_game": t_forward / max(len(ok), 1),
           "diagnostics": {
               "pass_share": {"any_pass": float(np.mean([r["n_pass"] > 0 for r in ok])), "last_two_pass": float(np.mean([r["last_two_pass"] for r in ok])),
                              "last_pass": float(np.mean([r["last_pass"] for r in ok]))},
               "hd_counts": {str(h): int(sum(r["hd"] == h for r in ok)) for h in sorted({r["hd"] for r in ok})},
               "last_mover_ahead_share": float(np.mean([(r["v"][0] > 0.5) == r["last_mover_is_target"] for r in ok])),
               "sign_agreement": {"T_vs_T-10": float(np.mean(np.sign(v_T[:len(v_10)] - 0.5) == np.sign(v_10 - 0.5))),
                                  "T_vs_T-30": float(np.mean(np.sign(v_T[:len(v_30)] - 0.5) == np.sign(v_30 - 0.5))),
                                  "by_confidence_T_vs_T-10": {f"{lo}-{hi}": float(np.mean((np.sign(v_T - 0.5) == np.sign(v_10 - 0.5))[(conf >= lo) & (conf < hi)]))
                                                              for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.51)) if ((conf >= lo) & (conf < hi)).any()},
                                  "spearman_T_vs_T-10": spearman(v_T[:len(v_10)], v_10), "spearman_T_vs_T-30": spearman(v_T[:len(v_30)], v_30)},
               "v_T_deciles": np.quantile(v_T, np.linspace(0, 1, 11)).round(3).tolist(),
               "decisive_share_abs_gt_0.4": float(np.mean(conf > 0.4)),
               "noresult_mean": float(np.mean([r["noresult"][0] for r in ok])),
               "komi_sensitivity": {"hd0_same_sign_6.5_vs_7.5": float(np.mean([(r["v"][0] > 0.5) == (r["v_komi75"][0] > 0.5) for r in ok if r["hd"] == 0])) if any(r["hd"] == 0 for r in ok) else None,
                                    "hd0_abs_dv_quantiles": np.quantile([abs(r["v"][0] - r["v_komi75"][0]) for r in ok if r["hd"] == 0], [0.5, 0.9, 1.0]).round(4).tolist() if any(r["hd"] == 0 for r in ok) else None,
                                    "hd1plus_same_sign_0_vs_7.5": float(np.mean([(r["v"][0] > 0.5) == (r["v_komi75"][0] > 0.5) for r in ok if r["hd"] >= 1])) if any(r["hd"] >= 1 for r in ok) else None,
                                    "hd1plus_abs_dv_quantiles": np.quantile([abs(r["v"][0] - r["v_komi75"][0]) for r in ok if r["hd"] >= 1], [0.5, 0.9, 1.0]).round(4).tolist() if any(r["hd"] >= 1 for r in ok) else None},
               "metadata_sensitivity_abs_dv_quantiles": np.quantile([abs(r["v"][0] - r["v_meta1d"][0]) for r in ok], [0.5, 0.9, 1.0]).round(4).tolist(),
               "black_win_initial_by_hd": {str(h): float(np.mean([r["black_win_initial"] for r in ok if r["hd"] == h])) for h in sorted({r["hd"] for r in ok})},
               "perspective_flip_sum_abs_dev_max": float(max(abs(x - 1.0) for r in ok for x in r["flip_sum"])),
               "score_T_quantiles": np.quantile([r["score"][0] for r in ok], [0.1, 0.5, 0.9]).round(2).tolist(),
               "score_sign_matches_v_T": float(np.mean([(r["score"][0] > 0) == (r["v"][0] > 0.5) for r in ok]))},
           "games_detail": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    d = rep["diagnostics"]
    print(f"stage 0 {'PASS' if rep['stage0_pass'] else 'FAIL'}: {rep['replayed']}/{rep['games']} games, hard stops {hard}; replay {rep['replay_s_per_game']:.2f} s/game, "
          f"forward {rep['forward_s_per_game']:.3f} s/game (5 batches of <= 3 positions)")
    print(f"passes: {d['pass_share']}; HD counts {d['hd_counts']}; perspective max |win + swapped win + noresult - 1| = {d['perspective_flip_sum_abs_dev_max']:.3f}")
    print(f"last mover ahead {d['last_mover_ahead_share']:.3f}; sign agreement {d['sign_agreement']}")
    print(f"v_T deciles {d['v_T_deciles']}; decisive share {d['decisive_share_abs_gt_0.4']:.3f}; noresult mean {d['noresult_mean']:.4f}")
    print(f"komi sensitivity {d['komi_sensitivity']}; metadata |dv| quantiles {d['metadata_sensitivity_abs_dv_quantiles']}")
    print(f"black win at move 0 by HD {d['black_win_initial_by_hd']}; score_T quantiles {d['score_T_quantiles']}; score sign = v sign {d['score_sign_matches_v_T']:.3f}")
    return 0 if rep["stage0_pass"] else 1


_FEATURES = None


def _init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def _replay_task(task):
    """(key, sgf, komi) -> inputs at the last position and OFFSETS plies before it: (key, bins uint8 [P, C, 19, 19], globs [P, G],
    side to move [P], ply index [P], n_moves, n_pass, last mover's colour)."""
    key, sgf, komi = task
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, rules_with_komi(komi))
    n = len(moves)
    want = {n - o for o in OFFSETS if n - o >= 0}
    bins, globs, pla, at = [], [], [], []
    for t, (c, x, y) in enumerate(moves):
        p = Board.BLACK if c == "B" else Board.WHITE
        if gs.board.pla != p:
            raise ValueError(f"{key} ply {t}: {c} to play but the board expects {gs.board.pla}")
        gs.play(p, Board.PASS_LOC if x is None else gs.board.loc(x, y))
        if t + 1 in want:
            b, g = gs.get_input_features(_FEATURES)
            bins.append(b[0].astype(np.uint8))
            globs.append(g[0].astype(np.float32))
            pla.append(1 if gs.board.pla == Board.WHITE else 0)
            at.append(t + 1)
    return key, np.stack(bins), np.stack(globs), np.asarray(pla, np.int8), np.asarray(at, np.int32), n, int(sum(x is None for _, x, _ in moves)), moves[-1][0]


def _forward(model, rows, bins, globs, pla, batch=256):
    """value probabilities [N, 3] (side to move), scoremean [N] and lead [N] (side to move's view) for stacked positions."""
    device = next(model.parameters()).device
    vals, scores, leads = [], [], []
    with torch.no_grad():
        for i in range(0, len(pla), batch):
            b = torch.as_tensor(bins[i:i + batch], device=device).float()
            g = torch.as_tensor(globs[i:i + batch], device=device)
            m = torch.as_tensor(rows[pla[i:i + batch].astype(int)], device=device)
            post = model.postprocess_output(model(b, g, input_meta=m))[0]
            vals.append(torch.softmax(post[1].float(), dim=1).cpu().numpy())
            scores.append(post[8].float().cpu().numpy())
            leads.append(post[10].float().cpu().numpy())
    return np.concatenate(vals), np.concatenate(scores), np.concatenate(leads)


def extract(a):
    """Stage 0 -> full pilot: every game of the folds' scored selection (fold_players, up to 20 per player), the data-convention
    komi, neutral metadata; one CSV per fold with the target-view proxy at T, T-10, T-30. Resumes per fold (a finished fold's
    CSV is kept)."""
    import multiprocessing as mp
    from aicup.l1b import neutral_rows
    if 0 in a.folds:
        raise SystemExit("fold 0 is not used here")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds_df = player_games(), pd.read_csv(FOLDS_CSV)
    hd = pd.read_csv(HD_TABLE, usecols=["game_key", "hd"]).drop_duplicates("game_key").set_index("game_key")["hd"]
    rows = neutral_rows()
    pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda")
        for fold in a.folds:
            path = out / f"games_fold{fold}.csv"
            if path.exists():
                print(f"fold {fold}: {path} exists, kept", flush=True)
                continue
            players = fold_players(all_games, folds_df, fold, max_games=20, min_games=5)
            meta = {k: (r.player_id, r.rank) for r in players.itertuples() for k in r.game_keys}
            keys = sorted(meta)
            t0, recs, n_fail = time.time(), [], 0
            for i in range(0, len(keys), a.part_size):
                part = keys[i:i + a.part_size]
                sgfs = load_games(part)
                tasks = [(k, sgfs[k][0], data_komi(int(hd.loc[k]))) for k in part]
                done = []
                for res in pool.imap(_replay_task, tasks, chunksize=4):
                    done.append(res)
                bins = np.concatenate([d[1] for d in done])
                globs = np.concatenate([d[2] for d in done])
                pla = np.concatenate([d[3] for d in done])
                value, score, lead = _forward(model, rows, bins, globs, pla, a.batch)
                if not (np.isfinite(value).all() and np.isfinite(score).all()):
                    raise FloatingPointError(f"fold {fold} part at {i}: non-finite head outputs")
                j = 0
                for k, _, _, pla_k, at_k, n, n_pass, last_c in done:
                    color = sgfs[k][1]
                    target_white = color == "W"
                    rec = {"game_key": k, "player_id": meta[k][0], "rank": meta[k][1], "color": color, "hd": int(hd.loc[k]), "komi": data_komi(int(hd.loc[k])),
                           "n_moves": n, "n_pass": n_pass, "last_mover_is_target": int(last_c == color)}
                    for o in OFFSETS:
                        rec[f"v_{o}"] = rec[f"score_{o}"] = rec[f"lead_{o}"] = rec[f"nores_{o}"] = np.nan
                    for pla_i, at_i in zip(pla_k, at_k):
                        o = n - int(at_i)
                        mover_is_target = bool(pla_i == 1) == target_white
                        rec[f"v_{o}"] = float(value[j, 0] if mover_is_target else value[j, 1])
                        rec[f"nores_{o}"] = float(value[j, 2])
                        rec[f"score_{o}"] = float(score[j] if mover_is_target else -score[j])
                        rec[f"lead_{o}"] = float(lead[j] if mover_is_target else -lead[j])
                        j += 1
                    recs.append(rec)
                assert j == len(pla)
                print(f"fold {fold}: {len(recs)}/{len(keys)} games, {len(recs) / (time.time() - t0):.1f} games/s", flush=True)
            pd.DataFrame(recs).to_csv(path, index=False)
            print(f"fold {fold} done: {len(recs)} games, {n_fail} failures, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("probe", "extract"))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part-size", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        a.out = a.out or str(ROOT / "results/endgame" / f"probe_fold{a.fold}.json")
        return probe(a)
    a.out = a.out or str(ROOT / "results/endgame")
    return extract(a)


if __name__ == "__main__":
    sys.exit(main())
