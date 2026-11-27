"""Section 4M M73 (2026-11-10): single-game identification of the opponent with the five habit components of competition 1
(opening, style, local shape, orientation, HD class; ident_pool.py's vectorised models), fused with weights fitted on held-out OWN games.
Pools: all training rows except 2 held-out games per player (the weight-fitting queries). Queries, test regime (only the own player
excluded): the 20 games of every validation player's n = 20 question, opponent side and target side (impostor); the test games.
Saves per query, for the fused score and for the opening component alone: top-1 candidate label, margin, candidate.
  python -I logs/nano4/m73_fused_id.py
"""
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COMPS = ip.COMPS
G = {}
OTHER = {"B": "W", "W": "B"}


def raw_scores(i):
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v])
    return np.stack([sc[c] for c in COMPS])  # [5, C]


def top2(x, ex):
    if ex >= 0:
        x = x.copy(); x[ex] = -np.inf
    o = np.argpartition(-x, 1)[:2]; o = o[np.argsort(-x[o])]
    return int(o[0]), float(x[o[0]] - x[o[1]])


def score_chunk(idx):
    out = []; w = G["w"]; C = G["C"]
    for i in idx:
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)
        if ex >= 0:
            valid[ex] = False
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)
        bf, mf = top2(w @ Z, ex); bo, mo = top2(S[0], ex)
        out.append((i, bf, mf, bo, mo))
    return out


def full_chunk(idx):
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rng = np.random.default_rng(20261110)
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k: i for i, k in enumerate(keys)}
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); hd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0 for s in sgf])
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")
    # validation queries
    qrow, qfold, qy, qpid = [], [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k in gk:
                    qrow.append(key_index[k]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid)
    qrow = np.array(qrow)
    with ctx.Pool(12) as pool:
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))
    # held-out own games for the weights: 2 per player for a sample of players, never a validation-question game
    used = set(qrow.tolist()); held = []
    by_player = rows.groupby("player_id").indices
    for p_ in rng.choice(players, 2400, replace=False):
        cand = [r for r in by_player[p_] if r not in used and hd[r] < 2]
        if len(cand) >= 8:
            held += rng.choice(cand, 2, replace=False).tolist()
    held = np.array(held); include = np.ones(len(rows), bool); include[held] = False
    # test
    questions = ip.read_tests([ROOT / "data/tests/rank_prediction_test_public.csv", ROOT / "data/tests/rank_prediction_test_private.csv"])
    tg = [(q["question_id"], g) for q in questions for g in q["games"]]
    with ctx.Pool(12) as pool:
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))
    log(f"query views: validation {len(qrow)} x 2, held-out {len(held)}, test {len(tg)} x 2")
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * len(held) + [-1] * (2 * len(tg))
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)
    G.update(qviews=qviews, pools=pools, excl=excl, C=C)
    # weights on the held-out own games
    n0 = 2 * len(qrow); hidx = np.arange(n0, n0 + len(held)); truth = pl[held]
    with ctx.Pool(12) as pool:
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # [5, Q, C]
    Z = [ip.standardise(S[j]) for j in range(len(COMPS))]; w = ip.fit_weights(Z, truth); rep = {"weights": dict(zip(COMPS, map(float, w))), "held_queries": int(len(held))}
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731
    acc = {}
    for name, F in [(c, Z[j]) for j, c in enumerate(COMPS)] + [("fused", np.einsum("j,jqc->qc", w, np.stack(Z)))]:
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}
    rep["heldout_accuracy"] = acc
    log("weights " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w)) + " | held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k, v in acc.items()))
    del full, S, Z
    G["w"] = w
    with ctx.Pool(12) as pool:
        res = [x for part in pool.map(score_chunk, np.array_split(np.concatenate([np.arange(n0), np.arange(n0 + len(held), len(qviews))]), 480)) for x in part]
    res = {i: r for i, *r in res}; log("all queries scored")
    Q = len(qrow); get = lambda rng_, j: np.array([res[i][j] for i in rng_])  # noqa: E731
    o, t_ = range(0, Q), range(Q, 2 * Q); base = dict(key=np.array([keys[r] for r in qrow]), pid=np.array(qpid), fold=np.array(qfold), y=np.array(qy), hd=hd[qrow], target_black=np.array([col[r] for r in qrow]) == "B")
    np.savez_compressed(ROOT / "results/r_variants/m73_val.npz", **base,
                        f_opp_lab=plab[get(o, 0)], f_opp_m=get(o, 1), f_opp_best=players[get(o, 0)], f_imp_lab=plab[get(t_, 0)], f_imp_m=get(t_, 1),
                        o_opp_lab=plab[get(o, 2)], o_opp_m=get(o, 3), o_opp_best=players[get(o, 2)], o_imp_lab=plab[get(t_, 2)], o_imp_m=get(t_, 3))
    s0 = n0 + len(held); to, tt = range(s0, s0 + len(tg)), range(s0 + len(tg), s0 + 2 * len(tg))
    thd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", g[0])) else 0 for _, g in tg])
    np.savez_compressed(ROOT / "results/r_variants/m73_test.npz", qid=np.array([q for q, _ in tg]), hd=thd, target_black=np.array([g[1] for _, g in tg]) == "B",
                        f_opp_lab=plab[get(to, 0)], f_opp_m=get(to, 1), f_opp_best=players[get(to, 0)], f_tgt_lab=plab[get(tt, 0)], f_tgt_m=get(tt, 1), f_tgt_best=players[get(tt, 0)],
                        o_opp_lab=plab[get(to, 2)], o_opp_m=get(to, 3), o_opp_best=players[get(to, 2)], o_tgt_lab=plab[get(tt, 2)], o_tgt_m=get(tt, 3))
    Path(ROOT / "results/r_variants/m73_fused_id.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
