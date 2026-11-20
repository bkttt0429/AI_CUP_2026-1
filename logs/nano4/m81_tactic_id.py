"""Section 4M M81 (2026-11-10): a sixth identification component -- tactical events -- next to M73's five (opening, style, local,
orient, hd). Every game is replayed on KataGo's Board (HD >= 2: handicap stones restored, actual colours); each non-pass move from ply 12
on becomes one token = (stones captured 0 / 1 / 2+) x (gives atari) x (response to an atari the opponent's last move made on an own group:
none / saved without capturing / saved by capturing / played on the liberty and still in atari / ignored) x (liberties of the played
stone's group 1 / 2 / 3 / 4+) x (ko: none / ko capture / move under a ko ban) x (own groups adjacent 0 / 1 / 2+) x (opponent groups
adjacent 0 / 1 / 2+) x (phase < 60 / < 150 / rest): 9,720 values, invariant under board symmetries and colour.
Per training player a multinomial shrunk towards the pool background; the shrinkage strength is the Dirichlet-multinomial precision
fitted on the pool (Minka's fixed point; no validation outcome involved). One pool for both colours.
Everything else is m73_fused_id.py verbatim (pools, held-out own games, queries, seed), so the five-component outputs must reproduce M73.
  --stage tac                 tokens of both sides of every training row and test game -> results/m81/tac_{train,test}_{ids,off}.npy
  --stage score --shard k/K   five- and six-component fusion of shard k of the queries -> results/m81/part{k}of{K}.npz (+ report, shard 0)
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.sgf import board_size, parse_moves, setup_and_moves  # noqa: E402
from katago.game.board import Board  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COMPS = ip.COMPS
OUT = ROOT / "results/m81"
G = {}
OTHER = {"B": "W", "W": "B"}
FIELDS = (("cap", 3), ("atk", 2), ("threat", 5), ("libs", 4), ("ko", 3), ("own", 3), ("opp", 3), ("phase", 3))
V_TAC = int(np.prod([n for _, n in FIELDS]))
START = 12
_ARR = ("board", "group_head", "group_stone_count", "group_liberty_count", "group_next", "group_prev")


def decode(tid):
    out = {}
    for name, n in reversed(FIELDS):
        out[name] = int(tid % n); tid //= n
    return out


def tactic_ids(sgf, start=START, fast=True):
    """{actual colour: [token ids]} of both sides' non-pass moves at ply >= start. fast: the Board's arrays as Python lists (same
    values, about 3x faster scalar access; the class only indexes them)."""
    try:
        stones, moves = setup_and_moves(sgf)
    except ValueError:
        stones, moves = [], parse_moves(sgf)
    b = Board(board_size(sgf))
    if fast:
        for a in _ARR:
            setattr(b, a, getattr(b, a).tolist())
    for x, y in stones:
        b.set_stone(Board.BLACK, b.loc(x, y))
    bd, gh, gl, gs, adjs = b.board, b.group_head, b.group_liberty_count, b.group_stone_count, b.adj
    out = {"B": [], "W": []}; last = [None, None, None]
    for t, (c, x, y) in enumerate(moves):
        pla = 1 if c == "B" else 2; opp = 3 - pla
        if x is None:
            b.playUnsafe(pla, Board.PASS_LOC); last[pla] = None
            continue
        loc = b.loc(x, y)
        if bd[loc] != 0:  # a move on an occupied point cannot be replayed: no token
            last[pla] = None
            continue
        own, oppg = set(), set()
        for d in adjs:
            v = bd[loc + d]
            if v == pla:
                own.add(int(gh[loc + d]))
            elif v == opp:
                oppg.add(int(gh[loc + d]))
        cap = sum(int(gs[h]) for h in oppg if gl[h] == 1)
        ko = 1 if (cap == 1 and b.wouldBeKoCapture(loc, pla)) else 2 if b.simple_ko_point is not None else 0
        thr, thr_libs = [], []
        lo = last[opp]
        if lo is not None and bd[lo] == opp:
            for d in adjs:
                a = lo + d
                if bd[a] == pla and gl[gh[a]] == 1:
                    thr.append(a); b.findLiberties(a, thr_libs)
        b.playUnsafe(pla, loc)
        libs = min(max(int(gl[gh[loc]]), 1), 4) if bd[loc] == pla else 1
        atk = int(any(bd[loc + d] == opp and gl[gh[loc + d]] == 1 for d in adjs))
        if not thr:
            th = 0
        elif any(bd[a] == pla and gl[gh[a]] >= 2 for a in thr):
            th = 2 if cap > 0 else 1
        elif loc in thr_libs:
            th = 3
        else:
            th = 4
        if t >= start:
            tid = 0
            for val, (_, n) in zip((min(cap, 2), atk, th, libs - 1, ko, min(len(own), 2), min(len(oppg), 2), 0 if t < 60 else 1 if t < 150 else 2), FIELDS):
                tid = tid * n + val
            out[c].append(tid)
        last[pla] = loc
    return out


def _tac_row(args):
    sgf, col = args; d = tactic_ids(sgf)
    return np.asarray(d[col], np.int16), np.asarray(d[OTHER[col]], np.int16)


def self_test(sgfs):
    xy = lambda x, y: chr(97 + x) + chr(97 + y)  # noqa: E731
    game = lambda seq: "(;SZ[19]HD[0]" + "".join(f";{c}[{xy(x, y)}]" for c, (x, y) in zip("BW" * 40, seq)) + ")"  # noqa: E731
    a = tactic_ids(game([(3, 3), (2, 3), (15, 15), (4, 3), (15, 3), (3, 2), (3, 4)]), start=0); t = decode(a["B"][3])
    assert (t["threat"], t["cap"], t["libs"], t["own"], t["opp"], t["atk"], t["ko"]) == (1, 0, 2, 1, 0, 0, 0), t  # extends out of atari
    a = tactic_ids(game([(3, 3), (2, 3), (15, 15), (4, 3), (15, 3), (3, 2), (3, 15)]), start=0); assert decode(a["B"][3])["threat"] == 4  # ignores the atari
    a = tactic_ids(game([(1, 0), (0, 0), (0, 1)]), start=0); t = decode(a["B"][1]); w = decode(a["W"][0])
    assert (t["cap"], t["ko"], t["own"], t["opp"], t["libs"]) == (1, 0, 0, 1, 2), t  # corner capture, not a ko
    assert (w["libs"], w["atk"]) == (0, 0), w  # the captured stone was a self-atari
    a = tactic_ids(game([(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2), (16, 16), (1, 1), (2, 1), (15, 15)]), start=0)
    w = decode(a["W"][3]); t = decode(a["B"][4]); w2 = decode(a["W"][4])
    assert (w["atk"], w["libs"], w["own"], w["opp"], w["cap"]) == (1, 0, 0, 2, 0), w  # throw-in that gives atari
    assert (t["cap"], t["ko"], t["libs"], t["threat"]) == (1, 1, 0, 2), t  # ko capture that also saves the stone in atari
    assert w2["ko"] == 2, w2  # the next move is made under the ko ban
    t0 = time.time(); slow = [tactic_ids(s, fast=False) for s in sgfs]; t1 = time.time(); quick = [tactic_ids(s) for s in sgfs]; t2 = time.time()
    assert all(x == y for x, y in zip(slow, quick)), "list-backed Board differs from the numpy-backed one"
    print(f"self-test passed: 4 hand-made positions; list-backed = numpy-backed Board on {len(sgfs)} games ({(t1 - t0) / len(sgfs) * 1e3:.1f} vs {(t2 - t1) / len(sgfs) * 1e3:.1f} ms per game)", flush=True)


def pack(arrs):
    off = np.zeros(len(arrs) + 1, np.int64); off[1:] = np.cumsum([len(a) for a in arrs])
    return np.concatenate(arrs).astype(np.int16), off


class Flat:
    def __init__(self, name):
        self.ids = np.load(OUT / f"{name}_ids.npy"); self.off = np.load(OUT / f"{name}_off.npy")

    def get(self, j):
        return self.ids[self.off[j]:self.off[j + 1]]


def load_rows():
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    return rows


def load_tests():
    questions = ip.read_tests([ROOT / "data/tests/rank_prediction_test_public.csv", ROOT / "data/tests/rank_prediction_test_private.csv"])
    return [(q["question_id"], g) for q in questions for g in q["games"]]


def stage_tac():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); tg = load_tests(); log(f"{len(rows)} rows, {len(tg)} test games")
    self_test(sgf[:200])
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        res = list(pool.imap(_tac_row, zip(sgf, col), chunksize=512)); log("training rows tokenised")
        rest = list(pool.imap(_tac_row, [g for _, g in tg], chunksize=128))
    for name, r in (("tac_train", res), ("tac_test", rest)):
        ids, off = pack([a for pair in r for a in pair]); np.save(OUT / f"{name}_ids.npy", ids); np.save(OUT / f"{name}_off.npy", off)
        log(f"{name}: {len(r)} games, {len(ids)} tokens ({len(ids) / max(2 * len(r), 1):.1f} per side), {len(np.unique(ids))} of {V_TAC} token values used")
    ids = np.load(OUT / "tac_train_ids.npy")[::7].astype(np.int64); div = V_TAC
    for name, n in FIELDS:
        div //= n; v = (ids // div) % n; print(f"  {name:7s} " + " ".join(f"{k}: {(v == k).mean():.4f}" for k in range(n)), flush=True)
    log("written")


def eb_precision(counts, p_bg, s=50.0, iters=200):
    """Dirichlet-multinomial precision s for a fixed mean p_bg (Minka 2000, fixed point): the per-player counts are draws from
    Dirichlet(s * p_bg)-multinomials."""
    r, c = np.nonzero(counts); x = counts[r, c]; m = p_bg[c]; n = counts.sum(1); n = n[n > 0]
    for _ in range(iters):
        new = s * float((m * (digamma(x + s * m) - digamma(s * m))).sum()) / float((digamma(n + s) - digamma(s)).sum())
        done = abs(new - s) < 1e-4 * s; s = float(np.clip(new, 1.0, 1e6))
        if done:
            break
    return s


def raw_scores(i):
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); u, cnt = np.unique(G["qtac"][i], return_counts=True)
    tac = cnt.astype(np.float64) @ G["logpT"][u] if len(u) else np.zeros(G["C"])
    return np.vstack([np.stack([sc[c] for c in COMPS]), tac[None]])  # [6, C]


def top2(x, ex):
    if ex >= 0:
        x = x.copy(); x[ex] = -np.inf
    o = np.argpartition(-x, 1)[:2]; o = o[np.argsort(-x[o])]
    return int(o[0]), float(x[o[0]] - x[o[1]])


def score_chunk(idx):
    out = []; w6, w5 = G["w6"], G["w5"]; C = G["C"]
    for i in idx:
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)
        if ex >= 0:
            valid[ex] = False
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)
        b6, m6 = top2(w6 @ Z, ex); b5, m5 = top2(w5 @ Z[:5], ex)
        out.append((i, b6, m6, b5, m5))
    return out


def full_chunk(idx):
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]


def stage_score(k, K):
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rng = np.random.default_rng(20261110)
    rows = load_rows()
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); hd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0 for s in sgf])
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")
    qrow, qfold, qy, qpid = [], [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k_ in gk:
                    qrow.append(key_index[k_]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid)
    qrow = np.array(qrow)
    with ctx.Pool(12) as pool:
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))
    used = set(qrow.tolist()); held = []
    by_player = rows.groupby("player_id").indices
    for p_ in rng.choice(players, 2400, replace=False):
        cand = [r for r in by_player[p_] if r not in used and hd[r] < 2]
        if len(cand) >= 8:
            held += rng.choice(cand, 2, replace=False).tolist()
    held = np.array(held); include = np.ones(len(rows), bool); include[held] = False
    tg = load_tests()
    with ctx.Pool(12) as pool:
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))
    log(f"query views: validation {len(qrow)} x 2, held-out {len(held)}, test {len(tg)} x 2")
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * len(held) + [-1] * (2 * len(tg))
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)
    # the tactic component: one pool, both colours
    train, test = Flat("tac_train"), Flat("tac_test"); assert len(train.off) == 2 * len(rows) + 1 and len(test.off) == 2 * len(tg) + 1
    idx = np.flatnonzero(include); ids = np.concatenate([train.get(2 * r) for r in idx]).astype(np.int64); ptok = np.repeat(pl[idx], train.off[2 * idx + 1] - train.off[2 * idx])
    counts = np.bincount(ptok * V_TAC + ids, minlength=C * V_TAC).reshape(C, V_TAC).astype(np.float64); del ids, ptok
    bg = counts.sum(0); p_bg = (bg + 5.0 / V_TAC) / (bg.sum() + 5.0); s = eb_precision(counts, p_bg)
    logpT = np.ascontiguousarray(np.log((counts + s * p_bg) / (counts.sum(1, keepdims=True) + s)).T.astype(np.float32)); del counts
    log(f"tactic pool: {int(bg.sum())} tokens, {int((bg > 0).sum())} token values in use, Dirichlet precision {s:.1f}")
    qtac = [train.get(2 * r + 1) for r in qrow] + [train.get(2 * r) for r in qrow] + [train.get(2 * r) for r in held] + [test.get(2 * j + 1) for j in range(len(tg))] + [test.get(2 * j) for j in range(len(tg))]
    assert len(qtac) == len(qviews)
    G.update(qviews=qviews, pools=pools, excl=excl, C=C, qtac=qtac, logpT=logpT)
    n0 = 2 * len(qrow); hidx = np.arange(n0, n0 + len(held)); truth = pl[held]
    with ctx.Pool(12) as pool:
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # [6, Q, C]
    Z = [ip.standardise(S[j]) for j in range(6)]; w6 = ip.fit_weights(Z, truth); w5 = ip.fit_weights(Z[:5], truth); names = (*COMPS, "tactic")
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731
    acc = {}
    for name, F in [(c, Z[j]) for j, c in enumerate(names)] + [("fused5", np.einsum("j,jqc->qc", w5, np.stack(Z[:5]))), ("fused6", np.einsum("j,jqc->qc", w6, np.stack(Z)))]:
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}
    log("weights (six) " + " ".join(f"{c} {x:.2f}" for c, x in zip(names, w6)) + " | (five) " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w5)))
    log("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k_} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k_, v in acc.items()))
    del full, S, Z
    G.update(w6=w6, w5=w5)
    if k == 0:
        Path(OUT / "report.json").write_text(json.dumps({"weights6": dict(zip(names, map(float, w6))), "weights5": dict(zip(COMPS, map(float, w5))), "held_queries": int(len(held)), "heldout_accuracy": acc,
                                                         "dirichlet_precision": s, "tokens": int(bg.sum()), "token_values_used": int((bg > 0).sum())}, indent=1))
        np.savez_compressed(OUT / "meta.npz", players=players, plab=plab, Q=len(qrow), H=len(held), T=len(tg))
    rest = np.concatenate([np.arange(n0), np.arange(n0 + len(held), len(qviews))]); mine = np.array_split(rest, K)[k]
    with ctx.Pool(12) as pool:
        res = [x for part in pool.map(score_chunk, np.array_split(mine, 240)) for x in part]
    res.sort(key=lambda x: x[0]); a = np.array([x[0] for x in res]); assert np.array_equal(a, mine)
    np.savez_compressed(OUT / f"part{k}of{K}.npz", idx=a, b6=np.array([x[1] for x in res]), m6=np.array([x[2] for x in res]), b5=np.array([x[3] for x in res]), m5=np.array([x[4] for x in res]))
    log(f"shard {k} of {K}: {len(mine)} queries scored and written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("tac", "score")); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args()
    if a_.stage == "tac":
        stage_tac()
    else:
        stage_score(*map(int, a_.shard.split("/")))
