"""Section 4M M82 (2026-11-10): a LEARNED conditional matcher as an identification component: each candidate is described by his own
games. One 256-dim row per game side = mean and standard deviation of that side's per-ply trunk vectors (64 pooled + 64 played point);
competition 1's B8 network and loss (aicup_pid.embed.EmbedNet): a row -> a unit vector; an episode takes 512 players, one game each as the
query and the sum of the vectors of his other games (up to 39) as his prototype; cross-entropy both ways. Cross-fitted: the queries of
validation fold v use the network trained WITHOUT fold v's players; test and the fusion weights use the all-players network.
Fusion as in M73 / M81, with the held-out own games drawn from the games that have vectors (games 21-40), and four fusions in one run:
five, six (+ tactic, M81), five + matcher, seven.
  --stage rows                 rows of both sides of every game with trunk vectors + the held-out keys -> results/m82/
  --stage embed                the five networks and their embeddings of every row (GPU)
  --stage score --shard k/K    the fusions of shard k of the queries -> results/m82/part{k}of{K}.npz (+ report, shard 0)
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COMPS = ip.COMPS
NAMES = (*COMPS, "tactic", "matcher")
FUS = {"5": [0, 1, 2, 3, 4], "6": [0, 1, 2, 3, 4, 5], "5e": [0, 1, 2, 3, 4, 6], "7": [0, 1, 2, 3, 4, 5, 6]}
OUT = ROOT / "results/m82"
VEC = (("base", ROOT / "results/trunk_vec"), ("x40", ROOT / "results/trunk_vec_x40"))
TEST_VEC = ROOT / "results/ct_submission/vec"
NETS = (1, 2, 3, 4, -1)  # the fold whose players a network never saw; -1 = all players (test, fusion weights)
OTHER = {"B": "W", "W": "B"}
STEPS, EPISODE, SUPPORT, MIN_GAMES, SEED = 6000, 512, 40, 8, 0
G = {}


def part_rows(csv_path):
    """(games table, rows [2, G, 256] (0 = the target's plies, 1 = the opponent's), ply counts [2, G]) of one trunk-vector part."""
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:
        X = np.hstack([z["g"], z["l"]]).astype(np.float64); target, row = z["target"], z["game_row"]
    n_g = len(tab); out = np.zeros((2, n_g, 256), np.float32); cnt = np.zeros((2, n_g), np.int32)
    for side, m in enumerate((target, ~target)):
        r = row[m]; order = np.argsort(r, kind="stable"); xs = X[m][order]; n = np.bincount(r, minlength=n_g); nz = n > 0; starts = (np.cumsum(n) - n)[nz]
        if not nz.any():
            continue
        s1 = np.add.reduceat(xs, starts, axis=0); s2 = np.add.reduceat(xs * xs, starts, axis=0); mean = s1 / n[nz][:, None]
        out[side][nz] = np.hstack([mean, np.sqrt(np.maximum(s2 / n[nz][:, None] - mean * mean, 0.0))]); cnt[side] = n
    return tab, out, cnt


def stage_rows():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True); ctx = mp.get_context("fork"); cols = defaultdict(list)
    with ctx.Pool(12) as pool:
        for src, d in VEC:
            for f in range(5):
                res = pool.map(part_rows, sorted((d / f"fold{f}").glob("games_s*_p*.csv")), chunksize=4)
                for tab, out, cnt in res:
                    cols["key"].append(tab["game_key"].to_numpy()); cols["pid"].append(tab["player_id"].to_numpy()); cols["fold"].append(np.full(len(tab), f, np.int8)); cols["x40"].append(np.full(len(tab), src == "x40"))
                    cols["T"].append(out[0]); cols["O"].append(out[1]); cols["nT"].append(cnt[0]); cols["nO"].append(cnt[1])
                log(f"{src} fold {f}: {sum(len(t) for t, _, _ in res)} games")
        res = pool.map(part_rows, sorted(TEST_VEC.glob("games_s*_p*.csv")), chunksize=4)
    R = {k: np.concatenate(v) for k, v in cols.items()}; R["key"] = R["key"].astype(str); R["pid"] = R["pid"].astype(str)
    assert len(np.unique(R["key"])) == len(R["key"]), "a game key appears in two parts"
    np.savez(OUT / "rows_train.npz", **R); log(f"training rows: {len(R['key'])} games, {len(np.unique(R['pid']))} players, {int((R['nO'] < 2).sum())} opponent sides with < 2 plies")
    task = np.concatenate([t["task"].to_numpy().astype(int) for t, _, _ in res]); o = np.argsort(task); assert np.array_equal(task[o], np.arange(len(task)))
    np.savez(OUT / "rows_test.npz", qid=np.concatenate([t["question_id"].to_numpy() for t, _, _ in res]).astype(str)[o], T=np.concatenate([x[0] for _, x, _ in res])[o], O=np.concatenate([x[1] for _, x, _ in res])[o])
    log(f"test rows: {len(task)} games")
    # held-out own games for the fusion weights: 2 per player among the games 21-40 (they have vectors), never a validation-question game
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); hd = {k: (int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0) for k, s in zip(rows.game_key, rows.sgf_content)}
    used = set()
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for ng, gk in zip(m["n_games"], m["game_keys"]):
            if ng == 20:
                used.update(gk)
    by_player = defaultdict(list)
    for k, p_, x in zip(R["key"], R["pid"], R["x40"]):
        if x and k not in used and hd.get(k, 9) < 2:
            by_player[p_].append(k)
    rng = np.random.default_rng(20261110); held = []
    for p_ in rng.choice(players, 2400, replace=False):
        cand = sorted(by_player.get(p_, []))
        if len(cand) >= 8:
            held += rng.choice(cand, 2, replace=False).tolist()
    (OUT / "held_keys.json").write_text(json.dumps(held)); log(f"held-out own games: {len(held)} of {len(held) // 2} players; validation-question games with vectors: {len(used & set(R['key']))} of {len(used)}")


def stage_embed():
    import torch
    from aicup_pid.embed import EmbedNet
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; R = np.load(OUT / "rows_train.npz"); key, pid, fold = R["key"], R["pid"], R["fold"]; T, O = R["T"], R["O"]; Rt = np.load(OUT / "rows_test.npz")
    held = set(json.load(open(OUT / "held_keys.json"))); is_held = np.array([k in held for k in key]); players = np.array(sorted(set(pid))); pidx = {p_: i for i, p_ in enumerate(players)}; owner = np.array([pidx[p_] for p_ in pid])
    pfold = np.zeros(len(players), np.int8); pfold[owner] = fold; rep = {}; emb = {}
    for v in NETS:
        torch.manual_seed(SEED); use = ~is_held & (fold != v); n_rows = np.bincount(owner[use], minlength=len(players)); tp = np.flatnonzero(n_rows >= MIN_GAMES)
        assert n_rows.max() <= SUPPORT, n_rows.max()
        tab = np.full((len(players), SUPPORT), -1, np.int64); fill = np.zeros(len(players), np.int64)
        for g in np.flatnonzero(use):
            tab[owner[g], fill[owner[g]]] = g; fill[owner[g]] += 1
        mu = T[use].mean(0); sd = T[use].std(0) + 1e-6; std = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731
        X = std(T); tab_t = torch.as_tensor(tab[tp], device=dev); cnt_t = torch.as_tensor(fill[tp], device=dev); P = min(EPISODE, len(tp))
        net = EmbedNet(256, 512, 128, 0.1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
        ar = torch.arange(SUPPORT, device=dev); tgt = torch.arange(P, device=dev); net.train()
        for step in range(STEPS):
            b = torch.randperm(len(tp), device=dev)[:P]; c = cnt_t[b]; u = torch.rand(P, SUPPORT, device=dev).masked_fill(ar[None] >= c[:, None], 2.0)
            rows = tab_t[b].gather(1, u.argsort(dim=1))  # the player's games in random order, padding last
            zq = net(X[rows[:, 0]]); sm = ar[None, 1:] < c[:, None]
            zs = net(X[rows[:, 1:].clamp(min=0)].reshape(-1, 256)).reshape(P, SUPPORT - 1, -1) * sm[..., None]
            logits = net.log_scale.exp() * zq @ torch.nn.functional.normalize(zs.sum(1), dim=-1).T
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 1000 == 0 or step == STEPS - 1:
                log(f"  net {v}: step {step}, loss {float(loss):.4f}, episode top-1 {float((logits.argmax(1) == tgt).float().mean()):.3f} of {P}, scale {float(net.log_scale.exp()):.1f}")
        net.eval()
        with torch.no_grad():
            enc = lambda A: np.concatenate([net(std(A[i:i + 65536])).cpu().numpy() for i in range(0, len(A), 65536)])  # noqa: E731
            eT, eO = enc(T), enc(O); np.save(OUT / f"emb_T_{v}.npy", eT); np.save(OUT / f"emb_O_{v}.npy", eO)
            if v == -1:
                np.save(OUT / "emb_test_T.npy", enc(Rt["T"])); np.save(OUT / "emb_test_O.npy", enc(Rt["O"]))
        emb[v] = eT; rep[f"net_{v}"] = {"players_trained": int(len(tp)), "rows": int(use.sum()), "final_loss": float(loss)}
        log(f"net {v}: {len(tp)} players, {int(use.sum())} rows; embeddings written")
    # report: the held-out own games against all candidates' prototypes (held rows excluded): in-sample players (net -1) and, per fold v,
    # the same queries of fold v's players under net v (players the network never saw)
    hi = np.flatnonzero(is_held); truth = owner[hi]

    def ident(v, sel):
        Pm = np.zeros((len(players), 128)); np.add.at(Pm, owner[~is_held], emb[v][~is_held]); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        S = emb[v][hi[sel]] @ Pm.T; r = (S > S[np.arange(sel.sum()), truth[sel]][:, None]).sum(1)
        return {"queries": int(sel.sum()), "top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean())}
    rep["heldout_all_net"] = ident(-1, np.ones(len(hi), bool)); log(f"held-out own games, all-players net: top-1 / top-5 / top-20 {rep['heldout_all_net']['top1']:.3f} / {rep['heldout_all_net']['top5']:.3f} / {rep['heldout_all_net']['top20']:.3f} ({len(hi)} queries, {len(players)} candidates)")
    for v in (1, 2, 3, 4):
        sel = pfold[truth] == v; a, b = ident(v, sel), ident(-1, sel); rep[f"unseen_fold{v}"] = {"unseen": a, "seen": b}
        log(f"  fold {v} players' held-out games ({a['queries']}): network that never saw them top-1 {a['top1']:.3f} / top-20 {a['top20']:.3f}; all-players network {b['top1']:.3f} / {b['top20']:.3f}")
    (OUT / "embed_report.json").write_text(json.dumps(rep, indent=1)); log("written")


def raw_scores(i):
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); u, cnt = np.unique(G["qtac"][i], return_counts=True)
    tac = cnt.astype(np.float64) @ G["logpT"][u] if len(u) else np.zeros(G["C"])
    return np.vstack([np.stack([sc[c] for c in COMPS]), tac[None], (G["protos"][G["qnet"][i]] @ G["qvec"][i]).astype(np.float64)[None]])  # [7, C]


def score_chunk(idx):
    out = []; C = G["C"]
    for i in idx:
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)
        if ex >= 0:
            valid[ex] = False
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)
        out.append((i, *[x for name, sel in FUS.items() for x in m81.top2(G["w"][name] @ Z[sel], ex)]))
    return out


def full_chunk(idx):
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]


def stage_score(k, K):
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rows = m81.load_rows()
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist()
    ctx = mp.get_context("fork")
    with ctx.Pool(12) as pool:
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")
    qrow, qfold, qpid = [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, ng, gk in zip(m["player_id"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k_ in gk:
                    qrow.append(key_index[k_]); qfold.append(v); qpid.append(pid)
    qrow = np.array(qrow); qfold = np.array(qfold); Q = len(qrow)
    with ctx.Pool(12) as pool:
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))
    held_keys = json.load(open(OUT / "held_keys.json")); held = np.array([key_index[k_] for k_ in held_keys]); H = len(held); include = np.ones(len(rows), bool); include[held] = False
    tg = m81.load_tests(); T = len(tg)
    with ctx.Pool(12) as pool:
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))
    log(f"query views: validation {Q} x 2, held-out {H}, test {T} x 2")
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * H + [-1] * (2 * T)
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)
    # tactic component (M81)
    train, test = m81.Flat("tac_train"), m81.Flat("tac_test"); V = m81.V_TAC
    idx = np.flatnonzero(include); ids = np.concatenate([train.get(2 * r) for r in idx]).astype(np.int64); ptok = np.repeat(pl[idx], train.off[2 * idx + 1] - train.off[2 * idx])
    counts = np.bincount(ptok * V + ids, minlength=C * V).reshape(C, V).astype(np.float64); del ids, ptok
    bg = counts.sum(0); p_bg = (bg + 5.0 / V) / (bg.sum() + 5.0); s = m81.eb_precision(counts, p_bg)
    logpT = np.ascontiguousarray(np.log((counts + s * p_bg) / (counts.sum(1, keepdims=True) + s)).T.astype(np.float32)); del counts
    qtac = [train.get(2 * r + 1) for r in qrow] + [train.get(2 * r) for r in qrow] + [train.get(2 * r) for r in held] + [test.get(2 * j + 1) for j in range(T)] + [test.get(2 * j) for j in range(T)]
    log(f"tactic pool fitted (Dirichlet precision {s:.1f})")
    # matcher component: prototypes per network, one vector per query
    R = np.load(OUT / "rows_train.npz"); rkey = R["key"]; gi = {k_: i for i, k_ in enumerate(rkey)}; heldset = set(held_keys); keep = np.array([k_ not in heldset for k_ in rkey]); cand = np.array([pidx[p_] for p_ in R["pid"]])
    Rt = np.load(OUT / "rows_test.npz"); assert np.array_equal(Rt["qid"], np.array([q for q, _ in tg])), "test rows are not in the order of the test games"
    protos = np.zeros((len(NETS), C, 128), np.float32); qvec = np.zeros((len(qviews), 128), np.float32); qnet = np.zeros(len(qviews), np.int8); missing = 0
    for j, v in enumerate(NETS):
        eT = np.load(OUT / f"emb_T_{v}.npy"); Pm = np.zeros((C, 128)); np.add.at(Pm, cand[keep], eT[keep]); protos[j] = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        if v > 0:
            eO = np.load(OUT / f"emb_O_{v}.npy"); sel = np.flatnonzero(qfold == v); g_ = np.array([gi.get(keys[r], -1) for r in qrow[sel]]); ok = g_ >= 0; missing += int((~ok).sum())
            qvec[sel[ok]] = eO[g_[ok]]; qvec[Q + sel[ok]] = eT[g_[ok]]; qnet[sel] = qnet[Q + sel] = j
        else:
            qvec[2 * Q:2 * Q + H] = eT[[gi[k_] for k_ in held_keys]]; qvec[2 * Q + H:2 * Q + H + T] = np.load(OUT / "emb_test_O.npy"); qvec[2 * Q + H + T:] = np.load(OUT / "emb_test_T.npy"); qnet[2 * Q:] = j
    log(f"matcher: prototypes of {int((np.linalg.norm(protos[-1], axis=1) > 0).sum())} of {C} candidates; validation query games without vectors: {missing}")
    G.update(qviews=qviews, pools=pools, excl=excl, C=C, qtac=qtac, logpT=logpT, protos=protos, qvec=qvec, qnet=qnet)
    n0 = 2 * Q; hidx = np.arange(n0, n0 + H); truth = pl[held]
    with ctx.Pool(12) as pool:
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # [7, H, C]
    Z = [ip.standardise(S[j]) for j in range(7)]; w = {name: ip.fit_weights([Z[j] for j in sel], truth) for name, sel in FUS.items()}
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731
    acc = {}
    for name, F in [(c, Z[j]) for j, c in enumerate(NAMES)] + [(f"fused{name}", np.einsum("j,jqc->qc", w[name], np.stack([Z[j] for j in sel]))) for name, sel in FUS.items()]:
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}
    for name, sel in FUS.items():
        log(f"weights ({name}) " + " ".join(f"{NAMES[j]} {x:.2f}" for j, x in zip(sel, w[name])))
    log("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k_} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k_, v in acc.items()))
    del full, S, Z
    G.update(w=w)
    if k == 0:
        Path(OUT / "report.json").write_text(json.dumps({"weights": {name: {NAMES[j]: float(x) for j, x in zip(sel, w[name])} for name, sel in FUS.items()}, "held_queries": int(H), "heldout_accuracy": acc, "dirichlet_precision": s}, indent=1))
        np.savez_compressed(OUT / "meta.npz", players=players, plab=plab, Q=Q, H=H, T=T)
    rest = np.concatenate([np.arange(n0), np.arange(n0 + H, len(qviews))]); mine = np.array_split(rest, K)[k]
    with ctx.Pool(12) as pool:
        res = [x for part in pool.map(score_chunk, np.array_split(mine, 240)) for x in part]
    res.sort(key=lambda x: x[0]); a = np.array([x[0] for x in res]); assert np.array_equal(a, mine)
    out = {"idx": a}
    for j, name in enumerate(FUS):
        out[f"b{name}"] = np.array([x[1 + 2 * j] for x in res]); out[f"m{name}"] = np.array([x[2 + 2 * j] for x in res])
    np.savez_compressed(OUT / f"part{k}of{K}.npz", **out); log(f"shard {k} of {K}: {len(mine)} queries scored and written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("rows", "embed", "score")); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args()
    if a_.stage == "rows":
        stage_rows()
    elif a_.stage == "embed":
        stage_embed()
    else:
        stage_score(*map(int, a_.shard.split("/")))
