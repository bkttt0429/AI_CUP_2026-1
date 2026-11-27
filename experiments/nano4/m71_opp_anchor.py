"""Section 4M M71 step 1 (2026-11-10): does the label of the training player an opponent is matched to agree with the target's label?
Opening models (depth 12, Black / White) fitted on ALL training rows (the test-time setting). Queries: the 20 games of every validation
player's n = 20 question (folds 1-4); candidates exclude the player's whole fold (the target is never in the pool, as in the test).
Per game: (a) the OPPONENT side against the opposite-colour models; (b) the TARGET side against the same-colour models = impostor
control (the true player is excluded: only the style -> level effect is left). Margin bands at the impostor quantiles (per colour).
Reports per band, HD0 and HD1: share of queries, exact-level and within-one agreement of the top-1 candidate's label with the target's.
Saves the per-query table for step 2 (results/r_variants/m71_queries.npz).
  python -I logs/nano4/m71_opp_anchor.py
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import TRAIN_DIR, load_games  # noqa: E402
import opening_id_pool as oip  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
G = {}


def score_chunk(idx):
    out = []
    for i in idx:
        key, col, fold = G["qkey"][i], G["qcol"][i], G["qfold"][i]
        tb, tw = G["tok"][key]; allow = G["pfold"] != fold; rec = []
        for M, t in (((G["MW"], tw) if col == "B" else (G["MB"], tb)), ((G["MB"], tb) if col == "B" else (G["MW"], tw))):  # opponent side, then target side
            s = M.scores(t); s = np.where(allow, s, -np.inf)
            o = np.argpartition(-s, 1)[:2]; o = o[np.argsort(-s[o])]
            w = np.exp(s - s[o[0]]); post = np.bincount(G["plab"], weights=w, minlength=10); post = post / post.sum()
            rec.append((int(G["plab"][o[0]]), float(s[o[0]] - s[o[1]]), post, int(o[0])))
        out.append((i, rec))
    return out


def main():
    t0 = time.time()
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]) for p in sorted(Path(TRAIN_DIR).glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; n = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; pfold = folds.reindex(players).to_numpy()
    keys = rows.game_key.tolist(); sgfs = load_games(keys); print(f"{len(rows)} rows, {n} players, SGFs loaded ({time.time() - t0:.0f} s)", flush=True)
    with mp.get_context("fork").Pool(12) as pool:
        toks = pool.map(oip.both_tokens, [sgfs[k][0] for k in keys], chunksize=512)
    tok = dict(zip(keys, toks)); del toks
    hd = {k: int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", sgfs[k][0])) else 0 for k in keys}
    MB, MW = oip.PoolModel(n), oip.PoolModel(n)
    for r in rows.itertuples():
        tb, tw = tok[r.game_key]
        (MB if r.color == "B" else MW).add(pidx[r.player_id], tb if r.color == "B" else tw)
    MB.finalize(); MW.finalize(); print(f"models fitted on all rows ({time.time() - t0:.0f} s)", flush=True)
    colour = dict(zip(rows.game_key, rows.color))
    qkey, qcol, qfold, qy, qpid, qhd = [], [], [], [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):
            if ng != 20:
                continue
            for k in gk:
                qkey.append(k); qcol.append(colour[k]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid); qhd.append(hd[k])
    Q = len(qkey); print(f"{Q} queries from {len(set(qpid))} validation players", flush=True)
    G.update(tok=tok, MB=MB, MW=MW, plab=plab, pfold=pfold, qkey=qkey, qcol=qcol, qfold=qfold)
    chunks = np.array_split(np.arange(Q), 240)
    with mp.get_context("fork").Pool(12) as pool:
        res = [x for part in pool.map(score_chunk, chunks) for x in part]
    res.sort(key=lambda x: x[0]); print(f"scored ({time.time() - t0:.0f} s)", flush=True)
    opp_lab = np.array([r[1][0][0] for r in res]); opp_m = np.array([r[1][0][1] for r in res]); opp_post = np.stack([r[1][0][2] for r in res]); opp_best = np.array([r[1][0][3] for r in res])
    imp_lab = np.array([r[1][1][0] for r in res]); imp_m = np.array([r[1][1][1] for r in res]); imp_post = np.stack([r[1][1][2] for r in res])
    qy, qhd, qfold = np.array(qy), np.array(qhd), np.array(qfold); isB = np.array(qcol) == "B"
    np.savez_compressed(ROOT / "results/r_variants/m71_queries.npz", key=np.array(qkey), pid=np.array(qpid), fold=qfold, y=qy, hd=qhd, target_black=isB,
                        opp_lab=opp_lab, opp_m=opp_m, opp_post=opp_post.astype(np.float32), opp_best=players[opp_best], imp_lab=imp_lab, imp_m=imp_m, imp_post=imp_post.astype(np.float32))
    # margin bands at the impostor quantiles; the opponent query uses the OTHER colour's model, so its reference is the impostor of that colour
    qs = [0.5, 0.9, 0.95, 0.99, 0.999]; names = ["<=q50", "q50-90", "q90-95", "q95-99", "q99-99.9", ">q99.9"]
    edges = {True: np.quantile(imp_m[isB], qs), False: np.quantile(imp_m[~isB], qs)}  # keyed by "the model is Black's"
    opp_band = np.where(isB, np.searchsorted(edges[False], opp_m), np.searchsorted(edges[True], opp_m))  # target Black -> opponent White -> White model
    imp_band = np.where(isB, np.searchsorted(edges[True], imp_m), np.searchsorted(edges[False], imp_m))
    rep = {"queries": int(Q), "impostor_q99": {"black_model": float(edges[True][3]), "white_model": float(edges[False][3])}, "bands": {}}
    for hdv in (0, 1):
        s = qhd == hdv; print(f"HD{hdv}: {int(s.sum())} queries"); rep["bands"][f"HD{hdv}"] = {}
        print(f"  {'band':9s} | opp share  exact  within1 | impostor share  exact  within1 | opp label - target label (-2..+2)")
        for b, nm in enumerate(names):
            so, si = s & (opp_band == b), s & (imp_band == b)
            d = opp_lab[so] - qy[so]; di = imp_lab[si] - qy[si]
            row = {"opp_share": float(so.sum() / s.sum()), "opp_exact": float((d == 0).mean()) if so.any() else None, "opp_within1": float((np.abs(d) <= 1).mean()) if so.any() else None, "opp_n": int(so.sum()),
                   "imp_share": float(si.sum() / s.sum()), "imp_exact": float((di == 0).mean()) if si.any() else None, "imp_within1": float((np.abs(di) <= 1).mean()) if si.any() else None,
                   "opp_diff_hist": {str(k): float((d == k).mean()) for k in (-2, -1, 0, 1, 2)} if so.any() else None}
            rep["bands"][f"HD{hdv}"][nm] = row
            print(f"  {nm:9s} |   {row['opp_share']:.4f}   {row['opp_exact'] or 0:.3f}   {row['opp_within1'] or 0:.3f}  |      {row['imp_share']:.4f}    {row['imp_exact'] or 0:.3f}   {row['imp_within1'] or 0:.3f}  | "
                  + (" ".join(f"{v:.2f}" for v in row["opp_diff_hist"].values()) if row["opp_diff_hist"] else "") + f"  (n {row['opp_n']})")
    # pre-registered continuation test: HD0, margin above the impostor q99
    s = qhd == 0; so, si = s & (opp_band >= 4), s & (imp_band >= 4)
    oe, ie, share = float((opp_lab[so] == qy[so]).mean()), float((imp_lab[si] == qy[si]).mean()), float(so.sum() / s.sum())
    go = (oe >= ie + 0.15) and (share >= 0.015)
    print(f"CONTINUATION (HD0, > impostor q99): opponent exact agreement {oe:.3f} vs impostor {ie:.3f} (need +0.15), opponent share {share:.4f} (need >= 0.015) -> {'GO to step 2' if go else 'STOP'}")
    # per-player view: how many of a player's 20 games have an opponent above q99, by fold
    hits = pd.Series((opp_band >= 4)).groupby(np.array(qpid)).sum()
    print("validation players by number of games (of 20) with an opponent above q99: " + " ".join(f"{k}: {v:.3f}" for k, v in (hits.clip(upper=4).value_counts(normalize=True).sort_index()).items()) + " (4 = 4 or more)")
    rep["continuation"] = {"opp_exact": oe, "imp_exact": ie, "opp_share": share, "go": bool(go)}
    Path(ROOT / "results/r_variants/m71_opp_anchor.json").write_text(json.dumps(rep, indent=1, default=float)); print(f"written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
