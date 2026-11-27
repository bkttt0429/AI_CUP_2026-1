"""Section 4M M71 test side (2026-11-10): opponent-anchor evidence on the 800 test questions (design doc, M71 test side).
Pool = all 8,007 training players. Calibration in the test regime (validation queries with only the own player excluded), the anchor
density ratio rho from the test's own opponent-side margin bands (label-free), per-game evidence, ct9 posterior x evidence ->
submission_rank_ct37[_proba].csv; then the two bias pushes on the new posterior -> submission_rank_ct38.csv.
  python -I logs/nano4/m71_test_apply.py
"""
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import TRAIN_DIR, load_games  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
import m71_step2 as s2  # noqa: E402
import opening_id_pool as oip  # noqa: E402
import predict_submission as ps  # noqa: E402
from m43_gate import decide  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
G = {}


def score_chunk(idx):
    """Per query: ((opp label, opp margin, opp best), (target-side label, margin)); exclude = a player index or -1."""
    out = []
    for i in idx:
        (tb, tw), col, ex = G["toks"][i], G["col"][i], G["excl"][i]; rec = []
        for M, t in (((G["MW"], tw) if col == "B" else (G["MB"], tb)), ((G["MB"], tb) if col == "B" else (G["MW"], tw))):
            s = M.scores(t)
            if ex >= 0:
                s[ex] = -np.inf
            o = np.argpartition(-s, 1)[:2]; o = o[np.argsort(-s[o])]
            rec.append((int(G["plab"][o[0]]), float(s[o[0]] - s[o[1]]), int(o[0])))
        out.append((i, rec))
    return out


def score_all(toks, col, excl):
    G.update(toks=toks, col=col, excl=excl)
    with mp.get_context("fork").Pool(12) as pool:
        res = [x for part in pool.map(score_chunk, np.array_split(np.arange(len(toks)), 240)) for x in part]
    res.sort(key=lambda x: x[0])
    G["last_tgt_best"] = np.array([r[1][1][2] for r in res])
    return (np.array([r[1][0][0] for r in res]), np.array([r[1][0][1] for r in res]), np.array([r[1][0][2] for r in res]), np.array([r[1][1][0] for r in res]), np.array([r[1][1][1] for r in res]))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--a-scale", type=float, default=None, help="use a_b = scale x the validation a_b (agreement-based density) instead of the share-based rho formula")
    ap.add_argument("--tag", nargs=2, default=["37", "38"], help="numbers of the anchor-only file and of the file with the two pushes")
    args = ap.parse_args(); A, B = args.tag
    t0 = time.time()
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]) for p in sorted(Path(TRAIN_DIR).glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; n = len(players)
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    keys = rows.game_key.tolist(); sgfs = load_games(keys)
    with mp.get_context("fork").Pool(12) as pool:
        toks = pool.map(oip.both_tokens, [sgfs[k][0] for k in keys], chunksize=512)
    tok = dict(zip(keys, toks)); del toks
    hd = {k: int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", sgfs[k][0])) else 0 for k in keys}
    MB, MW = oip.PoolModel(n), oip.PoolModel(n)
    for r in rows.itertuples():
        tb, tw = tok[r.game_key]
        (MB if r.color == "B" else MW).add(pidx[r.player_id], tb if r.color == "B" else tw)
    MB.finalize(); MW.finalize(); G.update(MB=MB, MW=MW, plab=plab); print(f"pool fitted on {len(rows)} rows ({time.time() - t0:.0f} s)", flush=True)
    colour = dict(zip(rows.game_key, rows.color))
    # validation queries, test regime: only the own player excluded
    qkey, qcol, qfold, qy, qpid, qhd = [], [], [], [], [], []
    for v in (1, 2, 3, 4):
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k in gk:
                    qkey.append(k); qcol.append(colour[k]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid); qhd.append(hd[k])
    ol, om, ob, il, im = score_all([tok[k] for k in qkey], qcol, [pidx[p] for p in qpid])
    np.savez_compressed(ROOT / "results/r_variants/m71_val_regime.npz", key=np.array(qkey), pid=np.array(qpid), fold=np.array(qfold), y=np.array(qy), hd=np.array(qhd), target_black=np.array(qcol) == "B",
                        opp_lab=ol, opp_m=om, opp_best=players[ob], imp_lab=il, imp_m=im, imp_best=players[G["last_tgt_best"]])
    z = {"key": np.array(qkey), "fold": np.array(qfold), "y": np.array(qy), "hd": np.array(qhd), "target_black": np.array(qcol) == "B", "opp_lab": ol, "opp_m": om, "imp_m": im}
    print(f"validation queries scored in the test regime: {len(qkey)} ({time.time() - t0:.0f} s)", flush=True)
    rep = {}
    print("CV in the test regime (cross-fitted, same rule as step 2):")
    p1 = s2.run("base seeds 0-2", (0, 1, 2), z, rep); p2 = s2.run("replication seeds 6-8", (6, 7, 8), z, rep)
    edges, cal, vband, vg = s2.fit(z, np.ones(len(qy), bool)); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    # test queries
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS); tasks, owner, _, _ = ps.build_tasks(questions)
    with mp.get_context("fork").Pool(12) as pool:
        ttoks = pool.map(oip.both_tokens, [sgf for _, sgf, _ in tasks], chunksize=256)
    tcol = [c for _, _, c in tasks]; thd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", sgf)) else 0 for _, sgf, _ in tasks]); tb_ = np.array(tcol) == "B"
    tl, tm, tbest, til, tim = score_all(ttoks, tcol, [-1] * len(tasks)); qid = np.array([questions[i]["question_id"] for i in owner])
    np.savez_compressed(ROOT / "results/r_variants/m71_test_queries.npz", qid=qid, hd=thd, target_black=tb_, opp_lab=tl, opp_m=tm, opp_best=players[tbest], tgt_lab=til, tgt_m=tim, tgt_best=players[G["last_tgt_best"]])
    band = np.where(tb_, np.searchsorted(edges[False], tm), np.searchsorted(edges[True], tm)); tband_t = np.where(tb_, np.searchsorted(edges[True], tim), np.searchsorted(edges[False], tim))
    g = s2.group_of(thd, tb_); nb = len(nominal)
    sh_opp = np.array([(band == b).mean() for b in range(nb)]); sh_tgt = np.array([(tband_t == b).mean() for b in range(nb)]); sh_val = np.array([(vband[vg >= 0] == b).mean() for b in range(nb)])
    print("band shares      " + " ".join(f"{x:7.4f}" for x in nominal) + "  (nominal)"); print("validation opp   " + " ".join(f"{x:7.4f}" for x in sh_val))
    print("test opp         " + " ".join(f"{x:7.4f}" for x in sh_opp)); print("test target side " + " ".join(f"{x:7.4f}" for x in sh_tgt) + "  (should be near nominal: the test players are not in the pool)")
    ex_test, ex_val = float(sh_opp[3:].sum() - nominal[3:].sum()), float(sh_val[3:].sum() - nominal[3:].sum()); rho = float(np.clip(ex_test / ex_val, 0, 1.2))
    tgt_excess = float(sh_tgt[3:].sum() - nominal[3:].sum())
    print(f"excess share above the impostor q90: test opponents {ex_test:+.4f}, validation opponents {ex_val:+.4f} -> rho {rho:.3f}; test target side {tgt_excess:+.4f}")
    for half in ("pub_", "priv_"):
        s = np.char.startswith(qid.astype(str), half); print(f"   {half:5s} opponents above q90: excess {float((band[s] >= 3).mean() - nominal[3:].sum()):+.4f}; above q99: share {float((band[s] >= 6).mean()):.4f} (nominal 0.0100)")
    # per-game evidence with a_b = rho ex_b / (nominal_b + rho ex_b)
    logE = np.zeros((len(tasks), 10)); a_used = np.zeros(len(tasks))
    for k in (0, 1, 2):
        s = g == k; ex = np.clip(cal[k]["share"] - nominal, 0, None); a = rho * ex / (nominal + rho * ex)
        if args.a_scale is not None:
            a = args.a_scale * cal[k]["a"]
        dd = tl[s][:, None] - np.arange(10)[None, :]
        logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9]); a_used[s] = a[band[s]]
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); cols = [f"p_{lv}" for lv in LEVELS]
    L = pd.DataFrame(logE).groupby(qid).sum().reindex(pr.question_id).fillna(0.0).to_numpy()
    P = pr[cols].to_numpy(float); Pn = P * np.exp(L - L.max(axis=1, keepdims=True)); Pn = Pn / Pn.sum(axis=1, keepdims=True)
    d0, d1 = np.asarray(decide(P), int), np.asarray(decide(Pn), int)
    out = pr.copy(); out[cols] = Pn; out.to_csv(ROOT / f"submission_rank_ct{A}_proba.csv", index=False)
    write_submission([(q, LEVELS[k]) for q, k in zip(pr.question_id, d1)], ROOT / f"submission_rank_ct{A}.csv")
    pub = pr.question_id.str.startswith("pub_").to_numpy(); ch = d0 != d1
    amax = pd.Series(a_used).groupby(qid).max().reindex(pr.question_id).fillna(0).to_numpy()
    print(f"questions with an anchor of weight >= 0.2: {int((amax >= 0.2).sum())}/800 (public {int((amax[pub] >= 0.2).sum())}, private {int((amax[~pub] >= 0.2).sum())}); >= 0.5: {int((amax >= 0.5).sum())}")
    print(f"ct{A} (ct9 x anchors) vs ct9: {int(ch.sum())} decisions differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}); moves " + str(pd.Series([f"{LEVELS[a]}->{LEVELS[b]}" for a, b in zip(d0[ch], d1[ch])]).value_counts().to_dict()))
    old = pd.read_csv(ROOT / "results/r_variants/opening_id_test_matches.csv", dtype=str)
    if len(old) == len(tasks):
        print(f"sanity: top-1 opponent candidate equals the 11-06 run's (models held out a few games there) in {float((old.opp_best.to_numpy() == players[tbest]).mean()):.3f} of test games")
    # the two bias pushes on the new posterior
    for base, a_, b_, o in ((f"submission_rank_ct{A}.csv", "1D", "A", f"submission_rank_ct{B}a.csv"), (f"submission_rank_ct{B}a.csv", "4D", "5D", f"submission_rank_ct{B}.csv")):
        r = subprocess.run([sys.executable, "-I", "logs/nano4/m53_write_probe.py", "--base", base, "--push", a_, b_, "--r", "2", "--out", o, "--proba", f"submission_rank_ct{A}_proba.csv"], capture_output=True, text=True, cwd=ROOT)
        print("  " + "\n  ".join(l for l in r.stdout.splitlines() if l.startswith("push") or l.startswith("md5")) + (("\n  ERR " + r.stderr[-300:]) if r.returncode else ""))
    new = pd.read_csv(ROOT / f"submission_rank_ct{B}.csv", dtype=str).set_index("question_id").pred_rank
    for ref in ("ct28", "ct9"):
        o = pd.read_csv(ROOT / f"submission_rank_{ref}.csv", dtype=str).set_index("question_id").pred_rank.reindex(new.index); c = new[new != o]
        print(f"ct{B} vs {ref}: {len(c)} differ (public {int(c.index.str.startswith('pub_').sum())}, private {int(c.index.str.startswith('priv_').sum())}); moves " + str(pd.Series([f"{o[i]}->{c[i]}" for i in c.index]).value_counts().head(14).to_dict()))
    rep.update(rho=rho, excess_test=ex_test, excess_val=ex_val, target_side_excess=tgt_excess, shares={"nominal": nominal.tolist(), "validation": sh_val.tolist(), "test_opp": sh_opp.tolist(), "test_target": sh_tgt.tolist()},
               cv_regime_pass=bool(p1 and p2), ct37_vs_ct9=int(ch.sum()), recommend_ct38=bool(p1 and p2 and rho >= 0.3))
    print(f"RECOMMENDATION RULE: regime-matched CV {'passes' if p1 and p2 else 'fails'}, rho {rho:.2f} ({'>=' if rho >= 0.3 else '<'} 0.3) -> {'recommend ct38 over ct28' if rep['recommend_ct38'] else 'keep ct28'}")
    Path(ROOT / f"results/r_variants/m71_test_apply_ct{B}.json").write_text(json.dumps(rep, indent=1, default=float)); print(f"written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
