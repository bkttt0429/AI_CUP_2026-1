"""Section 4M M106 (2026-11-13, report only, CPU): where do two external (Fox) rank groups land on the competition's level scale,
how concentrated are they, and do their games look like the competition's to a level model? (design doc M106)
Per game: the first 256 plies' trunk vectors (128 dims, standardised with results/m90/glstats.npz) pooled into the target player's
mean and standard deviation and the opponent's mean (384 dims). A question = the mean over its games. A small MLP (384-256-10,
cross-entropy) on the training players of folds 1-4 (fold 0 never read), four-fold CV for the probe's own accuracy; the probe fitted
on folds 1-4 then reads every external account (20 games each, /work/u2507115/ext_fox/pilot: per-ply parts and hashed accounts only).
Report: (a) predicted-level distribution of each external rank group, mean and between-account sd of the expected level, against
the sd within one true label among the competition's 20-game validation questions; (b) AUC of a linear domain classifier (external
group vs the competition players whose label is one of the group's two most predicted levels), against the AUC of telling those two
levels apart.
  python -I logs/nano4/m106_scale_pilot.py
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

T_MAX = 256; EXT = Path("/work/u2507115/ext_fox/pilot"); RANK = {lv: i for i, lv in enumerate(LEVELS)}; T0 = time.time()
ST = np.load(ROOT / "results/m90/glstats.npz"); MU, SD = ST["mu"].astype(np.float32), ST["sd"].astype(np.float32)


def log(s):
    print(f"[{time.time() - T0:6.0f} s] {s}", flush=True)


def pooled(X, row, own, n):
    """Per game: mean and sd of the target player's plies, mean of the opponent's plies -> [n, 384]."""
    out = np.zeros((n, 384), np.float32)
    for sel, kind in ((own, 0), (~own, 1)):
        r, x = row[sel], X[sel].astype(np.float64); c = np.maximum(np.bincount(r, minlength=n), 1)[:, None]; s = np.zeros((n, 128)); np.add.at(s, r, x); m = s / c
        if kind == 0:
            s2 = np.zeros((n, 128)); np.add.at(s2, r, x * x); out[:, :128] = m; out[:, 128:256] = np.sqrt(np.maximum(s2 / c - m * m, 0.0))
        else:
            out[:, 256:] = m
    return out


def comp_part(csv_path):
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:
        g, l, target, row, t = z["g"], z["l"], z["target"], z["game_row"], z["t"]
    m = t < T_MAX; X = (np.concatenate([g[m], l[m]], 1).astype(np.float32) - MU) / SD
    return tab["game_key"].to_numpy().astype(str), tab["player_id"].to_numpy().astype(str), tab["rank"].to_numpy().astype(str), pooled(X, row[m], target[m], len(tab))


def ext_part(path):
    with np.load(path) as z:
        key, L, subj, mov, q = z["key"].astype(str), z["LEN"].astype(np.int64), z["SUBJ"], z["MOV"], z["GLQ"]
    row = np.repeat(np.arange(len(L)), L); return key, pooled(q.astype(np.float32) / 20.0, row, mov == subj[row], len(L))


def fit_mlp(X, y, n_out, seed=0, epochs=40, hidden=256, wd=1e-2):
    torch.manual_seed(seed); net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden, n_out)) if hidden else torch.nn.Linear(X.shape[1], n_out)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd); Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xt) // 256 + 1))
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for a in range(0, len(Xt), 256):
            i = perm[a:a + 256]; loss = torch.nn.functional.cross_entropy(net(Xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); return lambda Z: torch.softmax(net(torch.from_numpy(Z.astype(np.float32))), 1).detach().numpy().astype(np.float64)


def auc(s, y):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def domain_auc(A, B, seed=0):
    """Five-fold AUC of a linear classifier telling the rows of A from the rows of B."""
    X = np.concatenate([A, B]); y = np.concatenate([np.zeros(len(A), int), np.ones(len(B), int)]); rng = np.random.default_rng(seed); f = rng.integers(0, 5, len(y)); s = np.zeros(len(y))
    for k in range(5):
        tr = f != k; s[~tr] = fit_mlp(X[tr], y[tr], 2, seed=k, epochs=30, hidden=0, wd=1e-1)(X[~tr])[:, 1]
    return auc(s, y)


def main():
    import m103_stability as m3  # noqa: E402  (fold loader, score matrix, decision rule)
    mb, SM = m3.mb, m3.SM; torch.set_num_threads(12)
    files = [f for v in (1, 2, 3, 4) for f in sorted((ROOT / f"results/trunk_vec/fold{v}").glob("games_s*_p*.csv"))]; fold_of = [int(f.parent.name[4:]) for f in files]; fx = sorted(f for f in (EXT / "ply").glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz"))
    with mp.get_context("fork").Pool(12) as pool:
        rc = pool.map(comp_part, files, chunksize=4); rx = pool.map(ext_part, fx, chunksize=4)
    key = np.concatenate([r[0] for r in rc]); pid = np.concatenate([r[1] for r in rc]); lab = np.array([RANK[x] for x in np.concatenate([r[2] for r in rc])]); F = np.concatenate([r[3] for r in rc]); fold = np.concatenate([np.full(len(r[0]), v) for r, v in zip(rc, fold_of)])
    kx = np.concatenate([r[0] for r in rx]); Fx = np.concatenate([r[1] for r in rx]); sel = pd.read_csv(EXT / "select.csv", usecols=["game_key", "account"], dtype=str).set_index("game_key")["account"]; acc = sel.reindex(kx).to_numpy().astype(str); grp = np.array([k.split(":")[1] for k in kx])
    mu, sd = F.mean(0), F.std(0) + 1e-6; F = (F - mu) / sd; Fx = (Fx - mu) / sd; log(f"competition games (folds 1-4): {len(key)} of {len(set(pid))} players; external games: {len(kx)} of {len(set(acc))} accounts, groups {dict(pd.Series(grp).value_counts())}")
    row_of = {k: i for i, k in enumerate(key)}; players = pd.DataFrame({"pid": pid, "lab": lab, "fold": fold, "row": np.arange(len(key))}).groupby("pid").agg(lab=("lab", "first"), fold=("fold", "first"), rows=("row", list)); rng = np.random.default_rng(0)

    def train_questions(mask):
        X, y = [], []
        for rows, lb in zip(players.rows[mask], players.lab[mask]):
            rows = np.asarray(rows)
            for n, reps in ((5, 4), (10, 4), (15, 2), (20, 1)):
                for _ in range(reps):
                    if len(rows) >= n:
                        X.append(F[rng.choice(rows, n, replace=False)].mean(0)); y.append(lb)
        return np.stack(X), np.array(y)

    rep = {"cv": {}}; allP, allY, allN = [], [], []
    for v in (1, 2, 3, 4):
        Xtr, ytr = train_questions((players.fold != v).to_numpy()); pred = fit_mlp(Xtr, ytr, 10, seed=v); y, n, _, q = m3.load_fold(v)
        Xv = np.stack([F[[row_of[k] for k in g_]].mean(0) for g_ in q["game_keys"]]); P = pred(Xv); d = mb.decisions(P, np.zeros_like(P), 0.5, 3.0, 0.6)[0]; sc = balanced_mean_over_n(SM[d, y], y, n, sizes=SIZES)
        rep["cv"][f"fold{v}"] = float(sc); allP.append(P); allY.append(y); allN.append(n); log(f"probe, validation fold {v}: official score {sc:.4f} ({len(y)} questions; {len(ytr)} training questions)")
    P, y, n = np.concatenate(allP), np.concatenate(allY), np.concatenate(allN); E = P @ np.arange(10); s20 = n == 20; within = float(np.sqrt(np.mean([E[s20 & (y == c)].var() for c in range(10)]))); rep["cv"]["mean"] = float(np.mean(list(rep["cv"].values()))); rep["within_label_sd_n20"] = within
    print(f"probe (trunk vectors only, pooled): mean official score {rep['cv']['mean']:.4f} (the full single arms are about 0.71, the ensemble with anchors 0.775)")
    print("competition validation questions of 20 games, expected level (0 = D ... 9 = 6D) by true label: " + "  ".join(f"{LEVELS[c]} {E[s20 & (y == c)].mean():.2f}±{E[s20 & (y == c)].std():.2f}" for c in range(10)) + f" | pooled sd within a label {within:.2f}")
    # external accounts
    Xall, yall = train_questions(np.ones(len(players), bool)); pred = fit_mlp(Xall, yall, 10, seed=0); A = pd.DataFrame(Fx).groupby(acc).mean(); ng = pd.Series(acc).value_counts().reindex(A.index).to_numpy(); ag = pd.Series(grp).groupby(acc).first().reindex(A.index).to_numpy(); Pa = pred(A.to_numpy()); Ea = Pa @ np.arange(10); da = mb.decisions(Pa, np.zeros_like(Pa), 0.5, 3.0, 0.6)[0]
    PL = np.stack([F[np.asarray(r)].mean(0) for r in players.rows]); pl = players.lab.to_numpy(); rep["groups"] = {}
    for g_ in sorted(set(ag)):
        s = (ag == g_) & (ng >= 15); h = np.bincount(da[s], minlength=10) / max(int(s.sum()), 1); top2 = np.argsort(-h)[:2]; c1, c2 = sorted(top2)
        a_dom = domain_auc(PL[np.isin(pl, top2)], A.to_numpy()[s]); a_adj = domain_auc(PL[pl == c1], PL[pl == c2]) if c1 != c2 else float("nan")
        rep["groups"][g_] = {"accounts": int(s.sum()), "decision_hist": h.tolist(), "expected_mean": float(Ea[s].mean()), "expected_sd": float(Ea[s].std()), "max_prob_mean": float(Pa[s].max(1).mean()), "domain_auc": a_dom, "adjacent_auc": a_adj, "top2": [LEVELS[c1], LEVELS[c2]]}
        print(f"external group {g_}: {int(s.sum())} accounts | predicted level: " + "  ".join(f"{LEVELS[c]} {h[c]:.3f}" for c in range(10)))
        print(f"    expected level {Ea[s].mean():.2f}, sd between accounts {Ea[s].std():.2f} (competition, within one label: {within:.2f}; ratio {Ea[s].std() / within:.2f}); mean top probability {Pa[s].max(1).mean():.3f} (competition 20-game questions {P[s20].max(1).mean():.3f})")
        print(f"    domain classifier, group vs competition players labelled {LEVELS[c1]} or {LEVELS[c2]}: AUC {a_dom:.3f}; for comparison, competition {LEVELS[c1]} vs {LEVELS[c2]}: AUC {a_adj:.3f}", flush=True)
    (ROOT / "results/r_variants/m106_scale_pilot.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
