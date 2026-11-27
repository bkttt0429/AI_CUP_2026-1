"""Section 4M M107 (2026-11-13, CPU): does training together with external players that carry a rank of ANOTHER server help a
level probe on the competition's players? (design doc M107 and its amendments; features and probe of m106_scale_pilot.py)
Arms, same architecture, same training questions:
  A  control: 384 -> 256 -> 10, competition training players only;
  B  multi-task: shared body, competition head (10 levels) + external head (one class per external rank); every step 256
     competition questions + 256 external questions; loss = CE_competition + lambda x CE_external (lambda 1 = the registered test).
Training length: each arm picks its own number of epochs from EPOCH_GRID on 10% inner hold-out training players (seed 0), then
trains three seeds on all training players; the validation fold is scored once.
External unit = (account, rank folder): the rank label is the folder of each game's key (an account's games may sit in two
folders); units with at least 5 games; up to --per-group units per rank; external questions of 5 and 10 games for every rank
(four of the eleven ranks come from selections with 10 games per account).
Report only: lambda 0.3 and 3; a larger body (384 -> 512 -> 512) for A and B; the mapped-label arm (each external rank turned into
a distribution over the competition's levels by arm A's predictions; external units train the competition head with those soft
labels); the external head's own accuracy on held-out accounts; gains by level.
Four-fold CV (fold 0 never read), official score; B - A paired, player bootstrap.
Gate for a full build: difference >= +0.005, CI lower bound > 0, at least three of four folds positive.
Speed (2026-11-14, no change of the test): the (fold, arm) trainings are independent given their seeds, so they run as twelve
single-threaded processes instead of one after another; a real run refuses to start when a source's per-ply vectors are incomplete.
With GPUs in the allocation (sbatch --gres=gpu:8 -c 96) every training runs on a GPU, three processes per GPU, and the file
readers use up to 48 processes; the parent never touches CUDA (the workers are forked).
  python -I logs/nano4/m107_multitask_probe.py --src SELECT.csv PLYDIR[,PLYDIR...] [RANK,RANK...] [--src ...] [--per-group 1000] [--smoke]
  (--smoke: one fold, one seed, short training; prints that every arm ran, never a score)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m106_scale_pilot as p6  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

T0 = time.time(); QPLAN = ((5, 4), (10, 4), (15, 2), (20, 1)); XPLAN = ((5, 4), (10, 4)); BATCH = 256; EPOCH_GRID = (2, 4, 8, 16, 40)
G = {}  # shared with the forked workers: pooled features, tables, per-fold data, run settings
N_GPU = 0 if "--cpu" in sys.argv else len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]); N_CPU = len(os.sched_getaffinity(0))


def worker_device():
    return torch.device(f"cuda:{(mp.current_process()._identity[0] - 1) % N_GPU}") if N_GPU else torch.device("cpu")


def log(s):
    print(f"[{time.time() - T0:6.0f} s] {s}", flush=True)


def rank_order(g):
    """Sort key of an external rank name: 18k ... 1k, 1d ... 9d."""
    return -int(g[:-1]) if g.endswith("k") else int(g[:-1]) - 1


def ext_part_f(job):
    """One external per-ply part pooled per game; only the games whose key's folder is wanted (None: all)."""
    path, want = job; key, F = p6.ext_part(path)
    if want is not None:
        m = np.array([k.split(":")[1] in want for k in key], bool); key, F = key[m], F[m]
    return key, F


def questions(F, rows_list, labels, rng, plan=QPLAN):
    X, y, ns = [], [], []
    for rows, lb in zip(rows_list, labels):
        rows = np.asarray(rows)
        for n, reps in plan:
            for _ in range(reps):
                if len(rows) >= n:
                    X.append(F[rng.choice(rows, n, replace=False)].mean(0)); y.append(lb); ns.append(n)
    return torch.from_numpy(np.stack(X).astype(np.float32)), torch.from_numpy(np.array(y, np.int64)), np.array(ns)


class Net(torch.nn.Module):
    def __init__(self, d_in, hid, n_ext):
        super().__init__(); layers, d = [], d_in
        for h in hid:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(0.2)]; d = h
        self.body = torch.nn.Sequential(*layers); self.hc = torch.nn.Linear(d, 10); self.hx = torch.nn.Linear(d, n_ext) if n_ext else None


def train(Xc, yc, Xe, ye, n_ext, hid, lam, seed, soft=None, epochs=40, dev="cpu"):
    """One probe. Xe None: arm A. soft None: external head (arm B). soft [n_ext, 10]: external questions train the competition head."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = Net(Xc.shape[1], hid, n_ext if (Xe is not None and soft is None) else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xc) // BATCH + 1)); ce = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xc), generator=g).to(dev)
        for a in range(0, len(Xc), BATCH):
            i = perm[a:a + BATCH]; loss = ce(net.hc(net.body(Xc[i])), yc[i])
            if Xe is not None:
                j = torch.randint(len(Xe), (BATCH,), generator=g).to(dev); he = net.body(Xe[j])
                loss = loss + lam * (ce(net.hx(he), ye[j]) if soft is None else -(soft[ye[j]] * torch.log_softmax(net.hc(he), 1)).sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); return net


def proba(net, X, head="c"):
    with torch.no_grad():
        h = net.body(torch.from_numpy(np.asarray(X, np.float32)).to(next(net.parameters()).device)); return torch.softmax(net.hc(h) if head == "c" else net.hx(h), 1).cpu().numpy().astype(np.float64)


def prep_fold(v):
    """Everything one validation fold needs (same random draws in the same order as the sequential version)."""
    F0, Fx0, fold, players, xa, hold, row_of = G["F0"], G["Fx0"], G["fold"], G["players"], G["xa"], G["hold"], G["row_of"]; m3 = G["m3"]
    tr = fold != v; mu, sd = F0[tr].mean(0), F0[tr].std(0) + 1e-6; F = (F0 - mu) / sd; Fx = (Fx0 - mu) / sd; rng = np.random.default_rng(100 + v); m = (players.fold != v).to_numpy()
    Xc, yc, _ = questions(F, players.rows[m], players.lab[m], rng); Xe, ye, _ = questions(Fx, xa.rows[~hold], xa.g[~hold], rng, XPLAN); y, n, pidv, q = m3.load_fold(v)
    inner = rng.random(int(m.sum())) < 0.10; Xc_in, yc_in, _ = questions(F, players.rows[m][~inner], players.lab[m][~inner], rng); Xi, yi_t, ni = questions(F, players.rows[m][inner], players.lab[m][inner], rng)
    Xv = np.stack([F[[row_of[k] for k in g_]].mean(0) for g_ in q["game_keys"]]).astype(np.float32); Xh = np.stack([Fx[np.asarray(r)].mean(0) for r in xa.rows[hold]]).astype(np.float32); Xt = np.stack([Fx[np.asarray(r)].mean(0) for r in xa.rows[~hold]]).astype(np.float32)
    return v, dict(Xc=Xc, yc=yc, Xe=Xe, ye=ye, Xc_in=Xc_in, yc_in=yc_in, Xi=Xi.numpy(), yi=yi_t.numpy(), ni=ni, Xv=Xv, Xh=Xh, Xt=Xt, y=np.asarray(y), n=np.asarray(n), pid=np.asarray(pidv), gh=xa.g[hold].to_numpy(), gt=xa.g[~hold].to_numpy())


def run_arm(job):
    """One (fold, arm): pick the training length on the inner hold-out players, train the seeds, predict."""
    v, name, soft = job; torch.set_num_threads(1); d = G["folds"][v]; ext, hid, lam, mapped = G["arms"][name]; mb, SM = G["m3"].mb, G["m3"].SM; ng = G["n_groups"]; dev = worker_device(); soft_t = torch.from_numpy(soft).to(dev) if mapped else None
    Xe, ye = (d["Xe"].to(dev), d["ye"].to(dev)) if ext else (None, None); Xc, yc, Xc_in, yc_in = (d[k].to(dev) for k in ("Xc", "yc", "Xc_in", "yc_in"))

    def inner_score(net):
        Pi = proba(net, d["Xi"]); return float(balanced_mean_over_n(SM[mb.decisions(Pi, np.zeros_like(Pi), 0.5, 3.0, 0.6)[0], d["yi"]], d["yi"], d["ni"], sizes=SIZES))
    sc = [inner_score(train(Xc_in, yc_in, Xe, ye, ng, hid, lam, 0, soft=soft_t, epochs=e, dev=dev)) for e in G["grid"]]; epochs = G["grid"][int(np.argmax(sc))]
    nets = [train(Xc, yc, Xe, ye, ng, hid, lam, s, soft=soft_t, epochs=epochs, dev=dev) for s in G["seeds"]]; P = np.mean([proba(nt, d["Xv"]) for nt in nets], 0); assert np.isfinite(P).all()
    out = {"v": v, "name": name, "epochs": epochs, "P": P}
    if name == "A":
        out["PA_ext"] = np.mean([proba(nt, d["Xt"]) for nt in nets], 0)
    if name == "B":
        out["Ph"] = np.mean([proba(nt, d["Xh"], "x") for nt in nets], 0)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", action="append", required=True, metavar="SELECT PLYDIRS [RANKS]"); ap.add_argument("--per-group", type=int, default=1000); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true", help="ignore the GPUs (fallback)"); ap.add_argument("--save-proba", default=None, help="npz for every arm's validation probabilities (M110)"); a_ = ap.parse_args()
    import m103_stability as m3  # noqa: E402
    mb, SM = m3.mb, m3.SM; seeds = (0,) if a_.smoke else (0, 1, 2); folds = (1,) if a_.smoke else (1, 2, 3, 4); grid = (1, 3) if a_.smoke else EPOCH_GRID
    files = [f for v in (1, 2, 3, 4) for f in sorted((ROOT / f"results/trunk_vec/fold{v}").glob("games_s*_p*.csv"))]; fold_of = [int(f.parent.name[4:]) for f in files]; jobs, sels = [], []
    for src in a_.src:
        want = frozenset(src[2].split(",")) if len(src) > 2 else None; fs = [f for d in src[1].split(",") for f in sorted(Path(d).glob("ply_x_*.npz")) if not f.name.endswith(".tmp.npz")]; assert fs, f"no parts under {src[1]}"
        jobs += [(f, want) for f in fs]; s_ = pd.read_csv(src[0], usecols=["game_key", "account"], dtype=str); sels.append(s_ if want is None else s_[s_.game_key.str.split(":").str[1].isin(want)]); log(f"source {src[0]}: {len(fs)} parts, ranks {sorted(want) if want else 'all'}, {len(sels[-1])} selected games")
    log(f"allocation: {N_CPU} CPUs, {N_GPU} GPUs")
    with mp.get_context("fork").Pool(min(N_CPU, 48)) as pool:
        rc = pool.map(p6.comp_part, files, chunksize=4); rx = pool.map(ext_part_f, jobs, chunksize=4)
    key = np.concatenate([r[0] for r in rc]); pid = np.concatenate([r[1] for r in rc]); lab = np.array([p6.RANK[x] for x in np.concatenate([r[2] for r in rc])]); F0 = np.concatenate([r[3] for r in rc]); fold = np.concatenate([np.full(len(r[0]), v) for r, v in zip(rc, fold_of)])
    kx = np.concatenate([r[0] for r in rx]); Fx0 = np.concatenate([r[1] for r in rx]); del rc, rx; _, first = np.unique(kx, return_index=True); first.sort(); kx, Fx0 = kx[first], Fx0[first]; have = set(kx.tolist()); bad = False
    for src, s_ in zip(a_.src, sels):
        miss = int((~s_.game_key.isin(have)).sum()); bad |= miss > 0.005 * len(s_); log(f"    {src[0]}: selected games without per-ply vectors {miss} of {len(s_)} ({miss / max(len(s_), 1):.3%})")
    if bad and not a_.smoke:
        print("INCOMPLETE: a source's per-ply vectors are not all there yet; a real run needs complete sources (rerun when the extraction has finished)"); sys.exit(2)
    sel = pd.concat(sels).drop_duplicates("game_key").set_index("game_key")["account"]; acc = sel.reindex(kx).to_numpy(); has = pd.notna(acc); kx, Fx0, acc = kx[has], Fx0[has], acc[has].astype(str)
    grp = np.array([k.split(":")[1] for k in kx]); groups = sorted(set(grp), key=rank_order); gi = {g: i for i, g in enumerate(groups)}; rng0 = np.random.default_rng(0)
    xa = pd.DataFrame({"acc": acc, "g": [gi[g] for g in grp], "row": np.arange(len(kx))}).groupby(["acc", "g"]).agg(rows=("row", list)).reset_index(); xa = xa[xa.rows.map(len) >= 5].reset_index(drop=True)  # unit = (account, rank folder)
    keep = np.concatenate([rng0.permutation(np.flatnonzero(xa.g.to_numpy() == i))[:a_.per_group] for i in range(len(groups))]); xa = xa.iloc[np.sort(keep)].reset_index(drop=True); ua = np.unique(xa.acc); held = set(ua[rng0.random(len(ua)) < 0.10]); hold = xa.acc.isin(held).to_numpy(); ng = xa.rows.map(len).to_numpy()
    log(f"competition: {len(key)} games, {len(set(pid))} players | external: {len(groups)} ranks {[str(g) for g in groups]}; units per rank {np.bincount(xa.g.to_numpy(), minlength=len(groups)).tolist()}; median games per unit by rank {[int(np.median(ng[xa.g.to_numpy() == i])) for i in range(len(groups))]}; {int(hold.sum())} units of held-out accounts")
    players = pd.DataFrame({"pid": pid, "lab": lab, "fold": fold, "row": np.arange(len(key))}).groupby("pid").agg(lab=("lab", "first"), fold=("fold", "first"), rows=("row", list))
    ARMS = {"A": (None, (256,), 0.0, False), "B": ("x", (256,), 1.0, False)}
    if not a_.smoke:
        ARMS.update({"B lambda 0.3": ("x", (256,), 0.3, False), "B lambda 3": ("x", (256,), 3.0, False), "A big": (None, (512, 512), 0.0, False), "B big": ("x", (512, 512), 1.0, False), "C mapped labels": ("x", (256,), 1.0, True)})
    G.update(F0=F0, Fx0=Fx0, fold=fold, players=players, xa=xa, hold=hold, row_of={k: i for i, k in enumerate(key)}, m3=m3, arms=ARMS, grid=grid, seeds=seeds, n_groups=len(groups))
    with mp.get_context("fork").Pool(min(4, len(folds))) as pool:
        G["folds"] = dict(pool.map(prep_fold, folds))
    log(f"fold data ready ({', '.join(str(len(G['folds'][v]['Xc'])) + ' competition / ' + str(len(G['folds'][v]['Xe'])) + ' external training questions' for v in folds[:1])} in the first fold)"); R = {}
    with mp.get_context("fork").Pool(min(N_CPU, 12 * N_GPU, 28) if N_GPU else min(N_CPU, 28)) as pool:  # forked after G holds the fold data; the parent has not touched CUDA
        first_jobs = sorted(((v, k, None) for v in folds for k in ARMS if not ARMS[k][3]), key=lambda j: -(len(ARMS[j[1]][1]) * 2 + (ARMS[j[1]][0] is not None)))  # the long ones first
        for o in pool.imap_unordered(run_arm, first_jobs, chunksize=1):
            R[(o["v"], o["name"])] = o
        log(f"{len(R)} (fold, arm) trainings done"); maps = {v: np.stack([R[(v, "A")]["PA_ext"][G["folds"][v]["gt"] == i].mean(0) for i in range(len(groups))]).astype(np.float32) for v in folds}  # each external rank as a distribution over the competition's levels
        for o in pool.imap_unordered(run_arm, [(v, k, maps[v]) for v in folds for k in ARMS if ARMS[k][3]], chunksize=1):
            R[(o["v"], o["name"])] = o
    if a_.smoke:
        log(f"SMOKE: every arm trained and predicted in parallel ({sorted(R)}); scores are not printed in a smoke run"); return
    if a_.save_proba:
        np.savez_compressed(a_.save_proba, **{f"{k.replace(' ', '_')}__{v}": R[(v, k)]["P"].astype(np.float32) for v in folds for k in ARMS})
    S = {k: [] for k in ARMS}; picked = {k: [] for k in ARMS}; ys, ns, ps, ext_acc = [], [], [], []
    for v in folds:
        d = G["folds"][v]; y, n = d["y"], d["n"]; ys.append(y); ns.append(n); ps.append(d["pid"])
        for k in ARMS:
            P = R[(v, k)]["P"]; S[k].append(SM[mb.decisions(P, np.zeros_like(P), 0.5, 3.0, 0.6)[0], y]); picked[k].append(int(R[(v, k)]["epochs"]))
        dh = R[(v, "B")]["Ph"].argmax(1); ext_acc.append((float((dh == d["gh"]).mean()), float((np.abs(dh - d["gh"]) <= 1).mean())))
        log(f"fold {v}: " + "  ".join(f"{k} {balanced_mean_over_n(S[k][-1], y, n, sizes=SIZES):.4f} (epochs {picked[k][-1]})" for k in ARMS) + f" | external head on held-out accounts: exact {ext_acc[-1][0]:.3f}, within one rank {ext_acc[-1][1]:.3f}")
    y, n, pidv = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps); rep = {"groups": [str(g) for g in groups], "ext_head": ext_acc, "epochs": picked, "arms": {}}

    def cmp(new, old):
        d = paired_diff(np.concatenate(S[new]), np.concatenate(S[old]), y, pidv, n, sizes=SIZES, n=3000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(S[new], S[old], ys, ns)]
        print(f"{new:16s} vs {old:6s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True); rep["arms"][f"{new} vs {old}"] = {**d, "per_fold": pf}; return d, pf
    for k in ARMS:
        print(f"{k:16s} official score {np.mean([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k], ys, ns)]):.4f}")
    d, pf = cmp("B", "A"); gate = (d["diff"] >= 0.005) and (d["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3)
    cmp("B lambda 0.3", "A"); cmp("B lambda 3", "A"); cmp("B big", "A big"); cmp("A big", "A"); cmp("C mapped labels", "A"); sA, sB = np.concatenate(S["A"]), np.concatenate(S["B"])
    print("B - A by true level: " + "  ".join(f"{LEVELS[c]} {float((sB - sA)[y == c].mean()):+.4f}" for c in range(10)))
    M = np.mean([maps[v] for v in folds], 0); print("external rank -> expected competition level (arm A's reading, 0 = D ... 9 = 6D): " + "  ".join(f"{g} {float(M[i] @ np.arange(10)):.2f}" for i, g in enumerate(groups))); rep["rank_map"] = M.tolist()
    print(f"REGISTERED GATE (B vs A: >= +0.005, CI lower bound > 0, >= 3 of 4 folds): {'PASS' if gate else 'NO'}"); rep["gate"] = bool(gate)
    (ROOT / "results/r_variants/m107_multitask_probe.json").write_text(json.dumps(rep, indent=1, default=float)); log("written")


if __name__ == "__main__":
    main()
