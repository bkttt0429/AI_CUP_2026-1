"""Section 4M M85 (2026-11-11): the mirror channel. Forward anchors look for a training player among the opponents inside a question's
own 5-20 games. Mirror: a question's player also appears as the OPPONENT inside training rows, whose owner's level is known for certain;
the uncertain part is whether the opponent side of that row is this player. Every game belongs to one player only (M83: no duplicated
games), so these are different games from the forward channel's.
First version, matcher only: candidate = a question (prototype = sum of the matcher vectors of the question's own-side rows), query = the
opponent-side row of a training row (one game), score = cosine, top-1 and margin.
Validation (fold v, network trained without fold v's players): every fold-v player gets ONE of his questions (sizes 5 / 10 / 20 with
probability .25 / .25 / .5), the players are split into two half-pools (about 800, as the test); queries = the rows whose owner is not
in fold v; null = the own-side rows of the OTHER half-pool's players (never seen by the network, certainly not in the pool); bands at
the null's 0.9 / 0.99 / 0.999 / 0.9999 quantiles; REPS different assignments. Mixture calibration as M71 (groups HD0 / HD1 x target
colour, a_b, P0, K), cross-fitted over folds. Primary: base + forward 5e anchors + mirror vs base + forward 5e anchors on the assigned
questions, M37, seeds 0-2 and 6-8. Test side (only if it passes): 800 questions, queries = all training rows, all-players network,
validation's mean band edges, density ratio, two safety checks -> ct56 = ct44's posterior x mirror evidence, same composition.
  python -I logs/nano4/m85_mirror.py [--rows stored|all] [--tag smoke]
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
import m82_cond_id as m82  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
RV, M82 = ROOT / "results/r_variants", ROOT / "results/m82"
QS = np.array([0.9, 0.99, 0.999, 0.9999]); NOM = np.diff(np.concatenate([[0.0], QS, [1.0]])); NB = len(QS) + 1
REPS, STEPS, EPISODE, S_EP, W_MAX, MIN_GAMES = 4, 6000, 512, 64, 256, 8
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)


def load_all(which, log):
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}
    meta = pd.DataFrame({"owner": rows.player_id.map(pidx).to_numpy(), "lab": rows["rank"].map(RANK).to_numpy(), "black": (rows.color == "B").to_numpy(),
                         "hd": rows.sgf_content.str.extract(r"HD\[(\d+)\]")[0].fillna("0").astype(int).to_numpy()}, index=rows.game_key.to_numpy()); del rows
    fo = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; pfold = np.array([int(fo.get(p, -9)) for p in players])
    B = np.load(M82 / "rows_train.npz"); keys, Ts, Os = [B["key"].astype(str)], [B["T"]], [B["O"]]
    with mp.get_context("fork").Pool(12) as pool:
        res = pool.map(m82.part_rows, sorted((ROOT / "results/trunk_vec_x80").glob("fold*/games_s*_p*.csv")), chunksize=4)
    keys.append(np.concatenate([t["game_key"].to_numpy() for t, _, _ in res]).astype(str)); Ts.append(np.concatenate([o[0] for _, o, _ in res])); Os.append(np.concatenate([o[1] for _, o, _ in res]))
    if which == "all":
        for f in sorted((ROOT / "results/m84_rowsp").glob("rows_*.npz")):
            if not f.name.endswith(".tmp.npz"):
                z = np.load(f); keys.append(z["key"].astype(str)); Ts.append(z["T"]); Os.append(z["O"])
    key = np.concatenate(keys); _, first = np.unique(key, return_index=True); first.sort(); key = key[first]; T = np.concatenate(Ts)[first]; O = np.concatenate(Os)[first]; m = meta.loc[key]; assert len(m) == len(key)
    D = dict(key=key, T=T, O=O, owner=m.owner.to_numpy(), lab=m.lab.to_numpy(), black=m.black.to_numpy(), hd=m.hd.to_numpy(), players=players, pidx=pidx, pfold=pfold)
    D["fold"] = pfold[D["owner"]]; D["grp"] = s2.group_of(D["hd"], ~D["black"])  # the "target" of a mirror query is the opponent side of the row
    log(f"rows ({which}): {len(key)} games of {len(np.unique(D['owner']))} players; HD0 {float((D['hd'] == 0).mean()):.3f}, HD1 {float((D['hd'] == 1).mean()):.3f}")
    return D


def train(D, v, seed, dev):
    """The matcher (M82 / M84 recipe) on the own-side rows of the players outside fold v; returns enc(rows [n, 256]) -> unit vectors on the GPU."""
    import torch
    from aicup_pid.embed import EmbedNet
    T, owner, C = D["T"], D["owner"], len(D["players"]); torch.manual_seed(7000 + 10 * seed + v); rng = np.random.default_rng(7000 + 10 * seed + v)
    use = D["fold"] != v; n_rows = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(n_rows >= MIN_GAMES); W = int(min(n_rows.max(), W_MAX))
    idx = rng.permutation(np.flatnonzero(use)); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos = np.arange(len(o)) - np.searchsorted(o, o, side="left"); k_ = pos < W
    tab = np.full((C, W), -1, np.int64); tab[o[k_], pos[k_]] = idx[k_]; cnt = np.minimum(n_rows, W); mu = T[use].mean(0); sd = T[use].std(0) + 1e-6
    std = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731
    X = std(T); tab_t = torch.as_tensor(tab[tp], device=dev); cnt_t = torch.as_tensor(cnt[tp], device=dev); P = min(EPISODE, len(tp)); S = min(S_EP + 1, W)
    net = EmbedNet(256, 512, 128, 0.1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
    ar = torch.arange(W, device=dev); tgt = torch.arange(P, device=dev); net.train()
    for _ in range(STEPS):
        b = torch.randperm(len(tp), device=dev)[:P]; c = cnt_t[b]; u = torch.rand(P, W, device=dev).masked_fill(ar[None] >= c[:, None], 2.0)
        rows = tab_t[b].gather(1, u.argsort(dim=1)[:, :S]); zq = net(X[rows[:, 0]]); sm = ar[None, 1:S] < c[:, None]
        zs = net(X[rows[:, 1:].clamp(min=0)].reshape(-1, 256)).reshape(P, S - 1, -1) * sm[..., None]
        logits = net.log_scale.exp() * zq @ torch.nn.functional.normalize(zs.sum(1), dim=-1).T
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); del X

    def enc(A):
        with torch.no_grad():
            return torch.cat([net(std(A[i:i + 65536])) for i in range(0, len(A), 65536)])
    return enc, float(loss.detach())


def top2(E, P, chunk=65536):
    """Top-1 candidate and margin of every row of E against the prototypes P (cosine; both unit vectors)."""
    import torch
    j, m = [], []
    for i in range(0, len(E), chunk):
        t = (E[i:i + chunk] @ P.T).topk(2, dim=1); j.append(t.indices[:, 0]); m.append(t.values[:, 0] - t.values[:, 1])
    return torch.cat(j), torch.cat(m)


def calibrate(counts, hist0, htop):
    """Per group: a by band, K / P0 ratio by label difference (M71's mixture; counts [3, NB], hist0 / htop [3, 19])."""
    cal = {}
    for g in range(3):
        share = counts[g] / max(counts[g].sum(), 1); a = np.clip(1 - NOM / np.maximum(share, 1e-12), 0, 0.98); P0 = (hist0[g] + 1.0) / (hist0[g].sum() + 19)
        a_top = float(np.clip(1 - NOM[3:].sum() / max(share[3:].sum(), 1e-12), 0, 0.98)); K = np.clip((htop[g] + 1.0) / (htop[g].sum() + 19) - (1 - a_top) * P0, 0, None); K = K / K.sum()
        if a_top < 0.05:  # no excess over the null in the top bands: K is not identified, so this group carries no evidence (added after the smoke run)
            a = np.zeros(NB); K = P0.copy()
        cal[g] = {"a": a, "ratio": np.minimum(K / P0, 30.0), "share": share, "a_top": a_top, "K": K, "P0": P0, "n_top": int(htop[g].sum())}
    return cal


def evidence(cal, lab, grp, band, scale=1.0):
    """log e(c) [n, 10] of matches (owner label, group, band)."""
    out = np.zeros((len(lab), 10))
    for g in range(3):
        s = grp == g
        if s.any():
            a = (scale * cal[g]["a"][band[s]])[:, None]; out[s] = np.log((1 - a) + a * cal[g]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9])
    return out


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="all", choices=("stored", "all")); ap.add_argument("--tag", default="mirror"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; D = load_all(a_.rows, log); G = len(D["key"]); kidx = {k: i for i, k in enumerate(D["key"])}; lab_t = torch.as_tensor(D["lab"], device=dev); grp_t = torch.as_tensor(D["grp"], device=dev)
    REC = {}; EDGES = []
    for v in FOLDS:
        enc, loss = train(D, v, 0, dev); E_T, E_O = enc(D["T"]), enc(D["O"]); y, n, pid, q = load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(q["game_keys"])
        qrows = [torch.as_tensor([kidx[k] for k in g_ if k in kidx], dtype=torch.long, device=dev) for g_ in gk]; by_p = defaultdict(lambda: defaultdict(list))
        for i, (p_, m_) in enumerate(zip(pid, n)):
            by_p[p_][int(m_)].append(i)
        plist = np.array(sorted(by_p)); pown = np.array([D["pidx"][p_] for p_ in plist]); qi = torch.as_tensor(np.flatnonzero(D["fold"] != v), device=dev); EQ = E_O[qi]; ql, qg = lab_t[qi], grp_t[qi]; y_t = torch.as_tensor(y, device=dev)
        counts = np.zeros((3, NB)); hist0 = np.zeros((3, 19)); M = defaultdict(list); assigned = []
        for rep in range(REPS):
            rng = np.random.default_rng(100 * v + rep); perm = rng.permutation(len(plist)); halves = (perm[:len(perm) // 2], perm[len(perm) // 2:]); asg = np.zeros(len(plist), np.int64)
            for i_, p_ in enumerate(plist):
                size = int(rng.choice([5, 10, 20], p=[0.25, 0.25, 0.5])); size = size if size in by_p[p_] else sorted(by_p[p_])[-1]; asg[i_] = int(rng.choice(by_p[p_][size]))
            assigned.append(asg)
            for h in (0, 1):
                cq = torch.as_tensor(asg[halves[h]], device=dev); P = torch.stack([torch.nn.functional.normalize(E_T[qrows[i]].sum(0), dim=0) for i in asg[halves[h]]])
                null = torch.as_tensor(np.flatnonzero(np.isin(D["owner"], pown[halves[1 - h]])), device=dev); _, mn = top2(E_T[null], P); edges = torch.quantile(mn[torch.randperm(len(mn), device=dev)[:4_000_000]], torch.as_tensor(QS, device=dev, dtype=mn.dtype))
                j, m = top2(EQ, P); band = torch.bucketize(m, edges); d = (ql - y_t[cq[j]] + 9).clamp(0, 18)
                for g in range(3):
                    s = qg == g; counts[g] += torch.bincount(band[s], minlength=NB).cpu().numpy(); hist0[g] += torch.bincount(d[s & (band == 0)], minlength=19).cpu().numpy()
                k = (band >= 1) & (qg >= 0)
                for name, arr in (("rep", torch.full((int(k.sum()),), rep, device=dev)), ("q", cq[j][k]), ("lab", ql[k]), ("grp", qg[k]), ("band", band[k])):
                    M[name].append(arr.cpu().numpy())
                EDGES.append(edges.cpu().numpy())
        REC[v] = {"y": y, "n": n, "pid": pid, "gk": gk, "counts": counts, "hist0": hist0, "assigned": assigned, **{k_: np.concatenate(x) for k_, x in M.items()}}
        htop = np.zeros((3, 19)); r = REC[v]; s = r["band"] >= 3; np.add.at(htop, (r["grp"][s], np.clip(r["lab"][s] - y[r["q"][s]] + 9, 0, 18)), 1); r["htop"] = htop
        log(f"fold {v}: matcher loss {loss:.3f}; {len(qi)} queries x {REPS} assignments x 2 half-pools of ~{len(plist) // 2}; HD0 share by band " + " ".join(f"{x:.5f}" for x in counts[0] / counts[0].sum()) + f" (nominal {' '.join(f'{x:.5f}' for x in NOM)}); matches at >= q99.9: {int(s.sum())}")
        del E_T, E_O, EQ
    cal_all = calibrate(sum(REC[v]["counts"] for v in FOLDS), sum(REC[v]["hist0"] for v in FOLDS), sum(REC[v]["htop"] for v in FOLDS)); c0 = cal_all[0]
    log("HD0 calibration (all folds): a by band " + " ".join(f"{x:.2f}" for x in c0["a"]) + f" | a_top {c0['a_top']:.2f}, top matches {c0['n_top']} | K(-2..2) " + " ".join(f"{c0['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)) + " | P0(-2..2) " + " ".join(f"{c0['P0'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)))
    # evaluation: mirror evidence (cross-fitted calibration) on top of the forward 5e anchors, on the assigned questions
    d82 = dict(np.load(RV / "m82_val.npz")); Z = {**{k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}
    LM, LF = {}, {}
    for v in FOLDS:
        oth = [u for u in FOLDS if u != v]; cal = calibrate(sum(REC[u]["counts"] for u in oth), sum(REC[u]["hist0"] for u in oth), sum(REC[u]["htop"] for u in oth)); r = REC[v]; logE = evidence(cal, r["lab"], r["grp"], r["band"])
        LM[v] = np.zeros((REPS, len(r["y"]), 10)); np.add.at(LM[v], (r["rep"], r["q"]), logE)
        te, lf, _ = s2.evidence_per_query(Z, v); key_row = {k: i for i, k in enumerate(Z["key"][te])}; LF[v] = np.zeros((len(r["y"]), 10))
        for i, g_ in enumerate(r["gk"]):
            rr = [key_row[k] for k in g_ if k in key_row]
            if rr:
                LF[v][i] = lf[rr].sum(axis=0)
    rep_out = {"rows": int(G), "calibration_hd0": {"a": c0["a"].tolist(), "a_top": c0["a_top"], "n_top": c0["n_top"], "share": c0["share"].tolist()}}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        cols = defaultdict(list)
        for v in FOLDS:
            r = REC[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            sc = lambda L, sel: SM[((P[sel] * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), r["y"][sel]]  # noqa: E731
            for rp in range(REPS):
                sel = r["assigned"][rp]; cols["base"].append(sc(np.zeros((len(sel), 10)), sel)); cols["fwd"].append(sc(LF[v][sel], sel)); cols["mir"].append(sc(LM[v][rp][sel], sel)); cols["both"].append(sc(LF[v][sel] + LM[v][rp][sel], sel))
                cols["y"].append(r["y"][sel]); cols["n"].append(r["n"][sel]); cols["pid"].append(r["pid"][sel]); cols["fold"].append(np.full(len(sel), v))
        A = {k: np.concatenate(x) for k, x in cols.items()}; res = {}
        for label, a, b in (("forward_vs_base", "fwd", "base"), ("mirror_vs_base", "mir", "base"), ("both_vs_forward", "both", "fwd")):
            dd = paired_diff(A[a], A[b], A["y"], A["pid"], A["n"], sizes=SIZES, n=5000); pf = [balanced_mean_over_n(A[a][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) - balanced_mean_over_n(A[b][A["fold"] == v], A["y"][A["fold"] == v], A["n"][A["fold"] == v], sizes=SIZES) for v in FOLDS]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}
            log(f"  seeds {seeds} {label:16s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x:+.4f}" for m_, x in dd["by_n"].items()) + f"  changed {float((A[a] != A[b]).mean()):.3f} -> {'PASS' if good else 'no'}")
        rep_out[f"seeds_{seeds[0]}"] = res; ok.append(res["both_vs_forward"]["pass"])
    passed = bool(all(ok)); log(f"PRE-REGISTERED: mirror evidence on top of the forward anchors: {'PASS' if passed else 'NO'}")
    # test side
    enc, _ = train(D, -1, 0, dev); E_O = enc(D["O"]); Rt = np.load(M82 / "rows_test.npz"); pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); ET = enc(Rt["T"]); tq = Rt["qid"]
    P = torch.stack([torch.nn.functional.normalize(ET[torch.as_tensor(np.flatnonzero(tq == x), device=dev)].sum(0), dim=0) for x in qids]); edges = torch.as_tensor(np.mean(EDGES, axis=0), device=dev, dtype=E_O.dtype)
    j, m = top2(E_O, P); band = torch.bucketize(m, edges); g0 = grp_t == 0; sh_t = (torch.bincount(band[g0], minlength=NB).float() / g0.sum()).cpu().numpy(); sh_v = c0["share"]
    ratios = [((sh_t[b] - NOM[b]) / (sh_v[b] - NOM[b]) if sh_v[b] - NOM[b] > 1e-5 else 0.0) for b in (2, 3, 4)]; r_m = float(np.clip(min(ratios), 0, 1)); null_ok = abs(sh_t[1] / sh_v[1] - 1) <= 0.25
    log("test, HD0 share by band " + " ".join(f"{x:.5f}" for x in sh_t) + " | validation " + " ".join(f"{x:.5f}" for x in sh_v) + f" | excess ratios at >= q99: {' '.join(f'{x:.2f}' for x in ratios)} -> r = {r_m:.3f}; q90-q99 band test / validation {sh_t[1] / sh_v[1]:.3f} -> null check {'OK' if null_ok else 'FAILED'}")
    k = ((band >= 1) & (grp_t >= 0)).cpu().numpy(); jn = j.cpu().numpy()[k]; logE = evidence(cal_all, D["lab"][k], D["grp"][k], band.cpu().numpy()[k], scale=r_m); LT = np.zeros((len(qids), 10)); np.add.at(LT, jn, logE)
    cols_ = [f"p_{lv}" for lv in LEVELS]; P9 = pr[cols_].to_numpy(float); P44 = pd.read_csv(ROOT / "submission_rank_ct44_proba.csv", dtype={"question_id": str}).set_index("question_id").loc[qids, cols_].to_numpy(float)
    sub = lambda name: pd.read_csv(ROOT / f"submission_rank_{name}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44 = sub("ct9"), sub("ct28"), sub("ct44"); lev = np.array(LEVELS); locked = c28 != c9; LF_t = np.log(np.maximum(P44, 1e-300)) - np.log(np.maximum(P9, 1e-300))
    base44 = np.where(locked, c28, lev[mb.decisions(P9, LF_t, 0.5, 3.0, 0.6)[2]]); repro = bool((base44 == c44).all()); final = np.where(locked, c28, lev[mb.decisions(P9, LF_t + LT, 0.5, 3.0, 0.6)[2]]); ch = final != c44; pub = np.char.startswith(qids.astype(str), "pub_")
    jump = np.abs(np.array([RANK[x] for x in final]) - np.array([RANK[x] for x in c44]))
    log(f"reproduction of ct44 without mirror evidence: {'exact' if repro else f'FAILED ({int((base44 != c44).sum())} questions differ)'}; questions with any mirror evidence {int((np.abs(LT).sum(1) > 0).sum())} of {len(qids)}; with mirror: {int(ch.sum())} differ from ct44 (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}; one level {int((jump == 1).sum())}, two or more {int((jump >= 2).sum())})")
    rep_out.update(test={"share": sh_t.tolist(), "r": r_m, "null_ok": bool(null_ok), "repro_ct44": repro, "changed_vs_ct44": int(ch.sum()), "public": int(ch[pub].sum()), "private": int(ch[~pub].sum())}, passed_cv=passed)
    adopt = bool(passed and repro and null_ok and a_.tag == "mirror" and a_.rows == "all"); rep_out["adopt"] = adopt
    if adopt:
        f = ROOT / "submission_rank_ct56.csv"; write_submission(list(zip(qids, final)), f); rep_out["md5"] = hashlib.md5(f.read_bytes()).hexdigest()[:12]; log(f"ct56 written, md5 {rep_out['md5']}")
    (RV / f"m85_{a_.tag}_eval.json").write_text(json.dumps(rep_out, indent=1, default=float)); log("written")
    if a_.tag == "mirror":
        subprocess.run([sys.executable, "-I", "logs/nano4/m84_variant.py", "--stage", "finalize"], cwd=ROOT)


if __name__ == "__main__":
    main()
