#!/usr/bin/env python
"""Section 4M M150 (2026-11-15, existence pilot, report only): are there RECURRING opponents without a label? (design doc M150)

About 30-50% of the opponents are not labelled training players; the anchors read nothing from them. If some of them are
regulars (the same person on the opponent side of many training players' games), they can be found without labels: the
opponent-side identity vectors of their rows form a tight cluster. Such a cluster would be a new anchor candidate whose reading is
the labels of the training players who met it.

One GPU. The all-players symmetry networks (seeds --seeds) embed the opponent side of every training row (subject colour
flipped); per row: the best labelled candidate (owner masked, per-row standardised scores averaged over the members, m15 margin)
and the 16 most similar rows of OTHER owners (cosine, averaged over the members). Then on the CPU:
  - precision of an edge by similarity, judged on pairs of rows that are both confidently recognised as a labelled player;
  - clusters = connected components of the mutual-neighbour graph at a similarity threshold (rows with >= 2 such edges);
  - a cluster is "labelled" when one labelled candidate is the best candidate of >= 40% of its rows, "unlabelled" when no
    candidate reaches 15%; counts, rows covered, number of distinct owners, spread of the owners' labels (share within one
    level of the cluster's median) against random groups; recall of the known regulars.
Registered reading: unlabelled clusters with >= 5 distinct owners cover >= 5% of the rows and their owners' labels are within
one level of the median in >= 0.80 -> build the formal version (cross-fitted, a new pool source, M37); under 1% -> closed;
in between -> reported to the user.

  python -I logs/nano4/m150_outsiders.py [--seeds 0,1,3] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402

CK, M120, OUT = Path("/work/u2507115/m120_ckpt"), ROOT / "results/m120", ROOT / "results/m150"; K = 16


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,3"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--analysis-only", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True); seeds = [int(x) for x in a_.seeds.split(",")]; q = np.load(base.CACHE / "queries.npz"); players = q["players"].astype(str); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); plab = pf["rank"].map({r: i for i, r in enumerate(["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"])}).reindex(players).to_numpy().astype(int)
    if not a_.analysis_only:
        dev = "cuda"; base.FULL, base.EXT = True, None; G, _ = base.load_games(False, log); N = len(G["key"]) if not a_.limit else min(a_.limit, len(G["key"]))
        owner = np.array([pidx[p] for p in G["pid"][:N]]); tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
        E = []; PR = []
        for s in seeds:
            tag = f"-1{f'_s{s}' if s else ''}"; st = torch.load(CK / f"ckpt_{tag}.pt", map_location=dev, weights_only=False); assert "opt" not in st; net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); e = torch.zeros((N, base.D_OUT), dtype=torch.float16, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, N, 1024):
                    r = slice(i, min(i + 1024, N)); e[r] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb((1 - G["SUBJ"][r]).astype(np.int64))).to(torch.float16)
            E.append(e); PR.append(tb(np.load(M120 / f"out_{tag}.npz")["protos"]).float()); del net; log(f"seed {s}: opponent side of {N} rows embedded")
        own_t = tb(owner); c1 = np.zeros(N, np.int64); m15 = np.zeros(N, np.float32); nn_i = np.zeros((N, K), np.int64); nn_s = np.zeros((N, K), np.float32); B = 2048
        with torch.no_grad():
            for i in range(0, N, B):
                r = slice(i, min(i + B, N)); z = 0.0; S = 0.0
                for e, pr in zip(E, PR):
                    x = e[r].float(); sc = x @ pr.T; sc[torch.arange(sc.shape[0], device=dev), own_t[r]] = float("nan"); mu = sc.nanmean(1, keepdim=True); sd = ((sc - mu) ** 2).nanmean(1, keepdim=True).sqrt().clamp(min=1e-9); z = z + ((sc - mu) / sd).nan_to_num(nan=-1e9) / len(E)
                    S = S + (e[r] @ e.T).float() / len(E)
                top = z.topk(5, dim=1); c1[r] = top.indices[:, 0].cpu().numpy(); m15[r] = (top.values[:, 0] - top.values[:, 1:5].mean(1)).cpu().numpy()
                S.masked_fill_(own_t[r][:, None] == own_t[None, :], -2.0); tk = S.topk(K, dim=1); nn_i[r] = tk.indices.cpu().numpy(); nn_s[r] = tk.values.cpu().numpy()
                if (i // B) % 100 == 0:
                    log(f"  rows {i} / {N}")
        np.savez(OUT / "pilot.npz", owner=owner, c1=c1, m15=m15, nn_i=nn_i.astype(np.int32), nn_s=nn_s.astype(np.float16), key=G["key"][:N]); log("neighbours written"); del G, E
    zz = np.load(OUT / "pilot.npz", allow_pickle=True); owner, c1, m15, nn_i, nn_s = zz["owner"], zz["c1"], zz["m15"], zz["nn_i"].astype(np.int64), zz["nn_s"].astype(np.float32); N = len(owner); rep = {"rows": int(N)}
    lab_o = plab[owner]; conf = m15 >= np.quantile(m15, 0.97); print(f"\nrows {N}; owners {len(np.unique(owner))}; similarity of the nearest row of another owner: quartiles " + " / ".join(f"{x:.3f}" for x in np.quantile(nn_s[:, 0], [0.25, 0.5, 0.75, 0.9, 0.99])) + f"; m15 margin of the best labelled candidate: q50 {np.quantile(m15, 0.5):.2f}, q90 {np.quantile(m15, 0.9):.2f}, q97 {np.quantile(m15, 0.97):.2f}")
    # precision of an edge by similarity, on pairs of rows that are both confidently recognised
    src = np.repeat(np.arange(N), K); dst = nn_i.reshape(-1); sim = nn_s.reshape(-1); both = conf[src] & conf[dst]; same = c1[src] == c1[dst]
    print("edge precision among pairs of confidently recognised rows (same best candidate), by similarity: " + "  ".join(f"[{lo:.2f},{hi:.2f}): {float(same[both & (sim >= lo) & (sim < hi)].mean()) if (both & (sim >= lo) & (sim < hi)).any() else float('nan'):.3f} (n {int((both & (sim >= lo) & (sim < hi)).sum())})" for lo, hi in ((0.5, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.9), (0.9, 1.01))))
    print("share of all rows whose nearest other-owner row has similarity >= t: " + "  ".join(f"{t}: {float((nn_s[:, 0] >= t).mean()):.3f}" for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85)) + " | among confidently recognised rows: " + "  ".join(f"{t}: {float((nn_s[conf, 0] >= t).mean()):.3f}" for t in (0.6, 0.7, 0.8)) + " | among the least recognised half: " + "  ".join(f"{t}: {float((nn_s[m15 < np.median(m15), 0] >= t).mean()):.3f}" for t in (0.6, 0.7, 0.8)))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(0); key_fwd = src * N + dst  # mutual edges via sorting
    order = np.argsort(key_fwd); kf = key_fwd[order]; kb = dst * N + src; pos = np.searchsorted(kf, kb); pos[pos >= len(kf)] = 0; mutual = kf[pos] == kb
    for tau in (0.65, 0.70, 0.75, 0.80):
        e_ok = mutual & (sim >= tau); deg = np.bincount(src[e_ok], minlength=N); keep = e_ok & (deg[src] >= 2) & (deg[dst] >= 2)
        _, root = connected_components(coo_matrix((np.ones(int(keep.sum()), np.int8), (src[keep], dst[keep])), shape=(N, N)).tocsr(), directed=False)
        inv, cnt = np.unique(root, return_counts=True); big = inv[cnt >= 8]; member = np.isin(root, big)
        rows = pd.DataFrame({"root": root[member], "c1": c1[member], "owner": owner[member], "lab": lab_o[member], "conf": conf[member]})
        g = rows.groupby("root"); size = g.size(); top_share = g.c1.agg(lambda x: x.value_counts().iloc[0] / len(x)); top_c = g.c1.agg(lambda x: x.value_counts().index[0]); n_own = g.owner.nunique(); med = g.lab.median(); rows["med"] = rows.root.map(med); rows["w1"] = (rows.lab - rows.med).abs() <= 1
        w1 = rows.groupby("root").w1.mean(); cand_gap = (pd.Series(plab[top_c.to_numpy()], index=top_c.index) - med).abs()
        labelled = top_share >= 0.40; unlab = top_share < 0.15; multi = n_own >= 5
        rw = np.concatenate([np.abs(x - np.median(x)) <= 1 for x in (rng.choice(lab_o, int(s_)) for s_ in size.to_numpy()[:400])]).mean() if len(size) else float("nan")
        out = {"clusters": int(len(size)), "rows_in_clusters": int(size.sum()), "labelled": int(labelled.sum()), "rows_labelled": int(size[labelled].sum()), "unlabelled": int(unlab.sum()), "rows_unlabelled": int(size[unlab].sum()), "unlabelled_multi_owner": int((unlab & multi).sum()), "rows_unlabelled_multi_owner": int(size[unlab & multi].sum()),
               "within1_labelled": float((w1[labelled] * size[labelled]).sum() / max(size[labelled].sum(), 1)), "within1_unlabelled_multi": float((w1[unlab & multi] * size[unlab & multi]).sum() / max(size[unlab & multi].sum(), 1)), "within1_random": float(rw)}
        rep[f"tau_{tau}"] = out
        print(f"\nthreshold {tau}: clusters of >= 8 rows: {out['clusters']} with {out['rows_in_clusters']} rows ({out['rows_in_clusters'] / N:.3f} of all rows)")
        print(f"   labelled (one candidate is the best candidate of >= 40% of the rows): {out['labelled']} clusters, {out['rows_labelled']} rows ({out['rows_labelled'] / N:.3f}); owners' labels within one level of the cluster median {out['within1_labelled']:.3f}; the candidate's own label within one of that median in {float((cand_gap[labelled] <= 1).mean()) if labelled.any() else float('nan'):.3f} of the clusters")
        print(f"   unlabelled (no candidate reaches 15%): {out['unlabelled']} clusters, {out['rows_unlabelled']} rows ({out['rows_unlabelled'] / N:.3f}); with >= 5 distinct owners: {out['unlabelled_multi_owner']} clusters, {out['rows_unlabelled_multi_owner']} rows ({out['rows_unlabelled_multi_owner'] / N:.4f}); owners' labels within one of the median {out['within1_unlabelled_multi']:.3f} (random groups {out['within1_random']:.3f})")
        print(f"   in between (15-40%): {int((~labelled & ~unlab).sum())} clusters, {int(size[~labelled & ~unlab].sum())} rows; sizes of the unlabelled multi-owner clusters: " + (" / ".join(f"{x:.0f}" for x in size[unlab & multi].quantile([0.5, 0.9, 0.99, 1.0]).to_numpy()) if (unlab & multi).any() else "-") + f"; mean share of confidently recognised rows inside them {float(rows[rows.root.isin(size[unlab & multi].index)].conf.mean()) if (unlab & multi).any() else float('nan'):.3f}")
        # recall of the known regulars: candidates that are the confident best candidate of >= 40 rows
        cc = pd.Series(c1[conf]).value_counts(); hubs = cc[cc >= 40].index.to_numpy(); dom = dict(zip(top_c.index[labelled], top_c[labelled])); hit = np.array([dom.get(r_) == c_ for r_, c_ in zip(root[conf], c1[conf])]); ish = np.isin(c1[conf], hubs)
        print(f"   known regulars (best candidate of >= 40 confidently recognised rows): {len(hubs)} players; share of their confident rows that sit in a cluster dominated by them {float(hit[ish].mean()) if ish.any() else float('nan'):.3f}")
    (ROOT / "results/r_variants/m150_outsiders.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nwritten results/r_variants/m150_outsiders.json")


if __name__ == "__main__":
    main()
