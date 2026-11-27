#!/usr/bin/env python
"""Section 4M M200 (copy of m161r_tta.py for the networks of m200_prod.py): orientation-averaged queries, with the choice
residual as the extra input, and the learned centres as a second source.

m124_tta_queries.py is reused for the job handling and the merge; only its per-network function is replaced by a copy that
passes the residual rows (unchanged by the board symmetry, like the engine vectors), and its loader attaches them.

  python -I logs/nano4/m200_tta.py --seeds 30 --name S9t
      reads results/m200 and /work/u2507115/m200_ckpt (width 512 / feed-forward 2,048), parts in results/m200_tta, writes
      results/m84_cache/var_S9t.npz (prototypes; m84_variant's format) and var_S9tc.npz (protos = the learned centres,
      has = the players with a centre [members, 5, C], the same query vectors) for m164_ident.py.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402
import m200_prod as m200  # noqa: E402

base = t.base
t.OUT, t.CK, t.TMP = ROOT / "results/m200", Path("/work/u2507115/m200_ckpt"), ROOT / "results/m200_tta"
base.D, base.FFN = 512, 2048


def one(v, seed, G, Tt, log):
    """m124_tta_queries.one with the residual input (the same steps, outputs and report)."""
    import torch
    dev = "cuda"; tag = f"{v}{f'_s{seed}' if seed else ''}"; D_OUT = base.D_OUT; SYM = t.sym_table()
    st = torch.load(t.CK / f"ckpt_{tag}.pt", map_location=dev, weights_only=False); assert "opt" not in st, f"{tag}: the network is still training"
    net = m200.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval()
    q = np.load(base.CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); gidx = {k: i for i, k in enumerate(G["key"])}; j = base.NETS.index(v); mine = np.flatnonzero(q["net"] == j)
    tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731

    def embed(src, rows, subj, k):
        out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(SYM[k][src["POS"][r].astype(np.int64)]), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64)), tb(src["CR"][r])).float().cpu().numpy()
        return out
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    te = mine[q["qtest"][mine] >= 0]; rows_t = q["qtest"][te].astype(np.int64); subj_t = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows_t], Tt["SUBJ"][rows_t]) if len(te) else np.zeros(0, np.int64)
    loc = {g_: i for i, g_ in enumerate(mine)}; pos_tr = np.array([loc[g_] for g_ in tr[ok]], dtype=np.int64); pos_te = np.array([loc[g_] for g_ in te], dtype=np.int64)
    acc = np.zeros((len(mine), D_OUT), np.float64); first = None; cos = []
    for k in range(8):
        E = np.zeros((len(mine), D_OUT), np.float32)
        if len(rows):
            E[pos_tr] = embed(G, rows, subj, k)
        if len(te):
            E[pos_te] = embed(Tt, rows_t, subj_t, k)
        if k == 0:
            first = E.copy()
        else:
            cos.append(float((E * first).sum(1)[np.abs(first).sum(1) > 0].mean()))
        acc += E
        log(f"  net {tag}: orientation {k} done")
    qv = (acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    z = np.load(t.OUT / f"out_{tag}.npz"); assert np.array_equal(z["qidx"], mine); dmax = float(np.abs(z["qvec"] - first).max())
    rep = {"queries": int(len(mine)), "identity_vs_stored_max_abs_diff": dmax, "cos_between_orientations": cos, "mean_cos": float(np.mean(cos))}
    held = (mine >= 2 * Q) & (mine < 2 * Q + H)
    if held.any():  # held-out own games of this network's queries (network -1 holds them all)
        truth = q["held_truth"][mine[held] - 2 * Q]; P = z["protos"]
        for nm, X in (("plain", first), ("averaged", qv)):
            S = X[held] @ P.T; r = (S > S[np.arange(len(truth)), truth][:, None]).sum(1); rep[f"heldout_top1_{nm}"] = float((r == 0).mean()); rep[f"heldout_top20_{nm}"] = float((r < 20).mean())
    t.TMP.mkdir(parents=True, exist_ok=True); np.savez(t.TMP / f"tta_{tag}.npz", qidx=mine, qvec=qv); (t.TMP / f"tta_{tag}.json").write_text(json.dumps(rep, indent=1))
    log(f"net {tag}: {len(mine)} queries; identity orientation vs stored vectors max |diff| {dmax:.4f}; mean cosine between orientations {rep['mean_cos']:.4f}" + (f"; held-out top-1 plain {rep['heldout_top1_plain']:.3f} -> averaged {rep['heldout_top1_averaged']:.3f}" if held.any() else ""))


_load_games = base.load_games


def load_games(smoke, log):
    pool = mp.get_context("fork").Pool(12)  # before anything large exists (strict overcommit on the nodes)
    G, Tt = _load_games(smoke, log); m200.attach_choice(G, Tt, pool, log); pool.close(); pool.join()
    return G, Tt


t.one = one; base.load_games = load_games

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="30"); ap.add_argument("--name", default="S9t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
