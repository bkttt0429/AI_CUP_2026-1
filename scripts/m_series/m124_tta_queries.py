#!/usr/bin/env python
"""Section 4M M124 (2026-11-14): test-time symmetry averaging of the QUERY vectors of the symmetry-augmented identity networks.

M120's preview: the anchors follow the share of opponents recognised with a clear margin, and that share reacts to averaging
(several seeds) much more than to one network's held-out accuracy. A network trained with a random board symmetry per episode is
roughly, not exactly, invariant to the orientation of a game; the mean of its vectors over the eight orientations of the query
game is a free average of eight correlated readings. Here every finished network of results/m120 re-embeds its queries (the
opponent and target sides of its validation games, the held-out games, the test games) under the eight symmetries of the move
coordinates (trunk vectors unchanged, as in training), the unit vectors are averaged and renormalised; prototypes unchanged.
Output: results/m84_cache/var_<name>.npz in m84_variant.py's format (protos from results/m120, the averaged query vectors), and a
report (the identity orientation must reproduce the stored vectors; cosine between orientations; held-out top-1 of the plain and
the averaged queries per network).

  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch python -I scripts/m_series/m124_tta_queries.py --seeds 0 --name S4t
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m88_seqid as base  # noqa: E402
from m112_idaug import sym_table  # noqa: E402

OUT, CK, TMP = ROOT / "results/m120", Path("/work/u2507115/m120_ckpt"), ROOT / "results/m124_parts"


def one(v, seed, G, Tt, log):
    import torch
    dev = "cuda"; tag = f"{v}{f'_s{seed}' if seed else ''}"; D_OUT = base.D_OUT; SYM = sym_table()
    st = torch.load(CK / f"ckpt_{tag}.pt", map_location=dev, weights_only=False); assert "opt" not in st, f"{tag}: the network is still training"
    net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval()
    q = np.load(base.CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); gidx = {k: i for i, k in enumerate(G["key"])}; j = base.NETS.index(v); mine = np.flatnonzero(q["net"] == j)
    tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731

    def embed(src, rows, subj, k):
        out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(SYM[k][src["POS"][r].astype(np.int64)]), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
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
    z = np.load(OUT / f"out_{tag}.npz"); assert np.array_equal(z["qidx"], mine); dmax = float(np.abs(z["qvec"] - first).max())
    rep = {"queries": int(len(mine)), "identity_vs_stored_max_abs_diff": dmax, "cos_between_orientations": cos, "mean_cos": float(np.mean(cos))}
    held = (mine >= 2 * Q) & (mine < 2 * Q + H)
    if held.any():  # held-out own games of this network's queries (network -1 holds them all)
        truth = q["held_truth"][mine[held] - 2 * Q]; P = z["protos"]
        for nm, X in (("plain", first), ("averaged", qv)):
            S = X[held] @ P.T; r = (S > S[np.arange(len(truth)), truth][:, None]).sum(1); rep[f"heldout_top1_{nm}"] = float((r == 0).mean()); rep[f"heldout_top20_{nm}"] = float((r < 20).mean())
    TMP.mkdir(parents=True, exist_ok=True); np.savez(TMP / f"tta_{tag}.npz", qidx=mine, qvec=qv); (TMP / f"tta_{tag}.json").write_text(json.dumps(rep, indent=1))
    log(f"net {tag}: {len(mine)} queries; identity orientation vs stored vectors max |diff| {dmax:.4f}; mean cosine between orientations {rep['mean_cos']:.4f}" + (f"; held-out top-1 plain {rep['heldout_top1_plain']:.3f} -> averaged {rep['heldout_top1_averaged']:.3f}" if held.any() else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--name", default="S4t")
    ap.add_argument("--merge-only", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    seeds = [int(x) for x in a_.seeds.split(",")]; jobs = [(v, s) for s in seeds for v in base.NETS]; base.FULL, base.EXT = True, None
    if not a_.merge_only:
        gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
        G, Tt = base.load_games(False, log); kids = {}; queue = list(jobs); free = list(gpus); bad = 0
        while queue or kids:
            while queue and free:
                v, s = queue.pop(0)
                if (TMP / f"tta_{v}{f'_s{s}' if s else ''}.npz").exists():  # done by an earlier run
                    continue
                gpu = free.pop(0); pid = os.fork()
                if pid == 0:
                    code = 0
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu); t1 = time.time()
                        one(v, s, G, Tt, lambda m: print(f"{m} ({time.time() - t1:.0f} s)", flush=True))
                    except BaseException:  # noqa: BLE001
                        import traceback
                        traceback.print_exc(); code = 1
                    finally:
                        sys.stdout.flush(); os._exit(code)
                kids[pid] = gpu
            if not kids:
                continue
            pid, status = os.wait(); free.append(kids.pop(pid)); bad += status != 0
        if bad:
            sys.exit(f"{bad} networks failed")
    q = np.load(base.CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, base.D_OUT), np.float32); qvec = np.zeros((len(seeds), N, base.D_OUT), np.float16); rep = {}
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            tag = f"{v}{f'_s{s}' if s else ''}"; z = np.load(OUT / f"out_{tag}.npz"); t = np.load(TMP / f"tta_{tag}.npz"); assert np.array_equal(z["qidx"], t["qidx"]); protos[si, j] = z["protos"]; qvec[si, t["qidx"]] = t["qvec"]; rep[tag] = json.loads((TMP / f"tta_{tag}.json").read_text())
    np.savez(base.CACHE / f"var_{a_.name}.npz", protos=protos, qvec=qvec); (base.CACHE / f"var_{a_.name}.json").write_text(json.dumps(rep, indent=1))
    log(f"var_{a_.name} written ({len(seeds)} seed(s))")
    for tag, r in rep.items():
        print(f"  net {tag}: mean cosine between orientations {r['mean_cos']:.4f}" + (f"; held-out top-1 {r['heldout_top1_plain']:.3f} -> {r['heldout_top1_averaged']:.3f}, top-20 {r['heldout_top20_plain']:.3f} -> {r['heldout_top20_averaged']:.3f}" if "heldout_top1_plain" in r else ""))


if __name__ == "__main__":
    main()
