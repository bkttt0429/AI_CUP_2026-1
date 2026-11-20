#!/usr/bin/env python
"""M255 (2026-11-20): orientation-averaged S7t query vectors (m161q networks -1, seeds 30 and 31; m124_tta_queries.py's recipe)
of the OPPONENT side and the TARGET side of every game in results/m255/base/vec (design doc M255).

-> results/m255/s7t_new.npz: qv_opp, qv_tgt [2, games, 128] (task order), task, qid. Check: the control games (original test
games) against the cached var_S7t.npz query vectors.

  python -I logs/nano4/m255_vec.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as m  # noqa: E402
from m112_idaug import sym_table  # noqa: E402

m.D, m.FFN = 512, 2048; CK = Path("/work/u2507115/m161q_ckpt"); OUT = ROOT / "results/m255"; SEEDS = (30, 31)


def main():
    import torch
    dev = "cuda"; tf = sorted((OUT / "base/vec").glob("games_s*_p*.csv")); rt = [m.read_part(f) for f in tf]; cat = lambda j: np.concatenate([x[j] for x in rt])  # noqa: E731
    o = np.argsort(cat(0)); assert np.array_equal(cat(0)[o], np.arange(len(o))); qid = cat(1)[o].astype(str); Tt = dict(GL=cat(2)[o], POS=cat(3)[o], MOV=cat(4)[o], LEN=cat(5)[o], SUBJ=cat(6)[o]); N = len(qid); SYM = sym_table(); tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    print(f"{N} games of {len(set(qid))} questions; plies median {int(np.median(Tt['LEN']))}", flush=True); qv = {"opp": np.zeros((len(SEEDS), N, m.D_OUT), np.float32), "tgt": np.zeros((len(SEEDS), N, m.D_OUT), np.float32)}; cosr = []
    for si, s in enumerate(SEEDS):
        st = torch.load(CK / f"ckpt_-1_s{s}.pt", map_location=dev, weights_only=False); assert "opt" not in st; net = m.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); del st
        for side, subj in (("opp", 1 - Tt["SUBJ"]), ("tgt", Tt["SUBJ"])):
            acc = np.zeros((N, m.D_OUT), np.float64); first = None
            for k in range(8):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    E = net(tb(Tt["GL"]), tb(SYM[k][Tt["POS"].astype(np.int64)]), tb(Tt["MOV"].astype(np.int64)), tb(Tt["LEN"].astype(np.int64)), tb(subj.astype(np.int64))).float().cpu().numpy()
                acc += E
                if k == 0:
                    first = E
                else:
                    cosr.append(float((E * first).sum(1).mean()))
            qv[side][si] = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
        print(f"network -1_s{s}: done; mean cosine between orientations {np.mean(cosr):.4f}", flush=True)
    np.savez(OUT / "s7t_new.npz", qv_opp=qv["opp"], qv_tgt=qv["tgt"], task=np.arange(N), qid=qid)
    # ---------------------------------------------------------------- check against the cache
    q = np.load(ROOT / "results/m84_cache/queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); tq = np.load(ROOT / "results/r_variants/m82_test.npz")["qid"].astype(str); cm = pd.read_csv(OUT / "inputs/ctl_map.csv", dtype=str); orig = dict(zip(cm.ctl_question, cm.test_question))
    V = np.load(ROOT / "results/m84_cache/var_S7t.npz", mmap_mode="r")["qvec"]; seen = {}; cs = {"opp": [], "tgt": []}
    for i, x in enumerate(qid):
        if x not in orig:
            continue
        j = seen.get(x, 0); seen[x] = j + 1; task = int(np.flatnonzero(tq == orig[x])[j])
        for side, off in (("opp", 2 * Q + H), ("tgt", 2 * Q + H + T)):
            for si in range(len(SEEDS)):
                old = np.asarray(V[si, off + task], np.float32); cs[side].append(float(old @ qv[side][si, i] / max(np.linalg.norm(old), 1e-9)))
    rep = {side: {"n": len(v), "min_cos": float(np.min(v)), "mean_cos": float(np.mean(v))} for side, v in cs.items()}; (OUT / "s7t_new.json").write_text(json.dumps(rep, indent=1))
    print("CHECK against the cached S7t query vectors (control games): " + "; ".join(f"{side}: min cosine {r['min_cos']:.4f}, mean {r['mean_cos']:.4f} ({r['n']})" for side, r in rep.items()) + f" -> {'PASS' if min(r['min_cos'] for r in rep.values()) >= 0.995 else 'FAIL'}"); print("written results/m255/s7t_new.npz")


if __name__ == "__main__":
    main()
