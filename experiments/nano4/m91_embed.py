"""Section 4M M91 (2026-11-13): identity vectors of BOTH sides of every training game and every test game under the M90 sequence
identity networks (the state kept by training), for the mirror channel and other uses of "who is on the other side of a training
row". One file per network and seed: results/m91/emb_{tag}.npz with own / opp [N, 128] float16 (training games in the loader's
order: own = the row's owner as the subject, opp = the other colour as the subject), t_own / t_opp (test games by task number);
results/m91/rows_{net}.npz keeps the game keys, owners, folds of that order.
  python -I logs/nano4/m91_embed.py --net 1 [--seeds 0,1,2] [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m88_seqid as m  # noqa: E402

OUT = ROOT / "results/m91"


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--net", type=int, required=True); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--batch", type=int, default=2048); a = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    m.FULL = True; OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; log(f"net {a.net}: GPU {torch.cuda.get_device_name(0)}")
    G, Tt = m.load_games(a.smoke, log); n = len(G["key"]); sfx = "_smoke" if a.smoke else ""
    np.savez(OUT / f"rows_{a.net}{sfx}.npz", key=G["key"], pid=G["pid"], fold=G["fold"], subj=G["SUBJ"], length=G["LEN"])
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def embed(net, src, subj):
        out = np.zeros((len(subj), m.D_OUT), np.float16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(subj), a.batch):
                s = slice(i, i + a.batch); out[s] = net(tb(src["GL"][s]), tb(src["POS"][s].astype(np.int64)), tb(src["MOV"][s].astype(np.int64)), tb(src["LEN"][s].astype(np.int64)), tb(subj[s].astype(np.int64))).float().cpu().numpy()
        return out
    for seed in [int(x) for x in a.seeds.split(",")]:
        tag = f"{a.net}{f'_s{seed}' if seed else ''}"; f = OUT / f"emb_{tag}{sfx}.npz"
        if f.exists():
            log(f"{f.name} exists, skipped"); continue
        st = torch.load(ROOT / f"results/m90/ckpt_{tag}.pt", map_location="cpu", weights_only=False); net = m.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); log(f"{tag}: state of step {st['best']['step']} (held-out top-1 {st['best']['top1']:.3f})"); del st
        own = embed(net, G, G["SUBJ"]); log(f"{tag}: own side of {n} training games"); opp = embed(net, G, 1 - G["SUBJ"]); log(f"{tag}: opponent side")
        t_own = embed(net, Tt, Tt["SUBJ"]); t_opp = embed(net, Tt, 1 - Tt["SUBJ"]); tmp = f.with_name(f.name + ".tmp.npz"); np.savez(tmp, own=own, opp=opp, t_own=t_own, t_opp=t_opp); tmp.replace(f)
        log(f"{tag}: written ({f.stat().st_size / 1e6:.0f} MB); norms own {float(np.linalg.norm(own[:1000].astype(np.float32), axis=1).mean()):.3f}, opp {float(np.linalg.norm(opp[:1000].astype(np.float32), axis=1).mean()):.3f}"); del net, own, opp
    log("done")


if __name__ == "__main__":
    main()
