"""Section 4M M95 (2026-11-13; user: "我判斷規則允許"): identity vectors of the competition-1 (player identification) games under the
M90 sequence identity networks, to check whether rank-test players are among competition 1's candidates. Read-only use of
(a) results/b1_public/vec: the old Public pool, 400 candidates x 100 games and its 400 questions (per-ply vectors made for the M62
support run), (b) /work/u2507115/pid_v2/ply_official: the added pools pub2 / priv2, 600 candidates x 10 games each and their
questions (per-ply vectors of the player-identification session, same layout). Output results/m95/xemb_{tag}.npz: own [G, 128]
float16 (the player's side) per network and seed; results/m95/games.npz: key, group ("oldpub:c:<player>", "oldpub:q:<question>",
"pub2:c:<player>", "pub2:q:<question>", "priv2:..."), length.
  python -I logs/nano4/m95_xembed.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m88_seqid as m  # noqa: E402

OUT, B1, PLY, T1 = ROOT / "results/m95", ROOT / "results/b1_public", Path("/work/u2507115/pid_v2/ply_official"), Path.home() / "BK_AICUP_2026-2/data/tests_1111"


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; files = sorted((B1 / "vec").glob("games_s*_p*.csv"))
    with mp.get_context("fork").Pool(12) as pool:
        parts = pool.map(m.read_part, files, chunksize=4)
    cmap = pd.read_csv(B1 / "inputs/cands_map.csv", dtype=str).set_index("question_id")["player_id"].to_dict(); key, grp, GL, POS, MOV, LEN, SUBJ = [], [], [], [], [], [], []
    for task, qid, gl, pos, mov, ln, sj in parts:
        key += [f"oldpub:{int(t)}" for t in task]; grp += [f"oldpub:c:{cmap[q]}" if q in cmap else f"oldpub:q:{q}" for q in qid]; GL.append(gl); POS.append(pos); MOV.append(mov); LEN.append(ln); SUBJ.append(sj)
    del parts; log(f"old Public pool: {len(key)} games")
    pl = {tag: pd.read_csv(T1 / f"player_identification_candidates2_{tag}.csv", usecols=["player_id"], dtype=str)["player_id"].to_numpy() for tag in ("public", "private")}
    for f in sorted(PLY.glob("ply_s*_p*.npz")):
        z = np.load(f); GL.append(z["GL"]); POS.append(z["POS"]); MOV.append(z["MOV"]); LEN.append(z["LEN"]); SUBJ.append(z["SUBJ"])
        for k in z["key"].astype(str):
            a = k.split(":"); key.append("new:" + k)
            if a[0] == "c":
                pool_ = "pub2" if a[2].startswith("pub2") else "priv2"; grp.append(f"{pool_}:c:{pl['public' if pool_ == 'pub2' else 'private'][int(a[1])]}")
            else:
                grp.append(f"{'pub2' if a[1].startswith('pub2') else 'priv2'}:q:{a[1]}")
    src = dict(GL=np.concatenate(GL), POS=np.concatenate(POS), MOV=np.concatenate(MOV), LEN=np.concatenate(LEN), SUBJ=np.concatenate(SUBJ)); del GL; G = len(key); grp = np.array(grp); log(f"all pools: {G} games, {len(set(grp))} groups; " + ", ".join(f"{p}: {len(set(x for x in grp if x.startswith(p + ':c:')))} candidates / {len(set(x for x in grp if x.startswith(p + ':q:')))} questions" for p in ("oldpub", "pub2", "priv2")))
    np.savez(OUT / "games.npz", key=np.array(key), group=grp, length=src["LEN"], subj=src["SUBJ"]); tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    for seed in (0, 1, 2):
        for v in m.NETS:
            tag = f"{v}{f'_s{seed}' if seed else ''}"; st = torch.load(ROOT / f"results/m90/ckpt_{tag}.pt", map_location="cpu", weights_only=False); net = m.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval(); del st; out = np.zeros((G, m.D_OUT), np.float16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, G, 2048):
                    s = slice(i, i + 2048); out[s] = net(tb(src["GL"][s]), tb(src["POS"][s].astype(np.int64)), tb(src["MOV"][s].astype(np.int64)), tb(src["LEN"][s].astype(np.int64)), tb(src["SUBJ"][s].astype(np.int64))).float().cpu().numpy()
            np.savez(OUT / f"xemb_{tag}.npz", own=out); log(f"{tag}: written"); del net
    log("done")


if __name__ == "__main__":
    main()
