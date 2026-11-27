"""Section 4M M61 (2026-11-08): is the test population low-activity players? In validation, players with <= 25 training games have HD1 share
0.245 (test 0.228, all training 0.171) and an A|1D first-band precision of 0.70 (all 0.41). If the test is drawn from such players, CV restricted
to them should reproduce the decoded public probe outcomes -- and then predict the untested boundaries. Report only.
For each push i -> j at r = 2 (m53 push rule: only i -> j moves), on validation questions (ct9, test-like: no HD >= 2 game) of players with
<= G games (G = 25, 40) and for all players: flips, precision (truth = j), share truth = i; under the natural composition of the subset and
class-balanced (each true class weighted equally). Compared with the public decoded precision of the probes already uploaded.
  python -I logs/nano4/m61_lowact_curves.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS  # noqa: E402
from m53_boundary_curves import push  # noqa: E402

R = {lv: i for i, lv in enumerate(LEVELS)}
# public decoded precision of uploaded single pushes (truth = target share among flipped; ledger MAP / exact counts)
OBSERVED = {"1D->A": 8 / 8, "B->A": 5.8 / 13, "5D->6D": 1 / 7, "2D->1D": 5 / 12, "3D->2D": 0 / 8, "1D->2D": 6 / 15, "A->1D": 2 / 9}


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str, "player_id": str}, keep_default_na=False, na_values={"hd": [""]})
    hd["hd"] = pd.to_numeric(hd["hd"], errors="coerce").fillna(0)
    games = hd.groupby("player_id").size(); hdg = hd.drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    ys, Ps, gs = [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        keep = ~(np.array([max(hdg.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)
        ys.append(y[keep]); Ps.append(P[keep]); gs.append(pd.Series(pid).map(games).to_numpy()[keep])
    y, P, g = np.concatenate(ys), np.concatenate(Ps), np.concatenate(gs)
    rep = {}
    pushes = [(a, b) for k in range(9) for a, b in ((k, k + 1), (k + 1, k))]
    subsets = {"all": np.ones(len(y), bool), "games<=40": g <= 40, "games<=25": g <= 25}
    print("precision of the r = 2 push (truth = target among flipped): natural composition | class-balanced; observed public where uploaded")
    for i, j in pushes:
        name = f"{LEVELS[i]}->{LEVELS[j]}"; row = {}
        for sname, m in subsets.items():
            _, mv = push(P[m], i, j, 2.0); yy = y[m][mv]
            if len(yy) == 0:
                row[sname] = (0, float("nan"), float("nan")); continue
            nat = float((yy == j).mean())
            cnt = np.bincount(y[m], minlength=10).astype(float); w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)[yy]
            bal = float(w[yy == j].sum() / w.sum())
            row[sname] = (int(mv.sum()), nat, bal)
        rep[name] = row
        obs = OBSERVED.get(name)
        print(f"{name:7s} " + "  ".join(f"{s}: n {row[s][0]:4d} nat {row[s][1]:.2f} bal {row[s][2]:.2f}" for s in subsets)
              + (f"   | public {obs:.2f}" if obs is not None else ""), flush=True)
    for sname in subsets:
        xs = [rep[k][sname][1] for k in OBSERVED]; xb = [rep[k][sname][2] for k in OBSERVED]; ob = list(OBSERVED.values())
        print(f"{sname}: corr with public precision over {len(ob)} uploaded pushes: natural {np.corrcoef(xs, ob)[0, 1]:+.2f}, balanced {np.corrcoef(xb, ob)[0, 1]:+.2f}; "
              f"mean abs error natural {np.mean(np.abs(np.array(xs) - ob)):.3f}, balanced {np.mean(np.abs(np.array(xb) - ob)):.3f}")
    Path(ROOT / "results/trunk_seq/es/m61_lowact_curves.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
