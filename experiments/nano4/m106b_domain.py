"""Section 4M M106b (2026-11-13, report only, CPU): the domain check of M106 with the two known confounds removed. Only games
without handicap and with at least 80 plies on both sides (the external pilot's selection rules); player-level features from those
games (players with >= 10 of them); and same-level comparisons: external 6d vs competition 6D, external 5k vs competition D, next to
competition 6D vs 5D and D vs C under the same restriction, and to a split of ONE competition level into two random halves (0.5 by
construction; shows the estimator's floor). Also where the classifier's signal sits: the AUC with only the target player's mean,
only the sd, only the opponent's mean.
  python -I logs/nano4/m106b_domain.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m106_scale_pilot as p  # noqa: E402


def main():
    import torch
    torch.set_num_threads(12); files = [f for v in (1, 2, 3, 4) for f in sorted((ROOT / f"results/trunk_vec/fold{v}").glob("games_s*_p*.csv"))]; fx = sorted(f for f in (p.EXT / "ply").glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz"))
    with mp.get_context("fork").Pool(12) as pool:
        rc = pool.map(p.comp_part, files, chunksize=4); rx = pool.map(p.ext_part, fx, chunksize=4)
    key = np.concatenate([r[0] for r in rc]); pid = np.concatenate([r[1] for r in rc]); lab = np.array([p.RANK[x] for x in np.concatenate([r[2] for r in rc])]); F = np.concatenate([r[3] for r in rc]); npl = np.concatenate([pd.read_csv(f, usecols=["n_plies"]).n_plies.to_numpy() for f in files])
    d82 = np.load(ROOT / "results/r_variants/m82_val.npz"); hd = pd.Series(d82["hd"], index=d82["key"].astype(str)); hd = hd[~hd.index.duplicated()].reindex(key).to_numpy(); print(f"competition games {len(key)}; handicap known for {int(np.isfinite(hd).sum())}; without handicap {int((hd == 0).sum())}; of those with >= 80 plies {int(((hd == 0) & (npl >= 80)).sum())}")
    kx = np.concatenate([r[0] for r in rx]); Fx = np.concatenate([r[1] for r in rx]); sel = pd.read_csv(p.EXT / "select.csv", usecols=["game_key", "account"], dtype=str).set_index("game_key")["account"]; acc = sel.reindex(kx).to_numpy().astype(str); grp = np.array([k.split(":")[1] for k in kx])
    ok = (hd == 0) & (npl >= 80); mu, sd = F[ok].mean(0), F[ok].std(0) + 1e-6; F = (F - mu) / sd; Fx = (Fx - mu) / sd
    g = pd.DataFrame(F[ok]).groupby(pid[ok]); cnt = g.size(); PL = g.mean()[cnt >= 10].to_numpy(); pl = pd.Series(lab[ok]).groupby(pid[ok]).first()[cnt >= 10].to_numpy(); A = pd.DataFrame(Fx).groupby(acc).mean(); ag = pd.Series(grp).groupby(acc).first().reindex(A.index).to_numpy(); A = A.to_numpy()
    print(f"competition players with >= 10 such games: {len(pl)} (by level: " + " ".join(f"{p.LEVELS[c]} {int((pl == c).sum())}" for c in range(10)) + ")")
    rng = np.random.default_rng(0); blocks = {"all 384": slice(0, 384), "own mean": slice(0, 128), "own sd": slice(128, 256), "opponent mean": slice(256, 384)}
    for name, X1, X2 in (("external 6d vs competition 6D", A[ag == "6d"], PL[pl == 9]), ("external 6d vs competition 5D", A[ag == "6d"], PL[pl == 8]), ("competition 6D vs competition 5D", PL[pl == 9], PL[pl == 8]), ("competition 6D vs competition 4D", PL[pl == 9], PL[pl == 7]),
                         ("external 5k vs competition D", A[ag == "5k"], PL[pl == 0]), ("external 5k vs competition C", A[ag == "5k"], PL[pl == 1]), ("competition D vs competition C", PL[pl == 0], PL[pl == 1]), ("competition D vs competition B", PL[pl == 0], PL[pl == 2]),
                         ("competition 6D, two random halves", *np.array_split(PL[pl == 9][rng.permutation(int((pl == 9).sum()))], 2))):
        if len(X1) > 1500:
            X1 = X1[rng.choice(len(X1), 1500, replace=False)]
        print(f"  {name:36s} ({len(X1)} vs {len(X2)}): AUC " + "  ".join(f"{k} {p.domain_auc(X1[:, s], X2[:, s]):.3f}" for k, s in blocks.items()), flush=True)


if __name__ == "__main__":
    main()
