"""Band specialists (2026-11-05, user: 為什麼不對 3D/4D 專門訓一個更細的模型). Protocol of hd_check / r_variants (folds 1-4, 32 training
questions per player, 64-dim fused_opp_diff_hd, MLP h32, seeds averaged), validation = pilot questions.
  flat        the 10-class MLP (the baseline)
  oracle      flat's posterior restricted to the true band (an upper bound for any band-first scheme with a perfect first stage)
  specialist  an MLP trained only on the band's players (4 classes A-3D; 2 classes 3D-4D), evaluated with the band given (vs oracle)
  two-stage   realistic: the band from flat's posterior mass (argmax over bands), then the specialist inside that band; the official
              score on ALL validation questions against flat (paired player bootstrap).
Bands: kyu {D,C,B}, mid {A,1D,2D,3D}, high {4D,5D,6D}; also the 3D-4D pair. Pre-registered: two-stage vs flat >= +0.003 with the CI lower
bound > 0 would be interesting; expected ~0 or negative (the specialist sees the same evidence; the band must come from the same moves).
  python -I logs/nano4/band_specialist.py --out results/r_variants/band_specialist.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
BANDS = {"kyu": [0, 1, 2], "mid": [3, 4, 5, 6], "high": [7, 8, 9]}
PAIR = [6, 7]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True); a = ap.parse_args(); t0 = time.time()
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str}); hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    data = {}
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], "players": players, "val": l1b_cv.pilot_questions(players)}
        print(f"fold {f}: {len(players)} players ({time.time() - t0:.0f} s)", flush=True)
    S = {k: [] for k in ("flat", "two_stage")}; ys, pids, ns = [], [], []
    rep = {"folds": a.folds, "band_given": {}, "pair_given": {}}
    acc = {"oracle_mid": [], "spec_mid": [], "oracle_pair": [], "spec_pair": [], "flat_mid_exact": [], "band_acc": []}
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v); train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]; X_va, y_va, n_va, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        Xt, Xv = X_tr["fused_opp_diff_hd"], X_va["fused_opp_diff_hd"]
        P_flat = np.mean([MLPModel(hidden=32, seed=s).fit(Xt, y_tr).predict_proba(Xv) for s in a.seeds], axis=0)
        # specialists per band (classes relabelled 0..k-1 inside the band)
        P_spec = {}
        for name, cls in list(BANDS.items()) + [("pair", PAIR)]:
            m = np.isin(y_tr, cls); remap = {c: i for i, c in enumerate(cls)}
            yt = np.array([remap[c] for c in y_tr[m]])
            Pb = np.mean([MLPModel(hidden=32, seed=s).fit(Xt[m], yt).predict_proba(Xv) for s in a.seeds], axis=0)
            full = np.zeros((len(Xv), len(LEVELS))); full[:, cls] = Pb[:, :len(cls)] if Pb.shape[1] >= len(cls) else Pb
            P_spec[name] = full
        # oracle (true band given) vs specialist (true band given) on the mid band and the 3D-4D pair
        for name, cls, key in (("mid", BANDS["mid"], "mid"), ("pair", PAIR, "pair")):
            sub = np.isin(y_va, cls)
            Po = P_flat[sub][:, cls]; pick_o = np.array(cls)[Po.argmax(1)]
            pick_s = np.array(cls)[P_spec[name][sub][:, cls].argmax(1)]
            acc[f"oracle_{key}"].append(float((pick_o == y_va[sub]).mean())); acc[f"spec_{key}"].append(float((pick_s == y_va[sub]).mean()))
            if key == "mid":
                flat_dec = np.array([expected_score_decision(p) for p in P_flat[sub]]); acc["flat_mid_exact"].append(float((flat_dec == y_va[sub]).mean()))
        # realistic two-stage: band by flat's posterior mass, then the specialist of that band (its posterior inside the band, ES decision)
        band_mass = np.column_stack([P_flat[:, cls].sum(1) for cls in BANDS.values()]); band_pick = band_mass.argmax(1)
        true_band = np.array([next(i for i, cls in enumerate(BANDS.values()) if c in cls) for c in y_va]); acc["band_acc"].append(float((band_pick == true_band).mean()))
        P_two = np.zeros_like(P_flat)
        for i, (name, cls) in enumerate(BANDS.items()):
            rows = band_pick == i; P_two[rows] = P_spec[name][rows]
        P_two = P_two / P_two.sum(1, keepdims=True)
        _, s_flat = evaluate(P_flat, y_va, d["val"]); _, s_two = evaluate(P_two, y_va, d["val"])
        S["flat"].append(s_flat); S["two_stage"].append(s_two); ys.append(y_va); pids.append(d["val"]["player_id"].to_numpy(dtype=str)); ns.append(n_va)
        print(f"fold {v}: flat {balanced_mean_over_n(s_flat, y_va, n_va):.4f}  two-stage {balanced_mean_over_n(s_two, y_va, n_va):.4f}  band acc {acc['band_acc'][-1]:.3f}  "
              f"mid: oracle {acc['oracle_mid'][-1]:.3f} specialist {acc['spec_mid'][-1]:.3f} flat exact {acc['flat_mid_exact'][-1]:.3f}  pair 3D-4D: oracle {acc['oracle_pair'][-1]:.3f} specialist {acc['spec_pair'][-1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    y, pid, n = map(np.concatenate, (ys, pids, ns)); Sf, St = np.concatenate(S["flat"]), np.concatenate(S["two_stage"])
    dd = paired_diff(St, Sf, y, pid, n)
    rep.update({"flat": balanced_mean_over_n(Sf, y, n), "two_stage": balanced_mean_over_n(St, y, n), "two_stage_vs_flat": dd, "acc_means": {k: float(np.mean(v)) for k, v in acc.items()}})
    print(f"flat {rep['flat']:.4f}  two-stage {rep['two_stage']:.4f}  diff {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]")
    print("means: " + "  ".join(f"{k} {v:.3f}" for k, v in rep["acc_means"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
