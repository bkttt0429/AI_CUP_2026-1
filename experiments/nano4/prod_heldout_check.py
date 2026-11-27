"""ct3 vs ct6 production ensembles on the inner held-out players (2026-11-05). Every production model (val_fold all, with fold 0,
x40) held out the same 10% of training players (RandomState(12345 + 5)) and scored its stopping point on the same 4000 of their
questions (RandomState(5)); none of the 24 models trained on them. So these questions are a labelled, production-level comparison:
per-set ensemble official score (class-balanced over n), paired player bootstrap, E - y by class (the +0.05 offset seen on the test),
decision agreement.  GPU, ~30 min (tokens of folds 0-4 + x40 twice: plain and opponent-HSL).
  python -I logs/nano4/prod_heldout_check.py --out results/r_variants/prod_heldout.json
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def inner_questions(cache):
    a = argparse.Namespace(cache=str(cache), with_fold0=True, extra_games=40, mix=0.0, train_games=0, train_sizes=None, train_folds=0, l1b_set=3)
    X_tr, y_tr, _, _, q_tr, _ = ts.r_features(a, "all")
    pid_tr = q_tr["player_id"].to_numpy(dtype=str)
    players = np.array(sorted(set(pid_tr)))
    held = set(np.random.RandomState(12345 + ts.fold_code("all")).permutation(players)[:int(0.1 * len(players))])
    is_in = np.array([x in held for x in pid_tr])
    pick = np.random.RandomState(ts.fold_code("all")).permutation(np.flatnonzero(is_in))[:4000]
    q = q_tr.iloc[pick].reset_index(drop=True)
    return X_tr[pick], y_tr[pick], q["game_keys"].tolist(), q, len(held)


def predict_set(paths, X_raw, offsets, R_raw, keys, device):
    out = {}
    for p in paths:
        model, stats, (r_mean, r_std), side = ts.load_checkpoint(p, device)
        Xd = torch.as_tensor(ts.standardise_tokens(X_raw.copy(), stats), device=device)
        R = ((R_raw - r_mean) / r_std).astype(np.float32)
        out[Path(p).stem] = ts.predict(model, R, keys, Xd, offsets, device)
        del Xd, model
        torch.cuda.empty_cache()
        print(f"  {Path(p).stem}: done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq")); ap.add_argument("--out", required=True); a = ap.parse_args()
    t0 = time.time(); device = "cuda"
    R_in, y, keys, q, n_held = inner_questions(Path(a.cache))
    pid, n = q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy()
    print(f"{len(y)} inner questions of {n_held} held-out players ({len(set(pid))} present); n sizes {dict(zip(*np.unique(n, return_counts=True)))} ({time.time() - t0:.0f} s)", flush=True)
    folds = [0, 1, 2, 3, 4]
    sets = {"ct3": ([str(ROOT / f"results/models/ct/all_{arm}-f0-x40_s{s}.pt") for arm in ("C", "T") for s in range(6)], False),
            "ct6": ([str(ROOT / f"results/models/ct/all_{arm}-f0-x40-oh_s{s}.pt") for arm in ("C", "T") for s in range(6)], True)}
    P_all, rep = {}, {"questions": int(len(y)), "held_players": n_held}
    for name, (paths, oh) in sets.items():
        parts = [(ts.VEC / f"fold{f}", ts.scores_path(a.cache, f, ts.VEC, 3, oh)) for f in folds] + [(ts.EXTRAS[40]["vec"] / f"fold{f}", ts.scores_path(a.cache, f, ts.EXTRAS[40]["vec"], 3, oh)) for f in folds]
        X_raw, offsets = ts.raw_tokens(parts)
        print(f"{name}: tokens {X_raw.shape} loaded ({time.time() - t0:.0f} s)", flush=True)
        P_all[name] = predict_set(paths, X_raw, offsets, R_in, keys, device)
        del X_raw; gc.collect()
    S, E = {}, {}
    for name in sets:
        Ps = P_all[name]
        P = np.mean(list(Ps.values()), axis=0)
        res, sc = evaluate(P, y, q)
        S[name], E[name] = sc, P @ np.arange(len(LEVELS))
        per_model = {k: float(evaluate(v, y, q)[0]["balanced_mean_over_n"]) for k, v in Ps.items()}
        rep[name] = {"ensemble": float(res["balanced_mean_over_n"]), "by_n": {str(m): balanced_mean_over_n(sc, y, n, sizes=(m,)) for m in (5, 10, 20)}, "per_model": per_model,
                     "mean_E_minus_y": float((E[name] - y).mean()), "E_minus_y_by_class": {LEVELS[c]: float((E[name] - y)[y == c].mean()) for c in range(len(LEVELS))},
                     "pred_counts": {LEVELS[c]: int(v) for c, v in enumerate(np.bincount([expected_score_decision(p) for p in P], minlength=len(LEVELS)))}}
        print(f"{name}: ensemble {rep[name]['ensemble']:.4f}  by n {rep[name]['by_n']}  mean E-y {rep[name]['mean_E_minus_y']:+.3f}  per model {min(per_model.values()):.4f}-{max(per_model.values()):.4f}", flush=True)
    dd = paired_diff(S["ct6"], S["ct3"], y, pid, n)
    dec3, dec6 = np.array([expected_score_decision(p) for p in np.mean(list(P_all["ct3"].values()), axis=0)]), np.array([expected_score_decision(p) for p in np.mean(list(P_all["ct6"].values()), axis=0)])
    rep["ct6_vs_ct3"] = {**dd, "decisions_differ": int((dec3 != dec6).sum()), "shift_counts": {str(k): int(v) for k, v in zip(*np.unique((dec6 - dec3)[dec3 != dec6], return_counts=True))},
                         "dE_mean": float((E["ct6"] - E["ct3"]).mean()), "dE_by_class": {LEVELS[c]: float((E["ct6"] - E["ct3"])[y == c].mean()) for c in range(len(LEVELS))}}
    print(f"ct6 vs ct3 on the held-out inner questions: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  by n {dd['by_n']}  decisions differ {rep['ct6_vs_ct3']['decisions_differ']}/{len(y)} "
          f"shifts {rep['ct6_vs_ct3']['shift_counts']}  dE mean {rep['ct6_vs_ct3']['dE_mean']:+.3f}", flush=True)
    print("E - y by class: ct3 " + " ".join(f"{k}:{v:+.2f}" for k, v in rep["ct3"]["E_minus_y_by_class"].items()))
    print("                ct6 " + " ".join(f"{k}:{v:+.2f}" for k, v in rep["ct6"]["E_minus_y_by_class"].items()))
    np.savez_compressed(Path(a.out).with_suffix(".P.npz"), y=y, n=n, pid=pid, keys=np.array([json.dumps(k) for k in keys]),
                        **{f"{name}__{m}": P for name in P_all for m, P in P_all[name].items()})
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out, f"and {Path(a.out).with_suffix('.P.npz').name} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
