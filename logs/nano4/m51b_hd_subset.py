"""Section 4M M51b check 2 (2026-11-07; user: 「1 2 都做」): how much of the HD block's contribution to ct9 comes from questions holding an
HD >= 2 game, which the test lacks? On validation fold v with ct9's six members (C/T-pc-f0-x40-oh seeds 0-2): the question embedding Z
once, then the last MLP with the HD block (R 59-63) (a) intact, (b) permuted across questions (one fixed permutation, as
feature_importance.py), (c) set to the training mean (zero after standardisation: no HD information). Class-balanced score of each on
all questions, on the HD-free questions and on the HD >= 2 questions, with the paired player bootstrap of the drop; flows A->1D and
1D->A on the HD-free subset. Report only.
  python -I logs/nano4/m51b_hd_subset.py --val-fold 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_seq_check as ts  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from feature_importance import probs_from, z_and_logits_fn  # noqa: E402
from m43_gate import decide, qscore  # noqa: E402
from m51_shift import confusion  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
A, D1 = RANK["A"], RANK["1D"]
HD_COLS = list(range(59, 64))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val-fold", type=int, required=True); a = ap.parse_args()
    t0, v, device = time.time(), a.val_fold, torch.device("cuda")
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.json"))["val"]
    keys = m["game_keys"]; y = np.array([RANK[r] for r in m["rank"]]); n = np.array(m["n_games"]); pid = np.array(m["player_id"], dtype=str)
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    hd2 = np.array([max(hd.get(g, 0.0) for g in ks) for ks in keys]) >= 2
    R_raw = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
    X_raw, offsets = ts.raw_tokens([(ts.VEC / f"fold{v}", ts.scores_path(ROOT / "results/trunk_seq", v, ts.VEC, 3, True))])
    members = [ts.load_checkpoint(ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.pt", device) for arm in ("C", "T") for s in (0, 1, 2)]
    Z = []
    for model, tstats, _, _ in members:
        Xs = torch.as_tensor(ts.standardise_tokens(X_raw.copy(), tstats), device=device)
        Z.append(z_and_logits_fn(model, Xs, offsets, keys, device)); del Xs; torch.cuda.empty_cache()
    del X_raw
    print(f"fold {v}: {len(keys)} questions, {int(hd2.sum())} with an HD >= 2 game ({hd2.mean():.1%}); Z ready ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.RandomState(1000 + v); perm = rng.permutation(len(keys))

    def probs(mod_fn):
        out = []
        for (mod, _, rstats, _), Zi in zip(members, Z):
            R_std = ((R_raw - rstats[0]) / rstats[1]).astype(np.float32); R_std = mod_fn(R_std)
            out.append(probs_from(mod, R_std, Zi, device))
        return np.mean(out, axis=0)

    def permute(R_std):
        R2 = R_std.copy(); R2[:, HD_COLS] = R_std[perm][:, HD_COLS]; return R2

    def zero(R_std):
        R2 = R_std.copy(); R2[:, HD_COLS] = 0.0; return R2

    P = {"intact": probs(lambda R: R), "permuted": probs(permute), "zeroed": probs(zero)}
    S = {k: qscore(decide(p), y) for k, p in P.items()}
    rep = {"val_fold": v, "questions": int(len(y)), "hd2_questions": int(hd2.sum()), "subsets": {}}
    for name, keep in (("all", np.ones(len(y), bool)), ("hd_free", ~hd2), ("hd2", hd2)):
        r = {"questions": int(keep.sum()), "score_intact": float(balanced_mean_over_n(S["intact"][keep], y[keep], n[keep]))}
        for alt in ("permuted", "zeroed"):
            dd = paired_diff(S["intact"][keep], S[alt][keep], y[keep], pid[keep], n[keep], n=2000)
            r[f"drop_{alt}"] = dd
        C0, C1 = confusion(decide(P["intact"][keep]), y[keep], n[keep]), confusion(decide(P["zeroed"][keep]), y[keep], n[keep])
        r["flows_intact"] = {"A->1D": float(C0[D1, A]), "1D->A": float(C0[A, D1]), "A->A": float(C0[A, A])}
        r["flows_zeroed"] = {"A->1D": float(C1[D1, A]), "1D->A": float(C1[A, D1]), "A->A": float(C1[A, A])}
        rep["subsets"][name] = r
        print(f"{name:8s} ({int(keep.sum())} q): intact {r['score_intact']:.4f}; drop when HD permuted {r['drop_permuted']['diff']:+.4f} [{r['drop_permuted']['ci95'][0]:+.4f}, {r['drop_permuted']['ci95'][1]:+.4f}]"
              f"; drop when HD zeroed {r['drop_zeroed']['diff']:+.4f} [{r['drop_zeroed']['ci95'][0]:+.4f}, {r['drop_zeroed']['ci95'][1]:+.4f}]"
              f"; flows A->1D {r['flows_intact']['A->1D']:.3f} -> {r['flows_zeroed']['A->1D']:.3f}, 1D->A {r['flows_intact']['1D->A']:.3f} -> {r['flows_zeroed']['1D->A']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    Path(ROOT / f"results/trunk_seq/es/m51b_hd_subset_val{v}.json").write_text(json.dumps(rep, indent=1, default=float)); print("written", flush=True)


if __name__ == "__main__":
    main()
