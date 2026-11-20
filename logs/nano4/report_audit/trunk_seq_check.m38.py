#!/usr/bin/env python
"""Does an in-game temporal layer add to trunk vectors under a whole-question loss? (architecture v2 section 4A)

Three arms on one trainer, same questions, same seeds:
  R0  the 64 R features (fused_opp_diff_hd on the _hd scores) only;
  C   R + per-ply residual blocks (kernel 1) over trunk vectors and per-move log p, pooled per game, then per question;
  T   the same with five dilated kernel-3 blocks (63 plies).
Everything from the loss back to the blocks trains together; the trunk vectors are fixed (scripts/trunk_extract.py).
C's hidden size is chosen so its blocks hold about as many parameters as T's. With --inner-frac the stopping point is chosen on
held-out training players only and the validation fold is scored once, at that point. Fold 0 is refused.

  python -I scripts/trunk_seq_check.py build --out results/trunk_seq
  python -I scripts/trunk_seq_check.py train --arm T --val-fold 1 --seed 0 --out results/trunk_seq
  python -I scripts/trunk_seq_check.py rfeat --val-fold 2
  python -I scripts/trunk_seq_check.py compare --val-folds 2 3 4 --out results/trunk_seq/es
  --loss kernel | es-ft  (section 4F) cross-entropy against the official score's kernel (tag -k), or the expected-score
                         fine-tune after the cross-entropy stage, stopped on the inner players (tag -es)
  python -I scripts/trunk_seq_check.py train --arm C --val-fold all --seed 0 --inner-frac 0.1   # the submission's models (2026-11-04):
      every fold's players train, the 10% inner hold-out picks the stopping point, the weights and both standardisations go to
      results/models/ct/all_<tag>_s<seed>.pt (+ .json sidecar); --save-dir also keeps a validation-fold model's weights.
      scripts/trunk_predict.py reads them (load_checkpoint).
  Section 4M (2026-11-04) data-axis arms, same trainer: --train-folds 2 (tag -f2: two of the other folds train, fold (v mod 4) + 1
      dropped), --train-games 10 --train-sizes 5 10 (tag -g10s5-10: each training player keeps 10 scored games, questions of 5-10
      games; -s5-10 alone is its control), --vec-dir results/trunk_vec_ft (tag -ft: another trunk's vectors; `build --vec-dir` first
      writes their score arrays scores_ft_fold<f>.npy). compare --sizes 5 10 evaluates only those question sizes.
  Section 4M M4: --pooling pma (tag -pma) replaces both mean/std poolings by pooling-by-multihead-attention (Lee et al. 2019): four
      learned seeds attend over a game's plies, two over a question's games; the linear layers after them keep their shapes.
  Section 4M M14: --opp-hsl (tag -oh) fills the opponent plies' Human SL token block (scores columns 11-29, zeros before) with the
      centred log p of the OPPONENT's moves under the same 18 profiles (results/scores_opp_hd, _x40; score_folds.py --opponent);
      `build --opp-hsl` writes the scores_oh_* arrays. --opp-hsl-r (tag -ohr) also appends the 39 opponent-HSL question features
      (aicup.opphsl) to R (64 -> 103); the R cache of -oh is the plain one.
  Section 4M M8: --extra-games (tag -x40) adds every training player's games 21-40 (scored into the *_x40 directories, EXTRA) to the
      games its questions draw from; the validation questions stay the 20-game sample's. --with-fold0 (tag -f0, --val-fold all only)
      adds fold 0's players to the submission's training set (the user's decision). `build --build-folds 0` and
      `build --vec-dir results/trunk_vec_x40 --extra-games` write the matching score arrays.
  Section 4M M30 (2026-11-06): --token-enc dual (tag -dual) replaces the first Linear(162, d) by a board tower (trunk g, l: 128 ->
      48) and a score tower (L1b + Human SL: 30 -> 24), each Linear-ReLU-Linear-LayerNorm, then Linear(48 + 24 + 4 flags, d);
      --token-enc clip (tag -clip) adds CLIP-style alignment: both towers' outputs are projected to 32 dims and L2-normalised, and a
      symmetric InfoNCE over up to 2,048 real plies of the batch (the same ply's two views are the positive pair, learnable temperature)
      joins the loss with weight --clip-weight (0.1). The heads only shape training; evaluation never computes the InfoNCE.
  Section 4M M33 (2026-11-06): --player-contrast W (tag -pc, -pc<W> when W != 0.1) builds every batch from pairs of training
      questions of the same player (each player's questions paired at random each epoch) and adds W x a symmetric InfoNCE between the
      two questions' embeddings (the question layer's output before dropout, through a training-only Linear(32, 32), L2-normalised,
      temperature 0.1): other players in the batch are the negatives. The checkpoint holds the model only (the head is not saved).
      Diagnostics: --pair-batches (tag -pb) builds the same paired batches without the InfoNCE; --tag-extra X appends -X to the arm
      tag (a same-recipe re-run beside the original files).
  Section 4M M36 (2026-11-06): --r-extra DIR (tag -ra) appends to R the question mean and std of the cross-fitted spatial-attention
      heads' per-game log-softmax (aicup.attnfeat; DIR = logs/nano4/e2e_pilot.py score parts, --r-extra-games the games CSV): 64 -> 84.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.metric_loss import expected_score_loss_torch, kernel_ce_torch, kernel_targets, score_matrix  # noqa: E402
from aicup.rankmodel import class_weights  # noqa: E402
from aicup.opphsl import opp_hsl_features  # noqa: E402
from aicup.scoring import _part_moves, completed_parts, part_paths, read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402

FOLDS = (1, 2, 3, 4)
VEC, HSL = ROOT / "results/trunk_vec", ROOT / "results/scores_hd"
L1B, OPP = {1: "scores3_B_hd", 2: "scores3_B_hd", 3: "scores3_hd", 4: "scores3_hd"}, {1: "scores3_B_opp_hd", 2: "scores3_B_opp_hd", 3: "scores3_opp_hd", 4: "scores3_opp_hd"}
L1B[0], OPP[0] = "scores3_hd", "scores3_opp_hd"  # fold 0: both L1b models are out-of-fold; A's scores feed its tokens and R (section 4M M8)
L1B_SETS = {  # section 4M M9: which L1b heads' per-move log p feed the tokens and R; set 4 = heads4_A (folds 1, 3) / heads4_B (folds 2, 4)
    3: {"l1b": L1B, "opp": OPP, "heads": ["heads3_A", "heads3_B"]},
    4: {"l1b": {0: "scores4_hd", 2: "scores4_hd", 4: "scores4_hd", 1: "scores4_B_hd", 3: "scores4_B_hd"},
        "opp": {0: "scores4_opp_hd", 2: "scores4_opp_hd", 4: "scores4_opp_hd", 1: "scores4_B_opp_hd", 3: "scores4_B_opp_hd"}, "heads": ["heads4_A", "heads4_B"]}}
EXTRA_MAX_GAMES = 40  # section 4M M8: games 21-40 of fold_players' permutation, scored into these directories (logs/nano4/x40_keys.py)
EXTRAS = {k: {"hsl": ROOT / f"results/scores_hd_x{k}", "vec": ROOT / f"results/trunk_vec_x{k}", "suffix": f"_x{k}"} for k in (40, 80)}  # games 21-40 / 41-80
EXTRA = {**EXTRAS[40], "l1b": {f: v + "_x40" for f, v in L1B.items()}, "opp": {f: v + "_x40" for f, v in OPP.items()}}  # the x40 set (section 4M M8)
for _k, _d in L1B_SETS.items():  # the extra directories of each L1b set: <dir>_x40, <dir>_x80
    _d["l1b_x"], _d["opp_x"] = {f: v + "_x40" for f, v in _d["l1b"].items()}, {f: v + "_x40" for f, v in _d["opp"].items()}
    _d["l1b_x80"], _d["opp_x80"] = {f: v + "_x80" for f, v in _d["l1b"].items()}, {f: v + "_x80" for f, v in _d["opp"].items()}
OPP_HSL = ROOT / "results/scores_opp_hd"  # section 4M M14: the opponent's moves under the same 18 profiles (scripts/score_folds.py --opponent)
OPP_HSL_X = {40: ROOT / "results/scores_opp_hd_x40"}  # the x40 keys' opponent scores; games 41-80 were not scored this way
HSL_TAG = os.environ.get("AICUP_HSL", "")  # section 4M M27: e.g. "jp" -> results/scores_hd_jp, scores_hd_x40_jp, scores_opp_hd_jp, scores_opp_hd_x40_jp
if HSL_TAG:
    HSL = ROOT / f"results/scores_hd_{HSL_TAG}"
    EXTRAS[40]["hsl"] = ROOT / f"results/scores_hd_x40_{HSL_TAG}"
    OPP_HSL = ROOT / f"results/scores_opp_hd_{HSL_TAG}"
    OPP_HSL_X = {40: ROOT / f"results/scores_opp_hd_x40_{HSL_TAG}"}
    EXTRA["hsl"] = EXTRAS[40]["hsl"]


def opp_hsl_level(a):
    """0: no opponent HSL; 1 (--opp-hsl, tag -oh): the opponent plies' Human SL token block (scores columns 11-29) holds the centred
    log p of the opponent's moves instead of zeros; 2 (--opp-hsl-r, tag -ohr): and the 39 opponent-HSL question features join R."""
    return 2 if getattr(a, "opp_hsl_r", False) else (1 if getattr(a, "opp_hsl", False) else 0)


def opp_hsl_dir(level_k):
    """The opponent-HSL table directory of extra level k (0: the 20-game sample); SystemExit when those games were not scored."""
    if not level_k:
        return OPP_HSL
    if level_k not in OPP_HSL_X:
        raise SystemExit(f"--opp-hsl: the opponent's moves of the x{level_k} games were not scored under the 18 profiles (x{sorted(OPP_HSL_X)} were)")
    return OPP_HSL_X[level_k]


def opp_hsl_table(directory, f, hsl):
    """The opponent-HSL scored games of fold f under `directory`, restricted to the games of the target-side table `hsl` (which they must all cover)."""
    t = read_scored_games(directory / f"fold{f}")
    missing = set(hsl["game_key"]) - set(t["game_key"])
    if missing:
        raise ValueError(f"{directory}/fold{f}: {len(missing)} games of the target-side HSL table have no opponent scores, e.g. {sorted(missing)[:3]}")
    return t[t["game_key"].isin(set(hsl["game_key"]))].reset_index(drop=True)


def extra_level(a):
    """0 (the 20-game sample only), 40 (+ games 21-40) or 80 (+ games 21-80): --extra-games."""
    k = int(getattr(a, "extra_games", 0) or 0)
    if k not in (0, 40, 80):
        raise SystemExit(f"--extra-games {k}: 40 or 80")
    if HSL_TAG and k == 80:  # section 4M M27: games 41-80 were not scored under AICUP_RULES
        raise SystemExit(f"AICUP_HSL={HSL_TAG}: --extra-games 80 has no {HSL_TAG} Human SL tables")
    return k


def extra_sets(level):
    """The extra-game levels included up to `level`: [] / [40] / [40, 80]."""
    return [k for k in (40, 80) if k <= level]


def set_dirs_for(dirs, k):
    """(l1b map, opp map) of L1b set `dirs` for extra level k (40 or 80)."""
    return (dirs["l1b_x"], dirs["opp_x"]) if k == 40 else (dirs["l1b_x80"], dirs["opp_x80"])


def l1b_set_of(a):
    """The L1b set a run uses (3 unless --l1b-set says otherwise)."""
    k = int(getattr(a, "l1b_set", 3) or 3)
    if k not in L1B_SETS:
        raise SystemExit(f"--l1b-set {k}: known sets {sorted(L1B_SETS)}")
    return k
N_SCORES = 30  # L1b: level + 10 centred; Human SL: level + 18 centred, target plies only (zero elsewhere)
D_TOKEN = 64 + 64 + N_SCORES + 4  # + is_target, mover, t / T, pass
HIDDEN = {"C": 110, "T": 64, "S": 192}  # S (section 4M M21, 2026-11-05): a 4-layer pre-norm transformer over the game's plies, full attention
GAME_DIM, Q_DIM = 64, 32
ARMS = ("R0", "C", "T", "S")
S_LAYERS, S_HEADS, S_MAX_PLIES = 4, 4, 512


def centred(logp):
    level = logp.mean(axis=1, keepdims=True)
    return np.hstack([level, logp - level])


def build(a):
    """One [plies, 30] fp16 array per fold, in the order of the trunk-vector parts, checked ply by ply."""
    out = Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    vec_dir = Path(getattr(a, "vec_dir", VEC))
    extra = extra_level(a)
    dirs = L1B_SETS[l1b_set_of(a)]
    oh = opp_hsl_level(a) > 0
    if extra:  # the score arrays of ONE extra vec dir: its level comes from the directory name (trunk_vec_x40 / trunk_vec_x80)
        k = 80 if str(vec_dir).endswith("_x80") else 40
        l1b_dirs, opp_dirs = set_dirs_for(dirs, k)
        hsl_dir = EXTRAS[k]["hsl"]
        oh_dir = opp_hsl_dir(k) if oh else None
    else:
        l1b_dirs, opp_dirs, hsl_dir = dirs["l1b"], dirs["opp"], HSL
        oh_dir = OPP_HSL if oh else None
    for f in (getattr(a, "build_folds", None) or FOLDS):
        fold_dir = vec_dir / f"fold{f}"
        tgt, opp, hsl = (_part_moves(ROOT / "results/l1b" / l1b_dirs[f] / f"fold{f}"), _part_moves(ROOT / "results/l1b" / opp_dirs[f] / f"fold{f}"), _part_moves(hsl_dir / f"fold{f}"))
        ohsl = _part_moves(oh_dir / f"fold{f}") if oh else None  # section 4M M14: the opponent's plies under the 18 profiles
        blocks, bad = [], 0
        for s, k in completed_parts(fold_dir):
            csv_path, npz_path = part_paths(fold_dir, s, k)
            games = pd.read_csv(csv_path, usecols=["game_key"])["game_key"].tolist()
            with np.load(npz_path) as z:
                row, t, target = z["game_row"], z["t"], z["target"]
            sc = np.zeros((len(t), N_SCORES), np.float32)
            for g, key in enumerate(games):
                idx = np.flatnonzero(row == g)
                plies_t, plies_o = t[idx][target[idx]], t[idx][~target[idx]]
                if not (np.array_equal(plies_t, tgt[key][2]) and np.array_equal(plies_o, opp[key][2]) and np.array_equal(plies_t, hsl[key][2])) \
                        or (oh and not (key in ohsl and np.array_equal(plies_o, ohsl[key][2]))):
                    bad += 1
                    continue
                sc[idx[target[idx]], :11] = centred(tgt[key][1])
                sc[idx[~target[idx]], :11] = centred(opp[key][1])
                sc[idx[target[idx]], 11:] = centred(hsl[key][1])
                if oh:  # the opponent plies' Human SL block (zeros in the plain arrays)
                    sc[idx[~target[idx]], 11:] = centred(ohsl[key][1])
            blocks.append(sc.astype(np.float16))
        if bad:
            raise ValueError(f"fold {f}: {bad} games whose plies do not line up with the stored scores")
        arr = np.concatenate(blocks)
        path = scores_path(out, f, vec_dir, l1b_set_of(a), oh)
        np.save(path, arr)
        print(f"fold {f}: {len(arr)} plies, every game lined up ({path.name})", flush=True)


def raw_tokens(parts):
    """Unstandardised tokens (X [plies, D_TOKEN] float16: trunk g, l, the 30 scores, is_target, mover, t / T, pass) and the game
    table {game_key: (start, length)} of trunk-vector directories with their score arrays. parts: [(vec_dir, scores.npy)], the
    score array in the ply order of completed_parts(vec_dir) (what build and scripts/trunk_predict.py write)."""
    out, offsets, start = [], {}, 0
    for fold_dir, sc_path in parts:
        sc = np.load(sc_path, mmap_mode="r")
        o = 0
        for s, k in completed_parts(fold_dir):
            csv_path, npz_path = part_paths(fold_dir, s, k)
            tab = pd.read_csv(csv_path, usecols=["game_key", "n_plies"], dtype={"game_key": str}, keep_default_na=False)
            with np.load(npz_path) as z:
                g, l, pos, mover, target, t = z["g"], z["l"], z["pos"], z["mover"], z["target"], z["t"]
            lens = tab["n_plies"].to_numpy()
            T = np.repeat(lens, lens).astype(np.float32)
            flags = np.stack([target.astype(np.float32), mover.astype(np.float32), t / T, (pos == 361).astype(np.float32)], axis=1)
            out.append(np.hstack([g, l, np.asarray(sc[o:o + len(t)]), flags.astype(np.float16)]))
            for key, n in zip(tab["game_key"], lens):
                offsets[key] = (start, int(n))
                start += int(n)
            o += len(t)
        if o != len(sc):
            raise ValueError(f"{sc_path}: {len(sc)} plies, the parts of {fold_dir} hold {o}")
    return np.concatenate(out), offsets


def token_stats(X):
    """The training-side standardisation of the first D_TOKEN - 4 columns: (mean, std) of about 2M plies spread over X."""
    head = X[:: max(1, len(X) // 2_000_000), :D_TOKEN - 4].astype(np.float32)
    return head.mean(axis=0), head.std(axis=0) + 1e-6


def standardise_tokens(X, stats, drop=None):
    """In place: the first D_TOKEN - 4 columns standardised with `stats` (fp16 storage); an ablation zeroes the trunk or score columns."""
    X[:, :D_TOKEN - 4] = ((X[:, :D_TOKEN - 4].astype(np.float32) - stats[0]) / stats[1]).astype(np.float16)
    if drop == "trunk":  # ablation: the columns stay, so every arm has the same parameters
        X[:, :128] = 0
    elif drop == "scores":
        X[:, 128:128 + N_SCORES] = 0
    return X


def load_token_dirs(parts, device, stats=None, drop=None):
    """raw_tokens of `parts`, standardised (with `stats`, else with their own token_stats) in device memory: (X, offsets, stats)."""
    X, offsets = raw_tokens(parts)
    if stats is None:
        stats = token_stats(X)
    return torch.as_tensor(standardise_tokens(X, stats, drop), device=device), offsets, stats


def load_tokens(folds, cache, device, stats=None, drop=None, vec_dir=VEC, extra=0, l1b_set=3, opp_hsl=False):
    """Tokens of `folds` in device memory (fp16), the game table {key: (start, length)}, and the training-side standardisation;
    extra (section 4M M8: 40, x80: 80) also loads the folds' games 21-40 (and 41-80) from EXTRAS; l1b_set and opp_hsl pick the score arrays."""
    parts = [(Path(vec_dir) / f"fold{f}", scores_path(cache, f, vec_dir, l1b_set, opp_hsl)) for f in folds]
    for k in extra_sets(int(extra or 0)):
        parts += [(EXTRAS[k]["vec"] / f"fold{f}", scores_path(cache, f, EXTRAS[k]["vec"], l1b_set, opp_hsl)) for f in folds]
    return load_token_dirs(parts, device, stats, drop)


def fold_code(v):
    """The integer that seeds a run's draws: the validation fold, or 5 for a run on all four folds ("all")."""
    return 5 if v == "all" else int(v)


def training_folds(v, n_train=0, with_fold0=False):
    """The folds whose players train for validation fold v: every other fold, or (section 4M M1) only n_train of them, dropping
    fold (v mod 4) + 1 first, so that over v = 1..4 every fold is dropped once. "all": every fold, plus fold 0 with with_fold0
    (section 4M M8, the submission's models only)."""
    if v == "all":
        if n_train:
            raise SystemExit("--train-folds applies to a validation fold, not to 'all'")
        return ([0] if with_fold0 else []) + list(FOLDS)
    others = [f for f in FOLDS if f != int(v)]
    if with_fold0:  # section 4M M20 (2026-11-05): the production recipe cross-validated -- fold 0 trains, fold v is held out (tag -f0)
        others = [0] + others
    if not n_train or n_train >= len(others):
        return others
    if n_train < 1:
        raise SystemExit("--train-folds takes 1-3")
    drop = int(v) % 4 + 1
    out = [f for f in others if f != drop]
    return out[:n_train]


def cap_games(players, n_games, seed=4242):
    """Section 4M M2: every player keeps at most n_games of its scored games, chosen by a permutation seeded from the player id
    (independent of row order and of the other players, like scoring.fold_players); players with fewer games are unchanged."""
    if not n_games:
        return players
    out = players.copy()
    out["game_keys"] = [list(k) if len(k) <= n_games else [k[i] for i in np.random.RandomState((int(str(p)[:8], 16) + seed) % 2**32).permutation(len(k))[:n_games]]
                        for p, k in zip(out["player_id"], out["game_keys"])]
    return out


def data_segments(a):
    """The training-data part of a run's tag, as segments (sections 4L.4 and 4M): f<train folds>, g<games per player>[s<min>-<max>],
    s<min>-<max> (question sizes), mix<pct>. Joined with "-" in the arm tag and with "_" in the R cache tag."""
    n_train, games, sizes = int(getattr(a, "train_folds", 0) or 0), int(getattr(a, "train_games", 0) or 0), getattr(a, "train_sizes", None)
    mix = float(getattr(a, "mix", 0.0) or 0.0)
    seg = []
    if getattr(a, "with_fold0", False):
        seg.append("f0")
    if n_train:
        seg.append(f"f{n_train}")
    if int(getattr(a, "l1b_set", 3) or 3) != 3:
        seg.append(f"h{int(a.l1b_set)}")
    if games or sizes:
        seg.append((f"g{games}" if games else "") + (f"s{sizes[0]}-{sizes[1]}" if sizes else ""))
    if int(getattr(a, "extra_games", 0) or 0):
        seg.append(f"x{int(a.extra_games)}")
    if opp_hsl_level(a):  # section 4M M14
        seg.append("oh" if opp_hsl_level(a) == 1 else "ohr")
    if mix > 0:
        seg.append(f"mix{round(mix * 100)}")
    if HSL_TAG:  # section 4M M27: the Human SL tables of another replay setting
        seg.append(HSL_TAG)
    return seg


def vec_tag(vec_dir):
    """"" for the default trunk vectors, else "-<name>" with the trunk_vec_ prefix stripped (results/trunk_vec_ft -> "-ft")."""
    p = Path(vec_dir)
    if p.resolve() == VEC.resolve():
        return ""
    return "-" + (p.name[len("trunk_vec_"):] if p.name.startswith("trunk_vec_") else p.name)


def scores_path(cache, f, vec_dir=VEC, l1b_set=3, opp_hsl=False):
    """The per-ply score array of fold f laid out in the part order of vec_dir (build writes it): scores[_<vec tag>][_h<set>][_oh]_fold<f>.npy
    (section 4M M14: _oh = the opponent plies' Human SL block filled)."""
    return Path(cache) / f"scores{vec_tag(vec_dir).replace('-', '_')}{'' if l1b_set == 3 else f'_h{l1b_set}'}{'_oh' if opp_hsl else ''}{'_' + HSL_TAG if HSL_TAG else ''}_fold{f}.npy"


CKPT_FORMAT = 1


def save_checkpoint(path, model, token_stats_, r_stats, meta):
    """<path> (.pt): the model's state (cpu) with the token and R standardisation it was trained with; the sidecar <stem>.json
    records the sha256, the token layout constants and `meta` (the training facts). Returns the sidecar dict."""
    from aicup.l1b import atomic_save
    from aicup.ladder import environment, to_json
    from aicup.setmodel_io import file_sha256, sidecar_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_r = int(model.mlp[0].in_features - (0 if model.arm == "R0" else Q_DIM))
    obj = {"format": CKPT_FORMAT, "arm": model.arm, "n_r": n_r, "pool": model.pool, "token_enc": getattr(model, "token_enc", "linear"), "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "token_mean": None if token_stats_ is None else torch.as_tensor(np.asarray(token_stats_[0], dtype=np.float32)),
           "token_std": None if token_stats_ is None else torch.as_tensor(np.asarray(token_stats_[1], dtype=np.float32)),
           "r_mean": torch.as_tensor(np.asarray(r_stats[0], dtype=np.float32)), "r_std": torch.as_tensor(np.asarray(r_stats[1], dtype=np.float32))}
    atomic_save(obj, path)
    side = {"format": CKPT_FORMAT, "arm": model.arm, "n_r": n_r, "pool": model.pool, "token_enc": getattr(model, "token_enc", "linear"), "state_sha256": file_sha256(path), "d_token": D_TOKEN, "n_scores": N_SCORES,
            "hidden": HIDDEN.get(model.arm), "game_dim": GAME_DIM, "q_dim": Q_DIM, **meta,
            **environment("cuda" if torch.cuda.is_available() else "cpu")}
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")
    return side


def load_checkpoint(path, device="cpu"):
    """(Model in eval mode on `device`, token stats (mean, std) or None, (r_mean, r_std), sidecar) of a save_checkpoint file.
    ValueError when the sidecar is missing, its format is not this code's, the .pt sha256 differs, or the token layout differs."""
    from aicup.setmodel_io import file_sha256, sidecar_path
    path = Path(path)
    side_path = sidecar_path(path)
    if not side_path.exists():
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")
    side = json.loads(side_path.read_text())
    if side.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path}: sidecar format {side.get('format')!r}, this code reads {CKPT_FORMAT}")
    sha = file_sha256(path)
    if side.get("state_sha256") != sha:
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(side.get('state_sha256'))[:12]}")
    layout = (side.get("d_token"), side.get("n_scores"), side.get("game_dim"), side.get("q_dim"))
    if layout != (D_TOKEN, N_SCORES, GAME_DIM, Q_DIM):
        raise ValueError(f"{path}: token layout {layout} is not this code's {(D_TOKEN, N_SCORES, GAME_DIM, Q_DIM)}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    model = Model(obj["arm"], int(obj["n_r"]), pool=obj.get("pool", "meanstd"), token_enc=obj.get("token_enc", "linear"))
    model.load_state_dict(obj["state"])
    model.to(device).eval()
    stats = None if obj["token_mean"] is None else (obj["token_mean"].numpy(), obj["token_std"].numpy())
    return model, stats, (obj["r_mean"].numpy(), obj["r_std"].numpy()), side


class Block(torch.nn.Module):
    def __init__(self, d, kernel, dilation, dropout):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.conv = torch.nn.Conv1d(d, d, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.conv(torch.relu(self.norm(x)).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(h) * mask


def masked_mean_std(x, w, dim):
    n = w.sum(dim=dim, keepdim=True).clamp(min=1.0)
    m = (x * w).sum(dim=dim, keepdim=True) / n
    v = (((x - m) ** 2) * w).sum(dim=dim, keepdim=True) / n
    return m.squeeze(dim), (v + 1e-6).sqrt().squeeze(dim)


class PMA(torch.nn.Module):
    """Pooling by multihead attention (Lee et al. 2019, Set Transformer): k learned seed vectors attend over a masked set [B, N, d]
    (pad True where padding) and come back as one [B, k * d] vector (seed + attention output). Every row needs one unpadded element."""

    def __init__(self, d, k, heads, dropout):
        super().__init__()
        self.seed = torch.nn.Parameter(torch.randn(1, k, d) / d ** 0.5)
        self.norm = torch.nn.LayerNorm(d)
        self.att = torch.nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

    def forward(self, x, pad):
        q = self.seed.expand(x.shape[0], -1, -1)
        kv = self.norm(x)
        h, _ = self.att(q, kv, kv, key_padding_mask=pad, need_weights=False)
        return (q + h).flatten(1)


POOLINGS = ("meanstd", "pma")
TOKEN_ENCODERS = ("linear", "dual", "clip")  # section 4M M30: one Linear over the 162 columns, or a board tower and a score tower (+ CLIP alignment)
DUAL_H = (48, 24)  # M30: widths of the board tower (trunk g, l: 128 columns) and the score tower (L1b + Human SL: 30 columns)
CLIP_DIM, CLIP_PLIES = 32, 2048  # M30 clip: projection width of the alignment heads, plies sampled per batch for the InfoNCE
PC_TEMP = 0.1  # section 4M M33: temperature of the player-contrastive InfoNCE


def paired_perm(pid, rng):
    """Section 4M M33: a permutation of the question indices in which positions 2k and 2k + 1 hold two questions of the same player;
    each player's questions are shuffled and paired (an odd one out is dropped), then the pairs are shuffled."""
    pid = np.asarray(pid)
    order = np.argsort(pid, kind="stable")
    bounds = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1], True])
    pairs = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi][rng.permutation(hi - lo)]
        pairs += [idx[i:i + 2] for i in range(0, len(idx) - 1, 2)]
    return np.concatenate([pairs[i] for i in rng.permutation(len(pairs))]) if pairs else np.zeros(0, int)


def player_contrast_loss(z, head):
    """Section 4M M33: symmetric InfoNCE between rows 2k and 2k + 1 of z [2m, d] (two questions of one player), the other pairs as negatives."""
    h = torch.nn.functional.normalize(head(z), dim=1)
    a, b = h[0::2], h[1::2]
    logits = a @ b.T / PC_TEMP
    target = torch.arange(a.shape[0], device=z.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))


class Tower(torch.nn.Module):
    """Section 4M M30: one input block's own encoder, Linear -> ReLU -> Linear -> LayerNorm (each block is scaled on its own before mixing)."""

    def __init__(self, n_in, h):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, h))
        self.norm = torch.nn.LayerNorm(h)

    def forward(self, x):
        return self.norm(self.net(x))


class Model(torch.nn.Module):
    def __init__(self, arm, n_r, dropout=0.1, pool="meanstd", token_enc="linear"):
        super().__init__()
        if pool not in POOLINGS:
            raise ValueError(f"pooling {pool!r}, expected one of {POOLINGS}")
        if token_enc not in TOKEN_ENCODERS:
            raise ValueError(f"token encoder {token_enc!r}, expected one of {TOKEN_ENCODERS}")
        self.arm, self.pool, self.token_enc = arm, pool, ("linear" if arm == "R0" else token_enc)
        self.aux_loss = None  # M30 clip: the InfoNCE of the last training forward pass (None otherwise)
        self.q_embed = None  # M33: the question embedding of the last training forward pass
        if arm != "R0":
            d = HIDDEN[arm]
            kernel = 3 if arm == "T" else 1
            if self.token_enc == "linear":
                self.proj = torch.nn.Linear(D_TOKEN, d)
            else:  # M30: board tower and score tower, then their outputs and the 4 flags -> d
                hb, hs = DUAL_H
                self.board, self.score = Tower(128, hb), Tower(N_SCORES, hs)
                self.proj = torch.nn.Linear(hb + hs + 4, d)
                if self.token_enc == "clip":  # CLIP-style alignment heads (training loss only) and a learnable temperature
                    self.clip_board, self.clip_score = torch.nn.Linear(hb, CLIP_DIM), torch.nn.Linear(hs, CLIP_DIM)
                    self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            if arm == "S":  # section 4M M21: full self-attention over the plies of a game, learned ply-position embedding
                self.pos = torch.nn.Embedding(S_MAX_PLIES, d)
                self.blocks = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d, S_HEADS, dim_feedforward=4 * d, dropout=dropout, batch_first=True, norm_first=True)
                                                   for _ in range(S_LAYERS)])
                self.final_norm = torch.nn.LayerNorm(d)
            else:
                self.blocks = torch.nn.ModuleList([Block(d, kernel, 2 ** k if arm == "T" else 1, dropout) for k in range(5)])
            self.game = torch.nn.Linear(4 * d, GAME_DIM)
            self.question = torch.nn.Linear(2 * GAME_DIM, Q_DIM)
            self.drop = torch.nn.Dropout(dropout)
            if pool == "pma":  # section 4M M4: 4 seeds over a game's plies (-> 4d, as the two mean/std pairs), 2 seeds over a question's games (-> 2 * GAME_DIM)
                self.game_pool = PMA(d, 4, 2, dropout)
                self.q_pool = PMA(GAME_DIM, 2, 2, dropout)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_r + (0 if arm == "R0" else Q_DIM), 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))

    def embed(self, tokens):
        """The per-ply input projection [G, L, d] and, for the two-tower encoders, the tower outputs (board, score)."""
        if self.token_enc == "linear":
            return self.proj(tokens), None
        b, s = self.board(tokens[..., :128]), self.score(tokens[..., 128:128 + N_SCORES])
        return self.proj(torch.cat([b, s, tokens[..., D_TOKEN - 4:]], dim=-1)), (b, s)

    def clip_loss(self, views, mask):
        """M30: symmetric InfoNCE between the board and score views of up to CLIP_PLIES real plies (the same ply is the positive pair)."""
        b, s = views
        idx = torch.nonzero(mask[..., 0].reshape(-1) > 0).squeeze(1)
        if idx.numel() < 2:
            return None
        if idx.numel() > CLIP_PLIES:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:CLIP_PLIES]]
        zb = torch.nn.functional.normalize(self.clip_board(b.reshape(-1, b.shape[-1])[idx]), dim=1)
        zs = torch.nn.functional.normalize(self.clip_score(s.reshape(-1, s.shape[-1])[idx]), dim=1)
        logits = self.logit_scale.clamp(max=math.log(100.0)).exp() * zb @ zs.T
        target = torch.arange(idx.numel(), device=logits.device)
        return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))

    def encode_games(self, tokens, mask):
        h, views = self.embed(tokens)
        h = h * mask
        self.aux_loss = self.clip_loss(views, mask) if (self.token_enc == "clip" and self.training) else None
        if self.arm == "S":
            L = tokens.shape[1]
            h = h + self.pos(torch.arange(L, device=tokens.device).clamp(max=S_MAX_PLIES - 1))[None]
            pad = ~mask[:, :, 0].bool()
            for b in self.blocks:
                h = b(h, src_key_padding_mask=pad)
            h = self.final_norm(h) * mask
        else:
            for b in self.blocks:
                h = b(h, mask)
        if self.pool == "pma":
            return torch.relu(self.game(self.game_pool(h, ~mask[:, :, 0].bool())))
        target = tokens[:, :, D_TOKEN - 4:D_TOKEN - 3] * mask
        other = (1.0 - tokens[:, :, D_TOKEN - 4:D_TOKEN - 3]) * mask
        tm, ts = masked_mean_std(h, target, 1)
        om, os_ = masked_mean_std(h, other, 1)
        return torch.relu(self.game(torch.cat([tm, ts, om, os_], dim=1)))

    def pool_games(self, v, game_of, B):
        """[B, 2 * GAME_DIM]: the mean and std of each question's game vectors, or (pma) two seeds' attention over them."""
        owner = game_of[:, 0]
        if self.pool == "pma":
            counts = torch.bincount(owner, minlength=B)
            first = torch.cumsum(counts, 0) - counts
            pos = torch.arange(owner.shape[0], device=v.device) - first[owner]  # games of a question are consecutive in game_of
            padded = torch.zeros(B, int(counts.max()), v.shape[1], device=v.device, dtype=v.dtype)
            padded[owner, pos] = v
            pad = torch.arange(padded.shape[1], device=v.device)[None, :] >= counts[:, None]
            return self.q_pool(padded, pad)
        w = torch.zeros(B, owner.shape[0], device=v.device)
        w[owner, torch.arange(owner.shape[0], device=v.device)] = 1.0
        n = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        m = (w @ v) / n
        s = ((w @ (v ** 2)) / n - m ** 2).clamp(min=0.0).add(1e-6).sqrt()
        return torch.cat([m, s], dim=1)

    def forward(self, r, tokens=None, mask=None, game_of=None):
        if self.arm == "R0":
            return self.mlp(r)
        v = self.encode_games(tokens, mask)  # [G, 64]
        zq = torch.relu(self.question(self.pool_games(v, game_of, r.shape[0])))
        self.q_embed = zq if self.training else None  # section 4M M33: the question embedding before dropout, for the player-contrastive loss
        z = self.drop(zq)
        return self.mlp(torch.cat([r, z], dim=1))


def swa_average(eval_states, best_step, k):
    """Uniform average of the k evaluation-time state dicts centred on the one recorded at best_step (clamped to the ends)."""
    steps = [s for s, _ in eval_states]
    i = steps.index(best_step) if best_step in steps else len(steps) - 1
    lo = max(0, i - k // 2)
    window = eval_states[lo:lo + k] if lo + k <= len(eval_states) else eval_states[max(0, len(eval_states) - k):]
    avg = {}
    for key in window[0][1]:
        avg[key] = sum(st[key].float() for _, st in window) / len(window)
        avg[key] = avg[key].to(window[0][1][key].dtype)
    return avg


def batch_tokens(X, offsets, questions, device):
    """Padded tokens of the games of a batch of questions; game_of[:, 0] is each game's question (a game may repeat across questions)."""
    keys, owner = [], []
    for i, ks in enumerate(questions):
        keys += list(ks)
        owner += [i] * len(ks)
    starts = torch.tensor([offsets[k][0] for k in keys], device=device)
    lens = torch.tensor([offsets[k][1] for k in keys], device=device)
    L = int(lens.max())
    idx = starts[:, None] + torch.arange(L, device=device)[None]
    mask = (torch.arange(L, device=device)[None] < lens[:, None])
    tokens = X[idx.clamp(max=X.shape[0] - 1)].float() * mask[:, :, None]
    return tokens, mask[:, :, None].float(), torch.tensor(owner, device=device)[:, None]


def predict(model, R, questions, X, offsets, device, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(R), batch):
            r = torch.as_tensor(R[i:i + batch], device=device)
            if model.arm == "R0":
                logits = model(r)
            else:
                tokens, mask, game_of = batch_tokens(X, offsets, questions[i:i + batch], device)
                logits = model(r, tokens, mask, game_of)
            out.append(torch.softmax(logits, dim=1).double().cpu().numpy())
    return np.concatenate(out)


def mixed_questions(train_q, data, train_folds, v, mix, inner_frac):
    """Section 4L.4: a seeded share `mix` of the training questions become same-rank mixed-player questions -- n distinct other players
    of the same rank (training folds only, never the inner held-out players train() will hold out), one random game each. player_id,
    rank and n_games stay the originating player's (train() still drops held-out players' questions and never scores a mixed question
    on the inner set); game_keys change and a `mixed` column marks the rows. A question whose rank has too few other players is left as is."""
    rng = np.random.RandomState(4242 + fold_code(v))
    players = np.array(sorted(set(train_q["player_id"].to_numpy(dtype=str))))
    held = set(np.random.RandomState(12345 + fold_code(v)).permutation(players)[:int(inner_frac * len(players))])  # train()'s rule
    pool = {}
    for f in train_folds:
        for r in data[f]["players"].itertuples():
            if str(r.player_id) not in held:
                pool.setdefault(r.rank, []).append((str(r.player_id), list(r.game_keys)))
    q = train_q.reset_index(drop=True).copy()
    keys_col = [list(k) for k in q["game_keys"].tolist()]
    mixed = np.zeros(len(q), dtype=bool)
    pick = np.flatnonzero(rng.rand(len(q)) < mix)
    pids, ranks = q["player_id"].to_numpy(dtype=str), q["rank"].to_numpy(dtype=str)
    for i in pick:
        if pids[i] in held:  # a held-out player's questions stay as they are: they are the inner set, never training data
            continue
        cands = [c for c in pool.get(ranks[i], []) if c[0] != pids[i]]
        n = len(keys_col[i])
        if len(cands) < n:
            continue
        chosen = rng.choice(len(cands), n, replace=False)
        keys_col[i] = [cands[j][1][rng.randint(len(cands[j][1]))] for j in chosen]
        mixed[i] = True
    q["game_keys"] = keys_col
    q["mixed"] = mixed
    return q


def extra_tables(f, base, folds_df, hd_all, dirs=None, level=40):
    """Section 4M M8 / x80: fold f's training tables with the players' games 21-40 (and 41-80 at level 80) appended to the 20-game
    sample; the players' game_keys come back in fold_players' order over up to `level` games. ValueError when a game is scored
    twice or an extra directory is incomplete (players_from_scores checks every player's count against the selection)."""
    dirs = dirs or L1B_SETS[3]
    hsl, l1b, opp, ohsl = base["hsl"], base["l1b"], base["opp"], base.get("opp_hsl")
    for k in extra_sets(level):
        l1b_map, opp_map = set_dirs_for(dirs, k)
        hsl_x, l1b_x = l1b_cv.load_fold(EXTRAS[k]["hsl"], ROOT / "results/l1b" / l1b_map[f], f)
        opp_x = read_scored_games(ROOT / "results/l1b" / opp_map[f] / f"fold{f}")
        hsl, l1b, opp = (pd.concat([t, x], ignore_index=True) for t, x in ((hsl, hsl_x), (l1b, l1b_x), (opp, opp_x)))
        if ohsl is not None:
            ohsl = pd.concat([ohsl, opp_hsl_table(opp_hsl_dir(k), f, hsl_x)], ignore_index=True)
    for name, t in (("hsl", hsl), ("l1b", l1b), ("opp", opp)):
        if not t["game_key"].is_unique:
            raise ValueError(f"fold {f} {name}: {int(t['game_key'].duplicated().sum())} games appear in both the base and the extra directory")
    if set(hsl["game_key"]) != set(opp["game_key"]):
        raise ValueError(f"fold {f}: the extra target and opponent directories score different games")
    players = l1b_cv.players_from_scores(hsl, folds_df, max_games=level)
    return {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], "players": players,
            **({"opp_hsl": ohsl} if ohsl is not None else {})}


def r_features(a, v, pool=0):
    """The 64 R features of the training and validation questions of validation fold v, cached, in the protocol of scripts/hd_check.py.
    v = "all" (the submission's models): the training questions of all four folds and no validation questions (empty arrays).
    a.mix > 0 (section 4L.4): that share of the training questions are same-rank mixed-player questions (mixed_questions), tag _mix<pct>."""
    mix = float(getattr(a, "mix", 0.0) or 0.0)
    games, sizes = int(getattr(a, "train_games", 0) or 0), getattr(a, "train_sizes", None)
    extra = extra_level(a)
    train_folds = training_folds(v, int(getattr(a, "train_folds", 0) or 0), bool(getattr(a, "with_fold0", False)))
    tag = ("r_all" if v == "all" else f"r_val{v}") + (f"_p{pool}" if pool else "") + "".join("_" + x for x in data_segments(a) if x != "oh")  # pool k > 0: another draw of the 32 training questions per player; -oh changes the tokens only
    path = Path(a.cache) / f"{tag}.npz"
    meta_path = Path(a.cache) / f"{tag}.json"
    if path.exists() and meta_path.exists():
        z = np.load(path)
        m = json.loads(meta_path.read_text())
        tr = pd.DataFrame(m["train"])
        va = pd.DataFrame(m["val"])
        return z["X_tr"], z["y_tr"], z["X_va"], z["y_va"], tr, va
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    data = {}
    dirs = L1B_SETS[l1b_set_of(a)]
    for f in sorted(set(train_folds) | ({int(v)} if v != "all" else set())):
        hsl, l1b = l1b_cv.load_fold(HSL, ROOT / "results/l1b" / dirs["l1b"][f], f)
        opp = read_scored_games(ROOT / "results/l1b" / dirs["opp"][f] / f"fold{f}")
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}
        if opp_hsl_level(a) == 2:  # section 4M M14: the opponent's moves under the 18 profiles, the games of the HSL table exactly
            data[f]["opp_hsl"] = opp_hsl_table(OPP_HSL, f, hsl)
        data[f]["train"] = extra_tables(f, data[f], folds_df, hd_all, dirs, extra) if (extra and f in train_folds) else {k: data[f][k] for k in ("hsl", "l1b", "opp", "hd", "players", "opp_hsl") if k in data[f]}
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + fold_code(v) + 1000 * pool)
    size_range = range(int(sizes[0]), int(sizes[1]) + 1) if sizes else range(5, 21)
    train_q = pd.concat([sample_questions(cap_games(data[f]["train"]["players"], games), 32, rng, sizes=size_range) for f in train_folds], ignore_index=True)
    if extra:
        print(f"extra games: training players draw from up to {extra} games ("
              + ", ".join(f"fold {f}: {sum(len(k) for k in data[f]['train']['players']['game_keys'])} games" for f in train_folds) + ")", flush=True)
    if mix > 0:
        train_q = mixed_questions(train_q, data, train_folds, v, mix, float(getattr(a, "inner_frac", 0.1)))
        print(f"mixed-player questions: {int(train_q['mixed'].sum())} of {len(train_q)} (share {mix})", flush=True)
    cat = lambda key: pd.concat([data[f]["train"][key] for f in train_folds], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
    Xt = X_tr["fused_opp_diff_hd"]
    if opp_hsl_level(a) == 2:  # section 4M M14: + the 39 opponent-HSL question features (64 -> 103)
        Xt = np.hstack([Xt, opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q)])
    cols = ["player_id", "rank", "n_games", "game_keys"]
    tcols = cols + (["mixed"] if "mixed" in train_q.columns else [])
    if v == "all":
        Xv, y_va, val_q = np.zeros((0, Xt.shape[1]), dtype=Xt.dtype), np.zeros(0, dtype=y_tr.dtype), pd.DataFrame(columns=cols)
    else:
        d = data[v]
        X_va, y_va, _, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        Xv, val_q = X_va["fused_opp_diff_hd"], d["val"]
        if opp_hsl_level(a) == 2:
            Xv = np.hstack([Xv, opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])])
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    np.savez(path, X_tr=Xt, y_tr=y_tr, X_va=Xv, y_va=y_va)
    meta_path.write_text(json.dumps({"train": train_q[tcols].to_dict("list"), "val": val_q[cols].to_dict("list")}))
    return Xt, y_tr, Xv, y_va, train_q[tcols], val_q[cols]


def train(a):
    sd = str(getattr(a, "save_dir", "") or "")
    if sd and (bool(HSL_TAG) != (f"_{HSL_TAG}" in sd if HSL_TAG else "_jp" in sd)):  # section 4M M27: jp models only into *_jp, never into the ct7 directories
        raise SystemExit(f"--save-dir {sd} does not match AICUP_HSL={HSL_TAG!r}")
    if a.val_fold == 0:
        raise SystemExit("fold 0 is not used here")
    is_all = a.val_fold == "all"  # the submission's models: every fold's players train, the inner hold-out alone picks the stopping point
    train_folds = training_folds(a.val_fold, a.train_folds, a.with_fold0)
    device = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    X_tr, y_tr, X_va, y_va, q_tr, q_va = r_features(a, a.val_fold)
    if a.r_extra:  # section 4M M36: + 20 question columns from the cross-fitted attention heads
        from aicup.attnfeat import load_scores, question_features
        sc = load_scores(a.r_extra, a.r_extra_games)
        X_tr = np.hstack([X_tr, question_features(q_tr["game_keys"], sc)]).astype(X_tr.dtype)
        X_va = np.hstack([X_va, question_features(q_va["game_keys"], sc)]).astype(X_va.dtype) if len(q_va) else np.zeros((0, X_tr.shape[1]), X_tr.dtype)
        print(f"r-extra: {len(sc)} scored games, R {X_tr.shape[1]} columns", flush=True)
    y_all = y_tr.copy()
    held = set()
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    std[std == 0] = 1.0
    R_tr, R_va = ((X_tr - mean) / std).astype(np.float32), ((X_va - mean) / std).astype(np.float32)
    keys_tr, keys_va = q_tr["game_keys"].tolist(), q_va["game_keys"].tolist()
    R_in, y_in, keys_in, q_in = None, None, None, None
    if a.inner_frac > 0:  # the same held-out players for every arm and seed of this validation fold
        pid_tr = q_tr["player_id"].to_numpy(dtype=str)
        players = np.array(sorted(set(pid_tr)))
        held = set(np.random.RandomState(12345 + fold_code(a.val_fold)).permutation(players)[:int(a.inner_frac * len(players))])
        is_in = np.array([x in held for x in pid_tr])
        mixed = q_tr["mixed"].to_numpy(dtype=bool) if "mixed" in q_tr.columns else np.zeros(len(q_tr), dtype=bool)  # a mixed question holds training games
        pick = np.random.RandomState(fold_code(a.val_fold)).permutation(np.flatnonzero(is_in & ~mixed))[:a.inner_questions]
        R_in, y_in, keys_in, q_in = R_tr[pick], y_tr[pick], [keys_tr[i] for i in pick], q_tr.iloc[pick]
        keep = np.flatnonzero(~is_in)
        R_tr, y_tr, keys_tr = R_tr[keep], y_tr[keep], [keys_tr[i] for i in keep]
        print(f"inner: {len(held)} of {len(players)} training players held out, {len(pick)} questions score the stopping point", flush=True)
    bagged = None
    if a.bag < 1:  # bagging over players: this model sees a seeded share of the training players
        cand = np.array(sorted(set(q_tr["player_id"].astype(str)) - held))
        bagged = set(np.random.RandomState(777 + a.seed + 100 * fold_code(a.val_fold)).permutation(cand)[:int(a.bag * len(cand))])

    pool_pids = []  # section 4M M33: the player of every question of each draw, in the draw's order

    def prep(Xp, yp, qp):
        """One draw of training questions without the held-out players (and, with --bag, only the bagged ones), standardised as pool 0;
        weights are class-balanced, times each question's mean game uniqueness (1 / the number of questions using the game) with --uniqueness."""
        pid = qp["player_id"].to_numpy(dtype=str)
        ok = np.array([(x not in held) and (bagged is None or x in bagged) for x in pid])
        Rp = ((Xp[ok] - mean) / std).astype(np.float32)
        kp = [k for k, o in zip(qp["game_keys"].tolist(), ok) if o]
        pool_pids.append(pid[ok])
        w = class_weights(yp[ok])
        if a.uniqueness:
            count = {}
            for ks in kp:
                for g in ks:
                    count[g] = count.get(g, 0) + 1
            u = np.array([np.mean([1.0 / count[g] for g in ks]) for ks in kp])
            w = w * u / u.mean()
            w = w / w.mean()
        return Rp, yp[ok], kp, w

    pools = [prep(X_tr, y_all, q_tr)]
    for k in range(1, a.resample):  # --resample K: epoch e trains on draw e mod K
        Xk, yk, _, _, qk, _ = r_features(a, a.val_fold, k)
        pools.append(prep(Xk, yk, qk))
    soft0 = None
    if a.pseudo:  # CV transduction: the validation questions join the first draw with soft targets (no labels), weight --pseudo-weight
        if is_all:
            raise SystemExit("--pseudo is the CV simulation (validation questions as unlabelled data); not for --val-fold all")
        pz = np.load(a.pseudo)
        Pp = np.asarray(pz["proba"], dtype=np.float32)
        if Pp.shape != (len(R_va), 10):
            raise SystemExit(f"--pseudo proba {Pp.shape} does not match the {len(R_va)} validation questions")
        Rp, yp, kp, wp = pools[0]
        soft0 = np.vstack([np.eye(10, dtype=np.float32)[yp], Pp])
        pools[0] = (np.vstack([Rp, R_va]), np.concatenate([yp, Pp.argmax(axis=1)]), kp + list(keys_va), np.concatenate([wp, np.full(len(R_va), a.pseudo_weight, dtype=wp.dtype)]))
        print(f"pseudo: {len(R_va)} validation questions added with soft targets (weight {a.pseudo_weight})", flush=True)
    if a.limit:  # smoke runs only
        keep = np.random.RandomState(0).permutation(len(pools[0][1]))[:a.limit]
        pools[0] = (pools[0][0][keep], pools[0][1][keep], [pools[0][2][i] for i in keep], pools[0][3][keep])
        pool_pids[0] = pool_pids[0][keep]
    R_tr, y_tr, keys_tr, _ = pools[0]
    print(f"training draws: {[len(x[1]) for x in pools]} questions" + (f"; {len(bagged)} bagged players" if bagged else ""), flush=True)
    X, offsets, Xv, offsets_v, stats = None, None, None, None, None
    if a.arm != "R0":
        X, offsets, stats = load_tokens(train_folds, a.cache, device, drop=a.drop, vec_dir=a.vec_dir, extra=extra_level(a), l1b_set=l1b_set_of(a), opp_hsl=opp_hsl_level(a) > 0)
        if not is_all:
            Xv, offsets_v, _ = load_tokens([a.val_fold], a.cache, device, stats, drop=a.drop, vec_dir=a.vec_dir, l1b_set=l1b_set_of(a), opp_hsl=opp_hsl_level(a) > 0)
        print(f"tokens loaded: {X.shape[0]} + {0 if is_all else Xv.shape[0]} plies ({time.time() - t0:.0f} s)", flush=True)
    if a.arm == "S":  # section 4M M21b: configurable width / depth
        HIDDEN["S"] = int(a.s_dim); globals()["S_LAYERS"] = int(a.s_layers)
    model = Model(a.arm, R_tr.shape[1], a.dropout, pool=a.pooling, token_enc=a.token_enc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    blocks = sum(p.numel() for p in model.blocks.parameters()) if a.arm != "R0" else 0
    pool_t = [(torch.as_tensor(Rp, device=device), torch.as_tensor(yp, device=device).long(), torch.as_tensor(wp, device=device, dtype=torch.float32), kp,
               torch.as_tensor(kernel_targets(yp), device=device, dtype=torch.float32) if (soft0 is None or j > 0) else torch.as_tensor(soft0, device=device))
              for j, (Rp, yp, kp, wp) in enumerate(pools)]
    if a.pseudo and a.arm != "R0":  # the pseudo questions' tokens live in the validation fold's tensor: merge the two tables
        shift = X.shape[0]
        offsets = {**offsets, **{k: (s_ + shift, n_) for k, (s_, n_) in offsets_v.items()}}
        X = torch.cat([X, Xv])
    pc_head = torch.nn.Linear(Q_DIM, 32).to(device) if a.player_contrast > 0 else None  # M33: training-only projection head
    train_params = list(model.parameters()) + (list(pc_head.parameters()) if pc_head is not None else [])
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(R_tr) // a.batch)
    if a.warmup_steps > 0:  # section 4M M21b: linear warmup then cosine
        wu = int(a.warmup_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + math.cos(math.pi * min(1.0, (t - wu) / max(1, steps - wu)))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.RandomState(a.seed)
    history, step, eval_states = [], 0, []
    run_aux, n_aux = 0.0, 0  # M30 clip: the running InfoNCE, for the log
    every = max(1, (len(R_tr) // a.batch) // a.evals_per_epoch)
    best, best_state, best_step, since = -1.0, None, 0, 0
    stop = False

    def logits_of(R_t, keys_tr, b):
        bt = torch.as_tensor(b, device=device)
        if a.arm == "R0":
            return model(R_t[bt]), bt
        tokens, mask, game_of = batch_tokens(X, offsets, [keys_tr[j] for j in b], device)
        return model(R_t[bt], tokens, mask, game_of), bt

    for epoch in range(a.epochs):
        if stop:
            break
        model.train()
        R_t, y_t, weight, keys_tr, targets = pool_t[epoch % len(pool_t)]
        paired = pc_head is not None or a.pair_batches  # M33 (-pb: the pairing alone, as a diagnostic)
        perm = paired_perm(pool_pids[epoch % len(pool_t)], rng) if paired else rng.permutation(len(keys_tr))
        run, n_run, t_ep = 0.0, 0, time.time()
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            logits, bt = logits_of(R_t, keys_tr, perm[i:i + a.batch])
            if a.loss == "kernel" or (a.pseudo and epoch % len(pool_t) == 0):  # soft targets (one-hot for real questions, proba for pseudo)
                ce = kernel_ce_torch(logits, targets[bt])
            else:
                ce = torch.nn.functional.cross_entropy(logits, y_t[bt], reduction="none")
            loss = (ce * weight[bt]).sum() / weight[bt].sum()
            if model.aux_loss is not None:  # M30 clip: + clip_weight x InfoNCE
                run_aux, n_aux = run_aux + model.aux_loss.item(), n_aux + 1
                loss = loss + a.clip_weight * model.aux_loss
            if pc_head is not None:  # M33: + player_contrast x InfoNCE over the batch's same-player pairs
                nce = player_contrast_loss(model.q_embed, pc_head)
                run_aux, n_aux = run_aux + nce.item(), n_aux + 1
                loss = loss + a.player_contrast * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)
            opt.step()
            sched.step()
            run, n_run, step = run + loss.item(), n_run + 1, step + 1
            if a.inner_frac > 0 and step % every == 0:
                Pi = predict(model, R_in, keys_in, X, offsets, device)
                _, si = evaluate(Pi, y_in, q_in, sizes=tuple(range(5, 21)))
                inner = float(np.mean([si[y_in == c].mean() for c in np.unique(y_in)]))
                history.append({"step": step, "epoch": step / (len(R_tr) // a.batch), "train_loss": run / max(n_run, 1), "inner": inner, **({"clip": run_aux / n_aux} if n_aux else {})})
                print(f"{a.arm} seed {a.seed} step {step}/{steps} (epoch {history[-1]['epoch']:.2f}): loss {run / max(n_run, 1):.4f}" + (f" (InfoNCE {run_aux / n_aux:.3f})" if n_aux else "")
                      + f"  inner {inner:.4f}  {time.time() - t0:.0f} s", flush=True)
                run, n_run, run_aux, n_aux = 0.0, 0, 0.0, 0
                if a.swa:
                    eval_states.append((step, {k: v.detach().clone() for k, v in model.state_dict().items()}))
                if inner > best:
                    best, best_step, since = inner, step, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    since += 1
                    if since >= a.patience:
                        stop = True
                        break
                model.train()
        if a.inner_frac <= 0:
            history.append({"epoch": epoch + 1, "train_loss": run / max(n_run, 1)})
            print(f"{a.arm} seed {a.seed} epoch {epoch + 1}/{a.epochs}: loss {run / max(n_run, 1):.4f}  {time.time() - t_ep:.0f} s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if a.swa and eval_states:  # the window of a.swa evaluation states centred on the best one, averaged
        model.load_state_dict(swa_average(eval_states, best_step, a.swa))
        print(f"swa: averaged {min(a.swa, len(eval_states))} evaluation states around step {best_step}", flush=True)
    ft_history, ft_best_step = [], None
    if a.loss == "es-ft":
        # section 4F's second stage: from the best CE state, the negative expected official score at a small constant lr; the inner
        # official score decides which state is kept, the CE one included
        if a.inner_frac <= 0:
            raise SystemExit("--loss es-ft stops on held-out players: give --inner-frac")
        if best_state is None:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        S_t = torch.as_tensor(score_matrix(), device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=a.ft_lr, weight_decay=a.wd)
        R_t, y_t, weight, keys_tr, _ = pool_t[0]
        since, stop = 0, False
        for epoch in range(a.ft_epochs):
            if stop:
                break
            model.train()
            perm = rng.permutation(len(keys_tr))
            run, n_run = 0.0, 0
            for i in range(0, len(perm) - a.batch + 1, a.batch):
                logits, bt = logits_of(R_t, keys_tr, perm[i:i + a.batch])
                es = expected_score_loss_torch(logits, y_t[bt], S_t)
                loss = (es * weight[bt]).sum() / weight[bt].sum()
                if model.aux_loss is not None:
                    loss = loss + a.clip_weight * model.aux_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                run, n_run, step = run + loss.item(), n_run + 1, step + 1
                if step % every == 0:
                    Pi = predict(model, R_in, keys_in, X, offsets, device)
                    _, si = evaluate(Pi, y_in, q_in, sizes=tuple(range(5, 21)))
                    inner = float(np.mean([si[y_in == c].mean() for c in np.unique(y_in)]))
                    ft_history.append({"step": step, "train_loss": run / max(n_run, 1), "inner": inner})
                    print(f"{a.arm} seed {a.seed} ES step {step}: loss {run / max(n_run, 1):.4f}  inner {inner:.4f} (best {best:.4f})  {time.time() - t0:.0f} s", flush=True)
                    run, n_run = 0.0, 0
                    if inner > best:
                        best, best_step, ft_best_step, since = inner, step, step, 0
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        since += 1
                        if since >= a.ft_patience:
                            stop = True
                            break
                    model.train()
        model.load_state_dict(best_state)
    tag = (a.arm + (f"-d{int(a.s_dim)}l{int(a.s_layers)}" if a.arm == "S" and (int(a.s_dim), int(a.s_layers)) != (192, 4) else "") + ("-wu" if a.warmup_steps > 0 else "") + (f"-no{a.drop}" if a.drop else "") + ("-rs" if a.resample > 1 else "") + ("-uq" if a.uniqueness else "") + ("-bg" if a.bag < 1 else "")
           + {"kernel": "-k", "es-ft": "-es"}.get(a.loss, "") + ("-pma" if a.pooling == "pma" else "") + ("" if a.token_enc == "linear" else f"-{a.token_enc}")
           + ("" if a.player_contrast <= 0 else ("-pc" if a.player_contrast == 0.1 else f"-pc{a.player_contrast:g}")) + ("-pb" if a.pair_batches and a.player_contrast <= 0 else "")
           + (f"w{a.clip_weight:g}" if a.token_enc == "clip" and a.clip_weight != 0.1 else "") + (f"-swa{a.swa}" if a.swa else "") + ("-pl" if a.pseudo else "")
           + "".join("-" + x for x in data_segments(a)) + vec_tag(a.vec_dir) + ("-ra" if a.r_extra else "") + (f"-{a.tag_extra}" if a.tag_extra else ""))
    out = Path(a.out) / ("all" if is_all else f"val{a.val_fold}")
    out.mkdir(parents=True, exist_ok=True)
    final = {"final": None, "final_nll": None}
    if is_all:
        print(f"{a.arm} seed {a.seed}: stopped at step {best_step or step} (inner {best:.4f}); trained on folds {train_folds}, no validation fold", flush=True)
    else:
        P = predict(model, R_va, keys_va, Xv, offsets_v, device)
        res, scores = evaluate(P, y_va, q_va)
        print(f"{a.arm} seed {a.seed}: stopped at step {best_step or step} (inner {best:.4f}); validation {res['balanced_mean_over_n']:.4f}  NLL {res['nll_balanced']:.4f}", flush=True)
        np.savez(out / f"{tag}_s{a.seed}.npz", proba=P.astype(np.float32), scores=scores)
        final = {"final": res["balanced_mean_over_n"], "final_nll": res["nll_balanced"]}
    (out / f"{tag}_s{a.seed}.json").write_text(json.dumps({"arm": tag, "drop": a.drop, "resample": a.resample, "uniqueness": a.uniqueness, "bag": a.bag, "mix": a.mix, "loss": a.loss, "pooling": a.pooling, "token_enc": a.token_enc, "clip_weight": a.clip_weight, "player_contrast": a.player_contrast, "pair_batches": a.pair_batches, "tag_extra": a.tag_extra, "ft_lr": a.ft_lr, "ft_epochs": a.ft_epochs, "ft_history": ft_history, "ft_best_step": ft_best_step, "seed": a.seed, "val_fold": a.val_fold, "n_params": n_params, "block_params": blocks,
                                                             "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "wd": a.wd, "dropout": a.dropout, "history": history, "inner_frac": a.inner_frac,
                                                             "best_step": best_step, "best_inner": best, "steps_per_epoch": len(R_tr) // a.batch,
                                                             **final, "wall_s": time.time() - t0}, indent=1))
    if is_all or a.save_dir:  # the weights, for scripts/trunk_predict.py (the submission's models always; a validation-fold model on request)
        save_dir = Path(a.save_dir) if a.save_dir else ROOT / "results" / "models" / "ct"
        ck = save_dir / f"{'all' if is_all else f'val{a.val_fold}'}_{tag}_s{a.seed}.pt"
        side = save_checkpoint(ck, model, stats, (mean, std), {
            "tag": tag, "seed": a.seed, "val_fold": a.val_fold, "train_folds": train_folds, "drop": a.drop, "mix": a.mix, "loss": a.loss, "pooling": a.pooling, "token_enc": a.token_enc, "clip_weight": a.clip_weight, "player_contrast": a.player_contrast, "pair_batches": a.pair_batches, "tag_extra": a.tag_extra, "r_extra": a.r_extra, "epochs": a.epochs, "batch": a.batch,
            "lr": a.lr, "wd": a.wd, "dropout": a.dropout, "inner_frac": a.inner_frac, "inner_questions": a.inner_questions, "held_players": len(held),
            "train_questions": int(len(R_tr)), "best_step": best_step, "best_inner": best, "steps_per_epoch": len(R_tr) // a.batch, "n_params": n_params,
            "block_params": blocks, "l1b_set": l1b_set_of(a), "l1b_heads": L1B_SETS[l1b_set_of(a)]["heads"],
            "l1b_dirs": {str(f): L1B_SETS[l1b_set_of(a)]["l1b"][f] for f in train_folds}, "opp_dirs": {str(f): L1B_SETS[l1b_set_of(a)]["opp"][f] for f in train_folds},
            "hsl_dir": str(HSL.relative_to(ROOT)), "vec_dir": str(Path(a.vec_dir).resolve().relative_to(ROOT)), "cache": str(a.cache),
            "train_folds_n": a.train_folds, "train_games": a.train_games, "train_sizes": a.train_sizes, "with_fold0": a.with_fold0,
            "extra_games": extra_level(a), "extra_dirs": ({str(k): {"hsl": str(EXTRAS[k]["hsl"]), "vec": str(EXTRAS[k]["vec"])} for k in extra_sets(extra_level(a))} or None),
            "opp_hsl": opp_hsl_level(a) > 0, "opp_hsl_r": opp_hsl_level(a) == 2, "opp_hsl_dirs": ([str(OPP_HSL.relative_to(ROOT))] + [str(opp_hsl_dir(k).relative_to(ROOT)) for k in extra_sets(extra_level(a))]) if opp_hsl_level(a) else None,
            "swa": a.swa, "pseudo": a.pseudo, "pseudo_weight": a.pseudo_weight, **final, "wall_s": time.time() - t0})
        print(f"saved {ck} (sha256 {side['state_sha256'][:12]})", flush=True)
    return 0


def compare(a):
    """Per-question scores averaged over seeds, per fold and pooled; conditions (1) CI lower bound > 0, (3) > 0 in all but one fold, (4) >= 0.005."""
    folds = a.val_folds or [a.val_fold]
    arms = a.arms
    pairs_spec = [tuple(x.split(":")) for x in a.pairs]
    sizes = tuple(a.sizes) if getattr(a, "sizes", None) else SIZES
    base = argparse.Namespace(**{**vars(a), "train_folds": 0, "train_games": 0, "train_sizes": None, "mix": 0.0, "extra_games": False, "with_fold0": False, "l1b_set": 3, "opp_hsl": False, "opp_hsl_r": False})  # the validation questions are the same in every data arm
    ys, pids, ns, sc, per_fold, seeds, stops = [], [], [], {}, {}, {}, {}
    for v in folds:
        _, _, _, y, _, q = r_features(base, v)
        d = Path(a.out) / f"val{v}"
        pid, n = q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy()
        ys.append(y); pids.append(pid); ns.append(n)
        for arm in arms:
            files = [f for f in sorted(d.glob(f"{arm}_s*.npz")) if a.seeds is None or int(f.stem.rsplit("_s", 1)[1]) in a.seeds]
            if not files:
                raise SystemExit(f"compare: no {arm}_s*.npz for validation fold {v} in {d}" + (f" with seeds {a.seeds}" if a.seeds else ""))
            s = [np.load(f)["scores"] for f in files]
            sc.setdefault(arm, []).append(np.mean(s, axis=0))
            per_fold.setdefault(arm, []).append(float(np.mean([balanced_mean_over_n(x, y, n, sizes=sizes) for x in s])))
            seeds.setdefault(arm, {})[str(v)] = [balanced_mean_over_n(x, y, n, sizes=sizes) for x in s]
            metas = [json.loads(f.with_suffix(".json").read_text()) for f in files]
            stops.setdefault(arm, {})[str(v)] = [m["best_step"] / m["steps_per_epoch"] for m in metas]
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
    S = {arm: np.concatenate(sc[arm]) for arm in arms}
    rep = {}
    for arm in arms:
        rep[arm] = {"score": float(np.mean(per_fold[arm])), "per_fold": per_fold[arm], "seeds": seeds[arm], "stop_epoch": stops[arm], "by_n": {str(m): balanced_mean_over_n(S[arm], y, n, sizes=(m,)) for m in SIZES}}
        print(f"{arm}: {rep[arm]['score']:.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[arm]) + "  by n " + " ".join(f"{m}: {rep[arm]['by_n'][str(m)]:.4f}" for m in SIZES)
              + "  stop epoch " + " ".join(f"{np.mean(v):.2f}" for v in stops[arm].values()))
    pairs, need = {}, max(1, len(folds) - 1)
    for x, z in pairs_spec:
        dd = paired_diff(S[x], S[z], y, pid, n, sizes=sizes, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= 0.005)}
        pairs[f"{x} vs {z}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{x} vs {z}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items())
              + f"  -> {'PASS' if pairs[f'{x} vs {z}']['pass'] else 'no'}")
    if sizes != SIZES:
        print(f"evaluated on questions of {sizes} games only")
    Path(a.out, f"compare_{a.tag or ''}{'_'.join(map(str, folds))}.json").write_text(json.dumps({"folds": folds, "sizes": list(sizes), "arms": rep, "pairs": pairs}, indent=1, default=float))
    return 0


def rfeat(a):
    r_features(a, a.val_fold, a.pool)
    print(f"R features of validation fold {a.val_fold}, draw {a.pool} cached", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "rfeat", "train", "compare"))
    ap.add_argument("--out", default=str(ROOT / "results/trunk_seq"))
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"), help="the score arrays and R features")
    ap.add_argument("--val-folds", type=int, nargs="+", default=None)
    ap.add_argument("--drop", choices=("trunk", "scores"), default=None, help="ablation: zero the trunk vectors or the per-move scores")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--pairs", nargs="+", default=["T:C", "C:R0", "T:R0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="compare: only these seeds")
    ap.add_argument("--tag", default=None, help="compare: prefix of the output file")
    ap.add_argument("--inner-frac", type=float, default=0.0)
    ap.add_argument("--resample", type=int, default=1, help="number of training-question draws, cycled by epoch")
    ap.add_argument("--uniqueness", action="store_true", help="weight questions by the mean uniqueness of their games")
    ap.add_argument("--bag", type=float, default=1.0, help="share of training players this model sees")
    ap.add_argument("--mix", type=float, default=0.0, help="section 4L.4: share of training questions made of same-rank games of distinct other players (tag -mix<pct>)")
    ap.add_argument("--pool", type=int, default=0, help="rfeat: which draw of training questions")
    ap.add_argument("--train-folds", type=int, default=0, help="section 4M M1: only this many of the other folds train (2: fold (v mod 4) + 1 is dropped; tag -f2)")
    ap.add_argument("--train-games", type=int, default=0, help="section 4M M2: each training player keeps at most this many scored games (tag -g<n>)")
    ap.add_argument("--train-sizes", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="section 4M M2: training question sizes (default 5 20; tag s<min>-<max>)")
    ap.add_argument("--vec-dir", default=str(VEC), help="trunk vectors (section 4M M3: results/trunk_vec_ft, tag -ft); `build --vec-dir` writes their score arrays")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="compare: evaluate only questions of these sizes (default 5 10 20)")
    ap.add_argument("--pooling", choices=POOLINGS, default="meanstd", help="section 4M M4: pma = attention pooling over plies and over games (tag -pma)")
    ap.add_argument("--token-enc", choices=TOKEN_ENCODERS, default="linear", help="section 4M M30: dual = board and score towers (tag -dual); clip = dual + CLIP InfoNCE between them (tag -clip)")
    ap.add_argument("--clip-weight", type=float, default=0.1, help="M30 clip: weight of the InfoNCE in the training loss (tag -clipw<w> when not 0.1)")
    ap.add_argument("--player-contrast", type=float, default=0.0, help="section 4M M33: weight of the same-player InfoNCE between paired questions (tag -pc)")
    ap.add_argument("--pair-batches", action="store_true", help="M33 diagnostic: same-player paired batches without the InfoNCE (tag -pb)")
    ap.add_argument("--r-extra", default="", help="section 4M M36: score parts of the attention heads; R gets their 20 question columns (tag -ra)")
    ap.add_argument("--r-extra-games", default=str(ROOT / "results/r_variants/attn_games.csv"), help="M36: the games CSV (game_key, fold)")
    ap.add_argument("--tag-extra", default="", help="appended as -X to the arm tag (a re-run beside the original files)")
    ap.add_argument("--extra-games", type=int, nargs="?", const=40, default=0, choices=(0, 40, 80), help="section 4M M8: training players also draw from their games 21-40 (40, tag -x40) or 21-80 (80, tag -x80)")
    ap.add_argument("--swa", type=int, default=0, help="average the weights of the K inner evaluations centred on the best one instead of the best alone (tag -swaK)")
    ap.add_argument("--pseudo", default=None, help="CV transduction: an npz with proba [Q_val, 10] aligned to the validation questions, added as soft-labelled training questions (tag -pl)")
    ap.add_argument("--pseudo-weight", type=float, default=1.0, help="weight of each pseudo-labelled question relative to a real one")
    ap.add_argument("--with-fold0", action="store_true", help="section 4M M8, --val-fold all only: fold 0's players join the training set (tag -f0)")
    ap.add_argument("--build-folds", type=int, nargs="+", default=None, help="build: only these folds (default 1-4; 0 for the fold-0 vectors)")
    ap.add_argument("--l1b-set", type=int, choices=sorted(L1B_SETS), default=3, help="section 4M M9: which L1b heads' log p feed the tokens and R (4 = heads4; tag -h4)")
    ap.add_argument("--opp-hsl", action="store_true", help="section 4M M14: the opponent plies' Human SL token block holds the opponent's centred log p (results/scores_opp_hd[_x40]; tag -oh)")
    ap.add_argument("--opp-hsl-r", action="store_true", help="section 4M M14: --opp-hsl, and the 39 opponent-HSL question features join R (64 -> 103; tag -ohr)")
    ap.add_argument("--s-dim", type=int, default=192, help="S arm width (section 4M M21b; tag -d<dim>l<layers> when not 192/4)")
    ap.add_argument("--s-layers", type=int, default=4, help="S arm depth")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup steps before the cosine schedule (tag -wu)")
    ap.add_argument("--inner-questions", type=int, default=4000)
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--arm", choices=("R0", "C", "T", "S"), help="S (section 4M M21): 4-layer transformer over plies, d 192, full attention")
    ap.add_argument("--val-fold", default="1", help="1-4, or 'all': train on every fold's players (the submission's models), no validation fold")
    ap.add_argument("--save-dir", default=None, help="train: write the weights (.pt + .json) here; 'all' runs always save, to results/models/ct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", choices=("ce", "kernel", "es-ft"), default="ce", help="section 4F: the score kernel's soft targets, or the expected-score fine-tune")
    ap.add_argument("--ft-epochs", type=int, default=2, help="es-ft: epochs cap of the second stage")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="es-ft: constant learning rate of the second stage")
    ap.add_argument("--ft-patience", type=int, default=8, help="es-ft: inner evaluations without improvement before stopping")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only this many training questions")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)
    if a.opp_hsl_r:
        a.opp_hsl = True
    if a.cmd == "train" and a.token_enc != "linear" and a.arm == "R0":
        ap.error("--token-enc is for the token arms (C, T, S)")
    if a.cmd == "train" and a.r_extra and a.resample > 1:
        ap.error("--r-extra takes a single training draw (no --resample)")
    if a.cmd == "train" and (a.player_contrast > 0 or a.pair_batches) and (a.arm == "R0" or a.pseudo or a.batch % 2 or a.loss != "ce"):
        ap.error("--player-contrast needs a token arm, an even --batch, --loss ce and no --pseudo")
    if a.val_fold != "all":
        try:
            a.val_fold = int(a.val_fold)
        except ValueError:
            ap.error("--val-fold takes 1-4 or 'all'")
    if a.val_fold == 0 or 0 in (a.val_folds or []):
        ap.error("fold 0 is not used here")
    if a.val_fold == "all" and a.cmd == "compare":
        ap.error("compare takes validation folds")
    if a.train_sizes and not (5 <= a.train_sizes[0] <= a.train_sizes[1] <= 20):
        ap.error("--train-sizes takes 5 <= MIN <= MAX <= 20")
    if a.train_games and a.train_games < (a.train_sizes[1] if a.train_sizes else 20):
        ap.error("--train-games must cover the largest training question size (give --train-sizes)")
    return {"build": build, "rfeat": rfeat, "train": train, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
