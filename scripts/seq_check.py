#!/usr/bin/env python
"""Folds 1-4 CV: do sequence-encoder outputs (scripts/seq_crossfit.py) beat the plain L1b sums in the fused_opp features?

Protocol as scripts/l1b_opp_check.py (pilot validation questions, sample_questions per player with RandomState(42 + v),
MLPModel(hidden=32), player bootstrap). Feature sets: fused_opp (the A/B sums of --l1b/--opp; must reproduce 0.6735) and,
per --seq NAME:DIR, fused_opp_seq_<NAME>: the same 56 features with the target's L1b sums replaced by the encoder's
base + Δ (the opponent's sums stay the A/B ones). Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4
folds, (4) >= 0.005): fused_opp_seq_<--pre> vs fused_opp (default --pre gru; several encoders can be named). Reported only:
every other encoder vs fused_opp, fused_opp_seq_{ctm_seq, transformer, mamba, mamba3} vs fused_opp_seq_gru when both are given, and mamba3 vs mamba. The "sum" encoder's set must equal
fused_opp (a check of the plumbing). Fold 0 is refused.

  python -I scripts/seq_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --seq sum:results/seq/sum gru:results/seq/gru conv:results/seq/conv ctm_seq:results/seq/ctm_seq --out results/seq/seq_check.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402


def parse_seq(spec):
    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"--seq expects NAME:DIR, got {spec!r}")
    return parts[0], parts[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="+", required=True)
    ap.add_argument("--seq", nargs="+", required=True, metavar="NAME:DIR", help="seq_crossfit.py output directories")
    ap.add_argument("--pre", nargs="+", default=["gru"], help="the encoders whose comparisons with fused_opp are pre-specified")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    try:
        seq_dirs = dict(parse_seq(s) for s in a.seq)
        if len(seq_dirs) != len(a.seq):
            raise ValueError("--seq names must be distinct")
        if not set(a.pre) <= set(seq_dirs):
            raise ValueError(f"--pre {a.pre!r} are not all among the --seq names {sorted(seq_dirs)}")
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
        where_s = {name: l1b_cv.locate_folds([d], folds) for name, d in seq_dirs.items()}
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            seqs = {}
            for name in seq_dirs:
                t = read_scored_games(Path(where_s[name][f]) / f"fold{f}")
                if set(t["game_key"]) != set(hsl["game_key"]):
                    raise ValueError(f"fold {f}: {seq_dirs[name]} covers other games than {a.hsl}")
                seqs[name] = t
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "seq": seqs, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    sets = ["fused_opp"] + [f"fused_opp_seq_{n}" for n in seq_dirs]
    rows, per_fold, nll = [], {s: [] for s in sets}, {s: [] for s in sets}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key, name=None: pd.concat([(data[f][key] if name is None else data[f][key][name]) for f in folds if f != v], ignore_index=True)  # noqa: E731
        val = data[v]["val"]
        sc = {}
        tables_tr = {"fused_opp": cat("l1b"), **{f"fused_opp_seq_{n}": cat("seq", n) for n in seq_dirs}}
        tables_va = {"fused_opp": data[v]["l1b"], **{f"fused_opp_seq_{n}": data[v]["seq"][n] for n in seq_dirs}}
        y_va = n_va = None
        for s in sets:
            X_tr, y_tr, _ = oc.features(cat("hsl"), tables_tr[s], cat("opp"), train_q)
            X_va, y_va, n_va = oc.features(data[v]["hsl"], tables_va[s], data[v]["opp"], val)
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr).predict_proba(X_va["fused_opp"]), y_va, val)
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in sets), flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in sets}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in sets}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    pre = [(f"fused_opp_seq_{n}", "fused_opp") for n in a.pre]
    report = [(f"fused_opp_seq_{n}", "fused_opp") for n in seq_dirs if n not in a.pre]
    if "gru" in seq_dirs:
        report += [(f"fused_opp_seq_{n}", "fused_opp_seq_gru") for n in ("ctm_seq", "transformer", "mamba", "mamba3") if n in seq_dirs]
    if "mamba" in seq_dirs and "mamba3" in seq_dirs:
        report.append(("fused_opp_seq_mamba3", "fused_opp_seq_mamba"))
    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in pre}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in report}}
    for s, h in head.items():
        print(f"{s:22s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp,
           "seq_dirs": seq_dirs, "pre": a.pre[0] if len(a.pre) == 1 else a.pre, "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
