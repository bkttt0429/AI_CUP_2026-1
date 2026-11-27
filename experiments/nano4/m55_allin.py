"""Section 4M M55 (A): every stored pc-family member stacked, against ct9 (pc seeds 0-2). Report only; out-of-fold probabilities of
results/trunk_seq/es/val{v}/<tag>_s<seed>.npz, validation folds 1-4. For each candidate: the class-balanced paired difference with the
player bootstrap (aicup.qeval.paired_diff), per-fold differences, the test-like plain difference (plain within n, mean over n), on all
questions and on the HD-free subset (no HD >= 2 game, the test-like protocol); the new rule (CI lower > 0, >= 3/4 folds > 0, >= 0.0025).
  E1  pc seeds 0-29
  E2  all members, equal weight per member;   E3  family-equal (mean within family, then over families)
  E2A pc 0-14 + 6-seed families 0-2 + 3-seed families   vs pc 0-2  (= ct9)
  E2B pc 15-29 + 6-seed families 3-5                    vs pc 3-5  (fresh-seed replication)
  E4  E2 + the S-pc members (when --s-tag is given and present)
  python -I logs/nano4/m55_allin.py [--s-tag S-pc-d128l3-wu-f0-x40-oh]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m47_prior import boot_players, nplain  # noqa: E402

PC = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh")
SIX = {"h4": ("C-pc-f0-h4-x40-oh", "T-pc-f0-h4-x40-oh"), "ohr": ("C-pc-f0-x40-ohr", "T-pc-f0-x40-ohr"),
       "uq": ("C-uq-pc-f0-x40-oh", "T-uq-pc-f0-x40-oh"), "jp": ("C-pc-f0-x40-oh-jp", "T-pc-f0-x40-oh-jp")}
THREE = {"rh": ("C-pc-f0-x40-oh-rh", "T-pc-f0-x40-oh-rh"), "ra": ("C-pc-f0-x40-oh-ra", "T-pc-f0-x40-oh-ra"),
         **{k: (f"C-{k}-f0-x40-oh", f"T-{k}-f0-x40-oh") for k in ("pc0.05", "pc0.2", "pc0.3", "pct0.05", "pct0.2", "pcv4")}}


def load_member(v, tag, s, cache):
    key = (v, tag, s)
    if key not in cache:
        cache[key] = np.load(ES / f"val{v}/{tag}_s{s}.npz")["proba"].astype(np.float64)
    return cache[key]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--s-tag", default=None); ap.add_argument("--s-seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    folds, cache = {}, {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        folds[v] = {"y": y, "n": n, "pid": pid, "hdfree": ~(np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)}

    def fam(tags, seeds):
        return [(t, s) for t in tags for s in seeds]

    specs = {
        "ct9": {"pc": fam(PC, range(3))},
        "pc3-5": {"pc": fam(PC, range(3, 6))},
        "E1": {"pc": fam(PC, range(30))},
        "E2": {"pc": fam(PC, range(30)), **{k: fam(t, range(6)) for k, t in SIX.items()}, **{k: fam(t, range(3)) for k, t in THREE.items()}},
        "E2A": {"pc": fam(PC, range(15)), **{k: fam(t, range(3)) for k, t in SIX.items()}, **{k: fam(t, range(3)) for k, t in THREE.items()}},
        "E2B": {"pc": fam(PC, range(15, 30)), **{k: fam(t, range(3, 6)) for k, t in SIX.items()}},
    }
    if a.s_tag:
        have = all((ES / f"val{v}/{a.s_tag}_s{s}.npz").exists() for v in FOLDS for s in a.s_seeds)
        print(f"S members {a.s_tag} seeds {a.s_seeds}: {'present' if have else 'MISSING'}", flush=True)
        if have:
            specs["E4"] = {**specs["E2"], "S": [(a.s_tag, s) for s in a.s_seeds]}
            specs["S-only"] = {"S": [(a.s_tag, s) for s in a.s_seeds]}
            specs["ct9+S"] = {"pc": fam(PC, range(3)), "S": [(a.s_tag, s) for s in a.s_seeds]}

    def probs(v, spec, family_equal=False):
        if family_equal:
            return np.mean([np.mean([load_member(v, t, s, cache) for t, s in mem], axis=0) for mem in spec.values()], axis=0)
        return np.mean([load_member(v, t, s, cache) for mem in spec.values() for t, s in mem], axis=0)

    S = {}
    for name, spec in specs.items():
        S[name] = {v: qscore(decide(probs(v, spec)), folds[v]["y"]) for v in FOLDS}
        if name == "E2":
            S["E3"] = {v: qscore(decide(probs(v, spec, family_equal=True)), folds[v]["y"]) for v in FOLDS}
    n_members = {k: sum(len(m) for m in sp.values()) for k, sp in specs.items()}
    print("members: " + ", ".join(f"{k} {n}" for k, n in n_members.items()), flush=True)
    rep = {"members": n_members, "comparisons": {}}
    comps = [("E1", "ct9"), ("E2", "ct9"), ("E3", "ct9"), ("E2A", "ct9"), ("E2B", "pc3-5")] + ([("E4", "ct9"), ("E4", "E2"), ("ct9+S", "ct9"), ("S-only", "ct9")] if "E4" in S else [])
    for new, old in comps:
        for subset in ("all", "hdfree"):
            k = {v: (np.ones(len(folds[v]["y"]), bool) if subset == "all" else folds[v]["hdfree"]) for v in FOLDS}
            cat = lambda key: np.concatenate([folds[v][key][k[v]] for v in FOLDS])  # noqa: E731
            y, n, pid = cat("y"), cat("n"), cat("pid")
            sn = np.concatenate([S[new][v][k[v]] for v in FOLDS]); so = np.concatenate([S[old][v][k[v]] for v in FOLDS])
            dd = paired_diff(sn, so, y, pid, n, n=3000)
            pf = [balanced_mean_over_n(S[new][v][k[v]], folds[v]["y"][k[v]], folds[v]["n"][k[v]]) - balanced_mean_over_n(S[old][v][k[v]], folds[v]["y"][k[v]], folds[v]["n"][k[v]]) for v in FOLDS]
            plain = nplain(sn, n) - nplain(so, n)
            a_c = [float(np.mean([(sn - so)[(y == c) & (n == m)].mean() for m in (5, 10, 20)])) for c in range(10)]
            ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
            rep["comparisons"][f"{new} vs {old} [{subset}]"] = {**dd, "per_fold": pf, "plain": plain, "a_by_class": dict(zip(LEVELS, a_c)), "pass_rule": bool(ok),
                                                                  "score_new": float(balanced_mean_over_n(sn, y, n)), "score_old": float(balanced_mean_over_n(so, y, n))}
            print(f"{new:6s} vs {old:5s} [{subset:6s}]: {rep['comparisons'][f'{new} vs {old} [{subset}]']['score_new']:.4f} vs {rep['comparisons'][f'{new} vs {old} [{subset}]']['score_old']:.4f}  "
                  f"{dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  plain {plain:+.4f}"
                  + ("  -> PASS" if ok else "  -> no") + "   a_c " + " ".join(f"{lv} {x:+.3f}" for lv, x in zip(LEVELS, a_c)), flush=True)
    Path(ROOT / "results/trunk_seq/es/m55_allin.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
