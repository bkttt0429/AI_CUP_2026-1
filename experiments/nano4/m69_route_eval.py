"""Section 4M M69 (2026-11-10): routing by question size on the current recipe. Short-question members (C/T-pc-f0-s5-10-x40-oh)
vote together with the standard members (C/T-pc-f0-x40-oh) on questions of n <= 10; n = 20 is the standard members alone.
Primary, pre-registered (design doc M69): on the n in {5, 10} validation questions,
  discovery   : [standard seeds 0-2 + short seeds 0-2] (12) vs [standard seeds 0-5] (12)
  replication : [standard seeds 6-8 + short seeds 3-5] (12) vs [standard seeds 6-11] (12)
metric = test-weighted gain 0.25 d5 + 0.25 d10 = half of the balanced mean over n in (5, 10); paired player bootstrap; conditions
(1) CI lower bound > 0, (2) >= 3 of 4 folds > 0, (3) gain >= 0.0025; adopt only if discovery AND replication pass all three.
Report only: pure routing (6 short alone on n <= 10) vs standard 6; pooled 24 v 24; by n; the short members on n = 20.
  python -I logs/nano4/m69_route_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

STD, SHORT = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"), ("C-pc-f0-s5-10-x40-oh", "T-pc-f0-s5-10-x40-oh")
SMALL = (5, 10)


def mean_p(v, arms, seeds):
    return np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds], axis=0)


def check_members(seeds):
    bad = []
    for v in FOLDS:
        ref = np.load(ES / f"val{v}/{STD[0]}_s0.npz")["proba"].shape
        for arm in SHORT:
            for s in seeds:
                f = ES / f"val{v}/{arm}_s{s}"
                if not f.with_suffix(".npz").exists():
                    bad.append(f"missing {f.name}"); continue
                side = json.load(open(f.with_suffix(".json")))
                if np.load(f.with_suffix(".npz"))["proba"].shape != ref or side.get("arm") != arm or side.get("seed") != s or side.get("val_fold") != v or abs(float(side.get("player_contrast", 0)) - 0.1) > 1e-9:
                    bad.append(f"protocol {f.name}: {dict((k, side.get(k)) for k in ('arm', 'seed', 'val_fold', 'player_contrast'))}")
    return bad


def run(name, new_small, ctl_small, base_seeds, out):
    """new_small / ctl_small: lists of (arms, seeds) averaged member-wise on n <= 10; n = 20 uses the standard base_seeds in both."""
    S_new, S_ctl, pf_new, pf_ctl, ys, ns, pids = [], [], [], [], [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        base = mean_p(v, STD, base_seeds)
        def mix(parts):
            Ps = [np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arms, seeds in parts for arm in arms for s in seeds]
            return np.where((n <= 10)[:, None], np.mean(Ps, axis=0), base)
        sn, sc = evaluate(mix(new_small), y, q, sizes=SIZES)[1], evaluate(mix(ctl_small), y, q, sizes=SIZES)[1]
        S_new.append(sn); S_ctl.append(sc); ys.append(y); ns.append(n); pids.append(pid)
        pf_new.append(balanced_mean_over_n(sn, y, n, sizes=SMALL)); pf_ctl.append(balanced_mean_over_n(sc, y, n, sizes=SMALL))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); sn, sc = np.concatenate(S_new), np.concatenate(S_ctl)
    d = paired_diff(sn, sc, y, pid, n, sizes=SMALL, n=5000)
    gain, ci = 0.5 * d["diff"], [0.5 * d["ci95"][0], 0.5 * d["ci95"][1]]; pf = [0.5 * (a - b) for a, b in zip(pf_new, pf_ctl)]
    cond = {"1_ci_lower_gt_0": bool(ci[0] > 0), "2_folds_3of4": bool(sum(e > 0 for e in pf) >= 3), "3_gain_ge_0.0025": bool(gain >= 0.0025)}
    by_n = {str(m): float(balanced_mean_over_n(sn, y, n, sizes=(m,)) - balanced_mean_over_n(sc, y, n, sizes=(m,))) for m in SMALL}
    out[name] = {"test_weighted_gain": gain, "ci95": ci, "per_fold": pf, "by_n_diff": by_n, "new_small_score": float(np.mean(pf_new)), "ctl_small_score": float(np.mean(pf_ctl)),
                 "conditions": cond, "pass": all(cond.values())}
    print(f"{name:28s} test-weighted gain {gain:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  d5 %+.4f d10 %+.4f" % (by_n["5"], by_n["10"])
          + f"  (n<=10 score {np.mean(pf_new):.4f} vs {np.mean(pf_ctl):.4f})  -> {'PASS' if all(cond.values()) else 'no'}")
    return all(cond.values())


def main():
    out = {}
    bad = check_members(range(6))
    print("member check:", "all 48 short members present and consistent" if not bad else bad[:10]); out["member_check"] = bad
    if any(b.startswith("missing") and "_s%d" % s in b for b in bad for s in (0, 1, 2)):
        print("discovery members missing: stop"); Path(ES / "m69_route_eval.json").write_text(json.dumps(out, indent=1)); return 1
    have_rep = not any(b.startswith("missing") for b in bad)
    print("PRIMARY (pre-registered)")
    p1 = run("discovery 12 v 12", [(STD, (0, 1, 2)), (SHORT, (0, 1, 2))], [(STD, (0, 1, 2, 3, 4, 5))], (0, 1, 2), out)
    p2 = run("replication 12 v 12", [(STD, (6, 7, 8)), (SHORT, (3, 4, 5))], [(STD, (6, 7, 8, 9, 10, 11))], (6, 7, 8), out) if have_rep else False
    out["adopt"] = bool(p1 and p2); print(f"ADOPT: {'YES' if p1 and p2 else 'NO'} (discovery {'pass' if p1 else 'no'}, replication {'pass' if p2 else 'no'})")
    print("REPORT ONLY")
    run("pure routing 6 v 6 (s0-2)", [(SHORT, (0, 1, 2))], [(STD, (0, 1, 2))], (0, 1, 2), out)
    if have_rep:
        run("pure routing 6 v 6 (s3-5)", [(SHORT, (3, 4, 5))], [(STD, (6, 7, 8))], (6, 7, 8), out)
        run("pooled 24 v 24", [(STD, range(6)), (SHORT, range(6))], [(STD, range(12))], tuple(range(6)), out)
    # the short members on the n = 20 questions, against the standard ones (expected worse)
    sc20 = {}
    for name, arms in (("short", SHORT), ("standard", STD)):
        vals = []
        for v in FOLDS:
            y, n, pid, q = load_fold(v); s = evaluate(mean_p(v, arms, (0, 1, 2)), y, q, sizes=SIZES)[1]
            vals.append({str(m): float(balanced_mean_over_n(s, y, n, sizes=(m,))) for m in SIZES})
        sc20[name] = {m: float(np.mean([x[m] for x in vals])) for m in vals[0]}
    out["six_member_score_by_n"] = sc20
    print("6-member score by n: " + "; ".join(f"{k} " + " ".join(f"n{m} {v:.4f}" for m, v in d.items()) for k, d in sc20.items()))
    Path(ES / "m69_route_eval.json").write_text(json.dumps(out, indent=1, default=float)); print("written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
