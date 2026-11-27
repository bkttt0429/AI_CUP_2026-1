"""Section 4M M58: the A|1D shift scan of m55_shift_scan.py run on each test half separately (public 400, private 400), to see whether the
private half shows the same unlabelled signature as the public half (where the shift is confirmed). Report only.
  python -I logs/nano4/m58_half_scan.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide  # noqa: E402
from m55_shift_scan import DELTAS, em  # noqa: E402


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    ys, ns, Ps = [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        keep = ~(np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)
        ys.append(y[keep]); ns.append(n[keep]); Ps.append(P[keep])
    y, n, P = np.concatenate(ys), np.concatenate(ns), np.concatenate(Ps)
    wq = np.zeros(len(y))
    for c in range(10):
        for m in (5, 10, 20):
            sel = (y == c) & (n == m); wq[sel] = 1.0 / max(sel.sum(), 1)
    d = decide(P)
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); Pt = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); dt = decide(Pt)
    pub = pr["question_id"].str.startswith("pub").to_numpy()
    i, j = LEVELS.index("A"), LEVELS.index("1D")
    s_val = np.log(np.clip(P[:, j], 1e-9, 1)) - np.log(np.clip(P[:, i], 1e-9, 1)); s_t = np.log(np.clip(Pt[:, j], 1e-9, 1)) - np.log(np.clip(Pt[:, i], 1e-9, 1))
    rv = (d == i) | (d == j)
    classes = [c for c in (i - 1, i, j, j + 1)]
    kdes = {c: gaussian_kde(s_val[rv & (y == c)], weights=wq[rv & (y == c)]) for c in classes}
    pools = {c: np.flatnonzero(rv & (y == c)) for c in classes}; pc = {c: wq[pools[c]] / wq[pools[c]].sum() for c in classes}
    rng = np.random.default_rng(0)

    def fit(xs):
        F0 = np.column_stack([kdes[c](xs) for c in classes]); w0, l0 = em(F0); best = (0.0, l0); k = classes.index(i)
        for dl in DELTAS:
            F = F0.copy(); F[:, k] = kdes[i](xs - dl); _, ll = em(F)
            if ll > best[1]:
                best = (float(dl), ll)
        return w0, l0, best

    rep = {}
    for half, m in (("public", pub), ("private", ~pub), ("both", np.ones(len(pub), bool))):
        x = s_t[m & ((dt == i) | (dt == j))]
        w0, l0, best = fit(x); lr = 2 * (best[1] - l0)
        null = []
        for _ in range(300):
            counts = rng.multinomial(len(x), w0)
            xs = np.concatenate([s_val[rng.choice(pools[c], cnt, p=pc[c])] for c, cnt in zip(classes, counts) if cnt > 0])
            _, a0, b = fit(xs); null.append(2 * (b[1] - a0))
        p = float((np.array(null) >= lr).mean())
        rep[half] = {"routed": int(len(x)), "delta": best[0], "lr": lr, "p": p, "composition": dict(zip([LEVELS[c] for c in classes], w0.tolist()))}
        print(f"{half:8s} routed {len(x):3d}: true-A shift delta {best[0]:+.1f}, LR {lr:.2f}, p {p:.3f}; composition " + " ".join(f"{LEVELS[c]} {w:.2f}" for c, w in zip(classes, w0)), flush=True)
    Path(ROOT / "results/trunk_seq/es/m58_half_scan.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
