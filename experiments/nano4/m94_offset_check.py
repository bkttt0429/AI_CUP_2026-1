"""Section 4M M94 (2026-11-13, report only): signed offset of the forward anchors, validation against test. M93 showed that games
which involve test players were taken out of the training rows, so a test player's games against training players are ALL in his
question, while a validation player's games against other training players are split between the two owners. If that split depends
on the levels, the label difference (identified opponent's level minus the player's level) seen in validation would not hold on the
test. Label-free comparison: for the confident forward matches (SP2e identifier, by band, even games and the two handicap-1 groups),
the identified opponent's level minus the base model's decision for that player (validation: the n = 20 question's decision from the
standard six members; test: ct9's decision), as a distribution and a mean; validation also against the true level.
  python -I logs/nano4/m94_offset_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m71_step2 as s2  # noqa: E402
import m86_bundle as b  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

V, T, key_pid, used = b.sources(["5e", "SP2e"]); dec = {}
for f in FOLDS:
    y, n, pid, q = load_fold(f); P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); d0 = (P @ b.SM).argmax(1); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, d0) if n_ == 20})
ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK)
for name in ("SP2e", "5e"):
    v, t = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"])
    vd = np.array([dec.get(key_pid[k], -1) for k in v["key"]]); td = ct9.reindex(t["qid"]).to_numpy(); nb = int(vband.max()) + 1; print(f"\nidentifier {name}: {nb} bands; validation queries {len(vband)}, test queries {len(tb)}")
    for g, gname in ((0, "even games"), (1, "handicap 1, player Black"), (2, "handicap 1, player White")):
        for lo, hi, bname in ((nb - 2, nb - 1, "top two bands"), (nb - 4, nb - 3, "next two bands"), (0, 1, "lowest two bands")):
            sv = (vg == g) & (vband >= lo) & (vband <= hi) & (vd >= 0); st = (tg == g) & (tb >= lo) & (tb <= hi)
            if sv.sum() < 30 or st.sum() < 30:
                continue
            dv, dvt, dt = v["opp_lab"][sv] - vd[sv], v["opp_lab"][sv] - v["y"][sv], t["opp_lab"][st] - td[st]; h = lambda x: " ".join(f"{float((x == k).mean()):.3f}" for k in (-2, -1, 0, 1, 2))  # noqa: E731
            se = float(np.sqrt(dv.var() / len(dv) + dt.var() / len(dt)))
            print(f"  {gname}, {bname}: opponent's level minus the base decision, share at -2..+2 | validation ({int(sv.sum())}) {h(dv)} mean {dv.mean():+.3f} | test ({int(st.sum())}) {h(dt)} mean {dt.mean():+.3f} | test - validation {dt.mean() - dv.mean():+.3f} (se {se:.3f}) | validation vs the true level {h(dvt)} mean {dvt.mean():+.3f}")
