#!/usr/bin/env python
"""M260 step 1b (2026-11-20, report only): how many validation anchors are "pairs the networks have seen" (the recognised opponent c
owns training rows whose opponent is the validation player P), and is their label relation tighter? (design doc M260)

Flags from results/m260/rows.npz (m260_rows.py): loose = some row of c has P as its best opponent-side candidate; strict = with a
margin >= q80 of all rows; top3 = P among the three best. Control = P replaced by another validation player of the same fold and level.
  R1  share of flagged games by SQ4 band (0-8); SP2e (an older identifier) for comparison
  R2  even games, top bands (7-8) and middle bands (4-6): candidate label == own label / within one, flagged against not flagged
  R3  validation games left in the top / middle bands without the flagged ones (M259: test / validation 0.865 / 0.986)

  python -I logs/nano4/m260_flags.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402

s2 = m3.s2; RV = ROOT / "results/r_variants"; TOP, MID = (7, 8), (4, 5, 6); B = 500


def main():
    rng = np.random.default_rng(260); z = np.load(ROOT / "results/m260/rows.npz"); owner = z["owner"].astype(np.int64); top_i = z["top_i"].astype(np.int64); m15 = z["m15"].astype(np.float32); N = len(owner)
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); pidx = {p: i for i, p in enumerate(players)}; C = len(players); thr = float(np.quantile(m15, 0.8))
    print(f"rows {N}; own-side check of the row job {float(z['hit_own']):.3f}; margin q80 {thr:.2f}; pairs (owner, best opponent candidate): {len(np.unique(owner * C + top_i[:, 0]))}", flush=True)
    sets = {"loose": np.unique(owner * C + top_i[:, 0]), "strict": np.unique((owner * C + top_i[:, 0])[m15 >= thr]), "top3": np.unique((owner[:, None] * C + top_i).ravel())}
    dv = np.load(RV / "m82_val.npz"); pid = dv["pid"].astype(str); y = dv["y"].astype(int); fold = dv["fold"].astype(int); hd = dv["hd"].astype(int); tb = dv["target_black"].astype(bool); P = np.array([pidx[p] for p in pid])
    pfold = np.full(C, -1); pfold[P] = fold; up, uv = np.unique(pid, return_inverse=True); rep = {}
    # control: another validation player of the same fold and level
    Pc = P.copy(); pl = {}
    for p_, f_, y_ in zip(P, fold, y):
        pl.setdefault((f_, y_), set()).add(p_)
    pl = {k: np.array(sorted(v)) for k, v in pl.items()}; shift = {k: rng.integers(1, max(len(v), 2)) for k, v in pl.items()}; posn = {k: {p_: i for i, p_ in enumerate(v)} for k, v in pl.items()}
    for i, (p_, f_, y_) in enumerate(zip(P, fold, y)):
        v = pl[(f_, y_)]; Pc[i] = v[(posn[(f_, y_)][p_] + shift[(f_, y_)]) % len(v)]
    for name in ("SQ4~m15", "SP2e"):
        f = RV / f"m84_{name}_val.npz"
        if not f.exists():
            print(f"{name}: no validation file"); continue
        sv = np.load(f); assert np.array_equal(sv["key"], dv["key"])
        if "f_opp_best" not in sv.files:
            print(f"{name}: no candidate ids stored"); continue
        v = {"key": dv["key"], "y": y, "fold": fold, "hd": hd, "target_black": tb, "opp_m": sv["f_opp_m"], "opp_lab": sv["f_opp_lab"], "imp_m": sv["f_imp_m"], "imp_lab": sv["f_imp_lab"]}
        ones = np.ones(len(y), bool); edges, cal, band, g = s2.fit(v, ones); ok = g >= 0; cand = np.array([pidx[c] for c in sv["f_opp_best"].astype(str)]); lab = sv["f_opp_lab"].astype(int)
        fl = {k: np.isin(cand * C + P, s) for k, s in sets.items()}; flc = np.isin(cand * C + Pc, sets["loose"]); other = pfold[cand] != fold; r_ = rep[name] = {}
        print(f"\n== {name}: validation games with HD <= 1: {int(ok.sum())}; flagged loose {float(fl['loose'][ok].mean()):.4f}, strict {float(fl['strict'][ok].mean()):.4f}, top3 {float(fl['top3'][ok].mean()):.4f}, control {float(flc[ok].mean()):.4f}; best candidate outside the player's fold {float(other[ok].mean()):.3f}")
        print("   R1 band | games | flagged loose | strict | top3 | control (another player of the same fold and level) | loose among candidates outside the fold | inside the fold")
        for b_ in range(9):
            m = ok & (band == b_); mo, mi = m & other, m & ~other
            print(f"   {b_} | {int(m.sum()):6d} | {float(fl['loose'][m].mean()):.3f} | {float(fl['strict'][m].mean()):.3f} | {float(fl['top3'][m].mean()):.3f} | {float(flc[m].mean()):.3f} | {float(fl['loose'][mo].mean()) if mo.any() else float('nan'):.3f} | {float(fl['loose'][mi].mean()) if mi.any() else float('nan'):.3f}")
        r_["flag_by_band"] = [{"games": int((ok & (band == b_)).sum()), "loose": float(fl["loose"][ok & (band == b_)].mean()), "strict": float(fl["strict"][ok & (band == b_)].mean()), "control": float(flc[ok & (band == b_)].mean())} for b_ in range(9)]
        F = fl["loose"]; d = lab - y
        for nm, bs in (("top (7-8)", TOP), ("middle (4-6)", MID)):
            m = ok & np.isin(band, bs); share, ctl = float(F[m].mean()), float(flc[m].mean()); e = m & (g == 0)

            def rel(w):
                a1, a0 = e & F, e & ~F; n1, n0 = w[a1].sum(), w[a0].sum(); s1, s0 = (w * (d == 0))[a1].sum() / max(n1, 1e-9), (w * (d == 0))[a0].sum() / max(n0, 1e-9); w1, w0 = (w * (np.abs(d) <= 1))[a1].sum() / max(n1, 1e-9), (w * (np.abs(d) <= 1))[a0].sum() / max(n0, 1e-9)
                return s1, s0, s1 - s0, w1, w0, w1 - w0
            o = rel(np.ones(len(y))); bt = np.array([rel(np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[uv]) for _ in range(B)]); ci = lambda j: (float(np.percentile(bt[:, j], 2.5)), float(np.percentile(bt[:, j], 97.5)))  # noqa: E731
            left = float((m & ~F).sum() / max(m.sum(), 1)); leftS = float((m & ~fl["top3"]).sum() / max(m.sum(), 1))
            print(f"   {nm}: flagged {share:.3f} (control {ctl:.3f}, difference {share - ctl:+.3f}); R2 even games flagged {int((e & F).sum())} / not {int((e & ~F).sum())}: same level {o[0]:.3f} vs {o[1]:.3f}, difference {o[2]:+.3f} [{ci(2)[0]:+.3f}, {ci(2)[1]:+.3f}]; within one {o[3]:.3f} vs {o[4]:.3f}, difference {o[5]:+.3f} [{ci(5)[0]:+.3f}, {ci(5)[1]:+.3f}]; R3 games left without the flagged ones {left:.3f} (without top3-flagged {leftS:.3f})")
            r_[nm] = {"flagged": share, "control": ctl, "same_flag": o[0], "same_not": o[1], "same_diff": o[2], "same_ci": ci(2), "within1_diff": o[5], "within1_ci": ci(5), "left": left}
        t_ = r_["top (7-8)"]; cond1 = t_["flagged"] - t_["control"] >= 0.15; cond2 = t_["same_diff"] >= 0.05 and t_["same_ci"][0] > 0
        verdict = "validation anchors are more optimistic than the test's: second step (registered separately)" if cond1 and cond2 else ("closed: the extra anchors have the same quality" if (t_["flagged"] - t_["control"] < 0.15 or abs(t_["same_diff"]) <= 0.03) else "in between: reported, no second step")
        print(f"   REGISTERED ({name}{', primary' if name.startswith('SQ4') else ', comparison only'}): top bands flagged - control {t_['flagged'] - t_['control']:+.3f} (>= 0.15: {cond1}); same-level share flagged - not flagged {t_['same_diff']:+.3f} [{t_['same_ci'][0]:+.3f}, {t_['same_ci'][1]:+.3f}] (>= +0.05 and interval above 0: {cond2})  ->  {verdict}", flush=True); r_["verdict"] = verdict
    (RV / "m260_flags.json").write_text(json.dumps(rep, indent=1)); print("\nwritten results/r_variants/m260_flags.json")


if __name__ == "__main__":
    main()
