#!/usr/bin/env python
"""M262 step 4 (2026-11-20): the PRIVATE PROPOSAL of the clean-win-share shift under the user's standing rules, and what the same
shift costs on the validation questions. (design doc M262; the user decides whether anything is adopted)

Setting as registered: pooled posterior of the ct112 rule, log p_k + gamma * x * k with x = standardised win share of the even
games (strong-net outcomes, results/r_variants/m262_outcomes.npz), gamma = -0.4, answers by the ct112 rule (locks + guards).
A Private answer changes only if: ct116 still carries the ct112-rule answer (the user's own eight decisions and the Public-informed
edits stay), the question has >= 3 even games, the change does not leave A, is not 5D -> 6D, is one level, and its expected score
under the shifted posterior is higher. Public rows = ct118e (the best Public file). Nothing is written into the project root.

  python -I logs/nano4/m262_final.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m262_outcome as mo  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402

LEV = led.LEVELS; K = np.arange(10.0); RV = ROOT / "results/r_variants"; GAMMA = -0.4; m3, b, s2, ES, FOLDS = mo.m3, mo.b, mo.s2, mo.ES, mo.FOLDS


def soft(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def main():
    z = np.load(RV / "m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P0 = z["P"][names.index("current")].astype(np.float64); qid = z["qid"].astype(str); nq = z["nq"].astype(int)
    oc = np.load(RV / "m262_outcomes.npz"); op = {str(x): i for i, x in enumerate(oc["qid"])}; oi = np.array([op[x] for x in qid]); ne = oc["n_even"][oi].astype(float); wins = oc["wins_even"][oi].astype(float)
    sh = np.where(ne > 0, wins / np.maximum(ne, 1), 0.5) - 0.5; mu, sd = float(sh.mean()), float(sh.std()); x = (sh - mu) / sd
    dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(b.decide(P), int)), z["c106"])  # noqa: E731
    LP = np.log(np.maximum(P0, 1e-300)); P1 = soft(LP + GAMMA * x[:, None] * K[None, :]); d0, d1 = dec(P0), dec(P1); S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    a116, a118e, a112 = rd("submission_rank_ct116.csv"), rd("submission_rank_ct118e.csv"), rd("submission_rank_ct112.csv"); assert all(a112[q] == LEV[d0[i]] for i, q in enumerate(qid))
    print(f"standardisation of the win share (all 800 questions): mean {mu + 0.5:.3f}, sd {sd:.3f}; gamma {GAMMA}; a question one sd above the mean moves its expected level by about {float(((P1 @ K) - (P0 @ K))[np.abs(x - 1) < 0.25].mean()):+.2f} levels")
    A_, D5, D6 = LEV.index("A"), LEV.index("5D"), LEV.index("6D"); out = {q: (a118e[q] if q.startswith("pub") else a116[q]) for q in qid}; rows = []; why = {}
    for i in np.flatnonzero((d1 != d0) & ~np.char.startswith(qid, "pub")):
        q = str(qid[i]); old, new = int(d0[i]), int(d1[i]); g1 = float(P1[i] @ (S[new] - S[old])); g0 = float(P0[i] @ (S[new] - S[old]))
        reason = ("ct116 does not carry the ct112-rule answer" if a116[q] != LEV[old] else "fewer than 3 even games" if ne[i] < 3 else "leaves A" if old == A_ else "5D -> 6D" if (old == D5 and new == D6) else "two levels" if abs(new - old) >= 2 else "not better under the shifted posterior" if g1 <= 0 else "")
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            out[q] = LEV[new]
        rows.append({"question_id": q, "games": int(nq[i]), "even_games": int(ne[i]), "wins_even": int(wins[i]), "ct116": a116[q], "shifted": LEV[new], "applied": not reason, "left_out_because": reason, "gain_shifted": round(g1, 3), "gain_current": round(g0, 3),
                     "p_old_current": round(float(P0[i, old]), 3), "p_new_current": round(float(P0[i, new]), 3), "p_old_shifted": round(float(P1[i, old]), 3), "p_new_shifted": round(float(P1[i, new]), 3)})
    ap_ = [r for r in rows if r["applied"]]; kinds = {}
    for r in ap_:
        kinds[f"{r['ct116']}>{r['shifted']}"] = kinds.get(f"{r['ct116']}>{r['shifted']}", 0) + 1
    print(f"\n== PRIVATE PROPOSAL: {len(ap_)} of {len(rows)} shifted answers pass the standing rules (left out: {why})")
    print("   question | games (even, won) | ct116 -> proposal | P(old), P(new) under the current posterior | under the shifted posterior | expected gain shifted / current")
    for r in sorted(rows, key=lambda r: r["question_id"]):
        print(f"   {r['question_id']} | {r['games']} ({r['even_games']}, {r['wins_even']}) | {r['ct116']} -> {r['shifted']} | {r['p_old_current']:.2f}, {r['p_new_current']:.2f} | {r['p_old_shifted']:.2f}, {r['p_new_shifted']:.2f} | {r['gain_shifted']:+.3f} / {r['gain_current']:+.3f}" + ("" if r["applied"] else f"   LEFT OUT: {r['left_out_because']}"))
    print(f"   kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + f"; down {sum(LEV.index(r['shifted']) < LEV.index(r['ct116']) for r in ap_)}, up {sum(LEV.index(r['shifted']) > LEV.index(r['ct116']) for r in ap_)}")
    print(f"   self-estimate of the {len(ap_)} changes: {sum(r['gain_shifted'] for r in ap_):+.2f} points under the shifted posterior ({sum(r['gain_shifted'] for r in ap_) / 400:+.4f} of the Private score), {sum(r['gain_current'] for r in ap_):+.2f} under the current posterior ({sum(r['gain_current'] for r in ap_) / 400:+.4f})")
    od = ROOT / "results/m262"; od.mkdir(parents=True, exist_ok=True); order = [r["question_id"] for r in csv.DictReader(open(ROOT / "submission_rank_ct118e.csv"))]; f1 = od / "submission_rank_ct122_proposal.csv"
    with open(f1, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["question_id", "pred_rank"]); [w.writerow([q, out[q]]) for q in order]
    md5 = hashlib.md5(open(f1, "rb").read()).hexdigest(); assert all(out[q] == a118e[q] for q in order if q.startswith("pub")); print(f"   written {f1.relative_to(ROOT)} (md5 {md5}): Public rows = ct118e, Private rows = ct116 with {sum(out[q] != a116[q] for q in order if not q.startswith('pub'))} answers changed")
    pd.DataFrame(rows).to_csv(od / "m262_private_changes.csv", index=False)
    # ------------------------------------------------------------ what the same shift costs on the validation questions
    hd_all = mo.read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"].astype(float)
    V = pd.concat([mo.games_of(ROOT / f"results/trunk_vec/fold{f}", mo.AI / f"base/fold{f}") for f in FOLDS], ignore_index=True); V = mo.outcome(V.assign(hd=hd_all.reindex(V["game_key"]).to_numpy())); won_of = dict(zip(V["game_key"], V["won"])); hd_of = dict(zip(V["game_key"], V["hd"]))
    zp = np.load(RV / "m251_val_post.npz"); off = 0; rec = []; folds_df = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid).astype(str); assert np.array_equal(zp["y"][off:off + len(y)], y) and np.array_equal(zp["pid"][off:off + len(y)].astype(str), pid)
        post = zp["post"][off:off + len(y)].astype(np.float64); post = post / post.sum(1, keepdims=True); off += len(y)
        for i, gk in enumerate(qd["game_keys"]):
            ev = [k for k in gk if hd_of.get(k, 9) == 0]; rec.append((pid[i], int(y[i]), int(n[i]), len(ev), sum(won_of[k] for k in ev), *post[i]))
    R = np.array([r[1:] for r in rec], float); pidv = np.array([r[0] for r in rec]); yv, nv, nev, wv = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3]; Pv = R[:, 4:]; xv = ((np.where(nev > 0, wv / np.maximum(nev, 1), 0.5) - 0.5) - mu) / sd
    tot = folds_df["n_games"].reindex(pidv).to_numpy(); SM = S
    print("\n== the same shift on the validation questions (pooled posterior of the ct112 rule, plain expected-score decisions, standardised with the TEST mean and sd; class-balanced score, sizes 5 / 10 / 20)")
    base_dec = np.asarray(b.decide(Pv), int); s0 = SM[base_dec, yv]; print(f"   no shift: {balanced_mean_over_n(s0, yv, nv, sizes=(5, 10, 20)):.4f}")
    rep = {"proposal_md5": md5, "applied": len(ap_), "self_estimate_shifted": sum(r["gain_shifted"] for r in ap_), "self_estimate_current": sum(r["gain_current"] for r in ap_), "validation": {}}
    for g in (-0.2, -0.4, 0.2):
        dg = np.asarray(b.decide(soft(np.log(np.maximum(Pv, 1e-300)) + g * xv[:, None] * K[None, :])), int); sg = SM[dg, yv]; dd = paired_diff(sg, s0, yv, pidv, nv, n=2000)
        line = f"   gamma {g:+.1f}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] (answers changed {float((dg != base_dec).mean()):.3f}); by size " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items())
        for nm, m in (("< 20 listed games", tot < 20), ("20-39", (tot >= 20) & (tot < 40)), (">= 40", tot >= 40)):
            mm = m & np.isin(nv, (5, 10)); line += f" | {nm} (5, 10 games): {float(sg[mm].mean() - s0[mm].mean()):+.4f}"
        print(line); rep["validation"][str(g)] = {"diff": dd["diff"], "ci": dd["ci95"]}
    (RV / "m262_final.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nwritten results/r_variants/m262_final.json")


if __name__ == "__main__":
    main()
