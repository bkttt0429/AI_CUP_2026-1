#!/usr/bin/env python
"""M251b (2026-11-19), report only, label-free: do a few test players own two or more test questions? Game-by-game votes.
(design doc M251b, registered before any reading)

Six identity networks (S7t x2, S6at, the three members of var_S4e). Every game's own-side vector is scored against the
prototypes of the OTHER units (standardised over the candidates per query, mean of the networks); the most similar unit gets
the game's vote. Two units of one player should give each other most of their votes; strangers scatter theirs.
Validation controls (each fold with its cross-fitted networks): P0 = 800 units of 800 different players (5 or 10 games, one
colour); Ps = 760 strangers + 20 players with two units of the same colour (5 games each); Pc = 760 strangers + 20 players
with a black and a white unit.
Also: the alignment of M185's stored test rows, and the opponent side against the other questions' prototypes re-read.

  python -I logs/nano4/m251b_votes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402

CACHE = mb7.CACHE; LEV = led.LEVELS; RNG = np.random.default_rng(2511); REPS = 5; NETS = ("S7t", "S6at", "S4e")


def scores(vecs, rows_q, rows_p, owner, nu, own_of_query):
    """Mean over the networks of the per-query standardised similarity of the query rows to the units' prototypes (prototype = normalised
    sum of the vectors rows_p of each unit); a query's own unit is excluded."""
    F = 0
    for V in vecs:
        T = np.zeros((nu, V.shape[1])); np.add.at(T, owner, V[rows_p].astype(np.float64)); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        S = V[rows_q].astype(np.float64) @ T.T; S[np.arange(len(rows_q)), own_of_query] = np.nan; F = F + (S - np.nanmean(S, 1, keepdims=True)) / np.maximum(np.nanstd(S, 1, keepdims=True), 1e-9)
    F = F / len(vecs); F[np.arange(len(rows_q)), own_of_query] = -1e9; return F


def votes(F, owner, nu):
    """Vote matrix [nu, nu], per unit: most voted other unit, its share of the unit's games, mutual flag; per game: best unit and margin over ranks 2-5."""
    best = F.argmax(1); top = -np.partition(-F, 5, axis=1)[:, :5]; top.sort(axis=1); top = top[:, ::-1]; mg = top[:, 0] - top[:, 1:5].mean(1)
    W = np.zeros((nu, nu)); np.add.at(W, (owner, best), 1.0); k = np.bincount(owner, minlength=nu).astype(float); fav = W.argmax(1); share = W[np.arange(nu), fav] / np.maximum(k, 1); mut = fav[fav] == np.arange(nu)
    return W, fav, share, mut, best, mg, k


def pairs_of(fav, share, mut, lo):
    return [(int(a), int(fav[a])) for a in range(len(fav)) if mut[a] and a < fav[a] and min(share[a], share[fav[a]]) >= lo]


def main():
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); vecs = []
    for name in NETS:
        V = np.load(CACHE / f"var_{name}.npz")["qvec"]; vecs += [V[m] for m in range(V.shape[0])]
    print(f"identity vectors: {len(vecs)} networks ({', '.join(NETS)})", flush=True)
    dv = np.load(ROOT / "results/r_variants/m82_val.npz"); keyv = dv["key"].astype(str); pidv = dv["pid"].astype(str); hdv = dv["hd"].astype(int); tbv = dv["target_black"].astype(bool); foldv = dv["fold"].astype(int); assert np.array_equal(q["qkey"][:Q].astype(str), keyv)
    dt = np.load(ROOT / "results/r_variants/m82_test.npz"); qt = dt["qid"].astype(str); tbt = dt["target_black"].astype(bool); hdt = dt["hd"].astype(int)
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; opp = ti[q["side"][ti] == 1]; opp = opp[np.argsort(q["qtest"][opp])]; assert len(tgt) == len(qt) == len(opp)
    ut, names = pd.factorize(qt); nq = len(names); size = np.bincount(ut, minlength=nq); col = np.zeros(nq, int); col[ut] = tbt.astype(int); pub = np.char.startswith(names.astype(str), "pub"); ans = led.decisions("submission_rank_ct116.csv").reindex(names).to_numpy().astype(int)
    # alignment of M185's stored rows
    z5 = np.load(ROOT / "results/r_variants/m185_tt_test.npz", allow_pickle=True); g5 = z5["game_qid"].astype(str)
    print(f"alignment: M185's stored question id of every test game equals the id used here in {float((g5 == qt).mean()):.4f} of the {len(qt)} rows" if len(g5) == len(qt) else f"alignment: M185 stored {len(g5)} rows, here {len(qt)}", flush=True)
    if len(g5) == len(qt):
        o5 = z5["owner"].astype(int); u5 = z5["qid"].astype(str); bn5 = z5["best_null"].astype(int); a5 = led.decisions("submission_rank_ct112.csv").reindex(u5).to_numpy().astype(int); d5 = np.abs(a5[o5] - a5[bn5])
        print(f"   M185's stored null matches (own side -> best other question): ct112 answers within one level, all games {float((d5 <= 1).mean()):.3f}; statistic above its 99 % quantile {float((d5 <= 1)[z5['stat_null'] > np.quantile(z5['stat_null'], 0.99)].mean()):.3f}", flush=True)
    # ------------------------------------------------------------ test: own side
    Ft = scores(vecs, tgt, tgt, ut, nq, ut); W, fav, share, mut, best, mg, k = votes(Ft, ut, nq)
    near = np.abs(ans[ut] - ans[best]) <= 1; print(f"\ntest, own side: answers (ct116) of a game's question and of the question it votes for within one level {float(near.mean()):.3f} (random pairs {float((np.abs(ans[:, None] - ans[None, :]) <= 1).mean()):.3f})")
    # ------------------------------------------------------------ validation controls
    ok = hdv <= 1; C = dict(p0_share=[], p0_pairs5=[], p0_pairs7=[], p0_mg=[], ps_hit5=[], ps_hit7=[], ps_share=[], pc_hit5=[], pc_hit7=[], pc_share=[], ps_false5=[], pc_false5=[], p0_k=[])
    for f in sorted(set(foldv.tolist())):
        idx = np.flatnonzero(ok & (foldv == f)); df = pd.DataFrame({"i": idx, "p": pidv[idx], "c": tbv[idx].astype(int)}); grp = {k_: g_.i.to_numpy() for k_, g_ in df.groupby(["p", "c"])}
        elig = {k_: (10 if len(g_) >= 10 else 5) for k_, g_ in grp.items() if len(g_) >= 5}; players = sorted({p_ for p_, _ in elig}); both = sorted({p_ for p_, c_ in elig if (p_, 1 - c_) in elig}); ten = sorted({p_ for (p_, c_), k_ in elig.items() if k_ == 10})

        def run(units):
            """units: list of (player, colour, rows)."""
            rows = np.concatenate([u[2] for u in units]); owner = np.concatenate([np.full(len(u[2]), j) for j, u in enumerate(units)]); F = scores(vecs, Q + rows, Q + rows, owner, len(units), owner); return votes(F, owner, len(units))

        def one(p_):
            cs = [c_ for c_ in (0, 1) if (p_, c_) in elig]; c_ = int(RNG.choice(cs)); return (p_, c_, RNG.choice(grp[(p_, c_)], elig[(p_, c_)], replace=False))
        for _ in range(REPS):
            pl = RNG.choice(players, 800, replace=False); U = [one(p_) for p_ in pl]; _, fv, sh, mu, _, mg0, k0 = run(U); C["p0_share"] += sh.tolist(); C["p0_k"] += k0.tolist(); C["p0_pairs5"].append(len(pairs_of(fv, sh, mu, 0.5))); C["p0_pairs7"].append(len(pairs_of(fv, sh, mu, 0.7))); C["p0_mg"] += mg0.tolist()
            sp = RNG.choice(ten, 20, replace=False); rest = RNG.choice([p_ for p_ in players if p_ not in set(sp)], 760, replace=False); U = []
            for p_ in sp:
                c_ = int(RNG.choice([c for c in (0, 1) if elig.get((p_, c)) == 10])); g_ = RNG.choice(grp[(p_, c_)], 10, replace=False); U += [(p_, c_, g_[:5]), (p_, c_, g_[5:])]
            U += [one(p_) for p_ in rest]; _, fv, sh, mu, _, _, _ = run(U); pr5 = set(pairs_of(fv, sh, mu, 0.5)); pr7 = set(pairs_of(fv, sh, mu, 0.7)); true = {(2 * j, 2 * j + 1) for j in range(20)}
            C["ps_hit5"].append(len(pr5 & true) / 20); C["ps_hit7"].append(len(pr7 & true) / 20); C["ps_false5"].append(len(pr5 - true)); C["ps_share"] += [float(min(sh[a], sh[b_])) if fv[a] == b_ and fv[b_] == a else 0.0 for a, b_ in true]
            cp = RNG.choice(both, 20, replace=False); rest = RNG.choice([p_ for p_ in players if p_ not in set(cp)], 760, replace=False); U = []
            for p_ in cp:
                U += [(p_, c_, RNG.choice(grp[(p_, c_)], elig[(p_, c_)], replace=False)) for c_ in (1, 0)]
            U += [one(p_) for p_ in rest]; _, fv, sh, mu, _, _, _ = run(U); pr5 = set(pairs_of(fv, sh, mu, 0.5)); pr7 = set(pairs_of(fv, sh, mu, 0.7))
            C["pc_hit5"].append(len(pr5 & true) / 20); C["pc_hit7"].append(len(pr7 & true) / 20); C["pc_false5"].append(len(pr5 - true)); C["pc_share"] += [float(min(sh[a], sh[b_])) if fv[a] == b_ and fv[b_] == a else 0.0 for a, b_ in true]
        print(f"fold {f}: controls done ({len(players)} players with a unit, {len(ten)} with a 10-game unit, {len(both)} with both colours)", flush=True)
    s0 = np.array(C["p0_share"]); k0 = np.array(C["p0_k"]); qs = lambda x: " / ".join(f"{float(np.quantile(x, q_)):.2f}" for q_ in (0.5, 0.9, 0.99, 0.999))  # noqa: E731
    print(f"\n== P0 (strangers only, {len(C['p0_pairs5'])} pools of 800): share of a unit's games voting for its most voted unit, 50 / 90 / 99 / 99.9 %: {qs(s0)} (units of 5 games {qs(s0[k0 == 5])}; of 10 {qs(s0[k0 == 10])})")
    print(f"   mutual pairs with both shares >= 0.5 per pool: mean {np.mean(C['p0_pairs5']):.2f} (max {max(C['p0_pairs5'])}); >= 0.7: mean {np.mean(C['p0_pairs7']):.2f} (max {max(C['p0_pairs7'])})")
    print(f"== Ps (two units of 5 games, same colour, same player, among 760 strangers): found as a mutual pair with both shares >= 0.5: {np.mean(C['ps_hit5']):.3f}; >= 0.7: {np.mean(C['ps_hit7']):.3f}; false pairs per pool {np.mean(C['ps_false5']):.2f}")
    print(f"== Pc (a black and a white unit of one player among 760 strangers): >= 0.5: {np.mean(C['pc_hit5']):.3f}; >= 0.7: {np.mean(C['pc_hit7']):.3f}; false pairs per pool {np.mean(C['pc_false5']):.2f}")
    # ------------------------------------------------------------ test reading
    p5 = pairs_of(fav, share, mut, 0.5); p7 = pairs_of(fav, share, mut, 0.7); print(f"\n== TEST: share of a question's games voting for its most voted question, 50 / 90 / 99 / 99.9 %: {qs(share)}; by size: " + "; ".join(f"{s}: {qs(share[size == s])}" for s in sorted(set(size.tolist()))))
    print(f"   mutual pairs with both shares >= 0.5: {len(p5)} (P0 expects {np.mean(C['p0_pairs5']):.2f}); >= 0.7: {len(p7)} (P0 expects {np.mean(C['p0_pairs7']):.2f})")
    one_way = [(a, int(fav[a])) for a in range(nq) if share[a] >= 0.5 and not (mut[a] and share[fav[a]] >= 0.5)]
    print("   all mutual pairs with both shares >= 0.5 (votes a->b of a's games | b->a of b's games; colour, games, half, ct116 answer):")
    rows_out = []
    for a, b_ in sorted(p5, key=lambda x: -min(share[x[0]], share[x[1]])):
        print(f"      {names[a]} ({'B' if col[a] else 'W'}, {size[a]}, {LEV[ans[a]]}) <-> {names[b_]} ({'B' if col[b_] else 'W'}, {size[b_]}, {LEV[ans[b_]]}):  {int(W[a, b_])}/{size[a]} | {int(W[b_, a])}/{size[b_]}   same colour {bool(col[a] == col[b_])}, answers differ by {abs(int(ans[a]) - int(ans[b_]))}")
        rows_out.append({"a": str(names[a]), "b": str(names[b_]), "votes_ab": int(W[a, b_]), "votes_ba": int(W[b_, a]), "size_a": int(size[a]), "size_b": int(size[b_]), "col_a": int(col[a]), "col_b": int(col[b_]), "ans_a": LEV[ans[a]], "ans_b": LEV[ans[b_]]})
    print(f"   one-way cases (a question gives >= 0.5 of its votes to one question that does not return them): {len(one_way)}: " + "; ".join(f"{names[a]}->{names[b_]} {int(W[a, b_])}/{size[a]} (back {int(W[b_, a])}/{size[b_]})" for a, b_ in one_way[:25]))
    if p5:
        A_ = np.array(p5); print(f"   of the {len(p5)} pairs: same colour {int((col[A_[:, 0]] == col[A_[:, 1]]).sum())}; Public+Private {int((pub[A_[:, 0]] != pub[A_[:, 1]]).sum())}, both Public {int((pub[A_[:, 0]] & pub[A_[:, 1]]).sum())}, both Private {int((~pub[A_[:, 0]] & ~pub[A_[:, 1]]).sum())}; answers equal {int((ans[A_[:, 0]] == ans[A_[:, 1]]).sum())}, differ by one {int((np.abs(ans[A_[:, 0]] - ans[A_[:, 1]]) == 1).sum())}, by more {int((np.abs(ans[A_[:, 0]] - ans[A_[:, 1]]) > 1).sum())}")
        num = lambda s_: int(str(s_).split("_")[-1])  # noqa: E731
        print("   question numbers of the pairs (a, b): " + ", ".join(f"({'P' if pub[a] else 'p'}{num(names[a])}, {'P' if pub[b_] else 'p'}{num(names[b_])})" for a, b_ in p5))
    # components of the graph with an edge wherever a question gives >= 0.5 of its votes (groups of three or more questions)
    par = list(range(nq))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a in range(nq):
        if share[a] >= 0.5:
            par[find(a)] = find(int(fav[a]))
    comp = {}
    for a in range(nq):
        comp.setdefault(find(a), []).append(a)
    big = [c for c in comp.values() if len(c) >= 2]; print(f"   groups linked by a >= 0.5 vote share: {len(big)} groups, sizes " + ", ".join(str(len(c)) for c in sorted(big, key=len, reverse=True)[:30]) + f"; questions in a group {sum(len(c) for c in big)}")
    # ------------------------------------------------------------ test: opponent side against the other questions' prototypes
    Fo = scores(vecs, opp, tgt, ut, nq, ut); bo = Fo.argmax(1); topo = -np.partition(-Fo, 5, axis=1)[:, :5]; topo.sort(axis=1); topo = topo[:, ::-1]; mo = topo[:, 0] - topo[:, 1:5].mean(1); m0 = np.array(C["p0_mg"])
    print(f"\n== TEST, opponent side -> other questions' prototypes (games between two test players): margin 50 / 90 / 99 / 99.9 %: {qs(mo)}; validation P0 own-side margin (strangers): {qs(m0)}; test own side: {qs(mg)}")
    for qq in (0.99, 0.999, 0.9999):
        thr = float(np.quantile(m0, qq)); m_ = mo > thr; d_ = np.abs(ans[ut[m_]] - ans[bo[m_]]); print(f"   above the P0 {qq * 100:g} % line ({thr:.2f}): {int(m_.sum())} opponent-side games (strangers would give {len(mo) * (1 - qq):.1f}); answers of the two questions within one level {float((d_ <= 1).mean()) if m_.any() else float('nan'):.3f}; even games among them {float((hdt[m_] == 0).mean()) if m_.any() else float('nan'):.3f}")
    for i in np.argsort(-mo)[:12]:
        print(f"      {mo[i]:.2f}  a game of {names[ut[i]]} ({'B' if col[ut[i]] else 'W'}, {LEV[ans[ut[i]]]}, HD{hdt[i]}): the opponent looks like the player of {names[bo[i]]} ({'B' if col[bo[i]] else 'W'}, {LEV[ans[bo[i]]]})")
    excess = len(p5) - float(np.mean(C["p0_pairs5"])); power = min(float(np.mean(C["ps_hit5"])), float(np.mean(C["pc_hit5"])))
    verdict = "CONFIRMED: some test players own several questions" if excess >= 5 and power >= 0.5 else ("no: similar strangers only" if excess < 2 else "in between")
    print(f"\nREGISTERED: mutual pairs (both shares >= 0.5) minus the P0 expectation = {excess:+.1f}; power (same colour 5+5 / two colours) {np.mean(C['ps_hit5']):.3f} / {np.mean(C['pc_hit5']):.3f}  ->  {verdict}")
    (ROOT / "results/r_variants/m251b_votes.json").write_text(json.dumps({"pairs": rows_out, "n_pairs_05": len(p5), "n_pairs_07": len(p7), "p0_pairs_05": float(np.mean(C["p0_pairs5"])), "p0_pairs_07": float(np.mean(C["p0_pairs7"])), "power_same": float(np.mean(C["ps_hit5"])), "power_cross": float(np.mean(C["pc_hit5"])),
                                                                         "groups": [[str(names[a]) for a in c] for c in big], "verdict": verdict}, indent=1)); print("written results/r_variants/m251b_votes.json")


if __name__ == "__main__":
    main()
