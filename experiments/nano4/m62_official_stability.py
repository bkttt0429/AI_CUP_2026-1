"""Section 4M M62 diagnostics (2026-11-09): ct9's stability on official-platform data WITHOUT rank labels. Report only; nothing is trained.
Inputs: results/b1_public/submission_rank_ct9_proba.csv (ct9's 24 members on competition 1's old Public: 400 candidates x 5 chunks of 20
black games, and its 400 questions of 5/10/15/20 games), inputs/cands_map.csv (chunk -> player), inputs/quest.csv (question -> n), and the
identification session's certain pairings (~/BK_AICUP_2026-2/results/probe/upload_fix.csv top_1, proven by its 1.0000000 score).
Official side (unseen players, each with 100 games):
  (a) player-level floor: ICC(1) of the expected level over a player's 5 chunks of 20 games (between- vs within-player sd); agreement of a
      chunk's decision with the player's consensus (mean log p of all 5 chunks), with the leave-one-out consensus (other 4 chunks, 80 games),
      and pairwise between chunks; exact and within one level.
  (b) games -> reliability: the question's decision vs the true player's 100-game consensus (disjoint games), by n = 5/10/15/20.
  (c) drift: decision distribution (consensus, chunks, questions), A|1D first-band share among decided 1D, entropy and max p by n.
CV reference (ct9's six members per fold, validation design 1x20 + 2x10 + 4x5 on the SAME 20 games of each player): agreement of the
n = 5 / 10 decisions with the n = 20 decision and with the truth, ICC of the expected level for n = 10 (2 per player) and n = 5 (4 per
player), and the same distribution figures. The designs differ (official: disjoint games, 100-game consensus; CV: nested games, 20-game
reference), which the report must state. Also writes cands_consensus.csv and questions_decisions.csv for the identification session.
  python -I logs/nano4/m62_official_stability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

R = {lv: i for i, lv in enumerate(LEVELS)}
A, D1 = R["A"], R["1D"]
B1 = ROOT / "results/b1_public"
PC = [f"p_{lv}" for lv in LEVELS]
MEMBERS = [(f"{t}-pc-f0-x40-oh", s) for t in ("C", "T") for s in (0, 1, 2)]


def softmax(l):
    z = np.exp(l - l.max())
    return z / z.sum()


def summarize(P):
    E = (P * np.arange(10)).sum(1)
    H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    d = decide(P)
    band = (d == D1) & (P[:, A] / np.clip(P[:, D1], 1e-12, None) >= 0.5)
    return E, H, P.max(1), np.asarray(d, int), band


def icc(groups):
    """One-way random-effects ICC(1) of a quantity measured k times per player -> (icc, within sd, between sd)."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 1]
    k = np.array([len(g) for g in groups]); n, N = len(groups), int(k.sum())
    means = np.array([g.mean() for g in groups]); grand = np.concatenate(groups).mean()
    msb = (k * (means - grand) ** 2).sum() / (n - 1); msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1); vb = max(msb - msw, 0.0) / k0
    return float(vb / (vb + msw)), float(np.sqrt(msw)), float(np.sqrt(vb))


def agree(d, ref):
    d, ref = np.asarray(d, int), np.asarray(ref, int)
    return {"exact": float((d == ref).mean()), "adjacent": float((np.abs(d - ref) <= 1).mean()),
            "score_like": float(np.where(d == ref, 1.0, np.where(np.abs(d - ref) == 1, np.exp(-1), 0.0)).mean())}


def dist(d, band, H, M):
    d = np.asarray(d, int); cnt = np.bincount(d, minlength=10)
    return {"decisions": {lv: int(c) for lv, c in zip(LEVELS, cnt)}, "decided_1D": int((d == D1).sum()),
            "first_band_share_of_1D": float(band.sum() / max((d == D1).sum(), 1)), "entropy": float(np.mean(H)), "maxp": float(np.mean(M))}


def official():
    pr = pd.read_csv(B1 / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id")
    cm = pd.read_csv(B1 / "inputs/cands_map.csv", dtype=str)
    qs = pd.read_csv(B1 / "inputs/quest.csv", usecols=["question_id", "num_games"], dtype=str)
    truth = pd.read_csv(Path.home() / "BK_AICUP_2026-2/results/probe/upload_fix.csv", dtype=str).set_index("question_id")["top_1"]
    missing = set(cm.question_id) | set(qs.question_id); missing -= set(pr.index)
    assert not missing, f"{len(missing)} questions without probabilities"
    cons, rows_c, Es, dcs, dloo, pair, bands, Hs, Ms, out_c = {}, [], [], [], [], [], [], [], [], []
    for pid, g in cm.groupby("player_id", sort=True):
        ids = g.sort_values("chunk").question_id.tolist()
        P = pr.loc[ids, PC].to_numpy(float); L = np.log(np.clip(P, 1e-12, 1))
        c = softmax(L.mean(0)); cons[pid] = c; dc = int(decide(c[None])[0])
        E, H, M, d, band = summarize(P)
        loo = np.array([int(decide(softmax(np.delete(L, i, 0).mean(0))[None])[0]) for i in range(len(ids))])
        Es.append(E); rows_c.append((d, np.full(len(d), dc))); dloo.append((d, loo)); bands.append(band); Hs.append(H); Ms.append(M)
        pair.append(float(np.mean([d[i] == d[j] for i in range(len(d)) for j in range(i + 1, len(d))])))
        dcs.append(dc); out_c.append({"player_id": pid, "consensus_decision": LEVELS[dc], "n_chunks": len(ids), **{f"lp_{lv}": float(x) for lv, x in zip(LEVELS, L.mean(0))}})
    d_all = np.concatenate([d for d, _ in rows_c]); dc_all = np.concatenate([c for _, c in rows_c]); loo_all = np.concatenate([l for _, l in dloo])
    band_all, H_all, M_all = np.concatenate(bands), np.concatenate(Hs), np.concatenate(Ms)
    i20 = icc(Es)
    rep = {"players": len(cons), "chunks": int(len(d_all)),
           "chunk_vs_consensus": agree(d_all, dc_all), "chunk_vs_loo_consensus_80_games": agree(d_all, loo_all), "chunk_pairwise_exact": float(np.mean(pair)),
           "icc_expected_level_n20": {"icc": i20[0], "within_sd": i20[1], "between_sd": i20[2]},
           "consensus_distribution": {lv: int(c) for lv, c in zip(LEVELS, np.bincount(np.array(dcs), minlength=10))},
           "chunks_n20": dist(d_all, band_all, H_all, M_all), "questions_by_n": {}}
    out_q = []
    for qid, n in zip(qs.question_id, qs.num_games.astype(int)):
        P = pr.loc[[qid], PC].to_numpy(float); E, H, M, d, band = summarize(P)
        pid = truth[qid]; c = cons[pid]; dc = int(decide(c[None])[0]); Ec = float((c * np.arange(10)).sum())
        out_q.append({"question_id": qid, "n": n, "true_player": pid, "decision": LEVELS[int(d[0])], "player_consensus": LEVELS[dc], "E": float(E[0]), "E_consensus": Ec,
                      "entropy": float(H[0]), "maxp": float(M[0]), "first_band": bool(band[0])})
    q = pd.DataFrame(out_q)
    for n, g in q.groupby("n"):
        dd, dc = g.decision.map(R).to_numpy(), g.player_consensus.map(R).to_numpy()
        rep["questions_by_n"][int(n)] = {"questions": int(len(g)), "vs_100_game_consensus": agree(dd, dc), "mean_abs_dE": float((g.E - g.E_consensus).abs().mean()),
                                         **dist(dd, g.first_band.to_numpy(), g.entropy.to_numpy(), g.maxp.to_numpy())}
    pd.DataFrame(out_c).to_csv(B1 / "cands_consensus.csv", index=False); q.to_csv(B1 / "questions_decisions.csv", index=False)
    return rep


def cv_reference():
    frames = []
    for v in FOLDS:
        y, n, pid, _ = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t, s in MEMBERS], axis=0)
        E, H, M, d, band = summarize(P)
        frames.append(pd.DataFrame({"fold": v, "pid": pid, "n": n, "y": y, "d": d, "E": E, "H": H, "M": M, "band": band}))
    df = pd.concat(frames, ignore_index=True)
    ref = df[df.n == 20].drop_duplicates("pid").set_index("pid")
    df["d20"] = df.pid.map(ref.d); df["E20"] = df.pid.map(ref.E)
    rep = {"questions": int(len(df)), "players": int(df.pid.nunique()), "by_n": {}, "icc_expected_level": {}}
    for n, g in df.groupby("n"):
        gg = g.dropna(subset=["d20"])
        r = {"questions": int(len(g)), "vs_truth": agree(g.d, g.y), **dist(g.d.to_numpy(), g.band.to_numpy(), g.H.to_numpy(), g.M.to_numpy())}
        if n != 20:
            r["vs_same_player_n20_same_games"] = agree(gg.d, gg.d20.astype(int)); r["mean_abs_dE_vs_n20"] = float((gg.E - gg.E20).abs().mean())
        rep["by_n"][int(n)] = r
    for n, k in ((10, 2), (5, 4)):
        groups = [g.E.to_numpy() for _, g in df[df.n == n].groupby("pid") if len(g) == k]
        i = icc(groups); rep["icc_expected_level"][f"n{n}"] = {"icc": i[0], "within_sd": i[1], "between_sd": i[2], "players": len(groups)}
    rep["truth_distribution"] = {lv: int(c) for lv, c in zip(LEVELS, np.bincount(df.drop_duplicates("pid").y.to_numpy(), minlength=10))}
    return rep


def main():
    off = official(); cv = cv_reference()
    rep = {"official": off, "cv_reference": cv,
           "caveats": ["official players are competition 1's Public candidates: unseen by every ct9 member, high-activity (>= 100 black games), black games only; not the rank-test population",
                       "official consensus = 100 disjoint games (5 chunks of 20); the CV reference compares n = 5 / 10 questions with the n = 20 question on the SAME 20 games, so CV agreement is optimistic relative to the official design",
                       "no rank labels on the official side: 'score_like' uses the player's consensus as the reference, not the truth"]}
    (B1 / "m62_official_stability.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"official: {off['players']} players, {off['chunks']} chunks of 20; chunk vs consensus exact {off['chunk_vs_consensus']['exact']:.3f} adjacent {off['chunk_vs_consensus']['adjacent']:.3f}; "
          f"vs leave-one-out (80 games) exact {off['chunk_vs_loo_consensus_80_games']['exact']:.3f}; pairwise exact {off['chunk_pairwise_exact']:.3f}")
    i = off["icc_expected_level_n20"]; print(f"official ICC(expected level, 5 x 20 games): {i['icc']:.3f}; within-player sd {i['within_sd']:.3f} levels, between-player sd {i['between_sd']:.3f}")
    for n, r in sorted(off["questions_by_n"].items()):
        a = r["vs_100_game_consensus"]; print(f"official n={n:2d}: vs 100-game consensus exact {a['exact']:.3f} adjacent {a['adjacent']:.3f} score-like {a['score_like']:.3f}; |dE| {r['mean_abs_dE']:.3f}; "
                                            f"entropy {r['entropy']:.3f} maxp {r['maxp']:.3f}; decided 1D {r['decided_1D']}, first-band share {r['first_band_share_of_1D']:.2f}")
    print("official consensus distribution:", off["consensus_distribution"]); print("official chunk decisions:", off["chunks_n20"]["decisions"], f"first-band share of 1D {off['chunks_n20']['first_band_share_of_1D']:.2f}")
    for n, r in sorted(cv["by_n"].items()):
        s = f"cv n={n:2d}: vs truth exact {r['vs_truth']['exact']:.3f} adjacent {r['vs_truth']['adjacent']:.3f}; entropy {r['entropy']:.3f} maxp {r['maxp']:.3f}; first-band share of 1D {r['first_band_share_of_1D']:.2f}"
        if "vs_same_player_n20_same_games" in r:
            a = r["vs_same_player_n20_same_games"]; s += f"; vs n=20 (same games) exact {a['exact']:.3f} adjacent {a['adjacent']:.3f}; |dE| {r['mean_abs_dE_vs_n20']:.3f}"
        print(s)
    for k, i in cv["icc_expected_level"].items():
        print(f"cv ICC(expected level, {k}, same player's disjoint {k[1:]}-game questions): {i['icc']:.3f}; within sd {i['within_sd']:.3f}, between sd {i['between_sd']:.3f} ({i['players']} players)")
    print("written", B1 / "m62_official_stability.json", "cands_consensus.csv", "questions_decisions.csv")


if __name__ == "__main__":
    main()
