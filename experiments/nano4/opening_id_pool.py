"""Full-pool opening-model identification (2026-11-06): every training player (8,007), two opening models (the player's moves as Black; as
White via colour swap), vectorised scoring of a game's first-12-move tokens against all players.
  1 held-out queries (2 Black games per player, not fitted): top-1 / 5 / 20, margin, impostor null (true player excluded), recall at the null's q99
  2 training opponents (a sample of rows): the opponent side scored against the opposite-colour model, the row's own player excluded; share of
    margins above q99 - 1% = excess; prevalence of training-player opponents = excess / recall
  3 test (800 questions, 10,000 games): the TARGET side against its colour model -> per question, is one training player the confident top-1
    in several games (an overlap of test and training players: report, do not use); the OPPONENT side -> per-game match (player, label,
    margin) saved to results/r_variants/opening_id_test_matches.csv for a possible label-anchor feature.
  python -I logs/nano4/opening_id_pool.py --out results/r_variants/opening_id_pool.json
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path.home() / "BK_AICUP_2026-2/src")]
from aicup.scoring import TRAIN_DIR, load_games  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
from aicup_pid.opening import N_POINTS, tokens  # noqa: E402
import predict_submission as ps  # noqa: E402

DEPTH, ALPHA, BETA, ALPHA_FULL = 12, 2.0, 5.0, 1.0


def swap(moves):
    return [("W" if c == "B" else "B", x, y) for c, x, y in moves]


def both_tokens(sgf):
    m = parse_moves(sgf)
    return tokens(m, DEPTH), tokens(swap(m), DEPTH)  # (Black side's tokens, White side's tokens as Black)


class PoolModel:
    """Vectorised version of aicup_pid.opening.OpeningModel over many players."""

    def __init__(self, n_players):
        self.n = n_players; self.full = defaultdict(Counter); self.black = defaultdict(Counter); self.fullT = defaultdict(Counter); self.blackT = defaultdict(Counter)
        self.bg = Counter(); self.bgT = Counter()

    def add(self, i, toks):
        for fctx, bctx, mv in toks:
            self.full[(fctx, mv)][i] += 1; self.black[(bctx, mv)][i] += 1; self.fullT[fctx][i] += 1; self.blackT[bctx][i] += 1
            self.bg[(bctx, mv)] += 1; self.bgT[bctx] += 1

    def finalize(self):
        conv = lambda d: {k: (np.fromiter(c.keys(), dtype=np.int64, count=len(c)), np.fromiter(c.values(), dtype=np.float64, count=len(c))) for k, c in d.items()}  # noqa: E731
        self.full, self.black, self.fullT, self.blackT = conv(self.full), conv(self.black), conv(self.fullT), conv(self.blackT)
        return self

    def _dense(self, d, key):
        v = np.zeros(self.n)
        e = d.get(key)
        if e is not None:
            v[e[0]] = e[1]
        return v

    def scores(self, toks):
        out = np.zeros(self.n)
        for fctx, bctx, mv in toks:
            pb = (self.bg.get((bctx, mv), 0) + BETA / N_POINTS) / (self.bgT.get(bctx, 0) + BETA)
            p_black = (self._dense(self.black, (bctx, mv)) + ALPHA * pb) / (self._dense(self.blackT, bctx) + ALPHA)
            p_full = (self._dense(self.full, (fctx, mv)) + ALPHA_FULL * p_black) / (self._dense(self.fullT, fctx) + ALPHA_FULL)
            out += np.log(p_full)
        return out


def best2(s, exclude=None):
    if exclude is not None:
        s = s.copy(); s[exclude] = -np.inf
    o = np.argpartition(-s, 1)[:2]; o = o[np.argsort(-s[o])]
    return int(o[0]), float(s[o[0]] - s[o[1]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--opp-sample", type=int, default=30000)
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(0)
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]) for p in sorted(Path(TRAIN_DIR).glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; n = len(players)
    plabel = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).to_numpy()
    print(f"{len(rows)} rows, {n} players ({time.time() - t0:.0f} s)", flush=True)
    sgfs = load_games(rows.game_key.tolist()); print(f"SGFs loaded ({time.time() - t0:.0f} s)", flush=True)
    keys = rows.game_key.tolist()
    with mp.get_context("fork").Pool(a.workers) as pool:
        toks = pool.map(both_tokens, [sgfs[k][0] for k in keys], chunksize=512)
    tok = dict(zip(keys, toks)); del toks
    print(f"tokens for {len(tok)} games ({time.time() - t0:.0f} s)", flush=True)
    # held-out: 2 Black games per player
    held, heldW = set(), set()
    for col, store in (("B", held), ("W", heldW)):
        for pid, g in rows[rows.color == col].groupby("player_id"):
            ks = g.game_key.tolist()
            if len(ks) >= 6:
                store |= set(rng.choice(ks, 2, replace=False).tolist())
    held_all = held | heldW
    MB, MW = PoolModel(n), PoolModel(n)
    for r in rows.itertuples():
        if r.game_key in held_all:
            continue
        tb, tw = tok[r.game_key]
        (MB if r.color == "B" else MW).add(pidx[r.player_id], tb if r.color == "B" else tw)
    MB.finalize(); MW.finalize(); print(f"models fitted; held-out B {len(held)} W {len(heldW)} ({time.time() - t0:.0f} s)", flush=True)
    rep = {"players": n, "rows": int(len(rows))}
    # 1 held-out
    hk = rows[rows.game_key.isin(held)]
    ranks, margins, imp, bestB, mabsB, labB = [], [], [], [], [], []
    for r in hk.itertuples():
        s = MB.scores(tok[r.game_key][0]); i = pidx[r.player_id]
        ranks.append(int((s > s[i]).sum())); b, m = best2(s); margins.append(m if b == i else -m); mabsB.append(m); bestB.append(b); labB.append(r.rank); _, mi = best2(s, exclude=i); imp.append(mi)
    ranks, margins, imp, bestB, mabsB, labB = np.array(ranks), np.array(margins), np.array(imp), np.array(bestB), np.array(mabsB), np.array(labB); q99 = float(np.quantile(imp, 0.99)); q999 = float(np.quantile(imp, 0.999))
    true_conf = (ranks == 0) & (margins > q99)
    rep["heldout_black"] = {"queries": int(len(ranks)), "top1": float((ranks == 0).mean()), "top5": float((ranks < 5).mean()), "top20": float((ranks < 20).mean()), "median_rank": float(np.median(ranks) + 1),
                           "impostor_q99": q99, "impostor_q999": q999, "recall_at_q99": float(true_conf.mean()), "precision_at_q99": float((ranks[margins > q99] == 0).mean()) if (margins > q99).any() else None,
                           "recall_at_q999": float(((ranks == 0) & (margins > q999)).mean())}
    hkw = rows[rows.game_key.isin(heldW)]; ranksW, margW, impW = [], [], []
    for r in hkw.itertuples():
        s = MW.scores(tok[r.game_key][1]); i = pidx[r.player_id]
        ranksW.append(int((s > s[i]).sum())); b, m = best2(s); margW.append(m if b == i else -m); _, mi = best2(s, exclude=i); impW.append(mi)
    ranksW, margW, impW = np.array(ranksW), np.array(margW), np.array(impW); q99W = float(np.quantile(impW, 0.99)); mabsW = np.abs(margW)
    rep["heldout_white"] = {"queries": int(len(ranksW)), "top1": float((ranksW == 0).mean()), "top5": float((ranksW < 5).mean()), "impostor_q99": q99W, "recall_at_q99": float(((ranksW == 0) & (margW > q99W)).mean()),
                           "top1_margin_median_true": float(np.median(margW[ranksW == 0])) if (ranksW == 0).any() else None}
    print(f"1b held-out WHITE ({len(ranksW)} queries): top-1 {rep['heldout_white']['top1']:.3f} top-5 {rep['heldout_white']['top5']:.3f}; impostor q99 {q99W:.2f}; recall at q99 {rep['heldout_white']['recall_at_q99']:.3f}", flush=True)
    h = rep["heldout_black"]
    print(f"1 held-out ({h['queries']} queries, {n} candidates): top-1 {h['top1']:.3f} top-5 {h['top5']:.3f} top-20 {h['top20']:.3f} median rank {h['median_rank']:.0f}; impostor q99 {q99:.2f}; at q99: recall {h['recall_at_q99']:.3f} precision {h['precision_at_q99']:.3f}; recall at q99.9 {h['recall_at_q999']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    # 2 training opponents
    samp = rows.sample(min(a.opp_sample, len(rows)), random_state=0)
    om = []
    for r in samp.itertuples():
        tb, tw = tok[r.game_key]; M, t = (MW, tw) if r.color == "B" else (MB, tb)  # the opponent plays the other colour
        _, m = best2(M.scores(t), exclude=pidx[r.player_id]); om.append(m)
    om = np.array(om); colB = samp.color.to_numpy() == "B"  # the opponent is White when the row's player is Black -> W model -> q99W
    thr = np.where(colB, q99W, q99); share = float((om > thr).mean()); excess = share - 0.01
    rec_mix = 0.5 * (h["recall_at_q99"] + rep["heldout_white"]["recall_at_q99"])
    rep["train_opponents"] = {"games": int(len(om)), "share_above_colour_q99": share, "excess": excess, "recall_used": rec_mix, "prevalence_estimate": float(excess / max(rec_mix, 1e-9))}
    tr = rep["train_opponents"]
    print(f"2 training opponents ({len(om)} games): above the colour-matched q99 {share:.4f} (excess {excess:+.4f}); recall at q99 {rec_mix:.3f} -> prevalence estimate {tr['prevalence_estimate']:.3f} (unreliable when recall is tiny) ({time.time() - t0:.0f} s)", flush=True)
    # 3 test
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS); tasks, owner, _, _ = ps.build_tasks(questions)
    with mp.get_context("fork").Pool(a.workers) as pool:
        ttoks = pool.map(both_tokens, [sgf for _, sgf, _ in tasks], chunksize=256)
    recs = []; per_q = defaultdict(list)
    for (k, sgf, col), (tb, tw), qi in zip(tasks, ttoks, owner):
        hd = re.search(r"HD\[(\d+)\]", sgf); hdv = int(hd.group(1)) if hd else 0
        st = (MB if col == "B" else MW).scores(tb if col == "B" else tw); bt, mt = best2(st)
        so = (MW if col == "B" else MB).scores(tw if col == "B" else tb); bo, mo = best2(so)
        thr_t = q99 if col == "B" else q99W; thr_o = q99W if col == "B" else q99
        recs.append({"question_id": questions[qi]["question_id"], "task": k, "color": col, "hd": hdv, "target_best": players[bt], "target_label": plabel[bt], "target_margin": mt, "target_confident": bool(mt > thr_t),
                     "opp_best": players[bo], "opp_label": plabel[bo], "opp_margin": mo, "opp_confident_q99": bool(mo > thr_o), "opp_confident_q999": bool(mo > q999)})
        per_q[questions[qi]["question_id"]].append((bt, mt > thr_t))
    R = pd.DataFrame(recs); R.to_csv(ROOT / "results/r_variants/opening_id_test_matches.csv", index=False)
    # overlap check: per question, the most frequent confident target-side top-1
    overlap = []
    ct7 = pd.read_csv(ROOT / "submission_rank_ct7.csv", dtype=str).set_index("question_id")["pred_rank"]
    for qid, lst in per_q.items():
        allc = Counter(b for b, _ in lst); b_all, plur = allc.most_common(1)[0]
        conf = Counter(b for b, c in lst if c); b_c, cnt = conf.most_common(1)[0] if conf else (b_all, 0)
        overlap.append({"question_id": qid, "games": len(lst), "plurality_top1_count": plur, "plurality_player": players[b_all], "plurality_label": plabel[b_all],
                        "confident_same_player_hits": cnt, "confident_player": players[b_c], "confident_label": plabel[b_c], "ct7": ct7.get(qid)})
    ov = pd.DataFrame(overlap); ov.to_csv(ROOT / "results/r_variants/opening_id_test_questions.csv", index=False)
    # null for the plurality count: games reassigned to random questions of the same size
    sizes = ov.games.to_numpy(); allbest = np.array([b for lst in per_q.values() for b, _ in lst]); null_plur = []
    for _ in range(20):
        perm = rng.permutation(allbest); pos = 0
        for sz in sizes:
            null_plur.append(Counter(perm[pos:pos + sz].tolist()).most_common(1)[0][1]); pos += sz
    null_plur = np.array(null_plur)
    rep["plurality"] = {"test_share_ge3": float((ov.plurality_top1_count >= 3).mean()), "null_share_ge3": float((null_plur >= 3).mean()), "test_share_ge4": float((ov.plurality_top1_count >= 4).mean()), "null_share_ge4": float((null_plur >= 4).mean()),
                        "test_share_ge5": float((ov.plurality_top1_count >= 5).mean()), "null_share_ge5": float((null_plur >= 5).mean()), "questions_ge5": int((ov.plurality_top1_count >= 5).sum())}
    hi = ov[ov.plurality_top1_count >= 4]
    agree = float((hi.plurality_label == hi.ct7).mean()) if len(hi) else None; adj = float((hi.plurality_label.map(lambda x: ps.LEVELS.index(x) if hasattr(ps, "LEVELS") else 0) - hi.ct7.map(lambda x: ps.LEVELS.index(x) if hasattr(ps, "LEVELS") else 0)).abs().le(1).mean()) if len(hi) and hasattr(ps, "LEVELS") else None
    rep["plurality"]["ge4_questions"] = int(len(hi)); rep["plurality"]["ge4_label_equals_ct7"] = agree
    print(f"   plurality of the per-game top-1 player within a question: >=3 in {rep['plurality']['test_share_ge3']:.3f} of questions (null {rep['plurality']['null_share_ge3']:.3f}), >=4 {rep['plurality']['test_share_ge4']:.3f} (null {rep['plurality']['null_share_ge4']:.3f}), >=5 {rep['plurality']['test_share_ge5']:.3f} (null {rep['plurality']['null_share_ge5']:.3f}); "
          f"for the {len(hi)} questions with >= 4: the plurality player's label equals ct7's prediction in {agree}", flush=True)
    flagged = ov[ov.confident_same_player_hits >= 3]
    rep["test"] = {"games": int(len(R)), "target_side_share_above_q99": float(R.target_confident.mean()), "target_side_excess": float(R.target_confident.mean() - 0.01),
                   "opp_side_share_above_q99": float(R.opp_confident_q99.mean()), "opp_side_excess": float(R.opp_confident_q99.mean() - 0.01), "opp_side_share_above_q999": float(R.opp_confident_q999.mean()),
                   "questions_with_ge3_confident_same_player_hits": int(len(flagged)), "max_hits_in_a_question": int(ov.confident_same_player_hits.max()) if len(ov) else 0,
                   "questions_with_ge2_hits": int((ov.confident_same_player_hits >= 2).sum()) if len(ov) else 0}
    t = rep["test"]
    print(f"3 test ({t['games']} games): TARGET side above q99 {t['target_side_share_above_q99']:.4f} (excess {t['target_side_excess']:+.4f}); questions with >= 3 confident hits on the same training player: {t['questions_with_ge3_confident_same_player_hits']} (max hits {t['max_hits_in_a_question']}, >= 2 hits: {t['questions_with_ge2_hits']}); "
          f"OPPONENT side above q99 {t['opp_side_share_above_q99']:.4f} (excess {t['opp_side_excess']:+.4f}), above q99.9 {t['opp_side_share_above_q999']:.5f} ({time.time() - t0:.0f} s)", flush=True)
    # mixture estimate of the share of pool players in a population of games: F = p F_true + (1 - p) F_imp over many thresholds
    def fit_p(obs, f_true, f_imp):
        ts = np.quantile(f_imp, [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]); A, y = [], []
        for t in ts:
            A.append((f_true > t).mean() - (f_imp > t).mean()); y.append((obs > t).mean() - (f_imp > t).mean())
        A, y = np.array(A), np.array(y); p = float(np.clip((A @ y) / max(A @ A, 1e-12), 0, 1))
        return p
    pB = fit_p(om[colB == False] if (colB == False).any() else om, mabsB, imp); pW = fit_p(om[colB], mabsW, impW)  # opponent is Black when the row's player is White
    rep["mixture"] = {"train_opponents_black_side_p": pB, "train_opponents_white_side_p": pW,
                      "test_target_black_p": fit_p(R.target_margin[R.color == "B"].to_numpy(), mabsB, imp), "test_target_white_p": fit_p(R.target_margin[R.color == "W"].to_numpy(), mabsW, impW),
                      "test_opponent_black_p": fit_p(R.opp_margin[R.color == "W"].to_numpy(), mabsB, imp), "test_opponent_white_p": fit_p(R.opp_margin[R.color == "B"].to_numpy(), mabsW, impW)}
    print("mixture estimates of the share of games whose side is a training player: training opponents B {:.3f} W {:.3f}; TEST targets B {:.3f} W {:.3f}; TEST opponents B {:.3f} W {:.3f}".format(
        pB, pW, rep["mixture"]["test_target_black_p"], rep["mixture"]["test_target_white_p"], rep["mixture"]["test_opponent_black_p"], rep["mixture"]["test_opponent_white_p"]), flush=True)
    # same-rank null for the plurality statistic: pseudo-questions of 20 held-out Black games from 20 DISTINCT players of the same label
    counts = []
    for lab in np.unique(labB):
        idx_l = np.flatnonzero(labB == lab); rng.shuffle(idx_l)
        seen, groups, cur, curp = set(), [], [], set()
        for j in idx_l:
            pl = hk.player_id.iloc[j] if False else None
        # one game per player: hk rows align with ranks/bestB order
        pids_l = hk.player_id.to_numpy()[idx_l]; used = set(); cur = []
        for j, pl in zip(idx_l, pids_l):
            if pl in used:
                continue
            used.add(pl); cur.append(j)
            if len(cur) == 20:
                counts.append(Counter(bestB[cur].tolist()).most_common(1)[0][1]); cur, used = [], set()
    counts = np.array(counts)
    rep["plurality"]["same_rank_null_ge3"] = float((counts >= 3).mean()); rep["plurality"]["same_rank_null_ge4"] = float((counts >= 4).mean()); rep["plurality"]["same_rank_null_ge5"] = float((counts >= 5).mean()); rep["plurality"]["same_rank_null_groups"] = int(len(counts))
    n20 = ov[ov.games == 20]
    rep["plurality"]["test_n20_share_ge3"] = float((n20.plurality_top1_count >= 3).mean()); rep["plurality"]["test_n20_share_ge4"] = float((n20.plurality_top1_count >= 4).mean()); rep["plurality"]["test_n20_share_ge5"] = float((n20.plurality_top1_count >= 5).mean())
    print(f"   same-rank distinct-player null ({len(counts)} pseudo-questions of 20): >=3 {rep['plurality']['same_rank_null_ge3']:.3f}, >=4 {rep['plurality']['same_rank_null_ge4']:.3f}, >=5 {rep['plurality']['same_rank_null_ge5']:.3f}; TEST n=20 questions: >=3 {rep['plurality']['test_n20_share_ge3']:.3f}, >=4 {rep['plurality']['test_n20_share_ge4']:.3f}, >=5 {rep['plurality']['test_n20_share_ge5']:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
