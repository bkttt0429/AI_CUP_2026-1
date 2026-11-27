"""Question-level identification of a rank-prediction question's player among the training players (M24, 2026-11-06).
Five habit components of competition 1 (aicup_pid: opening repertoire, raw style statistics, local-shape tokens, board
orientation, HD class), each a per-player model fitted on the player's games in the view where the target's stones are "B"
(a game whose target mark is W is colour-swapped; separate pools per view, summed per question), fused with one weight per
component (per-question standardisation over the candidates, multinomial likelihood, Newton's method). Three fits:
  A  in-pool gate: the 4,000 inner held-out questions' games (prod_heldout.P.npz) leave the pool, their 788 players stay with
     their other games -> top-1 by component / fused, margin, correct; weights cross-fitted on two halves of the players;
     posterior over labels (softmax over the candidates summed by label)
  B  out-of-pool gate: the 788 players leave entirely -> the margin distribution of a question whose player is NOT in the
     pool (false-accept rates), and whether confident matches land on same-label players (duplicate ids in training?)
  C  test: all games -> the 800 test questions' fused scores, margins, label posteriors; mixture estimate of the share of
     test questions whose player is in the pool, per n (A margins = in-pool reference, B margins = out-of-pool reference)
Outputs: --out JSON, <out>.gate.npz and <out>.test.npz (read by ident_ct8.py). Portable: numpy + pandas only, CPU, fork or
spawn multiprocessing; data under --root (data/training/train_*.csv, data/tests/rank_prediction_test_{public,private}.csv);
the gate file and ct7 submission are looked up under --root (results/r_variants, project root) or next to this script (inputs/).
  python -I scripts/m_series/ident_pool.py --out results/r_variants/ident_pool.json [--limit-players 1200] [--workers 30] [--root DIR]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--root"), Path.cwd())).resolve()
for _p in (HERE, ROOT / "src", Path.home() / "BK_AICUP_2026-2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from aicup.metric import LEVELS  # noqa: E402
from aicup.sgf import handicap, parse_moves  # noqa: E402
from aicup_pid.hd import hd_class  # noqa: E402
from aicup_pid.local import tokens as local_tokens  # noqa: E402
from aicup_pid.opening import tokens as opening_tokens  # noqa: E402
from aicup_pid.orient import orient_tokens  # noqa: E402
from aicup_pid.style import FEATURE_NAMES, game_features  # noqa: E402

COMPS = ("opening", "style", "local", "orient", "hd")
RANK = {lv: i for i, lv in enumerate(LEVELS)}
NGROUP = {5: range(5, 8), 10: range(8, 13), 15: range(13, 18), 20: range(18, 21)}
DEPTH, ALPHA, BETA, ALPHA_FULL, N_POINTS = 12, 2.0, 5.0, 1.0, 362
_DCLS = {(-1, -1): 0, (4, 4): 1}
for _hi in range(4):
    for _lo in range(_hi + 1):
        _DCLS[(_hi, _lo)] = len(_DCLS)
V_LOCAL, V_ORIENT, V_HD = 12 * 12 * 5 * 3, 8 + 361, 3
D_STYLE = len(FEATURE_NAMES)
MOD = (1 << 61) - 1


def find_input(root, rel, name):
    for p in (Path(root) / rel / name, Path(root) / name, HERE / "inputs" / name, HERE / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}: not under {root}/{rel}, {root}, or {HERE}/inputs")


def swap(moves):
    return [("W" if c == "B" else "B", x, y) for c, x, y in moves]


def target_mark(sgf, color):
    """The record mark of the target's stones: the actual colour, except that HD >= 2 records mark the first mover (actual White) as B."""
    return color if (handicap(sgf) or 0) < 2 else ("W" if color == "B" else "B")


def _h(vals, seed):
    h = seed
    for v in vals:
        h = (h * 1000003 + v + 7) % MOD
    return h


def _pt(c):
    return c[0] * 20 + c[1]


def opening_keys(toks):
    """[t, 4] int64 hashes per Black opening move: (full ctx, mv), (Black ctx, mv), full ctx, Black ctx."""
    out = np.empty((len(toks), 4), dtype=np.int64)
    for i, (fctx, bctx, mv) in enumerate(toks):
        hf = _h([_pt(c) for c in fctx], 1); hb = _h([_pt(c) for c in bctx], 2); m = _pt(mv)
        out[i] = (_h([hf, m], 3), _h([hb, m], 4), hf, hb)
    return out


def local_ids(toks):
    return np.fromiter((((_DCLS[a] * 12 + _DCLS[b]) * 5 + (ln - 1)) * 3 + ph for a, b, ln, ph in toks), dtype=np.int16, count=len(toks))


def orient_ids(toks):
    out = []
    for t in toks:
        if t[0] == "sym":
            out.append(t[1])
        elif t[0] == "first":
            out.append(8 + t[1][0] * 19 + t[1][1])
    return np.array(out, dtype=np.int16)


def game_view(args):
    """(mark, opening keys [t,4], style features [31], local ids, orient ids, hd class) of one game seen from the target (target = B)."""
    sgf, color = args
    m = parse_moves(sgf); mark = target_mark(sgf, color)
    mo = m if mark == "B" else swap(m)
    return (mark, opening_keys(opening_tokens(mo, DEPTH)), game_features(mo).astype(np.float32), local_ids(local_tokens(mo, 12)), orient_ids(orient_tokens(mo, 12, 1)), int(hd_class(sgf)))


class PoolModel:
    """Competition 1's opening back-off model over all players, built only for the keys the queries need:
    p(mv | full) = (n(full, mv) + a_full p(mv | black)) / (n(full) + a_full); p(mv | black) = (n(black, mv) + alpha p_bg) / (n(black) + alpha)."""

    def __init__(self, C):
        self.C = C

    def fit(self, pl, keys, need):
        self.tabs = []
        for j in range(4):
            k = keys[:, j]; keep = np.isin(k, need[j]); k, p = k[keep], pl[keep]
            order = np.lexsort((p, k)); ks, ps = k[order], p[order]
            new = np.ones(len(ks), bool); new[1:] = (ks[1:] != ks[:-1]) | (ps[1:] != ps[:-1])
            starts = np.flatnonzero(new); cnt = np.diff(np.append(starts, len(ks))).astype(np.float64); uk, up = ks[starts], ps[starts]
            knew = np.ones(len(uk), bool); knew[1:] = uk[1:] != uk[:-1]; kstarts = np.flatnonzero(knew); kends = np.append(kstarts[1:], len(uk))
            self.tabs.append(({int(kv): (int(s), int(e)) for kv, s, e in zip(uk[kstarts], kstarts, kends)}, up, cnt))
        self.bg, self.bgT = {}, {}
        for j, d in ((1, self.bg), (3, self.bgT)):
            k = keys[:, j]; keep = np.isin(k, need[j]); u, c = np.unique(k[keep], return_counts=True); d.update(zip(u.tolist(), c.tolist()))
        return self

    def _dense(self, j, key):
        v = np.zeros(self.C); idx, up, cnt = self.tabs[j]; se = idx.get(key)
        if se is not None:
            v[up[se[0]:se[1]]] = cnt[se[0]:se[1]]
        return v

    def scores(self, keys):
        out = np.zeros(self.C)
        for fk, bk, fc, bc in keys.tolist():
            pb = (self.bg.get(bk, 0) + BETA / N_POINTS) / (self.bgT.get(bc, 0) + BETA)
            p_black = (self._dense(1, bk) + ALPHA * pb) / (self._dense(3, bc) + ALPHA)
            p_full = (self._dense(0, fk) + ALPHA_FULL * p_black) / (self._dense(2, fc) + ALPHA_FULL)
            out += np.log(p_full)
        return out


class Multi:
    """Per-player multinomial with a fixed vocabulary, shrunk towards the pool background (aicup_pid.local's form)."""

    def __init__(self, C, V, alpha=2.0, beta=5.0):
        self.C, self.V, self.alpha, self.beta = C, V, alpha, beta

    def fit(self, pl_tok, ids):
        counts = np.bincount(pl_tok.astype(np.int64) * self.V + ids.astype(np.int64), minlength=self.C * self.V).reshape(self.C, self.V).astype(np.float64)
        bg = counts.sum(0); p_bg = (bg + self.beta / self.V) / (bg.sum() + self.beta)
        self.logp = np.log((counts + self.alpha * p_bg) / (counts.sum(1, keepdims=True) + self.alpha))
        return self

    def scores(self, ids_list):
        if not ids_list:
            return np.zeros(self.C)
        return self.logp @ np.bincount(np.concatenate(ids_list).astype(np.int64), minlength=self.V)


class Gauss:
    """aicup_pid.style.StyleModel vectorised over players, means shrunk towards the pool mean with kappa pseudo-games."""

    def __init__(self, C, kappa=2.0, shrink=0.1):
        self.C, self.kappa, self.shrink = C, kappa, shrink

    def fit(self, pl, X):
        X = np.asarray(X, dtype=np.float64); D = X.shape[1]
        n = np.bincount(pl, minlength=self.C).astype(float)
        S1 = np.stack([np.bincount(pl, weights=X[:, j], minlength=self.C) for j in range(D)], axis=1)
        mu = S1 / np.maximum(n, 1)[:, None]
        cov = (X.T @ X - (mu * n[:, None]).T @ mu) / max(len(X) - int((n > 0).sum()), 1)
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov)); cov += np.eye(D) * 1e-9 * max(np.trace(cov) / D, 1e-12)
        self.prec = np.linalg.inv(cov); mu_bg = X.mean(0) if len(X) else np.zeros(D)
        self.mu = (S1 + self.kappa * mu_bg) / (n + self.kappa)[:, None]; self.n = n + self.kappa
        return self

    def scores(self, Q):
        if len(Q) == 0:
            return np.zeros(self.C)
        Q = np.asarray(Q, dtype=np.float64); m = len(Q)
        d = self.mu - Q.mean(0); scale = 1.0 / m + 1.0 / self.n
        return -0.5 * np.einsum("cd,de,ce->c", d, self.prec, d) / scale


class ViewPool:
    """All five components for one view (target = B first mover, or target = B second mover after the swap)."""

    def __init__(self, C):
        self.C = C

    def fit(self, views, idx, pl, need):
        pl = pl[idx]
        okeys = np.concatenate([views[i][1] for i in idx]) if len(idx) else np.zeros((0, 4), np.int64); opl = np.repeat(pl, [len(views[i][1]) for i in idx])
        self.open = PoolModel(self.C).fit(opl, okeys, need); del okeys, opl
        self.style = Gauss(self.C).fit(pl, np.stack([views[i][2] for i in idx]) if len(idx) else np.zeros((0, D_STYLE)))
        for name, V, col in (("local", V_LOCAL, 3), ("orient", V_ORIENT, 4)):
            ids = np.concatenate([views[i][col] for i in idx]) if len(idx) else np.zeros(0, np.int16); ptok = np.repeat(pl, [len(views[i][col]) for i in idx])
            setattr(self, name, Multi(self.C, V).fit(ptok, ids))
        self.hd = Multi(self.C, V_HD).fit(pl, np.array([views[i][5] for i in idx], dtype=np.int64))
        return self

    def scores(self, vs):
        out = {c: np.zeros(self.C) for c in COMPS}
        if not vs:
            return out
        for v in vs:
            out["opening"] += self.open.scores(v[1])
        out["style"] = self.style.scores(np.stack([v[2] for v in vs])); out["local"] = self.local.scores([v[3] for v in vs])
        out["orient"] = self.orient.scores([v[4] for v in vs]); out["hd"] = self.hd.scores([np.array([v[5]], dtype=np.int16) for v in vs])
        return out


def fit_scenario(views, marks, pl, include, need, C, log):
    t0 = time.time(); pools = {}
    for mark in ("B", "W"):
        idx = np.flatnonzero(include & (marks == mark)); pools[mark] = ViewPool(C).fit(views, idx, pl, need)
        log(f"   {mark}-view pool fitted on {len(idx)} games")
    log(f"   scenario fitted in {time.time() - t0:.0f} s")
    return pools


def score_questions(pools, qviews, C):
    S = {c: np.zeros((len(qviews), C)) for c in COMPS}
    for qi, vs in enumerate(qviews):
        for mark in ("B", "W"):
            sub = [v for v in vs if v[0] == mark]
            if sub:
                sc = pools[mark].scores(sub)
                for c in COMPS:
                    S[c][qi] += sc[c]
    return S


def standardise(S):
    S = np.asarray(S, dtype=np.float64); mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); sd[sd < 1e-12] = 1.0
    return (S - mu) / sd


def fit_weights(Zs, truth, l2=1e-3, iters=60):
    """One weight per component: maximise the multinomial log-likelihood of the true candidate under softmax(sum_j w_j Z_j) (Newton)."""
    Z = np.stack(Zs).astype(np.float32); J, Q, _ = Z.shape; w = np.ones(J); qi = np.arange(Q)
    for _ in range(iters):
        logits = np.einsum("j,jqc->qc", w, Z); logits -= logits.max(1, keepdims=True); P = np.exp(logits); P /= P.sum(1, keepdims=True)
        EZ = np.einsum("jqc,qc->jq", Z, P); Zt = Z[:, qi, truth]
        grad = (EZ - Zt).mean(1) + 2 * l2 * w
        A = (Z * P[None]).reshape(J, -1); H = (A @ Z.reshape(J, -1).T) / Q - (EZ @ EZ.T) / Q + 2 * l2 * np.eye(J)
        step = np.linalg.solve(H, grad); w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return w


def fuse(S, valid, w):
    return np.einsum("j,jqc->qc", np.asarray(w), np.stack([standardise(S[c][:, valid]) for c in COMPS]))


def top2(F):
    o = np.argpartition(-F, 1, axis=1)[:, :2]; a = np.take_along_axis(F, o, 1); sw = a[:, 1] > a[:, 0]
    o[sw] = o[sw][:, ::-1]; a[sw] = a[sw][:, ::-1]
    return o[:, 0], o[:, 1], a[:, 0] - a[:, 1]


def label_post(F, lab_valid, K=len(LEVELS)):
    Z = F - F.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return np.stack([np.bincount(lab_valid, weights=P[q], minlength=K) for q in range(len(F))])


def fit_p(obs, f_true, f_imp):
    ts = np.quantile(f_imp, [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]); A, y = [], []
    for t in ts:
        A.append((f_true > t).mean() - (f_imp > t).mean()); y.append((obs > t).mean() - (f_imp > t).mean())
    A, y = np.array(A), np.array(y)
    return float(np.clip((A @ y) / max(A @ A, 1e-12), 0, 1))


def read_tests(paths):
    qs = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        for d in df.to_dict("records"):
            n = int(d["num_games"]); games = [(d.get(f"sgf_{i}", ""), d.get(f"color_{i}", "")) for i in range(1, n + 1)]
            qs.append({"question_id": d["question_id"], "n": n, "games": [(s, c) for s, c in games if s and c in ("B", "W")]})
    return qs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--root", default=str(ROOT)); ap.add_argument("--gate", default=None); ap.add_argument("--ct7", default=None); ap.add_argument("--limit-players", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(0); root = Path(a.root)
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    gate_path = Path(a.gate) if a.gate else find_input(root, "results/r_variants", "prod_heldout.P.npz"); ct7_path = Path(a.ct7) if a.ct7 else find_input(root, ".", "submission_rank_ct7.csv")
    train_files = sorted((root / "data/training").glob("train_*.csv"))
    if not train_files:
        raise SystemExit(f"no data/training/train_*.csv under {root}")
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in train_files], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    z = np.load(gate_path, allow_pickle=True); gate_keys = [json.loads(k) for k in z["keys"]]; gy, gn, gpid = z["y"].astype(int), z["n"].astype(int), z["pid"]
    gate_games = frozenset(k for ks in gate_keys for k in ks); gate_players = frozenset(gpid.tolist())
    if a.limit_players:
        others = np.array(sorted(set(rows.player_id) - gate_players)); keep = set(rng.choice(others, min(a.limit_players, len(others)), replace=False).tolist()) | gate_players
        rows = rows[rows.player_id.isin(keep)].reset_index(drop=True)
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)
    plabel = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k: i for i, k in enumerate(keys)}
    log(f"{len(rows)} rows, {C} players; gate: {len(gate_keys)} questions, {len(gate_players)} players, {len(gate_games)} games; workers {a.workers}")
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    with ctx.Pool(a.workers) as pool:
        views = list(pool.imap(game_view, zip(rows.sgf_content.tolist(), rows.color.tolist()), chunksize=512))
    rows = rows.drop(columns=["sgf_content"]); marks = np.array([v[0] for v in views]); log(f"views for {len(views)} games")
    questions = read_tests([root / "data/tests/rank_prediction_test_public.csv", root / "data/tests/rank_prediction_test_private.csv"])
    with ctx.Pool(a.workers) as pool:
        tviews = list(pool.imap(game_view, [g for q in questions for g in q["games"]], chunksize=256))
    test_views, pos = [], 0
    for q in questions:
        test_views.append(tviews[pos:pos + len(q["games"])]); pos += len(q["games"])
    test_n = np.array([q["n"] for q in questions]); test_qid = np.array([q["question_id"] for q in questions]); log(f"test views: {len(tviews)} games, {len(questions)} questions")
    gate_views = [[views[key_index[k]] for k in ks] for ks in gate_keys]; truth = np.array([pidx[p] for p in gpid])
    need = [np.unique(np.concatenate([v[1][:, j] for vs in gate_views + test_views for v in vs])) for j in range(4)]
    rep = {"players": C, "rows": int(len(rows)), "gate_questions": len(gate_keys), "gate_players": len(gate_players), "comps": list(COMPS)}
    # ---- A: in-pool
    log("A: fitting without the gate questions' games")
    pools = fit_scenario(views, marks, pl, ~rows.game_key.isin(gate_games).to_numpy(), need, C, log)
    SA = score_questions(pools, gate_views, C); del pools; log("A: scored")
    valid = np.ones(C, bool)
    half = np.array([int(p[-1], 16) % 2 if p[-1] in "0123456789abcdef" else len(p) % 2 for p in gpid])
    ws, FA = [], np.zeros((len(gate_keys), C))
    for h in (0, 1):
        fit_q = np.flatnonzero(half == h); app_q = np.flatnonzero(half != h)
        w = fit_weights([standardise(SA[c][fit_q]) for c in COMPS], truth[fit_q]); ws.append(w.tolist())
        FA[app_q] = fuse({c: SA[c][app_q] for c in COMPS}, valid, w)
    w_mean = np.mean(ws, axis=0); rep["fusion_weights_halves"] = ws; rep["fusion_weights_mean"] = w_mean.tolist()
    A1, A2, Am = top2(FA); A_correct = A1 == truth; A_post = label_post(FA, plabel)
    comp_top1 = {c: float((np.argmax(SA[c], 1) == truth).mean()) for c in COMPS}
    rep["A"] = {"top1_by_component": comp_top1, "fused_top1": float(A_correct.mean()), "fused_top5": float(np.mean([(FA[q] > FA[q, truth[q]]).sum() < 5 for q in range(len(truth))])),
                "fused_top1_by_n_group": {str(g): float(A_correct[np.isin(gn, list(r))].mean()) for g, r in NGROUP.items()},
                "median_margin_correct": float(np.median(Am[A_correct])) if A_correct.any() else None, "median_margin_wrong": float(np.median(Am[~A_correct])) if (~A_correct).any() else None,
                "wrong_match_label_equal": float((plabel[A1[~A_correct]] == gy[~A_correct]).mean()) if (~A_correct).any() else None,
                "wrong_match_label_within1": float((np.abs(plabel[A1[~A_correct]] - gy[~A_correct]) <= 1).mean()) if (~A_correct).any() else None}
    log(f"A: top-1 by component {json.dumps({k: round(v, 3) for k, v in comp_top1.items()})}; fused top-1 {A_correct.mean():.3f}, by n group {json.dumps({k: round(v, 3) for k, v in rep['A']['fused_top1_by_n_group'].items()})}; weights {np.round(w_mean, 2).tolist()}")
    # ---- B: out-of-pool
    log("B: fitting without the gate players")
    pools = fit_scenario(views, marks, pl, ~rows.player_id.isin(gate_players).to_numpy(), need, C, log)
    SB = score_questions(pools, gate_views, C); del pools
    validB = ~np.isin(players, list(gate_players)); vB = np.flatnonzero(validB)
    FB = fuse(SB, validB, w_mean); B1v, _, Bm = top2(FB); B1 = vB[B1v]; B_post = label_post(FB, plabel[validB])
    rep["B"] = {"candidates": int(validB.sum()), "margin_quantiles": {str(q): float(np.quantile(Bm, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
                "label_equal_all": float((plabel[B1] == gy).mean()), "label_within1_all": float((np.abs(plabel[B1] - gy) <= 1).mean())}
    for share in (0.01, 0.02, 0.05, 0.1):
        top = Bm >= np.quantile(Bm, 1 - share)
        rep["B"][f"top{int(share * 100)}pct_label_equal"] = float((plabel[B1[top]] == gy[top]).mean()); rep["B"][f"top{int(share * 100)}pct_label_within1"] = float((np.abs(plabel[B1[top]] - gy[top]) <= 1).mean())
    rep["thresholds"] = {}
    for g, r in NGROUP.items():
        sel = np.isin(gn, list(r)); d = {}
        for q in (0.9, 0.95, 0.99, 0.995):
            t = float(np.quantile(Bm[sel], q)); above = Am[sel] > t
            d[str(q)] = {"t": t, "A_share_above": float(above.mean()), "A_precision_above": float(A_correct[sel][above].mean()) if above.any() else None, "A_recall": float((above & A_correct[sel]).mean())}
        rep["thresholds"][str(g)] = d
    log("B: margin q50/q90/q99 {:.2f}/{:.2f}/{:.2f}; top-1% matches label-equal {:.3f} within-1 {:.3f} (all B: {:.3f}/{:.3f}); A (precision, recall) above B q99 by n group {}".format(
        rep["B"]["margin_quantiles"]["0.5"], rep["B"]["margin_quantiles"]["0.9"], rep["B"]["margin_quantiles"]["0.99"], rep["B"]["top1pct_label_equal"], rep["B"]["top1pct_label_within1"], rep["B"]["label_equal_all"], rep["B"]["label_within1_all"],
        json.dumps({g: (None if d["0.99"]["A_precision_above"] is None else round(d["0.99"]["A_precision_above"], 3), round(d["0.99"]["A_recall"], 3)) for g, d in rep["thresholds"].items()})))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".gate.npz"), y=gy, n=gn, pid=gpid, half=half, players=players, plabel=plabel, truth=truth,
                        A_top1=A1, A_top2=A2, A_margin=Am, A_correct=A_correct, A_post=A_post, A_top1_label=plabel[A1],
                        B_top1=B1, B_margin=Bm, B_post=B_post, B_top1_label=plabel[B1], w=w_mean)
    # ---- C: test
    log("C: fitting on all games")
    pools = fit_scenario(views, marks, pl, np.ones(len(rows), bool), need, C, log)
    ST = score_questions(pools, test_views, C); del pools
    FT = fuse(ST, valid, w_mean); T1, T2, Tm = top2(FT); T_post = label_post(FT, plabel)
    ct7 = pd.read_csv(ct7_path, dtype=str).set_index("question_id")["pred_rank"].reindex(test_qid).map(RANK).to_numpy()
    rep["test"] = {"questions": int(len(test_views)), "games": int(len(tviews)), "mixture_in_pool_share_by_n": {}, "ct7_agreement_top1_label_by_margin_decile": {}}
    for g, r in NGROUP.items():
        sel = np.isin(gn, list(r)); obs = Tm[test_n == g]
        rep["test"]["mixture_in_pool_share_by_n"][str(g)] = fit_p(obs, Am[sel], Bm[sel]) if len(obs) else None
    order = np.argsort(-Tm)
    for i, idx in enumerate(np.array_split(order, 10)):
        rep["test"]["ct7_agreement_top1_label_by_margin_decile"][str(i)] = {"margin_min": float(Tm[idx].min()), "label_equal_ct7": float((plabel[T1[idx]] == ct7[idx]).mean()), "within1": float((np.abs(plabel[T1[idx]] - ct7[idx]) <= 1).mean())}
    for g in NGROUP:
        tB = rep["thresholds"][str(g)]["0.99"]["t"]; sel = test_n == g
        rep["test"][f"n{g}_share_above_B_q99"] = float((Tm[sel] > tB).mean()) if sel.any() else None
    log("C: mixture in-pool share by n {}; share above B q99 by n {}; label(top-1) = ct7 in the top margin decile {:.3f} vs bottom {:.3f}".format(
        json.dumps({k: (None if v is None else round(v, 3)) for k, v in rep["test"]["mixture_in_pool_share_by_n"].items()}), json.dumps({g: rep["test"][f"n{g}_share_above_B_q99"] for g in NGROUP}),
        rep["test"]["ct7_agreement_top1_label_by_margin_decile"]["0"]["label_equal_ct7"], rep["test"]["ct7_agreement_top1_label_by_margin_decile"]["9"]["label_equal_ct7"]))
    np.savez_compressed(out.with_suffix(".test.npz"), qid=test_qid, n=test_n, top1=T1, top2=T2, margin=Tm, post=T_post, top1_label=plabel[T1], top1_player=players[T1], players=players, plabel=plabel, w=w_mean)
    out.write_text(json.dumps(rep, indent=1, default=float)); log(f"written {out}")


if __name__ == "__main__":
    main()
