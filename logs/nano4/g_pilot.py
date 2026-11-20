"""Section 4M M28 (2026-11-06; user: "bert 有試過嗎" -> "好"): the BERT-style "future-move prediction gain" G of section 6.5.
A masked-move Transformer predicts the actual move at an anchor ply from (a) the pre-move board's pooled trunk vector g_t and the past
<= 32 plies' moves, and (b) optionally some FUTURE moves (raw coordinates only). Views at the anchor (the mover's own moves vs the
opponent's in the next 8 plies):
  past     no future tokens
  own      the mover's own next 4 moves (offsets +2, +4, +6, +8 when the colours alternate)
  opp      the opponent's next 4 moves
  unord    own's coordinates with their offsets replaced by one 'no order' index (same set, time order removed)
  both     all 8 (training only)
G_own = log p(a_t | own) - log p(a_t | past); G_opp, G_unord likewise; G_order = own - unord isolates the time-order part.
Anchors whose answer point recurs in the next 8 plies (ko recaptures, passes) are excluded from training and evaluation.
No shortcuts: the anchor's played-point vector l_t, any future board, rank and player id never enter; the masked move is only the label.
Training: folds 0, 2, 3, 4 base-sample plies (results/trunk_vec), random anchors, random view per sample, random board symmetry on the
coordinates (g_t is a board-pooled vector, treated as invariant). Evaluation: M23's fold-1 sample (60 players per class, 20 games).
  python -I logs/nano4/g_pilot.py --out results/g_pilot            (train + per-game features + player-level evaluation)
  python -I logs/nano4/g_pilot.py --out results/g_pilot --smoke    (a few thousand steps on fold 2 only, for a quick check)
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402

PAST, FUT = 32, 8
L = PAST + 1 + FUT
PASS, MASK, PAD = 361, 362, 363
VIEWS = {"past": 0, "own": 1, "opp": 2, "unord": 3, "both": 4}
UNORD = L  # the offset index of an own future token whose time order was removed (unord view)
RANK = {lv: i for i, lv in enumerate(LEVELS)}


def sym_table():
    """[8, 362] the tensor position of each point under the 8 board symmetries (pass maps to pass)."""
    T = np.zeros((8, 362), dtype=np.int64)
    for k in range(8):
        for p in range(361):
            x, y = p % 19, p // 19
            if k & 1:
                x = 18 - x
            if k & 2:
                y = 18 - y
            if k & 4:
                x, y = y, x
            T[k, p] = y * 19 + x
        T[k, PASS] = PASS
    return T


def load_plies(folds, keys=None, log=print):
    """Concatenated per-ply arrays of results/trunk_vec/fold<f> (only games in `keys` when given), ordered by game then t."""
    P, M, TG, G, GK, GI, meta = [], [], [], [], [], [], []
    gid = 0
    for f in folds:
        for npz in sorted(glob.glob(str(ROOT / f"results/trunk_vec/fold{f}/moves_s*_p*.npz"))):
            csv = npz.replace("moves_", "games_").replace(".npz", ".csv")
            games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str})
            z = np.load(npz)
            row = z["game_row"]
            keep_rows = np.arange(len(games)) if keys is None else np.flatnonzero(games.game_key.isin(keys).to_numpy())
            if len(keep_rows) == 0:
                continue
            sel = np.isin(row, keep_rows)
            order = np.lexsort((z["t"][sel], row[sel]))
            r = row[sel][order]
            P.append(z["pos"][sel][order].astype(np.int16)); M.append(z["mover"][sel][order].astype(np.int8)); TG.append(z["target"][sel][order])
            G.append(z["g"][sel][order].astype(np.float16)); GI.append((np.searchsorted(keep_rows, r) + gid).astype(np.int32))
            sub = games.iloc[keep_rows]
            meta.append(sub.assign(gid=np.arange(gid, gid + len(sub))))
            gid += len(keep_rows)
        log(f"fold {f}: {gid} games so far")
    pos, mover, tgt, g, game = (np.concatenate(x) for x in (P, M, TG, G, GI))
    starts = np.zeros(gid + 1, dtype=np.int64)
    np.add.at(starts, game + 1, 1)
    starts = np.cumsum(starts)
    return {"pos": pos, "mover": mover, "target": tgt, "g": g, "game": game, "start": starts[:-1][game], "end": starts[1:][game], "meta": pd.concat(meta, ignore_index=True)}


class GModel(torch.nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ffn=512, dropout=0.1):
        super().__init__()
        self.act = torch.nn.Embedding(364, d); self.side = torch.nn.Embedding(3, d); self.off = torch.nn.Embedding(L + 1, d)
        self.gproj = torch.nn.Linear(64, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = torch.nn.LayerNorm(d); self.head = torch.nn.Linear(d, 362)

    def forward(self, act, side, off, g, pad):
        x = self.act(act) + self.side(side) + self.off(off)
        x[:, PAST] = x[:, PAST] + self.gproj(g)
        h = self.enc(x, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, PAST]))


def make_batch(D, idx, views, sym_k, SYM, gen=None):
    """Token tensors for anchors idx (global ply indices) under per-sample views and symmetries, plus view-independent anchor flags:
    rep (the answer's point recurs among the next 8 plies: ko recaptures, re-plays, passes -> excluded everywhere), n_own / n_opp
    (future plies of the mover / the opponent within the next 8)."""
    dev = idx.device
    B = len(idx)
    pos, mover, start, end = D["pos_t"], D["mover_t"], D["start_t"], D["end_t"]
    rel = torch.arange(-PAST, FUT + 1, device=dev)  # slot -> offset
    j = idx[:, None] + rel[None, :]
    valid = (j >= start[idx][:, None]) & (j < end[idx][:, None])
    jc = j.clamp(0, len(pos) - 1)
    act = torch.where(valid, pos[jc].long(), torch.full_like(jc, PAD))
    same = mover[jc] == mover[idx][:, None]
    side = torch.where(valid, (~same).long(), torch.full_like(jc, 2))
    fut = rel > 0
    own_f = valid & fut[None, :] & same
    opp_f = valid & fut[None, :] & ~same
    rep_ = (valid & fut[None, :] & (pos[jc].long() == pos[idx].long()[:, None])).any(1)
    n_own, n_opp = own_f.sum(1), opp_f.sum(1)
    v = views[:, None]
    keep_fut = torch.where(v == VIEWS["past"], torch.zeros_like(own_f), torch.where((v == VIEWS["own"]) | (v == VIEWS["unord"]), own_f,
               torch.where(v == VIEWS["opp"], opp_f, own_f | opp_f)))
    keep = (valid & ~fut[None, :]) | keep_fut
    keep[:, PAST] = True
    act[:, PAST] = MASK
    off = (rel.expand(B, -1) + PAST).clone()
    off = torch.where((v == VIEWS["unord"]) & own_f, torch.full_like(off, UNORD), off)  # unord: coordinates kept, time order removed
    pad = ~keep
    act = torch.where(keep, act, torch.full_like(act, PAD)); side = torch.where(keep, side, torch.full_like(side, 2))
    is_pt = act < 362  # symmetry on coordinates (pad / mask untouched)
    act = torch.where(is_pt, torch.gather(SYM[sym_k], 1, act.clamp(max=361)), act)
    label = torch.gather(SYM[sym_k], 1, pos[idx].long()[:, None])[:, 0]
    return act, side, off, D["g_t"][idx].float(), pad, label, rep_, n_own, n_opp


def train(D, a, dev, log):
    torch.manual_seed(0); gen = torch.Generator(device=dev); gen.manual_seed(0)
    model = GModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n = len(D["pos"]); steps_per_epoch = a.anchors // a.batch; total = steps_per_epoch * a.epochs
    warm = min(2000, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    SYM = torch.as_tensor(sym_table(), device=dev)
    probs = torch.tensor([0.3, 0.2, 0.2, 0.15, 0.15], device=dev)  # past, own, opp, unord, both
    held = torch.randint(0, n, (100_000,), device=dev, generator=gen)
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, n, (a.batch,), device=dev, generator=gen)
            views = torch.multinomial(probs, a.batch, replacement=True, generator=gen)
            symk = torch.randint(0, 8, (a.batch,), device=dev, generator=gen)
            act, side, off, g, pad, label, rp, _, _ = make_batch(D, idx, views, symk, SYM, gen)
            w = (~rp).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce = torch.nn.functional.cross_entropy(model(act, side, off, g, pad).float(), label, reduction="none")
            loss = (ce * w).sum() / w.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 1000 == 0:
                log(f"epoch {ep} step {step}/{total} loss {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            res = {}
            for vname in ("past", "own", "opp", "unord"):
                nll = []
                for b0 in range(0, len(held), 4096):
                    idx = held[b0:b0 + 4096]; views = torch.full((len(idx),), VIEWS[vname], device=dev); symk = torch.zeros(len(idx), dtype=torch.long, device=dev)
                    act, side, off, g, pad, label, rp, _, _ = make_batch(D, idx, views, symk, SYM, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        nll.append(torch.nn.functional.cross_entropy(model(act, side, off, g, pad).float(), label, reduction="none")[~rp])
                res[vname] = float(torch.cat(nll).mean())
        log(f"epoch {ep} done ({time.time() - t0:.0f} s): NLL on 100k training anchors " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()))
    return model, res


def to_device(D, dev):
    D["pos_t"] = torch.as_tensor(D["pos"], device=dev); D["mover_t"] = torch.as_tensor(D["mover"], device=dev)
    D["start_t"] = torch.as_tensor(D["start"], device=dev); D["end_t"] = torch.as_tensor(D["end"], device=dev)
    D["g_t"] = torch.as_tensor(D["g"], device=dev)
    return D


def game_features(model, D, dev):
    """Per-ply log p of the actual move under past / own / opp / unord, on anchors with 4 own + 4 opponent future plies and no recurrence
    of the answer's point; per game and side the means of nll_past, g_own, g_opp, g_unord, g_order (= own - unord), nll_own, g_own_pos."""
    SYM = torch.as_tensor(sym_table(), device=dev)
    n = len(D["pos"]); lp = {k: np.zeros(n, dtype=np.float32) for k in ("past", "own", "opp", "unord")}
    rep_a, nown, nopp = np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.int64)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, n, 8192):
            idx = torch.arange(b0, min(n, b0 + 8192), device=dev); symk = torch.zeros(len(idx), dtype=torch.long, device=dev)
            for vname in ("past", "own", "opp", "unord"):
                act, side, off, g, pad, label, rp, no, nq = make_batch(D, idx, torch.full((len(idx),), VIEWS[vname], device=dev), symk, SYM)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = torch.log_softmax(model(act, side, off, g, pad).float(), 1).gather(1, label[:, None])[:, 0]
                lp[vname][b0:b0 + len(idx)] = out.cpu().numpy()
            rep_a[b0:b0 + len(idx)] = rp.cpu().numpy(); nown[b0:b0 + len(idx)] = no.cpu().numpy(); nopp[b0:b0 + len(idx)] = nq.cpu().numpy()
    ok = (nown == 4) & (nopp == 4)
    df = pd.DataFrame({"gid": D["game"], "target": D["target"], "ok": ok, "rep": rep_a, **lp})
    df["rank"] = D["meta"].set_index("gid").loc[df.gid, "rank"].to_numpy()
    rep_share = df[df.ok].groupby("rank")["rep"].mean().to_dict()
    use = df.ok & ~df.rep
    eval_nll = {k: float(-df.loc[use, k].mean()) for k in ("past", "own", "opp", "unord")}
    eval_nll.update({"anchors_used": int(use.sum()), "anchors_full_window": int(df.ok.sum()), "rep_share_by_rank": rep_share})
    df = df[use].copy()
    df["g_own"], df["g_opp"], df["g_unord"] = df.own - df.past, df.opp - df.past, df.unord - df.past
    df["g_order"] = df.own - df.unord
    rows = []
    for (gid, tg), sub in df.groupby(["gid", "target"]):
        p = "t_" if tg else "o_"
        rows.append({"gid": gid, p + "nll_past": -sub.past.mean(), p + "g_own": sub.g_own.mean(), p + "g_opp": sub.g_opp.mean(), p + "g_unord": sub.g_unord.mean(),
                     p + "g_order": sub.g_order.mean(), p + "nll_own": -sub.own.mean(), p + "g_own_pos": (sub.g_own > 0.5).mean()})
    per = pd.DataFrame(rows).groupby("gid").first().reset_index()
    return D["meta"].merge(per, on="gid", how="left"), eval_nll


def evaluate_players(G, out, log):
    from ai_quality_eval import ridge_oof
    from oppview_eval import boot, partial
    from pilot_stats import spearman
    feats = [f"{p}{c}" for p in ("t_", "o_") for c in ("nll_past", "g_own", "g_opp", "g_unord", "g_order")]   # the registered 10 columns
    report_only = [f"{p}{c}" for p in ("t_", "o_") for c in ("nll_own", "g_own_pos")]
    G["y"] = G["rank"].map(RANK)
    g = G.groupby("player_id").agg({**{c: "mean" for c in feats + report_only}, "y": "first"})
    meta = pd.DataFrame(json.load(open(ROOT / "results/trunk_seq/r_val1.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val1/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    g["E"] = E.reindex(g.index); g = g.dropna(subset=["E"]); y = g.y.to_numpy(float); Ev = g.E.to_numpy(float)
    rep = {"players": int(len(g)), "features": {}}
    for c in feats + report_only:
        x = g[c].to_numpy(float); ok = np.isfinite(x)
        pe = partial(x[ok], y[ok], [Ev[ok]]); ci = boot(lambda idx: partial(x[ok][idx], y[ok][idx], [Ev[ok][idx]]), int(ok.sum()))
        rep["features"][c] = {"spearman_y": spearman(x[ok], y[ok]), "partial_given_E": pe, "ci95": ci}
        log(f"{c:14s} Spearman y {rep['features'][c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    X = g[feats].to_numpy(float); X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))
    comp = ridge_oof(X, y); est = partial(comp, y, [Ev]); ci = boot(lambda idx: partial(comp[idx], y[idx], [Ev[idx]]), len(y))
    rep["composite"] = {"features": feats, "spearman_y": spearman(comp, y), "partial_given_E": est, "ci95": ci}
    rep["continue"] = bool(ci[0] > 0 and est >= 0.08)
    sub = ["t_g_order", "t_nll_past", "o_g_order", "o_nll_past"]; Xs = np.where(np.isfinite(g[sub].to_numpy(float)), g[sub].to_numpy(float), 0.0)
    cs = ridge_oof(Xs, y); es_ = partial(cs, y, [Ev]); cis = boot(lambda idx: partial(cs[idx], y[idx], [Ev[idx]]), len(y))
    rep["composite_order"] = {"features": sub, "spearman_y": spearman(cs, y), "partial_given_E": es_, "ci95": cis}
    log(f"order sub-composite {sub}: Spearman y {rep['composite_order']['spearman_y']:.3f}  partial|E {es_:+.3f} [{cis[0]:+.3f}, {cis[1]:+.3f}]")
    log(f"G composite ({len(feats)} features): Spearman y {rep['composite']['spearman_y']:.3f}  partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
        f"continue (CI lower > 0 and >= 0.08): {'YES' if rep['continue'] else 'no'}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--anchors", type=int, default=8_000_000); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    if a.smoke:
        a.anchors, a.epochs = 2_000_000, 1
    D = to_device(load_plies([2] if a.smoke else [0, 2, 3, 4], log=log), dev)
    log(f"training plies {len(D['pos'])}, games {len(D['meta'])}")
    model, held = train(D, a, dev, log)
    torch.save(model.state_dict(), out / ("g_model_smoke.pt" if a.smoke else "g_model.pt"))
    del D; torch.cuda.empty_cache()
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, 1, max_games=20); players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:60]] for _, grp in players.groupby("rank")])
    keys = set(k for p in pick.itertuples() for k in p.game_keys)
    if a.smoke:
        pick = pd.concat([grp.iloc[:2] for _, grp in pick.groupby("rank")])
        keys = set(k for p in pick.itertuples() for k in p.game_keys)
    E = to_device(load_plies([1], keys=keys, log=log), dev)
    log(f"evaluation plies {len(E['pos'])}, games {len(E['meta'])} of {len(keys)} keys")
    G, eval_nll = game_features(model, E, dev)
    log("evaluation-set NLL by view (4 own + 4 opponent future plies, no answer recurrence): " + "  ".join(f"{k} {eval_nll[k]:.4f}" for k in ("past", "own", "opp", "unord"))
        + f"; anchors used {eval_nll['anchors_used']} of {eval_nll['anchors_full_window']}; answer-recurrence share by rank " + json.dumps({k: round(v, 4) for k, v in eval_nll["rep_share_by_rank"].items()}))
    G.to_csv(out / ("games_smoke.csv" if a.smoke else "games.csv"), index=False)
    rep = {"train_sample_nll": held, "eval_nll": eval_nll, "smoke": a.smoke}
    sane = eval_nll["past"] < math.log(362) - 1 and eval_nll["own"] < eval_nll["past"] and eval_nll["opp"] < eval_nll["past"]
    rep["sanity"] = bool(sane); log(f"sanity (past NLL < ln 362 - 1, own and opp below past): {'PASS' if sane else 'FAIL'}")
    log("per-game feature means " + json.dumps({c: round(float(G[c].mean()), 4) for c in G.columns if c[:2] in ("t_", "o_")}))
    rep.update(evaluate_players(G, out, log))
    (ROOT / f"results/r_variants/g_pilot{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(rep, indent=1, default=float)); log("written")


if __name__ == "__main__":
    main()
