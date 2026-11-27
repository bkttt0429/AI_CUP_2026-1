"""Section 4M M77 (2026-11-10): whole-game bidirectional encoder that recognises WHICH training player one side of a game is.
Input: the first 256 plies as tokens (point, side relative to the subject, absolute colour, ply index, offset to the previous ply,
offset to the subject side's previous ply) + [CLS]; pre-norm Transformer encoder; closed-set classification over the training players.
--exclude-fold v (1-4): every row of fold v's players is left out (cross-fitting: their games are never seen, so recognising the
opponents in them is as on the test set); -1: all rows (the test model). One held-out game per player (>= 8 games) measures the
identification itself. Outputs under results/m77/: val<v>.npz (the validation players' 20-game questions: opponent side and target
side = impostor) or test.npz, heldout_<v>.json, model_<v>.pt.
  python -I scripts/m_series/m77_idbert.py --exclude-fold 1
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path.cwd()
OUT = ROOT / "results/m77"


class Enc(nn.Module):
    def __init__(self, n_classes, d=256, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.pos, self.side, self.col, self.t = nn.Embedding(363, d), nn.Embedding(3, d), nn.Embedding(3, d), nn.Embedding(256, d)
        self.o1, self.o2 = nn.Embedding(1370, d), nn.Embedding(1370, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.ln = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    @staticmethod
    def offset(pos, k):
        prev = torch.roll(pos, k, 1); ok = (pos < 361) & (prev < 361); ok[:, :k] = False
        return torch.where(ok, (pos % 19 - prev % 19 + 18) * 37 + (pos // 19 - prev // 19 + 18), torch.full_like(pos, 1369))

    def forward(self, pos, col, subj):
        pad = pos == 362
        side = torch.where(pad, torch.full_like(col, 2), (col != subj[:, None]).long())
        h = self.pos(pos) + self.side(side) + self.col(col) + self.t(torch.arange(pos.shape[1], device=pos.device))[None] + self.o1(self.offset(pos, 1)) + self.o2(self.offset(pos, 2))
        h = torch.cat([self.cls.expand(len(pos), -1, -1), h], 1); mask = torch.cat([torch.zeros(len(pos), 1, dtype=torch.bool, device=pos.device), pad], 1)
        return self.out(self.enc(self.ln(h), src_key_padding_mask=mask)[:, 0])


def infer(model, POS, COL, subj, idx, batch=1024):
    """top-20 (class index, logit) of every row in idx with the given subject colours."""
    model.eval(); tops, vals = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), batch):
            j = torch.as_tensor(idx[i:i + batch], device="cuda")
            lg = model(POS[j].long(), COL[j].long(), torch.as_tensor(subj[i:i + batch], device="cuda").long()).float()
            v, k = lg.topk(20, dim=1); tops.append(k.cpu().numpy()); vals.append(v.cpu().numpy())
    return np.concatenate(tops), np.concatenate(vals)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exclude-fold", type=int, required=True); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); v = a.exclude_fold; t0 = time.time(); torch.manual_seed(20261110 + v); rng = np.random.default_rng(20261110 + v)
    z = np.load(OUT / "data.npz", allow_pickle=False)
    player, fold, hd, subj, plab = z["player"], z["fold"], z["hd"], z["subj"], z["plab"]
    POS, COL = torch.as_tensor(z["POS"].astype(np.int16), device="cuda"), torch.as_tensor(z["COL"], device="cuda")
    allowed = (hd < 2) & (fold != v)
    by = {}
    for r in np.flatnonzero(allowed):
        by.setdefault(int(player[r]), []).append(r)
    cls_players = np.array(sorted(by)); cmap = np.full(len(z["players"]), -1); cmap[cls_players] = np.arange(len(cls_players))
    held = np.array([rng.choice(rs) for rs in by.values() if len(rs) >= 8]); is_held = np.zeros(len(player), bool); is_held[held] = True
    train = np.flatnonzero(allowed & ~is_held)
    if a.limit:
        train = rng.choice(train, a.limit, replace=False)
    y_all = torch.as_tensor(cmap[player], device="cuda")
    model = Enc(len(cls_players), a.d, a.layers).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(train) // a.batch); warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps, 1)))))
    print(f"exclude fold {v}: {len(train)} training games, {len(cls_players)} classes, {len(held)} held-out games; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    best, best_state, hist = -1.0, None, []
    for ep in range(a.epochs):
        model.train(); perm = torch.as_tensor(rng.permutation(train), device="cuda"); tot, nb = 0.0, 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            j = perm[i:i + a.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(POS[j].long(), COL[j].long(), torch.as_tensor(subj, device="cuda")[j].long())
            loss = nn.functional.cross_entropy(lg.float(), y_all[j], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += float(loss.detach()); nb += 1
        tk, _ = infer(model, POS, COL, subj[held], held); truth = cmap[player[held]]
        acc = {k: float((tk[:, :k] == truth[:, None]).any(1).mean()) for k in (1, 5, 20)}; hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), **{f"top{k}": x for k, x in acc.items()}})
        print(f"  epoch {ep + 1}: loss {tot / max(nb, 1):.3f}; held-out own games top-1 {acc[1]:.3f} top-5 {acc[5]:.3f} top-20 {acc[20]:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if acc[1] > best:
            best, best_state = acc[1], {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save({"state": best_state, "classes": cls_players, "args": vars(a)}, OUT / f"model_{v}.pt")
    json.dump({"exclude_fold": v, "classes": int(len(cls_players)), "held": int(len(held)), "best_top1": best, "history": hist}, open(OUT / f"heldout_{v}.json", "w"), indent=1)

    def side_pack(idx, subject):
        tk, vl = infer(model, POS, COL, subject, idx); best_p = cls_players[tk[:, 0]]
        return plab[best_p], (vl[:, 0] - vl[:, 1]).astype(np.float32), best_p
    if v > 0:  # the validation players' n = 20 questions of fold v
        key_row = {k: i for i, k in enumerate(z["key"])}; m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]; rows, pids, ys = [], [], []
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):
            if ng == 20:
                for k in gk:
                    rows.append(key_row[k]); pids.append(pid); ys.append(int(plab[player[key_row[k]]]))
        rows = np.array(rows); ol, om, ob = side_pack(rows, 1 - subj[rows]); il, im, _ = side_pack(rows, subj[rows])
        np.savez_compressed(OUT / f"val{v}.npz", key=z["key"][rows], pid=np.array(pids), fold=np.full(len(rows), v), y=np.array(ys), hd=hd[rows], target_black=subj[rows] == 0,
                            opp_lab=ol, opp_m=om, opp_best=z["players"][ob], imp_lab=il, imp_m=im)
        print(f"validation fold {v}: {len(rows)} queries written ({time.time() - t0:.0f} s)")
    else:  # the test games
        TPOS, TCOL = torch.as_tensor(z["TPOS"].astype(np.int16), device="cuda"), torch.as_tensor(z["TCOL"], device="cuda"); ts = z["tsubj"]; idx = np.arange(len(ts))
        def tp(subject):
            tk, vl = infer(model, TPOS, TCOL, subject, idx); bp = cls_players[tk[:, 0]]; return plab[bp], (vl[:, 0] - vl[:, 1]).astype(np.float32), bp
        ol, om, ob = tp(1 - ts); tl, tm, tb = tp(ts)
        np.savez_compressed(OUT / "test.npz", qid=z["tqid"], hd=z["thd"], target_black=ts == 0, opp_lab=ol, opp_m=om, opp_best=z["players"][ob], tgt_lab=tl, tgt_m=tm, tgt_best=z["players"][tb])
        print(f"test: {len(idx)} games written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
