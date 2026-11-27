"""Section 4M M99 stage 0 (2026-11-13, identification metric only, no level labels): matching against a candidate's individual games
instead of one mean vector per candidate, and a mutual-nearest-neighbour filter (person re-identification's k-reciprocal idea).
Queries = the 3,244 held-out own games (true owner known), all-players network, seeds 0 / 1 / 2 (the vectors of
results/m91/emb_-1*.npz). Gallery = every training game except the held-out ones.
  prototype     cosine with the normalised sum of the candidate's games (the current way), standardised per query
  top-m         the query's K nearest gallery games; a candidate's score = the sum of his m best similarities among them
  lse           log-sum-exp over the candidate's games among the K nearest (beta 20, 40)
  mutual        as top-m, but a gallery game counts only if the query is at least as close to it as its own k1-th nearest
                gallery neighbour
Reported per seed and for the three seeds together (prototype: mean of the standardised scores; game-level: sum of the scores).
Continue only if the best variant beats the prototype's three-seed top-1 by >= 0.03.
  python -I logs/nano4/m99_gameknn.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m84_variant as mv  # noqa: E402
import m91_mirror as mm  # noqa: E402

K_NN, K1S = 512, (20, 50)


def main():
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; q = np.load(mv.CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); players = q["players"].astype(str); C = len(players); pcol = {p: i for i, p in enumerate(players)}; hkeys = q["qkey"][2 * Q:2 * Q + H].astype(str); truth = q["held_truth"].astype(np.int64)
    R = np.load(mm.EMB / "rows_-1.npz"); key = R["key"].astype(str); owner = np.array([pcol.get(p, -1) for p in R["pid"].astype(str)]); kpos = {k: i for i, k in enumerate(key)}; hrow = np.array([kpos[k] for k in hkeys]); is_held = np.zeros(len(key), bool); is_held[hrow] = True; gal = np.flatnonzero(~is_held & (owner >= 0)); gown = owner[gal]
    assert (owner[hrow] == truth).all(); log(f"{H} held-out queries, gallery {len(gal)} games of {len(np.unique(gown))} candidates")
    gal_t = torch.as_tensor(gal, device=dev); gown_t = torch.as_tensor(gown, device=dev); tr_t = torch.as_tensor(truth, device=dev); names = ["prototype", "top-1", "top-3", "top-5", "top-10", "lse20", "lse40"] + [f"mutual{k1} top-5" for k1 in K1S] + [f"mutual{k1} lse20" for k1 in K1S]; TOT = {n_: torch.zeros((H, C), device=dev) for n_ in names}; rep = {}

    def report(tag, S):
        r = (S > S[torch.arange(H, device=dev), tr_t][:, None]).sum(1); out = {"top1": float((r == 0).float().mean()), "top5": float((r < 5).float().mean()), "top20": float((r < 20).float().mean())}; rep[tag] = out
        return f"{out['top1']:.3f} / {out['top5']:.3f} / {out['top20']:.3f}"
    for s in (0, 1, 2):
        z = np.load(mm.EMB / f"emb_-1{f'_s{s}' if s else ''}.npz"); E = torch.as_tensor(z["own"].astype(np.float32), device=dev); del z; Gv = E[gal_t]; Qv = E[torch.as_tensor(hrow, device=dev)]; del E
        P = torch.nn.functional.normalize(torch.zeros((C, 128), device=dev).index_add_(0, gown_t, Gv), dim=1); Sp = Qv @ P.T; Sp = (Sp - Sp.mean(1, keepdim=True)) / Sp.std(1, keepdim=True); sc = {"prototype": Sp}
        val = torch.zeros((H, K_NN), device=dev); idx = torch.zeros((H, K_NN), dtype=torch.long, device=dev)
        for i in range(0, H, 512):
            t = (Qv[i:i + 512] @ Gv.T).topk(K_NN, dim=1); val[i:i + 512] = t.values; idx[i:i + 512] = t.indices
        own_k = gown_t[idx]; kk = own_k + C * torch.arange(H, device=dev)[:, None]  # (query, owner) key of each retrieved game, already in descending similarity
        # rank of each retrieved game among the same owner's retrieved games of that query
        flat = kk.flatten(); order = torch.argsort(flat, stable=True); fs = flat[order]; start = torch.ones_like(fs, dtype=torch.bool); start[1:] = fs[1:] != fs[:-1]; pos_in = torch.arange(len(fs), device=dev) - torch.cummax(torch.where(start, torch.arange(len(fs), device=dev), torch.zeros_like(fs)), 0).values; rk = torch.zeros_like(flat); rk[order] = pos_in; rk = rk.view(H, K_NN)

        def agg(v_, mask):
            return torch.zeros(H * C, device=dev).index_add_(0, flat, (v_ * mask).flatten()).view(H, C)
        for m_ in (1, 3, 5, 10):
            sc[f"top-{m_}"] = agg(val, (rk < m_).float())
        for beta in (20, 40):
            sc[f"lse{beta}"] = torch.log1p(agg(torch.exp(beta * (val - 1.0)), torch.ones_like(val)) * 1e6)
        # mutual filter: the k1-th nearest gallery neighbour of every retrieved gallery game
        ug = torch.unique(idx); kth = {k1: torch.zeros(len(gal), device=dev) for k1 in K1S}
        for i in range(0, len(ug), 2000):
            g_ = ug[i:i + 2000]; tk = (Gv[g_] @ Gv.T).topk(max(K1S) + 1, dim=1).values  # the game itself is the first
            for k1 in K1S:
                kth[k1][g_] = tk[:, k1]
        for k1 in K1S:
            mut = (val >= kth[k1][idx]).float(); order2 = torch.argsort((kk * 2 + (1 - mut.long())).flatten() * K_NN + torch.arange(K_NN, device=dev).repeat(H), stable=True)  # mutual games first inside each (query, owner), similarity order kept
            f2 = flat[order2]; st2 = torch.ones_like(f2, dtype=torch.bool); st2[1:] = f2[1:] != f2[:-1]; p2 = torch.arange(len(f2), device=dev) - torch.cummax(torch.where(st2, torch.arange(len(f2), device=dev), torch.zeros_like(f2)), 0).values; rk2 = torch.zeros_like(flat); rk2[order2] = p2; rk2 = rk2.view(H, K_NN)
            sc[f"mutual{k1} top-5"] = agg(val, ((rk2 < 5) & (mut > 0)).float()) + 1e-3 * sc["top-5"]; sc[f"mutual{k1} lse20"] = torch.log1p(agg(torch.exp(20 * (val - 1.0)), mut) * 1e6) + 1e-3 * sc["lse20"]
            if s == 0:
                log(f"   seed 0, k1 = {k1}: share of the retrieved games that are mutual neighbours {float(mut.mean()):.3f}; among the true owner's retrieved games {float(mut[own_k == tr_t[:, None]].mean()):.3f}")
        log(f"seed {s}: top-1 / top-5 / top-20 | " + " | ".join(f"{n_} {report(f'{n_}_s{s}', sc[n_])}" for n_ in names))
        for n_ in names:
            TOT[n_] += sc[n_]
        del Gv, Qv, P, val, idx
    log("three seeds: top-1 / top-5 / top-20 | " + " | ".join(f"{n_} {report(f'{n_}_3seeds', TOT[n_])}" for n_ in names))
    base = rep["prototype_3seeds"]["top1"]; best = max((n_ for n_ in names if n_ != "prototype"), key=lambda n_: rep[f"{n_}_3seeds"]["top1"]); gain = rep[f"{best}_3seeds"]["top1"] - base
    for n_ in ("top-5", "lse20"):
        r = rep  # prototype + game-level together
        S = TOT["prototype"] / 3 + (TOT[n_] - TOT[n_].mean(1, keepdim=True)) / TOT[n_].std(1, keepdim=True); log(f"three seeds, prototype plus standardised {n_}: {report(f'proto+{n_}', S)}")
    log(f"best game-level variant: {best}, top-1 {rep[f'{best}_3seeds']['top1']:.3f} against the prototype's {base:.3f} ({gain:+.3f}) -> {'CONTINUE' if gain >= 0.03 else 'STOP (below +0.03)'}"); (mm.RV / "m99_gameknn.json").write_text(json.dumps(rep, indent=1)); log("written")


if __name__ == "__main__":
    main()
