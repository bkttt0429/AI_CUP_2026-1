"""Token coupling diagnostic (2026-11-06; user: 「這裡輸入解耦丟進去，後面流程會比較好嗎」). How redundant are the five blocks of the
162-d ply token -- board trunk g (64), played-point trunk l (64), L1b (level + 10 centred), Human SL (level + 18 centred), 4 flags?
On fold 1 in the production layout (-oh score arrays), standardised exactly as the trainer does (trunk_seq_check.token_stats),
a random sample of plies, split into two halves BY GAME (fit on A, score on B), separately for target and opponent plies:
  spectrum  eigenvalues of the 158 continuous columns' correlation: participation ratio, components for 90 / 95 / 99 % of variance;
            the same inside each block
  cca       canonical correlations trunk (g + l) vs scores (30), g vs l, L1b (11) vs Human SL (19)
  r2        held-out R^2 of each score group from the trunk (least squares, and a 2-layer MLP), from g alone and l alone;
            and of the trunk from the scores (least squares)
Report only; no decision rides on it.
  python -I logs/nano4/token_coupling.py --out results/r_variants/token_coupling.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from trunk_seq_check import D_TOKEN, load_tokens  # noqa: E402

BLOCKS = {"g": slice(0, 64), "l": slice(64, 128), "l1b": slice(128, 139), "hsl": slice(139, 158)}
TRUNK, SCORES = slice(0, 128), slice(128, 158)


def spectrum(Z):
    C = np.corrcoef(Z, rowvar=False)
    C = C[np.isfinite(C).all(axis=1)][:, np.isfinite(C).all(axis=0)]
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    cum = np.cumsum(lam) / lam.sum()
    return {"dims": int(len(lam)), "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            **{f"k{p}": int(np.searchsorted(cum, p / 100) + 1) for p in (90, 95, 99)}, "top5_share": [float(x) for x in (lam[:5] / lam.sum())]}


def cca(A, B, k=10, ridge=1e-4):
    A = A - A.mean(0); B = B - B.mean(0)
    def isqrt(S):
        w, V = np.linalg.eigh(S + ridge * np.eye(len(S)) * np.trace(S) / len(S))
        return V @ np.diag(w ** -0.5) @ V.T
    n = len(A)
    M = isqrt(A.T @ A / n) @ (A.T @ B / n) @ isqrt(B.T @ B / n)
    return [float(s) for s in np.linalg.svd(M, compute_uv=False)[:k]]


def r2_cols(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def lstsq_r2(Xa, Ya, Xb, Yb):
    Xa1, Xb1 = np.hstack([Xa, np.ones((len(Xa), 1))]), np.hstack([Xb, np.ones((len(Xb), 1))])
    W = np.linalg.solve(Xa1.T @ Xa1 + 1e-3 * np.eye(Xa1.shape[1]), Xa1.T @ Ya)
    return r2_cols(Yb, Xb1 @ W)


def mlp_r2(Xa, Ya, Xb, Yb, device, epochs=12, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xa.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, Ya.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xa, ya = torch.as_tensor(Xa, device=device), torch.as_tensor(Ya, device=device)
    for ep in range(epochs):
        perm = torch.randperm(len(xa), device=device)
        for i in range(0, len(xa), 4096):
            j = perm[i:i + 4096]
            loss = torch.nn.functional.mse_loss(net(xa[j]), ya[j]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.cat([net(torch.as_tensor(Xb[i:i + 65536], device=device)) for i in range(0, len(Xb), 65536)]).cpu().numpy()
    return r2_cols(Yb, P)


def group_r2(r2):
    """Mean held-out R^2 by score group: L1b level, L1b centred, HSL level, HSL centred."""
    return {"l1b_level": float(r2[0]), "l1b_centred": float(np.mean(r2[1:11])), "hsl_level": float(r2[11]), "hsl_centred": float(np.mean(r2[12:30]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--n", type=int, default=600_000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, offsets, _ = load_tokens([a.fold], ROOT / "results/trunk_seq", "cpu", opp_hsl=True)
    lens = np.array([n for _, n in sorted(offsets.values())]); starts = np.array([s for s, _ in sorted(offsets.values())])
    game = np.repeat(np.arange(len(lens)), lens); assert len(game) == len(X) and starts[0] == 0
    rng = np.random.default_rng(0); half = rng.random(len(lens)) < 0.5
    tgt = X[:, D_TOKEN - 4].float().numpy() > 0.5
    print(f"fold {a.fold}: {len(X)} plies, {len(lens)} games; target plies {tgt.mean():.3f}", flush=True)
    rep = {"fold": a.fold, "plies": int(len(X)), "games": int(len(lens)), "roles": {}}
    for role, m in (("target", tgt), ("opponent", ~tgt)):
        idx = np.flatnonzero(m); idx = np.sort(rng.choice(idx, min(a.n, len(idx)), replace=False))
        Z = X[torch.as_tensor(idx)].float().numpy()[:, :158]; A = half[game[idx]]
        r = {"n": int(len(idx)), "spectrum_all": spectrum(Z), "spectrum_blocks": {k: spectrum(Z[:, s]) for k, s in BLOCKS.items()}}
        r["cca"] = {"trunk_vs_scores": cca(Z[A][:, TRUNK], Z[A][:, SCORES]), "g_vs_l": cca(Z[A][:, BLOCKS["g"]], Z[A][:, BLOCKS["l"]]),
                    "l1b_vs_hsl": cca(Z[A][:, BLOCKS["l1b"]], Z[A][:, BLOCKS["hsl"]])}
        Ya, Yb = Z[A][:, SCORES], Z[~A][:, SCORES]
        r["r2_scores_from"] = {name: group_r2(lstsq_r2(Z[A][:, s], Ya, Z[~A][:, s], Yb)) for name, s in (("trunk_linear", TRUNK), ("g_linear", BLOCKS["g"]), ("l_linear", BLOCKS["l"]))}
        r["r2_scores_from"]["trunk_mlp"] = group_r2(mlp_r2(Z[A][:, TRUNK], Ya, Z[~A][:, TRUNK], Yb, device))
        r["r2_hsl_from_l1b_linear"] = group_r2(np.r_[np.full(11, np.nan), lstsq_r2(Z[A][:, BLOCKS["l1b"]], Z[A][:, BLOCKS["hsl"]], Z[~A][:, BLOCKS["l1b"]], Z[~A][:, BLOCKS["hsl"]])])
        rt = lstsq_r2(Z[A][:, SCORES], Z[A][:, TRUNK], Z[~A][:, SCORES], Z[~A][:, TRUNK])
        r["r2_trunk_from_scores_linear"] = {"g_mean": float(rt[:64].mean()), "l_mean": float(rt[64:].mean()), "l_max": float(rt[64:].max())}
        rep["roles"][role] = r
        print(f"== {role} plies (n {len(idx)})", flush=True)
        print(f"  spectrum 158 cols: PR {r['spectrum_all']['participation_ratio']:.1f}, k90/95/99 {r['spectrum_all']['k90']}/{r['spectrum_all']['k95']}/{r['spectrum_all']['k99']}; "
              + "  ".join(f"{k} PR {v['participation_ratio']:.1f} k95 {v['k95']}/{v['dims']}" for k, v in r["spectrum_blocks"].items()), flush=True)
        print("  CCA top-5  trunk~scores " + " ".join(f"{x:.3f}" for x in r["cca"]["trunk_vs_scores"][:5]) + "  g~l " + " ".join(f"{x:.3f}" for x in r["cca"]["g_vs_l"][:5])
              + "  l1b~hsl " + " ".join(f"{x:.3f}" for x in r["cca"]["l1b_vs_hsl"][:5]), flush=True)
        for k, v in r["r2_scores_from"].items():
            print(f"  R2 scores from {k:13s} " + "  ".join(f"{g} {x:+.3f}" for g, x in v.items()), flush=True)
        print(f"  R2 HSL from L1b (linear)  hsl_level {r['r2_hsl_from_l1b_linear']['hsl_level']:+.3f}  hsl_centred {r['r2_hsl_from_l1b_linear']['hsl_centred']:+.3f}; "
              f"R2 trunk from scores (linear) g {r['r2_trunk_from_scores_linear']['g_mean']:+.3f}  l {r['r2_trunk_from_scores_linear']['l_mean']:+.3f}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1)); print("written", a.out)


if __name__ == "__main__":
    main()
