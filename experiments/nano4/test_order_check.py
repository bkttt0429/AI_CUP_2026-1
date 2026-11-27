"""Are the official test's sgf_1..sgf_n columns in time order? Training game_ids look chronological (per-player anchor rises with id,
Spearman +0.058 over 80 games). Per test question (n >= 10): the per-game opponent anchor (opponent HSL pmi from the ct6 shards + the HD rule)
against the column index -> mean within-question Spearman; the same for the target's own pmi; reference: training players' 20 base games
ordered by game_id (folds 1-4), same statistics. A test mean near the training reference => columns chronological; near 0 => shuffled.
  python -I logs/nano4/test_order_check.py --out results/r_variants/test_order_check.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table, signed_gap  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from pilot_stats import spearman  # noqa: E402

COLS = [f"sum_{p}" for p in PROFILES]


def summarize(rhos):
    r = np.array([x for x in rhos if np.isfinite(x)]); return {"questions": int(len(r)), "mean_rho": float(r.mean()), "se": float(r.std() / np.sqrt(len(r)))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS); tasks, owner, _, _ = ps.build_tasks(questions)
    own, opp = {}, {}
    for i in range(8):
        with np.load(ROOT / "results/ct6" / f"submission_rank_shard{i}of8.npz", allow_pickle=False) as z:
            for k, s, o in zip(z["task"].tolist(), z["sums"], z["hsl_opp_sums"]):
                own[k], opp[k] = s, o
    rho_a, rho_o, rho_a15 = [], [], []
    for qi, q in enumerate(questions):
        ks = [k for k, o in enumerate(owner) if o == qi]  # in column order (build_tasks keeps the input order)
        if len(ks) < 10:
            continue
        pm_o = posterior_mean_index(np.stack([opp[k] for k in ks])); pm_own = posterior_mean_index(np.stack([own[k] for k in ks]))
        hd = np.array([float(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", tasks[k][1])) else 0.0 for k in ks]); col = np.array([tasks[k][2] for k in ks])
        anchor = pm_o + signed_gap(hd, col); idx = np.arange(len(ks))
        rho_a.append(spearman(idx, anchor)); rho_o.append(spearman(idx, pm_own))
        if len(ks) >= 15:
            rho_a15.append(spearman(idx, anchor))
    rep = {"test": {"anchor_vs_column": summarize(rho_a), "own_vs_column": summarize(rho_o), "anchor_vs_column_n_ge_15": summarize(rho_a15)}}
    print(f"test (n >= 10, {rep['test']['anchor_vs_column']['questions']} questions): mean within-question Spearman(column index, opponent anchor) {rep['test']['anchor_vs_column']['mean_rho']:+.4f} ± {rep['test']['anchor_vs_column']['se']:.4f}; own pmi {rep['test']['own_vs_column']['mean_rho']:+.4f} ± {rep['test']['own_vs_column']['se']:.4f}")
    # training reference: 20 base games per player ordered by game_id
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    ra, ro = [], []
    for f in (1, 2, 3, 4):
        h = read_scored_games(ts.HSL / f"fold{f}"); o = read_scored_games(ts.OPP_HSL / f"fold{f}").set_index("game_key")
        h["pm_own"] = posterior_mean_index(h[COLS].to_numpy(float)); h["pm_opp"] = posterior_mean_index(o.loc[h.game_key, COLS].to_numpy(float))
        h["anchor"] = h.pm_opp + signed_gap(h.game_key.map(hd).fillna(0).to_numpy(), h.color.to_numpy()); h["gid"] = h.game_key.str.split("g_").str[1].astype(int)
        for pid, s in h.groupby("player_id"):
            if len(s) < 10:
                continue
            s = s.sort_values("gid"); idx = np.arange(len(s)); ra.append(spearman(idx, s.anchor.to_numpy())); ro.append(spearman(idx, s.pm_own.to_numpy()))
    rep["train_reference_20_games_by_game_id"] = {"anchor": summarize(ra), "own": summarize(ro)}
    print(f"training reference (20 games by game_id, {len(ra)} players): anchor {rep['train_reference_20_games_by_game_id']['anchor']['mean_rho']:+.4f} ± {rep['train_reference_20_games_by_game_id']['anchor']['se']:.4f}; own {rep['train_reference_20_games_by_game_id']['own']['mean_rho']:+.4f} ± {rep['train_reference_20_games_by_game_id']['own']['se']:.4f}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
