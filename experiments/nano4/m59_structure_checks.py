"""Section 4M M59 (2026-11-08): data-structure checks recommended by the research note data_structure_signals.md. Report only; counts and
statistics, nothing is used to identify anyone and no record is linked to a person.
  C2.1 per test question: are all games one colour? colour x num_games crosstab per half; ct9 predicted classes per (colour, n) cell.
  C2.4 training: share of player_ids whose games are all one colour.
  C4.1 exact duplicate move sequences (root properties stripped): train<->test and test<->test counts only.
  S1   question order: Spearman of question index vs ct9's expected level per half; lag-1 autocorrelation; pub_q_k vs priv_q_k level corr.
  python -I logs/nano4/m59_structure_checks.py
"""
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from m43_gate import decide  # noqa: E402

LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
MOVE = re.compile(r";([BW])\[([a-s]{0,2})\]")


def seq_hash(sgf):
    return hashlib.sha1("".join(f"{c}{m}" for c, m in MOVE.findall(sgf)).encode()).hexdigest()


def main():
    tests = {h: pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{h}.csv", dtype=str, keep_default_na=False) for h in ("public", "private")}
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id"); P = pr[[f"p_{l}" for l in LV]]
    E = (P.to_numpy() * np.arange(10)).sum(1); E = pd.Series(E, index=pr.index); D9 = pd.Series(decide(P.to_numpy()), index=pr.index)
    print("== C2.1 colour structure")
    test_hash = {}
    for h, df in tests.items():
        rows = []
        for _, r in df.iterrows():
            n = int(r["num_games"]); cols = [r[f"color_{k}"] for k in range(1, n + 1)]
            rows.append((r["question_id"], n, "mixed" if len(set(cols)) > 1 else cols[0]))
            for k in range(1, n + 1):
                test_hash.setdefault(seq_hash(r[f"sgf_{k}"]), []).append(r["question_id"])
        t = pd.DataFrame(rows, columns=["q", "n", "colour"])
        print(f"{h}: single-colour questions {int((t.colour != 'mixed').sum())}/{len(t)}; colour counts {t.colour.value_counts().to_dict()}")
        print(pd.crosstab(t.colour, t.n).to_string())
        t["pred"] = t.q.map(D9).map(lambda i: LV[i])
        ct = pd.crosstab([t.colour, t.n], t.pred).reindex(columns=LV, fill_value=0)
        print(f"{h}: ct9 predicted classes per (colour, n) cell:\n{ct.to_string()}")
        tests[h] = t
    print("== C2.4 training player colour")
    frames = []
    for p in sorted((ROOT / "data/training").glob("train_*.csv")):
        frames.append(pd.read_csv(p, usecols=["player_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False))
    tr = pd.concat(frames, ignore_index=True)
    nc = tr.groupby("player_id")["color"].nunique()
    print(f"training players {len(nc)}; single-colour players {int((nc == 1).sum())} ({(nc == 1).mean():.3f})")
    print("== C4.1 duplicate move sequences (counts only)")
    tr_hash = Counter(seq_hash(s) for s in tr["sgf_content"])
    dup_tt = sum(1 for k in test_hash if k in tr_hash)
    within = sum(1 for k, v in test_hash.items() if len(v) > 1)
    within_cross_q = sum(1 for k, v in test_hash.items() if len(set(v)) > 1)
    print(f"test games {sum(len(v) for v in test_hash.values())}, distinct sequences {len(test_hash)}; sequences also in training: {dup_tt}; "
          f"sequences repeated within the test: {within} (across different questions: {within_cross_q}); training sequences repeated within training: {sum(1 for v in tr_hash.values() if v > 1)}")
    print("== S1 question order")
    for h in ("public", "private"):
        q = tests[h].q.tolist(); e = E.reindex(q).to_numpy()
        rho, p = spearmanr(np.arange(len(e)), e)
        ac = float(np.corrcoef(e[:-1], e[1:])[0, 1])
        print(f"{h}: Spearman(index, expected level) {rho:+.3f} (p {p:.3f}); lag-1 autocorrelation {ac:+.3f}")
    pq = [f"pub_q_{k:04d}" for k in range(1, 401)]; vq = [f"priv_q_{k:04d}" for k in range(1, 401)]
    if all(x in E.index for x in pq + vq):
        r, p = spearmanr(E.reindex(pq).to_numpy(), E.reindex(vq).to_numpy())
        same_n = float(np.mean(tests["public"].set_index("q").n.reindex(pq).to_numpy() == tests["private"].set_index("q").n.reindex(vq).to_numpy()))
        same_c = float(np.mean(tests["public"].set_index("q").colour.reindex(pq).to_numpy() == tests["private"].set_index("q").colour.reindex(vq).to_numpy()))
        print(f"pub_q_k vs priv_q_k: Spearman of expected level {r:+.3f} (p {p:.3f}); same num_games {same_n:.3f}; same colour {same_c:.3f}")


if __name__ == "__main__":
    main()
