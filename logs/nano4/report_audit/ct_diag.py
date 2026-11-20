"""Why does the fold-1 mock's test-side C+T probability differ from the offline one by up to 0.0127 on some question?
Per-question |dp| against n, rank and handicap mix; then token-by-token comparison (trunk vectors, scores, flags) of the
worst question's games between the test-side vec parts and the training-side fold-1 parts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402
from aicup.sgf import handicap  # noqa: E402
from aicup.submission import read_test  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_predict as tp  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

M, D = ROOT / "results/mock_fold1_100", ROOT / "data/mock_tests/fold1_100"
A = pd.read_csv(M / "submission_rank_ct_proba.csv").set_index("question_id")
B = pd.read_csv(M / "offline_ct_proba.csv").set_index("question_id")
ans = pd.read_csv(D / "answers.csv", dtype=str).set_index("question_id")
d = (A[tp.P_COLS] - B.loc[A.index, tp.P_COLS]).abs().max(axis=1)
qs = read_test(D / "rank_prediction_test_public.csv") + read_test(D / "rank_prediction_test_private.csv")
hd = {q["question_id"]: [handicap(s) for s, _ in q["games"]] for q in qs}
tab = pd.DataFrame({"dp": d, "n": ans.loc[A.index, "n_games"].astype(int), "rank": ans.loc[A.index, "rank"],
                    "hd_max": [max(hd[q]) for q in A.index], "n_hd2": [sum(h >= 2 for h in hd[q]) for q in A.index],
                    "pmax": A[tp.P_COLS].max(axis=1)}).sort_values("dp", ascending=False)
pd.set_option("display.width", 200)
print(tab.head(12))
print("dp quantiles", {k: round(v, 6) for k, v in tab.dp.quantile([0.5, 0.9, 0.95, 0.99, 1]).to_dict().items()})
print(tab.groupby(tab.n_hd2 > 0).dp.agg(["mean", "max", "count"]))
print(tab.groupby("n").dp.agg(["mean", "max", "count"]))

players = fold_players(player_games(), pd.read_csv(FOLDS_CSV), 1)
pq = pilot_questions(players).set_index("question_id")
Xtr, off_tr = ts.raw_tokens([(ts.VEC / "fold1", ROOT / "results/trunk_seq/scores_fold1.npy")])
Xte, off_te = ts.raw_tokens([(M / "vec", M / "vec/scores_heads3_B.npy")])
tasks, owner, _, _ = ps.build_tasks(qs)
ids = [q["question_id"] for q in qs]
agg = []
for label, qid in (("worst", tab.index[0]), ("second", tab.index[1]), ("median", tab.index[len(tab) // 2])):
    keys = pq.loc[ans.loc[qid, "source_question"], "game_keys"]
    qi = ids.index(qid)
    ks = [k for k, o in enumerate(owner) if o == qi]
    print(f"== {label} question {qid}: dp {tab.loc[qid, 'dp']:.5f}, {len(keys)} games")
    for key, k in zip(keys, ks):
        s1, n1 = off_tr[key]
        s2, n2 = off_te[tp.task_key(k)]
        if n1 != n2:
            print(f"  {key} task {k}: plies differ {n1} vs {n2}")
            continue
        a, b = Xtr[s1:s1 + n1].astype(np.float32), Xte[s2:s2 + n2].astype(np.float32)
        dg, dsc, dfl = np.abs(a[:, :128] - b[:, :128]), np.abs(a[:, 128:158] - b[:, 128:158]), np.abs(a[:, 158:] - b[:, 158:])
        agg.append({"q": label, "key": key, "trunk_max": dg.max(), "trunk_mean": dg.mean(), "trunk_scale": np.abs(a[:, :128]).mean(),
                    "scores_max": dsc.max(), "scores_mean": dsc.mean(), "flags_max": dfl.max(), "hd": handicap(tasks[k][1])})
        print(f"  {key} task {k} n {n1} HD {handicap(tasks[k][1])}: trunk max {dg.max():.4f} mean {dg.mean():.5f} (|x| {np.abs(a[:, :128]).mean():.3f}), "
              f"scores max {dsc.max():.4f} mean {dsc.mean():.5f}, flags max {dfl.max():.4f}; worst ply {int(dg.max(axis=1).argmax())}")
pd.DataFrame(agg).to_csv(M / "ct_diag_tokens.csv", index=False)
# a global view: every game of the mock, trunk-vector difference summary
rows = []
for qid in ids:
    keys = pq.loc[ans.loc[qid, "source_question"], "game_keys"]
    qi = ids.index(qid)
    for key, k in zip(keys, [k for k, o in enumerate(owner) if o == qi]):
        s1, n1 = off_tr[key]
        s2, n2 = off_te[tp.task_key(k)]
        if n1 != n2:
            rows.append({"q": qid, "key": key, "plies_differ": True})
            continue
        a, b = Xtr[s1:s1 + n1, :158].astype(np.float32), Xte[s2:s2 + n2, :158].astype(np.float32)
        dd = np.abs(a - b)
        rows.append({"q": qid, "key": key, "plies_differ": False, "trunk_max": dd[:, :128].max(), "trunk_mean": dd[:, :128].mean(),
                     "scores_max": dd[:, 128:].max(), "scores_mean": dd[:, 128:].mean(), "hd": handicap(tasks[k][1])})
g = pd.DataFrame(rows)
g.to_csv(M / "ct_diag_games.csv", index=False)
print("games", len(g), "plies differ", int(g.plies_differ.sum()))
print(g[["trunk_max", "trunk_mean", "scores_max", "scores_mean"]].describe().T)
print(g.groupby(g.hd.fillna(-1) >= 2)[["trunk_max", "trunk_mean", "scores_max"]].mean())
per_q = g.groupby("q")[["trunk_mean", "scores_max"]].mean()
print("corr(question dp, mean trunk diff)", np.corrcoef(tab.loc[per_q.index, "dp"], per_q["trunk_mean"])[0, 1])
