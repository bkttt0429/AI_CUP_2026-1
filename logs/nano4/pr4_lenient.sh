#!/usr/bin/env bash
# PR #4 @ 22d1e4d on nano4: the strict default path again, then --lenient-inputs on a deliberately malformed
# copy of the mock test (pub_q_0002 declares 9 games but fills 10 sgf columns). Expected exit codes:
# strict mock 0; strict malformed 2; lenient malformed 4; merge of that shard with --allow-input-warnings 0.
set -u
FIX=tests/fixtures/mock_fold0_40
BAD=logs/nano4/mock_bad
EXP="--expect-rows 40 --expect-public 20 --expect-private 20"
mkdir -p "$BAD"
python -I - "$FIX" "$BAD" <<'EOF'
import sys
import pandas as pd
fix, bad = sys.argv[1], sys.argv[2]
pub = pd.read_csv(f"{fix}/rank_prediction_test_public.csv", dtype=str, keep_default_na=False)
assert pub.loc[pub.question_id == "pub_q_0002", "num_games"].item() == "10"
pub.loc[pub.question_id == "pub_q_0002", "num_games"] = "9"
pub.to_csv(f"{bad}/rank_prediction_test_public.csv", index=False)
pd.read_csv(f"{fix}/rank_prediction_test_private.csv", dtype=str, keep_default_na=False) \
    .to_csv(f"{bad}/rank_prediction_test_private.csv", index=False)
print("malformed copy written: pub_q_0002 num_games 10 -> 9")
EOF
run() { echo "+ $*"; "$@"; local c=$?; echo "EXIT $c"; return 0; }
run python -I scripts/predict_submission.py --inputs $FIX/rank_prediction_test_public.csv $FIX/rank_prediction_test_private.csv $EXP --out results/mock_fold0_40_v2/submission_rank.csv
run python -I scripts/predict_submission.py --inputs $BAD/rank_prediction_test_public.csv $BAD/rank_prediction_test_private.csv $EXP --out results/mock_bad_strict/submission_rank.csv
run python -I scripts/predict_submission.py --lenient-inputs --inputs $BAD/rank_prediction_test_public.csv $BAD/rank_prediction_test_private.csv $EXP --out results/mock_bad_lenient/submission_rank.csv
run python -I scripts/predict_submission.py --merge --nshards 1 --lenient-inputs --allow-input-warnings --inputs $BAD/rank_prediction_test_public.csv $BAD/rank_prediction_test_private.csv $EXP --out results/mock_bad_lenient/submission_rank.csv
python -I - <<'EOF'
import json
import numpy as np
import pandas as pd
def sub(d): return pd.read_csv(f"results/{d}/submission_rank.csv", dtype=str)
def pr(d): return pd.read_csv(f"results/{d}/submission_rank_proba.csv").set_index("question_id")
v2, v1, bad = sub("mock_fold0_40_v2"), sub("mock_fold0_40"), sub("mock_bad_lenient")
fix = pd.read_csv("tests/fixtures/mock_fold0_40/offline_predictions.csv", dtype=str)
for name, a, b in (("v2 strict vs 5c540a7 single", v2, v1), ("v2 strict vs fixture offline", v2, fix), ("lenient malformed vs v2 strict", bad, v2)):
    print(f"{name}: {(a.pred_rank.values == b.pred_rank.values).sum()}/{len(a)} equal, order same {list(a.question_id) == list(b.question_id)}")
cols = [c for c in pr("mock_fold0_40_v2").columns if c.startswith("p_")]
print("proba max |v2 - 5c540a7|", f"{np.abs(pr('mock_fold0_40_v2')[cols].values - pr('mock_fold0_40')[cols].values).max():.3e}")
print("proba max |lenient - v2|", f"{np.abs(pr('mock_bad_lenient')[cols].values - pr('mock_fold0_40_v2')[cols].values).max():.3e}")
r = json.load(open("results/mock_bad_lenient/submission_rank_report.json"))
print("lenient report (after the --allow merge):", {k: r.get(k) for k in ("mode", "lenient_inputs", "input_warnings", "allow_input_warnings", "exit_code", "problems", "games_used")})
EOF
