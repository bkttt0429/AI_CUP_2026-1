#!/usr/bin/env bash
# Build the mock-40 from the v1.1 training files (SGF with HD[]) and compare it with the v1.0 fixture (compute node).
set -euo pipefail
python -I scripts/make_mock_test.py --out-dir data/mock_tests/fold0_40_v11
cd data/mock_tests/fold0_40_v11
FX=../../../tests/fixtures/mock_fold0_40
for f in rank_prediction_test_public.csv rank_prediction_test_private.csv; do
  echo "$f: rows with HD[ $(grep -c 'HD\[' "$f") of $(( $(wc -l < "$f") - 1 )); HD values: $(grep -o 'HD\[[0-9]*\]' "$f" | sort | uniq -c | tr '\n' ' ')"
  echo "$f minus HD[] vs fixture: $(diff <(sed -E 's/HD\[[0-9]+\]//g' "$f") "$FX/$f" | wc -l) differing lines"
done
echo "answers vs fixture: $(diff <(cut -d, -f1-3 answers.csv) <(cut -d, -f1-3 "$FX/answers.csv") | wc -l) differing lines"
echo "offline_predictions vs fixture: $(diff offline_predictions.csv "$FX/offline_predictions.csv" | wc -l) differing lines"
