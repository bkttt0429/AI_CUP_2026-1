# mock_fold0_40

40-question mock test in the official test format (20 `pub_` + 20 `priv_` questions), cut from fold 0 by `experiments/make_mock_test.py`. It is training data, not an official test file. The submission, probability and report files come from the 2026-10-16 `scripts/predict_submission.py` run on lab01 (production `rank_model.json`): it matched the offline path on 40/40 questions with 0 games dropped. That is why `model` in the report is a lab01 path. Copied unchanged from `data/mock_tests/fold0_40/`, which stays git-ignored.

| file | content |
| --- | --- |
| `rank_prediction_test_public.csv`, `rank_prediction_test_private.csv` | inputs: `question_id,num_games,sgf_1..sgf_20,color_1..color_20` |
| `answers.csv` | true ranks |
| `offline_predictions.csv` | offline-path predictions |
| `submission_rank.csv`, `submission_rank_proba.csv`, `submission_rank_report.json` | pipeline outputs |
