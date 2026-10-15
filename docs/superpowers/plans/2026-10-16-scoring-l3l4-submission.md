# 擴大評分、L3/L4 學習與第一份提交管線 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 對 5 個 fold 每位（≥ 5 局）玩家各評 ≤ 20 局的 Human SL 似然（18 profile），用 18 維似然曲線在 folds 1–4 學 L3/L4，fold 0 只驗證一次，並做出能吃官方測試檔、產出 800 列 `submission_rank.csv` 的管線。

**Architecture:** 評分沿用 pilot 的 CPU 抽取 + GPU 18 profile 評分，改成可續跑的分段輸出，兩張 GPU 各一個 shard 依序跑 fold 1→2→3→4→0。題目特徵由每局 18 個 log p 總和組成；模型（單調門檻、多類邏輯迴歸、小 MLP）用 folds 1–4 的留一 fold 交叉驗證挑選，再用全部四個 fold 重訓，最後在 fold 0 評一次。提交腳本重用同一評分程式與模型。

**Tech Stack:** Python 3.12、torch 2.14（FP32 + 預設 TF32 卷積）、numpy、pandas（無 sklearn，模型用 torch 寫）。

**Spec:** [METHOD_NOTES.md](../../../METHOD_NOTES.md) §2.1、§2.3、§2.4.1、§2.7；[docs/ARCHITECTURE.md](../../ARCHITECTURE.md) L2–L6；官方 Tutorial README（測試檔與提交格式）。

## Global Constraints

- 每位玩家只評 `min(20, n_games)` 局，`n_games < 5` 的玩家不評（測試題至少 5 局）；局的挑選用以 player_id 衍生的 seed 打亂，與處理順序無關。
- 18 個 profile 與 pilot 相同（`humansl.PROFILES`）；FP32 + cuDNN 預設 TF32，batch 64；若冒煙測試證明 `cudnn.benchmark=True` 更快就開啟（數值差 ≤ 1e-3/手）。
- 評估題：每人巢狀切 1×20、⌊k/10⌋×10、⌊k/5⌋×5（k = 評分局數）。
- 模型挑選只用 folds 1–4 的留一 fold 交叉驗證；fold 0 只在最後評一次，所有候選的 fold 0 分數都要列出，但挑選依據是 CV。
- 決策一律用 `metric.expected_score_decision`。
- 提交檔：欄位 `question_id,pred_rank`，public 在前、private 在後，`pred_rank ∈ LEVELS`，題號不得缺漏或重複。
- 大檔（`*.npz`、分段 CSV）留在 lab01、不進 git；彙總 CSV 與 JSON 進 git。

## Review Focus

1. 長時間執行中斷：重跑必須跳過已完成的分段，不重算也不重複（Task 1 測試）。
2. 測試檔裡空的 `sgf_k` 欄（`num_games` 之後）必須忽略，不能變成 0 手的局（Task 5 測試）。
3. 測試 SGF 可能含虛手、非 19 路或解析失敗：逐局記錄，該題仍要輸出答案（Task 5 測試）。
4. 特徵在題目只有 5 局、總手數很少時仍有限值（無 NaN、無 log 0）（Task 3 測試）。
5. 模型輸出的機率每列加總為 1、沒有 NaN；決策落在 0–9（Task 3 測試）。

---

### Task 1: 全 fold 選局與可續跑評分

**Files:** Create `src/aicup/scoring.py`（把 pilot 的 worker、`load_games`、`score_stream` 搬進來）；Modify `scripts/pilot_score.py`（改為 import）；Create `scripts/score_folds.py`；Test `tests/test_scoring.py`

**Interfaces — Produces:**
- `fold_players(games: DataFrame[player_id, rank, game_key], folds: DataFrame[rank, player_id, n_games, fold], fold: int, max_games=20, min_games=5, seed=42) -> DataFrame[rank, player_id, game_keys]`
- `plan_parts(n_items: int, part_size: int) -> list[tuple[int, int]]`；`pending_parts(out_dir: Path, shard: int, parts) -> list[int]`（分段 CSV 與 npz 都存在才算完成）
- `score_stream(pool, model, tasks, rows, flush_at=1024, log_every=200)`、`load_games(keys)`、`init_worker(config)`、`extract(task)`（行為與 pilot 版相同）
- CLI `scripts/score_folds.py --folds 1 2 3 4 0 --shard i --nshards 2 [--workers 10] [--part-size 500] [--out results/scores] [--limit-games K] [--cudnn-benchmark]` → `results/scores/fold{F}/games_s{i}_p{k:03d}.csv`、`moves_s{i}_p{k:03d}.npz`、完成時 `done_s{i}`

- [ ] 測試：同一玩家的選局與其他玩家、處理順序無關；`n_games < 5` 被排除；`k = min(20, n)`；`plan_parts(1001, 500) == [(0,500),(500,1000),(1000,1001)]`；只有 CSV 沒有 npz 的分段仍算未完成。
- [ ] FAIL → 實作 → PASS；pilot 測試仍全過。
- [ ] lab01 冒煙：40 局，有／無 `--cudnn-benchmark` 各一次，記錄速度與兩者 log p 差 → commit。

### Task 2: 過夜執行

- [ ] lab01 以 `setsid nohup` 啟動兩個 shard（GPU 0／1），順序 fold 1→2→3→4→0；本機背景監看程序在兩者結束時通知。
- [ ] 完成後驗證：每 fold 局數與 Task 1 的選局數相同、無 NaN、無重複；合併成 `results/scores/games_fold{F}.csv`（進 git）。

### Task 3: 題目特徵與模型

**Files:** Create `src/aicup/qfeatures.py`、`src/aicup/rankmodel.py`；Test `tests/test_qfeatures.py`、`tests/test_rankmodel.py`

**Interfaces — Produces:**
- `question_features(sums: ndarray[m,18], n_moves: ndarray[m]) -> ndarray[F]`：相對曲線（每手平均 log p 減最大值，18）、最佳 profile 的每手 log p、posterior-mean index、log 總手數、log 局數、逐局 posterior-mean index 的標準差。`FEATURE_NAMES` 與之對齊。
- `build_matrix(games: DataFrame, questions: DataFrame) -> (X, y, n_games)`。
- `eval_questions(players)`（= `pilot.pilot_questions`）；`sample_questions(players, per_player, rng, sizes=range(5, 21)) -> DataFrame`（訓練用隨機子集，大小不超過玩家評分局數）。
- `ThresholdModel`、`SoftmaxModel(l2)`、`MLPModel(hidden, l2, epochs, seed)`，皆有 `fit(X, y)`、`predict_proba(X) -> [N,10]`、`state_dict()`／`from_state(...)`。

- [ ] 測試：特徵在 5 局 × 少手數時有限；相對曲線最大值為 0；可線性分開的合成資料上 SoftmaxModel 準確率 ≥ 0.95；機率列和為 1；同 seed 結果相同；ThresholdModel 與 `evidence.fit_thresholds` 一致。
- [ ] FAIL → 實作 → PASS → commit。

### Task 4: 交叉驗證、重訓、fold 0 評估

**Files:** Create `scripts/fit_rank_model.py`；Output `results/rank_model_eval.json`、`results/models/rank_model.json`

- [ ] 對每個候選（Threshold、Softmax l2 ∈ {1e-4, 1e-3, 1e-2}、MLP hidden ∈ {32, 64}），folds 1–4 留一 fold CV：訓練用 `sample_questions`（每人 8 題）＋巢狀題，驗證用巢狀題；記 n=5／10／20 與平均官方分。
- [ ] 依 CV 平均挑最佳；用 folds 1–4 全部重訓；fold 0 評一次（含所有候選的 fold 0 分數、玩家 bootstrap CI、混淆矩陣、每級分數）→ commit。

### Task 5: 提交管線

**Files:** Create `scripts/predict_submission.py`、`scripts/make_mock_test.py`；Test `tests/test_submission.py`

**Interfaces — Produces:**
- `read_test(path) -> list[dict(question_id, games=[(sgf, color), ...])]`（忽略 `num_games` 之後的空欄）
- `write_submission(rows, path)`、`validate_submission(path, question_ids) -> list[str]`（回傳問題清單，空 = 合格）
- CLI：`predict_submission.py --public tests/rank_prediction_test_public.csv --private tests/rank_prediction_test_private.csv --out submission_rank.csv`；同時寫 `submission_report.json`（每題局數、虛手數、解析失敗數、預測分布）。
- `make_mock_test.py`：從 fold 0 巢狀題做出官方格式的假測試檔（`pub_q_0001`… 與 `priv_q_0001`…）及答案。

- [ ] 測試：空 sgf 欄被忽略；`validate_submission` 抓到缺題、重複、非法等級；虛手局能評分。
- [ ] FAIL → 實作 → PASS；lab01 上用 40 題假測試檔端到端跑，分數與 Task 4 對同一批題目的離線評估一致 → commit。

### Task 6: 記錄

- [ ] METHOD_NOTES（新節：L3/L4 結果與第一份提交管線）、ARCHITECTURE（L3、L4、L6 現況）、記憶 → commit。
