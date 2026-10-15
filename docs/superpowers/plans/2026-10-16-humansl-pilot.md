# Human SL pilot（步驟 2）— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 回答 H1：KataGo humanv0 在 18 個 rank profile 下的目標落子似然，能不能在**未見玩家**上分開本競賽的 10 個等級（特別是相鄰段位），並量出對手等級假設的影響。

**Architecture:** CPU 多進程重播 SGF、算一次盤面特徵與合法步遮罩（`src/aicup/pilot.py`）；GPU 端同一批特徵換 18 組 metadata 列各跑一次前向（`humansl.score_positions`）；每手 18 個 log p 存檔；分析（`src/aicup/evidence.py`）只用不需擬合的統計量評 fold 0，唯一擬合（單調門檻）只用 fold 1。兩張 GPU 各跑一個 shard。

**Tech Stack:** Python 3.12、torch 2.14.1（FP32）、numpy、pandas、multiprocessing（fork，CUDA 初始化前建 Pool）、KataGo `python/katago`。

**Spec:** [METHOD_NOTES.md](../../../METHOD_NOTES.md) §2.4（pilot）、§2.1（驗證協議）；[docs/ARCHITECTURE.md](../../ARCHITECTURE.md) H1。

## Global Constraints

- `PROFILES = ['12k','11k',…,'1k','1d',…,'6d']`（18 個，index 0–17，弱 → 強）。
- 硬分箱：D = {12k,11k,10k}、C = {9k,8k,7k}、B = {6k,5k,4k}、A = {3k,2k,1k}、1D…6D 各對應 1d…6d。
- 評分一律 FP32（不用 autocast），因為 0A 一致性只在 FP32 驗證過；batch 64（0B 最快）。
- 只計目標玩家（`color` 欄）的落子；log p 為遮罩非法步後的 FP32 `log_softmax`。
- 盤面特徵實測為 0/1 值 → 以 uint8 傳遞，worker 內 assert 二值，否則報錯。
- 玩家選取 seed 42；每級 20 人；只選在該 fold 有 `_n20_0` 題（≥ 35 局）的玩家，用該題的 20 局。
- fold 0 = 評估集，**不得在上面擬合任何參數**；fold 1 = 校準集（只用於單調門檻）。
- 多進程 Pool 必須在 CUDA 初始化前建立（fork）。
- 輸出：`results/pilot/`（大檔 `*.npz` 不進 git）、`results/pilot_humansl.json`（摘要，進 git）。

## Review Focus

1. 測試集可能有虛手：`game_positions` 遇到目標玩家虛手要輸出 target = 361、pass 合法（Task 2 測試）。
2. 目標執白時 metadata 我方／對方互換：`score_positions` 與 `target_move_logps` 對一局執白目標比對（Task 1 測試）。
3. AUC 遇到同分（posterior mean 接近整數）要算 0.5（Task 5 測試）。
4. 玩家不足 20 局：選取只收有 `_n20_0` 題者（Task 3 測試）。
5. 批次路徑與 0A 驗證過的單局面路徑必須一致到 1e-4（Task 1 GPU 測試）。

---

### Task 1: 批次評分核心

**Files:** Modify `src/aicup/humansl.py`；Test `tests/test_humansl_score.py`

**Interfaces — Produces:**
- `PROFILES: list[str]`
- `profile_meta_for(own: str, opp: str, target_color: str) -> SGFMetadata`：target 執黑 → black=own、white=opp；執白相反。
- `meta_rows(specs: list[tuple[str, str]]) -> np.ndarray`，形狀 `[S, 2, 192]` float32；`rows[s, c]` = 目標顏色 c（0=B, 1=W）且輪到目標下時的 metadata 列。
- `score_positions(model, bins, globs, legal, target, colors, rows, batch=64) -> np.ndarray[N, S] float32`：bins `[N,22,19,19]`（uint8 或 float）、globs `[N,19]`、legal `[N,362]` bool、target `[N]` int、colors `[N]` int（0/1）。

- [ ] 測試：`profile_meta_for('3d','5k','W')` 的 `inverseWRank==7`、`inverseBRank==14`；`meta_rows([('3d','3d')])[0,1]` 等於 `make_profile_meta('3d').get_metadata_row(nextPlayer=Board.WHITE, boardArea=361)`；GPU（skipif 無 ckpt）：train_1D 第一局黑目標、第一局白目標，`score_positions` 對 `('1d','1d')`、`('5k','5k')` 的結果與 `target_move_logps` 相差 < 1e-4。
- [ ] 跑測試看 FAIL → 實作 → PASS → commit。

### Task 2: 位置抽取（CPU worker）

**Files:** Create `src/aicup/pilot.py`；Test `tests/test_pilot.py`

**Interfaces — Produces:** `game_positions(sgf: str, target_color: str, features) -> dict`，鍵 `bins` uint8 `[n,22,19,19]`、`globs` float32 `[n,19]`、`legal` bool `[n,362]`、`target` int16 `[n]`、`t` int16 `[n]`、`color` int8 `[n]`；目標落子前的盤面，合法性用 `Board.would_be_legal`。

- [ ] 測試：`(;SZ[19];B[pd];W[dp];B[pq];W[];B[dd])` 目標 B → `t == [0,2,4]`、`legal[i].sum() == 362 - 石數`、`bins[1]` 等於重播到第 2 手時 `get_input_features` 的結果；目標 W → `t == [1,3]`，第 2 筆 target == 361。
- [ ] FAIL → 實作 → PASS → commit。

### Task 3: 玩家選取、題目、評分腳本

**Files:** Modify `src/aicup/pilot.py`；Create `scripts/pilot_score.py`；Test `tests/test_pilot.py`

**Interfaces — Produces:**
- `select_players(questions: DataFrame, fold: int, per_rank: int = 20, seed: int = 42) -> DataFrame[rank, player_id, game_keys]`（`game_keys` 為 list，來自該玩家 `_n20_0` 題）。
- `pilot_questions(players) -> DataFrame[question_id, rank, player_id, n_games, game_keys]`：每人 1 題 20 局、2 題 10 局（前後半）、4 題 5 局（四等分）。
- CLI：`scripts/pilot_score.py --fold F --shard i --nshards n [--workers W] [--limit-games K] [--opp-sweep]`，輸出 `results/pilot/games_fold{F}_shard{i}.csv`（game_key, player_id, rank, color, n_moves, 18 欄 `sum_<profile>`）與 `moves_fold{F}_shard{i}.npz`；`--opp-sweep` 輸出 `results/pilot/opp_sweep.csv`（rank, game_key, own, opp, n_moves, sum_logp）。

- [ ] 測試：選取只收有 n20 題的玩家、每級 ≤ per_rank、同 seed 結果相同；`pilot_questions` 題數 1/2/4、同大小題目棋局互斥、題號格式。
- [ ] FAIL → 實作 → PASS；lab01 上 `--limit-games 20` 冒煙跑，記錄每秒局面數 → commit。

### Task 4: 正式執行（lab01，兩張 GPU）

- [ ] fold 0 與 fold 1 各兩個 shard（GPU 0/1）；再跑 `--opp-sweep`。驗證：每 fold 200 人 × 20 局 = 4,000 局全部有結果，每局 18 欄無 NaN。
- [ ] 把 `results/pilot/*.csv` 拉回本機 → commit（npz 不進 git）。

### Task 5: 分析

**Files:** Create `src/aicup/evidence.py`、`scripts/pilot_analyze.py`；Test `tests/test_evidence.py`

**Interfaces — Produces:** `HARD_BINS`、`class_scores_hard_bins(L) -> [...,10]`、`posterior_mean_index(L) -> [...]`、`auc(pos, neg) -> float`（同分 0.5）、`fit_thresholds(s, y) -> np.ndarray[9]`（單調、最大化官方分，DP）、`apply_thresholds(s, cuts) -> int array`、`bootstrap_ci(per_player_scores, n=1000, seed=42) -> (lo, hi)`。

- [ ] 測試：硬分箱取箱內最大；AUC 完全分開 = 1、全同分 = 0.5；可完全分開的資料 `fit_thresholds` 得滿分且門檻單調；`posterior_mean_index` 對極端 L 回傳該 index。
- [ ] FAIL → 實作 → PASS。
- [ ] `pilot_analyze.py` 產出 `results/pilot_humansl.json`：(a) 各級似然曲線與最佳 profile 分布（10×18）；(b) 硬分箱零樣本官方分（n=5/10/20，含玩家 bootstrap 95% CI）與 10×10 混淆矩陣；(c) 每局正規化版本；(d) 相鄰等級 AUC；(e) fold 1 擬合單調門檻 → fold 0 官方分；(f) 對手等級敏感度表 → commit。

### Task 6: 記錄結果

- [ ] METHOD_NOTES §2.4 換成實測數字與 H1 判讀；ARCHITECTURE H1 狀態；記憶 → commit。
