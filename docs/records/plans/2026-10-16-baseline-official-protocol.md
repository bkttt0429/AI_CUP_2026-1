# 官方 baseline 在本專案驗證協議下重跑 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 METHOD_NOTES §2.1 的玩家分組協議重訓官方 Tutorial 棋力 baseline，並在 Human SL 管線使用的同一批 fold 0 題目上評分，得到可以直接比較的參考分。

**Architecture:** 特徵與網路直接以檔案路徑載入官方 `repo/AICup-2026-Tutorial/{utils,network}.py`（一字不改）；notebook 的抽取（多進程 webdataset 分片）、訓練（5 epoch、依 val accuracy 存最佳）與推論（每局 softmax 平均後 argmax）搬成三支腳本，程式碼與超參數照抄。唯一的協議改動是資料：訓練池只有 folds 1–4 的玩家，驗證集是 folds 1–4 裡整人抽出、每級約 10,000 局（合計 ≈ notebook 的 `VAL_SIZE` 100,000），fold 0 只在最後評估。

**Tech Stack:** Python 3.12、torch 2.14.1+cu130（FP32，cuDNN 預設 TF32 卷積）、webdataset 1.0.2、sgfmill、numpy、pandas、pytest。

**Spec:** 上層 session 的任務說明（要點抄在 Global Constraints）；[METHOD_NOTES.md](../../../METHOD_NOTES.md) §2.1「先在新協議下重跑原始 baseline 取參考分」；`repo/AICup-2026-Tutorial/rank-prediction.ipynb`（commit `359e06fc`）。

## Global Constraints

- 官方程式不改：`SGFParseRankPrediction(history_length=250)`、`GoRankResNet(in_channels=500, num_channels=256, num_blocks=20, num_classes=10)`，從 `repo/AICup-2026-Tutorial` 載入；結果 JSON 記錄該 repo 的 commit 與 `utils.py`、`network.py`、`rank-prediction.ipynb` 的 sha256。
- notebook 超參數照抄：`SEED=42`（`np.random.seed`、`torch.manual_seed`、`torch.cuda.manual_seed_all`）、`PER_SHARD=1000`、`BATCH_SIZE=256`、`LEARNING_RATE=1e-3`（`optim.AdamW` 其餘預設）、`NUM_EPOCHS=5`、`CrossEntropyLoss()`、`WebDataset(shards, shardshuffle=1 if shuffle else 0)` + `.shuffle(1000)` + `.decode().map(decode_sample).batched(BATCH_SIZE, partial=True)`、`DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)`、`pick_workers(candidate=4)`、最佳 checkpoint = val accuracy 最高的 epoch、推論 `GPU_BATCH=64`、每局 softmax 算術平均後 argmax。
- 抽取流程照 notebook：`sorted(glob("train_*.csv"))` 串接 → `df.sample(frac=1.0, random_state=42).reset_index(drop=True)` → 依玩家切 val／train（保留打亂後順序，先 val 後 train）→ worker i 取 `iloc[i::NPROC]` 自寫分片 `<tag>-w{i}-%06d.tar.gz`，`__key__ = f"{rank}_{game_id}"`，欄位 `features.npy`／`label.txt`／`rank_idx.txt`／`game_name.txt`；解析失敗跳過並計數；分片平均 < 10 KB 視為空分片而失敗。
- 協議：任何 fold 0 玩家不得出現在訓練或驗證；驗證集 = folds 1–4 依玩家整人抽出、每級累積到 ≥ 10,000 局為止（folds 1–4 每級約 79,300 局，訓練剩約 694,000 局，notebook 為 900,000）。
- 評估題 = `pilot_questions(fold_players(player_games(), player_folds, 0))`（1,594 人、29,649 局；n=5／10／20 各 5,795／2,798／1,279 題）。另附 pilot 的 fold 0 題（`pilot_questions(select_players(questions_fold0, 0))`，200 人）供直接對照 §2.4.1(e) 的 0.586／0.601／0.647。
- 兩種決策都報：notebook 的「平均 softmax → argmax」與同一平均機率的 `metric.expected_score_decision`。
- 指標：官方分的類別平衡（每級平均再平均）與題目平均，分 n=5／10／20，玩家層級 bootstrap 95% CI（`evidence.bootstrap_ci`；類別平衡用 `balanced=True`、題目平均用預設），n=1000、seed 42。頂層欄位名與 `scripts/fit_rank_model.py` 寫的 `results/rank_model_eval.json` 相同（`balanced_by_n`、`balanced_mean_over_n`、`question_mean`、`balanced_ci95`、`question_mean_ci95`、`per_class`、`exact`、`within1`、`confusion_rows_truth_cols_pred`）。
- 算力窗口：`scripts/scores_ctl.sh status` 顯示兩個 shard 都 stopped、每個 fold done markers 2/2 之前，不跑 GPU 訓練、不跑大量 CPU 抽取；冒煙測試只用 CPU、≤ 2 進程、`nice -n 19`、`CUDA_VISIBLE_DEVICES=`。不碰 `results/scores/`。
- 大檔留在 lab01：分片 `data/baseline_official/`、checkpoint `models/baseline_official/`（皆在 .gitignore）。進 git：程式、測試、`results/baseline_official_eval.json`、`results/baseline_official/game_probs_fold0.csv`。
- 在 lab01 的 `~/AICUP_2026-1` 以 `.venv/bin/python -I` 執行；只把本計畫新增／修改的檔案 scp 過去（不整包同步，避免蓋到其他 session 的檔案）。

## Review Focus

1. fold 0 洩漏：train／val 名單含任一 fold 0 玩家時抽取必須中止；評估腳本從 `{train,val}_games.csv` 再查一次並寫入 JSON（Task 1 測試、Task 3）。
2. webdataset 多 worker 重複讀或讀空：`pick_workers` 要對照「實際寫入的 val 筆數」而非 100,000；全空分片要被平均大小檢查擋下（Task 1、Task 2 測試）。
3. fold 0 某局解析失敗：照 notebook 從該題平均中剔除，整題無局時預測 `LEVELS[0]`，失敗局數記入 JSON（Task 3 測試）。
4. 聚合要平均機率（softmax），不是 logits；兩種決策用同一個平均（Task 3 測試）。
5. 重跑不能混到舊分片：輸出目錄非空時抽取拒絕執行（Task 1 測試）。

---

### Task 1: 玩家切分與分片抽取

**Files:** Create `src/aicup/baseline.py`、`scripts/baseline_extract.py`；Test `tests/test_baseline.py`

**Interfaces — Produces:**
- `baseline.TUTORIAL_DIR = ROOT / "repo" / "AICup-2026-Tutorial"`
- `baseline.load_tutorial(tutorial_dir=TUTORIAL_DIR) -> tuple[type, type]`：`(SGFParseRankPrediction, GoRankResNet)`，以 `importlib.util.spec_from_file_location` 載入為 `aicup_tutorial_utils`／`aicup_tutorial_network`，同一 process 只載一次。
- `baseline.game_features(sgf: str) -> np.ndarray | None`：notebook 的 `sgf_to_features`（uint8 `[500, 19, 19]`，解析失敗回 None）；parser 每個 process 延遲建立一次，可直接給 `Pool` 用。
- `baseline.split_players(folds: DataFrame[rank, player_id, n_games, fold], train_folds=(1, 2, 3, 4), val_games_per_rank=10_000, seed=42) -> tuple[list[str], list[str]]`：(train_ids, val_ids)，皆排序。
- `baseline_extract.extract_split(df_part: DataFrame[player_id, game_id, rank, sgf_content], out_dir: Path, tag: str, nproc: int, per_shard: int, min_shard_bytes: int = 10 * 1024) -> dict`：`{written, bad, shards, bytes}`。
- CLI `scripts/baseline_extract.py [--out data/baseline_official] [--nproc 16] [--per-shard 1000] [--val-games-per-rank 10000] [--seed 42] [--limit N]` → `<out>/{val,train}/<tag>-w{i}-%06d.tar.gz`、`<out>/{val,train}_games.csv`（game_key, player_id, rank, fold）、`<out>/split.json`（各 split 的玩家數、局數、written、bad、分片數、bytes；val 每級局數）。`--limit` 只取每個 split 打亂後的前 N 局（冒煙用）。

`split_players` 的抽法（測試定不了順序，所以寫明）：`rng = np.random.RandomState(seed)`；依 `metric.LEVELS` 順序處理 train_folds 中出現的每一級；該級玩家依 `player_id` 排序後以 `rng.permutation` 打亂，依序加入 val 直到累積 `n_games ≥ val_games_per_rank`；若該級全部局數都進了 val 則 `ValueError`。

- [ ] **Step 1: 寫失敗測試**（fixture `FOLDS`：D、6D 各 8 位 folds 1–4 玩家，局數 1…8（每級 36 局），外加每級 1 位 fold 0 玩家 50 局；`GAME = "(;SZ[19];B[pd];W[dp];B[];W[dd])"`；`DF5` = 4 局好棋（D 兩局、6D 兩局）+ 1 局 `"not an sgf"`）

```python
def test_split_players_excludes_other_folds_and_is_player_disjoint():
    train, val = split_players(FOLDS, val_games_per_rank=10)
    pool = set(FOLDS.loc[FOLDS["fold"].isin([1, 2, 3, 4]), "player_id"])
    assert set(train) | set(val) == pool and not set(train) & set(val)
    assert train == sorted(train) and val == sorted(val)

def test_split_players_takes_just_enough_players_per_rank():
    train, val = split_players(FOLDS, val_games_per_rank=10)
    g, r = FOLDS.set_index("player_id")["n_games"], FOLDS.set_index("player_id")["rank"]
    for rank in ("D", "6D"):
        v = [p for p in val if r[p] == rank]
        assert g[v].sum() >= 10 and g[v].sum() - g[v].max() < 10
        assert any(r[p] == rank for p in train)

def test_split_players_is_deterministic_and_ignores_row_order():
    assert split_players(FOLDS.sample(frac=1.0, random_state=3), val_games_per_rank=10) == \
        split_players(FOLDS, val_games_per_rank=10)

def test_split_players_refuses_to_take_a_whole_rank():
    with pytest.raises(ValueError):
        split_players(FOLDS, val_games_per_rank=36)

def test_game_features_is_the_official_parser_output():          # tutorial 不在時 skip
    f = game_features(GAME)
    ref, err = load_tutorial()[0]().features_from_sgf_content(GAME)
    assert err is None and np.array_equal(f, ref) and f.shape == (500, 19, 19) and f.dtype == np.uint8
    assert f[:492].sum() == 0 and f[492].sum() == 1 and f[498].sum() == 1 and f[499].sum() == 2
    assert np.array_equal(f[494:496], f[496:498])                 # 虛手不改盤面
    assert game_features("not an sgf") is None

def test_tutorial_network_outputs_ten_logits():
    net = load_tutorial()[1](num_classes=10).eval()
    assert tuple(net(torch.zeros(2, 500, 19, 19)).shape) == (2, 10)

def test_extract_split_round_trip(tmp_path):
    stats = extract_split(DF5, tmp_path, "val", nproc=2, per_shard=1, min_shard_bytes=0)
    assert (stats["written"], stats["bad"], stats["shards"]) == (4, 1, 4)
    samples = list(wds.WebDataset(sorted(str(p) for p in (tmp_path / "val").glob("*.tar.gz")), shardshuffle=0).decode())
    assert sorted(s["__key__"] for s in samples) == ["6D_g_1", "6D_g_3", "D_g_1", "D_g_2"]
    for s in samples:
        assert s["rank_idx.txt"] == str(RANK_TO_IDX[s["label.txt"]])
        assert np.array_equal(s["features.npy"], game_features(SGF_OF[s["__key__"]]))

def test_extract_split_refuses_old_shards_and_flags_empty_ones(tmp_path):
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "old.tar.gz").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        extract_split(DF5, tmp_path, "val", nproc=1, per_shard=1)
    with pytest.raises(SystemExit):
        extract_split(DF5, tmp_path, "train", nproc=1, per_shard=1, min_shard_bytes=10**9)
```

- [ ] **Step 2:** lab01 `.venv/bin/python -I -m pytest tests/test_baseline.py -q` → FAIL（`aicup.baseline` 不存在）。
- [ ] **Step 3:** 實作 `baseline.py` 的四個介面與 `baseline_extract.py`：`extract_worker` 照 notebook，結束時把 `{written, bad}` 寫到 `<out>/<tag>/<tag>-w{i}.json`；`Process` 用 fork context；主程式寫分片前 assert train／val 玩家與非訓練 fold 的玩家無交集。
- [ ] **Step 4:** 同一指令 → PASS；`tests/` 其餘測試仍過。
- [ ] **Step 5:** lab01 冒煙（CPU 輕量）：`nice -n 19 .venv/bin/python -I scripts/baseline_extract.py --out data/baseline_smoke --limit 64 --nproc 2 --per-shard 8` → `split.json` 中 val、train 各 written 64、bad 0、各 8 個分片。
- [ ] **Step 6:** commit `feat: player-grouped split and shard extraction for the official baseline`

### Task 2: 訓練腳本

**Files:** Create `scripts/baseline_train.py`；Test `tests/test_baseline.py`（追加）

**Interfaces — Consumes:** Task 1 的分片與 `split.json` 的 `val.written`、`load_tutorial`、`extract_split`（測試用）。
**Produces:**
- `decode_sample(sample) -> (Tensor float32 [500,19,19], Tensor long, str)`、`make_loader(shards, workers, shuffle, batch_size) -> DataLoader`、`pick_workers(val_shards, expected, batch_size, candidate=4) -> int`、`validate(model, loader, criterion, device) -> (loss, acc_percent)`：本體照 notebook，只把 `BATCH_SIZE`、`VAL_SIZE`、`DEVICE` 改成參數。
- CLI `scripts/baseline_train.py [--features data/baseline_official] [--ckpt models/baseline_official] [--epochs 5] [--batch-size 256] [--lr 1e-3] [--seed 42] [--workers 4] [--device cuda]` → `<ckpt>/best_model.pth`、`epoch_{k}.pth`、`history.csv`（epoch, train_acc, val_loss, val_acc, minutes）、`run.json`（參數、裝置名、torch 版本、TF32 設定、實際 worker 數、train／val 筆數、best_epoch、best_val_acc、總分鐘）。

- [ ] **Step 1: 寫失敗測試**（`DF6` = 6 局好棋）

```python
def test_make_loader_reads_every_sample_once_and_pick_workers_falls_back(tmp_path):
    extract_split(DF6, tmp_path, "val", nproc=2, per_shard=1, min_shard_bytes=0)     # 6 個分片
    shards = sorted(str(p) for p in (tmp_path / "val").glob("*.tar.gz"))
    assert sum(len(y) for _x, y, _n in make_loader(shards, 2, False, batch_size=4)) == 6
    assert pick_workers(shards, expected=6, batch_size=4, candidate=2) == 2
    assert pick_workers(shards, expected=7, batch_size=4, candidate=2) == 0
    x, y, _ = next(iter(make_loader(shards, 0, True, batch_size=4)))
    assert x.dtype == torch.float32 and tuple(x.shape) == (4, 500, 19, 19) and y.dtype == torch.long
```

- [ ] **Step 2:** → FAIL（`baseline_train` 不存在）。
- [ ] **Step 3:** 實作 `scripts/baseline_train.py`，順序照 notebook：設 seed → 建 loader、`pick_workers` → 建模型與 optimizer → 訓練迴圈（每 500 step 印一次）→ 每 epoch `validate`，`val_acc > best` 存 `best_model.pth`，每 epoch 存 `epoch_{k}.pth`。
- [ ] **Step 4:** → PASS。
- [ ] **Step 5:** lab01 冒煙（CPU）：`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -I scripts/baseline_train.py --features data/baseline_smoke --ckpt models/baseline_smoke --epochs 2 --batch-size 8 --device cpu` → `history.csv` 2 列、`best_model.pth` 存在、log 顯示 4 個 worker 驗證通過。
- [ ] **Step 6:** commit `feat: official baseline training script under the player-grouped split`

### Task 3: fold 0 評估

**Files:** Modify `src/aicup/baseline.py`；Create `scripts/baseline_eval.py`；Test `tests/test_baseline.py`（追加）

**Interfaces — Consumes:** `load_tutorial`、`game_features`、`scoring.{player_games, fold_players, load_games}`、`pilot.{pilot_questions, select_players}`、`evidence.bootstrap_ci`、`metric.{expected_score_decision, score, LEVELS, RANK_TO_IDX}`、Task 1 的 `{train,val}_games.csv` 與 `split.json`、Task 2 的 `best_model.pth`、`run.json`、`history.csv`。
**Produces:**
- `baseline.mean_probs(game_probs: dict[str, np.ndarray], questions: Iterable[list[str]]) -> np.ndarray[Q, 10]`：每題可用局的 softmax 算術平均；整題無可用局 → 該列 NaN。
- `baseline.decide(probs: np.ndarray[Q, 10], rule: str) -> np.ndarray[int]`：`rule ∈ {"argmax", "expected_score"}`；NaN 列 → 0（notebook 的 `LEVELS[0]`）。
- `baseline.report(pred, y, qs: DataFrame[player_id, n_games], sizes=(5, 10, 20)) -> dict`：頂層 `balanced_by_n`、`balanced_mean_over_n`、`question_mean`、`balanced_ci95`、`question_mean_ci95`（全部大小合併，與 `fit_rank_model.py` 同義）、`per_class`、`exact`、`within1`、`confusion_rows_truth_cols_pred`；`by_n[n] = {balanced, balanced_ci95, question_mean, question_mean_ci95, n_questions, exact, within1, per_class}`。
- CLI `scripts/baseline_eval.py [--ckpt models/baseline_official/best_model.pth] [--features data/baseline_official] [--nproc 16] [--device cuda] [--limit-players N] [--out results/baseline_official_eval.json] [--probs-out results/baseline_official/game_probs_fold0.csv]`。

- [ ] **Step 1: 寫失敗測試**

```python
def test_mean_probs_averages_probabilities_and_marks_empty_questions():
    a, b = np.eye(10)[0] * 0.8 + np.eye(10)[1] * 0.2, np.eye(10)[0] * 0.2 + np.eye(10)[1] * 0.8
    P = mean_probs({"a": a, "b": b}, [["a", "b"], ["missing"], ["a", "missing"]])
    assert np.allclose(P[0], (a + b) / 2) and np.isnan(P[1]).all() and np.allclose(P[2], a)

def test_decide_rules_differ_on_neighbour_mass_and_fall_back_to_class_0():
    P = np.array([[0.30, 0.30, 0.40] + [0.0] * 7, [np.nan] * 10])
    assert decide(P, "argmax").tolist() == [2, 0]
    assert decide(P, "expected_score").tolist() == [1, 0]

def test_report_separates_class_balanced_from_question_mean():
    qs = pd.DataFrame({"player_id": ["p1", "p2", "p3", "p4"], "n_games": [5, 5, 5, 5]})
    r = report(np.array([0, 0, 0, 3]), np.array([0, 0, 0, 1]), qs, sizes=(5,))
    assert r["question_mean"] == pytest.approx(0.75) and r["balanced_by_n"][5] == pytest.approx(0.5)
    assert r["balanced_mean_over_n"] == pytest.approx(0.5) and r["by_n"][5]["n_questions"] == 4
    perfect = report(np.array([0, 1]), np.array([0, 1]), qs.iloc[:2], sizes=(5,))
    assert perfect["by_n"][5]["balanced_ci95"] == [1.0, 1.0] and perfect["exact"] == 1.0
```

- [ ] **Step 2:** → FAIL。
- [ ] **Step 3:** 實作三個函式與 `scripts/baseline_eval.py`：
  - 題目：`fold0` 與 `pilot_fold0`（見 Global Constraints）；`--limit-players N` 各取前 N 位玩家（冒煙用）。
  - 洩漏檢查：`{train,val}_games.csv` 的玩家與 fold 0 玩家交集必須為空，否則中止；交集大小寫入 JSON。
  - `Pool(nproc)` 在 CUDA 初始化前建立；`pool.imap(game_features, sgfs, chunksize=16)` 串流，每 64 局一個 batch，`softmax(1)`；解析失敗的局不進 `game_probs` 並計數。
  - 兩組題 × 兩種決策各一份 `report`；另記 fold 0 每局 argmax 準確率。
  - JSON 另含 tutorial commit 與 sha256、與 notebook 的差異清單、`split.json` 計數、`run.json`、`history.csv`、checkpoint 路徑、失敗局數；每局機率寫 `--probs-out`（`game_key`、`p_D`…`p_6D`，6 位小數）。
- [ ] **Step 4:** → PASS。
- [ ] **Step 5:** lab01 冒煙（CPU）：用 Task 2 的冒煙 checkpoint，`--features data/baseline_smoke --ckpt models/baseline_smoke/best_model.pth --limit-players 5 --device cpu --nproc 2 --out /tmp/baseline_eval_smoke.json --probs-out /tmp/baseline_probs_smoke.csv` → JSON 兩組題、兩種決策俱全、`failed_games` 0。
- [ ] **Step 6:** commit `feat: fold-0 evaluation of the official baseline with both decision rules`

### Task 4: 正式執行與記錄

- [ ] **Step 1:** 等 `scripts/scores_ctl.sh status` 顯示兩個 shard stopped、每個 fold done markers 2/2；`nvidia-smi` 確認 GPU 0 閒置。
- [ ] **Step 2:** 抽取 `setsid nohup .venv/bin/python -I scripts/baseline_extract.py --nproc 16 > data/baseline_official_extract.log 2>&1 &`；檢查 `split.json`：val 每級 ≥ 10,000 局、train + val 玩家 = folds 1–4 的 6,402 人、written + bad = 各 split 局數。
- [ ] **Step 3:** 訓練 `CUDA_VISIBLE_DEVICES=0 setsid nohup .venv/bin/python -I scripts/baseline_train.py > models/baseline_official/train_log.txt 2>&1 &`（先 `mkdir -p models/baseline_official`）；完成後看 `history.csv`。
- [ ] **Step 4:** 評估 `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I scripts/baseline_eval.py` → `results/baseline_official_eval.json`、`results/baseline_official/game_probs_fold0.csv`；scp 回本機。
- [ ] **Step 5:** METHOD_NOTES §2.1「先在新協議下重跑原始 baseline」那一點補一行結果：fold 0 n=5／10／20 類別平衡分與 CI、`balanced_mean_over_n`、兩種決策、pilot 題對照 §2.4.1(e)、訓練規模與最佳 epoch。
- [ ] **Step 6:** commit `data: official baseline under the player-grouped protocol`（JSON、CSV、METHOD_NOTES）
