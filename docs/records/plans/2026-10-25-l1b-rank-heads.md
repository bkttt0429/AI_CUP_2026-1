# L1b 每手證據模型（第一階段：凍結 humanv0 trunk + 10 個等級 head）— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用本資料的等級標籤訓練第一個 L1b 模型（humanv0 trunk 凍結、「等級未知」metadata、10 個從 humanv0 policy head 複製出來的等級 head），以兩路 out-of-fold 給 folds 1–4 出每手 10 維 log p，在 11/04 前以同一個聚合器、同一批題目對 Human SL 特徵做成對 CV 比較，回答 H2 的方向。

**Architecture:** 三段式。(1) **特徵快取**：每位訓練玩家 ≤ 40 局，KataGo 的 22 個二值輸入平面壓成 bit、連同全域特徵、合法步遮罩與目標落子存成分段檔，之後每個 epoch 都不再重播。(2) **訓練**：trunk 在 bf16、`no_grad` 下只做前向，輸出 `trunkfinal`；10 個 `PolicyHead` 的深拷貝各自只收自己等級的落子（類別平衡 CE、合法步遮 −inf），內層依玩家留出 10% 做提早停止。(3) **評估**：模型 A（folds 1、2）給 folds 3、4 出特徵，模型 B（folds 3、4）給 folds 1、2 出特徵，輸出與 `scripts/score_folds.py` 同一種檔案佈局；先看便宜的早期訊號（每手 NLL、相鄰等級 AUC），再跑 folds 1–4 的留一 fold CV，用事先指定的比較與門檻決定要不要往第二階段走。

**Tech Stack:** Python 3.12、torch 2.14.1、numpy、pandas；KataGo 的 Python 套件（`katago.train.model_pytorch.Model`／`PolicyHead`／`ExtraOutputs`、`katago.game.features.Features`、`katago.game.sgfmetadata.SGFMetadata`）；既有的 `aicup.scoring`、`aicup.pilot.game_positions`、`aicup.qfeatures`、`aicup.evidence`、`aicup.rankmodel`、`aicup.ladder.environment`。不加新套件。單張 H200 就夠，第一階段不寫多卡訓練。

**Spec:** GitHub issue #6（使用者決定與四點補充）；[METHOD_NOTES.md](../../../METHOD_NOTES.md) §2.2（晚注入）、§2.4.1(d)(e)（AUC 算法與 Human SL 基準）、§2.7「自訓 policy 的特徵必須 out-of-fold」與訓練長度那條、§2.8（CV 協議與 `rank_model.json` 的訓練題）、§3（0B 成本）；[docs/ARCHITECTURE.md](../../ARCHITECTURE.md) §2.1 L1b 列、§2.2「主提交候選」與「晚期融合」。

## Global Constraints

- **trunk 凍結、metadata 為「等級未知」**：`SGFMetadata` 與 `aicup.humansl.make_profile_meta` 相同（human、KGS、2020-03-01、讀秒 1200+30×5、ratedness unknown），但 `bRankIsUnknown = wRankIsUnknown = True`、`inverseBRank = inverseWRank = 0`（thermometer 全 0），所以 metadata 列第 4、5 欄為 1、第 6–73 欄為 0，黑白到手的列相同。
- **head 初始化**：10 個 head 都是 `model.policy_head` 的 `copy.deepcopy`；第 0 步每個 head 在同一批局面上的 policy log p 與 humanv0 在等級未知 metadata 下的 policy log p 相同（trunk 用 fp32 時 max |Δ| ≤ 1e-4，0A 的量級是 3.8e-4）。
- **只用目標玩家的落子**：每個局面是目標執色落子前的盤面，標籤是該玩家的等級；對手的落子不進訓練。
- **訓練資料**：folds 的所有玩家（`min_games=1`），每人 `fold_players(..., max_games=40)` 的前 40 局（同一個依 player_id 的 seed 42 排列，所以前 20 局就是評估用的那 20 局）。模型 A 用 folds 1、2（3,206 人、102,019 局、約 1.1×10⁷ 個目標落子），模型 B 用 folds 3、4（3,196 人、101,711 局）。
- **評估資料**：`fold_players(..., max_games=20, min_games=5)`，與 `results/scores/fold*/` 同一批局（每 fold 約 1,590 人、29,500 局）。A 給 folds 3、4（必要）與 0（選用），B 給 folds 1、2 與 0。
- **損失**：類別平衡 CE（`rankmodel.class_weights` 的語意套在局面上：每個等級的總權重相同、平均 1），合法步以外的點在 log-softmax 前設 −inf（與 `humansl._masked_policy_logp` 一致）。
- **訓練長度**：最多 3 個 epoch；內層驗證依玩家分層留出 10%（`settrain.INNER_SEED`，同一份快取的所有 seed 共用），每 1/4 epoch 在固定的 ≤ 4×10⁵ 個內層局面上算類別平衡每手 NLL，保留最好的權重（含第 0 步），連續 4 次沒改善就停。
- **最佳化**：Adam、lr 1e-4、無 weight decay、batch 256、heads fp32、trunk bf16 autocast 且 `torch.no_grad()`、seed 0；只有 head 的參數進 optimizer。
- **評分輸出佈局**：與 `scripts/score_folds.py` 相同：`<out>/fold{F}/games_s{shard}_p{k:03d}.csv`（`game_key, player_id, rank, color, n_moves, sum_<D…6D>` 10 欄）、`moves_s{shard}_p{k:03d}.npz`（`logp [M, 10]`、`game_row`、`t`），tmp 檔寫完才改名，可續跑；`sum_` 欄名用 `aicup.metric.LEVELS`。
- **CV 協議**：folds 1–4 留一 fold；驗證題 `pilot.pilot_questions`，訓練題 `qfeatures.sample_questions` 每人 32 題、`RandomState(42 + v)` 依訓練 fold 升冪抽（與 `fit_rank_model.train_set`、梯子相同）；模型 `MLPModel(hidden=32)`（單次確定性擬合）；主要指標 `balanced_mean_over_n`；成對 CI 用 `evidence.bootstrap_ci_mean_over_sizes`（以玩家為單位，n = 5000）。
- **事先指定的比較**（只有這兩組；不取 max）：`l1b` vs `hsl`、`fused` vs `hsl`。採用條件沿用梯子：① 成對 CI 下界 > 0；③ 4 個驗證 fold 至少 3 個點差 > 0；④ 點差 ≥ 0.005；② 不適用（單次確定性擬合），報告註明。
- **fold 0**：不在任何選擇裡用；A、B 都沒看過它，可以出特徵，但只當 L3 之後的訓練資料，不當評估。
- **nano4**：所有工作經 `scripts/nano4/run.sbatch`、加 `--exclude=25a-hgpn[001,003,004,007,062]`；輸出全部在 `results/l1b/`；快取、`.pt`、逐手 `.npz` 不進 git，`.json` 摘要與側檔進 git。新腳本都要有 `score_folds.py` 的 ROOT／`sys.path` 區塊；共用函式放 `src/aicup/`，不得 import `scripts/`。
- **git**：分支 `claude/festive-ptolemy-org8pz`（從 `main` @ `718911a`）；每個 task 一個 commit；PR 開在 `main`。

## Review Focus

1. **虛手**：目標玩家 pass 時 `target = 361`，合法步遮罩的 pass 位恆為 True，10 個 head 在 pass 位都要有有限的 log p（Task 1 `test_pass_move_has_finite_logp`）。
2. **沒有目標落子的局**（例如只有對手下了一手的殘局）：快取寫入器寫 `n_moves = 0`、不產生局面，讀回時所有陣列對齊（Task 2 `test_game_without_target_moves`）。
3. **內層子樣本缺某個等級**：類別權重永遠從完整訓練集算，不從內層子樣本算，所以不會除以 0；內層 NLL 缺級時只對有的等級平均並在側檔記 `inner_ranks_present`（Task 4 `test_inner_eval_with_missing_rank`）。
4. **續跑**：快取與評分的 part 檔以 tmp 名寫完再改名；中斷留下的 `*.tmp` 不算完成、重跑時覆寫（Task 2 `test_resume_ignores_tmp_parts`）。
5. **兩套評分的對齊**：早期訊號與 CV 要把 L1b 與 Human SL 在**相同的局、相同的落子**上比；game_key 或 t 對不上就報錯而不是靜默取交集（Task 6 `test_misaligned_scores_are_refused`）。

---

## 檔案結構

| 檔案 | 責任 |
| --- | --- |
| `src/aicup/l1b.py`（新） | 等級未知 metadata、`RankHeads`（10 個 head）、trunk 前向（`ExtraOutputs("trunkfinal")`）、遮罩 log-softmax、評分函式、head 的存取與側檔 |
| `src/aicup/featcache.py`（新） | 局面的 bit 打包／解包、part 檔的寫入（原子）與讀取、續跑計畫 |
| `src/aicup/l1btrain.py`（新） | 訓練設定 `HeadsCfg`、內層玩家留出、類別權重、訓練迴圈、checkpoint |
| `scripts/l1b_build_cache.py`（新） | CLI：folds → 快取 |
| `scripts/l1b_train.py`（新） | CLI：快取 → `heads_{A,B}.pt` + `.json`；`--check-init` |
| `scripts/l1b_score.py`（新） | CLI：heads + folds → `results/l1b/scores/fold{F}/`（`score_folds.py` 佈局） |
| `scripts/l1b_early_signal.py`（新） | CLI：每手 NLL 依等級、相鄰等級 AUC，L1b 對 Human SL |
| `scripts/l1b_cv.py`（新） | CLI：folds 1–4 留一 fold CV，`hsl`／`l1b`／`fused` 三組特徵與兩組事先指定的比較 |
| `src/aicup/scoring.py`（改） | `score_stream` 改收評分函式與欄名，`score_folds.py` 行為不變 |
| `src/aicup/qfeatures.py`（改） | `sum_cols(profiles)`、`feature_names(profiles)`、`build_matrix(..., profiles=PROFILES)` |
| `docs/NANO4.md`、`METHOD_NOTES.md`（改） | §4.2 指令；§2.9 L1b 設計與結果欄位 |
| `tests/test_l1b.py`、`tests/test_featcache.py`、`tests/test_l1btrain.py`、`tests/test_l1b_scripts.py`（新） | 測試；沒有 humanv0 權重時用 `modelconfigs.config_of_name["b2c16"]` 隨機初始化的 `Model`（沒有 metadata encoder，`input_meta=None`） |

---

### Task 1: `aicup.l1b` — 等級未知 metadata、RankHeads、trunk 前向、評分

**Files:**
- Create: `src/aicup/l1b.py`
- Test: `tests/test_l1b.py`

**Interfaces:**
- Consumes: `aicup.humansl.make_profile_meta`、`katago.train.model_pytorch.{Model, PolicyHead, ExtraOutputs}`、`aicup.metric.LEVELS`、`aicup.pilot.game_positions`。
- Produces:
  - `neutral_meta() -> SGFMetadata`；`neutral_rows() -> np.ndarray [2, 192] float32`（第 0 列黑到手、第 1 列白到手）。
  - `class RankHeads(torch.nn.Module)`：`__init__(self, policy_head: PolicyHead, n_ranks: int = 10)`；`forward(self, trunk, mask, mask_sum_hw, mask_sum) -> torch.Tensor [B, n_ranks, 362]`（每個 head 輸出的 policy 通道 0）。
  - `trunk_features(model, bins, globs, colors, rows=None) -> (trunk [B, C, 19, 19] float32, mask [B,1,19,19], mask_sum_hw [B,1,1,1], mask_sum)`：`rows` 為 `neutral_rows()`（模型沒有 metadata encoder 時忽略），`ExtraOutputs(["trunkfinal"])`，呼叫端決定 `no_grad`／autocast。
  - `masked_logp(logits, legal) -> [B, n_ranks, 362]`：非法點 −inf 後沿最後一軸 log-softmax。
  - `score_positions(model, heads, bins, globs, legal, target, colors, batch=64) -> np.ndarray [N, n_ranks] float32`：目標落子在每個 head 下的 log p（與 `humansl.score_positions` 同形）。
  - `save_heads(heads, path, meta: dict) -> dict`、`load_heads(path, model, device="cpu") -> (RankHeads, meta, warnings)`：`.pt` 是 `state_dict`，側檔 `path.with_suffix(".json")` 含 `format=1`、`n_ranks`、`levels`、`state_sha256`、`ladder.environment(device)` 的欄位與呼叫端給的 `meta`；sha256 不符或 `n_ranks` 不符 → `ValueError`。

- [ ] **Step 1: 寫失敗的測試**

```python
# tests/test_l1b.py
def _tiny_model():  # b2c16, no metadata encoder, random init, eval()
def test_neutral_meta_row_marks_rank_unknown_and_empty_thermometer():
    rows = neutral_rows()
    assert rows.shape == (2, 192) and np.array_equal(rows[0], rows[1])
    assert rows[0, 0] == rows[0, 1] == 1 and rows[0, 4] == rows[0, 5] == 1 and rows[0, 6:74].sum() == 0
def test_heads_start_equal_to_the_policy_head(tiny_model, positions):
    # 每個 head 的 masked log p == model 自己 policy 的 masked log p，fp32，atol 1e-6
def test_heads_are_independent_copies(tiny_model):
    # 改 head 3 的一個參數，head 0 與 model.policy_head 不變
def test_pass_move_has_finite_logp(tiny_model):
    # 一局含 pass 的 SGF（game_positions），target == 361，score_positions 的 10 欄都有限
def test_save_load_heads_round_trip(tmp_path, tiny_model):
    # 讀回後 score_positions 逐位元相同；改 .pt 一個 byte → ValueError("sha256")
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_l1b.py`
Expected: FAIL（`ModuleNotFoundError: aicup.l1b`）

- [ ] **Step 3: 實作 `src/aicup/l1b.py`**

`neutral_meta()` 從 `make_profile_meta("1d")` 改 4 個欄位。`RankHeads.forward` 對每個 head 呼叫 `head(trunk, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)`，取 `[:, 0, :]` 後 `torch.stack(dim=1)`。`trunk_features` 把 `mask = bins[:, 0:1]`、`mask_sum_hw = mask.sum((2, 3), keepdim=True)`、`mask_sum = mask.sum()` 與 `Model.forward` 同一算法，`extra.returned["trunkfinal"].float()`。`score_positions` 分批、`torch.no_grad()`，trunk 在 `torch.autocast("cuda", dtype=torch.bfloat16)`（device 為 cuda 時），heads fp32，`masked_logp(...).gather` 目標。

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_l1b.py`
Expected: PASS（5 個）

- [ ] **Step 5: Commit**

```bash
git add src/aicup/l1b.py tests/test_l1b.py
git commit -m "feat(l1b): rank-unknown metadata, RankHeads copied from the policy head, trunk features, scoring, save/load"
```

---

### Task 2: `aicup.featcache` — bit 打包的局面快取

**Files:**
- Create: `src/aicup/featcache.py`
- Test: `tests/test_featcache.py`

**Interfaces:**
- Consumes: `aicup.pilot.game_positions(sgf, target_color, features) -> {"bins" uint8 [n,22,19,19], "globs" float32 [n,19], "legal" bool [n,362], "target" int16 [n], "t" int16 [n]}`。
- Produces:
  - `N_PLANES = 22`、`N_GLOBAL = 19`、`N_MOVES = 362`、`BIN_BYTES = 993`（= ceil(22×361/8)）、`LEGAL_BYTES = 46`。
  - `pack_positions(pos: dict) -> dict`：`bins_packed uint8 [n, 993]`（`np.packbits(bins.reshape(n, -1), axis=1)`）、`legal_packed uint8 [n, 46]`、`globs float32 [n, 19]`、`target int16`、`t int16`。
  - `unpack_bins(packed) -> uint8 [n, 22, 19, 19]`、`unpack_legal(packed) -> bool [n, 362]`。
  - `write_part(part_stem: Path, games: pd.DataFrame, parts: list[dict]) -> None`：`games` 欄 `game_key, player_id, rank, color, n_moves`；位置陣列串接後加 `game_row int32 [N]`；寫 `<stem>.npz.tmp` 與 `<stem>.csv.tmp` 再 rename（npz 先）。
  - `completed_parts(cache_dir) -> list[(shard, k)]`、`read_cache(cache_dir) -> (games: pd.DataFrame, arrays: dict)`：所有完成的 part 串接，`arrays["game_row"]` 改成全域列號，`arrays["rank_idx"] int8` 由 `games["rank"]` 映射 `RANK_TO_IDX`，`arrays["player_id"]` 不展開（用 `games` 查）。
  - `plan_parts(keys, part_size, cache_dir, shard) -> list[(k, keys_k)]`：未完成的 part（與 `scoring.resume_plan` 同思路，但以 part 為單位：part k 的局固定為 `keys[k*part_size:(k+1)*part_size]`）。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_pack_unpack_round_trip():  # 隨機 0/1 平面與遮罩，解包逐位元相同；BIN_BYTES == 993
def test_write_read_part_aligns_games_and_positions(tmp_path):  # 3 局、各 5/0/7 個局面
def test_game_without_target_moves(tmp_path):  # n_moves 0 的局在 games 裡、不在 arrays 裡；read_cache 的 game_row 不指到它
def test_resume_ignores_tmp_parts(tmp_path):  # 寫一個 .npz.tmp；completed_parts 不含它；plan_parts 重排它
def test_read_cache_concatenates_parts_with_global_rows(tmp_path):
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_featcache.py`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 實作 `src/aicup/featcache.py`**

part 檔名 `cache_s{shard}_p{k:03d}.npz` / `games_s{shard}_p{k:03d}.csv`；`np.savez`（不壓縮，讀取快）。

- [ ] **Step 4: 跑測試確認通過**

Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_featcache.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aicup/featcache.py tests/test_featcache.py
git commit -m "feat(l1b): bit-packed position cache with atomic parts and resume"
```

---

### Task 3: `scripts/l1b_build_cache.py`

**Files:**
- Create: `scripts/l1b_build_cache.py`
- Test: `tests/test_l1b_scripts.py`（`test_build_cache_cli`）

**Interfaces:**
- Consumes: `scoring.player_games`、`scoring.fold_players(games, folds_df, fold, max_games, min_games)`、`scoring.load_games`、`scoring.init_worker`（Pool initializer；config 來自 humanv0 ckpt 或 `--model-config b2c16`）、`featcache.*`。
- Produces: CLI `python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A [--max-games 40] [--min-games 1] [--workers 48] [--part-size 200] [--shard i --nshards N] [--model-config NAME]`；`<out>/meta.json`（folds、max_games、min_games、players、games、positions、per_rank_positions、`questions`-style 的 `games_sha256`、`ladder.environment("cpu")`）由最後一個完成的進程寫（每個 shard 寫自己的 `meta_s{shard}.json`，`--merge-meta` 合併；單進程直接寫 `meta.json`）。
- 局的順序：`fold_players` 的輸出依 `(rank, player_id)` 排序後展開 game_keys，再 `RandomState(42).permutation` 打亂（讓每個 part 混等級），shard i 取 `keys[i::nshards]`。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_build_cache_cli(tmp_path, monkeypatch):
    # monkeypatch player_games / fold_players / load_games 給 6 位假玩家、12 局合成 SGF（含一局只有對手落子）
    # --model-config b2c16 --workers 2 --part-size 4 → 3 個 part；read_cache 的局數 == 11 有局面的局 + 1 個 n_moves 0；meta.json 的 positions == arrays 長度
    # 再跑一次 → 什麼都不重算（parts 已完成）、exit 0
```

- [ ] **Step 2: 跑測試確認失敗**　Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_l1b_scripts.py::test_build_cache_cli`　Expected: FAIL

- [ ] **Step 3: 實作**：Pool 在任何 CUDA 之前建立（fork）；`imap_unordered(extract, tasks)` 以 part 為單位收齊再 `write_part`；印每 part 一行（局數、局面數、秒數）。

- [ ] **Step 4: 跑測試確認通過**　Expected: PASS

- [ ] **Step 5: Commit**　`git commit -m "feat(l1b): cache builder CLI (folds -> packed positions, resumable, sharded)"`

---

### Task 4: `aicup.l1btrain` + `scripts/l1b_train.py`

**Files:**
- Create: `src/aicup/l1btrain.py`、`scripts/l1b_train.py`
- Test: `tests/test_l1btrain.py`、`tests/test_l1b_scripts.py`（`test_train_cli_and_check_init`）

**Interfaces:**
- Consumes: `featcache.read_cache`、`l1b.{RankHeads, trunk_features, masked_logp, save_heads, neutral_rows}`、`settrain.INNER_SEED`、`rankmodel.class_weights` 的語意、`ladder.environment`。
- Produces:
  - `@dataclass HeadsCfg(epochs=3, batch=256, lr=1e-4, eval_every=0.25, patience=4, inner_frac=0.1, inner_max_positions=400_000, seed=0, device="cuda")`。
  - `inner_split(games: pd.DataFrame, frac, seed=INNER_SEED) -> (train_game_rows, inner_game_rows)`：依玩家、按等級分層留出。
  - `position_weights(rank_idx: np.ndarray, n_ranks=10) -> np.ndarray`：每個等級總權重相同、平均 1；**用完整訓練局面算**。
  - `batches(arrays, rows, batch, rng) -> iterator of (bins float32 tensor, globs, legal bool, target long, rank long, colors long)`：解包在 CPU，再 `.to(device, non_blocking=True)`。
  - `evaluate_nll(model, heads, arrays, rows, cfg) -> {"nll": float, "nll_by_rank": [10], "ranks_present": [..]}`：類別平衡每手 NLL（真等級 head）。
  - `train_heads(model, cache_dir, cfg, out_pt, log=print, resume=True) -> (RankHeads, meta)`：訓練迴圈、每 `eval_every` epoch 評內層、保留最好、`patience`、checkpoint `<out_pt>.ckpt`（optimizer、step、best）；結束時 `save_heads` 加 `meta`：cfg、`train_folds`、`max_games`、`n_train_positions`、`n_inner_positions`、`per_rank_positions`、`inner_nll_init`、`inner_nll_best`、`best_epoch`、`stopped_early`、`inner_ranks_present`、`wall_seconds`、`humanv0_sha256`、`cache_meta`、`head_init="humanv0_policy_head"`。
  - `check_init(model, heads, arrays, rows, n=256) -> float`：trunk fp32、heads fp32，10 個 head 對 humanv0 等級未知 policy 的 max |Δ log p|。
  - CLI：`python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt [--ckpt models/katago/b18c384nbt-humanv0.ckpt] [--epochs 3] [--batch 256] [--lr 1e-4] [--seed 0] [--device cuda] [--check-init] [--overwrite]`；`--check-init` 只印 max |Δ| 並以 1e-4 為界 exit 0/1；輸出存在而沒有 `--overwrite` → exit 2。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_position_weights_balance_ranks():  # 每級總權重相同、平均 1；缺級不除以 0
def test_inner_split_is_by_player_and_stratified():  # 同一玩家只在一邊；每級內層玩家數 ≈ 10%
def test_inner_eval_with_missing_rank(tiny_cache):  # 內層只有 3 個等級 → nll 有限、ranks_present 長度 3
def test_train_heads_improves_inner_nll_and_resumes(tmp_path, tiny_model, tiny_cache):
    # cfg epochs=2, batch=8, eval_every=0.5, device cpu：inner_nll_best <= inner_nll_init；側檔欄位齊；中斷後 resume 從 ckpt 接著跑（step 數連續）
def test_check_init_is_zero_at_step_zero(tiny_model, tiny_cache):  # < 1e-6
def test_train_cli_and_check_init(tmp_path, ...):  # --check-init exit 0；訓練 → .pt + .json；再跑無 --overwrite → exit 2
```

- [ ] **Step 2: 跑測試確認失敗**　Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_l1btrain.py tests/test_l1b_scripts.py::test_train_cli_and_check_init`

- [ ] **Step 3: 實作**：損失 `(-logp_true_head * w).sum() / w.sum()`；`torch.manual_seed(cfg.seed)`、資料順序 `RandomState(cfg.seed)`；trunk `model.eval()`、`requires_grad_(False)`；每次評估印一行 `epoch x.xx inner NLL y (best z @ e)`。

- [ ] **Step 4: 跑測試確認通過**

- [ ] **Step 5: Commit**　`git commit -m "feat(l1b): head training with class-balanced CE, inner early stopping, checkpoints and --check-init"`

---

### Task 5: `scripts/l1b_score.py` 與 `scoring.score_stream` 的評分函式參數

**Files:**
- Modify: `src/aicup/scoring.py:50-90`（`score_stream`）
- Modify: `scripts/score_folds.py`（呼叫端改傳 `humansl` 的評分函式與 `PROFILES`；行為不變）
- Create: `scripts/l1b_score.py`
- Test: `tests/test_l1b_scripts.py`（`test_score_cli_matches_score_folds_layout`）、既有 `tests/test_scoring*.py` 必須全綠

**Interfaces:**
- `score_stream(pool, scorer, tasks, rows, flush_at=1024, log_every=200, log=print)`：`scorer(bins, globs, legal, target, colors, rows) -> [N, S]`；`humansl` 版本 `lambda *a: humansl.score_positions(model, *a)`；L1b 版本忽略 `rows`、呼叫 `l1b.score_positions(model, heads, ...)`。回傳值不變。
- CLI：`python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 --out results/l1b/scores [--max-games 20 --min-games 5] [--workers 10] [--shard i --nshards N] [--part-size 100]`；輸出佈局與 `score_folds.py` 相同、`sum_` 欄用 `LEVELS`；`<out>/fold{F}/meta_s{shard}.json` 記 heads 側檔摘要、環境、局數、局面數、秒數。
- 拒絕條件：heads 側檔的 `train_folds` 與 `--folds` 有交集 → exit 2（防止用看過的模型出特徵；`--allow-seen` 放行，只用於診斷，輸出目錄名要帶 `_seen`）。

- [ ] **Step 1: 寫失敗的測試**（tiny model、合成 SGF、monkeypatch 玩家）：輸出檔名與欄位和 `read_scored_games` 相容；初始 heads 的 10 欄 `sum_` 全部相等且等於 tiny model 等級未知 policy 的總和；`--folds` 含訓練 fold → exit 2。
- [ ] **Step 2: 跑測試確認失敗**
- [ ] **Step 3: 實作**（`score_stream` 重構後先跑 `tests/test_scoring*.py` 確認不變）
- [ ] **Step 4: 跑測試確認通過**　Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_l1b_scripts.py tests/test_scoring*.py`
- [ ] **Step 5: Commit**　`git commit -m "feat(l1b): out-of-fold scoring of the evaluation games under the 10 heads (score_folds layout)"`

---

### Task 6: `scripts/l1b_early_signal.py`

**Files:**
- Create: `scripts/l1b_early_signal.py`、`src/aicup/l1beval.py`（對齊與統計）
- Test: `tests/test_l1b_scripts.py`（`test_early_signal`、`test_misaligned_scores_are_refused`）

**Interfaces:**
- `l1beval.align_moves(l1b_dir, hsl_dir, fold) -> (games: pd.DataFrame, logp_l1b [M,10], logp_hsl [M,18], game_row [M], t [M])`：兩邊 `read_scored_games` + `moves_*.npz`；game_key 集合、每局 n_moves、每手 t 必須完全相同，否則 `ValueError`（Review Focus 5）。
- `per_rank_nll(logp_l1b, logp_hsl, rank_idx_per_move) -> {"l1b": [10], "hsl_best": [10], "hsl_best_profile": [10 個名稱], "hsl_pilot_peak": [10]}`：L1b 取真等級 head；`hsl_best` 為該等級在 18 個 profile 中平均 NLL 最低者（對 Human SL 樂觀）；`hsl_pilot_peak` 用 §2.4.1(a) 的峰值對應（D 6k、C 5k、B 4k、A 3k、1D 2k、2D 2k、3D 2k、4D 1k、5D 1d、6D 2d）。
- `adjacent_auc(sums_by_question, n_moves_by_question, rank) -> {pair: auc}`：每題相對曲線的總和 → `evidence.posterior_mean_index` → 相鄰等級 `evidence.auc`；題目用 `pilot.pilot_questions(fold_players)` 的 n = 20 題（與 §2.4.1(d) 相同），另報每局版本。
- CLI：`python -I scripts/l1b_early_signal.py --l1b results/l1b/scores --hsl results/scores --folds 3 4 --out results/l1b/early_signal_A.json`：印兩張表（每級 NLL：l1b／hsl_best／差；相鄰 AUC：l1b／hsl，加 pilot 基準 1D|2D 0.62、2D|3D 0.73、3D|4D 0.65 當參考列）。

- [ ] **Step 1: 寫失敗的測試**（合成兩套評分檔；一套 game_key 少一局 → `ValueError`；AUC 在完全可分的合成資料上為 1.0）
- [ ] **Step 2–4: 失敗 → 實作 → 通過**
- [ ] **Step 5: Commit**　`git commit -m "feat(l1b): early signal -- per-rank NLL and adjacent-rank AUC against Human SL on the same moves"`

---

### Task 7: `qfeatures` 一般化 + `scripts/l1b_cv.py`

**Files:**
- Modify: `src/aicup/qfeatures.py:13-60`
- Create: `scripts/l1b_cv.py`
- Test: `tests/test_qfeatures.py`（既有 + `test_profiles_parameter`）、`tests/test_l1b_scripts.py`（`test_cv_cli`）

**Interfaces:**
- `qfeatures.sum_cols(profiles) -> list[str]`、`qfeatures.feature_names(profiles) -> list[str]`（`FEATURE_NAMES == feature_names(PROFILES)`、`SUM_COLS == sum_cols(PROFILES)` 保持）；`build_matrix(games, questions, profiles=PROFILES)`。
- `l1b_cv.feature_sets(hsl_games, l1b_games, questions) -> {"hsl": X24, "l1b": X16, "fused": X40}`（`fused` = 兩者 `question_features` 串接）。
- CLI：`python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 [--per-player 32] [--n-boot 5000] --out results/l1b/cv_summary.json`（`--l1b` 可給多個目錄，每個 fold 只能在其中一個出現，重複 → exit 2）：每個驗證 fold v 三組特徵各一次 `MLPModel(hidden=32)` 擬合；摘要含每組的 `balanced_mean_over_n`、`balanced_by_n`、`per_fold`、類別平衡 NLL，與兩組事先指定比較的 `diff`、`ci95`、`per_fold_diff`、`conditions {"1": bool, "3": bool, "4": bool, "2": "n/a"}`、`pass`。
- L1b 特徵缺的 fold（例如 A 只給 3、4）：四個 fold 都必須有 L1b 評分，否則 exit 2 並列出缺的 fold。

- [ ] **Step 1: 寫失敗的測試**（合成 games／questions；`feature_names(["a","b"])` 長度 8；CLI 在合成資料上寫出摘要、兩組比較都有 `ci95`；缺 fold → exit 2）
- [ ] **Step 2–4: 失敗 → 實作 → 通過**　Run: `python3 -I -m pytest -q -p no:cacheprovider tests/test_qfeatures.py tests/test_l1b_scripts.py`
- [ ] **Step 5: Commit**　`git commit -m "feat(l1b): folds 1-4 CV of hsl / l1b / fused question features with the two pre-specified comparisons"`

---

### Task 8: 文件、全套測試、PR

**Files:**
- Modify: `docs/NANO4.md`（新增 §4.2「L1b 第一階段」）、`METHOD_NOTES.md`（新增 §2.9「L1b」：設計、事先指定的比較與門檻、結果欄位留白）

- [ ] **Step 1: NANO4 §4.2**：六個步驟的 `sbatch` 指令（見下方「nano4 工作」），每步的預期輸出與時間。
- [ ] **Step 2: METHOD_NOTES §2.9**：設計摘要（本計畫的 Global Constraints）、與 Human SL 的關係（微調而非蒸餾：推論成本不是瓶頸，蒸餾不排進；晚注入；L1a 與 L1b 的晚期融合留給 `fused` 特徵與之後的 L3）、兩路 OOF 與其代價、事先指定的比較與條件、「只寫成選項」的 A′／B′、合規風險影響整條管線。
- [ ] **Step 3: 全套測試**　Run: `python3 -I -m pytest -q -p no:cacheprovider tests`　Expected: 全綠（本機跳過的仍是缺資料／權重／引擎那些）
- [ ] **Step 4: Commit、push、開 PR（base `main`），PR 內文列 nano4 工作與預期值；在 issue #6 留 `[cloud Claude]`**

---

## nano4 工作（PR 合併後，順序固定；11/04 當天 `ens_top3` 正式提交優先）

| 步 | 指令（都加 `--exclude=25a-hgpn[001,003,004,007,062]`） | 預估 | 預期 |
| --- | --- | --- | --- |
| 0 | `sbatch -J tests run.sbatch python -m pytest -q` | 6 分 | 全綠 |
| 1 | `sbatch -c 48 -t 02:00:00 -J cacheA run.sbatch python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A --workers 48`；B 同（`--folds 3 4`、`cache_B`） | 每半 15–30 分（1.1×10⁷ 局面 ÷ 48 核 × 226/秒）、約 13 GB | `meta.json` 的 games ≈ 102,019／101,711 |
| 2 | `sbatch -t 01:00:00 -J initA run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt --check-init` | 2 分 | max \|Δ\| ≤ 1e-4 |
| 3 | `sbatch -t 08:00:00 -J trainA run.sbatch python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt`；B 同 | 每半 3 epoch ≤ 3 小時（單卡 bf16 前向約 4,000–6,000 局面/秒） | `inner_nll_best < inner_nll_init`；側檔進 git |
| 4 | `sbatch -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J scoreA run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 0 --out results/l1b/scores --workers 10`；B：`--heads heads_B.pt --folds 1 2 0 --out results/l1b/scores_B`（fold 0 兩邊都出，分開存） | 每半 10–30 分 | 每 fold 約 29,500 局；`meta_s*.json` |
| 5 | `sbatch -t 01:00:00 -J early run.sbatch python -I scripts/l1b_early_signal.py --l1b results/l1b/scores --hsl results/scores --folds 3 4 --out results/l1b/early_signal_A.json`；B 用 `--l1b results/l1b/scores_B --folds 1 2` | 幾分鐘 | 兩張表貼回 issue #6 → **決策點 1** |
| 6 | 把 `scores_B/fold{1,2}` 併入 `scores/`（或 `l1b_cv.py --l1b` 給兩個目錄）；`sbatch -t 02:00:00 -J l1bcv run.sbatch python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 --out results/l1b/cv_summary.json` | 20–40 分 CPU | 兩組比較的 `diff [ci95]` 與條件 → **決策點 2** |

決策點 1（早期訊號，一天內）：每級 NLL L1b 低於 `hsl_best` 的等級數、1D|2D／3D|4D 的 AUC 對 Human SL 的差。建議的讀法：L1b 在 ≥ 7 級較低、或兩個弱邊界的 AUC 都升 → 進 CV；兩者都沒有 → 第一階段的 head 容量不夠，直接規劃第二階段（解凍）而不跑 CV。決定由使用者下。
決策點 2（CV）：`l1b` vs `hsl`、`fused` vs `hsl` 以 ①③④ 判；任一組成立 → L1b 特徵進 L3（下一個計畫）；都不成立 → 看 `fused` 的點差方向決定要不要做第二階段。

## 算力與時程

- 快取：A、B 各約 1.1×10⁷ 局面 × 1,123 B ≈ 12–13 GB，建置 15–30 分（CPU 226 局面/秒/核 × 48 核；0B 端到端 45 落子/秒/進程的瓶頸就是這一步，快取後每個 epoch 不再付）。不快取的代價：每 epoch 重播 1.1×10⁷ 局面 ≈ 48 核 17 分，三個 epoch 近一小時 CPU，而且訓練時 GPU 等 CPU；快取後 GPU 滿載。trunk 輸出不快取（384 × 361 × bf16 ≈ 277 KB/局面，1.1×10⁷ 局面 ≈ 3 TB）。
- 訓練：單卡；trunk 前向 bf16 約 4,000–6,000 局面/秒（0B 在 RTX PRO 6000 是 4,296/秒；H200 未實測，第 2 步的 log 會給實際值），1.1×10⁷ 局面一個 epoch 30–45 分，3 epoch 加內層評估 ≤ 3 小時。兩半共 ≤ 6 GPU 小時。
- 評分：每半 3 個 fold 約 88,500 局、約 1×10⁷ 局面；8 卡分片 10–30 分。
- 時程（今天 10/25）：10/26–10/27 寫程式與測試、開 PR；10/28–10/30 nano4 跑步 1–5、決策點 1；10/31–11/02 CV、決策點 2；11/03 報名截止（使用者）；11/04 正式測試集，`ens_top3` 提交優先；11/05–11/14 依決策做第二階段或把 L1b 特徵接進 L3 與提交管線；11/15–11/21 凍結；11/22–11/24 最終上傳。nano4 單一工作 2 天上限對第一階段不是問題（每個工作 ≤ 8 小時）。

## 風險

1. **預訓練權重合規**（使用者要去官方表單問）：不只 L1b，現在的 `ens_top3` 提交路徑整條建在 humanv0 上。若不准：路線 2（自己的 trunk 從零訓練）是唯一出路。本計畫的 `RankHeads` 只依賴 `PolicyHead` 介面與 `trunk_features` 的回傳形狀，trunk 可換；但從零訓練的成本與分數都未知，要另開計畫、最晚 11/05 動工。
2. **沒有乾淨的 held-out**：fold 0 已用掉、也是集合模型的訓練資料；所有選擇只靠 folds 1–4 CV。CV 的 L3 用 OOF 特徵訓練、測試時用 A、B 各出一套特徵各走一次 L3 再平均，分布一致；若之後改成全量 L1b，要重新檢查特徵分布。
3. **CPU 特徵瓶頸**：由快取解決；快取格式與 `game_positions` 綁定（版本 10+ 的 22 個二值平面），KataGo 版本不能換。
4. **重度玩家**：每人 ≤ 40 局的上限與依玩家的內層留出；每級總權重相同。仍然是「每局一樣重」，不是「每人一樣重」，玩家層級的加權留給第二階段比較。
5. **head 容量**：只動 head 可能改不動 1D–3D；早期訊號一天內能看出來，再決定解凍。
6. **KGS 先驗**：凍結 trunk 保留 humanv0 的 KGS 2020 表徵；若 L1b 只是在重學等級尺度的對應，CV 增益會很小——這正是決策點 2 要回答的。

## 後續（不在第一階段；各開自己的計畫）

- 第二階段：解凍 trunk、小學習率、DDP 多卡、每人上限放寬或全量，checkpoint 以跨 2 天時限。
- A′／B′ 選項（issue #6）：A′ 用 folds 0、1、2、B′ 用 0、3、4，各看 60% 玩家；fold 0 因此沒有 OOF 特徵，L3 只能用 folds 1–4 訓練。用早期訊號與 A／B 比一次再決定。
- L3：`gamevec`／`ladder_build_vectors` 一般化到 10 個 profile，把 L1b（與 fused）向量餵給 `r3_pma`／`ens_top3`。
- 提交管線：`predict_submission.py` 的評分函式可換（humanv0 18 profile → L1b 10 head，或兩者都算、晚期融合）；分片檔多存 10 維 `move_logp`。
- 路線 2（合規備案）：自己的 trunk（例如 b6c96／b10c128 的 KataGo 配置、隨機初始化）接同一組 head；訓練程式只換 `--ckpt` 為 `--model-config`。
