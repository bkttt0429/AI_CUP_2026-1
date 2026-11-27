# 步驟 0：Human SL 一致性檢查、吞吐量測與玩家分組驗證 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 lab01 上建立三樣東西：(a) 用 PyTorch 直讀 `b18c384nbt-humanv0.ckpt` 算出的目標落子 log p 與 KataGo analysis engine 一致（0A）；(b) 純推論與端到端吞吐的實測數字（0B）；(c) 依玩家分組的 5-fold 切分與 5/10/20 局假題，以及官方指標的實作。

**Architecture:** 程式碼放本機 `D:\AICUP_2026-1\src\aicup\`（套件）與 `scripts\`（可執行腳本），每次執行前 `scp` 到 lab01 `~/AICUP_2026-1/` 同路徑，在 `.venv` 內跑。KataGo 的 Python 套件（`repo/KataGo/python/katago`）負責盤面重建、特徵、metadata 編碼與模型載入；我們只寫 SGF 解析、profile 重現、批次與比較。模型與二進位檔放 `models/` 與 `tools/`，與程式碼分開。

**Tech Stack:** Python 3.12、torch 2.14（cu130）、numpy、pandas、pytest；KataGo v1.15.0（humanv0 同版釋出）Eigen-AVX2 CPU 二進位當引擎參考；KataGo master 的 `python/katago`。

**Spec:** [METHOD_NOTES.md](../../../METHOD_NOTES.md) §1（事實）、§2.4–2.5（pilot 與步驟 0）、§2.1（驗證協議）。

## Global Constraints

- 所有讀資料的 Python 以 `.venv/bin/python -I` 執行；資料目錄 `data/`、`models/`、`tools/`、`repo/` 內不放我們的程式。
- 等級順序固定 `LEVELS = ['D','C','B','A','1D','2D','3D','4D','5D','6D']`（index 0–9）。
- 官方指標：完全正確 1、相鄰 `math.exp(-1)`、其他 0；越界鄰居視為 0。
- Human SL profile 必須完整重現引擎 `makeBasicRankProfile`：雙方同級、`bIsHuman=wIsHuman=True`、`gameRatednessIsUnknown=True`、`tcIsUnknown=False`、`tcIsByoYomi=True`、`mainTimeSeconds=1200`、`periodTimeSeconds=30`、`byoYomiPeriods=5`、`gameDate=2020-03-01`（`preaz` 為 2016-09-01）、`source=SGFMetadata.SOURCE_KGS`（=2）。inverse rank：`Nd → 10-N`，`Nk → 9+N`（`3d→7`、`5k→14`、`1d→9`、`1k→10`、`20k→29`）。
- 規則兩側一致：Python 用 `GameState.RULES_CHINESE`，引擎 query 用 `"rules":"chinese","komi":7.5`。
- 引擎設定：`nnRandomize = false`、`numSearchThreads = 1`、query 帶 `maxVisits:1, includePolicy:true, overrideSettings:{humanSLProfile:<profile>, ignorePreRootHistory:false}`。
- log p 一律 FP32 `torch.log_softmax`，非法步先遮成 `-inf`。
- 隨機種子 `SEED = 42`。

## Review Focus

1. 測試集 SGF 可能含虛手 `B[]`／`B[tt]`：解析器須回傳 pass，重播用 `Board.PASS_LOC`（Task 2 測試）。
2. 引擎 `policy` 陣列的座標方向若與 Python `features.loc_to_tensor_pos` 不同，單看目標落子抓不到：Task 5 比較整個 362 維向量。
3. 目標玩家執白時 metadata 的我方／對方交換由 `get_metadata_row(nextPlayer)` 處理：Task 5 的位置集必含執白目標（兩局）。
4. 引擎對非法步輸出 −1 且合法步機率總和 1，Python 原始 softmax 含非法步質量：Task 3 先遮非法步再 `log_softmax`，Task 5 若不符則記錄差異來源。
5. 提子後的歷史平面：Task 5 的位置集含至少一個「剛發生提子」的局面。

---

### Task 1: lab01 環境與 KataGo 資產

**Files:**
- Create (lab01): `~/AICUP_2026-1/tools/katago/`（二進位）、`~/AICUP_2026-1/models/katago/`（兩個模型檔）、`~/AICUP_2026-1/repo/KataGo/`（git clone，只用 `python/`）
- Create (lab01): `.venv/lib/python3.12/site-packages/katago.pth`，內容一行 `/home/lab01/AICUP_2026-1/repo/KataGo/python`
- Create (local + lab01): `scripts/sync_to_lab01.sh` — `scp -r src scripts tests lab01:~/AICUP_2026-1/`

**下載清單（來源皆為 github.com/lightvector/KataGo Releases v1.15.0）：**

| 檔名 | 大小 |
| --- | --- |
| `b18c384nbt-humanv0.bin.gz` | 99,066,230 B |
| `b18c384nbt-humanv0.ckpt` | 323,752,318 B |
| `katago-v1.15.0-eigenavx2-linux-x64.zip` | 41,250,172 B |

- [ ] **Step 1:** 下載三個檔案到各自目錄；`unzip` 二進位到 `tools/katago/`；`chmod +x tools/katago/katago`。
- [ ] **Step 2:** 驗證：`tools/katago/katago version` 印出 `1.15.0`。
- [ ] **Step 3:** `git clone --depth 1 https://github.com/lightvector/KataGo repo/KataGo`；寫入 `katago.pth`。
- [ ] **Step 4:** 驗證：`.venv/bin/python -I -c "from katago.train.load_model import load_model; from katago.game.gamestate import GameState; from katago.game.sgfmetadata import SGFMetadata; print(SGFMetadata.SOURCE_KGS)"` 印出 `2`。
- [ ] **Step 5:** 驗證引擎能載入人類模型：送一筆 `{"id":"t","moves":[],"rules":"chinese","komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":1,"includePolicy":true,"overrideSettings":{"humanSLProfile":"rank_3d"}}` 給 `tools/katago/katago analysis -model models/katago/b18c384nbt-humanv0.bin.gz -config tools/katago/analysis_example.cfg -override-config nnRandomize=false,numSearchThreads=1`，回傳含 `"policy"` 的 JSON，長度 362。
- [ ] **Step 6:** 安裝 `pytest` 到 `.venv`；本機建立 `src/aicup/__init__.py`、`tests/__init__.py`。

### Task 2: SGF 解析與座標轉換

**Files:**
- Create: `src/aicup/sgf.py`
- Test: `tests/test_sgf.py`

**Interfaces:**
- Produces: `parse_moves(sgf: str) -> list[tuple[str, int | None, int | None]]`，每項 `(color, x, y)`，`color ∈ {'B','W'}`，`x` 欄 0–18 由左到右、`y` 列 0–18 由上到下；pass 為 `(color, None, None)`。
- Produces: `board_size(sgf: str) -> int`。
- Produces: `to_gtp(x: int, y: int, size: int = 19) -> str`（欄字母 `ABCDEFGHJKLMNOPQRST`，列 `size - y`）；`from_gtp(s: str, size: int = 19) -> tuple[int, int]`。

- [ ] **Step 1:** 寫測試：`parse_moves("(;SZ[19];B[pd];W[dp];B[])") == [('B',15,3),('W',3,15),('B',None,None)]`；`parse_moves("(;SZ[19];W[tt])") == [('W',None,None)]`；`to_gtp(15,3) == 'Q16'`、`to_gtp(3,15) == 'D4'`、`to_gtp(8,0) == 'J19'`；`from_gtp('Q16') == (15,3)`；`board_size("(;SZ[19];B[pd])") == 19`。
- [ ] **Step 2:** 跑 `pytest tests/test_sgf.py -v`，預期 FAIL（module not found）。
- [ ] **Step 3:** 實作（用 `re.finditer(r";([BW])\[([a-s]{2}|tt)?\]", sgf)`）。
- [ ] **Step 4:** 跑測試，預期 PASS。

### Task 3: Human SL profile 重現與目標落子 log p

**Files:**
- Create: `src/aicup/humansl.py`
- Test: `tests/test_humansl.py`（不需 GPU、不需模型檔的部分）

**Interfaces:**
- Consumes: Task 2 的 `parse_moves`。
- Produces: `inverse_rank(s: str) -> int`。
- Produces: `make_profile_meta(rank: str, opp_rank: str | None = None, preaz: bool = False) -> SGFMetadata`。
- Produces: `load_humansl(ckpt_path: str, device: str, use_swa: bool = False) -> tuple[Model, Features]`（呼叫 `load_model(ckpt, use_swa, device, pos_len=19)` 取 `model`，`Features(model.config, 19)`）。
- Produces: `target_move_logps(model, features, moves, target_color: str, meta: SGFMetadata, rules=None) -> list[dict]`，每個目標落子一筆 `{'t': 手數索引, 'gtp': str, 'logp': float, 'policy_logp': np.ndarray(362)}`；對每個 `t` 先在**落子前**的 `GameState` 上評估，再 `gs.play`。非法步用 `board.would_be_legal(pla, loc)` 遮 `-inf`（pass 永遠合法）；`policy_logp` 為遮罩後 `log_softmax` 的整個向量。

- [ ] **Step 1:** 寫測試：`inverse_rank('3d')==7`、`inverse_rank('5k')==14`、`inverse_rank('1d')==9`、`inverse_rank('1k')==10`、`inverse_rank('20k')==29`；`m = make_profile_meta('3d')` 檢查 `m.inverseBRank==m.inverseWRank==7`、`m.bIsHuman and m.wIsHuman`、`m.tcIsByoYomi and not m.tcIsUnknown`、`m.mainTimeSeconds==1200`、`m.periodTimeSeconds==30`、`m.byoYomiPeriods==5`、`m.gameDate==datetime.date(2020,3,1)`、`m.source==2`、`m.gameRatednessIsUnknown`；`make_profile_meta('3d','5k').inverseWRank==14`；`make_profile_meta('5k',preaz=True).gameDate==datetime.date(2016,9,1)`。另測 `row = m.get_metadata_row(nextPlayer=Board.BLACK, boardArea=361)` 的 `row[6:13].sum()==7` 且 `row[40:47].sum()==7`。
- [ ] **Step 2:** 跑測試，預期 FAIL。
- [ ] **Step 3:** 實作 `inverse_rank`、`make_profile_meta`、`load_humansl`、`target_move_logps`。`target_move_logps` 內部：`gs = GameState(19, rules or GameState.RULES_CHINESE)`；對每手 `(c,x,y)`：`pla = Board.BLACK if c=='B' else Board.WHITE`，`loc = Board.PASS_LOC if x is None else gs.board.loc(x,y)`；若 `c == target_color`：取 `bin, glob = gs.get_input_features(features)`、`meta_row = meta.get_metadata_row(nextPlayer=gs.board.pla, boardArea=361)`，呼叫 `model(bin_t, glob_t, input_meta=meta_t)`，取 policy logits 通道 0（與 `GameState.get_model_outputs` 的 `policy_logits[0,:]` 同一通道，透過 `model.postprocess_output` 取出），遮罩、`log_softmax`，索引 `features.loc_to_tensor_pos(loc, gs.board)`；然後 `gs.play(pla, loc)`。
- [ ] **Step 4:** 跑測試，預期 PASS（不含模型的部分）。
- [ ] **Step 5（lab01，手動驗證）：** 對 `(;SZ[19];B[pd];W[dp];B[pq])` 以 `rank_3d` 評估 B 的兩手，印出 `logp` 與 `exp(logp)`；兩個值都應在 (0.01, 0.9) 內且 `policy_logp` 以 `exp` 加總 ≈ 1（容差 1e-4）。

### Task 4: 引擎參考值

**Files:**
- Create: `scripts/katago_engine_logp.py`

**Interfaces:**
- Consumes: Task 2 `parse_moves`、`to_gtp`。
- Produces: `engine_target_logps(katago_bin, model_path, cfg_path, moves, target_color, profile: str) -> list[dict]`，格式同 Task 3 的回傳（`t`、`gtp`、`logp`、`policy_logp`）。實作：啟動一個 `katago analysis -model <bin.gz> -config <cfg> -override-config nnRandomize=false,numSearchThreads=1` 子程序，對每個目標手 `t` 送一筆 query：`{"id": f"q{t}", "moves": [[c, gtp], ...前 t 手], "initialPlayer": "B", "rules": "chinese", "komi": 7.5, "boardXSize": 19, "boardYSize": 19, "maxVisits": 1, "includePolicy": true, "overrideSettings": {"humanSLProfile": profile, "ignorePreRootHistory": false}}`；pass 的 GTP 字串為 `"pass"`。讀回 `policy`（362 浮點，非法步 −1），把 −1 轉成 `-inf`、其餘取 `log`；目標索引用引擎的 `y*19 + x`（`y` 從上到下，與 Task 2 的 `y` 同向），最後一格 361 為 pass。
- CLI：`python scripts/katago_engine_logp.py --sgf "<sgf>" --color B --profile rank_3d --out results/engine_<name>.json`。

- [ ] **Step 1:** 實作。
- [ ] **Step 2（lab01 驗證）：** 對同一個三手 SGF 跑 `rank_3d`，確認回傳兩筆、`logp` 在 (log 0.01, log 0.9) 內、`exp(policy_logp)` 合法步加總 ≈ 1。

### Task 5: 0A 一致性比較

**Files:**
- Create: `scripts/check_consistency.py`
- Create: `results/consistency_0A.json`（執行產物）

**Interfaces:**
- Consumes: Task 3 `load_humansl`、`make_profile_meta`、`target_move_logps`；Task 4 `engine_target_logps`。
- 位置集：從 `data/training/train_1D.csv` 取前 4 局（兩局 `color==B`、兩局 `color==W`），每局取目標手索引 `{10, 60, 150}` 以及第一個「上一手剛提子」的目標手（由 Python 重播偵測棋子總數減少）；profile 固定 `rank_1d`，另加一局以 `rank_5k` 重跑。
- 輸出：每個位置的 `logp_py`、`logp_engine`、`abs_diff`、整個合法步向量的 `max_abs_diff`、`argmax_py == argmax_engine`；彙總 `max_abs_diff_target`、`max_abs_diff_vector`、`n_positions`、`use_swa`。

- [ ] **Step 1:** 實作；先 `use_swa=False`。
- [ ] **Step 2（lab01）：** 執行。通過條件：`max_abs_diff_target < 1e-2` 且所有位置 `argmax` 相同。
- [ ] **Step 3:** 若未通過：依序嘗試 (a) `use_swa=True`；(b) 不遮非法步（判斷引擎是否也含非法步質量）；(c) 比較整向量的相關係數，若 < 0.5 判定座標方向錯誤，檢查 `to_gtp` 與引擎索引。每次嘗試都寫入 `results/consistency_0A.json` 的 `attempts` 清單，不覆蓋。
- [ ] **Step 4:** 把通過的設定（`use_swa`、遮罩方式、容差、最大差）寫進 `METHOD_NOTES.md` §1 新增一列。

### Task 6: 0B 吞吐量測

**Files:**
- Create: `scripts/bench_humansl.py`
- Create: `results/bench_0B.json`（執行產物）

**Interfaces:**
- Consumes: Task 3 `load_humansl`、`make_profile_meta`、`target_move_logps`；Task 2 `parse_moves`。
- 純推論：以真實局面（Task 5 的第一局重播所得）複製成 batch `B ∈ {64, 256, 1024}`，FP32 與 `torch.autocast(dtype=torch.bfloat16)` 各一；warm-up 10 次、計時 50 次、`torch.cuda.synchronize()`；輸出 `positions_per_s`。
- 端到端：`train_1D.csv` 前 200 局（不分執色），對每局呼叫 `target_move_logps`（單局面逐一前向，`rank_1d`），計 `target_moves_per_s` 與 `wall_seconds`；另記純 Python 重播＋特徵（不前向）的 `features_per_s`，以分離 CPU 瓶頸。
- 輸出 JSON 含 GPU 名稱、torch 版本、各設定數字。

- [ ] **Step 1:** 實作。
- [ ] **Step 2（lab01，單卡 `CUDA_VISIBLE_DEVICES=0`）：** 執行；確認 `positions_per_s` 隨 batch 單調非減，否則記錄並檢查是否 CPU 受限。
- [ ] **Step 3:** 把數字寫進 `METHOD_NOTES.md` §3，取代「未實測」字樣；pilot 時程用 bf16 的 `positions_per_s` 重估。

### Task 7: 玩家分組切分、假題與官方指標

**Files:**
- Create: `src/aicup/metric.py`
- Create: `scripts/make_splits.py`
- Create: `splits/player_folds.csv`、`splits/questions_fold{0..4}.csv`（執行產物，lab01 與本機各一份）
- Test: `tests/test_metric.py`

**Interfaces:**
- Produces: `LEVELS: list[str]`、`score(pred_idx: int, true_idx: int) -> float`、`mean_score(preds, trues) -> float`、`expected_score_decision(probs: np.ndarray) -> int`（`argmax_k [p_k + e^-1 (p_{k-1} + p_{k+1})]`，越界為 0）。
- Produces: `splits/player_folds.csv` 欄位 `rank, player_id, n_games, fold`；每級內玩家依 `n_games` 降冪後以 `seed=42` 打亂同分者，round-robin 分到 5 個 fold。
- Produces: `splits/questions_fold{f}.csv` 欄位 `question_id, rank, player_id, n_games, game_keys`（`game_keys` 為 `;` 分隔的 `rank:game_id`）；對 fold f 的每位玩家、每個 `n ∈ {5,10,20}`、若其盤數 ≥ n 則出最多 2 題，題內棋局互斥、題間亦互斥（順序取用打亂後的棋局清單）；`question_id = f"f{f}_{rank}_{player_id[:8]}_n{n}_{j}"`。

- [ ] **Step 1:** 寫測試：`score(3,3)==1.0`、`score(3,4)==pytest.approx(math.exp(-1))`、`score(3,5)==0.0`、`score(0,9)==0.0`；`expected_score_decision(np.array([0.30,0.30,0.40,0,0,0,0,0,0,0]))==1`（index 1 的期望 0.30+e⁻¹(0.30+0.40)=0.557 > index 2 的 0.40+e⁻¹·0.30=0.510）；`expected_score_decision(np.eye(10)[9])==9`。
- [ ] **Step 2:** 跑測試，預期 FAIL。
- [ ] **Step 3:** 實作 `metric.py`。
- [ ] **Step 4:** 跑測試，預期 PASS。
- [ ] **Step 5:** 實作 `make_splits.py`（pandas 讀 10 個 CSV 只取 `player_id, game_id, rank` 三欄）。
- [ ] **Step 6（lab01）：** 執行；驗證：`player_folds.csv` 共 8,007 列、每位玩家恰一個 fold、每級每 fold 的玩家數差 ≤ 1；每個 `questions_fold*.csv` 內無重複 `game_keys` 元素；印出每 fold 的題數分佈（按 rank × n）。
- [ ] **Step 7:** 把 fold 大小與題數寫進 `METHOD_NOTES.md` §2.1。
