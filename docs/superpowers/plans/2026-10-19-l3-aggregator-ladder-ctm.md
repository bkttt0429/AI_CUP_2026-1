# L3 聚合器梯子（含 CTM）— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用已算好的逐手 Human SL 似然（`results/scores/fold*/moves_*.npz`），在相同輸入、相同題目上依序比較聚合器梯子 (0)→(5)。要回答 H4「哪種聚合器最能分辨未見玩家的相鄰等級」，並判斷 CTM 值不值得留。

**Architecture:**
- **每局向量**：先把每局的逐手 log p 壓成固定長度的分階段證據向量（4 個手數區段 × 每 profile 的 mean／std／Q10／Q90，加區段手數與平均 surprisal），每 fold 存成一個 npz。
- **靜態梯子 (0)(1)(2)**：從這些向量算題目特徵，沿用 `rankmodel` 的 Softmax 和 MLP。
- **集合梯子 (3)(4)(5)，加兩個無交互對照**：把每局編成 128 維 token，輸出對目前正式模型 `mlp24` 的 logits 的修正量 Δ，Δ 正則化趨近零。基準 logits 用 cross-fit 產生。
- **評估**：只在 folds 1–4 做留一 fold 交叉驗證，4 個驗證 fold 合併後，以玩家為單位做成對 bootstrap。fold 0 只在最後對選定的 rung 評一次。

**Tech Stack:** Python 3.12、torch 2.14.1（集合模型 FP32 在 GPU 上；靜態模型照舊 float64）、numpy 2.5.4、pandas 3.0.6。不加新套件（沒有 sklearn、scipy）。在 nano4 上透過 `scripts/nano4/run.sbatch` 執行。

**Spec:** [METHOD_NOTES.md](../../../METHOD_NOTES.md) §2.7（梯子、輸入表示、CTM 規格、判讀規則）、§2.6（局長訊號 ablation）、§2.3（校準形式）、§2.8（現行 L3/L4 與評估協議）；[docs/ARCHITECTURE.md](../../ARCHITECTURE.md) L3 列與 §2.2「聚合器支線」。2026-10-19 使用者決定：聚合器比較改用 folds 1–4 交叉驗證，fold 0 維持只評一次。

## Global Constraints

- **評估協議**：
  - 每個 rung 在 folds 1–4 做留一 fold CV（訓練 3 fold、驗證 1 fold）。驗證題用 `pilot.pilot_questions`（巢狀 1×20／2×10／4×5）。
  - 主要指標 `balanced_mean_over_n`：題長 5／10／20 各算類別平衡分數，再取平均。
  - rung 間比較：4 個驗證 fold 合併後的成對差，CI 用 `evidence.bootstrap_ci_mean_over_sizes(s_a − s_b, …)`（以玩家為重抽樣單位；門檻判定用 `n=5000`）。
  - 另外報類別平衡的查詢級 NLL 和 Brier，只報告，不作門檻。
- **fold 0**：只評一次。`run_rung_cv` 和 `scripts/ladder_cv.py` 的 folds 若含 0 就報錯；`scripts/ladder_eval_fold0.py` 在輸出檔已存在時拒絕執行。
- **訓練題**：用 `qfeatures.sample_questions`。驗證 fold 為 v 時，`rng = np.random.RandomState(42 + v)` 只建一次，依訓練 fold 由小到大逐 fold 抽，和 `fit_rank_model.train_set` 相同。預設每人 32 題。同一個 v 的所有 rung、所有 seed 共用同一批訓練題。
- **決策與損失**：決策一律用 `metric.expected_score_decision`；損失用 `rankmodel.class_weights` 的類別平衡權重。
- **逐手證據**：`ll` 是 18 個 profile 的 log p。用相對分數 `d = ll − mean_r ll`，另外保留 `l̄ = mean_r ll`，**不做**單位長度正規化。
- **手數區段**：用棋局手數 `t+1`（`t` 是 npz 裡 0 起算的著手索引）切成 1–40／41–100／101–180／181+。每段存：
  - 每 profile 的 `d` 的 mean／std（`ddof=0`）／Q10／Q90，分位數用 `np.quantile(method="linear")`
  - `n_eff`（段內目標手數）
  - `mean_surprisal = −mean(l̄)`
  - `present`（缺段為 0，其他欄填 0）
- **token**：每局一個 128 維 token，一題 5–20 個。permutation-invariant，**不加**局索引位置編碼。明確輸入 `log m`、`log n_target` 和各段總手數。
- **標準化**：一律用 `rankmodel._standardise`（std < 1e-12 時設為 1）。標準化之後，把 padding 位置重新設為 0。
- **CTM（rung 5）**：
  - 規格：latent 256、neuron history 8、NLM hidden 16、4 heads、固定 8 ticks、action 與 output 各 512 個 sync pairs。
  - 輸出：對基準 logits 的修正 Δ，損失 `CE(z^T) + γ‖Δ^T‖²`，γ = 1e-2。
  - 對照：原始 CTM 的 min-loss／max-certainty 雙 tick 損失（rung `r5_ctm_minmax`）。
- **禁止**：
  - 把各 tick 輸出相乘，或對實際落子取 `max_τ` 當似然。
  - 在驗證集上挑 tick。主要分數一律取最後一個 tick。
  - 訓練 8 ticks、推論 16 ticks 的結果另列為「超出訓練長度的泛化」，不參與選擇。
- **訓練長度**：固定 epoch，不做 early stopping。驗證 fold 只用來報分。
- **順序與門檻**：(0)(1)(2) → (3) 加對照 → 只有 `G_set` 成立時才跑 (4)(5)。不成立時要加 `--force`。
- **門檻定義**：每個門檻的每組比較都要同時滿足以下四項；比較對象都事先指定，不取 max。
  1. seed 平均後的成對 CI 下界 > 0
  2. 3 個 seed 各自的點差都 > 0
  3. 4 個驗證 fold 中至少 3 個點差 > 0
  4. 點差 ≥ 0.005
- **門檻報告**：摘要另附 Nadeau–Bengio 校正後的 CI（變異數 × 2.33）當保守參考，不作門檻。
- **nano4**：
  - 每個工作都經 `run.sbatch`，並加 `--exclude=25a-hgpn[001,003,004,007,062]`。
  - 輸出全部寫到 `results/ladder/`，**不得**寫到 `results/scores/`。
  - 新腳本都要有 `score_folds.py` 的 ROOT／`sys.path` 區塊。共用函式放 `src/aicup/`，不得 import `scripts/`（`python -I` 不會把 scripts 加進 path）。
- **git**：`results/ladder/vectors/` 和 `results/ladder/cv/*.npz` 不進 git；JSON 摘要進 git。

## Review Focus

1. **極短棋局或缺段**：例如某段沒有目標落子。向量必須全部是有限值、缺段 present=0（Task 1 `test_empty_segments_are_zero_and_flagged`）。
2. **padding 與常數欄**：題目不滿 20 局時，padding 位置不論填什麼都不影響輸出。實際資料有常數欄，例如 `s1_present` 恆為 1，`counts=False` 時整欄是 0；標準化之後也不能出現 NaN（Task 4 `test_padding_invariance`、`test_constant_columns_standardise_finite`）。
3. **局的順序**：同一題的局打亂後，預測必須相同（Task 3 `test_static_features_order_free`；Tasks 4–6 `test_permutation_invariance`）。
4. **資料不一致**：向量涵蓋的局 ≠ `fold_players` 選局，或 npz 加總 ≠ CSV `sum_*`，都必須直接報錯（Task 1 `test_fold_vectors_checks_csv_sums`、建檔腳本的選局比對）。整條管線用精確重現檢查驗證（Task 7 `test_reproduces_rank_model_eval_per_fold`）。
5. **基準洩漏與 fold 0 誤用**：Δ 模型的訓練題必須用 cross-fit 基準；驗證題的基準必須逐位元等於 `r2_mlp24` 靜態 rung 的 proba（Task 4 `test_crossfit_never_sees_own_fold`、Task 7 斷言）。folds 含 0 時拒絕執行；fold 0 評過就不能重跑（Task 7 `test_cv_refuses_fold0`、Task 8 `test_refuses_second_fold0_eval`）。

---

### Task 1: 每局分階段證據向量

**Files:** Create `src/aicup/gamevec.py`, `scripts/ladder_build_vectors.py`；Modify `.gitignore`（加 `results/ladder/vectors/`、`results/ladder/cv/*.npz`）；Test `tests/test_gamevec.py`

**Interfaces — Produces:**
- **常數**：`SEGMENTS = ((1, 40), (41, 100), (101, 180), (181, None))`；`WHOLE = ((1, None),)`；`SEG_DIM = 75`。每段依序為 d_mean 18、d_std 18、d_q10 18、d_q90 18、n_eff、mean_surprisal、present。
- `game_vector(logp: ndarray[n, 18], t: ndarray[n], segments=SEGMENTS) -> ndarray[float32, 75 * len(segments)]`
- `vec_names(segments) -> list[str]`：例如 `s1_mean_12k`、`s4_present`；WHOLE 的前綴是 `w_`。
- `fold_vectors(fold_dir: Path) -> dict`：
  - 鍵：`game_key: ndarray[<U*, G]`（固定寬度 unicode，不是 object）、`staged: float32[G, 300]`、`whole: float32[G, 75]`、`sums: float64[G, 18]`、`n_moves: int64[G]`。
  - 讀每個完成的分段（`scoring.completed_parts` + `part_paths`），以 `game_row` 分組。
  - 先用逐手 `ll` 的 float64 加總比對 CSV 的 `sum_*`（每局每 profile |Δ| ≤ 1e-2），不一致就拋 `ValueError`；通過後 `sums` 存 **CSV 的值**，讓 q24 和 `build_matrix` 逐位元相同。
- **CLI**：`scripts/ladder_build_vectors.py --folds 0 1 2 3 4 [--scores results/scores] [--out results/ladder/vectors]`，輸出 `fold{F}.npz`（`np.savez_compressed`，鍵同上）。
  - 寫檔前比對 `game_key` 集合與 `scoring.fold_players(player_games(), player_folds, F)` 的選局，不相等就報錯，訊息格式照 `fit_rank_model.load_fold`。
  - 每個 fold 印一行 `fold F: G games, staged 300 / whole 75`。
- **讀取**：讀向量一律寫成 `dict(np.load(path))`（一次全讀進記憶體），不得在迴圈裡反覆存取 NpzFile。

- [ ] **Step 1: 寫失敗的測試** `tests/test_gamevec.py`
  - `test_segment_boundaries`：t = [0, 39, 40, 99, 100, 179, 180, 250] 依序落在段 1,1,2,2,3,3,4,4，各段 `n_eff` 都是 2。
  - `test_relative_scores_and_surprisal`：ll 每列加常數 c 時，d 的統計不變，`mean_surprisal` 減 c。每段 `d_mean` 的 18 維總和為 0（容差 1e-5）。
  - `test_empty_segments_are_zero_and_flagged`：只有 t < 40 的 5 手時，段 2–4 的 75 維全部是 0，s1_present = 1，全部有限值。只有一手的段 std = 0。
  - `test_whole_matches_single_segment`：`WHOLE` 的結果等於手動用 np 對全部手數計算的值。
  - `test_fold_vectors_checks_csv_sums`：在 tmp 目錄造 1 個分段（2 局，CSV + npz，格式同 `score_folds.py`）。`sums` 等於 CSV 的值；CSV 某格 +1 後要拋 `ValueError`。
  - `test_vectors_roundtrip_without_pickle`：`np.savez_compressed` 後用 `np.load(..., allow_pickle=False)` 讀回，`game_key` 相等。
- [ ] **Step 2:** `pytest tests/test_gamevec.py -v` → FAIL（module 不存在）
- [ ] **Step 3:** 實作 `gamevec.py` 和建檔腳本
- [ ] **Step 4:** `pytest tests/test_gamevec.py -v` → PASS；`pytest -q` 全過
- [ ] **Step 5:** Commit `feat: per-game staged evidence vectors for the aggregator ladder`

### Task 2: 題目評估與成對比較

**Files:** Create `src/aicup/qeval.py`；Modify `scripts/fit_rank_model.py`（刪掉 `evaluate`，改成 `from aicup.qeval import evaluate`）；Test `tests/test_qeval.py`

**Interfaces — Produces:**
- `SIZES = (5, 10, 20)`
- `evaluate(proba, y, qs, sizes=SIZES) -> (dict, scores ndarray)`：從 `fit_rank_model.evaluate` 搬過來，原有的鍵和值都不變。另外加兩個鍵：
  - `nll_balanced`：各類別的平均 −log p(真實類別)，再對類別取平均。
  - `brier`：10 類的 Brier 分數平均。
- `balanced_mean_over_n(scores, y, n_games, sizes=SIZES) -> float`
- `paired_diff(s_a, s_b, y, player_id, n_games, sizes=SIZES, n=5000, seed=42) -> {"diff": float, "ci95": [lo, hi], "by_n": {5: d5, 10: d10, 20: d20}}`
  - `diff` 是兩邊 `balanced_mean_over_n` 的差。
  - CI 用 `bootstrap_ci_mean_over_sizes(s_a − s_b, player_id, y, n_games, n=n, seed=seed)`。這個統計量對分數是線性的，所以差值的 CI 就是成對 CI。

- [ ] **Step 1: 先寫 golden 測試，再搬函式**：`test_evaluate_golden_against_fit_rank_model`。固定 seed 產生隨機的 proba／y／qs，對舊的 `fit_rank_model.evaluate` 存下輸出當作 fixture（存成 tests 內的常數或 `tests/data/qeval_golden.json`）。新的 `qeval.evaluate` 在原有的每個鍵上都要和 fixture 相同。
- [ ] **Step 2: 其他測試**：
  - `test_paired_identical_is_zero`：相同分數 → diff 0、CI (0, 0)。
  - `test_paired_constant_offset`：s_b = s_a − 0.05 → diff 0.05、CI (0.05, 0.05)，容差 1e-12。
  - `test_paired_matches_headline_difference`：diff 等於兩次 `evaluate` 的 `balanced_mean_over_n` 之差。
  - `test_nll_brier_values`：手造 2 題，NLL 和 Brier 與手算一致。
- [ ] **Step 3:** `pytest tests/test_qeval.py -v` → 除 golden 外 FAIL
- [ ] **Step 4:** 實作，並修改 `fit_rank_model.py` 的 import
- [ ] **Step 5:** `pytest tests/test_qeval.py -v` → PASS；`pytest -q` 全過
- [ ] **Step 6:** Commit `refactor: question evaluation, NLL/Brier and paired comparison in aicup.qeval`

### Task 3: 題目打包與靜態梯子 (0)(1)(2)、局長 baseline

**Files:** Create `src/aicup/ladder_data.py`；Test `tests/test_ladder_data.py`

**Interfaces — Consumes:** Task 1 的向量 dict；`qfeatures.question_features`、`FEATURE_NAMES`
**Interfaces — Produces:**
- `QGLOB_NAMES = ["log_games", "log_target_moves", "log1p_seg1", "log1p_seg2", "log1p_seg3", "log1p_seg4"]`
- `pack_questions(questions: DataFrame, vec: dict, rep: str = "staged", counts: bool = True, max_games: int = 20) -> dict`，回傳的鍵：
  - `tokens: float32[Q, 20, D+1]`：D = 300（staged）或 75（whole），最後一欄是 `log1p(n_moves_g)`；padding 位置填 0。
  - `mask: bool[Q, 20]`：True 表示有局。
  - `qglob: float32[Q, 6]`。
  - `y: int64[Q]`：沒有 rank 時為 −1。
  - `idx: list[ndarray]`：每題在 vec 裡的列號。
  - `player_id`、`n_games`、`question_id`：都是 ndarray。
  - `counts=False` 時，把 token 的 `n_eff`、`present`、`log1p(n_moves_g)` 欄和 `qglob[:, 1:]` 都設為 0。
- `static_features(packed: dict, vec: dict, kind: str, counts: bool = True, clip_k: float | None = None) -> ndarray[Q, F]`，全部用 float64 計算。記號：`L = Σ_g sums_g`（18）、`n = Σ_g n_moves_g`、`c_g = sums_g / n_g`、`r_g = c_g − mean_r c_g`。各 kind 的定義：
  - `"sum"`（§2.3 校準形式的線性近似）：`[(L − mean_r L)/√n（18）, (L − mean_r L)/n（18）, log m, log n]`
  - `"clip"`：`m = 逐 profile 的 median_g r_g`；特徵為 `[m + mean_g clip(r_g − m, −k, k)（18）, log m, log n]`。`k = clip_k`，由 `fit_clip_k(...)` 只在訓練題上估出，`None` 時報錯。
  - `"trim"`：`[逐 profile 對 r_g 取 20% trimmed mean（每側去掉 floor(0.2 m) 局）（18）, log m, log n]`
  - `"q24"`：`question_features(sums[idx], n_moves[idx])`（與 `build_matrix` 相同）
  - `"pooled"`：`[token 在有效局上的平均（D+1）, qglob（6）, q24（24）]`
  - `"length"`（§2.6 只用局長統計）：`[log m, log n, 每局 n_moves 的 mean 和 std, 各段 present 的題內比例（4）, 各段 n_eff 的題內平均（4）]`
  - `counts=False` 時：`sum`／`clip`／`trim` 去掉 `log n`；`q24` 去掉 `log_moves` 欄（`FEATURE_NAMES.index("log_moves")`）；`pooled` 跟著 token 和 q24 的規則；`length` 不允許。
- `fit_clip_k(packed, vec) -> float`：訓練題所有 `|r_g − m|` 的 95 分位數。
- `STATIC_RUNGS`，每個值是 `(kind, rep, make_model)`，`make_model` 是零參數的 lambda：
  - `"r0_sum": ("sum", "staged", lambda: SoftmaxModel(l2=1e-3))`
  - `"r1_clip": ("clip", "staged", lambda: SoftmaxModel(l2=1e-3))`
  - `"r1_trim": ("trim", "staged", lambda: SoftmaxModel(l2=1e-3))`
  - `"r2_mlp24": ("q24", "staged", lambda: MLPModel(hidden=32))`：就是目前的正式模型
  - `"r2_staged": ("pooled", "staged", lambda: MLPModel(hidden=64))`
  - `"r2_whole": ("pooled", "whole", lambda: MLPModel(hidden=64))`
  - `"r2_length_only": ("length", "staged", lambda: MLPModel(hidden=32))`
- rung (1) 只做截斷和 trimmed：規格把三種做法列成擇一，這兩種已經針對「一局撐起全部證據」，Student-t 不做。

- [ ] **Step 1:** 測試
  - `test_pack_layout_and_padding`：5 局的題目 mask 前 5 個 True、後 15 個 False，padding 全 0。
  - `test_counts_off_zeroes_length_columns`：包含 q24 少了 `log_moves`、sum 少了 `log n`。
  - `test_static_features_order_free`：打亂 `game_keys` 後，每種 kind 的特徵都不變（容差 1e-9）。
  - `test_clip_equals_consensus_without_outliers`：題內每局的 c 都相同時，clip 和 trim 的前 18 維都等於 r。
  - `test_clip_bounds_outlier`：一局離群時，該局的貢獻被限制在 ±k 以內。
  - `test_trim_rule`：m = 5 時每側剛好去掉 1 局。
  - `test_q24_matches_build_matrix`：與 `qfeatures.build_matrix` 逐位元相等。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS（`pytest -q` 全過）
- [ ] **Step 5:** Commit `feat: question packing, static ladder rungs (0)(1)(2) and the length-only baseline`

### Task 4: 集合模型 (3)、無交互對照與訓練框架（含 cross-fit 基準）

**Files:** Create `src/aicup/aggregators.py`, `src/aicup/settrain.py`；Test `tests/test_aggregators.py`, `tests/test_settrain.py`

**Interfaces — Consumes:** Task 3 `pack_questions`、`static_features`；`rankmodel.class_weights`、`rankmodel._standardise`
**Interfaces — Produces:**
- **共用介面**：每個集合模型的建構子第一個參數都是 `d_in`（= token 最後一維）。`forward(tokens [B, M, d_in], mask [B, M] bool, qglob [B, 6], ticks: int | None = None) -> deltas [B, T, 10]`。輸出層的權重和 bias 初始化為 0，所以起始 Δ ≡ 0。
- `GameEncoder(d_in, d_tok=128)`：Linear(d_in, 128) → ReLU → Linear(128, 128) → LayerNorm。
- `PooledDelta(d_in, d_glob=6, hidden=256)`：無交互對照 `r2_pooled_delta`。不用 encoder，直接對標準化後的 token 在有效局上取平均，接 qglob，再過 MLP(→256→10)，T = 1。
- `QglobDelta(d_in, d_glob=6, hidden=64)`：`r3_null`，只吃 qglob，T = 1。用來量「疊在基準上的再校準」本身的收益。
- `DeepSets(d_in, d_glob=6, d_tok=128, hidden=256)`：GameEncoder → 有效 token 的 mean 和 max pooling → 接 qglob → MLP(→256→256→10)，T = 1。
- `SetTransformer(d_in, d_glob=6, d_tok=128, heads=4, layers=2)`：GameEncoder → 2 層 SAB（`nn.MultiheadAttention`，`key_padding_mask=~mask`）→ PMA（1 個 seed）→ 接 qglob → MLP，T = 1。
- `n_params(model) -> int`
- `TrainCfg(epochs=20, batch=512, lr=1e-3, weight_decay=1e-4, gamma=1e-2, seed=0, loss="final", device="cuda")`：AdamW 加 cosine schedule，每個 epoch 依 seed 洗牌。
- `fit_set_model(make: Callable[[int], nn.Module], data: dict, base: ndarray[Q, 10], cfg: TrainCfg) -> Fitted`：
  - 呼叫 `make(d_in)` 建模型。
  - 只用訓練題的有效 token 估標準化參數（`_standardise`），qglob 也一樣；標準化後把 padding 設回 0。
  - `loss="final"`：`CE(base + Δ_T, y; class_weights)` 加 `γ · mean‖Δ_T‖²`。
- `predict_set_model(fitted, data, base, ticks=None, pick="final") -> proba [Q, 10]`：`softmax(base + Δ_T)`。
- `crossfit_base_logits(make_static, X_by_fold: dict[int, ndarray], y_by_fold: dict[int, ndarray], X_eval: ndarray) -> (dict[int, ndarray], ndarray)`：
  - 每個訓練 fold f 的 base，用其他訓練 fold 擬合。
  - 驗證題的 base，用全部訓練 fold 擬合；這個值等於靜態 rung `r2_mlp24` 的輸出。
  - 回傳 `log(clip(proba, 1e-9))`。

- [ ] **Step 1:** 測試（CPU、小尺寸；以 `@pytest.mark.parametrize` 涵蓋 PooledDelta、DeepSets、SetTransformer，Task 5、6 再把新模型加進參數清單）
  - `test_permutation_invariance`：打亂有效 token 後輸出相同（容差 1e-5）。
  - `test_padding_invariance`：padding 位置填亂數不影響輸出。
  - `test_zero_init_delta`：未訓練時 `predict_set_model` 等於 `softmax(base)`。
  - `test_constant_columns_standardise_finite`：token 含常數欄，且 `counts=False` 時，訓練一個 epoch 後 loss 和 proba 都是有限值。
  - `test_learns_set_interaction`：玩具資料，標籤 = 某一欄在題內的 max 是否 > 0.5。DeepSets 訓練 30 個 epoch 後訓練準確率 > 0.9；同 seed 兩次結果相同。
  - `test_crossfit_never_sees_own_fold`：假模型記錄 fit 時看到的列，fold f 的 base 從沒看過 fold f。
  - `test_proba_valid`：每列加總為 1，沒有 NaN。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS
- [ ] **Step 5:** Commit `feat: set aggregators (3), no-interaction Δ controls and cross-fitted base logits`

### Task 5: Rung (4) cross-attention + GRU

**Files:** Modify `src/aicup/aggregators.py`；Test `tests/test_aggregators.py`

**Interfaces — Produces:** `XAttnGRU(d_in, d_glob=6, d_tok=128, d_hidden=256, heads=4, ticks=8)`：
- GameEncoder 產生 token；可學的初始狀態 `h0 [256]`。
- 每個 tick：`q = W_q h`（→ 128），對 token 做 `MultiheadAttention(128, 4)`（加 key_padding_mask）得到 `o`；`h = GRUCell([o, qglob], h)`；`Δ_t = W_out h`（W_out 零初始化）。
- 回傳全部 ticks 的 `[B, T, 10]`；`ticks` 參數可以覆寫 T。

- [ ] **Step 1:** 測試 `test_xattn_shapes_and_tick_override`：ticks = 8 時輸出 [B, 8, 10]，`ticks=16` 時輸出 [B, 16, 10]。permutation／padding invariance 的參數清單加入 `XAttnGRU`。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS
- [ ] **Step 5:** Commit `feat: cross-attention + GRU aggregator (4), fixed 8 ticks`

### Task 6: Rung (5) CTM 與雙 tick 損失

**Files:** Modify `src/aicup/aggregators.py`, `src/aicup/settrain.py`；Test `tests/test_aggregators.py`, `tests/test_settrain.py`

**Interfaces — Produces:**
- `CTM(d_in, d_glob=6, d_tok=128, d_latent=256, history=8, nlm_hidden=16, heads=4, ticks=8, n_sync_action=512, n_sync_out=512, pair_seed=0)`：
  - **sync pairs**：從 256·257/2 = 32,896 個 i ≤ j 組合中**不放回**抽 512 個（`torch.randperm(…, generator=torch.Generator().manual_seed(pair_seed))` 取前 512 個），action 和 output 各一組，存成 buffer。
  - **decay**：沿用原版。每對一個參數 `r_p`，初始為 0（不衰減），每次 forward 都 clamp 到 [0, 15]，權重是 `exp(−r_p (τ_now − τ))`。
  - **可學初始值**：`A0 [256, 8]`、`z0 [256]`。
  - **synapse**：`Linear(256 + 128 + 6, 512) → GLU → LayerNorm(256)`。
  - **NLM**：每個 neuron 自己的兩層 MLP（8 → 16 → 1，ReLU），用 einsum 一次算完 256 個 neuron。
  - `W_out` 零初始化。
- `ctm_minmax_loss(logits_ticks [B, T, 10], deltas [B, T, 10], y, w, gamma) -> scalar`：
  - 每個樣本挑兩個 tick：CE 最小的 `t1`，以及 certainty = 1 − H/log 10 最大的 `t2`。平手時取較後面的 tick。
  - 損失 = `(CE_t1 + CE_t2)/2 + γ(‖Δ_t1‖² + ‖Δ_t2‖²)/2`，每個樣本乘權重 w。
- `TrainCfg.loss` 接受 `"ctm_minmax"`。
- `predict_set_model(..., pick="certain")`：每題取自己 certainty 最大的 tick（逐題、不看標籤）。只用於報告欄位 `r5_ctm_minmax@certain`，不參與選擇。

遞迴（每個 tick；`tokens` 是 GameEncoder 輸出，`kpm = ~mask`）：

```python
A = A0.expand(B, -1, -1).clone(); z = z0.expand(B, -1)
Z_act, Z_out = [z], []                                        # action 的歷史從 z0 開始；output 的不含 z0
for _ in range(T):
    s_act = sync(Z_act, idx_act, r_act)                       # [B, 512]
    o, _ = attn(W_q(s_act)[:, None], tokens, tokens, key_padding_mask=kpm)   # [B, 1, 128]
    a = synapse(torch.cat([z, o[:, 0], qglob], -1))           # [B, 256]
    A = torch.cat([A[:, :, 1:], a[:, :, None]], -1)           # 只留最近 8 個 pre-activation
    z = nlm(A)                                                # [B, 256]
    Z_act.append(z); Z_out.append(z)
    deltas.append(W_out(sync(Z_out, idx_out, r_out)))         # [B, 10]
# sync(Z, idx, r)：Zs = stack(Z, 1) [B, τ, 256]；w_τ = exp(−clamp(r,0,15) · (τ_now − τ))；
# S_p = Σ_τ w_τ z_i^τ z_j^τ / sqrt(Σ_τ w_τ)
```

- [ ] **Step 1:** 測試
  - `test_ctm_shapes_history_and_pairs`：輸出 [B, 8, 10]；`idx_act`、`idx_out` 各 512 對，互不重複，且全部 i ≤ j；同一個 `pair_seed` 兩次建構結果相同；`ticks=16` 可以執行。
  - `test_ctm_action_sync_includes_z0`：ticks = 2，r = 0 時，第 2 個 tick 的 action sync 等於用 [z0, z1] 手算的值。
  - `test_ctm_norms_bounded_at_16_ticks`：隨機權重、ticks = 16 時，z 和 sync 的範數都是有限值，且小於 1e3。
  - permutation／padding invariance 的參數清單加入 `CTM`。
  - `test_minmax_loss_picks_ticks`：手造 logits，t1、t2 已知（包含平手的情況），損失值和手算一致。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS
- [ ] **Step 5: 成本對齊檢查**：`python -I -c` 印出預設 `XAttnGRU` 和 `CTM` 的 `n_params`。預估約 46 萬對 57 萬，比值約 1.25。比值不在 [0.75, 1.33] 時，只調 `XAttnGRU.d_hidden`，最終值寫進 Task 7 的 `SET_RUNGS`。
- [ ] **Step 6:** Commit `feat: CTM aggregator (5) with sync-pair readout and the original min-loss/max-certainty loss as control`

### Task 7: CV 執行、摘要與門檻

**Files:** Create `src/aicup/ladder.py`, `scripts/ladder_cv.py`；Test `tests/test_ladder.py`

**Interfaces — Consumes:** Tasks 1–6
**Interfaces — Produces:**
- `SET_RUNGS`，每個值是 `(rep, make(d_in), TrainCfg 覆寫)`：
  - `r2_pooled_delta`（PooledDelta）、`r3_null`（QglobDelta）
  - `r3_deepsets`、`r3_settf`、`r3_deepsets_whole`（rep = whole）
  - `r4_xattn`、`r5_ctm`、`r5_ctm_minmax`（loss = ctm_minmax）
- `RUNGS = STATIC_RUNGS | SET_RUNGS`
- `run_rung_cv(rung, vectors_dir, folds=(1, 2, 3, 4), seed=0, per_player=32, counts=True, device="cuda", base_cache=None) -> DataFrame`：
  - 回傳欄位：`question_id, fold, player_id, y, n_games, p0..p9, score`。
  - folds 含 0 時拋 `ValueError`。
  - 集合 rung 的 base 用 `crossfit_base_logits(STATIC_RUNGS["r2_mlp24"][2], …)`，輸入是 `static_features(..., "q24", counts=counts)`。每個 (v, counts) 只算一次，快取到 `results/ladder/cv/base_v{v}[_nocounts].npz`。
  - 斷言：驗證題 base 的 softmax 等於同一個 v 的 `r2_mlp24` 靜態 proba（容差 1e-12）。
  - r1_clip 的 `clip_k` 用 `fit_clip_k` 只在訓練題上估，記錄到摘要裡。
  - rung 4、5 額外輸出 `@T16`：同一個模型用 `ticks=16` 推論。`r5_ctm_minmax` 額外輸出 `@certain`。
- `summarize(cv_dir) -> dict`：
  - 每個 rung：各驗證 fold 的 `balanced_mean_over_n`、合併後的值、`per_class`、`nll_balanced`、`brier`。
  - 集合 rung：同一題的分數先在 seed 間平均，再算指標；另外報 seed 間的 SD。
  - 成對比較：每個 rung 都對 `r2_mlp24`；門檻用的各組比較；`r5_ctm_minmax@certain` 對 `r5_ctm`；`r2_mlp24@pp8` 對 `r2_mlp24`（量化每人 8 題改 32 題的效應）。
  - 每組比較都附 4 個 fold 各自的點差、seed 間的點差，以及 Nadeau–Bengio 校正後的 CI。
  - 診斷：base 在訓練題（cross-fit）和驗證題上的加權 CE。
- `gates(summary) -> dict`：每組比較都要滿足 Global Constraints 的四項條件，比較對象事先指定。
  - `G_rep`：`r2_staged > r2_whole`，且 `r3_deepsets > r3_deepsets_whole`。
  - `G_set`：`r3_settf > r3_deepsets`，且 `r3_settf > r2_pooled_delta`。前者量局與局的交互，後者是同框架的無交互對照。
  - `G_iter`：`r4_xattn > r3_settf`，且 `r5_ctm > r3_settf`。
  - `G_sync`：`r5_ctm > r4_xattn`。
  - 報告但不作門檻：`r3_deepsets` 對 `r2_pooled_delta`（每局非線性編碼的效應）、`r2_staged` 對 `r2_pooled_delta`（訓練框架的效應）、`r3_null` 對 `r2_mlp24`（再校準的效應）。
- **CLI**：`scripts/ladder_cv.py --rungs ... [--seeds 0 1 2] [--folds 1 2 3 4] [--per-player 32] [--vectors results/ladder/vectors] [--out results/ladder/cv] [--no-counts] [--force] [--summarize-only]`
  - 每個 (rung, seed) 寫 `out/{rung}[_pp{N}][_nocounts]_s{seed}.npz`，字串欄位用固定寬度 unicode；已經存在就跳過，可以續跑。
  - 結束時寫 `results/ladder/cv_summary.json`。
  - 要求 r4／r5 時，若摘要裡沒有 r3 和對照的結果，或 `G_set` 不成立，就印出原因並 exit 2；加 `--force` 才繼續。
  - 靜態 rung 只用 seed 0（它們是確定性的）；集合 rung 預設跑 seeds 0、1、2。
  - `--folds` 含 0 時 exit 2。

- [ ] **Step 1:** 測試
  - `test_reproduces_rank_model_eval_per_fold`：真實資料，向量檔不存在時 skip。用 `run_rung_cv("r2_mlp24", per_player=8)` 重跑，4 個 fold 的 `balanced_mean_over_n` 必須等於 `results/rank_model_eval.json` 的 `cv.mlp_h32.per_fold`（0.6131／0.6100／0.6312／0.6258，以 JSON 原值比對，容差 1e-9）。
  - `test_cv_static_and_deepsets_run_end_to_end`：合成的小向量、2 個 fold、CPU。每題都有分數、proba 有效；同一個驗證 fold 下，不同 rung 的 `question_id` 完全相同。
  - `test_cv_refuses_fold0`。
  - `test_summarize_pools_folds_and_seed_mean`：手造 npz，合併指標、seed 平均和逐 fold 點差都和手算一致。
  - `test_gates_logic`：手造摘要，涵蓋四項條件各自不成立的情況。
  - `test_force_required_for_r4_without_g_set`：exit code 2。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS；`pytest -q` 全過
- [ ] **Step 5:** Commit `feat: aggregator ladder CV runner, pooled paired summary and gating rules`

### Task 8: fold 0 一次評估與文件

**Files:** Create `scripts/ladder_eval_fold0.py`；Modify `METHOD_NOTES.md` §2.7、`docs/ARCHITECTURE.md` §3（只改聚合器那一列）、`docs/NANO4.md` §4；Test `tests/test_ladder.py`

**Interfaces — Produces:** CLI `scripts/ladder_eval_fold0.py --rung NAME [--seeds 0 1 2] [--out results/ladder/fold0_eval.json]`：
- 用 folds 1–4 重訓（集合 rung 的 base 用 4 fold cross-fit），對 fold 0 的 `pilot_questions` 評一次。集合 rung 的 3 個 seed 各訓練一次，每題分數取 seed 平均，和 CV 用同一個統計量。
- 輸出 headline 和 `bootstrap_ci_mean_over_sizes` 的 CI，加上兩組 `paired_diff`：
  - 對同一次重訓的 4 fold 基準（Δ ≡ 0）：量 Δ 本身的效應。
  - 對目前的正式模型（`results/models/rank_model.json`，經 `rankmodel.from_state` 加 `build_matrix`）：附註正式模型是每人 8 題訓練的，參考 CV 的 `r2_mlp24@pp8` 比較。
- JSON 記錄 `rung`、`seeds`、`chosen_from` 和 `cv_summary.json` 的 sha256。
- `--out` 已存在時拒絕執行（exit 2）。

- [ ] **Step 1:** 測試
  - `test_refuses_second_fold0_eval`。
  - `test_fold0_paired_alignment`：兩個模型的題號完全對齊才算差值，不對齊時拋錯。
- [ ] **Step 2:** FAIL → **Step 3:** 實作 → **Step 4:** PASS
- [ ] **Step 5: 文件**：
  - `METHOD_NOTES.md` §2.7 加一行：「2026-10-19 起：聚合器比較用 folds 1–4 CV 的合併成對差值與穩定性門檻；fold 0 只評一次（`scripts/ladder_eval_fold0.py`）」。
  - `docs/ARCHITECTURE.md` §3：只在聚合器支線那一列加註「以 folds 1–4 CV 成對檢定，fold 0 只評一次」，其他列的通則不動。
  - `docs/NANO4.md` §4：加 ladder 的 sbatch 指令（見 Task 9）。
- [ ] **Step 6:** Commit `feat: one-shot fold-0 evaluation for the chosen ladder rung; docs: CV-based aggregator protocol`

### Task 9: 在 nano4 執行

每個指令都在 `~/BK_AICUP_2026-1` 下送出，並加 `--exclude=25a-hgpn[001,003,004,007,062]`（以下用 `$EX` 表示）。

- [ ] **Step 1: 建向量**：`sbatch $EX -t 01:00:00 -J ladvec scripts/nano4/run.sbatch python -I scripts/ladder_build_vectors.py --folds 0 1 2 3 4`。
  - 預期 5 行 `fold F: … games`，總計 147,780 局。
  - 接著在登入節點跑 `pytest tests/test_ladder.py::test_reproduces_rank_model_eval_per_fold -v` → PASS。
- [ ] **Step 2: 跑靜態梯子、(3) 和對照**：`sbatch $EX -p 8gpus -t 12:00:00 -J ladder3 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r0_sum r1_clip r1_trim r2_mlp24 r2_staged r2_whole r2_length_only r2_pooled_delta r3_null r3_deepsets r3_deepsets_whole r3_settf`
  - 再跑一次 `--rungs r2_mlp24 --per-player 8`，這是 `@pp8` 的參考。
- [ ] **Step 3: 看門檻**：印出 `gates` 和各組比較。`G_set` 成立就跑 Step 4；不成立就停下，把摘要交給使用者決定要不要 `--force`。
- [ ] **Step 4: 跑 (4)(5)**：`sbatch $EX -p 8gpus -t 24:00:00 -J ladder45 scripts/nano4/run.sbatch python -I scripts/ladder_cv.py --rungs r4_xattn r5_ctm r5_ctm_minmax`
- [ ] **Step 5: 局長 ablation**：
  - 對 CV 最好的一階和 `r2_mlp24`，加 `--no-counts` 各跑一次。
  - staged 和 whole 都要跑：whole 沒有缺段模式，staged 的缺段全零仍然帶有手數區間。
  - 和 `r2_length_only` 一起列出「拿掉局長」和「只有局長」兩端。
- [ ] **Step 6: 交給使用者**：依 §2.7 的判讀規則寫摘要：
  - `G_rep` 成立 → 貢獻在輸入表示。
  - `G_set` 成立 → 局與局的交互有價值。
  - `G_iter` 成立 → 反覆讀取有價值。
  - `G_sync` 成立 → 才深究 CTM。
  - **由使用者選定 rung 後**，才執行 `scripts/ladder_eval_fold0.py --rung <選定>`。
  - 結果寫進 `METHOD_NOTES.md` §2.7，commit `data: aggregator ladder CV and the one fold-0 evaluation`。
