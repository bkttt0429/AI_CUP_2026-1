#!/usr/bin/env python
"""Official test CSVs -> submission_rank.csv using the Human SL features and the saved L3/L4 model.

For every question: drop unusable games (wrong size, illegal replay, non-alternating colours, no
target moves), score the rest under the 18 rank profiles, build the question features, predict with
the saved model and the expected-score decision. Public questions are written first, then private,
and the file is validated before the script exits. Also writes a JSON report (games used / dropped
and why, passes, shard files, prediction counts, problems, exit code) and the per-question class
probabilities.

Scoring is sharded over GPUs (2026-10-25, item 1 of the to-do list). The task list -- every usable
game in input order -- is built identically by every process from the inputs alone; shard i of N
scores the tasks i, i+N, ... and writes <out stem>_shard{i}of{N}.npz next to --out: per game its
n_moves and 18 log p sums, per move the log p under the 18 profiles with its task and move index (the
input of later per-game vectors), and a JSON meta with a hash of the questions. The merge reads the N
files, checks that they were made from the same questions and sharding and that every task was scored
exactly once, then predicts, writes and validates. The single-GPU run is shard 0 of 1 followed by the
merge in the same process, so it writes the same files.

  python -I scripts/predict_submission.py ...                                  # one GPU
  bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py ...   # 8 GPUs: adds --shard i --nshards 8
  python -I scripts/predict_submission.py --merge --nshards 8 ...              # then, on any node, no GPU

With --set-models (item 2) the decision comes from set models instead of the saved L3/L4 model: each
.pt + .json (aicup.setmodel_io) is a Δ-correction trained by experiments/train_set_model.py on top of the
paired base --set-base (a rank_model.json fitted on the same training questions; a sidecar's training
question hash must equal the base's, exit 2 otherwise). The merge reads the
shards' per-move log p, builds every game's staged and whole vectors (aicup.gamevec) and the packed
question sets (aicup.ladder_data.pack_questions), takes the base logits log(clip(base probabilities)),
runs every model and averages the probabilities first over the seeds of a rung, then over the rungs
(uniform), before the expected-score decision -- as the CV ensemble ens_top3 does. Three r3_pma seeds are
r3_pma; adding r4_xattn_t1 and r5_ctm makes ens_top3. A question whose set-model probabilities are not
finite falls back to the base's probabilities and is listed in the report (set_model_fallback_questions);
a question with no usable game still gets FALLBACK. Single-GPU and sharded runs give the same decisions,
not bit-identical vectors (TF32 and batch composition move log p by up to ~4e-3 nats).

With --l1b-heads (issue #9) the shard step also scores every position under each L1b heads file's 10 rank
heads (aicup.l1b; a fine-tuned trunk named in the sidecar is restored into its own model instance), one
trunk forward per model next to the 18 profile passes, and the shard file gains l1b_sums [n, K, 10] and
l1b_move_logp [M, K, 10] plus the models' names and sha256s. --fused-model (rank_model_fused.json,
scripts/fit_fused_model.py) then decides at the merge: the shards' L1b models must be exactly the
document's (state and trunk sha256), the fused MLP runs once per model on that model's fused features
(aicup.fused) and the probabilities are averaged before the expected-score decision; a question whose
fused probabilities are not finite gets --model's and is listed (fused_fallback_questions). --fused-model
and --set-models exclude each other; shards scored without --l1b-heads cannot feed a fused decision (exit 2).

With --l1b-heads the shard step also scores each game's *opponent* (issue #12): a second pass of the same CPU
pool over (k, sgf, the other colour) -- as l1b_score.py --opponent -- under the L1b heads only (the 18 profiles
see the target's moves only), giving the shard file l1b_opp_sums [n, K, 10] and l1b_opp_n_moves [n] (the
opponent's move counts, which the features use) and the meta l1b_opponent / opp_positions; --no-l1b-opponent
skips it. A fused_opp document (rank_model_fused_opp.json, 56 features, scripts/fit_fused_model.py --opp) or a
fused_opp_diff one (rank_model_fused_opp_diff.json, 59 features, --opp --diff; the differences are computed here
from the shards' per-game sums, issue #25)
decides from the target's and the opponent's sums; shards without the opponent's scores cannot feed it
(exit 2), while the 40-feature rank_model_fused.json ignores them, so one set of shard files serves both. Since
2026-11-04 the opponent pass also keeps its per-move log p in the shard file (l1b_opp_move_logp / opp_move_task /
opp_move_t, meta l1b_opp_moves): nothing here reads them; scripts/trunk_predict.py (the trunk + TCN ensemble) does.
Since 2026-11-05 (section 4M M14) the opponent pass also scores the opponents' moves under the 18 profiles (hsl_opp_sums [n, 18],
hsl_opp_move_logp [Mo, 18], meta hsl_opponent), the opponent-HSL channel of the ct6 trunk models; --no-opp-hsl skips it (the
pass then takes about a sixth of the time, as before). Nothing here reads those arrays either.

--expect-rows / --expect-public / --expect-private (defaults: the official 800 = 400 + 400; 0 disables
one) are checked against the questions read, so a truncated input fails validation; a mock test passes
its own counts. A malformed input (a question whose num_games disagrees with its filled sgf columns)
stops the run before scoring; --lenient-inputs, the deadline-day escape hatch, reads such a question
from its non-empty columns instead and lists it in the report's input_warnings (every shard and the
merge must pass the flag alike). Exit codes: 0 ok; 1 the submission failed validation; 2 malformed
inputs, a model trained on other features, or missing / inconsistent shard files; 3 a question with no
usable game was filled with FALLBACK (--allow-fallback accepts that with a warning and exit 0); 4 the
inputs were read leniently with warnings (--allow-input-warnings accepts). The submission, the
probabilities and the report are written in every case the merge reaches; a shard run exits 0 once its
file is written, the merge decides the final code.
"""
import argparse  # 解析命令列參數
import hashlib  # 計算題目雜湊
import json  # 讀寫側檔、meta 與報告
import multiprocessing as mp  # 計算特徵的 CPU 程序池
import os  # 原子改名
import sys  # 設定匯入路徑、輸出錯誤與結束碼
import time  # 計時
from collections import Counter  # 計數 HD 值與預測等級
from datetime import datetime  # 分片 meta 的寫出時間
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算與 npz 分片檔
import pandas as pd  # 題目表與機率檔

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（本檔在 scripts/）
if str(ROOT / "src") not in sys.path:  # src 還不在匯入路徑上時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src，才能匯入 aicup 套件
from aicup.fused import feature_set_of as fused_feature_set, fused_predict, load_fused_model, pairing_order  # noqa: E402  融合 MLP 的載入、配對與預測
from aicup.gamevec import SEGMENTS, WHOLE, game_vector  # noqa: E402  集合模型用的每局向量
from aicup.humansl import PROFILES  # noqa: E402  18 個等級剖面
from aicup.ladder_data import pack_questions  # noqa: E402  把題目打包成集合模型的輸入
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402  10 個等級與期望分數決策
from aicup.qfeatures import FEATURE_NAMES, question_features  # noqa: E402  q24 題目特徵
from aicup.rankmodel import _softmax, from_state  # noqa: E402  題目層模型（L3/L4）的還原與 softmax
from aicup.setmodel_io import file_sha256, load_set_model, sidecar_path, summary as set_model_summary  # noqa: E402  集合模型與側檔的讀取
from aicup.settrain import BASE_CLIP, predict_set_model  # noqa: E402  集合模型的推論
from aicup.handicap import question_hd_features_from_games  # noqa: E402  題目的讓子特徵
from aicup.sgf import handicap  # noqa: E402  讀取棋譜的 HD[]
from aicup.submission import check_game, read_test, read_test_lenient, validate_submission, write_submission  # noqa: E402  讀測試檔、檢查棋局與寫出／驗證提交檔

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"  # Human SL 網路（KataGo humanv0）的檢查點
FALLBACK = "1D"  # 只用在沒有可用棋局的題目：內部的等級兩側都有相鄰等級（差一級仍有部分分數）
EXIT_OK, EXIT_INVALID, EXIT_INPUTS, EXIT_FALLBACK, EXIT_INPUT_WARNINGS = 0, 1, 2, 3, 4  # 結束碼（見模組說明）
DEFAULT_INPUTS = [str(ROOT / "data" / "tests" / "rank_prediction_test_public.csv"),  # 預設輸入：先 Public
                  str(ROOT / "data" / "tests" / "rank_prediction_test_private.csv")]  # 再 Private


def load_questions(inputs, lenient=False):  # 讀入所有測試題目
    """(questions of the input files in order, input warnings). Strict (the default): aicup.submission.read_test,
    ValueError on a malformed file and no warnings. lenient: read_test_lenient, a malformed question read
    from its non-empty sgf columns and one warning per such question."""
    questions, warnings = [], []  # 題目與警告
    for path in inputs:  # 依序讀每個輸入檔（Public 在前）
        if lenient:  # 寬鬆模式
            qs, ws = read_test_lenient(path)  # 格式錯誤的題目也照非空的欄讀入
            warnings += ws  # 收集警告
        else:  # 嚴格模式（預設）
            qs = read_test(path)  # 格式錯誤就丟出 ValueError
        questions += qs  # 依輸入順序接上
    return questions, warnings  # 回傳題目與警告


def build_tasks(questions):  # 建立評分任務清單
    """(tasks [(k, sgf, colour)] over the usable games in input order, owner [question index of task k],
    dropped [{question_id, game, error}], passes). Deterministic: the same in every process reading the
    same inputs, which is what lets shards score disjoint slices of it."""
    tasks, owner, dropped, passes = [], [], [], 0  # 任務、所屬題目、捨棄的局與 pass 數
    for qi, q in enumerate(questions):  # 依輸入順序逐題
        for gi, (sgf, color) in enumerate(q["games"]):  # 逐局
            ok, info = check_game(sgf, color)  # 檢查這一局能否使用
            passes += info["passes"]  # 累計棋譜中的 pass 數
            if ok:  # 可用的局
                owner.append(qi)  # 記下所屬題目
                tasks.append((len(tasks), sgf, color))  # 任務編號依序遞增
            else:  # 不可用的局
                dropped.append({"question_id": q["question_id"], "game": gi + 1, "error": info["error"]})  # 記下捨棄原因（局號從 1 起算）
    return tasks, owner, dropped, passes  # 回傳任務清單與統計


def handicap_counts(questions):  # 統計各 HD 值的局數
    """{"absent" | HD: games} over every game read (usable or not): dataset v1.1 marks handicap games with HD[n] and the
    scoring replays HD >= 2 with the stones placed and the colours swapped (aicup.sgf.setup_and_moves); "absent" games are
    replayed as written, so a test set without HD[] is scored exactly as before v1.1."""
    c = Counter("absent" if (h := handicap(sgf)) is None else str(h) for q in questions for sgf, _ in q["games"])  # 每局的 HD 值，沒有 HD[] 記為 absent
    return {k: c[k] for k in sorted(c, key=lambda k: (k != "absent", k))}  # absent 排最前，其餘依字串排序


def questions_sha(inputs, questions):  # 題目雜湊
    """sha256 over the input file names and every question's id and game count: the shards of one run
    must agree on it, so a merge never mixes files made from other inputs."""
    h = hashlib.sha256()  # 依序累積輸入檔名與每題的題號、局數
    for path in inputs:  # 逐個輸入檔
        h.update(Path(path).name.encode() + b"\n")  # 只納入檔名（不含資料夾）
    for q in questions:  # 逐題
        h.update(f"{q['question_id']}:{len(q['games'])}\n".encode())  # 題號與讀入的局數
    return h.hexdigest()  # 十六進位字串


def shard_path(out, shard, nshards):  # 分片檔的路徑
    """<out stem>_shard{shard}of{nshards}.npz next to `out`."""
    out = Path(out)  # 也接受字串路徑，下一行要用 with_name
    return out.with_name(f"{out.stem}_shard{shard}of{nshards}.npz")  # 與提交檔放在同一個資料夾


def l1b_models_meta(paths):  # 記錄要用的 L1b 等級頭
    """The shard meta's record of the L1b heads files: name (stem), path, sha256s and training folds from their sidecars
    (aicup.l1b.save_heads); ValueError when a sidecar is missing."""
    out = []  # 每個等級頭檔案一筆
    for p in paths:  # 逐個檔案
        side = sidecar_path(p)  # 對應的側檔路徑
        if not side.exists():  # 沒有側檔
            raise ValueError(f"{p}: sidecar {side.name} is missing")  # 無法核對，停止
        meta = json.loads(side.read_text())  # 讀入側檔
        out.append({"name": Path(p).stem, "path": str(p), "state_sha256": meta.get("state_sha256"), "trunk_file": meta.get("trunk_file"),  # 名稱、路徑、權重雜湊與主幹檔
                    "trunk_sha256": meta.get("trunk_sha256"), "train_folds": meta.get("train_folds"), "levels": meta.get("levels")})  # 主幹的 sha256、訓練折與等級
    return out  # 回傳紀錄清單


def score_tasks(tasks, workers, log=print, l1b_heads=(), opponent=True, opponent_hsl=True):  # 分片步驟的評分
    """Score (k, sgf, colour) tasks on this GPU under the 18 profiles and, per L1b heads file, under its 10 rank
    heads: ({k: (n_moves, sums[18 + 10K])}, (move_logp [M, 18 + 10K], move_task [M], move_t [M]) or None, positions,
    wall seconds, opp). Each heads file gets its own model instance (load_heads restores a fine-tuned trunk in place);
    the positions' features are computed once for every model. opp is None, or with l1b_heads and opponent
    ({k: (the opponent's n_moves, sums[10K])}, positions, wall seconds): a second pass over the same games with the
    colours swapped (l1b_score.py --opponent), under the L1b heads -- and, with opponent_hsl (section 4M M14), under the 18
    profiles too: the pass's columns are then [18 + 10K] (profiles first), else [10K]."""
    import torch  # 讀檢查點設定、在 GPU 上評分要用 torch
    from aicup.humansl import load_humansl, meta_rows, score_positions as hsl_score  # Human SL 的載入與評分
    from aicup.l1b import load_heads, score_positions as l1b_score  # L1b 等級頭的載入與評分
    from aicup.scoring import init_worker, score_stream  # 程序池初始化與串流評分
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]  # 檢查點的模型設定，給 worker 建特徵器
    log(f"torch CPU threads: {torch.get_num_threads()} (scripts/nano4/shards.sh caps them per shard)")  # 回報 CPU 執行緒數（shards.sh 會限制）
    pool = mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(config,))  # 在初始化 CUDA 之前 fork 出程序池
    try:  # 確保最後關閉程序池
        model, _ = load_humansl(str(CKPT), "cuda")  # Human SL 網路
        l1b = []  # (模型實例, 等級頭) 的清單
        for p in l1b_heads:  # 逐個 L1b 等級頭檔案
            model_k, _ = load_humansl(str(CKPT), "cuda")  # 每個等級頭檔案用自己的模型實例
            heads_k, meta_k, warnings = load_heads(p, model_k, "cuda")  # 載入等級頭；微調主幹原地還原到 model_k
            for w in warnings:  # 逐一印出載入警告
                log(f"WARNING: {w}")  # 警告內容
            l1b.append((model_k, heads_k))  # 加入清單
        rows = meta_rows([(p, p) for p in PROFILES])  # 18 個剖面的 metadata 列

        def scorer(bins, globs, legal, target, colors, rows_):  # 同一批局面一次給所有模型評分
            parts = [hsl_score(model, bins, globs, legal, target, colors, rows_)]  # 18 個剖面的 log p
            parts += [l1b_score(m, h, bins, globs, legal, target, colors) for m, h in l1b]  # 每個 L1b 模型的 10 個等級頭
            return np.concatenate(parts, axis=1)  # 欄位依序接起來：[18 + 10K]

        per_task, moves, n_pos, wall = score_stream(pool, model, tasks, rows, log_every=500, log=log, scorer=scorer if l1b else None)  # 評本人的著手
        opp = None  # 對手評分趟的結果
        if l1b and opponent:  # 有 L1b 等級頭而且要評對手
            def opp_scorer(bins, globs, legal, target, colors, rows_):  # 對手評分趟的評分函式
                parts = [hsl_score(model, bins, globs, legal, target, colors, rows_)] if opponent_hsl else []  # 需要時也用 18 個剖面
                parts += [l1b_score(m, h, bins, globs, legal, target, colors) for m, h in l1b]  # 每個 L1b 模型的 10 個等級頭
                return np.concatenate(parts, axis=1)  # [18 + 10K] 或 [10K]

            opp_tasks = [(k, sgf, "W" if colour == "B" else "B") for k, sgf, colour in tasks]  # 同樣的局，顏色對調
            log(f"scoring the opponents' moves of {len(opp_tasks)} games under {'the 18 profiles and ' if opponent_hsl else ''}the {len(l1b)} L1b models", flush=True)  # 回報對手評分趟的內容
            o_per_task, o_moves, o_pos, o_wall = score_stream(pool, model if opponent_hsl else None, opp_tasks, rows if opponent_hsl else None,  # 評對手的著手，沿用同一個程序池
                                                              log_every=500, log=log, scorer=opp_scorer)  # 進度間隔與評分函式
            opp = (o_per_task, o_pos, o_wall, o_moves)  # 每局總和、局面數、秒數與逐手資料
        return per_task, moves, n_pos, wall, opp  # 回傳兩趟的結果
    finally:  # 無論成功或失敗
        pool.close()  # 關閉程序池


def save_shard(path, shard, nshards, tasks, per_task, moves, n_pos, wall, inputs, sha, n_tasks_total, l1b_models=(), opp=None):  # 寫出一個分片檔
    """Write a shard file atomically: task (the global indices scored), n_moves, sums [n, 18] (float64),
    move_logp [M, 18], move_task [M], move_t [M] and __meta__ (JSON). With l1b_models (K heads files, scored as
    columns 18.. of the sums and per-move log p) also l1b_sums [n, K, 10] (float64) and l1b_move_logp [M, K, 10]
    (float32), and the meta lists the models; with opp (score_tasks' opponent pass: ({k: (n, sums[10K])}, positions,
    wall[, moves])) also l1b_opp_sums [n, K, 10] (float64) and l1b_opp_n_moves [n], and the meta's l1b_opponent is true;
    when the pass also hands over its per-move arrays (moves: (logp [Mo, 10K], task [Mo], t [Mo])) they are written as
    l1b_opp_move_logp [Mo, K, 10] (float32), opp_move_task [Mo] and opp_move_t [Mo] -- the opponent plies' L1b log p the
    trunk + TCN path (scripts/trunk_predict.py, 2026-11-04) reads; the meta's l1b_opp_moves says whether they are there.
    An opponent pass with 18 + 10K columns (score_tasks' opponent_hsl, section 4M M14) also gives hsl_opp_sums [n, 18] (float64)
    and hsl_opp_move_logp [Mo, 18] (float32), and the meta's hsl_opponent is true. Returns the meta."""
    P, L, K = len(PROFILES), len(LEVELS), len(l1b_models)  # 剖面數、等級數與 L1b 模型數
    idx = np.array([k for k, _, _ in tasks], dtype=np.int64)  # 本分片評過的全域任務編號
    n_moves = np.array([per_task[k][0] for k in idx], dtype=np.int64)  # 每局本人的手數
    all_sums = np.stack([per_task[k][1] for k in idx]).astype(np.float64) if len(idx) else np.zeros((0, P + L * K))  # 每局的 log p 總和（空分片給空陣列）
    if all_sums.shape[1] != P + L * K:  # 欄數不符
        raise ValueError(f"scored {all_sums.shape[1]} columns per game, expected {P} profiles + {K} x {L} L1b heads")  # 停止
    if moves is None:  # 沒有任何一手（空分片）
        moves = (np.zeros((0, P + L * K), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))  # 補上空陣列
    with_opp = bool(K and opp is not None)  # 是否有對手評分趟
    o_moves = opp[3] if with_opp and len(opp) > 3 else None  # 對手的逐手陣列（有的話）
    meta = {"shard": shard, "nshards": nshards, "n_tasks_total": n_tasks_total, "n_tasks": int(len(idx)),  # meta：分片、任務總數與本分片的任務數
            "inputs": [str(p) for p in inputs], "questions_sha256": sha, "positions": int(n_pos), "scoring_wall_s": float(wall),  # 輸入、題目雜湊、局面數與秒數
            "profiles": list(PROFILES), "ckpt": CKPT.name, "l1b_models": [dict(m) for m in l1b_models], "levels": list(LEVELS),  # 剖面、檢查點、L1b 模型與等級
            "l1b_opponent": with_opp, "opp_positions": int(opp[1]) if with_opp else 0, "opp_scoring_wall_s": float(opp[2]) if with_opp else 0.0,  # 對手評分趟的資訊
            "l1b_opp_moves": o_moves is not None, "hsl_opponent": False, "written": datetime.now().isoformat(timespec="seconds")}  # 對手逐手資料是否存在、對手 HSL（稍後更新）與寫出時間
    arrays = {"task": idx, "n_moves": n_moves, "sums": all_sums[:, :P], "move_logp": moves[0][:, :P], "move_task": moves[1], "move_t": moves[2]}  # Human SL 的每局與逐手陣列
    if K:  # 有 L1b 模型
        arrays["l1b_sums"] = all_sums[:, P:].reshape(len(idx), K, L)  # 18 個剖面之後的欄是 L1b 總和，整理成 [n, K, 10]
        arrays["l1b_move_logp"] = moves[0][:, P:].reshape(len(moves[1]), K, L).astype(np.float32)  # L1b 的逐手 log p
    if with_opp:  # 有對手評分趟
        o_per_task = opp[0]  # 對手的每局結果
        missing = [int(k) for k in idx if k not in o_per_task]  # 缺少對手分數的任務
        if missing:  # 有缺
            raise ValueError(f"{len(missing)} tasks have no opponent scores (e.g. {missing[:3]})")  # 停止
        o_sums = np.stack([o_per_task[k][1] for k in idx]).astype(np.float64) if len(idx) else np.zeros((0, L * K))  # 對手的每局總和
        P_o = o_sums.shape[1] - L * K  # 0：只有 L1b 等級頭；P：18 個剖面在前（section 4M M14）
        if P_o not in (0, P):  # 欄數不符
            raise ValueError(f"the opponent pass scored {o_sums.shape[1]} columns per game, expected {K} x {L} L1b heads (+ {P} profiles)")  # 停止
        meta["hsl_opponent"] = P_o == P  # 對手是否也以 18 個剖面評過
        arrays["l1b_opp_sums"] = o_sums[:, P_o:].reshape(len(idx), K, L)  # 對手的 L1b 總和
        arrays["l1b_opp_n_moves"] = np.array([o_per_task[k][0] for k in idx], dtype=np.int64)  # 對手的手數
        if P_o:  # 有對手的 Human SL 分數
            arrays["hsl_opp_sums"] = o_sums[:, :P]  # 對手的 18 剖面總和
        if o_moves is not None:  # 有對手的逐手資料
            if o_moves[0].shape[1] != P_o + L * K or len(o_moves[0]) != int(arrays["l1b_opp_n_moves"].sum()):  # 欄數或手數對不上
                raise ValueError(f"the opponent pass gives {o_moves[0].shape} per-move log p for {int(arrays['l1b_opp_n_moves'].sum())} moves "  # 說明實際與預期的形狀
                                 f"of {K} x {L} heads{f' + {P} profiles' if P_o else ''}")  # 錯誤訊息的後半
            arrays["l1b_opp_move_logp"] = o_moves[0][:, P_o:].reshape(len(o_moves[1]), K, L).astype(np.float32)  # 對手各手的 L1b log p
            if P_o:  # 有對手的 Human SL 分數
                arrays["hsl_opp_move_logp"] = o_moves[0][:, :P].astype(np.float32)  # 對手各手的 18 剖面 log p
            arrays["opp_move_task"] = np.asarray(o_moves[1], dtype=np.int32)  # 對手各手所屬的任務
            arrays["opp_move_t"] = np.asarray(o_moves[2], dtype=np.int32)  # 對手各手的手序
    path = Path(path)  # 也接受字串路徑，下一行要組出暫存檔名
    tmp = path.with_name(path.stem + ".tmp.npz")  # 暫存檔名
    np.savez_compressed(tmp, __meta__=np.asarray(json.dumps(meta)), **arrays)  # 寫出壓縮的 npz，meta 存成 JSON 字串
    os.replace(tmp, path)  # 改名，完成原子寫入
    return meta  # 回傳 meta


def load_shards_l1b(out, nshards, with_moves=False):  # 讀取分片檔的 L1b 分數
    """([L1b model records], {k: l1b sums [K, 10]}, moves, opp) from the shard files: the models (name, state and trunk
    sha256) must be the same in every shard, and every shard must carry the opponent's scores or none; shards without
    L1b scores give ([], {}, None, None). moves is None, or with with_moves {k: (logp [n_moves, K, 10], t [n_moves])};
    opp is None, or {k: (the opponent's n_moves, sums [K, 10])} when the shards were scored with the opponent pass."""
    models, per_task, moves, opp, opponent = None, {}, {} if with_moves else None, {}, None  # 模型紀錄、本人總和、逐手資料、對手總和與是否有對手分數
    for i in range(nshards):  # 逐個分片檔
        path = shard_path(out, i, nshards)  # 第 i 個分片檔
        with np.load(path, allow_pickle=False) as z:  # 開啟分片檔（不允許 pickle）
            meta = json.loads(str(z["__meta__"]))  # 分片的 meta
            here = [(m.get("name"), m.get("state_sha256"), m.get("trunk_sha256")) for m in meta.get("l1b_models", [])]  # 這個分片用的 L1b 模型
            has_opp = bool(meta.get("l1b_opponent"))  # 這個分片有沒有對手分數
            if models is None:  # 第一個分片
                models, key, opponent = meta.get("l1b_models", []), here, has_opp  # 作為其他分片的比對基準
            elif here != key:  # 模型與分片 0 不同
                raise ValueError(f"{path}: L1b models {here} differ from shard 0's {key}; every shard must be scored with the same heads")  # 每個分片必須用相同的等級頭
            elif has_opp != opponent:  # 對手分數的有無與分片 0 不同
                raise ValueError(f"{path}: {'carries' if has_opp else 'lacks'} the opponents' L1b scores while shard 0 "  # 說明哪一邊有、哪一邊沒有
                                 f"{'carries' if opponent else 'lacks'} them; every shard must be scored alike (--no-l1b-opponent or not)")  # 每個分片必須以相同方式評分
            if not here:  # 沒有 L1b 分數
                continue  # 這個分片沒有東西可讀
            for k, s in zip(z["task"].tolist(), z["l1b_sums"]):  # 逐任務
                per_task[k] = s.astype(np.float64)  # 本人的 L1b 總和 [K, 10]
            if has_opp:  # 有對手分數
                for k, n, s in zip(z["task"].tolist(), z["l1b_opp_n_moves"].tolist(), z["l1b_opp_sums"]):  # 逐任務
                    opp[k] = (int(n), s.astype(np.float64))  # 對手的手數與 L1b 總和
            if with_moves:  # 需要逐手資料
                mt, logp, t = z["move_task"].astype(np.int64), z["l1b_move_logp"].astype(np.float64), z["move_t"].astype(np.int64)  # 每手所屬任務、log p 與手序
                order = np.argsort(mt, kind="stable")  # 依任務排序，保持同一任務內的順序
                mt, logp, t = mt[order], logp[order], t[order]  # 套用排序
                tasks_here, starts = np.unique(mt, return_index=True)  # 各任務的起始位置
                ends = np.append(starts[1:], len(mt))  # 各任務的結束位置
                for k, a, b in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):  # 逐任務切片
                    moves[k] = (logp[a:b], t[a:b])  # 這一局本人各手的 L1b log p 與手序
    return models or [], per_task, moves, (opp if opponent else None)  # 沒有對手分數時 opp 為 None


def load_shards_opp_hsl(out, nshards, with_moves=False):  # 讀取分片檔中對手的 Human SL 分數
    """({k: (the opponent's n_moves, sums [18])}, moves) from the shard files' opponent-HSL arrays (section 4M M14): moves is None,
    or with with_moves {k: (logp [n_opp, 18] float64, t [n_opp])}. ValueError when a shard was scored without the 18-profile
    opponent pass (meta hsl_opponent false, e.g. shards from before 2026-11-05 or --no-opp-hsl)."""
    per_task, moves = {}, {} if with_moves else None  # 對手的 Human SL 總和與逐手資料
    for i in range(nshards):  # 逐個分片檔
        path = shard_path(out, i, nshards)  # 第 i 個分片檔
        with np.load(path, allow_pickle=False) as z:  # 開啟分片檔（不允許 pickle）
            meta = json.loads(str(z["__meta__"]))  # 分片的 meta
            if not meta.get("hsl_opponent"):  # 這個分片沒有用 18 個剖面評對手
                raise ValueError(f"{path}: no opponent scores under the 18 profiles (hsl_opponent): the shards must come from predict_submission.py "  # 說明分片的版本要求
                                 "of 2026-11-05 or later, scored with --l1b-heads and the opponent pass, without --no-opp-hsl")  # 錯誤訊息的後半
            for k, n, s in zip(z["task"].tolist(), z["l1b_opp_n_moves"].tolist(), z["hsl_opp_sums"]):  # 逐任務
                per_task[k] = (int(n), s.astype(np.float64))  # 對手的手數與 18 剖面總和
            if with_moves:  # 需要逐手資料
                if not meta.get("l1b_opp_moves"):  # 沒有對手的逐手資料
                    raise ValueError(f"{path}: no per-move opponent log p (l1b_opp_moves)")  # 停止
                mt, logp, t = z["opp_move_task"].astype(np.int64), z["hsl_opp_move_logp"].astype(np.float64), z["opp_move_t"].astype(np.int64)  # 對手每手所屬任務、log p 與手序
                order = np.argsort(mt, kind="stable")  # 依任務排序，保持同一任務內的順序
                mt, logp, t = mt[order], logp[order], t[order]  # 套用排序
                tasks_here, starts = np.unique(mt, return_index=True)  # 各任務的起始位置
                ends = np.append(starts[1:], len(mt))  # 各任務的結束位置
                for k, a, b in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):  # 逐任務切片
                    moves[k] = (logp[a:b], t[a:b])  # 這一局對手各手的 log p 與手序
                for k in z["task"].tolist():  # 逐任務核對
                    got = len(moves[k][0]) if k in moves else 0  # 實際存的對手手數
                    if got != per_task[k][0]:  # 與對手的 n_moves 不符
                        raise ValueError(f"{path}: task {k} has {got} scored opponent moves, the opponent's n_moves says {per_task[k][0]}")  # 停止
    return per_task, moves  # 回傳對手的 Human SL 資料


def fused_probs(questions, owner, per_task, l1b_per_task, mlp, order, rank_model, opp_per_task=None, opponent=False, feature_set=None, task_games=None):  # 融合 MLP 的每題機率
    """({question index: probabilities [10]} for every question with a usable game, info): the fused MLP on each
    L1b model's fused features (shard columns `order`), averaged; a question whose averaged probabilities are not
    finite gets rank_model's (q24) probabilities and is listed in info["fused_fallback_questions"]. With opponent (a
    fused_opp document) the opponent's sums and move counts opp_per_task {k: (n, [K, 10])} join the features, and a
    fused_opp_diff document (feature_set) adds the opponent-minus-own differences; ValueError when the shards carry none.
    A fused_opp_diff_hd document (issue #47) also takes the question's handicap features, computed from the usable games'
    records task_games {k: (sgf, colour)} (aicup.handicap.question_hd_features_from_games; a record without HD[] counts as
    unknown, zeros -- main refuses such inputs unless --allow-missing-hd)."""
    feature_set = fused_feature_set(opponent, feature_set)  # 決定特徵集名稱（相容較早的 opponent 旗標）
    opponent = feature_set != "fused"  # 除了 fused 以外都需要對手分數
    if opponent and opp_per_task is None:  # 需要對手分數但分片沒有
        raise ValueError("the fused model takes the opponent's L1b features but the shards carry no opponent scores: score them "  # 說明要重評，或改用 40 個特徵的文件
                         "again with --l1b-heads (without --no-l1b-opponent), or decide with the 40-feature rank_model_fused.json")  # 錯誤訊息的後半
    with_hd = feature_set == "fused_opp_diff_hd"  # 是否需要讓子特徵
    if with_hd and task_games is None:  # 需要讓子特徵卻沒有棋譜
        raise ValueError("the fused_opp_diff_hd model needs the games' records (task_games) for the handicap features")  # 停止
    by_q = {}  # 題目 → 它的任務
    for k, qi in enumerate(owner):  # 依任務順序
        by_q.setdefault(qi, []).append(k)  # 歸到所屬的題目
    probs, fallback = {}, []  # 每題機率與退回 rank_model 的題目
    for qi, ks in by_q.items():  # 逐題
        hsl = np.stack([per_task[k][1] for k in ks])  # 各局的 Human SL 總和
        n = np.array([per_task[k][0] for k in ks])  # 各局的手數
        missing = [k for k in ks if k not in l1b_per_task] + ([k for k in ks if k not in opp_per_task] if opponent else [])  # 缺少 L1b 或對手分數的局
        if missing:  # 有缺
            raise ValueError(f"question {questions[qi]['question_id']}: {len(missing)} games have no L1b scores in the shards")  # 停止
        l1b = np.stack([l1b_per_task[k] for k in ks])  # 各局的 L1b 總和 [m, K, 10]
        if opponent:  # 需要對手特徵
            hd = question_hd_features_from_games([task_games[k] for k in ks]) if with_hd else None  # 需要時計算讓子特徵
            p = fused_predict(mlp, hsl, l1b, n, order, opp_sums=np.stack([opp_per_task[k][1] for k in ks]),  # 加入對手的總和
                              opp_n_moves=np.array([opp_per_task[k][0] for k in ks]), feature_set=feature_set, hd_feats=hd)  # 對手手數、特徵集與讓子特徵
        else:  # 40 個特徵的 fused 文件
            p = fused_predict(mlp, hsl, l1b, n, order)  # 只用本人的分數
        if not np.isfinite(p).all():  # 機率不是有限值
            p = rank_model.predict_proba(question_features(hsl, n)[None, :])[0]  # 改用 rank_model 的 q24 機率
            fallback.append(questions[qi]["question_id"])  # 記下這一題
        probs[qi] = p  # 這一題的機率
    return probs, {"fused_fallback_questions": fallback}  # 回傳機率與退回的題目


def load_shards(out, nshards, n_tasks_total, sha, with_moves=False):  # 合併時讀入全部分片檔
    """({k: (n_moves, sums)}, [meta per shard], moves) from the nshards files of `out`. Every file must exist,
    be shard i of nshards made from the same questions (questions_sha256), and together they must score every
    task 0..n_tasks_total-1 exactly once; FileNotFoundError / ValueError otherwise. moves is None, or with
    with_moves {k: (logp [n_moves, 18] float64, t [n_moves])} -- each task's per-move log p and move indices,
    checked to count n_moves."""
    per_task, metas, moves = {}, [], {} if with_moves else None  # 每局結果、各分片 meta 與逐手資料
    for i in range(nshards):  # 逐個分片檔
        path = shard_path(out, i, nshards)  # 第 i 個分片檔
        if not path.exists():  # 分片檔不存在
            raise FileNotFoundError(f"shard file {path} is missing ({nshards} shards expected)")  # 停止
        with np.load(path, allow_pickle=False) as z:  # 開啟分片檔（不允許 pickle）
            meta = json.loads(str(z["__meta__"]))  # 分片的 meta
            if (meta["shard"], meta["nshards"], meta["questions_sha256"]) != (i, nshards, sha):  # 分片編號、分片數或題目雜湊不符
                raise ValueError(f"{path}: made for another sharding or other inputs (shard {meta['shard']} of {meta['nshards']}, "  # 說明實際的分片與題目雜湊
                                 f"questions {meta['questions_sha256'][:12]}; expected {i} of {nshards}, {sha[:12]})")  # 以及預期值
            for k, n, s in zip(z["task"].tolist(), z["n_moves"].tolist(), z["sums"]):  # 逐任務
                if k in per_task:  # 同一任務出現在兩個分片
                    raise ValueError(f"task {k} was scored by two shards (shard {i} and an earlier one)")  # 停止
                per_task[k] = (int(n), s)  # 手數與 18 剖面總和
            if with_moves:  # 需要逐手資料
                mt, logp, t = z["move_task"].astype(np.int64), z["move_logp"].astype(np.float64), z["move_t"].astype(np.int64)  # 每手所屬任務、log p 與手序
                order = np.argsort(mt, kind="stable")  # 依任務排序，保持同一任務內的順序
                mt, logp, t = mt[order], logp[order], t[order]  # 套用排序
                tasks_here, starts = np.unique(mt, return_index=True)  # 各任務的起始位置
                ends = np.append(starts[1:], len(mt))  # 各任務的結束位置
                for k, a, b in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):  # 逐任務切片
                    moves[k] = (logp[a:b], t[a:b])  # 這一局各手的 log p 與手序
                for k in z["task"].tolist():  # 逐任務核對
                    got = len(moves[k][0]) if k in moves else 0  # 實際存的手數
                    if got != per_task[k][0]:  # 與 n_moves 不符
                        raise ValueError(f"{path}: task {k} has {got} scored moves, n_moves says {per_task[k][0]}")  # 停止
        metas.append(meta)  # 記下這個分片的 meta
    missing = sorted(set(range(n_tasks_total)) - set(per_task))  # 沒有被評到的任務
    unknown = sorted(set(per_task) - set(range(n_tasks_total)))  # 不在任務清單中的任務
    if missing or unknown:  # 分片沒有剛好涵蓋全部任務
        raise ValueError(f"the shards cover {len(per_task) - len(unknown)} of {n_tasks_total} tasks: {len(missing)} missing "  # 說明涵蓋的任務數與缺少數
                         f"(e.g. {missing[:3]}), {len(unknown)} unknown (e.g. {unknown[:3]})")  # 以及多出的任務數
    return per_task, metas, moves  # 回傳合併後的結果


def game_vectors(per_task, moves):  # 集合模型所需的每局向量
    """The vectors dict pack_questions reads, one row per task in index order: game_key (str(k)), staged [T, 300]
    and whole [T, 75] (aicup.gamevec.game_vector of the task's per-move log p), sums [T, 18] (float64) and
    n_moves [T]."""
    keys = sorted(per_task)  # 依任務索引排序
    staged = np.stack([game_vector(*moves[k], SEGMENTS) for k in keys]) if keys else np.zeros((0, 300), dtype=np.float32)  # 四個階段（SEGMENTS）的每局向量 [T, 300]
    whole = np.stack([game_vector(*moves[k], WHOLE) for k in keys]) if keys else np.zeros((0, 75), dtype=np.float32)  # 整局的每局向量
    return {"game_key": np.asarray([str(k) for k in keys], dtype=str), "staged": staged, "whole": whole,  # 鍵與兩種向量
            "sums": np.stack([np.asarray(per_task[k][1], dtype=np.float64) for k in keys]) if keys else np.zeros((0, len(PROFILES))),  # 18 剖面總和
            "n_moves": np.asarray([per_task[k][0] for k in keys], dtype=np.int64)}  # 每局手數


def load_set_models(paths, device="cpu"):  # 載入所有集合模型
    """[(Fitted, sidecar, path)] of the .pt files and the loader's warnings (aicup.setmodel_io.load_set_model)."""
    models, warnings = [], []  # 模型與載入警告
    for path in paths:  # 逐個 .pt 檔
        fitted, meta, w = load_set_model(path, device)  # 載入模型與側檔
        models.append((fitted, meta, str(path)))  # 加入清單
        warnings += w  # 收集警告
    return models, warnings  # 回傳模型與警告


def check_pairing(models, base_doc, base_path):  # 核對集合模型與配對基準
    """The set models must be Δ-corrections of --set-base. A sidecar that records its training questions
    (questions_sha256, experiments/train_set_model.py) needs a base document with the same value: ValueError
    otherwise, also when the base has none (results/models/rank_model.json is not a paired base). Returns
    warnings: a sidecar's base_model_sha256 that is not the file's (the same base rewritten, or another file
    fitted on the same questions)."""
    warnings, sha = [], file_sha256(base_path)  # 警告與基準檔的 sha256
    for _, meta, path in models:  # 逐個集合模型
        want = meta.get("questions_sha256")  # 側檔記錄的訓練題目雜湊
        if want and base_doc.get("questions_sha256") != want:  # 與基準的不同（或基準沒有這個值）
            have = base_doc.get("questions_sha256")  # 基準記錄的值
            raise ValueError(f"{path}: trained on other questions than the base {base_path} (questions_sha256 {want[:12]} vs "  # 說明兩者的雜湊
                             f"{have[:12] if have else 'none'}); --set-base must be the paired base its sidecar names ({meta.get('base_model')})")  # 提示應給的配對基準
        if meta.get("base_model_sha256") and meta["base_model_sha256"] != sha:  # 側檔記錄的基準檔雜湊與實際檔案不同
            warnings.append(f"{path}: its sidecar records the base file sha256 {meta['base_model_sha256'][:12]}, "  # 只記警告
                            f"{base_path} has {sha[:12]} (the same questions; a rewritten base file)")  # 題目相同、重寫過的基準檔
    return warnings  # 回傳警告


def average_set_models(frame, vec, X, base_model, models):  # 集合模型的平均機率
    """(P [Q, 10], info) of the set models on the questions of `frame` (question_id, player_id, n_games, game_keys
    into `vec`) with q24 features X [Q, 24]: the base logits are log(clip(base_model probabilities, BASE_CLIP));
    every model's predict_set_model gives probabilities, averaged first over the models of one rung (its seeds),
    then over the rungs. A question whose averaged probabilities are not finite gets the base's softmax instead
    and is listed in info["set_model_fallback_questions"]; info["rungs"] counts the models per rung."""
    base = np.log(np.clip(base_model.predict_proba(X), BASE_CLIP, None))  # 基準 logits：log(clip(基準機率))
    packed = {rep: pack_questions(frame, vec, rep, True) for rep in {meta["rep"] for _, meta, _ in models}}  # 每種表示各打包一次題目集合
    per_rung = {}  # 階 → 各模型的機率
    for fitted, meta, _ in models:  # 逐個集合模型
        per_rung.setdefault(meta["rung"], []).append(predict_set_model(fitted, packed[meta["rep"]], base))  # 依所屬的階收集機率
    P = np.mean([np.mean(ps, axis=0) for ps in per_rung.values()], axis=0)  # 先對階內的種子平均，再對各階平均
    bad = ~np.isfinite(P).all(axis=1)  # 機率不是有限值的題目
    if bad.any():  # 有這種題目
        P[bad] = _softmax(base[bad])  # 改用基準的 softmax
    info = {"set_model_fallback_questions": frame["question_id"][bad].tolist(), "rungs": {r: len(ps) for r, ps in per_rung.items()}}  # 退回的題目與各階的模型數
    return P, info  # 回傳機率與資訊


def question_frame(questions, owner, keys):  # 建立 pack_questions 需要的題目表
    """(frame for pack_questions, qis, by_q): the questions with a usable game, player_id = question_id, game_keys
    = keys[k] of their tasks k in task order; by_q[qi] lists those tasks."""
    by_q = {}  # 題目 → 它的任務
    for k, qi in enumerate(owner):  # 依任務順序
        by_q.setdefault(qi, []).append(k)  # 歸到所屬的題目
    qis = sorted(by_q)  # 有可用棋局的題目
    frame = pd.DataFrame({"question_id": [questions[qi]["question_id"] for qi in qis],  # 題號
                          "player_id": [questions[qi]["question_id"] for qi in qis],  # 以題號充當 player_id
                          "n_games": [len(by_q[qi]) for qi in qis], "game_keys": [[keys[k] for k in by_q[qi]] for qi in qis]})  # 局數與各局的鍵
    return frame, qis, by_q  # 回傳題目表與索引


def set_model_probs(questions, owner, per_task, moves, base_model, models):  # 集合模型的每題機率
    """({question index: probabilities [10]} for every question with a usable game, info) -- average_set_models
    over the per-game vectors of the shards (game_vectors; task k is game key str(k))."""
    frame, qis, by_q = question_frame(questions, owner, [str(k) for k in range(len(owner))])  # 題目表；任務 k 的鍵為 str(k)
    if not qis:  # 沒有任何可用的題目
        return {}, {"set_model_fallback_questions": [], "rungs": {}}  # 回傳空結果
    X = np.stack([question_features(np.stack([per_task[k][1] for k in by_q[qi]]), np.array([per_task[k][0] for k in by_q[qi]]))  # 每題的 q24 特徵
                  for qi in qis])  # 依題目順序
    P, info = average_set_models(frame, game_vectors(per_task, moves), X, base_model, models)  # 集合模型的平均機率
    return dict(zip(qis, P)), info  # 題目索引 → 機率


def load_rank_model(path):  # 載入 L3/L4 題目層模型
    """(saved JSON, model) of results/models/rank_model.json; ValueError when it was trained on other
    profiles or features than this pipeline computes."""
    saved = json.loads(Path(path).read_text())  # 讀入模型 JSON
    if saved["profiles"] != PROFILES or saved["features"] != FEATURE_NAMES:  # 剖面或特徵與本管線不同
        raise ValueError(f"{path}: the model was trained with different profiles or features than this pipeline computes")  # 停止
    return saved, from_state(saved["state"])  # 回傳 JSON 與還原的模型


def predict_questions(questions, owner, per_task, rank_model, set_probs=None):  # 每題的期望分數決策
    """(rows [(question_id, rank)], probs [[10 floats]] (NaN for a fallback question), fallback [question ids]):
    the expected-score decision on each question's probabilities -- set_probs[qi] when given (set_model_probs),
    else rank_model's on the question's q24 features; a question with no usable game gets FALLBACK."""
    by_q = {}  # 題目 → 各局的 (手數, 總和)
    for k, qi in enumerate(owner):  # 依任務順序
        by_q.setdefault(qi, []).append(per_task[k])  # 歸到所屬的題目
    rows, probs, fallback = [], [], []  # 提交列、機率與退路題
    for qi, q in enumerate(questions):  # 依輸入順序逐題
        games = by_q.get(qi, [])  # 這一題的可用局
        if not games:  # 沒有可用棋局
            rows.append((q["question_id"], FALLBACK))  # 填入 FALLBACK
            probs.append([float("nan")] * len(LEVELS))  # 機率記為 NaN
            fallback.append(q["question_id"])  # 列入退路題
            continue  # 下一題
        if set_probs is not None:  # 有集合模型或融合模型的機率
            p = set_probs[qi]  # 直接使用
        else:  # 用 rank_model
            x = question_features(np.stack([s for _, s in games]), np.array([n for n, _ in games]))  # 這一題的 q24 特徵
            p = rank_model.predict_proba(x[None, :])[0]  # rank_model 的機率
        rows.append((q["question_id"], LEVELS[expected_score_decision(p)]))  # 期望分數決策
        probs.append([float(v) for v in p])  # 記下機率
    return rows, probs, fallback  # 回傳決策、機率與退路題


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明的第一行作為描述
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="the test CSVs, public first")  # 測試 CSV（Public 在前）
    ap.add_argument("--model", default=str(ROOT / "results" / "models" / "rank_model.json"))  # L3/L4 題目層模型：預設由它決策，融合機率不是有限值時也退回它
    ap.add_argument("--out", default=str(ROOT / "submission_rank.csv"))  # 提交檔路徑，分片檔也寫在它旁邊
    ap.add_argument("--report", default=None, help="default: <out stem>_report.json")  # 報告路徑
    ap.add_argument("--workers", type=int, default=10, help="CPU feature workers of the scorer")  # 計算特徵的 CPU worker 數
    ap.add_argument("--shard", type=int, default=None, help="score only this shard (0-based) of --nshards and write its file")  # 只評這個分片並寫出分片檔
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--merge", action="store_true", help="predict from the --nshards shard files already written; no scoring")  # 只合併已寫好的分片檔，不評分
    ap.add_argument("--expect-rows", type=int, default=800, help="questions expected in the inputs (0: no check)")  # 預期的總題數
    ap.add_argument("--expect-public", type=int, default=400, help="pub_ questions expected (0: no check)")  # 預期的 Public 題數
    ap.add_argument("--expect-private", type=int, default=400, help="priv_ questions expected (0: no check)")  # 預期的 Private 題數
    ap.add_argument("--allow-fallback", action="store_true",  # 有退路題時仍以 0 結束
                    help=f"exit 0 even when a question had no usable game and was set to {FALLBACK}")  # 說明文字
    ap.add_argument("--lenient-inputs", action="store_true",  # 截止日的緊急出口：寬鬆讀入
                    help="read a question whose num_games disagrees with its filled sgf columns from the non-empty columns, "  # 說明文字
                         "with a warning in the report, instead of stopping (exit 4 unless --allow-input-warnings)")  # 說明文字的後半
    ap.add_argument("--allow-input-warnings", action="store_true", help="exit 0 despite --lenient-inputs warnings")  # 寬鬆讀入的警告不算失敗
    ap.add_argument("--allow-missing-hd", action="store_true", help="let a fused_opp_diff_hd --fused-model decide although some games carry no "  # 允許 _hd 文件處理沒有 HD[] 的局
                    "HD[] (their handicap features are zeros); without it such inputs are refused (exit 2)")  # 說明文字的後半
    ap.add_argument("--set-models", nargs="*", default=[], metavar="PT",  # 集合模型（待辦第 2 項）
                    help="set models (.pt with .json sidecars, experiments/train_set_model.py) whose averaged probabilities decide")  # 說明文字
    ap.add_argument("--set-base", default=None, help="the paired base of the set models (rank_model.json format); required with them")  # 集合模型的配對基準
    ap.add_argument("--set-device", default="cpu", help="device the set models run on")  # 集合模型執行的裝置
    ap.add_argument("--l1b-heads", nargs="*", default=[], metavar="PT",  # 分片步驟一併評分的 L1b 等級頭
                    help="L1b heads files (.pt + .json, with a fine-tuned trunk when the sidecar names one) scored alongside the "  # 說明文字
                         "18 profiles at the shard step; their sums go into the shard files for a fused decision")  # 說明文字的後半
    ap.add_argument("--fused-model", default=None,  # 融合文件：改以融合 MLP 決策
                    help="rank_model_fused.json or rank_model_fused_opp.json (scripts/fit_fused_model.py): decide with the fused MLP "  # 說明文字
                         "averaged over its L1b models, which must be exactly the models the shards were scored with")  # 說明文字的後半
    ap.add_argument("--no-l1b-opponent", dest="l1b_opponent", action="store_false",  # 關閉對手評分趟
                    help="with --l1b-heads: do not score the opponents' moves (the shards then cannot feed a fused_opp decision)")  # 說明文字
    ap.add_argument("--no-opp-hsl", dest="opp_hsl", action="store_false",  # 對手評分趟不用 18 個剖面
                    help="with the opponent pass: do not score the opponents' moves under the 18 profiles as well (section 4M M14; the shards "  # 說明文字（section 4M M14）
                         "then cannot feed the ct6 trunk models' opponent-HSL channel)")  # 說明文字的後半
    a = ap.parse_args(argv)  # 解析參數
    if a.set_models and not a.set_base:  # 集合模型需要配對基準
        ap.error("--set-models needs --set-base, the paired base they were trained on")  # 沒給就停止
    if a.fused_model and a.set_models:  # 兩種決策模型只能選一種
        ap.error("--fused-model and --set-models exclude each other (one decision model)")  # 同時給就停止
    if a.nshards < 1:  # 分片數至少為 1
        ap.error("--nshards must be >= 1")  # 不合理就停止
    if a.shard is not None and (a.merge or not 0 <= a.shard < a.nshards):  # 分片編號超出範圍，或與 --merge 同時使用
        ap.error("--shard needs 0 <= shard < --nshards and excludes --merge")  # 停止
    if a.shard is None and not a.merge and a.nshards != 1:  # 多個分片卻沒有指定分片或合併
        ap.error("--nshards > 1 needs --shard i (one process per shard) or --merge")  # 停止
    out = Path(a.out)  # 提交檔路徑
    report_path = Path(a.report) if a.report else out.with_name(out.stem + "_report.json")  # 報告路徑
    t0 = time.time()  # 計時起點

    try:  # 讀入測試題目
        questions, input_warnings = load_questions(a.inputs, a.lenient_inputs)  # 依參數選嚴格或寬鬆的讀法
    except ValueError as e:  # 輸入格式錯誤
        print(e, file=sys.stderr)  # 印出原因
        return EXIT_INPUTS  # 結束碼 2，什麼都不產出
    if input_warnings:  # 有寬鬆讀入的題目
        print(f"WARNING: {len(input_warnings)} questions read leniently: " + "; ".join(input_warnings[:5])  # 列出前 5 條警告
              + (f" (and {len(input_warnings) - 5} more)" if len(input_warnings) > 5 else ""), file=sys.stderr, flush=True)  # 其餘只報數量
    tasks, owner, dropped, passes = build_tasks(questions)  # 可用棋局的任務清單
    sha = questions_sha(a.inputs, questions)  # 題目雜湊，各分片必須一致
    hd_counts = handicap_counts(questions)  # 各 HD 值的局數
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, "  # 回報題目數、局數與可用局數
          f"{len(dropped)} dropped, {passes} passes; handicap HD[] " + " ".join(f"{k}:{v}" for k, v in hd_counts.items()), flush=True)  # 以及捨棄局數、pass 數與各 HD 值的局數

    if not a.merge:  # 評分步驟（單 GPU，或多 GPU 的其中一個分片）
        shard = 0 if a.shard is None else a.shard  # 單 GPU 執行視為 1 份中的分片 0
        mine = tasks[shard::a.nshards]  # 本分片負責的任務
        try:  # 讀取 L1b 等級頭的側檔
            l1b_models = l1b_models_meta(a.l1b_heads)  # 記錄要用的等級頭
        except ValueError as e:  # 缺少側檔
            print(e, file=sys.stderr)  # 印出原因
            return EXIT_INPUTS  # 結束碼 2
        print(f"scoring shard {shard} of {a.nshards}: {len(mine)} games"  # 回報本分片的局數
              + (f", L1b heads {[m['name'] for m in l1b_models]}" + ((" on the target's and the opponents' moves" + (" (the opponents' under the 18 profiles too)" if a.opp_hsl else ""))  # 以及使用的等級頭與是否評對手
                                                                     if a.l1b_opponent else "") if l1b_models else ""), flush=True)  # 沒有等級頭時不加說明
        per_task, moves, n_pos, wall, opp = score_tasks(mine, a.workers, l1b_heads=a.l1b_heads, opponent=a.l1b_opponent, opponent_hsl=a.opp_hsl)  # 在這張 GPU 上評分
        path = shard_path(out, shard, a.nshards)  # 分片檔路徑
        out.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        save_shard(path, shard, a.nshards, mine, per_task, moves, n_pos, wall, a.inputs, sha, len(tasks), l1b_models, opp)  # 寫出分片檔
        print(f"wrote {path}: {len(mine)} games, {n_pos} positions, {wall:.0f} s"  # 回報分片檔、局數、局面數與秒數
              + (f"; opponents {opp[1]} positions, {opp[2]:.0f} s" if opp else ""), flush=True)  # 以及對手評分趟的局面數與秒數
        if a.shard is not None:  # 多 GPU 執行中的一個分片
            return EXIT_OK  # 多 GPU 執行的其中一個分片；合併另外執行

    set_probs, set_info, set_models, set_warnings = None, {"set_model_fallback_questions": [], "rungs": {}}, [], []  # 集合模型的結果，預設為空
    fused_doc, fused_info, l1b_models = None, {"fused_fallback_questions": []}, []  # 融合模型的結果，預設為空
    try:  # 合併：讀入分片與模型；任何不一致都以結束碼 2 結束
        per_task, shard_metas, moves = load_shards(out, a.nshards, len(tasks), sha, with_moves=bool(a.set_models))  # Human SL 總和（集合模型另需逐手 log p）
        saved, rank_model = load_rank_model(a.model)  # L3/L4 模型：預設的決策模型，也是融合模型的退路
        l1b_models, l1b_per_task, _, opp_per_task = load_shards_l1b(out, a.nshards)  # 分片中的 L1b 模型與分數
        if a.set_models:  # 以集合模型決策
            set_models, set_warnings = load_set_models(a.set_models, a.set_device)  # 載入集合模型
            base_doc, set_base = load_rank_model(a.set_base)  # 載入配對基準
            set_warnings += check_pairing(set_models, base_doc, a.set_base)  # 核對配對並收集警告
            set_probs, set_info = set_model_probs(questions, owner, per_task, moves, set_base, set_models)  # 集合模型的每題機率
        if a.fused_model:  # 以融合 MLP 決策
            fused_doc, fused_mlp = load_fused_model(a.fused_model)  # 載入融合文件
            order = pairing_order(fused_doc, l1b_models)  # 文件中各模型對應到分片欄位的順序
            if fused_doc["feature_set"] == "fused_opp_diff_hd" and hd_counts.get("absent", 0) and not a.allow_missing_hd:  # _hd 文件遇到沒有 HD[] 的局
                raise ValueError(f"{a.fused_model} takes the handicap (HD) features but {hd_counts['absent']} of the games carry no HD[]: decide "  # 預設拒絕
                                 "with the fused_opp_diff document instead, or pass --allow-missing-hd to treat them as unknown (zeros)")  # 錯誤訊息的後半
            set_probs, fused_info = fused_probs(questions, owner, per_task, l1b_per_task, fused_mlp, order, rank_model,  # 融合 MLP 的每題機率
                                                opp_per_task, fused_doc["opponent"], fused_doc["feature_set"], {k: (sgf, c) for k, sgf, c in tasks})  # 對手分數、特徵集與各局棋譜
    except (FileNotFoundError, ValueError) as e:  # 分片或模型有問題
        print(e, file=sys.stderr)  # 印出原因
        return EXIT_INPUTS  # 結束碼 2
    for w in set_warnings:  # 集合模型的警告
        print(f"WARNING: {w}", file=sys.stderr, flush=True)  # 輸出到 stderr
    rows, probs, fallback = predict_questions(questions, owner, per_task, rank_model, set_probs)  # 每題的決策
    write_submission(rows, out)  # 寫出提交檔
    expect_prefix = {p: n for p, n in (("pub_", a.expect_public), ("priv_", a.expect_private)) if n > 0}  # Public／Private 題數的期望值
    problems = validate_submission(out, [q["question_id"] for q in questions], a.expect_rows or None, expect_prefix or None)  # 驗證提交檔
    (pd.DataFrame(probs, columns=[f"p_{lv}" for lv in LEVELS]).assign(question_id=[q["question_id"] for q in questions])  # 每題的機率表，附上題號
        .to_csv(out.with_name(out.stem + "_proba.csv"), index=False))  # 寫成 <out stem>_proba.csv
    code = (EXIT_INVALID if problems else EXIT_FALLBACK if fallback and not a.allow_fallback  # 結束碼：驗證失敗優先，其次是未接受的退路題
            else EXIT_INPUT_WARNINGS if input_warnings and not a.allow_input_warnings else EXIT_OK)  # 再其次是未接受的輸入警告，否則正常
    report = {"inputs": a.inputs, "model": a.model, "model_name": saved["name"], "mode": "merge" if a.merge else "single",  # 報告：輸入、模型與執行模式
              "nshards": a.nshards, "shards": shard_metas, "questions": len(questions),  # 分片數、各分片 meta 與題數
              "games_declared": int(sum(q["declared_games"] for q in questions)),  # 宣告的局數
              "games_present": int(sum(len(q["games"]) for q in questions)),  # 實際讀到的局數
              "games_used": len(tasks), "games_dropped": dropped, "passes": passes, "handicap": hd_counts,  # 使用與捨棄的局、pass 數與 HD 計數
              "positions": int(sum(m["positions"] for m in shard_metas)),  # 評分的局面總數
              "scoring_wall_s": float(sum(m["scoring_wall_s"] for m in shard_metas)), "total_wall_s": time.time() - t0,  # 評分秒數與總耗時
              "l1b_opponent": bool(l1b_models) and all(m.get("l1b_opponent") for m in shard_metas),  # 是否每個分片都評了對手
              "opp_positions": int(sum(m.get("opp_positions", 0) for m in shard_metas)),  # 對手評分趟的局面數
              "opp_scoring_wall_s": float(sum(m.get("opp_scoring_wall_s", 0.0) for m in shard_metas)),  # 對手評分趟的秒數
              "expectations": {"rows": a.expect_rows, "pub_": a.expect_public, "priv_": a.expect_private},  # 題數的期望值
              "fallback_questions": fallback, "allow_fallback": a.allow_fallback,  # 退路題與設定
              "lenient_inputs": a.lenient_inputs, "input_warnings": input_warnings, "allow_input_warnings": a.allow_input_warnings,  # 寬鬆讀入的設定與警告
              "decision_model": fused_doc["feature_set"] if a.fused_model else "set_models" if a.set_models else "rank_model",  # 實際用來決策的模型
              "set_models": [{"path": path, **set_model_summary(meta)} for _, meta, path in set_models], "set_base": a.set_base,  # 集合模型摘要與配對基準
              "set_rungs": set_info["rungs"], "set_model_warnings": set_warnings,  # 各階的模型數與警告
              "set_model_fallback_questions": set_info["set_model_fallback_questions"],  # 集合模型退回基準的題目
              "l1b_models": l1b_models,  # 分片中的 L1b 模型
              "fused_model": ({"path": a.fused_model, **{k: fused_doc.get(k) for k in ("name", "opponent", "feature_set", "file_sha256", "train_folds", "per_player",  # 融合文件的重點欄位
                                                                                         "questions_sha256", "l1b_models", "git_head")},  # 欄位清單的後半
                               "n_models": len(fused_doc["l1b_models"])} if fused_doc else None),  # 文件的 L1b 模型數；沒用融合模型時為 None
              "fused_fallback_questions": fused_info["fused_fallback_questions"],  # 融合模型退回 rank_model 的題目
              "predicted_counts": dict(Counter(r for _, r in rows)), "problems": problems, "exit_code": code}  # 各等級的預測數、問題與結束碼
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))  # 寫出報告 JSON
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}", flush=True)  # 回報輸出檔與驗證結果
    if fused_info["fused_fallback_questions"]:  # 有題目退回 rank_model
        print(f"WARNING: {len(fused_info['fused_fallback_questions'])} questions got {Path(a.model).name}'s probabilities because the "  # 警告哪些題用了 --model 的機率
              f"fused model's were not finite: {fused_info['fused_fallback_questions'][:5]}", file=sys.stderr, flush=True)  # 列出前 5 題
    if fallback:  # 有沒有可用棋局的題目
        print(f"WARNING: {len(fallback)} questions had no usable game and were set to {FALLBACK}: {fallback[:5]}"  # 警告這些題被填入 FALLBACK
              + ("" if a.allow_fallback else f" (exit {EXIT_FALLBACK}; pass --allow-fallback to accept)"), file=sys.stderr, flush=True)  # 並提示 --allow-fallback
    if set_info["set_model_fallback_questions"]:  # 有題目退回基準的機率
        print(f"WARNING: {len(set_info['set_model_fallback_questions'])} questions got the base's probabilities because the set "  # 警告哪些題用了基準的機率
              f"models' were not finite: {set_info['set_model_fallback_questions'][:5]}", file=sys.stderr, flush=True)  # 列出前 5 題
    if input_warnings and code == EXIT_INPUT_WARNINGS:  # 寬鬆讀入且未接受警告
        print(f"WARNING: {len(input_warnings)} questions were read leniently (exit {EXIT_INPUT_WARNINGS}; pass "  # 警告寬鬆讀入的題數
              f"--allow-input-warnings to accept)", file=sys.stderr, flush=True)  # 並提示 --allow-input-warnings
    return code  # 回傳結束碼


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
