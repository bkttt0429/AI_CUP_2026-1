"""One-shot fold-0 evaluation of the aggregator-ladder rung chosen from the folds 1-4 CV.

Fold 0 is the held-out evaluation fold: its result is spent once, on the configuration chosen from
the CV summary, and never feeds back into a choice. run() therefore raises Refused (the CLI's exit
2) before reading any data when the output or the canonical results/ladder/fold0_eval.json
(CANONICAL_OUT) exists already, when the CV summary the rung was chosen from (chosen_from) is
missing, when that summary has no CV results for the requested configuration
(ladder._label(rung, per_player, counts)), or, for a set rung, when the seeds differ from that CV
label's seeds. The output is written once, atomically, and never over an existing file (write_once).

Protocol (plan 2026-10-19-l3-aggregator-ladder-ctm, Task 8): the rung is retrained on folds 1-4 with
the training questions of split v = 0 (ladder.training_questions: one RandomState(42), folds 1-4
ascending -- fit_rank_model's final model) and scored on fold 0's pilot_questions. A static rung is
fitted once (deterministic; seeds null). A set rung trains once per seed on the 4-fold cross-fitted
base (ladder.split_base with v = 0, computed once and shared by the seeds), and the per-question
score is the mean over seeds -- the statistic of the CV summary. Only the rung's own configuration (its
TrainCfg.pick tick) is scored: the report-only EXTRAS of the CV (more ticks, another tick) are not computed on fold 0, so
that nothing can be picked on it after the fact. Reported: the headline
balanced_mean_over_n with its player-bootstrap CI (evidence.bootstrap_ci_mean_over_sizes, n = 5000)
and two paired differences (qeval.paired_diff), both through paired(), which refuses frames whose
question ids differ in content or order:
- vs_base: the same retrain's base -- for a set rung softmax of split_base's validation base
  (Δ ≡ 0); for a static rung r2_mlp24 refitted on the same training questions (same per_player and
  counts), so the difference is exactly 0 for r2_mlp24 itself;
- vs_production: results/models/rank_model.json (rankmodel.from_state on qfeatures.build_matrix
  features), trained with 8 questions per player (PRODUCTION_NOTE).
"""
import hashlib  # 記錄輸入檔的 sha256
import json  # 讀取 CV 摘要與當時的提交模型
import os  # 原子寫入（fsync、硬連結）
import tempfile  # 基準 logits 的暫存快取資料夾
import time  # 記錄每個種子的訓練時間
from datetime import datetime  # 記錄結果的建立時間
from pathlib import Path  # 路徑處理

import numpy as np  # 數值運算
import pandas as pd  # 讀取分折表

from aicup.evidence import bootstrap_ci_mean_over_sizes  # 標題分數的玩家 bootstrap 信賴區間
from aicup.ladder import (BASE_RUNG, CV_FOLDS, DEFAULT_PER_PLAYER, FOLDS_CSV, KEY_COLS, N_BOOT, ROOT, RUNGS, SET_RUNGS,  # 沿用 CV 的常數
                          _check_players, _frame, _label, _plain, _same_keys, _set_split, _static_split, environment,  # 沿用 CV 的訓練與評分步驟
                          load_vectors, split_base, to_json, training_questions, validation_questions)  # 與 CV 相同的資料與題目
from aicup.qeval import SIZES, balanced, balanced_mean_over_n, nll_brier, paired_diff  # 題目層評估
from aicup.qfeatures import FEATURE_NAMES, SUM_COLS, build_matrix  # 當時提交模型（rank_model.json）的特徵
from aicup.rankmodel import N_CLASSES, _softmax, from_state  # 還原當時的提交模型與 softmax
from aicup.scoring import TRAIN_DIR, fold_players, player_games  # 訓練資料與各折玩家

EVAL_FOLD = 0  # 留出的評估折
ALL_FOLDS = (EVAL_FOLD, *CV_FOLDS)  # 第 0–4 折
# 標準結果：不論 `out` 是什麼，只要它存在 check_request 就拒絕（測試會把它移開）
CANONICAL_OUT = ROOT / "results" / "ladder" / "fold0_eval.json"  # 第 0 折評估的標準結果檔
OUT = CANONICAL_OUT  # 預設的輸出
CHOSEN_FROM = ROOT / "results" / "ladder" / "cv_summary.json"  # 選定梯級所依據的 CV 摘要
VECTORS = ROOT / "results" / "ladder" / "vectors"  # 各折每局向量的資料夾
RANK_MODEL = ROOT / "results" / "models" / "rank_model.json"  # 當時的提交模型（production model），當作對照
PCOLS = [f"p{k}" for k in range(N_CLASSES)]  # 機率欄名 p0..p9
PRODUCTION_NOTE = (f"The production model (rank_model.json) was trained with 8 questions per player and this retrain with "  # 寫進結果的說明：對當時提交模型的差混有訓練題數的效應
                   f"per_player of them, so vs_production mixes the rung with the training size; the training size "  # 說明續行
                   f"alone is the CV comparison '{BASE_RUNG}@pp8 vs {BASE_RUNG}' of chosen_from "  # 訓練題數單獨的效應見 CV 比較
                   f"(cv_pp8_vs_base, when present).")  # 說明結束


class Refused(Exception):  # 不允許評估第 0 折時的例外
    """Fold 0 must not be evaluated for this request (the CLI exits with 2); raised before any data is read."""


def repo_path(path):  # 記錄用的路徑
    """The path relative to the repository root when inside it, else absolute."""
    p = Path(path).resolve()  # 絕對路徑
    try:  # 嘗試轉成相對於儲存庫的路徑
        return str(p.relative_to(ROOT))  # 儲存庫內的相對路徑
    except ValueError:  # 不在儲存庫內
        return str(p)  # 保留絕對路徑


def _sha256(raw):  # 位元組內容的 sha256
    return hashlib.sha256(raw).hexdigest()  # 十六進位字串


def check_request(rung, out, chosen_from, per_player=DEFAULT_PER_PLAYER, counts=True, seeds=(0, 1, 2)):  # 確認第 0 折可以評估
    """Refused unless fold 0 may be evaluated: neither `out` nor CANONICAL_OUT may exist (checked
    first), `chosen_from` must be a CV summary with results for ladder._label(rung, per_player,
    counts) and, for a set rung, `seeds` must be that CV label's seeds (in any order; a static rung
    is fitted once and ignores them). Returns the provenance of the choice: chosen_from
    (repo-relative), its sha256, the CV label, the CV entry of that label and the summary's
    r2_mlp24@pp8 vs r2_mlp24 comparison (None when absent)."""
    out, chosen_from = Path(out), Path(chosen_from)  # 轉成 Path
    if out.exists():  # 輸出已存在
        raise Refused(f"{out} exists: fold 0 has been evaluated already, and it is evaluated only once")  # 第 0 折只評一次
    if CANONICAL_OUT.exists():  # 標準結果已存在
        raise Refused(f"{CANONICAL_OUT} exists: fold 0 has been evaluated already, and it is evaluated only once "  # 不論 --out 為何都拒絕
                      f"(this canonical result is checked whatever --out is)")  # 訊息續行
    if rung not in RUNGS:  # 不認得的梯級
        raise Refused(f"unknown rung {rung!r}; known: {list(RUNGS)}")  # 拒絕
    if not chosen_from.is_file():  # 沒有 CV 摘要
        raise Refused(f"{chosen_from} not found: fold 0 is evaluated only for a rung chosen from a CV summary "  # 只評估由 CV 選出的梯級
                      f"(experiments/ladder_cv.py)")  # 訊息續行
    raw = chosen_from.read_bytes()  # 摘要的原始內容（也用來算 sha256）
    try:  # 解析摘要
        summary = json.loads(raw)  # 讀成字典
    except ValueError as e:  # 不是合法的 JSON
        raise Refused(f"{chosen_from} is not a CV summary ({e})") from e  # 拒絕
    rungs = summary.get("rungs") if isinstance(summary, dict) else None  # 摘要中各標籤的結果
    label = _label(rung, per_player, counts)  # 這次設定對應的 CV 標籤
    if not isinstance(rungs, dict) or label not in rungs:  # 摘要裡沒有這個設定
        have = sorted(rungs) if isinstance(rungs, dict) else []  # 摘要裡有的標籤
        raise Refused(f"{chosen_from} has no CV results for {label!r} (it has {have}): fold 0 is evaluated only for a "  # 只評估在 CV 中跑過的設定
                      f"configuration chosen from the CV; give --per-player / --no-counts as in that CV run")  # 提示要用與 CV 相同的參數
    cv = rungs[label] if isinstance(rungs[label], dict) else {}  # 該標籤的 CV 條目
    if rung in SET_RUNGS:  # 標題分數是種子平均：要與選定它的 CV 統計量用相同的種子
        cv_seeds = cv.get("seeds")  # CV 用過的種子
        if not isinstance(cv_seeds, list) or sorted(int(s) for s in seeds) != sorted(int(s) for s in cv_seeds):  # 種子不一致
            raise Refused(f"--seeds {sorted(int(s) for s in seeds)} differ from the seeds {cv_seeds} of {label!r} in "  # 拒絕
                          f"{chosen_from}: a set rung is evaluated on fold 0 with the seeds of its CV results")  # 訊息續行
    pp8 = (summary.get("comparisons") or {}).get(f"{BASE_RUNG}@pp8 vs {BASE_RUNG}")  # 訓練題數效應的 CV 比較
    return {"chosen_from": repo_path(chosen_from), "cv_summary_sha256": _sha256(raw), "cv_label": label,  # 選擇的來源紀錄
            "cv": {k: cv.get(k) for k in ("balanced_mean_over_n", "per_fold", "seeds")},  # 選定時看到的 CV 分數
            "cv_pp8_vs_base": None if pp8 is None else {k: pp8.get(k) for k in ("diff", "ci95")}}  # 訓練題數單獨的效應


# ---------------------------------------------------------------- 輸入

def load_eval_players(folds=ALL_FOLDS, train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):  # 讀取含第 0 折的各折玩家
    """{fold: scoring.fold_players selection}, fold 0 included (ladder.load_players refuses fold 0 so
    that the CV never reads it); reads the training CSVs, ~30 s."""
    games, folds_df = player_games(train_dir), pd.read_csv(folds_csv)  # 訓練棋局清單與分折表
    return {f: fold_players(games, folds_df, f) for f in folds}  # 每折選出的玩家


def load_inputs(vectors_dir, folds=ALL_FOLDS, players=None):  # 讀取向量與玩家並互相核對
    """(vec, fold_of, players) of the folds: the vectors first (a missing file fails before the ~30 s of
    players), then the players (load_eval_players when None), checked against each other."""
    vec, fold_of = load_vectors(vectors_dir, folds)  # 先讀向量，缺檔時盡早失敗
    players = load_eval_players(folds) if players is None else players  # 再讀玩家
    _check_players(players, vec, fold_of, folds)  # 確認向量恰好是這些玩家的棋局
    return vec, fold_of, players  # 評估所需的輸入


def check_production(doc, where="the production model"):  # 檢查當時的提交模型能否當對照
    """The production model must use qfeatures.FEATURE_NAMES and must not have seen fold 0."""
    if doc.get("features") != FEATURE_NAMES:  # 特徵定義不同
        raise ValueError(f"{where}: its features are not qfeatures.FEATURE_NAMES")  # 拒絕
    if sorted(doc.get("train_folds") or []) != list(CV_FOLDS):  # 訓練折不是第 1–4 折
        raise ValueError(f"{where}: trained on folds {doc.get('train_folds')}, not {list(CV_FOLDS)}")  # 看過第 0 折就不能當對照


def _games(vec):  # 由向量組出棋局表
    """qfeatures.build_matrix's games frame from the vectors (their sums and n_moves are the score CSVs' values)."""
    return pd.DataFrame({"game_key": vec["game_key"], **{c: vec["sums"][:, i] for i, c in enumerate(SUM_COLS)},  # 棋局鍵與 18 個剖面的總和
                         "n_moves": vec["n_moves"]})  # 每局手數


def production_frame(doc, vec, val_q):  # 當時提交模型在第 0 折題目上的結果
    """Frame (ladder._frame) of the production model `doc` (rank_model.json's content) on the questions:
    rankmodel.from_state(doc["state"]) on qfeatures.build_matrix features, as fit_rank_model scored it."""
    check_production(doc)  # 先確認模型合格
    X, _, _ = build_matrix(_games(vec), val_q)  # 24 維題目特徵
    return _frame(val_q, EVAL_FOLD, {"": from_state(doc["state"]).predict_proba(X)})  # 預測後整理成與 CV 相同格式的資料表


# ---------------------------------------------------------------- 評估

def paired(a, b, what, n_boot=N_BOOT):  # 兩個結果的成對比較
    """qeval.paired_diff of a's scores minus b's (frames with KEY_COLS and score) with its n_boot. The
    frames must hold identical questions in the same order -- question_id equal in content and order,
    and so every key column -- else ValueError (ladder._same_keys)."""
    _same_keys(a, b, what)  # 題目必須完全相同且同序
    y, n = a["y"].to_numpy(dtype=np.int64), a["n_games"].to_numpy(dtype=np.int64)  # 標籤與局數
    d = paired_diff(a["score"].to_numpy(dtype=float), b["score"].to_numpy(dtype=float), y,  # 逐題分數的成對差
                    a["player_id"].to_numpy(dtype=str), n, n=n_boot)  # 以玩家為單位 bootstrap
    return {**d, "n_boot": n_boot}  # 附上重抽次數


def _block(frames):  # 一個模型的指標
    """(metrics, frame) of one model's frames on the same questions, one per seed (one for a static
    model): per question the mean score over seeds, then balanced_mean_over_n and balanced_by_n, and
    the class-balanced NLL / Brier of the seed-mean per-question values (as ladder._rung_summary)."""
    for i, f in enumerate(frames[1:], 1):  # 逐一比對其他種子的資料表
        _same_keys(frames[0], f, f"frame {i}")  # 題目必須相同
    keys = frames[0][KEY_COLS]  # 題目鍵欄
    y, n = keys["y"].to_numpy(dtype=np.int64), keys["n_games"].to_numpy(dtype=np.int64)  # 標籤與局數
    s = np.mean([f["score"].to_numpy(dtype=float) for f in frames], axis=0)  # 逐題分數的種子平均
    nll, brier = (np.mean(v, axis=0) for v in zip(*(nll_brier(f[PCOLS].to_numpy(dtype=float), y) for f in frames)))  # 逐題 NLL 與 Brier 的種子平均
    metrics = {"balanced_mean_over_n": balanced_mean_over_n(s, y, n),  # 標題分數
               "balanced_by_n": {m: balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},  # 各題目大小的分數
               "nll_balanced": balanced(nll, y), "brier": balanced(brier, y)}  # 類別平衡 NLL 與 Brier
    return metrics, keys.assign(score=s)  # 指標與帶種子平均分數的資料表


def evaluate_fold0(rung, vec, players, production, seeds=(0, 1, 2), per_player=DEFAULT_PER_PLAYER, counts=True,  # 在第 0 折上評估一個梯級
                   device="cuda", n_boot=N_BOOT):  # 裝置與 bootstrap 次數
    """Retrain `rung` on folds 1-4 and evaluate it on fold 0 (see the module docstring). vec / players:
    the vectors and fold_players of folds 0-4; production: rank_model.json's content. Returns the
    result in its JSON form (string keys): settings, headline (with ci95, n_boot and, for a set rung,
    per_seed), base, production, vs_base, vs_production, the production note and diagnostics."""
    if rung not in RUNGS:  # 不認得的梯級
        raise KeyError(f"unknown rung {rung!r}; known: {list(RUNGS)}")  # 拒絕
    is_set = rung in SET_RUNGS  # 是否為集合梯級
    check_production(production)  # 確認當時的提交模型合格
    train_q, val_q = training_questions(players, EVAL_FOLD, CV_FOLDS, per_player), validation_questions(players, EVAL_FOLD)  # 第 1–4 折訓練題與第 0 折驗證題
    if is_set:  # 集合梯級：每個種子訓練一次
        seeds = [int(s) for s in seeds]  # 種子轉成整數
        if not seeds or len(set(seeds)) != len(seeds):  # 沒有種子或有重複
            raise ValueError(f"a set rung needs distinct seeds, got {seeds}")  # 拒絕
        runs, diag = {}, {"train_loss_last": {}}  # 各種子的結果與診斷
        with tempfile.TemporaryDirectory(prefix="ladder_fold0_") as cache:  # 基準只擬合一次，供所有種子使用
            _, base_val, diag["base"] = split_base(EVAL_FOLD, vec, train_q, val_q, counts, per_player, cache)  # 基準算一次並快取（訓練列為交叉擬合）；這裡只取驗證基準
            for seed in seeds:  # 逐個種子
                t0 = time.time()  # 開始計時
                # 第 0 折只報告選定的設定（該梯級 TrainCfg.pick 的 tick）；EXTRAS 只放在
                # CV 摘要裡，所以事後無法在第 0 折上挑選任何東西
                probas, info = _set_split(rung, EVAL_FOLD, vec, train_q, val_q, counts, per_player, seed, device, cache,  # 訓練並預測
                                          extras=False)  # 不計算 EXTRAS
                runs[seed] = _frame(val_q, EVAL_FOLD, probas)  # 該種子的逐題結果
                diag["train_loss_last"][seed] = info["train_loss"][-1] if info["train_loss"] else None  # 最後一個 epoch 的訓練損失
                print(f"{rung} seed {seed} trained ({time.time() - t0:.0f} s)", flush=True)  # 進度訊息
        frames, base = list(runs.values()), _frame(val_q, EVAL_FOLD, {"": _softmax(base_val)})  # 各種子的結果與 Δ ≡ 0 的基準
        base_def = f"softmax of the 4-fold cross-fitted {BASE_RUNG} validation base of the same training questions (Δ ≡ 0)"  # 基準的定義，寫進結果
    else:  # 靜態梯級：確定性擬合一次
        seeds = None  # 靜態梯級沒有種子
        probas, diag = _static_split(rung, vec, train_q, val_q, counts)  # 擬合並預測
        frames = [_frame(val_q, EVAL_FOLD, probas)]  # 唯一的結果
        base = frames[0] if rung == BASE_RUNG else _frame(val_q, EVAL_FOLD,  # 本身就是基準時直接沿用
                                                          _static_split(BASE_RUNG, vec, train_q, val_q, counts)[0])  # 否則在同一批訓練題上重新擬合 r2_mlp24
        base_def = f"{BASE_RUNG} refitted on the same training questions (folds 1-4, same per_player and counts)"  # 基準的定義，寫進結果

    head, rung_frame = _block(frames)  # 梯級的標題指標
    base_m, base_frame = _block([base])  # 基準的指標
    prod_m, prod_frame = _block([production_frame(production, vec, val_q)])  # 當時提交模型的指標
    y, n = rung_frame["y"].to_numpy(dtype=np.int64), rung_frame["n_games"].to_numpy(dtype=np.int64)  # 標籤與局數
    pid = rung_frame["player_id"].to_numpy(dtype=str)  # 玩家 id（bootstrap 的單位）
    head["ci95"] = list(bootstrap_ci_mean_over_sizes(rung_frame["score"].to_numpy(dtype=float), pid, y, n, n=n_boot))  # 標題分數的信賴區間
    head["n_boot"] = n_boot  # 記錄重抽次數
    vs_base, vs_prod = (paired(rung_frame, base_frame, f"{rung} vs base", n_boot),  # 對基準的成對差
                        paired(rung_frame, prod_frame, f"{rung} vs production", n_boot))  # 對當時提交模型的成對差
    if is_set:  # 集合梯級另報各種子
        head["per_seed"] = {s: balanced_mean_over_n(f["score"], y, n) for s, f in runs.items()}  # 各種子的標題分數
        head["seed_sd"] = float(np.std(list(head["per_seed"].values()), ddof=1)) if len(seeds) > 1 else None  # 種子間的標準差
        for c, ref in ((vs_base, base_m), (vs_prod, prod_m)):  # 兩個比較
            c["per_seed"] = {s: v - ref["balanced_mean_over_n"] for s, v in head["per_seed"].items()}  # 各種子對參考的點差
    return _plain({"rung": rung, "kind": "set" if is_set else "static", "seeds": seeds, "per_player": int(per_player),  # 設定
                   "counts": bool(counts), "eval_fold": EVAL_FOLD, "train_folds": list(CV_FOLDS),  # 計數設定與折
                   "n_train_questions": len(train_q), "n_questions": len(val_q),  # 題數
                   "n_players": int(val_q["player_id"].nunique()),  # 評估的玩家數
                   "headline": head, "base": {**base_m, "definition": base_def}, "production": prod_m,  # 標題分數、基準與當時提交模型的指標
                   "vs_base": vs_base, "vs_production": vs_prod, "note_production": PRODUCTION_NOTE, "diagnostics": diag})  # 成對比較、說明與診斷


# ---------------------------------------------------------------- 輸出

def write_once(path, obj):  # 只寫一次的原子寫入
    """Write strict JSON (ladder.to_json) to `path` atomically and never over an existing file: the text
    goes to a temporary file beside it, hard-linked to `path` (FileExistsError when `path` exists,
    e.g. written by another run meanwhile) and then removed."""
    path = Path(path)  # 轉成 Path
    path.parent.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # 同資料夾、以行程 id 區分的暫存檔
    try:  # 確保暫存檔最後被刪除
        with open(tmp, "x") as f:  # 只允許新建
            f.write(to_json(obj, indent=1) + "\n")  # 寫入嚴格 JSON
            f.flush()  # 先把 Python 緩衝區的內容交給作業系統
            os.fsync(f.fileno())  # 確實寫入磁碟
        os.link(tmp, path)  # 硬連結到目標；目標已存在時失敗，不會覆蓋
    finally:  # 不論成功與否
        tmp.unlink(missing_ok=True)  # 刪除暫存檔


def run(rung, out=OUT, chosen_from=CHOSEN_FROM, vectors_dir=VECTORS, rank_model=RANK_MODEL, seeds=(0, 1, 2),  # 第 0 折評估的進入點
        per_player=DEFAULT_PER_PLAYER, counts=True, device="cuda", n_boot=N_BOOT, players=None):  # 其餘設定
    """The one fold-0 evaluation: check_request (Refused before anything is read), then the inputs (a
    missing file fails before the players are loaded), evaluate_fold0, provenance (created; the code
    that runs -- ladder.environment: git_head, git_dirty, code_sha256, code_files, torch_version,
    device_name; chosen_from with its sha256; the production model's path and sha256) and write_once
    to `out`. players: {fold: fold_players frame} of folds 0-4, loaded when None. Returns the written result."""
    chosen = check_request(rung, out, chosen_from, per_player, counts, seeds)  # 先檢查是否允許，之後才讀資料
    missing = [str(p) for p in [*(Path(vectors_dir) / f"fold{f}.npz" for f in ALL_FOLDS), Path(rank_model)] if not p.exists()]  # 缺少的輸入檔
    if missing:  # 有缺檔
        raise FileNotFoundError(f"inputs not found: {', '.join(missing)} (vectors: experiments/ladder_build_vectors.py)")  # 提示產生向量的腳本
    env = environment(device if rung in SET_RUNGS else "cpu")  # 執行中的程式，在數小時的訓練之前先讀取
    raw = Path(rank_model).read_bytes()  # 當時提交模型的原始內容（也用來算 sha256）
    production = json.loads(raw)  # 解析模型的 JSON 內容
    check_production(production, str(rank_model))  # 確認模型合格
    vec, _, players = load_inputs(vectors_dir, ALL_FOLDS, players)  # 讀向量與玩家
    res = evaluate_fold0(rung, vec, players, production, seeds, per_player, counts, device, n_boot)  # 重訓並評估
    result = {"created": datetime.now().isoformat(timespec="seconds"), **env, **res, **chosen,  # 建立時間、程式版本、結果與選擇的來源
              "production_model": {"path": repo_path(rank_model), "sha256": _sha256(raw)}}  # 當時提交模型的路徑與 sha256
    write_once(out, result)  # 只寫一次
    return result  # 寫出的結果
