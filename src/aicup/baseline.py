"""The official AI CUP 2026 rank baseline (Tutorial rank-prediction.ipynb) under the project protocol.

The parser and the network are the Tutorial's own files, imported by path and never edited,
so features and model are the official ones. Only the data split is ours (METHOD_NOTES §2.1):
training and validation players come from folds 1-4, whole players at a time, and fold 0 is
used for evaluation only.
"""
import importlib.util  # 依檔案路徑載入 Tutorial 的模組
import sys  # 登記載入的模組
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列

from aicup.evidence import bootstrap_ci, bootstrap_ci_mean_over_sizes  # 以玩家重抽樣的信賴區間
from aicup.metric import LEVELS, expected_score_decision, score  # 等級清單、期望得分決策與官方計分

ROOT = Path(__file__).resolve().parents[2]  # 儲存庫根目錄
TUTORIAL_DIR = ROOT / "repo" / "AICup-2026-Tutorial"  # 官方 Tutorial 的本地副本
_TUTORIAL = {}  # 每個 Tutorial 路徑只載入一次的快取
_PARSER = None  # 共用的 SGF 解析器，第一次使用時才建立


def load_tutorial(tutorial_dir=TUTORIAL_DIR):  # 載入官方的解析器與網路類別
    """(SGFParseRankPrediction, GoRankResNet) from the Tutorial checkout, loaded once per process."""
    key = str(Path(tutorial_dir).resolve())  # 以絕對路徑作為快取鍵
    if key not in _TUTORIAL:  # 尚未載入過
        mods = []  # 依序收集 utils 與 network 模組
        for name in ("utils", "network"):  # Tutorial 的兩個檔案
            spec = importlib.util.spec_from_file_location(f"aicup_tutorial_{name}", Path(tutorial_dir) / f"{name}.py")  # 依路徑建立模組規格，改名以免與其他模組衝突
            mod = importlib.util.module_from_spec(spec)  # 建立模組物件
            sys.modules[spec.name] = mod  # 先登記模組名稱，依名稱查找的程式才找得到它
            spec.loader.exec_module(mod)  # 執行原檔的內容（不修改）
            mods.append(mod)  # 收進清單
        _TUTORIAL[key] = (mods[0].SGFParseRankPrediction, mods[1].GoRankResNet)  # 記下兩個類別
    return _TUTORIAL[key]  # 回傳快取的類別


def game_features(sgf):  # 以官方解析器抽出一局的特徵
    """The notebook's sgf_to_features: uint8 [500, 19, 19] for one game, None if it does not parse."""
    global _PARSER  # 使用模組層的共用解析器
    if _PARSER is None:  # 第一次呼叫
        _PARSER = load_tutorial()[0]()  # 建立官方解析器
    feat, err = _PARSER.features_from_sgf_content(sgf)  # 解析 SGF 內容
    return None if err is not None else feat  # 解析失敗時回傳 None


def split_players(folds, train_folds=(1, 2, 3, 4), val_games_per_rank=10_000, seed=42):  # 切出訓練與驗證玩家
    """(train_ids, val_ids), both sorted: whole players of `train_folds`, validation >= target games per rank.

    Ranks go in LEVELS order; a rank's players are sorted by id, permuted with one shared
    RandomState(seed) and taken in that order until their games reach `val_games_per_rank`.
    Players of other folds are in neither list.
    """
    pool = folds[folds["fold"].isin(list(train_folds))]  # 只取訓練折的玩家
    rng = np.random.RandomState(seed)  # 所有等級共用的亂數產生器
    val = []  # 驗證玩家
    for rank in [r for r in LEVELS if r in set(pool["rank"])]:  # 依 LEVELS 的順序逐級處理
        cand = pool[pool["rank"] == rank].sort_values("player_id")  # 這一級的玩家依 id 排序
        ids, games = cand["player_id"].to_numpy(), cand["n_games"].to_numpy()  # 玩家 id 與局數
        taken = 0  # 已取的局數
        for i in rng.permutation(len(cand)):  # 依打亂後的順序取玩家
            if taken >= val_games_per_rank:  # 局數已達目標
                break  # 這一級取完
            val.append(ids[i])  # 整位玩家放進驗證集
            taken += int(games[i])  # 累計局數
        if taken >= games.sum():  # 驗證集會拿走這一級全部的局
            raise ValueError(f"rank {rank}: validation would take all {int(games.sum())} training games")  # 不能讓這一級沒有訓練資料
    val_set = set(val)  # 方便查詢
    return sorted(p for p in pool["player_id"] if p not in val_set), sorted(val)  # 其餘玩家為訓練集


def eval_questions(games, folds, questions, fold=0):  # 組出評估用的題目集
    """The question sets of experiments/baseline_eval.py on one fold, carved 1x20 / 2x10 / 4x5 by pilot.pilot_questions.

    'fold{fold}': every player of the fold with >= 5 games, up to 20 of them (scoring.fold_players
    on `games`, the rows of scoring.player_games); 'pilot_fold{fold}': the 200-player pilot set
    of METHOD_NOTES §2.4.1 (pilot.select_players on `questions`, splits/questions_fold{fold}.csv).
    """
    from aicup.pilot import pilot_questions, select_players  # 延後到呼叫時才匯入切題函式：pilot 依賴 KataGo 的 Python 模組
    from aicup.scoring import fold_players  # 每折玩家的局清單；同樣延後匯入（scoring 也會匯入 pilot）

    return {f"fold{fold}": pilot_questions(fold_players(games, folds, fold)),  # 全折玩家的題目
            f"pilot_fold{fold}": pilot_questions(select_players(questions, fold))}  # pilot 200 位玩家的題目


def mean_probs(game_probs, questions):  # 每題平均各局的機率
    """[Q, 10]: per question, the mean softmax over its games that have one (the notebook's vote); NaN if none."""
    rows = []  # 每題一列
    for keys in questions:  # 逐題處理
        ps = [game_probs[k] for k in keys if k in game_probs]  # 有機率的局
        rows.append(np.mean(ps, axis=0) if ps else np.full(len(LEVELS), np.nan))  # 沒有局時填 NaN
    return np.array(rows, dtype=float).reshape(-1, len(LEVELS))  # 沒有題目時仍為 [0, 10]


def decide(probs, rule):  # 由機率決定每題的答案
    """Class per row: 'argmax' (the notebook) or 'expected_score' (metric.expected_score_decision).

    A NaN row (no game of the question parsed) gets class 0, the notebook's LEVELS[0] answer.
    """
    if rule not in ("argmax", "expected_score"):  # 只接受兩種規則
        raise ValueError(f"unknown decision rule {rule!r}")  # 拒絕未知的規則
    out = []  # 每列的答案
    for p in np.asarray(probs, dtype=float):  # 逐列處理
        if np.isnan(p).any():  # 沒有機率的題目
            out.append(0)  # 比照 notebook 回答 LEVELS[0]
        else:  # 有機率
            out.append(int(np.argmax(p)) if rule == "argmax" else expected_score_decision(p))  # 依規則決策
    return np.array(out, dtype=int)  # 整數答案陣列


def report(pred, y, qs, sizes=(5, 10, 20)):  # 彙整官方分數與信賴區間
    """Official scores of question decisions, with the keys of results/rank_model_eval.json plus `by_n` details.

    The headline `balanced_mean_over_n` gets the CI of that same statistic (one player resample
    per replicate shared by every size), as experiments/fit_rank_model.py reports it; `by_n` and
    `balanced_by_n_ci95` hold each size on its own. CIs resample players within each rank.
    """
    pred, y = np.asarray(pred, dtype=int), np.asarray(y, dtype=int)  # 預測與真實等級
    s = np.array([score(p, t) for p, t in zip(pred, y)])  # 每題的官方分數
    players, n = qs["player_id"].to_numpy(), qs["n_games"].to_numpy()  # 每題的玩家與局數
    classes = range(len(LEVELS))  # 10 個等級索引

    def balanced(mask):  # 各級平均分數
        return float(np.mean([s[mask & (y == k)].mean() for k in classes if (mask & (y == k)).any()]))  # 只對有題目的等級取平均

    def summary(mask):  # 完全正確率、差一級以內的比例與各級分數
        return {"exact": float((pred[mask] == y[mask]).mean()),  # 完全正確的比例
                "within1": float((np.abs(pred[mask] - y[mask]) <= 1).mean()),  # 差一級以內的比例
                "per_class": {LEVELS[k]: float(s[mask & (y == k)].mean()) for k in classes if (mask & (y == k)).any()}}  # 各級的平均分數

    by_n = {}  # 各題長的結果
    for m in sizes:  # 逐一題長
        mask = n == m  # 這個題長的題目
        by_n[m] = {"balanced": balanced(mask),  # 各級平均
                   "balanced_ci95": list(bootstrap_ci(s[mask], players[mask], y[mask], balanced=True)),  # 各級平均的 CI
                   "question_mean": float(s[mask].mean()),  # 題目平均
                   "question_mean_ci95": list(bootstrap_ci(s[mask], players[mask], y[mask])),  # 題目平均的 CI
                   "n_questions": int(mask.sum()), **summary(mask)}  # 題數與其他摘要
    conf = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)  # 混淆矩陣
    np.add.at(conf, (y, pred), 1)  # 列為真實等級、欄為預測
    keep = np.isin(n, sizes)  # 只算指定題長的題目
    return {"balanced_by_n": {m: by_n[m]["balanced"] for m in sizes},  # 各題長的各級平均
            "balanced_by_n_ci95": {m: by_n[m]["balanced_ci95"] for m in sizes},  # 各題長的 CI
            "balanced_mean_over_n": float(np.mean([by_n[m]["balanced"] for m in sizes])),  # 主要指標：各題長的各級平均再平均
            "balanced_mean_over_n_ci95": list(bootstrap_ci_mean_over_sizes(s[keep], players[keep], y[keep], n[keep])),  # 主要指標的 CI
            "question_mean": float(s.mean()),  # 全部題目的平均
            "question_mean_ci95": list(bootstrap_ci(s, players, y)),  # 全部題目平均的 CI
            **summary(np.ones(len(s), dtype=bool)),  # 全部題目的摘要
            "confusion_rows_truth_cols_pred": conf.tolist(),  # 混淆矩陣
            "by_n": by_n}  # 各題長的細節
