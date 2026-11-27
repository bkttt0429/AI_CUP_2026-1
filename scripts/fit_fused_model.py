#!/usr/bin/env python
"""Fit the submission's fused MLP (Human SL + L1b question features) exactly as experiments/l1b_fold0_check.py did.

Issue #9. Training questions: qfeatures.sample_questions, --per-player per player, one RandomState(42) over
--train-folds in ascending order (the L3 ladder's and the fold-0 check's rule). Features of a training fold
come from the L1b model that did not see it (the sidecar's train_folds): A (folds 1-2) feeds folds 3-4, B
(folds 3-4) feeds folds 1-2; a fold with no such model, or with several, is refused. The MLP is
rankmodel.MLPModel(hidden=32) on the 40 fused features (aicup.fused). The document written to --out
(rank_model_fused.json) records the features, the training folds and question hash, and the L1b models
(heads file, state and trunk sha256, training folds, feature folds) so predict_submission.py can check that the
shards were scored with exactly those models.

Fold 0 is re-scored as the fold-0 check did -- hsl alone, fused with each model's scores, and the mean of the
models' probabilities -- and the numbers go into the document (fold0_check) and to stdout; with --reference they
are compared with results/l1b/fold0_check.json. --fold0-proba writes the per-question fold-0 probabilities (the
offline side of experiments/check_fused_submission.py). Only the models that did not train on fold 0 take part (A and
B both; of the five cross-fit models of issue #16 -- f0..f4, each trained on the other four folds and feeding its
own fold -- only f0), so for a cross-fit document the fold-0 averages are over that one model and the submission's
average over all five has no held-out check (the document says so in fold0_check.models_seen_fold0).

With --opp NAME:OPP_SCORES_DIR --diff (issue #25) the document's MLP is the fused_opp_diff one (59 features: fused_opp + the three
opponent-minus-own pmi differences of aicup.fused.diff_features), the fold-0 table also carries fused_opp_diff_* and
fused_opp_diff_avg vs fused_opp_avg, and the default output is rank_model_fused_opp_diff.json.
--no-fold0 skips the fold-0 check altogether (fold0_check.skipped = true in the document; no --fold0-proba / --reference): for a refit
that must not spend another look at fold 0, such as the current feature set on the re-scored handicap games (issue #42 (b)).
With --opp ... --diff --hd-table HD_TABLE_CSV (issue #47; scripts/hd_table.py's table of dataset v1.1's HD[n]) the document's MLP is
the fused_opp_diff_hd one (64 features: + aicup.handicap's five handicap question features, looked up per game), the fold-0
table also carries fused_opp_diff_hd_* and fused_opp_diff_hd_avg vs fused_opp_diff_avg, the fused_opp_diff MLP's state is kept
(fused_opp_diff_state) and the default output is rank_model_fused_opp_diff_hd.json. predict_submission.py computes the same
features from the test files' HD[] and refuses the document when a game carries none.
With --opp NAME:OPP_SCORES_DIR (issue #12; one per --l1b model, l1b_score.py --opponent directories) the
document's MLP is the fused_opp one: the 40 fused features plus the 16 of the L1b sums over the opponent's
moves, built by scripts/l1b_opp_check.features exactly as experiments/l1b_opp_fold0_check.py did (the same
model feeds the target's and the opponent's features of a fold). The hsl and fused MLPs are still fitted
for the fold-0 table (hsl, fused_*, fused_opp_*; --reference then takes results/l1b/opp_fold0_check.json),
and the fused MLP's state is kept in the document (fused_state) as a reference.

  python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json \\
      B:results/l1b/scores3_B:results/l1b/heads3_B.json --out results/models/rank_model_fused.json \\
      --fold0-proba results/l1b/fold0_fused_proba.csv --reference results/l1b/fold0_check.json --reference-run disc
  python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json \\
      B:results/l1b/scores3_B:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp B:results/l1b/scores3_B_opp \\
      --out results/models/rank_model_fused_opp.json --fold0-proba results/l1b/fold0_fused_opp_proba.csv \\
      --reference results/l1b/opp_fold0_check.json
"""
import argparse  # 解析命令列參數
import hashlib  # 計算訓練題的雜湊
import json  # 讀取側檔與參照檔
import sys  # 設定匯入路徑與結束碼
from pathlib import Path  # 處理路徑

import numpy as np  # 數值運算
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
for d in (ROOT / "src", ROOT / "scripts"):  # 需要 src/ 的套件與 scripts/ 的共用腳本
    if str(d) not in sys.path:  # 還不在匯入路徑時
        sys.path.insert(0, str(d))  # 加入匯入路徑
from aicup import fused  # noqa: E402  融合模型文件的建立與儲存
from aicup.handicap import hd_summary, question_hd_features, read_hd_table  # noqa: E402  讓子特徵與 HD 表
from aicup.ladder import FOLDS_CSV  # noqa: E402  分折表
from aicup.metric import LEVELS  # noqa: E402  10 個等級
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數評估與成對比較
from aicup.qfeatures import sample_questions  # noqa: E402  訓練題抽樣
from aicup.rankmodel import MLPModel  # noqa: E402  題目層的 MLP
from aicup.scoring import read_scored_games  # noqa: E402  讀取逐局評分結果
import l1b_cv  # noqa: E402  共用 l1b_cv 的讀檔、特徵與協定
import l1b_opp_check as oc  # noqa: E402  共用 fused_opp 特徵的算法

DEFAULT_TRAIN_FOLDS = (1, 2, 3, 4)  # 預設訓練折（fold 0 留作檢查）
DEFAULT_OUT = {"fused": ROOT / "results" / "models" / "rank_model_fused.json", "fused_opp": ROOT / "results" / "models" / "rank_model_fused_opp.json",  # 各特徵集的預設輸出檔
               "fused_opp_diff": ROOT / "results" / "models" / "rank_model_fused_opp_diff.json",  # 加差值特徵的文件
               "fused_opp_diff_hd": ROOT / "results" / "models" / "rank_model_fused_opp_diff_hd.json"}  # 再加讓子特徵的文件


def parse_l1b(spec):  # 解析一個 --l1b 參數
    """"NAME:SCORES_DIR:HEADS_SIDECAR" -> (name, scores dir, sidecar path, sidecar dict)."""
    parts = spec.split(":")  # 以冒號拆開
    if len(parts) != 3:  # 格式不符
        raise ValueError(f"--l1b expects NAME:SCORES_DIR:HEADS_SIDECAR, got {spec!r}")  # 拒絕
    name, scores, side = parts  # 名稱、評分資料夾、側檔路徑
    sidecar = json.loads(Path(side).read_text())  # 讀入等級頭的側檔
    for key in ("state_sha256", "train_folds"):  # 側檔必須有的欄位（記進文件並決定特徵來源）
        if sidecar.get(key) is None:  # 側檔缺少該欄
            raise ValueError(f"{side}: sidecar has no {key}")  # 拒絕
    return name, Path(scores), Path(side), sidecar  # 回傳模型的四項資訊


def parse_opp(spec, names):  # 解析一個 --opp 參數
    """"NAME:OPP_SCORES_DIR" -> (name, dir); the name must be one of the --l1b models'."""
    parts = spec.split(":")  # 以冒號拆開
    if len(parts) != 2:  # 格式不符
        raise ValueError(f"--opp expects NAME:OPP_SCORES_DIR, got {spec!r}")  # 拒絕
    name, d = parts  # 名稱與對手評分資料夾
    if name not in names:  # 名稱對不上任何 --l1b 模型
        raise ValueError(f"--opp {spec!r}: no --l1b model is named {name!r} ({sorted(names)})")  # 拒絕
    return name, Path(d)  # 回傳名稱與資料夾


def opponent_table(opp_dir, fold, hsl):  # 讀入一折的對手評分表
    """The l1b_score.py --opponent table of one fold; ValueError unless it covers exactly the Human SL table's games."""
    opp = read_scored_games(Path(opp_dir) / f"fold{fold}")  # 對手落子的逐局評分
    if set(opp["game_key"]) != set(hsl["game_key"]):  # 棋局與 Human SL 表不同
        raise ValueError(f"fold {fold}: the opponent scores in {opp_dir} cover other games than the Human SL scores "  # 拒絕：對手表必須涵蓋相同的棋局
                         f"({len(set(opp['game_key']) ^ set(hsl['game_key']))} differ)")  # 並列出不同的局數
    return opp  # 回傳對手評分表


def feature_sources(models, folds):  # 決定每一折的特徵由哪個模型提供
    """{fold: model name} -- the one model whose train_folds exclude the fold; ValueError when none or several."""
    out = {}  # 折 → 模型名稱
    for f in folds:  # 逐折決定
        cands = [name for name, _, _, side in models if f not in set(side["train_folds"])]  # 沒用這一折訓練的模型
        if len(cands) != 1:  # 必須恰好一個
            raise ValueError(f"fold {f}: {len(cands)} L1b models did not train on it ({cands}); exactly one must supply its features")  # 否則拒絕
        out[f] = cands[0]  # 記下提供特徵的模型
    return out  # 回傳對照表


def questions_sha(train_q):  # 訓練題的雜湊（記進文件以便核對）
    h = hashlib.sha256()  # 建立雜湊物件
    for qid, keys in zip(train_q["question_id"], train_q["game_keys"]):  # 依序走過每一題
        h.update(f"{qid}:{';'.join(keys)}\n".encode())  # 加入題號與其棋局
    return h.hexdigest()  # 回傳十六進位雜湊


def feature_sets(hsl, l1b, questions, opp=None, diff=False, hd=None):  # 依選項算出各特徵集
    """l1b_cv.feature_sets (hsl, fused), plus l1b_opp_check.features' fused_opp when the opponent table is given and,
    with diff, fused_opp_diff (+ aicup.fused.question_diff_features; issue #25) and, with an HD table, fused_opp_diff_hd
    (+ aicup.handicap.question_hd_features; issue #47)."""
    if opp is None:  # 沒有對手表
        return l1b_cv.feature_sets(hsl, l1b, questions)  # 只有 hsl、l1b、fused
    X, y, n = oc.features(hsl, l1b, opp, questions)  # 含 fused_opp 的特徵集
    if diff:  # 需要差值特徵
        X["fused_opp_diff"] = np.hstack([X["fused_opp"], fused.question_diff_features(l1b, opp, questions)])  # fused_opp 加三個差值特徵
    if hd is not None:  # 有 HD 表
        X["fused_opp_diff_hd"] = np.hstack([X["fused_opp_diff"], question_hd_features(hd, questions)])  # 再加五個讓子特徵
    return X, y, n  # 回傳特徵、標籤與題目大小


def fit(models, hsl_dir, folds, per_player, folds_df, log=print, opp_dirs=None, diff=False, hd=None):  # 在訓練折上擬合各 MLP
    """({"fused": MLP, "hsl": MLP[, "fused_opp": MLP[, "fused_opp_diff": MLP[, "fused_opp_diff_hd": MLP]]]}, info) fitted on the
    training folds with the fold-0 checks' protocol; opp_dirs {model name: l1b_score.py --opponent directory} adds the fused_opp
    fit (the fold's opponent features come from the same model as its target features), diff the fused_opp_diff fit and an HD
    table (aicup.handicap.read_hd_table) the fused_opp_diff_hd fit."""
    by_name = {name: (scores, side) for name, scores, _, side in models}  # 模型名稱 → (評分資料夾, 側檔)
    src = feature_sources(models, folds)  # 每一折的特徵來源模型
    data = {}  # 每折的評分表與玩家
    for f in folds:  # 讀入每一折
        hsl, l1b = l1b_cv.load_fold(hsl_dir, by_name[src[f]][0], f)  # Human SL 與折外模型的 L1b 評分
        data[f] = {"hsl": hsl, "l1b": l1b, "players": l1b_cv.players_from_scores(hsl, folds_df),  # 還原玩家與選局順序
                   "opp": opponent_table(opp_dirs[src[f]], f, hsl) if opp_dirs else None}  # 同一模型的對手評分
        log(f"fold {f}: {len(data[f]['players'])} players, {len(hsl)} games, L1b features from {src[f]}"  # 回報這一折的規模與特徵來源
            + (f" (opponent moves {int(data[f]['opp']['n_moves'].sum())} vs target {int(hsl['n_moves'].sum())})" if opp_dirs else ""), flush=True)  # 有對手特徵時另印手數
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)  # 所有訓練折共用一個 RandomState(42)
    train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds)], ignore_index=True)  # 依折號遞增抽訓練題
    cat = lambda key: pd.concat([data[f][key] for f in sorted(folds)], ignore_index=True)  # noqa: E731  合併各訓練折的某種表
    X, y, _ = feature_sets(cat("hsl"), cat("l1b"), train_q, cat("opp") if opp_dirs else None, diff, hd)  # 訓練題的各特徵集
    sets = ("fused", "hsl") + (("fused_opp",) if opp_dirs else ()) + (("fused_opp_diff",) if diff else ()) + (("fused_opp_diff_hd",) if hd is not None else ())  # 依選項決定要擬合的特徵集
    log(f"{len(train_q)} training questions; fitting the {', '.join(sets)} MLPs", flush=True)  # 回報訓練題數與要擬合的 MLP
    mlps = {s: MLPModel(hidden=32).fit(X[s], y) for s in sets}  # 每個特徵集擬合一個 MLP
    return mlps, {"n_train": int(len(train_q)), "questions_sha256": questions_sha(train_q), "sources": src}  # 回傳模型與訓練資訊


def fold0(models, hsl_dir, mlps, folds_df, n_boot, opp_dirs=None, diff=False, hd=None):  # fold 0 的留出檢查
    """The fold-0 checks' numbers for these fits (hsl, fused_<model>, fused_avg and, with opp_dirs, fused_opp_<model>,
    fused_opp_avg; with diff also fused_opp_diff_<model>, fused_opp_diff_avg; with hd also fused_opp_diff_hd_<model>,
    fused_opp_diff_hd_avg) and the per-question probabilities table. Only the models out-of-fold for fold 0 take part (A and B
    both; of the five cross-fit models of issue #16 only the one that left fold 0 out), and the averages are over those;
    the result names them (fold0_models) and the models left aside (models_seen_fold0). ValueError when none is OOF."""
    oof = [m for m in models if 0 not in set(m[3]["train_folds"])]  # 沒用 fold 0 訓練的模型
    seen = [name for name, _, _, side in models if 0 in set(side["train_folds"])]  # 看過 fold 0 的模型（排除）
    if not oof:  # 沒有任何折外模型
        raise ValueError("no L1b model is out-of-fold for fold 0 (every sidecar's train_folds has 0); the fold-0 check cannot be made")  # 無法做 fold-0 檢查
    hsl0 = l1b_cv.read_scored_games(Path(hsl_dir) / "fold0")  # fold 0 的 Human SL 評分
    val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))  # fold 0 的 pilot 驗證題
    P, y, n = {}, None, None  # 各組機率、標籤與題目大小
    for name, scores, _, _ in oof:  # 每個折外模型
        _, l1b0 = l1b_cv.load_fold(hsl_dir, scores, 0)  # 該模型在 fold 0 的 L1b 評分
        X, y_, n_ = feature_sets(hsl0, l1b0, val, opponent_table(opp_dirs[name], 0, hsl0) if opp_dirs else None, diff, hd)  # fold 0 驗證題的特徵
        if y is None:  # 第一個模型時
            y, n = y_, n_  # 記下標籤與題目大小
            P["hsl"] = mlps["hsl"].predict_proba(X["hsl"])  # hsl 與模型無關，只算一次
        assert np.array_equal(y, y_)  # 各模型的題目標籤必須一致
        P[f"fused_{name}"] = mlps["fused"].predict_proba(X["fused"])  # 這個模型的 fused 機率
        if opp_dirs:  # 有對手特徵時
            P[f"fused_opp_{name}"] = mlps["fused_opp"].predict_proba(X["fused_opp"])  # fused_opp 機率
        if diff:  # 有差值特徵時
            P[f"fused_opp_diff_{name}"] = mlps["fused_opp_diff"].predict_proba(X["fused_opp_diff"])  # fused_opp_diff 機率
        if hd is not None:  # 有讓子特徵時
            P[f"fused_opp_diff_hd_{name}"] = mlps["fused_opp_diff_hd"].predict_proba(X["fused_opp_diff_hd"])  # fused_opp_diff_hd 機率
    P["fused_avg"] = np.mean([P[f"fused_{name}"] for name, *_ in oof], axis=0)  # 只平均折外模型的機率（對應提交時對各 L1b 模型取平均）
    if opp_dirs:  # 有對手特徵時
        P["fused_opp_avg"] = np.mean([P[f"fused_opp_{name}"] for name, *_ in oof], axis=0)  # fused_opp 的模型平均
    if diff:  # 有差值特徵時
        P["fused_opp_diff_avg"] = np.mean([P[f"fused_opp_diff_{name}"] for name, *_ in oof], axis=0)  # fused_opp_diff 的模型平均
    if hd is not None:  # 有讓子特徵時
        P["fused_opp_diff_hd_avg"] = np.mean([P[f"fused_opp_diff_hd_{name}"] for name, *_ in oof], axis=0)  # fused_opp_diff_hd 的模型平均
    pid = val["player_id"].to_numpy(dtype=str)  # 每題的玩家（bootstrap 單位）
    res = {k: evaluate(p, y, val) for k, p in P.items()}  # 每組機率的官方分數與診斷
    s = {k: r[1] for k, r in res.items()}  # 每組的逐題分數
    out = {"n_questions": int(len(y)), "n_players": int(len(set(pid))), "fold0_models": [name for name, *_ in oof], "models_seen_fold0": seen,  # 題數、玩家數、參與與被排除的模型
           "scores": {}, "vs_hsl": {}}  # 分數與相對 hsl 的差異
    for k in P:  # 每一組機率
        out["scores"][k] = {"balanced_mean_over_n": balanced_mean_over_n(s[k], y, n), "nll_balanced": res[k][0]["nll_balanced"],  # 總分與 NLL
                            "balanced_by_n": {str(m): balanced_mean_over_n(s[k], y, n, sizes=(m,)) for m in SIZES},  # 各題目大小的分數
                            "per_class": {LEVELS[c]: float(np.mean(s[k][y == c])) for c in range(len(LEVELS)) if np.any(y == c)}}  # 各等級的平均分數
        if k != "hsl":  # hsl 以外的每一組
            out["vs_hsl"][k] = paired_diff(s[k], s["hsl"], y, pid, n, n=n_boot)  # 與 hsl 的成對差異
    if opp_dirs:  # 有對手特徵時
        out["fused_opp_avg_vs_fused_avg"] = paired_diff(s["fused_opp_avg"], s["fused_avg"], y, pid, n, n=n_boot)  # 對手特徵帶來的差異
    if diff:  # 有差值特徵時
        out["fused_opp_diff_avg_vs_fused_opp_avg"] = paired_diff(s["fused_opp_diff_avg"], s["fused_opp_avg"], y, pid, n, n=n_boot)  # 差值特徵帶來的差異
    if hd is not None:  # 有讓子特徵時
        out["fused_opp_diff_hd_avg_vs_fused_opp_diff_avg"] = paired_diff(s["fused_opp_diff_hd_avg"], s["fused_opp_diff_avg"], y, pid, n, n=n_boot)  # 讓子特徵帶來的差異
    table = val[["question_id", "rank", "n_games", "player_id"]].copy()  # 逐題機率表的基本欄位
    for k, p in P.items():  # 每一組機率
        for j, lv in enumerate(LEVELS):  # 每個等級
            table[f"{k}_p_{lv}"] = p[:, j]  # 一個等級一欄機率
    return out, table  # 回傳檢查結果與逐題機率表


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--l1b", nargs="+", required=True, metavar="NAME:SCORES_DIR:HEADS_SIDECAR")  # 各 L1b 模型的名稱、評分資料夾與側檔
    ap.add_argument("--opp", nargs="*", default=[], metavar="NAME:OPP_SCORES_DIR",  # 各模型的對手評分資料夾
                    help="l1b_score.py --opponent directories, one per --l1b model: fit the fused_opp MLP instead (issue #12)")  # 給了就改擬合 fused_opp MLP
    ap.add_argument("--diff", action="store_true", help="with --opp: fit the fused_opp_diff MLP instead (+3 opponent-minus-own pmi features; issue #25)")  # 改擬合 fused_opp_diff
    ap.add_argument("--hd-table", default=None, help="with --opp --diff: fit the fused_opp_diff_hd MLP instead (+5 handicap question features from "  # 給 HD 表時改擬合 fused_opp_diff_hd
                    "scripts/hd_table.py's CSV of dataset v1.1's HD[n]; issue #47)")  # 參數說明的後半
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))  # Human SL 評分資料夾
    ap.add_argument("--train-folds", type=int, nargs="+", default=list(DEFAULT_TRAIN_FOLDS))  # 訓練折
    ap.add_argument("--per-player", type=int, default=32)  # 每位玩家抽的訓練題數
    ap.add_argument("--n-boot", type=int, default=5000)  # bootstrap 次數
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--out", default=None, help=f"default {DEFAULT_OUT['fused'].name}, {DEFAULT_OUT['fused_opp'].name} with --opp, "  # 文件的輸出路徑
                    f"{DEFAULT_OUT['fused_opp_diff'].name} with --opp --diff")  # 參數說明的後半
    ap.add_argument("--fold0-proba", default=None, help="write fold 0's per-question probabilities (CSV) here")  # fold 0 逐題機率的輸出
    ap.add_argument("--no-fold0", action="store_true", help="skip the fold-0 check (the document records fold0_check.skipped = true): a refit that must not "  # 略過 fold-0 檢查（最終用的 rank_model_fused_opp_diff_hd_r.json 即如此擬合）
                    "spend another look at fold 0, e.g. the same feature set on re-scored data (issue #42 (b)); incompatible with --fold0-proba / --reference")  # 參數說明的後半
    ap.add_argument("--reference", default=None,  # fold-0 數字的參照檔
                    help="results/l1b/fold0_check.json (its runs[--reference-run]) or results/l1b/opp_fold0_check.json to compare the fold-0 numbers with")  # 可給 fold0_check.json（取 --reference-run 那一組）或 opp_fold0_check.json
    ap.add_argument("--reference-run", default="disc")  # 參照檔中要比對的 run
    ap.add_argument("--overwrite", action="store_true")  # 允許覆寫既有文件
    a = ap.parse_args(argv)  # 解析參數
    opponent = bool(a.opp)  # 是否使用對手特徵
    if a.diff and not opponent:  # --diff 需要對手評分
        ap.error("--diff needs --opp (the differences are between the opponent's and the target's L1b sums)")  # 拒絕
    if a.hd_table and not a.diff:  # --hd-table 需要 --diff
        ap.error("--hd-table needs --opp --diff (the handicap features are appended to fused_opp_diff)")  # 拒絕
    if a.no_fold0 and (a.fold0_proba or a.reference):  # 略過 fold 0 時不能要求其輸出或比對
        ap.error("--no-fold0 skips the fold-0 check, so --fold0-proba and --reference have nothing to write or compare")  # 拒絕
    kind = "fused_opp_diff_hd" if a.hd_table else "fused_opp_diff" if a.diff else "fused_opp" if opponent else "fused"  # 文件採用的特徵集
    out = Path(a.out) if a.out else DEFAULT_OUT[kind]  # 輸出路徑（未指定時用預設）
    if out.exists() and not a.overwrite:  # 文件已存在
        print(f"{out} exists; pass --overwrite to refit", file=sys.stderr)  # 提示需要 --overwrite
        return 2  # 拒絕覆寫
    folds = sorted(set(a.train_folds))  # 去重並遞增排列（文件記錄的訓練折）
    if 0 in folds:  # fold 0 是檢查用的
        ap.error("fold 0 is the check, not a training fold (propose adding it separately)")  # 不可當訓練折
    try:  # 輸入有問題時改為回報錯誤
        models = [parse_l1b(spec) for spec in a.l1b]  # 解析各 L1b 模型
        names = [name for name, *_ in models]  # 模型名稱
        opp_dirs = dict(parse_opp(spec, set(names)) for spec in a.opp) if opponent else None  # 模型名稱 → 對手評分資料夾
        if opponent and sorted(opp_dirs) != sorted(names):  # 每個模型都要恰好一個對手資料夾
            raise ValueError(f"--opp must name every --l1b model exactly once: got {sorted(opp_dirs)}, models {sorted(names)}")  # 否則拒絕
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})  # 讀入分折表（玩家 id 保持字串）
        hd = read_hd_table(a.hd_table) if a.hd_table else None  # 讀入 HD 表
        if hd is not None:  # 有 HD 表時
            cov = hd_summary(hd)  # HD 覆蓋率
            print(f"HD table {a.hd_table}: {cov['games']} games, HD>=1 {cov['share_hd1']:.3f}, HD>=2 {cov['share_hd2']:.3f}, absent {cov['hd'].get('absent', 0)}", flush=True)  # 印出 HD 表摘要
        mlps, info = fit(models, a.hsl, folds, a.per_player, folds_df, opp_dirs=opp_dirs, diff=a.diff, hd=hd)  # 在訓練折上擬合
        if a.no_fold0:  # 略過 fold-0 檢查
            check, table = {"skipped": True, "reason": "--no-fold0: fold 0 not evaluated by this fit", "scores": {}, "vs_hsl": {}, "models_seen_fold0": []}, None  # 文件記錄已略過
            print("fold 0: skipped (--no-fold0); the document carries no fold-0 check", flush=True)  # 回報已略過
        else:  # 做 fold-0 檢查
            check, table = fold0(models, a.hsl, mlps, folds_df, a.n_boot, opp_dirs=opp_dirs, diff=a.diff, hd=hd)  # fold 0 的留出檢查
    except (FileNotFoundError, ValueError, KeyError) as e:  # 缺檔或資料不一致
        print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示輸入有誤
    if check["models_seen_fold0"]:  # 有模型看過 fold 0
        print(f"fold 0: models {check['models_seen_fold0']} trained on it and are left out of the check; the averages are over "  # 提醒這些模型被排除
              f"{check['fold0_models']} only -- the submission's average over all {len(models)} models has no held-out check", flush=True)  # 提交時的平均沒有留出檢查
    for k, v in check["scores"].items():  # 印出 fold 0 各組分數
        d = check["vs_hsl"].get(k)  # 與 hsl 的差異
        extra = f"  vs hsl {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""  # 有差異時附上
        print(f"fold 0 {k:14s} {v['balanced_mean_over_n']:.4f}  NLL {v['nll_balanced']:.4f}  by n "  # 總分與 NLL
              + " ".join(f"{m}: {v['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + extra, flush=True)  # 各題目大小與相對 hsl 的差異
    if opponent and not a.no_fold0:  # 有對手特徵且做了檢查
        d = check["fused_opp_avg_vs_fused_avg"]  # 對手特徵帶來的差異
        print(f"fold 0 fused_opp_avg vs fused_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)  # 印出差異與區間
    if a.diff and not a.no_fold0:  # 有差值特徵且做了檢查
        d = check["fused_opp_diff_avg_vs_fused_opp_avg"]  # 差值特徵帶來的差異
        print(f"fold 0 fused_opp_diff_avg vs fused_opp_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)  # 印出差異與區間
    if a.hd_table and not a.no_fold0:  # 有讓子特徵且做了檢查
        d = check["fused_opp_diff_hd_avg_vs_fused_opp_diff_avg"]  # 讓子特徵帶來的差異
        print(f"fold 0 fused_opp_diff_hd_avg vs fused_opp_diff_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)  # 印出差異與區間
    if a.reference:  # 有參照檔時
        ref = json.loads(Path(a.reference).read_text())  # 讀入參照檔
        ref = ref["runs"][a.reference_run]["scores"] if "runs" in ref else ref["scores"]  # 前者是 fold0_check.json，後者是 opp_fold0_check.json
        for k, v in check["scores"].items():  # 逐組比對
            if k in ref:  # 參照檔有這一組
                delta = v["balanced_mean_over_n"] - ref[k]["balanced_mean_over_n"]  # 與參照的差
                print(f"  vs {a.reference} {k}: {delta:+.2e}" + ("  WARNING: differs" if abs(delta) > 1e-6 else ""))  # 差超過 1e-6 就警告
    records = [fused.l1b_model_record(name, side_path.with_suffix(".pt").name, side, [f for f, s in info["sources"].items() if s == name])  # 每個 L1b 模型的紀錄（含提供特徵的折）
               for name, _, side_path, side in models]  # 逐一處理每個模型
    extra = {"feature_source_by_fold": {str(f): s for f, s in info["sources"].items()}, "hsl_dir": str(a.hsl),  # 各折的特徵來源與 Human SL 資料夾
             "l1b_scores_dirs": {name: str(scores) for name, scores, _, _ in models}, "fold0_check": check, "hsl_state": mlps["hsl"].state_dict()}  # 評分資料夾、fold-0 檢查與 hsl MLP 的 state
    if opponent:  # 有對手特徵時
        extra.update({"opp_scores_dirs": {name: str(d) for name, d in opp_dirs.items()}, "fused_state": mlps["fused"].state_dict()})  # 記錄對手資料夾與 fused MLP 的 state
    if a.diff:  # 有差值特徵時
        extra["fused_opp_state"] = mlps["fused_opp"].state_dict()  # 保留 fused_opp MLP 的 state
    if a.hd_table:  # 有讓子特徵時
        extra.update({"fused_opp_diff_state": mlps["fused_opp_diff"].state_dict(), "hd_table": str(a.hd_table), "hd_coverage": hd_summary(hd)})  # 保留 fused_opp_diff 的 state 與 HD 表資訊
    doc = fused.fused_document(mlps[kind], records, folds, a.per_player, info["n_train"], info["questions_sha256"], extra=extra, feature_set=kind)  # 組成融合模型文件
    fused.save_fused_model(doc, out)  # 寫出文件
    print(f"wrote {out}: {doc['name']}, {info['n_train']} training questions, L1b models {[r['name'] for r in records]}")  # 回報輸出
    if a.fold0_proba and table is not None:  # 需要 fold 0 逐題機率時
        Path(a.fold0_proba).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        table.to_csv(a.fold0_proba, index=False)  # 寫出逐題機率表
        print(f"wrote {a.fold0_proba} ({len(table)} fold-0 questions)")  # 回報輸出
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
