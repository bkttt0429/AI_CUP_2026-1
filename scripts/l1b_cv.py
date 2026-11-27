#!/usr/bin/env python
"""Folds 1-4 leave-one-fold-out CV of question features from Human SL, from the L1b heads, and fused.

Plan 2026-10-25-l1b-rank-heads, Task 7 (decision point 2). The protocol is the L3 ladder's: validation
questions pilot.pilot_questions (1 x 20, 2 x 10, 4 x 5 games per player), training questions
qfeatures.sample_questions with --per-player per player and RandomState(42 + v) over the training folds
in ascending order, one deterministic MLPModel(hidden=32) fit per feature set and validation fold,
scores pooled over the four validation folds, player bootstrap (n = --n-boot) for the paired
differences. Feature sets: hsl = the 24 question features of the 18 Human SL profiles (the production
model's), l1b = the 16 of the 10 L1b heads, fused = both (40). The two pre-specified comparisons are
l1b vs hsl and fused vs hsl; conditions as the ladder's: (1) CI lower bound > 0, (3) the point difference
> 0 in at least len(folds) - 1 folds, (4) point difference >= 0.005; (2) (every seed > 0) does not apply
to a deterministic fit and is reported as "n/a".

--l1b takes one or more l1b_score.py directories (model A's and model B's); every fold must be in exactly
one of them, and its games must be the same as --hsl's.

  python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 --out results/l1b/cv_summary.json
"""
import argparse  # 解析命令列參數
import sys  # 設定匯入路徑與結束碼
from pathlib import Path  # 處理路徑

import numpy as np  # 數值運算
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.humansl import PROFILES  # noqa: E402  18 個 Human SL 棋力設定
from aicup.l1bcv import players_from_scores  # noqa: E402  由評分表還原玩家的選局順序
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402  分折表、執行環境紀錄與 JSON 輸出
from aicup.metric import LEVELS  # noqa: E402  10 個等級（也是等級頭的順序）
from aicup.pilot import pilot_questions  # noqa: E402  驗證題的切法
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數評估與成對比較
from aicup.qfeatures import build_matrix, feature_names, sample_questions  # noqa: E402  題目特徵與訓練題抽樣
from aicup.rankmodel import MLPModel  # noqa: E402  題目層的 MLP
from aicup.scoring import completed_parts, read_scored_games  # noqa: E402  讀取逐局評分結果

FEATURE_SETS = ("hsl", "l1b", "fused")  # 要比較的三個特徵集
COMPARISONS = (("l1b", "hsl"), ("fused", "hsl"))  # 預先指定的兩個比較
MIN_DIFF = 0.005  # 條件 (4) 的最小點差異
TRAIN_SEED = 42  # 抽訓練題的種子（交叉驗證時再加驗證折編號）


def locate_folds(l1b_dirs, folds):  # 找出每一折的 L1b 評分在哪個資料夾
    """{fold: directory} with exactly one l1b directory per fold; ValueError listing duplicates and missing folds."""
    where = {}  # 折 → 資料夾
    dups = []  # 出現在多個資料夾的折
    for d in l1b_dirs:  # 逐一檢查每個 --l1b 資料夾
        for f in folds:  # 逐一檢查每一折
            if completed_parts(Path(d) / f"fold{f}"):  # 這個資料夾有該折已完成的評分
                if f in where:  # 已在別的資料夾找到過
                    dups.append(f)  # 記為重複
                where.setdefault(f, d)  # 記下第一個找到的資料夾
    missing = [f for f in folds if f not in where]  # 沒有任何資料夾涵蓋的折
    if dups or missing:  # 有重複或缺少
        raise ValueError(" ".join(filter(None, [f"fold {sorted(set(dups))} in more than one --l1b directory;" if dups else "",  # 錯誤訊息：重複的折
                                                 f"no L1b scores for folds {missing} in {list(l1b_dirs)}" if missing else ""])))  # 錯誤訊息：缺少的折
    return where  # 回傳每折的來源資料夾


def load_fold(hsl_dir, l1b_dir, fold):  # 讀入一折的兩種評分表
    """(hsl games, l1b games) of one fold with the same game keys (ValueError otherwise)."""
    hsl = read_scored_games(Path(hsl_dir) / f"fold{fold}")  # Human SL 的逐局評分
    l1b = read_scored_games(Path(l1b_dir) / f"fold{fold}")  # L1b 等級頭的逐局評分
    if set(hsl["game_key"]) != set(l1b["game_key"]):  # 兩邊評分的棋局不同
        raise ValueError(f"fold {fold}: {hsl_dir} and {l1b_dir} score different games")  # 兩邊棋局不一致時中止
    return hsl, l1b  # 回傳兩張表


def feature_sets(hsl_games, l1b_games, questions):  # 算出三個特徵集
    """{"hsl": X [Q, 24], "l1b": X [Q, 16], "fused": X [Q, 40]} of the questions, plus (y, n_games)."""
    X_h, y, n = build_matrix(hsl_games, questions, PROFILES)  # 18 個 Human SL 設定的 24 維特徵
    X_l, y2, _ = build_matrix(l1b_games, questions, LEVELS)  # 10 個等級頭的 16 維特徵
    assert np.array_equal(y, y2)  # 兩邊的題目標籤必須一致
    return {"hsl": X_h, "l1b": X_l, "fused": np.hstack([X_h, X_l])}, y, n  # 融合特徵為兩者橫向串接


def run_cv(hsl_dir, where, folds, folds_df, per_player, log=print):  # 留一折交叉驗證
    """Per validation fold the three fits; returns per-question tables and per-fold metrics."""
    data = {}  # 每折的評分表、玩家與驗證題
    for f in folds:  # 先讀入每一折
        hsl, l1b = load_fold(hsl_dir, where[f], f)  # 這一折的兩種評分
        players = players_from_scores(hsl, folds_df)  # 由評分表還原玩家與選局順序
        data[f] = {"hsl": hsl, "l1b": l1b, "players": players, "val": pilot_questions(players)}  # 驗證題用 pilot 的切法
        log(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)  # 回報這一折的規模
    rows, per_fold = [], {name: [] for name in FEATURE_SETS}  # 逐題結果與各特徵集的逐折分數
    nll = {name: [] for name in FEATURE_SETS}  # 各特徵集的逐折 NLL
    for v in folds:  # 每一折輪流當驗證折
        rng = np.random.RandomState(TRAIN_SEED + v)  # 每個驗證折固定的抽題亂數
        train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds) if f != v], ignore_index=True)  # 依折號遞增從其他折抽訓練題
        train_hsl = pd.concat([data[f]["hsl"] for f in folds if f != v], ignore_index=True)  # 訓練折的 Human SL 評分
        train_l1b = pd.concat([data[f]["l1b"] for f in folds if f != v], ignore_index=True)  # 訓練折的 L1b 評分
        X_tr, y_tr, _ = feature_sets(train_hsl, train_l1b, train_q)  # 訓練題的特徵
        X_va, y_va, n_va = feature_sets(data[v]["hsl"], data[v]["l1b"], data[v]["val"])  # 驗證題的特徵
        val = data[v]["val"]  # 驗證題表
        scores = {}  # 各特徵集的逐題分數
        for name in FEATURE_SETS:  # 每個特徵集各擬合一次
            proba = MLPModel(hidden=32).fit(X_tr[name], y_tr).predict_proba(X_va[name])  # 擬合 MLP 並預測驗證題機率
            res, s = evaluate(proba, y_va, val)  # 官方分數與診斷指標
            per_fold[name].append(res["balanced_mean_over_n"])  # 記錄這一折的平衡分數
            nll[name].append(res["nll_balanced"])  # 記錄這一折的 NLL
            scores[name] = s  # 逐題分數供合併比較
        log(f"CV held-out fold {v} ({len(train_q)} training questions): " + "  ".join(f"{n_} {per_fold[n_][-1]:.4f}" for n_ in FEATURE_SETS), flush=True)  # 印出這一折的結果
        rows.append(pd.DataFrame({"fold": v, "player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n_games": n_va,  # 這一折的逐題表
                                  **{f"s_{name}": scores[name] for name in FEATURE_SETS}}))  # 各特徵集的逐題分數欄
    return pd.concat(rows, ignore_index=True), per_fold, nll  # 回傳合併的逐題表與逐折指標


def summarise(table, per_fold, nll, folds, n_boot, n_features):  # 整理分數與預先指定的比較
    y, pid, n = table["y"].to_numpy(), table["player_id"].to_numpy(dtype=str), table["n_games"].to_numpy()  # 標籤、玩家與題目大小
    out = {"feature_sets": {}, "comparisons": {}}  # 輸出結構
    for name in FEATURE_SETS:  # 每個特徵集的分數摘要
        s = table[f"s_{name}"].to_numpy()  # 這個特徵集的逐題分數
        out["feature_sets"][name] = {"balanced_mean_over_n": balanced_mean_over_n(s, y, n),  # 各驗證折合併後的平衡分數
                                     "balanced_by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},  # 各題目大小的分數
                                     "per_fold": [float(x) for x in per_fold[name]], "nll_balanced": float(np.mean(nll[name])),  # 逐折分數與平均 NLL
                                     "nll_balanced_per_fold": [float(x) for x in nll[name]], "n_features": n_features[name]}  # 逐折 NLL 與特徵數
    need = max(1, len(folds) - 1)  # 條件 (3) 需要為正的折數
    for a, b in COMPARISONS:  # 逐一做預先指定的比較
        d = paired_diff(table[f"s_{a}"].to_numpy(), table[f"s_{b}"].to_numpy(), y, pid, n, n=n_boot)  # 成對差異與玩家 bootstrap CI
        per_fold_diff = [float(x - z) for x, z in zip(per_fold[a], per_fold[b])]  # 逐折的點差異
        cond = {"1": bool(d["ci95"][0] > 0), "2": "n/a", "3": bool(sum(x > 0 for x in per_fold_diff) >= need), "4": bool(d["diff"] >= MIN_DIFF)}  # 四個條件（(2) 不適用）
        out["comparisons"][f"{a}_vs_{b}"] = {**d, "per_fold_diff": per_fold_diff, "conditions": cond,  # 記錄比較結果
                                             "pass": bool(cond["1"] and cond["3"] and cond["4"])}  # 條件 1、3、4 都成立才通過
    return out  # 回傳摘要


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py output directories (each fold in exactly one)")  # L1b 評分資料夾
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))  # Human SL 評分資料夾
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])  # 參與交叉驗證的折
    ap.add_argument("--per-player", type=int, default=32)  # 每位玩家抽的訓練題數
    ap.add_argument("--n-boot", type=int, default=5000)  # bootstrap 次數
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--out", required=True)  # 摘要 JSON 的輸出路徑
    a = ap.parse_args(argv)  # 解析參數
    folds = sorted(set(a.folds))  # 去重並遞增排列（各折依此順序輪流驗證）
    if 0 in folds:  # fold 0 留給 fold-0 檢查，不在這裡評估
        ap.error("fold 0 is not evaluated")  # 拒絕評估 fold 0
    try:  # 輸入有問題時改為回報錯誤
        where = locate_folds(a.l1b, folds)  # 每折的 L1b 評分來源
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})  # 讀入分折表（玩家 id 保持字串）
        table, per_fold, nll = run_cv(a.hsl, where, folds, folds_df, a.per_player)  # 執行交叉驗證
    except (FileNotFoundError, ValueError, KeyError) as e:  # 缺檔或資料不一致
        print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示輸入有誤
    n_features = {"hsl": len(feature_names(PROFILES)), "l1b": len(feature_names(LEVELS)), "fused": len(feature_names(PROFILES)) + len(feature_names(LEVELS))}  # 各特徵集的維度
    summary = {"folds": folds, "per_player": a.per_player, "n_boot": a.n_boot, "questions": int(len(table)), "hsl_dir": a.hsl,  # 執行設定
               "l1b_dirs": {str(f): str(d) for f, d in where.items()}, **summarise(table, per_fold, nll, folds, a.n_boot, n_features),  # 資料來源與結果摘要
               "min_diff": MIN_DIFF, **environment("cpu")}  # 門檻與執行環境
    for name in FEATURE_SETS:  # 印出各特徵集的分數
        fs = summary["feature_sets"][name]  # 這個特徵集的摘要
        print(f"{name:6s} {fs['balanced_mean_over_n']:.4f}  by n " + " ".join(f"{m}: {fs['balanced_by_n'][str(m)]:.4f}" for m in SIZES)  # 總分與各題目大小的分數
              + f"  NLL {fs['nll_balanced']:.4f}  folds " + " ".join(f"{x:.4f}" for x in fs["per_fold"]))  # NLL 與逐折分數
    for a_, b_ in COMPARISONS:  # 印出每個比較
        c = summary["comparisons"][f"{a_}_vs_{b_}"]  # 這個比較的結果
        lo, hi = c["ci95"]  # 95% 信賴區間
        print(f"{a_} vs {b_}: {c['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])  # 差異、區間與逐折差異
              + f"  conditions 1 {'✓' if c['conditions']['1'] else '✗'} 2 n/a 3 {'✓' if c['conditions']['3'] else '✗'} 4 {'✓' if c['conditions']['4'] else '✗'}"  # 各條件是否成立
              + f"  -> {'PASS' if c['pass'] else 'no'}")  # 是否通過
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(summary, indent=1) + "\n")  # 寫出摘要 JSON
    print(f"wrote {a.out}")  # 回報輸出位置
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
