#!/usr/bin/env python
"""Do the opponent's moves add evidence? Folds 1-4 CV of the fused features plus the L1b features of the opponent's moves.

Online games pair players of similar ranks, so the opponent's moves are a second, independent look at the target
player's rank. l1b_score.py --opponent scores the opponent's moves (the other colour) of the same evaluation games
under the same out-of-fold heads (A for folds 3-4, B for 1-2); question_features over those sums gives 16 more
features. Protocol as scripts/l1b_cv.py: validation pilot.pilot_questions, training questions sample_questions
(--per-player, RandomState(42 + v)), one MLPModel(hidden=32) fit per feature set and validation fold.

Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp (hsl 24 +
l1b 16 + opp 16) vs fused (hsl + l1b). Reported only: hsl_opp (hsl + opp) vs hsl, opp alone.

  python -I scripts/l1b_opp_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/l1b/opp_check.json
"""
import argparse  # 解析命令列參數
import sys  # 設定匯入路徑與結束碼
from pathlib import Path  # 處理路徑

import numpy as np  # 數值運算
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
for d in (ROOT / "src", ROOT / "scripts"):  # 需要 src/ 的套件與 scripts/ 的 l1b_cv
    if str(d) not in sys.path:  # 還不在匯入路徑時
        sys.path.insert(0, str(d))  # 加入匯入路徑
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402  分折表、執行環境紀錄與 JSON 輸出
from aicup.metric import LEVELS  # noqa: E402  10 個等級（也是等級頭的順序）
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數評估與成對比較
from aicup.qfeatures import build_matrix, sample_questions  # noqa: E402  題目特徵與訓練題抽樣
from aicup.rankmodel import MLPModel  # noqa: E402  題目層的 MLP
from aicup.scoring import read_scored_games  # noqa: E402  讀取逐局評分結果
import l1b_cv  # noqa: E402  共用 l1b_cv 的讀檔、特徵與協定

SETS = ("hsl", "fused", "opp", "hsl_opp", "fused_opp")  # 要比較的五個特徵集
PRE = [("fused_opp", "fused")]  # 預先指定的比較
REPORT = [("hsl_opp", "hsl"), ("fused", "hsl"), ("fused_opp", "hsl")]  # 只報告的比較


def features(hsl, l1b, opp, questions):  # 五個特徵集（hd_check 與 fit_fused_model 也呼叫）
    base, y, n = l1b_cv.feature_sets(hsl, l1b, questions)  # 目標玩家的 hsl 與 fused 特徵
    X_o, y2, _ = build_matrix(opp, questions, LEVELS)  # 對手落子的 16 維 L1b 特徵
    assert np.array_equal(y, y2)  # 兩邊的題目標籤必須一致
    return {"hsl": base["hsl"], "fused": base["fused"], "opp": X_o, "hsl_opp": np.hstack([base["hsl"], X_o]),  # 各特徵集的組合
            "fused_opp": np.hstack([base["fused"], X_o])}, y, n  # fused 後接對手特徵（56 維）


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--l1b", nargs="+", required=True)  # 目標玩家落子的 L1b 評分資料夾
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")  # 對手落子的評分資料夾
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))  # Human SL 評分資料夾
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])  # 參與交叉驗證的折
    ap.add_argument("--per-player", type=int, default=32)  # 每位玩家抽的訓練題數
    ap.add_argument("--n-boot", type=int, default=5000)  # bootstrap 次數
    ap.add_argument("--out", required=True)  # 結果 JSON 的輸出路徑
    a = ap.parse_args(argv)  # 解析參數
    folds = sorted(set(a.folds))  # 去重並遞增排列（訓練題依此順序抽）
    where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)  # 每折的目標與對手評分來源
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 讀入分折表（玩家 id 保持字串）
    data = {}  # 每折的評分表、玩家與驗證題
    for f in folds:  # 讀入每一折
        hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)  # 目標玩家的兩種評分
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")  # 對手落子的評分
        if set(opp["game_key"]) != set(hsl["game_key"]):  # 對手評分涵蓋的棋局不同
            raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")  # 兩邊棋局不一致時中止
        players = l1b_cv.players_from_scores(hsl, folds_df)  # 還原玩家與選局順序
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players, "val": l1b_cv.pilot_questions(players)}  # 驗證題用 pilot 的切法
        print(f"fold {f}: {len(players)} players, {len(hsl)} games; opponent moves {int(opp['n_moves'].sum())} vs target "  # 回報規模與對手手數
              f"{int(hsl['n_moves'].sum())}", flush=True)  # 以及目標玩家的手數
    rows, per_fold, nll = [], {s: [] for s in SETS}, {s: [] for s in SETS}  # 逐題結果、逐折分數與 NLL
    for v in folds:  # 每一折輪流當驗證折
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)  # 與 l1b_cv 相同的抽題亂數
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)  # 從其他折抽訓練題
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731  合併訓練折的某種評分表
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), train_q)  # 訓練題的特徵
        val = data[v]["val"]  # 驗證題表
        X_va, y_va, n_va = features(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], val)  # 驗證題的特徵
        sc = {}  # 各特徵集的逐題分數
        for s in SETS:  # 每個特徵集各擬合一次
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, val)  # 擬合 MLP 並評估驗證題
            per_fold[s].append(res["balanced_mean_over_n"])  # 記錄這一折的平衡分數
            nll[s].append(res["nll_balanced"])  # 記錄這一折的 NLL
        print(f"fold {v}: " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)  # 印出這一折的結果
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))  # 這一折的逐題表
    t = pd.concat(rows, ignore_index=True)  # 合併各折的逐題表
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()  # 標籤、玩家與題目大小
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),  # 各特徵集的總分與平均 NLL
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},  # 各題目大小的分數
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS))},  # 各等級的平均分數
                "per_fold": per_fold[s]} for s in SETS}  # 逐折分數

    def compare(x, z):  # 比較兩個特徵集並判定條件
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)  # 成對差異與玩家 bootstrap CI
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]  # 逐折的點差異
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= len(folds) - 1), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}  # 條件 1、3、4
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}  # 三個條件都成立才通過

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}  # 預先指定與只報告的比較
    for s, h in head.items():  # 印出各特徵集的分數
        print(f"{s:10s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)  # 總分、NLL 與各題目大小
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]) + "  1D-3D " + " ".join(f"{h['per_class'][c]:.3f}" for c in ("1D", "2D", "3D")))  # 逐折分數與 1D-3D 的分數
    for g, cs in comps.items():  # 印出每組比較
        for k, c in cs.items():  # 組內的每個比較
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])  # 差異、區間與逐折差異
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")  # 各條件與是否通過
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "l1b_dirs": a.l1b, "opp_dirs": a.opp, "scores": head,  # 設定與分數
           "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}  # 比較結果與執行環境
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫出結果 JSON
    print(f"wrote {a.out}")  # 回報輸出位置
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
