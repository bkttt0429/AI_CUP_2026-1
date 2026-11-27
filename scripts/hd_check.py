#!/usr/bin/env python
"""Folds 1-4 CV: do the handicap (HD) features of dataset v1.1 add to fused_opp_diff? (issue #42, proposal (a))

The server's handicap is set from the two players' ranks (HD[1] 讓先 about one rank apart, HD[n] stones about n apart, HD[0]
equal), so it is label-side information like the opponent channel, but direct. aicup.handicap turns each question's games into
five means (hd_even, hd_sente, hd_stones, hd_gap_up, hd_gap_down; the signed gap seen from the labelled player uses the actual
colour, W = upper hand) that are appended to the 59 fused_opp_diff features (fused_opp_diff_hd, 64) and to the 56 fused_opp
features (fused_opp_hd, 61). Protocol as experiments/hist_check.py (pilot questions, 32 training questions per player, MLP h32,
player bootstrap): fused_opp must reproduce 0.6735 and fused_opp_diff 0.6916. Pre-specified (conditions (1) CI lower bound > 0,
(3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_hd vs fused_opp_diff. Reported only: fused_opp_hd vs fused_opp,
fused_opp_diff_hd vs fused_opp_hd, fused_opp_diff vs fused_opp. Diagnostics: the HD coverage of the scored games; every set's
score on the questions without any HD game / with 讓先 only / with a stones game; the Spearman correlation of fused_opp_diff's
residual (predicted - true rank index) with the question's signed gap. The HD table comes from scripts/hd_table.py (CPU); the
scores are the existing ones (no re-scoring): HD is read from the CSVs only.

  python -I scripts/hd_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --hd-table results/data/hd_table.csv --out results/seq/hd_check.json
"""
import argparse  # 解析命令列參數
import sys  # 設定匯入路徑與結束碼
import time  # 計算耗時
from pathlib import Path  # 處理路徑

import numpy as np  # 數值運算
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
for d in (ROOT / "src", ROOT / "scripts"):  # 需要 src/ 的套件與 scripts/ 的共用腳本
    if str(d) not in sys.path:  # 還不在匯入路徑時
        sys.path.insert(0, str(d))  # 加入匯入路徑
from aicup.fused import question_diff_features  # noqa: E402  三個「對手 − 自己」差值特徵
from aicup.handicap import HD_FEATURE_NAMES, hd_summary, question_hd_features, read_hd_table  # noqa: E402  讓子特徵與 HD 表
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402  分折表、執行環境紀錄與 JSON 輸出
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402  等級順序與期望分數決策
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數評估與成對比較
from aicup.qfeatures import sample_questions  # noqa: E402  訓練題抽樣
from aicup.rankmodel import MLPModel  # noqa: E402  題目層的 MLP
from aicup.scoring import read_scored_games  # noqa: E402  讀取逐局評分結果
import l1b_cv  # noqa: E402  共用 l1b_cv 的讀檔與協定
import l1b_opp_check as oc  # noqa: E402  共用 fused_opp 特徵的算法

SETS = ("fused_opp", "fused_opp_diff", "fused_opp_hd", "fused_opp_diff_hd")  # 要比較的四個特徵集
PRE = [("fused_opp_diff_hd", "fused_opp_diff")]  # 預先指定的比較
REPORT = [("fused_opp_hd", "fused_opp"), ("fused_opp_diff_hd", "fused_opp_hd"), ("fused_opp_diff", "fused_opp")]  # 只報告的比較
BUCKETS = ("no_hd", "sente_only", "stones")  # 依讓子情形分的三類題目
COL_SENTE, COL_STONES, COL_UP, COL_DOWN = (HD_FEATURE_NAMES.index(n) for n in ("hd_sente", "hd_stones", "hd_gap_up", "hd_gap_down"))  # HD 區塊中各欄的位置


def features(hsl, l1b, opp, hd, questions):  # 四個特徵集（trunk_seq_check 用它算 64 個題目層特徵）
    """The four feature sets of `questions`: fused_opp (56), fused_opp_diff (59), fused_opp + HD (61), fused_opp_diff + HD (64); also the HD block."""
    X, y, n = oc.features(hsl, l1b, opp, questions)  # 先算 fused_opp 等既有特徵
    diff = np.hstack([X["fused_opp"], question_diff_features(l1b, opp, questions)])  # 加上三個差值特徵即 fused_opp_diff
    H = question_hd_features(hd, questions)  # 每題 5 個讓子特徵
    return {"fused_opp": X["fused_opp"], "fused_opp_diff": diff, "fused_opp_hd": np.hstack([X["fused_opp"], H]), "fused_opp_diff_hd": np.hstack([diff, H])}, y, n, H  # 有無 HD 的各種組合與 HD 區塊


def bucket_of(H):  # 依 HD 區塊把題目分類
    """no_hd: no game with HD >= 1; stones: at least one HD >= 2 game; sente_only: HD = 1 games but no stones game."""
    stones, sente = H[:, COL_STONES] > 0, H[:, COL_SENTE] > 0  # 題目是否含讓子局、讓先局
    return np.where(stones, "stones", np.where(sente, "sente_only", "no_hd"))  # 讓子局優先，其次讓先


def _spearman(a, b):  # Spearman 等級相關
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()  # 轉成名次
    if a.std() == 0 or b.std() == 0:  # 任一方沒有變異時
        return float("nan")  # 相關係數無定義
    return float(np.corrcoef(a, b)[0, 1])  # 名次的 Pearson 相關即 Spearman 相關


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")  # 目標玩家落子的評分資料夾
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")  # 對手落子的評分資料夾
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))  # Human SL 評分資料夾
    ap.add_argument("--hd-table", required=True, help="scripts/hd_table.py's CSV")  # 訓練集的 HD 表
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])  # 參與交叉驗證的折
    ap.add_argument("--per-player", type=int, default=32)  # 每位玩家抽的訓練題數
    ap.add_argument("--n-boot", type=int, default=5000)  # bootstrap 次數
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--out", required=True)  # 結果 JSON 的輸出路徑
    a = ap.parse_args(argv)  # 解析參數
    folds = sorted(set(a.folds))  # 去重並遞增排列（訓練題依此順序抽）
    if 0 in folds:  # fold 0 留給 fold-0 檢查，不在這裡評估
        ap.error("fold 0 is not evaluated here")  # 拒絕評估 fold 0
    t0 = time.time()  # 開始計時
    try:  # 輸入有問題時改為回報錯誤
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)  # 每折的目標與對手評分來源
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})  # 讀入分折表（玩家 id 保持字串）
        hd_all = read_hd_table(a.hd_table)  # 讀入整個訓練集的 HD 表
        data, coverage = {}, {}  # 每折的資料與 HD 覆蓋率
        for f in folds:  # 讀入每一折
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)  # 目標玩家的兩種評分
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")  # 對手落子的評分
            if set(opp["game_key"]) != set(hsl["game_key"]):  # 對手評分涵蓋的棋局不同
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")  # 兩邊棋局不一致時中止
            hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]  # 只取這一折已評分的棋局
            missing = set(hsl["game_key"]) - set(hd["game_key"])  # HD 表中缺少的棋局
            if missing:  # 有棋局查不到 HD
                raise ValueError(f"fold {f}: {len(missing)} scored games are not in {a.hd_table}, e.g. {sorted(missing)[:3]}")  # 中止並列出例子
            players = l1b_cv.players_from_scores(hsl, folds_df)  # 還原玩家與選局順序
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}  # 驗證題用 pilot 的切法
            coverage[str(f)] = hd_summary(hd)  # 這一折的 HD 覆蓋率
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions; HD>=1 {coverage[str(f)]['share_hd1']:.3f}, "  # 回報規模與讓先以上的比例
                  f"HD>=2 {coverage[str(f)]['share_hd2']:.3f}, absent {coverage[str(f)]['hd'].get('absent', 0)}", flush=True)  # 讓子局比例與缺 HD 的局數
    except (FileNotFoundError, ValueError, KeyError) as e:  # 缺檔或資料不一致
        print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示輸入有誤
    rows, per_fold, nll, dims, oof = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}, []  # 逐題結果、逐折指標、特徵維度與折外預測
    for v in folds:  # 每一折輪流當驗證折
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)  # 與 l1b_cv 相同的抽題亂數
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)  # 從其他折抽訓練題
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731  合併訓練折的某種表
        X_tr, y_tr, _, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)  # 訓練題的特徵
        d = data[v]  # 驗證折的資料
        X_va, y_va, n_va, H_va = features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])  # 驗證題的特徵與 HD 區塊
        sc = {}  # 各特徵集的逐題分數
        for s in SETS:  # 每個特徵集各擬合一次
            dims[s] = int(X_tr[s].shape[1])  # 記錄特徵維度
            P = MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s])  # 擬合 MLP 並預測驗證題機率
            res, sc[s] = evaluate(P, y_va, d["val"])  # 官方分數與診斷指標
            per_fold[s].append(res["balanced_mean_over_n"])  # 記錄這一折的平衡分數
            nll[s].append(res["nll_balanced"])  # 記錄這一折的 NLL
            if s == "fused_opp_diff":  # 只對 fused_opp_diff 收集殘差診斷用的折外預測
                oof.append(pd.DataFrame({"fold": v, "n_games": n_va, "y": y_va, "pred": [expected_score_decision(p) for p in P],  # 折外的期望分數決策
                                         "q_gap": H_va[:, COL_UP] - H_va[:, COL_DOWN]}))  # 題目的有號差距
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS) + f"  ({time.time() - t0:.0f} s)", flush=True)  # 印出這一折的結果與耗時
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, "bucket": bucket_of(H_va), **{s: sc[s] for s in SETS}}))  # 這一折的逐題表（含讓子分類）
    t = pd.concat(rows, ignore_index=True)  # 合併各折的逐題表
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()  # 標籤、玩家與題目大小
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])), "n_features": dims[s],  # 各特徵集的總分、NLL 與維度
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},  # 各題目大小的分數
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},  # 各等級的平均分數（只算有題目的等級）
                "per_fold": per_fold[s]} for s in SETS}  # 逐折分數
    need = max(1, len(folds) - 1)  # 條件 (3) 需要為正的折數

    def compare(x, z):  # 比較兩個特徵集並判定條件
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)  # 成對差異與玩家 bootstrap CI
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]  # 逐折的點差異
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}  # 條件 1、3、4
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}  # 三個條件都成立才通過

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}  # 預先指定與只報告的比較
    buckets = {b: {"questions": int((t["bucket"] == b).sum()), **{s: float(t.loc[t["bucket"] == b, s].mean()) if (t["bucket"] == b).any() else None for s in SETS}}  # 各類題目的題數與平均分數
               for b in BUCKETS}  # 三類讓子情形
    o = pd.concat(oof, ignore_index=True)  # 合併各折的折外預測
    resid = o["pred"].to_numpy(dtype=float) - o["y"].to_numpy(dtype=float)  # 殘差 = 預測 − 真實等級索引
    residual = {"spearman_residual_vs_gap": _spearman(resid, o["q_gap"].to_numpy()),  # 殘差與有號差距的相關
                "by_n": {str(m): _spearman(resid[o["n_games"] == m], o.loc[o["n_games"] == m, "q_gap"].to_numpy()) for m in SIZES},  # 各題目大小的相關
                "mean_residual_by_gap_sign": {k: float(resid[m].mean()) if m.any() else None  # 依差距正負分組的平均殘差
                                              for k, m in (("gap<0", o["q_gap"] < 0), ("gap=0", o["q_gap"] == 0), ("gap>0", o["q_gap"] > 0))}}  # 差距 < 0、= 0、> 0 三組
    for s, h in head.items():  # 印出各特徵集的分數
        print(f"{s:18s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "  # 維度、總分與 NLL
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))  # 各題目大小與逐折分數
    for g, cs in comps.items():  # 印出每組比較
        for k, c in cs.items():  # 組內的每個比較
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])  # 差異、區間與逐折差異
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")  # 各條件與是否通過
    for b, r in buckets.items():  # 印出各類題目的分數
        print(f"bucket {b:11s} {r['questions']:5d} questions  " + "  ".join(f"{s} {r[s]:.4f}" if r[s] is not None else f"{s} -" for s in SETS))  # 題數與各特徵集的分數
    print(f"fused_opp_diff residual vs signed gap: Spearman {residual['spearman_residual_vs_gap']:.3f}; by n "  # 殘差與差距的相關
          + " ".join(f"{m}: {residual['by_n'][str(m)]:.3f}" for m in SIZES) + "; mean residual " + " ".join(f"{k} {v:+.3f}" if v is not None else f"{k} -" for k, v in residual["mean_residual_by_gap_sign"].items()))  # 各題目大小的相關與分組平均殘差
    out = {"folds": folds, "per_player": a.per_player, "n_boot": a.n_boot, "min_diff": l1b_cv.MIN_DIFF, "hd_table": a.hd_table, "hd_feature_names": HD_FEATURE_NAMES,  # 設定與 HD 特徵名稱
           "coverage": coverage, "scores": head, "comparisons": comps, "buckets": buckets, "residual": residual, "wall_s": time.time() - t0, **environment("cpu")}  # 各項結果、耗時與執行環境
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫出結果 JSON
    print(f"wrote {a.out}")  # 回報輸出位置
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
