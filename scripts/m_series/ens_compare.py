"""Ensemble-level adoption check (sections 4L.4 / 4M): the probability average of the `--new` arms' members against that of the
`--old` arms' members over the given seeds, folds 1-4, expected-score decision, paired player bootstrap; conditions (1) CI lower
bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005. Also each new arm against its old counterpart (same position in the lists).
--sizes restricts the evaluation to questions of those sizes (section 4M M2: 5 10). --new-seeds (default --seeds) are the new
arms' seeds; with unequal arm lists or seed lists only the combined pair is tested (section 4M M9: 12 = C T C-h4 T-h4 vs 6 = C T,
and the same-source control 12 = C T seeds 0-5 vs 6 = C T seeds 0-2).

  python -I scripts/m_series/ens_compare.py --old C T --new C-ft T-ft --seeds 0 1 2 --out results/trunk_seq/es/ens_m3.json
"""
import argparse  # 解析命令列參數
import json  # 讀取驗證折定義、寫出報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 組合結果檔路徑

import numpy as np  # 機率與分數的陣列運算
import pandas as pd  # 題目表格

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src")]  # 讓 aicup 套件可被匯入
from aicup.metric import LEVELS  # noqa: E402  匯入前須先設定 sys.path；10 個等級名稱
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  題目層評分與成對比較工具

ES = ROOT / "results/trunk_seq/es"  # 各成員驗證折機率檔的資料夾
RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引 0-9
FOLDS = (1, 2, 3, 4)  # 預設使用的驗證折


def load_fold(v):  # 讀取一個驗證折的題目與標籤（供其他實驗匯入）
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 該折驗證題的定義
    return np.array([RANK[r] for r in m["rank"]]), np.array(m["n_games"]), np.array(m["player_id"], dtype=str), pd.DataFrame(m)  # 回傳等級索引、每題局數、玩家 id 與題目表


def ensemble_scores(v, members, y, q, sizes):  # 一組成員集成後的逐題分數
    P = np.mean([np.load(ES / f"val{v}/{tag}_s{s}.npz")["proba"].astype(np.float64) for tag, s in members], axis=0)  # 各成員（臂, 種子）的機率取平均
    return evaluate(P, y, q, sizes=sizes)[1]  # 以期望分數決策評分，只取逐題分數


def main():  # 命令列進入點
    ap = argparse.ArgumentParser()  # 建立參數解析器
    ap.add_argument("--old", nargs="+", required=True)  # 舊集成的臂名稱
    ap.add_argument("--new", nargs="+", required=True)  # 新集成的臂名稱
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="the old arms' seeds")  # 舊臂的種子
    ap.add_argument("--new-seeds", type=int, nargs="+", default=None, help="the new arms' seeds (default: --seeds)")  # 新臂的種子
    ap.add_argument("--sizes", type=int, nargs="+", default=None)  # 只評這些局數（題目大小）的題目
    ap.add_argument("--out", required=True)  # 報告 JSON 的輸出路徑
    ap.add_argument("--n-boot", type=int, default=5000)  # 玩家 bootstrap 的次數
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS), help="validation folds (default 1 2 3 4)")  # 使用的驗證折
    a = ap.parse_args()  # 解析參數
    new_seeds = a.new_seeds or a.seeds  # 未指定時新臂沿用舊臂的種子
    sizes = tuple(a.sizes) if a.sizes else SIZES  # 參與評估的題目局數（預設 5、10、20）
    k, kn = len(a.seeds), len(new_seeds)  # 舊臂與新臂各自的種子數
    sets = {f"{kn * len(a.new)}_new": [(arm, s) for arm in a.new for s in new_seeds], f"{k * len(a.old)}_old": [(arm, s) for arm in a.old for s in a.seeds]}  # 新、舊兩個合併集成的成員清單
    pairs = [(f"{kn * len(a.new)}_new", f"{k * len(a.old)}_old")]  # 一定檢定的一對：新集成對舊集成
    if len(a.old) == len(a.new) and new_seeds == a.seeds:  # 臂數與種子相同時才逐臂比較
        for n_arm, o_arm in zip(a.new, a.old):  # 依清單位置配對新舊臂
            sets[f"{k}_{n_arm}"], sets[f"{k}_{o_arm}"] = [(n_arm, s) for s in a.seeds], [(o_arm, s) for s in a.seeds]  # 單臂集成的成員
            pairs.append((f"{k}_{n_arm}", f"{k}_{o_arm}"))  # 加入單臂對單臂的比較
    S, per_fold, ys, ns, pids = {x: [] for x in sets}, {x: [] for x in sets}, [], [], []  # 收集逐題分數、各折分數與題目資訊
    for v in a.folds:  # 逐一處理驗證折
        y, n, pid, q = load_fold(v)  # 該折的標籤、每題局數、玩家 id 與題目表
        ys.append(y); ns.append(n); pids.append(pid)  # 留待跨折合併
        for x, members in sets.items():  # 每個集成各評一次
            s = ensemble_scores(v, members, y, q, sizes)  # 該集成在此折的逐題分數
            S[x].append(s)  # 保存逐題分數供成對檢定
            per_fold[x].append(balanced_mean_over_n(s, y, n, sizes=sizes))  # 該折分數：各局數內類別平衡，再對局數平均
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)  # 合併各折的題目資訊
    rep = {"old": a.old, "new": a.new, "seeds": a.seeds, "new_seeds": new_seeds, "sizes": list(sizes), "ensembles": {}, "pairs": {}}  # 報告的基本設定
    for x in sets:  # 彙整每個集成的分數
        s = np.concatenate(S[x])  # 跨折的逐題分數
        rep["ensembles"][x] = {"members": sets[x], "score": float(np.mean(per_fold[x])), "per_fold": per_fold[x],  # 成員、各折平均與各折分數
                               "by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in sizes}}  # 各局數的類別平衡分數
        print(f"{x:16s} {rep['ensembles'][x]['score']:.4f}  folds {[round(e, 4) for e in per_fold[x]]}  by n " + " ".join(f"{m}: {rep['ensembles'][x]['by_n'][str(m)]:.4f}" for m in sizes))  # 印出集成分數摘要
    for x, z in pairs:  # 逐一檢定每一對
        d = paired_diff(np.concatenate(S[x]), np.concatenate(S[z]), y, pid, n, sizes=sizes, n=a.n_boot)  # 成對差與玩家 bootstrap 信賴區間
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]  # 各折的分數差
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= max(1, len(a.folds) - 1)), "4": bool(d["diff"] >= 0.005)}  # 採用條件 (1)(3)(4)
        rep["pairs"][f"{x} vs {z}"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}  # 記錄比較結果與是否通過
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)  # 印出差值、區間與各折差
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")  # 接著印各局數的差與判定
    if sizes != SIZES:  # 只評部分局數時
        print(f"evaluated on questions of {sizes} games only")  # 提醒評估範圍
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告 JSON
    print("written", a.out)  # 回報輸出位置


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
