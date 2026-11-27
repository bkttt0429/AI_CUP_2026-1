"""Section 4M M43 (2026-11-07; user: 「把錯的 30% 再 train 一個模型去修正，然後用類似貝氏的方式選相信誰」): an error gate and an
error-only corrector on top of the pc ensemble's out-of-fold probabilities, cross-fitted over the four validation folds.

For seed set S (0 1 2 = ct9, or 3 4 5): P = mean of C/T-pc-f0-x40-oh seeds S on each validation fold; y-hat = the expected-score decision.
Features X: log P (10), entropy, top-2 margin, n_games, and the 64 R columns of r_val{v}_f0_x40 (X_va, the validation questions' order).
Trained on three folds' validation questions, scored on the fourth:
  gate      g = P(y-hat != y | X), MLP h32, class-balanced;
  corrector q(y | X, y-hat wrong), MLP h32 on the wrong questions only; at prediction q(y-hat) = 0, renormalised;
  bayes     P'(y) = (1 - g) 1[y = y-hat] + g q(y), then the expected-score decision;
  stack     (control) a 10-class MLP h32 on all questions, expected-score decision.
Against ct9's decision: paired player bootstrap (class-balanced official score) and the plain average; the new rule (CI lower > 0,
>= 3 of 4 folds > 0, diff >= 0.0025) must hold for both seed sets.
  python -I scripts/m_series/m43_gate.py --seeds 0 1 2 --out results/trunk_seq/es/m43_s012.json
"""
import argparse  # 解析命令列參數
import json  # 寫出 JSON 報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import torch  # 訓練閘門與修正用的 MLP

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與配對 bootstrap
from ens_compare import ES, load_fold  # noqa: E402  折外機率目錄與驗證折載入

E1 = float(np.exp(-1.0))  # 差一級時的官方得分 e^-1
FOLDS = (1, 2, 3, 4)  # 四個驗證折


def decide(P):  # 期望分決策（多支腳本共用）
    es = P.copy(); es[:, 1:] += E1 * P[:, :-1]; es[:, :-1] += E1 * P[:, 1:]  # 每級期望分＝本級機率＋e^-1×相鄰兩級機率
    return es.argmax(1)  # 取期望分最高的等級


def qscore(d, y):  # 每題的官方得分
    k = np.abs(d - y)  # 決策與真值差幾級
    return np.where(k == 0, 1.0, np.where(k == 1, E1, 0.0))  # 答對 1、差一級 e^-1、其餘 0


def features(P, n, R):  # 組出閘門與修正模型的輸入 X
    lp = np.log(np.clip(P, 1e-9, 1)); ent = -(P * lp).sum(1, keepdims=True); s = np.sort(P, 1)  # 對數機率、熵與排序後的機率
    return np.hstack([lp, ent, (s[:, -1] - s[:, -2])[:, None], n[:, None].astype(float), R]).astype(np.float32)  # 共 77 欄：10＋熵＋前兩名差＋局數＋64


def train_mlp(X, y, n_out, groups, seed, hidden=32, epochs=80, lr=1e-3, wd=1e-4, batch=512, patience=6):  # 訓練 MLP 並回傳預測函式
    torch.manual_seed(seed); rng = np.random.RandomState(seed)  # 固定種子以便重現
    players = np.unique(groups); held = set(rng.permutation(players)[:len(players) // 10]); h = np.array([g in held for g in groups])  # 依玩家留出一成做提早停止
    mu, sd = X[~h].mean(0), X[~h].std(0) + 1e-6  # 只用訓練部分估標準化參數
    Xt, Xh = torch.as_tensor((X[~h] - mu) / sd), torch.as_tensor((X[h] - mu) / sd)  # 標準化後轉成張量
    yt, yh = torch.as_tensor(y[~h]).long(), torch.as_tensor(y[h]).long()  # 訓練與留出部分的標籤
    cnt = np.bincount(y[~h], minlength=n_out).astype(float); cw = torch.as_tensor(np.where(cnt > 0, cnt.sum() / (n_out * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)  # 類別平衡權重，沒出現的類別給 0
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, n_out))  # 單一隱藏層的 MLP
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)  # Adam 最佳化器
    best, best_state, since = float("inf"), None, 0  # 提早停止的狀態
    for ep in range(epochs):  # 逐 epoch 訓練
        net.train(); perm = torch.randperm(len(Xt))  # 切到訓練模式並打亂順序
        for i in range(0, len(Xt), batch):  # 逐批次
            j = perm[i:i + batch]  # 本批次的索引
            loss = torch.nn.functional.cross_entropy(net(Xt[j]), yt[j], weight=cw)  # 類別加權的交叉熵
            opt.zero_grad(); loss.backward(); opt.step()  # 反向傳播並更新權重
        net.eval()  # 切到評估模式
        with torch.no_grad():  # 留出損失不需要梯度
            vl = float(torch.nn.functional.cross_entropy(net(Xh), yh, weight=cw))  # 留出玩家的損失
        if vl < best - 1e-5:  # 損失有明顯改善
            best, best_state, since = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0  # 記下目前最佳的權重
        else:  # 沒有改善
            since += 1  # 累計未改善的 epoch 數
            if since >= patience:  # 達到耐心上限
                break  # 提早停止
    net.load_state_dict(best_state); net.eval()  # 載回最佳權重
    return lambda Z: torch.softmax(net(torch.as_tensor((Z - mu) / sd)), 1).detach().numpy()  # 以相同標準化輸出各類機率


def auc(score, label):  # 以排名計算 AUC
    import pandas as pd  # 只在這裡用到 pandas 的排名
    r = pd.Series(score).rank().to_numpy(); n1 = label.sum(); n0 = len(label) - n1  # 分數排名與正負例數
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))  # Mann-Whitney 形式的 AUC


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()  # 種子組與輸出路徑
    D = {}  # 各折的資料
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(v)  # 真值、局數、玩家 ID 與題目表
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in a.seeds], axis=0)  # C、T 兩臂各種子的折外機率平均
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)  # 64 個 R 欄，與驗證題同序
        assert len(R) == len(y), f"fold {v}: R rows {len(R)} != {len(y)} questions"  # 確認 R 與題目對齊
        d = decide(P)  # 主模型的期望分決策
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "X": features(P, n, R), "d": d}  # 存下本折所需的資料
    yall = np.concatenate([D[v]["y"] for v in FOLDS]); dall = np.concatenate([D[v]["d"] for v in FOLDS]); k = np.abs(dall - yall)  # 四折合併的真值、決策與差距
    rep = {"seeds": a.seeds, "main": {"exact": float((k == 0).mean()), "adjacent": float((k == 1).mean()), "two_plus": float((k >= 2).mean())}, "folds": {}}  # 主模型猜中、差一級、差兩級以上的比例
    print(f"seeds {a.seeds}: main decision exact {rep['main']['exact']:.3f}, adjacent {rep['main']['adjacent']:.3f}, two or more off {rep['main']['two_plus']:.3f}", flush=True)  # 印出主模型的命中分布
    S = {"main": {}, "bayes": {}, "stack": {}}  # 各做法在每折的逐題得分
    gates, wrongs = [], []  # 收集閘門輸出與實際對錯
    for v in FOLDS:  # 交叉擬合：輪流留出第 v 折
        tr = [u for u in FOLDS if u != v]  # 其餘三折當訓練
        X = np.concatenate([D[u]["X"] for u in tr]); y = np.concatenate([D[u]["y"] for u in tr]); d = np.concatenate([D[u]["d"] for u in tr])  # 訓練折的特徵、真值與決策
        groups = np.concatenate([D[u]["pid"] for u in tr]); wrong = (d != y).astype(int)  # 玩家分組與主模型是否答錯
        gate = train_mlp(X, wrong, 2, groups, seed=v)  # 閘門：預測主模型是否答錯
        corr = train_mlp(X[wrong == 1], y[wrong == 1], 10, groups[wrong == 1], seed=10 + v)  # 修正模型：只用答錯的題
        stack = train_mlp(X, y, 10, groups, seed=20 + v)  # 對照組：全部題目的 10 類疊加器
        Xv, yv, dv = D[v]["X"], D[v]["y"], D[v]["d"]  # 留出折的特徵、真值與決策
        g = gate(Xv)[:, 1]; qv = corr(Xv); qv[np.arange(len(dv)), dv] = 0.0; qv /= qv.sum(1, keepdims=True)  # 答錯機率 g；修正分布去掉 y-hat 後正規化
        Pb = g[:, None] * qv; Pb[np.arange(len(dv)), dv] += 1.0 - g  # 貝氏合併 (1 - g)·1[y = y-hat] + g·q
        S["main"][v] = qscore(dv, yv); S["bayes"][v] = qscore(decide(Pb), yv); S["stack"][v] = qscore(decide(stack(Xv)), yv)  # 三種做法的逐題得分
        gates.append(g); wrongs.append((dv != yv).astype(int))  # 留給 AUC 計算
        print(f"fold {v}: gate mean {g.mean():.3f} vs wrong rate {(dv != yv).mean():.3f}; bayes changes {(decide(Pb) != dv).mean():.3f} of the decisions", flush=True)  # 閘門平均、實際錯誤率與改動比例
    g_all, w_all = np.concatenate(gates), np.concatenate(wrongs)  # 合併四折的閘門輸出與對錯
    pmax = np.concatenate([D[v]["P"].max(1) for v in FOLDS])  # 主模型自己的最大機率
    rep["gate"] = {"auc": auc(g_all, w_all), "auc_of_1_minus_pmax": auc(1 - pmax, w_all), "mean_g": float(g_all.mean()), "wrong_rate": float(w_all.mean())}  # 閘門 AUC 與主模型信心的對照
    print(f"gate AUC {rep['gate']['auc']:.4f} (the main model's own 1 - max p: {rep['gate']['auc_of_1_minus_pmax']:.4f})", flush=True)  # 印出兩個 AUC
    y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in ("y", "n", "pid"))  # 四折合併的真值、局數與玩家
    for arm in ("bayes", "stack"):  # 逐一比較兩種做法
        s_new, s_old = np.concatenate([S[arm][v] for v in FOLDS]), np.concatenate([S["main"][v] for v in FOLDS])  # 新做法與主模型決策（種子 0 1 2 時即 ct9）的逐題得分
        dd = paired_diff(s_new, s_old, y, pid, n, n=5000)  # 玩家層配對 bootstrap
        pf = [balanced_mean_over_n(S[arm][v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(S["main"][v], D[v]["y"], D[v]["n"]) for v in FOLDS]  # 各折的類別平衡分數差
        ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025  # 新規則：CI 下界 > 0、至少 3 折為正、差 >= 0.0025
        rep["folds"][arm] = {**dd, "per_fold_diff": pf, "plain_diff": float((s_new - s_old).mean()), "pass_new_rule": bool(ok)}  # 記錄比較結果
        print(f"{arm} vs ct9 decision: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)  # 印出差值、信賴區間與各折
              + f"  plain {rep['folds'][arm]['plain_diff']:+.4f}" + ("  -> PASS (new rule)" if ok else "  -> no"), flush=True)  # 再印直接平均與判定
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
