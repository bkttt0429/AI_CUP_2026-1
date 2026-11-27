"""Section 4M M71 step 2 (2026-11-10): fuse the opponent-anchor evidence into the ensemble posterior, cross-fitted over validation folds.
Reads results/r_variants/m71_queries.npz (step 1). For validation fold v the calibration uses the other three folds only:
band edges = impostor-margin quantiles (per model colour); groups = HD0, HD1 x target colour; P0(d) = label-difference distribution
(candidate label - target label) of the bands <= q80; a_b = clip(1 - impostor share / opponent share, 0, 0.98);
K(d) = the pooled bands > q98 minus (1 - a_top) P0, clipped at 0, normalised; per-game evidence e(c) = (1 - a_b) + a_b min(K(d)/P0(d), 30)
with d = candidate label - c. Question posterior = ensemble posterior x prod_g e_g(c); expected-score decision.
Base = standard 6 members (seeds 0-2); replication = standard seeds 6-8. Rule (M37): CI lower bound > 0, >= 3 of 4 folds > 0, diff >= 0.0025, in both.
  python -I scripts/m_series/m71_step2.py
"""
import json  # 寫出 JSON 報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數與配對比較
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折

STD = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh")  # 標準成員的兩種結構（C、T）
QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]  # 帶界所用的冒充者 margin 分位
D = np.arange(-9, 10)  # 標籤差 d 的範圍 -9..9


def group_of(hd, tb):  # 依讓子數與本人執色分組
    return np.where(hd == 0, 0, np.where(hd == 1, np.where(tb, 1, 2), -1))  # HD0；HD1 本人執黑；HD1 本人執白；-1 = 不給證據


def fit(z, tr):  # 以訓練查詢校準證據模型
    """Calibration from the queries in `tr`: band edges per model colour, and per group P0, K, a_b."""
    edges = {True: np.quantile(z["imp_m"][tr & z["target_black"]], QS), False: np.quantile(z["imp_m"][tr & ~z["target_black"]], QS)}  # 依模型顏色取冒充者分位當帶界
    nominal = np.diff(np.concatenate([[0.0], QS, [1.0]]))  # 各帶的名目占比
    band = np.where(z["target_black"], np.searchsorted(edges[False], z["opp_m"]), np.searchsorted(edges[True], z["opp_m"]))  # 對手執另一色
    g = group_of(z["hd"], z["target_black"]); d = z["opp_lab"] - z["y"]; cal = {}  # 分組與標籤差 d
    for k in (0, 1, 2):  # 逐分組校準
        s = tr & (g == k); nb = len(QS) + 1  # 本組的訓練查詢；帶數
        share = np.array([(s & (band == b)).sum() for b in range(nb)]) / max(s.sum(), 1)  # 對手查詢落在各帶的占比
        a = np.clip(1 - nominal / np.maximum(share, 1e-9), 0, 0.98)  # a_b = 1 - 冒充者占比 / 對手占比
        hist = lambda m: (np.array([(d[m] == x).sum() for x in D]) + 1.0) / (m.sum() + len(D))  # noqa: E731  加 1 平滑的標籤差分布
        P0 = hist(s & (band <= 1)); top = s & (band >= 5); a_top = float(np.clip(1 - nominal[5:].sum() / max(top.sum() / max(s.sum(), 1), 1e-9), 0, 0.98))  # 背景 P0（<= q80）；> q98 合併帶的權重
        K = np.clip(hist(top) - (1 - a_top) * P0, 0, None); K = K / K.sum()  # 尖峰核 K：扣除背景、截 0 後正規化
        cal[k] = {"a": a, "ratio": np.minimum(K / P0, 30.0), "share": share, "a_top": a_top, "K": K, "P0": P0}  # 存下本組的校準
    return edges, cal, band, g  # 回傳帶界、校準表、帶與分組


def evidence_per_query(z, v):  # 第 v 折各查詢的對數證據
    """log e_g(c) [Q, 10] for the queries of fold v, calibrated on the other folds."""
    tr = z["fold"] != v; edges, cal, _, _ = fit(z, tr)  # 只用其餘三折校準
    te = np.flatnonzero(z["fold"] == v); out = np.zeros((len(te), 10))  # 本折的查詢與輸出陣列
    band = np.where(z["target_black"][te], np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te]))  # 對手查詢的帶
    g = group_of(z["hd"][te], z["target_black"][te]); lab = z["opp_lab"][te]  # 分組與配對到的候選人標籤
    for k in (0, 1, 2):  # 逐分組
        s = g == k  # 本組的查詢
        dd = lab[s][:, None] - np.arange(10)[None, :]  # 候選人標籤 - c
        out[s] = np.log((1 - cal[k]["a"][band[s]])[:, None] + cal[k]["a"][band[s]][:, None] * cal[k]["ratio"][dd + 9])  # 對數證據 log((1 - a_b) + a_b·K/P0)
    return te, out, cal  # 回傳查詢索引、對數證據與校準


def run(name, seeds, z, rep):  # 以一組種子做交叉擬合評估
    S_new, S_old, pf, ys, ns, pids, changed, total = [], [], [], [], [], [], 0, 0  # 收集結果與改動計數
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(v)  # 本折的真值、局數、玩家與題目表
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in seeds], axis=0)  # 標準成員的折外機率平均
        te, logE, cal = evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}  # 本折查詢的證據；棋局鍵對應列
        L = np.zeros_like(P)  # 每題的對數證據
        for i, gk in enumerate(q["game_keys"]):  # 逐題
            rows = [key_row[k] for k in gk if k in key_row]  # 這題中有查詢的局
            if rows:  # 有錨點證據
                L[i] = logE[rows].sum(axis=0)  # 各局證據相乘（對數相加）
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf = Pf / Pf.sum(axis=1, keepdims=True)  # 題目後驗 = 集成後驗 × 證據
        so, sn = evaluate(P, y, q, sizes=SIZES)[1], evaluate(Pf, y, q, sizes=SIZES)[1]  # 加證據前後的逐題得分
        changed += int((np.abs(sn - so) > 1e-12).sum()); total += len(y)  # 統計得分改變的題數
        S_new.append(sn); S_old.append(so); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(sn, y, n, sizes=SIZES) - balanced_mean_over_n(so, y, n, sizes=SIZES))  # 收集結果與本折的差
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); sn, so = np.concatenate(S_new), np.concatenate(S_old)  # 合併四折
    d = paired_diff(sn, so, y, pid, n, sizes=SIZES, n=5000)  # 玩家層配對 bootstrap
    cond = {"1_ci_lower_gt_0": bool(d["ci95"][0] > 0), "2_folds_3of4": bool(sum(e > 0 for e in pf) >= 3), "3_diff_ge_0.0025": bool(d["diff"] >= 0.0025)}  # M37 規則的三個條件
    rep[name] = {**d, "per_fold": pf, "base": float(balanced_mean_over_n(so, y, n, sizes=SIZES)), "questions_with_changed_score": changed, "questions": total, "conditions": cond, "pass": all(cond.values())}  # 記錄結果
    print(f"{name:22s} base {rep[name]['base']:.4f}  diff {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)  # 印出基礎集成的分數、差值與各折
          + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  score changed on {changed}/{total} questions -> {'PASS' if all(cond.values()) else 'no'}")  # 再印依局數的差與判定
    return all(cond.values())  # 回傳是否通過


def main():  # 命令列進入點
    z = dict(np.load(ROOT / "results/r_variants/m71_queries.npz", allow_pickle=False)); rep = {}  # 第 1 步的查詢表
    _, cal, _, _ = fit(z, np.ones(len(z["y"]), bool))  # 以全部四折校準（只供報告）
    for k, nm in ((0, "HD0"), (1, "HD1 target Black"), (2, "HD1 target White")):  # 逐分組列出校準表
        c = cal[k]; print(f"{nm}: a by band " + " ".join(f"{x:.2f}" for x in c["a"]) + f" | a_top {c['a_top']:.2f} | K(d=-2..2) " + " ".join(f"{c['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2))  # 印出各帶 a、a_top 與 K
                          + " | P0(d=-2..2) " + " ".join(f"{c['P0'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)) + " | K/P0 " + " ".join(f"{c['ratio'][9 + x]:.1f}" for x in (-2, -1, 0, 1, 2)))  # 再印 P0 與 K/P0
        rep[f"cal_{nm}"] = {"a": c["a"].tolist(), "a_top": c["a_top"], "K": c["K"].tolist(), "P0": c["P0"].tolist()}  # 記錄校準表
    p1 = run("base seeds 0-2", (0, 1, 2), z, rep); p2 = run("replication seeds 6-8", (6, 7, 8), z, rep)  # 基準組（種子 0-2）與複製組（種子 6-8）
    rep["adopt"] = bool(p1 and p2); print(f"ADOPT: {'YES' if p1 and p2 else 'NO'}")  # 兩組都通過才採用
    Path(ROOT / "results/r_variants/m71_step2.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
