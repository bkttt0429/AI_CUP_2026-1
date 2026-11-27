"""Section 4M M87 (2026-11-11): the mixture weight a_b = 1 - nominal_b / share_b treats every query as null when it computes the
excess; with a share pi of real matches the null mass in band b is (1 - pi) nominal_b. Correction: 1 - pi = min(1, share_0 / nominal_0)
(the lowest band taken as pure null), a_b = clip(1 - (1 - pi) nominal_b / share_b, 0, 0.98), a_top and K recomputed the same way.
Identifier 5e (m82_val.npz). Primary: corrected vs current evidence, M37, seeds 0-2 and 6-8.
  python -I scripts/m_series/m87_pi0.py
"""
import json  # 寫出結果 JSON
import sys  # 設定匯入路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 加入 src 與實驗腳本的路徑
from aicup.metric import LEVELS  # noqa: E402  十個等級的名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與配對差檢定
import m71_step2 as s2  # noqa: E402  錨點證據的現行校正
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  驗證折的成員機率與題目

SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)  # 計分矩陣：完全正確 1、差一級 1/e
NOM = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # 各 margin 帶的名目占比


def corrected(z, tr):  # 依真配對占比修正的混合權重
    """s2.fit with the null mass scaled by (1 - pi) per group; returns edges, cal (a, ratio replaced), and the (1 - pi) used."""
    edges, cal, band, g = s2.fit(z, tr); d = z["opp_lab"] - z["y"]; om = {}  # 先做現行的校正；d＝對手等級減本人等級
    for k in (0, 1, 2):  # 三組：HD0、HD1 本人執黑、HD1 本人執白
        c = cal[k]; share = c["share"]; q = float(min(1.0, share[0] / NOM[0])); om[k] = q; s = tr & (g == k)  # 1 − pi＝最低帶的占比除以名目占比（上限 1）
        a = np.clip(1 - q * NOM / np.maximum(share, 1e-9), 0, 0.98); top = s & (band >= 5); a_top = float(np.clip(1 - q * NOM[5:].sum() / max(top.sum() / max(s.sum(), 1), 1e-9), 0, 0.98))  # 修正後的各帶權重與高帶合併的權重
        hist = (np.array([(d[top] == x).sum() for x in s2.D]) + 1.0) / (top.sum() + len(s2.D)); K = np.clip(hist - (1 - a_top) * c["P0"], 0, None); K = K / K.sum()  # 高帶的標籤差分布扣掉零假設的部分得到 K
        cal[k] = {**c, "a_old": c["a"], "a": a, "ratio": np.minimum(K / c["P0"], 30.0), "K": K, "a_top": a_top}  # 保留舊權重，換上修正後的 a 與比值（上限 30）
    return edges, cal, om  # 帶界、校正表與各組的 1 − pi


def evidence(z, v, fix):  # 驗證折 v 的每局對數證據（現行或修正）
    tr = z["fold"] != v; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]  # 以其餘三折校正，在折 v 上評估
    if fix:  # 修正版
        edges, cal, _ = corrected(z, tr)  # 用修正後的權重
    else:  # 現行版
        edges, cal, _, _ = s2.fit(z, tr)  # 用現行的校正
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))  # 每個查詢的 margin 帶、組別與對手等級
    for k in (0, 1, 2):  # 三組
        s = g == k; a = cal[k]["a"][band[s]][:, None]; out[s] = np.log((1 - a) + a * cal[k]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9])  # e(c)＝(1 − a)＋a·比值，取對數
    return dict(zip(z["key"][te], out))  # 局鍵→對數證據


def main():  # 修正對現行的檢定
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}  # 5e 識別器的驗證端輸出
    _, cal, om = corrected(z, np.ones(len(z["y"]), bool))  # 以全部驗證查詢看修正前後的權重
    for k, nm in ((0, "HD0"), (1, "HD1 target Black"), (2, "HD1 target White")):  # 逐組印出
        print(f"{nm}: 1 - pi = {om[k]:.3f}; a before " + " ".join(f"{x:.2f}" for x in cal[k]["a_old"]) + " | after " + " ".join(f"{x:.2f}" for x in cal[k]["a"]) + f" | a_top {cal[k]['a_top']:.2f} | K(-2..2) " + " ".join(f"{cal[k]['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)))  # 1 − pi、修正前後的 a、a_top 與 K
    EV = {fix: {v: evidence(z, v, fix) for v in FOLDS} for fix in (False, True)}; rep = {}; ok = []  # 現行與修正兩版的每局證據
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        S = {"base": [], False: [], True: []}; ys, ns, pids = [], [], []  # 基礎、現行、修正三種得分
        for v in FOLDS:  # 逐折
            y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); y = np.asarray(y)  # 題目與基礎集成的平均機率
            S["base"].append(SM[(P @ SM).argmax(1), y]); ys.append(y); ns.append(np.asarray(n)); pids.append(np.asarray(pid))  # 基礎的期望分數決策得分
            for fix in (False, True):  # 現行與修正
                L = np.zeros_like(P)  # 每題的對數證據
                for i, gk in enumerate(q["game_keys"]):  # 逐題
                    for k in gk:  # 逐局
                        if k in EV[fix][v]:  # 該局有證據
                            L[i] += EV[fix][v][k]  # 累加
                S[fix].append(SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), y])  # 融合後期望分數決策的得分
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); res = {}  # 合併四折
        for label, a, b in (("current vs base", False, "base"), ("corrected vs base", True, "base"), ("CORRECTED vs CURRENT (primary)", True, False)):  # 三組比較，主檢定為修正對現行
            dd = paired_diff(np.concatenate(S[a]), np.concatenate(S[b]), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w, yy, nn, sizes=SIZES) for x, w, yy, nn in zip(S[a], S[b], ys, ns)]  # 配對差與各折差
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}  # M37 門檻
            print(f"  seeds {seeds} {label:32s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((np.concatenate(S[a]) != np.concatenate(S[b])).mean()):.3f} -> {'PASS' if good else 'no'}")  # 記錄差、各折、各題目局數的差與改動比例
        gain = np.concatenate(S[True]) - np.concatenate(S[False]); print("  corrected - current by true class: " + " ".join(f"{LEVELS[c]} {np.mean([gain[(y == c) & (n == m)].mean() for m in SIZES]):+.4f}" for c in range(10)))  # 依真實等級的增益（各局數平均）
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["CORRECTED vs CURRENT (primary)"]["pass"])  # 收集主檢定結果
    rep["adopt"] = bool(all(ok)); print(f"PRE-REGISTERED: corrected mixture weights replace the current ones: {'YES' if rep['adopt'] else 'NO'}")  # 兩組種子都過才採用
    Path(ROOT / "results/r_variants/m87_pi0.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出結果 JSON


if __name__ == "__main__":  # 直接執行時
    main()  # 執行檢定
