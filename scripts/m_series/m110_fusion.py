"""Section 4M M110 (2026-11-14): how much of the external-rank gain of M107 is left at the level of the base ensemble.
The probe's question probabilities (m107 --save-proba) are pooled into the base ensemble, P_f ~ P_base x P_probe^w, w cross-fitted
on fold halves (seeds 0-2; seeds 6-8 reuse the picks). Registered test: base + B (with external players) against base + A (same
architecture, competition players only), M37. Report: each against the base alone; the larger-body pair and lambda 3; and the
same differences after the ct75u rule (anchors unchanged) is applied on top of the pooled probabilities.
  python -I scripts/m_series/m110_fusion.py [--proba results/r_variants/m107_proba.npz]
"""
import argparse  # 解析命令列參數
import json  # 寫出報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m103_stability as m3  # noqa: E402  匯入前須先設定 sys.path；每局證據與配方設定（M103）
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0); ARMS = ("A", "B", "A_big", "B_big", "B_lambda_3")  # 沿用 m103 的模組與配方；五個投票配方、融合權重網格、探針的各臂


def rule75u(P, Ls):  # 套用 ct75u 規則（供其他實驗匯入）
    """Base decision and the ct75u rule's decision for base probabilities P (as m109_error_budget.py)."""
    dR, dA = {}, {}  # 各配方的邊界區決策與不受限決策
    for name in REC:  # 逐一配方
        d0, dA[name], dR[name], _, region = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)  # 基礎決策、兩種決策與邊界區
    d44 = dR["v44"]; keep = d44 != d0; d = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44))  # ct75s 規則：保留 ct44 的改動，SP 與 SP2e 一致才改
    for votes, ref, mask in ((np.stack([dR[k] for k in FIVE]), d44, keep), (np.stack([dA[k] for k in FIVE]), d0, ~region & ~keep)):  # 兩類題：ct44 改過的題、邊界區以外的題
        alt = ref.copy(); cnt = np.zeros(len(P), int)  # 除了參考答案以外的多數答案與票數
        for c in range(10):  # 逐一答案
            k_ = (votes == c).sum(0); better = (c != ref) & (k_ > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k_, cnt)  # 更新票數最多的其他答案
        s = mask & (cnt >= 5); d[s] = alt[s]  # 五個配方全票才改
    return d0, d  # 回傳基礎決策與 ct75u 規則決策


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--proba", default="results/r_variants/m107_proba.npz"); a_ = ap.parse_args(); Z = np.load(ROOT / a_.proba)  # 讀入 M107 各臂的驗證機率
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); Fd = {}  # 讀入各識別器的輸出
    for v in FOLDS:  # 逐一驗證折準備錨點證據
        y, n, pid, q = m3.load_fold(v); gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}  # 題目、每題的對局鍵與依局數的溫度
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():  # 逐一配方
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}  # 主識別器與編碼器（選用）的每局證據
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L  # 每題的總證據，必要時溫度縮放
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)  # 存下本折的資料
    half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; picks = None; rep = {}; ok = []  # 交叉擬合的兩半折與結果
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        S0, SU = {}, {}  # 基礎層與套用 ct75u 規則後的分數，以 (臂, w) 為鍵 -> 各折的清單
        for v in FOLDS:  # 逐一驗證折
            y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); d0, du = rule75u(P, Ls); S0.setdefault(("base", 0), []).append(SM[d0, y]); SU.setdefault(("base", 0), []).append(SM[du, y])  # 不併入探針的基準分數
            for arm in ARMS:  # 探針的每一臂
                Q = np.log(np.clip(Z[f"{arm}__{v}"].astype(np.float64), 1e-9, None)); assert len(Q) == len(P)  # 探針的對數機率，題目須對齊
                for w in WS:  # 每個融合權重
                    Pf = P * np.exp(w * (Q - Q.max(1, keepdims=True))); Pf /= Pf.sum(1, keepdims=True); d0, du = rule75u(Pf, Ls); S0.setdefault((arm, w), []).append(SM[d0, y]); SU.setdefault((arm, w), []).append(SM[du, y])  # 對數線性併入後的兩種分數
        ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); fs = lambda S, k: np.array([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731  各折的類別平衡分數
        if picks is None:  # 只在第一組種子上選權重
            picks = {}  # 每臂每折的權重
            for arm in ARMS:  # 逐一臂
                best = lambda fl: max(WS, key=lambda w: fs(S0, (arm, w))[[fi[v] for v in fl]].mean())  # noqa: E731  在指定折上基礎層分數最高的權重
                picks[arm] = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}  # 每折用另一半選出的權重
            print("weights picked on the other half of the folds (seeds 0-2): " + "  ".join(f"{a}: {[picks[a][v] for v in FOLDS]}" for a in ARMS))  # 印出選到的權重
            for arm in ARMS:  # 逐一臂
                print(f"    {arm:11s} base-level score by weight: " + "  ".join(f"w={w}: {fs(S0, (arm, w)).mean():.4f}" for w in WS))  # 各權重的基礎層分數
        cf = lambda S, arm: [S[(arm, picks[arm][v])][fi[v]] for v in FOLDS] if arm != "base" else S[("base", 0)]  # noqa: E731  取交叉擬合權重下各折的逐題分數
        print(f"\n=== seeds {seeds}: base ensemble {fs(S0, ('base', 0)).mean():.4f}; after the ct75u rule {fs(SU, ('base', 0)).mean():.4f} ===")  # 基準分數

        def cmp(S, new, old, tag):  # 比較兩臂的成對差
            A_, B_ = cf(S, new), cf(S, old); d = paired_diff(np.concatenate(A_), np.concatenate(B_), y, pid, n, sizes=SIZES, n=3000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(A_, B_, ys, ns)]  # 成對差、信賴區間與各折差
            print(f"  [{tag}] base+{new:10s} vs {('base+' + old) if old != 'base' else 'base':15s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True); rep[f"{seeds[0]}|{tag}|{new} vs {old}"] = {**d, "per_fold": pf}; return d, pf  # 印出、記錄並回傳
        d, pf = cmp(S0, "B", "A", "base level"); ok.append((d["ci95"][0] > 0) and (d["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3))  # 登記的檢定：基礎 + B 對基礎 + A（M37）
        for new, old in (("B", "base"), ("A", "base"), ("B_big", "A_big"), ("B_big", "base"), ("A_big", "base"), ("B_lambda_3", "A"), ("B_lambda_3", "base")):  # 只報告的基礎層比較
            cmp(S0, new, old, "base level")  # 基礎層的差
        for new, old in (("B", "A"), ("B", "base"), ("B_big", "A_big"), ("B_big", "base"), ("B_lambda_3", "base")):  # 只報告的 ct75u 規則後比較
            cmp(SU, new, old, "after ct75u rule")  # 套用錨點規則後的差
    passed = bool(all(ok)); print(f"\nREGISTERED TEST (base + B vs base + A at the base level, M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep["passed"] = passed; rep["picks"] = {a: {str(v): w for v, w in picks[a].items()} for a in ARMS}  # 兩組種子都過才算通過
    (ROOT / "results/r_variants/m110_fusion.json").write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
