"""Section 4M M77 evaluation (2026-11-10): anchors from the whole-game identity encoder (results/m77/val1-4.npz, cross-fitted: the model
for fold v never saw fold v's players' games) against the opening-model anchors (m71_queries.npz; same regime, same queries), through
M71 step 2's cross-fitted mixture model. Stage 1: held-out identification of the all-rows model (>= 0.20 top-1). Stage 2 (adoption):
encoder vs opening anchors, paired, M37 rule on seeds 0-2 and 6-8.
  python -I scripts/m_series/m77_eval.py
"""
import json  # 讀取留出準度、寫出報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數與配對比較
import m71_step2 as s2  # noqa: E402  錨點證據的校準
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折

M = ROOT / "results/m77"; rep = {}  # M77 的輸出資料夾與報告
for v in (1, 2, 3, 4, -1):  # 四個排除一折的模型與全列模型
    f = M / f"heldout_{v}.json"  # 該模型的留出準度紀錄
    if f.exists():  # 模型已訓練完成
        h = json.load(open(f)); best = max(h["history"], key=lambda e: e["top1"]); rep[f"heldout_{v}"] = best  # 取 top-1 最好的 epoch
        print(f"identification, model excluding fold {v}: {h['classes']} classes, held-out own games top-1 {best['top1']:.3f} top-5 {best['top5']:.3f} top-20 {best['top20']:.3f} (epoch {best['epoch']})")  # 印出認人準度
if "heldout_-1" in rep:  # 全列模型有結果
    t1 = rep["heldout_-1"]["top1"]; print(f"STAGE 1 (all-rows model, 8,007 candidates; opening 0.091, five-component fusion 0.125): top-1 {t1:.3f} -> {'PASS (>= 0.20)' if t1 >= 0.20 else ('fuse with the five components' if t1 >= 0.125 else 'below the fusion: close')}")  # 第 1 關的判定
parts = [dict(np.load(M / f"val{v}.npz")) for v in FOLDS]  # 四個交叉擬合模型的驗證查詢
enc = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}  # 編碼器的查詢表
o = dict(np.load(ROOT / "results/r_variants/m71_queries.npz")); opening = {k: o[k] for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}  # 開局模型的查詢表
Z = {"encoder": enc, "opening": opening}  # 兩種錨點來源
for name, z in Z.items():  # 逐來源報告帶的統計
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0  # 以全部查詢校準；HD0 組
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"])); hi, ihi = s & (band >= 6), s & (ib >= 6)  # 冒充者的帶；高於 q99 的對手與冒充者查詢
    rep[name] = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),  # 高於 q99 的占比與同級、鄰級內比例
                 "impostor_exact": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "share_above_q90": float((s & (band >= 3)).sum() / s.sum())}  # 冒充者同級率與高於 q90 的占比
    r = rep[name]; print(f"{name:8s} HD0 opponent queries above the impostor q99: {r['hd0_share_above_q99']:.4f} (above q90: {r['share_above_q90']:.3f}; impostor 0.0100 / 0.100); label agreement there exact {r['hd0_exact']:.3f} within one {r['hd0_within1']:.3f} "  # 印出帶的統計
                     f"(impostor exact {r['impostor_exact']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]) + " | K(-1,0,+1) " + " ".join(f"{x:.2f}" for x in cal[0]["K"][8:11]))  # 以及各帶 a 與 K


def scores_for(z, seeds):  # 一種錨點來源在各折的逐題得分
    out = []  # 每折一筆
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(v)  # 本折的真值、局數、玩家與題目表
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 標準成員的折外機率平均
        te, logE, _ = s2.evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}; L = np.zeros_like(P)  # 交叉擬合的查詢證據；題目的對數證據
        for i, gk in enumerate(q["game_keys"]):  # 逐題
            rows = [key_row[k] for k in gk if k in key_row]  # 這題中有查詢的局
            if rows:  # 有錨點證據
                L[i] = logE[rows].sum(axis=0)  # 各局證據相乘（對數相加）
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)  # 後驗乘上證據後正規化
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], evaluate(P, y, q, sizes=SIZES)[1], y, n, pid, L))  # 加證據與不加的得分、真值、局數、玩家、對數證據
    return out  # 回傳四折結果


def compare(a, b, label):  # 配對比較兩組得分；b 為 None 時與不加證據的基礎集成比較
    y, n, pid = (np.concatenate([x[k] for x in a]) for k in (2, 3, 4)); new = np.concatenate([x[0] for x in a]); old = np.concatenate([(w[0] if b is not None else x[1]) for x, w in zip(a, b if b is not None else a)])  # 新得分與比較基準的得分
    dd = paired_diff(new, old, y, pid, n, sizes=SIZES, n=5000)  # 玩家層配對 bootstrap
    pf = [balanced_mean_over_n(x[0], x[2], x[3], sizes=SIZES) - balanced_mean_over_n((w[0] if b is not None else x[1]), x[2], x[3], sizes=SIZES) for x, w in zip(a, b if b is not None else a)]  # 各折的差
    ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)  # M37 規則
    print(f"  {label:34s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")  # 印出比較結果
    return {**dd, "per_fold": pf, "pass": bool(ok)}  # 回傳結果與判定


def both(E, O, seeds):  # 兩種錨點證據合用（只報告）
    """Evidence of both identifiers multiplied (report only: are they complementary?)."""
    out = []  # 每折一筆
    for v, e, o_ in zip(FOLDS, E, O):  # 逐折取兩個來源的結果
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 本折資料與集成後驗
        L = 0.5 * (e[5] + o_[5]); Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)  # 兩來源的對數證據取平均後套用
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], e[1], y, n, pid, L))  # 格式同 scores_for
    return out  # 回傳四折結果


ok = []  # 各組種子是否通過第 2 關
for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
    print(f"seeds {seeds}:"); E, O = scores_for(Z["encoder"], seeds), scores_for(Z["opening"], seeds)  # 編碼器與開局錨點的得分
    r = {"encoder_vs_base": compare(E, None, "encoder anchors vs base"), "opening_vs_base": compare(O, None, "opening anchors vs base"), "encoder_vs_opening": compare(E, O, "ENCODER vs OPENING (adoption)"),  # 各自對基礎集成，以及編碼器對開局（採用依據）
         "mean_of_both_vs_opening": compare(both(E, O, seeds), O, "average of both evidences vs opening")}  # 兩者平均對開局（只報告）
    rep[f"seeds_{seeds[0]}"] = r; ok.append(r["encoder_vs_opening"]["pass"])  # 記錄並收集判定
rep["adopt_encoder"] = bool(all(ok)); print(f"STAGE 2 (pre-registered): the encoder replaces the opening model as the anchor source: {'YES' if all(ok) else 'NO'}")  # 第 2 關：兩組都通過才取代
Path(M / "m77_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告
