"""Section 4M (2026-11-10): test-side candidate from a given opponent identifier (fused five components: m73_*.npz; later the encoder).
 1 label-free density ratio r: within-one agreement of the matched label with the model's own decision, bands above the impostor q95,
   test opponents vs validation opponents, both net of the impostors (as m71_test_diag A)
 2 CV expectation at density r (weights x r): anchors on all levels / inside the boundary region only / pairwise act-at-0.6
 3 files: ct<A> = ct28 locked on the pushed questions + anchored decision inside the boundary region (bands >= q90, a = r x validation a);
          ct<B> = the same with the pairwise posterior >= 0.6 rule
  python -I scripts/m_series/m78_build_candidate.py --source fused --out 42 43
"""
import argparse  # 解析命令列參數
import hashlib  # 計算提交檔的 md5
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀寫提交檔與機率檔

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序
from aicup.qeval import SIZES, paired_diff  # noqa: E402  題目局數清單與配對 bootstrap
from aicup.submission import write_submission  # noqa: E402  寫出提交檔
import m71_cv_density as cd  # noqa: E402  錨點稀疏化
import m71_step2 as s2  # noqa: E402  錨點證據的校準
import m76_boundary as mb  # noqa: E402  邊界區域的決策規則
from ens_compare import ES, load_fold  # noqa: E402  折外機率與驗證折
from m43_gate import decide  # noqa: E402  期望分決策

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=("fused", "encoder", "matcher", "file"), default="fused"); ap.add_argument("--out", nargs=2, default=["42", "43"])  # 識別器來源與兩個輸出檔的編號
ap.add_argument("--val"); ap.add_argument("--test"); ap.add_argument("--prefix", default="f_"); ap.add_argument("--ref", default=None); a = ap.parse_args()  # file 來源的路徑、欄位前綴與比較用的參考檔
REF = a.ref or ("ct42" if a.source == "matcher" else "ct41")  # 預設的參考檔
if a.source == "file":  # M84：任一識別器、採 m82_eval.py 格式的檔案（--val、--test、--prefix）
    d = dict(np.load(a.val)); t = dict(np.load(a.test)); p_ = a.prefix  # 讀入驗證與測試檔
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d[p_ + "opp_lab"], "opp_m": d[p_ + "opp_m"], "imp_lab": d[p_ + "imp_lab"], "imp_m": d[p_ + "imp_m"]}  # 驗證查詢表
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t[p_ + "opp_lab"], "opp_m": t[p_ + "opp_m"], "tgt_lab": t[p_ + "tgt_lab"], "tgt_m": t[p_ + "tgt_m"]}  # 測試查詢表
elif a.source == "matcher":  # M82：五成分＋學習式條件化配對器（m82_eval.py 的檔案，前綴 f5e_）
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); t = dict(np.load(ROOT / "results/r_variants/m82_test.npz"))  # 讀入 M82 的輸出
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}  # 驗證查詢表
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t["f5e_opp_lab"], "opp_m": t["f5e_opp_m"], "tgt_lab": t["f5e_tgt_lab"], "tgt_m": t["f5e_tgt_m"]}  # 測試查詢表
elif a.source == "fused":  # M73 的五成分融合
    d = dict(np.load(ROOT / "results/r_variants/m73_val.npz")); t = dict(np.load(ROOT / "results/r_variants/m73_test.npz"))  # 讀入 M73 的輸出
    v = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f_opp_lab"], "opp_m": d["f_opp_m"], "imp_lab": d["f_imp_lab"], "imp_m": d["f_imp_m"]}  # 驗證查詢表（融合欄位 f_）
    T = {"qid": t["qid"], "hd": t["hd"], "target_black": t["target_black"], "opp_lab": t["f_opp_lab"], "opp_m": t["f_opp_m"], "tgt_lab": t["f_tgt_lab"], "tgt_m": t["f_tgt_m"]}  # 測試查詢表
else:  # encoder：M77 的整局編碼器
    parts = [dict(np.load(ROOT / f"results/m77/val{f}.npz")) for f in (1, 2, 3, 4)]; v = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "opp_best"}  # 合併四折的驗證查詢
    t = dict(np.load(ROOT / "results/m77/test.npz")); T = {k: t[k] for k in ("qid", "hd", "target_black", "opp_lab", "opp_m", "tgt_lab", "tgt_m")}  # 測試查詢表
edges, cal, vband, vg = s2.fit(v, np.ones(len(v["y"]), bool))  # 以全部驗證查詢校準，供測試端使用
band_of = lambda m, model_black: np.where(model_black, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731  依模型顏色把 margin 換成帶
vb_imp = band_of(v["imp_m"], v["target_black"]); tb = band_of(T["opp_m"], ~T["target_black"]); tg = s2.group_of(T["hd"], T["target_black"])  # 驗證冒充者的帶；測試對手的帶與分組
dec = {}  # 驗證玩家 n = 20 題的模型決策
for f in (1, 2, 3, 4):  # 逐一驗證折
    y, n, pid, q = load_fold(f); P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dd = np.asarray(decide(P), int)  # 種子 0-2 的期望分決策
    dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, dd) if n_ == 20})  # 只記 n = 20 的題
vd = np.array([dec[p_] for p_ in v["pid"]]); ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(RANK); td = ct9.reindex(T["qid"]).to_numpy()  # 驗證查詢取所屬 n = 20 題的模型決策；測試查詢取 ct9 檔中所屬題目的答案
num = den = 0.0  # 吻合法密度比的分子與分母
for b in range(4, 9):  # 高於冒充者 q95 的各帶
    w1 = lambda lab, bb, d_, s: float((np.abs(lab[s & (bb == b)] - d_[s & (bb == b)]) <= 1).mean()) if (s & (bb == b)).any() else np.nan  # noqa: E731  本帶的鄰級內吻合率
    av, ai, at = w1(v["opp_lab"], vband, vd, vg >= 0), w1(v["imp_lab"], vb_imp, vd, vg >= 0), w1(T["opp_lab"], tb, td, tg >= 0); w = int(((tg >= 0) & (tb == b)).sum())  # 驗證對手、驗證冒充者、測試對手；本帶的測試查詢數
    if w and np.isfinite([av, ai, at]).all():  # 三者都有值才計入
        num += w * (at - ai); den += w * (av - ai)  # 扣掉冒充者水準後加權累加
r_agree = float(np.clip(num / den, 0, 1)); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # 吻合法的密度比；各帶的名目占比
sh_v, sh_t = float((vband[vg >= 0] >= 3).mean() - nominal[3:].sum()), float((tb[tg >= 0] >= 3).mean() - nominal[3:].sum())  # 驗證與測試在 q90 以上的超額占比
# 在 CV 中權重偏低幾乎沒有損失（滿密度、權重 x0.74：+0.0080，權重相符為 +0.0078），權重偏高則約損失 0.001，所以取兩個不用標籤的估計中「較低」的那一個
r = float(min(r_agree, np.clip(sh_t / sh_v, 0, 1)))  # 吻合法與占比法取較低者
print(f"source {a.source}: density ratio: agreement-based {r_agree:.3f}, share-based {sh_t / sh_v:.3f} -> used r = {r:.3f} (the lower one); excess above q90: test {sh_t:+.4f}, validation {sh_v:+.4f}")  # 印出兩種密度比與採用的 r
# 只報告：冒充者的虛無分布能否套用到測試端？測試題「本人」那一側對候選池（他們的玩家不在池內）
tt_b = band_of(T["tgt_m"], T["target_black"]); print("null check, share of own-side queries at or above the impostor q90 / q99 / q99.9 (nominal 0.100 / 0.010 / 0.001): test " + " / ".join(f"{float((tt_b[tg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8))  # 測試本人側的帶；印出各分位以上的占比
                                                      + "; validation " + " / ".join(f"{float((vb_imp[vg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8)) + "; opponent side: test " + " / ".join(f"{float((tb[tg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8))  # 驗證冒充者與測試對手側
                                                      + "; validation " + " / ".join(f"{float((vband[vg >= 0] >= b).mean()):.4f}" for b in (3, 6, 8)))  # 驗證對手側
print(f"CV expectation at density {r:.2f} with weights x{r:.2f} (three thinning draws):")  # CV 期望的標題
for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
    acc = {"all levels": [], "region only": [], "pair >= 0.6": []}  # 三種決策方式的結果
    for draw in (1, 2, 3):  # 三次稀疏抽樣
        data = mb.fold_data(v, cd.thin(v, r, draw), r, seeds); y, n, pid = (np.concatenate([x[k] for x in data]) for k in (0, 1, 2)); cols = {k: [] for k in acc}; d0s = []  # 密度 r、權重 × r 的各折資料
        for yy, nn, pp, P, L in data:  # 逐折
            d0, dA, dAr, dB, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); d0s.append(d0); cols["all levels"].append(dA); cols["region only"].append(dAr); cols["pair >= 0.6"].append(dB)  # 收集各種決策
        d0 = np.concatenate(d0s)  # 基礎集成的決策
        for k in acc:  # 逐一決策方式
            dk = np.concatenate(cols[k]); q = paired_diff(mb.S[dk, y], mb.S[d0, y], y, pid, n, sizes=SIZES, n=2000); acc[k].append((q["diff"], q["ci95"][0], float((dk != d0).mean())))  # 對基礎集成決策的配對差與改動比例
    print(f"  seeds {seeds}: " + " | ".join(f"{k}: {np.mean(x, axis=0)[0]:+.4f} (CI lower {np.mean(x, axis=0)[1]:+.4f}, changed {np.mean(x, axis=0)[2]:.3f})" for k, x in acc.items()))  # 印出三次抽樣的平均
# 測試端的證據
logE = np.zeros((len(tb), 10))  # 測試查詢的對數證據
for k in (0, 1, 2):  # 三個分組
    s = tg == k; aw = r * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= 3); dd = T["opp_lab"][s][:, None] - np.arange(10)[None, :]  # 權重乘 r、只用 >= q90 的帶；標籤差
    logE[s] = np.log((1 - aw[tb[s]])[:, None] + aw[tb[s]][:, None] * cal[k]["ratio"][dd + 9])  # 測試查詢的對數證據 log e(c)
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); cols_ = [f"p_{lv}" for lv in LEVELS]  # ct9 的測試機率
L = pd.DataFrame(logE).groupby(T["qid"]).sum().reindex(pr.question_id).fillna(0.0).to_numpy(); P = pr[cols_].to_numpy(float)  # 依題號加總證據並對齊機率
d0, dA, dAr, dB, region = mb.decisions(P, L, 0.5, 3.0, 0.6)  # 測試題的各種決策
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(pr.question_id).to_numpy()  # noqa: E731  讀入提交檔並對齊題目順序
c9, c28, c41 = sub("ct9"), sub("ct28"), sub(REF); lev = np.array(LEVELS); assert (lev[d0] == c9).all()  # 參考檔；確認基礎集成的決策重現 ct9
locked = c28 != c9; pub = pr.question_id.str.startswith("pub_").to_numpy()  # 鎖定 ct28 推移翻動的題；Public 題遮罩
for name, dk in ((a.out[0], dAr), (a.out[1], dB)):  # 兩個輸出：只在區域內、成對 0.6
    final = np.where(locked, c28, lev[dk]); write_submission(list(zip(pr.question_id, final)), ROOT / f"submission_rank_ct{name}.csv")  # 鎖定題用 ct28，其餘用新決策
    for ref_n, ref in (("ct28", c28), (REF, c41)):  # 與 ct28 及參考檔比較
        ch = final != ref; print(f"ct{name} vs {ref_n}: {int(ch.sum())} differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())})" + (("; moves " + str(pd.Series([f'{x}->{y_}' for x, y_ in zip(ref[ch], final[ch])]).value_counts().head(12).to_dict())) if ref_n == "ct28" else ""))  # 印出差異題數與推移方向
    jump = np.abs(np.array([RANK[x] for x in final]) - np.array([RANK[x] for x in c28])); print(f"   changes vs ct28 by size: one level {int((jump == 1).sum())}, two or more {int((jump >= 2).sum())}; md5 {hashlib.md5((ROOT / f'submission_rank_ct{name}.csv').read_bytes()).hexdigest()[:12]}")  # 改動級數與 md5
Pn = P * np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); out = pr.copy(); out[cols_] = Pn; out.to_csv(ROOT / f"submission_rank_ct{a.out[0]}_proba.csv", index=False)  # 寫出加入錨點證據後的機率檔
print(f"boundary region covers {float(region.mean()):.3f} of the test questions; locked {int(locked.sum())}")  # 邊界區域涵蓋率與鎖定題數
