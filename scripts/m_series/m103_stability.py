"""Section 4M M103 (2026-11-13, report only): which new changes are stable? ct75s (changes shared by two recipes) reads +1.0 point
above ct44 on the Public half while the changes of a single recipe lose points. Generalisation as stability selection: a change of
the reference recipe (bundle v5: SP2e + encoder + class-conditional tables + tempering + pi0, all bands, test-side rules, no reversal
of ct44's changes) is characterised by
  support   how many of four other recipes, built on other identifier networks, make the same change
            (SP = M88 nets + matcher, S2 = M90 seed 0 alone, S3c = seed-3 20k-step nets, S3 = the same with external players);
  loo       the share of leave-one-game-out versions of the question in which the reference recipe still makes that change;
  margin    the expected-score margin of the change under the reference posterior.
CV under the test rules (as m90_testmimic.py): precision and mean gain of the changes by each characteristic, and the scores of
composition rules against "shared with SP" (ct75s's rule). Then the same characteristics on the test set for the 20 Public
questions whose group outcome is known (A shared: +1.0 point; B: -1.26; C: -1.90) - the back-test of every rule.
  python -I scripts/m_series/m103_stability.py
"""
import json  # 寫出報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取提交檔與整理表格

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
from aicup.metric import LEVELS  # noqa: E402  匯入前須先設定 sys.path；10 個等級名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較
import m71_step2 as s2  # noqa: E402  錨點證據的分帶與校準（M71 第 2 步）
import m76_boundary as mb  # noqa: E402  邊界區決策（M76）
import m83_classcond as m83  # noqa: E402  依等級的條件證據表（M83）
import m86_bundle as b  # noqa: E402  各識別器來源與共用常數（M86）
import m87_pi0 as m87  # noqa: E402  修正後的混合權重（M87）
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  驗證折的題目與成員機率

R_ENC = 0.713; REC = {"v44": ("5e", 0, 0, 0, 0, 3, 0.851), "v3": ("SP", 1, 1, 1, 1, 1, 0.875), "v5": ("SP2e", 1, 1, 1, 1, 1, 0.852), "s2": ("S2", 1, 1, 1, 1, 1, 0.734), "s3c": ("S3c", 1, 1, 1, 1, 1, 0.855), "s3": ("S3", 1, 1, 1, 1, 1, 0.85)}  # 編碼器的測試端密度比；各配方 = (識別器, 加編碼器, 條件表, 降溫, pi0 修正, 最低分帶, 測試端密度比＝混合權重的倍數)
OTH = ("v3", "s2", "s3c", "s3"); SM = b.SM; Z10 = np.zeros(10)  # 計算支持數的四個其他配方、官方計分矩陣、無證據時的零向量


def evidence(z, v, cc, pi0, band_min, scale):  # 驗證折 v 每局的錨點對數證據，以其他折校準
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)  # 用其他折擬合分帶界線與校準
    if pi0:  # 使用 M87 修正後的混合權重
        _, cal, _ = m87.corrected(z, tr)  # 取代校準結果
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]  # 條件表（選用）與本折的查詢
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))  # 每局的邊際分帶、組別與對手標籤
    for k in (0, 1, 2):  # 三個組別（HD0、HD1 目標執黑、HD1 目標執白）
        s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)  # 混合權重 × 似然比，只用最低分帶以上
    return dict(zip(z["key"][te], out))  # {對局鍵: 對數證據 [10]}


def test_evidence(V, T, name, cc, pi0, r, floor):  # 測試端每局的錨點對數證據，以全部驗證查詢校準
    v, tt = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1] if pi0 else cal  # 校準使用全部驗證查詢
    tb = np.where(~tt["target_black"], np.searchsorted(edges[True], tt["opp_m"]), np.searchsorted(edges[False], tt["opp_m"])); tg = s2.group_of(tt["hd"], tt["target_black"]); R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10))  # 測試局的分帶、組別與條件表
    for k in (0, 1, 2):  # 三個組別
        s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = tt["opp_lab"][s]; ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio)  # 混合權重乘上測試端密度比 r，只用 floor 以上的分帶
    return logE  # [測試局數, 10]


def loo_share(P_row, G, T, new):  # 去一局穩定度
    """Share of leave-one-game-out versions (evidence of one game removed) whose region decision is still `new`."""
    if len(G) < 2:  # 少於兩局無法去一局
        return 0.0  # 視為不穩定
    L = (G.sum(0)[None, :] - G) * T; d = mb.decisions(np.repeat(P_row[None, :], len(G), 0), L, 0.5, 3.0, 0.6)[2]  # 每個版本的證據與邊界區決策
    return float((d == new).mean())  # 仍做出同一改動的比例


def margin_of(P, L, new, old):  # 改動在加入證據後驗下的期望分數差
    Pa = P * np.exp(L - L.max(1, keepdims=True)); Pa /= Pa.sum(1, keepdims=True); E = Pa @ mb.S.T; r = np.arange(len(P))  # 後驗與各答案的期望分數
    return E[r, new] - E[r, old]  # 新答案減舊答案


def table(title, key, gain, bins, names):  # 依某項特徵分組印出改動的得失
    print(f"  {title}:")  # 表格標題
    for bn, nm in zip(bins, names):  # 逐一分組
        s = bn(key); n = int(s.sum())  # 此組的改動與個數
        if n:  # 有改動才印
            print(f"      {nm:26s} {n:6d} changes ({n / len(key):.3f}); better {float((gain[s] > 0).mean()):.3f}, worse {float((gain[s] < 0).mean()):.3f}; mean gain {float(gain[s].mean()):+.3f}; total {float(gain[s].sum()):+.1f}")  # 改對、改錯比例與平均得失


def main():  # 命令列進入點
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); assert all(x[0] in V for x in REC.values()), used; Fd = {}  # 讀入六個識別器（另含編碼器）的驗證與測試輸出，確認配方用到的都在
    for v in FOLDS:  # 逐一驗證折準備證據
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; G5 = None  # 題目、每題的對局鍵與依局數的溫度
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():  # 逐一配方
            ev = evidence(V[idn], v, cc, pi0, bmin, scale); ee = evidence(V["enc"], v, 0, pi0, bmin, R_ENC) if enc else {}  # 主識別器與編碼器（選用）的每局證據
            G = [np.array([ev.get(k, Z10) + b.U_ENC * ee.get(k, Z10) for k in g_]) if len(g_) else np.zeros((0, 10)) for g_ in gk]; L = np.stack([g_.sum(0) if len(g_) else Z10 for g_ in G]); Ls[name] = L * Tn if temp else L  # 每題每局的證據、題目總證據與溫度縮放
            if name == "v5":  # 參考配方
                G5 = G  # 保留每局證據供去一局使用
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls, G5, Tn)  # 存下本折的資料
    print("CV under the test rules; reference = bundle v5's recipe; new changes = questions the ct44 recipe leaves at the base decision", flush=True); rep = {}  # CV 段的說明
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子（主評估與複製）
        rows = {k: [] for k in ("gain", "sup", "loo", "mar", "v3")}; S = {k: [] for k in ("v44", "all v5", "shared with SP (ct75s rule)", "support >= 1", "support >= 2", "support >= 3", "support = 4", "loo = 1", "loo = 1 and support >= 1", "loo = 1 or support >= 2", "shared with SP or support >= 3", "majority of the five")}; ys, ns, ps = [], [], []  # 改動的特徵與各組合規則的逐題分數
        for v in FOLDS:  # 逐一驗證折
            y, n, pid, Ls, G5, Tn = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = {}  # 標準成員的基礎機率
            for name in REC:  # 每個配方的決策
                d0, _, dec[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)  # 基礎決策與邊界區決策
            d44 = dec["v44"]; keep = d44 != d0; nr = {k: np.where(keep, d44, dec[k]) for k in REC if k != "v44"}; n5 = nr["v5"]; sel = n5 != d44; sup = sum((nr[k] == n5).astype(int) for k in OTH); mar = margin_of(P, Ls["v5"], n5, d44)  # 不改回 ct44 的改動；參考配方的新改動、支持數與期望分數差
            loo = np.zeros(len(y)); idx = np.flatnonzero(sel)  # 只對新改動算去一局穩定度
            for i in idx:  # 逐題計算
                loo[i] = loo_share(P[i], G5[i], float(Tn[i, 0]), n5[i])  # 去一局後仍做同一改動的比例
            g = SM[n5, y] - SM[d44, y]; rows["gain"].append(g[sel]); rows["sup"].append(sup[sel]); rows["loo"].append(loo[sel]); rows["mar"].append(mar[sel]); rows["v3"].append((nr["v3"] == n5)[sel])  # 每個新改動的得失、三項特徵（支持數、去一局比例、期望分數差）與是否與 SP 相同
            votes = np.stack([nr[k] for k in ("v3", "v5", "s2", "s3c", "s3")]); maj = d44.copy()  # 五個配方的投票
            for c in range(10):  # 逐一答案
                cnt = (votes == c).sum(0); maj = np.where((cnt >= 3) & (c != d44), c, maj)  # 過半數投給不同答案時改用該答案
            rule = {"v44": d44, "all v5": n5, "shared with SP (ct75s rule)": np.where(nr["v3"] == n5, n5, d44), "support >= 1": np.where(sup >= 1, n5, d44), "support >= 2": np.where(sup >= 2, n5, d44), "support >= 3": np.where(sup >= 3, n5, d44), "support = 4": np.where(sup >= 4, n5, d44),  # 各組合規則的決策：依支持數
                    "loo = 1": np.where(loo >= 1, n5, d44), "loo = 1 and support >= 1": np.where((loo >= 1) & (sup >= 1), n5, d44), "loo = 1 or support >= 2": np.where((loo >= 1) | (sup >= 2), n5, d44), "shared with SP or support >= 3": np.where((nr["v3"] == n5) | (sup >= 3), n5, d44), "majority of the five": maj}  # 依去一局穩定度、組合條件與多數決
            for k, d in rule.items():  # 每條規則
                S[k].append(SM[d, y])  # 逐題官方分數
            ys.append(y); ns.append(n); ps.append(pid)  # 留待跨折合併
        R = {k: np.concatenate(x) for k, x in rows.items()}; y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps); gain = R["gain"]; print(f"seeds {seeds}: {len(gain)} new changes of the reference recipe ({len(gain) / len(y):.4f} of the questions); mean gain {gain.mean():+.3f}")  # 合併各折並印出新改動的總覽
        table("by support (other recipes making the same change)", R["sup"], gain, [lambda k, c=c: k == c for c in range(5)], [f"support {c}" for c in range(5)])  # 依支持數分組
        table("by leave-one-game-out share", R["loo"], gain, [lambda k: k >= 1, lambda k: (k >= 0.75) & (k < 1), lambda k: (k >= 0.5) & (k < 0.75), lambda k: k < 0.5], ["stable in all", "0.75 to 1", "0.5 to 0.75", "below 0.5"])  # 依去一局穩定度分組
        qs_ = np.quantile(R["mar"], [1 / 3, 2 / 3]); table("by expected-score margin (thirds)", R["mar"], gain, [lambda k: k <= qs_[0], lambda k: (k > qs_[0]) & (k <= qs_[1]), lambda k: k > qs_[1]], [f"low (<= {qs_[0]:.3f})", "middle", f"high (> {qs_[1]:.3f})"])  # 依期望分數差三等分
        table("shared with SP or not", R["v3"], gain, [lambda k: k, lambda k: ~k], ["shared with SP", "not shared"])  # 依是否與 SP 配方相同
        print("  composition rules, class-balanced score against the ct44 recipe and against the ct75s rule:"); res = {}  # 組合規則比較的標題
        for k in S:  # 逐一組合規則
            if k == "v44":  # ct44 配方本身是比較基準
                continue  # 略過
            A_ = np.concatenate(S[k]); d1 = paired_diff(A_, np.concatenate(S["v44"]), y, pid, n, sizes=SIZES, n=3000); d2 = paired_diff(A_, np.concatenate(S["shared with SP (ct75s rule)"]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(a_, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(c_, Fd[v][0], Fd[v][1], sizes=SIZES) for a_, c_, v in zip(S[k], S["shared with SP (ct75s rule)"], FOLDS)]  # 對 ct44 配方與對 ct75s 規則的成對差，及各折差
            print(f"      {k:32s} vs ct44 recipe {d1['diff']:+.4f} | vs ct75s rule {d2['diff']:+.4f} [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" | changed vs ct44 recipe {float((A_ != np.concatenate(S['v44'])).mean()):.4f}", flush=True); res[k] = {"vs_ct44": d1["diff"], "vs_ct75s_rule": d2["diff"], "ci": d2["ci95"]}  # 印出並記錄結果
        rep[f"seeds_{seeds[0]}"] = res  # 依種子組存入報告
    # 測試端：對參考配方的新改動計算同樣的特徵
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 測試端的基礎機率（ct9）與等級轉換
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  依題號讀取某個提交檔的答案
    c9, c28, c44, c73, c75n, c75s = (sub(k) for k in ("ct9", "ct28", "ct44", "ct73", "ct75n", "ct75s")); locked = c28 != c9; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; qpos = {q: i for i, q in enumerate(qids)}  # 歷次提交檔；ct28 的推移鎖定；每題局數與溫度
    Le = test_evidence(V, T, "enc", 0, 1, R_ENC, 1); Leq = pd.DataFrame(Le).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); tdec = {}; G5t = {}; L5 = None; same_order = bool(np.array_equal(T["SP2e"]["qid"], T["enc"]["qid"]))  # 編碼器的測試證據（每題加總）；確認與 SP2e 的對局順序一致
    for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():  # 逐一配方在測試端重建決策
        if name == "v44":  # ct44 配方直接用 ct44 檔
            continue  # 略過
        Eg = test_evidence(V, T, idn, cc, pi0, scale, bmin); Lq = (pd.DataFrame(Eg).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq  # 每題總證據（含編碼器）再乘溫度
        d = np.where(locked, c28, lev[mb.decisions(P9, Lq, 0.5, 3.0, 0.6)[2]]); tdec[name] = np.where(c44 != c28, c44, d)  # 鎖定的推移照舊，ct44 的改動不改回
        if name == "v3":  # SP 配方
            print(f"check: the SP recipe rebuilt on the test differs from ct73 in {int((tdec[name] != c73).sum())} questions")  # 核對與 ct73 的差異
        if name == "v5":  # 參考配方
            L5 = Lq; assert (tdec[name] == c75n).all(), int((tdec[name] != c75n).sum()); Eg5 = Eg + (b.U_ENC * Le if same_order else 0.0); qv = T[idn]["qid"]  # 必須與 ct75n 逐題相同；組出每局證據
            for q_ in np.unique(qv):  # 逐題
                G5t[q_] = Eg5[qv == q_]  # 該題的每局證據
            print(f"check: the reference recipe rebuilt on the test equals ct75n; encoder evidence aligned game by game: {same_order}")  # 回報核對結果
    n5 = tdec["v5"]; sel = np.flatnonzero(n5 != c44); sup = sum((tdec[k] == n5).astype(int) for k in OTH); mar = margin_of(P9, L5, rk(n5), rk(c44)); pub = np.char.startswith(qids.astype(str), "pub_"); out = []  # 測試端的新改動、支持數、期望分數差與 Public 標記
    print(f"\ntest set: the reference recipe's {len(sel)} new changes (Public {int(pub[sel].sum())}, Private {int((~pub[sel]).sum())}); group of the Public ones: A = also in ct73 (8 questions, +1.0 point together), C = only in ct75n (8 questions, -1.9 points together)")  # 測試端段落的說明
    for i in sel:  # 逐一新改動
        loo = loo_share(P9[i], G5t.get(qids[i], np.zeros((0, 10))), float(Tq[i, 0]), int(rk(n5[i]))); grp = ("A" if c73[i] == n5[i] else "C") if pub[i] else ("a" if c73[i] == n5[i] else "c"); out.append((qids[i], grp, c44[i], n5[i], int(sup[i]), loo, float(mar[i]), int(nq_t[i])))  # 去一局穩定度與組別（大寫 Public、小寫 Private）
    df = pd.DataFrame(out, columns=["question", "group", "ct44", "new", "support", "loo", "margin", "games"])  # 整理成表格
    for g_ in ("A", "C", "a", "c"):  # 逐一組別
        d_ = df[df.group == g_]; print(f"  group {g_} ({len(d_)}): support counts {dict(d_.support.value_counts().sort_index())}; loo = 1: {int((d_.loo >= 1).sum())}; mean margin {d_.margin.mean():.3f}")  # 各組的特徵摘要
    print(df[df.group.isin(["A", "C"])].sort_values(["group", "support"], ascending=[True, False]).to_string(index=False))  # 列出 Public 已知結果的題目
    for name, keep in (("support >= 1", df.support >= 1), ("support >= 2", df.support >= 2), ("support >= 3", df.support >= 3), ("support = 4", df.support >= 4), ("loo = 1", df.loo >= 1), ("loo = 1 or support >= 2", (df.loo >= 1) | (df.support >= 2))):  # 回測每條規則
        print(f"  rule {name:26s} keeps of the Public A group {int((keep & (df.group == 'A')).sum())}/8, of C {int((keep & (df.group == 'C')).sum())}/8; Private: of the shared {int((keep & (df.group == 'a')).sum())}/{int((df.group == 'a').sum())}, of the others {int((keep & (df.group == 'c')).sum())}/{int((df.group == 'c').sum())}")  # 好規則應留下 A 組、丟掉 C 組
    df.to_csv(ROOT / "results/r_variants/m103_test_changes.csv", index=False); (ROOT / "results/r_variants/m103_stability.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出測試端改動表與 CV 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
