"""Section 4M M82 evaluation (2026-11-10): merge the shards of m82_cond_id.py and compare the anchors of the four fusions (five, six =
+ tactic, 5e = five + matcher, seven) through M71 step 2's cross-fitted mixture model. Pre-registered primary: seven vs six when M81
adopted the six-component identifier (results/r_variants/m81_eval.json), else five + matcher vs five; M37 rule on seeds 0-2 and 6-8.
Sanity (pre-registered): the five-component anchors against the base must land in +0.010 .. +0.016.
  python -I scripts/m_series/m82_eval.py
"""
import json  # 讀取報告、寫出評估結果
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數與配對比較
import m71_step2 as s2  # noqa: E402  錨點證據的校準
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折

OUT = ROOT / "results/m82"; FUS = ("5", "6", "5e", "7")  # M82 的輸出資料夾與四種融合
meta = np.load(OUT / "meta.npz"); players, plab = meta["players"], meta["plab"]; Q, H, T = int(meta["Q"]), int(meta["H"]), int(meta["T"])  # 候選人、標籤與各類查詢數
parts = [np.load(p) for p in sorted(OUT.glob("part*of*.npz"))]; idx = np.concatenate([p["idx"] for p in parts]); order = np.argsort(idx)  # 讀入全部分片並依查詢索引排序
assert len(idx) == 2 * Q + 2 * T and np.array_equal(idx[order], np.concatenate([np.arange(2 * Q), np.arange(2 * Q + H, 2 * Q + H + 2 * T)])), "shards do not cover the queries"  # 分片必須剛好涵蓋全部驗證與測試查詢
col = {k: np.concatenate([p[k] for p in parts])[order] for k in parts[0].files if k != "idx"}  # 合併各欄位（b5、m5 等）
o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)  # 驗證對手側、驗證本人側、測試對手側、測試本人側的範圍
d73 = dict(np.load(ROOT / "results/r_variants/m73_val.npz")); t73 = dict(np.load(ROOT / "results/r_variants/m73_test.npz")); base = {k: d73[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}  # 沿用 M73 查詢表的基本欄位
assert len(base["key"]) == Q and len(t73["qid"]) == T  # 查詢數必須與 M73 相同
val, tst = dict(base), dict(qid=t73["qid"], hd=t73["hd"], target_black=t73["target_black"])  # 驗證與測試輸出的基本欄位
for f in FUS:  # 逐一融合
    b, m = col[f"b{f}"], col[f"m{f}"]  # top-1 候選人與 margin
    val.update({f"f{f}_opp_lab": plab[b[o]], f"f{f}_opp_m": m[o], f"f{f}_opp_best": players[b[o]], f"f{f}_imp_lab": plab[b[t_]], f"f{f}_imp_m": m[t_]})  # 驗證：對手側與冒充者的標籤、margin、候選人
    tst.update({f"f{f}_opp_lab": plab[b[to]], f"f{f}_opp_m": m[to], f"f{f}_opp_best": players[b[to]], f"f{f}_tgt_lab": plab[b[tt]], f"f{f}_tgt_m": m[tt], f"f{f}_tgt_best": players[b[tt]]})  # 測試：對手側與本人側的標籤、margin、候選人
np.savez_compressed(ROOT / "results/r_variants/m82_val.npz", **val); np.savez_compressed(ROOT / "results/r_variants/m82_test.npz", **tst)  # 寫出合併後的驗證與測試檔
rep = {"report": json.loads((OUT / "report.json").read_text()), "embed": json.loads((OUT / "embed_report.json").read_text())}; r = rep["report"]; e = rep["embed"]  # 讀入融合與嵌入兩份報告
print("matcher alone, held-out own games (all-players network): top-1 / top-5 / top-20 " + " / ".join(f"{e['heldout_all_net'][k]:.3f}" for k in ("top1", "top5", "top20"))  # 印出配對器單獨的準度
      + " | players the network never saw vs the same players seen, top-1: " + "; ".join(f"fold {v}: {e[f'unseen_fold{v}']['unseen']['top1']:.3f} vs {e[f'unseen_fold{v}']['seen']['top1']:.3f}" for v in (1, 2, 3, 4)))  # 以及沒看過對看過的比較
for name, w in r["weights"].items():  # 逐一融合
    print(f"weights ({name}): " + " ".join(f"{k} {v:.2f}" for k, v in w.items()))  # 印出融合權重
print("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k, v in r["heldout_accuracy"].items()))  # 印出各成分與融合的認人準度
print(f"same top candidate as M73's five-component run (different held-out games): validation opponent side {float((val['f5_opp_best'] == d73['f_opp_best']).mean()):.3f}")  # 驗證對手側與 M73 五成分 top-1 候選人的一致率（兩者的留出局不同，只看相近程度）
Z = {f: {**base, "opp_lab": val[f"f{f}_opp_lab"], "opp_m": val[f"f{f}_opp_m"], "imp_m": val[f"f{f}_imp_m"], "imp_lab": val[f"f{f}_imp_lab"]} for f in FUS}  # 每種融合的錨點查詢表
for name, z in Z.items():  # 逐一融合報告帶的統計
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0  # 以全部查詢校準；HD0 組
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"]))  # 冒充者查詢的帶
    hi, ihi = s & (band >= 6), s & (ib >= 6)  # 高於 q99 的對手與冒充者查詢
    rep[f"bands_{name}"] = q = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact_above_q99": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1_above_q99": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),  # 高於 q99 的占比與同級、鄰級內比例
                                "impostor_exact_above_q99": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "a_by_band_hd0": cal[0]["a"].tolist()}  # 冒充者同級率與 HD0 各帶的 a
    print(f"{name:3s} HD0: opponent queries above the impostor q99 {q['hd0_share_above_q99']:.4f}; exact / within-one agreement there {q['hd0_exact_above_q99']:.3f} / {q['hd0_within1_above_q99']:.3f} "  # 印出帶的統計
          f"(impostor exact {q['impostor_exact_above_q99']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]))  # 以及各帶的 a


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
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], evaluate(P, y, q, sizes=SIZES)[1], y, n, pid))  # 加證據與不加的得分、真值、局數、玩家
    return out  # 回傳四折結果


def compare(a, b, label):  # 配對比較兩組得分；b 為 None 時與不加證據的基礎集成比較
    y, n, pid = (np.concatenate([x[k] for x in a]) for k in (2, 3, 4))  # 合併四折的真值、局數與玩家
    dd = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] if b is not None else x[1] for x in (b if b is not None else a)]), y, pid, n, sizes=SIZES, n=5000)  # 玩家層配對 bootstrap
    pf = [balanced_mean_over_n(x[0], x[2], x[3], sizes=SIZES) - balanced_mean_over_n((w[0] if b is not None else x[1]), x[2], x[3], sizes=SIZES) for x, w in zip(a, b if b is not None else a)]  # 各折的差
    cond = (dd["ci95"][0] > 0) and (sum(e_ > 0 for e_ in pf) >= 3) and (dd["diff"] >= 0.0025)  # M37 規則
    print(f"  {label:34s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e_:+.4f}" for e_ in pf) + "  by n " + " ".join(f"{m}: {v_:+.4f}" for m, v_ in dd["by_n"].items()) + f"  -> {'PASS' if cond else 'no'}")  # 印出比較結果
    return {**dd, "per_fold": pf, "pass": bool(cond)}  # 回傳結果與判定


m81 = ROOT / "results/r_variants/m81_eval.json"; six = bool(json.loads(m81.read_text())["adopt_six"]) if m81.exists() else False  # M81 是否採用了六成分識別器
prim = ("7", "6") if six else ("5e", "5"); ok, sane = [], []  # 事先登錄的主比較；判定與合理性檢查
print(f"M81 adopted the six-component identifier: {'yes' if six else 'no'} -> primary = {prim[0]} vs {prim[1]}")  # 印出主比較
for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
    print(f"seeds {seeds}:"); S = {f: scores_for(Z[f], seeds) for f in FUS}  # 四種融合的逐題得分
    res = {f"{f}_vs_base": compare(S[f], None, f"{f} anchors vs base") for f in FUS}  # 各融合對基礎集成
    res["5e_vs_5"] = compare(S["5e"], S["5"], "five + matcher vs five" + (" (PRIMARY)" if not six else "")); res["7_vs_6"] = compare(S["7"], S["6"], "seven vs six" + (" (PRIMARY)" if six else "")); res["6_vs_5"] = compare(S["6"], S["5"], "six vs five (M81's contrast)")  # 加配對器、加戰術成分的比較
    rep[f"seeds_{seeds[0]}"] = res; ok.append(res[f"{prim[0]}_vs_{prim[1]}"]["pass"]); sane.append(0.010 <= res["5_vs_base"]["diff"] <= 0.016)  # 記錄；主比較判定與五成分增益的合理範圍
rep["sanity_five"] = bool(all(sane)); rep["primary"] = f"{prim[0]}_vs_{prim[1]}"; rep["adopt_matcher"] = bool(all(ok) and all(sane))  # 兩組都通過且合理才採用配對器
print(f"sanity (five-component anchors vs base inside +0.010 .. +0.016 on both seed sets): {'OK' if all(sane) else 'FAILED'}")  # 印出合理性檢查
print(f"PRE-REGISTERED: add the matcher to the identifier: {'YES' if rep['adopt_matcher'] else 'NO'}")  # 印出事先登錄的採用判定
Path(ROOT / "results/r_variants/m82_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告
