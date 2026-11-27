"""Section 4M M86 (2026-11-11): bundle test of small gains that act at different stages of the anchor pipeline. Switches on top of
ct44's pipeline (5e identifier, pooled evidence, current mixture weights, expected-score decision):
  id    identifier: one of --ids (5e = ct44's; P3 = all games + phase rows + 3 seeds; S / SP = M88's sequence model, alone / with P3)
  enc   the whole-game encoder's (M77) evidence added with weight 0.5 (its own cross-fitted mixture calibration)
  cc    class-conditional evidence tables (M83, kappa 50) for the main identifier
  temp  the question's anchor log-evidence times T_n = 1.0 / 1.0 / 0.6 for n = 5 / 10 / 20 (15 -> 0.8 on the test)
  pi0   (with --pi0) M87's corrected mixture weights, for both evidence sources
Selection and estimation use different folds: the best combination on folds {1, 2} (class-balanced score, seeds 0-2) is scored on folds
{3, 4} against "all off", and the other way round; the two halves together are the cross-fitted estimate. Rule: CI lower bound > 0,
difference >= 0.0025, >= 3 of 4 folds positive, on seeds 0-2 and 6-8 (same picks). If it passes, the combination chosen on all four
folds is written as --out (ct28 locked + boundary region, each source with its own lower density ratio, evidence from the margin bands
>= --floor: 3 = q90 as ct44, 1 = every band with a positive weight, which is what the CV comparison scores); the builder must reproduce
ct44 exactly with every switch off and floor 3.
  v1 (ct64):  python -I scripts/m_series/m86_bundle.py --ids 5e,P3 --floor 3 --out ct64 --json m86_bundle.json
  v2 (ct70):  ... --ids 5e,P3 --pi0 --floor 3 --out ct70 --json m86b_bundle.json
  v2, all bands (ct71):  ... --ids 5e,P3 --pi0 --floor 1 --out ct71 --json m86c_bundle.json
  v3 (ct72):  ... --ids 5e,P3,S,SP --pi0 --floor 1 --out ct72 --json m86d_bundle.json
"""
import argparse  # 解析命令列參數
import hashlib  # 計算候選檔的 md5
import itertools  # 列舉所有開關組合
import json  # 寫出結果 JSON
import subprocess  # 結束時呼叫 m84_variant.py 更新上傳指示
import sys  # 設定匯入路徑、取得目前的直譯器
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀寫提交檔與機率檔

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 加入 src 與實驗腳本的路徑
from aicup.metric import LEVELS  # noqa: E402  十個等級的名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與配對差檢定
from aicup.submission import write_submission  # noqa: E402  寫出提交檔
import m71_step2 as s2  # noqa: E402  錨點證據的校正（不分等級）
import m76_boundary as mb  # noqa: E402  邊界區的決策規則
import m83_classcond as m83  # noqa: E402  依等級的證據表
import m87_pi0 as m87  # noqa: E402  修正後的混合權重
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  驗證折的成員機率與題目
from m43_gate import decide  # noqa: E402  基礎的期望分數決策

RV = ROOT / "results/r_variants"; RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 識別器輸出的資料夾；等級→索引
TN = {5: 1.0, 10: 1.0, 15: 0.8, 20: 0.6}; U_ENC = 0.5  # 依題目局數的降溫係數；編碼器證據的權重
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)  # 計分矩陣：完全正確 1、差一級 1/e


def sources(ids):  # 讀入各識別器在驗證端與測試端的輸出
    d82 = dict(np.load(RV / "m82_val.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; t82 = dict(np.load(RV / "m82_test.npz")); tb_ = {k: t82[k] for k in ("qid", "hd", "target_black")}  # M82（5e）的驗證／測試輸出與共用欄位
    parts = [dict(np.load(ROOT / f"results/m77/val{v}.npz")) for v in FOLDS]; e = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}; te = dict(np.load(ROOT / "results/m77/test.npz"))  # M77 整局編碼器的驗證（各折合併）與測試輸出
    V = {"5e": {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_lab": d82["f5e_imp_lab"], "imp_m": d82["f5e_imp_m"]}, "enc": e}  # 驗證端：5e 與編碼器
    T = {"5e": {**tb_, "opp_lab": t82["f5e_opp_lab"], "opp_m": t82["f5e_opp_m"]}, "enc": {k: te[k] for k in ("qid", "hd", "target_black", "opp_lab", "opp_m")}}; used = ["5e"]  # 測試端：5e 與編碼器；used 記錄實際可用的識別器
    for name in ids:  # 逐一要求的識別器
        if name == "5e":  # 5e 已經讀入
            continue  # 略過
        fv, ft = RV / f"m84_{name}_val.npz", RV / f"m84_{name}_test.npz"  # m84_variant.py 寫出的識別器輸出
        if not (fv.exists() and ft.exists()):  # 缺檔時
            print(f"identifier {name}: files missing, left out", flush=True); continue  # 提示後略過該識別器
        dp, tp = dict(np.load(fv)), dict(np.load(ft)); assert np.array_equal(dp["key"], base["key"])  # 確認查詢順序與 M82 相同
        V[name] = {**base, "opp_lab": dp["f_opp_lab"], "opp_m": dp["f_opp_m"], "imp_lab": dp["f_imp_lab"], "imp_m": dp["f_imp_m"]}; T[name] = {**tb_, "opp_lab": tp["f_opp_lab"], "opp_m": tp["f_opp_m"]}; used.append(name)  # 該識別器在驗證端與測試端的等級與 margin（基本欄位沿用 M82），並記為可用
    return V, T, dict(zip(base["key"], base["pid"])), used  # 驗證、測試、局鍵→玩家、可用的識別器


def evidence(z, v, cc, pi0):  # 驗證折 v 的每局對數證據
    """{game_key: log e [10]} of fold v, calibrated on the other folds: pooled or class-conditional tables, current or corrected weights."""
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)  # 以其餘三折擬合帶界與校正表
    if pi0:  # pi0 開關
        _, cal, _ = m87.corrected(z, tr)  # 換成 M87 修正後的混合權重
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]  # cc 開關時建依等級的證據表；取折 v 的查詢
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))  # 每個查詢的 margin 帶、組別與對手等級
    for k in (0, 1, 2):  # 三組：HD0、HD1 本人執黑、HD1 本人執白
        s_ = g == k; a = cal[k]["a"][band[s_]][:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)  # e(c)＝(1 − a)＋a·比值，取對數
    return dict(zip(z["key"][te], out))  # 局鍵→對數證據


def main():  # 打包檢定與建檔
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", default="5e,P3"); ap.add_argument("--pi0", action="store_true"); ap.add_argument("--floor", type=int, default=3); ap.add_argument("--out", default="ct64"); ap.add_argument("--json", default="m86_bundle.json"); ap.add_argument("--require", default="", help="identifiers that must be present; the run stops if one is missing"); a_ = ap.parse_args()  # 命令列：識別器清單、pi0 開關、證據帶下限、輸出檔名、結果 JSON、必要的識別器
    V, T, key_pid, IDS = sources(a_.ids.split(",")); missing = [x for x in a_.require.split(",") if x and x not in IDS]  # 讀入識別器並檢查必要者是否齊全
    if missing:  # 缺少必要的識別器
        sys.exit(f"required identifier(s) missing: {missing}; nothing written")  # 停止，不寫任何檔
    PS = (0, 1) if a_.pi0 else (0,); COMBOS = list(itertools.product(range(len(IDS)), (0, 1), (0, 1), (0, 1), PS)); off = COMBOS[0]; rep = {"ids": IDS, "floor": a_.floor}  # 所有組合（識別器、enc、cc、temp、pi0）；off＝全關
    label = lambda c: "+".join(x for x in ((f"id={IDS[c[0]]}" if c[0] else ""), "enc" * c[1], "cc" * c[2], "temp" * c[3], "pi0" * c[4]) if x) or "all off"  # noqa: E731  組合的文字標籤
    F = {}  # 各折預先算好的每題證據
    for v in FOLDS:  # 驗證折 1–4
        y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y)  # 該折的題目與各題的對局

        def qsum(ev):  # 把每局證據加總成每題證據
            L = np.zeros((nq, 10))  # 每題 10 級的對數證據
            for i, g in enumerate(gk):  # 逐題
                for k in g:  # 逐局
                    if k in ev:  # 該局有證據
                        L[i] += ev[k]  # 累加
            return L  # 回傳每題證據
        F[v] = {"y": np.asarray(y), "n": np.asarray(n), "pid": np.asarray(pid), "main": {(a, c, p_): qsum(evidence(V[IDS[a]], v, c, p_)) for a in range(len(IDS)) for c in (0, 1) for p_ in PS}, "enc": {p_: qsum(evidence(V["enc"], v, 0, p_)) for p_ in PS}}  # 主識別器（各 id／cc／pi0）與編碼器的每題證據
        F[v]["T"] = np.array([TN[int(m)] for m in F[v]["n"]])[:, None]  # 依題目局數的降溫係數
    L_of = lambda v, c: (F[v]["main"][(c[0], c[2], c[4])] + c[1] * U_ENC * F[v]["enc"][c[4]]) * (F[v]["T"] if c[3] else 1.0)  # noqa: E731  組合 c 在折 v 的每題證據
    half = (list(FOLDS[:2]), list(FOLDS[2:])); picks = None; ok = []  # 兩半：折 {1, 2} 與 {3, 4}
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子（第二組沿用第一組的選擇）
        S = {}  # (組合, 折) -> 每題得分
        for v in FOLDS:  # 逐折
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 基礎集成的平均機率
            for c in COMBOS:  # 逐一組合
                L = L_of(v, c); S[(c, v)] = SM[((P * np.exp(L - L.max(axis=1, keepdims=True))) @ SM).argmax(1), F[v]["y"]]  # 融合後期望分數決策的得分
        fs = {c: np.array([balanced_mean_over_n(S[(c, v)], F[v]["y"], F[v]["n"], sizes=SIZES) for v in FOLDS]) for c in COMBOS}  # 各組合在各折的類別平衡分數
        if picks is None:  # 只在種子 0-2 上選擇
            fi = {v: i for i, v in enumerate(FOLDS)}; best = lambda fl: max(COMBOS, key=lambda c: fs[c][[fi[v] for v in fl]].mean())  # noqa: E731  在指定折上平均分數最高的組合
            picks = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}; final = best(list(FOLDS))  # 每折用另一半選出的組合；另取四折全體的最佳者
            print(f"scores of the {len(COMBOS)} combinations (mean over the four folds, seeds 0-2; difference to all off; the best 14 and the single switches):")  # 分數表的標題
            singles = [c for c in COMBOS if sum(1 for x in c if x) == 1]  # 只開一個開關的組合
            for c in sorted(COMBOS, key=lambda c: -fs[c].mean())[:14] + singles:  # 前 14 名與各單一開關
                print(f"  {label(c):30s} {fs[c].mean():.4f}  {fs[c].mean() - fs[off].mean():+.4f}   folds " + " ".join(f"{x:+.4f}" for x in fs[c] - fs[off]))  # 平均分數、對全關的差與各折差
            print(f"picked on folds {half[0]}: {label(best(half[0]))} | picked on folds {half[1]}: {label(best(half[1]))} | on all four: {label(final)}")  # 兩半與全體各自選到的組合
            rep["table"] = {label(c): {"mean": float(fs[c].mean()), "diff": float(fs[c].mean() - fs[off].mean()), "per_fold_diff": (fs[c] - fs[off]).tolist()} for c in COMBOS}  # 全部組合的分數表
            rep["picks"] = {"from_first_half": label(best(half[0])), "from_second_half": label(best(half[1])), "all_folds": label(final)}  # 記下選擇
        y, n, pid = (np.concatenate([F[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); new = np.concatenate([S[(picks[v], v)] for v in FOLDS]); old = np.concatenate([S[(off, v)] for v in FOLDS])  # 交叉擬合的組合與全關的每題得分
        dd = paired_diff(new, old, y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[picks[v]][i] - fs[off][i]) for i, v in enumerate(FOLDS)]; good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)  # 判定：CI 下界 > 0、差 ≥ 0.0025、≥ 3/4 折為正
        print(f"seeds {seeds}: cross-fitted bundle vs all off {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((new != old).mean()):.3f} -> {'PASS' if good else 'no'}")  # 記錄差、各折、各題目局數的差與改動比例
        fb = np.concatenate([S[(final, v)] for v in FOLDS]); df = paired_diff(fb, old, y, pid, n, sizes=SIZES, n=5000)  # 全體選出的組合對全關（未交叉擬合，只報告）
        print(f"seeds {seeds}: (not cross-fitted, report) the all-folds pick {label(final)} vs all off {df['diff']:+.4f} [{df['ci95'][0]:+.4f}, {df['ci95'][1]:+.4f}]")  # 印出只報告的比較
        rep[f"seeds_{seeds[0]}"] = {"bundle_vs_off": {**dd, "per_fold": pf, "pass": bool(good)}, "final_pick_vs_off_not_crossfitted": df}; ok.append(good)  # 收集結果
    passed = bool(all(ok) and final != off); rep["passed_cv"] = passed; print(f"PRE-REGISTERED: bundle passes: {'YES' if passed else 'NO'}")  # 兩組都過且選到的不是全關才算通過
    # ---- 測試端
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); dec = {}  # ct9（基礎集成）的測試題機率
    for f in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(f); P = np.mean([np.load(ES / f"val{f}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(decide(P), int)) if n_ == 20})  # 各驗證玩家 20 局題的基礎決策（估計密度比用）
    ct9 = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # ct9 的測試決策；各 margin 帶的名目占比

    def test_evidence(name, cc, pi0, floor):  # 一個來源在測試端的每題證據與密度比
        """Question-level log-evidence [800, 10] of one source on the test (m78_build_candidate.py's rules) and its density ratio."""
        v, t = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1] if pi0 else cal  # 以全部驗證查詢校正；pi0 時用修正後的權重
        band_of = lambda m, mb_: np.where(mb_, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731  依顏色取帶界算 margin 帶
        vb_imp = band_of(v["imp_m"], v["target_black"]); tb = band_of(t["opp_m"], ~t["target_black"]); tg = s2.group_of(t["hd"], t["target_black"]); assert all(k in key_pid for k in v["key"]), f"{name}: validation keys outside the anchor query set"  # 冒充者與測試對手的帶；確認驗證查詢都屬於錨點題集
        vd = np.array([dec[key_pid[k]] for k in v["key"]]); td = ct9.reindex(t["qid"]).to_numpy(); num = den = 0.0  # 驗證與測試查詢所屬題目的基礎決策
        for b in range(4, 9):  # 高把握的帶（索引 4–8）
            w1 = lambda lab, bb, d_, s: float((np.abs(lab[s & (bb == b)] - d_[s & (bb == b)]) <= 1).mean()) if (s & (bb == b)).any() else np.nan  # noqa: E731  該帶裡配到的等級與決策相差 ≤ 1 的比例
            av, ai, at = w1(v["opp_lab"], vband, vd, vg >= 0), w1(v["imp_lab"], vb_imp, vd, vg >= 0), w1(t["opp_lab"], tb, td, tg >= 0); w = int(((tg >= 0) & (tb == b)).sum())  # 驗證對手、冒充者、測試對手的一致率；測試查詢數當權重
            if w and np.isfinite([av, ai, at]).all():  # 三個比例都有值才計入
                num += w * (at - ai); den += w * (av - ai)  # 高於冒充者的一致率：測試對驗證
        r_agree = float(np.clip(num / den, 0, 1)); sh_v, sh_t = float((vband[vg >= 0] >= 3).mean() - nominal[3:].sum()), float((tb[tg >= 0] >= 3).mean() - nominal[3:].sum()); r = float(min(r_agree, np.clip(sh_t / sh_v, 0, 1)))  # 兩種密度比估計（一致率、高帶超額占比）取較低者
        R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10))  # cc 時的依等級證據表；測試查詢的對數證據
        for k in (0, 1, 2):  # 三組：HD0、HD1 本人執黑、HD1 本人執白
            s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = t["opp_lab"][s]  # 權重乘上密度比，低於 floor 的帶歸零
            ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio)  # 每局的對數證據
        return pd.DataFrame(logE).groupby(t["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(), r  # 依題加總成題目層證據，並回傳密度比
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  讀提交檔並對齊到 qids（ct9 機率檔）的題目順序
    c9, c28, c44 = sub("ct9"), sub("ct28"), sub("ct44"); lev = np.array(LEVELS); locked = c28 != c9; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([TN[int(m)] for m in nq_t])[:, None]; cache = {}  # ct9／ct28／ct44；ct28 改過的題鎖定；測試題的降溫係數

    def build(c, floor):  # 依組合 c 建立測試端的答案
        km = (IDS[c[0]], c[2], c[4], floor)  # 主來源的快取鍵
        if km not in cache:  # 尚未計算
            cache[km] = test_evidence(*km)  # 計算並快取
        L, r_main = cache[km]; r_enc = None  # 主來源的證據與密度比
        if c[1]:  # enc 開關
            ke = ("enc", 0, c[4], floor)  # 編碼器的快取鍵
            if ke not in cache:  # 尚未計算
                cache[ke] = test_evidence(*ke)  # 計算並快取
            L = L + U_ENC * cache[ke][0]; r_enc = cache[ke][1]  # 以權重 0.5 加上編碼器證據
        L = L * Tq if c[3] else L  # temp 開關：依局數降溫
        return np.where(locked, c28, lev[mb.decisions(P9, L, 0.5, 3.0, 0.6)[2]]), r_main, r_enc  # 鎖定題用 ct28，其餘只在邊界區內依融合後驗改決策
    base44, r0, _ = build(off, 3); repro = bool((base44 == c44).all()); print(f"builder with every switch off (floor q90) reproduces ct44: {'exact' if repro else f'FAILED, {int((base44 != c44).sum())} questions differ'} (density ratio {r0:.3f})")  # 全關＋q90 必須逐題重現 ct44
    fin, r_main, r_enc = build(final, a_.floor); ch = fin != c44; pub = np.char.startswith(qids.astype(str), "pub_"); jump = np.abs(np.array([RANK[x] for x in fin]) - np.array([RANK[x] for x in c44]))  # 以全體選出的組合建檔；與 ct44 的差異與跨級數
    print(f"test, combination {label(final)}, bands >= index {a_.floor}: density ratio main {r_main:.3f}" + (f", encoder {r_enc:.3f}" if r_enc is not None else "") + f"; {int(ch.sum())} questions differ from ct44 (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}; one level {int((jump == 1).sum())}, two or more {int((jump >= 2).sum())}); vs ct28 {int((fin != c28).sum())}")  # 記錄密度比與改動題數
    rep["test"] = {"repro_ct44": repro, "r_main": r_main, "r_enc": r_enc, "changed_vs_ct44": int(ch.sum()), "public": int(ch[pub].sum()), "private": int(ch[~pub].sum()), "two_or_more": int((jump >= 2).sum())}  # 測試端摘要
    rep["adopt"] = bool(passed and repro); rep["file"] = a_.out  # CV 通過且能重現 ct44 才採用
    if rep["adopt"]:  # 採用時寫出候選檔
        f = ROOT / f"submission_rank_{a_.out}.csv"; write_submission(list(zip(qids, fin)), f); rep["md5"] = hashlib.md5(f.read_bytes()).hexdigest()[:12]; print(f"{a_.out} written, md5 {rep['md5']}")  # 寫出提交檔並記錄 md5
        for other in ("ct64", "ct70", "ct71"):  # 與先前的打包檔比較
            if other != a_.out and (ROOT / f"submission_rank_{other}.csv").exists():  # 檔案存在且不是自己
                co = sub(other); print(f"   vs {other}: {int((fin != co).sum())} questions differ (public {int(((fin != co) & pub).sum())}, private {int(((fin != co) & ~pub).sum())})"); rep["test"][f"changed_vs_{other}"] = int((fin != co).sum())  # 記錄不同的題數
    (RV / a_.json).write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出結果 JSON
    subprocess.run([sys.executable, "-I", "scripts/m_series/m84_variant.py", "--stage", "finalize"], cwd=ROOT)  # 重寫 upload_next/ 的上傳指示


if __name__ == "__main__":  # 直接執行時
    main()  # 執行打包檢定
