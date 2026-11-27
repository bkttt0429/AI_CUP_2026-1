#!/usr/bin/env python
"""Section 4M M120 (2026-11-14): the formal test of the symmetry-augmented identity networks in the anchor pipeline, and the candidate.

v7 = bundle v5's recipe (encoder evidence + class-conditional tables + tempering + corrected mixture weights, every band, test-side
rules) with identifier --id in place of SP2e (default SP2y4: the phase matcher, S2y's five sequence members and the three
symmetry-augmented seeds, each source with its own fusion weight; m84_variant.py --stage eval --tag SP2y4 writes its files).
Registered (design doc M120 and its amendment of 07:06), both needed for a file:
  (1) recipe level   v7's region decisions against v5's: CI lower bound > 0, >= 3 of 4 folds positive, difference >= 0.0025,
                     on seeds 0-2 and 6-8;
  (2) file level     the ct75u composition with v7 in place of v5 (new changes need v3 = v7; reversals of ct44's changes and
                     changes outside the boundary region need all of v3, v7, s2, s3c, s3) against the ct75u composition:
                     difference >= 0 on both seed sets.
Report only: by question size, the players with <= 40 games, the identifiers of --also as v7.
--build (only if both hold): ct100 (ct76-ct99 are taken) by m104_build.py's code path with v7 in place of v5; the same path with v5 must reproduce
ct75u exactly; the feasible Public readings relative to ct75u.

  python -I scripts/m_series/m121_sym_anchor_test.py --id SP2y4 --also SP4e SP2e4 [--build]
"""
import argparse  # 解析命令列參數
import csv  # 寫出提交檔
import hashlib  # 計算提交檔的 md5
import json  # 讀取評估檔、寫出報告
import math  # 指數函式
import sys  # 調整模組搜尋路徑與錯誤結束
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取提交檔與證據彙總

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m103_stability as m3  # noqa: E402  匯入前須先設定 sys.path；每局證據與配方設定（M103）
import m105_voters as m5  # noqa: E402  密度比與多數答案（M105）
from aicup.metric import LEVELS  # noqa: E402  10 個等級名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM  # 沿用 m103 的模組、配方與計分矩陣
FIVE = ("v3", "v5", "s2", "s3c", "s3")  # ct75u 組合的五個投票配方
RV = ROOT / "results/r_variants"  # 識別器檔與報告的資料夾
E1 = math.exp(-1.0)  # 差一級的分數


def compose(d0, d44, dR, dA, region, five, locked=None):  # 投票配方的 ct75u 組合
    """The ct75u composition for the voters `five` (the first two are the pair whose shared new changes are taken)."""
    keep = d44 != d0  # ct44 配方改過的題
    free = ~region & ~keep if locked is None else ~region & ~keep & ~locked  # 邊界區以外、可以動的題
    r = np.where(keep, d44, np.where(dR[five[0]] == dR[five[1]], dR[five[1]], d44))  # 保留 ct44 的改動；前兩個配方一致才改
    a5, c5 = m5.modal(np.stack([dR[k] for k in five]), d44)  # 推翻 ct44 改動的多數票：不同於 ct44 答案的最多票答案與票數
    o5, k5 = m5.modal(np.stack([dA[k] for k in five]), d0)  # 邊界區以外改動的多數票
    s = keep & (c5 >= len(five)); r[s] = a5[s]  # 全票才推翻 ct44 的改動
    s = free & (k5 >= len(five)); r[s] = o5[s]  # 全票才動邊界區以外的題
    return r  # 回傳組合後的決策


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--id", default="SP2y4")  # 主要檢定的識別器
    ap.add_argument("--also", nargs="*", default=[])  # 只報告的其他識別器
    ap.add_argument("--build", action="store_true")  # 兩項都通過時建檔
    ap.add_argument("--name", default="ct100")  # 候選檔名稱
    a_ = ap.parse_args()  # 解析參數
    new_ids = [a_.id, *a_.also]  # 要當作 v7 的識別器
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3", *new_ids])  # 讀入現有與新識別器的輸出
    missing = [x for x in new_ids if x not in used]  # 缺檔的識別器
    if a_.id in missing:  # 主要識別器缺檔時
        sys.exit(f"identifier {a_.id}: evaluation files missing; run m84_variant.py --stage eval --tag {a_.id} first")  # 中止並提示先評估
    new_ids = [x for x in new_ids if x in used]  # 只保留有檔的識別器
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}  # 各驗證折與基礎決策
    for v in FOLDS:  # 驗證端的基礎決策（密度比用）
        y, n, pid, q = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)  # 標準成員的基礎機率
        dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})  # 每位玩家 20 局題的基礎決策
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # ct9 的答案與各帶的名目占比
    rec = dict(REC); rep = {"id": a_.id, "density_ratio": {}, "heldout": {}}  # 複製配方表，加入 v7 版本
    for x in new_ids:  # 每個新識別器
        r = m5.density_ratio(V[x], T[x], key_pid, dec, ct9s, nominal); rec[f"v7:{x}"] = (x, 1, 1, 1, 1, 1, round(r, 3)); rep["density_ratio"][x] = r  # 算出密度比，建立開啟全部開關的 v7 配方
        ev = RV / f"m84_{x}_eval.json"  # 該識別器的評估檔
        if ev.exists():  # 有評估檔時
            rep["heldout"][x] = json.loads(ev.read_text()).get("heldout", {}).get("fused")  # 留出對局的融合成績
        print(f"identifier {x}: test-side density ratio {r:.3f}; held-out fused top-1 / top-5 / top-20 {rep['heldout'].get(x)}", flush=True)  # 回報密度比與留出成績
    for x in ("SP2e",):  # 核對：v5 的識別器
        print(f"identifier {x} (v5): density ratio recorded {REC['v5'][6]}, computed {m5.density_ratio(V[x], T[x], key_pid, dec, ct9s, nominal):.3f}")  # 記錄值與重算值並列
    ngames = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id").n_games  # 每位玩家的局數
    Fd = {}  # 各折的資料
    for v in FOLDS:  # 逐一驗證折準備證據
        y, n, pid, q = LF[v]; gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}  # 題目、每題的對局鍵與依局數的溫度
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in rec.items():  # 逐一配方（含 v7 版本）
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}  # 主識別器與編碼器（選用）的每局證據
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L  # 每題的總證據，必要時溫度縮放
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)  # 存下本折的資料
    yy, nn, pp = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); low = ngames.reindex(pp).to_numpy() <= 40  # 合併各折；標出局數 <= 40 的玩家
    verdict = {}  # 各識別器的判定
    for x in new_ids:  # 每個新識別器當作 v7
        v7 = f"v7:{x}"; five7 = ("v3", v7, "s2", "s3c", "s3"); ok1, ok2 = [], []  # 以 v7 取代 v5 的五個投票配方
        for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
            S = {k: [] for k in ("v5", "v7", "u5", "u7", "base", "v44")}; changed = 0  # 各決策的逐題分數；組合改變的題數
            for v in FOLDS:  # 逐一驗證折
                y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR, dA = {}, {}  # 標準成員的基礎機率
                d0, _, d44, _, region = mb.decisions(P, Ls["v44"], 0.5, 3.0, 0.6)  # 基礎決策、ct44 配方決策與邊界區
                for name in (*FIVE, v7):  # 五個投票配方與 v7
                    _, dA[name], dR[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)  # 不受限的決策與邊界區決策
                u5, u7 = compose(d0, d44, dR, dA, region, FIVE), compose(d0, d44, dR, dA, region, five7); changed += int((u5 != u7).sum())  # 現行 ct75u 組合與換成 v7 的組合
                for k, d in (("v5", dR["v5"]), ("v7", dR[v7]), ("u5", u5), ("u7", u7), ("base", d0), ("v44", d44)):  # 每種決策
                    S[k].append(SM[d, y])  # 逐題官方分數
            fs = lambda k: np.array([balanced_mean_over_n(s_, Fd[v][0], Fd[v][1], sizes=SIZES) for s_, v in zip(S[k], FOLDS)])  # noqa: E731  各折的類別平衡分數
            print(f"\n=== {x} as v7, seeds {seeds}: base {fs('base').mean():.4f}, ct44 rule {fs('v44').mean():.4f}, v5 recipe {fs('v5').mean():.4f}, v7 recipe {fs('v7').mean():.4f}, ct75u rule {fs('u5').mean():.4f}, with v7 {fs('u7').mean():.4f} ===")  # 各決策的平均分數
            out = {}  # 兩項檢定的結果
            for tag, new, old in (("recipe", "v7", "v5"), ("file", "u7", "u5")):  # (1) 配方層級、(2) 檔案層級
                A_, B_ = np.concatenate(S[new]), np.concatenate(S[old]); d = paired_diff(A_, B_, yy, pp, nn, sizes=SIZES, n=5000); pf = (fs(new) - fs(old)).tolist()  # 成對差與各折差
                dl = paired_diff(A_[low], B_[low], yy[low], pp[low], nn[low], sizes=SIZES, n=3000)  # 只報告：局數少的玩家
                print(f"  [{tag}] {new} vs {old}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {v_:+.4f}" for m_, v_ in d["by_n"].items())  # 印出差值、各折與各局數的差
                      + f"  changed {float((A_ != B_).mean()):.4f}  | players with <= 40 games {dl['diff']:+.4f} [{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]", flush=True)  # 接著印改變比例與局數少的玩家
                out[tag] = {**d, "per_fold": pf, "low_activity": dl, "share_changed": float((A_ != B_).mean())}  # 記錄結果
            ok1.append((out["recipe"]["ci95"][0] > 0) and (out["recipe"]["diff"] >= 0.0025) and (sum(e > 0 for e in out["recipe"]["per_fold"]) >= 3)); ok2.append(out["file"]["diff"] >= 0)  # (1) 依 M37；(2) 差不為負
            rep[f"{x}|seeds_{seeds[0]}"] = {**out, "decisions_changed_by_the_composition": changed}  # 存入報告
        verdict[x] = {"recipe_M37": bool(all(ok1)), "file_not_worse": bool(all(ok2))}  # 兩組種子都要成立
        print(f"REGISTERED ({'primary' if x == a_.id else 'report only'}) {x}: (1) recipe level {'PASS' if all(ok1) else 'NO'}; (2) file level {'PASS' if all(ok2) else 'NO'}")  # 印出判定
    rep["verdict"] = verdict; rep["adopt"] = bool(verdict[a_.id]["recipe_M37"] and verdict[a_.id]["file_not_worse"])  # 只有主要識別器兩項都過才採用
    (RV / "m121_sym_anchor_test.json").write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告
    if not a_.build:  # 沒要求建檔
        return  # 到此結束
    if not rep["adopt"]:  # 未通過登記的條件
        print("no file: the registered conditions are not both met"); return  # 不建檔
    # ---- 測試端：m104_build.py 的路徑，以 v7 取代 v5
    v7 = f"v7:{a_.id}"; five7 = ("v3", v7, "s2", "s3c", "s3")  # 主要識別器的 v7 與投票配方
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 測試端的基礎機率（ct9）與等級轉換
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  依題號讀取某個提交檔的答案
    c9, c28, c44, c75s, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75s", "ct75u")); locked = c28 != c9; keep = c44 != c28  # 歷次提交檔；ct28 的推移鎖定；ct44 的改動
    nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 每題局數、溫度與 Public 標記
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); dR, dA = {}, {}; region = None  # 編碼器的測試證據（每題加總）
    for name in (*FIVE, v7):  # 五個投票配方與 v7 在測試端的決策
        idn, enc, cc, temp, pi0, bmin, scale = rec[name]; Lq = (pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, scale, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq  # 每題總證據（含編碼器）再乘溫度
        d0, da, dr, _, region = mb.decisions(P9, Lq, 0.5, 3.0, 0.6); dR[name] = np.where(locked, c28, lev[dr]); dA[name] = np.where(locked, c28, lev[da])  # 兩種決策；鎖定題維持 ct28

    def build(five):  # 以 m104_build.py 的路徑組出答案
        r = np.where(keep, c44, np.where(dR[five[0]] == dR[five[1]], dR[five[1]], c44)); vR = np.stack([dR[k] for k in five]); vA = np.stack([dA[k] for k in five])  # ct75s 型規則與五個配方的票
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != c44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != c9); fin = r.copy(); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]  # 全票才推翻 ct44 的改動、全票才動邊界區以外的題
        return fin, r, rev, out  # 回傳最終答案與中間結果
    chk, chk_s, _, _ = build(FIVE); assert (chk_s == c75s).all() and (chk == c75u).all(), (int((chk_s != c75s).sum()), int((chk != c75u).sum())); print("\nchecks: the same path with v5 reproduces ct75s and ct75u exactly")  # 用 v5 必須逐題重現 ct75s 與 ct75u
    fin, r7, rev, out = build(five7); assert (fin[locked] == c28[locked]).all()  # 以 v7 建檔；鎖定題不可變
    for nm, o in (("ct75u", c75u), ("ct75s", c75s), ("ct44", c44), ("ct28", c28)):  # 與先前幾個檔比較
        ch = fin != o; j = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((j == 1).sum())}, two or more {int((j >= 2).sum())}")  # 不同題數與差幾級
    for i in np.flatnonzero(fin != c75u):  # 逐一列出與 ct75u 不同的題
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct44 {c44[i]}, ct75u {c75u[i]} -> {fin[i]}  (v3 {dR['v3'][i]}, v5 {dR['v5'][i]}, v7 {dR[v7][i]})")  # 各檔答案與三個配方的決策
    f = ROOT / f"submission_rank_{a_.name}.csv"  # 候選提交檔路徑
    if f.exists():  # 不覆寫既有檔案
        sys.exit(f"{f.name} exists already: nothing overwritten (choose another --name)")  # 中止並提示換名稱
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["question_id", "pred_rank"]); w.writerows(zip(qids, fin))  # 標頭與每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw)  # 計算 md5 並複製到待上傳資料夾
    print(f"{f.name} written, md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")  # 檢查列數、換行字元與等級名稱
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct75u": qids[fin != c75u].tolist(), "public": int(((fin != c75u) & pub).sum()), "private": int(((fin != c75u) & ~pub).sum())}  # 記錄候選檔資訊
    (RV / "m121_sym_anchor_test.json").write_text(json.dumps(rep, indent=1, default=float))  # 更新報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
