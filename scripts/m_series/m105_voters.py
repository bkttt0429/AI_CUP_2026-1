"""Section 4M M105 (2026-11-13): ten voters and a vote threshold instead of the unanimity of five (design doc M105).
Voters = recipes with bundle v5's four switches (encoder, class-conditional tables, tempering, pi0; all bands; test-side density
ratio) on ten identifier sources: 5e, P3, S, SP, SP2e, S2, S2b, S2c, S3c, S3. Rule family on top of the ct75s rule:
 (a) questions the ct44 recipe changed: at least tr of the ten region decisions equal the same answer other than ct44's -> it;
 (b) outside the boundary region (not changed by ct44): at least to of the ten unrestricted decisions equal the same change -> it.
CV under the test rules. Thresholds picked on one half of the folds (seeds 0-2), judged on the other half against the ct75u rule
(unanimity of SP, SP2e, S2, S3c, S3); seeds 6-8 reuse the picks. Pass: CI lower bound > 0, >= 3 of 4 folds > 0, difference
>= 0.0010 on both seed sets. If it passes: ct75v with the thresholds picked on all four folds, and its Public decode table.
  python -I scripts/m_series/m105_voters.py [--build]
"""
import argparse  # 解析命令列參數
import csv  # 寫出提交檔
import hashlib  # 計算提交檔的 md5
import itertools  # 門檻組合的網格
import json  # 讀取帳本、寫出報告
import math  # 指數與對數
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取提交檔與證據彙總

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m103_stability as m3  # noqa: E402  匯入前須先設定 sys.path；每局證據與共用模組（M103）
import m58_decode as led  # noqa: E402  Public 帳本的解讀（M58）
from aicup.metric import LEVELS  # noqa: E402  10 個等級名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM  # 沿用 m103 已匯入的模組與常數
SCALE = {"5e": 0.851, "P3": 0.836, "S": 0.85, "SP": 0.875, "SP2e": 0.852, "S2": 0.734, "S2b": 0.734, "S2c": 0.734, "S3c": 0.855, "S3": 0.85}; TEN = tuple(SCALE); FIVE = ("SP", "SP2e", "S2", "S3c", "S3"); GRID = (6, 7, 8, 9, 10); E1 = math.exp(-1.0); NAME = "ct75v"  # 各來源記錄的密度比、十個與五個投票來源、門檻網格、差一級的分數、候選檔名


def density_ratio(v, t, key_pid, dec, ct9, nominal):  # 一個識別器的測試端密度比
    """Test-side density ratio of one identifier, as m86_bundle.py's test builder computes it (independent of the switches)."""
    ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones)  # 以全部驗證查詢擬合分帶
    band_of = lambda m, mb_: np.where(mb_, np.searchsorted(edges[True], m), np.searchsorted(edges[False], m))  # noqa: E731  依顏色選用分帶界線
    vb_imp = band_of(v["imp_m"], v["target_black"]); tb = band_of(t["opp_m"], ~t["target_black"]); tg = s2.group_of(t["hd"], t["target_black"])  # 驗證冒充者與測試對手的分帶、測試組別
    vd = np.array([dec[key_pid[k]] for k in v["key"]]); td = ct9.reindex(t["qid"]).to_numpy(); num = den = 0.0  # 驗證局玩家的基礎決策、測試局所屬題的 ct9 答案
    for q_ in range(4, 9):  # 第 4-8 帶
        w1 = lambda lab, bb, d_, s: float((np.abs(lab[s & (bb == q_)] - d_[s & (bb == q_)]) <= 1).mean()) if (s & (bb == q_)).any() else np.nan  # noqa: E731  此帶中標籤與決策差一級以內的比例
        av, ai, at = w1(v["opp_lab"], vband, vd, vg >= 0), w1(v["imp_lab"], vb_imp, vd, vg >= 0), w1(t["opp_lab"], tb, td, tg >= 0); w = int(((tg >= 0) & (tb == q_)).sum())  # 驗證真對手、冒充者、測試的一致率與測試局數
        if w and np.isfinite([av, ai, at]).all():  # 三者都有值時
            num += w * (at - ai); den += w * (av - ai)  # 以測試局數加權累計
    r_agree = float(np.clip(num / den, 0, 1)); sh_v, sh_t = float((vband[vg >= 0] >= 3).mean() - nominal[3:].sum()), float((tb[tg >= 0] >= 3).mean() - nominal[3:].sum())  # 一致率比；第 3 帶以上的超額占比
    return float(min(r_agree, np.clip(sh_t / sh_v, 0, 1)))  # 兩種估計取小，偏保守


def modal(votes, ref):  # 除了 ref 以外票數最多的答案
    """Most frequent answer other than ref among the rows of votes [m, nq], and its count."""
    alt = ref.copy(); cnt = np.zeros(votes.shape[1], int)  # 預設為 ref、票數 0
    for c in range(10):  # 逐一答案
        k = (votes == c).sum(0); better = (c != ref) & (k > cnt); alt = np.where(better, c, alt); cnt = np.where(better, k, cnt)  # 票數更多且不同於 ref 時取代
    return alt, cnt  # 回傳答案與票數


def rules(d0, d44, dR, dA, region, locked=None):  # 參考規則與投票規則族的材料
    """Decisions of the reference rules and the ingredients of the vote family."""
    keep = d44 != d0; free = ~region & ~keep if locked is None else ~region & ~keep & ~locked; r75s = np.where(keep, d44, np.where(dR["SP"] == dR["SP2e"], dR["SP2e"], d44))  # ct44 改過的題、可動的區外題；ct75s 規則 = SP 與 SP2e 一致才改
    a5, c5 = modal(np.stack([dR[k] for k in FIVE]), d44); o5, k5 = modal(np.stack([dA[k] for k in FIVE]), d0); r75u = r75s.copy(); s = keep & (c5 >= 5); r75u[s] = a5[s]; s = free & (k5 >= 5); r75u[s] = o5[s]  # ct75u 規則：五個配方全票才推翻 ct44 的改動或動區外題
    a10, c10 = modal(np.stack([dR[k] for k in TEN]), d44); o10, k10 = modal(np.stack([dA[k] for k in TEN]), d0)  # 十個投票來源的多數答案與票數
    return keep, free, r75s, r75u, (a10, c10), (o10, k10)  # 回傳規則族所需的全部材料


def family(r75s, keep, free, A, O, tr, to):  # 套用門檻 (tr, to) 的投票規則
    d = r75s.copy(); s = keep & (A[1] >= tr); d[s] = A[0][s]; s = free & (O[1] >= to); d[s] = O[0][s]  # 從 ct75s 規則出發，(a) 推翻 ct44 的改動、(b) 區外改動
    return d  # 回傳決策


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--s-computed", action="store_true", help="use the computed density ratio for voter S instead of the placeholder 0.85"); a_ = ap.parse_args()  # --build 即使沒過也建檔；--s-computed 改用算出的 S 密度比
    V, T, key_pid, used = b.sources(list(TEN)); assert all(k in V for k in TEN), used; Fd = {}; LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}  # 讀入十個識別器的輸出與各驗證折
    for v in FOLDS:  # 驗證端的基礎決策（密度比用）
        y, n, pid, q = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})  # 每位玩家 20 局題的基礎決策
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # ct9 的答案與各帶的名目占比
    for name in TEN:  # 核對每個來源的密度比
        r = density_ratio(V[name], T[name], key_pid, dec, ct9s, nominal); new_ = name in ("S2b", "S2c") or (name == "S" and a_.s_computed); warn = "" if new_ or abs(r - SCALE[name]) < 0.0015 else "  <-- DIFFERS from the recorded value (recorded value kept)"  # 新來源用算出的值，其餘與記錄值比對
        if new_:  # 沒有記錄值的來源
            SCALE[name] = round(r, 3)  # 改用算出的密度比
        print(f"density ratio {name:5s}: computed {r:.3f}, used {SCALE[name]:.3f}{warn}", flush=True)  # 回報算出與採用的值
    for v in FOLDS:  # 逐一驗證折準備證據
        y, n, pid, q = LF[v]; gk = list(q["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; ee = m3.evidence(V["enc"], v, 0, 1, 1, m3.R_ENC); Le = np.stack([sum((ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls = {}  # 依局數的溫度與編碼器的每題證據
        for name in ("v44",) + TEN:  # ct44 配方與十個投票來源
            ev = m3.evidence(V["5e"], v, 0, 0, 3, 0.851) if name == "v44" else m3.evidence(V[name], v, 1, 1, 1, SCALE[name]); L = np.stack([sum((ev.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L if name == "v44" else (L + b.U_ENC * Le) * Tn  # v44 不開任何開關；投票來源加編碼器再溫度縮放
        Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)  # 存下本折的資料
    half = (list(FOLDS[:2]), list(FOLDS[2:])); picks = None; ok = []; rep = {}  # 交叉擬合的兩半折
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        S = {}; tabR, tabO = [], []  # 各規則的逐題分數與兩類改動的得失表
        for v in FOLDS:  # 逐一驗證折
            y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dR, dA = {}, {}  # 標準成員的基礎機率
            d0, _, d44, _, region = mb.decisions(P, Ls["v44"], 0.5, 3.0, 0.6)  # 基礎決策、ct44 配方決策與邊界區
            for name in TEN:  # 每個投票來源
                _, dA[name], dR[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)  # 不受限的決策與邊界區決策
            keep, free, r75s, r75u, A, O = rules(d0, d44, dR, dA, region); S[("75s", v)] = SM[r75s, y]; S[("75u", v)] = SM[r75u, y]  # 兩條參考規則的逐題分數
            for tr, to in itertools.product(GRID + (99,), repeat=2):  # 全部門檻組合（99 = 從不）
                S[((tr, to), v)] = SM[family(r75s, keep, free, A, O, tr, to), y]  # 該組合的逐題分數
            s = keep & (A[1] >= 1); tabR.append((A[1][s], SM[A[0][s], y[s]] - SM[d44[s], y[s]])); s = free & (O[1] >= 1); tabO.append((O[1][s], SM[O[0][s], y[s]] - SM[d0[s], y[s]]))  # 依票數記錄兩類改動的得失
        yy, nn, pp = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); nq = len(yy); print(f"seeds {seeds}: {nq} questions")  # 合併各折的題目資訊
        for title, tab in (("(a) changing a ct44-recipe answer, by the number of the ten voters preferring the same other answer", tabR), ("(b) outside the boundary region, by the number of the ten voters making the same change", tabO)):  # 兩類改動各印一張表
            c_ = np.concatenate([x[0] for x in tab]); g_ = np.concatenate([x[1] for x in tab]); print(f"  {title}:")  # 票數與得失
            for k in range(1, 11):  # 依票數 1-10 分組
                s = c_ == k  # 此票數的改動
                if s.sum():  # 有改動才印
                    print(f"      {k:2d} of 10: {int(s.sum()):5d} ({s.sum() / nq:.4f}); better {float((g_[s] > 0).mean()):.3f}, worse {float((g_[s] < 0).mean()):.3f}; mean gain {float(g_[s].mean()):+.3f}; total {float(g_[s].sum()):+.1f}")  # 改對、改錯比例與平均得失
        fi = {v: i for i, v in enumerate(FOLDS)}; combos = list(itertools.product(GRID + (99,), repeat=2)); fs = {c: np.array([balanced_mean_over_n(S[(c, v)], Fd[v][0], Fd[v][1], sizes=SIZES) for v in FOLDS]) for c in combos}; f75u = np.array([balanced_mean_over_n(S[("75u", v)], Fd[v][0], Fd[v][1], sizes=SIZES) for v in FOLDS])  # 每個門檻組合與 ct75u 規則的各折分數
        if picks is None:  # 只在第一組種子上選門檻
            best = lambda fl: max(combos, key=lambda c: fs[c][[fi[v] for v in fl]].mean())  # noqa: E731  在指定折上平均分數最高的組合
            picks = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}; final = best(list(FOLDS)); print(f"  thresholds (reversal, outside; 99 = never): picked on folds {half[0]}: {best(half[0])} | on folds {half[1]}: {best(half[1])} | on all four: {final}")  # 每折用另一半選出的門檻；四折全體的選擇供建檔
            for c in sorted(combos, key=lambda c: -fs[c].mean())[:8]:  # 列出前 8 名組合
                print(f"      {str(c):10s} mean {fs[c].mean():.4f}  vs ct75u rule {fs[c].mean() - f75u.mean():+.4f}  folds " + " ".join(f"{x:+.4f}" for x in fs[c] - f75u))  # 平均分數與對 ct75u 規則的差
        new = np.concatenate([S[(picks[v], v)] for v in FOLDS]); old = np.concatenate([S[("75u", v)] for v in FOLDS]); dd = paired_diff(new, old, yy, pp, nn, sizes=SIZES, n=5000); pf = [float(fs[picks[v]][fi[v]] - f75u[fi[v]]) for v in FOLDS]; good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0010) and (sum(e > 0 for e in pf) >= 3)  # 交叉擬合的門檻對 ct75u 規則的判定
        fb = np.concatenate([S[(final, v)] for v in FOLDS]); df = paired_diff(fb, old, yy, pp, nn, sizes=SIZES, n=5000); d75s = paired_diff(old, np.concatenate([S[("75s", v)] for v in FOLDS]), yy, pp, nn, sizes=SIZES, n=3000)  # 只報告：四折選擇的差與 ct75u 對 ct75s 的差
        print(f"  seeds {seeds}: cross-fitted thresholds vs the ct75u rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  decisions changed {float((new != old).mean()):.4f} -> {'PASS' if good else 'no'}")  # 印出判定結果
        print(f"  seeds {seeds}: (not cross-fitted, report) the all-folds pick {final} vs the ct75u rule {df['diff']:+.4f} [{df['ci95'][0]:+.4f}, {df['ci95'][1]:+.4f}]; for reference the ct75u rule vs the ct75s rule {d75s['diff']:+.4f}", flush=True)  # 印出只報告的比較
        rep[f"seeds_{seeds[0]}"] = {**dd, "per_fold": pf, "pass": bool(good)}; ok.append(good)  # 記錄此組種子的結果
    passed = bool(all(ok)); rep.update(passed=passed, final=list(final)); print(f"PRE-REGISTERED: the ten-voter thresholds against the ct75u rule: {'PASS' if passed else 'NO'}"); (ROOT / ("results/r_variants/m105_voters_scomp.json" if a_.s_computed else "results/r_variants/m105_voters.json")).write_text(json.dumps(rep, indent=1, default=float))  # 兩組種子都過才算通過；寫出報告
    if not (passed or a_.build):  # 沒通過且沒要求建檔
        return  # 到此結束
    # 測試端
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 測試端的基礎機率（ct9）與等級轉換
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  依題號讀取某個提交檔的答案
    c9, c28, c44, c75s, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75s", "ct75u")); locked = c28 != c9; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 歷次提交檔；ct28 的推移鎖定；每題局數、溫度與 Public 標記
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); dR, dA = {}, {}  # 編碼器的測試證據（每題加總）
    for name in TEN:  # 每個投票來源在測試端的決策
        Lq = (pd.DataFrame(m3.test_evidence(V, T, name, 1, 1, SCALE[name], 1)).groupby(T[name]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq; d0, da, dr, _, region = mb.decisions(P9, Lq, 0.5, 3.0, 0.6); dR[name] = np.where(locked, rk(c28), dr); dA[name] = np.where(locked, rk(c28), da)  # 每題證據、決策；鎖定題維持 ct28
    keep, free, r75s, r75u, A, O = rules(rk(c9), rk(c44), dR, dA, region, locked); assert (lev[r75s] == c75s).all() and (lev[r75u] == c75u).all(), (int((lev[r75s] != c75s).sum()), int((lev[r75u] != c75u).sum())); print("checks: the ct75s and ct75u rules rebuilt here equal the two files")  # 重建的兩條參考規則必須與檔案逐題相同
    fin = lev[family(r75s, keep, free, A, O, final[0], final[1])]  # 以四折選出的門檻產生最終答案
    for nm, o in (("ct75u", c75u), ("ct75s", c75s), ("ct44", c44), ("ct28", c28)):  # 與先前幾個檔比較
        ch = fin != o; j = np.abs(rk(fin) - rk(o)); print(f"{NAME} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((j == 1).sum())}, two or more {int((j >= 2).sum())}")  # 不同題數與差幾級
    for i in np.flatnonzero(fin != c75u):  # 逐一列出與 ct75u 不同的題
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct44 {c44[i]}, ct75u {c75u[i]} -> {fin[i]}  (votes for a change of ct44's answer {int(A[1][i])}, outside the region {int(O[1][i])})")  # 各檔答案與票數
    f = ROOT / f"submission_rank_{NAME}.csv"  # 候選提交檔路徑
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["question_id", "pred_rank"]); w.writerows(zip(qids, fin))  # 標頭與每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw); print(f"{f.name} written, md5 {md5}; thresholds {final}")  # 計算 md5 並複製到待上傳資料夾
    ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}; prior = led.prior_from_ct9(); qs, cand, obs = led.build(up, prior); ou = next(o for o in obs if o["name"] == "submission_rank_ct75u.csv"); Su = up["submission_rank_ct75u.csv"]; dnew = led.decisions(f.name); k = int(((fin != c75u) & pub).sum()); rows = []  # 讀入 Public 帳本，準備列舉新檔可能的 Public 讀數
    for a in range(-k, k + 1):  # 完全正確題數的淨變動
        for bb in range(-k, k + 1):  # 差一級以內題數的淨變動
            if abs(a - bb) > k:  # 不可能的組合
                continue  # 略過
            qs2, cand2, obs2 = led.build(up, prior, [(dnew, ou["X"] + a, ou["Y"] + bb)]); sol = led.solve(qs2, cand2, obs2, prior, time_limit=20)  # 假設此讀數，檢查與全部上傳是否相容
            if sol is not None and sol != "timeout":  # 可行的讀數
                rows.append({"a": a, "b": bb, "score": Su + ((1 - E1) * a + E1 * bb) / 400, "points": (1 - E1) * a + E1 * bb, "cost": float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))})  # 讀數、分數差與先驗成本
    cmin = min(r["cost"] for r in rows); z = sum(math.exp(-(r["cost"] - cmin)) for r in rows)  # 正規化常數
    for r in rows:  # 每個可行讀數
        r["weight"] = math.exp(-(r["cost"] - cmin)) / z  # 粗估的可能性
    rows.sort(key=lambda r: -r["score"]); print(f"feasible Public readings of {NAME}: {len(rows)} ({k} Public questions differ from ct75u); rough probability of a reading not below ct75u: {sum(r['weight'] for r in rows if r['score'] >= Su - 1e-9):.2f}")  # 依分數排序並印出不低於 ct75u 的機率
    for r in sorted(rows, key=lambda r: -r["weight"])[:10]:  # 可能性最高的 10 個讀數
        print(f"    reading {r['score']:.7f}  exact {r['a']:+d}, within one {r['b']:+d} ({r['points']:+.2f} points vs ct75u)  weight {r['weight']:.3f}")  # 讀數與對應的題數變動
    L = [f"# `submission_rank_{NAME}.csv` 的 Public 讀數對照表（自動產生）", "", f"檔案 md5 `{md5}`：ct75s 的規則，加上十個配方中至少 {final[0]} 個同意才改掉 ct44 的答案、至少 {final[1]} 個同意才動邊界區以外的題。與 ct75u 不同的 Public 題 {k} 題。ct75u 是 0.8101304。", "", "| Public 讀數 | 對 ct75u：完全正確 | 差一級以內 | 分數差 | 粗估可能性 |", "|---|---|---|---|---|"]  # 解讀表的開頭
    L += [f"| {r['score']:.7f} | {r['a']:+d} | {r['b']:+d} | {r['points']:+.2f} | {r['weight']:.2f} |" for r in rows] + ["", f"事先寫下的規則：讀數 ≥ 0.8101304 → 最終檔改用 {NAME}；低於 → 傳回 ct75u。"]  # 每個讀數一列，最後附事先寫下的規則
    (ROOT / f"upload_next/{NAME}_decode.md").write_text("\n".join(L) + "\n"); print("written")  # 寫出解讀表


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
