"""Section 4M M123 (2026-11-14): M115 unchanged except for the identity vectors (var_S4e: the three symmetry-augmented seeds) and the
file name; registered as a single re-test (design doc M123). Original docstring:
Section 4M M115 (2026-11-14): M114's look-alike reading with (a) the average push per level removed (the log reading is centred
per level over the evaluated questions) and (b) a weight per question size picked by cross-fitting (design doc M115).
Below: M114's description; the grid is now sharpness 20 only and weights per size.
Section 4M M114: the target's look-alikes in the identity space as a second estimate of the level (design doc M114).
Reading of a question: mean target-side identity vector of its games (five members, var_S2y) against the prototypes of the training
players the network has seen (validation: the validation fold left out; test: all 8,007); level weights sum exp(beta x similarity)
divided by the level's number of candidates. Base probabilities P' ~ P x reading^w; then the whole ct75u rule on P'.
(beta, w) from {20, 30} x {0.2, 0.3, 0.5}, picked on one half of the folds (seeds 0-2) and judged on the other; seeds 6-8 reuse.
M37: the ct75u rule on P' against the ct75u rule on P - CI lower bound > 0, >= 3 of 4 folds, difference >= 0.0025, both seed sets.
On a pass (or --build): ct76 = the ct75u composition rebuilt on P9' (ct28's 38 locked answers untouched; with w = 0 the builder
must reproduce the files ct44 and ct75u), the reading's distribution on test against validation, and the feasible Public readings.
  python -I experiments/nano4/m114_lookalike_test.py [--build]
"""
import argparse  # 解析命令列參數
import csv  # 寫出提交檔
import hashlib  # 計算提交檔的 md5
import itertools  # 本檔未使用，維持原有的匯入
import json  # 寫出報告
import math  # 本檔未使用（E1 已註解掉），維持原有的匯入
import sys  # 調整模組搜尋路徑與錯誤結束
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取提交檔與分折表

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m103_stability as m3  # noqa: E402  匯入前須先設定 sys.path；每局證據與配方設定（M103）
import m110_fusion as mf  # noqa: E402  ct75u 規則（M110）
from aicup.metric import LEVELS  # noqa: E402  10 個等級名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM; CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0); BE = 20.0; NAME = "ct102"; VAR = "S4e"  # M123：在對稱擴增網路的向量（var_S4e）上做同一個檢定，候選檔名 ct102（ct76-ct99 已被占用）；E1 = math.exp(-1.0) 只留在註解，不再使用


def reading(tq_by_member, protos_by_member, cand_lab, beta):  # 「長得像的人」的等級讀數
    """log of the balanced level distribution of the look-alikes: tq [members, Q, 128] unit vectors, protos [members, C, 128]."""
    S = 0.0  # 各成員相似度的平均
    for m in range(len(tq_by_member)):  # 逐一成員
        S = S + tq_by_member[m] @ protos_by_member[m].T / len(tq_by_member)  # 題目向量對各候選人原型的相似度
    oh = np.eye(10)[cand_lab]; w = np.exp(beta * (S - S.max(1, keepdims=True))) @ oh / oh.sum(0); w = (w + 1e-9) / (w + 1e-9).sum(1, keepdims=True)  # 依等級加總 exp(beta x 相似度)，除以該等級的候選人數後正規化
    return np.log(w), S.max(1), -(w * np.log(w)).sum(1)  # 對數讀數、最高相似度、等級分布的熵


def pooled(P, lr, w):  # 把讀數以權重 w 併入基礎機率
    w = np.asarray(w, float).reshape(-1, 1) if np.ndim(w) else w; x = P * np.exp(w * (lr - lr.max(1, keepdims=True))); return x / x.sum(1, keepdims=True)  # w 可為每題一個；P' ~ P x 讀數^w 後正規化


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a_ = ap.parse_args()  # --build 即使沒過也建檔
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])  # 讀入識別器輸出與查詢檔，確認兩者的對局順序一致
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]  # 訓練玩家的標籤與所屬折；var_S4e 的查詢向量、原型與成員數
    Fd = {}; val_stats = []  # 各折資料與驗證端讀數的統計
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}  # 題目、每題的對局鍵與依局數的溫度
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():  # 逐一配方
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}  # 主識別器與編碼器（選用）的每局證據
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L  # 每題的總證據，必要時溫度縮放
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))  # 每局所屬的題與查詢列；候選人 = 不在本折的玩家
        for m in range(NM):  # 逐一成員
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))  # 目標方查詢向量（查詢檔第二段）加總到所屬題目
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); RD = {}  # 題目平均向量正規化成單位向量
        lr_, smax, ent = reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], BE); RD = lr_ - lr_.mean(0, keepdims=True); val_stats.append(np.stack([n, smax, ent], 1))  # 對沒看過本折的網路 v 的原型讀數；每個等級在本折的題目上置中
        Fd[v] = (y, n, pid, Ls, RD)  # 存下本折的資料
    half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; picks = None; ok = []; rep = {}; ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2))  # 交叉擬合的兩半折與合併後的題目資訊
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        S0, SU = {}, {}  # 基礎層與套用 ct75u 規則後的逐題分數，以權重為鍵
        for v in FOLDS:  # 逐一驗證折
            yv, nv, _, Ls, RD = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); d0, du = mf.rule75u(P, Ls); S0.setdefault("base", []).append(SM[d0, yv]); SU.setdefault("base", []).append(SM[du, yv])  # 不併入讀數的基準分數
            for w in WS[1:]:  # 每個正的權重
                d0, du = mf.rule75u(pooled(P, RD, w), Ls); S0.setdefault(w, []).append(SM[d0, yv]); SU.setdefault(w, []).append(SM[du, yv])  # 在 P' 上套用 ct75u 規則
            S0[0.0] = S0["base"]; SU[0.0] = SU["base"]  # 權重 0 即為基準
        fs = lambda S, k: np.array([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731  各折的類別平衡分數
        fsz = lambda S, k, m: np.array([balanced_mean_over_n(s_, yy, nn, sizes=(m,)) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731  各折在單一局數的分數
        if picks is None:  # 只在第一組種子上選權重
            best = lambda fl, m: max(WS, key=lambda c: fsz(SU, c, m)[[fi[v] for v in fl]].mean())  # noqa: E731  在指定折上該局數分數最高的權重
            picks = {**{v: {m: best(half[1], m) for m in SIZES} for v in half[0]}, **{v: {m: best(half[0], m) for m in SIZES} for v in half[1]}}; final = {m: best(list(FOLDS), m) for m in SIZES}; print(f"weights by question size picked on folds {half[0]}: { {m: best(half[0], m) for m in SIZES} } | on folds {half[1]}: { {m: best(half[1], m) for m in SIZES} } | on all four: {final}")  # 每折用另一半選出的權重；四折全體的選擇供建檔
            for m in SIZES:  # 逐一局數
                print(f"    n={m}: after the ct75u rule, by weight: " + "  ".join(f"{w}: {fsz(SU, w, m).mean() - fsz(SU, 'base', m).mean():+.4f}" for w in WS[1:]))  # 各權重相對基準的差
        new = [np.where(Fd[v][1] == SIZES[0], SU[picks[v][SIZES[0]]][fi[v]], np.where(Fd[v][1] == SIZES[1], SU[picks[v][SIZES[1]]][fi[v]], SU[picks[v][SIZES[2]]][fi[v]])) for v in FOLDS]; old = SU["base"]; dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new, old, ys, ns)]  # 每題依其局數用交叉擬合的權重；對基準的成對差與各折差
        good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); ok.append(good); A_, B_ = np.concatenate(new), np.concatenate(old)  # M37 判定
        print(f"seeds {seeds}: ct75u rule on P' against the ct75u rule on P (cross-fitted picks): {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}   (scores {fs(SU, 'base').mean():.4f} -> {np.mean([balanced_mean_over_n(a, yy, nn, sizes=SIZES) for a, yy, nn in zip(new, ys, ns)]):.4f}; answers changed {float((A_ != B_).mean()):.3f} of the questions by score)")  # 印出判定結果
        print("    by size: " + "  ".join(f"n={m}: {float(np.mean([balanced_mean_over_n(a, yy, nn, sizes=(m,)) - balanced_mean_over_n(c, yy, nn, sizes=(m,)) for a, c, yy, nn in zip(new, old, ys, ns)])):+.4f}" for m in SIZES) + " | by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)), flush=True)  # 依局數與真實等級的差
        ng_ = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(pid.astype(str)).to_numpy()  # 每題玩家在資料集中的局數
        print("    by the player's number of games in the data set (plain mean of the score difference; all sizes | 5-game questions): " + "  ".join(f"{nm_}: {float((A_ - B_)[s_].mean()):+.4f} | {float((A_ - B_)[s_ & (n == 5)].mean()):+.4f}" for nm_, s_ in (("1-25", ng_ <= 25), ("26-40", (ng_ > 25) & (ng_ <= 40)), ("41-70", (ng_ > 40) & (ng_ <= 70)), ("71-170", (ng_ > 70) & (ng_ <= 170)), ("171+", ng_ > 170))), flush=True)  # 依玩家局數分組的差
        rep[f"seeds_{seeds[0]}"] = {**dd, "per_fold": pf, "pass": bool(good)}  # 記錄此組種子的結果
    passed = bool(all(ok)); print(f"PRE-REGISTERED (M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep.update(passed=passed, final={str(k): v for k, v in final.items()}); (ROOT / "results/r_variants/m123_lookalike_sym.json").write_text(json.dumps(rep, indent=1, default=float))  # 兩組種子都過才算通過；寫出報告
    if not (passed or a_.build):  # 沒通過且沒要求建檔
        return  # 到此結束
    # ---------------------------------------------------------------- 測試端
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 測試端的基礎機率（ct9）與等級轉換
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  依題號讀取某個提交檔的答案
    c9, c28, c44, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75u")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 歷次提交檔；ct28 的推移鎖定；每局所屬題、每題局數、溫度與 Public 標記
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}  # 編碼器的測試證據（每題加總）
    for name in ("v44",) + FIVE:  # ct44 配方與五個投票配方
        idn, enc, cc, temp, pi0, bmin, scale = REC[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, scale, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()  # 主識別器的每題證據
        LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)  # 依配方加入編碼器並溫度縮放

    def compose(P):  # 在給定的基礎機率上組出 ct44 與 ct75u 的答案
        dR, dA = {}, {}  # 各配方的邊界區決策與不受限決策
        for name in LQ:  # 逐一配方
            d0, da, dr, _, region = mb.decisions(P, LQ[name], 0.5, 3.0, 0.6); dR[name] = np.where(locked, l28, dr); dA[name] = np.where(locked, l28, da)  # 兩種決策；鎖定題維持 ct28
        base = np.where(locked, l28, d0); d44 = dR["v44"]; keep = d44 != base; fin = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE])  # 基礎答案、ct44 答案與 ct75s 型規則；五個配方的票
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != d44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != base); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]  # 全票才推翻 ct44 的改動、全票才動邊界區以外的題
        return d44, fin  # 回傳 ct44 與 ct75u 的答案
    d44_0, fin_0 = compose(P9); assert (lev[d44_0] == c44).all(), int((lev[d44_0] != c44).sum()); assert (lev[fin_0] == c75u).all(), int((lev[fin_0] != c75u).sum()); print("checks: with weight 0 the builder reproduces the files ct44 and ct75u")  # 權重 0 時必須重現 ct44 與 ct75u
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))  # 測試端目標方查詢（依測試對局排序）與每局所屬的題
    for m in range(NM):  # 逐一成員
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))  # 加總到所屬題目
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); own = float(((QV[0, tgt].astype(np.float64) @ tq[0].T).argmax(1) == gq).mean()); print(f"alignment check (same member): a test game's target-side vector is closest to its own question's mean vector in {own:.3f} of the games"); assert own > 0.5, "test games and questions are not aligned"  # 正規化並檢查對局與題目確實對齊
    lr, smax20, ent20 = reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, BE); lr = lr - lr.mean(0, keepdims=True); wq = np.array([{5: final[5], 10: final[10], 15: (final[10] + final[20]) / 2, 20: final[20]}[int(m)] for m in nq_t]); vs = np.concatenate(val_stats)  # 對看過全部玩家的網路 -1 的原型讀數並置中；每題的權重（15 局取 10 與 20 的平均）
    print("the reading on test against validation (sharpness 20), by question size: mean top similarity | mean entropy of the level distribution")  # 測試端與驗證端讀數比較的標題
    for m in (5, 10, 15, 20):  # 逐一局數
        s_t = nq_t == m; s_v = vs[:, 0] == m; print(f"    n={m}: test {smax20[s_t].mean():.3f} | {ent20[s_t].mean():.3f}" + (f"   validation {vs[s_v, 1].mean():.3f} | {vs[s_v, 2].mean():.3f}" if s_v.any() else "   (no validation questions of this size)"))  # 平均最高相似度與平均熵
    d44_1, fin_1 = compose(pooled(P9, lr, wq)); fin = lev[fin_1]  # 在 P9' 上重建 ct75u 組合
    for nm, o in (("ct75u", c75u), ("ct44", c44), ("ct28", c28)):  # 與先前幾個檔比較
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{NAME} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")  # 不同題數與差幾級
    print("predicted levels, ct75u -> " + NAME + ": " + "  ".join(f"{lv} {int((c75u == lv).sum())}->{int((fin == lv).sum())}" for lv in LEVELS))  # 各等級答案數的變化
    f = ROOT / f"submission_rank_{NAME}.csv"  # 候選提交檔路徑
    if f.exists():  # 不覆寫既有檔案
        sys.exit(f"{f.name} exists already: nothing overwritten")  # 中止
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))  # 標頭與每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw); print(f"{f.name} written, md5 {md5}; weights by size = {final} (15 games: the mean of 10 and 20); passed the registered test: {passed}")  # 計算 md5、複製到待上傳資料夾並回報


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
