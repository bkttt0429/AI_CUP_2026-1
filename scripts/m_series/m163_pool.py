#!/usr/bin/env python
"""Section 4M M163 (2026-11-15): an identifier with the activity prior as one more source of the learned pool (design doc M163).

A copy of m143_pool.py with the look-alike reading left as it is in the ct106 rule (no new members):
  PRIMARY (the only test): pool + the anchor evidence of --new (statistic m15, test-side density ratio as in M129; one
  coefficient per question size, start 0) against the ct106 rule, the four guards relative to the ct106 rule, M37 on seeds 0-2
  and 6-8. Report only: the --also identifiers and the --control identifier, each alone as the extra source.
On a pass (unless --no-build): the test-side file --name (never overwrites, not copied to upload_next; ct28's locked answers
untouched; guards relative to ct106's answers; without the new input the builder must reproduce ct106).

  python -I scripts/m_series/m163_pool.py --new SPALm --also SPALn SPALh --control SPAL --name ct111
"""
import argparse  # 解析命令列參數
import csv  # 寫出提交檔
import hashlib  # 計算輸出檔的 md5
import json  # 讀寫結果 JSON
import sys  # 設定模組搜尋路徑與中止程式
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取 CSV 與依題目分組加總

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m105_voters as m5  # noqa: E402  M105 的測試端密度比計算
import m137b_build as mb7  # noqa: E402  M137b 的合併擬合、防護與共用設定
import m137f_build2 as mf7  # noqa: E402  修訂後的四條防護（guard2）
import m141_opp_lookalike as m41  # noqa: E402  （含額外輸入的 L0 合併的 fit／apply）
from aicup.metric import LEVELS  # noqa: E402  十個等級的名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  題目大小、類別平衡分數與成對差檢定

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM  # 沿用 M137b 匯入的模組與設定
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE  # 沿用 M137b 的來源名稱、權重與快取位置


def main():  # 主流程：交叉驗證的合併檢定與測試端建檔
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 命令列參數
    ap.add_argument("--new", required=True); ap.add_argument("--also", nargs="*", default=[]); ap.add_argument("--control", default=None)  # 主檢定的新識別器、只報告的其他識別器與對照
    ap.add_argument("--name", default="ct111"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default=None)  # 候選檔名、只檢定不建檔、報告檔名
    ap.add_argument("--build-anyway", action="store_true", help="dry run of the test-side code when the test did not pass: needs --out-dir (never the project root)"); ap.add_argument("--out-dir", default=None); a_ = ap.parse_args()  # 檢定沒過也演練建檔（只寫到 --out-dir）
    assert not a_.build_anyway or a_.out_dir, "--build-anyway writes outside the project only: give --out-dir"  # 演練建檔必須指定專案外的輸出資料夾
    out_json = ROOT / "results/r_variants" / (a_.json or f"m163_pool_{a_.new}.json")  # 報告檔路徑
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]  # M129 存下的五個投票來源密度比與其識別器名稱
    xs = [a_.new, *a_.also, *([a_.control] if a_.control else [])]  # 要評估的新識別器清單
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], *[f"{x}~{STAT}" for x in xs]])  # 載入各識別器（含 m15 版）的驗證端與測試端輸出
    if f"{a_.new}~{STAT}" not in used:  # 主檢定的識別器檔案不存在
        sys.exit(f"the identifier files of {a_.new}~{STAT} are missing")  # 中止
    xs = [x for x in xs if f"{x}~{STAT}" in used]  # 只保留檔案齊全的識別器
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}  # 各驗證折的題目；20 局題的基礎決策
    for v in FOLDS:  # 每個驗證折
        y, n, pid, q_ = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})  # 以種子 0-2 的基礎集成記下每位玩家 20 局題的決策
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # ct9 的測試端答案與各把握帶的名目占比
    dr = lambda x: round(m5.density_ratio(V[f"{x}~{STAT}"], T[f"{x}~{STAT}"], key_pid, dec, ct9s, nominal), 3)  # noqa: E731  一個識別器的測試端密度比（M105 的算法）
    chk = {x: dr(x) for x in ids}; assert all(abs(chk[x] - scale[x]) < 2e-3 for x in ids), (chk, scale)  # 五個投票來源存下的密度比必須能重現
    xscale = {x: dr(x) for x in xs}; print("test-side density ratio (m15): " + ", ".join(f"{x} {v:.3f}" for x, v in {**chk, **xscale}.items()), flush=True)  # 新識別器的密度比並印出
    rec0, rec2 = dict(REC), dict(REC)  # 兩套配方表：原設定與五個配方改用 m15
    for k in FIVE:  # 五個投票配方
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))  # 改用 m15 來源並換上它的密度比
    xspec = {x: (f"{x}~{STAT}", 1, 1, 1, 1, 1, float(xscale[x])) for x in xs}  # 新識別器的配方：開關與五個投票配方相同，配上各自的密度比
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])  # 局鍵索引與查詢表（確認順序一致）
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()  # 訓練玩家、等級與所屬折
    v_ = np.load(CACHE / f"var_{la.VAR}.npz")  # 長得像的人讀數用的身分向量（S4e）
    cur = [(v_["qvec"], v_["protos"], m) for m in range(v_["qvec"].shape[0])]  # 每個成員：查詢向量、原型與成員索引
    MEM = {"R": cur}; print(f"look-alike members: {len(cur)} ({la.VAR}), unchanged", flush=True)  # 讀數的成員維持不變

    def lookalike(mem, idx, owner, nq, net, cand):  # 長得像的人讀數
        """Centred log reading [nq, 10]: mean target-side vector of each question (queries idx, question index owner) against the prototypes of network `net`."""
        tq = np.zeros((len(mem), nq, 128))  # 每個成員每題的平均向量
        for j, (qv, _, m) in enumerate(mem):  # 每個成員
            np.add.at(tq[j], owner, qv[m, idx].astype(np.float64))  # 把各局向量加總到題目
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)  # 正規化成單位向量
        lr_, _, _ = la.reading(tq, [(pr[m, net] if cand is None else pr[m, net][cand]).astype(np.float64) for _, pr, m in mem], plab if cand is None else plab[cand], la.BE)  # 對網路 net 的原型讀數（驗證端只用候選人）
        return lr_ - lr_.mean(0, keepdims=True)  # 對各等級置中後回傳
    Fd = {}  # 各折的輸入
    for v in FOLDS:  # 逐一準備驗證折
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}  # 標籤、局數、玩家、各題局鍵與降溫係數

        def ev_of(spec):  # 依配方算這一折的每題錨點證據（有快取）
            if spec not in cache:  # 相同設定只算一次
                idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}  # 折外校準的錨點證據與編碼器證據
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L  # 每題加總各局證據，需要時乘上降溫係數
            return cache[spec]  # 回傳這個設定的證據
        for A, rec in ((0, rec0), (2, rec2)):  # 兩套配方表
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}  # 各配方的證據
        Lx = {x: ev_of(xspec[x]) for x in xs}  # 新識別器的證據
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)  # 每局的題號與查詢列；候選人排除驗證折玩家
        RD = {k: lookalike(mem, Q + gi, qi, len(y), v - 1, cand) for k, mem in MEM.items()}  # 本人側讀數：留出第 v 折的網路對其他折訓練玩家的原型
        cw = np.zeros(len(y))  # 類別平衡權重
        for s in SIZES:  # 每種題目大小
            for c in range(10):  # 每個等級
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)  # 權重＝該大小該等級題數的倒數
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, cw=cw, wq=np.array([W_FIN[int(m)] for m in n]), **RD)  # 存下這一折的輸入
        print(f"fold {v}: inputs ready ({len(y)} questions); look-alike reading alone, level of its maximum equals the truth: " + ", ".join(f"{k} {float((RD[k].argmax(1) == y).mean()):.3f}" for k in MEM), flush=True)  # 回報進度與讀數單獨的準度
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); cw_all = np.concatenate([Fd[v]["cw"] for v in FOLDS]); rep = {"density_ratio": xscale}; ok = []; keep = {}  # 合併四折的標籤與權重，準備報告
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731  四折類別平衡分數的平均
    base_of = lambda v, seeds: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # noqa: E731  一折的基礎集成機率（C、T 兩種結構乘種子的平均）

    def pool_decisions(X, extra=()):  # 交叉擬合合併：其餘三折擬合，在留出折上決策
        out = {}  # 各折的決策與係數
        for v in FOLDS:  # 輪流留出一折
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}  # 其餘三折當訓練資料
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra)  # 擬合含額外輸入的 L0 合併
            out[v] = (np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], extra)), int), coef)  # 以期望分數決策並保存係數
        return out  # 回傳各折結果
    VARI = [("PRIMARY: + new source", "R", (a_.new,)), *[(f"+ {x} only", "R", (x,)) for x in xs if x != a_.new]]  # 主檢定（加 --new）與只報告的其他識別器
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子各檢定一次
        X = {}  # 各折的合併輸入
        for v in FOLDS:  # 每個驗證折
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"]}  # 合併的輸入：基礎對數機率、讀數、六個來源與新識別器的證據
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)  # ct104b 規則的答案
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]  # 不加新來源的 L0＋四條防護＝ct106 規則，當參考
        print(f"\nseeds {seeds}: ct104b rule {bal([SM[Fd[v]['dc'], Fd[v]['y']] for v in FOLDS]):.4f}, ct106 rule {bal(S_old):.4f}")  # 印出兩個參考規則的分數
        if seeds == (0, 1, 2):  # 提交用的種子組
            keep = {"X": X, "ref": ref}  # 留給測試端擬合
        out = {}  # 這組種子的比較結果
        for name, rkey, extra in VARI:  # 每個版本
            Xv = {v: {**X[v], "R": Fd[v][rkey]} for v in FOLDS}; dd = pool_decisions(Xv, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]  # 加入新來源後的決策；相對於 ct106 規則加防護與不加防護的得分
            A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)  # 對 ct106 規則的成對差與信賴區間
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]  # 各折的差
            good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)  # M37：CI 下界 > 0、差 >= 0.0025、至少三折為正
            print(f"  {name:36s}: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items())  # 印出比較結果與各題目大小的差
                  + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}; coefficients (mean of the four fits) " + " | ".join(  # 接上改動比例、不加防護的分數與係數
                      f"n={s}: alpha {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}, v5 {np.mean([dd[v][1][s][3] for v in FOLDS]):.2f}, reading {np.mean([dd[v][1][s][7] for v in FOLDS]):.2f}" + (f", new {np.mean([dd[v][1][s][8] for v in FOLDS]):+.2f}" if extra else "") for s in SIZES), flush=True)  # 各題目大小的係數（四次擬合的平均）
            if name.startswith("PRIMARY"):  # 登錄的主檢定
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)), flush=True)  # 記錄是否通過並印出依真實等級的得失
            out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good), "score": bal(new_), "score_plain": bal(plain)}  # 存入報告
        rep[f"seeds_{seeds[0]}"] = out  # 存入報告
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets, the pool with the new source against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed  # 登錄的判定：兩組種子都要通過
    out_json.write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告
    if a_.no_build or not (passed or a_.build_anyway):  # 不建檔的情況
        return  # 到此結束
    out_dir = Path(a_.out_dir) if a_.out_dir else ROOT; post_dir = Path(a_.out_dir) if a_.out_dir else ROOT / "results/r_variants"  # 有 --out-dir 時檔案與後驗都寫到那裡
    if not passed:  # 只是演練
        print("DRY RUN OF THE BUILDER: the registered test did not pass, the file below is NOT a candidate")  # 標明產生的檔不是候選檔
    # ---------------------------------------------------------------- 測試端
    extra = (a_.new,); X012 = keep["X"]; Xcur = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}; Xnew = Xcur  # 四折合起來的輸入（種子 0-2）
    coef = m41.fit(Xnew, y, n, cw_all, extra); coef[15] = (coef[10] + coef[20]) / 2  # 以四折全部擬合；15 局題用 10 與 20 局係數的平均
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 讀入 ct9 的基礎機率與題號
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  讀取既有提交檔的答案
    c9, c28, c104b, c106 = (sub(k) for k in ("ct9", "ct28", "ct104b", "ct106")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 既有檔答案、ct28 鎖定題、各題局數與降溫係數、Public 題遮罩
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}  # 測試端編碼器證據（每題加總）
    for name, spec in [*[(k, rec2[k]) for k in NAMES], (a_.new, xspec[a_.new])]:  # 六個來源與新識別器
        idn, enc, cc, temp, pi0, bmin, sc = spec; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)  # 測試端每題證據（含編碼器與降溫）
    ti = np.flatnonzero(q["qtest"] >= 0); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000  # 測試局本人側的查詢列與所屬題目
    R_t = lookalike(MEM["R"], tgt, gq, 800, 4, None)  # 測試端讀數：全體玩家網路（-1）對全部原型
    tq0 = np.zeros((800, 128)); np.add.at(tq0, gq, cur[0][0][cur[0][2], tgt].astype(np.float64)); tq0 /= np.maximum(np.linalg.norm(tq0, axis=1, keepdims=True), 1e-9)  # 成員 0 的每題平均向量
    own = float(((cur[0][0][cur[0][2], tgt].astype(np.float64) @ tq0.T).argmax(1) == gq).mean())  # 測試局向量最接近自己題目平均的比例
    print(f"alignment check (member 0): a test game's target-side vector is closest to its own question's mean vector in {own:.3f} of the games"); assert own > 0.5, "test games and questions are not aligned"  # 對齊檢查：測試局與題目必須對得上
    X0 = {"logp": np.log(np.maximum(P9, 1e-12)), "R": R_t, **{k: LQ[k] for k in NAMES}}; c0 = mb7.fit(Xcur, y, n, cw_all, "L0"); c0[15] = ((c0[10][0] + c0[20][0]) / 2, (c0[10][1] + c0[20][1]) / 2)  # 不含新來源的輸入與 L0 係數，做法與 ct106 的建檔程式完全相同
    d0 = mf7.guard2(np.where(locked, l28, np.asarray(b.decide(mb7.apply(c0, X0, nq_t.astype(int))), int)), rk(c104b).astype(int)); assert (lev[d0] == c106).all(), int((lev[d0] != c106).sum()); print("checks: without the new inputs the builder reproduces ct106")  # 檢查：不加新輸入時要逐題重現 ct106
    Xt = {**X0, a_.new: LQ[a_.new]}; Pl = m41.apply(coef, Xt, nq_t.astype(int), extra); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin = lev[mf7.guard2(d_plain, rk(c106).astype(int))]; assert (fin[locked] == c28[locked]).all()  # 加新來源的合併決策，套上相對於 ct106 答案的防護；鎖定題不變
    np.savez(post_dir / f"m163_test_posterior_{a_.name}.npz", qid=qids, learned=Pl, base=P9)  # 存下測試端的後驗
    print("coefficients on all folds: " + " | ".join(f"n={s}: alpha {coef[s][0]:.2f}, v5 {coef[s][3]:.2f}, reading {coef[s][7]:.2f}, new source {coef[s][8]:+.2f}" for s in SIZES))  # 印出測試端用的係數
    print("test-side evidence of the new source: share of questions with a non-zero row " + f"{float((np.abs(LQ[a_.new]).sum(1) > 0).mean()):.3f} (v5: {float((np.abs(LQ['v5']).sum(1) > 0).mean()):.3f}); mean absolute log evidence {float(np.abs(LQ[a_.new]).mean()):.4f} (v5: {float(np.abs(LQ['v5']).mean()):.4f})")  # 新來源在測試端有證據的題目比例與平均強度（對照 v5）
    for nm, o in (("ct106", c106), ("ct104b", c104b)):  # 與既有各檔比較
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")  # 不同的題數與差幾級
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c106[fin != c106], fin[fin != c106])]).value_counts(); print("kinds of changes against ct106: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))  # 對 ct106 的改動類別
    for i in np.flatnonzero(fin != c106):  # 逐題列出改動
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct104b {c104b[i]}, ct106 {c106[i]} -> {fin[i]}")  # 題號、局數與各檔答案
    f = out_dir / f"submission_rank_{a_.name}.csv"  # 輸出檔路徑（演練時在 --out-dir）
    if f.exists():  # 不覆寫既有檔
        sys.exit(f"{f.name} exists already: nothing overwritten")  # 已存在就中止
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))  # 只用 LF 換行，表頭加每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]  # 計算 md5 前 12 碼
    print(f"{f.name} written (not copied to upload_next), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")  # 回報檔案與格式檢查
    if not passed:  # 演練時
        return  # 不把演練的檔寫進報告
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct106": qids[fin != c106].tolist(), "public": int(((fin != c106) & pub).sum()), "private": int(((fin != c106) & ~pub).sum())}  # 檔案資訊寫入報告
    out_json.write_text(json.dumps(rep, indent=1, default=float))  # 重寫報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
