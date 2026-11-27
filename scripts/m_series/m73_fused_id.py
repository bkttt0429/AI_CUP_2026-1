"""Section 4M M73 (2026-11-10): single-game identification of the opponent with the five habit components of competition 1
(opening, style, local shape, orientation, HD class; ident_pool.py's vectorised models), fused with weights fitted on held-out OWN games.
Pools: all training rows except 2 held-out games per player (the weight-fitting queries). Queries, test regime (only the own player
excluded): the 20 games of every validation player's n = 20 question, opponent side and target side (impostor); the test games.
Saves per query, for the fused score and for the opening component alone: top-1 candidate label, margin, candidate.
  python -I scripts/m_series/m73_fused_id.py
"""
import json  # 讀取驗證折題目、寫出報告
import multiprocessing as mp  # 平行建視角與評分
import re  # 從 SGF 取出讓子數 HD
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取訓練資料

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案、同目錄與競賽一 aicup_pid 的程式
import ident_pool as ip  # noqa: E402  五個習慣成分的向量化模型
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
COMPS = ip.COMPS  # 五個成分的名稱
G = {}  # 子行程共用的全域資料（fork 後繼承）
OTHER = {"B": "W", "W": "B"}  # 對手的顏色


def raw_scores(i):  # 第 i 個查詢對全部候選人的五成分原始分數
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v])  # 依查詢視角的標記選候選池評分
    return np.stack([sc[c] for c in COMPS])  # [5, C]，每個成分一列


def top2(x, ex):  # 取最高分的候選人與領先第二名的差
    if ex >= 0:  # 有要排除的候選人（本人）
        x = x.copy(); x[ex] = -np.inf  # 本人的分數設為負無限大
    o = np.argpartition(-x, 1)[:2]; o = o[np.argsort(-x[o])]  # 取前兩名並排序
    return int(o[0]), float(x[o[0]] - x[o[1]])  # 第一名索引與 margin


def score_chunk(idx):  # 一批查詢的融合與開局 top-1
    out = []; w = G["w"]; C = G["C"]  # 輸出、融合權重與候選人數
    for i in idx:  # 逐一查詢
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)  # 原始分數與要排除的候選人
        if ex >= 0:  # 驗證查詢要排除本人
            valid[ex] = False  # 本人不參與標準化
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)  # 每個成分對有效候選人標準化
        bf, mf = top2(w @ Z, ex); bo, mo = top2(S[0], ex)  # 融合與只用開局的 top-1、margin
        out.append((i, bf, mf, bo, mo))  # 記下結果
    return out  # 回傳這批結果


def full_chunk(idx):  # 一批留出查詢的完整原始分數，供擬合權重
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]  # 以 float32 節省記憶體


def main():  # 主流程
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    rng = np.random.default_rng(20261110)  # 固定種子抽留出局
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)  # 讀入全部訓練列
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]  # 棋局鍵＝等級:棋局 ID
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)  # 候選人（訓練玩家）清單與索引
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位候選人的等級標籤
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k: i for i, k in enumerate(keys)}  # 每列的玩家索引與棋局鍵
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); hd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0 for s in sgf])  # 棋譜、執色與讓子數
    ctx = mp.get_context("fork")  # 以 fork 讓子行程共用資料
    with ctx.Pool(12) as pool:  # 12 個工作行程
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))  # 每列以本人為主體的視角
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")  # 視角標記（B 或 W 候選池）
    # 驗證查詢
    qrow, qfold, qy, qpid = [], [], [], []  # 查詢列、折、真值與玩家
    for v in (1, 2, 3, 4):  # 四個驗證折
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 本折的驗證題
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # 只用 n = 20 的題
                for k in gk:  # 這題的 20 局
                    qrow.append(key_index[k]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid)  # 加入一個查詢
    qrow = np.array(qrow)  # 轉成陣列
    with ctx.Pool(12) as pool:  # 平行建對手側視角
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))  # 以對手為主體的視角
    # 擬合權重用的留出本人對局：從抽樣的玩家中每人取 2 局，絕不使用驗證題的局
    used = set(qrow.tolist()); held = []  # 驗證題用到的列；留出列
    by_player = rows.groupby("player_id").indices  # 每位玩家的列
    for p_ in rng.choice(players, 2400, replace=False):  # 抽 2,400 位玩家
        cand = [r for r in by_player[p_] if r not in used and hd[r] < 2]  # 不在驗證題且 HD < 2 的局
        if len(cand) >= 8:  # 至少 8 局才留出
            held += rng.choice(cand, 2, replace=False).tolist()  # 每人留出 2 局
    held = np.array(held); include = np.ones(len(rows), bool); include[held] = False  # 留出局不進候選池
    # 測試集
    questions = ip.read_tests([ROOT / "data/tests/rank_prediction_test_public.csv", ROOT / "data/tests/rank_prediction_test_private.csv"])  # 讀入 Public 與 Private 測試題
    tg = [(q["question_id"], g) for q in questions for g in q["games"]]  # 每局一筆（題號, (棋譜, 執色)）
    with ctx.Pool(12) as pool:  # 平行建測試局視角
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))  # 本人側與對手側
    log(f"query views: validation {len(qrow)} x 2, held-out {len(held)}, test {len(tg)} x 2")  # 記錄各類查詢數
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt  # 順序：驗證對手側、驗證本人側、留出局、測試對手側、測試本人側
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * len(held) + [-1] * (2 * len(tg))  # 驗證查詢排除本人，其餘不排除
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]  # 查詢用到的開局鍵，只為這些建表
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)  # 擬合 B、W 兩個視角的候選池
    G.update(qviews=qviews, pools=pools, excl=excl, C=C)  # 放進全域給子行程使用
    # 以留出的本人對局擬合權重
    n0 = 2 * len(qrow); hidx = np.arange(n0, n0 + len(held)); truth = pl[held]  # 留出查詢的索引與真正的玩家
    with ctx.Pool(12) as pool:  # 平行評分
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]  # 留出查詢的完整原始分數
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # 形狀 [5, Q, C]：成分、查詢、候選人
    Z = [ip.standardise(S[j]) for j in range(len(COMPS))]; w = ip.fit_weights(Z, truth); rep = {"weights": dict(zip(COMPS, map(float, w))), "held_queries": int(len(held))}  # 標準化後以多項式似然擬合權重
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731  真正玩家的名次（0 = 第一名）
    acc = {}  # 各成分與融合的認人準度
    for name, F in [(c, Z[j]) for j, c in enumerate(COMPS)] + [("fused", np.einsum("j,jqc->qc", w, np.stack(Z)))]:  # 五個成分各自與融合後
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}  # top-1／5／20 與名次中位數
    rep["heldout_accuracy"] = acc  # 存入報告
    log("weights " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w)) + " | held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k, v in acc.items()))  # 記錄權重與準度
    del full, S, Z  # 釋放記憶體
    G["w"] = w  # 融合權重給子行程使用
    with ctx.Pool(12) as pool:  # 平行評分其餘查詢
        res = [x for part in pool.map(score_chunk, np.array_split(np.concatenate([np.arange(n0), np.arange(n0 + len(held), len(qviews))]), 480)) for x in part]  # 驗證與測試查詢，跳過留出局
    res = {i: r for i, *r in res}; log("all queries scored")  # 查詢索引對應（融合 top-1、margin、開局 top-1、margin）
    Q = len(qrow); get = lambda rng_, j: np.array([res[i][j] for i in rng_])  # noqa: E731  取出一段查詢的某個欄位
    o, t_ = range(0, Q), range(Q, 2 * Q); base = dict(key=np.array([keys[r] for r in qrow]), pid=np.array(qpid), fold=np.array(qfold), y=np.array(qy), hd=hd[qrow], target_black=np.array([col[r] for r in qrow]) == "B")  # 對手側與本人側的範圍；驗證查詢的基本欄位
    np.savez_compressed(ROOT / "results/r_variants/m73_val.npz", **base,  # 寫出驗證查詢的結果
                        f_opp_lab=plab[get(o, 0)], f_opp_m=get(o, 1), f_opp_best=players[get(o, 0)], f_imp_lab=plab[get(t_, 0)], f_imp_m=get(t_, 1),  # f_ = 融合：對手側與冒充者
                        o_opp_lab=plab[get(o, 2)], o_opp_m=get(o, 3), o_opp_best=players[get(o, 2)], o_imp_lab=plab[get(t_, 2)], o_imp_m=get(t_, 3))  # o_ = 只用開局成分
    s0 = n0 + len(held); to, tt = range(s0, s0 + len(tg)), range(s0 + len(tg), s0 + 2 * len(tg))  # 測試對手側與本人側的範圍
    thd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", g[0])) else 0 for _, g in tg])  # 測試局的讓子數
    np.savez_compressed(ROOT / "results/r_variants/m73_test.npz", qid=np.array([q for q, _ in tg]), hd=thd, target_black=np.array([g[1] for _, g in tg]) == "B",  # 寫出測試查詢的結果
                        f_opp_lab=plab[get(to, 0)], f_opp_m=get(to, 1), f_opp_best=players[get(to, 0)], f_tgt_lab=plab[get(tt, 0)], f_tgt_m=get(tt, 1), f_tgt_best=players[get(tt, 0)],  # 融合：對手側與本人側
                        o_opp_lab=plab[get(to, 2)], o_opp_m=get(to, 3), o_opp_best=players[get(to, 2)], o_tgt_lab=plab[get(tt, 2)], o_tgt_m=get(tt, 3))  # 只用開局：對手側與本人側
    Path(ROOT / "results/r_variants/m73_fused_id.json").write_text(json.dumps(rep, indent=1)); log("written")  # 寫出權重與準度報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
