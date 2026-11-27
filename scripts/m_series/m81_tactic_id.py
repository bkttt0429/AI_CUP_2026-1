"""Section 4M M81 (2026-11-10): a sixth identification component -- tactical events -- next to M73's five (opening, style, local,
orient, hd). Every game is replayed on KataGo's Board (HD >= 2: handicap stones restored, actual colours); each non-pass move from ply 12
on becomes one token = (stones captured 0 / 1 / 2+) x (gives atari) x (response to an atari the opponent's last move made on an own group:
none / saved without capturing / saved by capturing / played on the liberty and still in atari / ignored) x (liberties of the played
stone's group 1 / 2 / 3 / 4+) x (ko: none / ko capture / move under a ko ban) x (own groups adjacent 0 / 1 / 2+) x (opponent groups
adjacent 0 / 1 / 2+) x (phase < 60 / < 150 / rest): 9,720 values, invariant under board symmetries and colour.
Per training player a multinomial shrunk towards the pool background; the shrinkage strength is the Dirichlet-multinomial precision
fitted on the pool (Minka's fixed point; no validation outcome involved). One pool for both colours.
Everything else is m73_fused_id.py verbatim (pools, held-out own games, queries, seed), so the five-component outputs must reproduce M73.
  --stage tac                 tokens of both sides of every training row and test game -> results/m81/tac_{train,test}_{ids,off}.npy
  --stage score --shard k/K   five- and six-component fusion of shard k of the queries -> results/m81/part{k}of{K}.npz (+ report, shard 0)
"""
import argparse  # 解析命令列參數
import json  # 讀取驗證題、寫出報告
import multiprocessing as mp  # 平行重放與評分
import re  # 從 SGF 取出讓子數 HD
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取訓練資料
from scipy.special import digamma  # 擬合 Dirichlet 精度用的 digamma

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案、同目錄與競賽一 aicup_pid 的程式
import ident_pool as ip  # noqa: E402  五個習慣成分的向量化模型
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序
from aicup.sgf import board_size, parse_moves, setup_and_moves  # noqa: E402  棋盤大小、著手與讓子石的解析
from katago.game.board import Board  # noqa: E402  KataGo 的棋盤，用來重放對局

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
COMPS = ip.COMPS  # 五個成分的名稱
OUT = ROOT / "results/m81"  # 戰術 token 與分片結果的資料夾
G = {}  # 子行程共用的全域資料（fork 後繼承）
OTHER = {"B": "W", "W": "B"}  # 對手的顏色
FIELDS = (("cap", 3), ("atk", 2), ("threat", 5), ("libs", 4), ("ko", 3), ("own", 3), ("opp", 3), ("phase", 3))  # token 的各欄位與取值個數
V_TAC = int(np.prod([n for _, n in FIELDS]))  # token 的取值總數（9,720）
START = 12  # 手數索引（0 起算）>= 12 才產生 token，即從第 13 手起
_ARR = ("board", "group_head", "group_stone_count", "group_liberty_count", "group_next", "group_prev")  # 要改成 Python list 加速存取的 Board 陣列


def decode(tid):  # 把 token 編號拆回各欄位（自我檢查用）
    out = {}  # 各欄位的值
    for name, n in reversed(FIELDS):  # 由最後一個欄位往前拆
        out[name] = int(tid % n); tid //= n  # 餘數是本欄位的值，再除掉這一位
    return out  # 回傳欄位字典


def tactic_ids(sgf, start=START, fast=True):  # 重放一局並產生雙方的戰術 token
    """{actual colour: [token ids]} of both sides' non-pass moves at ply >= start. fast: the Board's arrays as Python lists (same
    values, about 3x faster scalar access; the class only indexes them)."""
    try:  # 先試著取出讓子石與著手
        stones, moves = setup_and_moves(sgf)  # 讓子石（HD >= 2 時）與換回實際顏色的著手
    except ValueError:  # 讓子數不在讓子石表內（HD >= 10）時
        stones, moves = [], parse_moves(sgf)  # 不放讓子石，著手照記錄上的顏色
    b = Board(board_size(sgf))  # 建立空棋盤
    if fast:  # 改用 Python list 加速
        for a in _ARR:  # 逐一陣列
            setattr(b, a, getattr(b, a).tolist())  # numpy 陣列換成內容相同的 list
    for x, y in stones:  # 放上讓子石
        b.set_stone(Board.BLACK, b.loc(x, y))  # 讓子石為黑子
    bd, gh, gl, gs, adjs = b.board, b.group_head, b.group_liberty_count, b.group_stone_count, b.adj  # 常用陣列的區域別名
    out = {"B": [], "W": []}; last = [None, None, None]  # 每方的 token；每方上一手的位置（索引 1 黑、2 白）
    for t, (c, x, y) in enumerate(moves):  # 逐手重放
        pla = 1 if c == "B" else 2; opp = 3 - pla  # 本方與對方的編號
        if x is None:  # 虛手
            b.playUnsafe(pla, Board.PASS_LOC); last[pla] = None  # 下虛手並清除本方上一手
            continue  # 虛手不產生 token
        loc = b.loc(x, y)  # 落點位置
        if bd[loc] != 0:  # 落在已有子的點上無法重放：不產生 token
            last[pla] = None  # 清除本方上一手
            continue  # 跳過這一手
        own, oppg = set(), set()  # 相鄰的本方與對方棋串
        for d in adjs:  # 四個相鄰方向
            v = bd[loc + d]  # 相鄰點的狀態
            if v == pla:  # 本方的子
                own.add(int(gh[loc + d]))  # 記下所屬棋串
            elif v == opp:  # 對方的子
                oppg.add(int(gh[loc + d]))  # 記下所屬棋串
        cap = sum(int(gs[h]) for h in oppg if gl[h] == 1)  # 這手提掉的子數（只剩一氣的對方棋串）
        ko = 1 if (cap == 1 and b.wouldBeKoCapture(loc, pla)) else 2 if b.simple_ko_point is not None else 0  # 1 = 提劫、2 = 在劫爭禁著期間落子、0 = 無
        thr, thr_libs = [], []  # 被對方上一手叫吃的本方棋串與它們的氣
        lo = last[opp]  # 對方上一手的位置
        if lo is not None and bd[lo] == opp:  # 對方上一手的子仍在盤上
            for d in adjs:  # 檢查它的四個鄰點
                a = lo + d  # 鄰點位置
                if bd[a] == pla and gl[gh[a]] == 1:  # 本方棋串只剩一氣，即被叫吃
                    thr.append(a); b.findLiberties(a, thr_libs)  # 記下棋串與它的氣
        b.playUnsafe(pla, loc)  # 落子
        libs = min(max(int(gl[gh[loc]]), 1), 4) if bd[loc] == pla else 1  # 落子後所屬棋串的氣數（1 到 4+）
        atk = int(any(bd[loc + d] == opp and gl[gh[loc + d]] == 1 for d in adjs))  # 是否叫吃相鄰的對方棋串
        if not thr:  # 沒有被叫吃的棋串
            th = 0  # 無叫吃
        elif any(bd[a] == pla and gl[gh[a]] >= 2 for a in thr):  # 被叫吃的棋串已脫離叫吃
            th = 2 if cap > 0 else 1  # 靠提子救出為 2，否則為 1
        elif loc in thr_libs:  # 下在那口氣上但仍被叫吃
            th = 3  # 長出後仍被叫吃
        else:  # 沒有理會叫吃
            th = 4  # 不理會叫吃
        if t >= start:  # 手數索引（0 起算）達 start 才記錄
            tid = 0  # token 編號
            for val, (_, n) in zip((min(cap, 2), atk, th, libs - 1, ko, min(len(own), 2), min(len(oppg), 2), 0 if t < 60 else 1 if t < 150 else 2), FIELDS):  # 提子、叫吃、應對、氣、劫、相鄰本方、相鄰對方、階段
                tid = tid * n + val  # 混合進位編碼
            out[c].append(tid)  # 記到該方
        last[pla] = loc  # 更新本方上一手
    return out  # 回傳雙方的 token


def _tac_row(args):  # 一列訓練資料或一局測試的雙方 token
    sgf, col = args; d = tactic_ids(sgf)  # 重放整局
    return np.asarray(d[col], np.int16), np.asarray(d[OTHER[col]], np.int16)  # 先本人、後對手


def self_test(sgfs):  # 手工局面與兩種 Board 實作的自我檢查
    xy = lambda x, y: chr(97 + x) + chr(97 + y)  # noqa: E731  座標轉成 SGF 字母
    game = lambda seq: "(;SZ[19]HD[0]" + "".join(f";{c}[{xy(x, y)}]" for c, (x, y) in zip("BW" * 40, seq)) + ")"  # noqa: E731  由座標序列組出黑白交替的 SGF
    a = tactic_ids(game([(3, 3), (2, 3), (15, 15), (4, 3), (15, 3), (3, 2), (3, 4)]), start=0); t = decode(a["B"][3])  # 黑方第 4 手的 token
    assert (t["threat"], t["cap"], t["libs"], t["own"], t["opp"], t["atk"], t["ko"]) == (1, 0, 2, 1, 0, 0, 0), t  # 長出脫離叫吃
    a = tactic_ids(game([(3, 3), (2, 3), (15, 15), (4, 3), (15, 3), (3, 2), (3, 15)]), start=0); assert decode(a["B"][3])["threat"] == 4  # 不理會叫吃
    a = tactic_ids(game([(1, 0), (0, 0), (0, 1)]), start=0); t = decode(a["B"][1]); w = decode(a["W"][0])  # 角上提子的局面
    assert (t["cap"], t["ko"], t["own"], t["opp"], t["libs"]) == (1, 0, 0, 1, 2), t  # 角上提子，不是劫
    assert (w["libs"], w["atk"]) == (0, 0), w  # 被提的那顆子是自己送進叫吃
    a = tactic_ids(game([(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2), (16, 16), (1, 1), (2, 1), (15, 15)]), start=0)  # 劫的局面
    w = decode(a["W"][3]); t = decode(a["B"][4]); w2 = decode(a["W"][4])  # 取出要檢查的三手
    assert (w["atk"], w["libs"], w["own"], w["opp"], w["cap"]) == (1, 0, 0, 2, 0), w  # 撲入並叫吃
    assert (t["cap"], t["ko"], t["libs"], t["threat"]) == (1, 1, 0, 2), t  # 提劫，同時救出被叫吃的子
    assert w2["ko"] == 2, w2  # 下一手是在劫爭禁著期間落子
    t0 = time.time(); slow = [tactic_ids(s, fast=False) for s in sgfs]; t1 = time.time(); quick = [tactic_ids(s) for s in sgfs]; t2 = time.time()  # 兩種實作各跑一次並計時
    assert all(x == y for x, y in zip(slow, quick)), "list-backed Board differs from the numpy-backed one"  # list 版必須與 numpy 版完全相同
    print(f"self-test passed: 4 hand-made positions; list-backed = numpy-backed Board on {len(sgfs)} games ({(t1 - t0) / len(sgfs) * 1e3:.1f} vs {(t2 - t1) / len(sgfs) * 1e3:.1f} ms per game)", flush=True)  # 印出通過訊息與速度


def pack(arrs):  # 多段 token 陣列攤平成一維加偏移
    off = np.zeros(len(arrs) + 1, np.int64); off[1:] = np.cumsum([len(a) for a in arrs])  # 每段的起點偏移
    return np.concatenate(arrs).astype(np.int16), off  # 攤平的 token 與偏移


class Flat:  # 讀取攤平儲存的 token
    def __init__(self, name):  # 載入 token 與偏移
        self.ids = np.load(OUT / f"{name}_ids.npy"); self.off = np.load(OUT / f"{name}_off.npy")  # 攤平的 token 與每段偏移

    def get(self, j):  # 取第 j 段的 token
        return self.ids[self.off[j]:self.off[j + 1]]  # 依偏移切出


def load_rows():  # 讀入全部訓練列（其他腳本共用）
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)  # 合併所有訓練 CSV
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]  # 棋局鍵＝等級:棋局 ID
    return rows  # 回傳訓練列


def load_tests():  # 讀入測試局（其他腳本共用）
    questions = ip.read_tests([ROOT / "data/tests/rank_prediction_test_public.csv", ROOT / "data/tests/rank_prediction_test_private.csv"])  # Public 與 Private 測試題
    return [(q["question_id"], g) for q in questions for g in q["games"]]  # 每局一筆（題號, (棋譜, 執色)）


def stage_tac():  # tac 階段：產生全部對局的戰術 token
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    OUT.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    rows = load_rows(); sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); tg = load_tests(); log(f"{len(rows)} rows, {len(tg)} test games")  # 讀入訓練列與測試局
    self_test(sgf[:200])  # 先做自我檢查，以前 200 局比對兩種實作
    ctx = mp.get_context("fork")  # 以 fork 建立工作行程
    with ctx.Pool(12) as pool:  # 12 個工作行程
        res = list(pool.imap(_tac_row, zip(sgf, col), chunksize=512)); log("training rows tokenised")  # 訓練列的雙方 token
        rest = list(pool.imap(_tac_row, [g for _, g in tg], chunksize=128))  # 測試局的雙方 token
    for name, r in (("tac_train", res), ("tac_test", rest)):  # 訓練與測試分開存檔
        ids, off = pack([a for pair in r for a in pair]); np.save(OUT / f"{name}_ids.npy", ids); np.save(OUT / f"{name}_off.npy", off)  # 每局兩段（2j 本人、2j+1 對手）攤平後存檔
        log(f"{name}: {len(r)} games, {len(ids)} tokens ({len(ids) / max(2 * len(r), 1):.1f} per side), {len(np.unique(ids))} of {V_TAC} token values used")  # 記錄 token 數與用到的取值數
    ids = np.load(OUT / "tac_train_ids.npy")[::7].astype(np.int64); div = V_TAC  # 每 7 個取 1 統計欄位分布
    for name, n in FIELDS:  # 逐欄位
        div //= n; v = (ids // div) % n; print(f"  {name:7s} " + " ".join(f"{k}: {(v == k).mean():.4f}" for k in range(n)), flush=True)  # 拆出本欄位並印出各值的比例
    log("written")  # 完成


def eb_precision(counts, p_bg, s=50.0, iters=200):  # 以 Minka 固定點法估 Dirichlet 精度
    """Dirichlet-multinomial precision s for a fixed mean p_bg (Minka 2000, fixed point): the per-player counts are draws from
    Dirichlet(s * p_bg)-multinomials."""
    r, c = np.nonzero(counts); x = counts[r, c]; m = p_bg[c]; n = counts.sum(1); n = n[n > 0]  # 非零計數、對應的背景機率與每人總數
    for _ in range(iters):  # 固定點迭代
        new = s * float((m * (digamma(x + s * m) - digamma(s * m))).sum()) / float((digamma(n + s) - digamma(s)).sum())  # Minka 的更新式
        done = abs(new - s) < 1e-4 * s; s = float(np.clip(new, 1.0, 1e6))  # 判斷收斂並限制範圍
        if done:  # 已收斂
            break  # 結束迭代
    return s  # 回傳精度 s


def raw_scores(i):  # 第 i 個查詢的六成分原始分數
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); u, cnt = np.unique(G["qtac"][i], return_counts=True)  # 五成分分數；查詢的 token 計數
    tac = cnt.astype(np.float64) @ G["logpT"][u] if len(u) else np.zeros(G["C"])  # 戰術成分：各候選人的多項式對數似然
    return np.vstack([np.stack([sc[c] for c in COMPS]), tac[None]])  # 形狀 [6, C]：五成分加戰術成分


def top2(x, ex):  # 取最高分的候選人與領先第二名的差
    if ex >= 0:  # 有要排除的候選人（本人）
        x = x.copy(); x[ex] = -np.inf  # 本人的分數設為負無限大
    o = np.argpartition(-x, 1)[:2]; o = o[np.argsort(-x[o])]  # 取前兩名並排序
    return int(o[0]), float(x[o[0]] - x[o[1]])  # 第一名索引與 margin


def score_chunk(idx):  # 一批查詢的六成分與五成分融合 top-1
    out = []; w6, w5 = G["w6"], G["w5"]; C = G["C"]  # 兩組融合權重與候選人數
    for i in idx:  # 逐一查詢
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)  # 原始分數與要排除的候選人
        if ex >= 0:  # 驗證查詢要排除本人
            valid[ex] = False  # 本人不參與標準化
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)  # 每個成分對有效候選人標準化
        b6, m6 = top2(w6 @ Z, ex); b5, m5 = top2(w5 @ Z[:5], ex)  # 六成分與五成分融合的 top-1、margin
        out.append((i, b6, m6, b5, m5))  # 記下結果
    return out  # 回傳這批結果


def full_chunk(idx):  # 一批留出查詢的完整原始分數，供擬合權重
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]  # 以 float32 節省記憶體


def stage_score(k, K):  # score 階段：第 k 個分片的融合評分
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    rng = np.random.default_rng(20261110)  # 與 M73 相同的種子
    rows = load_rows()  # 讀入訓練列
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)  # 候選人（訓練玩家）清單與索引
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位候選人的等級標籤
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}  # 每列的玩家索引與棋局鍵
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist(); hd = np.array([int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0 for s in sgf])  # 棋譜、執色與讓子數
    ctx = mp.get_context("fork")  # 以 fork 讓子行程共用資料
    with ctx.Pool(12) as pool:  # 12 個工作行程
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))  # 每列以本人為主體的視角
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")  # 視角標記（B 或 W 候選池）
    qrow, qfold, qy, qpid = [], [], [], []  # 查詢列、折、真值與玩家
    for v in (1, 2, 3, 4):  # 四個驗證折
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 本折的驗證題
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # 只用 n = 20 的題
                for k_ in gk:  # 這題的 20 局
                    qrow.append(key_index[k_]); qfold.append(v); qy.append(RANK[rk]); qpid.append(pid)  # 加入一個查詢
    qrow = np.array(qrow)  # 轉成陣列
    with ctx.Pool(12) as pool:  # 平行建對手側視角
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))  # 以對手為主體的視角
    used = set(qrow.tolist()); held = []  # 驗證題用到的列；留出列
    by_player = rows.groupby("player_id").indices  # 每位玩家的列
    for p_ in rng.choice(players, 2400, replace=False):  # 抽 2,400 位玩家（同 M73）
        cand = [r for r in by_player[p_] if r not in used and hd[r] < 2]  # 不在驗證題且 HD < 2 的局
        if len(cand) >= 8:  # 至少 8 局才留出
            held += rng.choice(cand, 2, replace=False).tolist()  # 每人留出 2 局
    held = np.array(held); include = np.ones(len(rows), bool); include[held] = False  # 留出局不進候選池
    tg = load_tests()  # 測試局
    with ctx.Pool(12) as pool:  # 平行建測試局視角
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))  # 本人側與對手側
    log(f"query views: validation {len(qrow)} x 2, held-out {len(held)}, test {len(tg)} x 2")  # 記錄各類查詢數
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt  # 順序：驗證對手側、驗證本人側、留出局、測試對手側、測試本人側
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * len(held) + [-1] * (2 * len(tg))  # 驗證查詢排除本人，其餘不排除
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]  # 查詢用到的開局鍵，只為這些建表
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)  # 擬合 B、W 兩個視角的候選池
    # 戰術成分：一個候選池，黑白兩色共用
    train, test = Flat("tac_train"), Flat("tac_test"); assert len(train.off) == 2 * len(rows) + 1 and len(test.off) == 2 * len(tg) + 1  # 讀入 token，並確認每局兩段
    idx = np.flatnonzero(include); ids = np.concatenate([train.get(2 * r) for r in idx]).astype(np.int64); ptok = np.repeat(pl[idx], train.off[2 * idx + 1] - train.off[2 * idx])  # 候選池各列本人側的 token 與所屬玩家
    counts = np.bincount(ptok * V_TAC + ids, minlength=C * V_TAC).reshape(C, V_TAC).astype(np.float64); del ids, ptok  # 每位玩家各 token 值的次數
    bg = counts.sum(0); p_bg = (bg + 5.0 / V_TAC) / (bg.sum() + 5.0); s = eb_precision(counts, p_bg)  # 候選池的背景分布與 Dirichlet 精度
    logpT = np.ascontiguousarray(np.log((counts + s * p_bg) / (counts.sum(1, keepdims=True) + s)).T.astype(np.float32)); del counts  # 向背景收縮後的對數機率 [V, C]
    log(f"tactic pool: {int(bg.sum())} tokens, {int((bg > 0).sum())} token values in use, Dirichlet precision {s:.1f}")  # 記錄候選池規模與精度
    qtac = [train.get(2 * r + 1) for r in qrow] + [train.get(2 * r) for r in qrow] + [train.get(2 * r) for r in held] + [test.get(2 * j + 1) for j in range(len(tg))] + [test.get(2 * j) for j in range(len(tg))]  # 各查詢的 token，順序同 qviews
    assert len(qtac) == len(qviews)  # 確認與視角一一對應
    G.update(qviews=qviews, pools=pools, excl=excl, C=C, qtac=qtac, logpT=logpT)  # 放進全域給子行程使用
    n0 = 2 * len(qrow); hidx = np.arange(n0, n0 + len(held)); truth = pl[held]  # 留出查詢的索引與真正的玩家
    with ctx.Pool(12) as pool:  # 平行評分
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]  # 留出查詢的完整原始分數
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # 形狀 [6, Q, C]：成分、查詢、候選人
    Z = [ip.standardise(S[j]) for j in range(6)]; w6 = ip.fit_weights(Z, truth); w5 = ip.fit_weights(Z[:5], truth); names = (*COMPS, "tactic")  # 標準化後分別擬合六成分與五成分的權重
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731  真正玩家的名次（0 = 第一名）
    acc = {}  # 各成分與融合的認人準度
    for name, F in [(c, Z[j]) for j, c in enumerate(names)] + [("fused5", np.einsum("j,jqc->qc", w5, np.stack(Z[:5]))), ("fused6", np.einsum("j,jqc->qc", w6, np.stack(Z)))]:  # 六個成分各自與兩種融合
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}  # top-1／5／20 與名次中位數
    log("weights (six) " + " ".join(f"{c} {x:.2f}" for c, x in zip(names, w6)) + " | (five) " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w5)))  # 記錄兩組權重
    log("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k_} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k_, v in acc.items()))  # 記錄認人準度
    del full, S, Z  # 釋放記憶體
    G.update(w6=w6, w5=w5)  # 融合權重給子行程使用
    if k == 0:  # 只有分片 0 寫報告
        Path(OUT / "report.json").write_text(json.dumps({"weights6": dict(zip(names, map(float, w6))), "weights5": dict(zip(COMPS, map(float, w5))), "held_queries": int(len(held)), "heldout_accuracy": acc,  # 寫出權重與準度
                                                         "dirichlet_precision": s, "tokens": int(bg.sum()), "token_values_used": int((bg > 0).sum())}, indent=1))  # 以及戰術候選池的統計
        np.savez_compressed(OUT / "meta.npz", players=players, plab=plab, Q=len(qrow), H=len(held), T=len(tg))  # 候選人、標籤與各類查詢數
    rest = np.concatenate([np.arange(n0), np.arange(n0 + len(held), len(qviews))]); mine = np.array_split(rest, K)[k]  # 留出局以外的查詢，取本分片
    with ctx.Pool(12) as pool:  # 平行評分
        res = [x for part in pool.map(score_chunk, np.array_split(mine, 240)) for x in part]  # 本分片的融合結果
    res.sort(key=lambda x: x[0]); a = np.array([x[0] for x in res]); assert np.array_equal(a, mine)  # 依索引排序並確認完整
    np.savez_compressed(OUT / f"part{k}of{K}.npz", idx=a, b6=np.array([x[1] for x in res]), m6=np.array([x[2] for x in res]), b5=np.array([x[3] for x in res]), m5=np.array([x[4] for x in res]))  # 寫出本分片
    log(f"shard {k} of {K}: {len(mine)} queries scored and written")  # 記錄完成


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("tac", "score")); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args()  # 階段與分片參數
    if a_.stage == "tac":  # 產生 token
        stage_tac()  # 執行 tac 階段
    else:  # 融合評分
        stage_score(*map(int, a_.shard.split("/")))  # 解析 k/K 後執行 score 階段
