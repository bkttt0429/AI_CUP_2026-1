"""Section 4M M84 tooling (2026-11-11): cache of the five-component block score of every query, so that a new matcher variant is fused
in minutes instead of re-scoring the five components (40 minutes x 4 jobs). Same set-up as m82_cond_id.py's score stage (pools, held-out
games, queries, exclusions). block_i[c] = sum_j w_j z_ij[c] over the five components with M82's "5e" sub-weights (frozen), each z
standardised over the valid candidates of the query; the excluded candidate (a validation query's own player) is -inf. float16.
  --shard k/K -> results/m84_cache/block_{k}of{K}.npy (+ idx); shard 0 also writes block_held.npy and queries.npz (the query table:
  game key / test index, side, network, exclusion; players, labels, held-out truth)
"""
import argparse  # 解析分片參數
import json  # 讀取融合權重與驗證題
import multiprocessing as mp  # 平行建視角與評分
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案、同目錄與競賽一 aicup_pid 的程式
import ident_pool as ip  # noqa: E402  五個習慣成分的向量化模型
import m81_tactic_id as m81  # noqa: E402  訓練列與測試局的載入
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
COMPS = ip.COMPS  # 五個成分的名稱
M82 = ROOT / "results/m82"; CACHE = ROOT / "results/m84_cache"  # M82 的輸出與本快取的資料夾
OTHER = {"B": "W", "W": "B"}  # 對手的顏色
G = {}  # 子行程共用的全域資料（fork 後繼承）


def block_chunk(idx):  # 一批查詢的五成分區塊分數
    C = G["C"]; out = np.empty((len(idx), C), np.float16)  # 以 float16 存放
    for r, i in enumerate(idx):  # 逐一查詢
        v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); S = np.stack([sc[c] for c in COMPS]); ex = G["excl"][i]; valid = np.ones(C, bool)  # 五成分原始分數與要排除的候選人
        if ex >= 0:  # 驗證查詢要排除本人
            valid[ex] = False  # 本人不參與標準化
        b = G["w5"] @ ((S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12))  # 標準化後以凍結的子權重加總
        if ex >= 0:  # 被排除的本人
            b[ex] = -np.inf  # 設為負無限大
        out[r] = b  # 存入這一列
    return out  # 回傳區塊分數


def main(k, K):  # 計算第 k 個分片
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    w = json.loads((M82 / "report.json").read_text())["weights"]["5e"]; w5 = np.array([w[c] for c in COMPS]); log("frozen sub-weights " + " ".join(f"{c} {x:.2f}" for c, x in zip(COMPS, w5)))  # 凍結 M82「5e」融合中五個成分的子權重
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)  # 訓練列與候選人清單
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位候選人的等級標籤
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}; sgf = rows.sgf_content.tolist(); col = rows.color.tolist()  # 每列的玩家索引、棋局鍵、棋譜與執色
    ctx = mp.get_context("fork")  # 以 fork 讓子行程共用資料
    with ctx.Pool(12) as pool:  # 12 個工作行程
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))  # 每列以本人為主體的視角
    marks = np.array([v[0] for v in views]); qrow, qfold, qpid = [], [], []  # 視角標記；查詢列、折與玩家
    for v in (1, 2, 3, 4):  # 四個驗證折
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 本折的驗證題
        for pid, ng, gk in zip(m["player_id"], m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # 只用 n = 20 的題
                for k_ in gk:  # 這題的 20 局
                    qrow.append(key_index[k_]); qfold.append(v); qpid.append(pid)  # 加入一個查詢
    qrow = np.array(qrow); qfold = np.array(qfold); Q = len(qrow)  # 轉成陣列；驗證查詢數
    with ctx.Pool(12) as pool:  # 平行建對手側視角
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))  # 以對手為主體的視角
    held_keys = json.load(open(M82 / "held_keys.json")); held = np.array([key_index[k_] for k_ in held_keys]); H = len(held); include = np.ones(len(rows), bool); include[held] = False  # 沿用 M82 的留出局，不進候選池
    tg = m81.load_tests(); T = len(tg)  # 測試局
    with ctx.Pool(12) as pool:  # 平行建測試局視角
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))  # 本人側與對手側
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt; excl = np.array([pidx[p_] for p_ in qpid] * 2 + [-1] * H + [-1] * (2 * T))  # 查詢順序同 M82；驗證查詢排除本人
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]  # 查詢用到的開局鍵，只為這些建表
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log); G.update(qviews=qviews, pools=pools, excl=excl, C=C, w5=w5)  # 擬合候選池並放進全域
    n0 = 2 * Q; N = len(qviews)  # 驗證查詢總數與全部查詢數
    if k == 0:  # 只有分片 0 寫留出區塊與查詢表
        with ctx.Pool(12) as pool:  # 平行評分
            hb = np.concatenate(pool.map(block_chunk, np.array_split(np.arange(n0, n0 + H), 48)))  # 留出查詢的區塊分數
        np.save(CACHE / "block_held.npy", hb)  # 寫出留出局的區塊
        net = np.full(N, 4, np.int8); net[:Q] = net[Q:n0] = qfold - 1  # 在 (1, 2, 3, 4, -1) 中的網路索引
        side = np.zeros(N, np.int8); side[:Q] = 1; side[n0 + H:n0 + H + T] = 1  # 1 = 該局對手方的手
        qkey = np.array([keys[r] for r in qrow] * 2 + list(held_keys) + [""] * (2 * T)); qtest = np.full(N, -1, np.int32); qtest[n0 + H:n0 + H + T] = qtest[n0 + H + T:] = np.arange(T)  # 查詢的棋局鍵；測試查詢對應的測試局索引
        np.savez_compressed(CACHE / "queries.npz", qkey=qkey, qtest=qtest, side=side, net=net, excl=excl, players=players, plab=plab, Q=Q, H=H, T=T, held_truth=pl[held], w5=w5)  # 寫出查詢表
        log(f"held-out block and the query table written ({N} queries)")  # 記錄完成
    rest = np.concatenate([np.arange(n0), np.arange(n0 + H, N)]); mine = np.array_split(rest, K)[k]  # 留出局以外的查詢，取本分片
    with ctx.Pool(12) as pool:  # 平行評分
        blk = np.concatenate(pool.map(block_chunk, np.array_split(mine, 240)))  # 本分片的區塊分數
    np.save(CACHE / f"block_{k}of{K}.npy", blk); np.save(CACHE / f"idx_{k}of{K}.npy", mine); log(f"shard {k} of {K}: {len(mine)} queries, block {blk.shape} {blk.dtype} written")  # 寫出區塊與對應的查詢索引


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args(); main(*map(int, a_.shard.split("/")))  # 解析 k/K 後執行
