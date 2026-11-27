"""Section 4M M84 (2026-11-11): matcher variants with more games per candidate, fused through the cache of the five-component block
score (m84_cache.py), evaluated against the current identifier (M82's five + matcher, "5e") and turned into candidate files without
anyone attending. Network, loss and cross-fit as M82 (aicup_pid.embed.EmbedNet; fold v's queries use the network trained without fold
v's players); per episode one query game and up to 64 support games per player; a candidate's prototype = the sum over ALL his games.
  --stage embed --var ref|A|ALL    ref = M82's stored embeddings (cache check); A = base + x40 + x80 rows (up to 80 games per player);
                                   ALL = every training row (results/m84_rows from m84_rows_extract.py), seeds 0, 1, 2
  --stage eval --tag ref|A|B|B3    A: var A; B: var ALL member 0; B3: var ALL, the three members' standardised scores averaged
  --stage finalize                 upload_next/README.md + candidate files from the eval JSONs (also run at the end of every eval)
"""
import argparse  # 解析命令列參數
import hashlib  # 計算候選檔的 md5
import json  # 讀寫評估結果與報告
import multiprocessing as mp  # 平行處理主幹向量的分段檔
import shutil  # 複製候選檔到 upload_next/
import subprocess  # 呼叫 m78_build_candidate.py 建檔
import sys  # 設定匯入路徑、取得目前的直譯器
import time  # 計時與 README 的時間戳記
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案 src、實驗腳本與競賽一（aicup_pid）的 src
import ident_pool as ip  # noqa: E402  五成分辨識與融合權重擬合
import m71_step2 as s2  # noqa: E402  錨點證據的校正與融合
import m81_tactic_id as m81  # noqa: E402  讀取全部訓練對局
import m82_cond_id as m82  # noqa: E402  由主幹向量分段檔算每局列（M82 的定義）
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與配對差檢定
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  驗證折的成員機率與題目

CACHE, M82, RV, UP = ROOT / "results/m84_cache", ROOT / "results/m82", ROOT / "results/r_variants", ROOT / "upload_next"  # 快取、M82 輸出、識別器輸出與上傳指示的資料夾
NETS = (1, 2, 3, 4, -1)  # 交叉擬合的五個網路：v＝排除折 v 的玩家，-1＝看過全體玩家
TAGS = {"ref": ("ref", [0], None), "A": ("A", [0], ("46", "47")), "B": ("ALL", [0], ("48", "49")), "B3": ("ALL", [0, 1, 2], ("50", "51")),  # 評估標記：ref 快取檢查與 A、B、B3
        "P": ("P", [0], ("52", "53")), "P3": ("P", [0, 1, 2], ("54", "55")), "W": ("ref", [0], ("58", "59")), "R": ("R", [0], ("60", "61")), "C": ("PR", [0, 1, 2], ("62", "63")), "S": ("S", [0], ("66", "67")), "SP": ("S", [0], ("68", "69")), "S2": ("S2", [0], ("76", "77")), "SP2": ("S2", [0], ("78", "79")), "S2e": ("S2e", [0, 1, 2], ("80", "81")), "SP2e": ("S2e", [0, 1, 2], ("82", "83")),  # 分階段描述、權重、同等級回合、合併版與 M88／M90 逐手模型的標記
        "S3": ("S3", [0], ("86", "87")), "SP3": ("S3", [0], ("88", "89")), "S2x": ("S2x", [0, 1, 2, 3], ("90", "91")), "SP2x": ("S2x", [0, 1, 2, 3], ("92", "93")), "S3c": ("S3c", [0], ("94", "95")), "S2b": ("S2e", [1], None), "S2c": ("S2e", [2], None), "S2y": ("S2y", [0, 1, 2, 3, 4], ("96", "97")), "SP2y": ("S2y", [0, 1, 2, 3, 4], ("98", "99")),  # M101：S3＝M90 的配方加入外部玩家一起訓練（1 個種子）；S2x＝S2e 的三個種子加上 S3
        "S4": ("S4", [0], None), "SP4": ("S4", [0], None), "S4e": ("S4e", [0, 1, 2], None), "SP4e": ("S4e", [0, 1, 2], None), "S2y4": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4": ("S2y", [0, 1, 2, 3, 4], None), "SP2e4": ("S2e", [0, 1, 2], None), "SP4f": ("S4f", [0, 1, 2, 3], None), "SP2y4f": ("S2y", [0, 1, 2, 3, 4], None), "SP4et": ("S4et", [0, 1, 2], None), "SP2y4t1": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4t2": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4et": ("S2y", [0, 1, 2, 3, 4], None), "SP4ft": ("S4ft", [0, 1, 2, 3], None), "S4t": ("S4t", [0], None), "SP4t": ("S4t", [0], None), "SP2y4ft": ("S2y", [0, 1, 2, 3, 4], None), "S4s1": ("S4e", [1], None), "S4s2": ("S4e", [2], None), "S4s3": ("S4f", [3], None),  # M120／M122（S4f＝四個種子，第四個以遮手訓練）：對稱擴增的網路（沒有候選檔編號、不在 PRIORITY：正式檢定在 scripts/m_series/m121_sym_anchor_test.py）
        "Pp": ("P", [0], None), "Pp3": ("P", [0, 1, 2], None), "Cp": ("P", [0, 1, 2], None)}  # 標記 -> (var 檔、成員、ct 編號)；Pp／Pp3＝以目前已補算的列預覽 P（不建候選檔、不在 PRIORITY）
PRIORITY = ("SP2y", "S2y", "SP2x", "S2x", "SP2e", "S2e", "SP3", "S3", "S3c", "SP2", "S2", "SP", "S", "C", "P3", "P", "B3", "B", "R", "A", "W")  # 事先固定的採用順序：通過者中取最前面的一個
MULTI = {"S": [("S", [0])], "SP": [("P", [0, 1, 2]), ("S", [0])], "S2": [("S2", [0])], "SP2": [("P", [0, 1, 2]), ("S2", [0])],  # 多來源融合：標記 -> [(var 檔、成員), ...]
         "S2e": [("S2e", [0, 1, 2])], "SP2e": [("P", [0, 1, 2]), ("S2e", [0, 1, 2])],  # M90 三個種子平均，單獨或加分階段比對
         "S3": [("S3", [0])], "SP3": [("P", [0, 1, 2]), ("S3", [0])], "S2x": [("S2x", [0, 1, 2, 3])], "SP2x": [("P", [0, 1, 2]), ("S2x", [0, 1, 2, 3])],  # M101 的外部玩家版與四成員平均
         "S3c": [("S3c", [0])], "S2b": [("S2e", [1])], "S2c": [("S2e", [2])],  # M105：M90 的種子 1、2 各自當單一的投票來源（不在 PRIORITY：沒有候選檔）
         "S2y": [("S2y", [0, 1, 2, 3, 4])], "SP2y": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4])],  # 五成員平均（三個種子＋外部玩家版＋對照版）
         "S4": [("S4", [0])], "SP4": [("P", [0, 1, 2]), ("S4", [0])], "S4e": [("S4e", [0, 1, 2])], "SP4e": [("P", [0, 1, 2]), ("S4e", [0, 1, 2])],  # M120：對稱擴增的網路，種子 0／三個種子
         "S2y4": [("S2y", [0, 1, 2, 3, 4]), ("S4e", [0, 1, 2])], "SP2y4": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4e", [0, 1, 2])], "SP2e4": [("P", [0, 1, 2]), ("S2e", [0, 1, 2]), ("S4e", [0, 1, 2])],  # 舊逐手成員再加對稱擴增的三個種子
         "SP4et": [("P", [0, 1, 2]), ("S4et", [0, 1, 2])], "SP2y4t1": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0])], "SP2y4t2": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0, 1])],  # 方向平均的新種子逐一加入（1／2 個）
         "SP2y4et": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0, 1, 2])], "SP4ft": [("P", [0, 1, 2]), ("S4ft", [0, 1, 2, 3])],  # S4et／S4ft＝查詢向量做方向平均的三個／四個新種子（M124）
         "S4t": [("S4t", [0])], "SP4t": [("P", [0, 1, 2]), ("S4t", [0])], "SP2y4ft": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4ft", [0, 1, 2, 3])],  # M124：查詢向量對棋盤的八個方向取平均
         "S4s1": [("S4e", [1])], "S4s2": [("S4e", [2])], "S4s3": [("S4f", [3])],  # 新網路的單一種子（只報告：純對稱／＋藏開局／＋遮手）
         "SP4f": [("P", [0, 1, 2]), ("S4f", [0, 1, 2, 3])], "SP2y4f": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4f", [0, 1, 2, 3])]}  # M88 起：快取的區塊分數與一到三個配對來源融合，每個來源一個權重
CGRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # 階 C 的配對器權重候選值
WGRID = (1.0, 1.5, 2.0, None, 3.0, 4.0, 6.0)  # None＝M82 的權重（2.53），作為對照
DESC = {"A": "每位候選用到 80 局", "B": "每位候選用全部對局", "B3": "全部對局＋3 個 seed 平均", "P": "全部對局＋分階段（序盤／中盤／後半）的描述", "P3": "全部對局＋分階段的描述＋3 個 seed 平均", "W": "比對成分的融合權重改以錨點分數（交叉擬合）選定", "R": "比對網路一半的訓練回合只拿同等級的玩家互相區分（至多 80 局）", "C": "合併版：全部對局＋分階段的描述＋同等級回合＋3 個 seed 平均＋權重以錨點分數選定",  # README 用的各變體中文說明
        "S": "逐手的認人模型取代原本的比對", "SP": "逐手的認人模型＋分階段描述的比對，兩者並用",  # S、SP 的說明
        "S2": "用全部對局訓練的逐手認人模型取代原本的比對", "SP2": "用全部對局訓練的逐手認人模型＋分階段描述的比對",  # S2、SP2 的說明
        "S2e": "全部對局的逐手認人模型（3 個 seed 平均）取代原本的比對", "SP2e": "全部對局的逐手認人模型（3 個 seed 平均）＋分階段描述的比對",  # S2e、SP2e 的說明
        "S3": "加入外部公開棋譜玩家一起訓練的逐手認人模型（1 個 seed）取代原本的比對", "SP3": "加入外部玩家訓練的逐手認人模型（1 個 seed）＋分階段描述的比對",  # S3、SP3 的說明
        "S2x": "逐手認人模型 4 個成員平均（3 個 seed＋外部玩家版）取代原本的比對", "SP2x": "逐手認人模型 4 個成員平均（3 個 seed＋外部玩家版）＋分階段描述的比對",  # S2x、SP2x 的說明
        "S3c": "2 萬步排程的逐手認人模型（對照，不加外部玩家，1 個 seed）取代原本的比對", "S2y": "逐手認人模型 5 個成員平均（3 個 seed＋外部玩家版＋對照版）取代原本的比對", "SP2y": "逐手認人模型 5 個成員平均（3 個 seed＋外部玩家版＋對照版）＋分階段描述的比對"}  # S3c、S2y、SP2y 的說明
STEPS, EPISODE, S_EP, W_MAX, MIN_GAMES = 6000, 512, 64, 256, 8  # 訓練步數、每回合玩家數、支撐局數上限、每人抽樣表寬度上限、參與訓練的最少局數
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)  # 計分矩陣：完全正確 1、差一級 1/e


def load_rows(var, log):  # 讀取配對器訓練用的每局列
    """key, pid, fold, T of every game side with a trunk row, for the support set of `var` (base + x40 [+ x80 [+ the rest]])."""
    B = np.load(M82 / "rows_train.npz"); key, pid, fold, T = [B["key"]], [B["pid"]], [B["fold"]], [B["T"]]; pfold = dict(zip(B["pid"], B["fold"]))  # M82 的基礎列（base＋x40）與玩家→折的對照
    if var in ("A", "R", "ALL"):  # 這些變體再加上第 41–80 局（x80）
        with mp.get_context("fork").Pool(12) as pool:  # 12 個行程平行讀取
            res = pool.map(m82.part_rows, sorted((ROOT / "results/trunk_vec_x80").glob("fold*/games_s*_p*.csv")), chunksize=4)  # 每個 x80 分段檔的每局列
        k = np.concatenate([t["game_key"].to_numpy() for t, _, _ in res]).astype(str); p = np.concatenate([t["player_id"].to_numpy() for t, _, _ in res]).astype(str)  # 局鍵與玩家編號
        key.append(k); pid.append(p); fold.append(np.array([pfold.get(x, -9) for x in p], np.int8)); T.append(np.concatenate([o[0] for _, o, _ in res])); log(f"x80 rows: {len(k)} games")  # 併入 x80 的本人方列；查不到折的玩家記為 -9
    if var == "ALL":  # ALL 再加上其餘補算的對局
        rows = m81.load_rows(); owner = dict(zip(rows.game_key, rows.player_id)); del rows; ks, ts = [], []  # 局鍵→玩家的對照，隨即釋放原表
        for f in sorted((ROOT / "results/m84_rowsp").glob("rows_*.npz")):  # 逐一讀 m84_rows_extract.py 補算的列檔
            if f.name.endswith(".tmp.npz"):  # 尚未寫完的暫存檔
                continue  # 略過
            z = np.load(f); ks.append(z["key"].astype(str)); ts.append(z["T"])  # 收集局鍵與本人方的列
        k = np.concatenate(ks); p = np.array([owner[x] for x in k]); key.append(k); pid.append(p); fold.append(np.array([pfold.get(x, -9) for x in p], np.int8)); T.append(np.concatenate(ts)); log(f"extracted rows: {len(k)} games")  # 併入補算的列並記錄局數
    key, pid, fold, T = np.concatenate(key), np.concatenate(pid), np.concatenate(fold), np.concatenate(T); _, first = np.unique(key, return_index=True); first.sort()  # 合併後依局鍵去重，保留第一次出現且維持原順序
    return key[first], pid[first], fold[first], T[first]  # 回傳去重後的列


def part_phase(csv_path):  # 一個主幹向量分段檔的分階段平均
    """(game keys or test task numbers, phase means [2 sides, G, 384]) of one trunk-vector part: the mean 128-dim vector over a side's
    plies with t < 60, 60 <= t < 150, t >= 150 (zeros when a phase has no ply) -- m84_rows_extract.side_rows' definition."""
    import pandas as pd  # 讀分段檔的對局表
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)  # 分段檔的對局表
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:  # 對應的逐手向量檔
        X = np.hstack([z["g"], z["l"]]).astype(np.float64); target, row, t = z["target"], z["game_row"], z["t"]  # 每手 128 維（盤面 64＋落點 64）、是否本人、所屬局、手數
    n_g = len(tab); ph = np.zeros((2, n_g, 384), np.float16)  # 兩方 × 局數 × 三階段各 128 維
    for side, m in enumerate((target, ~target)):  # 0＝本人方、1＝對手方
        for k, pm in enumerate((t < 60, (t >= 60) & (t < 150), t >= 150)):  # 三個階段：序盤、中盤、後半
            mm = m & pm; r = row[mm]; n = np.bincount(r, minlength=n_g); nz = n > 0  # 該方在該階段的著手與每局手數
            if nz.any():  # 有任何著手才計算
                order = np.argsort(r, kind="stable"); ph[side][nz, 128 * k:128 * (k + 1)] = np.add.reduceat(X[mm][order], (np.cumsum(n) - n)[nz], axis=0) / n[nz][:, None]  # 依局分段加總再除以手數
    return (tab["game_key"].to_numpy().astype(str) if "player_id" in tab.columns else tab["task"].to_numpy().astype(int)), ph  # 訓練分段檔回傳局鍵，測試分段檔回傳測試棋局編號


def stage_embed(var):  # 訓練配對器變體並寫出原型與查詢向量
    import torch  # 需要 GPU 的階段才匯入
    from aicup_pid.embed import EmbedNet  # 競賽一的嵌入網路（與 M82 相同）
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); N = len(q["qkey"]); side, qnet, qtest = q["side"], q["net"], q["qtest"]  # 查詢表：候選人、查詢數、對局方、負責的網路、測試編號
    B = np.load(M82 / "rows_train.npz"); bkey = {k: i for i, k in enumerate(B["key"])}; Rt = np.load(M82 / "rows_test.npz"); held = set(json.load(open(M82 / "held_keys.json")))  # M82 的訓練列、測試列與留出局
    tr = np.flatnonzero(qtest < 0); src = np.array([bkey[k] for k in q["qkey"][tr]]); te = np.flatnonzero(qtest >= 0); QX = np.zeros((N, 256), np.float32)  # 非測試查詢（驗證與留出局）在 M82 訓練列的位置、測試查詢與查詢輸入列
    QX[tr] = np.where(side[tr][:, None] == 1, B["O"][src], B["T"][src]); QX[te] = np.where(side[te][:, None] == 1, Rt["O"][qtest[te]], Rt["T"][qtest[te]])  # 依查詢的一方取對手列或本人列
    if var == "ref":  # M82 自己的嵌入，用來檢查快取能否重現 M82 的融合
        protos = np.zeros((1, 5, C, 128), np.float32); qvec = np.zeros((1, N, 128), np.float16); keep = np.array([k not in held for k in B["key"]]); cand = np.array([pidx[p] for p in B["pid"]])  # 原型與查詢向量；原型排除留出局
        for j, v in enumerate(NETS):  # 五個交叉擬合網路
            eT = np.load(M82 / f"emb_T_{v}.npy"); Pm = np.zeros((C, 128)); np.add.at(Pm, cand[keep], eT[keep]); protos[0, j] = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 原型＝候選人各局嵌入之和再單位化
            if v > 0:  # 交叉擬合網路負責驗證查詢
                eO = np.load(M82 / f"emb_O_{v}.npy"); s = tr[qnet[tr] == j]; g = np.array([bkey[k] for k in q["qkey"][s]]); qvec[0, s] = np.where(side[s][:, None] == 1, eO[g], eT[g])  # 依一方取對手或本人的嵌入
            else:  # 全體網路（-1）
                s = tr[qnet[tr] == j]; qvec[0, s] = eT[[bkey[k] for k in q["qkey"][s]]]; eo, et = np.load(M82 / "emb_test_O.npy"), np.load(M82 / "emb_test_T.npy"); qvec[0, te] = np.where(side[te][:, None] == 1, eo[qtest[te]], et[qtest[te]])  # 留出局與測試查詢的嵌入
        np.savez(CACHE / "var_ref.npz", protos=protos, qvec=qvec); log("var ref written from M82's embeddings"); return  # 寫出 var_ref 後結束
    key, pid, fold, T = load_rows("ALL" if var in ("P", "PR") else var, log); is_held = np.array([k in held for k in key]); owner = np.array([pidx[p] for p in pid]); seeds = (0,) if var in ("A", "R") else (0, 1, 2)  # 載入支撐列；A、R 只訓練種子 0，其餘三個種子
    if var in ("P", "PR"):  # 分階段平均接到列後面：256 → 640 維，對局與查詢都接
        kidx = {k: i for i, k in enumerate(key)}; PT = np.zeros((len(key), 384), np.float16); PTB = np.zeros((len(B["key"]), 384), np.float16); POB = np.zeros_like(PTB); PTt = np.zeros((len(Rt["qid"]), 384), np.float16); POt = np.zeros_like(PTt)  # 支撐列、M82 訓練列（本人／對手）與測試列的階段平均
        csvs = [f for d_ in ("trunk_vec", "trunk_vec_x40", "trunk_vec_x80") for f in sorted((ROOT / "results" / d_).glob("fold*/games_s*_p*.csv"))]  # 已有逐手向量的訓練分段檔
        with mp.get_context("fork").Pool(12) as pool:  # 平行計算階段平均
            res = pool.map(part_phase, csvs, chunksize=4); rest = pool.map(part_phase, sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv")), chunksize=4)  # 訓練與測試的分段檔
        for ident, ph in res:  # 逐一訓練分段檔
            ii = np.array([kidx.get(k, -1) for k in ident]); ok = ii >= 0; PT[ii[ok]] = ph[0][ok]; jb = np.array([bkey.get(k, -1) for k in ident]); ok = jb >= 0; PTB[jb[ok]] = ph[0][ok]; POB[jb[ok]] = ph[1][ok]  # 填入支撐列與 M82 列的本人／對手階段平均
        for ident, ph in rest:  # 逐一測試分段檔
            PTt[ident] = ph[0]; POt[ident] = ph[1]  # 依測試棋局編號填入
        n_ex = 0  # 由補算檔取得的局數
        for f in sorted((ROOT / "results/m84_rowsp").glob("rows_*.npz")):  # 沒有逐手向量的對局：用補算時存下的階段平均
            if f.name.endswith(".tmp.npz"):  # 尚未寫完的暫存檔
                continue  # 略過
            z = np.load(f); ii = np.array([kidx.get(k, -1) for k in z["key"].astype(str)]); ok = ii >= 0; PT[ii[ok]] = z["PT"][ok]; n_ex += int(ok.sum())  # 填入並計數
        log(f"phase means: {sum(len(i_) for i_, _ in res)} games from stored parts, {n_ex} extracted, {len(PTt)} test; games with no phase mean at all: {int((np.abs(PT).sum(1) == 0).sum())}")  # 記錄各來源局數與仍缺階段平均的局數
        T = np.hstack([T, PT.astype(np.float32)]); del PT; QP = np.zeros((N, 384), np.float32)  # 支撐列接上階段平均
        QP[tr] = np.where(side[tr][:, None] == 1, POB[src], PTB[src]); QP[te] = np.where(side[te][:, None] == 1, POt[qtest[te]], PTt[qtest[te]]); QX = np.hstack([QX, QP])  # 查詢列也接上對應一方的階段平均
    D_IN = T.shape[1]  # 網路輸入維度（256 或 640）
    log(f"var {var}: {len(key)} games of {len(np.unique(owner))} players; games per player: median {int(np.median(np.bincount(owner, minlength=C)))}, max {int(np.bincount(owner, minlength=C).max())}")  # 記錄局數、玩家數與每人局數
    protos = np.zeros((len(seeds), 5, C, 128), np.float32); qvec = np.zeros((len(seeds), N, 128), np.float16); owner_t = torch.as_tensor(owner, device=dev); rep = {}  # 各種子 × 五個網路的原型與查詢向量
    for si, seed in enumerate(seeds):  # 逐一種子
        for j, v in enumerate(NETS):  # 逐一交叉擬合網路
            torch.manual_seed(1000 * seed + j); rng = np.random.default_rng(1000 * seed + j); use = ~is_held & (fold != v); n_rows = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(n_rows >= MIN_GAMES); W = int(min(n_rows.max(), W_MAX))  # 排除留出局與折 v 的玩家；局數夠的玩家才參與訓練
            idx = rng.permutation(np.flatnonzero(use)); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos = np.arange(len(o)) - np.searchsorted(o, o, side="left"); k_ = pos < W  # 打亂後依玩家排序，算每局在該玩家內的位置
            tab = np.full((C, W), -1, np.int64); tab[o[k_], pos[k_]] = idx[k_]; cnt = np.minimum(n_rows, W)  # 玩家 × 局的索引表（-1＝空）與每人可用局數
            mu = T[use].mean(0); sd = T[use].std(0) + 1e-6; std = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731  以訓練列的平均與標準差標準化
            X = std(T); tab_t = torch.as_tensor(tab[tp], device=dev); cnt_t = torch.as_tensor(cnt[tp], device=dev); P = min(EPISODE, len(tp)); S = min(S_EP + 1, W)  # 每回合 P 位玩家，每人 1 局查詢＋至多 64 局支撐
            net = EmbedNet(D_IN, 512, 128, 0.1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)  # 網路、AdamW 與 OneCycle 排程（同 M82）
            ar = torch.arange(W, device=dev); tgt_all = torch.arange(P, device=dev); net.train()  # 位置索引與每回合的正解標籤
            rk_ = torch.as_tensor(q["plab"][tp], device=dev); by_rank = [torch.nonzero(rk_ == r_)[:, 0] for r_ in range(10)] if var in ("R", "PR") else None  # 階 R、C：每隔一個回合只在同一個等級內抽人
            for step in range(STEPS):  # 訓練迴圈
                if by_rank is not None and step % 2 == 1:  # 奇數步：同等級回合
                    pool_ = by_rank[(step // 2) % 10]; b = pool_[torch.randperm(len(pool_), device=dev)[:P]]  # 各等級輪流，從該等級抽玩家
                else:  # 偶數步或其他變體
                    b = torch.randperm(len(tp), device=dev)[:P]  # 跨等級隨機抽玩家
                Pn = len(b); tgt = tgt_all[:Pn]; c = cnt_t[b]; u = torch.rand(Pn, W, device=dev).masked_fill(ar[None] >= c[:, None], 2.0)  # 隨機排序鍵，空位設為 2 排到最後
                rows = tab_t[b].gather(1, u.argsort(dim=1)[:, :S]); zq = net(X[rows[:, 0]]); sm = ar[None, 1:S] < c[:, None]  # 每人隨機取 S 局：第一局當查詢，其餘當支撐
                zs = net(X[rows[:, 1:].clamp(min=0)].reshape(-1, D_IN)).reshape(Pn, S - 1, -1) * sm[..., None]  # 支撐局的向量，空位歸零
                logits = net.log_scale.exp() * zq @ torch.nn.functional.normalize(zs.sum(1), dim=-1).T  # 查詢對各原型（支撐之和單位化）的相似度
                loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))  # 雙向交叉熵
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()  # 更新參數與學習率
            net.eval()  # 切換到推論模式
            with torch.no_grad():  # 推論不需梯度
                enc = lambda A: torch.cat([net(std(A[i:i + 65536])) for i in range(0, len(A), 65536)])  # noqa: E731  分批編碼，每批 65,536 列
                eT = enc(T); keep = torch.as_tensor(~is_held, device=dev); Pm = torch.zeros(C, 128, device=dev).index_add_(0, owner_t[keep], eT[keep]); protos[si, j] = torch.nn.functional.normalize(Pm, dim=-1).cpu().numpy()  # 原型＝候選人全部非留出局的向量之和再單位化
                s = np.flatnonzero(qnet == j); qvec[si, s] = enc(QX[s]).cpu().numpy()  # 由這個網路負責的查詢向量
            log(f"  seed {seed} net {v}: {len(tp)} players, {int(use.sum())} rows, final loss {float(loss.detach()):.4f}, last episode top-1 {float((logits.argmax(1) == tgt).float().mean()):.3f}")  # 記錄訓練玩家數、列數、最後的損失與命中
        Q, H = int(q["Q"]), int(q["H"]); hq = np.arange(2 * Q, 2 * Q + H); S_ = qvec[si, hq].astype(np.float32) @ protos[si, 4].T; truth = q["held_truth"]; r = (S_ > S_[np.arange(H), truth][:, None]).sum(1)  # 留出本人局：以全體網路的原型算正解的名次
        rep[f"seed{seed}"] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean())}; log(f"seed {seed}: matcher alone on the held-out own games, top-1 / top-5 / top-20 {rep[f'seed{seed}']['top1']:.3f} / {rep[f'seed{seed}']['top5']:.3f} / {rep[f'seed{seed}']['top20']:.3f} (M82: 0.074 / 0.181 / 0.309)")  # 配對器單獨的 top-1／5／20，與 M82 對照
    np.savez(CACHE / f"var_{var}.npz", protos=protos, qvec=qvec); (CACHE / f"var_{var}.json").write_text(json.dumps({"games": int(len(key)), "heldout": rep}, indent=1)); log("written")  # 寫出變體的向量檔與摘要


def anchors(z, seeds):  # 各折的錨點分數
    """Per fold: scores of the ensemble posterior with the anchor evidence of identifier z (vectorised expected-score decision), and of the base."""
    out = []  # 每折一組結果
    for v in FOLDS:  # 驗證折 1–4
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 該折的題目與基礎集成的平均機率
        te, logE, _ = s2.evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}; L = np.zeros_like(P)  # 以其餘三折校正的每局對數證據
        for i, gk in enumerate(q["game_keys"]):  # 逐題
            rows = [key_row[k] for k in gk if k in key_row]  # 該題有證據的對局
            if rows:  # 有證據時
                L[i] = logE[rows].sum(axis=0)  # 整題的對數證據＝各局相加
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); out.append((SM[(Pf @ SM).argmax(1), y], SM[(P @ SM).argmax(1), y], np.asarray(y), np.asarray(n), np.asarray(pid)))  # 融合後與基礎的期望分數決策得分
    return out  # 回傳各折結果


def stage_eval(tag):  # 單一配對來源的評估（取代 5e 的配對器）
    import torch  # 需要 GPU 的階段才匯入
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    var, members, _ = TAGS[tag]; dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)  # 標記設定與查詢表
    V = np.load(CACHE / f"var_{var}.npz"); protos = torch.as_tensor(V["protos"][members], device=dev); qvec = V["qvec"][members]  # 所選成員的原型與查詢向量

    def zm(ids):  # 查詢的配對器分數
        """Matcher score of queries `ids` [n, C]: per member standardised over the valid candidates, averaged over the members."""
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool)  # 要排除的候選人（驗證查詢的本人）
        if (ex >= 0).any():  # 有需要排除的查詢
            r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False  # 標成無效
        out = torch.zeros(len(ids), C, device=dev); nj = torch.as_tensor(net[ids], device=dev).long()  # 累加的分數與每個查詢負責的網路
        for m in range(len(members)):  # 逐一成員
            S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)  # 該成員的查詢向量
            for j in range(5):  # 五個交叉擬合網路
                s = nj == j  # 由網路 j 負責的查詢
                if s.any():  # 有才計算
                    S[s] = qv[s] @ protos[m, j].T  # 與網路 j 原型的餘弦相似度
            cnt = valid.sum(1, keepdim=True); mu = (S * valid).sum(1, keepdim=True) / cnt; sd = (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12); out += (S - mu) / sd  # 在有效候選人上標準化後累加
        return out / len(members), valid  # 成員平均與有效遮罩
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy").astype(np.float32); zh = torch.cat([zm(hq[i:i + 4096])[0] for i in range(0, H, 4096)]).cpu().numpy(); truth = q["held_truth"]  # 留出本人局的區塊分數與配對器分數
    w5e = json.loads((M82 / "report.json").read_text())["weights"]["5e"]["matcher"]; w = np.array([1.0, w5e]) if tag == "ref" else ip.fit_weights([hb, zh], truth)  # ref 沿用 M82 的權重，其餘在留出局上重新擬合
    r = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1)  # noqa: E731  正解候選人的名次
    rk = r(w[0] * hb + w[1] * zh); acc = {"top1": float((rk == 0).mean()), "top5": float((rk < 5).mean()), "top20": float((rk < 20).mean())}  # 融合後的 top-1／5／20
    log(f"tag {tag}: weights block {w[0]:.2f}, matcher {w[1]:.2f}; held-out own games, fused top-1 / top-5 / top-20 {acc['top1']:.3f} / {acc['top5']:.3f} / {acc['top20']:.3f} (M82 5e: 0.268 / 0.437 / 0.588)")  # 記錄權重與留出局命中
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]  # 讀入四個分片的區塊分數，依查詢編號排序
    assert np.array_equal(idx, np.concatenate([np.arange(2 * Q), np.arange(2 * Q + H, 2 * Q + H + 2 * T)])); b = np.zeros(len(idx), np.int64); m = np.zeros(len(idx))  # 確認涵蓋留出局以外的所有查詢；準備最佳候選人與 margin
    for i in range(0, len(idx), 2048):  # 每批 2,048 個查詢
        ids = idx[i:i + 2048]; z, valid = zm(ids); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev) + w[1] * z; F = F.masked_fill(~valid, -float("inf")); top = F.topk(2, dim=1)  # 融合分數，排除無效候選人後取前兩名
        b[i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()  # 第一名與第一、二名之差（margin）
    del blk; log("all queries fused")  # 釋放區塊分數
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}  # 查詢表各段（驗證對手側、驗證本人側、測試對手側、測試本人側）與 M82 的基本欄位
    val = {**base, "f_opp_lab": plab[b[o]], "f_opp_m": m[o], "f_opp_best": players[b[o]], "f_imp_lab": plab[b[t_]], "f_imp_m": m[t_]}  # 驗證端：對手與冒充者的等級與 margin
    np.savez_compressed(RV / f"m84_{tag}_val.npz", **val); np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[to]], f_opp_m=m[to], f_opp_best=players[b[to]], f_tgt_lab=plab[b[tt]], f_tgt_m=m[tt], f_tgt_best=players[b[tt]])  # 寫出識別器輸出（驗證與測試）
    same = float((val["f_opp_best"] == d82["f5e_opp_best"]).mean()); same_t = float((players[b[to]] == t82["f5e_opp_best"]).mean()); rep = {"weights": w.tolist(), "heldout_fused": acc, "same_top_candidate_as_5e": {"validation": same, "test": same_t}}  # 與 5e 頂尖候選人相同的比例
    log(f"same top candidate as M82's 5e identifier: validation opponent side {same:.4f}, test opponent side {same_t:.4f}" + (f" -> cache check {'OK' if min(same, same_t) >= 0.98 else 'FAILED'}" if tag == "ref" else ""))  # ref 時一併判定快取檢查
    if tag == "ref":  # 快取檢查的分支
        rep["cache_ok"] = bool(min(same, same_t) >= 0.98); (RV / "m84_ref_eval.json").write_text(json.dumps(rep, indent=1)); return  # 一致率 ≥ 0.98 才算通過，寫出後結束
    ZN = {**base, "opp_lab": val["f_opp_lab"], "opp_m": val["f_opp_m"], "imp_m": val["f_imp_m"], "imp_lab": val["f_imp_lab"]}; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; ok = []  # 新識別器與 5e 的錨點輸入
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子（主檢定與重複驗證）
        A, B_ = anchors(ZN, seeds), anchors(ZO, seeds); y, n, pid = (np.concatenate([x[k] for x in A]) for k in (2, 3, 4)); res = {}  # 兩個識別器的各折錨點分數
        for label, a, c in (("new_vs_base", [x[0] for x in A], [x[1] for x in A]), ("5e_vs_base", [x[0] for x in B_], [x[1] for x in B_]), ("new_vs_5e", [x[0] for x in A], [x[0] for x in B_])):  # 三組比較：新對基礎、5e 對基礎、新對 5e
            dd = paired_diff(np.concatenate(a), np.concatenate(c), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(u, x[2], x[3], sizes=SIZES) - balanced_mean_over_n(v_, x[2], x[3], sizes=SIZES) for u, v_, x in zip(a, c, A)]  # 玩家層 bootstrap 的配對差與各折差
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}  # M37：CI 下界 > 0、≥ 3/4 折為正、差 ≥ 0.0025
            log(f"  seeds {seeds} {label:12s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")  # 記錄差、信賴區間與各折
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["new_vs_5e"]["pass"])  # 主檢定看新識別器對 5e
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False  # 讀快取檢查的結果
    rep["cache_ok"] = cache_ok; rep["adopt"] = bool(all(ok) and cache_ok); log(f"PRE-REGISTERED: variant {tag} passes against the 5e identifier: {'YES' if rep['adopt'] else 'NO'}" + ("" if cache_ok else " (cache check missing or failed)"))  # 兩組種子都過且快取檢查通過才採用
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float))  # 寫出評估結果
    if TAGS[tag][2] is not None:  # 有候選檔編號的變體
        finalize()  # 更新上傳指示與候選檔


def stage_eval_multi(tag):  # 多個配對來源的融合評估（M88 起）
    """M88: as stage_eval, for a fusion of the cached block with several matcher sources (MULTI[tag]); one weight per source, fitted on the
    held-out own games together with the block's."""
    import torch  # 需要 GPU 的階段才匯入
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []  # 查詢表；src 收集各來源
    for var, members in MULTI[tag]:  # 逐一配對來源
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))  # 只取實際存在的成員（少了種子時自動退回）

    def zs(ids):  # 各來源的配對器分數
        """Per source the matcher score of queries `ids` [n, C] (standardised over the valid candidates, averaged over the members) and the valid mask."""
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); cnt = valid.sum(1, keepdim=True); out = []  # 排除本人後的有效遮罩、負責的網路與有效人數
        for protos, qvec in src:  # 逐一來源
            z = torch.zeros(len(ids), C, device=dev)  # 該來源的累加分數
            for m in range(protos.shape[0]):  # 逐一成員
                S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)  # 該成員的查詢向量
                for j in range(5):  # 五個交叉擬合網路
                    s = nj == j  # 由網路 j 負責的查詢
                    if s.any():  # 有才計算
                        S[s] = qv[s] @ protos[m, j].T  # 與原型的相似度
                mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)  # 在有效候選人上標準化後累加
            out.append(z / protos.shape[0])  # 成員平均
        return out, valid  # 各來源分數與有效遮罩
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy").astype(np.float32); zh = [x.cpu().numpy() for x in zs(hq)[0]]; truth = q["held_truth"]; w = ip.fit_weights([hb, *zh], truth); rnk = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1)  # noqa: E731  在留出本人局上擬合區塊與各來源的權重
    acc = {}  # 留出局的命中
    for name, F in [(f"source{k} ({MULTI[tag][k][0]}) alone", zh[k]) for k in range(len(zh))] + [("fused", w[0] * hb + sum(w[k + 1] * zh[k] for k in range(len(zh))))]:  # 各來源單獨與融合後
        r_ = rnk(F); acc[name] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean())}  # 計算 top-1／5／20 命中率
    log(f"tag {tag}: weights block {w[0]:.2f}, " + ", ".join(f"{MULTI[tag][k][0]} {w[k + 1]:.2f}" for k in range(len(zh))) + " | held-out own games top-1 / top-5 / top-20: " + "; ".join(f"{k} {x['top1']:.3f}/{x['top5']:.3f}/{x['top20']:.3f}" for k, x in acc.items()) + " (M82 5e fused: 0.268/0.437/0.588)")  # 記錄權重與命中
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros(len(idx), np.int64); m = np.zeros(len(idx))  # 讀入四個分片的區塊分數並排序
    for i in range(0, len(idx), 2048):  # 每批 2,048 個查詢
        z, valid = zs(idx[i:i + 2048]); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)  # 區塊分數乘上權重
        for k in range(len(z)):  # 逐一加上各來源
            F = F + w[k + 1] * z[k]  # 加權相加
        top = F.masked_fill(~valid, -float("inf")).topk(2, dim=1); b[i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()  # 最佳候選人與 margin
    del blk; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}  # 釋放區塊分數；查詢表各段與 M82 的基本欄位
    val = {**base, "f_opp_lab": plab[b[o]], "f_opp_m": m[o], "f_opp_best": players[b[o]], "f_imp_lab": plab[b[t_]], "f_imp_m": m[t_]}  # 驗證端輸出
    np.savez_compressed(RV / f"m84_{tag}_val.npz", **val); np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[to]], f_opp_m=m[to], f_opp_best=players[b[to]], f_tgt_lab=plab[b[tt]], f_tgt_m=m[tt], f_tgt_best=players[b[tt]])  # 寫出識別器輸出（驗證與測試）
    rep = {"weights": w.tolist(), "heldout": acc, "same_top_candidate_as_5e": {"validation": float((val["f_opp_best"] == d82["f5e_opp_best"]).mean()), "test": float((players[b[to]] == t82["f5e_opp_best"]).mean())}}  # 摘要：權重、命中、與 5e 的一致率
    ZN = {**base, "opp_lab": val["f_opp_lab"], "opp_m": val["f_opp_m"], "imp_m": val["f_imp_m"], "imp_lab": val["f_imp_lab"]}; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; ok = []  # 新識別器與 5e 的錨點輸入
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        A, B_ = anchors(ZN, seeds), anchors(ZO, seeds); y, n, pid = (np.concatenate([x[k] for x in A]) for k in (2, 3, 4)); res = {}  # 兩個識別器的各折錨點分數
        for label, a, c in (("new_vs_base", [x[0] for x in A], [x[1] for x in A]), ("5e_vs_base", [x[0] for x in B_], [x[1] for x in B_]), ("new_vs_5e", [x[0] for x in A], [x[0] for x in B_])):  # 三組比較
            dd = paired_diff(np.concatenate(a), np.concatenate(c), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(u, x[2], x[3], sizes=SIZES) - balanced_mean_over_n(v_, x[2], x[3], sizes=SIZES) for u, v_, x in zip(a, c, A)]  # 配對差與各折差
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}  # M37 門檻
            log(f"  seeds {seeds} {label:12s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x:+.4f}" for m_, x in dd["by_n"].items()) + f"  -> {'PASS' if good else 'no'}")  # 記錄差、信賴區間、各折與各題目局數的差
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["new_vs_5e"]["pass"])  # 主檢定看新識別器對 5e
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False  # 快取檢查的結果
    rep["cache_ok"] = cache_ok; rep["adopt"] = bool(all(ok) and cache_ok); log(f"PRE-REGISTERED: identifier {tag} passes against the 5e identifier: {'YES' if rep['adopt'] else 'NO'}")  # 事先登錄的判定
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float)); finalize()  # 寫出結果並更新上傳指示


def stage_wgrid():  # 階 W：融合權重以錨點分數選定
    """Rung W: the matcher's fusion weight chosen by the anchors' score instead of the held-out identification likelihood. One pass fuses
    every weight of WGRID (M82's matcher, cached block); for validation fold v the weight is the one with the best class-balanced score
    (seeds 0-2) on the other three folds; primary = that cross-fitted choice vs M82's weight, M37 on seeds 0-2 and 6-8."""
    import torch  # 需要 GPU 的階段才匯入
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)  # 查詢表
    V = np.load(CACHE / "var_ref.npz"); protos = torch.as_tensor(V["protos"][0], device=dev); qvec = V["qvec"][0]; w5e = json.loads((M82 / "report.json").read_text())["weights"]["5e"]["matcher"]; grid = [w5e if g is None else g for g in WGRID]  # M82 的配對器；格點的 None 換成現行權重
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros((len(grid), len(idx)), np.int64); m = np.zeros((len(grid), len(idx)))  # 區塊分數；每個權重各一組結果
    for i in range(0, len(idx), 2048):  # 每批 2,048 個查詢
        ids = idx[i:i + 2048]; ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False  # 有效候選人遮罩
        S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[ids].astype(np.float32), device=dev); nj = torch.as_tensor(net[ids], device=dev).long()  # 查詢向量與負責的網路
        for j in range(5):  # 五個交叉擬合網路
            s = nj == j  # 由網路 j 負責的查詢
            if s.any():  # 有才計算
                S[s] = qv[s] @ protos[j].T  # 與原型的相似度
        cnt = valid.sum(1, keepdim=True); mu = (S * valid).sum(1, keepdim=True) / cnt; z = (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12); B_ = torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)  # 標準化的配對器分數與區塊分數
        for gi, beta in enumerate(grid):  # 逐一候選權重
            top = (B_ + beta * z).masked_fill(~valid, -float("inf")).topk(2, dim=1); b[gi, i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[gi, i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()  # 該權重下的最佳候選人與 margin
    del blk; log(f"fused {len(grid)} weights: " + " ".join(f"{x:.2f}" for x in grid))  # 釋放區塊分數並記錄
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}  # 查詢表各段與 M82 的基本欄位
    Z = [{**base, "opp_lab": plab[b[gi][o]], "opp_m": m[gi][o], "imp_m": m[gi][t_], "imp_lab": plab[b[gi][t_]]} for gi in range(len(grid))]; ref = grid.index(w5e); rep = {"grid": grid}; ok = []; pick = None  # 每個權重的錨點輸入；ref＝現行權重的位置
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        A = [anchors(Z[gi], seeds) for gi in range(len(grid))]; fs = np.array([[balanced_mean_over_n(A[gi][f][0], A[gi][f][2], A[gi][f][3], sizes=SIZES) for f in range(len(FOLDS))] for gi in range(len(grid))])  # [權重, 折] 的類別平衡分數
        if pick is None:  # 只在種子 0-2 上選擇一次
            pick = [int(np.argmax(fs[:, [u for u in range(len(FOLDS)) if u != f]].mean(1))) for f in range(len(FOLDS))]; final = int(np.argmax(fs.mean(1)))  # 各折取其餘三折最佳的權重；另取四折全體最佳者
            log("  score by weight (mean over folds, seeds 0-2): " + " ".join(f"{grid[gi]:.2f}: {fs[gi].mean():.4f}" for gi in range(len(grid))) + f" | cross-fitted picks per fold {[round(grid[x], 2) for x in pick]}; on all folds {grid[final]:.2f}")  # 記錄各權重的分數與選擇
        new = [A[pick[f]][f][0] for f in range(len(FOLDS))]; old = [A[ref][f][0] for f in range(len(FOLDS))]; y, n, pid = (np.concatenate([x[k] for x in A[ref]]) for k in (2, 3, 4))  # 交叉擬合選擇與現行權重的得分
        dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[pick[f], f] - fs[ref, f]) for f in range(len(FOLDS))]; good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)  # M37 門檻
        log(f"  seeds {seeds} cross-fitted weight vs M82's weight {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")  # 記錄比較結果
        rep[f"seeds_{seeds[0]}"] = {"new_vs_5e": {**dd, "per_fold": pf, "pass": bool(good)}, "score_by_weight": fs.mean(1).tolist()}; ok.append(good)  # 收集結果
    rep.update(picks=[grid[x] for x in pick], final_weight=grid[final], adopt=bool(all(ok) and grid[final] != w5e)); log(f"PRE-REGISTERED: rung W passes: {'YES' if rep['adopt'] else 'NO'}")  # 兩組都過且選到的權重不同於現行才採用
    gi = final; np.savez_compressed(RV / "m84_W_val.npz", **base, f_opp_lab=plab[b[gi][o]], f_opp_m=m[gi][o], f_opp_best=players[b[gi][o]], f_imp_lab=plab[b[gi][t_]], f_imp_m=m[gi][t_])  # 以四折全體選出的權重寫出驗證端輸出
    np.savez_compressed(RV / "m84_W_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[gi][to]], f_opp_m=m[gi][to], f_opp_best=players[b[gi][to]], f_tgt_lab=plab[b[gi][tt]], f_tgt_m=m[gi][tt], f_tgt_best=players[b[gi][tt]])  # 測試端輸出
    (RV / "m84_W_eval.json").write_text(json.dumps(rep, indent=1, default=float)); finalize()  # 寫出結果並更新上傳指示


def stage_cgrid(tag="C"):  # 階 C：合併版配對器加上權重交叉擬合
    """Rung C: the PR matcher (all games, phase rows, same-level episodes, three seeds averaged) with the fusion weight chosen by the
    anchors' score, cross-fitted over folds on seeds 0-2. Primary: that identifier vs M82's 5e, M37 on seeds 0-2 and 6-8."""
    import torch  # 需要 GPU 的階段才匯入
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    var, members, ct_ = TAGS[tag]; dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)  # 標記設定與查詢表
    V = np.load(CACHE / f"var_{var}.npz"); protos = torch.as_tensor(V["protos"][members], device=dev); qvec = V["qvec"][members]; grid = list(CGRID)  # 成員的原型、查詢向量與權重候選
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros((len(grid), len(idx)), np.int64); m = np.zeros((len(grid), len(idx)))  # 區塊分數；每個權重各一組結果
    for i in range(0, len(idx), 2048):  # 每批 2,048 個查詢
        ids = idx[i:i + 2048]; ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); z = torch.zeros(len(ids), C, device=dev); cnt = valid.sum(1, keepdim=True)  # 有效遮罩、負責的網路與累加分數
        for mi in range(len(members)):  # 逐一成員
            S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[mi, ids].astype(np.float32), device=dev)  # 該成員的查詢向量
            for j in range(5):  # 五個交叉擬合網路
                s = nj == j  # 由網路 j 負責的查詢
                if s.any():  # 有才計算
                    S[s] = qv[s] @ protos[mi, j].T  # 與原型的相似度
            mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)  # 在有效候選人上標準化後累加
        z /= len(members); B_ = torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)  # 成員平均；區塊分數
        for gi, beta in enumerate(grid):  # 逐一候選權重
            top = (B_ + beta * z).masked_fill(~valid, -float("inf")).topk(2, dim=1); b[gi, i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[gi, i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()  # 該權重下的最佳候選人與 margin
    del blk; log(f"fused {len(grid)} weights: " + " ".join(f"{x:.2f}" for x in grid))  # 釋放區塊分數並記錄
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}  # 查詢表各段與 M82 的基本欄位
    Z = [{**base, "opp_lab": plab[b[gi][o]], "opp_m": m[gi][o], "imp_m": m[gi][t_], "imp_lab": plab[b[gi][t_]]} for gi in range(len(grid))]; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; rep = {"grid": grid}; ok = []; pick = None  # 各權重與 5e 的錨點輸入
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        A = [anchors(Z[gi], seeds) for gi in range(len(grid))]; O = anchors(ZO, seeds); fs = np.array([[balanced_mean_over_n(A[gi][f][0], A[gi][f][2], A[gi][f][3], sizes=SIZES) for f in range(len(FOLDS))] for gi in range(len(grid))])  # [權重, 折] 的類別平衡分數
        fo = np.array([balanced_mean_over_n(O[f][0], O[f][2], O[f][3], sizes=SIZES) for f in range(len(FOLDS))])  # 5e 各折的分數
        if pick is None:  # 只在種子 0-2 上選擇一次
            pick = [int(np.argmax(fs[:, [u for u in range(len(FOLDS)) if u != f]].mean(1))) for f in range(len(FOLDS))]; final = int(np.argmax(fs.mean(1)))  # 各折取其餘三折最佳的權重；另取四折全體最佳者
            log("  score by weight (mean over folds, seeds 0-2): " + " ".join(f"{grid[gi]:.2f}: {fs[gi].mean():.4f}" for gi in range(len(grid))) + f" | 5e {fo.mean():.4f} | cross-fitted picks per fold {[grid[x] for x in pick]}; on all folds {grid[final]:.2f}")  # 記錄各權重的分數與選擇
        new = [A[pick[f]][f][0] for f in range(len(FOLDS))]; old = [O[f][0] for f in range(len(FOLDS))]; y, n, pid = (np.concatenate([x[k] for x in O]) for k in (2, 3, 4))  # 交叉擬合選擇與 5e 的得分
        dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[pick[f], f] - fo[f]) for f in range(len(FOLDS))]; good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)  # M37 門檻
        log(f"  seeds {seeds} C (cross-fitted weight) vs 5e {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")  # 記錄比較結果
        rep[f"seeds_{seeds[0]}"] = {"new_vs_5e": {**dd, "per_fold": pf, "pass": bool(good)}, "score_by_weight": fs.mean(1).tolist()}; ok.append(good)  # 收集結果
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False  # 快取檢查的結果
    rep.update(picks=[grid[x] for x in pick], final_weight=grid[final], cache_ok=cache_ok, adopt=bool(all(ok) and cache_ok)); log(f"PRE-REGISTERED: rung {tag} passes: {'YES' if rep['adopt'] else 'NO'}")  # 兩組都過且快取檢查通過才採用
    gi = final; np.savez_compressed(RV / f"m84_{tag}_val.npz", **base, f_opp_lab=plab[b[gi][o]], f_opp_m=m[gi][o], f_opp_best=players[b[gi][o]], f_imp_lab=plab[b[gi][t_]], f_imp_m=m[gi][t_])  # 以四折全體選出的權重寫出驗證端輸出
    np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[gi][to]], f_opp_m=m[gi][to], f_opp_best=players[b[gi][to]], f_tgt_lab=plab[b[gi][tt]], f_tgt_m=m[gi][tt], f_tgt_best=players[b[gi][tt]])  # 測試端輸出
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float))  # 寫出評估結果
    if ct_ is not None:  # 有候選檔編號（C 有，Cp 預覽沒有）
        finalize()  # 更新上傳指示與候選檔


def finalize():  # 產生 upload_next/ 的上傳指示與候選檔
    """upload_next/: the candidate of the highest-priority variant that passed, built with the unchanged composition, and a README for the user."""
    UP.mkdir(exist_ok=True); st = {t: json.loads((RV / f"m84_{t}_eval.json").read_text()) for t in PRIORITY if (RV / f"m84_{t}_eval.json").exists()}; passing = [t for t in PRIORITY if st.get(t, {}).get("adopt")]  # 讀各變體的評估結果，依順位列出通過者
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]  # noqa: E731  檔案 md5 的前 12 碼
    for t in passing:  # 逐一通過的變體
        a, b = TAGS[t][2]; f = ROOT / f"submission_rank_ct{a}.csv"  # 候選檔與保守版的編號
        if not f.exists():  # 還沒建檔才建
            r = subprocess.run([sys.executable, "-I", "scripts/m_series/m78_build_candidate.py", "--source", "file", "--val", str(RV / f"m84_{t}_val.npz"), "--test", str(RV / f"m84_{t}_test.npz"), "--prefix", "f_", "--ref", "ct44", "--out", a, b], capture_output=True, text=True, cwd=ROOT)  # 以 m78_build_candidate.py 建檔（組合方式不變，--ref ct44 只是比較基準）
            (UP / f"build_{t}.log").write_text(r.stdout + "\n" + r.stderr[-3000:]); print(r.stdout[-2500:], flush=True)  # 保存建檔紀錄
    desc = DESC; L = [f"# 上傳指示（自動產生，{time.strftime('%Y-%m-%d %H:%M')}）", "", "**2026-11-14 更新：建議的最終檔改為 ct104b**（`submission_rank_ct104b.csv`，md5 7e00dc32133c，Public 0.8126304，2026-11-14 上傳，歷來最高，已是線上最後一筆；說明見 `ct104_decode.md`）。備援依序為 ct75u（0.8101304）→ ct75s → ct44。以下是 11/13 的說明，保留備查。", "", "**（11/13）建議的最終檔：ct75u**（`submission_rank_ct75u.csv`，md5 9974e151b93e，Public 0.8101304，2026-11-13 上傳，歷來最高）。**注意：11/13 最後兩次上傳是拆解檔（ct75s_out 0.8092107、ct75s_out_up 0.8076304），線上最後一筆目前不是 ct75u——請再傳一次 ct75u，讀數應為 0.8101304。** ct75u＝ct75s＋五個配方全票通過的 16 題改動。依事先寫下的規則（讀數不低於 ct75s 即建議採用）。備援：ct75s（md5 926f0e7c2153，Public 0.8092107）→ ct44（md5 367ab951442f，Public 0.8067107）。其他讀數：ct73 0.8060501、ct75n 0.8044698。之後若再傳別的檔，最後一筆必須傳回選定的最終檔。", ""]  # README 的開頭：目前建議的最終檔與備援
    if (UP / "ct75s_decode.md").exists():  # M102：共同改動的檔案與其解碼表（由 m102_split.py 寫出，重寫這份 README 時保留）
        L += ["## 目前不用傳任何檔", "", "- `submission_rank_ct75s.csv` 已上傳，Public 0.8092107：共同的 8 題淨多對 1 題、多 1 題差一級以內（對 ct44 +1.0 分）；由此推得 ct73 獨有的 4 題少對 2 題、ct75n 獨有的 8 題少對 3 題。29 次上傳的讀數聯立仍然相容。對照表與事先寫下的建議見 `ct75s_decode.md`。", "- 第六版打包（加入與外部玩家一起訓練的認人網路、共五個成員）通過，但選到的配方與第五版相同，產生的檔與 ct75／ct75n 逐位元組相同，沒有新的候選檔。", ""]  # ct75s 的說明段落
    if (UP / "ct104_decode.md").exists():  # M129：2026-11-14 的候選檔（統計量＋「長得像」打包）；重寫這份 README 時保留這段說明
        L += ["## 2026-11-14 的新候選檔：ct104、ct104a、ct104b", "", "- 三個檔都通過事先登錄的交叉驗證檢定（對 ct75u 規則 +0.003 到 +0.005），Public 帳本回測沒有否決；說明、md5 與否決線見 `ct104_decode.md`。要不要上傳、最終用哪一個，由使用者決定。", ""]  # ct104 系列的說明段落
    if (UP / "ct75u_decode.md").exists():  # M104：全票通過的候選檔
        L += ["- `submission_rank_ct75u.csv` 已上傳，Public 0.8101304：對 ct75s 多 1 題差一級以內（+0.37 分），9 題不同。30 次上傳的讀數聯立仍然相容。交叉驗證對 ct75s 的規則 +0.0046／+0.0036（八折全正）。對照表在 `ct75u_decode.md`。", ""]  # ct75u 的說明段落
    plain = lambda s_: s_.replace("id=SP2y", "全部對局的逐手認人模型（5 個成員：3 seed＋外部玩家版＋對照版）＋分階段比對").replace("id=SP2x", "全部對局的逐手認人模型（3 seed＋外部玩家版）＋分階段比對").replace("id=SP2e", "全部對局的逐手認人模型（3 seed）＋分階段比對").replace("id=S2e", "全部對局的逐手認人模型（3 seed）").replace("id=SP2", "全部對局的逐手認人模型＋分階段比對").replace("id=S2", "全部對局的逐手認人模型").replace("id=SP", "逐手認人模型＋分階段比對").replace("id=S", "逐手認人模型").replace("id=P3", "比對改用全部對局＋分階段描述").replace("enc", "加入整局編碼器的證據").replace("cc", "證據表依等級分別估").replace("temp", "20 局的題目證據降溫").replace("pi0", "修正混合權重").replace("+", "；")  # noqa: E731  把組合標籤換成中文說明
    bundles = []  # 依順位排列的（json、說明）：第一個採用且有檔案者就是要上傳的檔
    for jn, note in (("m86g_bundle.json", "第六版（再加一組與外部公開棋譜玩家一起訓練的認人網路）"), ("m86f_bundle.json", "第五版（全部對局的逐手認人模型，3 個 seed 平均）"), ("m86e_bundle.json", "第四版（逐手認人模型用全部對局訓練）"), ("m86d_bundle.json", "第三版（含逐手認人模型，全部把握帶）"), ("m86c_bundle.json", "第二版（全部把握帶）"), ("m86b_bundle.json", "第二版（只用高把握帶）"), ("m86_bundle.json", "第一版")):  # 各版打包檢定的結果檔，由新到舊
        if (RV / jn).exists():  # 有結果檔才處理
            bj_ = json.loads((RV / jn).read_text()); f_ = ROOT / f"submission_rank_{bj_.get('file', 'ct64')}.csv"  # 讀結果與對應的候選檔
            if jn == "m86g_bundle.json" and not f_.exists():  # 第六版選到第五版的配方：其檔案與 ct75 逐位元組相同，已刪除
                continue  # 略過
            bundles.append((bj_, note, f_, bool(bj_.get("adopt") and f_.exists())))  # 記下是否採用且檔案存在
    has_bundle = any(x[3] for x in bundles); first = True  # 是否有可用的打包檔；first 標記第一個
    for bj_, note, f_, ok_ in bundles:  # 逐一把打包結果寫進 README
        d = [bj_[k]["bundle_vs_off"] for k in ("seeds_0", "seeds_6")]; up_ = {"ct64": 0.8038092, "ct71": 0.8053895, "ct72": 0.8038092}.get(bj_.get("file")); old_ = bj_.get("file") in ("ct64", "ct70", "ct71", "ct72")  # 已上傳過的檔與其 Public 讀數
        if ok_:  # 已採用且有檔案
            shutil.copy(f_, UP / f_.name)  # 複製到 upload_next/
            if first and bj_.get("file") not in ("ct64", "ct70", "ct71", "ct72"):  # 2026-11-12 起的固定組合方式：不改回 ct44 的改動
                import csv  # 讀寫提交檔
                rd = lambda n_: {r_["question_id"]: r_["pred_rank"] for r_ in csv.DictReader(open(ROOT / f"submission_rank_{n_}.csv"))}  # noqa: E731  讀提交檔成「題號→等級」
                c28_, c44_, cn_ = rd("ct28"), rd("ct44"), rd(bj_["file"]); fn_ = ROOT / f"submission_rank_{bj_['file']}n.csv"  # 讀 ct28、ct44 與新檔；「不改回」版的檔名結尾加 n
                with open(fn_, "w", newline="") as fh:  # 寫出不改回版
                    w_ = csv.writer(fh, lineterminator="\n"); w_.writerow(["question_id", "pred_rank"]); w_.writerows([k_, c44_[k_] if c44_[k_] != c28_[k_] else cn_[k_]] for k_ in c44_)  # ct44 已改的題維持 ct44，其餘用新檔
                shutil.copy(fn_, UP / fn_.name); nd_ = sum(1 for k_ in c44_ if (c44_[k_] if c44_[k_] != c28_[k_] else cn_[k_]) != c44_[k_])  # 複製並計算對 ct44 改了幾題
                same_ = fn_.read_bytes() == (ROOT / "submission_rank_ct73.csv").read_bytes()  # 是否與現任最終檔 ct73 完全相同
                upn_ = {"ct75n": 0.8044698}.get(fn_.stem.replace("submission_rank_", ""))  # 已上傳過的不改回版
                L += [f"## {'不必上傳（內容與現任最終檔 ct73 完全相同）' if same_ else '已上傳過（Public ' + format(upn_, '.7f') + '，高於 0.8027 的否決線）' if upn_ is not None else '要傳的檔'}：`{fn_.name}`（md5 {md5(fn_)}）", "", f"- 內容：`{f_.name}` 的「不改回」版：ct44 已改過的題目維持 ct44 的答案，其餘用新檔的答案（11/12 起的固定組合方式）。對 ct44 改 {nd_} 題。",  # 不改回版的標題與內容說明
                      (f"- 完整版 `{f_.name}`（md5 {md5(f_)}）已上傳（Public {up_:.7f}）；CV 上它比「不改回」版高約 0.002 到 0.004，Public 上因為改回了 ct44 的改動而偏低。最終檔選定後，最後一筆必須再傳那個檔。" if up_ is not None else f"- 完整版 `{f_.name}`（md5 {md5(f_)}）是 CV 最高的版本，兩者期望差約 0.002；有多的上傳次數可以也傳，但最後一筆請留「不改回」版或現任最終檔。"), ""]  # 完整版的說明
            if first:  # 第一個採用的打包檔
                L += [f"## {'完整版（已上傳，Public ' + format(up_, '.7f') + '）' if up_ is not None and not old_ else '目前不用傳。最新通過的打包檔（已上傳過，Public ' + format(up_, '.7f') + '）' if up_ is not None else '完整版' if bj_.get('file') not in ('ct64', 'ct70', 'ct71', 'ct72') else '要傳的檔'}：`{f_.name}`（md5 {md5(f_)}）", "", f"- 內容：ct44 加上一包小改良，{note}（{plain(bj_['picks']['all_folds'])}；設計文件 M86–M90）。", f"- 選與驗分開的 CV 估計：{d[0]['diff']:+.4f} [{d[0]['ci95'][0]:+.4f}, {d[0]['ci95'][1]:+.4f}]／{d[1]['diff']:+.4f} [{d[1]['ci95'][0]:+.4f}, {d[1]['ci95'][1]:+.4f}]；對 ct44 改 {bj_['test']['changed_vs_ct44']} 題（Public {bj_['test']['public']}／Private {bj_['test']['private']}）。",  # 標題、內容與選驗分開的估計
                      "- 已上傳過，不必再傳。讀數高於 0.8027 的否決線，但低於 ct73（0.8060501）與 ct44（0.8067107）。" if up_ is not None and not old_ else "- 已上傳過，不必再傳；它的「不改回」版就是現任最終檔 ct73。" if up_ is not None else "- 傳一次、記下讀數即可；回來後判定（讀數 ≥ 0.8027 才可能取代現任最終檔）。", ""]; first = False  # 上傳狀態的說明；之後的打包檔只列一行
            else:  # 其他已採用的打包檔
                L += [f"- 其他已通過的打包檔（不必傳）：`{f_.name}`（md5 {md5(f_)}，{note}，CV {d[0]['diff']:+.4f}／{d[1]['diff']:+.4f}{'，已上傳過，Public ' + format(up_, '.7f') if up_ is not None else ''}）"]  # 只列一行
        else:  # 沒採用或沒有檔案
            L += [f"- 打包檢定{note}：{'沒過' if not bj_.get('passed_cv') else '建檔檢查沒過'}（{d[0]['diff']:+.4f}／{d[1]['diff']:+.4f}）。"]  # 記下沒過的原因與差
    L += [""]  # 段落空行
    built = [t for t in passing if (ROOT / f"submission_rank_ct{TAGS[t][2][0]}.csv").exists()]  # 已建好候選檔的單項變體
    if built and has_bundle:  # 已有打包檔時，單項變體只列出不上傳
        for u in built:  # 逐一列出
            a2 = TAGS[u][2][0]; f2 = ROOT / f"submission_rank_ct{a2}.csv"; L += [f"- 單項通過 CV 的比對變體（不必傳，已包含在上面的打包檔裡）：`submission_rank_ct{a2}.csv`（md5 {md5(f2)}，{desc[u]}）"]  # 單項變體不必上傳
        L += [""]  # 段落空行
    elif built:  # 沒有打包檔但有單項候選
        t = built[0]; a = TAGS[t][2][0]; f = ROOT / f"submission_rank_ct{a}.csv"; shutil.copy(f, UP / f.name); d = [st[t][k]["new_vs_5e"]["diff"] for k in ("seeds_0", "seeds_6")]  # 順位最高的單項候選檔
        L += [f"## 要傳的檔：`submission_rank_ct{a}.csv`（md5 {md5(f)}）", "", f"- 內容：ct44 的做法，比對成分改成「{desc[t]}」。CV 對 ct44 的認人方法 {d[0]:+.4f}／{d[1]:+.4f}，兩組 seed 都通過。", "- 傳一次即可，把 Public 讀數記下來；不必當場判斷。",  # 要傳的檔與說明
              "- 規則（回來後由 Claude 解碼判定）：讀數 ≥ 0.8027（ct44 − 0.004）才取代 ct44。", "- 最後一筆只在 11/24 截止時才算數；有空的話最後再傳一次 ct44，沒空就不用。", ""]  # 判定規則與最後一筆的提醒
        for u in built[1:]:  # 其餘通過的單項候選
            a2 = TAGS[u][2][0]; f2 = ROOT / f"submission_rank_ct{a2}.csv"; shutil.copy(f2, UP / f2.name); L += [f"- 另一個也通過 CV 的檔（次要，有多的上傳次數再傳）：`submission_rank_ct{a2}.csv`（md5 {md5(f2)}，{desc[u]}）"]  # 列為次要候選
    elif not has_bundle:  # 完全沒有通過者
        L += ["## 目前不用傳", "", "- 還沒有任何新候選通過 CV 規則。ct44 維持最終檔。", ""]  # ct44 維持最終檔
    mj = RV / "m85_mirror_eval.json"; mir = json.loads(mj.read_text()) if mj.exists() else None; f56 = ROOT / "submission_rank_ct56.csv"  # M85 鏡像通道的結果與候選檔
    if mir is not None and mir.get("adopt") and f56.exists():  # 鏡像通道通過且已建檔
        shutil.copy(f56, UP / f56.name); d = [mir[k]["both_vs_forward"]["diff"] for k in ("seeds_0", "seeds_6")]  # 複製並取兩組種子的差
        L += ["", f"## 也要傳的檔：`submission_rank_ct56.csv`（md5 {md5(f56)}）", "", f"- 內容：ct44 再加上「鏡像通道」的證據（測試玩家以對手身分出現在訓練棋譜裡）。CV 在 ct44 的做法之上 {d[0]:+.4f}／{d[1]:+.4f}，兩組 seed 都通過；對 ct44 改 {mir['test']['changed_vs_ct44']} 題（Public {mir['test']['public']}／Private {mir['test']['private']}）。",  # 鏡像通道的說明
              "- 一樣傳一次、記下讀數；≥ 0.8027 才可能取代 ct44，回來後判定。", ""]  # 判定規則
    elif mir is not None:  # 跑完但沒過
        L += ["", f"- 鏡像通道（M85）：已跑完，{'CV 沒過' if not mir.get('passed_cv') else '測試端安全檢查沒過'}，沒有建檔。"]  # 記下沒過的原因
    else:  # 還沒有結果
        L += ["", "- 鏡像通道（M85）：尚未完成。"]  # 註明尚未完成
    L += ["", "## 各變體的狀態", "", "| 變體 | 內容 | 狀態 | 對 ct44 認人方法的錨點差（seed 0–2／6–8） |", "|---|---|---|---|"]  # 各變體狀態表的表頭
    for t in ("A", "R", "B", "B3", "P", "P3", "W", "C", "S", "SP", "S2", "SP2", "S2e", "SP2e"):  # 依序列出各變體
        if t in st:  # 已有評估結果
            d = [st[t][k]["new_vs_5e"] for k in ("seeds_0", "seeds_6")]; L.append(f"| {t} | {desc[t]} | {'通過' if st[t]['adopt'] else '沒過'} | {d[0]['diff']:+.4f} [{d[0]['ci95'][0]:+.4f}, {d[0]['ci95'][1]:+.4f}]／{d[1]['diff']:+.4f} [{d[1]['ci95'][0]:+.4f}, {d[1]['ci95'][1]:+.4f}] |")  # 是否通過與兩組種子的錨點差
        else:  # 尚未評估
            L.append(f"| {t} | {desc[t]} | 尚未完成 | — |")  # 標為尚未完成
    L += ["", "採用順序事先固定為 SP2e > S2e > SP2 > S2 > SP > S > C > P3 > P > B3 > B > R > A > W（設計文件 M84、M88、M90）。下載：`sftp -P 2222 ...:BK_AICUP_2026-1/upload_next/*`（一次 OTP）。"]  # 採用順序與下載方式
    (UP / "README.md").write_text("\n".join(L) + "\n"); shutil.copy(ROOT / "submission_rank_ct44.csv", UP / "submission_rank_ct44.csv"); print("upload_next/README.md written; passing: " + (", ".join(passing) or "none"), flush=True)  # 寫出 README，並附上 ct44 當備援


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("embed", "eval", "finalize", "wgrid", "cgrid")); ap.add_argument("--var", default="A"); ap.add_argument("--tag", default="A"); a_ = ap.parse_args()  # 命令列：階段、embed 的變體（--var）、評估的標記（--tag）
    if a_.stage == "embed":  # 訓練配對器變體
        stage_embed(a_.var)  # 訓練 --var 指定的變體（ref 只轉存 M82 的嵌入）
    elif a_.stage == "eval":  # 評估識別器
        stage_eval_multi(a_.tag) if a_.tag in MULTI else stage_eval(a_.tag)  # 多來源的標記走 stage_eval_multi
    elif a_.stage == "wgrid":  # 階 W
        stage_wgrid()  # 權重格點
    elif a_.stage == "cgrid":  # 階 C
        stage_cgrid(a_.tag if a_.tag in ("C", "Cp") else "C")  # 標記為 C 或 Cp，其他一律以 C 執行
    else:  # 其餘即 finalize 階段
        finalize()  # 重寫上傳指示（通過者若還沒有候選檔會先建檔）
