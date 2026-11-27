"""Question-level identification of a rank-prediction question's player among the training players (M24, 2026-11-06).
Five habit components of competition 1 (aicup_pid: opening repertoire, raw style statistics, local-shape tokens, board
orientation, HD class), each a per-player model fitted on the player's games in the view where the target's stones are "B"
(a game whose target mark is W is colour-swapped; separate pools per view, summed per question), fused with one weight per
component (per-question standardisation over the candidates, multinomial likelihood, Newton's method). Three fits:
  A  in-pool gate: the 4,000 inner held-out questions' games (prod_heldout.P.npz) leave the pool, their 788 players stay with
     their other games -> top-1 by component / fused, margin, correct; weights cross-fitted on two halves of the players;
     posterior over labels (softmax over the candidates summed by label)
  B  out-of-pool gate: the 788 players leave entirely -> the margin distribution of a question whose player is NOT in the
     pool (false-accept rates), and whether confident matches land on same-label players (duplicate ids in training?)
  C  test: all games -> the 800 test questions' fused scores, margins, label posteriors; mixture estimate of the share of
     test questions whose player is in the pool, per n (A margins = in-pool reference, B margins = out-of-pool reference)
Outputs: --out JSON, <out>.gate.npz and <out>.test.npz (read by ident_ct8.py). Portable: numpy + pandas only, CPU, fork or
spawn multiprocessing; data under --root (data/training/train_*.csv, data/tests/rank_prediction_test_{public,private}.csv);
the gate file and ct7 submission are looked up under --root (results/r_variants, project root) or next to this script (inputs/).
  python -I scripts/m_series/ident_pool.py --out results/r_variants/ident_pool.json [--limit-players 1200] [--workers 30] [--root DIR]
"""
import argparse  # 解析命令列參數
import json  # 讀取 gate 題目的對局鍵、寫出報告
import multiprocessing as mp  # 多行程平行解析棋譜
import os  # 查詢 CPU 數與是否支援 fork
import sys  # 讀取 --root 與調整模組搜尋路徑
import time  # 記錄經過時間
from collections import defaultdict  # 本檔未使用，維持原有的匯入
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取訓練與測試 CSV

HERE = Path(__file__).resolve().parent  # 本程式所在資料夾（找 inputs/ 用）
ROOT = Path(next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--root"), Path.cwd())).resolve()  # 在 argparse 之前先取 --root，才能設定匯入路徑
for _p in (HERE, ROOT / "src", Path.home() / "BK_AICUP_2026-2/src"):  # 本資料夾、本專案 src、競賽一儲存庫的 src（aicup_pid）
    if str(_p) not in sys.path:  # 避免重複加入
        sys.path.insert(0, str(_p))  # 放到搜尋路徑最前面
from aicup.metric import LEVELS  # noqa: E402  匯入前須先設定 sys.path；10 個等級名稱
from aicup.sgf import handicap, parse_moves  # noqa: E402  讀讓子數與著手序列
from aicup_pid.hd import hd_class  # noqa: E402  競賽一的 HD 類別成分
from aicup_pid.local import tokens as local_tokens  # noqa: E402  競賽一的局部棋形 token
from aicup_pid.opening import tokens as opening_tokens  # noqa: E402  競賽一的開局 token
from aicup_pid.orient import orient_tokens  # noqa: E402  競賽一的盤面方位 token
from aicup_pid.style import FEATURE_NAMES, game_features  # noqa: E402  競賽一的風格統計特徵

COMPS = ("opening", "style", "local", "orient", "hd")  # 五個習慣成分的名稱
RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引 0-9
NGROUP = {5: range(5, 8), 10: range(8, 13), 15: range(13, 18), 20: range(18, 21)}  # gate 題依局數歸到最接近的測試題目局數（5、10、15、20）
DEPTH, ALPHA, BETA, ALPHA_FULL, N_POINTS = 12, 2.0, 5.0, 1.0, 362  # 開局深度、退讓模型的平滑常數、點數（361 點＋虛手）
_DCLS = {(-1, -1): 0, (4, 4): 1}  # 位移類別編號：無前一手、距離遠
for _hi in range(4):  # 其餘的位移類別 (大, 小)，各分量截在 3
    for _lo in range(_hi + 1):  # 小分量不超過大分量
        _DCLS[(_hi, _lo)] = len(_DCLS)  # 依序編號，共 12 類
V_LOCAL, V_ORIENT, V_HD = 12 * 12 * 5 * 3, 8 + 361, 3  # 詞彙大小：局部棋形、方位（8 種對稱＋首著座標）、HD 類別
D_STYLE = len(FEATURE_NAMES)  # 風格特徵的維度
MOD = (1 << 61) - 1  # 雜湊用的梅森質數模數


def find_input(root, rel, name):  # 依序在幾個位置尋找輸入檔
    for p in (Path(root) / rel / name, Path(root) / name, HERE / "inputs" / name, HERE / name):  # 依序找 --root/rel、--root、本程式旁的 inputs/ 與本程式資料夾
        if p.exists():  # 找到就用
            return p  # 回傳第一個存在的路徑
    raise FileNotFoundError(f"{name}: not under {root}/{rel}, {root}, or {HERE}/inputs")  # 都找不到時中止


def swap(moves):  # 交換全部著手的顏色
    return [("W" if c == "B" else "B", x, y) for c, x, y in moves]  # B 與 W 互換，座標不變


def target_mark(sgf, color):  # 目標方在棋譜中的記號
    """The record mark of the target's stones: the actual colour, except that HD >= 2 records mark the first mover (actual White) as B."""
    return color if (handicap(sgf) or 0) < 2 else ("W" if color == "B" else "B")  # 讓 2 子以上的棋譜記號與實際顏色相反


def _h(vals, seed):  # 整數序列的多項式雜湊
    h = seed  # 以種子區分不同用途的雜湊
    for v in vals:  # 逐一併入每個值
        h = (h * 1000003 + v + 7) % MOD  # 滾動雜湊
    return h  # 回傳雜湊值


def _pt(c):  # 座標轉成單一點編號
    return c[0] * 20 + c[1]  # 以 20 為底，虛手 (19, 19) 也有編號


def opening_keys(toks):  # 開局 token 轉成雜湊鍵
    """[t, 4] int64 hashes per Black opening move: (full ctx, mv), (Black ctx, mv), full ctx, Black ctx."""
    out = np.empty((len(toks), 4), dtype=np.int64)  # 每個 token 四個鍵
    for i, (fctx, bctx, mv) in enumerate(toks):  # 逐一處理開局 token
        hf = _h([_pt(c) for c in fctx], 1); hb = _h([_pt(c) for c in bctx], 2); m = _pt(mv)  # 完整脈絡、黑方脈絡與著手的編碼
        out[i] = (_h([hf, m], 3), _h([hb, m], 4), hf, hb)  # 計數用的分子鍵與分母鍵
    return out  # 回傳 [t, 4] 陣列


def local_ids(toks):  # 局部棋形 token 轉成整數 id
    return np.fromiter((((_DCLS[a] * 12 + _DCLS[b]) * 5 + (ln - 1)) * 3 + ph for a, b, ln, ph in toks), dtype=np.int16, count=len(toks))  # 兩個位移類別、線位、階段合成一個 id


def orient_ids(toks):  # 方位 token 轉成整數 id
    out = []  # 收集 id
    for t in toks:  # 逐一處理 token
        if t[0] == "sym":  # 對稱指標 token
            out.append(t[1])  # 對稱指標，id 0-7
        elif t[0] == "first":  # 首著原座標 token
            out.append(8 + t[1][0] * 19 + t[1][1])  # id 8 起，對應 361 個點
    return np.array(out, dtype=np.int16)  # 回傳 id 陣列


def game_view(args):  # 從目標方的視角萃取一局的成分輸入（多行程工作函式）
    """(mark, opening keys [t,4], style features [31], local ids, orient ids, hd class) of one game seen from the target (target = B)."""
    sgf, color = args  # 棋譜與目標方實際顏色
    m = parse_moves(sgf); mark = target_mark(sgf, color)  # 著手序列與目標方的記號
    mo = m if mark == "B" else swap(m)  # 目標記號為 W 時交換顏色，讓目標方一律是 B
    return (mark, opening_keys(opening_tokens(mo, DEPTH)), game_features(mo).astype(np.float32), local_ids(local_tokens(mo, 12)), orient_ids(orient_tokens(mo, 12, 1)), int(hd_class(sgf)))  # 目標方記號與五個成分所需的輸入


class PoolModel:  # 開局成分：一次算全部候選玩家的退讓模型，只為查詢用到的鍵建表
    """Competition 1's opening back-off model over all players, built only for the keys the queries need:
    p(mv | full) = (n(full, mv) + a_full p(mv | black)) / (n(full) + a_full); p(mv | black) = (n(black, mv) + alpha p_bg) / (n(black) + alpha)."""

    def __init__(self, C):  # 建立模型
        self.C = C  # 候選玩家數

    def fit(self, pl, keys, need):  # 統計每個鍵在各玩家的次數
        self.tabs = []  # 四種鍵各一張稀疏表
        for j in range(4):  # 逐一處理四種鍵
            k = keys[:, j]; keep = np.isin(k, need[j]); k, p = k[keep], pl[keep]  # 只保留查詢會用到的鍵
            order = np.lexsort((p, k)); ks, ps = k[order], p[order]  # 依（鍵, 玩家）排序
            new = np.ones(len(ks), bool); new[1:] = (ks[1:] != ks[:-1]) | (ps[1:] != ps[:-1])  # 標出每個（鍵, 玩家）組合的起點
            starts = np.flatnonzero(new); cnt = np.diff(np.append(starts, len(ks))).astype(np.float64); uk, up = ks[starts], ps[starts]  # 每個組合的次數與其鍵、玩家
            knew = np.ones(len(uk), bool); knew[1:] = uk[1:] != uk[:-1]; kstarts = np.flatnonzero(knew); kends = np.append(kstarts[1:], len(uk))  # 每個鍵在組合陣列中的起訖位置
            self.tabs.append(({int(kv): (int(s), int(e)) for kv, s, e in zip(uk[kstarts], kstarts, kends)}, up, cnt))  # 鍵 -> 範圍、玩家、次數
        self.bg, self.bgT = {}, {}  # 全部玩家合計的背景次數
        for j, d in ((1, self.bg), (3, self.bgT)):  # 黑方脈絡＋著手、黑方脈絡
            k = keys[:, j]; keep = np.isin(k, need[j]); u, c = np.unique(k[keep], return_counts=True); d.update(zip(u.tolist(), c.tolist()))  # 計算背景次數
        return self  # 方便串接呼叫

    def _dense(self, j, key):  # 取出一個鍵在全部玩家的次數向量
        v = np.zeros(self.C); idx, up, cnt = self.tabs[j]; se = idx.get(key)  # 預設全為 0
        if se is not None:  # 有玩家出現過此鍵時
            v[up[se[0]:se[1]]] = cnt[se[0]:se[1]]  # 填入這些玩家的次數
        return v  # 回傳 [C] 向量

    def scores(self, keys):  # 一組開局著手對每位玩家的對數似然
        out = np.zeros(self.C)  # 每位玩家的累計分數
        for fk, bk, fc, bc in keys.tolist():  # 逐一處理每個開局著手
            pb = (self.bg.get(bk, 0) + BETA / N_POINTS) / (self.bgT.get(bc, 0) + BETA)  # 背景機率，向均勻分布平滑
            p_black = (self._dense(1, bk) + ALPHA * pb) / (self._dense(3, bc) + ALPHA)  # 只看黑方脈絡的逐玩家機率
            p_full = (self._dense(0, fk) + ALPHA_FULL * p_black) / (self._dense(2, fc) + ALPHA_FULL)  # 完整脈絡的機率，退讓到黑方脈絡
            out += np.log(p_full)  # 累加對數機率
        return out  # 回傳 [C] 分數


class Multi:  # 局部棋形、方位、HD 共用的多項分布成分
    """Per-player multinomial with a fixed vocabulary, shrunk towards the pool background (aicup_pid.local's form)."""

    def __init__(self, C, V, alpha=2.0, beta=5.0):  # 設定玩家數、詞彙數與平滑常數
        self.C, self.V, self.alpha, self.beta = C, V, alpha, beta  # 保存設定

    def fit(self, pl_tok, ids):  # 由每個 token 的玩家與 id 擬合
        counts = np.bincount(pl_tok.astype(np.int64) * self.V + ids.astype(np.int64), minlength=self.C * self.V).reshape(self.C, self.V).astype(np.float64)  # 玩家 × 詞彙的次數表
        bg = counts.sum(0); p_bg = (bg + self.beta / self.V) / (bg.sum() + self.beta)  # 全部候選玩家合計的背景分布（平滑）
        self.logp = np.log((counts + self.alpha * p_bg) / (counts.sum(1, keepdims=True) + self.alpha))  # 每位玩家向背景收縮後的對數機率
        return self  # 方便串接呼叫

    def scores(self, ids_list):  # 查詢的 token 對每位玩家的對數似然
        if not ids_list:  # 沒有任何局時
            return np.zeros(self.C)  # 全部玩家同分
        return self.logp @ np.bincount(np.concatenate(ids_list).astype(np.int64), minlength=self.V)  # 次數向量乘上對數機率表


class Gauss:  # 風格成分：逐玩家的高斯模型
    """aicup_pid.style.StyleModel vectorised over players, means shrunk towards the pool mean with kappa pseudo-games."""

    def __init__(self, C, kappa=2.0, shrink=0.1):  # 設定玩家數與收縮強度
        self.C, self.kappa, self.shrink = C, kappa, shrink  # 保存設定

    def fit(self, pl, X):  # 由每局的玩家與風格特徵擬合
        X = np.asarray(X, dtype=np.float64); D = X.shape[1]  # 特徵矩陣與維度
        n = np.bincount(pl, minlength=self.C).astype(float)  # 每位玩家的局數
        S1 = np.stack([np.bincount(pl, weights=X[:, j], minlength=self.C) for j in range(D)], axis=1)  # 每位玩家的特徵總和
        mu = S1 / np.maximum(n, 1)[:, None]  # 每位玩家的平均
        cov = (X.T @ X - (mu * n[:, None]).T @ mu) / max(len(X) - int((n > 0).sum()), 1)  # 合併的玩家內共變異數
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov)); cov += np.eye(D) * 1e-9 * max(np.trace(cov) / D, 1e-12)  # 向對角收縮並加微小嶺值，確保可逆
        self.prec = np.linalg.inv(cov); mu_bg = X.mean(0) if len(X) else np.zeros(D)  # 精度矩陣與全部對局的平均
        self.mu = (S1 + self.kappa * mu_bg) / (n + self.kappa)[:, None]; self.n = n + self.kappa  # 平均向池平均收縮，局數加上虛擬局
        return self  # 方便串接呼叫

    def scores(self, Q):  # 查詢局的平均與每位玩家平均的馬氏距離分數
        if len(Q) == 0:  # 沒有任何局時
            return np.zeros(self.C)  # 全部玩家同分
        Q = np.asarray(Q, dtype=np.float64); m = len(Q)  # 查詢特徵與局數
        d = self.mu - Q.mean(0); scale = 1.0 / m + 1.0 / self.n  # 平均之差與其變異的倍數
        return -0.5 * np.einsum("cd,de,ce->c", d, self.prec, d) / scale  # 高斯對數似然（省略常數）


class ViewPool:  # 一個視角下五個成分的集合
    """All five components for one view (target = B first mover, or target = B second mover after the swap)."""

    def __init__(self, C):  # 建立視角池
        self.C = C  # 候選玩家數

    def fit(self, views, idx, pl, need):  # 以選定的對局擬合五個成分
        pl = pl[idx]  # 選定對局的玩家索引
        okeys = np.concatenate([views[i][1] for i in idx]) if len(idx) else np.zeros((0, 4), np.int64); opl = np.repeat(pl, [len(views[i][1]) for i in idx])  # 開局鍵與每個鍵的玩家
        self.open = PoolModel(self.C).fit(opl, okeys, need); del okeys, opl  # 擬合開局成分後釋放記憶體
        self.style = Gauss(self.C).fit(pl, np.stack([views[i][2] for i in idx]) if len(idx) else np.zeros((0, D_STYLE)))  # 擬合風格成分
        for name, V, col in (("local", V_LOCAL, 3), ("orient", V_ORIENT, 4)):  # 局部棋形與方位兩個 token 成分
            ids = np.concatenate([views[i][col] for i in idx]) if len(idx) else np.zeros(0, np.int16); ptok = np.repeat(pl, [len(views[i][col]) for i in idx])  # token id 與每個 token 的玩家
            setattr(self, name, Multi(self.C, V).fit(ptok, ids))  # 擬合並存成同名屬性
        self.hd = Multi(self.C, V_HD).fit(pl, np.array([views[i][5] for i in idx], dtype=np.int64))  # HD 類別成分，每局一個 token
        return self  # 方便串接呼叫

    def scores(self, vs):  # 一題中此視角的對局對每位玩家的五個成分分數
        out = {c: np.zeros(self.C) for c in COMPS}  # 每個成分一個 [C] 向量
        if not vs:  # 此視角沒有對局時
            return out  # 回傳全 0
        for v in vs:  # 開局成分逐局累加
            out["opening"] += self.open.scores(v[1])  # 加上該局開局的對數似然
        out["style"] = self.style.scores(np.stack([v[2] for v in vs])); out["local"] = self.local.scores([v[3] for v in vs])  # 風格與局部棋形以整題計算
        out["orient"] = self.orient.scores([v[4] for v in vs]); out["hd"] = self.hd.scores([np.array([v[5]], dtype=np.int16) for v in vs])  # 方位與 HD 類別
        return out  # 回傳各成分分數


def fit_scenario(views, marks, pl, include, need, C, log):  # 在一個情境（納入的對局）下擬合 B、W 兩個視角池
    t0 = time.time(); pools = {}  # 計時並收集兩個池
    for mark in ("B", "W"):  # 兩個視角分開擬合
        idx = np.flatnonzero(include & (marks == mark)); pools[mark] = ViewPool(C).fit(views, idx, pl, need)  # 只用納入且同記號的對局
        log(f"   {mark}-view pool fitted on {len(idx)} games")  # 回報擬合的局數
    log(f"   scenario fitted in {time.time() - t0:.0f} s")  # 回報耗時
    return pools  # 回傳 {記號: ViewPool}


def score_questions(pools, qviews, C):  # 每題對每位候選人的五個成分分數
    S = {c: np.zeros((len(qviews), C)) for c in COMPS}  # 每個成分一個 [題數, C] 矩陣
    for qi, vs in enumerate(qviews):  # 逐題處理
        for mark in ("B", "W"):  # 兩個視角
            sub = [v for v in vs if v[0] == mark]  # 題內屬於此視角的對局
            if sub:  # 有對局才評分
                sc = pools[mark].scores(sub)  # 用對應視角的池評分
                for c in COMPS:  # 逐一成分
                    S[c][qi] += sc[c]  # 兩個視角的分數相加
    return S  # 回傳各成分分數


def standardise(S):  # 每題對候選人做標準化
    S = np.asarray(S, dtype=np.float64); mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); sd[sd < 1e-12] = 1.0  # 每列的平均與標準差，避免除以 0
    return (S - mu) / sd  # 回傳 z 分數


def fit_weights(Zs, truth, l2=1e-3, iters=60):  # 擬合融合權重（供其他實驗匯入）
    """One weight per component: maximise the multinomial log-likelihood of the true candidate under softmax(sum_j w_j Z_j) (Newton)."""
    Z = np.stack(Zs).astype(np.float32); J, Q, _ = Z.shape; w = np.ones(J); qi = np.arange(Q)  # [成分, 題, 候選人]，權重初值為 1
    for _ in range(iters):  # 牛頓法迭代
        logits = np.einsum("j,jqc->qc", w, Z); logits -= logits.max(1, keepdims=True); P = np.exp(logits); P /= P.sum(1, keepdims=True)  # 目前權重下的候選人 softmax
        EZ = np.einsum("jqc,qc->jq", Z, P); Zt = Z[:, qi, truth]  # Z 的期望值與真實候選人的 Z
        grad = (EZ - Zt).mean(1) + 2 * l2 * w  # 負對數似然的梯度加 L2
        A = (Z * P[None]).reshape(J, -1); H = (A @ Z.reshape(J, -1).T) / Q - (EZ @ EZ.T) / Q + 2 * l2 * np.eye(J)  # 海森矩陣：Z 在 softmax 下的共變異數加 L2
        step = np.linalg.solve(H, grad); w = w - step  # 牛頓步
        if np.abs(step).max() < 1e-7:  # 已收斂
            break  # 提早結束
    return w  # 回傳 [J] 權重


def fuse(S, valid, w):  # 依權重融合標準化後的成分分數
    return np.einsum("j,jqc->qc", np.asarray(w), np.stack([standardise(S[c][:, valid]) for c in COMPS]))  # 只在有效候選人上標準化後加權相加


def top2(F):  # 每題分數最高的兩位候選人與其差距
    o = np.argpartition(-F, 1, axis=1)[:, :2]; a = np.take_along_axis(F, o, 1); sw = a[:, 1] > a[:, 0]  # 取前兩名，標出順序顛倒者
    o[sw] = o[sw][:, ::-1]; a[sw] = a[sw][:, ::-1]  # 讓第一欄為最高分
    return o[:, 0], o[:, 1], a[:, 0] - a[:, 1]  # 第一名、第二名、邊際


def label_post(F, lab_valid, K=len(LEVELS)):  # 由候選人分數得到標籤後驗
    Z = F - F.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)  # 候選人上的 softmax
    return np.stack([np.bincount(lab_valid, weights=P[q], minlength=K) for q in range(len(F))])  # 依候選人的標籤加總機率


def fit_p(obs, f_true, f_imp):  # 混合模型估計觀測邊際中「玩家在池內」的占比
    ts = np.quantile(f_imp, [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]); A, y = [], []  # 以池外邊際的分位當門檻
    for t in ts:  # 每個門檻一個方程式
        A.append((f_true > t).mean() - (f_imp > t).mean()); y.append((obs > t).mean() - (f_imp > t).mean())  # 超過門檻的比例：池內參考與觀測各減池外參考
    A, y = np.array(A), np.array(y)  # 轉成陣列
    return float(np.clip((A @ y) / max(A @ A, 1e-12), 0, 1))  # 最小平方解，截在 [0, 1]


def read_tests(paths):  # 讀取測試題的對局
    qs = []  # 收集題目
    for p in paths:  # Public 與 Private 兩個檔
        df = pd.read_csv(p, dtype=str, keep_default_na=False)  # 全部欄位以字串讀入
        for d in df.to_dict("records"):  # 逐題處理
            n = int(d["num_games"]); games = [(d.get(f"sgf_{i}", ""), d.get(f"color_{i}", "")) for i in range(1, n + 1)]  # 該題局數與每局的棋譜、顏色
            qs.append({"question_id": d["question_id"], "n": n, "games": [(s, c) for s, c in games if s and c in ("B", "W")]})  # 只保留有棋譜且顏色有效的對局
    return qs  # 回傳題目清單


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))  # 輸出路徑與工作行程數
    ap.add_argument("--root", default=str(ROOT)); ap.add_argument("--gate", default=None); ap.add_argument("--ct7", default=None); ap.add_argument("--limit-players", type=int, default=0)  # 資料根目錄、gate 檔、ct7 檔、玩家數上限
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(0); root = Path(a.root)  # 解析參數、計時、固定亂數
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    gate_path = Path(a.gate) if a.gate else find_input(root, "results/r_variants", "prod_heldout.P.npz"); ct7_path = Path(a.ct7) if a.ct7 else find_input(root, ".", "submission_rank_ct7.csv")  # 找 gate 題與 ct7 提交檔
    train_files = sorted((root / "data/training").glob("train_*.csv"))  # 訓練集 CSV
    if not train_files:  # 沒有訓練資料時
        raise SystemExit(f"no data/training/train_*.csv under {root}")  # 中止並說明
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in train_files], ignore_index=True)  # 讀入全部訓練對局
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]  # 對局的唯一鍵（與 gate 檔相同格式）
    z = np.load(gate_path, allow_pickle=True); gate_keys = [json.loads(k) for k in z["keys"]]; gy, gn, gpid = z["y"].astype(int), z["n"].astype(int), z["pid"]  # gate 題的對局鍵、標籤、局數與玩家
    gate_games = frozenset(k for ks in gate_keys for k in ks); gate_players = frozenset(gpid.tolist())  # gate 題用到的對局與玩家
    if a.limit_players:  # 快速試跑：只取部分玩家
        others = np.array(sorted(set(rows.player_id) - gate_players)); keep = set(rng.choice(others, min(a.limit_players, len(others)), replace=False).tolist()) | gate_players  # 抽樣其他玩家並保留全部 gate 玩家
        rows = rows[rows.player_id.isin(keep)].reset_index(drop=True)  # 只留這些玩家的對局
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)  # 候選玩家清單與索引
    plabel = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位候選玩家的等級索引
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k: i for i, k in enumerate(keys)}  # 每局的玩家索引與對局鍵的位置
    log(f"{len(rows)} rows, {C} players; gate: {len(gate_keys)} questions, {len(gate_players)} players, {len(gate_games)} games; workers {a.workers}")  # 回報資料規模
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")  # 可用 fork 時用 fork，否則 spawn
    with ctx.Pool(a.workers) as pool:  # 平行萃取訓練對局的視角
        views = list(pool.imap(game_view, zip(rows.sgf_content.tolist(), rows.color.tolist()), chunksize=512))  # 每局的成分輸入
    rows = rows.drop(columns=["sgf_content"]); marks = np.array([v[0] for v in views]); log(f"views for {len(views)} games")  # 丟掉棋譜文字省記憶體，取出每局的記號
    questions = read_tests([root / "data/tests/rank_prediction_test_public.csv", root / "data/tests/rank_prediction_test_private.csv"])  # 讀入 800 個測試題
    with ctx.Pool(a.workers) as pool:  # 平行萃取測試對局的視角
        tviews = list(pool.imap(game_view, [g for q in questions for g in q["games"]], chunksize=256))  # 全部測試對局攤平後處理
    test_views, pos = [], 0  # 依題目切回
    for q in questions:  # 逐題取回自己的對局
        test_views.append(tviews[pos:pos + len(q["games"])]); pos += len(q["games"])  # 依局數切片
    test_n = np.array([q["n"] for q in questions]); test_qid = np.array([q["question_id"] for q in questions]); log(f"test views: {len(tviews)} games, {len(questions)} questions")  # 測試題的局數與題號
    gate_views = [[views[key_index[k]] for k in ks] for ks in gate_keys]; truth = np.array([pidx[p] for p in gpid])  # gate 題的對局視角與正確玩家索引
    need = [np.unique(np.concatenate([v[1][:, j] for vs in gate_views + test_views for v in vs])) for j in range(4)]  # 查詢會用到的開局鍵，只為這些鍵建表
    rep = {"players": C, "rows": int(len(rows)), "gate_questions": len(gate_keys), "gate_players": len(gate_players), "comps": list(COMPS)}  # 報告的基本資訊
    # ---- A：池內
    log("A: fitting without the gate questions' games")  # 情境 A 開始
    pools = fit_scenario(views, marks, pl, ~rows.game_key.isin(gate_games).to_numpy(), need, C, log)  # 只扣掉 gate 題本身的對局
    SA = score_questions(pools, gate_views, C); del pools; log("A: scored")  # gate 題的成分分數，之後釋放池
    valid = np.ones(C, bool)  # 全部候選人都有效
    half = np.array([int(p[-1], 16) % 2 if p[-1] in "0123456789abcdef" else len(p) % 2 for p in gpid])  # 依玩家 id 末碼把 gate 玩家分成兩半
    ws, FA = [], np.zeros((len(gate_keys), C))  # 兩半的權重與交叉擬合的融合分數
    for h in (0, 1):  # 交叉擬合：一半擬合、另一半套用
        fit_q = np.flatnonzero(half == h); app_q = np.flatnonzero(half != h)  # 擬合用與套用的題目
        w = fit_weights([standardise(SA[c][fit_q]) for c in COMPS], truth[fit_q]); ws.append(w.tolist())  # 在這一半上擬合權重
        FA[app_q] = fuse({c: SA[c][app_q] for c in COMPS}, valid, w)  # 套用到另一半
    w_mean = np.mean(ws, axis=0); rep["fusion_weights_halves"] = ws; rep["fusion_weights_mean"] = w_mean.tolist()  # 兩半權重的平均供 B、C 使用
    A1, A2, Am = top2(FA); A_correct = A1 == truth; A_post = label_post(FA, plabel)  # 前兩名、邊際、是否認對與標籤後驗
    comp_top1 = {c: float((np.argmax(SA[c], 1) == truth).mean()) for c in COMPS}  # 各成分單獨的 top-1 正確率
    rep["A"] = {"top1_by_component": comp_top1, "fused_top1": float(A_correct.mean()), "fused_top5": float(np.mean([(FA[q] > FA[q, truth[q]]).sum() < 5 for q in range(len(truth))])),  # 各成分與融合的 top-1、融合的 top-5
                "fused_top1_by_n_group": {str(g): float(A_correct[np.isin(gn, list(r))].mean()) for g, r in NGROUP.items()},  # 各局數組的融合 top-1
                "median_margin_correct": float(np.median(Am[A_correct])) if A_correct.any() else None, "median_margin_wrong": float(np.median(Am[~A_correct])) if (~A_correct).any() else None,  # 認對與認錯時的邊際中位數
                "wrong_match_label_equal": float((plabel[A1[~A_correct]] == gy[~A_correct]).mean()) if (~A_correct).any() else None,  # 認錯時配到同級玩家的比例
                "wrong_match_label_within1": float((np.abs(plabel[A1[~A_correct]] - gy[~A_correct]) <= 1).mean()) if (~A_correct).any() else None}  # 認錯時配到差一級以內的比例
    log(f"A: top-1 by component {json.dumps({k: round(v, 3) for k, v in comp_top1.items()})}; fused top-1 {A_correct.mean():.3f}, by n group {json.dumps({k: round(v, 3) for k, v in rep['A']['fused_top1_by_n_group'].items()})}; weights {np.round(w_mean, 2).tolist()}")  # 回報情境 A 的結果
    # ---- B：池外
    log("B: fitting without the gate players")  # 情境 B 開始
    pools = fit_scenario(views, marks, pl, ~rows.player_id.isin(gate_players).to_numpy(), need, C, log)  # gate 玩家的全部對局都扣掉
    SB = score_questions(pools, gate_views, C); del pools  # gate 題的成分分數，之後釋放池
    validB = ~np.isin(players, list(gate_players)); vB = np.flatnonzero(validB)  # 候選人只剩非 gate 玩家
    FB = fuse(SB, validB, w_mean); B1v, _, Bm = top2(FB); B1 = vB[B1v]; B_post = label_post(FB, plabel[validB])  # 融合、第一名（換回全域索引）、邊際與標籤後驗
    rep["B"] = {"candidates": int(validB.sum()), "margin_quantiles": {str(q): float(np.quantile(Bm, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},  # 候選人數與池外邊際的分位數
                "label_equal_all": float((plabel[B1] == gy).mean()), "label_within1_all": float((np.abs(plabel[B1] - gy) <= 1).mean())}  # 池外題的第一名（必為他人）與題目玩家同級、差一級以內的比例
    for share in (0.01, 0.02, 0.05, 0.1):  # 只看邊際最高的一小部分
        top = Bm >= np.quantile(Bm, 1 - share)  # 邊際在前 share 的題目
        rep["B"][f"top{int(share * 100)}pct_label_equal"] = float((plabel[B1[top]] == gy[top]).mean()); rep["B"][f"top{int(share * 100)}pct_label_within1"] = float((np.abs(plabel[B1[top]] - gy[top]) <= 1).mean())  # 有把握的誤配是否落在同級玩家
    rep["thresholds"] = {}  # 各局數組的接受門檻
    for g, r in NGROUP.items():  # 逐一局數組
        sel = np.isin(gn, list(r)); d = {}  # 屬於此組的 gate 題
        for q in (0.9, 0.95, 0.99, 0.995):  # 以池外邊際的分位數當門檻
            t = float(np.quantile(Bm[sel], q)); above = Am[sel] > t  # 門檻與池內題超過門檻者
            d[str(q)] = {"t": t, "A_share_above": float(above.mean()), "A_precision_above": float(A_correct[sel][above].mean()) if above.any() else None, "A_recall": float((above & A_correct[sel]).mean())}  # 超過比例、精確率與召回率
        rep["thresholds"][str(g)] = d  # 存入報告
    log("B: margin q50/q90/q99 {:.2f}/{:.2f}/{:.2f}; top-1% matches label-equal {:.3f} within-1 {:.3f} (all B: {:.3f}/{:.3f}); A (precision, recall) above B q99 by n group {}".format(  # 回報情境 B 的結果
        rep["B"]["margin_quantiles"]["0.5"], rep["B"]["margin_quantiles"]["0.9"], rep["B"]["margin_quantiles"]["0.99"], rep["B"]["top1pct_label_equal"], rep["B"]["top1pct_label_within1"], rep["B"]["label_equal_all"], rep["B"]["label_within1_all"],  # 邊際分位與同級率
        json.dumps({g: (None if d["0.99"]["A_precision_above"] is None else round(d["0.99"]["A_precision_above"], 3), round(d["0.99"]["A_recall"], 3)) for g, d in rep["thresholds"].items()})))  # 各局數組在 q99 門檻的精確率與召回率
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    np.savez_compressed(out.with_suffix(".gate.npz"), y=gy, n=gn, pid=gpid, half=half, players=players, plabel=plabel, truth=truth,  # 寫出 gate 結果：題目資訊與候選人
                        A_top1=A1, A_top2=A2, A_margin=Am, A_correct=A_correct, A_post=A_post, A_top1_label=plabel[A1],  # 情境 A 的配對結果
                        B_top1=B1, B_margin=Bm, B_post=B_post, B_top1_label=plabel[B1], w=w_mean)  # 情境 B 的配對結果與權重
    # ---- C：測試
    log("C: fitting on all games")  # 情境 C 開始
    pools = fit_scenario(views, marks, pl, np.ones(len(rows), bool), need, C, log)  # 以全部訓練對局擬合
    ST = score_questions(pools, test_views, C); del pools  # 測試題的成分分數，之後釋放池
    FT = fuse(ST, valid, w_mean); T1, T2, Tm = top2(FT); T_post = label_post(FT, plabel)  # 融合、前兩名、邊際與標籤後驗
    ct7 = pd.read_csv(ct7_path, dtype=str).set_index("question_id")["pred_rank"].reindex(test_qid).map(RANK).to_numpy()  # ct7 提交檔的答案（對照用）
    rep["test"] = {"questions": int(len(test_views)), "games": int(len(tviews)), "mixture_in_pool_share_by_n": {}, "ct7_agreement_top1_label_by_margin_decile": {}}  # 測試端報告
    for g, r in NGROUP.items():  # 逐一局數組
        sel = np.isin(gn, list(r)); obs = Tm[test_n == g]  # 對應的 gate 題與該局數的測試邊際
        rep["test"]["mixture_in_pool_share_by_n"][str(g)] = fit_p(obs, Am[sel], Bm[sel]) if len(obs) else None  # 估計測試題玩家在池內的占比
    order = np.argsort(-Tm)  # 依邊際由高到低排序
    for i, idx in enumerate(np.array_split(order, 10)):  # 分成十等分
        rep["test"]["ct7_agreement_top1_label_by_margin_decile"][str(i)] = {"margin_min": float(Tm[idx].min()), "label_equal_ct7": float((plabel[T1[idx]] == ct7[idx]).mean()), "within1": float((np.abs(plabel[T1[idx]] - ct7[idx]) <= 1).mean())}  # 第一名的標籤與 ct7 答案的吻合度
    for g in NGROUP:  # 逐一局數組
        tB = rep["thresholds"][str(g)]["0.99"]["t"]; sel = test_n == g  # 池外 q99 門檻與該局數的測試題
        rep["test"][f"n{g}_share_above_B_q99"] = float((Tm[sel] > tB).mean()) if sel.any() else None  # 超過門檻的測試題占比
    log("C: mixture in-pool share by n {}; share above B q99 by n {}; label(top-1) = ct7 in the top margin decile {:.3f} vs bottom {:.3f}".format(  # 回報情境 C 的結果
        json.dumps({k: (None if v is None else round(v, 3)) for k, v in rep["test"]["mixture_in_pool_share_by_n"].items()}), json.dumps({g: rep["test"][f"n{g}_share_above_B_q99"] for g in NGROUP}),  # 池內占比與超過門檻占比
        rep["test"]["ct7_agreement_top1_label_by_margin_decile"]["0"]["label_equal_ct7"], rep["test"]["ct7_agreement_top1_label_by_margin_decile"]["9"]["label_equal_ct7"]))  # 最高與最低十分位的吻合度
    np.savez_compressed(out.with_suffix(".test.npz"), qid=test_qid, n=test_n, top1=T1, top2=T2, margin=Tm, post=T_post, top1_label=plabel[T1], top1_player=players[T1], players=players, plabel=plabel, w=w_mean)  # 寫出測試端結果（ident_ct8.py 讀取）
    out.write_text(json.dumps(rep, indent=1, default=float)); log(f"written {out}")  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
