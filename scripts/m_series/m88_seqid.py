"""Section 4M M88 (2026-11-11): single-game identification as the main model -- a sequence-level conditional matcher.
M87's funnel: the opponent is a training player in more than half of the games, the current identifier gets him at top-1 in about a
fifth, and the anchor gain is close to linear in that rate. The M82 / M84 matcher only sees a game side as mean / std (and phase means)
of the trunk vectors. Here a Transformer reads the whole game ply by ply (both players' plies, up to 256): per ply the two 64-dim trunk
vectors (board-pooled, played point), the played point, colour, ply index, the displacement to the previous ply and to the mover's own
previous move, and whether the ply belongs to the SUBJECT side (the side to identify). Output: a unit vector for the subject side.
Training is conditional, as M82: an episode takes P players x M games each; every game is a query against the prototypes (normalised
sums of the players' game vectors, its own left out); cross-entropy over the P players.
Cross-fit as M82: net v never sees fold v's players (validation queries of fold v use it), net -1 sees every player (test, held-out).
  --stage train --net v [--smoke]   trains one network (resumable), keeps the state with the best held-out top-1, writes results/m88/out_{v}.npz
  --stage merge                     results/m84_cache/var_S.npz in m84_variant.py's format (then: m84_variant.py --stage eval --tag S / SP)
"""
import argparse  # 解析命令列參數
import json  # 讀寫訓練紀錄與摘要
import math  # 學習率排程的 cos 與尺度的 log
import multiprocessing as mp  # 平行讀取分段檔
import sys  # 設定匯入路徑
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取分段檔的對局表與訓練 CSV

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 加入 src 與實驗腳本的路徑
CACHE, M82, OUT = ROOT / "results/m84_cache", ROOT / "results/m82", ROOT / "results/m88"  # 查詢表快取、M82 的留出局清單、本模型的輸出
DIRS = (ROOT / "results/trunk_vec", ROOT / "results/trunk_vec_x40", ROOT / "results/trunk_vec_x80"); TEST = ROOT / "results/ct_submission/vec"  # 有逐手向量的訓練分段檔資料夾（base、x40、x80）與測試分段檔資料夾
T_MAX, NETS = 256, (1, 2, 3, 4, -1)  # 每局最多讀 256 手；五個交叉擬合網路
D, LAYERS, HEADS, FFN, DROP, D_OUT = 384, 8, 8, 1536, 0.1, 128  # Transformer 的寬度、層數、注意力頭數、FFN 寬度、dropout 與輸出維度
P_EP, M_EP, STEPS, LR, WD, WARM, EVAL_EVERY, MIN_GAMES = 256, 8, 40000, 3e-4, 0.05, 1000, 5000, 8  # 每回合 256 人 × 8 局、步數、學習率、權重衰減、暖身、評估間隔、參與訓練的最少局數
NONE = 37 * 37  # 位移索引的「無」（虛手、填充或不存在）


def read_part(csv_path):  # 把一個主幹向量分段檔讀成補齊的陣列
    """One trunk-vector part as padded per-game arrays: trunk vectors [G, T_MAX, 128] fp16, point [G, T_MAX] (362 = padding), mover,
    length, the colour of the part's target player (0 Black / 1 White), and the game keys (training) or task numbers (test)."""
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)  # 分段檔的對局表
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:  # 對應的逐手向量檔
        g, l, pos, mover, target, row, t = z["g"], z["l"], z["pos"], z["mover"], z["target"], z["game_row"], z["t"]  # 盤面／落點向量、落點、執子方、是否本人、所屬局、手數
    n = len(tab); GL = np.zeros((n, T_MAX, 128), np.float16); POS = np.full((n, T_MAX), 362, np.int16); MOV = np.zeros((n, T_MAX), np.int8); SUBJ = np.zeros(n, np.int8)  # 預先配置補齊的陣列（落點以 362 填充）
    m = t < T_MAX; GL[row[m], t[m], :64] = g[m]; GL[row[m], t[m], 64:] = l[m]; POS[row[m], t[m]] = pos[m]; MOV[row[m], t[m]] = mover[m]; SUBJ[row[target]] = mover[target]  # 前 256 手填入；SUBJ＝目標玩家的顏色
    train = "player_id" in tab.columns  # 有 player_id 欄的是訓練分段檔
    return (tab["game_key"].to_numpy().astype(str) if train else tab["task"].to_numpy().astype(int)), (tab["player_id"].to_numpy().astype(str) if train else tab["question_id"].to_numpy().astype(str)), GL, POS, MOV, np.minimum(np.bincount(row, minlength=n), T_MAX).astype(np.int16), SUBJ  # 局鍵／棋局編號、玩家／題號、各陣列與截斷後的局長


FULL = False  # --full（M90）：再加入 m90_ply_extract.py 補算的 595k 局；輸出到 results/m90，變體名 S2，60,000 步
VAR = "S"  # 寫進快取的變體名
EXT = None  # --ext-ply／--ext-select（M101）：外部玩家的逐手分段檔（M90 的格式）與其 game_key -> 帳號對照表；他們在每個網路都參與訓練，但永遠不當候選人
EXT_SHARE = None  # --ext-share：每個回合的玩家中外部玩家所占的比例（None：不設配額）
SEED = 0  # --seed（M90 補登）：同一網路的其他訓練種子，供其標準化分數取平均


def read_m90(path):  # 讀一個 m90_ply_extract.py 的分段檔
    """One part of m90_ply_extract.py as padded arrays (trunk vectors dequantised to fp16)."""
    st = np.load(ROOT / "results/m90/glstats.npz"); z = np.load(path); L = z["LEN"].astype(np.int64); n = len(L); row = np.repeat(np.arange(n), L); t = np.concatenate([np.arange(x) for x in L]) if n else np.zeros(0, np.int64)  # 量化用的平均／標準差；每手所屬的局與手數
    GL = np.zeros((n, T_MAX, 128), np.float16); POS = np.full((n, T_MAX), 362, np.int16); MOV = np.zeros((n, T_MAX), np.int8)  # 預先配置補齊的陣列
    GL[row, t] = (z["GLQ"].astype(np.float32) / 20.0 * st["sd"] + st["mu"]).astype(np.float16); POS[row, t] = z["POS"]; MOV[row, t] = z["MOV"]  # int8 反量化：x＝q / 20 · sd＋mu
    return z["key"].astype(str), GL, POS, MOV, z["LEN"].astype(np.int16), z["SUBJ"].astype(np.int8)  # 局鍵、各陣列、局長與目標玩家的顏色


def load_games_v1(smoke, log):  # 第一版載入程式（保留給 m90_loader_check.py 與新版逐欄比對）
    files, folds = [], []  # 訓練分段檔與其所屬的折
    for d in DIRS:  # base、x40、x80 三個資料夾
        for f in range(5):  # 折 0–4
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv")); fs = fs[:12] if smoke else fs; files += fs; folds += [f] * len(fs)  # 冒煙測試每折只取 12 個分段檔
    tf = sorted(TEST.glob("games_s*_p*.csv"))  # 測試分段檔
    with mp.get_context("fork").Pool(12) as pool:  # 12 個行程平行讀取
        res = pool.map(read_part, files, chunksize=2); rt = pool.map(read_part, tf, chunksize=2)  # 讀入訓練與測試分段檔
    cat = lambda r, j: np.concatenate([x[j] for x in r])  # noqa: E731  串接各分段檔的第 j 個欄位
    G = dict(key=cat(res, 0), pid=cat(res, 1), GL=cat(res, 2), POS=cat(res, 3), MOV=cat(res, 4), LEN=cat(res, 5), SUBJ=cat(res, 6), fold=np.concatenate([np.full(len(x[0]), f, np.int8) for x, f in zip(res, folds)]))  # 訓練對局的全部欄位
    if FULL:  # --full：再加入補算的對局
        files2 = [f for k in (0, 1, 2) for f in sorted((ROOT / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]; files2 = files2[:20] if smoke else files2  # m90_ply0–2 的分段檔（冒煙測試只取 20 個）
        with mp.get_context("fork").Pool(12) as pool:  # 第二個行程池（此時主行程已持有大量記憶體，見 load_games 的說明）
            r2 = pool.map(read_m90, files2, chunksize=2)  # 讀入補算的分段檔
        tab = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; own = {}  # 玩家→折的對照
        for f_ in sorted((ROOT / "data/training").glob("train_*.csv")):  # 逐一官方訓練 CSV
            c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str); own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))  # 局鍵（等級:game_id）→玩家
        k2 = cat(r2, 0); p2 = np.array([own[k] for k in k2]); f2 = np.array([int(tab.get(p, -9)) for p in p2], np.int8); n_old = len(G["key"])  # 補算對局的玩家與折（不在分折表者記 -9）
        G = dict(key=np.concatenate([G["key"], k2]), pid=np.concatenate([G["pid"], p2]), GL=np.concatenate([G["GL"]] + [x[1] for x in r2]), POS=np.concatenate([G["POS"]] + [x[2] for x in r2]), MOV=np.concatenate([G["MOV"]] + [x[3] for x in r2]),  # 把補算的對局接在既有對局之後
                 LEN=np.concatenate([G["LEN"], cat(r2, 4)]), SUBJ=np.concatenate([G["SUBJ"], cat(r2, 5)]), fold=np.concatenate([G["fold"], f2])); del r2  # 其餘欄位；釋放暫存
        log(f"added {len(k2)} extracted games to the {n_old} stored ones")  # 記錄加入的局數
    o = np.argsort(cat(rt, 0)); Tt = dict(task=cat(rt, 0)[o], GL=cat(rt, 2)[o], POS=cat(rt, 3)[o], MOV=cat(rt, 4)[o], LEN=cat(rt, 5)[o], SUBJ=cat(rt, 6)[o]); assert np.array_equal(Tt["task"], np.arange(len(o)))  # 測試對局依棋局編號排序並確認連號
    log(f"games loaded: {len(G['key'])} training games ({G['GL'].nbytes / 1e9:.1f} GB), {len(o)} test games; mean plies {float(G['LEN'].mean()):.0f}, truncated at {T_MAX}: {float((G['LEN'] >= T_MAX).mean()):.3f}")  # 記錄局數、記憶體用量與手數
    return G, Tt  # 訓練與測試對局


def count_part(csv_path):  # 數一個主幹向量分段檔有幾局
    return len(pd.read_csv(csv_path, usecols=[0], dtype=str, keep_default_na=False))  # 只讀第一欄來計列數


def count_m90(path):  # 數一個補算分段檔有幾局
    with np.load(path) as z:  # 開啟分段檔
        return len(z["LEN"])  # 局長陣列的長度即局數


def load_games(smoke, log):  # 載入全部對局（只開一個行程池的新版）
    """All games as padded arrays. One worker pool, started before anything large exists, and every part written straight into
    preallocated arrays. The compute nodes run with strict overcommit (vm.overcommit_memory = 2, one commit limit for the whole node):
    the first version (load_games_v1) forked a second pool of twelve workers from a parent already holding ~50 GB and kept every part
    in a list next to the concatenated copy; with five loaders on a node that exhausted the node's commit limit (2026-11-12, 13 of 15
    networks died with MemoryError / ENOSPC). Same arrays in the same order as load_games_v1."""
    files, folds = [], []  # 訓練分段檔與其所屬的折
    for d in DIRS:  # base、x40、x80 三個資料夾
        for f in range(5):  # 折 0–4
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv")); fs = fs[:12] if smoke else fs; files += fs; folds += [f] * len(fs)  # 冒煙測試每折只取 12 個分段檔
    tf = sorted(TEST.glob("games_s*_p*.csv")); files2 = []  # 測試分段檔；補算分段檔的清單
    if FULL:  # --full：加入補算的對局
        files2 = [f for k in (0, 1, 2) for f in sorted((ROOT / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]; files2 = files2[:20] if smoke else files2  # m90_ply0–2 的分段檔（冒煙測試只取 20 個）
        if EXT:  # 有外部玩家時
            fx = sorted(f for f in Path(EXT[0]).glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz")); files2 += fx  # 冒煙測試也全部讀入：一個帳號的 20 局分散在各分段檔裡
    cat = lambda r, j: np.concatenate([x[j] for x in r])  # noqa: E731  串接各分段檔的第 j 個欄位
    with mp.get_context("fork").Pool(12) as pool:  # 唯一的行程池，在配置大陣列之前啟動
        n1 = pool.map(count_part, files, chunksize=8); n2 = pool.map(count_m90, files2, chunksize=8) if files2 else []; n_old = int(sum(n1)); N = n_old + int(sum(n2))  # 先數局數以便預先配置
        GL = np.zeros((N, T_MAX, 128), np.float16); POS = np.full((N, T_MAX), 362, np.int16); MOV = np.zeros((N, T_MAX), np.int8); LEN = np.zeros(N, np.int16); SUBJ = np.zeros(N, np.int8); keys, pids, a = [], [], 0  # 一次配置全部陣列
        for x in pool.imap(read_part, files, chunksize=2):  # 依序讀入既有的分段檔
            b = a + len(x[0]); keys.append(x[0]); pids.append(x[1]); GL[a:b] = x[2]; POS[a:b] = x[3]; MOV[a:b] = x[4]; LEN[a:b] = x[5]; SUBJ[a:b] = x[6]; a = b  # 直接寫進預先配置的位置
        assert a == n_old, (a, n_old)  # 確認局數與預先計數相符
        for x in pool.imap(read_m90, files2, chunksize=2):  # 依序讀入補算（與外部帳號）的分段檔
            b = a + len(x[0]); keys.append(x[0]); GL[a:b] = x[1]; POS[a:b] = x[2]; MOV[a:b] = x[3]; LEN[a:b] = x[4]; SUBJ[a:b] = x[5]; a = b  # 直接寫進預先配置的位置
        assert a == N, (a, N)  # 確認總局數相符
        rt = pool.map(read_part, tf, chunksize=2)  # 讀入測試分段檔
    key = np.concatenate(keys); pid = np.concatenate(pids); fold = np.concatenate([np.full(n, f, np.int8) for n, f in zip(n1, folds)])  # 既有對局的局鍵、玩家與折
    if FULL:  # 補上補算對局的玩家與折
        tab = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; own = {}  # 玩家→折的對照
        for f_ in sorted((ROOT / "data/training").glob("train_*.csv")):  # 逐一官方訓練 CSV
            c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str); own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))  # 局鍵（等級:game_id）→玩家
        if EXT:  # 有外部玩家時
            xs = pd.read_csv(EXT[1], usecols=["game_key", "account"], dtype=str); own.update(zip(xs["game_key"], "x:" + xs["account"]))  # 外部帳號加上前綴 x:，與競賽玩家區分
        k2 = key[n_old:]; p2 = np.array([own[k] for k in k2]); f2 = np.array([int(tab.get(p, -9)) for p in p2], np.int8); pid = np.concatenate([pid, p2]); fold = np.concatenate([fold, f2])  # 補算對局的玩家與折（不在分折表者記 -9，例如外部帳號）
        log(f"added {len(k2)} extracted games to the {n_old} stored ones")  # 記錄加入的局數
    G = dict(key=key, pid=pid, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, fold=fold)  # 訓練對局的全部欄位
    o = np.argsort(cat(rt, 0)); Tt = dict(task=cat(rt, 0)[o], GL=cat(rt, 2)[o], POS=cat(rt, 3)[o], MOV=cat(rt, 4)[o], LEN=cat(rt, 5)[o], SUBJ=cat(rt, 6)[o]); assert np.array_equal(Tt["task"], np.arange(len(o)))  # 測試對局依棋局編號排序並確認連號
    log(f"games loaded: {len(G['key'])} training games ({G['GL'].nbytes / 1e9:.1f} GB), {len(o)} test games; mean plies {float(G['LEN'].mean()):.0f}, truncated at {T_MAX}: {float((G['LEN'] >= T_MAX).mean()):.3f}")  # 記錄局數、記憶體用量與手數
    return G, Tt  # 訓練與測試對局


def build_model():  # 建立序列辨識網路
    import torch  # 需要時才匯入 PyTorch
    from torch import nn  # 神經網路模組

    class SeqID(nn.Module):  # 逐手讀整局的 Transformer 認人網路
        def __init__(self):  # 建立各層
            super().__init__()  # 初始化父類別
            self.gl = nn.Linear(128, D); self.pos = nn.Embedding(363, D); self.side = nn.Embedding(2, D); self.col = nn.Embedding(2, D); self.tpos = nn.Embedding(T_MAX, D)  # 主幹向量投影；落點、是否 SUBJECT 方、顏色、手數的嵌入
            self.dprev = nn.Embedding(NONE + 1, D); self.down = nn.Embedding(NONE + 1, D); self.cls = nn.Parameter(torch.zeros(1, 1, D)); self.norm = nn.LayerNorm(D)  # 相對上一手、相對同方上一手的位移嵌入；[CLS] 與正規化
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, FFN, DROP, activation="gelu", batch_first=True, norm_first=True), LAYERS, enable_nested_tensor=False)  # 8 層 pre-norm 的 Transformer 編碼器
            self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D_OUT)); self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))  # 輸出頭與可學的相似度尺度
            self.register_buffer("mu", torch.zeros(128)); self.register_buffer("sd", torch.ones(128))  # 主幹向量標準化用的平均與標準差

        @staticmethod  # 不需要網路狀態
        def disp(pos, k):  # 相對 k 手前的位移索引
            """Index of the displacement from the ply k steps earlier to this ply ((dx + 18) * 37 + dy + 18), NONE when either is a pass / padding / missing."""
            import torch  # 靜態方法內另行匯入
            prev = torch.full_like(pos, 362); prev[:, k:] = pos[:, :-k]; ok = (pos < 361) & (prev < 361)  # k 手前的落點；兩手都在盤上才有位移
            return torch.where(ok, (pos % 19 - prev % 19 + 18) * 37 + (pos // 19 - prev // 19 + 18), torch.full_like(pos, NONE))  # 位移索引，否則為 NONE

        def forward(self, gl, pos, mov, length, subj):  # 前向：回傳每局 SUBJECT 方的單位向量
            import torch  # 前向內另行匯入
            B, T = pos.shape; x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))  # 每手的輸入：各項嵌入相加
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # 位置 0 是 [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]  # 取 [CLS] 的輸出
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)  # 經輸出頭後單位化
    return SeqID()  # 回傳新網路


def stage_train(v, smoke):  # 訓練一個交叉擬合網路
    import torch  # 需要 GPU 的階段才匯入
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}{'_smoke' if smoke else ''}{f'_s{SEED}' if SEED else ''}"; steps = 300 if smoke else STEPS; eval_every = 100 if smoke else EVAL_EVERY  # 輸出標記含網路、冒煙與種子；冒煙測試縮短步數
    log(f"net {v}: GPU {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")  # 記錄 GPU 型號與記憶體
    G, Tt = load_games(smoke, log); q = np.load(CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(M82 / "held_keys.json")))  # 載入對局、查詢表、候選人與留出局
    ext_p = sorted(set(p for p in G["pid"] if p.startswith("x:"))); pidx.update({p: C + i for i, p in enumerate(ext_p)}); CT = C + len(ext_p)  # 外部玩家的索引排在競賽玩家之後；原型與查詢仍只有競賽玩家
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}; n_g = len(owner)  # 每局的主人、是否留出、局鍵→列
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=CT); tp = np.flatnonzero(cnt >= MIN_GAMES); W = int(cnt.max()); tp_ext, tp_own = tp[tp >= C], tp[tp < C]; n_ext = 0 if EXT_SHARE is None or not len(tp_ext) else min(int(round(P_EP * EXT_SHARE)), len(tp_ext))  # 排除留出局與折 v；局數夠的玩家參與訓練；每回合的外部玩家數
    if len(ext_p):  # 有外部玩家時
        log(f"external players: {len(tp_ext)} with {int((use & (owner >= C)).sum())} games" + ("" if EXT_SHARE is None else f", {n_ext} of the {P_EP} players of every episode"))  # 記錄外部玩家數、局數與配額
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((CT, W), -1, np.int64); tab[o, pos_] = idx  # 玩家 × 局的索引表（-1＝空）
    rng = np.random.default_rng(880 + v + 1000 * SEED); torch.manual_seed(880 + v + 1000 * SEED); net = build_model().to(dev)  # 依網路與種子固定亂數並建立網路
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32); lens = G["LEN"][:1]  # noqa: F841  抽 4,000 局估計標準化參數
    flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat  # 去掉填充手後的平均與標準差寫進網路
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))  # AdamW；線性暖身後 cosine 衰減
    ck = OUT / f"ckpt_{tag}.pt"; start = 0; best = {"top1": -1.0, "step": -1}; best_state = None  # 檢查點路徑與最佳狀態
    if ck.exists():  # 有檢查點就續跑
        st = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; log(f"resumed at step {start}, best so far {best}")  # 還原網路、最佳化器、排程與最佳狀態
    tb = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731  轉成 GPU 張量

    def embed(src, rows, subj):  # 分批計算對局的單位向量
        """Unit vectors of games `rows` of `src` (G or the test dict) with subject colours `subj`."""
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)  # 推論模式與輸出陣列
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不需梯度，以 bf16 計算
            for i in range(0, len(rows), 1024):  # 每批 1,024 局
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()  # 前向後存回 CPU
        net.train(); return out  # 切回訓練模式並回傳

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx]); h_truth = owner[hk]; seen = G["fold"][hk] != v  # 留出本人局、正解與「看過的玩家」標記

    def evaluate(step):  # 在留出本人局上量 top-1／5／20
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 原型＝競賽玩家全部非留出局的向量之和再單位化
        S = embed(G, hk, G["SUBJ"][hk]) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res = {"step": step}  # 留出局對原型的相似度與正解的名次
        for name, s in (("seen", seen), ("unseen", ~seen)):  # 看過與沒看過的玩家分開統計
            if s.any():  # 有該類留出局才計算
                res[name] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}  # 局數與 top-1／5／20
        log(f"  step {step}: held-out own games " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if k != "step"))  # 記錄評估結果
        return res, Pm  # 評估結果與原型
    hist = []; net.train(); ar = np.arange(P_EP); tl = 0.0; ta = 0.0; t_last = time.time()  # 評估紀錄、損失與命中的累計
    for step in range(start, steps):  # 訓練迴圈（可由檢查點續跑）
        pl = rng.choice(tp, P_EP, replace=False) if not n_ext else np.concatenate([rng.choice(tp_ext, n_ext, replace=False), rng.choice(tp_own, P_EP - n_ext, replace=False)]); u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1)  # 抽 256 位玩家（含外部配額），每人隨機取 8 局
        with torch.autocast("cuda", dtype=torch.bfloat16):  # 以 bf16 前向
            z = net(tb(G["GL"][rows]), tb(G["POS"][rows].astype(np.int64)), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))  # 2,048 局的單位向量
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T  # 每局對 256 個原型（各玩家向量和）的相似度
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])  # 自己玩家的原型改用留一法（扣掉這一局）
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()  # 交叉熵、梯度裁剪 1.0、更新參數與學習率
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())  # 累計損失與回合內命中
        if (step + 1) % (50 if smoke else 500) == 0:  # 每 500 步（冒煙測試 50 步）記錄一次
            k_ = 50 if smoke else 500; log(f"  step {step + 1}/{steps}: loss {tl / k_:.4f}, episode top-1 {ta / k_:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / k_:.2f} s/step"); tl = ta = 0.0; t_last = time.time()  # 平均損失、命中、尺度與每步秒數
        if (step + 1) % eval_every == 0 or step + 1 == steps:  # 每 5,000 步（冒煙測試 100 步）與最後一步評估
            res, _ = evaluate(step + 1) if len(hk) else ({"step": step + 1, "seen": {"n": 0, "top1": float(step + 1), "top5": 0.0, "top20": 0.0}}, None); hist.append(res); key_ = "seen" if "seen" in res else "unseen"  # 沒有留出局時以步數當假分數；以看過的玩家為準
            if res[key_]["top1"] > best["top1"]:  # 留出 top-1 創新高
                best = {"top1": res[key_]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}  # 記下最佳狀態（複製到 CPU）
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state}, ck); t_last = time.time()  # 存檢查點以便續跑
    net.load_state_dict(best_state); log(f"net {v}: best held-out top-1 {best['top1']:.3f} at step {best['step']}")  # 換回留出局最佳的狀態
    if not len(hk):  # 在子集上的冒煙測試可能沒有任何留出局：用一組假查詢讓程式路徑能跑完
        hk = np.flatnonzero(~is_held)[:8]; h_truth = owner[hk]; seen = np.ones(len(hk), bool)  # 取 8 局當假的留出局
    res, Pm = evaluate(best["step"]); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)  # 最佳狀態的原型；由這個網路負責的查詢
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])  # 非測試查詢（驗證或留出局）：對手側時 SUBJECT 取另一個顏色
    loc = {g_: i for i, g_ in enumerate(mine)}  # 查詢編號→輸出列
    if len(rows):  # 有找得到對局的查詢
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)  # 計算查詢向量
    te = mine[q["qtest"][mine] >= 0]  # 測試端查詢（由全體網路負責）
    if len(te):  # 有測試查詢時
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)  # 測試對局的查詢向量
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv); (OUT / f"train_{tag}.json").write_text(json.dumps({"best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum())}, indent=1))  # 寫出原型、查詢向量與訓練紀錄
    log(f"net {v}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game)")  # 記錄查詢數與找不到對局的查詢數


def stage_merge(seeds):  # 把各網路的輸出合成 m84_variant.py 的快取格式
    q = np.load(CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, D_OUT), np.float32); qvec = np.zeros((len(seeds), N, D_OUT), np.float16); rep = {}; var = VAR if len(seeds) == 1 else VAR + "e"  # 多個種子時變體名加 e（如 S2e）
    for si, sd_ in enumerate(seeds):  # 逐一種子
        for j, v in enumerate(NETS):  # 逐一交叉擬合網路
            t_ = f"{v}{f'_s{sd_}' if sd_ else ''}"; z = np.load(OUT / f"out_{t_}.npz"); protos[si, j] = z["protos"]; qvec[si, z["qidx"]] = z["qvec"]; rep[t_] = json.loads((OUT / f"train_{t_}.json").read_text())["best"]  # 讀入原型與查詢向量，記下最佳的 top-1 與步數
    np.savez(CACHE / f"var_{var}.npz", protos=protos, qvec=qvec); (CACHE / f"var_{var}.json").write_text(json.dumps(rep, indent=1)); print(f"var_{var} written ({len(seeds)} seed(s)); best held-out top-1 per net: " + ", ".join(f"{k}: {x['top1']:.3f} (step {x['step']})" for k, x in rep.items()), flush=True)  # 寫出 var 檔並列出各網路的最佳 top-1


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("train", "merge")); ap.add_argument("--net", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", default="0"); ap.add_argument("--ext-ply", default=None); ap.add_argument("--ext-select", default=None); ap.add_argument("--ext-share", type=float, default=None); ap.add_argument("--out", default=None, help="output folder under results/ and variant name, as folder:VAR (M101: m101:S3)"); ap.add_argument("--steps", type=int, default=None, help="training steps = end of the cosine schedule (default 40,000; 60,000 with --full)"); a_ = ap.parse_args(); SEED = a_.seed  # 解析參數；--seed 設定全域種子
    if a_.full:  # M90：使用全部對局
        FULL, VAR, OUT, STEPS = True, "S2", ROOT / "results/m90", 60000  # 輸出到 results/m90、變體 S2、60,000 步
    if a_.ext_ply:  # M101：加入外部玩家
        assert a_.full and a_.ext_select and a_.out, "--ext-ply needs --full, --ext-select and --out"; EXT, EXT_SHARE = (a_.ext_ply, a_.ext_select), a_.ext_share  # 外部玩家須搭配 --full、--ext-select 與 --out
    if a_.out:  # 指定輸出資料夾與變體名
        OUT, VAR = ROOT / "results" / a_.out.split(":")[0], a_.out.split(":")[1]  # 格式為「資料夾:變體名」
    if a_.steps:  # 指定訓練步數
        STEPS = a_.steps  # 覆寫 cosine 排程的終點
    if a_.stage == "train":  # 訓練階段
        stage_train(a_.net, a_.smoke)  # 訓練 --net 指定的網路
    else:  # 合併階段
        stage_merge([int(x) for x in a_.seeds.split(",")])  # 合併 --seeds 列出的種子
