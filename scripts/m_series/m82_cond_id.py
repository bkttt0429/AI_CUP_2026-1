"""Section 4M M82 (2026-11-10): a LEARNED conditional matcher as an identification component: each candidate is described by his own
games. One 256-dim row per game side = mean and standard deviation of that side's per-ply trunk vectors (64 pooled + 64 played point);
competition 1's B8 network and loss (aicup_pid.embed.EmbedNet): a row -> a unit vector; an episode takes 512 players, one game each as the
query and the sum of the vectors of his other games (up to 39) as his prototype; cross-entropy both ways. Cross-fitted: the queries of
validation fold v use the network trained WITHOUT fold v's players; test and the fusion weights use the all-players network.
Fusion as in M73 / M81, with the held-out own games drawn from the games that have vectors (games 21-40), and four fusions in one run:
five, six (+ tactic, M81), five + matcher, seven.
  --stage rows                 rows of both sides of every game with trunk vectors + the held-out keys -> results/m82/
  --stage embed                the five networks and their embeddings of every row (GPU)
  --stage score --shard k/K    the fusions of shard k of the queries -> results/m82/part{k}of{K}.npz (+ report, shard 0)
"""
import argparse  # 解析命令列參數
import json  # 讀寫留出局清單與報告
import multiprocessing as mp  # 平行讀取向量與評分
import re  # 從 SGF 取出讓子數 HD
import sys  # 調整模組搜尋路徑
import time  # 計時
from collections import defaultdict  # 依鍵收集清單
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取棋局表

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案、同目錄與競賽一 aicup_pid 的程式
import ident_pool as ip  # noqa: E402  五個習慣成分的向量化模型
import m81_tactic_id as m81  # noqa: E402  訓練列與測試局的載入、戰術成分
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
COMPS = ip.COMPS  # 五個成分的名稱
NAMES = (*COMPS, "tactic", "matcher")  # 七個成分：五成分、戰術、配對器
FUS = {"5": [0, 1, 2, 3, 4], "6": [0, 1, 2, 3, 4, 5], "5e": [0, 1, 2, 3, 4, 6], "7": [0, 1, 2, 3, 4, 5, 6]}  # 四種融合各用哪些成分
OUT = ROOT / "results/m82"  # M82 的輸出資料夾
VEC = (("base", ROOT / "results/trunk_vec"), ("x40", ROOT / "results/trunk_vec_x40"))  # 有主幹向量的訓練局：前 20 局與第 21-40 局
TEST_VEC = ROOT / "results/ct_submission/vec"  # 測試局的主幹向量
NETS = (1, 2, 3, 4, -1)  # 網路從未看過其玩家的那一折；-1 = 全部玩家（測試與融合權重用）
OTHER = {"B": "W", "W": "B"}  # 對手的顏色
STEPS, EPISODE, SUPPORT, MIN_GAMES, SEED = 6000, 512, 40, 8, 0  # 訓練步數、每回合玩家數、每人最多局數、最少局數、種子
G = {}  # 子行程共用的全域資料（fork 後繼承）


def part_rows(csv_path):  # 一個主幹向量分段檔中每局雙方的列
    """(games table, rows [2, G, 256] (0 = the target's plies, 1 = the opponent's), ply counts [2, G]) of one trunk-vector part."""
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)  # 讀入棋局表
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:  # 對應的逐手向量檔
        X = np.hstack([z["g"], z["l"]]).astype(np.float64); target, row = z["target"], z["game_row"]  # 每手 128 維向量、是否本人的手、所屬棋局
    n_g = len(tab); out = np.zeros((2, n_g, 256), np.float32); cnt = np.zeros((2, n_g), np.int32)  # 輸出的列與手數
    for side, m in enumerate((target, ~target)):  # 0 = 本人、1 = 對手
        r = row[m]; order = np.argsort(r, kind="stable"); xs = X[m][order]; n = np.bincount(r, minlength=n_g); nz = n > 0; starts = (np.cumsum(n) - n)[nz]  # 依棋局排序並找出每局的起點
        if not nz.any():  # 這一方沒有任何手
            continue  # 略過
        s1 = np.add.reduceat(xs, starts, axis=0); s2 = np.add.reduceat(xs * xs, starts, axis=0); mean = s1 / n[nz][:, None]  # 每局的一次和、平方和與平均
        out[side][nz] = np.hstack([mean, np.sqrt(np.maximum(s2 / n[nz][:, None] - mean * mean, 0.0))]); cnt[side] = n  # 列＝平均接標準差，共 256 維
    return tab, out, cnt  # 回傳棋局表、列與手數


def stage_rows():  # rows 階段：建立每局雙方的列與留出局清單
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    OUT.mkdir(parents=True, exist_ok=True); ctx = mp.get_context("fork"); cols = defaultdict(list)  # 輸出資料夾、工作行程與各欄位的收集
    with ctx.Pool(12) as pool:  # 12 個工作行程
        for src, d in VEC:  # 兩批訓練向量
            for f in range(5):  # 五個折，含第 0 折
                res = pool.map(part_rows, sorted((d / f"fold{f}").glob("games_s*_p*.csv")), chunksize=4)  # 本折各分段檔的列
                for tab, out, cnt in res:  # 逐一分段檔收集
                    cols["key"].append(tab["game_key"].to_numpy()); cols["pid"].append(tab["player_id"].to_numpy()); cols["fold"].append(np.full(len(tab), f, np.int8)); cols["x40"].append(np.full(len(tab), src == "x40"))  # 棋局鍵、玩家、折、是否為第 21-40 局
                    cols["T"].append(out[0]); cols["O"].append(out[1]); cols["nT"].append(cnt[0]); cols["nO"].append(cnt[1])  # 本人與對手的列及手數
                log(f"{src} fold {f}: {sum(len(t) for t, _, _ in res)} games")  # 記錄局數
        res = pool.map(part_rows, sorted(TEST_VEC.glob("games_s*_p*.csv")), chunksize=4)  # 測試局的列
    R = {k: np.concatenate(v) for k, v in cols.items()}; R["key"] = R["key"].astype(str); R["pid"] = R["pid"].astype(str)  # 合併各分段檔的結果
    assert len(np.unique(R["key"])) == len(R["key"]), "a game key appears in two parts"  # 每局只能出現一次
    np.savez(OUT / "rows_train.npz", **R); log(f"training rows: {len(R['key'])} games, {len(np.unique(R['pid']))} players, {int((R['nO'] < 2).sum())} opponent sides with < 2 plies")  # 寫出訓練列
    task = np.concatenate([t["task"].to_numpy().astype(int) for t, _, _ in res]); o = np.argsort(task); assert np.array_equal(task[o], np.arange(len(task)))  # 依測試局的原始順序排序並確認完整
    np.savez(OUT / "rows_test.npz", qid=np.concatenate([t["question_id"].to_numpy() for t, _, _ in res]).astype(str)[o], T=np.concatenate([x[0] for _, x, _ in res])[o], O=np.concatenate([x[1] for _, x, _ in res])[o])  # 寫出測試列
    log(f"test rows: {len(task)} games")  # 記錄測試局數
    # 融合權重用的留出本人對局：每位玩家從第 21-40 局（有向量）中取 2 局，絕不使用驗證題的局
    rows = m81.load_rows(); players = np.array(sorted(rows.player_id.unique())); hd = {k: (int(m.group(1)) if (m := re.search(r"HD\[(\d+)\]", s)) else 0) for k, s in zip(rows.game_key, rows.sgf_content)}  # 訓練列、玩家清單與每局讓子數
    used = set()  # 驗證題用到的棋局鍵
    for v in (1, 2, 3, 4):  # 四個驗證折
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 本折的驗證題
        for ng, gk in zip(m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # n = 20 的題
                used.update(gk)  # 記下它的局
    by_player = defaultdict(list)  # 每位玩家可留出的局
    for k, p_, x in zip(R["key"], R["pid"], R["x40"]):  # 逐一有向量的局
        if x and k not in used and hd.get(k, 9) < 2:  # 第 21-40 局、不在驗證題、HD < 2
            by_player[p_].append(k)  # 列為可留出
    rng = np.random.default_rng(20261110); held = []  # 固定種子；留出局
    for p_ in rng.choice(players, 2400, replace=False):  # 抽 2,400 位玩家
        cand = sorted(by_player.get(p_, []))  # 這位玩家可留出的局
        if len(cand) >= 8:  # 至少 8 局才留出
            held += rng.choice(cand, 2, replace=False).tolist()  # 每人留出 2 局
    (OUT / "held_keys.json").write_text(json.dumps(held)); log(f"held-out own games: {len(held)} of {len(held) // 2} players; validation-question games with vectors: {len(used & set(R['key']))} of {len(used)}")  # 寫出留出局清單


def stage_embed():  # embed 階段：訓練五個網路並輸出每列的嵌入（GPU）
    import torch  # 只有這個階段需要 PyTorch
    from aicup_pid.embed import EmbedNet  # 競賽一的 B8 嵌入網路
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    dev = "cuda"; R = np.load(OUT / "rows_train.npz"); key, pid, fold = R["key"], R["pid"], R["fold"]; T, O = R["T"], R["O"]; Rt = np.load(OUT / "rows_test.npz")  # 讀入訓練列與測試列
    held = set(json.load(open(OUT / "held_keys.json"))); is_held = np.array([k in held for k in key]); players = np.array(sorted(set(pid))); pidx = {p_: i for i, p_ in enumerate(players)}; owner = np.array([pidx[p_] for p_ in pid])  # 留出局遮罩與每列所屬玩家
    pfold = np.zeros(len(players), np.int8); pfold[owner] = fold; rep = {}; emb = {}  # 每位玩家的折；報告與嵌入
    for v in NETS:  # 五個網路
        torch.manual_seed(SEED); use = ~is_held & (fold != v); n_rows = np.bincount(owner[use], minlength=len(players)); tp = np.flatnonzero(n_rows >= MIN_GAMES)  # 訓練列排除留出局與折 v；局數足夠的玩家
        assert n_rows.max() <= SUPPORT, n_rows.max()  # 每人不超過 40 局
        tab = np.full((len(players), SUPPORT), -1, np.int64); fill = np.zeros(len(players), np.int64)  # 每位玩家的局索引表（-1 為空位）
        for g in np.flatnonzero(use):  # 逐一訓練列
            tab[owner[g], fill[owner[g]]] = g; fill[owner[g]] += 1  # 填入所屬玩家的表
        mu = T[use].mean(0); sd = T[use].std(0) + 1e-6; std = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731  以訓練列標準化並放上 GPU
        X = std(T); tab_t = torch.as_tensor(tab[tp], device=dev); cnt_t = torch.as_tensor(fill[tp], device=dev); P = min(EPISODE, len(tp))  # 本人側的列、訓練玩家的局表與局數；每回合玩家數
        net = EmbedNet(256, 512, 128, 0.1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)  # 嵌入網路、AdamW 與 OneCycle 排程
        ar = torch.arange(SUPPORT, device=dev); tgt = torch.arange(P, device=dev); net.train()  # 位置索引與對角線目標
        for step in range(STEPS):  # 逐步訓練
            b = torch.randperm(len(tp), device=dev)[:P]; c = cnt_t[b]; u = torch.rand(P, SUPPORT, device=dev).masked_fill(ar[None] >= c[:, None], 2.0)  # 抽 P 位玩家；空位填 2.0（大於任何亂數），排序後落在最後
            rows = tab_t[b].gather(1, u.argsort(dim=1))  # 玩家的局以隨機順序排列，空位排在最後
            zq = net(X[rows[:, 0]]); sm = ar[None, 1:] < c[:, None]  # 第一局當查詢；其餘有效局的遮罩
            zs = net(X[rows[:, 1:].clamp(min=0)].reshape(-1, 256)).reshape(P, SUPPORT - 1, -1) * sm[..., None]  # 其他局的向量，空位歸 0
            logits = net.log_scale.exp() * zq @ torch.nn.functional.normalize(zs.sum(1), dim=-1).T  # 查詢對各玩家原型的相似度
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))  # 雙向交叉熵
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()  # 反向傳播並更新
            if step % 1000 == 0 or step == STEPS - 1:  # 定期記錄
                log(f"  net {v}: step {step}, loss {float(loss):.4f}, episode top-1 {float((logits.argmax(1) == tgt).float().mean()):.3f} of {P}, scale {float(net.log_scale.exp()):.1f}")  # 損失、回合內 top-1 與縮放
        net.eval()  # 切到評估模式
        with torch.no_grad():  # 推論不需梯度
            enc = lambda A: np.concatenate([net(std(A[i:i + 65536])).cpu().numpy() for i in range(0, len(A), 65536)])  # noqa: E731  分批計算嵌入
            eT, eO = enc(T), enc(O); np.save(OUT / f"emb_T_{v}.npy", eT); np.save(OUT / f"emb_O_{v}.npy", eO)  # 每列本人側與對手側的嵌入
            if v == -1:  # 全部玩家的網路
                np.save(OUT / "emb_test_T.npy", enc(Rt["T"])); np.save(OUT / "emb_test_O.npy", enc(Rt["O"]))  # 另外輸出測試局的嵌入
        emb[v] = eT; rep[f"net_{v}"] = {"players_trained": int(len(tp)), "rows": int(use.sum()), "final_loss": float(loss)}  # 保留嵌入供報告；記錄訓練規模
        log(f"net {v}: {len(tp)} players, {int(use.sum())} rows; embeddings written")  # 記錄完成
    # 報告：留出的本人對局對所有候選人的原型（不含留出列）：樣本內的玩家（網路 -1），以及每個折 v，
    # 折 v 玩家的同一批查詢在網路 v 下的結果（網路從未看過的玩家）
    hi = np.flatnonzero(is_held); truth = owner[hi]  # 留出列與其真正的玩家

    def ident(v, sel):  # 以網路 v 認出留出局的準度
        Pm = np.zeros((len(players), 128)); np.add.at(Pm, owner[~is_held], emb[v][~is_held]); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 每位玩家的原型：非留出列嵌入之和再正規化
        S = emb[v][hi[sel]] @ Pm.T; r = (S > S[np.arange(sel.sum()), truth[sel]][:, None]).sum(1)  # 相似度與真正玩家的名次
        return {"queries": int(sel.sum()), "top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean())}  # 查詢數與 top-1／5／20
    rep["heldout_all_net"] = ident(-1, np.ones(len(hi), bool)); log(f"held-out own games, all-players net: top-1 / top-5 / top-20 {rep['heldout_all_net']['top1']:.3f} / {rep['heldout_all_net']['top5']:.3f} / {rep['heldout_all_net']['top20']:.3f} ({len(hi)} queries, {len(players)} candidates)")  # 全部玩家網路的準度
    for v in (1, 2, 3, 4):  # 逐折比較沒看過與看過
        sel = pfold[truth] == v; a, b = ident(v, sel), ident(-1, sel); rep[f"unseen_fold{v}"] = {"unseen": a, "seen": b}  # 折 v 玩家：網路 v（沒看過）對網路 -1（看過）
        log(f"  fold {v} players' held-out games ({a['queries']}): network that never saw them top-1 {a['top1']:.3f} / top-20 {a['top20']:.3f}; all-players network {b['top1']:.3f} / {b['top20']:.3f}")  # 記錄兩者的準度
    (OUT / "embed_report.json").write_text(json.dumps(rep, indent=1)); log("written")  # 寫出嵌入報告


def raw_scores(i):  # 第 i 個查詢的七成分原始分數
    v = G["qviews"][i]; sc = G["pools"][v[0]].scores([v]); u, cnt = np.unique(G["qtac"][i], return_counts=True)  # 五成分分數；查詢的戰術 token 計數
    tac = cnt.astype(np.float64) @ G["logpT"][u] if len(u) else np.zeros(G["C"])  # 戰術成分（M81）
    return np.vstack([np.stack([sc[c] for c in COMPS]), tac[None], (G["protos"][G["qnet"][i]] @ G["qvec"][i]).astype(np.float64)[None]])  # 形狀 [7, C]：五成分、戰術、配對器


def score_chunk(idx):  # 一批查詢在四種融合下的 top-1
    out = []; C = G["C"]  # 輸出與候選人數
    for i in idx:  # 逐一查詢
        S = raw_scores(i); ex = G["excl"][i]; valid = np.ones(C, bool)  # 原始分數與要排除的候選人
        if ex >= 0:  # 驗證查詢要排除本人
            valid[ex] = False  # 本人不參與標準化
        Z = (S - S[:, valid].mean(1, keepdims=True)) / np.maximum(S[:, valid].std(1, keepdims=True), 1e-12)  # 每個成分對有效候選人標準化
        out.append((i, *[x for name, sel in FUS.items() for x in m81.top2(G["w"][name] @ Z[sel], ex)]))  # 每種融合的 top-1 與 margin
    return out  # 回傳這批結果


def full_chunk(idx):  # 一批留出查詢的完整原始分數，供擬合權重
    return [(i, raw_scores(i).astype(np.float32)) for i in idx]  # 以 float32 節省記憶體


def stage_score(k, K):  # score 階段：第 k 個分片的融合評分
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附經過秒數的記錄函式
    rows = m81.load_rows()  # 讀入訓練列
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}; C = len(players)  # 候選人（訓練玩家）清單與索引
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位候選人的等級標籤
    pl = rows.player_id.map(pidx).to_numpy(dtype=np.int64); keys = rows.game_key.tolist(); key_index = {k_: i for i, k_ in enumerate(keys)}  # 每列的玩家索引與棋局鍵
    sgf = rows.sgf_content.tolist(); col = rows.color.tolist()  # 棋譜與執色
    ctx = mp.get_context("fork")  # 以 fork 讓子行程共用資料
    with ctx.Pool(12) as pool:  # 12 個工作行程
        views = list(pool.imap(ip.game_view, zip(sgf, col), chunksize=512))  # 每列以本人為主體的視角
    marks = np.array([v[0] for v in views]); log(f"{len(rows)} rows, {C} players; target-side views built")  # 視角標記（B 或 W 候選池）
    qrow, qfold, qpid = [], [], []  # 查詢列、折與玩家
    for v in (1, 2, 3, 4):  # 四個驗證折
        m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]  # 本折的驗證題
        for pid, ng, gk in zip(m["player_id"], m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # 只用 n = 20 的題
                for k_ in gk:  # 這題的 20 局
                    qrow.append(key_index[k_]); qfold.append(v); qpid.append(pid)  # 加入一個查詢
    qrow = np.array(qrow); qfold = np.array(qfold); Q = len(qrow)  # 轉成陣列；驗證查詢數
    with ctx.Pool(12) as pool:  # 平行建對手側視角
        opp_views = list(pool.imap(ip.game_view, [(sgf[r], OTHER[col[r]]) for r in qrow], chunksize=512))  # 以對手為主體的視角
    held_keys = json.load(open(OUT / "held_keys.json")); held = np.array([key_index[k_] for k_ in held_keys]); H = len(held); include = np.ones(len(rows), bool); include[held] = False  # rows 階段選出的留出局，不進候選池
    tg = m81.load_tests(); T = len(tg)  # 測試局
    with ctx.Pool(12) as pool:  # 平行建測試局視角
        t_tgt = list(pool.imap(ip.game_view, [g for _, g in tg], chunksize=256)); t_opp = list(pool.imap(ip.game_view, [(g[0], OTHER[g[1]]) for _, g in tg], chunksize=256))  # 本人側與對手側
    log(f"query views: validation {Q} x 2, held-out {H}, test {T} x 2")  # 記錄各類查詢數
    qviews = opp_views + [views[r] for r in qrow] + [views[r] for r in held] + t_opp + t_tgt  # 順序：驗證對手側、驗證本人側、留出局、測試對手側、測試本人側
    excl = [pidx[p_] for p_ in qpid] * 2 + [-1] * H + [-1] * (2 * T)  # 驗證查詢排除本人，其餘不排除
    need = [np.unique(np.concatenate([v[1][:, j] for v in qviews])) for j in range(4)]  # 查詢用到的開局鍵，只為這些建表
    pools = ip.fit_scenario(views, marks, pl, include, need, C, log)  # 擬合 B、W 兩個視角的候選池
    # 戰術成分（M81）
    train, test = m81.Flat("tac_train"), m81.Flat("tac_test"); V = m81.V_TAC  # 讀入戰術 token
    idx = np.flatnonzero(include); ids = np.concatenate([train.get(2 * r) for r in idx]).astype(np.int64); ptok = np.repeat(pl[idx], train.off[2 * idx + 1] - train.off[2 * idx])  # 候選池各列本人側的 token 與所屬玩家
    counts = np.bincount(ptok * V + ids, minlength=C * V).reshape(C, V).astype(np.float64); del ids, ptok  # 每位玩家各 token 值的次數
    bg = counts.sum(0); p_bg = (bg + 5.0 / V) / (bg.sum() + 5.0); s = m81.eb_precision(counts, p_bg)  # 候選池的背景分布與 Dirichlet 精度
    logpT = np.ascontiguousarray(np.log((counts + s * p_bg) / (counts.sum(1, keepdims=True) + s)).T.astype(np.float32)); del counts  # 向背景收縮後的對數機率 [V, C]
    qtac = [train.get(2 * r + 1) for r in qrow] + [train.get(2 * r) for r in qrow] + [train.get(2 * r) for r in held] + [test.get(2 * j + 1) for j in range(T)] + [test.get(2 * j) for j in range(T)]  # 各查詢的 token，順序同 qviews
    log(f"tactic pool fitted (Dirichlet precision {s:.1f})")  # 記錄精度
    # 配對器成分：每個網路一組原型，每個查詢一個向量
    R = np.load(OUT / "rows_train.npz"); rkey = R["key"]; gi = {k_: i for i, k_ in enumerate(rkey)}; heldset = set(held_keys); keep = np.array([k_ not in heldset for k_ in rkey]); cand = np.array([pidx[p_] for p_ in R["pid"]])  # 訓練列的鍵、非留出遮罩與所屬候選人
    Rt = np.load(OUT / "rows_test.npz"); assert np.array_equal(Rt["qid"], np.array([q for q, _ in tg])), "test rows are not in the order of the test games"  # 測試列必須與測試局同序
    protos = np.zeros((len(NETS), C, 128), np.float32); qvec = np.zeros((len(qviews), 128), np.float32); qnet = np.zeros(len(qviews), np.int8); missing = 0  # 各網路的原型、查詢向量與所用網路
    for j, v in enumerate(NETS):  # 逐一網路
        eT = np.load(OUT / f"emb_T_{v}.npy"); Pm = np.zeros((C, 128)); np.add.at(Pm, cand[keep], eT[keep]); protos[j] = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 候選人原型：非留出局本人側嵌入之和再正規化
        if v > 0:  # 排除一折的網路：給該折的驗證查詢用
            eO = np.load(OUT / f"emb_O_{v}.npy"); sel = np.flatnonzero(qfold == v); g_ = np.array([gi.get(keys[r], -1) for r in qrow[sel]]); ok = g_ >= 0; missing += int((~ok).sum())  # 折 v 查詢對應的列；沒有向量的局計數
            qvec[sel[ok]] = eO[g_[ok]]; qvec[Q + sel[ok]] = eT[g_[ok]]; qnet[sel] = qnet[Q + sel] = j  # 對手側用對手嵌入、本人側用本人嵌入
        else:  # 全部玩家的網路：留出局與測試查詢
            qvec[2 * Q:2 * Q + H] = eT[[gi[k_] for k_ in held_keys]]; qvec[2 * Q + H:2 * Q + H + T] = np.load(OUT / "emb_test_O.npy"); qvec[2 * Q + H + T:] = np.load(OUT / "emb_test_T.npy"); qnet[2 * Q:] = j  # 留出局、測試對手側、測試本人側
    log(f"matcher: prototypes of {int((np.linalg.norm(protos[-1], axis=1) > 0).sum())} of {C} candidates; validation query games without vectors: {missing}")  # 記錄原型數與缺向量的查詢
    G.update(qviews=qviews, pools=pools, excl=excl, C=C, qtac=qtac, logpT=logpT, protos=protos, qvec=qvec, qnet=qnet)  # 放進全域給子行程使用
    n0 = 2 * Q; hidx = np.arange(n0, n0 + H); truth = pl[held]  # 留出查詢的索引與真正的玩家
    with ctx.Pool(12) as pool:  # 平行評分
        full = [x for part in pool.map(full_chunk, np.array_split(hidx, 48)) for x in part]  # 留出查詢的完整原始分數
    full.sort(key=lambda x: x[0]); S = np.stack([x[1] for x in full]).transpose(1, 0, 2)  # 形狀 [7, H, C]：成分、留出查詢、候選人
    Z = [ip.standardise(S[j]) for j in range(7)]; w = {name: ip.fit_weights([Z[j] for j in sel], truth) for name, sel in FUS.items()}  # 標準化後為四種融合各擬合權重
    rk = lambda F: (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # noqa: E731  真正玩家的名次（0 = 第一名）
    acc = {}  # 各成分與融合的認人準度
    for name, F in [(c, Z[j]) for j, c in enumerate(NAMES)] + [(f"fused{name}", np.einsum("j,jqc->qc", w[name], np.stack([Z[j] for j in sel]))) for name, sel in FUS.items()]:  # 七個成分各自與四種融合
        r = rk(F); acc[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "median_rank": float(np.median(r) + 1)}  # top-1／5／20 與名次中位數
    for name, sel in FUS.items():  # 逐一融合
        log(f"weights ({name}) " + " ".join(f"{NAMES[j]} {x:.2f}" for j, x in zip(sel, w[name])))  # 記錄權重
    log("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k_} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k_, v in acc.items()))  # 記錄認人準度
    del full, S, Z  # 釋放記憶體
    G.update(w=w)  # 融合權重給子行程使用
    if k == 0:  # 只有分片 0 寫報告
        Path(OUT / "report.json").write_text(json.dumps({"weights": {name: {NAMES[j]: float(x) for j, x in zip(sel, w[name])} for name, sel in FUS.items()}, "held_queries": int(H), "heldout_accuracy": acc, "dirichlet_precision": s}, indent=1))  # 寫出權重、準度與精度
        np.savez_compressed(OUT / "meta.npz", players=players, plab=plab, Q=Q, H=H, T=T)  # 候選人、標籤與各類查詢數
    rest = np.concatenate([np.arange(n0), np.arange(n0 + H, len(qviews))]); mine = np.array_split(rest, K)[k]  # 留出局以外的查詢，取本分片
    with ctx.Pool(12) as pool:  # 平行評分
        res = [x for part in pool.map(score_chunk, np.array_split(mine, 240)) for x in part]  # 本分片的融合結果
    res.sort(key=lambda x: x[0]); a = np.array([x[0] for x in res]); assert np.array_equal(a, mine)  # 依索引排序並確認完整
    out = {"idx": a}  # 輸出欄位
    for j, name in enumerate(FUS):  # 每種融合兩欄
        out[f"b{name}"] = np.array([x[1 + 2 * j] for x in res]); out[f"m{name}"] = np.array([x[2 + 2 * j] for x in res])  # top-1 候選人與 margin
    np.savez_compressed(OUT / f"part{k}of{K}.npz", **out); log(f"shard {k} of {K}: {len(mine)} queries scored and written")  # 寫出本分片


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("rows", "embed", "score")); ap.add_argument("--shard", default="0/1"); a_ = ap.parse_args()  # 階段與分片參數
    if a_.stage == "rows":  # 建立列
        stage_rows()  # 執行 rows 階段
    elif a_.stage == "embed":  # 訓練網路與嵌入
        stage_embed()  # 執行 embed 階段
    else:  # 融合評分
        stage_score(*map(int, a_.shard.split("/")))  # 解析 k/K 後執行 score 階段
