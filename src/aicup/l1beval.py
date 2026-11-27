"""The L1b early signal (plan 2026-10-25-l1b-rank-heads, Task 6): L1b against Human SL on the same moves.

Both scorings are in scripts/score_folds.py's layout (games_s*_p*.csv + moves_s*_p*.npz per fold).
align_moves reads a fold from both directories and refuses anything but the same games with the same
moves (game_key set, n_moves and move indices t per game), so every comparison is paired move by move.
per_rank_nll gives each rank's mean per-move NLL under its own L1b head and under Human SL's best (and
pilot-peak) profile; question_sums turns per-move log p into per-question sums (one 20-game question
per player with exactly 20 scored games, as the pilot's n = 20 questions) and adjacent_auc is the
pilot's §2.4.1(d) statistic: posterior-mean index of the summed curve, Mann-Whitney AUC of each pair of
adjacent ranks (P(the higher rank's index is higher)).
"""
import numpy as np  # 數值陣列
import pandas as pd  # 讀取局表

from aicup.evidence import auc, posterior_mean_index  # AUC 與後驗平均索引
from aicup.humansl import PROFILES  # 18 個 Human SL 剖面
from aicup.metric import LEVELS, RANK_TO_IDX  # 10 個等級與其索引
from aicup.scoring import completed_parts, part_paths  # 評分分段檔的列舉與路徑

# §2.4.1(a)：每個等級的似然曲線達到峰值的 Human SL 剖面（KGS 尺度被壓縮）
PILOT_PEAK = {"D": "6k", "C": "5k", "B": "4k", "A": "3k", "1D": "2k", "2D": "2k", "3D": "2k", "4D": "1k", "5D": "1d", "6D": "2d"}  # 等級 -> 峰值剖面
# §2.4.1(d)，n = 20 題，凍結的 Human SL pilot 的相鄰等級 AUC（第 0 折，每級 20 位玩家）
PILOT_AUC_REFERENCE = {"D|C": 0.87, "C|B": 0.77, "B|A": 0.79, "A|1D": 0.88, "1D|2D": 0.62, "2D|3D": 0.73, "3D|4D": 0.65,  # 相鄰等級對 -> 參考 AUC
                       "4D|5D": 0.86, "5D|6D": 0.89}  # 最後兩對
ADJACENT_PAIRS = [f"{a}|{b}" for a, b in zip(LEVELS, LEVELS[1:])]  # 9 對相鄰等級


def read_moves(fold_dir):  # 讀入一折的逐手分數
    """(games, logp [M, S] float32, game_row [M] into games, t [M]) of every finished part of one fold, sorted by
    (game_row, t). FileNotFoundError when the fold has no finished part."""
    parts = completed_parts(fold_dir)  # 已完成的分段
    if not parts:  # 沒有任何已完成的分段
        raise FileNotFoundError(f"no finished parts in {fold_dir}")  # 停止並說明
    frames, logps, rows, ts, offset = [], [], [], [], 0  # 收集清單與局列號位移
    for s, k in parts:  # 逐段讀入
        csv_path, npz_path = part_paths(fold_dir, s, k)  # 兩個檔的路徑
        games = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)  # 局鍵、玩家、等級與顏色欄以字串讀入，且不把 "NA" 等字串當成缺值
        with np.load(npz_path, allow_pickle=False) as z:  # 安全讀取逐手陣列
            logps.append(z["logp"].astype(np.float32))  # 逐手 log p
            rows.append(z["game_row"].astype(np.int64) + offset)  # 轉成全域的局列號
            ts.append(z["t"].astype(np.int64))  # 手數索引
        offset += len(games)  # 推進位移
        frames.append(games)  # 收集局表
    games = pd.concat(frames, ignore_index=True)  # 合併局表
    if not games["game_key"].is_unique:  # 同一局被評分多次
        raise ValueError(f"{fold_dir}: {int(games['game_key'].duplicated().sum())} games scored more than once")  # 拒絕
    logp, game_row, t = np.concatenate(logps), np.concatenate(rows), np.concatenate(ts)  # 合併逐手陣列
    order = np.lexsort((t, game_row))  # 先依局、再依手數排序
    return games, logp[order], game_row[order], t[order]  # 回傳排序後的結果


def align_moves(l1b_dir, hsl_dir, fold):  # 逐手對齊兩套評分的同一折
    """(games, logp_l1b [M, 10], logp_hsl [M, 18], game_row [M], t [M]) of fold `fold` scored in both directories,
    in the L1b games' order. ValueError unless both have exactly the same games with the same moves."""
    games, lp_a, row_a, t_a = read_moves(f"{l1b_dir}/fold{fold}")  # L1b 的評分
    games_b, lp_b, row_b, t_b = read_moves(f"{hsl_dir}/fold{fold}")  # Human SL 的評分
    keys_a, keys_b = set(games["game_key"]), set(games_b["game_key"])  # 兩邊的局鍵集合
    if keys_a != keys_b:  # 涵蓋的局不同
        raise ValueError(f"fold {fold}: the two scorings cover different games ({len(keys_a - keys_b)} only in {l1b_dir}, "  # 直接拒絕，不默默取交集
                         f"{len(keys_b - keys_a)} only in {hsl_dir}; e.g. {sorted(keys_a ^ keys_b)[:3]})")  # 錯誤訊息的後半
    if lp_a.shape[1] != len(LEVELS) or lp_b.shape[1] != len(PROFILES):  # 欄數須分別為 10 與 18
        raise ValueError(f"fold {fold}: expected {len(LEVELS)} L1b heads and {len(PROFILES)} Human SL profiles, got {lp_a.shape[1]} and {lp_b.shape[1]}")  # 拒絕
    # 把 Human SL 的列對應到 L1b 的局順序
    pos_b = {k: i for i, k in enumerate(games_b["game_key"])}  # Human SL 的局鍵 -> 列號
    remap = np.array([pos_b[k] for k in games["game_key"]], dtype=np.int64)  # L1b 列 -> HSL 列
    n_a = np.bincount(row_a, minlength=len(games))  # L1b 每局的手數
    n_b = np.bincount(row_b, minlength=len(games_b))[remap]  # HSL 每局的手數（依 L1b 的局順序）
    if not np.array_equal(n_a, n_b):  # 評分的手數不同
        bad = games["game_key"][np.flatnonzero(n_a != n_b)[:3]].tolist()  # 舉出最多三局
        raise ValueError(f"fold {fold}: {int((n_a != n_b).sum())} games have different numbers of scored moves, e.g. {bad}")  # 拒絕
    inv = np.empty(len(games_b), dtype=np.int64)  # 反向對照表
    inv[remap] = np.arange(len(games))  # HSL 列 -> L1b 列
    row_b_in_a = inv[row_b]  # HSL 每一手所屬的局換成 L1b 的列號
    order = np.lexsort((t_b, row_b_in_a))  # 依 L1b 的局與手數排序
    lp_b, row_b_in_a, t_b = lp_b[order], row_b_in_a[order], t_b[order]  # 重排 HSL 的逐手資料
    if not np.array_equal(row_b_in_a, row_a) or not np.array_equal(t_b, t_a):  # 逐手的位置須完全相同
        diff = np.flatnonzero((row_b_in_a != row_a) | (t_b != t_a))[:1]  # 第一個不同的手
        where = games["game_key"][row_a[diff[0]]] if len(diff) else "?"  # 所屬的局
        raise ValueError(f"fold {fold}: the move indices differ between the two scorings (first at game {where})")  # 拒絕
    return games, lp_a, lp_b, row_a, t_a  # 回傳逐手配對的結果


def per_rank_nll(logp_l1b, logp_hsl, rank_idx):  # 各等級的平均逐手 NLL
    """Mean per-move NLL by true rank: {"l1b": [10] (own head), "hsl_best": [10] (lowest-NLL profile of the rank),
    "hsl_best_profile": [10 names], "hsl_pilot_peak": [10] (PILOT_PEAK profile), "n_moves": [10]}; None where a
    rank has no moves."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)  # 每一手的真實等級
    out = {"l1b": [], "hsl_best": [], "hsl_best_profile": [], "hsl_pilot_peak": [], "n_moves": []}  # 各欄的清單
    for k, lv in enumerate(LEVELS):  # 逐級處理
        sel = rank_idx == k  # 這一級的手
        n = int(sel.sum())  # 手數
        out["n_moves"].append(n)  # 記下手數
        if n == 0:  # 這一級沒有手
            for key in ("l1b", "hsl_best", "hsl_best_profile", "hsl_pilot_peak"):  # 其餘欄位
                out[key].append(None)  # 記為 None
            continue  # 處理下一級
        out["l1b"].append(float(-logp_l1b[sel, k].mean()))  # 自己等級之頭下的 NLL
        hsl = -np.asarray(logp_hsl[sel], dtype=np.float64).mean(axis=0)  # 各剖面下的 NLL
        j = int(np.argmin(hsl))  # NLL 最低的剖面
        out["hsl_best"].append(float(hsl[j]))  # 最佳剖面的 NLL
        out["hsl_best_profile"].append(PROFILES[j])  # 最佳剖面的名稱
        out["hsl_pilot_peak"].append(float(hsl[PROFILES.index(PILOT_PEAK[lv])]))  # pilot 峰值剖面的 NLL
    return out  # 回傳結果


def game_sums(games, logp, game_row):  # 每局的 log p 總和
    """Per game the summed log p [G, S] and move count [G]."""
    sums = np.zeros((len(games), logp.shape[1]), dtype=np.float64)  # 每局一列
    np.add.at(sums, game_row, logp.astype(np.float64))  # 依局累加
    return sums, np.bincount(game_row, minlength=len(games))  # 總和與手數


def question_sums(games, logp, game_row, n_games=20):  # 每位玩家一題的總和
    """(sums [Q, S], rank_idx [Q], n_moves [Q]) of one question per player with exactly n_games scored games (the
    pilot's n = 20 questions); players with another number of games are left out."""
    g_sums, g_n = game_sums(games, logp, game_row)  # 先算每局的總和
    sums, ranks, n_moves = [], [], []  # 收集清單
    for (pid, rank), idx in games.groupby(["player_id", "rank"], sort=True).indices.items():  # 依玩家分組
        if len(idx) != n_games:  # 局數不是 n_games
            continue  # 略過這位玩家
        sums.append(g_sums[idx].sum(axis=0))  # 該玩家各局加總
        ranks.append(RANK_TO_IDX[rank])  # 真實等級索引
        n_moves.append(int(g_n[idx].sum()))  # 總手數
    if not sums:  # 沒有符合的玩家
        return np.zeros((0, logp.shape[1])), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)  # 回傳空陣列
    return np.stack(sums), np.asarray(ranks, dtype=np.int64), np.asarray(n_moves, dtype=np.int64)  # 回傳三個陣列


def adjacent_auc(sums, rank_idx):  # 相鄰等級的 AUC
    """{pair: AUC} for every adjacent pair of ranks: posterior-mean index of the summed curve (softmax over heads or
    profiles), P(index of the higher rank > index of the lower) with ties half; None when a rank has no rows."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)  # 每列的真實等級
    pmi = posterior_mean_index(np.asarray(sums, dtype=np.float64)) if len(rank_idx) else np.zeros(0)  # 每列的後驗平均索引
    out = {}  # 結果字典
    for k, pair in enumerate(ADJACENT_PAIRS):  # 逐對相鄰等級
        lo, hi = pmi[rank_idx == k], pmi[rank_idx == k + 1]  # 較低與較高等級的索引
        out[pair] = float(auc(hi, lo)) if len(lo) and len(hi) else None  # 兩邊都有資料才計算
    return out  # 回傳結果


def early_signal(l1b_dir, hsl_dir, folds):  # 彙整多折的早期訊號報告
    """The whole report over several folds (moves and questions pooled): per-rank NLL, adjacent-rank AUC on 20-game
    questions and on single games, for L1b and Human SL, with the pilot reference."""
    lp_a, lp_b, ranks = [], [], []  # 逐手的 L1b 分數、HSL 分數與等級
    q_sums = {"l1b": [], "hsl": []}  # 每題的總和
    q_ranks = []  # 每題的等級
    g_sums = {"l1b": [], "hsl": []}  # 每局的總和
    g_ranks = []  # 每局的等級
    players_by_rank = {lv: 0 for lv in LEVELS}  # 每級有 20 局題的玩家數
    for fold in folds:  # 逐折處理
        games, a, b, game_row, _ = align_moves(l1b_dir, hsl_dir, fold)  # 逐手對齊兩套評分
        rank_idx = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64)  # 每局的等級索引
        lp_a.append(a)  # 收集 L1b 的逐手分數
        lp_b.append(b)  # 收集 HSL 的逐手分數
        ranks.append(rank_idx[game_row])  # 每一手的等級
        sa, ra, _ = question_sums(games, a, game_row)  # L1b 的 20 局題總和
        sb, rb, _ = question_sums(games, b, game_row)  # HSL 的 20 局題總和
        assert np.array_equal(ra, rb)  # 兩邊選出的題目（以等級序列核對）必須相同
        q_sums["l1b"].append(sa)  # 收集 L1b 的題目總和
        q_sums["hsl"].append(sb)  # 收集 HSL 的題目總和
        q_ranks.append(ra)  # 收集題目的等級
        for k in ra:  # 逐題
            players_by_rank[LEVELS[k]] += 1  # 累計每級的玩家數
        ga, _ = game_sums(games, a, game_row)  # L1b 每局的總和
        gb, _ = game_sums(games, b, game_row)  # HSL 每局的總和
        g_sums["l1b"].append(ga)  # 收集 L1b 的單局總和
        g_sums["hsl"].append(gb)  # 收集 HSL 的單局總和
        g_ranks.append(rank_idx)  # 收集每局的等級
    rank_moves = np.concatenate(ranks)  # 全部手的等級
    nll = per_rank_nll(np.concatenate(lp_a), np.concatenate(lp_b), rank_moves)  # 每級的 NLL
    qr, gr = np.concatenate(q_ranks), np.concatenate(g_ranks)  # 題目與單局的等級
    return {"folds": [int(f) for f in folds], "l1b_dir": str(l1b_dir), "hsl_dir": str(hsl_dir), "n_moves": int(len(rank_moves)),  # 來源與總手數
            "nll": nll, "n_players_by_rank": players_by_rank,  # 每級的 NLL 與玩家數
            "auc_question20": {src: adjacent_auc(np.concatenate(q_sums[src]), qr) for src in ("l1b", "hsl")},  # 20 局題的相鄰等級 AUC
            "auc_game": {src: adjacent_auc(np.concatenate(g_sums[src]), gr) for src in ("l1b", "hsl")},  # 單局的相鄰等級 AUC
            "pilot_reference": dict(PILOT_AUC_REFERENCE)}  # pilot 的參考值


def format_report(res):  # 把報告排成文字表格
    """The two tables as text."""
    lines = ["per-move NLL by true rank (L1b: own head; Human SL: the rank's best of 18 profiles, and the pilot-peak profile)",  # 第一張表的標題
             f"{'rank':>4s} {'moves':>8s} {'l1b':>8s} {'hsl_best':>9s} {'profile':>7s} {'peak':>8s} {'l1b-best':>9s}"]  # 第一張表的欄名列
    nll = res["nll"]  # 每級的 NLL
    for k, lv in enumerate(LEVELS):  # 每級一列
        if nll["l1b"][k] is None:  # 這一級沒有手
            lines.append(f"{lv:>4s} {0:>8d} {'-':>8s}")  # 只印出手數 0
            continue  # 處理下一級
        lines.append(f"{lv:>4s} {nll['n_moves'][k]:>8d} {nll['l1b'][k]:>8.4f} {nll['hsl_best'][k]:>9.4f} {nll['hsl_best_profile'][k]:>7s} "  # 手數、兩種 NLL 與最佳剖面
                     f"{nll['hsl_pilot_peak'][k]:>8.4f} {nll['l1b'][k] - nll['hsl_best'][k]:>+9.4f}")  # 峰值剖面的 NLL 與差值
    lines += ["", "adjacent-rank AUC of the posterior-mean index (20-game questions; single games in brackets); pilot n=20 reference",  # 空一行後接第二張表的標題
              f"{'pair':>6s} {'l1b':>7s} {'hsl':>7s} {'l1b-hsl':>8s} {'pilot':>6s} {'(l1b':>7s} {'hsl)':>7s}"]  # 第二張表的欄名列
    for pair in ADJACENT_PAIRS:  # 每對相鄰等級一列
        a, b = res["auc_question20"]["l1b"][pair], res["auc_question20"]["hsl"][pair]  # 20 局題的兩個 AUC
        ga, gb = res["auc_game"]["l1b"][pair], res["auc_game"]["hsl"][pair]  # 單局的兩個 AUC
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731  沒有值時印成 "-"
        d = "-" if a is None or b is None else f"{a - b:+.3f}"  # L1b 減 HSL 的差
        lines.append(f"{pair:>6s} {f(a):>7s} {f(b):>7s} {d:>8s} {res['pilot_reference'].get(pair, float('nan')):>6.2f} {f(ga):>7s} {f(gb):>7s}")  # 一列結果
    return "\n".join(lines)  # 合成一段文字
