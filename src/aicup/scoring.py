"""Shared Human SL scoring machinery: game loading, CPU workers, streaming GPU scoring,
per-fold game selection and resumable part bookkeeping.

Used by the pilot, the full-fold scoring run and the submission pipeline, so every
log-likelihood in the project comes from the same code path.
"""
import json  # 寫出 patch_meta.json 摘要
import os  # 讀取環境變數 AICUP_TRAIN_DIR
import re  # 比對分段檔名
import shutil  # 複製基準資料夾的其他檔案
import time  # 計算評分耗時與速度
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算與 npz 存取
import pandas as pd  # 讀寫棋局表 CSV

from aicup.pilot import game_positions  # 由 SGF 產生指定顏色每一手的網路輸入

ROOT = Path(__file__).resolve().parents[2]  # 儲存庫根目錄
TRAIN_DIR = Path(os.environ["AICUP_TRAIN_DIR"]) if os.environ.get("AICUP_TRAIN_DIR") else ROOT / "data" / "training"  # 訓練棋譜資料夾；設定 AICUP_TRAIN_DIR 可改用另一份棋局表（section 4M M135：外部玩家），檔案格式相同
_FEATURES = None  # 每個工作程序的 KataGo 特徵物件，由 init_worker 建立


def init_worker(config):  # 工作程序的初始化函式
    """Pool initializer: build KataGo Features once per worker process."""
    global _FEATURES  # 要寫入模組層級的特徵物件
    from katago.game.features import Features  # 在工作程序內匯入 KataGo 特徵類別
    _FEATURES = Features(config, 19)  # 依模型設定建立 19 路棋盤的特徵物件


def extract(task):  # 工作程序：抽出一局的逐手輸入
    idx, sgf, color = task  # 任務編號、棋譜、要評分的顏色
    return idx, game_positions(sgf, color, _FEATURES)  # 回傳編號與該顏色每手的輸入


def load_games(keys, train_dir=TRAIN_DIR):  # 依棋局鍵讀取棋譜
    """sgf and colour for each 'RANK:game_id' key, reading only the rank files needed."""
    by_rank = {}  # 依等級分組的棋局編號
    for k in keys:  # 逐一處理棋局鍵
        r, g = k.split(":")  # 拆成等級與棋局編號
        by_rank.setdefault(r, set()).add(g)  # 加入該等級的集合
    out = {}  # 棋局鍵對應 (sgf, 顏色)
    for r, ids in by_rank.items():  # 每個等級的檔案只讀一次
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["game_id", "color", "sgf_content"])  # 只讀需要的三個欄位
        for row in df[df["game_id"].isin(ids)].itertuples():  # 只留下要求的棋局
            out[f"{r}:{row.game_id}"] = (row.sgf_content, row.color)  # 記下棋譜與玩家執子顏色
    missing = [k for k in keys if k not in out]  # 找不到的棋局鍵
    if missing:  # 有缺漏就中止
        raise KeyError(f"{len(missing)} game keys not found, e.g. {missing[:3]}")  # 回報缺漏數量與範例
    return out  # 回傳對照表


def score_stream(pool, model, tasks, rows, flush_at=1024, log_every=200, log=print, scorer=None):  # 串流評分多局棋
    """Score (idx, sgf, colour) tasks under every metadata row.

    The default scorer is humansl.score_positions(model, ...) under every row of `rows`; `scorer`
    replaces it: scorer(bins, globs, legal, target, colors, rows) -> [N, S] (the L1b heads pass one
    that ignores `model` and `rows`). Returns per-task {idx: (n_moves, sums[S])}, stacked per-move
    arrays (logp [M, S], task idx [M], move index t [M]) or None, number of positions, wall seconds.
    """
    if scorer is None:  # 沒有指定評分函式時
        from aicup import humansl  # 匯入 Human SL 評分模組

        def scorer(bins, globs, legal, target, colors, rows_):  # 預設的 Human SL 評分函式
            return humansl.score_positions(model, bins, globs, legal, target, colors, rows_)  # 每手在每個 metadata 列下的對數機率 [N, S]

    per_task, move_logp, move_task, move_t = {}, [], [], []  # 每局總和與逐手結果的收集處
    buf, t0, done, n_pos = [], time.time(), 0, 0  # 緩衝區、起始時間、完成局數、局面數

    def flush():  # 把緩衝區的局面送上 GPU 評分
        nonlocal buf  # 要清空外層的緩衝區
        if not buf:  # 緩衝區是空的
            return  # 不需評分
        cat = {k: np.concatenate([p[k] for _, p in buf]) for k in ("bins", "globs", "legal", "target", "color")}  # 合併成一個大批次
        logp = scorer(cat["bins"], cat["globs"], cat["legal"], cat["target"], cat["color"], rows)  # 一次評完整個批次
        o = 0  # 目前在批次中的位移
        for idx, p in buf:  # 依原順序拆回每一局
            n = len(p["t"])  # 這一局的手數
            part = logp[o:o + n]  # 取出這一局的逐手結果
            o += n  # 移到下一局的起點
            per_task[idx] = (n, part.sum(axis=0))  # 記下手數與各列的總和
            move_logp.append(part)  # 保留逐手對數機率
            move_task.append(np.full(n, idx, dtype=np.int32))  # 每手所屬的任務編號
            move_t.append(p["t"])  # 每手在棋局中的手數索引
        buf = []  # 清空緩衝區

    for idx, pos in pool.imap(extract, tasks, chunksize=1):  # CPU 工作程序依序產生每局的輸入
        buf.append((idx, pos))  # 放入緩衝區
        n_pos += len(pos["t"])  # 累計局面數
        done += 1  # 累計完成局數
        if sum(len(p["t"]) for _, p in buf) >= flush_at:  # 緩衝的局面已足夠
            flush()  # 送上 GPU 評分
        if done % log_every == 0:  # 定期回報進度
            log(f"  {done}/{len(tasks)} games, {n_pos} positions, {n_pos / (time.time() - t0):.0f} positions/s")  # 輸出進度與速度
    flush()  # 評完剩下的局面
    arrays = (np.concatenate(move_logp), np.concatenate(move_task), np.concatenate(move_t)) if move_logp else None  # 堆疊逐手陣列
    return per_task, arrays, n_pos, time.time() - t0  # 回傳結果與耗時


def player_games(train_dir=TRAIN_DIR):  # 列出訓練集所有玩家的棋局
    """All (player_id, rank, game_key) rows of the training set."""
    frames = []  # 各等級檔案的表格
    for p in sorted(Path(train_dir).glob("train_*.csv")):  # 逐一讀取每個等級的檔案
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank"])  # 只讀需要的欄位
        df["game_key"] = df["rank"] + ":" + df["game_id"]  # 組成 'RANK:game_id' 棋局鍵
        frames.append(df[["player_id", "rank", "game_key"]])  # 保留三個欄位
    return pd.concat(frames, ignore_index=True)  # 合併成一張表


def fold_players(games, folds, fold, max_games=20, min_games=5, seed=42):  # 挑出一折的玩家與棋局
    """Players of one fold with >= min_games games and up to max_games of them each.

    Each player's keys are sorted, then permuted with a seed derived from the player id,
    so a player's selection does not depend on row order or on any other player.
    """
    by_player = games.groupby("player_id")["game_key"].apply(sorted).to_dict()  # 每位玩家排序後的棋局鍵
    rows = []  # 收集每位玩家的結果
    for r in folds[folds["fold"] == fold].sort_values(["rank", "player_id"]).itertuples():  # 依等級與 id 的固定順序走訪該折玩家
        keys = by_player.get(r.player_id, [])  # 這位玩家的棋局
        if len(keys) < min_games:  # 局數不足
            continue  # 不列入
        rng = np.random.RandomState((int(r.player_id[:8], 16) + seed) % 2**32)  # 由玩家 id 前 8 個十六進位字元加上 seed 導出的固定種子
        rows.append({"rank": r.rank, "player_id": r.player_id,  # 記下等級與玩家
                     "game_keys": [keys[i] for i in rng.permutation(len(keys))[:max_games]]})  # 打亂後取前 max_games 局
    return pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])  # 回傳玩家表


def read_game_keys(path):  # 讀取棋局鍵清單檔
    """The game keys listed in a file: a CSV with a game_key column (any other columns ignored), or one key per line
    (blank lines and # comments skipped)."""
    text = Path(path).read_text()  # 讀入整個檔案
    lines = text.splitlines()  # 拆成行
    if lines and "game_key" in lines[0].split(","):  # 第一行是含 game_key 的表頭
        return pd.read_csv(path, usecols=["game_key"], dtype=str, keep_default_na=False)["game_key"].tolist()  # 以 CSV 讀取該欄
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]  # 純文字格式：每行一個鍵，略過空行與註解行


def restrict_players(players, keys):  # 只保留指定的棋局
    """fold_players' table with each player's game_keys cut to `keys` (same order); players left with none are dropped.
    Scoring some games of a fold again (the HD >= 2 games, issue #42): the selection stays the fold's, only the listed
    games are scored, into another directory that patch_fold then merges over the original."""
    want = set(keys)  # 要保留的棋局集合
    rows = [{"rank": r.rank, "player_id": r.player_id, "game_keys": [k for k in r.game_keys if k in want]} for r in players.itertuples()]  # 每位玩家只留指定棋局
    return pd.DataFrame([r for r in rows if r["game_keys"]], columns=["rank", "player_id", "game_keys"])  # 去掉沒有棋局的玩家


def _part_moves(fold_dir):  # 讀出一個評分資料夾的逐手結果
    """{game_key: (row of the games table, logp [n, S], t [n])} over the finished parts of one scored directory."""
    out = {}  # 棋局鍵對應該局的資料
    for s, k in completed_parts(fold_dir):  # 逐一讀取完成的分段
        csv_path, npz_path = part_paths(fold_dir, s, k)  # 分段的 CSV 與 npz 路徑
        games = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)  # 讀棋局表，id 維持字串
        with np.load(npz_path, allow_pickle=False) as z:  # 讀逐手陣列
            logp, row, t = z["logp"], z["game_row"].astype(np.int64), z["t"]  # 對數機率、所屬列、手數索引
        for g, rec in enumerate(games.to_dict("records")):  # 逐局處理
            m = row == g  # 屬於這一局的手
            if rec["game_key"] in out:  # 同一局出現兩次
                raise ValueError(f"{fold_dir}: game {rec['game_key']} scored more than once")  # 視為錯誤
            out[rec["game_key"]] = (rec, logp[m], t[m])  # 記下表列與逐手結果
    if not out:  # 沒有任何完成的分段
        raise FileNotFoundError(f"no finished parts in {fold_dir}")  # 回報找不到結果
    return out  # 回傳對照表


def patch_fold(base_dir, patch_dir, out_dir):  # 把重新評分的棋局併回一折
    """Write out_dir: base_dir's parts with the games scored in patch_dir replaced, game by game (the table's n_moves and sum
    columns and the per-move arrays; player_id, rank and color must agree, and the sum columns must be the same set); every
    other file of base_dir (meta_s*.json, done_s*) is copied. Every patch game must be in base; out_dir must not hold parts.
    Returns the summary (games, patched, n_moves before/after over the patched games, mean |change| and mean change of every
    sum column, the patched keys with their move counts), also written to out_dir/patch_meta.json."""
    base_dir, patch_dir, out_dir = Path(base_dir), Path(patch_dir), Path(out_dir)  # 統一成 Path 物件
    if out_dir.exists() and completed_parts(out_dir):  # 輸出資料夾已有分段
        raise FileExistsError(f"{out_dir} already holds scored parts")  # 拒絕覆蓋
    patch = _part_moves(patch_dir)  # 讀入重新評分的棋局
    base_keys = set()  # 基準資料夾的全部棋局鍵
    for s, k in completed_parts(base_dir):  # 逐一讀取基準的完成分段
        base_keys |= set(pd.read_csv(part_paths(base_dir, s, k)[0], usecols=["game_key"], dtype=str, keep_default_na=False)["game_key"])  # 收集棋局鍵
    if not base_keys:  # 基準沒有任何結果
        raise FileNotFoundError(f"no finished parts in {base_dir}")  # 回報錯誤
    missing = sorted(set(patch) - base_keys)  # 不在基準中的替換棋局
    if missing:  # 有不屬於基準的棋局
        raise ValueError(f"{len(missing)} patched games are not in {base_dir}, e.g. {missing[:3]}")  # 拒絕合併
    out_dir.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    sum_cols, deltas, before, after, patched = None, [], 0, 0, []  # 摘要用的累計變數
    for s, k in completed_parts(base_dir):  # 逐一處理基準的分段
        csv_path, npz_path = part_paths(base_dir, s, k)  # 分段的檔案路徑
        games = pd.read_csv(csv_path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)  # 讀棋局表，id 維持字串
        with np.load(npz_path, allow_pickle=False) as z:  # 讀逐手陣列
            logp, row, t = z["logp"], z["game_row"].astype(np.int64), z["t"]  # 對數機率、所屬列、手數索引
        cols = [c for c in games.columns if c.startswith("sum_")]  # 各剖面的總和欄位
        sum_cols = sum_cols or cols  # 記下第一個分段的欄位供摘要使用
        keep = np.ones(len(row), dtype=bool)  # 要保留的原始逐手資料
        new_logp, new_row, new_t = [], [], []  # 替換進來的逐手資料
        for g in np.flatnonzero(games["game_key"].isin(patch)):  # 這個分段中要替換的棋局
            key = games.at[g, "game_key"]  # 棋局鍵
            rec, lp, tt = patch[key]  # 重新評分的表列與逐手結果
            for c in ("player_id", "rank", "color"):  # 核對身分欄位
                if str(rec[c]) != str(games.at[g, c]):  # 兩邊不一致
                    raise ValueError(f"game {key}: {c} is {rec[c]!r} in the patch but {games.at[g, c]!r} in {base_dir}")  # 拒絕替換
            if set(c for c in rec if c.startswith("sum_")) != set(cols):  # 總和欄位的集合不同
                raise ValueError(f"game {key}: the patch has other sum columns than {base_dir}")  # 拒絕替換
            old_sums = games.loc[g, cols].to_numpy(dtype=float)  # 替換前的總和
            deltas.append(np.array([float(rec[c]) for c in cols]) - old_sums)  # 記錄總和的變化
            before, after = before + int(games.at[g, "n_moves"]), after + int(rec["n_moves"])  # 累計替換前後的手數
            patched.append({"game_key": key, "n_moves_before": int(games.at[g, "n_moves"]), "n_moves_after": int(rec["n_moves"])})  # 記下這一局的手數變化
            games.at[g, "n_moves"] = int(rec["n_moves"])  # 更新手數
            for c in cols:  # 逐一更新總和欄位
                games.at[g, c] = float(rec[c])  # 換成重新評分的值
            keep &= row != g  # 丟掉這一局原本的逐手資料
            new_logp.append(lp)  # 加入新的逐手對數機率
            new_row.append(np.full(len(tt), g, dtype=row.dtype))  # 新的逐手資料對應同一列
            new_t.append(tt)  # 新的手數索引
        logp = np.concatenate([logp[keep], *new_logp]) if new_logp else logp[keep]  # 合併保留與新增的對數機率
        row = np.concatenate([row[keep], *new_row]) if new_row else row[keep]  # 合併所屬列
        t = np.concatenate([t[keep], *new_t]) if new_t else t[keep]  # 合併手數索引
        order = np.lexsort((t, row))  # 依列、再依手數排序
        out_csv, out_npz = part_paths(out_dir, s, k)  # 輸出分段沿用原本的編號
        np.savez_compressed(out_npz, logp=logp[order].astype(np.float32), game_row=row[order].astype(np.int32), t=t[order].astype(np.int16))  # 以評分腳本寫分段時的型別（float32／int32／int16）壓縮寫出逐手陣列
        games.to_csv(out_csv, index=False)  # 寫出更新後的棋局表
    for p in base_dir.iterdir():  # 複製基準資料夾的其他檔案
        if p.is_file() and not _PART_RE.fullmatch(p.name) and not p.name.endswith((".tmp.csv", ".tmp.npz")):  # 不是分段檔也不是暫存檔
            shutil.copy2(p, out_dir / p.name)  # 連同時間戳記複製
    unused = sorted(set(patch) - {q["game_key"] for q in patched})  # 沒有寫入的替換棋局
    if unused:  # 通過上面「每局都在基準中」的檢查後不可能發生，保留作為防護
        raise ValueError(f"{len(unused)} patched games were not written, e.g. {unused[:3]}")  # 回報未寫入的棋局
    D = np.stack(deltas) if deltas else np.zeros((0, len(sum_cols or [])))  # 各替換棋局的總和變化矩陣
    summary = {"base": str(base_dir), "patch": str(patch_dir), "out": str(out_dir), "games": len(base_keys), "patched": len(patched),  # 摘要：資料夾與局數
               "n_moves_before": before, "n_moves_after": after,  # 替換前後的手數
               "sum_mean_abs_change": {c: float(np.abs(D[:, i]).mean()) if len(D) else 0.0 for i, c in enumerate(sum_cols or [])},  # 每欄的平均絕對變化
               "sum_mean_change": {c: float(D[:, i].mean()) if len(D) else 0.0 for i, c in enumerate(sum_cols or [])},  # 每欄的平均變化
               "patched_games": patched}  # 被替換棋局的清單
    (out_dir / "patch_meta.json").write_text(json.dumps(summary, indent=1) + "\n")  # 寫出摘要檔
    return summary  # 回傳摘要


def plan_parts(n_items, part_size):  # 把項目切成固定大小的分段
    return [(i, min(n_items, i + part_size)) for i in range(0, n_items, part_size)]  # 每段的起訖索引


def part_paths(out_dir, shard, k):  # 分段檔案的路徑
    out_dir = Path(out_dir)  # 轉成 Path 物件
    return out_dir / f"games_s{shard}_p{k:03d}.csv", out_dir / f"moves_s{shard}_p{k:03d}.npz"  # 棋局表與逐手陣列


def pending_parts(out_dir, shard, parts):  # 找出還沒完成的分段
    """Indices of parts whose CSV or npz output is missing."""
    return [k for k in range(len(parts)) if not all(p.exists() for p in part_paths(out_dir, shard, k))]  # 兩個檔案缺一即未完成


_PART_RE = re.compile(r"(games|moves)_s(\d+)_p(\d+)\.(csv|npz)")  # 正式分段檔名：種類、分片、編號


def _part_files(out_dir, shard=None):  # 列出資料夾內的分段檔
    """{(shard, index): kinds present} for final part files (temporary files never match)."""
    found = {}  # (分片, 索引) 對應種類集合
    for p in Path(out_dir).glob("*_s*_p*.*"):  # 先以萬用字元篩出候選檔名
        m = _PART_RE.fullmatch(p.name)  # 完整比對檔名格式
        if m and (shard is None or int(m.group(2)) == shard):  # 格式相符且屬於指定分片
            found.setdefault((int(m.group(2)), int(m.group(3))), set()).add(m.group(1))  # 記下 games 或 moves
    return found  # 回傳結果


def completed_parts(out_dir, shard=None):  # 已完成的分段
    """(shard, index) of parts whose CSV and npz both exist, sorted."""
    return sorted(k for k, kinds in _part_files(out_dir, shard).items() if kinds == {"games", "moves"})  # 兩種檔案都在才算完成


def done_game_keys(out_dir, shard):  # 已評分完成的棋局鍵
    done = set()  # 收集已完成的棋局
    for s, k in completed_parts(out_dir, shard):  # 逐一讀取完成的分段
        done |= set(pd.read_csv(part_paths(out_dir, s, k)[0], usecols=["game_key"])["game_key"])  # 加入該分段的棋局鍵
    return done  # 回傳集合


def next_part_index(out_dir, shard):  # 下一個可用的分段編號
    """First part index not used by any file of this shard, finished or not."""
    used = [k for _, k in _part_files(out_dir, shard)]  # 已使用的編號
    return max(used) + 1 if used else 0  # 接在最大編號之後


def resume_plan(keys, out_dir, shard, part_size):  # 規劃續跑的分段
    """[(part_index, keys)] covering every key not already in a finished part, in order.

    Resume is decided per game, not per part number, so the part size may change between
    runs and no game is ever scored twice; new parts are numbered after any existing file.
    """
    done = done_game_keys(out_dir, shard)  # 已評分完成的棋局
    stray = done - set(keys)  # 已完成但不在這次挑選中的棋局
    if stray:  # 這次的挑選與先前不同
        raise ValueError(f"{len(stray)} finished games are not in this selection, e.g. {sorted(stray)[:3]}")  # 拒絕混用不同的挑選
    pending = [k for k in keys if k not in done]  # 尚待評分的棋局（保持原順序）
    start = next_part_index(out_dir, shard)  # 新分段的起始編號
    return [(start + j, pending[i:i + part_size]) for j, i in enumerate(range(0, len(pending), part_size))]  # 切成分段並編號


def pause_requested(path):  # 檢查是否要求暫停
    return Path(path).exists()  # 暫停檔存在即暫停


def read_scored_games(fold_dir):  # 讀取一折的全部評分結果
    """All finished parts of one fold (every shard) as one table; each game must appear once."""
    paths = [part_paths(fold_dir, s, k)[0] for s, k in completed_parts(fold_dir)]  # 各完成分段的棋局表
    if not paths:  # 沒有任何結果
        raise FileNotFoundError(f"no finished parts in {fold_dir}")  # 回報錯誤
    str_cols = {"game_key": str, "player_id": str, "rank": str, "color": str}  # 全為數字的十六進位玩家 id 也必須維持字串
    games = pd.concat([pd.read_csv(p, dtype=str_cols, keep_default_na=False) for p in paths], ignore_index=True)  # 合併所有分段
    if not games["game_key"].is_unique:  # 有棋局被評分兩次以上
        raise ValueError(f"{fold_dir}: {int(games['game_key'].duplicated().sum())} games scored more than once")  # 回報重複的數量
    return games  # 回傳合併後的表


def compare_scored_games(new, ref):  # 比較兩份評分結果
    """Per-game log p sums of `new` against `ref` (same game_key), e.g. one host against another.

    Differences are absolute, per profile; the per-move figures divide by the reference move
    count. best_profile_agreement is the share of games whose highest-scoring profile is the same.
    """
    cols = [c for c in new.columns if c.startswith("sum_")]  # 各剖面的總和欄位
    m = new.merge(ref[["game_key", "n_moves", *cols]], on="game_key", how="left", suffixes=("", "_ref"), indicator=True)  # 依棋局鍵對齊兩邊
    missing = int((m["_merge"] == "left_only").sum())  # 參考端缺少的棋局數
    m = m[m["_merge"] == "both"]  # 只比較兩邊都有的棋局
    if m.empty:  # 沒有共同的棋局
        return {"games": 0, "missing_in_ref": missing}  # 只回報缺漏
    a, b = m[cols].to_numpy(dtype=float), m[[f"{c}_ref" for c in cols]].to_numpy(dtype=float)  # 兩邊的總和矩陣
    diff = np.abs(a - b)  # 每局每剖面的絕對差
    per_move = diff / m["n_moves_ref"].to_numpy(dtype=float)[:, None]  # 換算成每手的差
    return {"games": int(len(m)), "missing_in_ref": missing,  # 比較的局數與缺漏數
            "n_moves_mismatch": int((m["n_moves"] != m["n_moves_ref"]).sum()),  # 手數不同的局數
            "max_abs_sum": float(diff.max()), "max_abs_per_move": float(per_move.max()),  # 最大的差異
            "mean_abs_per_move": float(per_move.mean()),  # 每手的平均差異
            "best_profile_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean())}  # 最佳剖面一致的比例
