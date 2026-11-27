"""Bit-packed cache of KataGo input positions for L1b training (plan 2026-10-25-l1b-rank-heads, Task 2).

Every position is the board before one target-player move, as pilot.game_positions builds it: 22 binary
input planes (packed to 993 bytes), 19 global features (float32), the legal-move mask (362 bits -> 46
bytes), the target move, the move index and the colour to move. Positions are stored in parts, each a
games table (game_key, player_id, rank, color, n_moves) and an npz of concatenated arrays with game_row
pointing at the table; parts are written to .tmp names and renamed (npz first), so a part counts as done
only when both files exist. read_cache concatenates every finished part with global game rows and the
rank index of each position (read_caches: several directories as one cache, optionally without the games
of given players); plan_parts lists the parts a shard still has to write.
"""
import re  # 解析分段檔名
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列與位元壓縮
import pandas as pd  # 局表的讀寫

from aicup.metric import RANK_TO_IDX  # 等級名稱 -> 索引

N_PLANES, N_GLOBAL, N_MOVES, BOARD = 22, 19, 362, 19 * 19  # 輸入平面數、全域特徵數、落點數（含 pass）、盤面點數
BIN_BYTES = -(-N_PLANES * BOARD // 8)  # 每個局面的平面壓縮後為 993 bytes（無條件進位）
LEGAL_BYTES = -(-N_MOVES // 8)  # 合法手遮罩壓縮後為 46 bytes
ARRAYS = ("bins_packed", "legal_packed", "globs", "target", "t", "color")  # 每個局面儲存的陣列
_PART_RE = re.compile(r"cache_s(\d+)_p(\d+)\.npz$")  # 由檔名取出分片與分段編號


def pack_positions(pos):  # 把局面壓縮成快取格式
    """game_positions' dict -> packed arrays: bins_packed uint8 [n, 993], legal_packed uint8 [n, 46], globs
    float32 [n, 19], target int16, t int16, color uint8."""
    bins = np.asarray(pos["bins"], dtype=np.uint8)  # 二值輸入平面
    n = len(bins)  # 局面數
    legal = np.asarray(pos["legal"], dtype=bool).reshape(n, N_MOVES)  # 合法手遮罩
    return {"bins_packed": np.packbits(bins.reshape(n, N_PLANES * BOARD), axis=1).reshape(n, BIN_BYTES),  # 平面攤平後壓成位元
            "legal_packed": np.packbits(legal, axis=1).reshape(n, LEGAL_BYTES),  # 遮罩壓成位元
            "globs": np.asarray(pos["globs"], dtype=np.float32).reshape(n, N_GLOBAL),  # 全域特徵維持浮點
            "target": np.asarray(pos["target"], dtype=np.int16).reshape(n),  # 目標落點
            "t": np.asarray(pos["t"], dtype=np.int16).reshape(n),  # 局內的手數索引
            "color": np.asarray(pos["color"], dtype=np.uint8).reshape(n)}  # 輪到的顏色


def unpack_bins(packed):  # 還原輸入平面
    """uint8 [n, 993] -> uint8 [n, 22, 19, 19]."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, BIN_BYTES)  # 整理成 [n, 993]
    return np.unpackbits(packed, axis=1, count=N_PLANES * BOARD).reshape(len(packed), N_PLANES, BOARD // 19, 19)  # 解壓並去掉補齊的位元


def unpack_legal(packed):  # 還原合法手遮罩
    """uint8 [n, 46] -> bool [n, 362]."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, LEGAL_BYTES)  # 整理成 [n, 46]
    return np.unpackbits(packed, axis=1, count=N_MOVES).astype(bool)  # 只取前 362 個位元


def part_paths(cache_dir, shard, k):  # 一個分段的兩個檔案路徑
    """(npz path, csv path) of part k of a shard."""
    cache_dir = Path(cache_dir)  # 統一成 Path
    return cache_dir / f"cache_s{shard}_p{k:03d}.npz", cache_dir / f"games_s{shard}_p{k:03d}.csv"  # 依分片與分段編號命名


def write_part(cache_dir, shard, k, games, parts):  # 寫入一個分段
    """Write one part: `games` (game_key, player_id, rank, color, n_moves) and the packed positions of each game
    in the same order. Both files go to .tmp names first, the npz is renamed before the csv."""
    games = pd.DataFrame(games)[["game_key", "player_id", "rank", "color", "n_moves"]].reset_index(drop=True)  # 固定欄位與順序
    if len(games) != len(parts):  # 局數須等於局面區塊數
        raise ValueError(f"{len(games)} games but {len(parts)} position blocks")  # 拒絕不一致的輸入
    counts = [len(p["target"]) for p in parts]  # 每局的局面數
    if games["n_moves"].tolist() != counts:  # n_moves 欄須等於實際的局面數
        raise ValueError(f"n_moves column {games['n_moves'].tolist()} differs from the positions per game {counts}")  # 拒絕不一致的輸入
    arrays = {key: (np.concatenate([p[key] for p in parts]) if parts else np.zeros((0,), dtype=np.uint8)) for key in ARRAYS}  # 串接各局的陣列
    arrays["game_row"] = np.repeat(np.arange(len(games), dtype=np.int32), counts) if parts else np.zeros((0,), dtype=np.int32)  # 每個局面指向局表的列
    if sum(counts) == 0:  # 整個分段沒有任何局面
        arrays = _empty_arrays()  # 改用形狀正確的空陣列
    npz, csv = part_paths(cache_dir, shard, k)  # 正式檔名
    npz.parent.mkdir(parents=True, exist_ok=True)  # 建立快取資料夾
    npz_tmp, csv_tmp = npz.with_name(npz.name + ".tmp"), csv.with_name(csv.name + ".tmp")  # 暫存檔名
    with open(npz_tmp, "wb") as fh:  # 以檔案物件寫入，np.savez 才不會自動加上 .npz
        np.savez(fh, **arrays)  # 寫入陣列
    games.to_csv(csv_tmp, index=False)  # 寫入局表
    npz_tmp.rename(npz)  # npz 先改名
    csv_tmp.rename(csv)  # csv 最後改名，兩個檔齊全才算完成


def _empty_arrays():  # 沒有局面時的空陣列
    return {"bins_packed": np.zeros((0, BIN_BYTES), dtype=np.uint8), "legal_packed": np.zeros((0, LEGAL_BYTES), dtype=np.uint8),  # 二維欄位保留第二維的寬度
            "globs": np.zeros((0, N_GLOBAL), dtype=np.float32), "target": np.zeros((0,), dtype=np.int16),  # 全域特徵與目標落點
            "t": np.zeros((0,), dtype=np.int16), "color": np.zeros((0,), dtype=np.uint8), "game_row": np.zeros((0,), dtype=np.int32)}  # 其餘一維欄位


def completed_parts(cache_dir):  # 列出已完成的分段
    """Sorted (shard, k) of every part whose npz and csv both exist (tmp files are not parts)."""
    out = []  # 收集結果
    for p in Path(cache_dir).glob("cache_s*_p*.npz"):  # 只找正式的 npz 檔名
        m = _PART_RE.search(p.name)  # 解析分片與分段編號
        if m and part_paths(cache_dir, int(m.group(1)), int(m.group(2)))[1].exists():  # 對應的 csv 也存在
            out.append((int(m.group(1)), int(m.group(2))))  # 記為已完成
    return sorted(out)  # 依 (分片, 分段) 排序


def plan_parts(keys, part_size, cache_dir, shard):  # 規劃尚未完成的分段
    """[(k, keys of part k)] of the parts a shard has not finished; part k holds keys[k*part_size:(k+1)*part_size]."""
    done = {k for s, k in completed_parts(cache_dir) if s == shard}  # 這個分片已完成的分段
    return [(k, keys[k * part_size:(k + 1) * part_size]) for k in range(-(-len(keys) // part_size)) if k not in done]  # 只列出未完成的分段，讓中斷的建置可以續跑


def read_cache(cache_dir, drop_players=None):  # 讀入單一快取
    """(games, arrays) of every finished part in (shard, k) order; game_row is global, rank_idx int8 per position.
    drop_players: player ids whose games (and positions) are left out, e.g. the players of a held-out fold
    (issue #16). FileNotFoundError when the directory has no finished part."""
    return read_caches([cache_dir], drop_players)  # 當作只有一個資料夾的多快取讀取


def read_caches(cache_dirs, drop_players=None):  # 把多個快取當成一個讀入
    """read_cache over several cache directories in the given order, as one cache: game_row is global over all of
    them (issue #16: a model trained on folds 0-4 minus one reads cache_A, cache_B and cache_0 together). Every
    directory must have a finished part."""
    drop = set(map(str, drop_players)) if drop_players else None  # 要排除的玩家集合
    frames, blocks, offset = [], [], 0  # 局表、陣列區塊與目前的局列號位移
    for cache_dir in cache_dirs:  # 依序讀每個快取
        parts = completed_parts(cache_dir)  # 已完成的分段
        if not parts:  # 這個快取沒有已完成的分段
            raise FileNotFoundError(f"no finished cache parts in {cache_dir}")  # 停止並說明
        for shard, k in parts:  # 逐段讀入
            npz, csv = part_paths(cache_dir, shard, k)  # 兩個檔的路徑
            games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)  # 局鍵、玩家、等級與顏色欄以字串讀入，且不把 "NA" 等字串當成缺值（n_moves 仍為數字）
            with np.load(npz, allow_pickle=False) as z:  # 安全讀取陣列
                arr = {key: z[key] for key in (*ARRAYS, "game_row")}  # 取出所有欄位
            arr["game_row"] = arr["game_row"].astype(np.int32)  # 統一列號型別
            if drop:  # 需要排除玩家
                keep = ~games["player_id"].isin(drop).to_numpy()  # 要保留的局
                if not keep.all():  # 這個分段有要排除的局
                    pos_keep = keep[arr["game_row"]] if len(arr["game_row"]) else np.zeros((0,), dtype=bool)  # 要保留的局面
                    new_row = np.cumsum(keep) - 1  # 保留下來的每一局的新索引
                    arr = {key: arr[key][pos_keep] for key in arr}  # 刪掉被排除之局的局面
                    arr["game_row"] = new_row[arr["game_row"]].astype(np.int32)  # 列號改指向新的局表
                    games = games[keep].reset_index(drop=True)  # 局表也刪掉被排除的局
            arr["game_row"] = arr["game_row"] + offset  # 轉成全域列號
            offset += len(games)  # 推進位移
            frames.append(games)  # 收集局表
            blocks.append(arr)  # 收集陣列
    games = pd.concat(frames, ignore_index=True)  # 合併局表
    arrays = {key: np.concatenate([b[key] for b in blocks]) for key in (*ARRAYS, "game_row")}  # 合併陣列
    rank_of_game = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int8)  # 每局的等級索引
    arrays["rank_idx"] = rank_of_game[arrays["game_row"]] if len(arrays["game_row"]) else np.zeros((0,), dtype=np.int8)  # 每個局面的等級索引
    return games, arrays  # 回傳局表與陣列
