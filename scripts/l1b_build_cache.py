#!/usr/bin/env python
"""Build the L1b training cache: every target-player position of the chosen folds' players, bit-packed.

Plan 2026-10-25-l1b-rank-heads, Task 3. Players are every player of --folds with >= --min-games games
(default 1: all of them), each with up to --max-games games chosen by scoring.fold_players (a fixed
per-player permutation, seed 42, so the first 20 are the evaluation games of results/scores). The games
are shuffled once (RandomState(42)) so every part mixes ranks; shard i of N takes keys[i::N] and writes
its parts (featcache.write_part, --part-size games each) to --out, skipping parts it already finished.
Positions come from pilot.game_positions in a CPU pool (features need only the model config's input
version; --model-config b18c384nbt is humanv0's and the default, no checkpoint is read).

After its parts a shard writes <out>/meta_s{shard}.json from its finished parts' games tables; a single
process also writes <out>/meta.json. With several shards run `--out DIR --merge-meta` once at the end to
combine them into meta.json (folds, max_games, min_games, players, games, positions, per_rank_positions,
games_sha256, provenance).

  python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A --workers 48
  bash scripts/nano4/shards.sh 8 python -I scripts/l1b_build_cache.py --folds 3 4 --out results/l1b/cache_B --workers 12
  python -I scripts/l1b_build_cache.py --out results/l1b/cache_B --merge-meta

--opponent (issue #33, the opponent-head route): the positions before the OPPONENT's moves (the other colour) of the same
games, labelled with the target player's rank -- the training data of heads that read the labelled player's rank off the
opponent's play. The games table's color column is the side extracted (as in l1b_score.py --opponent) and the meta
records side = "opponent"; the trainer carries it into the heads' sidecar.
"""
import argparse  # 解析命令列參數
import hashlib  # 計算選局清單的雜湊
import json  # 讀寫 meta
import multiprocessing as mp  # CPU 抽取局面的行程池
import sys  # 設定匯入路徑與結束碼
import time  # 計算耗時
from pathlib import Path  # 處理路徑

import numpy as np  # 打亂棋局順序
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.featcache import completed_parts, pack_positions, part_paths, plan_parts, write_part  # noqa: E402  快取分段的規劃、位元壓縮與寫入
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402  分折表、執行環境紀錄與 JSON 輸出
from aicup.metric import LEVELS  # noqa: E402  10 個等級
from aicup.scoring import extract, fold_players, init_worker, load_games, player_games  # noqa: E402  與評分共用的選局與局面抽取

SHUFFLE_SEED = 42  # 打亂棋局順序的種子
DEFAULT_MODEL_CONFIG = "b18c384nbt"  # humanv0 的架構名稱：輸入特徵相同，不需要檢查點


def select_games(folds, max_games, min_games, folds_csv=FOLDS_CSV):  # 選出要建快取的棋局
    """DataFrame (game_key, player_id, rank) of the chosen games of every player of `folds`, shuffled with
    RandomState(SHUFFLE_SEED) after a (rank, player_id) sort; and the number of players."""
    games, folds_df = player_games(), pd.read_csv(folds_csv)  # 訓練集所有棋局與分折表
    players = pd.concat([fold_players(games, folds_df, f, max_games=max_games, min_games=min_games) for f in sorted(folds)],  # 各折依固定排列選局
                        ignore_index=True).sort_values(["rank", "player_id"], kind="stable")  # 依等級與玩家排序
    rows = [{"game_key": k, "player_id": p.player_id, "rank": p.rank} for p in players.itertuples() for k in p.game_keys]  # 展開成每局一列
    table = pd.DataFrame(rows, columns=["game_key", "player_id", "rank"])  # 選局表
    perm = np.random.RandomState(SHUFFLE_SEED).permutation(len(table))  # 固定種子的打亂順序
    return table.iloc[perm].reset_index(drop=True), int(len(players))  # 打亂後的選局表與玩家人數


def games_sha256(table):  # 選局表的雜湊，用來核對各分片的選局一致
    h = hashlib.sha256()  # 建立雜湊物件
    for r in table.itertuples():  # 依表中順序
        h.update(f"{r.game_key}:{r.player_id}:{r.rank}\n".encode())  # 加入每一局的鍵、玩家與等級
    return h.hexdigest()  # 回傳十六進位雜湊


def other_colour(c):  # 對手的顏色
    return "W" if c == "B" else "B"  # 黑白互換


def build_parts(table, out, shard, nshards, part_size, workers, config, log=print, opponent=False):  # 寫出本分片的快取分段
    """Write the unfinished parts of this shard; returns the number of parts written. opponent: extract the positions before
    the other colour's moves instead of the target player's (the table's color column is the side extracted)."""
    mine = table.iloc[shard::nshards].reset_index(drop=True)  # 本分片負責的棋局（keys[i::N]）
    keys = mine["game_key"].tolist()  # 棋局鍵清單
    info = {r.game_key: (r.player_id, r.rank) for r in mine.itertuples()}  # 棋局鍵 → (玩家, 等級)
    todo = plan_parts(keys, part_size, out, shard)  # 尚未完成的分段（中斷後可續跑）
    n_parts = -(-len(keys) // part_size)  # 分段總數（無條件進位）
    log(f"shard {shard} of {nshards}: {len(keys)} games in {n_parts} parts, {len(todo)} to do", flush=True)  # 回報本分片的工作量
    if not todo:  # 全部已完成
        log("nothing to do", flush=True)  # 回報無事可做
        return 0  # 沒有寫出任何分段
    sgfs = load_games([k for _, ks in todo for k in ks])  # 讀入待處理棋局的 SGF 與顏色
    ctx = mp.get_context("fork")  # 以 fork 建立子行程
    with ctx.Pool(workers, initializer=init_worker, initargs=(config,)) as pool:  # 每個子行程各自建立 KataGo 特徵物件
        for k, part_keys in todo:  # 逐一處理每個分段
            t0 = time.time()  # 開始計時
            side = {key: (other_colour(sgfs[key][1]) if opponent else sgfs[key][1]) for key in part_keys}  # 要抽取的顏色：對手模式取另一色
            tasks = [(i, sgfs[key][0], side[key]) for i, key in enumerate(part_keys)]  # 每局的 (序號, SGF, 顏色)
            got = dict(pool.imap_unordered(extract, tasks))  # 平行抽取每局的局面
            parts = [pack_positions(got[i]) for i in range(len(part_keys))]  # 依原順序做位元壓縮
            games = pd.DataFrame({"game_key": part_keys, "player_id": [info[k_][0] for k_ in part_keys],  # 這個分段的棋局表：鍵與玩家
                                  "rank": [info[k_][1] for k_ in part_keys], "color": [side[k_] for k_ in part_keys],  # 等級與被抽取的顏色
                                  "n_moves": [len(p["target"]) for p in parts]})  # 每局的局面數
            write_part(out, shard, k, games, parts)  # 寫出這個分段（先寫暫存檔再改名）
            log(f"part {k}: {len(part_keys)} games, {int(games['n_moves'].sum())} positions, {time.time() - t0:.0f} s", flush=True)  # 回報這個分段的規模與耗時
    return len(todo)  # 回傳寫出的分段數


def shard_meta(out, shard, table, folds, max_games, min_games, nshards, n_players, opponent=False):  # 本分片的 meta
    """Counts of this shard's finished parts from their games tables (no npz is read); side = the colour extracted."""
    frames = [pd.read_csv(part_paths(out, s, k)[1], dtype=str, keep_default_na=False) for s, k in completed_parts(out) if s == shard]  # 讀入本分片已完成分段的棋局表
    games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_key", "rank", "n_moves"])  # 合併；沒有分段時用空表
    n_moves = games["n_moves"].astype(int) if len(games) else pd.Series(dtype=int)  # 每局的局面數
    per_rank = {lv: int(n_moves[games["rank"] == lv].sum()) if len(games) else 0 for lv in LEVELS}  # 各等級的局面數
    return {"folds": [int(f) for f in sorted(folds)], "max_games": int(max_games), "min_games": int(min_games), "nshards": int(nshards),  # 選局設定
            "shards": [int(shard)], "players": int(n_players), "games_selected": int(len(table)), "games": int(len(games)),  # 分片、玩家與棋局數
            "positions": int(n_moves.sum()), "per_rank_positions": per_rank, "games_sha256": games_sha256(table),  # 局面數與選局雜湊
            "side": "opponent" if opponent else "target", "model_input_config": None, **environment("cpu")}  # 抽取端、輸入設定（由呼叫端補上）與執行環境


def merge_meta(out):  # 合併各分片的 meta
    """meta.json from every meta_s*.json of `out`; ValueError when they disagree on the selection."""
    metas = [json.loads(p.read_text()) for p in sorted(Path(out).glob("meta_s*.json"))]  # 讀入每個分片的 meta
    if not metas:  # 一個都沒有
        raise FileNotFoundError(f"no meta_s*.json in {out}")  # 無法合併
    for key in ("folds", "max_games", "min_games", "nshards", "players", "games_selected", "games_sha256"):  # 各分片必須相同的選局設定
        if len({json.dumps(m[key]) for m in metas}) != 1:  # 有分片不一致
            raise ValueError(f"the shard metas disagree on {key}")  # 中止，避免混到不同的選局
    if len({m.get("side", "target") for m in metas}) != 1:  # 抽取端不一致
        raise ValueError("the shard metas disagree on side (target / opponent)")  # 中止
    merged = {**metas[0], "shards": sorted(s for m in metas for s in m["shards"]), "games": sum(m["games"] for m in metas),  # 以第一個分片為底，合併分片清單與局數
              "positions": sum(m["positions"] for m in metas),  # 局面總數
              "per_rank_positions": {lv: sum(m["per_rank_positions"][lv] for m in metas) for lv in LEVELS}}  # 各等級的局面總數
    if len(merged["shards"]) != merged["nshards"]:  # 有分片的 meta 還沒寫出
        print(f"WARNING: {len(merged['shards'])} of {merged['nshards']} shard metas present", file=sys.stderr)  # 警告但仍寫出
    (Path(out) / "meta.json").write_text(to_json(merged, indent=1) + "\n")  # 寫出合併的 meta.json
    return merged  # 回傳合併結果


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--folds", type=int, nargs="*", default=[])  # 要建快取的折
    ap.add_argument("--out", required=True)  # 快取資料夾
    ap.add_argument("--max-games", type=int, default=40)  # 每位玩家最多取幾局
    ap.add_argument("--min-games", type=int, default=1)  # 局數下限（預設全收）
    ap.add_argument("--workers", type=int, default=12)  # CPU 行程數
    ap.add_argument("--part-size", type=int, default=200)  # 每個分段的局數
    ap.add_argument("--shard", type=int, default=None)  # 本行程的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="KataGo config name (katago.train.modelconfigs) for the input features")  # 決定輸入特徵的 KataGo 設定
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--merge-meta", action="store_true", help="only combine <out>/meta_s*.json into meta.json")  # 只合併各分片的 meta
    ap.add_argument("--opponent", action="store_true", help="the positions before the opponent's moves (the other colour), labelled with the target player's rank")  # 改抽對手落子前的局面
    a = ap.parse_args(argv)  # 解析參數
    out = Path(a.out)  # 快取資料夾
    if a.merge_meta:  # 只做合併
        m = merge_meta(out)  # 合併各分片的 meta
        print(f"wrote {out / 'meta.json'}: {m['games']} games, {m['positions']} positions, shards {m['shards']}")  # 回報合併結果
        return 0  # 結束
    if not a.folds:  # 沒指定折
        ap.error("--folds is required (unless --merge-meta)")  # 必須指定折
    if a.nshards > 1 and a.shard is None:  # 多分片卻沒指定編號
        ap.error("--nshards > 1 needs --shard i")  # 必須指定分片編號
    shard = a.shard or 0  # 單一行程時為分片 0
    from katago.train import modelconfigs  # KataGo 的模型設定表
    config = modelconfigs.config_of_name[a.model_config]  # 輸入特徵的設定
    table, n_players = select_games(a.folds, a.max_games, a.min_games, a.folds_csv)  # 選局並打亂
    print(f"{n_players} players, {len(table)} games over folds {sorted(a.folds)} (<= {a.max_games} per player)"  # 回報玩家與棋局數
          + ("; the OPPONENT's positions, labelled with the target player's rank" if a.opponent else ""), flush=True)  # 對手模式時另外標明
    out.mkdir(parents=True, exist_ok=True)  # 建立快取資料夾
    build_parts(table, out, shard, a.nshards, a.part_size, a.workers, config, opponent=a.opponent)  # 寫出本分片的分段
    meta = {**shard_meta(out, shard, table, a.folds, a.max_games, a.min_games, a.nshards, n_players, opponent=a.opponent), "model_input_config": a.model_config}  # 本分片的 meta，補上輸入設定
    (out / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")  # 寫出本分片的 meta
    if a.nshards == 1:  # 單一行程時
        merged = merge_meta(out)  # 直接合併成 meta.json
        print(f"wrote {out / 'meta.json'}: {merged['games']} games, {merged['positions']} positions")  # 回報結果
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
