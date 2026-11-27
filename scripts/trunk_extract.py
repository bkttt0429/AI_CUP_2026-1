#!/usr/bin/env python
"""Per-ply trunk vectors of the scored games: the pristine humanv0 trunk, neutral metadata, both players' plies.

Each ply stores the board-pooled and the played-point trunk vector, each through a fixed random projection to 64 dimensions
(fp16), taken from the position before the move; a pass has no point, so its played-point vector is zero. The games are the
selection of scripts/score_folds.py (up to 20 per player), replayed with the handicap fix at whiteKomi 7.5, so the plies line
up with the per-move scores by (game_key, t). Parts resume per game, as the scoring runs. Fold 0 is refused.

  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 1 2 3 4 --out results/trunk_vec --workers 10
Section 4M M3 (2026-11-04): --trunk <stage-2 heads .pt> swaps in that run's fine-tuned trunk (l1b.load_heads, sha256-checked) with
the same projection; the out-of-fold rule of the L1b scores applies (folds 1, 2 with heads3_B, folds 3, 4 with heads3_A), and each
fold directory records its trunk in trunk.json and refuses another one:
  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 1 2 --trunk results/l1b/heads3_B.pt --out results/trunk_vec_ft
"""
import argparse  # 解析命令列參數
import json  # 讀寫 trunk.json 與側檔
import multiprocessing as mp  # 重播棋局的 CPU 程序池
import os  # 原子改名
import sys  # 設定匯入路徑
import time  # 計算速度與耗時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算與 npz 輸出
import pandas as pd  # 讀分折表、寫每局表格
import torch  # 在 GPU 上執行主幹

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（本檔在 scripts/）
for d in (ROOT / "src", ROOT / "scripts"):  # 把 src 與 scripts 都加入匯入路徑
    if str(d) not in sys.path:  # 還不在匯入路徑上時
        sys.path.insert(0, str(d))  # 加到最前面
from katago.game.board import Board  # noqa: E402  KataGo 的棋盤與顏色常數
from katago.game.gamestate import GameState  # noqa: E402  KataGo 的局面狀態與規則
from aicup.humansl import load_humansl  # noqa: E402  載入 Human SL 網路
from aicup.l1b import load_heads, neutral_rows, trunk_features  # noqa: E402  匯入 L1b 等級頭、中性 metadata 與主幹輸出
from aicup.ladder import FOLDS_CSV  # noqa: E402  預設的玩家分折表
from aicup.pilot import start_state  # noqa: E402  建立開局局面（含讓子石）
from aicup.scoring import fold_players, load_games, part_paths, player_games, read_game_keys, restrict_players, resume_plan  # noqa: E402  與評分相同的選局與分段續跑
from aicup.sgf import setup_and_moves  # noqa: E402  讓子修正後的重播方式

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"  # 原始 humanv0 檢查點
RULES = dict(GameState.RULES_CHINESE, whiteKomi=7.5)  # 中國規則，白貼 7.5
PROJ_SEED = 20261103  # 隨機投影的固定種子（測試端 trunk_predict.py 也用同一組）
PROJ_DIM = 64  # 投影後的維度
PASS_POS = 361  # pass 的位置編號（棋盤 361 個點之外）
_FEATURES = None  # 每個 worker 程序的 KataGo 特徵器
QUANT = None  # section 4M M135（--quant8）：(mu [128], sd [128])；此時 g 與 l 以 sd / 20 為單位存成 int8（鍵 gq、lq）


def trunk_info(heads_path):  # 決定 trunk.json 的內容
    """What a fold directory records about the trunk its vectors came from: the pristine checkpoint, or a stage-2 run's heads file
    with its fine-tuned trunk's sha256 (from the sidecar)."""
    if heads_path is None:  # 沒有 --trunk
        return {"trunk": "pristine", "ckpt": CKPT.name}  # 原始主幹
    hp = Path(heads_path)  # 等級頭檔案的路徑
    meta = json.loads(hp.with_suffix(".json").read_text())  # 讀入它的側檔
    if not meta.get("trunk_file"):  # 側檔沒有微調主幹
        raise SystemExit(f"{hp}: not a stage-2 run (no fine-tuned trunk in its sidecar)")  # 不是 stage-2 的訓練，拒絕
    return {"trunk": "fine-tuned", "heads": hp.name, "trunk_file": meta["trunk_file"], "trunk_sha256": meta["trunk_sha256"]}  # 微調主幹：等級頭檔名、主幹檔與 sha256


def check_trunk_meta(fold_dir, info):  # 確保一個資料夾只放同一個主幹的向量
    """trunk.json of a fold directory: written when missing, compared when present (SystemExit on a different trunk), so a
    directory never mixes vectors of two trunks. A directory with parts but no trunk.json is the pristine trunk's."""
    fold_dir = Path(fold_dir)  # 轉成 Path
    path = fold_dir / "trunk.json"  # 主幹紀錄檔
    if path.exists():  # 已有紀錄
        old = json.loads(path.read_text())  # 讀入既有紀錄
    elif any(fold_dir.glob("games_s*_p*.csv")):  # 已有分段檔卻沒有 trunk.json
        old = {"trunk": "pristine", "ckpt": CKPT.name}  # 依約定視為原始主幹的向量
    else:  # 全新的資料夾
        old = None  # 沒有既有紀錄
    if old is not None and old != info:  # 既有主幹與這次不同
        raise SystemExit(f"{fold_dir}: holds vectors of trunk {old}, not {info}")  # 拒絕混用
    if old is None or not path.exists():  # 還沒有 trunk.json
        path.write_text(json.dumps(info, indent=1))  # 寫入這次的主幹紀錄


def init_worker(config):  # 程序池的初始化函式
    global _FEATURES  # 設定本程序的全域特徵器
    from katago.game.features import Features  # 在 worker 內匯入
    _FEATURES = Features(config, 19)  # 每個 worker 建一次 19 路的特徵器


def replay(task):  # 在 worker 中重播一局
    """Inputs before every ply of one game, in play order."""
    idx, sgf, color = task  # 任務鍵、棋譜與本人顏色
    stones, moves = setup_and_moves(sgf)  # 讓子石與實際顏色的著手
    gs = start_state(stones, RULES)  # 開局局面（擺好讓子石）
    bins, globs, pos, mover, target = [], [], [], [], []  # 每一手的盤面特徵、全域特徵、落點、下子方與是否本人
    for c, x, y in moves:  # 依序走每一手
        pla = Board.BLACK if c == "B" else Board.WHITE  # 下子方
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)  # 落點（pass 為 PASS_LOC）
        b, g = gs.get_input_features(_FEATURES)  # 落子前局面的網路輸入
        bins.append(b[0].astype(np.uint8))  # 盤面特徵存成 uint8 以節省記憶體
        globs.append(g[0].astype(np.float32))  # 全域特徵
        pos.append(PASS_POS if x is None else _FEATURES.loc_to_tensor_pos(loc, gs.board))  # 落點在張量中的位置
        mover.append(0 if c == "B" else 1)  # 0 為黑、1 為白
        target.append(c == color)  # 是否為本人的手
        gs.play(pla, loc)  # 走這一手，前進到下一個局面
    return idx, np.stack(bins), np.stack(globs), np.asarray(pos, np.int16), np.asarray(mover, np.int8), np.asarray(target, bool)  # 回傳整局堆疊好的陣列


def projections(channels, device):  # 建立兩個固定的隨機投影矩陣
    gen = torch.Generator().manual_seed(PROJ_SEED)  # 固定種子，訓練端與測試端得到同一組投影
    scale = channels ** 0.5  # 以 √通道數縮放，保持投影後的尺度
    return (torch.randn(channels, PROJ_DIM, generator=gen) / scale).to(device), (torch.randn(channels, PROJ_DIM, generator=gen) / scale).to(device)  # 池化向量與落點向量各用一個


def vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, batch):  # 計算一疊局面的兩種投影向量
    """[n, 64] fp16 pooled and played-point vectors of a stack of positions."""
    device = proj_g.device  # 與投影矩陣在同一個裝置
    n = len(pos)  # 局面數
    out_g, out_l = np.empty((n, PROJ_DIM), np.float16), np.empty((n, PROJ_DIM), np.float16)  # 輸出陣列
    with torch.no_grad():  # 只做推論
        for i in range(0, n, batch):  # 分批送進網路
            j = min(n, i + batch)  # 這一批的結尾
            b, g = torch.as_tensor(bins[i:j], device=device), torch.as_tensor(globs[i:j], device=device)  # 盤面與全域特徵
            c = torch.as_tensor(mover[i:j], device=device).long()  # 下子方，用來選 metadata 列
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):  # GPU 上以 bf16 加速
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)  # 主幹最後一層的特徵圖
            trunk = trunk.float()  # 確保池化與投影以 fp32 計算（trunk_features 本身已回傳 float32）
            p = torch.as_tensor(pos[i:j], device=device).long()  # 各局面的落點
            on = p < PASS_POS  # 不是 pass 的局面
            local = torch.zeros(j - i, trunk.shape[1], device=device)  # 落點向量，pass 保持為零
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]  # 取出落點位置的通道向量
            out_g[i:j] = (trunk.mean(dim=(2, 3)) @ proj_g).cpu().numpy().astype(np.float16)  # 全盤平均後投影
            out_l[i:j] = (local @ proj_l).cpu().numpy().astype(np.float16)  # 落點向量投影
    return out_g, out_l  # 回傳兩種向量


def write_part(fold_dir, shard, k, done, meta, model, proj_g, proj_l, rows, batch):  # 寫出一個分段的向量與每局表格
    bins = np.concatenate([d[1] for d in done])  # 串接這個分段各局的盤面特徵
    globs = np.concatenate([d[2] for d in done])  # 全域特徵
    pos = np.concatenate([d[3] for d in done])  # 落點
    mover = np.concatenate([d[4] for d in done])  # 下子方
    target = np.concatenate([d[5] for d in done])  # 是否為本人的手
    lens = np.array([len(d[3]) for d in done])  # 每局的手數
    g, l = vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, batch)  # 計算兩種向量
    if not (np.isfinite(g).all() and np.isfinite(l).all()):  # 檢查有沒有 NaN／inf
        raise ValueError(f"part {k}: non-finite vector")  # 有就不寫檔
    game_row = np.repeat(np.arange(len(done), dtype=np.int32), lens)  # 每一手屬於表格的第幾列
    t = np.concatenate([np.arange(n, dtype=np.int16) for n in lens])  # 每一手在局內的序號
    csv_path, npz_path = part_paths(fold_dir, shard, k)  # 這個分段的正式檔名
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")  # 暫存的 npz 檔名
    if QUANT is not None:  # section 4M M135：以 mu 為中心、sd / 20 為單位存成 int8（讀取端 trunk_seq_check.part_gl 會還原 g、l）
        mu, sd = QUANT  # 量化用的平均與標準差
        q8 = lambda x, m, s: np.clip(np.rint((x.astype(np.float32) - m) / s * 20.0), -127, 127).astype(np.int8)  # noqa: E731  量化函式：四捨五入並截在 ±127
        np.savez(tmp, gq=q8(g, mu[:PROJ_DIM], sd[:PROJ_DIM]), lq=q8(l, mu[PROJ_DIM:], sd[PROJ_DIM:]), pos=pos, mover=mover, target=target, game_row=game_row, t=t)  # 前 64 維統計量給 g、後 64 維給 l
    else:  # 不量化
        np.savez(tmp, g=g, l=l, pos=pos, mover=mover, target=target, game_row=game_row, t=t)  # 直接存 fp16 向量
    os.replace(tmp, npz_path)  # npz 先改成正式檔名
    table = pd.DataFrame([{**meta[d[0]], "n_plies": int(n)} for d, n in zip(done, lens)])  # 每局一列：表格欄位加上手數
    table.to_csv(csv_path.with_name(csv_path.name + ".tmp.csv"), index=False)  # 先寫暫存的 CSV
    os.replace(csv_path.with_name(csv_path.name + ".tmp.csv"), csv_path)  # 再改名，這個分段才算完成
    return int(lens.sum())  # 回傳這個分段的手數


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明的第一行作為描述
    ap.add_argument("--folds", type=int, nargs="+", required=True)  # 要抽向量的折
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--workers", type=int, default=10)  # 重播棋局的 CPU worker 數
    ap.add_argument("--shard", type=int, default=0)  # 本程序的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--part-size", type=int, default=100)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=256)  # 送進網路的批次大小
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 玩家分折表
    ap.add_argument("--trunk", default=None, help="section 4M M3: a stage-2 heads .pt whose fine-tuned trunk replaces the pristine one")  # 改用微調主幹（section 4M M3）
    ap.add_argument("--max-games", type=int, default=20, help="scoring.fold_players' cap per player (section 4M M8: 40 with --game-keys)")  # 每位玩家最多幾局
    ap.add_argument("--game-keys", default=None, help="only these games of each fold's selection (a CSV with a game_key column)")  # 只處理清單上的局
    ap.add_argument("--allow-fold0", action="store_true", help="section 4M (fold 0 joins the submission's training players, the user's decision)")  # 允許處理 fold 0（fold 0 併入提交模型的訓練玩家時用）
    ap.add_argument("--quant8", default=None, help="section 4M M135: an npz with mu, sd [128] (results/m90/glstats.npz); g and l are stored as int8 in units of sd / 20, the statistics are copied to <fold dir>/quant8.npz")  # 以 int8 量化儲存（section 4M M135）
    a = ap.parse_args(argv)  # 解析參數
    if 0 in a.folds and not a.allow_fold0:  # 預設保護 fold 0
        ap.error("fold 0 is not used here (--allow-fold0 for the submission's training set)")  # 沒有 --allow-fold0 就拒絕
    only = read_game_keys(a.game_keys) if a.game_keys else None  # 指定的局清單（None 表示全部）
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]  # 檢查點的模型設定，給 worker 建特徵器
    all_games, folds_df = player_games(), pd.read_csv(a.folds_csv)  # 訓練集所有局與分折表
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))  # 在初始化 CUDA 之前 fork 出程序池
    try:  # 確保最後關閉程序池
        model, _ = load_humansl(str(CKPT), "cuda")  # 在 GPU 上載入 humanv0
        info = trunk_info(a.trunk)  # 本次使用的主幹紀錄
        if a.quant8:  # section 4M M135：指定了 int8 量化的統計量時
            global QUANT  # 改寫模組層級的量化設定
            st = np.load(a.quant8)  # 讀入 mu、sd 統計量
            QUANT = (st["mu"].astype(np.float32), st["sd"].astype(np.float32))  # 轉成 float32
            if QUANT[0].shape != (2 * PROJ_DIM,) or QUANT[1].shape != (2 * PROJ_DIM,) or not (QUANT[1] > 0).all():  # 檢查形狀與 sd > 0
                raise SystemExit(f"--quant8 {a.quant8}: mu and sd must be [{2 * PROJ_DIM}] with sd > 0")  # 統計量不合格就停止
            info = {**info, "quant8": "sd/20"}  # 量化方式也記入 trunk.json
        if a.trunk:  # 指定了微調主幹
            _, side, warnings = load_heads(a.trunk, model, "cuda")  # 把微調主幹原地載入 `model`（核對 sha256）
            for w in warnings:  # 逐一印出載入警告
                print(f"warning: {w}", flush=True)  # 警告內容
            print(f"trunk: fine-tuned, {Path(a.trunk).name} -> {side['trunk_file']} (sha256 {side['trunk_sha256'][:12]})", flush=True)  # 回報使用的主幹檔與 sha256
        model.eval()  # 推論模式
        proj_g, proj_l = None, None  # 投影矩陣在第一個分段時依通道數建立
        rows = neutral_rows()  # 中性的 metadata 列（黑、白各一）
        for fold in a.folds:  # 逐折處理
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=5)  # 與評分相同的選局
            if only is not None:  # 有指定局清單時
                players = restrict_players(players, only)  # 只保留清單上的局
            meta = {k: {"game_key": k, "player_id": r.player_id, "rank": r.rank} for r in players.itertuples() for k in r.game_keys}  # 每局的表格欄位
            keys = sorted(meta)[a.shard::a.nshards]  # 本分片負責的局
            fold_dir = Path(a.out) / f"fold{fold}"  # 這一折的輸出資料夾
            fold_dir.mkdir(parents=True, exist_ok=True)  # 建立資料夾
            check_trunk_meta(fold_dir, info)  # 確認沒有混用不同的主幹
            if QUANT is not None and not (fold_dir / "quant8.npz").exists():  # 量化時要留下統計量給讀取端
                np.savez(fold_dir / f"quant8.tmp{a.shard}.npz", mu=QUANT[0], sd=QUANT[1])  # 每個分片用自己的暫存檔名
                os.replace(fold_dir / f"quant8.tmp{a.shard}.npz", fold_dir / "quant8.npz")  # 改名成 quant8.npz
            plan = resume_plan(keys, fold_dir, a.shard, a.part_size)  # 只排還沒完成的局
            t0, plies = time.time(), 0  # 計時與手數
            for n_part, (k, part_keys) in enumerate(plan):  # 逐個分段處理
                sgf = load_games(part_keys)  # 讀入這個分段的棋譜與本人顏色
                tasks = [(q, *sgf[q]) for q in part_keys]  # 重播任務：(鍵, 棋譜, 顏色)
                for q in part_keys:  # 逐局
                    meta[q]["color"] = sgf[q][1]  # 補上本人執色
                done = list(pool.imap(replay, tasks, chunksize=1))  # 以 CPU 程序池重播，保持順序
                if proj_g is None:  # 第一次才建立投影
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 推論、bf16
                        trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(),  # 用一個局面取得主幹的通道數
                                                        torch.as_tensor(done[0][4][:1]).cuda().long(), rows)  # 下子方與 metadata 列
                    proj_g, proj_l = projections(trunk.shape[1], trunk.device)  # 依通道數建立固定的投影
                plies += write_part(fold_dir, a.shard, k, done, meta, model, proj_g, proj_l, rows, a.batch)  # 計算並寫出這個分段
                print(f"fold {fold} part {n_part + 1}/{len(plan)}: {plies} plies, {plies / (time.time() - t0):.0f} plies/s", flush=True)  # 回報進度與速度
            print(f"fold {fold} done: {len(keys)} games in this shard, {plies} plies this run, {(time.time() - t0) / 60:.1f} min", flush=True)  # 回報這一折的局數、手數與耗時
    finally:  # 無論成功或失敗
        pool.close()  # 不再接受任務
        pool.join()  # 等待 worker 結束
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
