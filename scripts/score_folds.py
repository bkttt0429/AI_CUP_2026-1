#!/usr/bin/env python
"""Score every eligible player of the given folds under the 18 Human SL profiles.

Each player with >= 5 games contributes up to 20 games (scoring.fold_players). Work is
written in parts of --part-size games:
  <out>/fold{F}/games_s{shard}_p{k:03d}.csv   game_key, player_id, rank, color, n_moves, sum_<profile> x 18
  <out>/fold{F}/moves_s{shard}_p{k:03d}.npz   logp [moves, 18], game_row (row in the part CSV), t
Both files are written to temporary names and renamed, npz first, so a part counts as done
only when both exist.

Breakpoints:
  * resume is decided per game (scoring.resume_plan): rerunning the same command scores only
    games that are in no finished part, even if --part-size changed;
  * pause cleanly with `touch <out>/PAUSE` (or SIGTERM / Ctrl-C to the main process): the run
    stops after the part in progress, i.e. within about a minute at the default part size;
    a second signal stops at once and only the unfinished part is lost;
  * the run refuses to start while the pause file exists: delete it to resume.
The CPU Pool is created before CUDA initialises (fork).
"""
import argparse  # 解析命令列參數
import json  # 寫出完成標記檔
import multiprocessing as mp  # 計算特徵的 CPU 程序池
import os  # 讀環境變數、原子改名與送出訊號
import signal  # 攔截 SIGTERM／SIGINT 以便乾淨暫停
import sys  # 設定匯入路徑
import time  # 計時
from datetime import datetime  # log 與完成標記的時間戳
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 寫出逐手的 npz 陣列
import pandas as pd  # 讀分折表、寫出每局 CSV

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（本檔在 scripts/）
if str(ROOT / "src") not in sys.path:  # src 還不在匯入路徑上時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src，才能匯入 aicup 套件
from aicup.scoring import (fold_players, init_worker, load_games, part_paths, pause_requested,  # noqa: E402  評分流程共用的函式
                           player_games, read_game_keys, restrict_players, resume_plan, score_stream)  # 匯入清單的後半

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"  # Human SL 網路（KataGo humanv0）的檢查點
STOP = {"signal": None}  # 收到的停止訊號；訊號處理函式與主迴圈共用


def log(msg):  # 印出帶時間戳的 log
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)  # 立即輸出，讓 Slurm 的 log 即時更新


def _on_signal(signum, _frame):  # SIGTERM／SIGINT 的處理函式
    if STOP["signal"] is not None:  # 第二次收到訊號：立刻停止
        signal.signal(signum, signal.SIG_DFL)  # 還原預設的處理方式
        os.kill(os.getpid(), signum)  # 再對自己送一次同一個訊號，立即結束
    STOP["signal"] = signum  # 記下訊號，主迴圈在進行中的分段完成後停下
    log(f"signal {signum}: stopping after the part in progress (send again to stop now)")  # 告知何時會停


def main():  # 命令列進入點
    ap = argparse.ArgumentParser()  # 命令列參數
    ap.add_argument("--folds", type=int, nargs="+", required=True)  # 要評分的折
    ap.add_argument("--shard", type=int, default=0)  # 本程序負責的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數（shards.sh 會自動加上）
    ap.add_argument("--workers", type=int, default=10)  # 計算特徵的 CPU worker 數
    ap.add_argument("--part-size", type=int, default=100)  # 每個分段檔的局數
    ap.add_argument("--limit-games", type=int, default=0)  # 每折只評前幾局（0 表示不限；冒煙測試用）
    ap.add_argument("--max-games", type=int, default=20, help="scoring.fold_players' cap per player (section 4M M8: 40 with --game-keys of positions 21-40)")  # 每位玩家最多取幾局
    ap.add_argument("--cudnn-benchmark", action="store_true")  # 是否開啟 cuDNN benchmark
    ap.add_argument("--opponent", action="store_true", help="section 4M M14: score the OPPONENT's moves (colour flipped) under the same 18 profiles; the color column stays the target's")  # 改評對手的手（section 4M M14）
    ap.add_argument("--profiles", nargs="+", default=None, metavar="RANK",  # 改用其他 humanv0 剖面（section 4M M16）
                    help="section 4M M16: score under these humanv0 profiles instead of the 18 (12k-6d), e.g. 7d 8d 9d; the sums columns are sum_<rank>")  # 說明文字
    ap.add_argument("--out", default=str(ROOT / "results" / "scores"))  # 輸出目錄
    ap.add_argument("--pause-file", default=None, help="default: <out>/PAUSE")  # 暫停檔的位置
    ap.add_argument("--game-keys", default=None, help="score only these games of each fold's selection (a CSV with a game_key column or one key "  # 只評清單上的局（例如 issue #42 的讓子局重評）
                    "per line), e.g. the HD >= 2 games again into a patch directory that scripts/patch_scored.py merges (issue #42)")  # 說明文字的後半
    a = ap.parse_args()  # 解析參數
    pause_file = Path(a.pause_file) if a.pause_file else Path(a.out) / "PAUSE"  # 決定暫停檔的路徑
    if pause_requested(pause_file):  # 暫停檔已經存在
        log(f"pause file {pause_file} exists; delete it to run")  # 提示先刪除暫停檔
        return  # 不啟動

    all_games = player_games()  # 訓練集所有的（玩家、等級、局）列
    folds = pd.read_csv(os.environ.get("AICUP_FOLDS_CSV") or ROOT / "splits" / "player_folds.csv")  # 玩家分折表；section 4M M135：AICUP_FOLDS_CSV 與 AICUP_TRAIN_DIR 搭配使用

    import torch  # 讀檢查點設定與在 GPU 上評分
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]  # 只取用檢查點中的模型設定，給 worker 建特徵器（網路稍後才載入 GPU）
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))  # 在初始化 CUDA 之前 fork 出 CPU 程序池
    for sig in (signal.SIGTERM, signal.SIGINT):  # 在 fork 之後才設定：worker 保留預設的處理方式
        signal.signal(sig, _on_signal)  # 主程序收到訊號時改為乾淨暫停
    from aicup.humansl import PROFILES as DEFAULT_PROFILES, inverse_rank, load_humansl, meta_rows  # Human SL 的剖面與載入函式
    PROFILES = list(a.profiles) if a.profiles else list(DEFAULT_PROFILES)  # 本次使用的剖面：--profiles 或預設的 18 個
    for p_ in PROFILES:  # 逐一檢查剖面名稱
        inverse_rank(p_)  # humanv0 不認得的等級會丟出 ValueError
    model, _ = load_humansl(str(CKPT), "cuda")  # 在 GPU 上載入 Human SL 網路
    torch.backends.cudnn.benchmark = a.cudnn_benchmark  # 依參數開關 cuDNN benchmark
    rows = meta_rows([(p, p) for p in PROFILES])  # 每個剖面的 metadata 輸入列（本人與對手設為同一等級）
    log(f"shard {a.shard}/{a.nshards} on {torch.cuda.get_device_name(0)}; folds {a.folds}; part size {a.part_size}; "  # 回報分片、GPU、折與分段大小
        f"pause file {pause_file}; {len(PROFILES)} profiles {PROFILES[0]}..{PROFILES[-1]}" + ("; the OPPONENT's moves" if a.opponent else ""))  # 以及暫停檔、剖面範圍與是否評對手

    def stop_now():  # 是否該停下：收到訊號或出現暫停檔
        return STOP["signal"] is not None or pause_requested(pause_file)  # 任一條件成立就停

    only = read_game_keys(a.game_keys) if a.game_keys else None  # --game-keys 指定的局（None 表示全部）
    for fold in a.folds:  # 逐折評分
        players = fold_players(all_games, folds, fold, max_games=a.max_games)  # 這一折的合格玩家與各自選出的局
        if only is not None:  # 有指定局清單時
            players = restrict_players(players, only)  # 只保留清單上的局
        games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][a.shard::a.nshards]  # 本分片負責的局（每 nshards 局取一局）
        if a.limit_games:  # 有限制局數時
            games = games[:a.limit_games]  # 只取前幾局
        meta = {g[2]: g for g in games}  # game_key → (等級, 玩家, 鍵)
        out = Path(a.out) / f"fold{fold}"  # 這一折的輸出資料夾
        out.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        plan = resume_plan([g[2] for g in games], out, a.shard, a.part_size)  # 只排還沒評過的局，並分成各個分段
        pending = sum(len(keys) for _, keys in plan)  # 待評的局數
        log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(games) - pending} already done, "  # 回報玩家數、本分片局數與已完成局數
            f"{pending} to score in {len(plan)} parts")  # 以及待評局數與分段數
        if plan:  # 還有待評的局
            data = load_games([k for _, keys in plan for k in keys])  # 讀入待評各局的棋譜與本人顏色
            t_fold, pos_fold = time.time(), 0  # 這一折的計時與局面數
            for j, (k, keys) in enumerate(plan):  # 逐個分段評分
                if stop_now():  # 收到暫停要求
                    log(f"paused before fold {fold} part index {k} ({len(plan) - j} parts left in this fold)")  # 回報停在哪裡、還剩幾個分段
                    pool.close()  # 關閉程序池
                    return  # 結束；重跑同一個指令即可續跑
                sub = [meta[key] for key in keys]  # 這個分段各局的（等級, 玩家, 鍵）
                tasks = [(i, data[key][0], ("W" if data[key][1] == "B" else "B") if a.opponent else data[key][1]) for i, key in enumerate(keys)]  # 評分任務；--opponent 時翻色改評對手的手
                per_task, (logp, task_idx, ts), n_pos, wall = score_stream(pool, model, tasks, rows, log_every=10**9)  # 在 GPU 上評分（不印中間進度），取得每局總和與逐手 log p
                recs = [{"game_key": key, "player_id": pid, "rank": rank, "color": data[key][1], "n_moves": per_task[i][0],  # 每局一列：鍵、玩家、等級、本人執色與評分的手數
                         **{f"sum_{p}": float(s) for p, s in zip(PROFILES, per_task[i][1])}}  # 以及每個剖面的 log p 總和
                        for i, (rank, pid, key) in enumerate(sub)]  # 依分段內的順序
                csv_path, npz_path = part_paths(out, a.shard, k)  # 這個分段的正式檔名
                tmp_csv, tmp_npz = csv_path.with_suffix(".tmp.csv"), npz_path.with_name(npz_path.stem + ".tmp.npz")  # 暫存檔名
                np.savez_compressed(tmp_npz, logp=logp, game_row=task_idx, t=ts)  # 先寫逐手陣列
                pd.DataFrame(recs).to_csv(tmp_csv, index=False)  # 再寫每局表格
                os.replace(tmp_npz, npz_path)  # npz 先改名
                os.replace(tmp_csv, csv_path)  # CSV 後改名：兩個都在，這個分段才算完成
                pos_fold += n_pos  # 累計局面數
                log(f"  fold {fold} part {j + 1}/{len(plan)} (index {k}): {len(sub)} games, {n_pos} positions, {n_pos / wall:.0f} positions/s")  # 回報進度與速度
            log(f"fold {fold}: scored {pos_fold} positions in {(time.time() - t_fold) / 60:.1f} min")  # 回報這一折的局面總數與耗時
        (out / f"done_s{a.shard}").write_text(json.dumps({  # 寫出本分片這一折的完成標記
            "fold": fold, "shard": a.shard, "nshards": a.nshards, "games": len(games), "part_size": a.part_size,  # 折、分片、局數與分段大小
            "cudnn_benchmark": a.cudnn_benchmark, "finished": datetime.now().isoformat(timespec="seconds")}))  # cuDNN 設定與完成時間
        log(f"fold {fold} done")  # 回報這一折完成
    pool.close()  # 全部完成後關閉程序池


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
