"""Section 4M M90 (2026-11-12): per-ply trunk vectors of the 595,379 training games that have none, in a compact form, so that the
sequence identity model (M88) can train on, and build prototypes from, all 1,000,000 games. Same network, projection and replay as
scripts/trunk_extract.py; kept per ply (t < 256): the 128 values quantised to int8 (round((x - mu) / sd * 20), clipped to +-127; mu, sd
per dimension from stored parts, results/m90/glstats.npz), the played point, the mover; per game: length, the owner's colour.
  --stats                               writes results/m90/glstats.npz from stored parts (CPU)
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I scripts/m_series/m90_ply_extract.py --job 0 --njobs 3 --workers 10
"""
import argparse  # 解析命令列參數
import multiprocessing as mp  # 平行重放棋譜
import os  # 以改名的方式換上寫完的檔案
import sys  # 設定匯入路徑
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取既有分段檔的局鍵
import torch  # 在 GPU 上跑 Human SL 主幹

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案 src、scripts、實驗腳本與競賽一（aicup_pid）的 src
import m81_tactic_id as m81  # noqa: E402  讀取全部訓練對局
import trunk_extract as te  # noqa: E402  沿用相同的網路、投影與重放
from aicup.humansl import load_humansl  # noqa: E402  載入 Human SL 網路
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402  中性的中繼資料列與主幹特徵
STATS = ROOT / "results/m90/glstats.npz"; OUTS = {0: ROOT / "results/m90_ply0", 1: ROOT / "results/m90_ply1", 2: ROOT / "results/m90_ply2"}  # 量化用的統計量；三個工作各自的輸出資料夾
VEC = [ROOT / "results/trunk_vec", ROOT / "results/trunk_vec_x40", ROOT / "results/trunk_vec_x80"]; T_MAX, Q = 256, 20.0  # 已有逐手向量的資料夾；每局保留 256 手、量化倍率 20


def safe_replay(task):  # 重放一局，失敗時不中斷整批
    try:  # 嘗試重放
        return te.replay(task)  # 每手的輸入特徵
    except Exception as e:  # 重放無法處理的棋譜：略過
        return (task[0], None, repr(e)[:80])  # 回傳局鍵與錯誤摘要


def main():  # 補算逐手向量，或只計算量化統計量
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", action="store_true"); ap.add_argument("--job", type=int, default=0); ap.add_argument("--njobs", type=int, default=3); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)  # 統計量模式；工作與分片的切分
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--part", type=int, default=200); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--sub", type=int, default=0); ap.add_argument("--nsub", type=int, default=1, help="repair runs: this copy takes the parts k with k %% nsub == sub"); a = ap.parse_args(); t0 = time.time()  # 行程數、每檔局數、批次大小、修補用的子切分
    if a.stats:  # --stats：只計算量化用的統計量
        s1 = np.zeros(128); s2 = np.zeros(128); n = 0  # 一次和、二次和與手數
        for f in sorted((VEC[0] / "fold1").glob("moves_s*_p*.npz"))[::3]:  # 折 1 的逐手向量檔，每 3 個取 1 個
            with np.load(f) as z:  # 開啟向量檔
                X = np.hstack([z["g"], z["l"]]).astype(np.float64)  # 每手 128 維（盤面 64＋落點 64）
            s1 += X.sum(0); s2 += (X * X).sum(0); n += len(X)  # 累加
        mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu * mu, 1e-12)); STATS.parent.mkdir(parents=True, exist_ok=True); np.savez(STATS, mu=mu.astype(np.float32), sd=sd.astype(np.float32)); print(f"stats from {n} plies written", flush=True); return  # 各維的平均與標準差寫入 glstats.npz 後結束
    st = np.load(STATS); mu, sd = st["mu"], st["sd"]; out_dir = OUTS[a.job]; out_dir.mkdir(parents=True, exist_ok=True); rows = m81.load_rows(); have = set()  # 量化統計量、本工作的輸出資料夾、全部訓練對局
    for d in VEC:  # 已有逐手向量的資料夾
        for p in d.glob("fold*/games_s*_p*.csv"):  # 逐一分段檔
            have.update(pd.read_csv(p, usecols=["game_key"], dtype=str)["game_key"])  # 記下已有向量的局鍵
    todo = np.array(sorted(set(rows.game_key) - have)); mine = todo[a.job::a.njobs][a.shard::a.nshards]; sgf = dict(zip(rows.game_key, zip(rows.sgf_content, rows.color))); del rows  # 待補算的對局；依工作與分片切出本份；局鍵→（棋譜、顏色）
    print(f"{len(todo)} games without stored vectors; this shard: {len(mine)} ({time.time() - t0:.0f} s)", flush=True)  # 記錄待補算與本份的局數
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]; pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 以網路設定初始化重放用的行程池
    try:  # 確保最後會關閉行程池
        model, _ = load_humansl(str(te.CKPT), "cuda"); model.eval(); nrows = neutral_rows(); proj = None; plies = 0; skipped = 0; t1 = time.time()  # 載入 Human SL 主幹；投影延後建立
        for k, start in enumerate(range(0, len(mine), a.part)):  # 每 --part 局寫成一個分段檔
            path = out_dir / f"ply_j{a.job}of{a.njobs}_s{a.shard}of{a.nshards}_p{k:05d}.npz"  # 分段檔名含工作、分片與序號
            if path.exists() or k % a.nsub != a.sub:  # 已經完成，或屬於其他修補副本
                continue  # 略過
            keys = mine[start:start + a.part]; res = list(pool.map(safe_replay, [(q, *sgf[q]) for q in keys], chunksize=2)); done = [d for d in res if d[1] is not None and len(d[3]) > 0]; skipped += len(res) - len(done)  # 平行重放；剔除失敗與沒有著手的棋譜
            if not done:  # 這一段沒有任何可用的對局：不寫檔
                continue  # 略過
            if proj is None:  # 第一次需要時才建立投影
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不需梯度，以 bf16 計算
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)  # 跑一個局面以取得主幹的通道數
                proj = te.projections(trunk.shape[1], trunk.device)  # 與 trunk_extract.py 相同的固定隨機投影
            lens = np.array([len(d[3]) for d in done]); bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done]); pos = np.concatenate([d[3] for d in done]); mover = np.concatenate([d[4] for d in done])  # 每局手數與串接後的網路輸入
            g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch); t = np.concatenate([np.arange(n) for n in lens]); keep = t < T_MAX  # 每手兩個 64 維向量；只保留前 256 手
            q = np.clip(np.rint((np.hstack([g, l])[keep].astype(np.float32) - mu) / sd * Q), -127, 127).astype(np.int8); subj = np.array([int(d[4][d[5]][0]) if d[5].any() else 0 for d in done], np.int8)  # 標準化後乘 20 量化成 int8（截到 ±127）；主人的顏色
            tmp = path.with_name(path.name + ".tmp.npz"); np.savez(tmp, key=np.array([d[0] for d in done]), LEN=np.minimum(lens, T_MAX).astype(np.int16), SUBJ=subj, POS=pos[keep].astype(np.int16), MOV=mover[keep].astype(np.int8), GLQ=q); os.replace(tmp, path); plies += int(lens.sum())  # 先寫暫存檔再改名，避免留下不完整的分段檔
            if k % 20 == 0:  # 分段檔序號為 20 的倍數時記錄進度
                print(f"part {k + 1}/{(len(mine) + a.part - 1) // a.part}: {plies} plies, {plies / (time.time() - t1):.0f} plies/s, skipped {skipped}", flush=True)  # 手數、速度與略過的局數
        print(f"done: {len(mine)} games, {plies} plies, skipped {skipped}, {(time.time() - t0) / 60:.1f} min", flush=True)  # 本份完成的摘要
    finally:  # 不論成功或失敗
        pool.close(); pool.join()  # 關閉行程池


if __name__ == "__main__":  # 直接執行時
    main()  # 執行補算
