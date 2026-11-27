"""Section 4M M84 (2026-11-11): the per-game-side trunk rows (mean and standard deviation of the 64 + 64 per-ply trunk vectors, as
m82_cond_id.py's rows stage) of every training row that has NO stored trunk vectors yet (results/trunk_vec, _x40, _x80 hold about 405k of
the 1,000,000 games). Same network, projection, replay and fp16 rounding as scripts/trunk_extract.py; the per-ply vectors are not kept
(disk), only the two 256-dim rows per game. Resumes per part.
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I scripts/m_series/m84_rows_extract.py --job 0 --njobs 3 --workers 10
"""
import argparse  # 解析命令列參數
import multiprocessing as mp  # 平行重放棋譜
import os  # 以 os.replace 完成寫檔
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取已有向量的棋局表
import torch  # 主幹網路推論

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 加入本專案、scripts、同目錄與競賽一的程式
import m81_tactic_id as m81  # noqa: E402  訓練列的載入
import trunk_extract as te  # noqa: E402  與 scripts/trunk_extract.py 相同的重放、投影與向量計算
from aicup.humansl import load_humansl  # noqa: E402  載入 KataGo Human SL 網路（humanv0）
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402  中性的後設資料列與主幹特徵

OUT = ROOT / "results/m84_rowsp"  # 第二次執行：另外存各階段的平均（第一次執行的部分檔案留在 results/m84_rows，未使用）
VEC = [ROOT / "results/trunk_vec", ROOT / "results/trunk_vec_x40", ROOT / "results/trunk_vec_x80"]  # 已存有主幹向量的三批對局


def safe_replay(task):  # 重放失敗時不中斷整批
    try:  # 嘗試重放
        return te.replay(task)  # 回傳每手的輸入
    except Exception as e:  # 重放無法處理的棋譜：這局不產生列
        return (task[0], None, repr(e)[:80])  # 回傳棋局鍵與錯誤摘要


def side_rows(X, row, target, t, n_g):  # 每局雙方的列與分階段平均（AI_CUP_2026-2 的 experiments/pid_v2_phase_rows.py 也匯入此函式）
    """Per game and side (0 = the target's plies, 1 = the opponent's): the 256-dim row (mean, std over the side's plies), the ply
    count, and the phase means [384] = the mean vector over the side's plies with t < 60, 60 <= t < 150, t >= 150 (zeros if none)."""
    out = np.zeros((2, n_g, 256), np.float32); ph = np.zeros((2, n_g, 384), np.float16); cnt = np.zeros((2, n_g), np.int32)  # 列、階段平均與手數
    for side, m in enumerate((target, ~target)):  # 0 = 本人、1 = 對手
        r = row[m]; order = np.argsort(r, kind="stable"); xs = X[m][order]; n = np.bincount(r, minlength=n_g); nz = n > 0  # 依棋局排序；每局手數
        if not nz.any():  # 這一方沒有任何手
            continue  # 略過
        starts = (np.cumsum(n) - n)[nz]; s1 = np.add.reduceat(xs, starts, axis=0); s2 = np.add.reduceat(xs * xs, starts, axis=0); mean = s1 / n[nz][:, None]  # 每局的一次和、平方和與平均
        out[side][nz] = np.hstack([mean, np.sqrt(np.maximum(s2 / n[nz][:, None] - mean * mean, 0.0))]); cnt[side] = n  # 列＝平均接標準差，共 256 維
        for k, pm in enumerate((t < 60, (t >= 60) & (t < 150), t >= 150)):  # 三個階段：手數索引 t（0 起算）< 60、60-149、>= 150
            mm = m & pm; r = row[mm]; n = np.bincount(r, minlength=n_g); nz = n > 0  # 這一方在本階段的手
            if nz.any():  # 有手才計算
                order = np.argsort(r, kind="stable"); ph[side][nz, 128 * k:128 * (k + 1)] = np.add.reduceat(X[mm][order], (np.cumsum(n) - n)[nz], axis=0) / n[nz][:, None]  # 本階段的平均向量（128 維）
    return out, ph, cnt  # 回傳列、階段平均與手數


def main():  # 主流程
    ap = argparse.ArgumentParser(); ap.add_argument("--job", type=int, default=0); ap.add_argument("--njobs", type=int, default=1); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)  # 工作與分片的切分
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--part", type=int, default=200); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()  # 工作行程數、每段局數、批次大小、局數上限
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rows = m81.load_rows(); have = set()  # 建立輸出資料夾並讀入訓練列
    for d in VEC:  # 三批已有向量的資料夾
        for p in d.glob("fold*/games_s*_p*.csv"):  # 各分段檔的棋局表
            have.update(pd.read_csv(p, usecols=["game_key"], dtype=str)["game_key"])  # 記下已有向量的棋局
    todo = np.array(sorted(set(rows.game_key) - have)); mine = todo[a.job::a.njobs][a.shard::a.nshards]; mine = mine[:a.limit] if a.limit else mine  # 尚無向量的局，依工作與分片取出本分片
    sgf = dict(zip(rows.game_key, zip(rows.sgf_content, rows.color))); del rows  # 棋局鍵對應（棋譜, 執色）；釋放訓練表
    print(f"{len(have)} games have vectors, {len(todo)} do not; this shard: {len(mine)} ({time.time() - t0:.0f} s)", flush=True)  # 印出待算局數
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]  # 讀出網路設定，給工作行程建特徵
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重放用的工作行程池
    try:  # 確保最後關閉行程池
        model, _ = load_humansl(str(te.CKPT), "cuda"); model.eval(); nrows = neutral_rows(); proj = None; plies = 0; skipped = 0; t1 = time.time()  # 載入網路與中性的後設資料列
        for k, start in enumerate(range(0, len(mine), a.part)):  # 逐段處理，每段 a.part 局
            path = OUT / f"rows_j{a.job}of{a.njobs}_s{a.shard}of{a.nshards}_p{k:05d}.npz"  # 本段的輸出檔
            if path.exists():  # 已完成的段（續跑）
                continue  # 略過
            keys = mine[start:start + a.part]; res = list(pool.imap(safe_replay, [(q, *sgf[q]) for q in keys], chunksize=2)); done = [d for d in res if d[1] is not None and len(d[3]) > 0]; skipped += len(res) - len(done)  # 平行重放；保留成功且有手的局
            if not done:  # 本段沒有可用的局
                continue  # 略過
            if proj is None:  # 第一次需要建立投影
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不算梯度，以 bf16 推論
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)  # 試算一個局面以取得主幹通道數
                proj = te.projections(trunk.shape[1], trunk.device)  # 與 trunk_extract.py 同一種子的隨機投影
            lens = np.array([len(d[3]) for d in done]); bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done]); pos = np.concatenate([d[3] for d in done])  # 各局手數與合併後的輸入
            mover = np.concatenate([d[4] for d in done]); target = np.concatenate([d[5] for d in done]); g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch)  # 每手的盤面平均與落點向量
            out, ph, cnt = side_rows(np.hstack([g, l]).astype(np.float64), np.repeat(np.arange(len(done)), lens), target, np.concatenate([np.arange(n) for n in lens]), len(done))  # 每局雙方的列與階段平均
            tmp = path.with_name(path.name + ".tmp.npz"); np.savez(tmp, key=np.array([d[0] for d in done]), T=out[0], O=out[1], PT=ph[0], PO=ph[1], nT=cnt[0], nO=cnt[1]); os.replace(tmp, path); plies += int(lens.sum())  # 先寫暫存檔再改名，避免留下不完整的檔
            if k % 20 == 0:  # 每 20 段報告一次
                print(f"part {k + 1}/{(len(mine) + a.part - 1) // a.part}: {plies} plies, {plies / (time.time() - t1):.0f} plies/s, skipped {skipped}", flush=True)  # 印出進度與速度
        print(f"done: {len(mine)} games, {plies} plies, skipped {skipped}, {(time.time() - t0) / 60:.1f} min", flush=True)  # 印出完成摘要
    finally:  # 無論成功與否
        pool.close(); pool.join()  # 關閉工作行程池


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
