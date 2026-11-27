#!/usr/bin/env python
"""Section 4M M124 (2026-11-14): test-time symmetry averaging of the QUERY vectors of the symmetry-augmented identity networks.

M120's preview: the anchors follow the share of opponents recognised with a clear margin, and that share reacts to averaging
(several seeds) much more than to one network's held-out accuracy. A network trained with a random board symmetry per episode is
roughly, not exactly, invariant to the orientation of a game; the mean of its vectors over the eight orientations of the query
game is a free average of eight correlated readings. Here every finished network of results/m120 re-embeds its queries (the
opponent and target sides of its validation games, the held-out games, the test games) under the eight symmetries of the move
coordinates (trunk vectors unchanged, as in training), the unit vectors are averaged and renormalised; prototypes unchanged.
Output: results/m84_cache/var_<name>.npz in m84_variant.py's format (protos from results/m120, the averaged query vectors), and a
report (the identity orientation must reproduce the stored vectors; cosine between orientations; held-out top-1 of the plain and
the averaged queries per network).

  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch python -I scripts/m_series/m124_tta_queries.py --seeds 0 --name S4t
"""
import argparse  # 解析命令列參數
import json  # 寫出並在合併時讀回每個網路的報告
import os  # fork 子行程、設定 GPU
import sys  # 調整模組搜尋路徑與錯誤結束
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m88_seqid as base  # noqa: E402  匯入前須先設定 sys.path；序列身分網路的載入器與模型
from m112_idaug import sym_table  # noqa: E402  八種盤面對稱的對照表

OUT, CK, TMP = ROOT / "results/m120", Path("/work/u2507115/m120_ckpt"), ROOT / "results/m124_parts"  # M120 的輸出與檢查點、本程式的中間檔資料夾


def one(v, seed, G, Tt, log):  # 一個網路的查詢在八個方位下重新嵌入並平均
    import torch  # 子行程內才匯入 torch
    dev = "cuda"; tag = f"{v}{f'_s{seed}' if seed else ''}"; D_OUT = base.D_OUT; SYM = sym_table()  # 網路標籤、輸出維度與對稱表
    st = torch.load(CK / f"ckpt_{tag}.pt", map_location=dev, weights_only=False); assert "opt" not in st, f"{tag}: the network is still training"  # 只用訓練完成後的精簡檢查點
    net = base.build_model().to(dev); net.load_state_dict(st["best_state"]); net.eval()  # 載入最佳狀態並切到評估模式
    q = np.load(base.CACHE / "queries.npz"); Q, H = int(q["Q"]), int(q["H"]); gidx = {k: i for i, k in enumerate(G["key"])}; j = base.NETS.index(v); mine = np.flatnonzero(q["net"] == j)  # 查詢檔；本網路負責的查詢
    tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731  陣列轉成 GPU 張量

    def embed(src, rows, subj, k):  # 在第 k 個方位下計算對局向量
        out = np.zeros((len(rows), D_OUT), np.float32)  # 輸出向量
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不記錄梯度、bf16 計算
            for i in range(0, len(rows), 1024):  # 每批 1024 局
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(SYM[k][src["POS"][r].astype(np.int64)]), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()  # 落點套用對稱 k，主幹向量不變（同訓練時）
        return out  # 回傳向量
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])  # 驗證端與留出查詢：對局列號與主角顏色（對手側查詢換另一方）
    te = mine[q["qtest"][mine] >= 0]; rows_t = q["qtest"][te].astype(np.int64); subj_t = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows_t], Tt["SUBJ"][rows_t]) if len(te) else np.zeros(0, np.int64)  # 測試端查詢的對局與主角顏色
    loc = {g_: i for i, g_ in enumerate(mine)}; pos_tr = np.array([loc[g_] for g_ in tr[ok]], dtype=np.int64); pos_te = np.array([loc[g_] for g_ in te], dtype=np.int64)  # 查詢在輸出陣列中的位置
    acc = np.zeros((len(mine), D_OUT), np.float64); first = None; cos = []  # 八個方位的累加、恆等方位的向量、方位間餘弦
    for k in range(8):  # 八種盤面對稱
        E = np.zeros((len(mine), D_OUT), np.float32)  # 此方位的查詢向量
        if len(rows):  # 有驗證端或留出查詢時
            E[pos_tr] = embed(G, rows, subj, k)  # 嵌入驗證端與留出查詢
        if len(te):  # 有測試端查詢時
            E[pos_te] = embed(Tt, rows_t, subj_t, k)  # 嵌入測試端查詢
        if k == 0:  # 恆等方位
            first = E.copy()  # 保留作為原始查詢向量
        else:  # 其他方位
            cos.append(float((E * first).sum(1)[np.abs(first).sum(1) > 0].mean()))  # 與恆等方位的平均餘弦（略過全零列）
        acc += E  # 累加各方位
        log(f"  net {tag}: orientation {k} done")  # 記錄進度
    qv = (acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)).astype(np.float32)  # 八個方位平均後重新正規化
    z = np.load(OUT / f"out_{tag}.npz"); assert np.array_equal(z["qidx"], mine); dmax = float(np.abs(z["qvec"] - first).max())  # 恆等方位應重現 M120 存下的查詢向量
    rep = {"queries": int(len(mine)), "identity_vs_stored_max_abs_diff": dmax, "cos_between_orientations": cos, "mean_cos": float(np.mean(cos))}  # 本網路的報告
    held = (mine >= 2 * Q) & (mine < 2 * Q + H)  # 查詢檔中留出本人對局那一段
    if held.any():  # 本網路查詢中的留出本人對局（網路 -1 擁有全部）
        truth = q["held_truth"][mine[held] - 2 * Q]; P = z["protos"]  # 正確玩家與原型
        for nm, X in (("plain", first), ("averaged", qv)):  # 原始與平均後的查詢向量
            S = X[held] @ P.T; r = (S > S[np.arange(len(truth)), truth][:, None]).sum(1); rep[f"heldout_top1_{nm}"] = float((r == 0).mean()); rep[f"heldout_top20_{nm}"] = float((r < 20).mean())  # 留出局的 top-1 與 top-20
    TMP.mkdir(parents=True, exist_ok=True); np.savez(TMP / f"tta_{tag}.npz", qidx=mine, qvec=qv); (TMP / f"tta_{tag}.json").write_text(json.dumps(rep, indent=1))  # 寫出中間檔與報告
    log(f"net {tag}: {len(mine)} queries; identity orientation vs stored vectors max |diff| {dmax:.4f}; mean cosine between orientations {rep['mean_cos']:.4f}" + (f"; held-out top-1 plain {rep['heldout_top1_plain']:.3f} -> averaged {rep['heldout_top1_averaged']:.3f}" if held.any() else ""))  # 回報核對結果與成績變化


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--seeds", default="0")  # 要處理的種子（以逗號分隔）
    ap.add_argument("--name", default="S4t")  # 輸出的變體名稱
    ap.add_argument("--merge-only", action="store_true")  # 只合併已存在的中間檔
    a_ = ap.parse_args()  # 解析參數
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    seeds = [int(x) for x in a_.seeds.split(",")]; jobs = [(v, s) for s in seeds for v in base.NETS]; base.FULL, base.EXT = True, None  # 每個種子的五個網路；全部對局、不加外部帳號
    if not a_.merge_only:  # 需要重新嵌入時
        gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]  # 可用的 GPU
        G, Tt = base.load_games(False, log); kids = {}; queue = list(jobs); free = list(gpus); bad = 0  # 對局只載入一次；工作佇列與空閒 GPU
        while queue or kids:  # 直到全部工作完成
            while queue and free:  # 有工作也有空閒 GPU 時
                v, s = queue.pop(0)  # 取出下一個網路
                if (TMP / f"tta_{v}{f'_s{s}' if s else ''}.npz").exists():  # 先前的執行已完成
                    continue  # 略過
                gpu = free.pop(0); pid = os.fork()  # 占用一張 GPU 並 fork 子行程
                if pid == 0:  # 子行程
                    code = 0  # 預設成功
                    try:  # 處理此網路
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu); t1 = time.time()  # 匯入 torch 前先指定 GPU
                        one(v, s, G, Tt, lambda m: print(f"{m} ({time.time() - t1:.0f} s)", flush=True))  # 計算八個方位的平均查詢向量
                    except BaseException:  # noqa: BLE001  任何例外都記錄後以失敗結束
                        import traceback  # 印出錯誤堆疊用
                        traceback.print_exc(); code = 1  # 印出錯誤並標記失敗
                    finally:  # 不論成敗
                        sys.stdout.flush(); os._exit(code)  # 直接結束子行程，不回到父行程的迴圈
                kids[pid] = gpu  # 父行程記下子行程占用的 GPU
            if not kids:  # 沒有執行中的子行程
                continue  # 回到迴圈開頭重新檢查
            pid, status = os.wait(); free.append(kids.pop(pid)); bad += status != 0  # 等一個子行程結束並釋放其 GPU
        if bad:  # 有網路失敗
            sys.exit(f"{bad} networks failed")  # 不合併，以錯誤結束
    q = np.load(base.CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, base.D_OUT), np.float32); qvec = np.zeros((len(seeds), N, base.D_OUT), np.float16); rep = {}  # 配對器快取格式的陣列
    for si, s in enumerate(seeds):  # 逐一種子
        for j, v in enumerate(base.NETS):  # 五個交叉擬合網路
            tag = f"{v}{f'_s{s}' if s else ''}"; z = np.load(OUT / f"out_{tag}.npz"); t = np.load(TMP / f"tta_{tag}.npz"); assert np.array_equal(z["qidx"], t["qidx"]); protos[si, j] = z["protos"]; qvec[si, t["qidx"]] = t["qvec"]; rep[tag] = json.loads((TMP / f"tta_{tag}.json").read_text())  # 原型沿用 M120，查詢向量換成平均後的
    np.savez(base.CACHE / f"var_{a_.name}.npz", protos=protos, qvec=qvec); (base.CACHE / f"var_{a_.name}.json").write_text(json.dumps(rep, indent=1))  # 寫出 var_<name>.npz 與報告
    log(f"var_{a_.name} written ({len(seeds)} seed(s))")  # 回報完成
    for tag, r in rep.items():  # 彙總每個網路
        print(f"  net {tag}: mean cosine between orientations {r['mean_cos']:.4f}" + (f"; held-out top-1 {r['heldout_top1_plain']:.3f} -> {r['heldout_top1_averaged']:.3f}, top-20 {r['heldout_top20_plain']:.3f} -> {r['heldout_top20_averaged']:.3f}" if "heldout_top1_plain" in r else ""))  # 方位間餘弦與留出成績的變化


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
