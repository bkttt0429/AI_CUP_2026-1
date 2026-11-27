#!/usr/bin/env python
"""Section 4M M127 (2026-11-14): a wider family of "was the opponent recognised" statistics, chosen without labels.

M125: replacing first-minus-second by first-minus-mean(2..5) of the fused score raises the label-free share of recognised
opponents from 0.204 to 0.226 on the production identifier; M126: alone that is worth about +0.002 at recipe level (below the
adoption threshold on one seed set). Here the same fused score vector is read in more ways:
  m1k     F1 - mean(F2 .. Fk), k = 3, 4, 5, 7, 10, 20          z10, z20   (F1 - mean(F2..Fk)) / sd(F2..Fk)
  lr      a logistic model of "the first candidate is the true player", fitted on the held-out own games only (their first
          candidate is right or wrong; never on the opponent-side or impostor queries, so it cannot learn the two query
          types apart), on: m12, m15, m110, m120, z20, F2 - F3, and per source (block, matcher, sequence nets) the margin of
          the fused first candidate inside that source and whether the source ranks it first.
  --stage fuse --tags ...     one GPU pass per identifier: top-20 fused values, the first candidate, the per-source read-outs,
                              for the validation / test queries and the held-out own games -> results/r_variants/m127_<tag>.npz
  --stage report --tags ...   per tag and statistic: the label-free share (m120_diag_mass.py's definition), the reading quality
                              of the top bands, anchors against m12 and against m15 (report only); --write STAT writes
                              m84_<tag>~<STAT>_{val,test}.npz for the bundle / m121 path.
"""
import argparse  # 解析命令列參數
import json  # 讀取融合權重、寫出報告
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 本專案 src、實驗模組與競賽一儲存庫的 src
import m84_variant as mv  # noqa: E402  匯入前須先設定 sys.path；配對器變體、快取位置與錨點評分（M84）
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  類別平衡分數與成對比較

CACHE, RV, s2 = mv.CACHE, mv.RV, mv.s2  # 配對器快取、結果資料夾、M71 第 2 步的分帶校準
KS = (3, 4, 5, 7, 10, 20)  # m1k 統計量的 k 值


def fuse(tag):  # 一個識別器的融合分數讀出（--stage fuse，需 GPU）
    import torch  # 只有這個階段需要 torch
    t0 = time.time(); dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []  # 查詢檔：候選人、標籤、每個查詢要排除的玩家與負責的網路
    for var, members in mv.MULTI[tag]:  # 此識別器的各個向量來源
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))  # 原型放到 GPU，查詢向量留在 CPU
    w = np.array(json.loads((RV / f"m84_{tag}_eval.json").read_text())["weights"]); n_src = len(src) + 1  # 評估時擬合的融合權重；來源數含五成分區塊

    def zs(ids):  # 一批查詢在各向量來源上的標準化相似度
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); cnt = valid.sum(1, keepdim=True); out = []  # 排除查詢本人後的有效候選人；每個查詢負責的交叉擬合網路
        for protos, qvec in src:  # 逐一來源
            z = torch.zeros(len(ids), C, device=dev)  # 此來源的平均 z 分數
            for m in range(protos.shape[0]):  # 逐一成員
                S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)  # 相似度矩陣與查詢向量
                for j in range(5):  # 五個交叉擬合網路
                    s = nj == j  # 由網路 j 負責的查詢
                    if s.any():  # 有這類查詢時
                        S[s] = qv[s] @ protos[m, j].T  # 對該網路的原型計算相似度
                mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)  # 在有效候選人上標準化後累加
            out.append(z / protos.shape[0])  # 成員平均
        return out, valid  # 各來源的 z 分數與有效候選人遮罩

    def run(ids, block):  # 融合並讀出前 20 名與各來源的讀數
        n = len(ids); TV = np.zeros((n, 20), np.float32); b = np.zeros(n, np.int64); SR = np.zeros((n, n_src), np.uint8); SG = np.zeros((n, n_src), np.float32)  # 前 20 名分數、第一名、各來源對第一名的名次與差距
        for i in range(0, n, 2048):  # 每批 2048 個查詢
            z, valid = zs(ids[i:i + 2048]); parts = [torch.as_tensor(block[i:i + 2048].astype(np.float32), device=dev), *z]; F = sum(w[k] * parts[k] for k in range(n_src))  # 五成分區塊與各來源加權融合
            F = F.masked_fill(~valid, -float("inf")); top = F.topk(20, dim=1); TV[i:i + 2048] = top.values.cpu().numpy(); t1 = top.indices[:, :1]; b[i:i + 2048] = t1[:, 0].cpu().numpy()  # 排除無效候選人後取前 20 名
            for k in range(n_src):  # 逐一來源
                Sk = parts[k].masked_fill(~valid, -float("inf")); own = Sk.gather(1, t1); SR[i:i + 2048, k] = (Sk > own).sum(1).clamp(max=255).cpu().numpy().astype(np.uint8)  # 融合第一名在此來源的名次（0 = 也排第一）
                rest = Sk.scatter(1, t1, -float("inf")); SG[i:i + 2048, k] = (own[:, 0] - rest.max(1).values).cpu().numpy()  # 融合第一名在此來源與其餘最佳者的差距
        return TV, b, SR, SG  # 回傳四種讀出
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]  # 讀入四個分片的五成分區塊分數並依查詢編號排序
    TV, b, SR, SG = run(idx, blk[order]); del blk  # 驗證與測試查詢的讀出，之後釋放區塊分數
    o = slice(0, Q); ref = dict(np.load(RV / f"m84_{tag}_val.npz")); assert np.array_equal(plab[b[o]], ref["f_opp_lab"]) and np.allclose((TV[:, 0] - TV[:, 1])[o], ref["f_opp_m"], atol=1e-3), "the fused pass does not reproduce the stored identifier"  # 核對：必須重現已存的識別器（標籤與 m12 邊際）
    hq = np.arange(2 * Q, 2 * Q + H); HV, hb, HR, HG = run(hq, np.load(CACHE / "block_held.npy")); hit = hb == q["held_truth"]  # 留出本人對局的讀出與第一名是否正確
    np.savez_compressed(RV / f"m127_{tag}.npz", TV=TV, best=b, lab=plab[b], SR=SR, SG=SG, HV=HV, hit=hit, HR=HR, HG=HG, held_truth=q["held_truth"])  # 寫出 m127_<tag>.npz
    print(f"tag {tag}: fused ({n_src} sources), stored identifier reproduced; held-out first candidate right {float(hit.mean()):.3f} ({time.time() - t0:.0f} s)", flush=True)  # 回報核對結果與留出正確率


def simple_stats(TV):  # 由前 20 名分數算出各種把握統計量
    out = {"m12": TV[:, 0] - TV[:, 1]}  # 第一名減第二名
    for k in KS:  # 各個 k
        out[f"m1{k}"] = TV[:, 0] - TV[:, 1:k].mean(1)  # 第一名減第 2 到第 k 名的平均
    for k in (10, 20):  # 標準化版本
        out[f"z{k}"] = (TV[:, 0] - TV[:, 1:k].mean(1)) / np.maximum(TV[:, 1:k].std(1), 1e-6)  # 再除以第 2 到第 k 名的標準差
    return out  # 回傳 {統計量名稱: 陣列}


def features(TV, SR, SG):  # 邏輯迴歸 lr 的特徵
    s = simple_stats(TV)  # 先算簡單統計量
    return np.column_stack([s["m12"], s["m15"], s["m110"], s["m120"], s["z20"], TV[:, 1] - TV[:, 2], np.clip(SG, -8, 8), (SR == 0).astype(np.float32)]).astype(np.float64)  # 統計量、第二名減第三名、各來源的差距與是否也排第一


def fit_logistic(X, y, l2=1.0, iters=200):  # 以牛頓法擬合 L2 正則化的邏輯迴歸
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.column_stack([(X - mu) / sd, np.ones(len(X))]); w = np.zeros(Z.shape[1]); R = l2 * np.eye(Z.shape[1]); R[-1, -1] = 0  # 標準化特徵加截距；截距不做正則化
    for _ in range(iters):  # 牛頓法迭代
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) + R @ w; Hm = (Z * (p * (1 - p))[:, None]).T @ Z + R; step = np.linalg.solve(Hm, g); w -= step  # 梯度、海森矩陣與牛頓步
        if np.abs(step).max() < 1e-8:  # 已收斂
            break  # 提早結束
    return lambda Xn: np.column_stack([(Xn - mu) / sd, np.ones(len(Xn))]) @ w  # 回傳計算 logit 的函式


def auc(s, y):  # 以名次計算 AUC
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1  # 分數的名次與正負例數
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))  # Mann-Whitney 統計量換算成 AUC


def quick_write(tags, k):  # 不做比較，直接寫出識別器檔（--stage write）
    """Identifier files m84_<tag>~<k>_{val,test}.npz for a simple statistic, without the comparisons."""
    q = np.load(CACHE / "queries.npz"); Q, T = int(q["Q"]), int(q["T"]); players = q["players"]; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)  # 讀出列的四段：驗證對手側、驗證目標方（已排除本人，當冒充者）、測試對手側、測試目標方
    d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {x: d82[x] for x in ("key", "pid", "fold", "y", "hd", "target_black")}  # 沿用 M82 檔的查詢資訊欄位
    for tag in tags:  # 逐一識別器
        z = dict(np.load(RV / f"m127_{tag}.npz")); st = simple_stats(z["TV"])  # 融合讀出與統計量
        np.savez_compressed(RV / f"m84_{tag}~{k}_val.npz", **base, f_opp_lab=z["lab"][o], f_opp_m=st[k][o], f_opp_best=players[z["best"][o]], f_imp_lab=z["lab"][t_], f_imp_m=st[k][t_])  # 驗證端識別器檔：以統計量 k 當邊際
        np.savez_compressed(RV / f"m84_{tag}~{k}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=z["lab"][to], f_opp_m=st[k][to], f_opp_best=players[z["best"][to]], f_tgt_lab=z["lab"][tt], f_tgt_m=st[k][tt])  # 測試端識別器檔
        print(f"identifier files m84_{tag}~{k}_{{val,test}}.npz written", flush=True)  # 回報寫出的檔案


def report(tags, write):  # 比較各統計量（--stage report），可選擇寫出其中一個
    q = np.load(CACHE / "queries.npz"); Q, T = int(q["Q"]), int(q["T"]); players = q["players"]; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)  # 讀出列的四段：驗證對手側、驗證目標方（已排除本人，當冒充者）、測試對手側、測試目標方
    d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}; nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); out = {}  # M82 的查詢資訊與各帶的名目占比
    for tag in tags:  # 逐一識別器
        z = dict(np.load(RV / f"m127_{tag}.npz")); st = simple_stats(z["TV"]); hs = simple_stats(z["HV"]); y = z["hit"].astype(float); out[tag] = {}  # 查詢與留出局的統計量；留出局第一名是否正確
        Xh = features(z["HV"], z["HR"], z["HG"]); half = np.array([int(str(players[t])[:8], 16) % 2 for t in z["held_truth"]]); cv = np.zeros(len(y))  # 留出局的特徵；依玩家 id 分成兩半
        for h in (0, 1):  # 交叉擬合
            cv[half == h] = fit_logistic(Xh[half != h], y[half != h])(Xh[half == h])  # 一半擬合、另一半評分
        st["lr"] = fit_logistic(Xh, y)(features(z["TV"], z["SR"], z["SG"])).astype(np.float32); hs["lr"] = cv  # 查詢用全部留出局擬合的模型；留出局用交叉擬合的分數
        print(f"\n=== {tag}: held-out own games, AUC of 'first candidate is right': " + "  ".join(f"{k} {auc(v, y):.4f}" for k, v in hs.items()))  # 各統計量在留出局上的 AUC
        print("  statistic: share of recognised opponents (all / even / HD1-B / HD1-W), top bands same / within one, anchors vs m12 and vs m15 (seeds 0-2 | 6-8)")  # 表頭
        A = {}  # 各統計量與種子組的錨點分數
        for k in ("m12", "m15", *[x for x in st if x not in ("m12", "m15")]):  # 先算兩個參考統計量，再算其他
            Z = {**base, "opp_lab": z["lab"][o], "opp_m": st[k][o], "imp_m": st[k][t_], "imp_lab": z["lab"][t_]}; ones = np.ones(Q, bool); edges, cal, band, g = s2.fit(Z, ones)  # 以統計量 k 當邊際組出識別器並擬合分帶
            mass = {j: float(np.maximum(cal[j]["share"] - nominal, 0).sum()) for j in (0, 1, 2)}; wg = np.array([(g == j).sum() for j in (0, 1, 2)], float); wg /= wg.sum(); top = (g == 0) & (band >= 5); dd = (Z["opp_lab"] - Z["y"])[top]  # 不需標籤的認出占比（同 m120_diag_mass.py）與最高幾帶的標籤差
            row = {"mass": float(sum(wg[j] * mass[j] for j in (0, 1, 2))), "mass_by_group": mass, "top_same": float((dd == 0).mean()), "top_within1": float((np.abs(dd) <= 1).mean()), "heldout_auc": auc(hs[k], y), "vs": {}}  # 此統計量的各項數字
            line = f"  {k:5s} {row['mass']:.4f} / {mass[0]:.4f} / {mass[1]:.4f} / {mass[2]:.4f}   {row['top_same']:.3f} / {row['top_within1']:.3f}  "  # 輸出列的前半
            for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
                A[(k, seeds)] = mv.anchors(Z, seeds)  # 加入此識別器錨點證據後的各折分數
            for ref in ("m12", "m15"):  # 對兩個參考統計量比較
                if k == ref or (ref == "m15" and k == "m12"):  # 不和自己比，m12 也不和 m15 比
                    continue  # 略過
                for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
                    a, c = A[(k, seeds)], A[(ref, seeds)]; yy, n, pid = (np.concatenate([x[j] for x in a]) for j in (2, 3, 4))  # 兩邊的各折結果與題目資訊
                    d = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] for x in c]), yy, pid, n, sizes=SIZES, n=2000); pf = [float(balanced_mean_over_n(u[0], u[2], u[3], sizes=SIZES) - balanced_mean_over_n(v_[0], v_[2], v_[3], sizes=SIZES)) for u, v_ in zip(a, c)]  # 成對差與各折差
                    row["vs"][f"{ref}|{seeds[0]}"] = {**d, "per_fold": pf}  # 記錄比較結果
                line += f" vs {ref}: " + " | ".join(f"{row['vs'][f'{ref}|{s_}']['diff']:+.4f} [{row['vs'][f'{ref}|{s_}']['ci95'][0]:+.4f}, {row['vs'][f'{ref}|{s_}']['ci95'][1]:+.4f}] ({sum(e > 0 for e in row['vs'][f'{ref}|{s_}']['per_fold'])}/4)" for s_ in (0, 6))  # 接上兩組種子的差、區間與正的折數
            out[tag][k] = row; print(line, flush=True)  # 存入報告並印出
            if write and k == write:  # 要求寫出這個統計量時
                np.savez_compressed(RV / f"m84_{tag}~{k}_val.npz", **base, f_opp_lab=z["lab"][o], f_opp_m=st[k][o], f_opp_best=players[z["best"][o]], f_imp_lab=z["lab"][t_], f_imp_m=st[k][t_])  # 驗證端識別器檔
                np.savez_compressed(RV / f"m84_{tag}~{k}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=z["lab"][to], f_opp_m=st[k][to], f_opp_best=players[z["best"][to]], f_tgt_lab=z["lab"][tt], f_tgt_m=st[k][tt])  # 測試端識別器檔
                print(f"    identifier files m84_{tag}~{k}_{{val,test}}.npz written")  # 回報寫出的檔案
    (RV / f"m127_conf_family{'_' + '_'.join(tags) if len(tags) < 3 else ''}.json").write_text(json.dumps(out, indent=1, default=float))  # 寫出報告（標籤少時檔名附上標籤）


if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("fuse", "report", "write")); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--write", default=None); a_ = ap.parse_args()  # 階段、識別器標籤與要寫出的統計量
    if a_.stage == "fuse":  # GPU 階段
        for t in a_.tags:  # 逐一識別器
            fuse(t)  # 計算融合讀出
    elif a_.stage == "write":  # 只寫檔
        quick_write(a_.tags, a_.write or "m15")  # 預設用 m15 統計量
    else:  # 報告階段
        report(a_.tags, a_.write)  # 比較各統計量，必要時寫檔
