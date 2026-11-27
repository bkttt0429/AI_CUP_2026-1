#!/usr/bin/env python
"""Section 4M M164 (2026-11-15): identifier builder for the by-games identity networks - any matcher sources, centre sources,
the activity prior (design doc M161q / M163).

For every --tag NAME:var[,var...] (sources fused with the cached block; every member of each variant file is used):
  per source the standardised score of every query against every candidate, averaged over the members (m84_variant's rule).
  A variant file that holds `has` [members, 5, C] (a centre source: only some candidates have a vector) is standardised over
  the candidates WITH a vector, the others score 0 there (neutral): centres and prototypes are never mixed in one matrix;
  one weight per source, fitted on the held-out own games (ident_pool.fit_weights; a damped refit if Newton diverges);
  the activity prior: fused score + beta * log n_c (n_c = the candidate's number of training games), beta fitted by the
  n_c-weighted likelihood of the held-out own games (--prior fit), a given number, or 0;
  all queries fused -> results/r_variants/m127_<NAME>.npz (m127's layout), m84_<NAME>_eval.json (weights, beta, held-out
  readings), m84_<NAME>~m15_{val,test}.npz (m127_conf_family.quick_write).
Nothing is overwritten: a NAME whose files exist is refused.

  python -I scripts/m_series/m164_ident.py --tag SQ1:S7t --tag SQ2:P,S2y,S4ft,S4Lt,S7t --prior fit
"""
import argparse  # 解析命令列參數
import json  # 寫出評估報告
import sys  # 設定模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取玩家折表

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series"), str(Path.home() / "BK_AICUP_2026-2/src")]  # 讓 aicup 套件、實驗模組與玩家辨識競賽的套件可匯入
import ident_pool as ip  # noqa: E402  來源權重的擬合（fit_weights）
import m127_conf_family as m127  # noqa: E402  把握統計量與識別器檔的寫出
import m84_variant as mv  # noqa: E402  快取位置與既有的識別器名稱

CACHE, RV = mv.CACHE, mv.RV  # 查詢快取與結果資料夾
BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]  # 依局數分帶，用於列數加權的 top-1


def fit_w(Zs, truth, log, l2=1e-3):  # 擬合各來源的融合權重
    """ident_pool.fit_weights; when its undamped Newton diverges, a damped Newton on the same objective."""
    w = ip.fit_weights(Zs, truth, l2=l2)  # 先用 ident_pool 的牛頓法
    if np.isfinite(w).all() and np.abs(w).max() < 20:  # 沒有發散就直接採用
        return w  # 回傳權重
    log(f"   fit_weights diverged (largest |w| {np.abs(w).max():.1f}): damped refit")  # 回報發散，改用阻尼版重新擬合
    Z = np.stack(Zs).astype(np.float32); J, Qn, _ = Z.shape; qi = np.arange(Qn); w = np.ones(J)  # 各來源分數疊成陣列，權重從 1 開始

    def obj(w_):  # 目標函式：負的平均對數概似加 L2 懲罰
        lg = np.einsum("j,jqc->qc", w_, Z); lg -= lg.max(1, keepdims=True); P = np.exp(lg); s = P.sum(1, keepdims=True)  # 融合分數的 softmax
        return float(-(lg[qi, truth] - np.log(s[:, 0])).mean() + l2 * (w_ ** 2).sum()), P / s  # 回傳目標值與機率
    f0, P = obj(w)  # 起點的目標值
    for _ in range(100):  # 最多 100 次阻尼牛頓迭代
        EZ = np.einsum("jqc,qc->jq", Z, P); grad = (EZ - Z[:, qi, truth]).mean(1) + 2 * l2 * w  # 梯度
        A = (Z * P[None]).reshape(J, -1); Hm = (A @ Z.reshape(J, -1).T) / Qn - (EZ @ EZ.T) / Qn + 2 * l2 * np.eye(J); step = np.linalg.solve(Hm + 1e-6 * np.eye(J), grad); t = 1.0  # 黑塞矩陣與牛頓步，步長從 1 開始
        while t > 1e-4:  # 回溯線搜尋
            f1, P1 = obj(w - t * step)  # 試走這一步
            if f1 <= f0:  # 目標沒有變差
                break  # 接受這個步長
            t /= 2  # 步長減半
        if t <= 1e-4 or abs(f0 - f1) < 1e-9:  # 步長太小或已收斂
            break  # 停止迭代
        w, f0, P = w - t * step, f1, P1  # 更新權重
    return w  # 回傳權重


def main():  # 主流程：逐一建出 --tag 指定的識別器
    import torch  # 用 GPU（若有）計算標準化分數
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--tag", action="append", required=True); ap.add_argument("--prior", default="fit"); ap.add_argument("--stat", default="m15"); ap.add_argument("--heldout-only", action="store_true"); a_ = ap.parse_args()  # 參數：識別器組成、先驗、把握統計量、只做留出局
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    dev = "cuda" if torch.cuda.is_available() else "cpu"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); truth = q["held_truth"].astype(np.int64)  # 查詢表：候選玩家、標籤、排除對象、負責的網路；留出局的真實玩家
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(float).reindex(players.astype(str)).to_numpy(); assert np.isfinite(ng).all() and (ng >= 1).all(); logn = np.log(ng); ln_t = torch.as_tensor(logn, device=dev, dtype=torch.float32)  # 每位候選人的訓練局數 n_c 與 log n_c（活躍度先驗）
    band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]]); share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum(); wb = share[band] / np.bincount(band, minlength=len(BANDS))[band]; wb /= wb.sum(); wa = ng[truth] / ng[truth].sum()  # 留出局的局數帶，以及列數加權與 n_c 加權的權重
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy"); assert (excl[hq] < 0).all(); blk = idx = None  # 留出本人局的查詢列與其區塊分數
    for spec in a_.tag:  # 每個要建的識別器
        name, vars_ = spec.split(":"); vars_ = vars_.split(","); assert name not in mv.TAGS and name not in mv.MULTI, f"{name} is an existing tag"  # 名稱與來源變體；不可與既有名稱重複
        assert a_.heldout_only or not ((RV / f"m127_{name}.npz").exists() or (RV / f"m84_{name}_eval.json").exists()), f"{name}: results exist already, nothing overwritten"  # 結果已存在就拒絕，絕不覆寫
        src = []  # 各來源的原型、查詢向量與中心遮罩
        for v in vars_:  # 每個來源變體檔
            V = np.load(CACHE / f"var_{v}.npz"); src.append((torch.as_tensor(V["protos"], device=dev), V["qvec"], torch.as_tensor(V["has"], device=dev) if "has" in V.files else None))  # 原型放上裝置；含 has 的是中心來源
        n_src = len(src) + 1; log(f"== identifier {name}: block + " + " + ".join(f"{v} ({s[0].shape[0]} member(s){', centre source' if s[2] is not None else ''})" for v, s in zip(vars_, src)) + f"; device {dev}")  # 來源數（含區塊分數）並回報組成

        def zs(ids):  # 一批查詢對每位候選人的各來源標準化分數
            ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); out = []  # 排除查詢本人；記下各查詢由哪個網路負責
            for protos, qvec, has in src:  # 每個來源
                z = torch.zeros(len(ids), C, device=dev)  # 這個來源的成員平均分數
                for m in range(protos.shape[0]):  # 每個成員
                    S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev); ok = valid.clone() if has is not None else valid  # 相似度、查詢向量與可比較的候選人
                    for j in range(5):  # 五個交叉擬合網路（1-4 與全體）
                        s = nj == j  # 由網路 j 負責的查詢
                        if s.any():  # 有這類查詢時
                            S[s] = qv[s] @ protos[m, j].T  # 對網路 j 原型的相似度
                            if has is not None:  # 中心來源
                                ok[s] &= has[m, j][None]  # 只在有中心的候選人上比較
                    cnt = ok.sum(1, keepdim=True).clamp(min=1); mu = (S * ok).sum(1, keepdim=True) / cnt; zz = (S - mu) / (((S - mu) ** 2 * ok).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)  # 在可比較的候選人上標準化
                    z += torch.where(ok, zz, torch.zeros_like(zz)) if has is not None else zz  # 中心來源裡沒有中心的候選人記 0（中性）
                out.append(z / protos.shape[0])  # 成員平均
            return out, valid  # 回傳各來源分數與有效遮罩
        # ---------------------------------------------------------------- 在留出本人局上擬合權重與先驗
        zh = [x.cpu().numpy() for x in zs(hq)[0]]; w = fit_w([hb.astype(np.float32), *zh], truth, log); Fh = w[0] * hb.astype(np.float32) + sum(w[k + 1] * zh[k] for k in range(len(zh)))  # 留出局的各來源分數、融合權重與融合分數
        rnk = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1); acc = {}  # noqa: E731  真實玩家的名次函式與準度紀錄
        for nm, F in [(f"source{k} ({vars_[k]}) alone", zh[k]) for k in range(len(zh))] + [("fused", Fh)]:  # 各來源單獨與融合後
            r_ = rnk(F); acc[nm] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean()), "top1_rows": float((wb * (r_ == 0)).sum()), "top1_nc": float((wa * (r_ == 0)).sum())}  # top-1／5／20 與兩種加權的 top-1
        Ft = torch.as_tensor(Fh, device=dev, dtype=torch.float32); tr = torch.as_tensor(truth, device=dev); wt = torch.as_tensor(wa, device=dev, dtype=torch.float32); lls = {}  # 融合分數、真實玩家與 n_c 權重放上裝置
        for beta in np.arange(0.0, 2.01, 0.05):  # 先驗強度 beta 的格點
            lls[round(float(beta), 2)] = float((wt * torch.log_softmax(Ft + float(beta) * ln_t[None], dim=1)[torch.arange(H, device=dev), tr]).sum())  # n_c 加權的對數概似
        beta = max(lls, key=lls.get) if a_.prior == "fit" else float(a_.prior); r_ = rnk(Fh + beta * logn[None].astype(np.float32)); acc["fused with the prior"] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean()), "top1_rows": float((wb * (r_ == 0)).sum()), "top1_nc": float((wa * (r_ == 0)).sum())}  # 取概似最高的 beta（或給定值），並報加上先驗後的準度
        log(f"   weights block {w[0]:.2f}, " + ", ".join(f"{vars_[k]} {w[k + 1]:.2f}" for k in range(len(zh))) + f"; beta {beta:.2f} ({'fitted' if a_.prior == 'fit' else 'given'}; likelihood-best {max(lls, key=lls.get):.2f})")  # 回報權重與 beta
        for nm, x in acc.items():  # 各讀數
            log(f"   held-out own games, {nm}: top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; weighted by rows {x['top1_rows']:.3f}, by n_c {x['top1_nc']:.3f}")  # 回報留出局的準度
        rep = {"sources": vars_, "weights": w.tolist(), "beta": beta, "beta_loglik": lls, "heldout": acc}  # 評估報告
        if a_.heldout_only:  # 只做留出局時
            continue  # 不算全部查詢
        # ---------------------------------------------------------------- 全部查詢
        if blk is None:  # 區塊分數只載入一次
            idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)])[order]; log(f"   blocks loaded ({len(idx)} queries)")  # 載入四個分片的區塊分數並依查詢列排序

        def run(ids, block):  # 對一組查詢做融合，留下前 20 名與各來源的讀數
            n = len(ids); TV = np.zeros((n, 20), np.float32); bb = np.zeros(n, np.int64); SR = np.zeros((n, n_src), np.uint8); SG = np.zeros((n, n_src), np.float32)  # 前 20 名融合分數、第一名候選人、各來源中的名次與邊際
            for i in range(0, n, 2048):  # 每批 2048 個查詢
                z, valid = zs(ids[i:i + 2048]); parts = [torch.as_tensor(block[i:i + 2048].astype(np.float32), device=dev), *z]; F = (sum(float(w[k]) * parts[k] for k in range(n_src)) + beta * ln_t[None]).masked_fill(~valid, -float("inf"))  # 融合分數＝加權和加上 beta x log n_c，排除無效候選人
                top = F.topk(20, dim=1); TV[i:i + 2048] = top.values.cpu().numpy(); t1 = top.indices[:, :1]; bb[i:i + 2048] = t1[:, 0].cpu().numpy()  # 前 20 名與第一名
                for k in range(n_src):  # 每個來源
                    Sk = parts[k].masked_fill(~valid, -float("inf")); own = Sk.gather(1, t1); SR[i:i + 2048, k] = (Sk > own).sum(1).clamp(max=255).cpu().numpy().astype(np.uint8)  # 第一名候選人在這個來源中的名次
                    rest = Sk.scatter(1, t1, -float("inf")); SG[i:i + 2048, k] = (own[:, 0] - rest.max(1).values).cpu().numpy()  # 第一名在這個來源中領先其他人的邊際
            return TV, bb, SR, SG  # 回傳
        TV, bb, SR, SG = run(idx, blk); HV, hbest, HR, HG = run(hq, hb); hit = hbest == truth  # 全部查詢與留出局各跑一次；留出局第一名是否正確
        np.savez_compressed(RV / f"m127_{name}.npz", TV=TV, best=bb, lab=plab[bb], SR=SR, SG=SG, HV=HV, hit=hit, HR=HR, HG=HG, held_truth=truth, beta=beta)  # 寫出 m127 格式的融合結果
        o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); st = m127.simple_stats(TV)[a_.stat]  # 驗證與測試的對手側、本人側；把握統計量（預設 m15）
        for nm, sl, null in (("validation opponent side", o, t_), ("test opponent side", to, tt)):  # 對手側對照本人側（零分布）
            ex = {qq: float((st[sl] > np.quantile(st[null], qq)).mean() - (1 - qq)) for qq in (0.90, 0.98, 0.995)}; grid = np.quantile(st[null], np.linspace(0.5, 0.999, 500)); sup = float(max((st[sl] > t).mean() - (st[null] > t).mean() for t in grid))  # 超過本人側分位數的超額比例與最大超額（不看標籤的認出比例下界）
            rep[nm] = {"excess": {str(k): v for k, v in ex.items()}, "largest_excess": sup, "median_games_of_first": float(np.median(ng[bb[sl]]))}  # 存入報告
            log(f"   {nm}: excess above the own-side q90 / q98 / q99.5 " + " / ".join(f"{ex[k]:+.3f}" for k in ex) + f", largest {sup:.3f}; median games of the first candidate {np.median(ng[bb[sl]]):.0f}")  # 回報超額與第一名候選人的局數中位數
        (RV / f"m84_{name}_eval.json").write_text(json.dumps(rep, indent=1)); m127.quick_write([name], a_.stat); log(f"   identifier {name} written (held-out first candidate right {float(hit.mean()):.3f})")  # 寫出評估報告與 m84_<NAME>~m15 的驗證／測試識別器檔


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
